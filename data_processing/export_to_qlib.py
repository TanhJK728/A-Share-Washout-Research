import pandas as pd
from clickhouse_driver import Client
from pathlib import Path
import subprocess
import requests
import shutil
import sys
import time

# Config
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_DB = "stock_data"

EXPORT_DIR = Path("qlib_data/cn_data") # Qlib 数据存储位置
CSV_TEMP_DIR = Path("qlib_data/csv_temp") # 临时 CSV 存放目录

DUMP_SCRIPT_URL = "https://raw.githubusercontent.com/microsoft/qlib/main/scripts/dump_bin.py"
DUMP_SCRIPT_PATH = Path("dump_bin.py")


def download_dump_script(force: bool = True) -> None:
    """更新 dump_bin.py 并自动打补丁以适配 macOS"""
    if not DUMP_SCRIPT_PATH.exists() or force:
        print(f"正在下载/更新 Qlib 数据转换脚本: {DUMP_SCRIPT_URL}")
        try:
            resp = requests.get(DUMP_SCRIPT_URL, timeout=20)
            resp.raise_for_status()
            DUMP_SCRIPT_PATH.write_bytes(resp.content)
            print("dump_bin.py 下载/更新成功")
        except Exception as e:
            if DUMP_SCRIPT_PATH.exists():
                print(f"下载失败: {e}，使用本地缓存副本。")
            else:
                raise RuntimeError(f"dump_bin.py 下载失败: {e}") from e

    try:
        content = DUMP_SCRIPT_PATH.read_text(encoding='utf-8')
        if ".iterdir()" in content:
            print("检测到原始脚本, 正在应用 macOS 兼容性补丁...")
            new_content = content.replace(".iterdir()", ".glob('*.csv')")
            DUMP_SCRIPT_PATH.write_text(new_content, encoding='utf-8')
        elif ".glob('*.csv')" in content:
            print("脚本已包含 macOS 兼容性补丁.")
    except Exception as e:
        print(f"应用补丁失败 (非致命错误): {e}")


def hard_reset_dir(dir_path: Path) -> None:
    """强制重建目录"""
    if dir_path.exists():
        try:
            shutil.rmtree(dir_path)
        except Exception as e:
            print(f"删除目录失败: {dir_path} ({e})，尝试忽略并继续")
    dir_path.mkdir(parents=True, exist_ok=True)


def sanitize_csv_temp_dir(dir_path: Path) -> None:
    """删除所有非 CSV 的文件/目录"""
    if not dir_path.exists():
        return
    removed = 0
    for p in list(dir_path.iterdir()):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
            elif p.is_file() and p.suffix.lower() != ".csv":
                p.unlink(missing_ok=True)
                removed += 1
        except Exception as e:
            pass
    if removed > 0:
        print(f"已清理 csv_temp 中 {removed} 个非 CSV 项")


def export_clickhouse_to_qlib():
    # 0) 更新 dump_bin.py
    download_dump_script(force=False)

    # 1) 读 ClickHouse
    client = Client(host=CLICKHOUSE_HOST, database=CLICKHOUSE_DB, settings={"use_numpy": True})

    print("正在从 ClickHouse 读取全量数据...")
    
    sql = """
    SELECT
        t1.ts_code    AS ts_code,
        t1.trade_date AS trade_date,
        t1.open       AS open, 
        t1.close      AS close, 
        t1.high       AS high, 
        t1.low        AS low, 
        t1.vol        AS volume, 
        t1.amount     AS amount,
        
        -- 使用 ifNull 防止空值报错
        ifNull(t1.turnover_rate, 0) AS turnover,

        -- 1. 新闻情绪 (Sentiment)
        ifNull(t_sent.avg_score, 0)   AS sentiment,
        
        -- 2. 板块得分 (Sector Score)
        ifNull(t_sector.alpha_score, 0) AS sector_score,
        
        -- 3. 最终合成Alpha (Total Alpha)
        ifNull(t_alpha.alpha_score, 0)  AS total_score

    FROM stock_daily t1
    
    -- 关联新闻表
    LEFT JOIN (
        SELECT ts_code, trade_date, avg(score) as avg_score
        FROM stock_news_sentiment
        GROUP BY ts_code, trade_date
    ) t_sent ON t1.ts_code = t_sent.ts_code AND t1.trade_date = t_sent.trade_date
    
    -- 关联板块轮动因子
    LEFT JOIN (
        SELECT ts_code, trade_date, alpha_score
        FROM stock_daily_alpha
        WHERE strategy_name = 'sector_rotation_v1'
    ) t_sector ON t1.ts_code = t_sector.ts_code AND t1.trade_date = t_sector.trade_date
    
    -- 关联最终合成因子
    LEFT JOIN (
        SELECT ts_code, trade_date, alpha_score
        FROM stock_daily_alpha
        WHERE strategy_name = 'multi_factor_v1'
    ) t_alpha ON t1.ts_code = t_alpha.ts_code AND t1.trade_date = t_alpha.trade_date
    
    ORDER BY t1.trade_date ASC
    """

    df = client.query_dataframe(sql)
    print(f"读取完成！共 {len(df)} 行数据。")
    
    # 2) 规范字段
    try:
        df["date"] = pd.to_datetime(df["trade_date"])
    except KeyError:
        # 兼容性处理
        for col in df.columns:
            if "trade_date" in col:
                df["trade_date"] = df[col]
                df["date"] = pd.to_datetime(df["trade_date"])
                break
        else:
            raise KeyError("无法找到 trade_date 列")

    if "ts_code" not in df.columns:
        for col in df.columns:
            if "ts_code" in col:
                df["ts_code"] = df[col]
                break

    df["symbol"] = df["ts_code"]
    df["volume"] = df["volume"].astype(float)
    df["factor"] = 1.0
    
    # 确保 turnover 是 float 类型
    if "turnover" in df.columns:
        df["turnover"] = df["turnover"].astype(float)
    else:
        print("警告：结果中没有找到 turnover 列, 将使用全0填充")
        df["turnover"] = 0.0

    # 3) 清理并生成 CSV
    print("正在重建临时目录...")
    hard_reset_dir(CSV_TEMP_DIR)

    print("正在生成临时 CSV 文件 (按股票拆分)...")
    
    cols_to_write = ["date", "open", "close", "high", "low", "volume", "amount", "factor", 
                     "turnover", "sentiment", "sector_score", "total_score"]

    grouped = df.groupby("symbol", observed=False)
    total = grouped.ngroups
    count = 0

    for symbol, g in grouped:
        symbol = str(symbol)
        safe_symbol = symbol.replace("/", "_").replace("\\", "_").strip()
        file_path = CSV_TEMP_DIR / f"{safe_symbol}.csv"

        g.to_csv(
            file_path,
            index=False,
            columns=cols_to_write,
            date_format="%Y-%m-%d",
        )

        count += 1
        if count % 1000 == 0:
            print(f" 已处理 {count}/{total} 只股票...")

    # 4) 二次清理
    sanitize_csv_temp_dir(CSV_TEMP_DIR)

    print("CSV 准备就绪，开始调用 Qlib 转换脚本...")

    # 5) 调用 dump_bin.py
    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(DUMP_SCRIPT_PATH),
        "dump_all",
        "--data_path", str(CSV_TEMP_DIR),
        "--qlib_dir", str(EXPORT_DIR),
        "--include_fields", "open,close,high,low,volume,amount,factor,turnover,sentiment,sector_score,total_score",
        "--date_field_name", "date",
        "--symbol_field_name", "symbol",
        "--file_suffix", ".csv",
    ]

    print(f"执行命令: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n转换完成. Qlib 数据已更新至: {EXPORT_DIR.resolve()}")
    except subprocess.CalledProcessError as e:
        print(f"\n转换失败: {e}")
        print("请检查控制台输出的错误信息.")
        raise


if __name__ == "__main__":
    export_clickhouse_to_qlib()
