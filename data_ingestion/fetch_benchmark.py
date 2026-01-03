import akshare as ak
import pandas as pd
from clickhouse_driver import Client
from datetime import datetime

# Config
CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_DB = 'stock_data'
BENCHMARK_CODE = "sh000300"  # AkShare 接口用的代码
TARGET_CODE = "SH000300"     # 存入数据库和 Qlib 用的代码

def fetch_and_save_benchmark():
    print(f"正在下载沪深300指数数据 ({BENCHMARK_CODE})...")
    
    # 1. 指数日K
    try:
        df = ak.stock_zh_index_daily(symbol=BENCHMARK_CODE)
        print(f"获取成功！共 {len(df)} 条历史记录")
    except Exception as e:
        print(f"AkShare 下载失败: {e}")
        return

    # 2. 数据清洗
    df['trade_date'] = pd.to_datetime(df['date'])
    df['ts_code'] = TARGET_CODE 
    
    # 统一字段名
    if 'vol' not in df.columns and 'volume' in df.columns:
        df = df.rename(columns={'volume': 'vol'})
    elif 'vol' in df.columns:
        pass # keep vol
    
    # 补充 Qlib 所需 fields
    df['pre_close'] = df['close'].shift(1).fillna(df['open'])
    df['change'] = df['close'] - df['pre_close']
    df['pct_chg'] = df['change'] / df['pre_close'] * 100
    
    # 确保 volume 存在且格式正确
    if 'vol' in df.columns:
         df['volume'] = df['vol']
    else:
         df['volume'] = 0

    df['amount'] = 0.0
    df['turnover_rate'] = 0.0

    # 3. 存入 ClickHouse
    print(f"正在写入 ClickHouse 表 {CLICKHOUSE_DB}.stock_daily ...")
    
    # 添加 settings 参数, 解除分区写入限制
    client = Client(
        host=CLICKHOUSE_HOST, 
        database=CLICKHOUSE_DB, 
        settings={
            'use_numpy': True,
            'max_partitions_per_insert_block': 2000
        }
    )
    
    # 删除旧数据
    client.execute(f"ALTER TABLE stock_daily DELETE WHERE ts_code = '{TARGET_CODE}'")
    
    # 插入的数据
    insert_df = df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                    'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']]
    
    client.insert_dataframe(
        'INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, turnover_rate) VALUES',
        insert_df
    )
    
    print(f"成功导入 {len(insert_df)} 条基准数据！")

if __name__ == "__main__":
    fetch_and_save_benchmark()