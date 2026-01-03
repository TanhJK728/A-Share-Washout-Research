import akshare as ak
import pandas as pd
import time
import random
from datetime import datetime
from clickhouse_driver import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Config
START_DATE = "20200101"
END_DATE = "20251231"
DB_HOST = 'localhost'
DB_DATABASE = 'stock_data'
MAX_WORKERS = 4  # 并发数量

# 连接 ClickHouse
client = Client(host=DB_HOST, database=DB_DATABASE, settings={'use_numpy': True})

def get_all_stock_codes():
    print("正在获取全市场股票列表...")
    try:
        df = ak.stock_zh_a_spot_em()
        return df['代码'].tolist()
    except Exception as e:
        print(f"获取列表失败: {e}")
        return []

def process_stock(code):
    """
    单个股票的处理逻辑（下载 -> 清洗 -> 入库）
    注意：每个线程需要独立的 ClickHouse 连接，或者短连接，但为了简单高效，我们在主线程统一写入。
    """
    # 每个线程内部建立连接，防止连接冲突
    local_client = Client(host=DB_HOST, database=DB_DATABASE, settings={'use_numpy': True})
    
    try:
        # 1. 下载
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
        if df is None or df.empty:
            return False

        # 2. 清洗
        rename_dict = {
            '日期': 'trade_date', '开盘': 'open', '最高': 'high', '最低': 'low', 
            '收盘': 'close', '成交量': 'vol', '成交额': 'amount', 
            '涨跌幅': 'pct_chg', '涨跌额': 'change', '换手率': 'turnover_rate'
        }
        df = df.rename(columns=rename_dict)
        df['ts_code'] = str(code)
        df['pre_close'] = df['close'].shift(1).fillna(df['open'])
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
        
        # 补全cols
        required_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 3. 排序与筛选
        final_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']
        df_save = df[final_cols].copy()

        # 4. Insert
        local_client.insert_dataframe(
            'INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, turnover_rate) VALUES',
            df_save
        )
        # 关闭连接
        local_client.disconnect()
        return True

    except Exception as e:
        return False

if __name__ == "__main__":
    # 1. 清空旧表
    print("正在清空旧数据(Truncate)...")
    client.execute("TRUNCATE TABLE stock_daily")
    print("旧数据已清空, 准备重新插入数据.")

    # 2. 获取列表
    all_codes = get_all_stock_codes()
    print(f"启动多线程下载, 线程数:{MAX_WORKERS}")
    
    # 3. 多线程执行
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_code = {executor.submit(process_stock, code): code for code in all_codes}
        
        # 进度条
        pbar = tqdm(total=len(all_codes))
        
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                pass
            
            pbar.update(1)
            pbar.set_description(f"Processing")
            
            # 随机休眠一点点
            if success_count % 10 == 0:
                time.sleep(random.uniform(0.1, 0.5))

    print(f"\n回填完成! 共入库 {success_count} 只股票.")
