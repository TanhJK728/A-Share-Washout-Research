import akshare as ak
import pandas as pd
from datetime import datetime
from clickhouse_driver import Client

# 连接 ClickHouse
print("正在连接 ClickHouse...")
client = Client(
    host='localhost', 
    user='default', 
    password='', 
    database='stock_data',
    settings={'use_numpy': True} 
)

def get_realtime_daily_data():
    print("正在通过 AKShare 从东方财富抓取全市场实时行情...")
    try:
        # 这个接口返回的列包含：代码,名称,最新价,涨跌幅,涨跌额,成交量,成交额,振幅,最高,最低,今开,昨收,量比,换手率,市盈率-动态,市净率...
        df = ak.stock_zh_a_spot_em()
    except Exception as e:
        print(f"网络请求失败: {e}")
        return None

    if df is None or df.empty:
        print("未获取到数据")
        return None

    # 时间处理
    today = datetime.now().date()
    
    # 检查重复
    check_sql = f"SELECT count() FROM stock_daily WHERE trade_date = '{today}'"
    try:
        count = client.execute(check_sql)[0][0]
        if count > 0:
            print(f"今日 ({today}) 的行情数据已经存在 ({count} 条)! 跳过入库, 防止重复.")
            return None 
    except Exception as e:
        print(f"检查重复失败: {e}")

    print(f"抓取成功! 原始数据 {len(df)} 行")
    
    # 映射
    rename_dict = {
        '代码': 'ts_code',
        '最新价': 'close',
        '今开': 'open',
        '最高': 'high',
        '最低': 'low',
        '昨收': 'pre_close',
        '涨跌额': 'change',
        '涨跌幅': 'pct_chg',
        '成交量': 'vol',
        '成交额': 'amount',
        '换手率': 'turnover_rate'
    }
    
    # 过滤掉不在 rename_dict 里的列，防止报错
    available_cols = set(df.columns)
    rename_keys = set(rename_dict.keys())
    
    cols_to_use = list(rename_keys.intersection(available_cols)) # 有时候 AKShare 返回的列会变, 做个交集处理更安全, 确保我们只取存在的列.
    
    df = df[cols_to_use].rename(columns=rename_dict)
    
    df['trade_date'] = today
    df['ts_code'] = df['ts_code'].astype(str)
    
    # 数值转换列表
    numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']
    
    for col in numeric_cols:
        # AKShare 的换手率是百分比(3.5代表3.5%)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 写入数据库的列顺序
    columns_to_db = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 'turnover_rate']
    
    # 按照指定顺序排列
    df_final = df[columns_to_db].copy()

    return df_final

def save_to_clickhouse(df):
    if df is None or df.empty:
        return

    print(f"正在写入 {len(df)} 条数据到 ClickHouse...")
    try:
        # Insert
        client.insert_dataframe(
            'INSERT INTO stock_daily (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount, turnover_rate) VALUES',
            df
        )
        print("入库成功!")
    except Exception as e:
        print(f"入库失败: {e}")

if __name__ == "__main__":
    data = get_realtime_daily_data()
    if data is not None:
        save_to_clickhouse(data)
        
        # 验证
        try:
            count = client.execute("SELECT count() FROM stock_daily")[0][0]
            print(f"数据库当前总行数: {count}")

        except Exception as e:
            print(f"查询失败: {e}")