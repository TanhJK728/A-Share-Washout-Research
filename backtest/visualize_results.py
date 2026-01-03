import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from pathlib import Path

# Config
MLRUNS_DIR = Path("mlruns")

def find_latest_report():
    """自动查找 mlruns 中最新生成的资金曲线报告"""
    latest_time = 0
    latest_file = None
    
    for root, dirs, files in os.walk(MLRUNS_DIR):
        for file in files:
            if file == "report_normal_1day.pkl":
                full_path = os.path.join(root, file)
                file_time = os.path.getmtime(full_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = full_path
    
    return latest_file

def calculate_metrics(series_returns, series_cum):
    """通用指标计算函数"""
    # 1. 累计收益
    total_ret = series_cum.iloc[-1] - 1
    
    # 2. 年化收益 (Assuming 252 trading days)
    days = len(series_returns)
    if days > 0:
        ann_ret = (1 + total_ret) ** (252 / days) - 1
    else:
        ann_ret = 0
        
    # 3. 夏普比率
    daily_mean = series_returns.mean()
    daily_std = series_returns.std()
    if daily_std != 0:
        sharpe = (daily_mean / daily_std) * np.sqrt(252)
    else:
        sharpe = 0
        
    # 4. 最大回撤
    max_nav = series_cum.cummax()
    drawdown = (series_cum - max_nav) / max_nav
    max_dd = drawdown.min()
    
    return total_ret, ann_ret, sharpe, max_dd

def plot_performance(pkl_path):
    print(f"正在读取回测报告: {pkl_path}")
    
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)
    
    # 计算净值曲线
    # 策略净值 (从1.0开始)
    df["strategy_cum"] = (df["return"] + 1).cumprod()
    # 基准净值 (从1.0开始)
    df["bench_cum"] = (df["bench"] + 1).cumprod()
    
    # 绘图
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["strategy_cum"], label="Strategy (Washout)", color='#d62728', linewidth=2)
    plt.plot(df.index, df["bench_cum"], label="Benchmark (CSI300)", color='#7f7f7f', linestyle='--', alpha=0.8)
    
    plt.title("Strategy vs Benchmark Equity Curve", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (1.0 = Start)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 保存图片
    save_dir = Path(__file__).parent
    output_img = save_dir / "strategy_comparison.png"
    plt.savefig(output_img)
    print(f"\n 对比图已保存至: {output_img}")

    # 计算双方指标
    # 策略指标
    strat_tot, strat_ann, strat_shp, strat_mdd = calculate_metrics(df["return"], df["strategy_cum"])
    # 基准指标
    bench_tot, bench_ann, bench_shp, bench_mdd = calculate_metrics(df["bench"], df["bench_cum"])

    print("\n" + "="*65)
    print(f"{'绩效对比 (Performance Comparison)':^65}")
    print("="*65)
    print(f"{'指标 (Metric)':<25} | {'你的策略 (Strategy)':<18} | {'沪深300 (Benchmark)':<18}")
    print("-" * 65)
    print(f"{'累计收益 (Total Return)':<25} | {strat_tot*100:>14.2f}%   | {bench_tot*100:>14.2f}%")
    print(f"{'年化收益 (Annualized)':<25} | {strat_ann*100:>14.2f}%   | {bench_ann*100:>14.2f}%")
    print(f"{'夏普比率 (Sharpe Ratio)':<25} | {strat_shp:>14.4f}    | {bench_shp:>14.4f}")
    print(f"{'最大回撤 (Max Drawdown)':<25} | {strat_mdd*100:>14.2f}%   | {bench_mdd*100:>14.2f}%")
    print("-" * 65)

if __name__ == "__main__":
    report_path = find_latest_report()
    if report_path:
        plot_performance(report_path)
    else:
        print("未找到回测报告文件！")