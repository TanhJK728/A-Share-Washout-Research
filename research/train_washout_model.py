# -*- coding: utf-8 -*-
import qlib
from qlib.data import D
from qlib.utils import init_instance_by_config
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Initialize Qlib
QLIB_DATA_DIR = str(Path("qlib_data/cn_data").resolve())
qlib.init(provider_uri=QLIB_DATA_DIR, region="cn")

# 定义"强力洗盘+爆发", 我们需要过去一个月的表现来判断是否在"洗盘"
fields = []
names = []

# A. 核心洗盘特征 (Price Action)
# 1. 振幅 (Amplitude): 洗盘通常伴随剧烈震荡
fields.append("($high - $low) / Ref($close, 1)")
names.append("amplitude")

# 2. 长下影线 (Lower Shadow): 主力试盘或支撑强度的标志
# 下影线长度 / 收盘价
fields.append("(If($open < $close, $open, $close) - $low) / $close")
names.append("lower_shadow_ratio")

# B. 量能与筹码 (Volume & Turnover)
# 1. 换手率 (Turnover)
fields.append("$turnover")
names.append("turnover")

# 2. 换手率变化: 今天换手率 / 过去5天均值
fields.append("$turnover / Mean($turnover, 5)")
names.append("turnover_ratio_5d")

# 3. 量能萎缩: 判断是否缩量洗盘 (True=1, False=0)
# 如果今天成交量小于过去20天均值, 可能是缩量洗盘
fields.append("$volume / Mean($volume, 20)")
names.append("vol_shrink_20d")

# C. 趋势与历史表现 (Past Month Behavior)
# 1. 过去20天涨跌幅: 也就是月度涨幅，判断股票是否处于活跃期
fields.append("$close / Ref($close, 20) - 1")
names.append("return_20d")

# 2. 波动率 (Volatility): 过去20天的标准差，寻找活跃股
fields.append("Std($close, 20) / Mean($close, 20)")
names.append("volatility_20d")

# 3. RSI (相对强弱指标): 简单的 RSI 近似，判断是否超卖
# (涨幅和 / 绝对涨幅和)
fields.append("Mean($close>$open, 14) / 14") 
names.append("rsi_sim_14")

# D. 目标 (Label)
# 预测：明天最高价相对于今天收盘价的涨幅 (捕捉盘中拉升)
label_expr = ["Ref($high, -1) / $close - 1"]
label_cols = ["label_max_ret"]

def get_data_handler():

    dh_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "start_time": "2020-01-01",
            "end_time": "2025-12-31",
            "instruments": "all",
            "infer_processors": [{"class": "Fillna", "kwargs": {"fields_group": "feature"}}],
            # 学习阶段丢弃 Label 为空的行
            "learn_processors": [{"class": "DropnaLabel"}],
            "data_loader": {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": (fields, names),
                        "label": (label_expr, label_cols),
                    },
                },
            },
        },
    }
    return init_instance_by_config(dh_config)

def train_and_predict():
    print("正在构建'游资洗盘'特征集")
    dh = get_data_handler()

    print("提取数据中...")
    df = dh.fetch(col_set=["feature", "label"])

    # 将多级列索引 (feature, amplitude) 展平为 (amplitude)
    df.columns = df.columns.droplevel(0)
    
    # 数据清洗：去极值或填充0   
    df = df.fillna(0)

    # 生成二分类标签: 明天最高涨幅 > 4% 视为"爆发" (1)，否则为 (0)
    # 对于游资票，我们想抓那个瞬间的冲高
    df["label_class"] = (df["label_max_ret"] > 0.04).astype(int)

    # 训练集：2020 to 2024.06
    # 测试集：2024.07 to Today
    split_date = pd.Timestamp("2024-07-01")
    
    # 这里的 reset_index 是为了方便后续根据日期筛选，但训练时我们需要纯数值
    train_df = df[df.index.get_level_values("datetime") < split_date]
    test_df = df[df.index.get_level_values("datetime") >= split_date]

    X_train = train_df.iloc[:, :-2] # 特征
    y_train = train_df["label_class"] # 标签

    X_test = test_df.iloc[:, :-2]
    y_test = test_df["label_class"]

    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    print(f"正样本(爆发)比例: {y_train.mean():.2%}")

    # 训练 LightGBM (GBDT 比 简单的深度学习在表格数据上往往更有效且快)
    print("开始训练模型...")
    clf = lgb.LGBMClassifier(
        n_estimators=500,  # 树的数量
        learning_rate=0.05,
        num_leaves=64, # 增加复杂度
        max_depth=7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="auc")

    # 评估
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\n 测试集 AUC: {auc:.4f} (大于 0.6 说明有一定预测能力)")

    # 特征重要性分析
    print("\n 模型认为最重要的洗盘指标:")
    imp_df = pd.DataFrame({"Feature": names, "Importance": clf.feature_importances_})
    print(imp_df.sort_values("Importance", ascending=False).head(10))

    return clf, df

def predict_next_day(model, df_all):
    """
    预测所有股票在数据集中'最后一天'的表现，给出明天的建议
    """
    print("\n 根据模型预测下一个交易日的爆发概率: ")
    
    # 1. 获取最新日期
    latest_date = df_all.index.get_level_values("datetime").max()
    print(f"基于历史数据日期: {latest_date.date()} (预测 T+1 日表现)")
    
    # 2. 提取当天数据
    latest_data = df_all[df_all.index.get_level_values("datetime") == latest_date].copy()
    
    # 3. 准备特征 (确保去掉 label 列)
    # df_all 最后两列是 label_max_ret 和 label_class，所以取 :-2
    X_latest = latest_data.iloc[:, :-2]
    
    # 4. 预测概率
    probs = model.predict_proba(X_latest)[:, 1]
    latest_data["prob_burst"] = probs
    
    # 5. 取概率最高的 Top 10
    top_picks = latest_data.sort_values(by="prob_burst", ascending=False).head(10)
    
    top_picks_display = top_picks.reset_index()
    
    print("\n[模型预测] 明日最可能拉升/爆发的 Top 10 股票：")
    print("-" * 80)
    
    print(f"{'股票代码':<12} {'爆发概率':<10} {'换手率':<10} {'振幅':<10} {'20日涨幅':<10}")
    print("-" * 80)
    for _, row in top_picks_display.iterrows():
        code = str(row['instrument'])
        print(f"{code:<12} {row['prob_burst']:.2%}      {row['turnover']:.4f}      {row['amplitude']:.4f}      {row['return_20d']:.4f}")
    
    print("-" * 80)
    print("策略建议：")
    print("1. 这些股票虽然模型预测爆发概率高，但必须结合明日开盘情况.")
    print("2. 如果明日[高开]且[量能配合], 则模型验证成功, 可跟进.")
    print("3. 如果明日[大幅低开], 说明主力洗盘未结束或出货, 需放弃.")


if __name__ == "__main__":
    trained_model, full_data = train_and_predict()

    predict_next_day(trained_model, full_data)
