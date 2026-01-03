import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
import sys
from pathlib import Path
import pandas as pd

# Initialize Qlib
provider_uri = str(Path("qlib_data/cn_data").resolve())
qlib.init(provider_uri=provider_uri, region=REG_CN)
print(f"Qlib 初始化完成, 数据源: {provider_uri}")

# 定义特征集 
fields = []
names = []

# A. 价格形态
fields.append("($high - $low) / Ref($close, 1)")
names.append("amplitude")
fields.append("(If($open < $close, $open, $close) - $low) / $close")
names.append("lower_shadow_ratio")

# B. 量能
fields.append("$turnover") 
names.append("turnover")
fields.append("$turnover / Mean($turnover, 5)")
names.append("turnover_ratio_5d")
fields.append("$volume / Mean($volume, 20)")
names.append("vol_shrink_20d")

# C. 趋势
fields.append("$close / Ref($close, 20) - 1")
names.append("return_20d")
fields.append("Std($close, 20) / Mean($close, 20)")
names.append("volatility_20d")
fields.append("Mean($close>$open, 14) / 14") 
names.append("rsi_sim_14")

# D. Label (预测目标: T+1 收盘涨幅)
label_fields = ["Ref($close, -1) / $close - 1"]
label_names = ["label"]

# Config
market = "all"
benchmark = "SH000300"

# 时间段配置
TRAIN_START = "2021-01-01"
TRAIN_END   = "2024-06-30"
TEST_START  = "2024-07-01"
TEST_END    = "2025-12-30"

conf = {
    "task": {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.05,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "DataHandlerLP",
                    "module_path": "qlib.data.dataset.handler",
                    "kwargs": {
                        "instruments": market,
                        "start_time": TRAIN_START,
                        "end_time": TEST_END,
                        "data_loader": {
                            "class": "QlibDataLoader",
                            "kwargs": {
                                "config": {
                                    "feature": (fields, names),
                                    "label": (label_fields, label_names),
                                },
                            },
                        },
                        # 用 CSRankNorm 把横截面排名标准化, 这将把所有特征转换为 0~1 之间的排名, 解决数值爆炸和异常值问题
                        "infer_processors": [
                             {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'feature'}},
                             {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}
                        ],
                        # 训练时对 Label 也做排名处理, 可以让模型更关注相对强弱, 而不是绝对涨幅
                        "learn_processors": [
                            {'class': 'DropnaLabel'},
                            {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}
                        ],
                    },
                },
                "segments": {
                    "train": (TRAIN_START, TRAIN_END),
                    "valid": (TEST_START, TEST_END),
                    "test":  (TEST_START, TEST_END),
                },
            },
        },
    },
    "record": [
        {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {
                "model": "<MODEL>",
                "dataset": "<DATASET>",
            },
        },
        {
            "class": "PortAnaRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {
                "config": {
                    "strategy": {
                        "class": "TopkDropoutStrategy",
                        "module_path": "qlib.contrib.strategy",
                        "kwargs": {
                            "signal": "<PRED>",
                            "topk": 5,
                            "n_drop": 5,
                            "hold_thresh": 1,
                        },
                    },
                    "backtest": {
                        "start_time": TEST_START,
                        "end_time": TEST_END,
                        "account": 1000000,
                        "benchmark": benchmark,
                        "exchange_kwargs": {
                            "limit_threshold": 0.095,
                            "deal_price": "close",
                        },
                    },
                },
            },
        },
    ],
}

if __name__ == "__main__":
    # 实验管理
    with R.start(experiment_name="washout_strategy_rank"):
        print("1. 构建数据集 & 训练模型...")
        model = init_instance_by_config(conf["task"]["model"])
        dataset = init_instance_by_config(conf["task"]["dataset"])
        model.fit(dataset)

        print("2. 生成预测结果...")
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        print("3. 执行回测...")
        par = PortAnaRecord(recorder, conf["record"][1]["kwargs"]["config"])
        par.generate()

        print(f"\n 回测完成！结果已保存.")
        
        try:
            # 加载回测指标文件 (DataFrame)
            # load_object 返回的是一个字典, key是频率 '1day'
            report_dict = recorder.load_object("portfolio_analysis/indicators_normal_1day_obj.pkl")
            
            # 获取 DataFrame
            indicators_df = report_dict['1day']
            
            print("\n====== 最终回测绩效 ======")

            cols_to_print = ['annualized_return', 'max_drawdown', 'sharpe', 'information_ratio']
            print(indicators_df[cols_to_print])
            
        except Exception as e:
            print(f"打印指标详情失败 (不影响文件生成): {e}")
            print("请直接去 mlruns 文件夹下查看 report_normal_1day.pkl报告")
