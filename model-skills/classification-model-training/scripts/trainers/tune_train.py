# -*- coding: utf-8 -*-
"""model-training 调参训练: 在 engines._xgb 引擎基础上注入超参重训。

engines._xgb 的入口用 XgbFitter() 无参(强正则默认参数)会欠拟合
(Train AUC <= OOT AUC)。本脚本复用引擎的 XgbFitter(params),注入加容量的超参。
数据切分由 run_build 按 model.split 内部完成, 本脚本直接读三档 parquet,
test 当 val 做 early stopping。
train_with_params 是给 run_build.py 编排器调用的纯函数,
本模块不提供 CLI(产物落盘由 run_build 统一负责)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from engines._xgb._core import XgbFitter, MetricsReport  # noqa: E402
from engines._xgb._data_utils import make_eval_pairs  # noqa: E402


# 调优超参: 针对欠拟合加容量(depth 4->6, min_child_weight 50->20,
# n_estimators 300->800 由 early stopping 兜底, lr 0.05->0.03)。
TUNED_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 5,
    "learning_rate": 0.02,
    "n_estimators": 1500,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,
    "reg_alpha": 0.5,
    "reg_lambda": 3.0,
    "random_state": 42,
    "verbosity": 0,
}


def train_with_params(
    train_path: str, test_path: str, oot_path: str,
    target: str, features: List[str], params: dict,
) -> tuple:
    """用指定超参训练 + 评估 Train/Val/OOT(三段式, test 当 val 做 early stopping)。

    Args:
        train_path: 训练段 parquet(已清洗)
        test_path: 测试段 parquet(已清洗),作为 early-stopping eval_set
        oot_path: OOT 段 parquet(已清洗)
        target: 标签列
        features: 入模特征
        params: XGBoost 超参字典

    Returns:
        (trainer, metrics) 其中 metrics 含 train/val/oot 的 auc/ks
    """
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(test_path)
    oot_df = pd.read_parquet(oot_path)
    print(f"[tune] 切分: train={len(train_df)} val={len(val_df)} oot={len(oot_df)}")

    X_train, y_train = train_df[features], train_df[target]
    # 防泄漏: early stopping 仅用 val(=test)
    eval_pairs = make_eval_pairs(
        train=train_df, val=val_df, features=features, target=target,
    )

    trainer = XgbFitter(params)
    trainer.train(X_train, y_train, eval_pairs[1][0], eval_pairs[1][1])

    reporter = MetricsReport()
    splits = {
        "train": (train_df[target].to_numpy(), trainer.predict_proba(train_df[features])),
        "val": (val_df[target].to_numpy(), trainer.predict_proba(val_df[features])),
        "oot": (oot_df[target].to_numpy(), trainer.predict_proba(oot_df[features])),
    }
    m = reporter.evaluate_splits(splits)
    print(f"[tune] best_iteration={getattr(trainer, 'best_iteration_', None)}")
    for k in ("train", "val", "oot"):
        print(f"[tune] {k}: AUC={m[k].auc:.4f} KS={m[k].ks:.4f}")
    return trainer, m
