# -*- coding: utf-8 -*-
"""XGB 训练/预测工具: baseline 与 G6 融合模型共用, 超参只有一处真相。

默认超参为固定值
(binary:logistic / lr=0.1 / max_depth=3 / n_estimators=50 / subsample=0.8 /
colsample_bytree=0.8 / reg_alpha=0.1 / reg_lambda=1.0 / seed=42)。
进化比的是「特征带来的增益」, 超参固定才能保证轮间可比——调参不是本环节职责。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 50,
    "random_state": 42,
}


def train_xgb(X: pd.DataFrame, y, params: dict | None = None) -> xgb.XGBClassifier:
    """固定超参训练 XGB 二分类模型。

    Args:
        X: 特征矩阵(数值列; NaN 由 XGB 原生处理)
        y: 0/1 标签
        params: 缺省用 DEFAULT_XGB_PARAMS; 传入时整体替换
    """
    p = dict(DEFAULT_XGB_PARAMS if params is None else params)
    model = xgb.XGBClassifier(**p)
    model.fit(X, y)
    return model


def predict_scores(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """正类概率打分。"""
    return model.predict_proba(X)[:, 1]


def select_numeric_features(df: pd.DataFrame, candidate_cols: list) -> list:
    """从候选列中筛出可数值化的列(to_numeric 不报错), 保持输入顺序。

    数值列筛选思路:
    文本列不进 baseline/融合模型(语义信息走 semantic-feature 的 semantic score 通道)。
    """
    numeric = []
    for c in candidate_cols:
        if c not in df.columns:
            continue
        try:
            pd.to_numeric(df[c].head(100), errors="raise")
            numeric.append(c)
        except (ValueError, TypeError):
            continue
    return numeric


def as_numeric_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """把选定列统一转 float(无法转换的置 NaN), 供 XGB 消费。"""
    return df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
