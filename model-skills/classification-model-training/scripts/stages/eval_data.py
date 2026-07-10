# -*- coding: utf-8 -*-
"""统一组装 train/test/oot 三段评估数据 + 特征重要性的中间产物。

evaluation/ predictions/ explainability/ 三个 stage 均从本模块拿数据,
避免每个 stage 各自 split + predict 一遍。

数据切分由 run_build 按 model.split 内部完成, 本模块直接读 train/test/oot 三档 parquet,
test 段同时用于早停评估集(口径对齐)。

边界: PSI / base 对比 / 月度漂移 等评估报告能力归 classification-model-evaluation /
classification-model-comparison skill, 本模块不算 PSI。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 引擎自举(与 compare_base 一致): 注入 scripts/ 到 sys.path, 让 xgb 子包可 import
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from engines._xgb._core import BinMetrics, MetricsReport  # noqa: E402


@dataclass
class EvalData:
    """单次 run 的评估数据中间产物(无 I/O)。"""

    # 切分
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    oot_df: pd.DataFrame
    # 预测分(本模型)
    train_score: np.ndarray
    val_score: np.ndarray
    oot_score: np.ndarray
    # 三段 AUC/KS(从训练阶段传入或这里重算)
    metrics: Dict[str, BinMetrics]
    # 特征重要性
    importance: Dict[str, float] = field(default_factory=dict)
    # 元信息
    target: str = "label"
    features: List[str] = field(default_factory=list)


def _evaluate_three_split(
    predictor, target: str, features: List[str],
    train_df: pd.DataFrame, val_df: pd.DataFrame, oot_df: pd.DataFrame,
) -> Dict[str, BinMetrics]:
    """对三段分别算 AUC/KS/Gini, 一次 evaluate_splits 出三档。"""
    reporter = MetricsReport()
    splits = {
        "train": (train_df[target].to_numpy(), predictor.predict_proba(train_df[features])),
        "val": (val_df[target].to_numpy(), predictor.predict_proba(val_df[features])),
        "oot": (oot_df[target].to_numpy(), predictor.predict_proba(oot_df[features])),
    }
    return reporter.evaluate_splits(splits)


def assemble(
    predictor,
    train_path: str,
    test_path: str,
    oot_path: str,
    target: str,
    features: List[str],
    metrics: Optional[Dict[str, BinMetrics]] = None,
) -> EvalData:
    """读三档 parquet + 预测 + 提取特征重要性,产 EvalData。

    Args:
        predictor: 已训练预测器,需暴露 predict_proba(df[features])
        train_path/test_path/oot_path: run_build 内部按 model.split 切出的三档 parquet
            (test 当 val 段,口径与训练阶段一致)
        target/features: 与训练一致的列名
        metrics: 训练阶段已算过的三段指标(避免重复评估);None 则自动重算

    Returns:
        EvalData
    """
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(test_path)
    oot_df = pd.read_parquet(oot_path)
    train_score = predictor.predict_proba(train_df[features])
    val_score = predictor.predict_proba(val_df[features])
    oot_score = predictor.predict_proba(oot_df[features])

    if metrics is None:
        metrics = _evaluate_three_split(
            predictor, target, features, train_df, val_df, oot_df
        )

    importance = getattr(predictor, "get_feature_importance", lambda: {})() or {}

    return EvalData(
        train_df=train_df, val_df=val_df, oot_df=oot_df,
        train_score=train_score, val_score=val_score, oot_score=oot_score,
        metrics=metrics,
        importance=importance, target=target, features=list(features),
    )

