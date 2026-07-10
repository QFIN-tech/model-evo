# -*- coding: utf-8 -*-
"""特征稳定性: 以训练段等频分箱边界为基准, 比较训练 vs OOT 段分布。

PSI = sum( (p_oot - p_train) * ln(p_oot / p_train) ), 0 频数桶按 1e-6 平滑。
> warn_threshold 在报告中标 [PSI_WARN], 不做剔除。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


_EPS = 1e-6


def _train_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    """以训练段的非空数值算等频边界(去重后可能少于 n_bins+1)。"""
    arr = series.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(arr, qs))
    if edges.size < 2:
        return np.array([])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bucket_dist(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    """把 series 按 edges 切桶, 缺失算一桶, 返回各桶占比(含 MISSING 桶在末)。"""
    n = len(series)
    if n == 0 or edges.size < 2:
        return np.array([])
    non_null = series.dropna()
    if non_null.size == 0:
        cuts = np.zeros(edges.size - 1)
    else:
        cats = pd.cut(non_null, bins=edges, include_lowest=True)
        cuts = cats.value_counts(sort=False).to_numpy()
    miss = n - int(cuts.sum())
    dist = np.concatenate([cuts, [miss]])
    return dist / n


def _categorical_psi(train_series: pd.Series, oot_series: pd.Series) -> float:
    """非数值列 PSI: 以 train 的 unique 值集合为基准 bucket, oot 未见类别归 "OTHER", NaN 单独一桶。

    PSI = sum( (p_oot - p_train) * ln(p_oot / p_train) ), 0 频按 _EPS 平滑。
    """
    n_train = len(train_series)
    n_oot = len(oot_series)
    if n_train == 0 or n_oot == 0:
        return float("nan")

    train_vc = train_series.value_counts(dropna=False)
    oot_vc = oot_series.value_counts(dropna=False)
    train_categories = set(train_vc.index)

    # train 见过的类别(含 NaN) + oot 里未见类别归 OTHER + train 里没有的 NaN 也归 NaN(已在 value_counts dropna=False)
    p_train: list = []
    p_oot: list = []
    for cat in train_categories:
        p_train.append(train_vc.get(cat, 0) / n_train)
        p_oot.append(oot_vc.get(cat, 0) / n_oot)
    # oot 里 train 未见类别合并为 OTHER 桶(p_train=0, p_oot=该部分占比)
    oot_unseen_mass = sum(oot_vc.get(c, 0) for c in oot_vc.index if c not in train_categories)
    if oot_unseen_mass > 0:
        p_train.append(0.0)
        p_oot.append(oot_unseen_mass / n_oot)

    pt = np.array(p_train, dtype=float)
    po = np.array(p_oot, dtype=float)
    pt = np.where(pt == 0, _EPS, pt)
    po = np.where(po == 0, _EPS, po)
    psi = float(np.sum((po - pt) * np.log(po / pt)))
    return round(psi, 6)


def compute_psi_for_feature(
    train_series: pd.Series, oot_series: pd.Series, n_bins: int = 10
) -> float:
    """以 train 的等频边界比较 train vs oot 的桶分布, 返回 PSI。

    数值列: 等频分箱 + 缺失独立桶, PSI 公式同上。
    非数值列: 委托 _categorical_psi 做类别分布对比, 不退化为缺失率差。
    """
    if not pd.api.types.is_numeric_dtype(train_series):
        return _categorical_psi(train_series, oot_series)
    edges = _train_edges(train_series, n_bins)
    if edges.size < 2:
        # 数值列但样本不足以算分箱边界(如全 NaN 或全相同) -> 退化为类别分布对比
        return _categorical_psi(train_series, oot_series)
    p_train = _bucket_dist(train_series, edges)
    p_oot = _bucket_dist(oot_series, edges)
    p_train = np.where(p_train == 0, _EPS, p_train)
    p_oot = np.where(p_oot == 0, _EPS, p_oot)
    psi = float(np.sum((p_oot - p_train) * np.log(p_oot / p_train)))
    return round(psi, 6)


def compute_psi_table(
    train_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    features: List[str],
    n_bins: int = 10,
    warn_threshold: float = 0.10,
) -> pd.DataFrame:
    """批量计算特征 PSI, 按 PSI 降序。

    Args:
        train_df: 训练段样本
        oot_df: OOT 段样本
        features: 特征清单
        n_bins: 训练段等频分箱数
        warn_threshold: 超此阈值则 warn=True

    Returns:
        DataFrame, 列: feature/psi/warn
    """
    rows = []
    for f in features:
        if f not in train_df.columns or f not in oot_df.columns:
            rows.append({"feature": f, "psi": np.nan, "warn": False})
            continue
        psi = compute_psi_for_feature(train_df[f], oot_df[f], n_bins=n_bins)
        rows.append({"feature": f, "psi": psi, "warn": bool(psi > warn_threshold)})
    out = pd.DataFrame(rows)
    return out.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
