# -*- coding: utf-8 -*-
"""特征基础统计: 缺失率/unique/均值/分位数/min/max/dtype。

仅做描述性统计, 不做任何剔除决策。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def compute_basic_stats(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """计算每个特征的基础统计。

    Args:
        df: 全样本(用于统计, 不区分 train/oot)
        features: 待统计的特征列名清单

    Returns:
        DataFrame, 每行一个特征, 列为:
        feature/dtype/count/missing_rate/unique/mean/std/min/q25/median/q75/max
        非数值列的 mean/std/分位数/min/max 置为 NaN。
    """
    rows = []
    n = len(df)
    for f in features:
        if f not in df.columns:
            rows.append(
                {
                    "feature": f,
                    "dtype": "MISSING_COL",
                    "count": 0,
                    "missing_rate": 1.0,
                    "unique": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                }
            )
            continue
        s = df[f]
        non_null = s.dropna()
        cnt = int(non_null.shape[0])
        miss = float(1 - cnt / n) if n else 1.0
        uniq = int(non_null.nunique())
        row = {
            "feature": f,
            "dtype": str(s.dtype),
            "count": cnt,
            "missing_rate": round(miss, 6),
            "unique": uniq,
        }
        if pd.api.types.is_numeric_dtype(s) and cnt > 0:
            arr = non_null.to_numpy(dtype=float)
            q25, med, q75 = np.percentile(arr, [25, 50, 75])
            row.update(
                {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if cnt > 1 else 0.0,
                    "min": float(np.min(arr)),
                    "q25": float(q25),
                    "median": float(med),
                    "q75": float(q75),
                    "max": float(np.max(arr)),
                }
            )
        else:
            row.update(
                {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "q25": np.nan,
                    "median": np.nan,
                    "q75": np.nan,
                    "max": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
