# -*- coding: utf-8 -*-
"""数据清洗工具: xgboost 不接受 object dtype 特征列,落临时 parquet。

供 model-training 与 model-tuning 共用 (后者从 baseline 拿到的 data_path 同样需要预清洗)。
"""
from __future__ import annotations

import os
import tempfile
from typing import List


def clean_object_features(data_path: str, features: List[str]) -> str:
    """若 features 中有 object dtype 列,转 float64 落到一份临时 parquet。

    Args:
        data_path: 原始 parquet 路径
        features: 入模特征列

    Returns:
        清洗后 parquet 路径 (无 object 列时返回原 data_path)
    """
    import pandas as pd
    raw = pd.read_parquet(data_path)
    obj_feats = [c for c in features if c in raw.columns and raw[c].dtype == object]
    if not obj_feats:
        return data_path
    print(f"[data_clean] 清洗 object 特征列 -> float64: {obj_feats}")
    for c in obj_feats:
        raw[c] = raw[c].astype(float)
    tmp = tempfile.mkdtemp(prefix="model-training_clean_")
    out = os.path.join(tmp, "sample_clean.parquet")
    raw.to_parquet(out, index=False)
    print(f"[data_clean] 清洗后数据: {out}")
    return out
