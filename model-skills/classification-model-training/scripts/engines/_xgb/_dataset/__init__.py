# -*- coding: utf-8 -*-
"""_dataset — 数据加载与切分子包。

按关注点从 _data_utils.py 分桶：
  - _load: 数据读取、特征推断、X/y 转换
  - _split: 经典三段式切分（train/val/oot）
  - _impute: 缺失值处理
  - _woe: WoE 编码
"""
from ._load import (
    load_table,
    infer_features,
    to_xy,
    to_indices,
    make_eval_pairs,
)
from ._split import (
    DatasetSplits,
    prepare_splits,
    format_split_md,
    SplitReport,
    SplitRatios,
    SplitCounts,
    SplitPosRates,
    DEFAULT_OOT_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_RANDOM_SEED,
)
from ._impute import (
    DNNImputer,
    NullAudit,
    ImputeReport,
    DEFAULT_INDICATOR_LOW,
    DEFAULT_INDICATOR_HIGH,
)
from ._woe import (
    WoeBinner,
    WoEFeatureMap,
    WoEReport,
)

__all__ = [
    "load_table",
    "infer_features",
    "to_xy",
    "to_indices",
    "make_eval_pairs",
    "DatasetSplits",
    "prepare_splits",
    "format_split_md",
    "SplitReport",
    "SplitRatios",
    "SplitCounts",
    "SplitPosRates",
    "DEFAULT_OOT_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_RANDOM_SEED",
    "DNNImputer",
    "NullAudit",
    "ImputeReport",
    "DEFAULT_INDICATOR_LOW",
    "DEFAULT_INDICATOR_HIGH",
    "WoeBinner",
    "WoEFeatureMap",
    "WoEReport",
]
