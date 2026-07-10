# -*- coding: utf-8 -*-
"""_data_utils: _dataset 子包公开接口的重导出门面。

调用方可 `from _data_utils import xxx` 或直接 `from engines._xgb._dataset import xxx`。
"""
from engines._xgb._dataset._load import (  # noqa: F401,F403
    load_table,
    infer_features,
    to_xy,
    to_indices,
    make_eval_pairs,
)
from engines._xgb._dataset._split import (  # noqa: F401,F403
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
    _validate_ratios,
    _split_by_strategy,
    _split_explicit,
    _split_by_time,
    _split_random,
    _split_train_val,
    _validate_splits,
    _adapt_query_for_dtypes,
    _safe_query,
)
from engines._xgb._dataset._impute import (  # noqa: F401,F403
    DNNImputer,
    NullAudit,
    ImputeReport,
    DEFAULT_INDICATOR_LOW,
    DEFAULT_INDICATOR_HIGH,
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
]
