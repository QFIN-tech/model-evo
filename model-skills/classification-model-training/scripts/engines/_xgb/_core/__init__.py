# -*- coding: utf-8 -*-
"""xgb 引擎核心能力 — fitter / metrics / drift 三件套。

通过 PYTHONPATH 暴露给 entry 使用，避免 sys.path.insert 反模式。
"""
from ._fit import XgbFitter
from ._metrics import BinMetrics, MetricsReport, _ks_via_roc
from ._drift import DriftProbe, SplitPair

__all__ = [
    "XgbFitter",
    "BinMetrics",
    "MetricsReport",
    "DriftProbe",
    "SplitPair",
    "_ks_via_roc",
]
