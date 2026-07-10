# -*- coding: utf-8 -*-
"""lr 引擎核心能力 — metrics / drift 二件套。

metrics 与 drift 算法无关, 直接转发 _xgb._core; lr fitter 由 sklearn LogisticRegression 提供,
不放在 _core 里。
"""
from engines._xgb._core import (
    BinMetrics,
    MetricsReport,
    DriftProbe,
    SplitPair,
    _ks_via_roc,
)

__all__ = [
    "BinMetrics",
    "MetricsReport",
    "DriftProbe",
    "SplitPair",
    "_ks_via_roc",
]
