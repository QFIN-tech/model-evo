# -*- coding: utf-8 -*-
"""feature_iv 单元测试: IV 单调性 + AUC 排序合理。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from feature_iv import compute_iv_for_feature, compute_iv_table  # noqa: E402


def _make_signal_noise(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # signal: 与 y 强相关
    signal = y + rng.normal(0, 0.5, size=n)
    # noise: 与 y 无关
    noise = rng.normal(0, 1, size=n)
    return pd.DataFrame({"signal": signal, "noise": noise, "y": y})


def test_iv_signal_beats_noise():
    df = _make_signal_noise()
    res_signal = compute_iv_for_feature(df["signal"], df["y"], n_bins=10)
    res_noise = compute_iv_for_feature(df["noise"], df["y"], n_bins=10)
    assert res_signal["iv"] > res_noise["iv"]
    assert res_signal["auc"] > 0.7
    assert 0.45 <= res_noise["auc"] <= 0.6


def test_iv_table_sorted_descending():
    df = _make_signal_noise()
    table = compute_iv_table(df, ["noise", "signal"], label_col="y", n_bins=10)
    # signal 应排在 noise 前
    assert list(table["feature"]) == ["signal", "noise"]


def test_iv_single_class_label_returns_nan():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1, 1, 1, 1]})
    res = compute_iv_for_feature(df["x"], df["y"], n_bins=5)
    assert np.isnan(res["iv"])
    assert np.isnan(res["auc"])
