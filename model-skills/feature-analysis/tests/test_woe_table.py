# -*- coding: utf-8 -*-
"""build_woe_table 单元测试: long-format 表结构 + 桶完整性 + iv_bin 求和≈iv。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from feature_iv import build_woe_table, compute_iv_for_feature  # noqa: E402


def _make_signal_with_missing(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    signal = y + rng.normal(0, 0.5, size=n)
    # 造 5% 缺失, 验证 MISSING 桶落表
    mask = rng.uniform(size=n) < 0.05
    signal = signal.astype(float)
    signal[mask] = np.nan
    return pd.DataFrame({"signal": signal, "y": y})


def test_woe_table_columns():
    df = _make_signal_with_missing()
    table = build_woe_table(df, ["signal"], label_col="y", n_bins=10)
    assert list(table.columns) == [
        "feature", "bin", "cnt", "pos", "neg", "pos_rate", "woe", "iv_bin",
    ]


def test_woe_table_has_missing_bucket():
    df = _make_signal_with_missing()
    table = build_woe_table(df, ["signal"], label_col="y", n_bins=10)
    bins = set(table["bin"].astype(str))
    assert any("MISSING" in b for b in bins), f"缺 MISSING 桶, 实际 bins={bins}"


def test_pos_plus_neg_equals_cnt():
    df = _make_signal_with_missing()
    table = build_woe_table(df, ["signal"], label_col="y", n_bins=10)
    assert (table["pos"] + table["neg"] == table["cnt"]).all()


def test_iv_bin_sum_approx_iv():
    """每特征各桶 iv_bin 求和应≈ compute_iv_for_feature 返回的 iv(浮点容差)。"""
    df = _make_signal_with_missing()
    table = build_woe_table(df, ["signal"], label_col="y", n_bins=10)
    res = compute_iv_for_feature(df["signal"], df["y"], n_bins=10)
    iv_sum = table["iv_bin"].sum()
    assert abs(iv_sum - res["iv"]) < 1e-4, f"iv_bin 求和={iv_sum} 与 iv={res['iv']} 偏差过大"


def test_woe_table_skips_missing_feature():
    """特征不在 df 中时该特征不出行, 不报错。"""
    df = _make_signal_with_missing()
    table = build_woe_table(df, ["signal", "ghost_col"], label_col="y", n_bins=10)
    assert "ghost_col" not in set(table["feature"])
    assert "signal" in set(table["feature"])
