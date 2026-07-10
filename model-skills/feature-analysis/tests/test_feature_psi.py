# -*- coding: utf-8 -*-
"""feature_psi 单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from feature_psi import compute_psi_for_feature, compute_psi_table  # noqa: E402


def test_psi_same_dist_near_zero():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, size=2000))
    b = pd.Series(rng.normal(0, 1, size=2000))
    psi = compute_psi_for_feature(a, b, n_bins=10)
    assert psi < 0.05  # 同分布近零


def test_psi_shifted_dist_large():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, size=2000))
    b = pd.Series(rng.normal(2, 1, size=2000))  # 均值偏移
    psi = compute_psi_for_feature(a, b, n_bins=10)
    assert psi > 0.5


def test_psi_table_warn_flag():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, size=2000),
            "drift": rng.normal(0, 1, size=2000),
        }
    )
    oot_df = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, size=2000),
            "drift": rng.normal(3, 1, size=2000),
        }
    )
    out = compute_psi_table(
        train_df, oot_df, ["stable", "drift"], n_bins=10, warn_threshold=0.1
    )
    stable_row = out[out["feature"] == "stable"].iloc[0]
    drift_row = out[out["feature"] == "drift"].iloc[0]
    assert stable_row["warn"] is False or stable_row["warn"] == False  # noqa
    assert drift_row["warn"] is True or drift_row["warn"] == True  # noqa
    assert drift_row["psi"] > stable_row["psi"]
