# -*- coding: utf-8 -*-
"""feature_stats 单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from feature_stats import compute_basic_stats  # noqa: E402


def test_basic_stats_numeric_and_missing():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, None],
            "b": ["x", "y", "x", None, "z"],
        }
    )
    out = compute_basic_stats(df, ["a", "b"])
    assert set(out["feature"]) == {"a", "b"}
    a_row = out[out["feature"] == "a"].iloc[0]
    assert a_row["count"] == 4
    assert a_row["missing_rate"] == pytest.approx(0.2, abs=1e-6)
    assert a_row["mean"] == pytest.approx(2.5, abs=1e-6)
    assert a_row["min"] == 1.0
    assert a_row["max"] == 4.0
    b_row = out[out["feature"] == "b"].iloc[0]
    assert b_row["count"] == 4
    assert np.isnan(b_row["mean"])


def test_basic_stats_missing_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = compute_basic_stats(df, ["a", "ghost"])
    ghost = out[out["feature"] == "ghost"].iloc[0]
    assert ghost["dtype"] == "MISSING_COL"
    assert ghost["missing_rate"] == 1.0
    assert ghost["count"] == 0
