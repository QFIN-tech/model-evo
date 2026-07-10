# -*- coding: utf-8 -*-
"""boundary_filter 单测: 4 条规则 + 编排器在阈值/缺数据/空表下的边界行为。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from boundary_filter import (
    BoundaryFilterResult,
    apply_constant_rule,
    apply_leakage_rule,
    apply_id_like_rule,
    apply_all_missing_rule,
    filter_boundary_features,
    DEFAULT_IV_MAX,
    DEFAULT_CONST_UNIQUE_MAX,
    DEFAULT_ID_LIKE_RATIO,
    DEFAULT_MISSING_MAX,
)


# -------------------- apply_constant_rule --------------------

def test_constant_drops_unique_le_one():
    stats_df = pd.DataFrame({"feature": ["a", "b", "c"], "unique": [1, 5, 0]})
    out = apply_constant_rule(["a", "b", "c"], stats_df, unique_max=1)
    assert out == {"a", "c"}


def test_constant_drops_std_zero():
    stats_df = pd.DataFrame({"feature": ["a", "b"], "std": [0.0, 1.5]})
    out = apply_constant_rule(["a", "b"], stats_df, unique_max=1)
    assert out == {"a"}


def test_constant_keeps_when_clean():
    stats_df = pd.DataFrame({"feature": ["a", "b"], "unique": [10, 20], "std": [1.0, 2.0]})
    out = apply_constant_rule(["a", "b"], stats_df, unique_max=1)
    assert out == set()


def test_constant_empty_table_no_drop():
    out = apply_constant_rule(["a", "b"], pd.DataFrame(), unique_max=1)
    assert out == set()


# -------------------- apply_leakage_rule --------------------

def test_leakage_drops_iv_above_max():
    iv_df = pd.DataFrame({"feature": ["a", "b", "c"], "iv": [0.5, 1.5, 2.0]})
    out = apply_leakage_rule(["a", "b", "c"], iv_df, iv_max=1.0)
    assert out == {"b", "c"}


def test_leakage_nan_not_dropped():
    """IV NaN 不删。"""
    iv_df = pd.DataFrame({"feature": ["a", "b"], "iv": [float("nan"), 1.5]})
    out = apply_leakage_rule(["a", "b"], iv_df, iv_max=1.0)
    assert out == {"b"}


def test_leakage_empty_table_no_drop():
    out = apply_leakage_rule(["a", "b"], pd.DataFrame(), iv_max=1.0)
    assert out == set()


def test_leakage_ignores_features_not_in_baseline():
    iv_df = pd.DataFrame({"feature": ["a", "z_unused"], "iv": [1.5, 2.0]})
    out = apply_leakage_rule(["a"], iv_df, iv_max=1.0)
    assert out == {"a"}


# -------------------- apply_id_like_rule --------------------

def test_id_like_drops_high_ratio():
    stats_df = pd.DataFrame({"feature": ["a", "b", "c"], "unique": [950, 100, 500]})
    out = apply_id_like_rule(["a", "b", "c"], stats_df, sample_total=1000, ratio=0.9)
    assert out == {"a"}


def test_id_like_skips_zero_sample_total():
    """sample_total <= 0 时跳过(无法计算比率)。"""
    stats_df = pd.DataFrame({"feature": ["a"], "unique": [999]})
    out = apply_id_like_rule(["a"], stats_df, sample_total=0, ratio=0.9)
    assert out == set()


def test_id_like_empty_table_no_drop():
    out = apply_id_like_rule(["a", "b"], pd.DataFrame(), sample_total=1000, ratio=0.9)
    assert out == set()


# -------------------- apply_all_missing_rule --------------------

def test_all_missing_drops_rate_one():
    stats_df = pd.DataFrame({"feature": ["a", "b", "c"], "missing_rate": [1.0, 0.5, 0.0]})
    out = apply_all_missing_rule(["a", "b", "c"], stats_df, missing_max=1.0)
    assert out == {"a"}


def test_all_missing_keeps_high_but_not_full():
    """missing_rate=0.94 不删。"""
    stats_df = pd.DataFrame({"feature": ["a", "b"], "missing_rate": [0.94, 0.5]})
    out = apply_all_missing_rule(["a", "b"], stats_df, missing_max=1.0)
    assert out == set()


def test_all_missing_empty_table_no_drop():
    out = apply_all_missing_rule(["a", "b"], pd.DataFrame(), missing_max=1.0)
    assert out == set()


# -------------------- filter_boundary_features() 编排 --------------------

@pytest.fixture
def analysis_csv_dir(tmp_path):
    """造 stats.csv + feature-quality.csv 模拟 feature-analysis 输出。

    - f0: unique=1 命中 constant
    - f1: iv=1.5 命中 leakage
    - f2: unique=950 / sample_total=1000 命中 id_like
    - f3: missing_rate=1.0 命中 all_missing
    - f4: 全合格保留
    """
    stats = pd.DataFrame({
        "feature": ["f0", "f1", "f2", "f3", "f4"],
        "unique": [1, 100, 950, 50, 80],
        "std": [0.0, 1.0, 2.0, 0.5, 1.5],
        "missing_rate": [0.1, 0.2, 0.3, 1.0, 0.05],
    })
    fq = pd.DataFrame({
        "feature": ["f0", "f1", "f2", "f3", "f4"],
        "iv": [0.3, 1.5, 0.5, 0.4, 0.6],
        "auc": [0.6, 0.99, 0.65, 0.62, 0.7],
    })
    stats.to_csv(tmp_path / "stats.csv", index=False)
    fq.to_csv(tmp_path / "feature-quality.csv", index=False)
    return tmp_path


def test_filter_all_rules_on(analysis_csv_dir):
    res = filter_boundary_features(
        ["f0", "f1", "f2", "f3", "f4"],
        str(analysis_csv_dir),
        sample_total=1000,
    )
    assert isinstance(res, BoundaryFilterResult)
    assert res.kept_features == ["f4"]
    assert set(res.dropped_features) == {"f0", "f1", "f2", "f3"}
    assert res.dropped_by_rule["constant"] == ["f0"]
    assert res.dropped_by_rule["leakage"] == ["f1"]
    assert res.dropped_by_rule["id_like"] == ["f2"]
    assert res.dropped_by_rule["all_missing"] == ["f3"]
    assert res.n_before == 5
    assert res.sample_total == 1000


def test_filter_preserves_baseline_order(analysis_csv_dir):
    res = filter_boundary_features(
        ["f4", "f3", "f2", "f1", "f0"],
        str(analysis_csv_dir),
        sample_total=1000,
    )
    assert res.kept_features == ["f4"]
    # dropped 按 baseline 顺序
    assert res.dropped_features == ["f3", "f2", "f1", "f0"]


def test_filter_disable_constant(analysis_csv_dir):
    """关 constant 规则后 f0 保留。"""
    res = filter_boundary_features(
        ["f0", "f1", "f2", "f3", "f4"],
        str(analysis_csv_dir),
        sample_total=1000,
        enable_constant=False,
    )
    assert "f0" in res.kept_features
    assert res.dropped_by_rule.get("constant") is None
    assert res.rules_enabled["constant"] is False


def test_filter_disable_leakage(analysis_csv_dir):
    """关 leakage 规则后 f1 保留。"""
    res = filter_boundary_features(
        ["f0", "f1", "f2", "f3", "f4"],
        str(analysis_csv_dir),
        sample_total=1000,
        enable_leakage=False,
    )
    assert "f1" in res.kept_features
    assert res.rules_enabled["leakage"] is False


def test_filter_custom_iv_max(analysis_csv_dir):
    """iv_max=2.0 后 f1(iv=1.5) 保留。"""
    res = filter_boundary_features(
        ["f0", "f1", "f2", "f3", "f4"],
        str(analysis_csv_dir),
        sample_total=1000,
        iv_max=2.0,
    )
    assert "f1" in res.kept_features


def test_filter_fallback_to_iv_table(tmp_path):
    """feature-quality.csv 缺失时回退 iv_table.csv。"""
    stats = pd.DataFrame({
        "feature": ["a", "b"],
        "unique": [10, 100],
        "std": [1.0, 2.0],
        "missing_rate": [0.1, 0.2],
    })
    iv_table = pd.DataFrame({
        "feature": ["a", "b"],
        "iv": [1.5, 0.5],   # a 命中 leakage
    })
    stats.to_csv(tmp_path / "stats.csv", index=False)
    iv_table.to_csv(tmp_path / "iv_table.csv", index=False)
    # 没有 feature-quality.csv, 应回退到 iv_table.csv
    res = filter_boundary_features(["a", "b"], str(tmp_path), sample_total=1000)
    assert res.dropped_by_rule["leakage"] == ["a"]


def test_filter_missing_csv_warns_and_skips(tmp_path):
    """analysis_dir 里 csv 全无 → 所有规则跳过,特征全保留。"""
    res = filter_boundary_features(["a", "b"], str(tmp_path), sample_total=1000)
    assert res.kept_features == ["a", "b"]
    assert res.dropped_features == []
    # 规则都跑了但都没剔除
    for rule in ("constant", "leakage", "id_like", "all_missing"):
        assert res.dropped_by_rule[rule] == []


def test_filter_id_like_skips_when_sample_total_zero(analysis_csv_dir):
    """sample_total=0 时 id_like 规则跳过,f2 保留。"""
    res = filter_boundary_features(
        ["f0", "f1", "f2", "f3", "f4"],
        str(analysis_csv_dir),
        sample_total=0,
    )
    assert "f2" in res.kept_features
    # 其他规则照常
    assert "f0" in res.dropped_features  # constant
    assert "f1" in res.dropped_features  # leakage
    assert "f3" in res.dropped_features  # all_missing


def test_result_as_dict_serializable(analysis_csv_dir):
    """as_dict 返回纯 list/dict 可 json 序列化。"""
    import json
    res = filter_boundary_features(
        ["f0", "f1", "f2", "f3", "f4"],
        str(analysis_csv_dir),
        sample_total=1000,
    )
    out = res.as_dict()
    assert json.dumps(out)  # 不抛错
    assert out["thresholds"]["iv_max"] == DEFAULT_IV_MAX
    assert out["thresholds"]["const_unique_max"] == DEFAULT_CONST_UNIQUE_MAX
    assert out["thresholds"]["id_like_ratio"] == DEFAULT_ID_LIKE_RATIO
    assert out["thresholds"]["missing_max"] == DEFAULT_MISSING_MAX
    assert out["n_before"] == 5
