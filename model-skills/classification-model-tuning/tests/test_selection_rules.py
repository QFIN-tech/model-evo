# -*- coding: utf-8 -*-
"""selection_rules 单测: 三条规则在阈值/缺数据/空表下的边界行为。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from selection_rules import (
    SelectionResult,
    apply_high_psi_rule,
    apply_low_iv_rule,
    apply_high_missing_rule,
    select,
    DEFAULT_PSI_THRESHOLD,
    DEFAULT_IV_THRESHOLD,
    DEFAULT_MISSING_THRESHOLD,
)


# -------------------- apply_high_psi_rule --------------------

def test_high_psi_drops_above_threshold():
    psi_df = pd.DataFrame({"feature": ["a", "b", "c"], "psi": [0.05, 0.20, 0.30]})
    out = apply_high_psi_rule(["a", "b", "c"], psi_df, threshold=0.10)
    assert out == {"b", "c"}


def test_high_psi_ignores_features_not_in_baseline():
    """psi_df 含 baseline 没用过的特征 → 不应被剔除集合记上。"""
    psi_df = pd.DataFrame({"feature": ["a", "z_unused"], "psi": [0.5, 0.9]})
    out = apply_high_psi_rule(["a"], psi_df, threshold=0.10)
    assert out == {"a"}


def test_high_psi_empty_table_no_drop():
    """psi_df 为空(无 oot) → 不剔除任何特征。"""
    out = apply_high_psi_rule(["a", "b"], pd.DataFrame(), threshold=0.10)
    assert out == set()


def test_high_psi_nan_treated_as_not_drop():
    psi_df = pd.DataFrame({"feature": ["a", "b"], "psi": [float("nan"), 0.5]})
    out = apply_high_psi_rule(["a", "b"], psi_df, threshold=0.10)
    assert out == {"b"}


# -------------------- apply_low_iv_rule --------------------

def test_low_iv_drops_below_threshold():
    iv_df = pd.DataFrame({"feature": ["a", "b", "c"], "iv": [0.50, 0.01, 0.10]})
    out = apply_low_iv_rule(["a", "b", "c"], iv_df, threshold=0.02)
    assert out == {"b"}


def test_low_iv_nan_treated_as_drop():
    """IV 算不出来(NaN, 比如常数列)→ 算低 IV 一并剔除。"""
    iv_df = pd.DataFrame({"feature": ["a", "b"], "iv": [float("nan"), 0.30]})
    out = apply_low_iv_rule(["a", "b"], iv_df, threshold=0.02)
    assert out == {"a"}


def test_low_iv_empty_table_no_drop():
    out = apply_low_iv_rule(["a", "b"], pd.DataFrame(), threshold=0.02)
    assert out == set()


# -------------------- apply_high_missing_rule --------------------

def test_high_missing_drops_above_threshold():
    stats_df = pd.DataFrame(
        {"feature": ["a", "b", "c"], "missing_rate": [0.05, 0.97, 0.50]}
    )
    out = apply_high_missing_rule(["a", "b", "c"], stats_df, threshold=0.95)
    assert out == {"b"}


def test_high_missing_empty_table_no_drop():
    out = apply_high_missing_rule(["a"], pd.DataFrame(), threshold=0.95)
    assert out == set()


# -------------------- select() 编排 --------------------

@pytest.fixture
def csv_dir(tmp_path):
    """造 3 个 csv 模拟 feature-analysis 输出。"""
    stats = pd.DataFrame({
        "feature": ["a", "b", "c", "d", "e"],
        "missing_rate": [0.02, 0.10, 0.99, 0.05, 0.30],
    })
    iv_df = pd.DataFrame({
        "feature": ["a", "b", "c", "d", "e"],
        "iv": [0.30, 0.005, 0.50, 0.40, float("nan")],
    })
    psi_df = pd.DataFrame({
        "feature": ["a", "b", "c", "d", "e"],
        "psi": [0.03, 0.02, 0.04, 0.25, 0.05],
    })
    stats.to_csv(tmp_path / "stats.csv", index=False)
    iv_df.to_csv(tmp_path / "iv_table.csv", index=False)
    psi_df.to_csv(tmp_path / "psi_table.csv", index=False)
    return tmp_path


def test_select_all_rules_on(csv_dir):
    """a 全合格 → 保留;b 低 IV;c 高缺失;d 高 PSI;e IV NaN → 剔除。"""
    res = select(["a", "b", "c", "d", "e"], str(csv_dir))
    assert isinstance(res, SelectionResult)
    assert res.kept_features == ["a"]
    assert set(res.dropped_features) == {"b", "c", "d", "e"}
    assert res.dropped_by_rule["high_psi"] == ["d"]
    assert res.dropped_by_rule["low_iv"] == ["b", "e"]
    assert res.dropped_by_rule["high_missing"] == ["c"]


def test_select_preserves_baseline_order(csv_dir):
    """kept/dropped 都按 baseline_features 顺序排,不重排。"""
    res = select(["e", "d", "c", "b", "a"], str(csv_dir))
    assert res.kept_features == ["a"]
    # dropped 顺序按 baseline 顺序: e d c b
    assert res.dropped_features == ["e", "d", "c", "b"]


def test_select_disable_psi_rule(csv_dir):
    """关 PSI 规则后, d 应被保留(只有它被 PSI 剔除)。"""
    res = select(["a", "b", "c", "d", "e"], str(csv_dir), enable_psi=False)
    assert "d" in res.kept_features
    assert res.dropped_by_rule.get("high_psi") is None  # 关了不写
    assert res.rules_enabled["high_psi"] is False


def test_select_disable_iv_rule(csv_dir):
    """关 IV 规则后, b 和 e 应被保留。"""
    res = select(["a", "b", "c", "d", "e"], str(csv_dir), enable_iv=False)
    assert "b" in res.kept_features
    assert "e" in res.kept_features
    # c d 仍被剔除(missing/psi)
    assert "c" in res.dropped_features
    assert "d" in res.dropped_features


def test_select_custom_thresholds_loosen(csv_dir):
    """放宽 PSI 阈值到 0.5, d (psi=0.25) 应被保留。"""
    res = select(
        ["a", "b", "c", "d", "e"], str(csv_dir),
        psi_threshold=0.5, iv_threshold=0.001, missing_threshold=0.999,
    )
    assert set(res.kept_features) == {"a", "b", "c", "d"}  # e 仍因 IV=NaN 被剔除
    assert res.dropped_features == ["e"]


def test_select_missing_csv_gracefully(tmp_path):
    """analysis_dir 里 csv 不全 → 规则跳过,所有特征都保留。"""
    res = select(["a", "b"], str(tmp_path))
    assert res.kept_features == ["a", "b"]
    assert res.dropped_features == []


def test_selection_result_as_dict_serializable(csv_dir):
    """as_dict 返回纯 list/dict 可 json 序列化。"""
    import json
    res = select(["a", "b", "c"], str(csv_dir))
    out = res.as_dict()
    assert json.dumps(out)  # 不抛错
    assert out["thresholds"]["psi"] == DEFAULT_PSI_THRESHOLD
    assert out["thresholds"]["iv"] == DEFAULT_IV_THRESHOLD
    assert out["thresholds"]["missing"] == DEFAULT_MISSING_THRESHOLD
