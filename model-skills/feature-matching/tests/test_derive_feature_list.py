# -*- coding: utf-8 -*-
"""derive_feature_list 派生特征清单单测(纯函数 derive_features / filter_by_list)。"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from derive_feature_list import derive_features, filter_by_list


def test_derive_excludes_non_feature_cols():
    """排除 id/dt/label 列后只剩特征列, 保序。"""
    all_cols = ["user_no", "pday", "label", "f0", "f1", "f2"]
    feats = derive_features(all_cols, ["user_no", "pday", "label"])
    assert feats == ["f0", "f1", "f2"]


def test_derive_keeps_order_and_dedup():
    """重复列去重, 顺序按 all_cols 首次出现。"""
    all_cols = ["f0", "f1", "f0", "f2"]
    feats = derive_features(all_cols, [])
    assert feats == ["f0", "f1", "f2"]


def test_derive_empty_exclude():
    """exclude 为空时全部列均为特征。"""
    all_cols = ["a", "b", "c"]
    assert derive_features(all_cols, []) == ["a", "b", "c"]


def test_derive_exclude_missing_cols_noop():
    """exclude 含不存在的列不影响结果。"""
    all_cols = ["f0", "f1"]
    assert derive_features(all_cols, ["nope", "user_no"]) == ["f0", "f1"]


def test_derive_exclude_all():
    """全部列都被排除时返回空。"""
    all_cols = ["user_no", "pday"]
    assert derive_features(all_cols, ["user_no", "pday"]) == []


# ---- filter_by_list 单测 ----

def test_filter_keeps_intersection_in_allow_list_order():
    """交集按 allow_list 顺序, 不按 derived 顺序。"""
    derived = ["f2", "f0", "f1"]  # sample 里的顺序
    allow = ["f0", "f1", "f2"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1", "f2"]  # 按 allow 顺序
    assert missing == []


def test_filter_drops_allow_features_not_in_sample():
    """allow_list 里不在 sample 中的特征进 missing, 不进 kept。"""
    derived = ["f0", "f1"]
    allow = ["f0", "f1", "f2", "f3"]  # f2/f3 不在 sample
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1"]
    assert missing == ["f2", "f3"]


def test_filter_drops_derived_features_not_in_allow():
    """derived 里不在 allow_list 中的特征被静默丢弃(不打 warn)。"""
    derived = ["f0", "f1", "extra_in_sample"]
    allow = ["f0", "f1"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1"]
    assert missing == []
    # extra_in_sample 不出现, 也不进 missing(missing 只跟踪 allow 侧)


def test_filter_allow_dedup():
    """allow_list 含重复特征时去重, 保首次出现顺序。"""
    derived = ["f0", "f1", "f2"]
    allow = ["f0", "f1", "f0", "f2", "f1"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == ["f0", "f1", "f2"]
    assert missing == []


def test_filter_empty_allow_raises_or_returns_empty():
    """allow_list 为空时 kept 为空(不会抛错, 由 _load_allow_list 在文件层校验非空)。"""
    derived = ["f0", "f1"]
    kept, missing = filter_by_list(derived, [])
    assert kept == []
    assert missing == []


def test_filter_no_overlap():
    """allow 与 derived 完全不相交: kept 空, missing = allow 全部。"""
    derived = ["a", "b"]
    allow = ["x", "y"]
    kept, missing = filter_by_list(derived, allow)
    assert kept == []
    assert missing == ["x", "y"]
