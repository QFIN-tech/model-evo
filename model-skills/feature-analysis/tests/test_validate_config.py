# -*- coding: utf-8 -*-
"""validate_config 交叉校验 / 配置校验单测。"""
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from validate_config import cross_validate_features  # noqa: E402


def test_cross_validate_all_valid(tmp_path):
    """全部特征在数据中, 返回全部 valid, missing 为空。"""
    data_csv = tmp_path / "feature-list.csv"
    data_csv.write_text("feature_name\nf0\nf1\nf2\n", encoding="utf-8")
    valid, missing = cross_validate_features(["f0", "f1", "f2"], str(data_csv))
    assert valid == ["f0", "f1", "f2"]
    assert missing == []


def test_cross_validate_partial_missing(tmp_path):
    """部分特征不在数据中, 返回 valid + missing 分列。"""
    data_csv = tmp_path / "feature-list.csv"
    data_csv.write_text("feature_name\nf0\nf2\n", encoding="utf-8")
    valid, missing = cross_validate_features(["f0", "f1", "f2"], str(data_csv))
    assert valid == ["f0", "f2"]
    assert missing == ["f1"]


def test_cross_validate_all_missing(tmp_path):
    """全部特征都不在数据中, valid 为空, missing 全量。"""
    data_csv = tmp_path / "feature-list.csv"
    data_csv.write_text("feature_name\na\nb\n", encoding="utf-8")
    valid, missing = cross_validate_features(["f0", "f1"], str(data_csv))
    assert valid == []
    assert missing == ["f0", "f1"]


def test_cross_validate_preserves_order(tmp_path):
    """valid 保持用户输入顺序。"""
    data_csv = tmp_path / "feature-list.csv"
    data_csv.write_text("feature_name\nc\na\nb\n", encoding="utf-8")
    valid, _ = cross_validate_features(["a", "b", "c"], str(data_csv))
    assert valid == ["a", "b", "c"]


def test_cross_validate_file_not_found():
    """数据 csv 不存在时抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        cross_validate_features(["f0"], "/nonexistent/path.csv")


def test_cross_validate_missing_column(tmp_path):
    """csv 没有 feature_name 列时抛 ValueError。"""
    data_csv = tmp_path / "bad.csv"
    data_csv.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature_name"):
        cross_validate_features(["f0"], str(data_csv))
