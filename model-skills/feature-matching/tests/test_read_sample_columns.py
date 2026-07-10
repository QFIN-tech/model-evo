# -*- coding: utf-8 -*-
"""read_sample_columns 单测: parquet/csv 两种输入都能正确返回列名。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from derive_feature_list import read_sample_columns, read_parquet_columns


def test_read_parquet_columns_returns_schema_names(tmp_path):
    """parquet 输入: 走 pyarrow schema, 列名按原顺序返回。"""
    p = tmp_path / "sample.parquet"
    pd.DataFrame({"z": [1], "a": [2], "m": [3]}).to_parquet(p)

    cols = read_parquet_columns(str(p))
    assert cols == ["z", "a", "m"]


def test_read_sample_columns_parquet(tmp_path):
    """read_sample_columns 对 .parquet 走 parquet 分支。"""
    p = tmp_path / "sample.parquet"
    pd.DataFrame({"user_no": ["u0"], "pday": [20260101], "label": [0], "f0": [1.0]}).to_parquet(p)

    cols = read_sample_columns(str(p))
    assert cols == ["user_no", "pday", "label", "f0"]


def test_read_sample_columns_csv(tmp_path):
    """read_sample_columns 对 .csv 走 pandas 读 header 分支, 不全量加载。"""
    p = tmp_path / "sample.csv"
    pd.DataFrame({"user_no": ["u0"], "pday": [20260101], "label": [0], "f0": [1.0]}).to_csv(p, index=False)

    cols = read_sample_columns(str(p))
    assert cols == ["user_no", "pday", "label", "f0"]


def test_read_sample_columns_csv_preserves_order(tmp_path):
    """csv 列顺序与原文件一致(不重排)。"""
    p = tmp_path / "sample.csv"
    pd.DataFrame({"z": [1], "a": [2], "m": [3]}).to_csv(p, index=False)

    cols = read_sample_columns(str(p))
    assert cols == ["z", "a", "m"]


def test_read_sample_columns_unsupported_extension(tmp_path):
    """非 .parquet / .csv 输入: 抛 ValueError。"""
    p = tmp_path / "sample.txt"
    p.write_text("user_no,label\nu0,0\n")

    with pytest.raises(ValueError, match="不支持的 sample 文件格式"):
        read_sample_columns(str(p))


def test_read_sample_columns_csv_uppercase_extension(tmp_path):
    """大写 .CSV 扩展名也应识别。"""
    p = tmp_path / "sample.CSV"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(p, index=False)

    cols = read_sample_columns(str(p))
    assert cols == ["a", "b"]
