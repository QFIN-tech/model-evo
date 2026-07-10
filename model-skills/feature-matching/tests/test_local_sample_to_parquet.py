# -*- coding: utf-8 -*-
"""_local_sample_to_parquet 单测: parquet 输入直接复制, csv 输入转写为 parquet。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from fetch_sample import _local_sample_to_parquet


def _make_df(n=20):
    """造一个含 id/dt/label/feature 的小 DataFrame。"""
    return pd.DataFrame(
        {
            "user_no": [f"u{i}" for i in range(n)],
            "pday": [20260101 + i for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "f0": [float(i) for i in range(n)],
            "f1": [float(i * 2) for i in range(n)],
        }
    )


def test_parquet_input_copied_as_parquet(tmp_path):
    """parquet 输入: shutil.copyfile 直接复制, 输出仍是 parquet 且内容一致。"""
    src = tmp_path / "src.parquet"
    df = _make_df()
    df.to_parquet(src)

    dst = tmp_path / "out" / "sample.parquet"
    dst.parent.mkdir(parents=True)
    _local_sample_to_parquet(str(src), str(dst))

    assert dst.exists()
    out = pd.read_parquet(dst)
    pd.testing.assert_frame_equal(out, df)


def test_csv_input_transcoded_to_parquet(tmp_path):
    """csv 输入: pandas 读后写 parquet, 输出列与数据一致(列顺序保留)。"""
    src = tmp_path / "src.csv"
    df = _make_df()
    df.to_csv(src, index=False)

    dst = tmp_path / "out" / "sample.parquet"
    dst.parent.mkdir(parents=True)
    _local_sample_to_parquet(str(src), str(dst))

    assert dst.exists()
    assert dst.suffix == ".parquet"
    out = pd.read_parquet(dst)
    pd.testing.assert_frame_equal(out, df)


def test_csv_input_preserves_column_order(tmp_path):
    """csv 输入: 列顺序与原 csv 一致(不重排)。"""
    src = tmp_path / "src.csv"
    df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
    df.to_csv(src, index=False)

    dst = tmp_path / "sample.parquet"
    _local_sample_to_parquet(str(src), str(dst))

    out = pd.read_parquet(dst)
    assert list(out.columns) == ["z", "a", "m"]


def test_unsupported_extension_raises(tmp_path):
    """非 .parquet / .csv 输入: 抛 ValueError。"""
    src = tmp_path / "src.txt"
    src.write_text("user_no,label\nu0,0\n")

    dst = tmp_path / "sample.parquet"
    with pytest.raises(ValueError, match="不支持的本地样本格式"):
        _local_sample_to_parquet(str(src), str(dst))


def test_csv_input_uppercase_extension(tmp_path):
    """大写 .CSV 扩展名也应识别(按 lower() 判断)。"""
    src = tmp_path / "src.CSV"
    df = _make_df(5)
    df.to_csv(src, index=False)

    dst = tmp_path / "sample.parquet"
    _local_sample_to_parquet(str(src), str(dst))

    out = pd.read_parquet(dst)
    pd.testing.assert_frame_equal(out, df)
