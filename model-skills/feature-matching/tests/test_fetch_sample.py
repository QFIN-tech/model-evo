# -*- coding: utf-8 -*-
"""fetch_sample _build_cfg 单测: 构造 args Namespace → 验证 cfg dict 结构。

本测试针对 _build_cfg 纯函数, 覆盖 spark / local_file 两种模式 + features 留空/指定两种分支。
"""
import sys
from argparse import Namespace
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

# 注入 _modelevo-shared/scripts (与 _bootstrap.py 同口径), 让 _build_cfg 在 spark 模式下
# 能 import spark_defaults; 测试直接调 _build_cfg 不走 main(), 不会触发 import _bootstrap。
_SHARED_SCRIPTS = Path(__file__).resolve().parents[3] / "_modelevo-shared" / "scripts"
sys.path.insert(0, str(_SHARED_SCRIPTS))

from fetch_sample import _build_cfg


def _args(**kw):
    """构造 _build_cfg 所需的 argparse Namespace 默认值, 测试用例只 override 关键字段。"""
    base = dict(
        mode="spark",
        model_name="m1",
        version="v1",
        sample_table="db.t",
        local_parquet_path=None,
        feature_table=None,
        join_keys=None,
        dt_col="pday",
        id_cols="user_no",
        fetch_start="20260101",
        fetch_end="20260131",
        where=None,
        features=None,
        feature_list_source=None,
        business_domain=None,
        label_col="label",
        label_expr=None,
        hdfs_base=None,
        feature_lag_day=0,
    )
    base.update(kw)
    return Namespace(**base)


def test_build_cfg_spark_basic():
    """spark 模式: 指定 features, cfg 应含 features 列表。"""
    cfg = _build_cfg(_args(features="f0,f1"))
    model = cfg["model"]
    assert model["name"] == "m1"
    assert model["sample_table"] == "db.t"
    assert model["fetch_dt"] == ["20260101", "20260131"]
    assert model["features"] == ["f0", "f1"]
    assert "split" not in model
    assert "spark_submit" in cfg


def test_build_cfg_spark_join_mode():
    """spark 模式 + feature_table: join_keys 默认 [user_no, pday]。"""
    cfg = _build_cfg(_args(feature_table="db.feat", join_keys=None))
    model = cfg["model"]
    assert model["feature_table"] == "db.feat"
    assert model["join_keys"] == ["user_no", "pday"]


def test_build_cfg_local_file_mode():
    """local_file 模式: hdfs_base 空, sample_table 占位, mode=local_file。"""
    cfg = _build_cfg(_args(
        mode="local_file",
        sample_table=None,
        local_parquet_path="/tmp/x.parquet",
        features=None,
    ))
    model = cfg["model"]
    assert model["mode"] == "local_file"
    assert model["sample_table"] == "local_file"
    assert model["local_parquet_path"] == "/tmp/x.parquet"
    assert cfg["spark_submit"]["hdfs_base"] == ""
    assert "split" not in model


def test_build_cfg_features_empty_list():
    """features 留空: cfg["model"]["features"] == [] (留给下游派生)。"""
    cfg = _build_cfg(_args(features=None))
    assert cfg["model"]["features"] == []


def test_build_cfg_features_dedup_whitespace():
    """features 字符串带空格: strip 后切分。"""
    cfg = _build_cfg(_args(features=" f0 , f1 , f2 "))
    assert cfg["model"]["features"] == ["f0", "f1", "f2"]


def test_build_cfg_join_keys_override():
    """join_keys 显式指定: 不走 user_no+pday 默认。"""
    cfg = _build_cfg(_args(
        feature_table="db.feat",
        join_keys="user_no,dt",
    ))
    assert cfg["model"]["join_keys"] == ["user_no", "dt"]


def test_build_cfg_feature_lag_day_default():
    """默认 feature_lag_day=0 写入 cfg。"""
    cfg = _build_cfg(_args(feature_table="db.feat"))
    assert cfg["model"]["feature_lag_day"] == 0


def test_build_cfg_feature_lag_day_one():
    """feature_lag_day=1 透传到 cfg (spark 模式)。"""
    cfg = _build_cfg(_args(feature_table="db.feat", feature_lag_day=1))
    assert cfg["model"]["feature_lag_day"] == 1


def test_build_cfg_feature_lag_day_local_file():
    """local_file 模式下 feature_lag_day 也落盘记录 (即使不走 spark)。"""
    cfg = _build_cfg(_args(
        mode="local_file",
        sample_table=None,
        local_parquet_path="/tmp/x.parquet",
        feature_lag_day=1,
    ))
    assert cfg["model"]["feature_lag_day"] == 1
