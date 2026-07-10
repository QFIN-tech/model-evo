# -*- coding: utf-8 -*-
"""build_sample_feature_sql t-1 滞后 JOIN 单测: SQL 字符串断言。

覆盖 lag=0/1 两种模式的 ON 子句 + 特征表窗口, 以及边界异常。
"""
import sys
from pathlib import Path

import pytest

_SHARED_SCRIPTS = Path(__file__).resolve().parents[3] / "_modelevo-shared" / "scripts"
sys.path.insert(0, str(_SHARED_SCRIPTS))

from fetch_spark import build_sample_feature_sql


def test_lag_zero_on_clause_same_day():
    """lag=0: ON 子句为 a.user_no=b.user_no AND a.pday=b.pday, 特征窗口 = 样本窗口。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0", "f1"],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=0,
    )
    assert "a.user_no=b.user_no" in sql
    assert "a.pday=b.pday" in sql
    # 特征表子查询窗口 = 样本窗口
    assert "FROM db.feat WHERE pday >= '20260101' AND pday <= '20260131'" in sql
    # 样本表子查询窗口
    assert "FROM db.sample WHERE pday >= '20260101' AND pday <= '20260131'" in sql
    # 不应出现日期算术
    assert "date_add" not in sql
    assert "date_format" not in sql


def test_lag_one_on_clause_date_arithmetic():
    """lag=1: ON 子句用日期算术对齐 a.pday = date_format(date_add(b.pday,1)),
    特征窗口 = 样本窗口 - 1 天。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0", "f1"],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=1,
    )
    # ON 子句: a.user_no=b.user_no + 日期算术 (无 a.pday=b.pday)
    assert "a.user_no=b.user_no" in sql
    assert "a.pday=b.pday" not in sql
    assert "a.pday = date_format(date_add(to_date(b.pday, 'yyyyMMdd'), 1), 'yyyyMMdd')" in sql
    # 特征表窗口平移 -1 天: 20251231 ~ 20260130
    assert "FROM db.feat WHERE pday >= '20251231' AND pday <= '20260130'" in sql
    # 样本表窗口不变
    assert "FROM db.sample WHERE pday >= '20260101' AND pday <= '20260131'" in sql


def test_lag_one_cross_year_boundary():
    """跨年边界: fetch_start=20260101 → feat_start=20251231 (datetime 跨年处理)。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0"],
        fetch_start="20260101", fetch_end="20260105", where=None,
        feature_lag_day=1,
    )
    assert "FROM db.feat WHERE pday >= '20251231' AND pday <= '20260104'" in sql


def test_lag_one_full_column_except_includes_dt_col():
    """lag=1 + 全列模式 (features=[]): b.* EXCEPT 子句含 dt_col, 避免重复列。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=[],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=1,
    )
    # lag=1: EXCEPT 子句需含 dt_col (因 b.pday 不再等值匹配 a.pday)
    assert "b.* EXCEPT (user_no, pday)" in sql


def test_lag_zero_full_column_except_excludes_dt_col():
    """lag=0 + 全列模式: EXCEPT 子句只含 join_keys (含 dt_col=pday, 仍出现在 EXCEPT 中)。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=[],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=0,
    )
    # lag=0: join_keys=user_no,pday, EXCEPT = (user_no, pday)
    assert "b.* EXCEPT (user_no, pday)" in sql


def test_lag_one_without_dt_col_in_join_keys_raises():
    """lag=1 + join_keys 不含 dt_col: 应抛 ValueError (无 dt_col 无法做日期对齐)。"""
    with pytest.raises(ValueError, match="feature_lag_day=1 要求 dt_col"):
        build_sample_feature_sql(
            sample_table="db.sample", feature_table="db.feat",
            join_keys=["user_no"], dt_col="pday",
            label_expr="label", id_cols=["user_no"], features=["f0"],
            fetch_start="20260101", fetch_end="20260131", where=None,
            feature_lag_day=1,
        )


def test_lag_two_raises():
    """lag=2 不支持: 仅 0/1, 抛 ValueError。"""
    with pytest.raises(ValueError, match="feature_lag_day 仅支持 0"):
        build_sample_feature_sql(
            sample_table="db.sample", feature_table="db.feat",
            join_keys=["user_no", "pday"], dt_col="pday",
            label_expr="label", id_cols=["user_no"], features=["f0"],
            fetch_start="20260101", fetch_end="20260131", where=None,
            feature_lag_day=2,
        )


def test_lag_negative_raises():
    """lag 负数: 仅 0/1, 抛 ValueError。"""
    with pytest.raises(ValueError, match="feature_lag_day 仅支持 0"):
        build_sample_feature_sql(
            sample_table="db.sample", feature_table="db.feat",
            join_keys=["user_no", "pday"], dt_col="pday",
            label_expr="label", id_cols=["user_no"], features=["f0"],
            fetch_start="20260101", fetch_end="20260131", where=None,
            feature_lag_day=-1,
        )


def test_lag_one_specified_features_drops_dt_col_from_b_cols():
    """lag=1 + 指定 features: b_cols 用 join_keys_eq 去掉 dt_col, 避免 b.pday 与 a.pday 冲突。"""
    sql = build_sample_feature_sql(
        sample_table="db.sample", feature_table="db.feat",
        join_keys=["user_no", "pday"], dt_col="pday",
        label_expr="label", id_cols=["user_no"], features=["f0", "f1"],
        fetch_start="20260101", fetch_end="20260131", where=None,
        feature_lag_day=1,
    )
    # 特征表子查询 b_cols = user_no, f0, f1 (不含 pday)
    # 查找特征表子查询段: FROM db.feat WHERE pday >= ...
    feat_sub_idx = sql.find("FROM db.feat")
    assert feat_sub_idx >= 0
    # 截取特征表子查询的 SELECT 列表 (在 SELECT 关键字与 FROM db.feat 之间)
    select_idx = sql.rfind("SELECT ", 0, feat_sub_idx)
    feat_select_cols = sql[select_idx:feat_sub_idx]
    assert "user_no" in feat_select_cols
    assert "f0" in feat_select_cols
    assert "f1" in feat_select_cols
    # 不应单独 SELECT b.pday (因 lag=1 时 b_cols 用 join_keys_eq 不含 dt_col)
    # feat_select_cols 中 pday 仅可能以 AS label 形式出现(标签列); 此处 label_expr="label" 是列名,
    # 不应混入 pday
    assert "pday" not in feat_select_cols
