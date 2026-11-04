"""回归:finalize-retrain 冻结轮数时不得触发 catboost 同义参数冲突。

实测 2026-11-02 批次:creditcard / amazon_employee_access 的 catboost HPO
冠军构造时带 iterations=4000,旧实现再 set_params(n_estimators=...) 后
fit 抛 CatBoostError('only one of the parameters ...'),交付链中断。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
for _p in (str(SKILL), str(SKILL / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_catboost_round_key_is_iterations():
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    from _mvo.cmd_deliver import _freeze_rounds

    m = CatBoostClassifier(iterations=4000, depth=7, verbose=False)
    assert _freeze_rounds(m, 269) == "iterations"
    assert int(m.get_params()["iterations"]) == 269
    assert "n_estimators" not in (m.get_params() or {})
    m.fit([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1], verbose=False)  # 不抛


def test_catboost_n_estimators_only_model():
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    from _mvo.cmd_deliver import _freeze_rounds

    m = CatBoostClassifier(n_estimators=4000, verbose=False)
    assert _freeze_rounds(m, 120) == "n_estimators"
    m.fit([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1], verbose=False)


def test_xgboost_and_lightgbm_keys():
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    import lightgbm as lgb
    import xgboost as xgb

    from _mvo.cmd_deliver import _freeze_rounds

    assert _freeze_rounds(xgb.XGBClassifier(n_estimators=100), 42) == "n_estimators"
    assert _freeze_rounds(lgb.LGBMClassifier(n_estimators=100), 42) == "n_estimators"


def test_non_round_estimator_returns_none():
    from sklearn.linear_model import LogisticRegression

    from _mvo.cmd_deliver import _freeze_rounds

    assert _freeze_rounds(LogisticRegression(), 10) is None


def test_catboost_early_stopping_synonym():
    """catboost 的 od_wait 与 early_stopping_rounds 同义:只设已有的那个。"""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    from _mvo.cmd_deliver import _clear_early_stopping

    m = CatBoostClassifier(iterations=100, od_wait=50, verbose=False)
    assert _clear_early_stopping(m) == "od_wait"
    assert m.get_params().get("od_wait") is None
    assert "early_stopping_rounds" not in (m.get_params() or {})
    m.fit([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1], verbose=False)


def test_xgboost_early_stopping_key():
    pytest.importorskip("xgboost")
    import xgboost as xgb

    from _mvo.cmd_deliver import _clear_early_stopping

    m = xgb.XGBClassifier(n_estimators=100, early_stopping_rounds=10)
    assert _clear_early_stopping(m) == "early_stopping_rounds"
    m.fit([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1], verbose=False)  # 无 eval_set 也不抛


def test_catboost_use_best_model_disabled():
    """无 eval_set 的全量重拟合必须关掉 use_best_model(catboost 只收布尔)。"""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    from _mvo.cmd_deliver import _clear_early_stopping, _freeze_rounds

    m = CatBoostClassifier(iterations=100, od_type="Iter", od_wait=50,
                           use_best_model=True, verbose=False)
    _freeze_rounds(m, 60)
    _clear_early_stopping(m)
    assert m.get_params().get("use_best_model") is False
    m.fit([[0.0], [1.0], [0.0], [1.0]], [0, 1, 0, 1], verbose=False)  # 不抛
