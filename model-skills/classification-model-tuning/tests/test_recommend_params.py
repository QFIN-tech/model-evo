# -*- coding: utf-8 -*-
"""recommend_params 单测: 各诊断状态下,推荐参数应按方向移动且在 BOUNDS 内。"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from diagnose import Diagnosis
from recommend_params import recommend, BOUNDS, BOUNDS_DNN, BOUNDS_LR


_BASE = {
    "max_depth": 6, "learning_rate": 0.03, "n_estimators": 800,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
}


def test_underfit_adds_capacity():
    """underfit: depth+1, mcw/2, lr/2, n_est×1.5。"""
    diag = Diagnosis(status="underfit", reasons=[], signals={})
    r = recommend(_BASE, diag)
    assert r["max_depth"] == 7
    assert r["min_child_weight"] == 10
    assert r["learning_rate"] == 0.015
    assert r["n_estimators"] == 1200


def test_overfit_adds_regularization():
    """overfit: depth-1, reg_lambda*2, mcw*2。"""
    diag = Diagnosis(status="overfit", reasons=[], signals={})
    r = recommend(_BASE, diag)
    assert r["max_depth"] == 5
    assert r["reg_lambda"] == 2.0
    assert r["min_child_weight"] == 40


def test_underconverged_doubles_n_estimators():
    diag = Diagnosis(status="underconverged", reasons=[], signals={})
    r = recommend(_BASE, diag)
    assert r["n_estimators"] == 1600


def test_unstable_psi_reduces_sampling():
    diag = Diagnosis(status="unstable_psi", reasons=[], signals={})
    r = recommend(_BASE, diag)
    assert r["subsample"] == 0.7
    assert r["colsample_bytree"] == 0.7


def test_well_fit_returns_baseline_copy():
    """well_fit: 不修改任何键(但仍是新 dict,避免别名)。"""
    diag = Diagnosis(status="well_fit", reasons=[], signals={})
    r = recommend(_BASE, diag)
    assert r == _BASE
    assert r is not _BASE


def test_bounds_clip_for_extreme_baseline():
    """baseline depth 已经在上界时,underfit 推荐不应越界。"""
    base = dict(_BASE, max_depth=10)
    diag = Diagnosis(status="underfit", reasons=[], signals={})
    r = recommend(base, diag)
    assert r["max_depth"] <= BOUNDS["max_depth"][1]


def test_bounds_clip_for_lower_extreme():
    """baseline subsample 已经在下界附近时,unstable_psi 推荐不应越界。"""
    base = dict(_BASE, subsample=0.55)
    diag = Diagnosis(status="unstable_psi", reasons=[], signals={})
    r = recommend(base, diag)
    assert r["subsample"] >= BOUNDS["subsample"][0]


# -------------------- dnn 推荐策略 --------------------

_BASE_DNN = {
    "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 1e-4,
    "batch_size": 512, "epochs": 100, "patience": 10,
}


def test_dnn_underfit_increases_lr_reduces_dropout():
    diag = Diagnosis(status="underfit", reasons=[], signals={})
    r = recommend(_BASE_DNN, diag, algo="dnn")
    assert r["learning_rate"] == 0.002
    assert r["dropout"] == round(0.3 * 0.7, 3)
    assert r["epochs"] == 150


def test_dnn_overfit_increases_regularization():
    diag = Diagnosis(status="overfit", reasons=[], signals={})
    r = recommend(_BASE_DNN, diag, algo="dnn")
    assert r["dropout"] == 0.4
    assert r["weight_decay"] == 2e-4


def test_dnn_underconverged_doubles_epochs_and_patience():
    diag = Diagnosis(status="underconverged", reasons=[], signals={})
    r = recommend(_BASE_DNN, diag, algo="dnn")
    assert r["epochs"] == 200
    assert r["patience"] == 15


def test_dnn_unstable_psi_increases_dropout_and_batch():
    diag = Diagnosis(status="unstable_psi", reasons=[], signals={})
    r = recommend(_BASE_DNN, diag, algo="dnn")
    assert r["dropout"] == 0.4
    assert r["batch_size"] == 1024


def test_dnn_well_fit_unchanged():
    diag = Diagnosis(status="well_fit", reasons=[], signals={})
    r = recommend(_BASE_DNN, diag, algo="dnn")
    assert r == _BASE_DNN
    assert r is not _BASE_DNN


def test_dnn_bounds_clip_high_dropout():
    """baseline dropout 已在上界时,overfit 推荐不应越界。"""
    base = dict(_BASE_DNN, dropout=0.55)
    diag = Diagnosis(status="overfit", reasons=[], signals={})
    r = recommend(base, diag, algo="dnn")
    assert r["dropout"] <= BOUNDS_DNN["dropout"][1]


# -------------------- lr 推荐策略 --------------------

_BASE_LR = {
    "C": 1.0, "max_n_bins": 8, "min_bin_size": 0.05,
    "max_iter": 1000, "regularization": "l2",
}


def test_lr_underfit_increases_C_and_bins():
    diag = Diagnosis(status="underfit", reasons=[], signals={})
    r = recommend(_BASE_LR, diag, algo="lr")
    assert r["C"] == 2.0
    assert r["max_n_bins"] == 10


def test_lr_overfit_decreases_C_and_bins():
    diag = Diagnosis(status="overfit", reasons=[], signals={})
    r = recommend(_BASE_LR, diag, algo="lr")
    assert r["C"] == 0.5
    assert r["max_n_bins"] == 6


def test_lr_underconverged_doubles_max_iter():
    diag = Diagnosis(status="underconverged", reasons=[], signals={})
    r = recommend(_BASE_LR, diag, algo="lr")
    assert r["max_iter"] == 2000


def test_lr_unstable_psi_coarsens_bins():
    diag = Diagnosis(status="unstable_psi", reasons=[], signals={})
    r = recommend(_BASE_LR, diag, algo="lr")
    assert r["max_n_bins"] == 6
    assert r["min_bin_size"] == 0.1


def test_lr_well_fit_unchanged():
    diag = Diagnosis(status="well_fit", reasons=[], signals={})
    r = recommend(_BASE_LR, diag, algo="lr")
    assert r == _BASE_LR
    assert r is not _BASE_LR


def test_lr_bounds_clip_high_C():
    """baseline C 已在上界时,underfit 推荐不应越界。"""
    base = dict(_BASE_LR, C=900.0)
    diag = Diagnosis(status="underfit", reasons=[], signals={})
    r = recommend(base, diag, algo="lr")
    assert r["C"] <= BOUNDS_LR["C"][1]


def test_unknown_algo_raises():
    """未知 algo 应抛 ValueError。"""
    diag = Diagnosis(status="well_fit", reasons=[], signals={})
    try:
        recommend(_BASE, diag, algo="unknown_algo")
        assert False, "应抛 ValueError"
    except ValueError:
        pass
