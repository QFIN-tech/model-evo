# -*- coding: utf-8 -*-
"""diagnose 规则诊断单测: 各种 metric 组合 → 期望 status。"""
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from diagnose import diagnose, Diagnosis


def _m(train_auc, val_auc, oot_auc, ks=0.3):
    """简便构造 metrics dict。"""
    return {
        "train": {"auc": train_auc, "ks": ks},
        "val": {"auc": val_auc, "ks": ks},
        "oot": {"auc": oot_auc, "ks": ks},
    }


def test_well_fit_when_gap_small_and_train_high():
    """train/oot 都不错且 gap 小 → well_fit。"""
    d = diagnose(_m(0.82, 0.79, 0.78), used_params={"n_estimators": 800},
                 best_iteration=200, new_psi=0.05)
    assert d.status == "well_fit"


def test_overfit_when_gap_large():
    """train-oot gap=0.10 > 0.05 → overfit。"""
    d = diagnose(_m(0.90, 0.83, 0.80), used_params={"n_estimators": 800},
                 best_iteration=200, new_psi=0.05)
    assert d.status == "overfit"
    assert any("过拟合" in r for r in d.reasons)
    assert d.signals["train_oot_gap"] > 0.05


def test_underfit_when_train_low():
    """train_auc=0.65 < 0.70 → underfit。"""
    d = diagnose(_m(0.65, 0.64, 0.63), used_params={"n_estimators": 800},
                 best_iteration=100, new_psi=0.05)
    assert d.status == "underfit"


def test_underfit_when_gap_too_small_and_train_low():
    """train 低 + gap 几乎为 0 → underfit (双触发, 文案含"且")。"""
    d = diagnose(_m(0.68, 0.68, 0.679), used_params={"n_estimators": 800},
                 best_iteration=100, new_psi=0.05)
    assert d.status == "underfit"
    assert any("且" in r for r in d.reasons)


def test_underfit_when_gap_too_small_only():
    """gap < 0.005 但 train_auc >= 0.70 → 仍触发 underfit (独立触发, 不要求 train 低)。

    SKILL.md 记载 underfit 触发条件为 `train_auc < 0.70 或 train-oot gap < 0.005`,
    故 train_auc=0.78 / oot_auc=0.777 (gap=0.003 < 0.005) 应判欠拟合。
    """
    d = diagnose(_m(0.78, 0.78, 0.777), used_params={"n_estimators": 800},
                 best_iteration=200, new_psi=0.05)
    assert d.status == "underfit"
    assert any("训练都没学进去" in r for r in d.reasons)


def test_underconverged_when_best_iter_at_limit():
    """best_iteration=780 / n=800 = 97.5% > 95% → underconverged(且 train 高,无 overfit)。"""
    d = diagnose(_m(0.78, 0.77, 0.76), used_params={"n_estimators": 800},
                 best_iteration=780, new_psi=0.05)
    assert d.status == "underconverged"


def test_unstable_psi_alone():
    """指标合理但 PSI 高 → unstable_psi。"""
    d = diagnose(_m(0.80, 0.78, 0.77), used_params={"n_estimators": 800},
                 best_iteration=200, new_psi=0.18)
    assert d.status == "unstable_psi"
    assert any("PSI_WARN" in r for r in d.reasons)


def test_overfit_wins_over_psi_priority():
    """同时过拟合 + PSI 高: status 取 overfit(在循环中先命中)。"""
    d = diagnose(_m(0.92, 0.85, 0.80), used_params={"n_estimators": 800},
                 best_iteration=200, new_psi=0.18)
    assert d.status == "overfit"


def test_diagnosis_as_dict_serializable():
    """as_dict 返回 status/reasons/signals 可序列化。"""
    d = diagnose(_m(0.90, 0.83, 0.80), used_params={"n_estimators": 800},
                 best_iteration=200, new_psi=0.05)
    out = d.as_dict()
    assert out["status"] == "overfit"
    assert isinstance(out["reasons"], list)
    assert "train_oot_gap" in out["signals"]


def test_missing_metrics_falls_back_gracefully():
    """metrics 缺 val 字段也不应抛错。"""
    metrics = {"train": {"auc": 0.78}, "oot": {"auc": 0.77}}
    d = diagnose(metrics, used_params={"n_estimators": 800},
                 best_iteration=200, new_psi=None)
    assert isinstance(d, Diagnosis)
    assert d.status in ("well_fit", "overfit", "underfit", "underconverged")


# -------------------- dnn algo: underconverged 分流 --------------------

def test_dnn_underconverged_when_early_stopped_false():
    """dnn: early_stopped=False (跑满 epochs) → underconverged。"""
    d = diagnose(
        _m(0.82, 0.79, 0.78),
        used_params={"epochs": 100, "early_stopped": False},
        best_iteration=100, new_psi=0.05, algo="dnn",
    )
    assert d.status == "underconverged"
    assert any("early_stopped=False" in r for r in d.reasons)


def test_dnn_underconverged_when_best_epoch_at_limit():
    """dnn: best_epoch/epochs >= 0.95 → underconverged。"""
    d = diagnose(
        _m(0.82, 0.79, 0.78),
        used_params={"epochs": 100, "early_stopped": True},
        best_iteration=98, new_psi=0.05, algo="dnn",
    )
    assert d.status == "underconverged"


def test_dnn_well_fit_when_converged_early():
    """dnn: early_stopped=True 且 best_epoch 远小于 epochs → well_fit。"""
    d = diagnose(
        _m(0.82, 0.79, 0.78),
        used_params={"epochs": 100, "early_stopped": True},
        best_iteration=30, new_psi=0.05, algo="dnn",
    )
    assert d.status == "well_fit"


def test_dnn_overfit_uses_auc_gap_only():
    """dnn: overfit 规则 algo-agnostic, 只看 AUC gap。"""
    d = diagnose(
        _m(0.92, 0.85, 0.80),
        used_params={"epochs": 100, "early_stopped": True},
        best_iteration=30, new_psi=0.05, algo="dnn",
    )
    assert d.status == "overfit"


# -------------------- lr algo: underconverged 分流 --------------------

def test_lr_underconverged_when_not_converged():
    """lr: converged=False → underconverged。"""
    d = diagnose(
        _m(0.82, 0.79, 0.78),
        used_params={"max_iter": 1000, "converged": False},
        best_iteration=1000, new_psi=0.05, algo="lr",
    )
    assert d.status == "underconverged"
    assert any("converged=False" in r for r in d.reasons)


def test_lr_well_fit_when_converged():
    """lr: converged=True → 不会触发 underconverged。"""
    d = diagnose(
        _m(0.82, 0.79, 0.78),
        used_params={"max_iter": 1000, "converged": True},
        best_iteration=200, new_psi=0.05, algo="lr",
    )
    assert d.status == "well_fit"


def test_lr_overfit_uses_auc_gap_only():
    """lr: overfit 规则 algo-agnostic。"""
    d = diagnose(
        _m(0.92, 0.85, 0.80),
        used_params={"max_iter": 1000, "converged": True},
        best_iteration=200, new_psi=0.05, algo="lr",
    )
    assert d.status == "overfit"


def test_lr_signals_no_early_stop_ratio():
    """lr: 不计算 early_stop_ratio signal (n_iter 语义不同)。"""
    d = diagnose(
        _m(0.82, 0.79, 0.78),
        used_params={"max_iter": 1000, "converged": True},
        best_iteration=200, new_psi=0.05, algo="lr",
    )
    assert "early_stop_ratio" not in d.signals
