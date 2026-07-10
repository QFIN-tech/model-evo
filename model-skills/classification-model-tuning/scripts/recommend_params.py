# -*- coding: utf-8 -*-
"""规则式超参推荐: 给定 baseline 超参 + Diagnosis,生成调优后的超参。

按 algo 分流到 _recommend_xgb / _recommend_dnn / _recommend_lr,
各自的策略表只动该算法相关的超参键, 其他键原样保留。

## xgb 策略表
| 状态 | 动作 |
|------|------|
| underfit | depth+1, mcw/2, lr/2, n_est×1.5(各受 bounds 约束) |
| overfit | depth-1, reg_lambda×2, mcw×2 |
| underconverged | n_estimators ×2 (受上限 2000 约束) |
| unstable_psi | subsample -0.1, colsample_bytree -0.1 (受下限 0.5 约束) |
| well_fit | 微调: 不动 |

## dnn 策略表
| 状态 | 动作 |
|------|------|
| underfit | lr×2, dropout×0.7, epochs×1.5 |
| overfit | dropout+0.1, weight_decay×2 |
| underconverged | epochs×2, patience+5 |
| unstable_psi | dropout+0.1, batch_size×2 (减梯度噪声) |
| well_fit | 不动 |

## lr 策略表
| 状态 | 动作 |
|------|------|
| underfit | C×2 (减弱正则), max_n_bins+2 (更细 WoE 分箱) |
| overfit | C/2 (加强正则), max_n_bins-2 |
| underconverged | max_iter×2 |
| unstable_psi | max_n_bins-2, min_bin_size×2 (粗分箱更稳) |
| well_fit | 不动 |
"""
from __future__ import annotations

from typing import Any, Dict

from diagnose import Diagnosis

# 边界约束(经验性,防止"加容量"加飞)
BOUNDS_XGB = {
    "max_depth": (3, 10),
    "min_child_weight": (1, 200),
    "learning_rate": (0.005, 0.3),
    "n_estimators": (100, 2000),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.1, 10.0),
}

BOUNDS_DNN = {
    "dropout": (0.0, 0.6),
    "learning_rate": (1e-4, 1e-2),
    "weight_decay": (1e-6, 1e-2),
    "batch_size": (64, 2048),
    "epochs": (20, 500),
    "patience": (3, 50),
}

BOUNDS_LR = {
    "C": (1e-3, 1e3),
    "max_n_bins": (2, 30),
    "min_bin_size": (0.005, 0.30),
    "max_iter": (100, 5000),
}

# BOUNDS 作为 BOUNDS_XGB 的别名导出
BOUNDS = BOUNDS_XGB


def _clip(bounds: Dict[str, tuple], name: str, val: float) -> float:
    """按 bounds 限制取值;名字不在表中原样返回。"""
    if name not in bounds:
        return val
    lo, hi = bounds[name]
    return max(lo, min(hi, val))


def _ci(bounds: Dict[str, tuple], name: str, val: float) -> int:
    """clip 后取 int。"""
    return int(round(_clip(bounds, name, val)))


def _recommend_xgb(
    baseline_params: Dict[str, Any], diagnosis: Diagnosis
) -> Dict[str, Any]:
    new = dict(baseline_params)
    status = diagnosis.status
    B = BOUNDS_XGB
    if status == "underfit":
        new["max_depth"] = _ci(B, "max_depth", new.get("max_depth", 6) + 1)
        new["min_child_weight"] = max(
            1, _ci(B, "min_child_weight", new.get("min_child_weight", 20) / 2)
        )
        new["learning_rate"] = round(
            _clip(B, "learning_rate", new.get("learning_rate", 0.03) / 2), 4
        )
        new["n_estimators"] = _ci(B, "n_estimators", new.get("n_estimators", 800) * 1.5)
    elif status == "overfit":
        new["max_depth"] = _ci(B, "max_depth", new.get("max_depth", 6) - 1)
        new["reg_lambda"] = round(
            _clip(B, "reg_lambda", new.get("reg_lambda", 1.0) * 2), 4
        )
        new["min_child_weight"] = _ci(
            B, "min_child_weight", new.get("min_child_weight", 20) * 2
        )
    elif status == "underconverged":
        new["n_estimators"] = _ci(B, "n_estimators", new.get("n_estimators", 800) * 2)
    elif status == "unstable_psi":
        new["subsample"] = round(
            _clip(B, "subsample", new.get("subsample", 0.8) - 0.1), 3
        )
        new["colsample_bytree"] = round(
            _clip(B, "colsample_bytree", new.get("colsample_bytree", 0.8) - 0.1), 3
        )
    # well_fit: 不动
    return new


def _recommend_dnn(
    baseline_params: Dict[str, Any], diagnosis: Diagnosis
) -> Dict[str, Any]:
    new = dict(baseline_params)
    status = diagnosis.status
    B = BOUNDS_DNN
    if status == "underfit":
        new["learning_rate"] = round(
            _clip(B, "learning_rate", new.get("learning_rate", 0.001) * 2), 6
        )
        new["dropout"] = round(
            _clip(B, "dropout", new.get("dropout", 0.3) * 0.7), 3
        )
        new["epochs"] = _ci(B, "epochs", new.get("epochs", 100) * 1.5)
    elif status == "overfit":
        new["dropout"] = round(
            _clip(B, "dropout", new.get("dropout", 0.3) + 0.1), 3
        )
        new["weight_decay"] = round(
            _clip(B, "weight_decay", new.get("weight_decay", 1e-4) * 2), 8
        )
    elif status == "underconverged":
        new["epochs"] = _ci(B, "epochs", new.get("epochs", 100) * 2)
        new["patience"] = _ci(B, "patience", new.get("patience", 10) + 5)
    elif status == "unstable_psi":
        new["dropout"] = round(
            _clip(B, "dropout", new.get("dropout", 0.3) + 0.1), 3
        )
        new["batch_size"] = _ci(B, "batch_size", new.get("batch_size", 512) * 2)
    # well_fit: 不动
    return new


def _recommend_lr(
    baseline_params: Dict[str, Any], diagnosis: Diagnosis
) -> Dict[str, Any]:
    new = dict(baseline_params)
    status = diagnosis.status
    B = BOUNDS_LR
    if status == "underfit":
        new["C"] = round(_clip(B, "C", new.get("C", 1.0) * 2), 4)
        new["max_n_bins"] = _ci(B, "max_n_bins", new.get("max_n_bins", 8) + 2)
    elif status == "overfit":
        new["C"] = round(_clip(B, "C", new.get("C", 1.0) / 2), 4)
        new["max_n_bins"] = _ci(B, "max_n_bins", new.get("max_n_bins", 8) - 2)
    elif status == "underconverged":
        new["max_iter"] = _ci(B, "max_iter", new.get("max_iter", 1000) * 2)
    elif status == "unstable_psi":
        new["max_n_bins"] = _ci(B, "max_n_bins", new.get("max_n_bins", 8) - 2)
        new["min_bin_size"] = round(
            _clip(B, "min_bin_size", new.get("min_bin_size", 0.05) * 2), 4
        )
    # well_fit: 不动
    return new


def recommend(
    baseline_params: Dict[str, Any],
    diagnosis: Diagnosis,
    algo: str = "xgb",
) -> Dict[str, Any]:
    """根据 Diagnosis 调整 baseline_params,返回新超参 dict。

    Args:
        baseline_params: baseline 训练用的完整超参
        diagnosis: 诊断结果
        algo: 'xgb' | 'dnn' | 'lr'; dispatch 到对应推荐策略

    Returns:
        新超参 dict(原 dict 浅拷贝后被修改的若干键)
    """
    algo = (algo or "xgb").lower()
    if algo == "xgb":
        return _recommend_xgb(baseline_params, diagnosis)
    if algo == "dnn":
        return _recommend_dnn(baseline_params, diagnosis)
    if algo == "lr":
        return _recommend_lr(baseline_params, diagnosis)
    raise ValueError(f"未知 algo={algo!r}, 仅支持 xgb|dnn|lr")
