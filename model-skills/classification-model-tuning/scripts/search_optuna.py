# -*- coding: utf-8 -*-
"""Optuna 贝叶斯搜索: 在 baseline 周围 ±ratio 的搜索空间内,
跑 n_trials 次训练,取 val_auc 最优的超参。

按 algo 分流到 _build_space_xgb / _build_space_dnn / _build_space_lr,
对应的 train_fn 由调用方传入 (与 trainer_dispatch 同协议):
- xgb: tune_train.train_with_params (返回 trainer, metrics)
- dnn: train_dnn.train_dnn_model (返回 predictor, metrics, info)
- lr : train_lr.train_lr_model (返回 predictor, metrics, info)

为统一调用, train_fn 必须满足: train_fn(params=params, **common) -> (predictor|trainer, metrics)
本模块对 dnn/lr 调用方传入的 lambda 自动丢弃 info (无需 info, 用 best metrics 即可)。

惰性 import optuna,缺包时抛清晰错误。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

# 各 algo 的搜索空间键 (仅这些键会被 optuna suggest)
XGB_SPACE_KEYS = [
    "max_depth", "min_child_weight", "learning_rate",
    "subsample", "colsample_bytree", "reg_lambda",
]
DNN_SPACE_KEYS = [
    "dropout", "learning_rate", "weight_decay",
    "batch_size", "epochs", "patience",
]
LR_SPACE_KEYS = [
    "C", "max_n_bins", "min_bin_size", "max_iter",
]


def _ri(base: int, lo_clip: int, hi_clip: int, ratio: float) -> Tuple[int, int, str]:
    lo = max(lo_clip, int(round(base * (1 - ratio))))
    hi = min(hi_clip, int(round(base * (1 + ratio))))
    if hi <= lo:
        hi = lo + 1
    return (lo, hi, "int")


def _rf(base: float, lo_clip: float, hi_clip: float, ratio: float) -> Tuple[float, float, str]:
    lo = max(lo_clip, base * (1 - ratio))
    hi = min(hi_clip, base * (1 + ratio))
    if hi <= lo:
        hi = lo + 1e-3
    return (lo, hi, "float")


def _build_space_xgb(baseline_params: Dict[str, Any], ratio: float) -> Dict[str, Tuple]:
    """xgb: 在 baseline 周围 ±ratio 构造搜索空间。"""
    return {
        "max_depth": _ri(int(baseline_params.get("max_depth", 6)), 3, 10, ratio),
        "min_child_weight": _ri(
            int(baseline_params.get("min_child_weight", 20)), 1, 200, ratio
        ),
        "learning_rate": _rf(
            float(baseline_params.get("learning_rate", 0.03)), 0.005, 0.3, ratio
        ),
        "subsample": _rf(float(baseline_params.get("subsample", 0.8)), 0.5, 1.0, ratio),
        "colsample_bytree": _rf(
            float(baseline_params.get("colsample_bytree", 0.8)), 0.5, 1.0, ratio
        ),
        "reg_lambda": _rf(float(baseline_params.get("reg_lambda", 1.0)), 0.1, 10.0, ratio),
    }


def _build_space_dnn(baseline_params: Dict[str, Any], ratio: float) -> Dict[str, Tuple]:
    """dnn: dropout/lr/weight_decay/batch_size/epochs/patience, ±ratio baseline。"""
    return {
        "dropout": _rf(float(baseline_params.get("dropout", 0.3)), 0.0, 0.6, ratio),
        "learning_rate": _rf(
            float(baseline_params.get("learning_rate", 0.001)), 1e-4, 1e-2, ratio
        ),
        "weight_decay": _rf(
            float(baseline_params.get("weight_decay", 1e-4)), 1e-6, 1e-2, ratio
        ),
        "batch_size": _ri(int(baseline_params.get("batch_size", 512)), 64, 2048, ratio),
        "epochs": _ri(int(baseline_params.get("epochs", 100)), 20, 500, ratio),
        "patience": _ri(int(baseline_params.get("patience", 10)), 3, 50, ratio),
    }


def _build_space_lr(baseline_params: Dict[str, Any], ratio: float) -> Dict[str, Tuple]:
    """lr: C/max_n_bins/min_bin_size/max_iter, ±ratio baseline。"""
    return {
        "C": _rf(float(baseline_params.get("C", 1.0)), 1e-3, 1e3, ratio),
        "max_n_bins": _ri(int(baseline_params.get("max_n_bins", 8)), 2, 30, ratio),
        "min_bin_size": _rf(
            float(baseline_params.get("min_bin_size", 0.05)), 0.005, 0.30, ratio
        ),
        "max_iter": _ri(int(baseline_params.get("max_iter", 1000)), 100, 5000, ratio),
    }


def _build_space(
    baseline_params: Dict[str, Any], ratio: float, algo: str = "xgb"
) -> Dict[str, Tuple]:
    """按 algo dispatch 到对应 space builder。"""
    algo = (algo or "xgb").lower()
    if algo == "xgb":
        return _build_space_xgb(baseline_params, ratio)
    if algo == "dnn":
        return _build_space_dnn(baseline_params, ratio)
    if algo == "lr":
        return _build_space_lr(baseline_params, ratio)
    raise ValueError(f"未知 algo={algo!r}, 仅支持 xgb|dnn|lr")


def search(
    baseline_params: Dict[str, Any],
    train_fn: Callable[..., Tuple[Any, Dict[str, Dict[str, float]]]],
    train_common: Dict[str, Any],
    n_trials: int = 30,
    ratio: float = 0.30,
    seed: int = 42,
    algo: str = "xgb",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """跑 Optuna 搜索,目标 = val_auc 最大化。

    Args:
        baseline_params: 搜索中心
        train_fn: 训练函数, 签名 train_fn(params=params, **train_common) -> (predictor, metrics)
            xgb 路径用 tune_train.train_with_params 原样即可;
            dnn/lr 路径调用方需用 lambda 包装丢弃 info:
                lambda params, **kw: train_dnn_model(params=params, **kw)[:2]
        train_common: 透传给 train_fn 的 train_path/test_path/oot_path/target/features 等
        n_trials: 搜索次数
        ratio: 各超参在 baseline 周围 ±ratio 内取值
        seed: TPE sampler 种子
        algo: 'xgb' | 'dnn' | 'lr'; 决定搜索空间

    Returns:
        (best_params, trials_log)
        - best_params: baseline_params 合并上 best 试验中的可调键
        - trials_log: [{trial_number, params, val_auc, oot_auc}]
    """
    try:
        import optuna
    except ImportError as e:
        raise SystemExit(
            "[search_optuna] 未安装 optuna,请: pip install --user 'optuna<4' 后再用 --method optuna"
        ) from e

    space = _build_space(baseline_params, ratio, algo=algo)
    trials_log: List[Dict[str, Any]] = []

    def objective(trial: "optuna.trial.Trial") -> float:
        params = dict(baseline_params)
        for name, (lo, hi, kind) in space.items():
            if kind == "int":
                params[name] = trial.suggest_int(name, lo, hi)
            else:
                params[name] = trial.suggest_float(name, lo, hi)
        _predictor, metrics = train_fn(params=params, **train_common)
        # metrics: Dict[str, BinMetrics] (dataclass, 通过 .auc/.ks 属性访问)
        val_auc = float(metrics["val"].auc)
        oot_auc = float(metrics["oot"].auc)
        trials_log.append({
            "trial_number": trial.number,
            "params": {k: params[k] for k in space},
            "val_auc": val_auc,
            "oot_auc": oot_auc,
        })
        return val_auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(baseline_params)
    best.update(study.best_params)
    return best, trials_log
