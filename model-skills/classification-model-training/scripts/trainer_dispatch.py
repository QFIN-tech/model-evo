# -*- coding: utf-8 -*-
"""算法分派: 按 model.algo 选择对应 trainer,返回统一四元组。

供 model-training 的 run_build.py 与 model-tuning 的 run_tuning.py 共用。
common 字典字段: train_path/test_path/oot_path/target/features
(数据切分由 run_build 按 model.split 内部完成, 本层直接读三档 parquet 路径)。

第四返回值 train_info 是各算法的训练细节字典(透传 trainer 的 info),
字段随 algo 不同:
- xgb: {"best_iteration": int|None}  (early stopping 最优轮)
- dnn: {"best_epoch": int, "total_epochs": int, "early_stopped": bool, "best_val_auc": float}
- lr:  {"n_iter": int, "converged": bool, ...}
下游 config.json.runtime / model/_manifest.json / report.md 按 algo 取字段展示。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def dispatch_train(
    algo: str,
    common: Dict[str, Any],
    params_override: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """按 algo 选 trainer 模块,返回 (predictor, metrics, used_params, train_info)。

    Args:
        algo: 'xgb' | 'dnn' | 'lr'
        common: 训练通用参数 (train_path/test_path/oot_path/target/features)
        params_override: 覆盖默认超参;None 时各 trainer 用自带 _PARAMS

    Returns:
        (predictor, metrics, used_params, train_info)
        - train_info: 训练细节字典, 字段见模块 docstring
    """
    if algo == "xgb":
        from trainers.tune_train import TUNED_PARAMS, train_with_params
        params = params_override if params_override is not None else TUNED_PARAMS
        trainer, metrics = train_with_params(params=params, **common)
        # xgb trainer 暴露 best_iteration_ (early stopping 最优轮), 透传为 dict
        train_info = {"best_iteration": getattr(trainer, "best_iteration_", None)}
        return trainer, metrics, params, train_info
    if algo == "dnn":
        from trainers.train_dnn import DNN_PARAMS, train_dnn_model
        params = params_override if params_override is not None else DNN_PARAMS
        predictor, metrics, info = train_dnn_model(params=params, **common)
        # info 含 best_epoch/total_epochs/early_stopped/best_val_auc, 整体透传
        return predictor, metrics, params, dict(info) if info else {}
    if algo == "lr":
        from trainers.train_lr import LR_PARAMS, train_lr_model
        params = params_override if params_override is not None else LR_PARAMS
        predictor, metrics, info = train_lr_model(params=params, **common)
        # info 含 n_iter/converged 等, 整体透传
        return predictor, metrics, params, dict(info) if info else {}
    raise ValueError(f"未知 algo={algo!r}, 仅支持 xgb|dnn|lr")
