# -*- coding: utf-8 -*-
"""model/ 子目录产出: 落盘训练好的模型 + _manifest.json。

按算法保留原生扩展名:
- xgb -> model.json (booster.save_model)
- dnn/lr -> model.pkl (pickle.dump)
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from stages.layout import RunLayout, write_manifest


def _ext_for(algo: str) -> str:
    if algo == "xgb":
        return "json"
    return "pkl"


def _write_meta_for_non_xgb(
    layout: RunLayout,
    predictor: Any,
    used_params: Dict[str, Any],
    train_info: Optional[Dict[str, Any]],
) -> Path:
    """为 dnn / lr 落 model_meta.json (xgb 由引擎自身落盘)。

    统一结构: {algo, feature_names, feature_importance, train_info, params, created_at}
    - feature_names: predictor.features
    - feature_importance: predictor.get_feature_importance() (dnn 空字典, lr 为 |coef|)
    - train_info: trainer 返回的训练细节 (dnn: best_epoch/...; lr: n_iter/converged/...)
    - params: used_params
    """
    meta_path = layout.model_dir / "model_meta.json"
    feature_names = list(getattr(predictor, "features", []) or [])
    try:
        feature_importance = predictor.get_feature_importance() or {}
    except Exception:
        feature_importance = {}
    payload = {
        "algo": layout.algo,
        "feature_names": feature_names,
        "feature_importance": dict(feature_importance),
        "train_info": train_info or {},
        "params": used_params or {},
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return meta_path


def save_model(predictor: Any, layout: RunLayout) -> Path:
    """按 layout.algo 落盘 predictor。

    Args:
        predictor: 训练好的预测器(xgb 暴露 save_model;其他 pickle 兜底)
        layout: RunLayout

    Returns:
        模型文件绝对路径
    """
    ext = _ext_for(layout.algo)
    path = layout.model_dir / f"model.{ext}"
    if layout.algo == "xgb":
        predictor.save_model(str(path))
    else:
        with path.open("wb") as f:
            pickle.dump(predictor, f)
    return path


def write_model_stage(
    layout: RunLayout,
    predictor: Any,
    used_params: Dict[str, Any],
    train_info: Optional[Dict[str, Any]] = None,
    produced_by: Optional[str] = None,
) -> Path:
    """落 model/ 阶段产物。

    Args:
        layout: RunLayout
        predictor: 训练好的预测器
        used_params: 实际训练超参(写入 manifest 便于追溯)
        train_info: 训练细节字典, 字段随 algo 不同
            (xgb: best_iteration; dnn: best_epoch/total_epochs/early_stopped/best_val_auc;
             lr: n_iter/converged)。整体写入 manifest, 下游按 algo 取字段。
        produced_by: manifest 来源标识(下游 skill 复用本函数时传值)

    Returns:
        模型文件路径

    Note:
        algo=lr 且 predictor 持有 scorecard_df 时, 额外落 model/scorecard.csv
        (列 [feature, bin, woe, coef, score]), 并在 manifest 标 has_scorecard=True。
    """
    model_path = save_model(predictor, layout)
    files: list = [model_path]
    # xgb 引擎在 save_model 时已落 model_meta.json; dnn / lr 在此补写
    meta_path = layout.model_dir / "model_meta.json"
    if not meta_path.exists() and layout.algo != "xgb":
        _write_meta_for_non_xgb(layout, predictor, used_params, train_info)
    if meta_path.exists():
        files.append(meta_path)

    # LR 评分卡: LrPredictor.scorecard_df 由 trainer 注入(algo=lr 时非空)
    scorecard_df = getattr(predictor, "scorecard_df", None)
    has_scorecard = scorecard_df is not None
    if has_scorecard:
        scorecard_path = layout.model_dir / "scorecard.csv"
        scorecard_df.to_csv(scorecard_path, index=False, encoding="utf-8")
        files.append(scorecard_path)

    extra: Dict[str, Any] = {
        "algo": layout.algo,
        "used_params": used_params,
        "has_scorecard": has_scorecard,
    }
    if train_info:
        extra["train_info"] = train_info
    write_manifest(layout.model_dir, stage="model", files=files, extra=extra,
                   produced_by=produced_by)
    return model_path
