# -*- coding: utf-8 -*-
"""explainability/ 子目录产出: feature-importance.csv + shap-summary.csv + _manifest.json。

- feature-importance.csv: 模型原生 importance(xgb gain / dnn 第一层权重均值 / lr 系数;
  各 trainer 暴露 get_feature_importance() 接口)
- shap-summary.csv: xgb 树模型用 shap.TreeExplainer 算 (mean_abs_shap / mean_shap);
  其他算法标 status=skipped
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from stages.eval_data import EvalData
from stages.layout import RunLayout, write_manifest

_SHAP_SAMPLE_SIZE = 1000  # OOT 采样上限,避免 KernelExplainer 类似的慢路径


def _write_importance_csv(importance: dict, out_path: Path) -> int:
    """写 feature-importance.csv,按 importance 降序。返回行数。"""
    rows = sorted(importance.items(), key=lambda kv: -float(kv[1]))
    df = pd.DataFrame(rows, columns=["feature", "importance"])
    df.to_csv(out_path, index=False)
    return len(df)


def _compute_shap_tree(
    predictor: Any, oot_df: pd.DataFrame, features: List[str]
) -> Optional[pd.DataFrame]:
    """对 xgb 用 shap.TreeExplainer 算 OOT 采样的 mean_abs_shap / mean_shap。"""
    try:
        import shap
    except ImportError:
        print("[explain] shap 未安装,跳过 shap-summary")
        return None
    # xgb: predictor.model(XGBClassifier) 或 predictor.booster
    inner = getattr(predictor, "model", None)
    booster = (
        getattr(predictor, "booster", None)   # xgb 直挂别名
        or inner                                # xgb.XGBClassifier 也可直接喂 TreeExplainer
    )
    if booster is None:
        print("[explain] predictor 未暴露 model/booster,跳过 shap")
        return None
    sample = oot_df[features]
    if len(sample) > _SHAP_SAMPLE_SIZE:
        sample = sample.sample(_SHAP_SAMPLE_SIZE, random_state=42)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(sample)
    if isinstance(values, list):  # 多分类返回 list, 二分类拿第一项
        values = values[0]
    arr = np.asarray(values)
    mean_abs = np.abs(arr).mean(axis=0)
    mean_sgn = arr.mean(axis=0)
    df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_sgn,
    }).sort_values("mean_abs_shap", ascending=False)
    return df


def write_explainability_stage(
    layout: RunLayout,
    predictor: Any,
    data: EvalData,
    produced_by: Optional[str] = None,
) -> None:
    """落 explainability/ 阶段产物: importance csv + shap csv(可选) + manifest。"""
    files: List[Path] = []
    extra: dict = {}

    imp_path = layout.explainability_dir / "feature-importance.csv"
    n_imp = _write_importance_csv(data.importance, imp_path)
    files.append(imp_path)
    extra["n_features_in_importance"] = n_imp

    shap_path = layout.explainability_dir / "shap-summary.csv"
    shap_df: Optional[pd.DataFrame] = None
    if layout.algo == "xgb":
        shap_df = _compute_shap_tree(predictor, data.oot_df, data.features)
        if shap_df is not None:
            shap_df.to_csv(shap_path, index=False)
            files.append(shap_path)
            extra["shap_sample_size"] = min(len(data.oot_df), _SHAP_SAMPLE_SIZE)
        else:
            extra["shap_skipped_reason"] = "未能从 predictor 取到 booster 或 shap 未装"
    else:
        extra["shap_skipped_reason"] = f"algo={layout.algo} 暂未接入 shap"

    write_manifest(
        layout.explainability_dir,
        stage="explainability",
        files=files,
        extra=extra,
        produced_by=produced_by,
    )
