# -*- coding: utf-8 -*-
"""Baseline 训练评估: 固定超参 XGB, 三档预测 + 三档评估落盘。

产物契约:
- baseline/predictions/{split}_predictions.parquet: id_cols + label_col + xgb_score
  (xgb_score 是 G6 融合模型的 benchmark 输入)
- baseline/evaluation/baseline_{split}_eval.{json,md}: AUC/KS/十分桶(model-skills 口径)
- baseline/model/model.json + model_meta.json(含特征列清单, 供复现)
"""
from __future__ import annotations

import pandas as pd

from evo_core import metrics, paths, state_io, xgb_utils

PRODUCED_BY = "feature-mining-skills/feature-evolution-orchestration"


def run_baseline(
    session_dir,
    splits: dict,
    contract: dict,
    feature_cols: list,
    mining_name: str,
    xgb_params: dict | None = None,
    baseline_feature_cols: list | None = None,
) -> dict:
    """训练 baseline 并落盘全部产物, 返回 {split: eval_bundle}。

    Args:
        session_dir: session 根目录
        splits: {train/test/oot: DataFrame}(temporal_split 产出)
        contract: sample_contract.parse_sample_contract 产出
        feature_cols: 候选特征列(内部再筛数值列)
        mining_name: 模型/任务名(报告标题用)
        xgb_params: 缺省用 evo_core.xgb_utils.DEFAULT_XGB_PARAMS
        baseline_feature_cols: 冻结 champion 为指定特征子集
            (如饱和场景下 = 既有强模型的入模特征)。未传(None)时走
            原逻辑(select_numeric_features(全部候选))。传了且未配 base_model_score_col
            时, baseline 特征固定为该子集(仍过 select_numeric 兜底校验数值化),
            model_meta.feature_cols 落该子集 -> probe_roi 冗余扫描与 G3 对照锚定该子集。
    """
    label_col = contract["label_col"]
    score_col = contract.get("base_model_score_col") or ""
    if score_col:
        # 评估模式(有既有模型): baseline 特征 = 既有模型打分列本身;
        # 融合判定 = 固定 XGB 在 [既有分 + 候选特征] 上 OOT 是否超既有分单独。
        numeric_features = [score_col]
    elif baseline_feature_cols:
        # 冻结 champion 为指定子集(饱和场景复现既有强模型口径)
        present = [c for c in baseline_feature_cols if c in splits["train"].columns]
        numeric_features = xgb_utils.select_numeric_features(splits["train"], present)
        if not numeric_features:
            raise ValueError("baseline_feature_cols 指定子集无数值特征可用(检查列名/数据)")
    else:
        numeric_features = xgb_utils.select_numeric_features(splits["train"], feature_cols)
    if not numeric_features:
        raise ValueError("baseline 无数值特征可用(候选列全部不可数值化)")

    X_train = xgb_utils.as_numeric_frame(splits["train"], numeric_features)
    y_train = splits["train"][label_col]
    model = xgb_utils.train_xgb(X_train, y_train, params=xgb_params)

    # 模型落盘 + meta(特征列清单是复现与 G6 融合模型对齐的关键)
    model_dir = paths.baseline_model_dir(session_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "model.json"))
    state_io.write_json(
        model_dir / "model_meta.json",
        {
            "schema_version": 1,
            "produced_by": PRODUCED_BY,
            "algo": "xgb",
            "mode": "base_score_fusion" if score_col else "raw_features",
            "base_model_score_col": score_col or None,
            "params": xgb_params or xgb_utils.DEFAULT_XGB_PARAMS,
            "feature_cols": numeric_features,
            "n_features": len(numeric_features),
            "train_rows": int(len(splits["train"])),
        },
    )

    bundles = {}
    for split, sub in splits.items():
        X = xgb_utils.as_numeric_frame(sub, numeric_features)
        scores = xgb_utils.predict_scores(model, X)

        pred = sub[contract["id_cols"] + [label_col]].copy()
        pred["xgb_score"] = scores
        pred_path = paths.baseline_predictions_parquet(session_dir, split)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(pred_path, index=False)

        bundle = metrics.eval_bundle(sub[label_col], scores)
        bundles[split] = bundle
        model_meta = {"name": mining_name, "version": "baseline", "algo": "xgb", "split": split}
        state_io.write_json(
            paths.baseline_eval_json(session_dir, split),
            metrics.build_eval_json(PRODUCED_BY, model_meta, split, bundle),
        )
        md = metrics.build_eval_md("%s(baseline)" % mining_name, split, bundle)
        emd = paths.baseline_eval_md(session_dir, split)
        emd.parent.mkdir(parents=True, exist_ok=True)
        emd.write_text(md, encoding="utf-8")
    return bundles
