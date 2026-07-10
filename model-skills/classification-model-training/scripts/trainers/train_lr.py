# -*- coding: utf-8 -*-
"""model-training LR 评分卡训练: 用 engines._lr 引擎的 WoE+LR + 三段式数据。

与 xgb(tune_train)/dnn(train_dnn)路径对齐, 提供算法无关的 predictor 协议产物
(predict_proba(df[features]) -> np.ndarray), 供 eval_data 复用。
本封装把 WoE 编码(WoeBinner) + LogisticRegression 打包进 LrPredictor,
使下游对比与 xgb/dnn 完全同构。

数据切分由 run_build 按 model.split 内部完成, 本脚本直接读三档 parquet,
test 当 val 做 LR 评估。

模块隔离: lr 引擎的代码是自洽超集(_core 含 MetricsReport/DriftProbe,
_dataset 含 WoeBinner, entry 内联 train_lr_model), import 时用
engines._lr._dataset / engines._lr._core / engines._lr._model 即可。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from engines._lr._dataset import WoeBinner  # noqa: E402
from engines._lr._core import MetricsReport  # noqa: E402
from engines._lr._model import train_lr_model as _engine_train_lr  # noqa: E402


# LR 默认超参: 对齐 lr 引擎 argparse 默认值(WoE 分箱 max_n_bins 8 /
# min_bin_size 0.05, L2 正则 C=1.0, max_iter 1000)。WoE 编码器只做编码,
# 不做 IV 自动筛选 —— model-training 按用户显式 features 入模, 不做自动剔除。
LR_PARAMS = {
    "max_n_bins": 8,
    "min_bin_size": 0.05,
    "regularization": "l2",
    "C": 1.0,
    "max_iter": 1000,
}


@dataclass
class LrPredictor:
    """算法无关预测器: 把 WoE 编码 + LR 打包, 暴露 predict_proba(df)。

    与 XgbFitter/DnnPredictor 同构: 下游 eval_data 只依赖
    predict_proba(df[features]) 与可选 get_feature_importance()。本对象可直接
    pickle(WoeBinner/LogisticRegression 均可序列化), 故模型落盘为 .pkl。

    Attributes:
        woe_encoder: 已 fit 的 WoeBinner(持有各特征 OptimalBinning 与 WoE 映射)
        model: 已训练的 LogisticRegression(输入为 WoE 编码后的特征)
        features: 入模原始特征(WoeBinner.fitted_features, 列名+顺序与训练一致)
        scorecard_df: 评分卡表(列 [feature, bin, woe, coef, score]);
            algo=lr 时由 trainer 注入, 其他场景为 None。落盘为 model/scorecard.csv。
    """

    woe_encoder: WoeBinner
    model: LogisticRegression
    features: List[str] = field(default_factory=list)
    scorecard_df: Optional[pd.DataFrame] = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """对原始特征 DataFrame 做 WoE 编码 -> LR 预测, 返回正类概率。

        Args:
            df: 含 self.features 列的 DataFrame(通常为 oot_df[features])

        Returns:
            正类概率 np.ndarray, shape=(n,)
        """
        X_woe = self.woe_encoder.transform(df)
        return self.model.predict_proba(X_woe)[:, 1]

    def get_feature_importance(self) -> dict:
        """以 |系数| 作为重要性(输入已 WoE 标准化, 系数可直接比较)。"""
        return {
            feat: float(abs(c))
            for feat, c in zip(self.features, self.model.coef_[0])
        }


def train_lr_model(
    train_path: str, test_path: str, oot_path: str,
    target: str, features: List[str], params: dict,
) -> tuple:
    """用 WoE+LR 训练 + 评估 Train/Val/OOT(三段式, test 当 val 仅报训练 AUC)。

    Args:
        train_path: 训练段 parquet(已清洗为数值列)
        test_path: 测试段 parquet(已清洗),作为 LR 评估集
        oot_path: OOT 段 parquet(已清洗)
        target: 标签列
        features: 入模特征
        params: LR 超参字典(见 LR_PARAMS)

    Returns:
        (predictor, metrics, info)
        - predictor: LrPredictor
        - metrics: {train,val,oot: {auc,ks}}
        - info: 引擎 train_lr_model 返回的训练信息(n_iter/converged 等)
    """
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(test_path)
    oot_df = pd.read_parquet(oot_path)
    print(f"[lr] 切分: train={len(train_df)} val={len(val_df)} oot={len(oot_df)}")

    # WoE 编码: 仅用训练段 fit(防泄漏), 编码器只做编码不做 IV 筛选
    woe = WoeBinner(
        max_n_bins=int(params["max_n_bins"]),
        min_bin_size=float(params["min_bin_size"]),
    )
    y_train = train_df[target].to_numpy()
    woe.fit(train_df, y_train, features)
    if not woe.fitted_features:
        raise RuntimeError("无特征通过 WoE 分箱, 无法训练 LR(检查特征是否全为常数/缺失)")
    print(f"[lr] WoE 编码: {len(woe.fitted_features)}/{len(features)} 个特征入模")

    X_train_woe = woe.transform(train_df)
    X_val_woe = woe.transform(val_df)
    y_val = val_df[target].to_numpy()

    # 复用引擎 train_lr_model: 训练 LR + 返回训练信息 + 评分卡 DataFrame
    # 传入 woe_encoder 触发 ScoreCardConverter 生成评分卡, 落盘为 model/scorecard.csv
    model, info, scorecard_df = _engine_train_lr(
        X_train_woe, y_train, X_val_woe, y_val,
        regularization=str(params["regularization"]),
        C=float(params["C"]), max_iter=int(params["max_iter"]),
        woe_encoder=woe,
    )

    predictor = LrPredictor(
        woe_encoder=woe, model=model,
        features=list(woe.fitted_features),
        scorecard_df=scorecard_df,
    )

    reporter = MetricsReport()
    splits = {
        "train": (train_df[target].to_numpy(), predictor.predict_proba(train_df)),
        "val": (val_df[target].to_numpy(), predictor.predict_proba(val_df)),
        "oot": (oot_df[target].to_numpy(), predictor.predict_proba(oot_df)),
    }
    m = reporter.evaluate_splits(splits)
    for k in ("train", "val", "oot"):
        print(f"[lr] {k}: AUC={m[k].auc:.4f} KS={m[k].ks:.4f}")
    return predictor, m, info
