# -*- coding: utf-8 -*-
"""LR 评分卡模型定义与训练函数。

从 `engines/_lr/entry.py` 提取, 供 `trainers/train_lr.py` 复用。
评估/报告逻辑已委托 `classification-model-evaluation`, 此处只保留训练核心。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from engines._lr._dataset import WoeBinner  # noqa: E402

logger = logging.getLogger(__name__)


class ScoreCardConverter:
    """将 LR 模型 + WoE 转换为标准评分卡。

    评分卡公式：
        Score = base_score - factor * ln(odds)
        factor = pdo / ln(2)

    每个特征每个分箱的分数贡献：
        score_i = -(coef_i * woe_ij + intercept/n) * factor
    """

    def __init__(
        self,
        base_score: int = 600,
        pdo: int = 50,
        base_odds: float = 50.0,
    ) -> None:
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def convert(
        self,
        lr_model: LogisticRegression,
        woe_encoder: WoeBinner,
    ) -> pd.DataFrame:
        """生成评分卡表。

        Returns:
            DataFrame with columns [feature, bin, woe, coef, score]
        """
        coefs = lr_model.coef_[0]
        intercept = lr_model.intercept_[0]
        n_features = len(woe_encoder.fitted_features)

        rows = []
        for i, feat in enumerate(woe_encoder.fitted_features):
            coef = coefs[i]
            fmap = woe_encoder.feature_maps[feat]

            for bin_label, woe_val in zip(fmap.bin_edges, fmap.woe_values):
                # 分数 = -(系数 × WoE + 截距均摊) × factor
                score_contribution = -(coef * woe_val + intercept / n_features) * self.factor
                rows.append({
                    "feature": feat,
                    "bin": bin_label,
                    "woe": round(woe_val, 6),
                    "coef": round(coef, 6),
                    "score": round(score_contribution, 2),
                })

        return pd.DataFrame(rows)

    def predict_score(
        self,
        lr_model: LogisticRegression,
        X_woe: pd.DataFrame,
    ) -> np.ndarray:
        """预测评分卡分数（越高越好/风险越低）。"""
        # log_odds = X @ coef + intercept
        log_odds = X_woe.values @ lr_model.coef_[0] + lr_model.intercept_[0]
        scores = self.offset - self.factor * log_odds
        return scores


def train_lr_model(
    X_train_woe: pd.DataFrame,
    y_train: np.ndarray,
    X_val_woe: pd.DataFrame,
    y_val: np.ndarray,
    regularization: str = "l2",
    C: float = 1.0,
    max_iter: int = 1000,
    woe_encoder: Optional[WoeBinner] = None,
    base_score: int = 600,
    pdo: int = 50,
    base_odds: float = 50.0,
) -> Tuple[LogisticRegression, Dict[str, Any], Optional[pd.DataFrame]]:
    """训练 LR 模型。

    Args:
        X_train_woe: WoE 编码后的训练集
        y_train: 训练集标签
        X_val_woe: WoE 编码后的验证集
        y_val: 验证集标签
        regularization: 正则化类型
        C: 正则化强度
        max_iter: 最大迭代次数
        woe_encoder: 已 fit 的 WoeBinner; 传入时同步生成评分卡 DataFrame, 否则返回 None
        base_score: 评分卡基准分(如 600)
        pdo: odds 翻倍对应的分数差(如 50)
        base_odds: 基准分对应的 odds(如 50:1)

    Returns:
        (lr_model, training_info, scorecard_df_or_None)
        - scorecard_df: 列 [feature, bin, woe, coef, score]; woe_encoder 为 None 时返回 None
    """
    # 选择 solver
    if regularization == "l1":
        solver = "saga"
        penalty = "l1"
    elif regularization == "elasticnet":
        solver = "saga"
        penalty = "elasticnet"
    else:
        solver = "lbfgs"
        penalty = "l2"

    lr_params = {
        "penalty": penalty,
        "C": C,
        "solver": solver,
        "max_iter": max_iter,
        "random_state": 42,
        "n_jobs": -1,
    }
    if penalty == "elasticnet":
        lr_params["l1_ratio"] = 0.5

    model = LogisticRegression(**lr_params)
    model.fit(X_train_woe, y_train)

    # 训练信息
    train_prob = model.predict_proba(X_train_woe)[:, 1]
    val_prob = model.predict_proba(X_val_woe)[:, 1]
    train_auc = roc_auc_score(y_train, train_prob)
    val_auc = roc_auc_score(y_val, val_prob)

    info = {
        "n_features": X_train_woe.shape[1],
        "regularization": regularization,
        "C": C,
        "train_auc": round(train_auc, 6),
        "val_auc": round(val_auc, 6),
        "converged": bool(model.n_iter_[0] < max_iter),
        "n_iter": int(model.n_iter_[0]),
    }

    logger.info(
        f"LR 训练完成: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, "
        f"迭代 {model.n_iter_[0]} 轮, {'收敛' if info['converged'] else '⚠️ 未收敛'}"
    )

    scorecard_df: Optional[pd.DataFrame] = None
    if woe_encoder is not None:
        converter = ScoreCardConverter(
            base_score=base_score, pdo=pdo, base_odds=base_odds,
        )
        scorecard_df = converter.convert(model, woe_encoder)
        logger.info(
            f"评分卡已生成: {len(scorecard_df)} 行 (base_score={base_score}, "
            f"pdo={pdo}, base_odds={base_odds})"
        )

    return model, info, scorecard_df
