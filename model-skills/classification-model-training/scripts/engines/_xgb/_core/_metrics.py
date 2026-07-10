# -*- coding: utf-8 -*-
"""分类模型评估指标集合。

设计要点:
- BinMetrics dataclass: 单档结果结构化, 支持 to_dict() 序列化兼容
- MetricsReport 状态化: 一次构造多次 evaluate
- KS 走 sklearn.roc_curve 单源实现, 小样本标记 is_degraded 不抛异常
- evaluate_splits: 三档 split 一致口径评估
- _ks_via_roc 模块级函数: 训练循环零开销调用, 不必实例化 MetricsReport

边界: 完整评估能力 (lift / ROC 曲线 / 校准 / Brier / 月度 PSI / 趋势检验) 已委托
classification-model-evaluation skill, 本模块仅保留训练阶段必需的 AUC/KS/Gini 三件套。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class BinMetrics:
    """单档评估结果 (immutable)。

    与 dict 相比: 属性访问 (m.auc) / 类型注解 / frozen 防误改 / to_dict 兼容序列化。
    """

    auc: float
    ks: float
    gini: float
    n_pos: int
    n_neg: int
    n_total: int
    positive_rate: float
    is_degraded: bool

    def to_dict(self) -> dict:
        """转 dict 用于 config.json / report.md 序列化。"""
        return {
            "auc": float(self.auc),
            "ks": float(self.ks),
            "gini": float(self.gini),
            "n_pos": int(self.n_pos),
            "n_neg": int(self.n_neg),
            "n_total": int(self.n_total),
            "positive_rate": float(self.positive_rate),
            "is_degraded": bool(self.is_degraded),
        }


def _ks_via_roc(y_true, y_prob) -> float:
    """KS 统计量: max(|TPR - FPR|)。

    走 sklearn.metrics.roc_curve 单源实现, 避免手写 cumsum 的边界 bug。
    供训练循环 (DNN epoch 监控) 与 MetricsReport.evaluate 共用,
    不必实例化 MetricsReport, 零开销。

    Returns:
        KS 值, 退化场景 (单类样本) 返回 0.0
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))


class MetricsReport:
    """状态化分类模型评估器 (训练阶段口径)。

    设计要点:
    - 构造时固定 n_bins / random_state, 多次 evaluate 复用配置
    - evaluate 返回 BinMetrics (dataclass), 非散列 dict
    - evaluate_splits 一次算三档

    边界: 本类只算训练阶段需要的 AUC/KS/Gini, 不堆 lift / ROC 曲线 / 校准 / Brier
    等评估报告能力 — 那些归 classification-model-evaluation skill, 这里堆积会变成死代码。
    """

    def __init__(self, n_bins: int = 10, random_state: int = 42) -> None:
        """构造评估器。

        Args:
            n_bins: 预留 (训练阶段未使用, 完整分档评估走 evaluation skill)
            random_state: 预留 bootstrap 用 (当前未启用)
        """
        self.n_bins = n_bins
        self.random_state = random_state

    def evaluate(self, y_true, y_prob) -> BinMetrics:
        """单档评估: AUC / KS / Gini + 退化标记。

        退化场景 (n_pos<30 或 n_neg<30 或单类) 返回 is_degraded=True, 不抛异常,
        让 caller 自行决定是否跳过该档报告。
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        n_total = len(y_true)
        n_pos = int(y_true.sum())
        n_neg = n_total - n_pos
        positive_rate = float(n_pos / n_total) if n_total > 0 else 0.0

        if n_pos == 0 or n_neg == 0:
            return BinMetrics(
                auc=0.5, ks=0.0, gini=0.0,
                n_pos=n_pos, n_neg=n_neg, n_total=n_total,
                positive_rate=round(positive_rate, 6), is_degraded=True,
            )

        auc = float(roc_auc_score(y_true, y_prob))
        ks = _ks_via_roc(y_true, y_prob)
        gini = 2.0 * auc - 1.0
        is_degraded = (n_pos < 30) or (n_neg < 30) or (ks == 0.0) or (auc == 0.5)

        return BinMetrics(
            auc=round(auc, 6), ks=round(ks, 6), gini=round(gini, 6),
            n_pos=n_pos, n_neg=n_neg, n_total=n_total,
            positive_rate=round(positive_rate, 6), is_degraded=is_degraded,
        )

    def evaluate_splits(
        self,
        splits: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, BinMetrics]:
        """一次评估多档 (train/val/oot)。

        替代 caller 写 for 循环逐档调 evaluate, 三档口径完全一致。

        Args:
            splits: {"train": (y_true, y_prob), "val": (...), "oot": (...)}

        Returns:
            {"train": BinMetrics, "val": BinMetrics, "oot": BinMetrics}
        """
        return {name: self.evaluate(yt, yp) for name, (yt, yp) in splits.items()}
