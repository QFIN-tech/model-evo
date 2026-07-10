# -*- coding: utf-8 -*-
"""分数漂移探针。

设计要点:
- SplitPair dataclass: 结构化承载 (baseline, comparison) 输入, 替代散列 ndarray 入参
- 单源 score_psi: 等频分箱(quantile histogram), 直接复用 baseline 分位数,
  避免边界去重后 bin 数缩水的隐式行为
- DriftProbe 单职责: 只算 PSI, 不混入 bootstrap/Mann-Kendall/月度分析
  (这些功能在本项目零消费, 留待需要时单独建模块, 不在 probe 里堆积)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitPair:
    """两组分数的对比对 (immutable)。

    命名清晰 / 可附 name 标签 / frozen 防误改 / 支持序列化 (to_dict 配套下游 report)。
    """

    baseline: np.ndarray
    comparison: np.ndarray
    name: str = ""

    def __post_init__(self) -> None:
        """校验: 至少 1 个样本, 否则 PSI 无定义。"""
        if len(self.baseline) == 0:
            raise ValueError(f"SplitPair '{self.name}': baseline 为空, PSI 无定义")
        if len(self.comparison) == 0:
            raise ValueError(f"SplitPair '{self.name}': comparison 为空, PSI 无定义")


class DriftProbe:
    """分数漂移探针 (单职责: 只算 PSI)。

    - 类只暴露 score_psi, 不堆积 bootstrap/Mann-Kendall/月度分析 (本仓库零消费)
    - 构造参数只有 n_bins, 无 epsilon (用 clip 防 log(0), 不污染信号)
    - score_psi 接 SplitPair 或裸 ndarray, 调用方两种风格都能用
    """

    def __init__(self, n_bins: int = 10) -> None:
        """构造探针。

        Args:
            n_bins: 等频分箱数 (baseline 分位数切分)
        """
        self.n_bins = n_bins

    def score_psi(
        self,
        baseline,
        comparison=None,
    ) -> float:
        """计算两组预测分数之间的 PSI (等频分箱单源实现)。

        算法: pd.qcut 等频分箱 + clip 下限 (不污染信号),
        等频分箱保证每箱 baseline 占比 = 1/n_bins, PSI 退化为对比集分布偏差的纯度量。

        Args:
            baseline: 基准期分数 (ndarray / Series / SplitPair)
            comparison: 对比期分数; baseline 为 SplitPair 时忽略

        Returns:
            PSI 值 (非负, 越大漂移越严重)
        """
        if isinstance(baseline, SplitPair):
            pair = baseline
        else:
            pair = SplitPair(
                baseline=np.asarray(baseline),
                comparison=np.asarray(comparison),
            )

        base = pair.baseline.astype(float)
        comp = pair.comparison.astype(float)

        # 等频分箱: 基于 baseline 分位数切 n_bins 个箱
        # duplicates='drop' 处理 baseline 重复值导致的边界合并
        bins = pd.qcut(base, q=self.n_bins, duplicates="drop", retbins=True)[1]
        if len(bins) < 3:
            # baseline 严重退化 (单值或两值), 退化为按唯一值分箱
            bins = np.unique(base)
            if len(bins) < 2:
                return 0.0
            bins = np.concatenate([[-np.inf], bins[1:-1] + 0.5 * np.diff(bins), [np.inf]])
        else:
            bins[0] = -np.inf
            bins[-1] = np.inf

        base_hist = np.histogram(base, bins=bins)[0]
        comp_hist = np.histogram(comp, bins=bins)[0]

        base_prop = base_hist / base_hist.sum()
        comp_prop = comp_hist / comp_hist.sum()

        # clip 防 log(0): 不加 epsilon, 避免小箱被噪声主导
        eps = 1e-12
        base_prop = np.clip(base_prop, eps, None)
        comp_prop = np.clip(comp_prop, eps, None)

        return float(np.sum((comp_prop - base_prop) * np.log(comp_prop / base_prop)))
