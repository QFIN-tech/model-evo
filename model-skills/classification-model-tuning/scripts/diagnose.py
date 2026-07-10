# -*- coding: utf-8 -*-
"""规则诊断: 基于 baseline 的 metrics + PSI + 训练动力学信号推断模型状态。

输出 `Diagnosis(status, reasons, signals)`,供 recommend_params 决定调参方向。

支持算法:
- xgb: 训练动力学信号 = best_iteration / n_estimators
- dnn: 训练动力学信号 = best_epoch / epochs / early_stopped
- lr : 训练动力学信号 = converged (凸优化极少 underconverged, 兜底用)

overfit / underfit / unstable_psi 三条规则 algo-agnostic, 只看 AUC gap 与 PSI。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 阈值: 与 CLAUDE.md 一致(PSI>0.10 警戒);其他来自调参经验
GAP_OVERFIT = 0.05         # train_auc - oot_auc 超过此值视为过拟合
GAP_UNDERFIT = 0.005       # train_auc - oot_auc 低于此值 → 欠拟合(独立触发, 不要求 train_auc 低)
TRAIN_LOW_AUC = 0.70       # train_auc 低于此值视为欠拟合 (信号叠加)
PSI_WARN = 0.10            # 与 CLAUDE.md 规范对齐
EARLY_STOP_RATIO = 0.95    # best_iteration / n_estimators 比例 (xgb/dnn 共用)


@dataclass(frozen=True)
class Diagnosis:
    """诊断结论(主要状态 + 多条触发原因 + 中间信号)。"""

    status: str                        # 'underfit' | 'overfit' | 'underconverged' | 'unstable_psi' | 'well_fit'
    reasons: List[str] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """用于落入 config.json runtime。"""
        return {"status": self.status, "reasons": list(self.reasons),
                "signals": dict(self.signals)}


def _safe_get(d: Dict[str, Any], *keys, default: Optional[float] = None) -> Optional[float]:
    """逐层取值,中间缺失返回默认。"""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _check_underconverged_xgb(
    used_params: Dict[str, Any], best_iteration: Optional[int]
) -> Optional[str]:
    """xgb: best_iteration / n_estimators >= 0.95 → 未收敛。"""
    n_estimators = used_params.get("n_estimators")
    if best_iteration is None or not n_estimators:
        return None
    ratio = best_iteration / n_estimators
    if ratio >= EARLY_STOP_RATIO:
        return (
            f"best_iteration={best_iteration} / n_estimators={n_estimators} "
            f"= {ratio:.2%} ≥ {EARLY_STOP_RATIO:.0%}(未收敛)"
        )
    return None


def _check_underconverged_dnn(
    used_params: Dict[str, Any], best_iteration: Optional[int]
) -> Optional[str]:
    """dnn: early_stopped=False 或 best_epoch/epochs >= 0.95 → 未收敛。

    best_iteration 在 dnn 路径里映射到 best_epoch (trainer_dispatch 已处理)。
    """
    epochs = used_params.get("epochs")
    early_stopped = used_params.get("early_stopped")
    # 显式 early_stopped=False 强烈提示未收敛 (跑了全部 epochs 仍未触 early stop)
    if early_stopped is False and epochs:
        return (
            f"early_stopped=False, 跑满 epochs={epochs} 未触 early stopping(未收敛)"
        )
    if best_iteration is not None and epochs:
        ratio = best_iteration / epochs
        if ratio >= EARLY_STOP_RATIO:
            return (
                f"best_epoch={best_iteration} / epochs={epochs} "
                f"= {ratio:.2%} ≥ {EARLY_STOP_RATIO:.0%}(未收敛)"
            )
    return None


def _check_underconverged_lr(
    used_params: Dict[str, Any], best_iteration: Optional[int]
) -> Optional[str]:
    """lr: converged=False → 未收敛 (凸优化罕见, 兜底)。

    best_iteration 在 lr 路径里映射到 n_iter (trainer_dispatch 已处理)。
    """
    converged = used_params.get("converged")
    max_iter = used_params.get("max_iter")
    n_iter = best_iteration
    if converged is False:
        return (
            f"converged=False, sklearn LR 在 max_iter={max_iter} 内未收敛(n_iter={n_iter})"
        )
    return None


def diagnose(
    metrics: Dict[str, Dict[str, float]],
    used_params: Dict[str, Any],
    best_iteration: Optional[int],
    new_psi: Optional[float] = None,
    algo: str = "xgb",
) -> Diagnosis:
    """规则诊断。优先级 overfit > underfit > underconverged > unstable_psi > well_fit
    (同一 baseline 可能多条触发,status 取最先命中的,其他作为 reasons)。

    Args:
        metrics: {"train": {"auc": ...}, "val": {...}, "oot": {...}}
        used_params: baseline 训练超参; underconverged 检查会读 algo 相关字段
            (xgb: n_estimators; dnn: epochs/early_stopped; lr: converged/max_iter)
        best_iteration: early stopping 命中的最佳 iteration/epoch/n_iter
        new_psi: 训练→OOT psi 分数稳定性
        algo: 'xgb' | 'dnn' | 'lr'; 影响训练动力学诊断分流

    Returns:
        Diagnosis 实例
    """
    algo = (algo or "xgb").lower()
    train_auc = _safe_get(metrics, "train", "auc")
    oot_auc = _safe_get(metrics, "oot", "auc")

    signals: Dict[str, float] = {}
    if train_auc is not None:
        signals["train_auc"] = float(train_auc)
    if oot_auc is not None:
        signals["oot_auc"] = float(oot_auc)
    if train_auc is not None and oot_auc is not None:
        signals["train_oot_gap"] = float(train_auc - oot_auc)
    if new_psi is not None:
        signals["new_psi"] = float(new_psi)
    # 早期收敛信号: xgb/dnn 用 ratio; lr 不算 (n_iter 语义不同)
    if algo == "xgb" and best_iteration is not None and used_params.get("n_estimators"):
        signals["early_stop_ratio"] = float(best_iteration) / float(used_params["n_estimators"])
    elif algo == "dnn" and best_iteration is not None and used_params.get("epochs"):
        signals["early_stop_ratio"] = float(best_iteration) / float(used_params["epochs"])

    reasons: List[str] = []
    statuses_hit: List[str] = []

    # overfit / underfit (algo-agnostic, 只看 AUC gap)
    if train_auc is not None and oot_auc is not None:
        gap = train_auc - oot_auc
        if gap > GAP_OVERFIT:
            statuses_hit.append("overfit")
            reasons.append(f"train-oot AUC gap={gap:.4f} > {GAP_OVERFIT}(过拟合)")
        elif gap < GAP_UNDERFIT or train_auc < TRAIN_LOW_AUC:
            statuses_hit.append("underfit")
            if gap < GAP_UNDERFIT and train_auc < TRAIN_LOW_AUC:
                reasons.append(
                    f"train AUC={train_auc:.4f} < {TRAIN_LOW_AUC} 且 train-oot gap={gap:.4f} "
                    f"< {GAP_UNDERFIT}(欠拟合)"
                )
            elif gap < GAP_UNDERFIT:
                reasons.append(
                    f"train-oot gap={gap:.4f} < {GAP_UNDERFIT}"
                    f"(欠拟合: 训练都没学进去)"
                )
            else:
                reasons.append(f"train AUC={train_auc:.4f} < {TRAIN_LOW_AUC}(欠拟合)")

    # underconverged (algo-specific)
    under_msg: Optional[str] = None
    if algo == "xgb":
        under_msg = _check_underconverged_xgb(used_params, best_iteration)
    elif algo == "dnn":
        under_msg = _check_underconverged_dnn(used_params, best_iteration)
    elif algo == "lr":
        under_msg = _check_underconverged_lr(used_params, best_iteration)
    if under_msg:
        statuses_hit.append("underconverged")
        reasons.append(under_msg)

    # unstable_psi (algo-agnostic)
    if new_psi is not None and new_psi > PSI_WARN:
        statuses_hit.append("unstable_psi")
        reasons.append(f"train→oot PSI={new_psi:.4f} > {PSI_WARN}(分数不稳定 [PSI_WARN])")

    if not statuses_hit:
        return Diagnosis(status="well_fit", reasons=["指标在合理区间"], signals=signals)
    return Diagnosis(status=statuses_hit[0], reasons=reasons, signals=signals)
