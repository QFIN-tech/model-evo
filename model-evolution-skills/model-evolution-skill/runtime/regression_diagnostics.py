"""v2.17.0:回归任务的水位与指标背离诊断(只诊断,不进 verdict)。

事故(run4 regression.2,原值金额回归):
- final_test 预测均值 2350 vs 实际 1965(+20%)——7 月水位下移,模型排序可用、
  绝对金额需校准,但 evaluation / shadow 预检 / 报告里都没有这类信号;
- 022/023/024 的 rmse 与 MAE 方向相反(如 023:rmse +5.1、MAE −10.6),主指标
  掩盖了中低额段的变化,只能靠 agent 手算发现。

本模块提供纯函数,由 evaluator(搜索期)、finalize/evaluate-final(交付期)与
report 消费。所有输出都是诊断字段,不改变任何裁决。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: 水位比告警阈:|pred_mean / y_mean − 1| 超过该值记 level_shift 告警
LEVEL_RATIO_WARN = 0.10
#: v2.17.0(D8):水位比噪声判据的 z。告警需 |ratio−1| > max(LEVEL_RATIO_WARN,
#: LEVEL_RATIO_Z·SE(ratio)) —— 重尾目标的期均值本身波动大(实测每期 n≈2223、
#: 独立同分布的合成数据,期均值波动约 18%),只看固定 ±10% 会把抽样噪声误报成
#: 水位偏离
LEVEL_RATIO_Z = 2.0
#: 单期最少行数(过小的期均值噪声大,不出告警)
MIN_PERIOD_ROWS = 50


def _f(x: Any) -> "float | None":
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def level_ratio_se(y, p) -> "float | None":
    """水位比 R = mean(p)/mean(y) 的标准误(delta 法,含 y/p 配对协方差)。

    比值估计量的线性化:R̂ − R ≈ mean(p − R·y)/ȳ,故
    SE(R̂) = sd(p_i − R̂·y_i) / (√n · |ȳ|)。等价于对 ȳ、p̄ 的样本均值标准误
    做 delta 法并计入二者协方差(预测与真实高度相关时不重复计噪声)。
    n<2 或 ȳ=0 时返回 None(调用方退回固定阈值)。
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = int(y.size)
    if n < 2:
        return None
    y_mean = float(y.mean())
    if y_mean == 0:
        return None
    r = float(p.mean()) / y_mean
    resid = p - r * y
    se = float(np.std(resid, ddof=1) / (math.sqrt(n) * abs(y_mean)))
    return se if math.isfinite(se) else None


def level_stats(y_true, y_pred, *, z: float | None = None) -> dict:
    """整体水位:预测均值、真实均值、比值(总额比)与均值偏置。

    v2.17.0(D8):附 ``level_ratio_se``(delta 法);告警判据
    |ratio−1| > max(LEVEL_RATIO_WARN, z·SE),``level_ratio_threshold`` 为生效阈。
    """
    z = LEVEL_RATIO_Z if z is None else float(z)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        return {"n": 0}
    y_mean, p_mean = float(y.mean()), float(p.mean())
    ratio = (p_mean / y_mean) if y_mean != 0 else None
    se = level_ratio_se(y, p) if ratio is not None else None
    thr = max(LEVEL_RATIO_WARN, z * se) if se is not None else LEVEL_RATIO_WARN
    return {
        "n": int(y.size),
        "y_mean": y_mean,
        "pred_mean": p_mean,
        "mean_bias": p_mean - y_mean,
        "level_ratio": ratio,
        "level_ratio_se": se,
        "level_ratio_z": z,
        "level_ratio_threshold": thr,
        # 超过固定阈但在噪声带内(z·SE)→ 不告警,单列标记便于呈现
        "level_within_noise": bool(
            ratio is not None and abs(ratio - 1.0) > LEVEL_RATIO_WARN
            and abs(ratio - 1.0) <= thr),
        "level_shift_warning": bool(
            ratio is not None and abs(ratio - 1.0) > thr),
    }


def level_by_period(y_true, y_pred, periods, *,
                    z: float | None = None) -> list[dict]:
    """按期(如 pday/月)分组的水位比;periods 与 y 同长。

    v2.17.0(D8):每期独立计算 ``level_ratio_se``,告警判据同
    :func:`level_stats`(|ratio−1| > max(LEVEL_RATIO_WARN, z·SE))。
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    per = np.asarray(periods)
    out: list[dict] = []
    if per.shape[0] != y.shape[0]:
        return out
    for key in sorted({str(v) for v in per}):
        mask = per.astype(str) == key
        if int(mask.sum()) < MIN_PERIOD_ROWS:
            continue
        st = level_stats(y[mask], p[mask], z=z)
        st["period"] = key
        out.append(st)
    return out


def metric_divergence(delta_primary: "float | None",
                      delta_secondary: "float | None", *,
                      primary: str = "rmse", secondary: str = "mae",
                      se_secondary: "float | None" = None,
                      z: float = 2.0) -> dict:
    """主/次指标对同一对照(如 vs_parent)方向相反时告警。

    Δ 为"当前 − 对照"的原始差;按各自指标方向换算成"改善量"后比较符号
    (rmse/mae/mape 越小越好,r2 越大越好)。次指标 Δ 只在超过 ``z·SE``
    (若给出 SE)时才算有方向,避免噪声级背离刷屏。
    """
    from .metrics import metric_direction

    dp, ds = _f(delta_primary), _f(delta_secondary)
    if dp is None or ds is None or dp == 0 or ds == 0:
        return {"diverged": False, "primary": primary, "secondary": secondary}
    gain_p = dp if metric_direction(primary) else -dp
    gain_s = ds if metric_direction(secondary) else -ds
    se = _f(se_secondary)
    significant = True if se is None or se <= 0 else abs(ds) > z * se
    diverged = bool(significant and (gain_p > 0) != (gain_s > 0))
    return {
        "diverged": diverged,
        "primary": primary, "secondary": secondary,
        "delta_primary": dp, "delta_secondary": ds,
        "secondary_significant": bool(significant),
        "note": ("主/次指标方向相反:主指标可能掩盖了另一部分样本(如中低额段)"
                 "的退化或改善,裁决仍只看主指标") if diverged else None,
    }


def regression_diagnostics(y_true, y_pred, *, periods=None,
                           parent_pred=None, primary: str = "rmse",
                           secondary: str = "mae") -> dict:
    """搜索期/交付期统一入口。``parent_pred`` 给出时附带主/次指标背离诊断。"""
    out: dict = {"level": level_stats(y_true, y_pred)}
    # v2.17.0(D8):整体与分期告警分开呈现(消费方不得混写)
    out["overall_level_shift"] = bool(out["level"].get("level_shift_warning"))
    if periods is not None:
        out["level_by_period"] = level_by_period(y_true, y_pred, periods)
        out["period_level_shift"] = any(
            r.get("level_shift_warning") for r in out["level_by_period"])
        out["period_level_shift_periods"] = [
            r.get("period") for r in out["level_by_period"]
            if r.get("level_shift_warning")]
    if parent_pred is not None:
        from .metrics import compute_metric

        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_pred, dtype=float)
        q = np.asarray(parent_pred, dtype=float)
        if q.shape == p.shape == y.shape:
            def _m(name, a):
                return float(compute_metric(name, y, a, score=a))

            se_sec = None
            if str(secondary).lower() == "mae" and y.size > 1:
                # MAE 差的配对 SE:逐行绝对误差差的标准误
                d_abs = np.abs(y - p) - np.abs(y - q)
                se_sec = float(np.std(d_abs, ddof=1) / math.sqrt(d_abs.size))
            try:
                out["vs_parent_divergence"] = metric_divergence(
                    _m(primary, p) - _m(primary, q),
                    _m(secondary, p) - _m(secondary, q),
                    primary=primary, secondary=secondary,
                    se_secondary=se_sec)
            except Exception as exc:  # noqa: BLE001 —— 诊断不阻塞
                out["vs_parent_divergence"] = {"diverged": False,
                                               "error": str(exc)}
    return out
