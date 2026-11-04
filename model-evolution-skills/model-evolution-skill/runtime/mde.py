"""v2.17.0:量纲感知的最小有意义提升(MDE)解析。

事故(run4 regression.2,原值金额回归):``evaluator.minimum_meaningful_improvement``
默认 0.001 是**绝对值**,对 AUC/log1p rmse 合理,对原值 rmse(≈12932 元)毫无
意义——不手工改成 10 元,所有 Δ 都"有意义",穷尽判定、HPO 边际递减门、
confirm 稳定性门与交付容差全部失效。

口径:
- ``minimum_meaningful_improvement``(绝对值,旧键,语义不变);
- ``minimum_meaningful_improvement_relative``(v2.17.0 新键,默认 None):
  设置时 ``MDE = max(absolute, relative × |baseline 主指标|)``。AUC≈0.81 时
  0.0008×0.81 < 0.001,绝对值仍生效;原值 rmse 12932 时 → 10.3。
- 新 run(skill_version ≥ 2.17.0)init 时把 relative 写进 run 配置
  (``runtime.protocol_gates.V2_17_0_RUN_CONFIG``);代码默认与
  assets/global-config.yaml 均为 None,旧 run 配置里没有该键,行为不变。

所有消费 MDE 的位置统一调 :func:`effective_mde`,不要再直接读配置键。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

#: 旧默认(绝对值)
DEFAULT_ABSOLUTE_MDE = 0.001
#: 量纲告警阈:absolute < ratio × |baseline| 且未设 relative → 视为量纲错配
SCALE_WARNING_RATIO = 1e-4


def _finite(x: Any) -> "float | None":
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def baseline_metric_from_run(run_dir: "Path | str | None") -> "float | None":
    """run-state.baseline_id → trials/<id>/evaluation.json 的主指标值。"""
    if not run_dir:
        return None
    rd = Path(run_dir)
    try:
        state = json.loads((rd / "run-state.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    bid = (state or {}).get("baseline_id")
    if not bid:
        return None
    ev_path = rd / "trials" / str(bid) / "evaluation.json"
    try:
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return _finite(((ev or {}).get("primary_metric") or {}).get("value"))


def mde_info(config, *, run_dir: "Path | str | None" = None,
             baseline_metric: "float | None" = None) -> dict:
    """返回 MDE 解析明细(供 planner-context / 报告呈现)。"""
    absolute = _finite(config.get(
        "evaluator.minimum_meaningful_improvement", DEFAULT_ABSOLUTE_MDE))
    if absolute is None or absolute <= 0:
        absolute = DEFAULT_ABSOLUTE_MDE
    relative = _finite(config.get(
        "evaluator.minimum_meaningful_improvement_relative", None))
    if baseline_metric is None:
        baseline_metric = baseline_metric_from_run(run_dir)
    base = _finite(baseline_metric)
    value, source = absolute, "absolute"
    if relative and relative > 0 and base is not None:
        rel_value = relative * abs(base)
        if rel_value > absolute:
            value, source = rel_value, "relative"
    warning = None
    if (not relative) and base is not None and \
            absolute < SCALE_WARNING_RATIO * abs(base):
        warning = (
            f"MDE 量纲疑似错配:绝对 MDE={absolute:g} < {SCALE_WARNING_RATIO:g}"
            f"×|baseline 主指标|={abs(base):g};原值量纲的回归指标请设置 "
            "evaluator.minimum_meaningful_improvement(绝对)或 "
            "evaluator.minimum_meaningful_improvement_relative(相对)")
    return {"value": float(value), "source": source,
            "absolute": float(absolute), "relative": relative,
            "baseline_metric": base, "scale_warning": warning}


def effective_mde(config, *, run_dir: "Path | str | None" = None,
                  baseline_metric: "float | None" = None) -> float:
    """当前 run 的有效 MDE(主指标单位,方向无关的正数)。"""
    return mde_info(config, run_dir=run_dir,
                    baseline_metric=baseline_metric)["value"]
