"""v2.19.0:天花板估计(ceiling estimate)的落盘契约与读侧工具。

生产者:``runtime.reference``(``modelevo.py ceiling --run``)在 baseline 之后
用固定墙钟跑等预算强参照(强默认 GBDT + seed bagging,可选 AutoGluon),
把结果写到 ``<run>/ceiling/ceiling-estimate.json``。

消费者:stop report(``gap_to_ref`` 判停)、planner-context(展示与预算分配)、
benchmark 报告(``lift_capture``)。消费方只经本模块读取,不直接解析文件。

口径:
- 参照只在 **validation** 面上评估(与搜索 trial 同一裁决面,不读 shadow /
  final_test 标签);
- 参照里允许 seed bagging / AutoML 集成——它只是"这份数据在当前特征面上
  大致能到哪里"的测量,不是可交付 trial,不受单模型拓扑门约束;
- ``feature_surface`` 记录参照用的特征面(``base`` = profile 材料化的特征面;
  ``trial:<id>`` = 该 trial 的特征矩阵)。base 面参照与加宽后的冠军不完全
  同口径,gap 只作为下界提示,读侧会带上 surface 供解释。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CEILING_DIR = "ceiling"
CEILING_FILE = "ceiling-estimate.json"


def ceiling_path(run_dir: "Path | str") -> Path:
    return Path(run_dir) / CEILING_DIR / CEILING_FILE


def write_ceiling(run_dir: "Path | str", payload: dict) -> Path:
    """原子写入 ceiling-estimate.json(补 schema 版本)。

    payload 必填键:``primary_metric``、``higher_is_better``、
    ``ceiling_estimate``(float)、``references``(dict:名称 → {value, ...})。
    """
    for key in ("primary_metric", "higher_is_better", "ceiling_estimate",
                "references"):
        if key not in payload:
            raise ValueError(f"ceiling payload 缺少必填键 {key}")
    out = {"schema": SCHEMA_VERSION, "surface": "validation", **payload}
    path = ceiling_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_ceiling(run_dir: "Path | str | None") -> dict | None:
    """读 ceiling-estimate.json;不存在/损坏/值非有限时返回 None。"""
    if not run_dir:
        return None
    path = ceiling_path(run_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 —— 读侧容错:没有天花板估计就当不存在
        return None
    try:
        val = float(data.get("ceiling_estimate"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    data["ceiling_estimate"] = val
    return data


def gap_to_ref(ceiling: "dict | None", value: Any,
               higher_is_better: "bool | None" = None) -> float | None:
    """方向感知的剩余差距:正数 = 离参照还差多少(主指标单位)。

    负数表示已超过参照。任一输入缺失时返回 None。
    """
    if not ceiling:
        return None
    try:
        v = float(value)
        c = float(ceiling["ceiling_estimate"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (math.isfinite(v) and math.isfinite(c)):
        return None
    hib = ceiling.get("higher_is_better") if higher_is_better is None \
        else higher_is_better
    return (c - v) if hib in (True, None) else (v - c)


def lift_capture(*, base: Any, achieved: Any, reference: Any,
                 higher_is_better: bool = True) -> float | None:
    """(achieved − base)/(reference − base),方向感知;分母 ≤ 0 返回 None。"""
    try:
        b, a, r = float(base), float(achieved), float(reference)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (b, a, r)):
        return None
    num, den = (a - b, r - b) if higher_is_better else (b - a, b - r)
    if den <= 0:
        return None
    return num / den


def ceiling_summary(run_dir: "Path | str | None", champion_value: Any = None,
                    *, mde: "float | None" = None) -> dict | None:
    """planner-context / stop report 共用的摘要;无估计时返回 None。"""
    c = load_ceiling(run_dir)
    if c is None:
        return None
    gap = gap_to_ref(c, champion_value) if champion_value is not None \
        else None
    out = {
        "ceiling_estimate": c["ceiling_estimate"],
        "best_reference": c.get("best_reference"),
        "feature_surface": c.get("feature_surface", "base"),
        "baseline_value": c.get("baseline_value"),
        "champion_value": champion_value,
        "gap_to_ref": gap,
        # v2.19.0:单模型口径参照(与交付同口径)及集成溢价
        "single_model_estimate": c.get("single_model_estimate"),
        "gap_to_single_model_ref": (
            gap_to_ref({**c, "ceiling_estimate": c["single_model_estimate"]},
                       champion_value)
            if champion_value is not None
            and c.get("single_model_estimate") is not None else None),
        "ensemble_premium": c.get("ensemble_premium"),
        "computed_at": c.get("computed_at"),
    }
    if gap is not None and mde:
        try:
            out["gap_in_mde"] = gap / float(mde)
            out["within_mde"] = bool(gap <= float(mde))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return out
