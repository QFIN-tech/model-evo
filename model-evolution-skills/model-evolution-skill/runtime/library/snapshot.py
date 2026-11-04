"""v2.18.0 经验库:快照载荷的契约与校验(dev 文档 §5.1)。

快照是一个 run(或一批历史数据)向库提交的不可变单元。``snapshot_id`` =
规范 JSON 的 sha256,重复提交同一内容幂等。
"""
from __future__ import annotations

import re

from .artifacts import canonical_json, is_sha256, sha256_hex

SNAPSHOT_SCHEMA = "modelevo.snapshot/1"
#: curation:v2.18.2 异步整理,由服务端把一个主题组的现行原则写成快照(consolidation.py)
SNAPSHOT_KINDS = ("run", "baseline", "legacy_principles", "legacy_evidence", "legacy_recipes",
                  "curation")
#: 不参与 snapshot_id 的易变字段
VOLATILE_KEYS = ("submitted_at",)

TRIAL_COLUMNS = (
    "run_id", "trial_id", "atid", "generation", "action", "algorithm",
    "parent_trial_ids", "changed_modules", "family_hints", "data_view",
    "topics", "hypothesis", "status", "verdict", "primary_metric",
    "higher_is_better", "metric_value", "delta_parent", "delta_parent_ci_low",
    "delta_parent_ci_high", "p_positive", "delta_baseline", "failure_reason",
    "experience_ids", "recipe_id")
RECIPE_COLUMNS = (
    "recipe_id", "atid", "algorithm", "kind", "source_run_id",
    "source_trial_id", "runtime_api_version", "derived_from", "derivation",
    "is_hpo", "n_trees")
RECIPE_KINDS = ("top_trial", "baseline")
MEASUREMENT_COLUMNS = (
    "recipe_id", "run_id", "trial_id", "kind", "protocol", "primary_metric",
    "higher_is_better", "value", "baseline_value", "delta_baseline", "gain",
    "ci_low", "ci_high", "p_positive", "confirmed")
MEASUREMENT_KINDS = ("export", "warm_remeasure", "baseline", "legacy")
GOVERNANCE_COLUMNS = ("subject_type", "subject_id", "action", "payload", "reason")

#: 任务指纹量级段(spec.py 的分桶)
SAMPLE_SCALES = ("lt10w", "10w_50w", "50w_200w", "200w_1000w", "ge1000w")
_HASH8 = re.compile(r"^[0-9a-f]{8}$")


def snapshot_id(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k not in VOLATILE_KEYS}
    return sha256_hex(canonical_json(body))


def run_id_for(run_dir) -> str:
    """run 的库内标识:``<目录名>@<绝对路径 sha256 前 8 位>``。

    目录名本身会重名(run1/run-req-t7-rand400w 与 run2/run-req-t7-rand400w),
    只用路径又太长;两者结合既可读又唯一。
    """
    import os

    path = os.path.abspath(str(run_dir)).rstrip("/")
    return f"{os.path.basename(path)}@{sha256_hex(path.encode('utf-8'))[:8]}"


def parse_atid(atid: str | None) -> dict | None:
    """``{model_type}-{target}-{scale}-{col_hash8}`` → 各段;target 可含 ``-``。

    ``data_base``(``{scale}-{col_hash8}``)只在量级与哈希都合法时给出 ——
    ``unknown-unverified``、``pending`` 等降级指纹没有可信底座。
    """
    if not atid or not isinstance(atid, str) or "-" not in atid:
        return None
    model_type, rest = atid.split("-", 1)
    parts = rest.rsplit("-", 2)
    if len(parts) != 3:
        return None
    target, scale, col_hash = parts
    valid = scale in SAMPLE_SCALES and bool(_HASH8.match(col_hash))
    return {"atid": atid, "model_type": model_type, "target_name": target,
            "sample_scale": scale, "col_hash8": col_hash,
            "data_base": f"{scale}-{col_hash}" if valid else None}


def _check_rows(rows, name: str, required: tuple, errors: list) -> None:
    if rows is None:
        return
    if not isinstance(rows, list):
        errors.append(f"{name} 必须是列表")
        return
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{name}[{i}] 必须是对象")
            continue
        for key in required:
            if row.get(key) in (None, ""):
                errors.append(f"{name}[{i}] 缺 {key}")


def validate_snapshot(payload) -> list[str]:
    """返回错误列表(空 = 合法)。只做结构校验;产物存在性由仓库在提交时检查。"""
    if not isinstance(payload, dict):
        return ["快照必须是 JSON 对象"]
    errors: list[str] = []
    if payload.get("schema") != SNAPSHOT_SCHEMA:
        errors.append(f"schema 必须是 {SNAPSHOT_SCHEMA}")
    if payload.get("kind") not in SNAPSHOT_KINDS:
        errors.append(f"kind 必须是 {SNAPSHOT_KINDS} 之一")
    run = payload.get("run")
    if not isinstance(run, dict):
        errors.append("run 必须是对象")
    _check_rows(payload.get("discoveries"), "discoveries", ("local_id",), errors)
    for i, d in enumerate(payload.get("discoveries") or []):
        if isinstance(d, dict) and not isinstance(d.get("payload"), dict):
            errors.append(f"discoveries[{i}].payload 必须是对象")
    _check_rows(payload.get("trial_outcomes"), "trial_outcomes",
                ("run_id", "trial_id"), errors)
    _check_rows(payload.get("recipes"), "recipes", ("recipe_id", "kind"), errors)
    for i, r in enumerate(payload.get("recipes") or []):
        if isinstance(r, dict):
            if r.get("kind") not in RECIPE_KINDS:
                errors.append(f"recipes[{i}].kind 必须是 {RECIPE_KINDS} 之一")
            if not is_sha256(r.get("recipe_id", "")):
                errors.append(f"recipes[{i}].recipe_id 不是 sha256")
    _check_rows(payload.get("measurements"), "measurements",
                ("recipe_id", "kind"), errors)
    for i, m in enumerate(payload.get("measurements") or []):
        if isinstance(m, dict) and m.get("kind") not in MEASUREMENT_KINDS:
            errors.append(f"measurements[{i}].kind 必须是 {MEASUREMENT_KINDS} 之一")
    _check_rows(payload.get("governance"), "governance",
                ("subject_type", "subject_id", "action"), errors)
    for key in ("discoveries", "trial_outcomes", "recipes", "measurements"):
        ids = [r.get({"discoveries": "local_id", "trial_outcomes": "trial_id",
                      "recipes": "recipe_id"}.get(key, "")) for r in
               payload.get(key) or [] if isinstance(r, dict)]
        ids = [x for x in ids if x]
        if key != "measurements" and len(ids) != len(set(ids)):
            errors.append(f"{key} 中有重复标识")
    return errors
