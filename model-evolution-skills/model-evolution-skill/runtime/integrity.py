"""Trial 源码与预测工件的可审计完整性协议。

搜索代码必须先冻结再执行；预测必须携带样本身份与其产生环境的哈希。
这些约束把“同一个 trial 被悄悄改写”和“跨视图按行号错配”从统计
问题前移成确定性的完整性错误。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import SAMPLE_KEY_COLUMN
from .logging_utils import dump_json, load_yaml, now_iso

SOURCE_MANIFEST = "source-manifest.json"
PREDICTION_META_SUFFIX = ".meta.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _source_files(trial_dir: Path) -> list[Path]:
    trial_dir = Path(trial_dir)
    files: list[Path] = []
    for dirname in ("code", "config"):
        root = trial_dir / dirname
        if not root.is_dir():
            continue
        files.extend(p for p in root.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and not p.name.endswith((".pyc", ".pyo")))
    return sorted(files, key=lambda p: p.relative_to(trial_dir).as_posix())


def build_source_manifest(trial_dir: Path) -> dict[str, Any]:
    trial_dir = Path(trial_dir)
    entries = {
        p.relative_to(trial_dir).as_posix(): {
            "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in _source_files(trial_dir)
    }
    canonical = json.dumps(entries, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "files": entries,
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def freeze_trial_source(trial_dir: Path) -> dict[str, Any]:
    """首次调用落盘源码清单；以后源码有任何变化都 fail-closed。"""
    trial_dir = Path(trial_dir)
    path = trial_dir / SOURCE_MANIFEST
    current = build_source_manifest(trial_dir)
    if not current["files"]:
        raise ValueError(f"Trial 无可冻结源码:{trial_dir}")
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("combined_sha256") != current["combined_sha256"]:
            raise PermissionError(
                "Trial 源码在冻结后发生变化；同一 trial 不允许改写。"
                "请从该 trial 创建新的子 trial，并保留新 lineage")
        return frozen
    current["frozen_at"] = now_iso()
    dump_json(path, current)
    return current


def verify_trial_source(trial_dir: Path) -> dict[str, Any]:
    trial_dir = Path(trial_dir)
    path = trial_dir / SOURCE_MANIFEST
    if not path.exists():
        raise PermissionError(
            f"Trial 缺少 {SOURCE_MANIFEST}，不能评估或交付未冻结源码")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    current = build_source_manifest(trial_dir)
    if frozen.get("combined_sha256") != current["combined_sha256"]:
        raise PermissionError(
            "Trial 当前源码与冻结清单不一致，预测与源码无法建立可信绑定")
    return frozen


def execution_seed(trial_dir: Path) -> int:
    """返回源码实际消费的 seed，并拒绝 config/代码双写漂移。"""
    trial_dir = Path(trial_dir)
    cfg = load_yaml(trial_dir / "config" / "trial.yaml") or {}
    if cfg.get("seed") is None:
        raise PermissionError("Trial config 未显式记录 seed")
    value = int(cfg["seed"])
    text = (trial_dir / "code" / "train.py").read_text(encoding="utf-8")
    # 通用 trial 模板运行期以 config.seed 为权威；baseline 模板把 seed
    # 编译进模块常量。后者必须逐值一致，不能只在 YAML 中伪装新 seed。
    if "cfg.get(\"seed\"" in text or "cfg.get('seed'" in text:
        return value
    import re

    match = re.search(r"(?m)^SEED\s*=\s*([-+]?\d+)\s*$", text)
    if match is None:
        raise PermissionError("Trial 代码未声明可审计的执行 seed")
    code_seed = int(match.group(1))
    if code_seed != value:
        raise PermissionError("Trial config.seed 与代码实际 SEED 不一致")
    return code_seed


def verify_confirmation_seed(trial_dir: Path, parent_dir: Path) -> dict:
    """确认 Trial 必须与父代使用不同的、可审计的执行 seed。"""
    current = execution_seed(trial_dir)
    parent = execution_seed(parent_dir)
    if current == parent:
        raise PermissionError("confirm 必须使用不同于父代的新 seed")
    return {"current_seed": current, "parent_seed": parent,
            "independent": True}


def prediction_meta_path(score_path: Path) -> Path:
    score_path = Path(score_path)
    return score_path.with_name(score_path.name + PREDICTION_META_SUFFIX)


def bind_prediction_artifact(
    score_path: Path,
    labels: pd.DataFrame,
    *,
    source_manifest: dict[str, Any],
    split_manifest_path: Path,
    data_view: str,
    model_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """给预测补全样本键并写不可含糊的 provenance sidecar。"""
    score_path = Path(score_path)
    pred = pd.read_parquet(score_path)
    if "score" not in pred.columns:
        raise ValueError(f"预测工件缺少 score 列:{score_path}")
    if len(pred) != len(labels):
        raise ValueError(
            f"预测工件 {len(pred)} 行 != 标签面 {len(labels)} 行")
    if SAMPLE_KEY_COLUMN not in labels.columns:
        raise ValueError("标签面缺稳定样本键，不能绑定预测身份")
    keys = labels[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True)
    if keys.duplicated().any():
        raise ValueError("标签面样本键重复，预测不可配对")
    if SAMPLE_KEY_COLUMN in pred.columns:
        existing = pred[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True)
        if not existing.equals(keys):
            raise ValueError("预测工件自带样本键与权威标签面不一致")
    else:
        pred[SAMPLE_KEY_COLUMN] = keys.values

    split_hash = sha256_file(split_manifest_path) \
        if Path(split_manifest_path).exists() else None
    model_hashes = {
        str(Path(p).name): sha256_file(Path(p))
        for p in (model_paths or []) if Path(p).is_file()
    }
    model_canonical = json.dumps(model_hashes, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
    model_hash = hashlib.sha256(model_canonical).hexdigest() \
        if model_hashes else None
    meta = {
        "schema_version": 1,
        "created_at": now_iso(),
        "prediction_file": score_path.name,
        "rows": len(pred),
        "sample_key_column": SAMPLE_KEY_COLUMN,
        "sample_key_sha256": hashlib.sha256(
            "\n".join(keys).encode("utf-8")).hexdigest(),
        "source_manifest_sha256": source_manifest.get("combined_sha256"),
        "split_manifest_sha256": split_hash,
        "data_view": str(data_view),
        "model_sha256": model_hash,
        "model_files": model_hashes,
    }
    # 预测文件本身只重复存样本键；其余标量放 sidecar，避免百万行上
    # 重复几十字节哈希造成不必要的磁盘放大。
    pred.to_parquet(score_path, index=False)
    meta["prediction_sha256"] = sha256_file(score_path)
    dump_json(prediction_meta_path(score_path), meta)
    return meta


def verify_prediction_artifact(score_path: Path) -> dict[str, Any]:
    score_path = Path(score_path)
    meta_path = prediction_meta_path(score_path)
    if not meta_path.exists():
        raise PermissionError(f"预测缺完整性 sidecar:{meta_path.name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("prediction_sha256") != sha256_file(score_path):
        raise PermissionError(f"预测文件在绑定后被修改:{score_path}")
    pred = pd.read_parquet(score_path, columns=[SAMPLE_KEY_COLUMN])
    if pred[SAMPLE_KEY_COLUMN].astype(str).duplicated().any():
        raise PermissionError("预测样本键重复")
    return meta
