# -*- coding: utf-8 -*-
"""session 状态读写: state.json / ledger.jsonl / _manifest.json / .done 标记。

- state.json: 断点续跑核心, 记录进化全局状态(champion / best_oot_auc / 轮次 / 连续无接受)
- ledger.jsonl: append-only 候选台账, 一行一条候选记录(含指纹去重)
- _manifest.json: 与 model-skills 同规范(schema_version / produced_by / files)
- .done: 阶段完成标记(model-skills 同款, 存在即完成, 缺失即半途中断)
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj: dict) -> None:
    """原子写 json(先写临时文件再替换), 避免中断留下半个文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---- .done 标记 ----
def mark_done(dir_path) -> None:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    (p / ".done").touch()


def clear_done(dir_path) -> None:
    p = Path(dir_path) / ".done"
    if p.exists():
        p.unlink()


def is_done(dir_path) -> bool:
    return (Path(dir_path) / ".done").exists()


# ---- _manifest.json ----
def write_manifest(dir_path, produced_by: str, files: list, overview: dict | None = None) -> None:
    """在目录下落 _manifest.json。

    Args:
        dir_path: 产物目录
        produced_by: 形如 feature-mining-skills/<skill_name>
        files: 该目录关键产物文件名列表
        overview: 概览信息(行数/指标等), 可选
    """
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "produced_by": produced_by,
        "produced_at": _now_iso(),
        "files": sorted(files),
        "overview": overview or {},
    }
    write_json(Path(dir_path) / "_manifest.json", manifest)


# ---- state.json ----
def init_state(session_dir, mining_name: str, config_snapshot: dict) -> dict:
    """初始化 state.json( baseline 指标由 prepare_session 后续回填)。"""
    from . import paths

    state = {
        "schema_version": SCHEMA_VERSION,
        "mining_name": mining_name,
        "status": "initialized",
        "created_at": _now_iso(),
        "current_round": 0,
        "champion_fids": [],
        "champion_oot_auc": None,
        "champion_oot_ks": None,
        "baseline_oot_auc": None,
        "baseline_oot_ks": None,
        "no_accept_streak": 0,
        "accepted_count": 0,
        "next_fid": 1,
        "semantic_columns": [],
        "config_snapshot": config_snapshot,
    }
    write_json(paths.state_json(session_dir), state)
    return state


def load_state(session_dir) -> dict:
    from . import paths

    p = paths.state_json(session_dir)
    if not p.exists():
        raise FileNotFoundError(
            "state.json 不存在: %s。session 尚未初始化, 请先跑 prepare_session.py" % p
        )
    return read_json(p)


def save_state(session_dir, state: dict) -> None:
    from . import paths

    write_json(paths.state_json(session_dir), state)


def update_state(session_dir, **fields) -> dict:
    """读-改-写 state.json 的便捷封装。"""
    state = load_state(session_dir)
    state.update(fields)
    save_state(session_dir, state)
    return state


# ---- ledger.jsonl(append-only) ----
def append_ledger(session_dir, record: dict) -> None:
    """追加一条候选记录。record 至少含 cid/verdict; ts 由本函数补齐。

    幂等约定: 同 cid 已存在记录时调用方负责先判重(见 ledger_has_cid),
    本函数只做追加, 不重写历史行。
    """
    from . import paths

    p = paths.ledger_jsonl(session_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("ts", _now_iso())
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_ledger(session_dir) -> list:
    from . import paths

    p = paths.ledger_jsonl(session_dir)
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ledger_has_cid(session_dir, cid: str) -> bool:
    return any(r.get("cid") == cid for r in read_ledger(session_dir))


def recent_fingerprints(session_dir, last_n_rounds: int = 10) -> set:
    """近 N 轮已尝试过的 hypothesis 指纹集合(生成黑名单用)。"""
    rows = read_ledger(session_dir)
    if not rows:
        return set()
    max_round = max(int(r.get("round", 0)) for r in rows)
    lo = max(0, max_round - last_n_rounds + 1)
    return {
        r["fingerprint"]
        for r in rows
        if r.get("fingerprint") and int(r.get("round", 0)) >= lo
    }
