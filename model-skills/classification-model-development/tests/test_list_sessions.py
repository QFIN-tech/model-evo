# -*- coding: utf-8 -*-
"""list_sessions 单测: 扫描 runs/ 下 sessions + 损坏文件跳过。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from list_sessions import scan_sessions, format_sessions


def _write_session(out_dir: Path, dir_name: str, data: dict) -> Path:
    """写一个 session 目录 + session.json。"""
    d = out_dir / dir_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "session.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return d


def test_scan_empty_outputs(tmp_path):
    """空 outputs 目录 → 返回空列表。"""
    out_dir = tmp_path / "runs"
    out_dir.mkdir()
    assert scan_sessions(out_dir) == []


def test_scan_valid_sessions(tmp_path):
    """合法 session → 返回 task_name/created_at/description, 按目录名倒序。"""
    out_dir = tmp_path / "runs"
    _write_session(out_dir, "20260617-111252-req_cnt_t7", {
        "session_id": "20260617-111252-req_cnt_t7",
        "task_name": "req_cnt_t7",
        "created_at": "2026-06-17T11:12:52+08:00",
        "description": "active user t7",
    })
    _write_session(out_dir, "20260620-080000-churn_v2", {
        "session_id": "20260620-080000-churn_v2",
        "task_name": "churn_v2",
        "created_at": "2026-06-20T08:00:00+08:00",
        "description": "churn v2",
    })

    sessions = scan_sessions(out_dir)
    assert len(sessions) == 2
    assert sessions[0]["dir_name"] == "20260620-080000-churn_v2"
    assert sessions[1]["dir_name"] == "20260617-111252-req_cnt_t7"
    assert sessions[1]["task_name"] == "req_cnt_t7"
    assert "active user t7" in sessions[1]["description"]

    md = format_sessions(sessions)
    assert "req_cnt_t7" in md
    assert "churn_v2" in md


def test_scan_skips_broken_sessions(tmp_path):
    """损坏 session.json (无文件 / JSON 非法) → 跳过不报错。"""
    out_dir = tmp_path / "runs"

    _write_session(out_dir, "20260617-111252-ok", {"task_name": "ok"})

    broken_no_json = out_dir / "20260618-000000-no_json"
    broken_no_json.mkdir()

    broken_bad_json = out_dir / "20260619-000000-bad_json"
    broken_bad_json.mkdir()
    (broken_bad_json / "session.json").write_text("not a json {{{", encoding="utf-8")

    sessions = scan_sessions(out_dir)
    assert len(sessions) == 1
    assert sessions[0]["dir_name"] == "20260617-111252-ok"
    assert sessions[0]["task_name"] == "ok"


def test_format_empty():
    """空列表 → 打印提示。"""
    assert format_sessions([]) == "无历史 session"
