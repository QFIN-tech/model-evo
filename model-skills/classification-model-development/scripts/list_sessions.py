# -*- coding: utf-8 -*-
"""扫描 runs/ 下历史 sessions, 格式化打印供 agent / 用户选择。

两种情形:
- 当前会话已有 session_dir (上游 feature-matching / feature-analysis 已跑过) → 直接复用, 不询问。
- 用户新启动一个会话直接调 model-training → 调本脚本列出历史 sessions,
  agent 用 AskUserQuestion 问: 选历史(列编号) 或 新建(追问 task_name/description)。

session 目录格式: runs/{YYYYMMDD-HHMMSS}-{task_name}/session.json
session.json 字段: session_id / task_name / created_at / description。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def _find_outputs_dir(start: Optional[Path] = None) -> Path:
    """从 start (默认脚本所在目录) 往上找 model_evo/runs/。"""
    p = (start or Path(__file__).resolve()).resolve()
    for _ in range(8):
        candidate = p / "runs"
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return Path("runs")


def scan_sessions(outputs_dir: Path) -> List[dict]:
    """扫描 outputs_dir 下 sessions, 按 mtime 倒序返回。

    每个 dict 字段: dir_name / task_name / created_at / description / path。
    损坏 session.json (无文件/解析失败) 跳过, 不报错。
    """
    if not outputs_dir.is_dir():
        return []

    sessions: List[dict] = []
    for child in outputs_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        session_json = child / "session.json"
        if not session_json.is_file():
            continue
        try:
            data = json.loads(session_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        sessions.append({
            "dir_name": child.name,
            "task_name": data.get("task_name", ""),
            "created_at": data.get("created_at", ""),
            "description": data.get("description", ""),
            "path": str(child),
        })

    sessions.sort(key=lambda s: s["dir_name"], reverse=True)
    return sessions


def format_sessions(sessions: List[dict]) -> str:
    """格式化为 markdown 表 (带序号), 供 agent / 用户阅读。"""
    if not sessions:
        return "无历史 session"

    lines = ["| 序号 | session_id | task_name | created_at | description |",
             "|------|------------|-----------|------------|-------------|"]
    for i, s in enumerate(sessions, 1):
        desc = (s["description"] or "").replace("|", "/").replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        lines.append(
            f"| {i} | {s['dir_name']} | {s['task_name']} | {s['created_at']} | {desc} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="列出 runs/ 下历史 sessions, 供选择")
    parser.add_argument(
        "--outputs_dir", default=None,
        help="outputs 目录 (默认自动从脚本位置向上查找 model_evo/runs/)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else _find_outputs_dir()
    sessions = scan_sessions(outputs_dir)
    print(f"[list_sessions] outputs_dir = {outputs_dir}")
    print(format_sessions(sessions))
    return 0


if __name__ == "__main__":
    sys.exit(main())
