# -*- coding: utf-8 -*-
"""list_sessions.py: 扫描 runs/ 下的进化 session, 推断进度, 供断点续跑选择。

用法:
    python list_sessions.py [--runs-dir runs] [--last 5]

进度推断规则(按产物存在性):
  initialized  —— data/profile/baseline 三处 .done + evolution/state.json 存在
  running      —— state.status == running(current_round > 0)
  finalized    —— state.status == finalized(report.md 已产)
  broken       —— 目录存在但关键产物缺失(半途中断, 需补齐或 --force 重跑)

退出码: 0 成功 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

from evo_core import paths, state_io

_SESSION_NAME_RE = re.compile(r"^\d{8}-\d{6}-.+$")


def infer_stage(session_dir: Path) -> str:
    """按产物存在性推断 session 进度阶段。"""
    if not paths.state_json(session_dir).exists():
        if paths.data_dir(session_dir).exists():
            return "broken(缺 evolution/state.json, 疑似准备阶段中断)"
        return "broken(非完整 session 目录)"
    try:
        state = state_io.load_state(session_dir)
    except Exception:
        return "broken(state.json 不可读)"
    status = state.get("status")
    if status == "finalized":
        return "finalized(report 已产, 可查看或继续加轮)"
    missing = [
        d
        for d in (paths.data_dir(session_dir), paths.splits_dir(session_dir), paths.baseline_dir(session_dir))
        if not state_io.is_done(d)
    ]
    if missing:
        return "broken(缺 .done: %s)" % ", ".join(m.name for m in missing)
    if status == "running":
        return "running(第 %s 轮后, champion_oot_auc=%s)" % (
            state.get("current_round"),
            state.get("champion_oot_auc"),
        )
    return "initialized(准备完成, 未开始进化)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="列出现有进化 session 并推断进度")
    parser.add_argument("--runs-dir", default="runs", help="session 根目录(默认 runs)")
    parser.add_argument("--last", type=int, default=5, help="最多显示最近 N 个(默认 5)")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print("runs 目录不存在: %s(无历史 session)" % runs_dir)
        return 0

    sessions = sorted(
        (d for d in runs_dir.iterdir() if d.is_dir() and _SESSION_NAME_RE.match(d.name)),
        key=lambda d: d.name,
        reverse=True,
    )[: args.last]
    if not sessions:
        print("%s 下无 {timestamp}-{mining_name} 命名的 session" % runs_dir)
        return 0

    print("| # | session | 阶段 |")
    print("|---|---|---|")
    for i, d in enumerate(sessions, 1):
        print("| %d | %s | %s |" % (i, d.name, infer_stage(d)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
