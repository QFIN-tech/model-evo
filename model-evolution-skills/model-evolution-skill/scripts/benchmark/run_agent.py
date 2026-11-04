#!/usr/bin/env python
"""v2.19.0 benchmark 冷启动 Agent 驱动:对每个 benchmark run 用 ``claude -p`` 端到端跑一次。

    python scripts/benchmark/run_agent.py --runs-root <dir> --model <模型> \
        [--only a,b] [--tier fast] [--max-generations 3] [--max-trials 10] \
        [--wall-hours 1.0] [--dry-run]

前提:run 目录由 ``benchmark prepare`` 生成(含 benchmark-task.json、task-spec.yaml、
data-loader.py,``library.enabled: false``),且已 validate/preflight 通过。

做什么:
1. 在 run 目录下建 ``.claude/skills/modelevo`` → skill 根的符号链接(与正式 run
   的安装方式一致;guard hook 已由 init 写入 run 的 .claude/settings.json);
2. 把预算写进 task-spec 的 resource_budget(套件 budget 段为默认);
3. 以 run 目录为 cwd 调用 ``claude -p``,提示词要求按 SKILL.md 标准工作流从
   profile 一路做到 report,**无人值守**:确认门按保守默认裁决并留痕、运行中
   裁决走预注册默认分支,不得等待用户;
4. **多轮续跑**:``claude -p`` 的主会话一结束回合进程就退出(实测 deepseek-v4-
   flash 派出 gen-01 subagent 后写"等回传后继续"即结束,run 停在 gen-01)。每轮
   结束后读 run-state:未到 FINAL_EVALUATED 且墙钟/轮数未用完,就起一个**新会话**
   按磁盘状态续跑(SKILL 约定"已有任务以磁盘状态为准",不用 --resume 累积上下文);
5. 每轮结果追加到 ``<runs-root>/agent-runs.jsonl``。

``--model`` 必填且原样传给 claude(本机别名可能被环境变量映射到别的模型)。
``--dry-run`` 只打印将执行的命令与提示词,不启动 claude。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

SKILL = Path(__file__).resolve().parents[2]

#: claude -p 放行的工具。Bash 只放行 python 调用(modelevo CLI、lint、trial 脚本);
#: 读写限于 run 目录由 guard hook 与 cwd 约束
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Task",
                 "Bash(python:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)",
                 "Bash(tail:*)", "Bash(mkdir:*)"]


def _prompt(run_dir: Path, task: dict, budget: dict) -> str:
    ds = task.get("dataset") or {}
    return (
        "你在使用 modelevo skill(.claude/skills/modelevo/SKILL.md)对本目录的 "
        f"benchmark 任务 `{ds.get('name')}` 做冷启动端到端建模。\n"
        f"- run 目录:{run_dir}(task-spec.yaml、data-loader.py 已就绪,"
        "validate/preflight 已通过;经验库已主动禁用,不要改动这一点)。\n"
        f"- 目标:{task.get('target_name')},主指标 {ds.get('primary_metric')},"
        f"任务类型 {ds.get('model_type')}。\n"
        + ("- 本目录有 DATA_DICTIONARY.md(序列列的字段顺序与业务含义),"
           "设计特征前先读它。\n" if (run_dir / "DATA_DICTIONARY.md").exists()
           else "")
        + ""
        f"- 预算:max_generations={budget['max_generations']},"
        f"max_trials={budget['max_trials']},墙钟 ≤ {budget['wall_hours']} h。\n"
        "要求:\n"
        "1. 先读 SKILL.md,按标准工作流执行:profile → baseline → "
        "代际演进 → 确认 → "
        "finalize-all → evaluate-final → report。export-experience 不要执行"
        "(冷启动 benchmark 不写经验库)。\n"
        "2. 这是无人值守运行:profile 命中确认门时按「机械提案复核清单」逐列"
        "裁决,写回 spec 并在 logs/user-decisions.md 记录豁免。这是公开 "
        "benchmark 数据,没有已知泄漏;机械提案判 label_equivalent / "
        "banned-certain 的列,若证据只有'单变量 AUC 高',且没有标识符、时间、"
        "未来窗或标签复制的语义,应裁为 kept_ambiguous(合法保留),不要 ban——"
        "误封强特征会永久丢掉信息通道(见 SKILL.md 机械提案复核);只有确有"
        "泄漏语义的列才 ban;运行中出现的裁决一律按预注册默认分支"
        "继续,不得等待用户;交付位 diverged/unconfirmed 时按默认分支对搜索冠军 "
        "confirm 后再收尾。\n"
        "3. 不得修改 skill 目录、不得手改状态产物、不得绕过任何门。\n"
        "4. 你是无头运行:回合一结束进程就退出、没有人会'等回传后继续'。subagent "
        "一律前台同步执行(不要 run_in_background),训练用 wait 前台等待;不要在"
        "等待中结束回合。持续推进直到 FINAL_EVALUATED 或预算耗尽。\n"
        "5. 结束时输出一行 BENCH_DONE 及 status/leaderboard 摘要。\n")


def _resume_prompt(run_dir: Path, task: dict, budget: dict, rnd: int,
                   left_h: float) -> str:
    return (_prompt(run_dir, task, budget)
            + f"\n[续跑第 {rnd} 轮] 本 run 之前的会话已结束。先运行 status / "
            "leaderboard / snapshot 了解磁盘上的真实进度(未 finalize 的代先 "
            "finalize-generation),然后从当前阶段继续,不要重做已完成且 hash 兼容的"
            f"工作。剩余墙钟约 {left_h:.2f} h。\n")


def _run_state(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "run-state.json").read_text(
            encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def _ensure_skill_link(run_dir: Path) -> None:
    link = run_dir / ".claude" / "skills" / "modelevo"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        return
    os.symlink(SKILL, link)


def _apply_budget(run_dir: Path, budget: dict) -> None:
    p = run_dir / "task-spec.yaml"
    spec = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rb = spec.setdefault("resource_budget", {})
    rb.update({"max_generations": int(budget["max_generations"]),
               "max_trials": int(budget["max_trials"]),
               "max_wall_time_hours": float(budget["wall_hours"])})
    p.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--model", required=True,
                    help="驱动模型(原样传给 claude --model)")
    ap.add_argument("--suite", default=str(SKILL / "assets"
                                           / "benchmark-suite.yaml"))
    ap.add_argument("--tier", default="fast", choices=("fast", "full", "large"))
    ap.add_argument("--only", default=None, help="逗号分隔数据集名")
    ap.add_argument("--max-generations", type=int, default=None)
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--wall-hours", type=float, default=None)
    ap.add_argument("--claude", default="claude", help="claude 可执行文件")
    ap.add_argument("--max-rounds", type=int, default=12,
                    help="每个 run 最多续跑的会话轮数(默认 12)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    suite = yaml.safe_load(Path(args.suite).read_text(encoding="utf-8")) or {}
    budget = dict((suite.get("budget") or {}).get(args.tier) or {})
    budget.setdefault("max_generations", 3)
    budget.setdefault("max_trials", 10)
    budget.setdefault("wall_hours", 1.0)
    for k, v in (("max_generations", args.max_generations),
                 ("max_trials", args.max_trials),
                 ("wall_hours", args.wall_hours)):
        if v is not None:
            budget[k] = v
    only = {x for x in (args.only or "").split(",") if x}
    root = Path(args.runs_root)
    ledger = root / "agent-runs.jsonl"
    rc_all = 0
    for ds in suite.get("datasets") or []:
        name = ds.get("name")
        if only and name not in only:
            continue
        if not only and args.tier not in (ds.get("tiers") or []):
            continue
        run_dir = (root / name).resolve()
        task_p = run_dir / "benchmark-task.json"
        if not task_p.exists():
            print(f"[skip] {name}: 无 benchmark-task.json(先 benchmark prepare)")
            continue
        task = json.loads(task_p.read_text(encoding="utf-8"))
        prompt = _prompt(run_dir, task, budget)
        cmd = [args.claude, "-p", prompt, "--model", args.model,
               "--allowedTools", *ALLOWED_TOOLS]
        if args.dry_run:
            print(f"=== {name}(dry-run)\ncwd: {run_dir}\nbudget: {budget}\n"
                  f"cmd: {args.claude} -p <prompt> --model {args.model} "
                  f"--allowedTools {' '.join(ALLOWED_TOOLS)}\n--- prompt ---\n"
                  f"{prompt}")
            continue
        _ensure_skill_link(run_dir)
        _apply_budget(run_dir, budget)
        t0 = time.time()
        wall_sec = float(budget["wall_hours"]) * 3600
        (run_dir / "logs").mkdir(exist_ok=True)
        rc = 0
        for rnd in range(1, int(args.max_rounds) + 1):
            st = _run_state(run_dir)
            used = time.time() - t0
            if st.get("phase") == "FINAL_EVALUATED":
                break
            # 墙钟用完仍未收尾:本轮作为收尾轮(见下方提示与循环末尾的 break)
            left_h = max(0.0, (wall_sec - used) / 3600)
            prompt = _prompt(run_dir, task, budget) if rnd == 1 else \
                _resume_prompt(run_dir, task, budget, rnd, left_h)
            if used >= wall_sec:
                prompt += ("\n墙钟预算已用完:不要再开新代,直接对搜索冠军走"
                           "确认与 finalize-all → evaluate-final → report 收尾。\n")
            cmd[2] = prompt
            try:
                proc = subprocess.run(
                    cmd, cwd=run_dir, capture_output=True, text=True,
                    timeout=int(max(wall_sec - used, 0) + 3600))
                rc, tail = proc.returncode, (proc.stdout or "")[-4000:]
            except subprocess.TimeoutExpired:
                rc, tail = 124, "timeout"
            with (run_dir / "logs" / "bench-agent.log").open(
                    "a", encoding="utf-8") as fh:
                fh.write(f"\n===== round {rnd} rc={rc} =====\n{tail}\n")
            st = _run_state(run_dir)
            rec = {"dataset": name, "model": args.model, "round": rnd,
                   "rc": rc, "wall_hours": round((time.time() - t0) / 3600, 3),
                   "phase": st.get("phase"), "budget": budget,
                   "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            if used >= wall_sec:
                break  # 收尾轮已给过
        rc_all = rc_all or rc
    return 0 if rc_all == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
