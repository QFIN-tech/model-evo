#!/usr/bin/env python
"""经验库异步整理 worker(v2.18.2 第 2 期,dev 文档 §23)。

逐组执行:``library curate-next`` 打开下一个需要整理的主题组 → 用 ``claude -p``(无头模式)
按 ``references/experience-library-curation.md`` 写决议并 ``curate-apply`` → 检查结果,
模型没有完成时 ``merge-abort`` 释放租约。与 run 解耦,由人或定时任务启动,不会自动运行。

    python scripts/curate_worker.py --model <模型> [--max-groups 5] [--work-dir DIR]

``--model`` 必填且原样传给 claude:本机的模型别名可能被环境变量映射到别的模型
(如 ANTHROPIC_DEFAULT_OPUS_MODEL),整理质量取决于它,所以不设默认值。
模型名同时记入经验库提交的 actor,更新日志里可以查到每次整理用的模型。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
MODELEVO = [sys.executable, str(SKILL / "scripts" / "modelevo.py")]
PROTOCOL = "references/experience-library-curation.md"


def _library(*args: str, model: str) -> dict:
    proc = subprocess.run([*MODELEVO, "library", *args, "--model", model], cwd=SKILL,
                          capture_output=True, text=True)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"status": "error", "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
    out["_exit"] = proc.returncode
    return out


def _snapshot_state(snapshot_id: str) -> str | None:
    """快照当前状态(经 HTTP 客户端,与 CLI 同一套配置)。"""
    sys.path.insert(0, str(SKILL / "scripts"))
    sys.path.insert(0, str(SKILL))
    from _mvo.cmd_library import _client, library_settings

    s = library_settings(argparse.Namespace(uri=None, namespace=None, root=None))
    snap = _client(s).get_snapshot(s["namespace"], snapshot_id)
    return (snap or {}).get("state")


def _allowed_commands() -> list[str]:
    """claude -p 的 Bash 放行规则按命令前缀精确匹配:模型会把脚本写成相对路径
    (cwd 是 skill 根)或绝对路径,解释器也可能写成 python / python3,各种写法都要列出,
    否则被拒后既不能提交也不能放弃(10-29 首轮 10 组里 4 组因此失败)。"""
    scripts = ("scripts/modelevo.py", MODELEVO[1])
    pythons = (sys.executable, "python", "python3")
    return [f"Bash({py} {sc} library {sub}:*)" for py in pythons for sc in scripts
            for sub in ("curate-apply", "merge-abort", "merge-renew")]


def _prompt(ctx_file: Path, decision_file: Path, opened: dict, model: str) -> str:
    m = "scripts/modelevo.py"           # 工作目录是 skill 根;放行规则见 _allowed_commands
    return (
        f"你是 ModelEvo 经验库的整理模型。先完整阅读 {PROTOCOL},再阅读整理上下文 {ctx_file}"
        f"(主题组 {opened['group']},{opened['principles']} 条原则,触发条件 "
        f"{', '.join(opened['triggers'])})。\n"
        f"按协议写整理决议到 {decision_file}(snapshot_id 与 lease_token 取自上下文),然后:\n"
        f"1. `{sys.executable} {m} library curate-apply --decision {decision_file} --dry-run "
        f"--model {model}` 校验,按报错逐条修正直到 status=valid;\n"
        f"2. 去掉 --dry-run 正式提交,确认 status=curated。\n"
        "上面的命令要原样单独执行(工作目录已是 skill 根),不要加 cd、&& 或管道,"
        "否则会被权限规则拒绝。宁可少改:没有把握的原则用 keep。不得编造数字。"
        f"若 20 分钟内无法得到合法决议,执行 `{sys.executable} {m} library merge-abort "
        f"--snapshot {opened['snapshot_id']} --lease-token {opened['lease_token']} "
        f"--reason <原因> --model {model}` 后结束。"
        "最后一行只输出 JSON:{\"status\": \"curated\"|\"aborted\", \"summary\": \"…\"}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="整理所用模型(原样传给 claude --model)")
    ap.add_argument("--max-groups", type=int, default=5)
    ap.add_argument("--claude", default="claude", help="claude 可执行文件")
    ap.add_argument("--timeout-sec", type=int, default=1800, help="每组上限(与租约同长)")
    ap.add_argument("--work-dir", default=None,
                    help="上下文 / 决议 / 日志目录;缺省 ~/.cache/modelevo-curation/<时间>")
    args = ap.parse_args(argv)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    work = Path(args.work_dir or Path.home() / ".cache" / "modelevo-curation" / stamp)
    work.mkdir(parents=True, exist_ok=True)
    log = work / "worker-log.jsonl"
    results = []
    for i in range(args.max_groups):
        ctx_file, dec_file = work / f"{i:02d}-context.json", work / f"{i:02d}-decision.json"
        opened = _library("curate-next", "--out", str(ctx_file), model=args.model)
        if opened.get("status") != "opened":
            results.append({"step": i, "status": opened.get("status"),
                            "detail": {k: v for k, v in opened.items() if k != "_exit"}})
            break
        t0 = time.time()
        allowed = ["Read", "Write", *_allowed_commands()]
        try:
            proc = subprocess.run([args.claude, "-p", _prompt(ctx_file, dec_file, opened,
                                                              args.model),
                                   "--model", args.model, "--allowedTools", *allowed],
                                  cwd=SKILL, capture_output=True, text=True,
                                  timeout=args.timeout_sec)
            tail = proc.stdout.strip().splitlines()[-1:] or [""]
        except subprocess.TimeoutExpired:
            tail = ["timeout"]
        state = _snapshot_state(opened["snapshot_id"])
        if state != "merged":
            # 模型没有提交:替它释放租约,快照记 merge_failed,下次重来。不能只看
            # "submitted":之前失败过的组重开时快照本来就是 merge_failed / reverted,
            # 只判 submitted 会漏放租约,下一组 409(10-29 首轮实测)
            _library("merge-abort", "--snapshot", opened["snapshot_id"], "--lease-token",
                     opened["lease_token"], "--reason", f"worker: 模型未完成({tail[0][:200]})",
                     model=args.model)
            state = "merge_failed"
        rec = {"step": i, "group": opened["group"], "snapshot_id": opened["snapshot_id"],
               "state": state, "sec": round(time.time() - t0), "model_tail": tail[0][:500]}
        results.append(rec)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"work_dir": str(work), "results": results}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
