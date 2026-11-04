"""v17.2 修复批次单元验证(wait 子命令 + guard 覆盖 subagent + 签名表)。

背景(2026-09-03 example-seq-v3-4-2.8.6-v17.1 run 审计回写):

1. **subagent 轮询等训练**:gen-01/gen-02 各 13~15 次
   `sleep 115; tail …` 循环等单个 trial,~3.5M tokens 信息增量≈0。
   根因:后台 `run` 启动后,"等 trial 终态"没有确定性命令,Agent
   退化成轮询日志。修复:`wait --trial <id>` 子命令——阻塞到
   trial.yaml 状态机终态,输出与 run 同格式 SUMMARY 行(格式单源
   _trial_summary_line);超时 WAIT_TIMEOUT + exit 3;已终态立即返回。
2. **guard 对 subagent 整类豁免**:gen-02 单代 Bash sed/grep runtime
   源码 32 次做源码刑侦(~6M tokens);SKILL 明令禁止 triage 刑侦,
   但 hook 只拦主会话。修复:guard 的 Read/Grep/Glob 源码拦截对
   subagent 同规生效(Bash 不拦——subagent 是执行一线);新增
   MODELEVO_ALLOW_SOURCE_READ=1 开发豁免。
3. **CLI --help 探索**:每 subagent 3~5 次 new-trial/run/finalize
   --help。修复:digest 增"常用子命令签名表"。

运行:python tests/test_v17dot2_fixes.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cli_source  # noqa: E402

import yaml

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "modelevo_cli_v17dot2", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "modelevo_guard_v17dot2", SKILL / "assets" / "guard_main_session.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


def _make_run_dir(tmp: Path) -> Path:
    """最小 run 目录:task-spec + 一个 trial 记录。"""
    run = tmp / "run"
    (run / "trials" / "trial-001-g01-lightgbm-1234" / "code").mkdir(
        parents=True, exist_ok=True)
    spec = {
        "task_id": "t17dot2", "task_name": "wait-test", "model_type":
        "binary_classification",
        "target": {"name": "y", "positive_definition": "y==1"},
        "primary_metric": "auc",
        "split_policy": {"type": "random", "train": 0.7, "validation": 0.3},
        "features": {"banned": {}, "reparse": {}},
    }
    (run / "task-spec.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    return run


def _write_trial(run: Path, status: str, history: list[str] | None = None,
                 evaluation: dict | None = None) -> str:
    tid = "trial-001-g01-lightgbm-1234"
    data = {
        "trial_id": tid, "algorithm": "lightgbm", "generation": 1,
        "status": status,
        "status_history": [{"state": s, "at": "2026-09-03T00:00:00"}
                           for s in (history or [status])],
        "hypothesis": {"statement": "wait 子命令验证用占位假设"},
        "execution": {"command": "python train.py"},
    }
    if evaluation is not None:
        data["evaluation"] = evaluation
    (run / "trials" / tid / "trial.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return tid


def _run_wait(run: Path, tid: str, extra: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "modelevo.py"),
         "wait", "--run", str(run), "--trial", tid, *extra],
        capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# 1. wait 子命令
# ---------------------------------------------------------------------------

def test_wait_terminal_immediate() -> None:
    print("== wait:已终态立即返回 ==")
    cli = _load_cli()
    check("_WAIT_TERMINAL_STATES 覆盖 EVALUATED/FAILED/REFLECTED/SUCCEEDED",
          {"EVALUATED", "FAILED", "REFLECTED", "SUCCEEDED"}
          <= cli._WAIT_TERMINAL_STATES)
    # 历史回退形态:force_rerun 把 EVALUATED 回写 CODE_WRITTEN,
    # status_history 有终态记录 → 视为已终态,不等
    t = type("T", (), {"status": "CODE_WRITTEN",
                       "data": {"status_history": [
                           {"state": "RUNNING"}, {"state": "EVALUATED"},
                           {"state": "CODE_WRITTEN"}]}})()
    check("force_rerun 回退态经 history 判定终态",
          cli._trial_reached_terminal(t))
    t2 = type("T", (), {"status": "RUNNING",
                        "data": {"status_history": [
                            {"state": "PROPOSED"},
                            {"state": "CODE_WRITTEN"},
                            {"state": "RUNNING"}]}})()
    check("RUNNING 活动态不判终态", not cli._trial_reached_terminal(t2))


def test_wait_cli_paths() -> None:
    print("== wait:CLI 行为 ==")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run = _make_run_dir(tmp)
        # (a) trial 不存在 → exit 1
        r = _run_wait(run, "trial-999-g99-nope-0000", ["--timeout", "2"])
        check("trial 不存在 → exit 1", r.returncode == 1, r.stderr[-200:])
        # (b) 已终态(EVALUATED)→ 立即返回 SUMMARY,exit 0
        ev = {"verdict": "accepted",
              "primary_metric": {"value": 0.7123},
              "delta": {"vs_parent": 0.01, "vs_baseline": 0.02},
              "shadow_test": {"auc": 0.70}}
        tid = _write_trial(run, "EVALUATED", evaluation=ev)
        t0 = time.monotonic()
        r = _run_wait(run, tid, ["--timeout", "5"])
        dt = time.monotonic() - t0
        check("终态立即返回(exit 0)", r.returncode == 0, r.stderr[-200:])
        check("终态等待 <10s(无轮询空转)", dt < 10, f"{dt:.1f}s")
        summary = [ln for ln in r.stdout.splitlines()
                   if ln.startswith("SUMMARY\t")]
        check("SUMMARY 行存在", len(summary) == 1)
        if summary:
            parts = summary[0].split("\t")
            check("SUMMARY 格式与 run 同字段序(trial_id/status/verdict/"
                  "metric/delta_parent/delta_baseline)",
                  parts[1] == tid and parts[2] == "EVALUATED"
                  and parts[3] == "accepted"
                  and parts[4] == "0.712300"
                  and parts[5] == "delta_parent=+0.010000"
                  and parts[6] == "delta_baseline=+0.020000",
                  summary[0][:120])
        # (c) 超时 → WAIT_TIMEOUT + exit 3,不 kill(状态仍是 RUNNING)
        tid2 = _write_trial(run, "RUNNING", history=[
            "PROPOSED", "CODE_WRITTEN", "RUNNING"])
        r = _run_wait(run, tid2, ["--timeout", "2", "--poll-interval", "5"])
        check("超时 → exit 3", r.returncode == 3, r.stdout[-200:])
        check("超时输出 WAIT_TIMEOUT 行",
              "WAIT_TIMEOUT" in r.stdout and tid2 in r.stdout)
        # (d) 轮询间隔收敛到 [5,60]
        r2 = _run_wait(run, tid2, ["--timeout", "0.05",
                                   "--poll-interval", "0.01"])
        check("poll-interval 下限 5s 生效(超时先于首轮 sleep 到期",
              r2.returncode == 3)


def test_summary_format_single_source() -> None:
    print("== SUMMARY 格式单源 ==")
    cli = _load_cli()
    src = cli_source()
    check("cmd_run_trial 不再内联 SUMMARY 格式串(单源 _trial_summary_line)",
          src.count('"SUMMARY\\t{}\\t{}\\t{}\\t{}\\tdelta_parent={}'
                     '\\tdelta_baseline={}"') == 1)


# ---------------------------------------------------------------------------
# 2. guard:v17.2 覆盖 subagent + wait 白名单
# ---------------------------------------------------------------------------

class _FakeSubproc:
    """拦截 _deny/_allow 的 stdout 探针。"""

def _guard_run(guard, tool: str, payload: dict, subagent: bool,
               env_extra: dict | None = None) -> tuple[bool, str]:
    """驱动 guard.main():返回 (allowed, deny_message)。

    main() 从 stdin 读 hook 输入,故用 StringIO 喂入;_deny 走
    sys.exit(code),捕获 SystemExit 判定放行/拒绝。"""
    import io
    hook_input = {"tool_name": tool, "tool_input": payload,
                  "transcript_path": None}
    if subagent:
        # _in_subagent 读 transcript_path;None → 按主会话。为测试
        # subagent 分支,直接打补丁。
        orig = guard._in_subagent
        guard._in_subagent = lambda _h: True
    saved = {k: os.environ.get(k) for k in
             ("CLAUDE_PROJECT_DIR", "MODELEVO_ALLOW_MAIN_EXEC",
              "MODELEVO_ALLOW_SOURCE_READ")}
    os.environ["CLAUDE_PROJECT_DIR"] = "/proj"
    for k in ("MODELEVO_ALLOW_MAIN_EXEC", "MODELEVO_ALLOW_SOURCE_READ"):
        os.environ.pop(k, None)
    for k, v in (env_extra or {}).items():
        os.environ[k] = v
    buf = io.StringIO()
    old_stdout, old_stdin = sys.stdout, sys.stdin
    captured = {"allowed": False}
    try:
        sys.stdout = buf
        sys.stdin = io.StringIO(json.dumps(hook_input))
        try:
            guard.main()
            captured["allowed"] = True  # main 正常返回 = _allow()
        except SystemExit as e:
            captured["allowed"] = e.code == 0
        out = buf.getvalue()
    finally:
        sys.stdout, sys.stdin = old_stdout, old_stdin
        if subagent:
            guard._in_subagent = orig
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return captured["allowed"], out


def test_guard_subagent_source_block() -> None:
    print("== guard:subagent 源码拦截(v17.2) ==")
    guard = _load_guard()
    read_runtime = {"file_path": "/x/model-evo-V2.8/modelevo/runtime/runner.py"}
    allowed, msg = _guard_run(guard, "Read", read_runtime, subagent=False)
    check("主会话 Read runtime → 拒", not allowed)
    allowed, msg = _guard_run(guard, "Read", read_runtime, subagent=True)
    check("subagent Read runtime → 拒(v17.2 新增)", not allowed,
          f"msg={msg[:80]}")
    allowed, _ = _guard_run(guard, "Read",
                            {"file_path": "/x/run/data/trials/t/code/train.py"},
                            subagent=True)
    check("subagent Read trial code → 放行", allowed)
    allowed, _ = _guard_run(guard, "Read",
                            {"file_path": "/x/skills/modelevo/references/"
                                          "template-api.md"},
                            subagent=True)
    check("subagent Read references → 放行", allowed)
    allowed, _ = _guard_run(guard, "Grep",
                            {"pattern": "eval_frame",
                             "path": "/x/model-evo-V2.8/modelevo/runtime/"
                                     "runner.py"},
                            subagent=True)
    check("subagent Grep runtime 目录 → 拒", not allowed)
    allowed, _ = _guard_run(guard, "Read", read_runtime, subagent=True,
                            env_extra={"MODELEVO_ALLOW_SOURCE_READ": "1"})
    check("MODELEVO_ALLOW_SOURCE_READ=1 → 豁免源码拦截", allowed)


def test_guard_wait_profile_allowlist_and_bash() -> None:
    print("== guard:wait/profile 白名单 + subagent Bash 放行 ==")
    guard = _load_guard()
    check("ALLOWED_SUBCOMMANDS 含 wait", "wait" in guard.ALLOWED_SUBCOMMANDS)
    check("ALLOWED_SUBCOMMANDS 含 profile",
          "profile" in guard.ALLOWED_SUBCOMMANDS)
    allowed, _ = _guard_run(
        guard, "Bash",
        {"command": "python modelevo.py wait --trial trial-001"},
        subagent=False)
    check("主会话 Bash wait → 放行(只读等待)", allowed)
    allowed, _ = _guard_run(
        guard, "Bash",
        {"command": "python modelevo.py profile --run /tmp/run"},
        subagent=False)
    check("主会话 Bash profile → 放行(机械材料化)", allowed)
    allowed, _ = _guard_run(
        guard, "Bash",
        {"command": "python modelevo.py run --trial trial-001"},
        subagent=False)
    check("主会话 Bash run → 拒(不变)", not allowed)
    allowed, _ = _guard_run(
        guard, "Bash",
        {"command": "sed -n '38,60p' /x/model-evo-V2.8/modelevo/runtime/"
                    "trial.py"},
        subagent=True)
    # v19.4 语义更新:Bash 文本读取命令直读 harness 源码改为拒绝
    # (v17.2 时代放行;实测单 run 26 次严格口径 Bash 源码刑侦全部
    # 走这条路,工具级拦截形同虚设)。
    check("subagent Bash sed runtime → 拒(v19.4;v17.2 曾放行)",
          not allowed)


# ---------------------------------------------------------------------------
# 3. 文档接线(digest 签名表 / SKILL wait 与 triage 同规)
# ---------------------------------------------------------------------------

def test_docs_wiring() -> None:
    print("== 文档接线 ==")
    digest = (SKILL / "references" / "protocol-digest.md").read_text(
        encoding="utf-8")
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    check("digest 含 wait 子命令条目",
          "wait --trial" in digest and "WAIT_TIMEOUT" in digest)
    check("digest 含常用子命令签名表(消 --help 探索)",
          "签名表" in digest and "new-trial" in digest
          and "--budget-configs" in digest)
    check("digest triage 同规(subagent 也禁源码刑侦)",
          "主会话与 subagent 同规" in digest)
    check("digest 等待姿势 (b) 改为 run_in_background + wait",
          "前台跟一条" in digest and "wait --trial" in digest)
    check("SKILL 等待纪律含 wait", "wait --trial" in skill)
    check("SKILL triage 节声明 hook 覆盖 subagent",
          "主会话与 subagent** 的 Read/Grep/Glob" in skill
          or "主会话与 subagent" in skill)
    check("SKILL guard 工具表不再写 subagent 不受影响",
          "subagent 不受影响" not in skill)
    check("SKILL 记录 SOURCE_READ 豁免通道",
          "MODELEVO_ALLOW_SOURCE_READ" in skill)


def main() -> int:
    test_wait_terminal_immediate()
    test_wait_cli_paths()
    test_summary_format_single_source()
    test_guard_subagent_source_block()
    test_guard_wait_allowlist_and_bash()
    test_docs_wiring()
    print(f"\n=== v17.2: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
