"""v17 修复批次单元验证(故障处置协议:diagnose + 主会话边界 hook)。

背景(2026-09-02 实测 triage 事故回写):trial-003/004 泄漏污染处置
时,主会话为弄清"verdict 为何不被 leaderboard 收录"在主会话内
grep/sed runtime 源码 30+ 次、手改 evaluation.json 与 trial.yaml
(口径分裂),未缓存输入烧掉 ~14 万 tokens。本批次把 triage 从
"LLM 刑侦"改为"确定性命令 + 硬边界":

1. diagnose 子命令:异常状态一键归因,输出 code/severity/detail/
   remedy,remedy 即处置命令;只读,不改任何状态;
2. diagnose 检出形态:leakage_suspect 强制 invalid / evaluation.json
   与 trial.yaml verdict 口径分裂(手改痕迹)/ 跨 epoch 排除 /
   baseline 锚定缺失 / empty scaffold / missing validation_score /
   harness_hash 漂移 / 泄漏分类漂移 / final_test gate 违规;
3. guard_main_session.py hook:主会话 Read/Grep harness 源码 → deny;
   演进循环 Bash 命令 → deny;diagnose/status/leaderboard → allow;
   subagent transcript / env 豁免 → allow(fail-open)。

运行:python tests/test_v17_fixes.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

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
        "modelevo_cli_v17", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _make_run(td: Path) -> Path:
    """最小 run 工作区:task-spec + run-state + leaderboard 目录。"""
    (td / "trials").mkdir(parents=True)
    (td / "leaderboard").mkdir()
    dump_yaml = yaml.safe_dump
    (td / "task-spec.yaml").write_text(dump_yaml({
        "task_id": "diag-test",
        "task_name": "diagnose 单测",
        "model_type": "binary_classification",
        "target": {"name": "label",
                   "positive_definition": "y=1"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based",
                         "train": "2025-01~2025-06",
                         "validation": "2025-07~2025-08",
                         "final_test": "2025-09~2025-09"},
    }), encoding="utf-8")
    (td / "run-state.json").write_text(json.dumps({
        "phase": "EVOLVING", "baseline_id": "t-root",
        "harness_epoch": 0, "harness_hash": "fake-hash",
    }), encoding="utf-8")
    return td


def _make_trial(td: Path, tid: str, *, status="EVALUATED",
                verdict=None, evaluation=None, epoch=None,
                with_score=True, with_code=True, leak_suspects=None) -> Path:
    tdir = td / "trials" / tid
    (tdir / "code").mkdir(parents=True, exist_ok=True)
    if with_code:
        (tdir / "code" / "train.py").write_text(
            "import lightgbm\n", encoding="utf-8")
    if with_score:
        (tdir / "validation_score.parquet").write_bytes(b"x")
    ev = dict(evaluation or {})
    if verdict is not None:
        ev.setdefault("verdict", verdict)
    if leak_suspects:
        ev["leakage_suspects"] = leak_suspects
    data = {"trial_id": tid, "status": status,
            "generation": 1, "algorithm": "lightgbm",
            "parent_trial_ids": ["t-root"] if tid != "t-root" else []}
    if ev:
        data["evaluation"] = ev
    if epoch is not None:
        data["harness_epoch"] = epoch
    (tdir / "trial.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8")
    return tdir


# ---------------------------------------------------------------------------
# 1. diagnose:检出形态逐项验证
# ---------------------------------------------------------------------------

def test_diagnose_detections() -> None:
    print("== diagnose 检出形态 ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run = _make_run(td)
        # 形态 A:泄漏哨兵强制 invalid
        _make_trial(td, "t-leak", verdict="invalid",
                    leak_suspects=[{"feature": "loan_cnt_t0",
                                    "train_auc": 1.0}])
        # 形态 B:evaluation.json 与 trial.yaml verdict 分裂(手改痕迹)
        tdir = _make_trial(td, "t-split", verdict="accepted")
        (tdir / "evaluation.json").write_text(
            json.dumps({"verdict": "invalid"}), encoding="utf-8")
        # 形态 C:跨 epoch 排除
        _make_trial(td, "t-epoch", verdict="accepted", epoch=3)
        # 形态 D:空脚手架
        _make_trial(td, "t-empty", status="PROPOSED", with_code=False,
                    with_score=False)
        # 形态 E:SUCCEEDED 但缺 validation_score
        _make_trial(td, "t-noscore", status="SUCCEEDED",
                    with_score=False)
        args = type("A", (), {"spec": str(run / "task-spec.yaml"),
                              "run": None, "trial": None})()
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.cmd_diagnose(args)
        out = json.loads(buf.getvalue())
        codes = set()
        for tid, findings in out["trial_findings"].items():
            codes |= {f["code"] for f in findings}
        check("rc=1(含 critical)", rc == 1, f"rc={rc}")
        check("leakage_suspect 检出", "leakage_suspect" in codes, codes)
        check("verdict 口径分裂检出",
              "evaluation_verdict_mismatch" in codes, codes)
        check("跨 epoch 排除检出", "cross_epoch_excluded" in codes, codes)
        check("空脚手架检出", "empty_scaffold" in codes, codes)
        check("缺 validation_score 检出",
              "missing_validation_score" in codes, codes)
        # remedy 契约:每条 finding 都有非空 remedy
        all_have_remedy = all(
            f.get("remedy") and f.get("code") and f.get("severity")
            for fl in out["trial_findings"].values() for f in fl)
        check("finding 三元组齐备(code/severity/remedy)",
              all_have_remedy)
        check("remedy 即处置提示在输出中",
              "remedy 即处置" in out["hint"]
              and "只读" in out["hint"])


def test_diagnose_env_and_clean() -> None:
    print("== diagnose 环境漂移 + 干净 run ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run = _make_run(td)
        # 形态 F:harness_hash 漂移(run-state 记录 ≠ 当前 runtime)
        # 形态 G:泄漏分类漂移
        (run / "run-state.json").write_text(json.dumps({
            "phase": "EVOLVING", "harness_epoch": 0,
            "harness_hash": "stale-hash",
            "leak_manifest_hash": "stale-leak",
        }), encoding="utf-8")
        # 形态 H:final_test gate 违规
        (run / "final-test-gate.json").write_text(
            json.dumps({"count": 2}), encoding="utf-8")
        args = type("A", (), {"spec": str(run / "task-spec.yaml"),
                              "run": None, "trial": None})()
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.cmd_diagnose(args)
        out = json.loads(buf.getvalue())
        codes = {f["code"] for f in out["env_findings"]}
        check("harness_hash 漂移检出",
              "harness_hash_drift" in codes, codes)
        check("泄漏分类漂移检出",
              "leak_manifest_drift" in codes, codes)
        check("final_test gate 违规检出",
              "final_test_gate_violation" in codes, codes)
        # 形态 I:干净 run → exit 0,零 finding
        # (run-state 不写 harness_hash → diagnose 视为"未盖章",
        #  不报漂移;真实 run 在 init/run 时由 CLI 盖章)
        with tempfile.TemporaryDirectory() as td2:
            run2 = _make_run(Path(td2))
            st = json.loads(
                (run2 / "run-state.json").read_text(encoding="utf-8"))
            st.pop("harness_hash", None)
            (run2 / "run-state.json").write_text(
                json.dumps(st), encoding="utf-8")
            args2 = type("A", (), {
                "spec": str(run2 / "task-spec.yaml"),
                "run": None, "trial": None})()
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                rc2 = cli.cmd_diagnose(args2)
            out2 = json.loads(buf2.getvalue())
            check("干净 run exit=0", rc2 == 0)
            check("干净 run 无 finding",
                  not out2["env_findings"] and not out2["trial_findings"])


def test_diagnose_missing_trial() -> None:
    print("== diagnose 聚焦与缺 trial 路径 ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as td:
        run = _make_run(Path(td))
        _make_trial(Path(td), "t1", verdict="accepted")
        args = type("A", (), {"spec": str(run / "task-spec.yaml"),
                              "run": None, "trial": "no-such"})()
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.cmd_diagnose(args)
        out = json.loads(buf.getvalue())
        check("缺 trial 报错并列出 existing", rc == 1
              and "no-such" in out.get("error", "")
              and "t1" in out.get("existing", []))
        # 聚焦存在且干净的 trial
        args2 = type("A", (), {"spec": str(run / "task-spec.yaml"),
                               "run": None, "trial": "t1"})()
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            rc2 = cli.cmd_diagnose(args2)
        out2 = json.loads(buf2.getvalue())
        check("聚焦单 trial", out2["n_trials"] == 1)
        # delta_vs_baseline 空的 accepted → baseline_anchor_missing
        _make_trial(Path(td), "t-anchor", verdict="promising")
        (run / "leaderboard" / "leaderboard.json").write_text(
            json.dumps({"rows": [{"trial_id": "t-anchor",
                                  "verdict": "promising",
                                  "delta_vs_baseline": None}]}),
            encoding="utf-8")
        args3 = type("A", (), {"spec": str(run / "task-spec.yaml"),
                               "run": None, "trial": "t-anchor"})()
        buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3):
            cli.cmd_diagnose(args3)
        out3 = json.loads(buf3.getvalue())
        codes = {f["code"] for fl in out3["trial_findings"].values()
                 for f in fl}
        check("baseline 锚定缺失检出",
              "baseline_anchor_missing" in codes, codes)


# ---------------------------------------------------------------------------
# 2. guard_main_session.py:主会话硬边界
# ---------------------------------------------------------------------------

def _run_hook(payload: dict, env_extra: dict | None = None,
              transcript: str | None = None):
    env = dict(os.environ)
    env["CLAUDE_PROJECT_DIR"] = "/tmp"
    env.pop("MODELEVO_ALLOW_MAIN_EXEC", None)
    if env_extra:
        env.update(env_extra)
    if transcript:
        payload = {**payload, "transcript_path": transcript}
    proc = subprocess.run(
        [sys.executable, str(SKILL / "assets" / "guard_main_session.py")],
        input=json.dumps(payload), capture_output=True, text=True,
        env=env, timeout=30)
    return proc


def test_hook_matrix() -> None:
    print("== guard_main_session hook ==")
    skill = SKILL
    # 主会话 Read runtime 源码 → deny
    proc = _run_hook({"tool_name": "Read", "tool_input": {
        "file_path": str(skill / "runtime" / "runner.py")}})
    out = json.loads(proc.stdout or "{}")
    check("Read runtime → deny",
          proc.returncode == 2
          and out.get("hookSpecificOutput", {}).get(
              "permissionDecision") == "deny")
    check("deny 理由指向 diagnose",
          "diagnose" in out.get("hookSpecificOutput", {}).get(
              "permissionDecisionReason", ""))
    # 主会话 Grep scripts/modelevo.py → deny
    proc = _run_hook({"tool_name": "Grep", "tool_input": {
        "pattern": "verdict",
        "path": str(skill / "scripts" / "modelevo.py")}})
    check("Grep scripts → deny", proc.returncode == 2)
    # 主会话 Read run 目录 trial.yaml → allow
    proc = _run_hook({"tool_name": "Read", "tool_input": {
        "file_path": "/data/run-x/trials/t1/trial.yaml"}})
    check("Read trial.yaml → allow", proc.returncode == 0)
    # 主会话 Bash run/new-trial → deny;diagnose/status → allow
    for cmd, expect_deny in (("modelevo.py run --trial t1", True),
                             ("modelevo.py new-trial --x", True),
                             ("modelevo.py finalize-generation "
                              "--generation 3", True),
                             ("modelevo.py diagnose --run r", False),
                             ("modelevo.py profile --run r", False),
                             ("modelevo.py status", False),
                             ("modelevo.py leaderboard --spec s", False)):
        proc = _run_hook({"tool_name": "Bash", "tool_input": {
            "command": f"python {cmd}"}})
        check(("deny" if expect_deny else "allow") + f": {cmd.split()[0]}",
              (proc.returncode == 2) == expect_deny,
              f"rc={proc.returncode}")
    # subagent transcript(v17:整类豁免;v17.2 起源码拦截同规覆盖
    # subagent,Read runtime → deny——本断言随 v17.2 语义翻转)
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td) / "t.jsonl"
        tpath.write_text('{"isSidechain":false}\n{"isSidechain":true}\n')
        proc = _run_hook({"tool_name": "Read", "tool_input": {
            "file_path": str(skill / "runtime" / "runner.py")}},
            transcript=str(tpath))
        check("subagent 上下文 Read runtime → deny(v17.2 同规)",
              proc.returncode == 2)
        # subagent 的合法读取面不受影响:trial code / references → allow
        proc = _run_hook({"tool_name": "Read", "tool_input": {
            "file_path": "/data/run-x/trials/t1/code/train.py"}},
            transcript=str(tpath))
        check("subagent Read trial code → allow", proc.returncode == 0)
        proc = _run_hook({"tool_name": "Read", "tool_input": {
            "file_path": str(skill / "references" / "template-api.md")}},
            transcript=str(tpath))
        check("subagent Read references → allow", proc.returncode == 0)
        # subagent Bash 不拦(v17.2:执行一线角色,写码/跑 run 合法)
        proc = _run_hook({"tool_name": "Bash", "tool_input": {
            "command": "python modelevo.py run --trial t1"}},
            transcript=str(tpath))
        check("subagent Bash run → allow(Bash 不拦)", proc.returncode == 0)
    # env 豁免 → allow
    proc = _run_hook({"tool_name": "Bash", "tool_input": {
        "command": "modelevo.py run --trial t1"}},
        env_extra={"MODELEVO_ALLOW_MAIN_EXEC": "1"})
    check("env 豁免 → allow", proc.returncode == 0)
    # 非 Claude Code 上下文 → allow(无 CLAUDE_PROJECT_DIR)
    env = {k: v for k, v in os.environ.items()
           if k != "CLAUDE_PROJECT_DIR"}
    proc = subprocess.run(
        [sys.executable, str(SKILL / "assets" / "guard_main_session.py")],
        input=json.dumps({"tool_name": "Read", "tool_input": {
            "file_path": str(skill / "runtime" / "runner.py")}}),
        capture_output=True, text=True, env=env, timeout=30)
    check("非 hook 上下文 fail-open", proc.returncode == 0)


def test_docs_sync() -> None:
    print("== 文档同步 ==")
    skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    digest = (SKILL / "references" / "protocol-digest.md").read_text(
        encoding="utf-8")
    check("SKILL.md 含故障处置节", "故障处置" in skill_md
          and "diagnose" in skill_md)
    check("SKILL.md 铁律 13 含 triage 约束",
          "异常状态(triage)也受本条约束" in skill_md)
    check("SKILL.md 含 hook 安装说明",
          "guard_main_session.py" in skill_md)
    check("digest 含 diagnose 速查", "diagnose" in digest
          and "禁止为" in digest)
    check("digest 含体量护栏", "Subagent 体量护栏" in digest)
    check("SKILL.md 主会话允许清单含 diagnose",
          "`diagnose`(读状态" in skill_md
          or "planner-context`/`diagnose`" in skill_md)


def main() -> int:
    test_diagnose_detections()
    test_diagnose_env_and_clean()
    test_diagnose_missing_trial()
    test_hook_matrix()
    test_docs_sync()
    print(f"\n=== v17: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
