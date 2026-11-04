"""v2.17.2 运行时修复批次回归测试(run4-v2.16.0-2.opus5.5 / req-t7-400w 复盘)。

- B2 run-generation 不得改写已有活 runner 的 trial(状态/历史不变,摘要 already_running);
- B1 特征块进程池:worker 被 SIGKILL → 带标记的 typed 错误(判 oom,infra 类,
  重入不计 fix_code);drain 超时;文件锁有界等待;并发受内存估计约束;
  信号退出 + cgroup oom_kill 增加 → oom;
- B5 compute_hours 计入 finalize-all / evaluate-final 收尾用时(finalize_hours 分列);
- B6 CLI 路径 resolve;残留 MODELEVO_PRESCREEN_ROOT 不再误映射(check-products);
- B7 停止报告按代归档 stop_history;FINAL_EVALUATED 覆盖 STOPPED;
- B8 planner/planner_context/report 归 plumbing,纯 plumbing 变更自动登记;
- C3 warm-start 预注册家族 BH → selection_correction(family=warm_start),可 confirm;
- C4 shadow 诊断位 gap_sanity_passed(读取侧兼容旧键 passed);
- D3 final_recipe 轮数取交付冻结值(frozen_rounds / best_iteration / 来源留痕)。

全部离线合成数据,不读真实 run。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic import bootstrap

import runtime.scheduler as sched  # noqa: E402
from runtime.config import Config  # noqa: E402
from runtime.gen_executor import GenerationExecutor  # noqa: E402
from runtime.logging_utils import dump_json, load_json, load_yaml  # noqa: E402
from runtime.runner import ConcurrentRunError, TrialRunner  # noqa: E402
from runtime.scheduler import (AttemptPolicy, ExecutionResult,  # noqa: E402
                               FailureCircuitBreaker, ResourcePlan, Scheduler,
                               classify_failure, is_infra_failure)
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402


# ---------------------------------------------------------------------------
# 公共夹具
# ---------------------------------------------------------------------------

def _cfg(**execution) -> Config:
    cfg = Config.load(skill_config=False)
    cfg._data.setdefault("execution", {}).update(execution)
    return cfg


def _mk_trial(run: Path, tid: str, *, generation: int = 1,
              status: str = "CODE_WRITTEN", **extra) -> TrialRecord:
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "code" / "train.py").write_text("print('noop')\n", encoding="utf-8")
    data = {"trial_id": tid, "algorithm": "xgboost", "status": status,
            "generation": generation, "parent_trial_ids": [],
            "search_action": "explore",
            "hypothesis": {"statement": f"h-{tid}"},
            "execution": {"code_entry": "train.py",
                          "command": "python train.py"},
            "status_history": [{"state": status,
                                "at": "2026-10-23T03:56:19+08:00"}]}
    data.update(extra)
    (d / "trial.yaml").write_text(yaml.safe_dump(data, allow_unicode=True),
                                  encoding="utf-8")
    return TrialRecord.load(d)


def _bare_runner(run: Path, cfg: Config) -> TrialRunner:
    r = TrialRunner.__new__(TrialRunner)
    r.run_dir = Path(run)
    r.config = cfg
    r.scheduler = Scheduler(cfg, {"cpu_physical": 32, "cpu_count": 64,
                                  "load1": 0.0, "ram_available_gb": 200.0,
                                  "gpus": []})
    r.attempt_policy = AttemptPolicy.from_config(cfg)
    r.breaker = FailureCircuitBreaker(cfg)
    return r


@pytest.fixture()
def live_pid():
    """别的进程的活 pid(模拟 `run --detach` 的 runner)。"""
    p = subprocess.Popen(["sleep", "120"])
    try:
        yield p.pid
    finally:
        p.kill()
        p.wait(timeout=10)


def _spec_reg() -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v2172", "task_name": "v2172", "model_type": "regression",
        "target": {"name": "amt_t7"}, "time": {"column": "pday"},
        "entity": {"id_column": "user_no"}, "primary_metric": "rmse",
        "resource_budget": {"max_generations": 8, "max_trials": 50},
        "data": {},
    })


def _evo_run(tmp: Path, *, skill_version: str = "2.17.0") -> Path:
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "run-state.json").write_text(json.dumps(
        {"baseline_id": "t-base", "skill_version": skill_version}),
        encoding="utf-8")
    _evo_trial(run, "t-base", 0, 100.0, "accepted",
               extra={"root_baseline_only": True})
    return run


def _evo_trial(run: Path, tid: str, gen: int, metric: float, verdict: str, *,
               action: str = "explore", parents=None, extra=None,
               result=None):
    d = run / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    ev = {"primary_metric": {"name": "rmse", "value": metric},
          "verdict": verdict, "integrity": {"valid": True}, "delta": {}}
    if verdict == "confirmed":
        ev["fresh_confirmation"] = {"passed": True}
    ev.update(extra or {})
    data = {"trial_id": tid, "algorithm": "xgboost", "generation": gen,
            "status": "EVALUATED", "search_action": action,
            "parent_trial_ids": parents or [], "data_view": "v1",
            "evaluation": ev}
    if result is not None:
        data["result"] = result
    rec = TrialRecord(data=data, trial_dir=d)
    rec.flush()
    return rec


def _engine(run: Path):
    from runtime.evolution import EvolutionEngine

    return EvolutionEngine(run, _spec_reg(), Config.load(skill_config=False))


# ===========================================================================
# B2 run-generation 不得改写在跑 trial
# ===========================================================================

def test_b2_executor_skips_trial_with_live_runner(tmp_path, live_pid):
    run = tmp_path / "run"
    busy = _mk_trial(run, "trial-006-g01-explore-xgboost-352",
                     status="RUNNING")
    (Path(busy.trial_dir) / ".runner.lock").write_text(f"{live_pid}\n")
    yaml_before = (Path(busy.trial_dir) / "trial.yaml").read_bytes()
    free = _mk_trial(run, "trial-007-g01-explore-xgboost-352")
    called = []

    def _run(trial, **kw):
        called.append(trial.trial_id)
        trial.data["status"] = "EVALUATED"
        trial.flush()
        return trial

    cfg = _cfg()
    runner = SimpleNamespace(
        run=_run, run_dir=run, config=cfg,
        spec=SimpleNamespace(time_column=None, id_column=None,
                             primary_metric="auc"),
        breaker=FailureCircuitBreaker(cfg),
        resolve_timeout_minutes=lambda override=None: 5.0)
    out = GenerationExecutor(runner).execute_generation(
        [busy, free], plan=ResourcePlan(n_parallel_trials=2), run_dir=run)
    assert called == [free.trial_id], "在跑 trial 不得交给 runner"
    assert out["already_running"] == [busy.trial_id]
    assert out["already_running_pids"] == {busy.trial_id: live_pid}
    assert out["n_failed"] == 0 and out["n_succeeded"] == 1
    assert busy.trial_id not in out["executed"]
    # 状态与历史逐字节不变(旧实现:FAILED/infra_error + barrier timeout)
    assert (Path(busy.trial_dir) / "trial.yaml").read_bytes() == yaml_before
    rec = load_yaml(Path(busy.trial_dir) / "trial.yaml")
    assert rec["status"] == "RUNNING" and "failure_reason" not in rec


def test_b2_all_trials_running_returns_clean_summary(tmp_path, live_pid):
    run = tmp_path / "run"
    busy = _mk_trial(run, "trial-001-g01-a", status="RUNNING")
    (Path(busy.trial_dir) / ".runner.lock").write_text(f"{live_pid}\n")
    runner = SimpleNamespace(run=lambda *a, **k: pytest.fail("不应执行"),
                             run_dir=run, config=_cfg())
    out = GenerationExecutor(runner).execute_generation(
        [busy], plan=ResourcePlan(n_parallel_trials=1), run_dir=run)
    assert out["already_running"] == [busy.trial_id]
    assert out["executed"] == [] and out["n_failed"] == 0


def test_b2_runner_refusal_inside_batch_is_skip_not_failure(tmp_path):
    """批前探测与启动之间的竞态:runner 抛 ConcurrentRunError → 跳过、不 flush。"""
    run = tmp_path / "run"
    t = _mk_trial(run, "trial-001-g01-a", status="RUNNING")
    before = (Path(t.trial_dir) / "trial.yaml").read_bytes()

    def _run(trial, **kw):
        raise ConcurrentRunError(trial.trial_id, 4242,
                                 Path(trial.trial_dir) / ".runner.lock")

    cfg = _cfg()
    runner = SimpleNamespace(
        run=_run, run_dir=run, config=cfg,
        spec=SimpleNamespace(time_column=None, id_column=None,
                             primary_metric="auc"),
        breaker=FailureCircuitBreaker(cfg),
        resolve_timeout_minutes=lambda override=None: 5.0)
    out = GenerationExecutor(runner).execute_generation(
        [t], plan=ResourcePlan(n_parallel_trials=1), run_dir=run)
    assert out["already_running"] == [t.trial_id]
    assert out["n_failed"] == 0
    assert (Path(t.trial_dir) / "trial.yaml").read_bytes() == before


def test_b2_summarize_only_corrects_trials_it_started(tmp_path):
    run = tmp_path / "run"
    mine = _mk_trial(run, "trial-001-g01-a", status="RUNNING")
    other = _mk_trial(run, "trial-002-g01-b", status="RUNNING")
    ex = GenerationExecutor(SimpleNamespace(run_dir=run, config=_cfg()))
    ex._started = {mine.trial_id}
    out = ex._summarize({mine.trial_id: mine, other.trial_id: other},
                        racing=None)
    assert mine.status == "FAILED" and mine.data["failure_reason"] == \
        "timeout"
    assert other.status == "RUNNING", "非本执行器启动的 trial 不做 barrier 兜底"
    assert out["timeout_trials"] == [mine.trial_id]


def test_b2_runner_refuses_before_any_mutation(tmp_path, live_pid):
    run = tmp_path / "run"
    t = _mk_trial(run, "trial-001-g01-a", status="FAILED",
                  failure_reason="code_error",
                  result={"status": "failed", "returncode": -15})
    (Path(t.trial_dir) / ".runner.lock").write_text(f"{live_pid}\n")
    before = (Path(t.trial_dir) / "trial.yaml").read_bytes()
    with pytest.raises(ConcurrentRunError) as ei:
        _bare_runner(run, _cfg()).run(t, plan=ResourcePlan())
    assert ei.value.pid == live_pid
    assert isinstance(ei.value, RuntimeError), "兼容旧的 RuntimeError 捕获"
    assert (Path(t.trial_dir) / "trial.yaml").read_bytes() == before
    assert not t.data.get("attempts"), "FAILED 重入不得记 attempt"


def test_b2_cli_split_already_running(tmp_path, live_pid):
    from _mvo.cmd_trial import _split_already_running

    run = tmp_path / "run"
    a = _mk_trial(run, "trial-001-g01-a", status="RUNNING")
    b = _mk_trial(run, "trial-002-g01-b")
    c = _mk_trial(run, "trial-003-g01-c", status="RUNNING")
    (Path(a.trial_dir) / ".runner.lock").write_text(f"{live_pid}\n")
    (Path(c.trial_dir) / ".runner.lock").write_text("4194303\n")  # 死锁
    keep, running = _split_already_running([a, b, c])
    assert [t.trial_id for t in keep] == [b.trial_id, c.trial_id]
    assert running == {a.trial_id: live_pid}
    src = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    body = src.split("def cmd_run_generation(")[1].split("\ndef ")[0]
    assert "_split_already_running(runnable)" in body
    assert body.index("_split_already_running(runnable)") < \
        body.index('t.data["harness_hash"] = stamp["harness_hash"]'), \
        "盖 harness_hash 的 flush 之前必须排除在跑 trial"


# ===========================================================================
# B1 特征块进程池鲁棒性
# ===========================================================================

def _kernel_kill_marker(batch):
    v = batch["v"]
    if 13 in v:
        os.kill(os.getpid(), signal.SIGKILL)  # 模拟 OOM-kill
    return np.asarray(v, dtype=np.float32).reshape(-1, 1)


def _kernel_hang_marker(batch):
    v = batch["v"]
    if 13 in v:
        time.sleep(120)
    return np.asarray(v, dtype=np.float32).reshape(-1, 1)


def _kernel_ok(batch):
    return np.asarray(batch["v"], dtype=np.float32).reshape(-1, 1)


def _parquet(tmp_path: Path, n: int = 100) -> Path:
    p = tmp_path / "src.parquet"
    pd.DataFrame({"v": np.arange(n, dtype=np.int64)}).to_parquet(p)
    return p


def test_b1_worker_sigkill_raises_typed_oom_error(tmp_path, monkeypatch):
    from runtime.feature_cache import (FeatureBlockWorkerLost,
                                       WORKER_LOST_MARK, parallel_map_batches)

    monkeypatch.delenv("MODELEVO_FEATURE_JOBS", raising=False)
    src = _parquet(tmp_path)
    t0 = time.time()
    with pytest.raises(FeatureBlockWorkerLost) as ei:
        parallel_map_batches(_kernel_kill_marker, src, ["v"], 1, n_jobs=2,
                             batch_rows=10, log=None, drain_timeout_sec=60)
    assert time.time() - t0 < 30, "worker 丢失必须立即暴露,不得挂死"
    msg = str(ei.value)
    assert WORKER_LOST_MARK in msg and "OOM" in msg
    # trial 进程以该异常退出(rc=1)时 stderr 尾部带标记 → oom(infra 类)
    tail = f"Traceback ...\nruntime.feature_cache.FeatureBlockWorkerLost: {msg}"
    assert classify_failure(1, tail) == "oom"
    assert classify_failure(-15, "concurrent.futures.process."
                                 "BrokenProcessPool: A process ...") == "oom"
    assert is_infra_failure("oom")
    # 正常内核不受影响(行序与值逐位一致)
    out = parallel_map_batches(_kernel_ok, src, ["v"], 1, n_jobs=2,
                               batch_rows=10, log=None)
    assert np.array_equal(out[:, 0], np.arange(100, dtype=np.float32))


def test_b1_drain_timeout_kills_pool(tmp_path, monkeypatch):
    from runtime.feature_cache import (DRAIN_TIMEOUT_MARK,
                                       FeatureBlockDrainTimeout,
                                       parallel_map_batches,
                                       resolve_drain_timeout)

    monkeypatch.delenv("MODELEVO_FEATURE_JOBS", raising=False)
    src = _parquet(tmp_path)
    t0 = time.time()
    with pytest.raises(FeatureBlockDrainTimeout) as ei:
        parallel_map_batches(_kernel_hang_marker, src, ["v"], 1, n_jobs=2,
                             batch_rows=10, log=None, drain_timeout_sec=1.0)
    assert time.time() - t0 < 30, "超时后杀池返回,不在 shutdown 上挂死"
    assert DRAIN_TIMEOUT_MARK in str(ei.value)
    assert classify_failure(1, str(ei.value)) == "infra_error"
    # 配置口径:调用方 > env > 默认;≤0 = 不限
    monkeypatch.setenv("MODELEVO_FEATURE_DRAIN_TIMEOUT_SEC", "12")
    assert resolve_drain_timeout() == 12.0
    assert resolve_drain_timeout(5) == 5.0
    assert resolve_drain_timeout(0) is None
    monkeypatch.delenv("MODELEVO_FEATURE_DRAIN_TIMEOUT_SEC")
    assert resolve_drain_timeout() == 3600.0


def test_b1_file_lock_bounded_wait_logs_holder(tmp_path):
    import fcntl

    from runtime.feature_cache import _FileLock

    lock_p = tmp_path / "train.lock"
    holder = open(lock_p, "a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    holder.truncate(0)
    holder.write("424242\n")
    holder.flush()
    msgs: list[str] = []
    t0 = time.time()
    try:
        with _FileLock(lock_p, timeout_sec=0.5, poll_sec=0.05,
                       log=msgs.append) as lk:
            assert lk.timed_out and lk.fh is None, "超时退化为无锁"
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()
    assert time.time() - t0 < 5
    assert any("424242" in m and "超时" in m for m in msgs), msgs
    # 锁空闲时正常拿到,并写入持有者 pid
    with _FileLock(lock_p, timeout_sec=1.0) as lk2:
        assert lk2.fh is not None and not lk2.timed_out
        assert lk2.holder_pid() == os.getpid()


def test_b1_jobs_capped_by_memory_estimate(monkeypatch):
    from runtime.feature_cache import cap_jobs_by_memory

    monkeypatch.delenv("MODELEVO_FEATURE_WORKER_GB", raising=False)
    assert cap_jobs_by_memory(8, None, available_gb=35.0) == (8, None), \
        "无单 worker 估计 → 行为不变"
    jobs, info = cap_jobs_by_memory(8, 10.0, available_gb=35.0)
    assert jobs == 3 and info["capped"] and info["worker_mem_gb"] == 10.0
    assert cap_jobs_by_memory(2, 10.0, available_gb=500.0) == (2, None)
    assert cap_jobs_by_memory(8, 50.0, available_gb=10.0)[0] == 1
    monkeypatch.setenv("MODELEVO_FEATURE_WORKER_GB", "20")
    assert cap_jobs_by_memory(8, available_gb=100.0)[0] == 5


def _fake_cgroup(tmp_path: Path, monkeypatch, oom_kill: int = 0) -> Path:
    cg = tmp_path / "cg"
    (cg / "user.slice").mkdir(parents=True)
    ev = cg / "user.slice" / "memory.events"
    ev.write_text(f"low 0\nhigh 0\nmax 3\noom 1\noom_kill {oom_kill}\n")
    pc = tmp_path / "proc_cgroup"
    pc.write_text("0::/user.slice\n")
    monkeypatch.setattr(sched, "CGROUP_ROOT", str(cg))
    monkeypatch.setattr(sched, "PROC_SELF_CGROUP", str(pc))
    monkeypatch.setattr(sched, "PROC_VMSTAT", str(tmp_path / "novmstat"))
    return ev


def test_b1_signal_exit_with_cgroup_oom_evidence_is_oom(tmp_path,
                                                        monkeypatch):
    ev = _fake_cgroup(tmp_path, monkeypatch)
    snap = sched.read_cgroup_oom_kills()
    assert list(snap.values()) == [0]
    bump = (f"import os,signal,pathlib;p=pathlib.Path({str(ev)!r});"
            "p.write_text(p.read_text().replace('oom_kill 0','oom_kill 1'));"
            "os.kill(os.getpid(), signal.SIGKILL)")
    res = sched.run_trial_process(
        [sys.executable, "-c", bump], cwd=tmp_path,
        stdout_path=tmp_path / "o1.log", stderr_path=tmp_path / "e1.log",
        timeout_sec=60, heartbeat_log=tmp_path / "hb1.log")
    assert res.returncode == -9 and res.failure_reason == "oom"
    assert "oom evidence" in (tmp_path / "hb1.log").read_text()
    # 对照:同样被信号杀,但 oom_kill 计数没动 → 不判 oom
    res2 = sched.run_trial_process(
        [sys.executable, "-c",
         "import os,signal;os.kill(os.getpid(), signal.SIGTERM)"],
        cwd=tmp_path, stdout_path=tmp_path / "o2.log",
        stderr_path=tmp_path / "e2.log", timeout_sec=60)
    assert res2.returncode == -15 and res2.failure_reason == "code_error"


def test_b1_oom_reentry_does_not_consume_fix_code(tmp_path):
    run = tmp_path / "run"
    r = _bare_runner(run, _cfg())
    t = _mk_trial(run, "trial-006-g01-a", status="FAILED",
                  failure_reason="oom",
                  attempts=[{"kind": "retry", "note": "自动重试(oom)"}],
                  result={"status": "failed", "returncode": -9})
    assert r.attempt_policy.next_action(t, "oom") == "give_up"
    assert r._reenter_failed_trial(t) == "oom"
    assert t.data["attempts"][-1]["kind"] == "requeue"
    assert not any(a["kind"] == "fix_code" for a in t.data["attempts"])
    # code_error 仍按 fix_code 计(语义不回退)
    t2 = _mk_trial(run, "trial-007-g01-b", status="FAILED",
                   failure_reason="code_error")
    r._reenter_failed_trial(t2)
    assert t2.data["attempts"][-1]["kind"] == "fix_code"


# ===========================================================================
# B5 compute_hours 计入收尾
# ===========================================================================

def test_b5_compute_hours_summary_includes_finalize(tmp_path):
    from runtime.trial import compute_hours_summary, record_timing_log

    run = tmp_path / "run"
    t1 = _mk_trial(run, "trial-001-g00-a", generation=0, status="EVALUATED",
                   result={"runtime_sec": 3600.0})
    t2 = _mk_trial(run, "trial-002-g01-b", status="EVALUATED",
                   result={"runtime_sec": 1800.0})
    acct = compute_hours_summary(run, [t1, t2])
    assert acct["finalize_hours"] == 0 and acct["compute_hours"] == 1.5
    # 旧格式 finalize-all.json(v2.17.1 写的 wall_min)
    (run / "logs").mkdir()
    dump_json(run / "logs" / "finalize-all.json",
              {"at": "x", "returncodes": {}, "wall_min": 90.0})
    acct = compute_hours_summary(run, [t1, t2])
    assert acct["finalize_hours"] == pytest.approx(1.5)
    assert acct["compute_hours"] == pytest.approx(3.0)
    assert acct["trial_compute_hours"] == pytest.approx(1.5)
    # 再跑一次 finalize-all(新格式:compute_sec 优先)+ evaluate-final 两次
    rec = record_timing_log(run / "logs" / "finalize-all.json",
                            {"wall_min": 30.0, "compute_sec": 2700.0})
    assert len(rec["history"]) == 2 and rec["wall_min"] == 30.0
    record_timing_log(run / "logs" / "evaluate-final.json",
                      {"wall_sec": 360.0})
    record_timing_log(run / "logs" / "evaluate-final.json",
                      {"wall_sec": 360.0})
    acct = compute_hours_summary(run, [t1, t2])
    assert acct["finalize_breakdown"] == {
        "finalize-all": pytest.approx(2.25), "evaluate-final":
            pytest.approx(0.2)}
    assert acct["finalize_hours"] == pytest.approx(2.45)
    assert acct["compute_hours"] == pytest.approx(1.5 + 2.45)


def test_b5_status_and_generation_budget_metrics(tmp_path, capsys):
    from _mvo.cmd_inspect import cmd_status

    run = _evo_run(tmp_path)
    _evo_trial(run, "t-g1", 1, 95.0, "rejected", parents=["t-base"],
               result={"runtime_sec": 7200.0})
    (run / "logs").mkdir()
    dump_json(run / "logs" / "finalize-all.json", {"wall_min": 60.0})
    _engine(run).finalize_generation(1, ["t-g1"], baseline_metric=100.0)
    gy = load_yaml(run / "generations" / "gen-01" / "generation.yaml")
    bm = gy["budget_metrics"]
    assert bm["finalize_hours"] == pytest.approx(1.0)
    assert bm["trial_compute_hours"] == pytest.approx(2.0)
    assert bm["compute_hours"] == pytest.approx(3.0)
    assert cmd_status(argparse.Namespace(run=str(run), spec=None)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["compute_hours"] == pytest.approx(3.0)
    assert out["finalize_hours"] == pytest.approx(1.0)
    assert out["trial_compute_hours"] == pytest.approx(2.0)


def test_b5_writers_wired():
    src_speed = (SKILL / "scripts" / "_mvo" / "cmd_speed.py").read_text(
        encoding="utf-8")
    assert 'record_timing_log(logs / "finalize-all.json"' in src_speed
    assert '"compute_sec"' in src_speed
    src_dlv = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    ef = src_dlv.split("def cmd_evaluate_final(")[1].split("\ndef ")[0]
    assert 'record_timing_log(run_dir / "logs" / "evaluate-final.json"' in ef
    src_trial = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    fg = src_trial.split("def cmd_finalize_generation(")[1].split("\ndef ")[0]
    assert "compute_hours_summary(" in fg


# ===========================================================================
# B6 路径规范化 + 残留 MODELEVO_PRESCREEN_ROOT
# ===========================================================================

def _splits_run(tmp_path: Path) -> tuple[Path, TaskSpec]:
    from runtime.data import write_splits
    from runtime.logging_utils import dump_yaml

    run = tmp_path / "run"
    run.mkdir(parents=True)
    dump_yaml(run / "task-spec.yaml", {
        "task_id": "b6", "task_name": "b6", "model_type": "regression",
        "target": {"name": "y"}, "time": {"column": "dt"},
        "entity": {"id_column": "uid"}, "primary_metric": "rmse",
        "split_policy": {"type": "time_based", "train": "2026-01~2026-03",
                         "validation": "2026-04", "shadow_test": "2026-05",
                         "final_test": "2026-06"}})
    spec = TaskSpec.load(run / "task-spec.yaml")
    rng = np.random.default_rng(0)
    rows = [(f"2026-0{m}-{1 + i % 28:02d}", f"u{m}{i:05d}")
            for m in range(1, 7) for i in range(120)]
    df = pd.DataFrame(rows, columns=["dt", "uid"])
    df["x"] = rng.normal(size=len(df))
    df["y"] = np.clip(50 + 20 * df["x"] + rng.normal(size=len(df)), 0, None)
    write_splits(df, spec, run / "sample-profile" / "splits")
    return run, spec


def test_b6_run_dir_resolved(tmp_path, monkeypatch):
    from _mvo.context import _load_spec, _resolve_splits_arg, _run_dir_from

    run, _ = _splits_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MODELEVO_RUN_DIR", raising=False)
    rd = _run_dir_from(argparse.Namespace(run="run", spec=None))
    assert rd.is_absolute() and rd == run.resolve()
    rd2 = _run_dir_from(argparse.Namespace(run=None,
                                           spec="run/task-spec.yaml"))
    assert rd2 == run.resolve()
    spec = _load_spec(argparse.Namespace(run=None, spec="run/task-spec.yaml"))
    assert Path(spec._spec_path).is_absolute()
    monkeypatch.setenv("MODELEVO_RUN_DIR", "run")
    assert _run_dir_from(argparse.Namespace(run=None, spec=None)) == \
        run.resolve()
    # --splits-dir:run 相对(旧口径)/ cwd 相对 / 绝对 → 一律绝对路径
    assert _resolve_splits_arg(rd, "sample-profile/splits") == str(
        (run / "sample-profile" / "splits").resolve())
    assert _resolve_splits_arg(rd, "run/sample-profile/splits") == str(
        (run / "sample-profile" / "splits").resolve())
    assert _resolve_splits_arg(rd, str(run / "sample-profile" / "splits")) \
        == str((run / "sample-profile" / "splits").resolve())


def test_b6_stale_prescreen_root_ignored_by_check_products(tmp_path,
                                                          monkeypatch):
    from _mvo.cmd_inspect import cmd_check_products
    from runtime.data import load_split_frames

    run, _ = _splits_run(tmp_path)
    sandbox = tmp_path / "old-prescreen-sandbox"
    (sandbox / "sample-profile" / "splits").mkdir(parents=True)
    monkeypatch.setenv("MODELEVO_PRESCREEN_ROOT", str(sandbox))
    # 复现:残留 env 把真实 run 的读取重映射进沙箱(没有 validation 面)
    assert "validation" not in load_split_frames(run)
    monkeypatch.chdir(tmp_path)
    rc = cmd_check_products(argparse.Namespace(
        run="run", spec="run/task-spec.yaml",
        splits_dir="sample-profile/splits"))
    assert rc == 0
    assert "MODELEVO_PRESCREEN_ROOT" not in os.environ
    assert "validation" in load_split_frames(run)


def test_b6_prescreen_root_kept_inside_sandbox(tmp_path, monkeypatch):
    from _mvo.context import _run_dir_from

    sandbox = tmp_path / "sb"
    (sandbox / "trials").mkdir(parents=True)
    monkeypatch.setenv("MODELEVO_PRESCREEN_ROOT", str(sandbox))
    _run_dir_from(argparse.Namespace(run=str(sandbox), spec=None))
    assert os.environ.get("MODELEVO_PRESCREEN_ROOT") == str(sandbox)


# ===========================================================================
# B7 停止状态
# ===========================================================================

def test_b7_stop_archived_per_generation_and_cleared(tmp_path):
    run = _evo_run(tmp_path)
    _evo_trial(run, "t-g1", 1, 95.0, "rejected", parents=["t-base"])
    eng = _engine(run)
    eng.finalize_generation(1, ["t-g1"], baseline_metric=100.0)
    # 模拟 gen-01 命中停止(旧报告无 generation,按 n_generations-1 推断)
    st = load_json(run / "evolution-state.json")
    st["stopped"] = {"stop_reason": "single_data_view_insufficient",
                     "n_generations": 2, "advisory": True}
    dump_json(run / "evolution-state.json", st)
    rs = load_json(run / "run-state.json")
    rs.update(phase="STOPPED", stop_reason="single_data_view_insufficient")
    dump_json(run / "run-state.json", rs)
    # 停止后继续演进:gen-02 有新证据
    _evo_trial(run, "t-g2", 2, 97.0, "rejected", parents=["t-base"])
    out = _engine(run).finalize_generation(2, ["t-g2"], baseline_metric=100.0)
    assert out["stop_superseded"] == {
        "generation": 1, "stop_reason": "single_data_view_insufficient",
        "superseded_by_generation": 2}
    st = load_json(run / "evolution-state.json")
    hist = st["stop_history"]
    assert len(hist) == 1 and hist[0]["generation"] == 1
    assert hist[0]["superseded_by_generation"] == 2
    rs = load_json(run / "run-state.json")
    if out["stop_report"] is None:
        assert "stopped" not in st
        assert rs["phase"] == "EVOLVING" and "stop_reason" not in rs
    else:  # 本代又命中停止:新报告带本代代号
        assert st["stopped"]["generation"] == 2
        assert rs["phase"] == "STOPPED"
    assert rs["superseded_stop_reason"] == "single_data_view_insufficient"
    # 重复 finalize 同一代不重复归档
    _engine(run).finalize_generation(2, ["t-g2"], baseline_metric=100.0)
    assert len(load_json(run / "evolution-state.json")["stop_history"]) == 1


def test_b7_final_evaluated_overrides_stopped(tmp_path):
    from _mvo.context import PHASE_ORDER, _update_run_state

    run = tmp_path / "run"
    run.mkdir()
    dump_json(run / "run-state.json", {"status": "PROFILED",
                                       "phase": "STOPPED",
                                       "stop_reason": "x"})
    st = _update_run_state(run, status="FINAL_EVALUATED")
    assert st["phase"] == "FINAL_EVALUATED"
    assert st["status"] == "FINAL_EVALUATED", "status 字段不再残留 PROFILED"
    st = _update_run_state(run, status="STOPPED")
    assert st["phase"] == "FINAL_EVALUATED", "STOPPED 不得倒回已收尾的 phase"
    assert PHASE_ORDER == TrialRunner.PHASE_ORDER


def test_b7_runner_running_supersedes_stop_reason(tmp_path):
    run = tmp_path / "run"
    t = _mk_trial(run, "trial-022-g07-a", generation=7)
    dump_json(run / "run-state.json", {"phase": "STOPPED",
                                       "stop_reason": "convergence"})
    _bare_runner(run, _cfg())._update_run_status("RUNNING", t)
    rs = load_json(run / "run-state.json")
    assert rs["phase"] == "EVOLVING" and "stop_reason" not in rs
    assert rs["superseded_stop_reason"] == "convergence"


def test_b7_status_output_coherent(tmp_path, capsys):
    from _mvo.cmd_inspect import cmd_status

    run = _evo_run(tmp_path)
    dump_json(run / "evolution-state.json", {
        "n_generations": 7, "stopped": {"stop_reason": "convergence",
                                        "n_generations": 7}})
    _evo_trial(run, "t-g6", 6, 97.0, "rejected", parents=["t-base"])
    assert cmd_status(argparse.Namespace(run=str(run), spec=None)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["stopped"] == "convergence" and out["stopped_generation"] == 6
    # 停止后又建了 gen-07 trial:当前停止失效,作为 superseded 呈现
    _evo_trial(run, "t-g7", 7, 96.0, "rejected", parents=["t-base"])
    assert cmd_status(argparse.Namespace(run=str(run), spec=None)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["stopped"] is None
    assert out["stop_superseded"]["stop_reason"] == "convergence"


def test_b7_refinalize_after_final_evaluation_keeps_phase(tmp_path):
    run = _evo_run(tmp_path)
    for g, m in ((1, 99.0), (2, 98.5), (3, 98.4)):
        _evo_trial(run, f"t-g{g}", g, m, "rejected", parents=["t-base"])
        _engine(run).finalize_generation(g, [f"t-g{g}"],
                                         baseline_metric=100.0)
    st = load_json(run / "evolution-state.json")
    if not st.get("stopped"):
        pytest.skip("合成序列未触发停止判定")
    rs = load_json(run / "run-state.json")
    rs["phase"] = "FINAL_EVALUATED"
    dump_json(run / "run-state.json", rs)
    _engine(run).finalize_generation(3, ["t-g3"], baseline_metric=100.0)
    assert load_json(run / "run-state.json")["phase"] == "FINAL_EVALUATED"


# ===========================================================================
# B8 plumbing 模块与自动登记
# ===========================================================================

def _ack_run(tmp_path: Path, changed: dict) -> Path:
    from runtime.harness import harness_file_hashes

    run = tmp_path / "run-x"
    run.mkdir(parents=True)
    files = dict(harness_file_hashes())
    files.update(changed)
    (run / "run-state.json").write_text(json.dumps({
        "harness_epoch": 1, "harness_hash": "deadbeef00000000",
        "harness_files": files}))
    return run


def test_b8_plumbing_set():
    from runtime.harness import PLUMBING_MODULES

    assert {"planner.py", "planner_context.py", "report.py"} <= \
        PLUMBING_MODULES
    assert "config.py" not in PLUMBING_MODULES, "config.py 持评分口径常量"
    assert "evaluator.py" not in PLUMBING_MODULES


def test_b8_plumbing_only_change_auto_registered(tmp_path):
    from _mvo.context import _auto_ack_plumbing
    from runtime.harness import protocol_stamp

    stamp = protocol_stamp(SKILL / "runtime")
    run = _ack_run(tmp_path, {"planner.py": "aaaa", "report.py": "bbbb",
                              "planner_context.py": "cccc"})
    st = load_json(run / "run-state.json")
    entry = _auto_ack_plumbing(run, st, stamp)
    assert entry is not None and entry["auto"] is True
    assert entry["measurement_impact"] is False
    assert sorted(entry["changed"]) == ["planner.py", "planner_context.py",
                                        "report.py"]
    st2 = load_json(run / "run-state.json")
    assert st2["harness_hash"] == stamp["harness_hash"]
    assert st2["harness_epoch"] == 1, "plumbing 变更 epoch 不变"
    assert st2["harness_change_log"][-1]["auto"] is True
    # 触及 measurement(config.py)→ 不自动登记
    run2 = _ack_run(tmp_path / "m", {"planner.py": "aaaa",
                                     "config.py": "dddd"})
    assert _auto_ack_plumbing(run2, load_json(run2 / "run-state.json"),
                              stamp) is None
    # 旧 run(无逐文件 hash)→ 不自动登记
    run3 = tmp_path / "legacy"
    run3.mkdir()
    dump_json(run3 / "run-state.json", {"harness_hash": "x"})
    assert _auto_ack_plumbing(run3, load_json(run3 / "run-state.json"),
                              stamp) is None


def test_b8_manual_ack_path1_no_flag_needed(tmp_path):
    from _mvo.cmd_inspect import cmd_acknowledge_harness_change

    run = _ack_run(tmp_path, {"planner.py": "aaaa"})
    a = argparse.Namespace(run=str(run), reason="planner fix",
                           allow_measurement_impact=False,
                           measurement_no_score_impact=False,
                           verify_against=None, spec=None)
    assert cmd_acknowledge_harness_change(a) == 0
    st = load_json(run / "run-state.json")
    assert st["harness_epoch"] == 1
    assert st["harness_change_log"][-1]["measurement_impact"] is False


def test_b8_gates_wired():
    src = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    rt = src.split("def cmd_run_trial(")[1].split("\ndef ")[0]
    rg = src.split("def cmd_run_generation(")[1].split("\ndef ")[0]
    for body in (rt, rg):
        assert "_auto_ack_plumbing(run_dir, st, stamp)" in body
        assert body.index("_auto_ack_plumbing(") < body.index(
            "Harness 代码已变更")


# ===========================================================================
# C3 warm-start 预注册家族
# ===========================================================================

def _warm(run: Path, tid: str, metric: float, p_better: float,
          verdict: str = "accepted"):
    return _evo_trial(
        run, tid, 0, metric, verdict, action="warm_start",
        extra={"delta": {"vs_baseline": 100.0 - metric,
                         "vs_baseline_paired_bootstrap": {
                             "p_better": p_better, "p_positive": p_better,
                             "delta_ci": [0.1, 2.0]}}})


def test_c3_warm_start_family_bh_and_eligibility(tmp_path):
    from runtime.trial import (WARM_START_NOT_CONFIRMABLE,
                               confirmation_parent_eligibility)

    run = _evo_run(tmp_path)
    _warm(run, "t-warm-a", 98.9, 1.0)        # p=0
    _warm(run, "t-warm-b", 99.5, 0.7)        # p=0.3 → 不过
    out = _engine(run).finalize_generation(0, ["t-base", "t-warm-a",
                                               "t-warm-b"],
                                           baseline_metric=100.0)
    fam = out["selection_correction"]["warm_start_family"]
    assert fam["family_size"] == 2 and fam["alpha_charged"] is False
    assert fam["passed"] == ["t-warm-a"]
    assert fam["not_confirmable"] == ["t-warm-b"]
    st = load_json(run / "evolution-state.json")
    assert float(st.get("alpha_spent") or 0.0) == 0.0, "预注册家族不扣 alpha"
    a = TrialRecord.load(run / "trials" / "t-warm-a")
    sc = a.data["evaluation"]["selection_correction"]
    assert sc["family"] == "warm_start" and sc["passed"] is True
    assert sc["family_size"] == 2 and sc["reference"] == "baseline"
    assert load_json(run / "trials" / "t-warm-a" / "evaluation.json")[
        "selection_correction"]["family"] == "warm_start"
    ok, why = confirmation_parent_eligibility(a)
    assert ok and why == "selection_correction_passed_warm_start"
    b = TrialRecord.load(run / "trials" / "t-warm-b")
    sc_b = b.data["evaluation"]["selection_correction"]
    assert sc_b["passed"] is False
    assert sc_b["not_confirmable_reason"] == WARM_START_NOT_CONFIRMABLE
    assert b.data["evaluation"]["verdict"] == "accepted", "不改 verdict"
    assert confirmation_parent_eligibility(b) == (
        False, WARM_START_NOT_CONFIRMABLE)


def test_c3_warm_start_backfilled_on_later_generation(tmp_path):
    from runtime.trial import confirmation_parent_eligibility

    run = _evo_run(tmp_path)
    w = _warm(run, "t-warm-a", 98.9, 1.0)
    # 旧 run:warm 没有校正记录 → 资格门给明确原因
    assert confirmation_parent_eligibility(w) == (
        False, "warm_start_not_confirmable")
    _evo_trial(run, "t-g1", 1, 99.0, "rejected", parents=["t-base"])
    _engine(run).finalize_generation(1, ["t-g1"], baseline_metric=100.0)
    w = TrialRecord.load(run / "trials" / "t-warm-a")
    assert confirmation_parent_eligibility(w)[0] is True


# ===========================================================================
# C4 shadow 诊断位命名
# ===========================================================================

def test_c4_writer_uses_gap_sanity_key():
    src = (SKILL / "runtime" / "runner.py").read_text(encoding="utf-8")
    blk = src.split('"gap_vs_validation": gap,')[1].split("})")[0]
    assert '"gap_sanity_passed": bool(passed)' in blk
    assert '"criterion": "gap_sanity"' in blk
    assert '"passed"' not in blk, "诊断位不再与确认门同名"


@pytest.mark.parametrize("key", ["gap_sanity_passed", "passed"])
def test_c4_generation_shadow_check_reads_new_and_legacy(tmp_path, key):
    run = _evo_run(tmp_path)
    _evo_trial(run, "t-g1", 1, 90.0, "confirmed", action="confirm",
               parents=["t-base"],
               extra={"shadow_test": {"rmse": 91.0, "gap_relative": 0.011,
                                      key: False}})
    out = _engine(run).finalize_generation(1, ["t-g1"],
                                           baseline_metric=100.0)
    sc = out["shadow_check"]
    assert sc is not None and sc["gap_sanity_passed"] is False
    assert sc["criterion"] == "gap_sanity"


# ===========================================================================
# D3 final_recipe 轮数取交付冻结值
# ===========================================================================

def _d3_meta(**kw):
    m = {"best_trial": "trial-020-g05-confirm-xgboost-2006",
         "algorithm": "confirm", "refit_mode": "frozen_rounds",
         "n_features": 1647, "final_model_sha256": "ce" * 32,
         "delivery_status": "confirmed", "seed": 20263856,
         "early_stopping_best_iteration": 5802,
         "hyperparameters": {"max_depth": 6, "learning_rate": 0.025,
                             "n_estimators": 5802}}
    m.update(kw)
    return m


def _champ(tmp_path: Path) -> Path:
    champ = tmp_path / "trials" / "trial-020-g05-confirm-xgboost-2006"
    (champ / "config").mkdir(parents=True)
    (champ / "config" / "trial.yaml").write_text(yaml.safe_dump(
        {"hyperparameters": {"max_depth": 6, "learning_rate": 0.025,
                             "n_estimators": 8000}}))
    return champ


def test_d3_final_recipe_uses_frozen_rounds(tmp_path):
    from _mvo.cmd_deliver import final_recipe_summary

    r = final_recipe_summary(_d3_meta(), _champ(tmp_path))
    assert r["hyperparameters"]["n_estimators"] == 5802
    assert r["frozen_rounds"] == 5802 and r["rounds_key"] == "n_estimators"
    assert r["best_iteration"] == 5802
    assert r["configured_rounds_cap"] == 8000
    assert r["hyperparameters_source"] == \
        "config.hyperparameters+model_meta.hyperparameters[n_estimators]"
    assert r["hyperparameters"]["max_depth"] == 6
    # catboost 键同样生效;meta 无轮数时不改(旧口径)
    r2 = final_recipe_summary(_d3_meta(hyperparameters={"iterations": 1234}),
                              None)
    assert r2["frozen_rounds"] == 1234 and r2["rounds_key"] == "iterations"
    assert r2["hyperparameters_source"] == "model_meta.hyperparameters"
    r3 = final_recipe_summary(_d3_meta(hyperparameters={"n_estimators": None},
                                       early_stopping_best_iteration=None),
                              None)
    assert "frozen_rounds" not in r3 and "best_iteration" not in r3


def test_d3_record_final_recipe_upgrades_legacy_entry(tmp_path):
    from _mvo.cmd_deliver import _record_final_recipe
    from runtime.memory import TaskExperience

    run = tmp_path / "run"
    te = TaskExperience(run)
    te.save({"task_id": "x", "key_findings": ["k"], "final_recipe": {
        "model_sha256": "ce" * 32, "refit_mode": "frozen_rounds",
        "hyperparameters": {"n_estimators": 8000},
        "hyperparameters_source": "config.hyperparameters"}})
    champ = _champ(tmp_path)
    _record_final_recipe(run, _d3_meta(), champ)
    fr = te.load()["final_recipe"]
    assert fr["hyperparameters"]["n_estimators"] == 5802
    assert fr["frozen_rounds"] == 5802
    assert te.load()["key_findings"] == ["k"]
    txt = te.path.read_text(encoding="utf-8")
    _record_final_recipe(run, _d3_meta(), champ)          # 之后照旧幂等
    assert te.path.read_text(encoding="utf-8") == txt
