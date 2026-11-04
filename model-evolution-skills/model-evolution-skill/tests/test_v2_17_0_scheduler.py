"""v2.17.0 WP5a:资源感知调度(内存准入 + 按实际占用选卡)回归测试。

背景(run4 regression.2,8×A800 / 960GB 与其它 run 共享):
  - 同机其它 run 常驻 300~740GB,本 run 单 trial 峰值 79~179GB;scheduler
    的 generation_batch 计划 8 并发却不看内存,agent 只能手工错峰;
  - wait 进程两次被 cgroup OOM 杀(exit 137);
  - trial-016 被派到与其它 run 2 个进程共享的 GPU 1,分类子模型 42 min;
  - finalize 把资源受限串行记成 avg_parallelism<1.5 的编排问题。

全部离线:/proc/meminfo、cgroup 文件、nvidia-smi 输出均为 mock;训练执行体
为 mock(不起真实训练,不占内存/GPU)。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic import bootstrap

import runtime.scheduler as sched  # noqa: E402
from runtime.config import Config  # noqa: E402
from runtime.gen_executor import GenerationExecutor  # noqa: E402
from runtime.runner import TrialRunner  # noqa: E402
from runtime.scheduler import (  # noqa: E402
    ADMISSION_TIMEOUT, AttemptPolicy, ExecutionResult, FailureCircuitBreaker,
    GpuLease, MemoryAdmission, ResourcePlan, Scheduler, admission_summary,
    estimate_trial_peak_gb, is_infra_failure, read_cgroup_memory,
    system_memory)
from runtime.trial import TrialRecord  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _cfg(**execution) -> Config:
    cfg = Config.load(skill_config=False)
    cfg._data.setdefault("execution", {}).update(execution)
    return cfg


def _fake_mem(available: float, capacity: float = 960.0):
    def _probe(**_kw):
        return {"available_gb": available, "capacity_gb": capacity,
                "mem_available_gb": available, "cgroup": None,
                "source": "mock"}
    return _probe


def _mk_trial(run: Path, tid: str, *, parents=(), generation: int = 1,
              status: str = "CODE_WRITTEN", **extra) -> TrialRecord:
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "code" / "train.py").write_text("print('noop')\n", encoding="utf-8")
    data = {"trial_id": tid, "algorithm": "xgboost", "status": status,
            "generation": generation, "parent_trial_ids": list(parents),
            "search_action": "explore",
            "hypothesis": {"statement": f"h-{tid}"},
            "execution": {"code_entry": "train.py",
                          "command": "python train.py"},
            "status_history": [{"state": status,
                                "at": "2026-10-23T10:00:00+08:00"}]}
    data.update(extra)
    (d / "trial.yaml").write_text(yaml.safe_dump(data, allow_unicode=True),
                                  encoding="utf-8")
    return TrialRecord.load(d)


def _bare_runner(run: Path, cfg: Config, gpus=()) -> TrialRunner:
    r = TrialRunner.__new__(TrialRunner)
    r.run_dir = Path(run)
    r.config = cfg
    r.scheduler = Scheduler(cfg, {"cpu_physical": 32, "cpu_count": 64,
                                  "load1": 0.0, "ram_available_gb": 200.0,
                                  "gpus": list(gpus)})
    r.attempt_policy = AttemptPolicy.from_config(cfg)
    r.breaker = FailureCircuitBreaker(cfg)
    return r


def _events(run: Path) -> list[dict]:
    p = run / "logs" / "admission.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln]


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(sched.ADMISSION_ENV, raising=False)
    monkeypatch.setitem(sched._GPU_OCC_CACHE, "data", None)
    monkeypatch.setitem(sched._GPU_OCC_CACHE, "at", 0.0)
    yield


# ---------------------------------------------------------------------------
# 1. 系统内存:/proc/meminfo + cgroup(v1 / v2)
# ---------------------------------------------------------------------------

def test_meminfo_and_cgroup_v1(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal:       1006632960 kB\n"
                  "MemFree:         10000000 kB\n"
                  "MemAvailable:   104857600 kB\n")   # 100 GiB
    cg = tmp_path / "cg"
    (cg / "memory").mkdir(parents=True)
    gib = 2 ** 30
    (cg / "memory" / "memory.limit_in_bytes").write_text(str(960 * gib))
    (cg / "memory" / "memory.usage_in_bytes").write_text(str(900 * gib))
    (cg / "memory" / "memory.stat").write_text(
        f"cache 1\ntotal_inactive_file {10 * gib}\n")
    pc = tmp_path / "proc_cgroup"
    pc.write_text("4:memory:/kubepods/podX/abc\n1:cpu:/kubepods/podX/abc\n")
    info = read_cgroup_memory(str(cg), str(pc))
    assert info["version"] == 1
    # 可用 = 960 − (900 − 10) = 70 GiB
    assert info["available_bytes"] == 70 * gib
    m = system_memory(meminfo_path=str(mi), cgroup_root=str(cg),
                      proc_cgroup=str(pc))
    assert m["mem_available_gb"] == 100.0
    assert m["cgroup"]["limit_gb"] == 960.0
    assert m["available_gb"] == 70.0, "有效可用取 MemAvailable 与 cgroup 的紧者"
    assert m["source"] == "meminfo+cgroup"


def test_cgroup_v2_unlimited_and_limited(tmp_path):
    gib = 2 ** 30
    cg = tmp_path / "cg2"
    (cg / "user.slice").mkdir(parents=True)
    pc = tmp_path / "proc_cgroup"
    pc.write_text("0::/user.slice\n")
    (cg / "user.slice" / "memory.max").write_text("max\n")
    (cg / "user.slice" / "memory.current").write_text(str(5 * gib))
    assert read_cgroup_memory(str(cg), str(pc)) is None, "max = 无限额"
    (cg / "user.slice" / "memory.max").write_text(str(64 * gib))
    (cg / "user.slice" / "memory.stat").write_text(
        f"anon 1\ninactive_file {4 * gib}\n")
    info = read_cgroup_memory(str(cg), str(pc))
    assert info["version"] == 2
    assert info["available_bytes"] == (64 - (5 - 4)) * gib
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 1048576000 kB\nMemAvailable: 524288000 kB\n")
    m = system_memory(meminfo_path=str(mi), cgroup_root=str(cg),
                      proc_cgroup=str(pc))
    assert m["available_gb"] == 63.0 and m["capacity_gb"] == 64.0


def test_meminfo_module_paths_are_mockable(tmp_path, monkeypatch):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 2097152 kB\nMemAvailable: 1048576 kB\n")
    monkeypatch.setattr(sched, "MEMINFO_PATH", str(mi))
    monkeypatch.setattr(sched, "CGROUP_ROOT", str(tmp_path / "nocg"))
    monkeypatch.setattr(sched, "PROC_SELF_CGROUP", str(tmp_path / "nopc"))
    m = system_memory()
    assert m["mem_total_gb"] == 2.0 and m["available_gb"] == 1.0
    assert m["cgroup"] is None and m["source"] == "meminfo"


# ---------------------------------------------------------------------------
# 2. 峰值估计来源优先级
# ---------------------------------------------------------------------------

def test_peak_estimate_priority(tmp_path):
    run = tmp_path / "run"
    cfg = _cfg(memory_estimate_default_gb=50)
    _mk_trial(run, "trial-001-g00-a", generation=0,
              status="EVALUATED", result={"peak_memory_gb": 80.0})
    _mk_trial(run, "trial-002-g01-b", parents=["trial-001-g00-a"],
              status="EVALUATED", result={"peak_memory_gb": 120.0},
              status_history=[{"state": "EVALUATED",
                               "at": "2026-10-23T12:00:00+08:00"}])
    # 1) 父代 × 1.15(多父代取最大)
    t = _mk_trial(run, "trial-003-g02-c",
                  parents=["trial-001-g00-a", "trial-002-g01-b"],
                  expected_resources={"peak_memory_gb": 10})
    e = estimate_trial_peak_gb(run, t.data, cfg)
    assert e["source"] == "parent" and e["basis_trial"] == "trial-002-g01-b"
    assert e["estimate_gb"] == pytest.approx(120 * 1.15, abs=0.01)

    # 2) 父代无峰值 → 同 lineage 最近 trial(祖先/亲缘中最近完成者)
    _mk_trial(run, "trial-004-g02-d", parents=["trial-001-g00-a"],
              status="FAILED")
    t5 = _mk_trial(run, "trial-005-g03-e", parents=["trial-004-g02-d"],
                   expected_resources={"peak_memory_gb": 10})
    e = estimate_trial_peak_gb(run, t5.data, cfg)
    assert e["source"] == "lineage"
    # 祖先 001(80GB)与亲缘 002(120GB,更近完成)中取最近者
    assert e["basis_trial"] == "trial-002-g01-b"
    assert e["estimate_gb"] == pytest.approx(120 * 1.15, abs=0.01)

    # 2b) 自身上一 attempt 的峰值优先于亲缘
    t5.data["attempts"] = [{"kind": "retry",
                            "previous_result": {"peak_memory_gb": 30.0}}]
    e = estimate_trial_peak_gb(run, t5.data, cfg)
    assert e["source"] == "lineage" and e["basis_gb"] == 30.0

    # 3) 无历史 → trial.yaml expected_resources.peak_memory_gb
    t6 = _mk_trial(run, "trial-006-g01-root",
                   expected_resources={"peak_memory_gb": 42})
    e = estimate_trial_peak_gb(run, t6.data, cfg)
    assert e["source"] == "declared" and e["estimate_gb"] == 42.0

    # 4) 都没有 → 缺省(无数据面时取配置值)
    t7 = _mk_trial(run, "trial-007-g01-root2")
    e = estimate_trial_peak_gb(run, t7.data, cfg)
    assert e["source"] == "default" and e["estimate_gb"] == 50.0

    # 4b) 有数据面:min(缺省, max(1, 体积×4))——小数据不按大缺省排队
    sp = run / "sample-profile" / "splits"
    sp.mkdir(parents=True)
    (sp / "train.parquet").write_bytes(b"x" * 4096)
    e = estimate_trial_peak_gb(run, t7.data, cfg)
    assert e["source"] == "default" and e["estimate_gb"] == 1.0
    assert e["data_plane_gb"] is not None


# ---------------------------------------------------------------------------
# 3. 准入:通过 / 排队 / 超时
# ---------------------------------------------------------------------------

def test_admission_pass_immediately(tmp_path):
    run = tmp_path / "run"
    adm = MemoryAdmission(run, _cfg(memory_safety_margin_gb=60),
                          memory_probe=_fake_mem(400.0))
    t = adm.acquire("t1", {"estimate_gb": 100.0, "source": "declared"},
                    generation=1)
    assert t.admitted and t.decision == "admitted" and t.wait_sec < 1
    assert t.path is not None and t.path.exists(), "准入后落预留"
    ev = _events(run)
    assert [e["event"] for e in ev] == ["admitted"]
    assert ev[0]["estimate_gb"] == 100.0 and ev[0]["generation"] == 1
    t.release()
    assert not list((run / ".admission").glob("*.json")), "释放后预留消失"
    t.release()  # 幂等


def test_admission_queues_then_admits(tmp_path):
    run = tmp_path / "run"
    avail = {"v": 120.0}
    clock = {"t": 1000.0}

    def _probe(**_kw):
        return {"available_gb": avail["v"], "capacity_gb": 960.0}

    def _sleep(sec):
        clock["t"] += sec
        if clock["t"] >= 1000.0 + 90:   # 90s 后同机其它 run 释放内存
            avail["v"] = 300.0

    adm = MemoryAdmission(run, _cfg(memory_safety_margin_gb=60,
                                    admission_poll_sec=30,
                                    admission_max_wait_min=10),
                          memory_probe=_probe, sleep=_sleep,
                          clock=lambda: clock["t"])
    queued = []
    t = adm.acquire("t-q", {"estimate_gb": 100.0, "source": "parent"},
                    generation=2, on_queue=queued.append)
    assert t.admitted and t.wait_sec == pytest.approx(90.0)
    assert len(queued) == 1, "首次被拒回调一次"
    evs = [e["event"] for e in _events(run)]
    assert evs == ["queued", "admitted"]
    assert _events(run)[1]["wait_sec"] == pytest.approx(90.0)
    t.release()


def test_admission_timeout(tmp_path):
    run = tmp_path / "run"
    clock = {"t": 0.0}

    def _sleep(sec):
        clock["t"] += sec

    adm = MemoryAdmission(run, _cfg(memory_safety_margin_gb=60,
                                    admission_poll_sec=30,
                                    admission_max_wait_min=2),
                          memory_probe=_fake_mem(80.0), sleep=_sleep,
                          clock=lambda: clock["t"])
    t = adm.acquire("t-x", {"estimate_gb": 100.0, "source": "default"},
                    generation=3)
    assert not t.admitted and t.decision == "timeout"
    assert t.wait_sec == pytest.approx(120.0)
    assert t.path is None and not list((run / ".admission").glob("*.json"))
    assert [e["event"] for e in _events(run)] == ["queued", "timeout"]


def test_reservation_blocks_second_trial_until_release(tmp_path):
    """MemAvailable 尚未反映刚起的 trial(内存爬升滞后)——预留兜底。"""
    run = tmp_path / "run"
    cfg = _cfg(memory_safety_margin_gb=60, admission_max_wait_min=0)
    adm = MemoryAdmission(run, cfg, memory_probe=_fake_mem(250.0))
    a = adm.acquire("a", {"estimate_gb": 100.0})
    assert a.admitted
    b = adm.acquire("b", {"estimate_gb": 100.0})
    assert not b.admitted, "250 − 预留 100 − 余量 60 = 90 < 100"
    a.release()
    c = adm.acquire("b", {"estimate_gb": 100.0})
    assert c.admitted
    c.release()


def test_stale_reservation_reclaimed_and_small_trial(tmp_path):
    run = tmp_path / "run"
    (run / ".admission").mkdir(parents=True)
    (run / ".admission" / "dead.4194303.x.json").write_text(json.dumps(
        {"trial_id": "dead", "pid": 4194303, "token": "x",
         "estimate_gb": 500}))
    adm = MemoryAdmission(run, _cfg(admission_max_wait_min=0),
                          memory_probe=_fake_mem(200.0))
    assert adm.reservations() == [], "持有进程已死 → 陈旧预留回收"
    small = adm.acquire("tiny", {"estimate_gb": 1.0})
    assert small.admitted
    starved = MemoryAdmission(run, _cfg(admission_max_wait_min=0),
                              memory_probe=_fake_mem(10.0))
    s2 = starved.acquire("tiny2", {"estimate_gb": 1.5})
    assert s2.admitted and s2.decision == "small", \
        "≤2GB 的 trial 单独无法触发 OOM,不排队"
    small.release()
    s2.release()


def test_admission_disabled_by_config_and_env(tmp_path, monkeypatch):
    run = tmp_path / "run"
    t = MemoryAdmission(run, _cfg(memory_admission=False),
                        memory_probe=_fake_mem(0.0)).acquire(
        "t", {"estimate_gb": 500})
    assert t.admitted and t.decision == "disabled"
    monkeypatch.setenv(sched.ADMISSION_ENV, "0")
    assert not MemoryAdmission(run, _cfg()).enabled


# ---------------------------------------------------------------------------
# 4. runner 接线:单 trial(run --detach worker 同路径)准入超时
# ---------------------------------------------------------------------------

def test_execute_once_admission_timeout_is_infra(tmp_path, monkeypatch):
    run = tmp_path / "run"
    cfg = _cfg(memory_safety_margin_gb=60, admission_poll_sec=0.01,
               admission_max_wait_min=0.001)
    trial = _mk_trial(run, "trial-001-g01-x",
                      expected_resources={"peak_memory_gb": 100})
    monkeypatch.setattr(sched, "system_memory", _fake_mem(50.0))
    called = []
    monkeypatch.setattr("runtime.runner.run_trial_process",
                        lambda cmd, **kw: called.append(kw) or None)
    r = _bare_runner(run, cfg)
    res = r._execute_once(trial, Path(trial.trial_dir) / "code",
                          Path(trial.trial_dir), ResourcePlan(), 1, 1)
    assert isinstance(res, ExecutionResult) and not res.ok
    assert res.failure_reason == ADMISSION_TIMEOUT
    assert called == [], "准入超时不得启动训练"
    assert trial.data["admission"]["state"] == "timeout"
    assert not list((run / ".admission").glob("*.json"))
    # 分类:infra、不进熔断、不自动重试、重入记 requeue
    assert is_infra_failure(ADMISSION_TIMEOUT)
    assert r.attempt_policy.next_action(trial, ADMISSION_TIMEOUT) == "requeue"
    for _ in range(5):
        r.breaker.record(ADMISSION_TIMEOUT)
    assert r.breaker.tripped(0, 0) is None, "准入超时不得触发连续失败熔断"
    trial.data["status"] = "FAILED"
    trial.data["failure_reason"] = ADMISSION_TIMEOUT
    trial.data["result"] = {"status": "failed", "peak_memory_gb": 0.0}
    assert r._reenter_failed_trial(trial) == ADMISSION_TIMEOUT
    assert trial.data["attempts"][-1]["kind"] == "requeue", \
        "重入不消耗 retry/fix 预算"
    trial_log = (Path(trial.trial_dir) / "trial.log").read_text()
    assert "内存准入排队" in trial_log


def test_execute_once_admitted_attaches_pid_and_releases(tmp_path,
                                                          monkeypatch):
    run = tmp_path / "run"
    cfg = _cfg(memory_safety_margin_gb=60)
    trial = _mk_trial(run, "trial-001-g01-y",
                      expected_resources={"peak_memory_gb": 50})
    monkeypatch.setattr(sched, "system_memory", _fake_mem(500.0))
    seen = {}

    def _fake(cmd, **kw):
        kw["on_spawn"](os.getpid())
        res = list((run / ".admission").glob("*.json"))
        seen["child_pid"] = json.loads(res[0].read_text())["child_pid"]
        return ExecutionResult(True, 0, 0.1, 0.0, None,
                               Path(trial.trial_dir) / "stdout.log",
                               Path(trial.trial_dir) / "stderr.log")

    monkeypatch.setattr("runtime.runner.run_trial_process", _fake)
    r = _bare_runner(run, cfg)
    res = r._execute_once(trial, Path(trial.trial_dir) / "code",
                          Path(trial.trial_dir), ResourcePlan(), 1, 1)
    assert res.ok and seen["child_pid"] == os.getpid()
    assert trial.data["admission"]["state"] == "admitted"
    assert not list((run / ".admission").glob("*.json")), "训练结束释放预留"


def test_duck_typed_trial_skips_admission(tmp_path, monkeypatch):
    """无 trial_dir 的鸭子类型替身(旧测试/诊断)不做准入,不碰真实内存。"""
    run = tmp_path / "run"
    monkeypatch.setattr(sched, "system_memory", _fake_mem(0.0))
    monkeypatch.setattr("runtime.runner.run_trial_process",
                        lambda cmd, **kw: "ran")
    trial = SimpleNamespace(trial_id="t-fake",
                            data={"execution": {"command": "python x.py"},
                                  "environment": {}})
    r = _bare_runner(run, _cfg())
    assert r._execute_once(trial, run, run, ResourcePlan(), 1, 1) == "ran"


# ---------------------------------------------------------------------------
# 5. run-generation:按准入逐个启动(mock 执行体)
# ---------------------------------------------------------------------------

def test_run_generation_admits_one_by_one(tmp_path, monkeypatch):
    run = tmp_path / "run"
    cfg = _cfg(memory_safety_margin_gb=60, admission_poll_sec=0.05,
               admission_max_wait_min=1)
    trials = [_mk_trial(run, f"trial-00{i}-g01-z{i}",
                        expected_resources={"peak_memory_gb": 100})
              for i in (1, 2, 3)]
    # MemAvailable 恒 250(模拟内存爬升滞后):只靠预留防止一次全起
    monkeypatch.setattr(sched, "system_memory", _fake_mem(250.0))
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "starts": []}

    def _fake_proc(cmd, **kw):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            state["starts"].append(time.monotonic())
        time.sleep(0.3)
        with lock:
            state["active"] -= 1
        return ExecutionResult(True, 0, 0.3, 0.0, None, Path("o"), Path("e"))

    monkeypatch.setattr("runtime.runner.run_trial_process", _fake_proc)
    real = _bare_runner(run, cfg)

    def _run(trial, **kw):
        trial.data["status"] = "RUNNING"
        trial.flush()
        res = real._execute_once(trial, Path(trial.trial_dir) / "code",
                                 Path(trial.trial_dir), kw["plan"], 1.0, 1,
                                 slot_index=kw.get("slot_index", 0))
        trial.data["status"] = "EVALUATED" if res.ok else "FAILED"
        if not res.ok:
            trial.data["failure_reason"] = res.failure_reason
        trial.flush()
        return trial

    runner = SimpleNamespace(
        run=_run, run_dir=run, config=cfg,
        spec=SimpleNamespace(time_column=None, id_column=None,
                             primary_metric="auc"),
        breaker=FailureCircuitBreaker(cfg),
        resolve_timeout_minutes=lambda override=None: 5.0)
    out = GenerationExecutor(runner).execute_generation(
        trials, plan=ResourcePlan(n_parallel_trials=3, threads_per_trial=4),
        run_dir=run)
    assert state["max_active"] == 1, \
        f"内存只够 1 个:应逐个启动,实际最大并发 {state['max_active']}"
    assert out["n_succeeded"] == 3 and out["n_failed"] == 0
    waits = out["admission_wait_sec"]
    assert sorted(1 for w in waits.values() if w and w > 0) == [1, 1], \
        f"两个 trial 排过队: {waits}"
    summ = out["admission"]
    assert summ["throttled"] is True and summ["n_waited"] == 2
    assert "资源受限串行" in summ["reason"]
    evs = _events(run)
    assert sum(1 for e in evs if e["event"] == "admitted") == 3
    assert sum(1 for e in evs if e["event"] == "queued") == 2


def test_generation_breaker_ignores_admission_timeouts(tmp_path):
    cfg = _cfg()
    runner = SimpleNamespace(run_dir=tmp_path, config=cfg,
                             breaker=FailureCircuitBreaker(cfg),
                             _check_breaker=lambda *a: pytest.fail(
                                 "准入超时不应触发代级熔断"))
    summary = {"n_failed": 3, "n_succeeded": 1, "n_admission_timeout": 3}
    GenerationExecutor(runner)._generation_breaker(summary, tmp_path)
    assert "breaker" not in summary


# ---------------------------------------------------------------------------
# 6. admission_summary
# ---------------------------------------------------------------------------

def test_admission_summary(tmp_path):
    run = tmp_path / "run"
    assert admission_summary(run) == {
        "throttled": False, "waits_sec": 0.0, "n_waited": 0, "reason": "",
        "n_timeout": 0, "max_wait_sec": 0.0, "min_available_gb": None,
        "trials": {}}
    rows = [
        {"event": "admitted", "trial_id": "a", "generation": 4,
         "wait_sec": 0.0, "epoch": 1},
        {"event": "queued", "trial_id": "b", "generation": 4,
         "available_gb": 37.0, "epoch": 2, "pid": 4194303},
        {"event": "admitted", "trial_id": "b", "generation": 4,
         "wait_sec": 600.0, "queued": True, "min_available_gb": 23.0},
        {"event": "queued", "trial_id": "c", "generation": 4, "epoch": 3},
        {"event": "timeout", "trial_id": "c", "generation": 4,
         "wait_sec": 7200.0, "min_available_gb": 30.0},
        {"event": "queued", "trial_id": "d", "generation": 5, "epoch": 4},
        {"event": "admitted", "trial_id": "d", "generation": 5,
         "wait_sec": 60.0, "queued": True},
        {"event": "gpu_select", "trial_id": "a", "generation": 4},
    ]
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "admission.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\nnot-json\n")
    s4 = admission_summary(run, generation=4)
    assert set(s4) >= {"throttled", "waits_sec", "n_waited", "reason"}
    assert s4["throttled"] is True and s4["n_waited"] == 2
    assert s4["waits_sec"] == pytest.approx(7800.0)
    assert s4["n_timeout"] == 1 and s4["min_available_gb"] == 23.0
    assert "资源受限串行" in s4["reason"] and "admission_timeout" in s4["reason"]
    s5 = admission_summary(run, 5)
    assert s5["n_waited"] == 1 and s5["waits_sec"] == 60.0
    sall = admission_summary(run)
    assert sall["n_waited"] == 3
    assert admission_summary(run, 9)["throttled"] is False


# ---------------------------------------------------------------------------
# 7. GPU 按实际占用选卡
# ---------------------------------------------------------------------------

_GPUS_CSV = ("0, GPU-a, 81920, 30000, 100\n"
             "1, GPU-b, 81920, 1000, 0\n"
             "2, GPU-c, 81920, 20000, 5\n")


def _mock_smi(monkeypatch, cards=_GPUS_CSV, apps="GPU-a, 4194000, 29000\n",
              counter=None):
    def _smi(args):
        if counter is not None:
            counter.append(args[0])
        if args[0].startswith("--query-gpu"):
            return cards
        return apps
    monkeypatch.setattr(sched, "_nvidia_smi", _smi)


def _live_runner(run: Path) -> TrialRunner:
    snap_gpus = [{"index": i, "memory_free_gb": f, "util_pct": 0}
                 for i, f in ((0, 79.0), (1, 70.0), (2, 60.0))]
    r = _bare_runner(run, _cfg(), gpus=snap_gpus)
    r.scheduler.live = True
    return r


def _duck(tid="t-gpu"):
    return SimpleNamespace(trial_id=tid, data={"algorithm": "xgboost",
                                               "generation": 4})


def test_gpu_avoids_foreign_process_card(tmp_path, monkeypatch):
    run = tmp_path / "run"
    _mock_smi(monkeypatch)
    r = _live_runner(run)
    trial = _duck()
    plan = ResourcePlan(gpu_index=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    lease, gi = r._acquire_gpu_lease(trial, plan, 0)
    assert gi == 1, "plan 首选的 gpu0 有外来进程(100%)→ 应换到干净卡 gpu1"
    sel = trial.data["environment"]["gpu_selection"]
    assert sel["mode"] == "occupancy" and sel["foreign_procs"] == 0
    assert any(o["index"] == 0 and o["foreign_procs"] == 1
               for o in sel["occupied"])
    ev = [e for e in _events(run) if e["event"] == "gpu_select"]
    assert ev and ev[-1]["gpu"] == 1
    # 第二个 trial:gpu1 被本 run 租约占用 → 下一张干净卡 gpu2
    lease2, gi2 = r._acquire_gpu_lease(_duck("t2"), plan, 1)
    assert gi2 == 2
    # 第三个:干净卡用尽 → 退回旧排序(plan 首选 gpu0)并留痕
    t3 = _duck("t3")
    lease3, gi3 = r._acquire_gpu_lease(t3, plan, 2)
    assert gi3 == 0
    assert t3.data["environment"]["gpu_selection"]["mode"] == \
        "fallback_all_occupied"
    assert _events(run)[-1]["mode"] == "fallback_all_occupied"
    for le in (lease, lease2, lease3):
        le.release()


def test_gpu_query_failure_falls_back(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setattr(sched, "_nvidia_smi", lambda args: None)
    r = _live_runner(run)
    trial = _duck()
    plan = ResourcePlan(gpu_index=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    lease, gi = r._acquire_gpu_lease(trial, plan, 0)
    assert gi == 0, "查询失败 → 旧逻辑(plan 首选卡)"
    assert trial.data["environment"]["gpu_selection"]["mode"] == \
        "query_failed"
    lease.release()


def test_gpu_occupancy_cache_and_injected_snapshot(tmp_path, monkeypatch):
    calls: list = []
    _mock_smi(monkeypatch, counter=calls)
    a = sched.query_gpu_occupancy()
    b = sched.query_gpu_occupancy()
    assert a == b and len(calls) == 2, "TTL 内复用缓存(只查 1 轮 = 2 条命令)"
    a[0]["procs"].append({"pid": 1})
    assert len(sched.query_gpu_occupancy()[0]["procs"]) == 1, "返回副本"
    # 注入快照的调度器(旧测试/离线回放)不碰真实设备
    r = _bare_runner(tmp_path, _cfg(), gpus=[{"index": 0,
                                              "memory_free_gb": 79,
                                              "util_pct": 0}])
    assert r.scheduler.live is False and r._gpu_occupancy() is None


def test_gpu_foreign_attribution(tmp_path, monkeypatch):
    run = tmp_path / "run"
    code = run / "trials" / "t1" / "code"
    code.mkdir(parents=True)
    own = subprocess.Popen([sys.executable, "-c",
                            "import time; time.sleep(30)"], cwd=str(code))
    try:
        apps = (f"GPU-a, {own.pid}, 1000\n"     # 可见且 cwd 在本 run → 本 run
                "GPU-b, 4194001, 1000\n"        # 不可见 + 本 run 持租约 → 本 run
                "GPU-c, 4194002, 1000\n")       # 不可见 + 无租约 → 外来
        _mock_smi(monkeypatch, apps=apps)
        lease = GpuLease.acquire(run / ".gpu-leases", 1)
        cards = sched.annotate_gpu_occupancy(sched.query_gpu_occupancy(),
                                             run)
        by = {c["index"]: c for c in cards}
        assert by[0]["own_procs"] == 1 and by[0]["foreign_procs"] == 0
        assert by[1]["own_procs"] == 1 and by[1]["leased_by_run"] is True
        assert by[2]["foreign_procs"] == 1 and by[2]["clean"] is False
        lease.release()
    finally:
        own.kill()
        own.wait()
    # 进程查询失败:归属未知,按"看起来空闲"判干净
    monkeypatch.setitem(sched._GPU_OCC_CACHE, "data", None)
    _mock_smi(monkeypatch, apps=None)
    cards = sched.annotate_gpu_occupancy(sched.query_gpu_occupancy(), run)
    by = {c["index"]: c for c in cards}
    assert by[1]["foreign_procs"] is None and by[1]["clean"] is True
    assert by[0]["clean"] is False


# ---------------------------------------------------------------------------
# 8. snapshot 新字段
# ---------------------------------------------------------------------------

def test_snapshot_new_fields(tmp_path, monkeypatch, capsys):
    import _mvo.cmd_inspect as ci

    run = tmp_path / "run"
    for i, pk in enumerate((80.0, 100.0, 90.0, 20.0), start=1):
        _mk_trial(run, f"trial-00{i}-g01-s{i}", status="EVALUATED",
                  result={"peak_memory_gb": pk},
                  status_history=[{"state": "EVALUATED",
                                   "at": f"2026-10-23T1{i}:00:00+08:00"}])
    monkeypatch.setattr(ci, "resource_snapshot", lambda: {
        "cpu_physical": 64, "cpu_count": 128, "load1": 0.0,
        "ram_available_gb": 400.0, "gpus": []})
    monkeypatch.setattr(sched, "system_memory", _fake_mem(400.0))
    _mock_smi(monkeypatch)
    rc = ci.cmd_snapshot(SimpleNamespace(run=str(run), spec=None))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["system_memory"]["available_gb"] == 400.0
    adm = out["admission"]
    # 最近 3 个有峰值的 trial(004/003/002)最大峰值 100 × 1.15 = 115
    assert adm["per_trial_estimate_gb"] == pytest.approx(115.0)
    assert adm["estimate_source"] == "recent_trials"
    assert adm["suggested_parallel"] == 2   # floor((400 − 0 − 60) / 115)
    assert adm["memory_bound"] is True
    gpu = out["gpu_occupancy"]
    assert gpu["available"] is True
    by = {c["index"]: c for c in gpu["cards"]}
    assert by[0]["foreign_procs"] == 1 and by[0]["clean"] is False
    assert gpu["clean_order"] == [1, 2]
    assert "generation_batch" in out["plans_by_scope"]


# ---------------------------------------------------------------------------
# 9. wait:轻量 import 与幂等
# ---------------------------------------------------------------------------

def _wait_fixture(tmp_path: Path) -> tuple[Path, str]:
    run = tmp_path / "run"
    tid = "trial-001-g01-w"
    (run / "trials" / tid).mkdir(parents=True)
    (run / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "t", "task_name": "t", "objective": "o",
        "model_type": "binary_classification",
        "target": {"name": "y", "positive_definition": "y==1"},
        "primary_metric": "auc", "data": {"source": "x.parquet"},
        "split_policy": {"type": "random", "train": 0.7,
                         "validation": 0.3},
        "features": {"banned": {}, "reparse": {}}}))
    (run / "trials" / tid / "trial.yaml").write_text(yaml.safe_dump({
        "trial_id": tid, "algorithm": "lightgbm", "generation": 1,
        "status": "EVALUATED",
        "status_history": [{"state": "EVALUATED",
                            "at": "2026-10-23T00:00:00+08:00"}],
        "execution": {"command": "python train.py"},
        "evaluation": {"verdict": "accepted",
                       "primary_metric": {"value": 0.7}}}))
    return run, tid


def test_wait_is_lightweight_and_idempotent(tmp_path):
    run, tid = _wait_fixture(tmp_path)
    before = {p.name: p.stat().st_mtime_ns
              for p in run.rglob("*") if p.is_file()}
    probe = (
        "import json, resource, runpy, sys\n"
        f"sys.argv = [{str(SKILL / 'scripts' / 'modelevo.py')!r}, 'wait',"
        f" '--run', {str(run)!r}, '--trial', {tid!r}, '--timeout', '3']\n"
        "rc = None\n"
        "try:\n"
        "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    rc = e.code\n"
        "heavy = [m for m in ('pandas', 'numpy', 'pyarrow', 'sklearn',\n"
        "         'scipy', 'psutil', 'runtime.runner', 'runtime.scheduler',\n"
        "         'runtime.evaluator', '_mvo.cmd_inspect', '_mvo.cmd_trial')\n"
        "         if m in sys.modules]\n"
        # 峰值 RSS 取 /proc/self/status 的 VmHWM(按地址空间计,exec 后重置);
        # ru_maxrss 会把 fork 出来、exec 之前的 pytest 镜像也算进去
        "hwm = None\n"
        "try:\n"
        "    for ln in open('/proc/self/status'):\n"
        "        if ln.startswith('VmHWM:'):\n"
        "            hwm = int(ln.split()[1]) / 1024\n"
        "except OSError:\n"
        "    pass\n"
        "if hwm is None:\n"
        "    hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024\n"
        "print('PROBE ' + json.dumps({'rc': rc, 'heavy': heavy,\n"
        "      'rss_mb': hwm}))\n")
    for _ in range(2):  # 重入两次:结果一致且不写任何文件
        cp = subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True, timeout=120)
        line = [ln for ln in cp.stdout.splitlines()
                if ln.startswith("PROBE ")]
        assert line, cp.stdout + cp.stderr
        info = json.loads(line[0][6:])
        assert info["rc"] == 0, cp.stdout + cp.stderr
        assert info["heavy"] == [], f"wait 不得 import 重模块: {info}"
        assert info["rss_mb"] < 60, f"wait 峰值 RSS 过高: {info}"
        assert "SUMMARY\t" + tid in cp.stdout
        assert "WAIT_HINT" in cp.stdout and "137" in cp.stdout
    after = {p.name: p.stat().st_mtime_ns
             for p in run.rglob("*") if p.is_file()}
    assert before == after, "wait 只读:不得写任何文件"


def test_wait_help_and_queued_hint(tmp_path):
    cp = subprocess.run([sys.executable, str(SKILL / "scripts" /
                                             "modelevo.py"), "wait", "-h"],
                        capture_output=True, text=True, timeout=60)
    assert cp.returncode == 0 and "137" in cp.stdout
    run, tid = _wait_fixture(tmp_path)
    p = run / "trials" / tid / "trial.yaml"
    d = yaml.safe_load(p.read_text())
    d.update(status="RUNNING", admission={"state": "queued",
                                          "since": "x", "estimate_gb": 100})
    p.write_text(yaml.safe_dump(d))
    cp = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "modelevo.py"), "wait",
         "--run", str(run), "--trial", tid, "--timeout", "1"],
        capture_output=True, text=True, timeout=60)
    assert cp.returncode == 3
    assert "ADMISSION_QUEUED\t" + tid in cp.stdout


def test_light_stale_probe_matches_runner(tmp_path):
    from runtime.daemon import stale_running_probe

    r = TrialRunner.__new__(TrialRunner)

    def _mk(name, hb_age, lock_pid):
        d = tmp_path / name
        d.mkdir()
        if lock_pid:
            (d / ".runner.lock").write_text(f"{lock_pid}\n")
        if hb_age is not None:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S+08:00",
                               time.localtime(time.time() - hb_age))
            (d / "trial.log").write_text(f"[{ts}] heartbeat: runtime=1s\n")
        return SimpleNamespace(status="RUNNING", trial_dir=d)

    cases = [_mk("live", 9999, os.getpid()), _mk("dead", 600, 4194303),
             _mk("fresh", 30, 4194303), _mk("nolog", None, None)]
    for t in cases:
        assert stale_running_probe(t.trial_dir, "RUNNING")[0] == \
            r.is_stale_running(t)[0], t.trial_dir.name
    assert stale_running_probe(cases[0].trial_dir, "EVALUATED") == \
        (False, "")


# ---------------------------------------------------------------------------
# 10. 配置键
# ---------------------------------------------------------------------------

def test_config_keys_in_defaults_and_global_config():
    keys = ("memory_admission", "memory_safety_margin_gb",
            "memory_estimate_default_gb", "admission_poll_sec",
            "admission_max_wait_min")
    d = Config.load(skill_config=False)
    g = yaml.safe_load((SKILL / "assets" / "global-config.yaml").read_text())
    gex = (g.get("modelevo", g))["execution"]
    for k in keys:
        assert d.get(f"execution.{k}") is not None, k
        assert gex.get(k) == d.get(f"execution.{k}"), k
    assert d.get("execution.memory_admission") is True
    assert d.get("execution.memory_safety_margin_gb") == 60
