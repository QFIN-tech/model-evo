"""v2.19.0 P1「效率」:并发 / 内存校准 / HPO 跨卡+限时 / confirm seed 分卡并行 /
finalize 复用冻结模型 / warm-start 去重。

全部离线、CPU、合成小数据;GPU 相关路径用注入的占用快照与探测函数模拟。
"""
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from runtime.config import Config  # noqa: E402
from runtime import hpo as H  # noqa: E402
from runtime import scheduler as S  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 并发:GPU trial 线程 = 可用核 / min(可用卡数, 本批 trial 数)
# ---------------------------------------------------------------------------

def _snap(n_gpu=8, cpu=128, load=0.0, free_gb=70.0, util=0):
    return {"cpu_count": cpu, "cpu_physical": cpu // 2, "load1": load,
            "ram_total_gb": 1000, "ram_available_gb": 600,
            "gpus": [{"index": i, "memory_total_gb": 80,
                      "memory_free_gb": free_gb, "util_pct": util}
                     for i in range(n_gpu)]}


def test_plan_gpu_eight_cards_reaches_eight_way():
    cfg = Config.load()
    p = S.Scheduler(cfg, _snap()).plan(planned_trials=8)
    usable = int(128 * 0.9) - 2
    assert p.n_parallel_trials == 8
    assert p.threads_per_trial == min(24, max(8, usable // 8))
    assert len(p.gpu_indices) == 8


def test_plan_gpu_slots_capped_by_cards_and_threads_clamped():
    cfg = Config.load()
    p = S.Scheduler(cfg, _snap(n_gpu=2)).plan(planned_trials=8)
    assert p.n_parallel_trials == 2
    assert p.threads_per_trial == 24, "线程钳到 gpu_trial_threads_max"
    cfg2 = Config({**cfg.snapshot(), "execution": {
        **cfg.get("execution"), "gpu_trial_threads_max": 40}})
    p2 = S.Scheduler(cfg2, _snap(n_gpu=2)).plan(planned_trials=8)
    assert p2.threads_per_trial == 40


def test_plan_gpu_floor_limits_slots_when_cores_scarce():
    cfg = Config.load()
    # usable = 32*0.9-2 = 26 → 下限 8 线程最多 3 路
    p = S.Scheduler(cfg, _snap(n_gpu=8, cpu=32)).plan(planned_trials=8)
    assert p.n_parallel_trials == 3
    assert p.threads_per_trial == 8


def test_plan_cpu_only_keeps_old_logic():
    cfg = Config.load()
    snap = _snap(n_gpu=0)
    new = S.Scheduler(cfg, snap).plan(planned_trials=8)
    cfg_off = Config({**cfg.snapshot(), "execution": {
        **cfg.get("execution"), "gpu_aware_parallel": False}})
    old = S.Scheduler(cfg_off, snap).plan(planned_trials=8)
    assert (new.n_parallel_trials, new.threads_per_trial) == \
        (old.n_parallel_trials, old.threads_per_trial)


def test_plan_busy_gpus_not_counted():
    cfg = Config.load()
    snap = _snap(n_gpu=8)
    for g in snap["gpus"][:6]:
        g["memory_free_gb"] = 5.0
    p = S.Scheduler(cfg, snap).plan(planned_trials=8)
    assert p.n_parallel_trials == 2


# ---------------------------------------------------------------------------
# 2. 内存预估校准:同类实测 × 1.25
# ---------------------------------------------------------------------------

def _trial(run, tid, peak=None, parents=(), view="v1", at="2026-11-01T00:00:00",
           status="EVALUATED", attempts=None):
    d = run / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    data = {"trial_id": tid, "parent_trial_ids": list(parents),
            "data_view": view, "status": status,
            "status_history": [{"status": status, "at": at}]}
    if peak is not None:
        data["result"] = {"peak_memory_gb": peak}
    if attempts:
        data["attempts"] = attempts
    (d / "trial.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return data


def test_memory_estimate_uses_peer_peaks(tmp_path):
    run = tmp_path / "run"
    _trial(run, "trial-002", peak=141.0)
    _trial(run, "trial-004", peak=60.0, parents=["trial-002"],
           at="2026-11-01T01:00:00")
    _trial(run, "trial-005", peak=64.0, parents=["trial-002"],
           at="2026-11-01T02:00:00")
    _trial(run, "trial-006", peak=200.0, parents=["trial-002"], view="v2",
           at="2026-11-01T03:00:00")  # 别的视图不算同类
    me = _trial(run, "trial-009", parents=["trial-002"], status="PROPOSED")
    est = S.estimate_trial_peak_gb(run, me, Config.load())
    assert est["source"] == "peer"
    assert est["estimate_gb"] == pytest.approx(64.0 * 1.25, abs=0.01)
    assert est["basis_trial"] == "trial-005"


def test_memory_estimate_peer_includes_grandchildren(tmp_path):
    run = tmp_path / "run"
    _trial(run, "trial-002", peak=141.0)
    _trial(run, "trial-004", peak=60.0, parents=["trial-002"])
    _trial(run, "trial-007", peak=70.0, parents=["trial-004"],
           at="2026-11-01T05:00:00")
    me = _trial(run, "trial-009", parents=["trial-002"], status="PROPOSED")
    est = S.estimate_trial_peak_gb(run, me, Config.load())
    assert est["source"] == "peer"
    assert est["estimate_gb"] == pytest.approx(70.0 * 1.25, abs=0.01)


def test_memory_estimate_falls_back_without_peers(tmp_path):
    run = tmp_path / "run"
    _trial(run, "trial-002", peak=100.0)
    me = _trial(run, "trial-004", parents=["trial-002"], status="PROPOSED")
    est = S.estimate_trial_peak_gb(run, me, Config.load())
    assert est["source"] == "parent"
    assert est["estimate_gb"] == pytest.approx(100.0 * S.PEAK_SAFETY_FACTOR,
                                               abs=0.01)


def test_memory_estimate_own_attempt_keeps_old_order(tmp_path):
    run = tmp_path / "run"
    _trial(run, "trial-002", peak=100.0)
    _trial(run, "trial-003", peak=40.0, parents=["trial-002"])
    me = _trial(run, "trial-004", parents=["trial-002"], status="FAILED",
                attempts=[{"previous_result": {"peak_memory_gb": 150.0}}])
    est = S.estimate_trial_peak_gb(run, me, Config.load())
    assert est["source"] == "parent", "有自身 attempt 实测时不走同类校准"


def test_admission_view_uses_pending_calibrated_estimate(tmp_path):
    run = tmp_path / "run"
    _trial(run, "trial-002", peak=141.0, at="2026-11-01T00:00:00")
    for i, pk in enumerate((60.0, 62.0, 61.0)):
        _trial(run, f"trial-00{4 + i}", peak=pk, parents=["trial-002"],
               at=f"2026-11-01T0{1 + i}:00:00")
    for i in range(8):
        _trial(run, f"trial-01{i}", parents=["trial-002"], status="PROPOSED")
    view = S.resource_admission_view(
        run, Config.load(), max_parallel=8,
        memory={"available_gb": 750.0})
    assert view["per_trial_estimate_gb"] == pytest.approx(62.0 * 1.25,
                                                          abs=0.01)
    assert view["suggested_parallel"] == 8


# ---------------------------------------------------------------------------
# 3. HPO:搜索墙钟上限 + 跨卡并行
# ---------------------------------------------------------------------------

class _FakeClock:
    def __init__(self, t0=1_000_000.0):
        self.t = t0
        self.lock = threading.Lock()

    def time(self):
        with self.lock:
            return self.t

    def advance(self, s):
        with self.lock:
            self.t += s


def test_hpo_search_cap_truncates_and_keeps_best(monkeypatch):
    clk = _FakeClock()
    monkeypatch.setattr(H.time, "time", clk.time)
    space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}

    def fit(p):
        clk.advance(25.0)            # 每个 config "耗时" 25 s
        return {"value": -abs(p["x"] - 0.3), "best_iteration": 10}

    r = H.run_hpo_search(space, 40, fit, metric_direction_up=True, seed=1,
                         max_search_minutes=1, multi_gpu="off")
    assert r["search_capped"] is True
    assert r["budget_truncated_by_deadline"] is False
    # 60 s 上限、25 s/config:第 2 个完成时 now+25 > 上限 → 停止派发
    assert 1 <= r["n_configs_evaluated"] <= 3 < 40
    vals = [t["inner_metric"] for t in r["trials"]]
    assert r["inner_score"] == max(vals), "用已完成 config 中最优者"
    assert r["max_search_minutes"] == 1


def test_hpo_search_cap_merges_with_deadline(monkeypatch):
    """deadline 更早时由 deadline 外推截断(不记 search_capped)。"""
    clk = _FakeClock()
    monkeypatch.setattr(H.time, "time", clk.time)
    space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}

    def fit(p):
        clk.advance(25.0)
        return -abs(p["x"] - 0.3)

    r = H.run_hpo_search(space, 40, fit, metric_direction_up=True, seed=1,
                         deadline_ts=clk.time() + 60, max_search_minutes=600,
                         multi_gpu="off")
    assert r["search_capped"] is False
    assert r["budget_truncated_by_deadline"] is True
    assert r["n_configs_evaluated"] < 40


def test_hpo_no_cap_without_run_dir_keeps_old_behavior():
    space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}
    r = H.run_hpo_search(space, 5, lambda p: -abs(p["x"] - 0.3),
                         metric_direction_up=True, seed=1)
    assert r["n_configs_evaluated"] == 5
    assert r["search_capped"] is False and r["max_search_minutes"] is None
    assert r["multi_gpu"]["reason"] == "config_off"


def test_hpo_cap_read_from_run_config(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "trials" / "trial-001").mkdir(parents=True)
    (run / "modelevo-config.yaml").write_text(yaml.safe_dump(
        {"modelevo": {"evolution": {"hpo": {"max_search_minutes": 7,
                                            "multi_gpu": "off"}}}}))
    space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}
    r = H.run_hpo_search(space, 3, lambda p: -abs(p["x"] - 0.3),
                         metric_direction_up=True, seed=1,
                         checkpoint_path=run / "trials" / "trial-001"
                         / "hpo-checkpoint.json")
    assert r["max_search_minutes"] == 7
    assert r["multi_gpu"]["reason"] == "config_off"


def test_bind_device_ordinals_one_worker_per_card():
    seen, active, peak = [], [0], [0]
    lk = threading.Lock()

    def fn(p):
        with lk:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        o = H.hpo_device_ordinal()
        if o:
            H._HPO_DEVICE.consumed = True   # 模拟 build_gbm 消费序号
        seen.append(o)
        import time as _t
        _t.sleep(0.02)
        with lk:
            active[0] -= 1
        return 0.0

    w, st = H._bind_device_ordinals(fn, [0, 1, 2, 3], fallback_jobs=2)
    ths = [threading.Thread(target=w, args=({},)) for _ in range(12)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert st["honored"] is True
    assert set(seen) == {0, 1, 2, 3}
    assert peak[0] <= 4
    assert H.hpo_device_ordinal() is None


def test_bind_device_ordinals_fallback_when_not_honored():
    active, peak = [0], [0]
    lk = threading.Lock()

    def fn(p):   # 不经 build_gbm:序号不被消费
        with lk:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        import time as _t
        _t.sleep(0.02)
        with lk:
            active[0] -= 1
        return 0.0

    w, st = H._bind_device_ordinals(fn, [0, 1, 2, 3], fallback_jobs=2)
    ths = [threading.Thread(target=w, args=({},)) for _ in range(12)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert st["honored"] is False
    assert peak[0] <= 2 + 1, "验证期最多主卡 + 探测卡,之后回落 2 路"


def test_build_gbm_honors_thread_ordinal():
    pytest.importorskip("xgboost")
    H._HPO_DEVICE.ordinal = 2
    try:
        m = H.build_gbm("xgboost", {}, 0, "binary_classification", 10, True)
        assert m.get_params()["device"] == "cuda:2"
        assert H._HPO_DEVICE.consumed is True
    finally:
        H._HPO_DEVICE.ordinal = None
    m = H.build_gbm("xgboost", {}, 0, "binary_classification", 10, True)
    assert m.get_params()["device"] == "cuda"


def _occ(n=4):
    return [{"index": i, "memory_free_gb": 70.0, "memory_used_gb": 0.1,
             "util_pct": 0, "procs": [], "procs_known": True}
            for i in range(n)]


def test_acquire_hpo_gpus_leases_and_widens_env(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "trials").mkdir(parents=True)
    monkeypatch.setenv("MODELEVO_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    rec = H.acquire_hpo_gpus(run, 3, probe=lambda o: True, occupancy=_occ())
    try:
        assert rec["reason"] == "ok"
        assert rec["gpus"][0] == 2 and len(rec["gpus"]) == 3
        assert os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0] == "2"
        assert rec["ordinals"] == [0, 1, 2]
        assert len(list((run / ".gpu-leases").glob("gpu-*.lock"))) == 2
    finally:
        H.release_hpo_gpus(rec)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert not list((run / ".gpu-leases").glob("gpu-*.lock"))


def test_acquire_hpo_gpus_cuda_initialized_falls_back(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "trials").mkdir(parents=True)
    monkeypatch.setenv("MODELEVO_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    rec = H.acquire_hpo_gpus(run, 4, probe=lambda o: False, occupancy=_occ())
    assert rec["reason"] == "cuda_already_initialized"
    assert rec["ordinals"] == []
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert not list((run / ".gpu-leases").glob("gpu-*.lock"))


def test_acquire_hpo_gpus_no_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELEVO_GPU", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert H.acquire_hpo_gpus(tmp_path, 4)["reason"] == "no_gpu_allocated"


def test_run_hpo_search_multi_gpu_records_parallel_gpus(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "trials" / "trial-001").mkdir(parents=True)
    fake = {"ordinals": [0, 1, 2], "gpus": [5, 6, 7], "reason": "ok"}
    monkeypatch.setattr(H, "acquire_hpo_gpus", lambda rd, n: dict(fake))
    monkeypatch.setenv("MODELEVO_GPU", "1")
    seen = set()

    def fit(p):
        o = H.hpo_device_ordinal()
        H._HPO_DEVICE.consumed = True
        seen.add(o)
        return -abs(p["x"] - 0.3)

    space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}
    r = H.run_hpo_search(space, 9, fit, metric_direction_up=True, seed=1,
                         checkpoint_path=run / "trials" / "trial-001"
                         / "hpo-checkpoint.json", n_jobs=2,
                         multi_gpu="auto", max_search_minutes=0)
    assert r["parallel_gpus"] == 3 and r["parallel"] == 3
    assert r["multi_gpu"]["gpus"] == [5, 6, 7]
    assert seen <= {0, 1, 2} and len(seen) >= 2
    assert r["n_configs_evaluated"] == 9


# ---------------------------------------------------------------------------
# 4. confirm seed 分卡并行(runtime.seed_parallel)
# ---------------------------------------------------------------------------

_FAKE_TRIAL = '''
import numpy as np
SEED = 0


class _M:
    def __init__(self, seed):
        self.seed = seed

    def predict_proba(self, X):
        p = (np.asarray(X)[:, 0] * 0 + (self.seed % 97) / 100.0)
        return np.c_[1 - p, p]


def _fit_seed_variant(n_jobs, X_fit, y_fit, eval_set):
    return _M(SEED)


def align_predict_device(m):
    pass


if __name__ == "__main__":
    raise SystemExit("不应被执行")
'''


def test_seed_parallel_worker_and_finish(tmp_path, monkeypatch):
    from runtime import seed_parallel as SP

    run = tmp_path / "run"
    td = run / "trials" / "trial-009-g02-confirm"
    (td / "code").mkdir(parents=True)
    script = td / "code" / "train.py"
    script.write_text(_FAKE_TRIAL, encoding="utf-8")
    plan_calls = []

    def _plan(rd, tdir, n_todo, occupancy=None, data_gb=0.0):
        plan_calls.append(n_todo)
        return {"mode": "parallel", "reason": "ok", "gpus": [3, 4][:n_todo],
                "n_parallel": 1 + min(2, n_todo)}

    monkeypatch.setattr(SP, "plan_seed_parallel", _plan)
    X = pd.DataFrame({"a": np.arange(5, dtype="float32")})
    y = pd.Series([0, 1, 0, 1, 1])
    h = SP.start_seed_workers(run, td, script, [1, 2], 7, 1, X, y,
                              [(X, y)], X, "binary_classification")
    assert h["assigned"] == [2], "todo[0] 留给主进程,其余给 worker"
    assert plan_calls == [1]
    rec = SP.finish_seed_workers(h)
    assert rec["workers"]["2"]["ok"] is True, rec
    assert rec["workers"]["2"]["gpu"] == 3
    out = pd.read_parquet(td / "seed_scores" / "seed-2.parquet")["score"]
    assert np.allclose(out, ((7 + 2000) % 97) / 100.0), "seed = seed0 + 1000k"
    assert h["data_path"] is None, "数据文件已清理"


def test_seed_parallel_serial_when_plan_rejects(tmp_path, monkeypatch):
    from runtime import seed_parallel as SP

    monkeypatch.setenv("MODELEVO_GPU", "0")
    td = tmp_path / "run" / "trials" / "t"
    td.mkdir(parents=True)
    X = pd.DataFrame({"a": [1.0, 2.0]})
    h = SP.start_seed_workers(tmp_path / "run", td, td / "x.py", [1, 2], 0, 1,
                              X, X["a"], [(X, X["a"])], X,
                              "binary_classification")
    assert h["assigned"] == []
    assert h["record"]["reason"] == "no_gpu_allocated"
    assert SP.finish_seed_workers(h)["mode"] == "serial"


def test_seed_parallel_plan_config_off(tmp_path, monkeypatch):
    from runtime import seed_parallel as SP

    monkeypatch.setenv("MODELEVO_GPU", "1")
    monkeypatch.setenv("MODELEVO_CONFIRM_PARALLEL_SEEDS", "off")
    assert SP.plan_seed_parallel(tmp_path, tmp_path, 2)["reason"] == \
        "config_off"


def test_seed_parallel_plan_leases_extra_clean_cards(tmp_path, monkeypatch):
    from runtime import seed_parallel as SP

    run = tmp_path / "run"
    td = run / "trials" / "t"
    td.mkdir(parents=True)
    monkeypatch.setenv("MODELEVO_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("MODELEVO_CONFIRM_PARALLEL_SEEDS", raising=False)
    monkeypatch.setattr(S, "system_memory",
                        lambda **kw: {"available_gb": 900.0})
    monkeypatch.setattr(S.MemoryAdmission, "evaluate",
                        lambda self, need: {"ok": True, "decision": "fits",
                                            "available_gb": 900.0,
                                            "headroom_gb": 800.0})
    plan = SP.plan_seed_parallel(run, td, 2, occupancy=_occ(4))
    try:
        assert plan["mode"] == "parallel", plan
        assert len(plan["gpus"]) == 2 and 0 not in plan["gpus"]
    finally:
        SP._release(plan)


def test_baseline_template_seed_segment_uses_parallel_helper():
    src = (SKILL / "runtime" / "baseline.py").read_text(encoding="utf-8")
    assert "from runtime.seed_parallel import start_seed_workers" in src
    assert "finish_seed_workers" in src


# ---------------------------------------------------------------------------
# 5. finalize 复用冻结模型
# ---------------------------------------------------------------------------

def _deliver():
    from _mvo import cmd_deliver
    return cmd_deliver


def _frozen_dir(tmp_path, model, X, name="trial-001", perturb=0.0):
    import joblib

    d = tmp_path / name
    (d / "model").mkdir(parents=True)
    joblib.dump(model, d / "model" / "model.joblib")
    sc = model.predict_proba(X)[:, 1] + perturb
    pd.DataFrame({"score": sc}).to_parquet(d / "validation_score.parquet",
                                           index=False)
    (d / "metrics.json").write_text(json.dumps({"best_iteration": 7}))
    return d


def _small_model():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0.5).astype(int)
    return LogisticRegression().fit(X, y), X


def test_frozen_reuse_passes_on_bitwise_match(tmp_path):
    cd = _deliver()
    m, X = _small_model()
    d = _frozen_dir(tmp_path, m, X)
    spec = types.SimpleNamespace(model_type="binary_classification")
    model, rep = cd._frozen_model_reuse([d], None, X, spec)
    assert model is not None and rep["passed"] is True
    assert rep["source_trial"] == "trial-001"
    assert rep["max_abs_score_diff"] == 0.0
    assert rep["attempts"][0]["trial_best_iteration"] == 7


def test_frozen_reuse_rejects_mismatch_and_tries_next(tmp_path):
    cd = _deliver()
    m, X = _small_model()
    bad = _frozen_dir(tmp_path, m, X, "trial-001", perturb=1e-6)
    good = _frozen_dir(tmp_path, m, X, "trial-002-confirm")
    spec = types.SimpleNamespace(model_type="binary_classification")
    model, rep = cd._frozen_model_reuse([bad], None, X, spec)
    assert model is None and rep["passed"] is False
    assert "不一致" in rep["attempts"][0]["reason"]
    model, rep = cd._frozen_model_reuse([bad, good], None, X, spec)
    assert rep["passed"] is True and rep["source_trial"] == "trial-002-confirm"


def test_frozen_reuse_feature_face_mismatch_falls_back(tmp_path):
    cd = _deliver()
    m, X = _small_model()
    d = _frozen_dir(tmp_path, m, X)
    X2 = X.copy()
    X2["a"] = X2["a"] + 0.01          # 交付管线重建的特征帧与冠军不同
    spec = types.SimpleNamespace(model_type="binary_classification")
    model, rep = cd._frozen_model_reuse([d], None, X2, spec)
    assert model is None and rep["passed"] is False


def test_frozen_reuse_missing_artifacts_and_uplift(tmp_path):
    cd = _deliver()
    m, X = _small_model()
    d = _frozen_dir(tmp_path, m, X)
    (d / "validation_score.parquet").unlink()
    spec = types.SimpleNamespace(model_type="binary_classification")
    assert cd._frozen_model_reuse([d], None, X, spec)[0] is None
    up = types.SimpleNamespace(model_type="uplift")
    assert cd._frozen_model_reuse([d], None, X, up)[1]["reason"].startswith(
        "uplift")


def test_reuse_candidates_include_confirm_children(tmp_path):
    cd = _deliver()
    T = types.SimpleNamespace
    trials = [T(trial_id="t1", parent_ids=[], data={}),
              T(trial_id="t2", parent_ids=["t1"],
                data={"search_action": "confirm"}),
              T(trial_id="t3", parent_ids=["t1"],
                data={"search_action": "explore"})]
    dirs = cd._reuse_candidate_dirs(tmp_path, "t1", trials)
    assert [p.name for p in dirs] == ["t1", "t2"]


def _load_delivery_fixture():
    p = pathlib.Path(__file__).resolve().parent / \
        "test_v2_17_0_delivery_report.py"
    spec = importlib.util.spec_from_file_location("_v2170_fixture", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finalize_reuse_equivalent_to_full_reproduction(tmp_path, monkeypatch):
    """复用路径与 --full-reproduction 产出同一交付模型(冻结轮数 + 预测一致)。"""
    pytest.importorskip("xgboost")
    import joblib

    fx = _load_delivery_fixture()
    out = {}
    for mode in ("reuse", "full"):
        base = tmp_path / mode
        base.mkdir()
        with monkeypatch.context() as mp:
            run, info = fx._make_run(base, kind="sk")
            fx._install_fakes(mp, run, info)
            rc = fx.cmd_deliver.cmd_finalize_retrain(fx._retrain_args(
                run, full_reproduction=(mode == "full")))
            assert rc == 0
            meta = fx._meta(run)
            model = joblib.load(run / "final-model" / "model" / "model.joblib")
            Xv = pd.read_parquet(info["champ_dir"] / "model"
                                 / "validation_input.parquet")
            out[mode] = (meta, model.predict(Xv), fx._calls(run))
    m_r, p_r, c_r = out["reuse"]
    m_f, p_f, c_f = out["full"]
    assert m_r["reproduction_mode"] == "reused_frozen_model"
    assert m_f["reproduction_mode"] == "full_reproduction_forced"
    assert c_r == [] and len(c_f) == 1
    assert m_r["frozen_rounds_params"] == m_f["frozen_rounds_params"]
    assert np.allclose(p_r, p_f, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. warm-start 去重指纹(代码 + config + 附带模块)
# ---------------------------------------------------------------------------

_CODE = '''import pathlib
RUN_DIR = pathlib.Path("/a/run")
SEED = 1
X = 3


def f():
    return X
'''


def _recipe(d, code=_CODE, cfg=None, extra=None):
    d.mkdir(parents=True)
    (d / "train.py").write_text(code, encoding="utf-8")
    (d / "trial.yaml").write_text(yaml.safe_dump(cfg or {
        "trial_id": "x", "notes": "n", "hyperparameters": {"max_depth": 8}}))
    if extra:
        (d / "code").mkdir()
        for k, v in extra.items():
            (d / "code" / k).write_text(v)
    return d


def _wfp():
    from _mvo.warmstart import warm_recipe_fingerprint
    return warm_recipe_fingerprint


def test_warm_fp_ignores_volatile_config_and_layout(tmp_path):
    fp = _wfp()
    a = _recipe(tmp_path / "a")
    code_b = _CODE.replace('"/a/run"', '"/b/run"').replace("SEED = 1",
                                                           "SEED = 9")
    code_b = "# 注释\n" + code_b
    b = _recipe(tmp_path / "b", code=code_b, cfg={
        "trial_id": "y", "notes": "另一条备注",
        "hyperparameters": {"max_depth": 8}})
    assert fp(a) == fp(b) != "unparsable"


def test_warm_fp_config_or_module_difference_not_deduped(tmp_path):
    fp = _wfp()
    a = _recipe(tmp_path / "a")
    b = _recipe(tmp_path / "b", cfg={"trial_id": "x", "notes": "n",
                                     "hyperparameters": {"max_depth": 6}})
    c = _recipe(tmp_path / "c", extra={"deep_seq.py": "K = 1\n"})
    d = _recipe(tmp_path / "d", extra={"deep_seq.py": "K = 2\n"})
    assert fp(a) != fp(b), "超参不同 = 训练行为不同,不能去重"
    assert fp(c) != fp(d) and fp(a) != fp(c)


def test_warm_fp_unparsable(tmp_path):
    fp = _wfp()
    a = _recipe(tmp_path / "a", code="def (:\n")
    assert fp(a) == "unparsable"


def test_cmd_startup_uses_warm_recipe_fingerprint():
    src = (SKILL / "scripts" / "_mvo" / "cmd_startup.py").read_text(
        encoding="utf-8")
    assert "warm_recipe_fingerprint(_rdir)" in src
    assert "不重复创建 trial" in src
