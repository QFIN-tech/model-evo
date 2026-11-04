"""v2.13 回归测试:E1 跨代动态 alpha spending + E2 代级 deadline 口径。

事故来源(2026-09,真实 run 排障):
  E1 旧 `_apply_generation_fdr` 按 ``alpha_total / 2^(gid+1)`` 几何收紧:
     gen-06 的 q=0.00078 < 2000 次 bootstrap 的单侧 p 分辨率 0.0005,
     真实 +2.5σ 提升(p≈0.006)结构性不可能通过;且当代只有 confirm
     trial(无提升主张)时 alpha 照样被消耗,后半程预算被空转烧光。
  E2 旧 `execute_generation` 的 barrier deadline 只看 wall 余量
     (实测只剩 214s),而单 trial 需 250~400s ⇒ 4/6 被 SIGTERM、
     触发熔断、17% compute 白烧;且批路径从不读 trial 的
     ``expected_resources.timeout_multiplier``,heavy/hpo 的 4× 放大
     只在单 trial 路径生效。

本文件锁定:
  E1a 有限代数动态均摊:8 代、alpha_total=0.10 → 每代 q_used=0.0125,
      总花费 ≤ alpha_total;
  E1b family_size==0 不扣费,且后续代 q_used 单调不减;
  E1c 同一代重复 finalize 幂等,不重复扣费;
  E1d 旧 evolution-state.json(无 alpha_spent 键)可加载并正常工作;
  E2a per_trial_timeout_sec 按 timeout_multiplier 放大(封顶生效按封顶);
  E2b wall 余量不足 → fail-closed,整代不启动;
  E2c wall_remaining_sec=None → 不预检,行为与旧版一致(会启动)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic import bootstrap

from runtime.config import Config
from runtime.evolution import EvolutionEngine
from runtime.gen_executor import GenerationExecutor
from runtime.scheduler import ResourcePlan
from runtime.spec import TaskSpec
from runtime.trial import TrialRecord


def _spec(max_generations: int = 8) -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v213", "task_name": "v213",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "resource_budget": {"max_generations": max_generations},
        "data": {},
    })


def _trial(run_dir: Path, tid: str, gen: int, *, p_better=None,
           verdict: str = "accepted", action: str = "explore",
           metric: float = 0.80) -> TrialRecord:
    """落盘 trial 桩(E1 需要工厂能从 trials/ 重读)。"""
    d = run_dir / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    ev: dict = {"primary_metric": {"name": "auc", "value": metric},
                "verdict": verdict, "integrity": {"valid": True}}
    if p_better is not None:
        ev["delta"] = {"vs_parent_paired_bootstrap": {"p_better": p_better}}
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "lightgbm", "generation": gen,
        "status": "EVALUATED", "search_action": action,
        "parent_trial_ids": [], "evaluation": ev,
        "status_history": [{"state": "EVALUATED", "at": "2026-09-19T00:00:00+08:00"}],
    }, trial_dir=d)
    rec.flush()
    return rec


def _engine(run_dir: Path, max_generations: int = 8) -> EvolutionEngine:
    return EvolutionEngine(run_dir, _spec(max_generations),
                           Config.load(skill_config=False))


# ---------------------------------------------------------------------------
# E1a: 有限代数动态均摊
# ---------------------------------------------------------------------------
def test_e1a_dynamic_alpha_amortized(tmp_path: Path):
    run = tmp_path / "run"
    eng = _engine(run, max_generations=8)
    # gen-00 baseline 先 finalize(占 finalized_generations,不扣费:
    # root trial 无 p_better → family_size=0)
    _trial(run, "t-base", 0, metric=0.78)
    eng.finalize_generation(0, ["t-base"], wall_time_hours=0.1)
    assert eng.state["alpha_spent"] == 0.0

    spent = 0.0
    for gid in (1, 2, 3, 4):
        _trial(run, f"t-g{gid}", gid, p_better=0.96, metric=0.80 + gid / 100)
        eng = _engine(run, max_generations=8)  # 重读 state(真实调用路径)
        report = eng.finalize_generation(
            gid, [f"t-g{gid}"], wall_time_hours=0.2)
        # 8 代均摊:alpha_remaining/gens_remaining = 0.10/(8-n_finalized)
        assert report is not None
        st = json.loads((run / "evolution-state.json").read_text())
        assert st["alpha_spent"] == pytest.approx(spent + 0.0125, abs=1e-9), \
            f"gen-{gid} 扣费错误: {st['alpha_spent']}"
        spent = st["alpha_spent"]
    assert spent == pytest.approx(0.05, abs=1e-9)
    assert spent <= 0.10


# ---------------------------------------------------------------------------
# E1b: family_size==0 不扣费;后代 q_used 单调不减
# ---------------------------------------------------------------------------
def test_e1b_zero_family_no_charge(tmp_path: Path):
    run = tmp_path / "run"
    eng = _engine(run, max_generations=8)
    # gen-01:只有 confirm trial(无提升主张)
    _trial(run, "t-confirm", 1, p_better=None, action="confirm")
    eng.finalize_generation(1, ["t-confirm"], wall_time_hours=0.1)
    st1 = json.loads((run / "evolution-state.json").read_text())
    assert st1["alpha_spent"] == 0.0
    assert st1["alpha_ledger"] == []

    # gen-02:有提升主张 → 扣费;剩余代数少一代 → q_used ≥ 上一代
    _trial(run, "t-g2", 2, p_better=0.96)
    eng = _engine(run, max_generations=8)
    eng.finalize_generation(2, ["t-g2"], wall_time_hours=0.1)
    st2 = json.loads((run / "evolution-state.json").read_text())
    assert st2["alpha_spent"] > 0.0
    # 单调性:gen-02(均摊分母 7)不小于假想的 gen-01(分母 8)口径
    q_gen2 = st2["alpha_spent"]
    assert q_gen2 >= 0.10 / 8 - 1e-9


# ---------------------------------------------------------------------------
# E1c: 同一代重复 finalize 幂等,不重复扣费
# ---------------------------------------------------------------------------
def test_e1c_idempotent_no_double_charge(tmp_path: Path):
    run = tmp_path / "run"
    eng = _engine(run, max_generations=8)
    _trial(run, "t-g1", 1, p_better=0.96)
    eng.finalize_generation(1, ["t-g1"], wall_time_hours=0.1)
    st1 = json.loads((run / "evolution-state.json").read_text())
    charged = st1["alpha_spent"]
    assert charged > 0.0

    # 重复 finalize 同一代:状态推进幂等守卫已有,这里锁 alpha 不双扣
    eng = _engine(run, max_generations=8)
    eng.finalize_generation(1, ["t-g1"], wall_time_hours=0.1)
    st2 = json.loads((run / "evolution-state.json").read_text())
    assert st2["alpha_spent"] == pytest.approx(charged, abs=1e-12)
    assert len([e for e in st2["alpha_ledger"]
                if e["generation"] == 1]) == 1


# ---------------------------------------------------------------------------
# E1d: 旧 evolution-state.json(无 alpha_spent 键)兼容加载
# ---------------------------------------------------------------------------
def test_e1d_legacy_state_loads(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir(parents=True, exist_ok=True)
    # 旧格式:无 alpha_spent / alpha_ledger
    (run / "evolution-state.json").write_text(json.dumps({
        "started_at": "2026-09-01T00:00:00+08:00",
        "n_generations": 2,
        "generation_best_deltas": [0.001, 0.002],
        "finalized_generations": [0, 1],
        "trials_accepted": 1,
    }), encoding="utf-8")
    eng = _engine(run, max_generations=8)
    assert eng.state["alpha_spent"] == 0.0
    assert eng.state["alpha_ledger"] == []
    # gen-02 finalize 正常工作(gen-01 已 finalize → 分母 7)
    _trial(run, "t-g2", 2, p_better=0.96)
    eng.finalize_generation(2, ["t-g2"], wall_time_hours=0.1)
    st = json.loads((run / "evolution-state.json").read_text())
    # finalize 返回 schema 不含 family_size;从 generation.yaml 校正块读
    gen_yaml = json.loads(json.dumps(__import__("yaml").safe_load(
        (run / "generations" / "gen-02" / "generation.yaml").read_text())))
    corr = gen_yaml["budget_metrics"]["selection_correction"]
    assert corr["family_size"] == 1
    assert st["alpha_spent"] == pytest.approx(0.10 / 7, abs=1e-9)


# ---------------------------------------------------------------------------
# E2 桩:mock runner(不真跑训练)
# ---------------------------------------------------------------------------
def _mock_runner(run_dir: Path, base_timeout_min: float = 90.0,
                 cap_minutes: float = 480.0) -> SimpleNamespace:
    """最小 runner 桩:只暴露 gen_executor 用到的接口。"""
    calls: list[dict] = []

    def run(trial, **kw):
        calls.append({"trial_id": trial.trial_id,
                      "timeout_minutes": kw.get("timeout_minutes")})
        trial.data["status"] = "RUNNING"
        trial.data["status_history"].append({"state": "RUNNING",
                                              "at": "now"})
        trial.flush()
        return trial

    cfg = Config.load(skill_config=False)
    cfg._data.setdefault("evolution", {}).setdefault(
        "hpo", {})["max_timeout_minutes"] = cap_minutes
    return SimpleNamespace(
        run=run, run_dir=run_dir, config=cfg, spec=_spec(),
        calls=calls,
        resolve_timeout_minutes=lambda override=None: (
            float(override) if override is not None
            else base_timeout_min),
    )


def _gen_trial(run_dir: Path, tid: str, *, timeout_multiplier=None) \
        -> TrialRecord:
    d = run_dir / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    er: dict = {}
    if timeout_multiplier is not None:
        er["timeout_multiplier"] = timeout_multiplier
        er["heavy"] = True
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "lightgbm", "generation": 1,
        "status": "CODE_WRITTEN", "search_action": "explore",
        "parent_trial_ids": [], "expected_resources": er,
        "status_history": [],
    }, trial_dir=d)
    rec.flush()
    return rec


# ---------------------------------------------------------------------------
# E2a: per_trial_timeout_sec = base × multiplier(封顶生效按封顶)
# ---------------------------------------------------------------------------
def test_e2a_per_trial_timeout_multiplier(tmp_path: Path):
    run = tmp_path / "run"
    t_plain = _gen_trial(run, "t-plain")
    t_heavy = _gen_trial(run, "t-heavy", timeout_multiplier=4)
    ex = GenerationExecutor(_mock_runner(run, base_timeout_min=90.0))
    sec = ex._per_trial_timeout_sec([t_plain, t_heavy])
    assert sec["t-plain"] == pytest.approx(90 * 60)
    # 4× 放大:heavy 是 plain 的 4 倍
    assert sec["t-heavy"] == pytest.approx(4 * sec["t-plain"])
    assert sec["t-heavy"] <= 480 * 60 + 1e-9

    # 封顶:multiplier 超大 → 封顶 evolution.hpo.max_timeout_minutes
    ex_cap = GenerationExecutor(
        _mock_runner(run, base_timeout_min=90.0, cap_minutes=100.0))
    sec_cap = ex_cap._per_trial_timeout_sec([t_heavy])
    assert sec_cap["t-heavy"] == pytest.approx(100 * 60)


# ---------------------------------------------------------------------------
# E2b: wall 余量不足 → fail-closed,整代不启动
# ---------------------------------------------------------------------------
def test_e2b_insufficient_wall_budget_rejects(tmp_path: Path):
    run = tmp_path / "run"
    t1 = _gen_trial(run, "t-a")
    t2 = _gen_trial(run, "t-b", timeout_multiplier=4)
    runner = _mock_runner(run)
    ex = GenerationExecutor(runner)
    plan = ResourcePlan(n_parallel_trials=2, threads_per_trial=4)
    out = ex.execute_generation(
        [t1, t2], plan=plan, run_dir=run,
        wall_remaining_sec=60.0)  # < 最小 90*60
    assert out["generation_not_started"] is True
    assert out["reason"] == "insufficient_wall_budget"
    assert out["wall_remaining_sec"] == pytest.approx(60.0)
    assert out["min_trial_timeout_sec"] == pytest.approx(90 * 60)
    assert "重试" in out["remedy"]
    # 无 trial 进入 RUNNING:mock 未被调用,状态未变
    assert runner.calls == []
    assert t1.data["status"] == "CODE_WRITTEN"
    assert t2.data["status"] == "CODE_WRITTEN"


# ---------------------------------------------------------------------------
# E2c: wall_remaining_sec=None → 不预检,行为与旧版一致(会启动)
# ---------------------------------------------------------------------------
def test_e2c_none_wall_skips_preflight(tmp_path: Path):
    run = tmp_path / "run"
    t1 = _gen_trial(run, "t-a")
    runner = _mock_runner(run)
    ex = GenerationExecutor(runner)
    plan = ResourcePlan(n_parallel_trials=1, threads_per_trial=4)
    out = ex.execute_generation([t1], plan=plan, run_dir=run)
    # 与旧版一致:启动了 trial,无 generation_not_started 键
    assert "generation_not_started" not in out
    assert len(runner.calls) == 1
    # 每 trial 用自己的超时(90min base,无放大)
    assert runner.calls[0]["timeout_minutes"] == pytest.approx(
        90.0, abs=1.0)
    # deadline 产物落盘
    art = run / "generations" / "gen-01" / "executor-deadline.json"
    assert art.exists()
    payload = json.loads(art.read_text())
    assert payload["per_trial_timeout_sec"]["t-a"] == pytest.approx(
        90 * 60, abs=1.0)
