"""v2.20.0 回归测试:warm-start 确认资格修正 + 先验豁免。

来源(docs/MODELEVO_V2_20_0_CONFIRM_20261102.md):

1. **确认资格分支顺序**:warm trial 没有父代,天然带 ``root_baseline_only``;
   旧口径把 ``root_baseline_only`` 豁免排在 warm 家族检查之前,v2.17.2(C3)
   的预注册 BH 因此形同虚设 —— 只有 ``p_better ∈ [0.95, 阈值)`` 的窄带才会
   真的被判 ``warm_start_not_confirmable``。
2. **先验豁免**:warm-start 的增益若"换 seed 后仍在"
   (``z = Δ_dir / sqrt(SE² + σ_seed²) ≥ z_required``,σ_seed 取**实测**),
   可不做 confirm 直接交付,``status=waived_by_prior``。confirm 门本身是
   等价性检验、通过率与数据无关,不能作为豁免判据。

全部离线合成数据,不读真实 run。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.config import Config  # noqa: E402
from runtime.confirmation import (  # noqa: E402
    measured_sigma_seed, seed_robust_gain)
from runtime.leaderboard import Leaderboard  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import (  # noqa: E402
    TrialRecord, WARM_START_NOT_CONFIRMABLE,
    confirmation_parent_eligibility)


def _spec(metric: str = "auc") -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v220", "task_name": "v220",
        "model_type": "binary_classification",
        "target": {"name": "y"}, "time": {"column": "dt"},
        "entity": {"id_column": "uid"}, "primary_metric": metric,
        "resource_budget": {"max_generations": 8}, "data": {},
    })


def _cfg(**evaluator) -> Config:
    cfg = Config.load(skill_config=False)
    cfg.raw["evaluator"].update(evaluator)
    return cfg


def _run(tmp_path: Path, *, baseline_id: str = "t-001") -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "run-state.json").write_text(
        json.dumps({"baseline_id": baseline_id, "skill_version": "2.20.0"}),
        encoding="utf-8")
    (run / "evolution-state.json").write_text(json.dumps({}), encoding="utf-8")
    return run


def _trial(run: Path, tid: str, *, metric: float, action: str = "explore",
           verdict: str = "accepted", gen: int = 0,
           delta: dict | None = None, root_baseline_only: bool = False,
           correction: dict | None = None,
           parents: list | None = None) -> TrialRecord:
    d = run / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    ev = {"primary_metric": {"name": "auc", "value": metric},
          "verdict": verdict, "integrity": {"valid": True},
          "stability": {"passed": True}, "delta": delta or {}}
    if root_baseline_only:
        ev["root_baseline_only"] = True
    if correction is not None:
        ev["selection_correction"] = correction
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "xgboost", "generation": gen,
        "status": "EVALUATED", "search_action": action,
        "parent_trial_ids": parents or [], "data_view": "v1",
        "evaluation": ev,
        "status_history": [{"state": "EVALUATED",
                            "at": "2026-11-02T00:00:00+08:00"}],
    }, trial_dir=d)
    rec.flush()
    return rec


def _vs_baseline(delta: float, se: float, *, sigma: float | None = None,
                 sigma_source: str | None = None) -> dict:
    pb = {"p_better": 0.99, "p_positive": 0.99, "se": se, "se_bootstrap": se}
    if sigma is not None:
        pb["sigma_seed_estimate"] = sigma
        pb["sigma_seed_estimate_source"] = sigma_source or "run_replicates"
    return {"vs_baseline": delta, "vs_baseline_paired_bootstrap": pb}


# ===========================================================================
# 1. 确认资格:root_baseline_only 不再豁免 warm-start
# ===========================================================================

def test_warm_start_root_baseline_only_no_longer_bypasses_bh(tmp_path):
    """warm trial 无父代 ⇒ 带 root_baseline_only;旧口径在此直接放行,
    新口径必须走 C3 的预注册家族校正(无记录 ⇒ 不可确认)。"""
    run = _run(tmp_path)
    warm = _trial(run, "t-002", metric=0.81, action="warm_start",
                  root_baseline_only=True)
    assert confirmation_parent_eligibility(warm) == (
        False, WARM_START_NOT_CONFIRMABLE)


def test_warm_start_passes_when_pregistered_family_passed(tmp_path):
    """C3 校正通过(p_better ≥ 0.95 且过 BH)⇒ 可确认。"""
    run = _run(tmp_path)
    warm = _trial(run, "t-002", metric=0.81, action="warm_start",
                  root_baseline_only=True,
                  correction={"family": "warm_start", "passed": True,
                              "reference": "baseline", "bh_passed": True})
    assert confirmation_parent_eligibility(warm) == (
        True, "selection_correction_passed_warm_start")


def test_non_warm_root_baseline_only_still_exempt(tmp_path):
    """非 warm 的根 trial 保持原语义:只建立跑线,可直接确认。"""
    run = _run(tmp_path)
    base = _trial(run, "t-001", metric=0.80, root_baseline_only=True)
    assert confirmation_parent_eligibility(base) == (True, "root_baseline")


def test_warm_root_fix_off_for_legacy_runs(tmp_path):
    """在途 run(skill_version < 2.20.0)不换尺:保持旧的 root_baseline 放行。"""
    run = _run(tmp_path)
    (run / "run-state.json").write_text(json.dumps(
        {"baseline_id": "t-001", "skill_version": "2.19.1"}), encoding="utf-8")
    warm = _trial(run, "t-002", metric=0.81, action="warm_start",
                  root_baseline_only=True)
    assert confirmation_parent_eligibility(warm) == (True, "root_baseline")


def test_warm_start_bh_failure_blocks_even_with_high_p(tmp_path):
    """p_better ≥ 0.95 但 BH 未过 ⇒ 明确不可确认(旧口径这条是唯一生效的
    窄带,现在与 p_better < 0.95 一视同仁走同一道门)。"""
    run = _run(tmp_path)
    warm = _trial(run, "t-002", metric=0.81, action="warm_start",
                  root_baseline_only=True,
                  correction={"family": "warm_start", "passed": False,
                              "not_confirmable_reason":
                                  WARM_START_NOT_CONFIRMABLE})
    assert confirmation_parent_eligibility(warm) == (
        False, WARM_START_NOT_CONFIRMABLE)


# ===========================================================================
# 2. 先验豁免判据
# ===========================================================================

def test_seed_robust_gain_passes_on_large_robust_delta(tmp_path):
    run = _run(tmp_path)
    t = _trial(run, "t-002", metric=0.81, action="warm_start",
               delta=_vs_baseline(0.010, 0.001, sigma=0.001))
    out = seed_robust_gain(t)
    assert out["available"] is True
    assert out["sigma_source"] == "trial_eval"
    # z = 0.010 / sqrt(0.001² + 0.001²) ≈ 7.07
    assert out["z"] == pytest.approx(0.010 / (0.001 ** 2 * 2) ** 0.5,
                                     rel=1e-6)
    assert out["passed"] is True


def test_seed_robust_gain_fails_when_gain_is_seed_noise(tmp_path):
    """增益与 σ_seed 同量级 ⇒ 不豁免(这正是有判别力的地方)。"""
    run = _run(tmp_path)
    t = _trial(run, "t-002", metric=0.81, action="warm_start",
               delta=_vs_baseline(0.0015, 0.001, sigma=0.001))
    out = seed_robust_gain(t)
    assert out["available"] is True
    assert out["passed"] is False


def test_seed_robust_gain_requires_measured_sigma(tmp_path):
    """没有实测 σ_seed ⇒ 不评估(不用公式先验自我证明)。"""
    run = _run(tmp_path)
    t = _trial(run, "t-002", metric=0.81, action="warm_start",
               delta=_vs_baseline(0.010, 0.001))
    out = seed_robust_gain(t)
    assert out["available"] is False
    assert out["passed"] is False


def test_measured_sigma_falls_back_to_library_ratio(tmp_path):
    run = _run(tmp_path)
    t = _trial(run, "t-002", metric=0.81, action="warm_start",
               delta=_vs_baseline(0.010, 0.001))
    sig = measured_sigma_seed(t, prior_ratio=1.2)
    assert sig["available"] is True
    assert sig["source"] == "library_ratio"
    assert sig["sigma_seed"] == pytest.approx(0.0012)


def test_seed_robust_gain_regression_direction(tmp_path):
    """越小越好的指标:Δ 为负才是提升。"""
    run = _run(tmp_path)
    t = _trial(run, "t-002", metric=0.10, action="warm_start",
               delta=_vs_baseline(-0.010, 0.001, sigma=0.001))
    assert seed_robust_gain(t, higher_better=False)["passed"] is True
    assert seed_robust_gain(t, higher_better=True)["passed"] is False


# ===========================================================================
# 3. 交付位解析:waived_by_prior
# ===========================================================================

def _delivery_run(tmp_path: Path) -> Path:
    run = _run(tmp_path)
    _trial(run, "t-001", metric=0.800)               # baseline
    _trial(run, "t-002", metric=0.811, action="warm_start",
           root_baseline_only=True,
           delta=_vs_baseline(0.011, 0.001, sigma=0.001))
    return run


def _load_trials(run: Path) -> list:
    return [TrialRecord.load(d) for d in sorted((run / "trials").iterdir())
            if (d / "trial.yaml").exists()]


def test_delivery_resolution_waived_by_prior(tmp_path):
    run = _delivery_run(tmp_path)
    trials = _load_trials(run)
    cfg = _cfg(prior_waiver={"enabled": True, "z_required": 2.0},
               prior_waiver_actions=["warm_start"])
    res = Leaderboard(run, _spec(), cfg).delivery_resolution(trials)
    assert res["status"] == "waived_by_prior"
    assert res["delivery_trial"] == "t-002"
    assert res["delivery_recipe"] == "t-002"
    assert res["prior_waiver"]["criterion"] == "seed_robust_gain_v1"
    assert res["prior_waiver"]["passed"] is True


def test_delivery_resolution_waiver_off_keeps_old_behaviour(tmp_path):
    """配置关闭(在途 run)⇒ 与旧口径一致:未确认 → unconfirmed。"""
    run = _delivery_run(tmp_path)
    trials = _load_trials(run)
    res = Leaderboard(run, _spec(), _cfg()).delivery_resolution(trials)
    assert res["status"] == "unconfirmed"


def test_delivery_resolution_waiver_not_applied_to_explore(tmp_path):
    """动作不在白名单(普通 explore)⇒ 不豁免。"""
    run = _run(tmp_path)
    _trial(run, "t-001", metric=0.800)
    _trial(run, "t-002", metric=0.811, action="explore",
           delta=_vs_baseline(0.011, 0.001, sigma=0.001))
    trials = _load_trials(run)
    cfg = _cfg(prior_waiver={"enabled": True, "z_required": 2.0},
               prior_waiver_actions=["warm_start"])
    res = Leaderboard(run, _spec(), cfg).delivery_resolution(trials)
    assert res["status"] == "unconfirmed"
    assert res["prior_waiver"]["reason"].startswith("动作 'explore'")


def test_delivery_resolution_confirmed_still_wins_over_waiver(tmp_path):
    """已有 fresh confirm 时走 bound,不因豁免改变既有口径。"""
    run = _run(tmp_path)
    _trial(run, "t-001", metric=0.800)
    _trial(run, "t-002", metric=0.811, action="warm_start",
           delta=_vs_baseline(0.011, 0.001, sigma=0.001))
    _trial(run, "t-003", metric=0.8105, action="confirm",
           verdict="confirmed", parents=["t-002"],
           delta=_vs_baseline(0.0, 0.001, sigma=0.001))
    rec = TrialRecord.load(run / "trials" / "t-003")
    rec.data["evaluation"]["fresh_confirmation"] = {
        "passed": True, "criterion": "noninferiority_seed_aware_v1"}
    rec.flush()
    trials = _load_trials(run)
    cfg = _cfg(prior_waiver={"enabled": True, "z_required": 2.0},
               prior_waiver_actions=["warm_start"])
    res = Leaderboard(run, _spec(), cfg).delivery_resolution(trials)
    assert res["status"] == "bound"
    assert res["delivery_trial"] == "t-003"


# ===========================================================================
# 4. 协议门
# ===========================================================================

def test_v2_20_run_config_written():
    from runtime.protocol_gates import V2_20_0, V2_20_0_RUN_CONFIG

    assert V2_20_0 == "2.20.0"
    ev = V2_20_0_RUN_CONFIG["evaluator"]
    assert ev["prior_waiver"]["enabled"] is True
    assert ev["prior_waiver"]["z_required"] == 2.0
    assert ev["prior_waiver_actions"] == ["warm_start"]


def test_prior_waiver_defaults_off():
    """代码 DEFAULTS 保持旧口径:在途 run 不换尺。"""
    assert Config.load(skill_config=False).get("evaluator.prior_waiver") is None
