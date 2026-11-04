"""v2.19.1 回归测试:σ_seed 规模先验 + 收尾整合通道 + shadow 样本键。

来源(docs/MODELEVO_V2_19_0_DEV_20261101.md §10):
- σ_seed/SE 实测比值随 validation 行数单调上升(5 万行中位 0.42、44.7 万行
  1.19),v2.19.0 的「σ_seed ≥ 1×SE」在小数据上过保守、在大数据上又缺 >1 的
  部分;
- 停止时被同代 BH 拦下的正增益候选随停止一起作废(benchmark 72 个被拦候选
  里 6 个在测试面上更好);
- shadow 分数落盘时不带样本键,trial 在评估前失败会留下无法消费的工件。

全部离线合成数据,不读真实 run。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.config import Config, _deep_merge  # noqa: E402
from runtime.confirmation import (resolve_sigma_seed,  # noqa: E402
                                  scaled_sigma_prior)
from runtime.evaluator import Evaluator  # noqa: E402
from runtime.protocol_gates import (V2_17_0_RUN_CONFIG,  # noqa: E402
                                    V2_19_0_RUN_CONFIG, V2_19_1_RUN_CONFIG)
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402


def _spec(metric: str = "auc", model_type: str = "binary_classification",
          max_generations: int = 8) -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v2191", "task_name": "v2191", "model_type": model_type,
        "target": {"name": "y"}, "time": {"column": "dt"},
        "entity": {"id_column": "uid"}, "primary_metric": metric,
        "resource_budget": {"max_generations": max_generations},
        "data": {},
    })


def _cfg(**evaluator) -> Config:
    cfg = Config.load(skill_config=False)
    cfg.raw["evaluator"].update(evaluator)
    return cfg


def _merge_into(cfg: Config, *layers) -> Config:
    merged = cfg.raw
    for lay in layers:
        merged = _deep_merge(merged, lay)
    for sec, vals in merged.items():
        if isinstance(vals, dict) and isinstance(cfg.raw.get(sec), dict):
            cfg.raw[sec].update(vals)
        else:
            cfg.raw[sec] = vals
    return cfg


def _new_run_cfg(**override) -> Config:
    cfg = _merge_into(Config.load(skill_config=False), V2_17_0_RUN_CONFIG,
                      V2_19_0_RUN_CONFIG, V2_19_1_RUN_CONFIG)
    cfg.raw["evaluator"].update(override)
    return cfg


# ---------------------------------------------------------------------------
# 1. σ_seed 规模先验
# ---------------------------------------------------------------------------

def test_scaled_prior_shape():
    se = 0.01
    # 返回值是**比值** r(n_val),调用方再乘该次检验自己的 SE
    assert scaled_sigma_prior(se, 50000) == pytest.approx(0.5)
    # 44.7 万行 → ≈1.2(req-t7 实测 1.19);400 万行 → 钳到上限 1.5
    assert scaled_sigma_prior(se, 447000) == pytest.approx(1.19, rel=0.03)
    assert scaled_sigma_prior(se, 4_000_000) == pytest.approx(1.5)
    # 比 5 万小 → 钳到下界,不再随规模下降到 0
    assert scaled_sigma_prior(se, 5000) == pytest.approx(0.5)
    # 无 SE / 无 n_val 时不给出
    assert scaled_sigma_prior(None, 100000) is None
    assert scaled_sigma_prior(se, None) is None
    assert scaled_sigma_prior(se, 0) is None


def test_measured_sigma_not_lifted_by_floor():
    """新口径:实测比值小于 1×SE 时原样使用(旧口径抬到 1×SE)。"""
    old = resolve_sigma_seed(replicate_sd=0.004, replicate_df=4, se=0.01,
                             floor_ratio=1.0)
    new = resolve_sigma_seed(replicate_sd=0.004, replicate_df=4, se=0.01,
                             floor_ratio=1.0, floor_measured=False)
    assert old["sigma_seed"] == pytest.approx(0.01)
    assert old["source"] == "run_replicates+floor"
    assert new["sigma_seed"] == pytest.approx(0.004)
    assert new["source"] == "run_replicates"


def test_ratio_prior_beats_absolute_prior_and_floor_kept_without_measure():
    """比值先验排在绝对先验之前;没有实测时 floor 仍然兜底。"""
    out = resolve_sigma_seed(prior_ratio=0.8, prior=0.05, se=0.01,
                             floor_ratio=0.0)
    assert out["sigma_seed"] == pytest.approx(0.008)
    assert out["source"] == "scaled_prior"
    # 无实测、无比值先验 → floor
    out2 = resolve_sigma_seed(se=0.01, floor_ratio=1.0, floor_measured=False)
    assert out2["sigma_seed"] == pytest.approx(0.01)
    assert out2["source"] == "floor_se_ratio"


def test_evaluator_applies_scaled_prior_only_in_new_run():
    rng = np.random.default_rng(7)
    n = 6000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y
    child = parent + rng.normal(scale=0.05, size=n)
    # v2.19.0 口径(只有 seed_variance_in_se):无比值先验 → σ=1×SE
    v2190 = _merge_into(Config.load(skill_config=False), V2_17_0_RUN_CONFIG,
                        V2_19_0_RUN_CONFIG)
    v2190.raw["evaluator"]["bootstrap_samples"] = 150
    v2190.raw["evaluator"]["seed_variance_factor"] = 1.0
    a = Evaluator(_spec(), v2190).evaluate(
        y, child, parent_score=parent, seed_sd_context={"n_val": n})
    pb_a = a["delta"]["vs_parent_paired_bootstrap"]
    assert pb_a["sigma_seed_source"] == "floor_se_ratio"
    # v2.19.1 口径:n_val 给出比值先验(6000 行 → r=0.5),factor=2
    v2191 = _new_run_cfg(bootstrap_samples=150)
    b = Evaluator(_spec(), v2191).evaluate(
        y, child, parent_score=parent, seed_sd_context={"n_val": n})
    pb_b = b["delta"]["vs_parent_paired_bootstrap"]
    assert pb_b["sigma_seed_source"] == "scaled_prior"
    assert pb_b["sigma_seed"] == pytest.approx(0.5 * pb_b["se_bootstrap"])
    assert pb_b["seed_variance_factor"] == pytest.approx(2.0)
    # 小数据(n_val 6000 → r=0.5,factor=2):se_eff = √(1+2·0.25)·SE ≈ 1.22·SE,
    # 比 v2.19.0 的 √2·SE **更宽松** —— 这正是"小数据不再过保守"的口径
    assert pb_a["se"] == pytest.approx(pb_a["se_bootstrap"] * np.sqrt(2),
                                       rel=0.02)
    assert pb_b["se"] == pytest.approx(pb_b["se_bootstrap"] * np.sqrt(1.5),
                                       rel=0.02)
    assert pb_b["se"] < pb_a["se"]


def test_evaluator_large_n_is_stricter_than_legacy():
    """n_val 400 万 → r=1.5,factor=2 ⇒ se_eff ≈ √5.5·SE,比旧 √2·SE 更严。"""
    rng = np.random.default_rng(19)
    n = 20000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y
    child = parent + rng.normal(scale=0.05, size=n)
    cfg = _new_run_cfg(bootstrap_samples=150)
    ev = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent,
        seed_sd_context={"n_val": 4_000_000})
    pb = ev["delta"]["vs_parent_paired_bootstrap"]
    assert pb["sigma_seed"] == pytest.approx(1.5 * pb["se_bootstrap"])
    assert pb["se"] == pytest.approx(pb["se_bootstrap"] * np.sqrt(5.5),
                                     rel=0.02)


def test_library_ratio_ignored_in_v2190_run():
    """在途 v2.19.0 run(无 confirm_seed_sd_prior_ratio)即便拿到库内比值也走旧口径。"""
    rng = np.random.default_rng(13)
    n = 6000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y
    child = parent + rng.normal(scale=0.05, size=n)
    v2190 = _merge_into(Config.load(skill_config=False), V2_17_0_RUN_CONFIG,
                        V2_19_0_RUN_CONFIG)
    v2190.raw["evaluator"]["bootstrap_samples"] = 150
    ev = Evaluator(_spec(), v2190).evaluate(
        y, child, parent_score=parent,
        seed_sd_context={"n_val": n, "prior_ratio": 3.0})
    pb = ev["delta"]["vs_parent_paired_bootstrap"]
    assert pb["sigma_seed_source"] == "floor_se_ratio"
    assert pb["sigma_seed"] == pytest.approx(pb["se_bootstrap"])


def test_evaluator_prefers_library_ratio_over_formula():
    rng = np.random.default_rng(11)
    n = 30000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y
    child = parent + rng.normal(scale=0.05, size=n)
    cfg = _new_run_cfg(bootstrap_samples=150)
    ev = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent,
        seed_sd_context={"n_val": n, "prior_ratio": 1.3})
    pb = ev["delta"]["vs_parent_paired_bootstrap"]
    assert pb["sigma_seed_source"] == "scaled_prior"
    assert pb["sigma_seed"] == pytest.approx(1.3 * pb["se_bootstrap"])


# ---------------------------------------------------------------------------
# 2. 配置门
# ---------------------------------------------------------------------------

def test_defaults_keep_legacy_and_v2191_gate():
    old = Config.load(skill_config=False)
    assert old.get("evaluator.seed_variance_floor_measured") is True
    assert old.get("evaluator.closing_consolidation") is False
    assert old.get("evaluator.confirm_seed_sd_prior_ratio") is None
    assert float(old.get("evaluator.seed_variance_factor")) == 1.0
    g = Config.load()   # skill 级 global-config 也不得启用
    assert not g.get("evaluator.closing_consolidation")
    assert g.get("evaluator.confirm_seed_sd_prior_ratio") is None
    new = _new_run_cfg()
    assert new.get("evaluator.seed_variance_floor_measured") is False
    assert new.get("evaluator.closing_consolidation") is True
    assert new.get("evaluator.confirm_seed_sd_prior_ratio") == "scaled"
    assert float(new.get("evaluator.seed_variance_factor")) == 2.0


def test_cmd_init_writes_v2191_keys(tmp_path):
    import subprocess

    run = tmp_path / "r"
    out = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / "modelevo.py"), "init",
         "--dir", str(run), "--task-id", "t"],
        capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    import yaml

    data = yaml.safe_load((run / "modelevo-config.yaml").read_text())
    ev = data["modelevo"]["evaluator"]
    assert ev["confirm_seed_sd_prior_ratio"] == "scaled"
    assert ev["seed_variance_floor_measured"] is False
    assert float(ev["seed_variance_factor"]) == 2.0
    assert ev["closing_consolidation"] is True
    # v2.19.0 的键仍在前一层写出
    assert ev["anchor_mode"] == "incumbent"


def test_confirm_gate_ratio_mode_parsed():
    from runtime.confirmation import confirm_config

    assert confirm_config(_new_run_cfg())["prior_ratio_mode"] == "scaled"
    fixed = Config.load(skill_config=False)
    fixed.raw["evaluator"]["confirm_seed_sd_prior_ratio"] = 0.9
    cc = confirm_config(fixed)
    assert cc["prior_ratio_mode"] == "fixed"
    assert cc["prior_ratio"] == pytest.approx(0.9)
    assert confirm_config(Config.load(skill_config=False))["prior_ratio_mode"] \
        is None


# ---------------------------------------------------------------------------
# 3. 收尾整合通道
# ---------------------------------------------------------------------------

def _run(tmp_path: Path, *, version: str = "2.19.1") -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "run-state.json").write_text(json.dumps(
        {"baseline_id": "t-001", "skill_version": version}), encoding="utf-8")
    (run / "evolution-state.json").write_text(json.dumps({}), encoding="utf-8")
    return run


def _bh_blocked(pre_verdict: str = "accepted") -> dict:
    """代级 BH 未通过的校正记录(收尾整合只收这类候选)。"""
    return {"method": "bh", "reference": "anchor", "family_size": 4,
            "q_used": 0.01, "p_value_one_sided": 0.019, "bh_passed": False,
            "passed": False, "promotion_blocked_by": None,
            "pre_correction_verdict": pre_verdict,
            "pre_correction_confidence": "high",
            "pre_correction_reason": "单试验显著"}


def _trial(run: Path, tid: str, gen: int, *, metric: float,
           verdict: str = "accepted", action: str = "explore",
           p_better: float | None = None, valid: bool = True,
           correction: dict | None = None, **ev_extra) -> TrialRecord:
    d = run / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    pb = {}
    if p_better is not None:
        pb = {"p_better": p_better, "p_positive": p_better, "se": 0.001}
    ev = {"primary_metric": {"name": "auc", "value": metric},
          "verdict": verdict, "integrity": {"valid": valid},
          "stability": {"passed": True},
          "delta": {"vs_parent_paired_bootstrap": pb,
                    "vs_anchor_paired_bootstrap": dict(pb)}, **ev_extra}
    if correction is None and verdict == "promising":
        correction = _bh_blocked()
    if correction:
        ev["selection_correction"] = correction
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "xgboost", "generation": gen,
        "status": "EVALUATED", "search_action": action,
        "parent_trial_ids": [], "data_view": "v1", "evaluation": ev,
        "status_history": [{"state": "EVALUATED",
                            "at": "2026-11-01T00:00:00+08:00"}],
    }, trial_dir=d)
    rec.flush()
    return rec


def _engine(run: Path, cfg: Config):
    from runtime.evolution import EvolutionEngine

    return EvolutionEngine(run, _spec(), cfg)


def test_closing_consolidation_grants_confirm_eligibility_only(tmp_path):
    """通过只授予 confirm 资格:verdict / 代级记录不变,alpha 扣费。"""
    from runtime.trial import confirmation_parent_eligibility

    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8600)                      # 冠军
    _trial(run, "t-002", 5, metric=0.8700, verdict="promising",
           p_better=0.981)                                      # 被拦候选
    eng = _engine(run, _new_run_cfg())
    before = TrialRecord.load(run / "trials" / "t-002")
    assert confirmation_parent_eligibility(before)[0] is False
    out = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert out["status"] == "confirm_eligible"
    assert out["candidate"] == "t-002"
    assert out["p_value_one_sided"] == pytest.approx(0.019)
    assert out["q_used"] == pytest.approx(0.10)
    assert eng.state["alpha_spent"] == pytest.approx(0.10)
    assert "confirm" in out["next_action"]
    rec = TrialRecord.load(run / "trials" / "t-002")
    ev = rec.data["evaluation"]
    assert ev["verdict"] == "promising"                 # 不当冠军
    assert ev["selection_correction"] == _bh_blocked()  # 代级记录不动
    assert ev["closing_consolidation"]["passed"] is True
    assert ev["closing_consolidation"]["grants"] == "confirm_eligibility"
    assert confirmation_parent_eligibility(rec) == (
        True, "closing_consolidation_confirm_only")
    # 一次性:再调返回 already_ran,不再扣费
    again = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert again["skipped"] == "already_ran"
    assert eng.state["alpha_spent"] == pytest.approx(0.10)


def test_closing_consolidation_q_never_exceeds_remaining_alpha(tmp_path):
    """q 严格 ≤ 剩余 alpha(无下限);剩余为 0 时不检验、不扣费。"""
    from runtime.trial import confirmation_parent_eligibility

    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8600)
    _trial(run, "t-002", 5, metric=0.8700, verdict="promising",
           p_better=0.981)                                      # p=0.019
    eng = _engine(run, _new_run_cfg())
    eng.state["alpha_spent"] = 0.09                             # 剩 0.01
    out = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert out["q_used"] == pytest.approx(0.01)
    assert out["status"] == "not_eligible"
    assert eng.state["alpha_spent"] == pytest.approx(0.10)
    rec = TrialRecord.load(run / "trials" / "t-002")
    assert rec.data["evaluation"]["closing_consolidation"]["passed"] is False
    assert confirmation_parent_eligibility(rec)[0] is False

    run2 = _run(tmp_path / "b")
    _trial(run2, "t-001", 0, metric=0.8600)
    _trial(run2, "t-002", 5, metric=0.8700, verdict="promising",
           p_better=0.999)
    eng2 = _engine(run2, _new_run_cfg())
    eng2.state["alpha_spent"] = 0.10                            # 已花完
    out2 = eng2._closing_consolidation(6, eng2.trial_factory.all_trials())
    assert out2["status"] == "no_alpha_remaining"
    assert eng2.state["alpha_spent"] == pytest.approx(0.10)
    assert "closing_consolidation" not in TrialRecord.load(
        run2 / "trials" / "t-002").data["evaluation"]


def test_closing_consolidation_never_revives_gate_failures(tmp_path):
    """rejected(稳定性/切片/时序门)、质量门失败、被护栏拦下、无 BH 记录的都不进。"""
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8600)
    _trial(run, "t-002", 5, metric=0.8900, verdict="rejected",
           p_better=0.999, slice_evaluation={"passed": False})
    _trial(run, "t-003", 5, metric=0.8800, verdict="promising",
           p_better=0.999, temporal_evaluation={"passed": False})
    _trial(run, "t-004", 5, metric=0.8750, verdict="promising",
           p_better=0.999,
           correction=dict(_bh_blocked(), promotion_blocked_by="quality_gate"))
    _trial(run, "t-005", 5, metric=0.8720, verdict="promising",
           p_better=0.999, correction={})
    eng = _engine(run, _new_run_cfg())
    out = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert out["status"] == "no_candidate"
    assert out["n_candidates_eligible"] == 0
    assert TrialRecord.load(run / "trials" / "t-002").data[
        "evaluation"]["verdict"] == "rejected"


def test_closing_consolidation_delta_is_the_candidates(tmp_path):
    """留痕的 Δ 属于选中的候选,不是最后遍历到的 trial。"""
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8600)
    _trial(run, "t-002", 5, metric=0.8700, verdict="promising",
           p_better=0.99)
    _trial(run, "t-003", 5, metric=0.8605, verdict="promising",
           p_better=0.99)
    eng = _engine(run, _new_run_cfg())
    out = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert out["candidate"] == "t-002"
    assert out["delta_vs_champion"] == pytest.approx(0.01)
    assert out["n_candidates_eligible"] == 2
    assert out["n_candidates_reaching_mde"] == 1
    rec = TrialRecord.load(run / "trials" / "t-002").data[
        "evaluation"]["closing_consolidation"]
    assert rec["delta_vs_champion"] == pytest.approx(0.01)


def test_closing_consolidation_requires_mde_and_passes_only_once(tmp_path):
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8600)
    # Δ=0.0005 < MDE 0.001 → 不进检验
    _trial(run, "t-002", 5, metric=0.8605, verdict="promising",
           p_better=0.999)
    eng = _engine(run, _new_run_cfg())
    out = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert out["status"] == "no_candidate"
    assert TrialRecord.load(run / "trials" / "t-002").data[
        "evaluation"]["verdict"] == "promising"


def test_closing_consolidation_blocked_by_p_value_or_parent_guard(tmp_path):
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8600)
    _trial(run, "t-002", 5, metric=0.8700, verdict="promising",
           p_better=0.50)     # p=0.5 ≫ q → 不通过
    eng = _engine(run, _new_run_cfg())
    out = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert out["status"] == "not_eligible"
    assert TrialRecord.load(run / "trials" / "t-002").data[
        "evaluation"]["verdict"] == "promising"
    # vs_parent 显著更差(p_better ≤ 0.80)→ 护栏拦下
    run2 = _run(tmp_path / "g")
    _trial(run2, "t-001", 0, metric=0.8600)
    _trial(run2, "t-002", 5, metric=0.8700, verdict="promising",
           p_better=0.30)
    eng2 = _engine(run2, _new_run_cfg())
    out2 = eng2._closing_consolidation(6, eng2.trial_factory.all_trials())
    assert out2["status"] == "not_eligible"
    assert out2["parent_guard"]["passed"] is False


def test_closing_consolidation_off_for_legacy_runs(tmp_path):
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.86)
    _trial(run, "t-002", 5, metric=0.87, verdict="promising", p_better=0.999)
    eng = _engine(run, _cfg())
    assert eng._closing_consolidation(6, eng.trial_factory.all_trials()) is None
    assert TrialRecord.load(run / "trials" / "t-002").data[
        "evaluation"]["verdict"] == "promising"


def test_closing_consolidation_regression_direction(tmp_path):
    """rmse 越小越好:候选必须比冠军低才够 MDE。"""
    from runtime.evolution import EvolutionEngine

    run = _run(tmp_path)
    for tid, v, verdict, pb in (("t-001", 12.0, "accepted", None),
                                ("t-002", 10.0, "promising", 0.99),
                                ("t-003", 12.5, "promising", 0.99)):
        _trial(run, tid, 5 if tid != "t-001" else 0, metric=v,
               verdict=verdict, p_better=pb)
    eng = EvolutionEngine(run, _spec(metric="rmse", model_type="regression"),
                          _new_run_cfg())
    out = eng._closing_consolidation(6, eng.trial_factory.all_trials())
    assert out["candidate"] == "t-002"


def test_closing_consolidation_written_into_stop_report(tmp_path, monkeypatch):
    """停止时 stop_report 带通道留痕,通过则 remaining 追加 confirm 待办,
    且候选 verdict 不变(交付位解析不受影响)。"""
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.86)
    # p=0.019:同代 BH(q=0.10/8 代=0.0125)拦下,收尾用剩余 0.0875 放行
    _trial(run, "t-002", 1, metric=0.87, verdict="promising", p_better=0.981)
    eng = _engine(run, _new_run_cfg())
    eng.state["n_generations"] = 1
    eng.state["finalized_generations"] = [0]
    eng.state["generation_best_deltas"] = [0.0]
    # 强制停止建议命中(停止判定本身不在本测试范围)
    monkeypatch.setattr(eng.stop, "check", lambda *a, **k: {
        "stop_reason": "forced", "advisory": True, "blocks": [],
        "remaining_high_potential_directions": []})
    rep = eng.finalize_generation(
        1, ["t-002"], baseline_metric=0.80, wall_time_hours=1.0,
        compute_hours=1.0)
    assert rep is not None
    stopped = eng.state.get("stopped") or {}
    cc = stopped["closing_consolidation"]
    assert cc["status"] == "confirm_eligible"
    assert cc["candidate"] == "t-002"
    assert "confirm_or_consolidate:t-002" in stopped[
        "remaining_high_potential_directions"]
    # 代级 + 收尾合计不超总预算
    assert eng.state["alpha_spent"] == pytest.approx(0.10)
    # 冠军与交付位解析不变(stop_report 不过期)
    assert stopped["delivery_resolution"]["search_champion"] == "t-001"
    assert TrialRecord.load(run / "trials" / "t-002").data[
        "evaluation"]["verdict"] == "promising"


# ---------------------------------------------------------------------------
# 4. shadow 分数落盘带样本键
# ---------------------------------------------------------------------------

def test_dump_shadow_score_includes_sample_key(tmp_path, monkeypatch):
    from runtime import templates

    src = templates.TRIAL_TEMPLATE if hasattr(templates, "TRIAL_TEMPLATE") \
        else None
    body = src if src else ""
    assert "__mvo_sample_key" in body, "trial 模板的 dump_shadow_score 未写入样本键"
    # baseline 模板同样
    from runtime import baseline

    assert "__mvo_sample_key" in baseline.BASELINE_TEMPLATE \
        if hasattr(baseline, "BASELINE_TEMPLATE") else True


def test_shadow_sample_key_written_at_dump(tmp_path, monkeypatch):
    """真实执行 dump_shadow_score:落盘文件必须带样本键。"""
    import os

    import pandas as pd

    from runtime import templates

    run = tmp_path / "run"
    tdir = run / "trials" / "t-001"
    (run / "sample-profile" / "splits" / "labels").mkdir(parents=True)
    tdir.mkdir(parents=True)
    pd.DataFrame({"target": [0, 1, 0],
                  "__mvo_sample_key": ["k1", "k2", "k3"]}).to_parquet(
        run / "sample-profile" / "splits" / "labels" / "shadow_test.parquet",
        index=False)
    # 渲染模板里的 dump_shadow_score 段并执行
    src = templates.TRIAL_TEMPLATE
    i = src.index("def dump_shadow_score")
    j = src.index("\n\n\n", i)
    code = (src[i:j].replace("{{", "{").replace("}}", "}")
            .replace("TRIAL_DIR", f"__import__('pathlib').Path({str(tdir)!r})")
            .replace("LABELS_DIR", f"__import__('pathlib').Path("
                                   f"{str(run / 'sample-profile' / 'splits' / 'labels')!r})"))
    monkeypatch.setenv("MODELEVO_TRIAL_EXEC", tdir.name)
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    monkeypatch.setenv("MODELEVO_QUICK_RATIO", "0")
    ns: dict = {}
    exec(compile(code, "dump_shadow_score", "exec"), ns)  # noqa: S102
    ns["dump_shadow_score"](np.array([0.1, 0.9, 0.3]))
    out = pd.read_parquet(tdir / "shadow_test_score.parquet")
    assert list(out.columns) == ["score", "__mvo_sample_key"]
    assert list(out["__mvo_sample_key"]) == ["k1", "k2", "k3"]
    os.environ.pop("MODELEVO_TRIAL_EXEC", None)


# ---------------------------------------------------------------------------
# 5. 经验库 σ_seed/SE 比值
# ---------------------------------------------------------------------------

def test_sigma_seed_ratio_extracted_from_evaluation():
    from runtime.library.legacy_import import _sigma_seed_ratio

    ev = {"delta": {"vs_parent_paired_bootstrap": {
        "sigma_seed": 0.002, "se_bootstrap": 0.001,
        "sigma_seed_estimate": 0.0005,
        "sigma_seed_estimate_source": "run_replicates"}}}
    # 取 floor 之前的实测估计,不取被抬升后的 sigma_seed
    assert _sigma_seed_ratio(ev) == pytest.approx(0.5)
    assert _sigma_seed_ratio({"delta": {}}) is None
    assert _sigma_seed_ratio({"confirmation_validation": {
        "sigma_seed": 0.0, "se_eval": 0.001}}) is None


@pytest.mark.parametrize("src", ["scaled_prior", "config_prior", None])
def test_sigma_seed_ratio_skips_non_measured(src):
    """先验 / floor / 旧记录(无原始估计)不得入库,否则库内中位数自我循环。"""
    from runtime.library.legacy_import import _sigma_seed_ratio

    pb = {"sigma_seed": 0.001, "se_bootstrap": 0.001,
          "sigma_seed_source": "floor_se_ratio"}
    if src is not None:
        pb.update(sigma_seed_estimate=0.0005,
                  sigma_seed_estimate_source=src)
    assert _sigma_seed_ratio(
        {"delta": {"vs_parent_paired_bootstrap": pb}}) is None


def test_evaluator_records_raw_sigma_estimate():
    """落盘带 floor 之前的估计与来源,经验库据此只收实测比值。"""
    from runtime.library.legacy_import import _sigma_seed_ratio

    rng = np.random.default_rng(5)
    n = 6000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y
    child = parent + rng.normal(scale=0.05, size=n)
    cfg = _new_run_cfg(bootstrap_samples=150)
    ev = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent,
        seed_sd_context={"n_val": n, "replicate_sd": 1e-4,
                         "replicate_df": 3})
    pb = ev["delta"]["vs_parent_paired_bootstrap"]
    assert pb["sigma_seed_estimate_source"] == "run_replicates"
    assert pb["sigma_seed_estimate"] == pytest.approx(1e-4)
    assert _sigma_seed_ratio(ev) == pytest.approx(
        1e-4 / pb["se_bootstrap"], rel=1e-4)
    # 公式先验的结果不入库
    ev2 = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent, seed_sd_context={"n_val": n})
    assert _sigma_seed_ratio(ev2) is None


def test_seed_variance_ratio_of_uses_data_base_median():
    from runtime.library.repository import seed_variance_ratio_of

    class _Store:
        def fingerprints(self, at=None):
            return [{"atid": "binary_classification-y-10w_50w-abcdef12",
                     "data_base": "10w_50w-abcdef12"},
                    {"atid": "binary_classification-z-10w_50w-abcdef12",
                     "data_base": "10w_50w-abcdef12"}]

        def trial_outcomes(self, ns, *, atid=None, at=None):
            if atid.endswith("-y-10w_50w-abcdef12"):
                return [{"extra": {"sigma_seed_ratio": 0.4}},
                        {"extra": {"sigma_seed_ratio": 0.6}},
                        {"extra": {}}]
            return [{"extra": {"sigma_seed_ratio": 0.5}}]

    out = seed_variance_ratio_of(_Store(), "ns", "10w_50w-abcdef12")
    assert out["ratio"] == pytest.approx(0.5)
    assert out["n"] == 3
    assert out["source"] == "library"
    # 样本不足 → None(让调用方退回公式先验)
    assert seed_variance_ratio_of(_Store(), "ns", "10w_50w-abcdef12",
                                  min_n=9) is None
    assert seed_variance_ratio_of(_Store(), "ns", None) is None


def test_library_view_seed_variance_ratio_reads_view(tmp_path):
    from runtime import library_view

    run = tmp_path / "run"
    (run / "experience").mkdir(parents=True)
    (run / "experience" / "library-view.json").write_text(json.dumps(
        {"watermark": 3, "seed_variance_ratio": {"ratio": 0.7, "n": 5,
                                                 "source": "library"}}),
        encoding="utf-8")
    r = library_view.seed_variance_ratio(run)
    assert r["ratio"] == pytest.approx(0.7)
    assert r["n"] == 5
    # 无键 → None
    (run / "experience" / "library-view.json").write_text(
        json.dumps({"watermark": 3}), encoding="utf-8")
    assert library_view.seed_variance_ratio(run) is None


def test_closing_eligibility_survives_generation_fdr_rerun(tmp_path):
    """重复 finalize 重算代级 BH 时,收尾整合的 confirm 资格不丢。"""
    from runtime.trial import confirmation_parent_eligibility

    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8600)
    _trial(run, "t-002", 5, metric=0.8700, verdict="promising",
           p_better=0.981)
    eng = _engine(run, _new_run_cfg())
    assert eng._closing_consolidation(
        6, eng.trial_factory.all_trials())["status"] == "confirm_eligible"
    eng._apply_generation_fdr(5, eng.trial_factory.all_trials(),
                              charge_alpha=False)
    rec = TrialRecord.load(run / "trials" / "t-002")
    ev = rec.data["evaluation"]
    assert ev["closing_consolidation"]["passed"] is True
    assert ev["selection_correction"]["method"] != "closing_consolidation"
    assert ev["verdict"] == "promising"
    assert confirmation_parent_eligibility(rec) == (
        True, "closing_consolidation_confirm_only")