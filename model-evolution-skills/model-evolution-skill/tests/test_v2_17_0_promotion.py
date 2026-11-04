"""v2.17.0 回归测试:统计晋级门与停止语义(WP1)。

事故来源:run4-v2.16.0-2.opus5.5.regression.2 / run-loan-amt-t7-reg(原值金额
回归,rmse 越小越好,MDE=10 元)。

- #1 016/018/020/025 单步 Δ vs_parent 只有 1~2.5×MDE,每代 q≈0.0071,每一步
  都过不了代级校正;020/025 相对已确认交付位 015 已好 49~58(z −3.67/−3.98),
  却永远进不了确认链 → 锚定晋级(promotion_reference=anchor)。
- #2 crossover 020 的 vs_parent 只取一个父代:vs 018 z −1.87(未过)/ vs 016
  z −3.10(本可过),裁决由参数顺序决定 → IUT(crossover_parent_test=iut)。
- #2.3 alpha 线性均摊 → 可选 batch-LORD++(alpha_schedule=lord)。
- #17 STOPPED 语义不清 → stop_report.advisory/blocks/reason_detail/
  undelivered_gain。

本文件全部离线合成数据,不读真实 run。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.config import Config  # noqa: E402
from runtime.evolution import (EvolutionEngine, alpha_schedule,  # noqa: E402
                               compute_undelivered_gain,
                               crossover_parent_test, lord_gamma,
                               lord_level, promotion_reference,
                               promotion_threshold_preview,
                               resolve_promotion_anchor)
from runtime.generation import StopCondition  # noqa: E402
from runtime.leaderboard import Leaderboard  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import (TrialRecord,  # noqa: E402
                           confirmation_parent_eligibility)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spec(max_generations: int = 8) -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v2170", "task_name": "v2170",
        "model_type": "regression",
        "target": {"name": "amt"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "rmse",
        "resource_budget": {"max_generations": max_generations},
        "data": {},
    })


def _cfg(**evaluator) -> Config:
    cfg = Config.load(skill_config=False)
    cfg.raw["evaluator"].update(evaluator)
    return cfg


def _pb(p_better: float, se: float = 10.0, delta: float = -20.0) -> dict:
    # rmse 越小越好:p_better = 1 − p_positive
    return {"p_better": p_better, "p_positive": 1.0 - p_better,
            "se": se, "noise": se, "delta_mean": delta,
            "delta_ci": [delta - 2 * se, delta + 2 * se],
            "z_score": delta / se}


def _write_state(run: Path, *, baseline_id: str = "t-base",
                 delivery: str | None = None, anchors: dict | None = None,
                 finalized: list[int] | None = None) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "run-state.json").write_text(json.dumps(
        {"baseline_id": baseline_id, "skill_version": "2.17.0"}),
        encoding="utf-8")
    st = {"started_at": "2026-10-23T00:00:00+08:00", "n_generations": 0,
          "generation_best_deltas": [],
          "finalized_generations": finalized or [],
          "alpha_spent": 0.0, "alpha_ledger": [], "crossover_combos": [],
          "stopped": None}
    if delivery:
        st["delivery_resolution"] = {"delivery_trial": delivery,
                                     "status": "bound"}
    if anchors:
        st["promotion_anchors"] = anchors
    (run / "evolution-state.json").write_text(json.dumps(st),
                                              encoding="utf-8")


def _trial(run: Path, tid: str, gen: int, *, metric: float,
           parents: list[str] | None = None, action: str = "explore",
           verdict: str = "promising", p_parent: float | None = None,
           p_anchor: float | None = None, anchor_id: str | None = None,
           se: float = 10.0, delta_anchor: float | None = None,
           confirmed: bool = False, extra_ev: dict | None = None
           ) -> TrialRecord:
    d = run / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    ev: dict = {"primary_metric": {"name": "rmse", "value": metric},
                "verdict": verdict, "confidence": "medium",
                "reason": "stub", "integrity": {"valid": True},
                "stability": {"passed": True}, "delta": {}}
    if p_parent is not None:
        ev["delta"]["vs_parent"] = -15.0 if p_parent >= 0.5 else 15.0
        ev["delta"]["vs_parent_paired_bootstrap"] = _pb(p_parent, se)
    if p_anchor is not None:
        ev["delta"]["vs_anchor"] = delta_anchor if delta_anchor is not None \
            else -45.0
        ev["delta"]["vs_anchor_paired_bootstrap"] = _pb(p_anchor, se, -45.0)
        ev["delta"]["anchor_id"] = anchor_id
    if confirmed:
        ev["verdict"] = "confirmed"
        ev["fresh_confirmation"] = {"passed": True}
    if extra_ev:
        ev.update(extra_ev)
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "xgboost", "generation": gen,
        "status": "EVALUATED", "search_action": action,
        "parent_trial_ids": parents or [], "data_view": "v1",
        "evaluation": ev,
        "status_history": [{"state": "EVALUATED",
                            "at": "2026-10-23T00:00:00+08:00"}],
    }, trial_dir=d)
    rec.flush()
    (d / "evaluation.json").write_text(json.dumps(ev), encoding="utf-8")
    return rec


def _reload(run: Path, tid: str) -> dict:
    return TrialRecord.load(run / "trials" / tid).data["evaluation"]


def _chain_run(run: Path) -> None:
    """小步链:锚点 A(已确认)→ T1(gen-1,Δ≈1.5σ)→ T2(gen-2,Δ≈1.6σ
    vs T1,但 vs 锚点 ≈3.5σ)。uniform 8 代 ⇒ 每代 q=0.0125。"""
    _write_state(run, delivery="t-A",
                 anchors={"1": {"anchor_id": "t-A"},
                          "2": {"anchor_id": "t-A"}})
    _trial(run, "t-base", 0, metric=12932.0, verdict="accepted",
           extra_ev={"root_baseline_only": True})
    _trial(run, "t-A", 0, metric=12681.0, parents=["t-base"],
           action="confirm", confirmed=True)


# ---------------------------------------------------------------------------
# 1. 配置门:旧 run(配置无键)保持旧口径
# ---------------------------------------------------------------------------

def test_defaults_keep_legacy_protocol():
    cfg = Config.load(skill_config=False)
    assert promotion_reference(cfg) == "parent"
    assert crossover_parent_test(cfg) == "first_parent"
    assert alpha_schedule(cfg) == "uniform"
    # skill 级 global-config 也必须保持旧口径(在途 run 不换尺)
    cfg2 = Config.load()
    assert promotion_reference(cfg2) == "parent"
    assert crossover_parent_test(cfg2) == "first_parent"
    assert alpha_schedule(cfg2) == "uniform"
    # 新 run 的覆写口径
    from runtime.protocol_gates import V2_17_0_RUN_CONFIG
    ev = V2_17_0_RUN_CONFIG["evaluator"]
    assert promotion_reference(Config({"evaluator": ev})) == "anchor"
    assert crossover_parent_test(Config({"evaluator": ev})) == "iut"


def test_legacy_run_verdict_unchanged_even_with_vs_anchor(tmp_path: Path):
    """旧 run:即便 evaluation 里有 vs_anchor 字段,也按 vs_parent 校正。"""
    run = tmp_path / "run"
    _chain_run(run)
    _trial(run, "t-T2", 1, metric=12632.0, parents=["t-A"],
           verdict="accepted", p_parent=0.97, p_anchor=0.9995,
           anchor_id="t-A")
    eng = EvolutionEngine(run, _spec(), Config.load(skill_config=False))
    rep = eng._apply_generation_fdr(1, eng.trial_factory.all_trials())
    ev = _reload(run, "t-T2")
    assert rep["reference"] == "parent" and rep["schedule"] == "uniform"
    assert rep["q_used"] == pytest.approx(0.0125)
    sc = ev["selection_correction"]
    assert sc["reference"] == "parent"
    assert sc["p_value_one_sided"] == pytest.approx(0.03)
    assert sc["passed"] is False
    assert ev["verdict"] == "promising"          # accepted → promising(旧)
    assert ev["parent_verdict"] == "accepted"    # 校正前 verdict
    assert eng.state["alpha_ledger"][0]["schedule"] == "uniform"


# ---------------------------------------------------------------------------
# 2. 锚定晋级:小步链能晋级;parent 模式保持旧裁决
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["anchor", "parent"])
def test_small_step_chain(tmp_path: Path, mode: str):
    run = tmp_path / mode
    _chain_run(run)
    # gen-1:T1 的父代就是锚点(vs_anchor = vs_parent,p=0.017 > q)
    _trial(run, "t-T1", 1, metric=12660.0, parents=["t-A"],
           verdict="accepted", p_parent=0.983, p_anchor=0.983,
           anchor_id="t-A")
    cfg = _cfg(promotion_reference=mode)
    eng = EvolutionEngine(run, _spec(), cfg)
    eng.finalize_generation(1, ["t-T1"], wall_time_hours=0.1)
    ev1 = _reload(run, "t-T1")
    assert ev1["verdict"] == "promising"
    assert ev1["selection_correction"]["reference"] == mode

    # gen-2:T2 vs T1 只有 p_better=0.972(单步不显著),vs 锚点 0.9995
    _trial(run, "t-T2", 2, metric=12632.0, parents=["t-T1"],
           verdict="promising", p_parent=0.972, p_anchor=0.9995,
           anchor_id="t-A")
    eng = EvolutionEngine(run, _spec(), cfg)
    out = eng.finalize_generation(2, ["t-T2"], wall_time_hours=0.2)
    ev2 = _reload(run, "t-T2")
    sc = ev2["selection_correction"]
    assert ev2["parent_verdict"] == "promising"
    t2 = TrialRecord.load(run / "trials" / "t-T2")
    if mode == "anchor":
        assert sc["reference"] == "anchor" and sc["anchor_id"] == "t-A"
        assert sc["p_value_one_sided"] == pytest.approx(0.0005)
        assert sc["passed"] is True and sc["parent_guard"]["passed"] is True
        assert ev2["verdict"] == "accepted"
        ok, why = confirmation_parent_eligibility(t2)
        assert ok and why == "selection_correction_passed_vs_anchor"
        # 冠军随之更换;未确认 ⇒ 交付位分叉,下一代锚点仍是已确认的 A
        lb = Leaderboard(run, _spec(), cfg)
        assert lb.best_trial(eng.trial_factory.all_trials()).trial_id \
            == "t-T2"
        st = json.loads((run / "evolution-state.json").read_text())
        assert st["delivery_resolution"]["search_champion"] == "t-T2"
        assert st["promotion_anchors"]["3"]["anchor_id"] == "t-A"
        assert resolve_promotion_anchor(run, 3)["anchor_id"] == "t-A"
        assert out["selection_correction"]["passed"] == ["t-T2"]
    else:
        assert sc["reference"] == "parent"
        assert sc["passed"] is False
        assert ev2["verdict"] == "promising"
        assert confirmation_parent_eligibility(t2)[0] is False


def test_anchor_idempotent_refinalize(tmp_path: Path):
    run = tmp_path / "run"
    _chain_run(run)
    _trial(run, "t-T2", 1, metric=12632.0, parents=["t-A"],
           verdict="promising", p_parent=0.972, p_anchor=0.9995,
           anchor_id="t-A")
    cfg = _cfg(promotion_reference="anchor")
    eng = EvolutionEngine(run, _spec(), cfg)
    eng.finalize_generation(1, ["t-T2"], wall_time_hours=0.1)
    spent = eng.state["alpha_spent"]
    eng = EvolutionEngine(run, _spec(), cfg)
    eng.finalize_generation(1, ["t-T2"], wall_time_hours=0.1)
    ev = _reload(run, "t-T2")
    assert ev["verdict"] == "accepted"
    assert ev["selection_correction"]["pre_correction_verdict"] == "promising"
    assert eng.state["alpha_spent"] == pytest.approx(spent)
    assert len(eng.state["alpha_ledger"]) == 1


def test_anchor_parent_guard_blocks_significantly_worse(tmp_path: Path):
    """vs 锚点显著更好,但 vs 直接父代显著更差(p_better ≤ 0.05)→ 不晋级。"""
    run = tmp_path / "run"
    _chain_run(run)
    _trial(run, "t-P", 1, metric=12600.0, parents=["t-A"],
           verdict="promising", p_parent=0.97, p_anchor=0.99,
           anchor_id="t-A")
    _trial(run, "t-C", 2, metric=12640.0, parents=["t-P"],
           verdict="rejected", p_parent=0.02, p_anchor=0.9995,
           anchor_id="t-A")
    cfg = _cfg(promotion_reference="anchor")
    eng = EvolutionEngine(run, _spec(), cfg)
    rep = eng._apply_generation_fdr(2, eng.trial_factory.all_trials())
    ev = _reload(run, "t-C")
    sc = ev["selection_correction"]
    assert sc["bh_passed"] is True
    assert sc["passed"] is False
    assert sc["promotion_blocked_by"] == "parent_guard"
    assert sc["parent_guard"]["vs_parent_p_better"] == pytest.approx(0.02)
    assert ev["verdict"] == "rejected"           # 回到 vs_parent 单试验裁决
    assert ev["parent_verdict"] == "rejected"
    assert rep["blocked"] == {"t-C": "parent_guard"}
    assert confirmation_parent_eligibility(
        TrialRecord.load(run / "trials" / "t-C"))[0] is False


def test_anchor_not_passed_demotes_accepted_and_fallback(tmp_path: Path):
    """anchor 未过 → accepted 降 promising;vs_anchor 缺失 → 按 parent 口径。"""
    run = tmp_path / "run"
    _chain_run(run)
    _trial(run, "t-X", 1, metric=12670.0, parents=["t-A"],
           verdict="accepted", p_parent=0.999, p_anchor=0.96,
           anchor_id="t-A")
    _trial(run, "t-Y", 1, metric=12675.0, parents=["t-A"],
           verdict="accepted", p_parent=0.9999,
           extra_ev={"delta": {
               "vs_parent": -10.0,
               "vs_parent_paired_bootstrap": _pb(0.9999),
               "anchor_id": "t-A", "anchor_incomparable": True,
               "anchor_note": {"status": "incomparable",
                               "reason": "锚点与 trial 不同数据视图"}}})
    cfg = _cfg(promotion_reference="anchor")
    eng = EvolutionEngine(run, _spec(), cfg)
    rep = eng._apply_generation_fdr(1, eng.trial_factory.all_trials())
    assert rep["family_size"] == 2               # 每 trial 仍只算一次检验
    evx, evy = _reload(run, "t-X"), _reload(run, "t-Y")
    assert evx["selection_correction"]["reference"] == "anchor"
    assert evx["verdict"] == "promising"
    assert evy["selection_correction"]["reference"] == "parent"
    assert "不同数据视图" in evy["selection_correction"][
        "reference_fallback_reason"]
    assert evy["verdict"] == "accepted"


# ---------------------------------------------------------------------------
# 3. LORD++ 排程
# ---------------------------------------------------------------------------

def test_lord_gamma_and_budget_without_discoveries():
    H = 14
    gam = [lord_gamma(j, H) for j in range(1, H + 1)]
    assert sum(gam) == pytest.approx(1.0)
    assert gam[0] > gam[1] > gam[2]                # 递减
    assert lord_gamma(H + 1, H) == 0.0 and lord_gamma(0, H) == 0.0
    ledger: list[dict] = []
    qs = []
    for g in range(1, H + 3):
        lv = lord_level(ledger, alpha_total=0.10, q_cap=0.10, horizon=H,
                        upto_generation=g)
        qs.append(lv["q"])
        ledger.append({"generation": g, "q_used": lv["q"], "family_size": 1,
                       "discoveries": 0})
    # 无发现:Σα ≤ W0 = α/2,预算不超限;α_t 单调不增
    assert sum(qs) <= 0.05 + 1e-12
    assert sum(qs) == pytest.approx(0.05)
    assert all(a >= b - 1e-15 for a, b in zip(qs, qs[1:]))
    assert qs[0] == pytest.approx(gam[0] * 0.05)


def test_lord_replenishes_after_discovery_and_wealth_nonnegative():
    H = 14
    base = lord_level([{"generation": 1, "q_used": 0.02, "family_size": 1,
                        "discoveries": 0}], alpha_total=0.1, q_cap=0.1,
                      horizon=H)
    disc = lord_level([{"generation": 1, "q_used": 0.02, "family_size": 1,
                        "discoveries": 1}], alpha_total=0.1, q_cap=0.1,
                      horizon=H)
    assert disc["q"] > base["q"]
    # α_2 = γ2·W0 + (α−W0)·γ1
    assert disc["alpha_t_raw"] == pytest.approx(
        lord_gamma(2, H) * 0.05 + 0.05 * lord_gamma(1, H))
    # 随机发现序列:每步 q ≤ wealth,累计花费 ≤ W0 + 已获回补(预算不超限)
    rng = np.random.default_rng(0)
    ledger: list[dict] = []
    for g in range(1, 30):
        lv = lord_level(ledger, alpha_total=0.1, q_cap=0.1, horizon=H,
                        upto_generation=g)
        assert lv["wealth_before"] >= 0
        assert lv["q"] <= lv["wealth_before"] + 1e-12
        R = sum(e["discoveries"] for e in ledger)
        earned = (0.05 + 0.1 * (R - 1)) if R else 0.0
        spent = sum(e["q_used"] for e in ledger) + lv["q"]
        assert spent <= 0.05 + earned + 1e-9
        ledger.append({"generation": g, "q_used": lv["q"], "family_size": 2,
                       "discoveries": int(rng.random() < 0.3)})


def test_lord_schedule_in_engine_ledger(tmp_path: Path):
    run = tmp_path / "run"
    _chain_run(run)
    cfg = _cfg(alpha_schedule="lord", promotion_reference="anchor")
    # gen-1:一个能过的候选 → 发现;gen-2 的 q 因回补而高于无发现情形
    _trial(run, "t-T1", 1, metric=12600.0, parents=["t-A"],
           verdict="accepted", p_parent=0.9999, p_anchor=0.9999,
           anchor_id="t-A")
    eng = EvolutionEngine(run, _spec(max_generations=14), cfg)
    eng.finalize_generation(1, ["t-T1"], wall_time_hours=0.1)
    e1 = eng.state["alpha_ledger"][0]
    assert e1["schedule"] == "lord" and e1["batch_index"] == 1
    assert e1["discoveries"] == 1
    assert e1["q_used"] == pytest.approx(lord_gamma(1, 14) * 0.05)
    assert _reload(run, "t-T1")["selection_correction"]["method"] \
        == "benjamini_hochberg_with_batch_lord_plus_plus"
    _trial(run, "t-T2", 2, metric=12590.0, parents=["t-T1"],
           verdict="promising", p_parent=0.9, p_anchor=0.97,
           anchor_id="t-A")
    eng = EvolutionEngine(run, _spec(max_generations=14), cfg)
    eng.finalize_generation(2, ["t-T2"], wall_time_hours=0.2)
    e2 = eng.state["alpha_ledger"][1]
    assert e2["batch_index"] == 2
    assert e2["q_used"] == pytest.approx(
        lord_gamma(2, 14) * 0.05 + 0.05 * lord_gamma(1, 14))


# ---------------------------------------------------------------------------
# 4. promotion_threshold_preview
# ---------------------------------------------------------------------------

def test_promotion_threshold_preview(tmp_path: Path):
    from statistics import NormalDist

    run = tmp_path / "run"
    _chain_run(run)
    for i, se in enumerate((10.0, 12.0, 11.0), start=1):
        _trial(run, f"t-{i}", i, metric=12650.0 - i, parents=["t-A"],
               action="exploit", p_parent=0.9, se=se)
    spec = _spec(max_generations=8)
    out = promotion_threshold_preview(run, spec, _cfg())
    assert out["schedule"] == "uniform" and out["reference"] == "parent"
    assert out["next_generation"] == 1
    assert out["q_next"] == pytest.approx(0.10 / 8)
    z1 = NormalDist().inv_cdf(1 - 0.0125)
    assert out["z_by_family"]["1"] == pytest.approx(z1)
    assert out["z_by_family"]["3"] == pytest.approx(
        NormalDist().inv_cdf(1 - 0.0125 / 3))
    assert out["se_reference"] == pytest.approx(11.0)
    assert out["delta_hint"]["1"] == pytest.approx(z1 * 11.0)
    assert out["delta_hint_by_action"]["exploit"] == pytest.approx(z1 * 11.0)
    assert [t["family"] for t in out["thresholds"]] == [1, 2, 3]
    assert out["anchor_id"] is None
    # 纯读取:不写任何文件
    before = sorted(p.name for p in run.rglob("*"))
    out2 = promotion_threshold_preview(
        run, spec, _cfg(promotion_reference="anchor", alpha_schedule="lord"))
    assert sorted(p.name for p in run.rglob("*")) == before
    assert out2["reference"] == "anchor" and out2["schedule"] == "lord"
    assert out2["anchor_id"] == "t-A"
    assert out2["q_next"] == pytest.approx(lord_gamma(1, 8) * 0.05)
    assert out2["lord"]["batch_index"] == 1
    # 无样本 → delta_hint None
    empty = tmp_path / "empty"
    _write_state(empty)
    assert promotion_threshold_preview(empty, spec, _cfg())["delta_hint"] \
        is None


# ---------------------------------------------------------------------------
# 5. stop_report 新字段
# ---------------------------------------------------------------------------

def _stop(spec, cfg, **kw):
    sc = StopCondition(spec, cfg)
    base = dict(n_trials_started=10, n_generations=5, wall_time_hours=1.0,
                trials_accepted=0, best_metric=12681.0,
                baseline_metric=12932.0,
                generation_best_deltas=[-5.0, -3.0, -2.0], noise=10.0,
                unexplored_high_potential=[], data_views_tried=["v1"])
    base.update(kw)
    return sc.check(**base)


def test_stop_report_semantics_fields():
    spec, cfg = _spec(), _cfg()
    ug = {"best_trial": "t-025", "gain": 58.0, "meets_mde": True,
          "reference": "anchor", "anchor_id": "t-015"}
    rep = _stop(spec, cfg, undelivered_gain=ug)
    assert "single_data_view_insufficient" in rep["stop_reason"]
    assert rep["advisory"] is True and rep["blocks"] == []
    detail = rep["reason_detail"]["single_data_view_insufficient"]
    assert "单口径护栏" in detail and "STOPPED" in detail
    assert rep["undelivered_gain"]["best_trial"] == "t-025"
    assert "confirm_or_consolidate:t-025" in \
        rep["remaining_high_potential_directions"]
    # parent 协议:不加条目,给"需 anchor 协议的新 run"提示
    rep2 = _stop(spec, cfg, undelivered_gain={**ug, "reference": "parent"})
    assert rep2["remaining_high_potential_directions"] == []
    assert "anchor 协议" in rep2["undelivered_gain"]["action_hint"]
    # 硬拦截:max_trials 耗尽
    rep3 = _stop(TaskSpec(data={**spec.data, "resource_budget": {
        "max_trials": 10, "max_generations": 8}}), cfg)
    assert "budget_max_trials" in rep3["stop_reason"]
    assert rep3["advisory"] is False
    assert any("new-trial" in b for b in rep3["blocks"])
    assert "BudgetExhausted" in rep3["reason_detail"]["budget_max_trials"]


def test_compute_undelivered_gain_prefers_vs_anchor(tmp_path: Path):
    run = tmp_path / "run"
    _chain_run(run)
    _trial(run, "t-020", 6, metric=12632.0, parents=["t-A"],
           p_parent=0.97, p_anchor=0.9999, anchor_id="t-A",
           delta_anchor=-49.3, se=13.4)
    _trial(run, "t-025", 11, metric=12623.5, parents=["t-020"],
           p_parent=0.9, se=7.0)                  # 无 vs_anchor → 点值差
    _trial(run, "t-bad", 11, metric=12500.0, parents=["t-A"],
           verdict="invalid")
    scr = _trial(run, "t-scr", 11, metric=12400.0, parents=["t-A"])
    scr.data["screen_only"] = True
    scr.flush()
    trials = [TrialRecord.load(p) for p in sorted((run / "trials").iterdir())]
    ug = compute_undelivered_gain(trials, anchor_id="t-A",
                                  anchor_source="delivery_trial",
                                  spec=_spec(), config=_cfg(),
                                  run_dir=run, reference="anchor")
    ids = [r["trial_id"] for r in ug["top_candidates"]]
    assert "t-bad" not in ids and "t-scr" not in ids
    assert ug["best_trial"] == "t-025"
    assert ug["source"] == "metric_difference"
    assert ug["gain"] == pytest.approx(12681.0 - 12623.5)
    t020 = next(r for r in ug["top_candidates"] if r["trial_id"] == "t-020")
    assert t020["source"] == "vs_anchor" and t020["gain"] == pytest.approx(49.3)
    assert t020["z"] == pytest.approx(49.3 / 13.4)
    assert ug["meets_mde"] is True               # MDE 绝对值 0.001


# ---------------------------------------------------------------------------
# 6. runner:IUT / 锚点 / 回归诊断(端到端 _evaluate_trial)
# ---------------------------------------------------------------------------

def _make_reg_run(tmp_path: Path, n: int = 1200, **evaluator):
    from runtime.data import write_splits

    spec = TaskSpec(data={
        "task_id": "v2170r", "task_name": "v2170r",
        "model_type": "regression",
        "target": {"name": "amt"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "rmse",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
    })
    rng = np.random.default_rng(1)
    days = [f"2026-02-{d:02d}" for d in range(1, 11)]
    df = pd.DataFrame({
        "dt": [days[i % len(days)] for i in range(n)],
        "uid": list(range(n)),
        "x": rng.normal(size=n),
        "amt": rng.gamma(2.0, 500.0, size=n),
    })
    run = tmp_path / "run"
    write_splits(df, spec, run / "sample-profile" / "splits")
    cfg = _cfg(bootstrap_samples=300, group_importance_policy="skip_rejected",
               **evaluator)
    cfg.raw["evaluator"]["skip_group_importance_on_generation_0"] = True
    return run, spec, cfg


def _labels(run: Path) -> pd.DataFrame:
    from runtime.data import view_splits_dir
    return pd.read_parquet(view_splits_dir(run, "v1") / "labels"
                           / "validation.parquet")


def _mk_scored_trial(run: Path, tid: str, pred: np.ndarray, *, gen: int,
                     parents: list[str] | None = None,
                     action: str = "explore") -> TrialRecord:
    from runtime.integrity import freeze_trial_source

    tdir = run / "trials" / tid
    (tdir / "code").mkdir(parents=True, exist_ok=True)
    (tdir / "config").mkdir(parents=True, exist_ok=True)
    (tdir / "code" / "train.py").write_text(f"print('{tid}')\n",
                                            encoding="utf-8")
    (tdir / "config" / "trial.yaml").write_text("seed: 1\n", encoding="utf-8")
    pd.DataFrame({"score": pred}).to_parquet(
        tdir / "validation_score.parquet", index=False)
    (tdir / "metrics.json").write_text("{}", encoding="utf-8")
    freeze_trial_source(tdir)
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "xgboost", "generation": gen,
        "search_action": action, "changed_modules": ["feature"],
        "data_view": "v1", "parent_trial_ids": parents or [],
        "status": "SUCCEEDED"}, trial_dir=tdir)
    rec.flush()
    return rec


def _evaluate(run, spec, cfg, trial, parent_scores=None):
    from runtime.runner import TrialRunner

    runner = TrialRunner(run, spec, cfg)
    ev = runner._evaluate_trial(trial, parent_scores=parent_scores,
                                include_shadow=False)
    trial.data["evaluation"] = ev
    trial.flush()
    return ev


def _crossover_setup(tmp_path: Path, **evaluator):
    from runtime.runner import load_parent_scores

    run, spec, cfg = _make_reg_run(tmp_path, **evaluator)
    y = _labels(run)["amt"].to_numpy(dtype=float)
    rng = np.random.default_rng(5)
    e = rng.normal(0, 300.0, size=len(y))
    child = y + e
    # 父代 A 远差于子代;父代 B 与子代几乎同分(最保守的那一个)
    pa = child + rng.normal(0, 250.0, size=len(y))
    pb_ = child + rng.normal(0, 25.0, size=len(y))
    for tid, pred in (("t-PA", pa), ("t-PB", pb_)):
        t = _mk_scored_trial(run, tid, pred, gen=1)
        _evaluate(run, spec, cfg, t)
    return run, spec, cfg, child, load_parent_scores


@pytest.mark.parametrize("order", [["t-PA", "t-PB"], ["t-PB", "t-PA"]])
def test_crossover_iut_picks_most_conservative_parent(tmp_path: Path, order):
    run, spec, cfg, child, lps = _crossover_setup(
        tmp_path, crossover_parent_test="iut")
    tr = _mk_scored_trial(run, "t-X", child, gen=2, parents=order,
                          action="crossover")
    # 调用方按旧口径传第一个父代的分数;IUT 必须改用最保守者
    ev = _evaluate(run, spec, cfg, tr, parent_scores=lps(run, order[0]))
    d = ev["delta"]
    assert d["crossover_parent_test"] == "iut"
    assert d["vs_parent_id"] == "t-PB"
    rows = {r["parent_id"]: r for r in d["vs_parents_iut"]}
    assert set(rows) == {"t-PA", "t-PB"}
    assert rows["t-PB"]["selected"] is True
    assert rows["t-PB"]["p_better"] <= rows["t-PA"]["p_better"]
    for r in rows.values():
        assert {"parent_id", "delta", "se", "p_better"} <= set(r)
    # 正式 vs_parent 即 IUT 选中的父代
    assert d["vs_parent"] == pytest.approx(rows["t-PB"]["delta"])


def test_crossover_first_parent_unchanged(tmp_path: Path):
    run, spec, cfg, child, lps = _crossover_setup(tmp_path)
    tr = _mk_scored_trial(run, "t-X", child, gen=2,
                          parents=["t-PA", "t-PB"], action="crossover")
    ev = _evaluate(run, spec, cfg, tr, parent_scores=lps(run, "t-PA"))
    d = ev["delta"]
    assert "vs_parents_iut" not in d and "crossover_parent_test" not in d
    # 仍用调用方传入的父代(PA):Δ 远大于 vs PB
    y = _labels(run)["amt"].to_numpy(dtype=float)
    pa = lps(run, "t-PA")["validation"]["score"].to_numpy()
    rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))  # noqa: E731
    assert d["vs_parent"] == pytest.approx(rmse(child) - rmse(pa), rel=1e-6)


def test_runner_vs_anchor_and_regression_diagnostics(tmp_path: Path):
    run, spec, cfg = _make_reg_run(tmp_path, promotion_reference="anchor")
    y = _labels(run)["amt"].to_numpy(dtype=float)
    rng = np.random.default_rng(9)
    e = rng.normal(0, 300.0, size=len(y))
    anchor = _mk_scored_trial(run, "t-A", y + e * 1.10, gen=0)
    _evaluate(run, spec, cfg, anchor)
    parent = _mk_scored_trial(run, "t-P", y + e * 1.05, gen=1,
                              parents=["t-A"])
    from runtime.runner import load_parent_scores
    _evaluate(run, spec, cfg, parent,
              parent_scores=load_parent_scores(run, "t-A"))
    _write_state(run, baseline_id="t-A",
                 anchors={"2": {"anchor_id": "t-A",
                                "source": "delivery_trial"}})
    child = _mk_scored_trial(run, "t-C", y + e * 1.00, gen=2,
                             parents=["t-P"])
    ev = _evaluate(run, spec, cfg, child,
                   parent_scores=load_parent_scores(run, "t-P"))
    d = ev["delta"]
    assert d["anchor_id"] == "t-A" and d["anchor_source"] == "delivery_trial"
    rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))  # noqa: E731
    assert d["vs_anchor"] == pytest.approx(rmse(y + e) - rmse(y + 1.1 * e),
                                           rel=1e-6)
    pba = d["vs_anchor_paired_bootstrap"]
    assert pba["p_better"] is not None and pba["se"] > 0
    assert d["vs_parent_id"] == "t-P"
    # vs_anchor 不改写 noise(仍是 vs_parent 的 SE)
    assert ev["noise"] == pytest.approx(d["vs_parent_paired_bootstrap"]["se"])
    assert ev["parent_verdict"] == ev["verdict"]
    diag = ev["regression_diagnostics"]
    assert diag["level"]["n"] == len(y)
    assert "level_by_period" in diag and diag["level_by_period"]
    assert "vs_parent_divergence" in diag
    # 锚点即父代 → 逐字复用 vs_parent
    _write_state(run, baseline_id="t-A",
                 anchors={"2": {"anchor_id": "t-P"}})
    child2 = _mk_scored_trial(run, "t-C2", y + e * 0.99, gen=2,
                              parents=["t-P"])
    ev2 = _evaluate(run, spec, cfg, child2,
                    parent_scores=load_parent_scores(run, "t-P"))
    assert ev2["delta"]["anchor_is_parent"] is True
    assert ev2["delta"]["vs_anchor"] == ev2["delta"]["vs_parent"]


def test_regression_diagnostics_never_affects_verdict(monkeypatch):
    from runtime import regression_diagnostics as rd
    from runtime.evaluator import Evaluator

    spec = _spec()
    cfg = _cfg(bootstrap_samples=200)
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 500.0, size=800)
    parent = y + rng.normal(0, 300, size=800)
    child = y + rng.normal(0, 250, size=800)
    periods = np.array([f"d{i % 4}" for i in range(800)])
    ok = Evaluator(spec, cfg).evaluate(y, child, parent_score=parent,
                                       periods=periods)
    assert "level" in ok["regression_diagnostics"]
    assert len(ok["regression_diagnostics"]["level_by_period"]) == 4

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(rd, "regression_diagnostics", _boom)
    bad = Evaluator(spec, cfg).evaluate(y, child, parent_score=parent,
                                        periods=periods)
    assert bad["regression_diagnostics"] == {"error": "boom"}
    for k in ("verdict", "confidence", "reason", "noise"):
        assert bad[k] == ok[k]


# ---------------------------------------------------------------------------
# 7. confirm 参照:父代经锚定晋级 → 参照 = anchor_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchored", [True, False])
def test_confirm_reference_uses_anchor_id(tmp_path: Path, monkeypatch,
                                          anchored: bool):
    import runtime.runner as runner_mod
    from runtime.runner import TrialRunner

    run = tmp_path / "run"
    _write_state(run)
    sc = ({"reference": "anchor", "passed": True, "anchor_id": "t-A"}
          if anchored else {"reference": "parent", "passed": True})
    _trial(run, "t-P", 3, metric=12632.0, parents=["t-G1", "t-G2"]
           if anchored else ["t-G1"], verdict="accepted",
           extra_ev={"selection_correction": sc})
    child = _trial(run, "t-C", 4, metric=12630.0, parents=["t-P"],
                   action="confirm")
    asked: list[str] = []

    def _fake_lps(run_dir, pid):
        asked.append(pid)
        return {"data_view": "v1"}             # 无 shadow → 跳过并留痕

    monkeypatch.setattr(runner_mod, "load_parent_scores", _fake_lps)
    obj = TrialRunner.__new__(TrialRunner)
    obj.run_dir, obj.spec, obj.config = run, _spec(), _cfg()
    out = obj._confirm_reference_check(
        child, pd.DataFrame(), np.zeros(1), pd.DataFrame(), None,
        sigma_info={}, cc={}, higher_better=False)
    assert out["available"] is False
    assert out["reason"] == "参照无 shadow_test_score"
    if anchored:
        assert asked == ["t-A"]
        assert out["reference_trial"] == "t-A"
        assert out["reference_kind"] == "anchor"
    else:
        assert asked == ["t-G1"]
        assert out["reference_kind"] == "grandparent"


# ---------------------------------------------------------------------------
# 8. MDE 接线:confirmation / StopCondition 走 effective_mde
# ---------------------------------------------------------------------------

def test_mde_wiring_effective(tmp_path: Path):
    from runtime.confirmation import confirm_config, delivery_tolerance

    run = tmp_path / "run"
    _write_state(run, baseline_id="t-base")
    _trial(run, "t-base", 0, metric=12932.0, verdict="accepted")
    old = _cfg()
    new = _cfg(minimum_meaningful_improvement_relative=0.0008)
    # 旧 run(无 relative):与绝对值完全一致
    assert confirm_config(old, run_dir=run)["mde"] == pytest.approx(0.001)
    assert delivery_tolerance(old, None, run_dir=run) == pytest.approx(0.001)
    # 新 run:max(absolute, 0.0008×12932)=10.35
    assert confirm_config(new, run_dir=run)["mde"] == pytest.approx(
        0.0008 * 12932.0)
    assert delivery_tolerance(new, 1.0, run_dir=run) == pytest.approx(
        0.0008 * 12932.0)
    assert confirm_config(new, baseline_metric=1000.0)["mde"] == \
        pytest.approx(0.8)
    assert StopCondition(_spec(), new, run_dir=run)._mde() == pytest.approx(
        0.0008 * 12932.0)
    assert StopCondition(_spec(), old, run_dir=run)._mde() == pytest.approx(
        0.001)


# ---------------------------------------------------------------------------
# 9. finalize 挂钩:函数缺失静默跳过;throttled 改写告警
# ---------------------------------------------------------------------------

def test_finalize_hooks_guarded(tmp_path: Path, monkeypatch):
    import runtime.inventory as inv
    import runtime.scheduler as sch

    run = tmp_path / "run"
    _chain_run(run)
    calls: dict = {}
    monkeypatch.setattr(inv, "backfill_from_trials",
                        lambda rd, trials, generation=None: calls.setdefault(
                            "backfill", generation), raising=False)
    monkeypatch.setattr(sch, "admission_summary",
                        lambda rd, generation=None: {
                            "throttled": True, "waits_sec": 120.0,
                            "reason": "MemAvailable 不足"}, raising=False)
    for i in (1, 2):
        t = _trial(run, f"t-s{i}", 1, metric=12670.0, parents=["t-A"],
                   p_parent=0.6)
        t.data["status_history"] = [
            {"state": "RUNNING", "at": f"2026-10-23T0{i}:00:00+08:00"},
            {"state": "EVALUATED", "at": f"2026-10-23T0{i}:50:00+08:00"}]
        t.flush()
    eng = EvolutionEngine(run, _spec(), _cfg())
    out = eng.finalize_generation(1, ["t-s1", "t-s2"], wall_time_hours=0.1)
    assert calls["backfill"] == 1
    par = out["parallelism"]
    assert par["throttled"] is True
    assert "资源受限串行" in par["warning"]
    from runtime.logging_utils import load_yaml
    gy = load_yaml(run / "generations" / "gen-01" / "generation.yaml")
    assert gy["parallelism"]["admission_summary"]["throttled"] is True
    # 函数不存在 → 静默跳过,旧告警
    monkeypatch.delattr(inv, "backfill_from_trials", raising=False)
    monkeypatch.delattr(sch, "admission_summary", raising=False)
    eng = EvolutionEngine(run, _spec(), _cfg())
    out2 = eng.finalize_generation(1, ["t-s1", "t-s2"], wall_time_hours=0.1)
    assert "反思必答" in out2["parallelism"]["warning"]
