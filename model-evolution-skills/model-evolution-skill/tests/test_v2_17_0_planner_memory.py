"""v2.17.0 WP2:Planner / 方向台账 / planner-context / 经验库 / 信息源清单。

依据 run4 regression.2(原值 rmse 回归,越小越好)的 skill-optimization-plan:

#3  方向台账不区分符号:rmse 任务 +10.7(变差)被当增益,已证伪的 feat:amt2
    连续 4 代成为 UCB 首选 → 台账改方向感知 gain;n_accepted 优先 parent_verdict;
#4  confirm 连续 3 代提名非冠军 trial-003 → 只提名冠军配方 / 容差内陈旧确认,
    附 rationale;物化命令 CLI 路径取调用路径;
MDE planner 全部改走 effective_mde;
ctx planner-context 呈现 MDE(+scale_warning)、晋级门预览、noise_precheck;
#18 warm baseline 只展示同口径;经验 condition 写 target_transform/primary_metric,
    冲突只在同口径内判;检索同口径优先;
#9  export-experience 接受 `experiences:` 包装、逐条 schema 校验;profile 对
    str context 健壮;
#22 information-inventory 按 trial family_hints 自动回填。

全部离线:临时 run 目录;经验库由 tests/conftest.py 指向不可达端口。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SKILL = Path(__file__).resolve().parents[1]
for _p in (str(SKILL), str(SKILL / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime.config import Config  # noqa: E402
from runtime.directions import DirectionLedger  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialFactory, TrialRecord  # noqa: E402

RAW_ATID = "reg-loan_amt_t7-200w_1000w-bc5329dc"
LOG_ATID = "reg-loan_amt_t7_log1p-200w_1000w-bc5329dc"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spec_dict(metric="rmse", mt="regression", target="loan_amt_t7",
               derived=None, reparse=None) -> dict:
    d = {"task_id": "wp2-demo", "task_name": "wp2-demo", "model_type": mt,
         "target": {"name": target}, "time": {"column": "dt"},
         "entity": {"id_column": "uid"}, "primary_metric": metric,
         "business_domains": ["user_operation"],
         "split_policy": {"type": "time_based", "train": "2026-01",
                          "validation": "2026-02", "shadow_test": "2026-03",
                          "final_test": "2026-04"},
         "data": {}, "resource_budget": {"max_trials": 60}}
    if derived:
        d["data"]["derived_columns"] = list(derived)
    if reparse:
        d["features"] = {"reparse": [{"column": c, "plan": "p"}
                                     for c in reparse]}
    return d


def _spec(**kw) -> TaskSpec:
    return TaskSpec(data=_spec_dict(**kw))


def _cfg(**evaluator) -> Config:
    cfg = Config.load(skill_config=False)
    cfg.raw.setdefault("evaluator", {}).update(evaluator)
    return cfg


def _run(tmp_path, spec: TaskSpec | None = None, baseline_id=None) -> Path:
    run = tmp_path / "run"
    (run / "trials").mkdir(parents=True, exist_ok=True)
    if spec is not None:
        (run / "task-spec.yaml").write_text(
            yaml.safe_dump(spec.data, allow_unicode=True), encoding="utf-8")
    if baseline_id:
        (run / "run-state.json").write_text(
            json.dumps({"baseline_id": baseline_id}), encoding="utf-8")
    return run


def _trial(run: Path, tid: str, *, gen=1, action="explore", parents=None,
           value=None, vs_parent=None, verdict="accepted",
           parent_verdict=None, passed=True, hints=None,
           modules=("feature",), runtime=600.0, fresh=None, root=False,
           noise=None, algo="xgboost", status="EVALUATED",
           metric="rmse") -> TrialRecord:
    ev = {"verdict": verdict, "trial_id": tid,
          "primary_metric": {"name": metric, "value": value},
          "delta": {"vs_parent": vs_parent}, "integrity": {"valid": True},
          "selection_correction": {"passed": passed}}
    if parent_verdict:
        ev["parent_verdict"] = parent_verdict
    if fresh is not None:
        ev["fresh_confirmation"] = {"passed": fresh}
    if root:
        ev["root_baseline_only"] = True
    if noise is not None:
        ev["noise"] = noise
    data = {"trial_id": tid, "algorithm": algo, "generation": gen,
            "status": status, "search_action": action,
            "parent_trial_ids": list(parents or []),
            "changed_modules": list(modules), "family_hints": hints,
            "hypothesis": {"statement": "测试用最小假设陈述"},
            "evaluation": ev, "result": {"runtime_sec": runtime}}
    rec = TrialRecord(data=data, trial_dir=run / "trials" / tid)
    rec.flush()
    (run / "trials" / tid / "evaluation.json").write_text(
        json.dumps(ev), encoding="utf-8")
    return rec


# ===========================================================================
# 1. #3 方向台账:方向感知 gain
# ===========================================================================

def _ledger_for(tmp_path, spec, amt2_delta, sub):
    run = tmp_path / sub
    (run / "trials").mkdir(parents=True)
    trials = [
        _trial(run, "t-seqd", hints=["seqd_"], vs_parent=-50.0,
               verdict="accepted", runtime=1800.0),
        _trial(run, "t-amt2", hints=["amt2_"], vs_parent=amt2_delta,
               verdict="rejected", runtime=1800.0),
    ]
    led = DirectionLedger(run)
    state = led.refresh(trials, spec)
    return led, state


def test_rmse_worsening_trial_does_not_raise_ucb(tmp_path):
    spec = _spec(metric="rmse")
    led, st = _ledger_for(tmp_path, spec, +10.7, "worse")
    amt2, seqd = st["arms"]["feat:amt2"], st["arms"]["feat:seqd"]
    assert st["gain_convention"].startswith("direction_aware")
    assert st["metric_higher_is_better"] is False
    # 变差 +10.7 → gain −10.7;有效 −50 → gain +50
    assert amt2["mean_gain"] == pytest.approx(-10.7)
    assert amt2["mean_delta"] == pytest.approx(-10.7), "mean_delta=方向感知增益"
    assert amt2["mean_raw_delta"] == pytest.approx(10.7)
    assert amt2["delta_per_hour"] == 0.0
    assert seqd["mean_gain"] == pytest.approx(50.0) and \
        seqd["delta_per_hour"] > 0
    scores = led.ucb_scores()
    assert scores["feat:seqd"] > scores["feat:amt2"]
    assert next(iter(led.ucb_allocation(1))) == "feat:seqd"
    # 变差的 trial 不得比"零增益"更抬高该方向的 UCB
    led0, _ = _ledger_for(tmp_path, spec, 0.0, "zero")
    assert scores["feat:amt2"] <= led0.ucb_scores()["feat:amt2"] + 1e-12
    # 对照:越大越好的指标上同一 +10.7 才是增益
    ledup, stup = _ledger_for(tmp_path, _spec(metric="auc",
                                              mt="binary_classification"),
                              +10.7, "auc")
    assert stup["arms"]["feat:amt2"]["delta_per_hour"] > 0


def test_auc_ledger_values_unchanged(tmp_path):
    run = _run(tmp_path)
    t = _trial(run, "t1", hints=["seqd_"], vs_parent=0.002, metric="auc")
    st = DirectionLedger(run).refresh([t], _spec(metric="auc",
                                                mt="binary_classification"))
    arm = st["arms"]["feat:seqd"]
    assert arm["mean_delta"] == arm["mean_gain"] == arm["mean_raw_delta"] \
        == pytest.approx(0.002)


def test_ledger_n_accepted_prefers_parent_verdict(tmp_path):
    run = _run(tmp_path)
    a = _trial(run, "t1", hints=["x_"], vs_parent=-5.0, verdict="accepted",
               parent_verdict="rejected")
    b = _trial(run, "t2", hints=["y_"], vs_parent=-5.0, verdict="promising",
               parent_verdict="accepted")
    c = _trial(run, "t3", hints=["z_"], vs_parent=-5.0, verdict="accepted")
    st = DirectionLedger(run).refresh([a, b, c], _spec())
    assert st["arms"]["feat:x"]["n_accepted"] == 0
    assert st["arms"]["feat:y"]["n_accepted"] == 1
    assert st["arms"]["feat:z"]["n_accepted"] == 1, "无 parent_verdict 回落 verdict"


def test_planner_falsified_families_use_gain(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    trials = [
        _trial(run, "t1", hints=["good_"], vs_parent=-30.0, verdict="promising"),
        _trial(run, "t2", hints=["good_"], vs_parent=-20.0, verdict="promising"),
        _trial(run, "t3", hints=["bad_"], vs_parent=5.0, verdict="rejected"),
        _trial(run, "t4", hints=["bad_"], vs_parent=8.0, verdict="rejected"),
    ]
    DirectionLedger(run).refresh(trials, spec)
    fal = AutonomousPlanner(run, spec, _cfg())._falsified_families()
    assert "feat:bad" in fal and "feat:good" not in fal


def test_planner_rebuilds_legacy_ledger_in_memory_for_rmse(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t-amt2", hints=["amt2_"], vs_parent=10.7, verdict="rejected",
           runtime=3340.0)
    legacy = {"updated_at": "x", "arms": {"feat:amt2": {
        "n_tried": 1, "n_accepted": 0, "cost_sec": 3340.0,
        "last_generation": 1, "mean_delta": 10.67, "std_delta": 0.0,
        "lcb_delta": 10.67, "n_eff": 1.0, "cost_hours": 0.928,
        "delta_per_hour": 11.5, "accept_rate": 0.0}}}
    (run / "directions.json").write_text(json.dumps(legacy))
    pl = AutonomousPlanner(run, spec, _cfg())
    state = pl._direction_ledger().state
    assert state.get("rebuilt_in_memory_for_gain_convention") is True
    assert state["arms"]["feat:amt2"]["delta_per_hour"] == 0.0
    assert state["arms"]["feat:amt2"]["mean_gain"] == pytest.approx(-10.7)
    cands = pl._direction_candidates(None, 2)
    assert all(c["acquisition"]["expected_gain"] <= 0 for c in cands)
    # 盘上旧台账不被 planner 改写(下一次 finalize 正式重写)
    assert json.loads((run / "directions.json").read_text()) == legacy


# ===========================================================================
# 2. #4 confirm 只提名冠军配方 / 容差内陈旧确认
# ===========================================================================

def _confirm_run(tmp_path, *, champion_confirmed: bool,
                 champion_passed: bool = True, with_close: bool = False):
    spec = _spec()
    run = _run(tmp_path, spec, baseline_id="t001")
    _trial(run, "t001", gen=0, value=12932.0, root=True, modules=())
    _trial(run, "t002", value=12847.0, vs_parent=-85.0, parents=["t001"])
    _trial(run, "t003", value=12868.0, vs_parent=-64.0, parents=["t001"])
    _trial(run, "t013", gen=3, value=12683.0, vs_parent=-134.8,
           parents=["t002"], noise=10.0, passed=champion_passed,
           modules=("model",))
    if with_close:
        _trial(run, "t016", gen=4, value=12695.0, vs_parent=-12.0,
               parents=["t002"])
    if champion_confirmed:
        _trial(run, "t015", gen=4, action="confirm", parents=["t013"],
               value=12681.5, verdict="confirmed", fresh=True, modules=())
    return run, spec


def test_confirm_not_nominated_for_non_champion_when_champion_bound(tmp_path):
    from runtime.planner import AutonomousPlanner

    run, spec = _confirm_run(tmp_path, champion_confirmed=True)
    pl = AutonomousPlanner(run, spec, _cfg())
    assert pl._confirmation_candidates() == []
    skipped = {s["trial_id"]: s["reason"] for s in pl._confirm_skipped}
    assert skipped.get("t003") == "champion_recipe_already_bound"
    batch = pl.plan(slots=4)
    assert not [p for p in batch["proposals"] if p["action"] == "confirm"]
    assert any(s["trial_id"] == "t003"
               for s in batch["confirm_nomination_skipped"])


def test_confirm_nominates_champion_recipe_with_rationale(tmp_path):
    from runtime.planner import AutonomousPlanner

    run, spec = _confirm_run(tmp_path, champion_confirmed=False)
    pl = AutonomousPlanner(run, spec, _cfg())
    cands = pl._confirmation_candidates()
    assert [c["parent_ids"] for c in cands] == [["t013"]]
    c = cands[0]
    assert c["confirm_nomination"]["role"] == "search_champion_recipe"
    assert c["confirm_nomination"]["search_champion"] == "t013"
    assert "t013" in c["rationale"] and "冠军" in c["rationale"]
    skipped = {s["trial_id"]: s for s in pl._confirm_skipped}
    assert skipped["t003"]["reason"] == "gap_exceeds_delivery_tolerance"
    assert skipped["t003"]["gap"] == pytest.approx(185.0)


def test_confirm_stale_only_within_delivery_tolerance(tmp_path):
    from runtime.planner import AutonomousPlanner

    # 冠军未过选择校正(不可确认)→ 只有落后 ≤ 交付容差(2×noise=20)的候选
    run, spec = _confirm_run(tmp_path, champion_confirmed=False,
                             champion_passed=False, with_close=True)
    pl = AutonomousPlanner(run, spec, _cfg())
    cands = pl._confirmation_candidates()
    assert [c["parent_ids"] for c in cands] == [["t016"]]
    nom = cands[0]["confirm_nomination"]
    assert nom["role"] == "stale_within_tolerance"
    assert nom["gap_to_champion"] == pytest.approx(12.0)
    assert nom["gap_to_champion"] <= nom["delivery_tolerance"]
    assert "容差" in cands[0]["rationale"]


def test_materialize_command_keeps_invocation_path(tmp_path, monkeypatch):
    from runtime import planner as planner_mod

    link = tmp_path / "skill-link"
    os.symlink(SKILL, link)
    cli = link / "scripts" / "modelevo.py"
    monkeypatch.setattr(sys, "argv", [str(cli), "plan"])
    assert planner_mod._cli_script_path() == cli, "软链写法不得被 resolve"
    run, spec = _confirm_run(tmp_path, champion_confirmed=False)
    pl = planner_mod.AutonomousPlanner(run, spec, _cfg())
    cand = pl._confirmation_candidates()[0]
    assert str(cli) in cand["materialize_command"]
    # 非 CLI 调用回落旧口径(绝对路径)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    assert planner_mod._cli_script_path().is_absolute()


# ===========================================================================
# 3. MDE:planner 全部走 effective_mde;HPO 递减门方向感知
# ===========================================================================

def test_planner_uses_effective_mde(tmp_path):
    import inspect

    from runtime.planner import AutonomousPlanner

    spec = _spec(reparse=["seq_d365"])
    run = _run(tmp_path, spec, baseline_id="t001")
    _trial(run, "t001", gen=0, value=12932.47, root=True, modules=())
    (run / "sample-profile").mkdir()
    (run / "sample-profile" / "information-inventory.yaml").write_text(
        yaml.safe_dump({"sources": [{
            "source_type": "reparse", "column": "seq_d365",
            "consumption_status": "unconsumed",
            "granularity_space": [{"dimension": "window", "hint": "h",
                                   "state": "unexplored"}]}]}),
        encoding="utf-8")
    cfg = _cfg(minimum_meaningful_improvement=0.001,
               minimum_meaningful_improvement_relative=0.0008)
    pl = AutonomousPlanner(run, spec, cfg)
    mde = 0.0008 * 12932.47
    assert pl._mde() == pytest.approx(mde)
    parent = pl._parent()
    inv = pl._inventory_candidates(parent)
    assert inv and inv[0]["acquisition"]["potential_gain_upper_bound"] == \
        pytest.approx(2 * mde)
    hpo = [p for p in pl._model_candidates(parent)
           if p["family"] == "model:hpo"][0]
    assert hpo["acquisition"]["expected_gain"] == pytest.approx(0.75 * mde)
    assert hpo["acquisition"]["potential_gain_upper_bound"] == \
        pytest.approx(mde)
    # 源码层面:只剩 _mde 的回落分支读旧键
    src = inspect.getsource(AutonomousPlanner)
    body = src.replace(inspect.getsource(AutonomousPlanner._mde), "")
    assert "evaluator.minimum_meaningful_improvement\"" not in body


def test_hpo_state_diminishing_is_direction_aware(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t001", gen=0, value=12932.0, root=True, modules=())
    good = _trial(run, "t011", gen=2, action="hpo", parents=["t001"],
                  value=12880.0, vs_parent=-51.8, modules=("model",))
    bad = _trial(run, "t019", gen=3, action="hpo", parents=["t001"],
                 value=12940.0, vs_parent=8.4, modules=("model",),
                 verdict="rejected")
    for t in (good, bad):
        (Path(t.trial_dir) / "metrics.json").write_text(
            json.dumps({"n_features": 10}))
    pl = AutonomousPlanner(run, spec, _cfg(minimum_meaningful_improvement=10.0))
    st = pl._champion_face_hpo_state(pl.by_id["t011"])
    assert st["last_gain"] == pytest.approx(51.8)
    assert st["diminishing"] is False, "rmse Δ=−51.8 是改善,不是递减"
    st2 = pl._champion_face_hpo_state(pl.by_id["t019"])
    assert st2["last_gain"] == pytest.approx(-8.4) and st2["diminishing"]


# ===========================================================================
# 4. planner-context:MDE / 晋级门预览 / noise_precheck / warm baseline
# ===========================================================================

def _ctx_run(tmp_path, spec=None):
    spec = spec or _spec()
    run = _run(tmp_path, spec, baseline_id="t001")
    _trial(run, "t001", gen=0, value=12932.47, root=True, modules=())
    return run, spec


def test_context_shows_mde_and_scale_warning(tmp_path, monkeypatch):
    from runtime import evolution as evo
    from runtime.planner_context import (build_planner_context,
                                         render_planner_context)

    monkeypatch.delattr(evo, "promotion_threshold_preview", raising=False)
    run, spec = _ctx_run(tmp_path)
    ctx = build_planner_context(run, spec, _cfg())
    assert ctx["mde"]["ready"] and ctx["mde"]["value"] == pytest.approx(0.001)
    assert ctx["mde"]["scale_warning"]
    md = render_planner_context(ctx)
    assert md.index("MDE 量纲告警") < md.index("## 1. TaskSpec")
    assert "**MDE = 0.001**" in md and "absolute" in md
    # 晋级门预览接口缺失 → 降级,不崩溃
    assert ctx["promotion_preview"]["ready"] is False
    assert "1b. 晋级门预览" in md and "不可用" in md
    assert ctx["target_scale"]["target_transform"] == "raw"


def test_context_renders_promotion_preview(tmp_path, monkeypatch):
    from runtime import evolution as evo
    from runtime.planner_context import (build_planner_context,
                                         render_planner_context)

    seen = {}

    def _preview(run_dir, spec, config):
        seen["called"] = True
        # WP1 promotion_threshold_preview 的返回结构
        return {"next_generation": 8, "q_next": 0.0071, "schedule": "uniform",
                "reference": "anchor", "anchor_id": "t015",
                "thresholds": [
                    {"family": 1, "p_one_sided": 0.0071, "z": 2.45,
                     "delta_hint": 24.5, "below_bootstrap_resolution": False},
                    {"family": 2, "p_one_sided": 0.00355, "z": 2.69,
                     "delta_hint": 26.9, "below_bootstrap_resolution": False}],
                "z_by_family": {"1": 2.45, "2": 2.69},
                "delta_hint": {"1": 24.5, "2": 26.9},
                "se_reference": 10.0, "se_n": 4, "se_source": "vs_anchor",
                "delta_hint_by_action": {"explore": 24.5},
                "note": "family=m 时本代第 1 小的单侧 p 需 ≤ q_next/m"}

    monkeypatch.setattr(evo, "promotion_threshold_preview", _preview,
                        raising=False)
    run, spec = _ctx_run(tmp_path)
    ctx = build_planner_context(
        run, spec, _cfg(minimum_meaningful_improvement=10.0))
    assert seen.get("called") and ctx["promotion_preview"]["ready"]
    assert not ctx["mde"]["scale_warning"]
    md = render_planner_context(ctx)
    assert "本代 q=0.0071" in md and "对照=anchor" in md and "锚点=t015" in md
    assert "需 z ≥ 2.45" in md and "Δ 估计 ≈ 24.5" in md
    assert "单侧 p ≤ 0.0071" in md and "SE 中位数 10" in md
    assert "{'1'" not in md, "字典型 delta_hint 不得原样塞进标量行"
    assert "MDE 量纲告警" not in md

    # 兼容 families 字典写法
    from runtime.planner_context import _render_promotion_preview
    alt = _render_promotion_preview(
        {"q_next": 0.01, "families": {"1": {"z": 2.33}}})
    assert any("需 z ≥ 2.33" in x for x in alt)

    def _boom(*a, **k):
        raise RuntimeError("wp1 not ready")

    monkeypatch.setattr(evo, "promotion_threshold_preview", _boom)
    ctx2 = build_planner_context(run, spec, _cfg())
    assert ctx2["promotion_preview"]["ready"] is False


def test_context_with_real_promotion_preview(tmp_path):
    """WP1 的真实接口在场时端到端可渲染(接口缺失则跳过)。"""
    from runtime import evolution as evo
    from runtime.planner_context import (build_planner_context,
                                         render_planner_context)

    if not hasattr(evo, "promotion_threshold_preview"):
        pytest.skip("WP1 promotion_threshold_preview 未就绪")
    run, spec = _ctx_run(tmp_path)
    ctx = build_planner_context(run, spec, _cfg())
    md = render_planner_context(ctx)
    assert "1b. 晋级门预览" in md
    if ctx["promotion_preview"].get("ready"):
        assert "family=1" in md


def test_context_renders_noise_precheck(tmp_path):
    from runtime.planner_context import (build_planner_context,
                                         render_planner_context)

    run, spec = _ctx_run(tmp_path)
    d = run / "sample-profile" / "splits-sample"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"noise_precheck": {
        "recommend_v1": True, "se_ratio": 4.2,
        "reason": "SE_sample 89 > 2×MDE"}}))
    ctx = build_planner_context(run, spec, _cfg())
    assert ctx["sample_noise_precheck"]["recommend_v1"] is True
    md = render_planner_context(ctx)
    assert "6c. 抽样面噪声预检" in md and "se_ratio=4.2" in md
    # 不存在则跳过
    (d / "manifest.json").write_text(json.dumps({}))
    md2 = render_planner_context(build_planner_context(run, spec, _cfg()))
    assert "6c." not in md2


def _tt_card(atid, rank, trial, metric_value, *, target, transform=None):
    cond = {"task_id": atid, "abstract_task_id": atid,
            "model_type": "regression", "algorithm": "xgboost"}
    if transform:
        cond.update(target_name=target, target_transform=transform,
                    primary_metric="rmse")
    return {"experience_id": f"top-trial-{atid}-rank{rank}",
            "experience_type": "top_trial_baseline", "rank": rank,
            "statement": f"rank-{rank}", "category": "top_trial_baseline",
            "outcome": "positive", "condition": cond,
            "context": {"target_name": target},
            "evidence": {"trial_ids": [trial], "abstract_task_ids": [atid],
                         "primary_metric": metric_value,
                         "primary_metric_name": "rmse",
                         "delta_vs_baseline": -1.0},
            "confidence": "high"}


# v2.18.0:test_warm_baseline_only_shows_comparable 已移除 —— 测的是旧 memory/ 全局库的接线(由经验库视图取代),新行为见 tests/test_v2_18_integration.py

def test_infer_target_transform():
    from runtime.memory import infer_target_transform

    assert infer_target_transform(_spec_dict()) == "raw"
    assert infer_target_transform(
        _spec_dict(target="loan_amt_t7_log1p")) == "log1p"
    assert infer_target_transform(
        _spec_dict(target="amt_l1p", derived=["amt_l1p"])) == "derived"
    d = _spec_dict()
    d["target"]["transform"] = "Log1p"
    assert infer_target_transform(d) == "log1p", "显式声明优先"
    assert infer_target_transform(TaskSpec(data=_spec_dict(
        target="y_log1p"))) == "log1p"


SEQG_OLD = ("seq_gap_d365 的段内正间隔统计（seqg_）在 seq_d365 族之上仍有"
            "独立增益；同源二分类 run 的结论在本回归目标上不成立。")
SEQG_NEW = "seq_gap_d365 相对 seq_d365 无独立增益(条件增量未确立)"


# ===========================================================================
# 6. #9 export-experience schema + profile 健壮性
# ===========================================================================

WRAPPED = """\
experiences:
  - claim: 两段式 hurdle 在原值金额上显著优于单模型
    category: architecture
    outcome: positive
    action_type: add
    context: 公共上下文
    evidence: "gen-03 013 vs 008 −134.8"
  - claim: tweedie 只改善 mae
    outcome: pos
    confidence: maybe
  - 这一条是裸字符串
"""


def test_load_experience_export_structures(tmp_path):
    from runtime.memory import load_experience_export

    p = tmp_path / "x.yaml"
    p.write_text("foo: 1\nbar: 2\n", encoding="utf-8")
    r = load_experience_export(p)
    assert r["structure"] == "invalid" and r["errors"][0]["fix"]
    p.write_text("claim: 单条\noutcome: neutral\n", encoding="utf-8")
    r = load_experience_export(p)
    assert r["structure"] == "single_mapping" and len(r["items"]) == 1
    p.write_text("- claim: a\n  outcome: [\n", encoding="utf-8")
    r = load_experience_export(p)
    assert r["structure"] == "yaml_error" and r["errors"][0]["line"]


def test_cli_export_returns_1_and_diagnose_reports(tmp_path, capsys):
    from _mvo.cmd_inspect import _diag_experience_export, cmd_export_experience

    run = _run(tmp_path, _spec())
    (run / "experience").mkdir(parents=True)
    (run / "experience" / "exported-global-experiences.yaml").write_text(
        WRAPPED, encoding="utf-8")
    rc = cmd_export_experience(SimpleNamespace(run=str(run), spec=None))
    assert rc == 1
    found = _diag_experience_export(run)
    f = next(x for x in found if x["code"] == "experience_export_schema_error")
    assert f["detail"]["structure"] == "wrapped:experiences"
    assert any(e["field"] == "outcome" for e in f["detail"]["errors"])
    assert "export-experience --run" in f["remedy"]
    # 修好后:exit 0、无 finding
    (run / "experience" / "exported-global-experiences.yaml").write_text(
        WRAPPED.split("  - claim: tweedie")[0], encoding="utf-8")
    # v2.18.0:导出 = 提交 run 快照;测试隔离下经验库不可达 → 写 outbox,exit 3
    assert cmd_export_experience(SimpleNamespace(
        run=str(run), spec=None)) == 3
    assert list((run / "experience" / "outbox").glob("snapshot-*.json"))
    assert not [x for x in _diag_experience_export(run)
                if x["code"] == "experience_export_schema_error"]


def test_profile_supports_column_robust_to_context_types():
    from runtime.profile import _experience_supports_column

    assert _experience_supports_column(
        {"claim": "", "context": "seq_d365 解析带来增益"}, "seq_d365")
    assert _experience_supports_column(
        {"claim": "seq 通道加宽", "context": None}, "seq_d365")
    assert not _experience_supports_column(
        {"claim": "无关", "context": ["list"]}, "seq_d365")
    assert not _experience_supports_column("not-a-dict", "seq_d365")
    assert _experience_supports_column(
        {"claim": "", "context": {"objective_description": "seq_d365"}},
        "seq_d365")


# ===========================================================================
# 7. #22 information-inventory 自动回填
# ===========================================================================

def _inventory(run: Path, dims_state: dict | None = None,
               status="unconsumed") -> Path:
    dims = ["window", "recency", "bucketing", "aggregation"]
    gs = []
    for d in dims:
        g = {"dimension": d, "hint": "h", "state": "unexplored"}
        if dims_state and d in dims_state:
            g.update(dims_state[d])
        gs.append(g)
    p = run / "sample-profile" / "information-inventory.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"sources": [{
        "source_type": "reparse", "column": "seq_d365",
        "consumption_status": status, "granularity_space": gs}],
        "summary": {"unconsumed": 1}}, allow_unicode=True), encoding="utf-8")
    return p


def _states(p: Path) -> dict:
    inv = yaml.safe_load(p.read_text(encoding="utf-8"))
    s = inv["sources"][0]
    return {"status": s["consumption_status"],
            **{g["dimension"]: (g["state"], g.get("state_source"))
               for g in s["granularity_space"]}}


def test_backfill_promotes_states_and_is_idempotent(tmp_path):
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path, _spec(reparse=["seq_d365"]))
    p = _inventory(run, {"recency": {"state": "partial",
                                     "state_source": "planner"}})
    trials = [
        _trial(run, "t1", hints=["info:seq_d365:window", "seqd_"],
               verdict="accepted"),
        _trial(run, "t2", hints=["feat:info:seq_d365:aggregation"],
               verdict="rejected"),
        _trial(run, "t3", hints=["info:seq_d365:recency"],
               verdict="promising"),
        _trial(run, "t4", hints=["info:seq_d365:merge"], verdict="accepted"),
        _trial(run, "t5", gen=2, hints=["info:seq_d365:bucketing"],
               verdict="accepted"),
        _trial(run, "t6", hints=["info:seq_d365:bucketing"],
               verdict="invalid"),
    ]
    rep = backfill_from_trials(run, trials, generation=1)
    assert rep["status"] == "updated"
    st = _states(p)
    assert st["window"] == ("implemented", "trial_backfill")
    assert st["aggregation"] == ("explicitly_falsified", "trial_backfill")
    assert st["recency"] == ("partial", "trial_backfill")
    assert st["bucketing"] == ("unexplored", None), "gen-2 / invalid 不计"
    assert st["status"] == "partially_consumed"
    assert rep["source_status_changes"] == [
        {"column": "seq_d365", "from": "unconsumed",
         "to": "partially_consumed"}]
    assert rep["unmatched_hints"][0]["hint"] == "info:seq_d365:merge"
    assert Path(rep["backup"]).exists()
    assert "unconsumed" in Path(rep["backup"]).read_text(encoding="utf-8")
    inv = yaml.safe_load(p.read_text(encoding="utf-8"))
    g = next(g for g in inv["sources"][0]["granularity_space"]
             if g["dimension"] == "window")
    assert g["state_evidence"] == ["t1"] and g["state_verdict"] == "accepted"
    # 幂等:再跑一次无变更、不写盘、不新增备份
    mtime = p.stat().st_mtime_ns
    bak_dir = Path(rep["backup"]).parent
    n_bak = len(list(bak_dir.iterdir()))
    rep2 = backfill_from_trials(run, trials, generation=1)
    assert rep2["status"] == "unchanged" and not rep2["changes"]
    assert p.stat().st_mtime_ns == mtime
    assert len(list(bak_dir.iterdir())) == n_bak
    # 后续代全量:t5(gen-2)补上 bucketing;此前回填的证伪可被 accepted 升级
    t7 = _trial(run, "t7", gen=3, hints=["info:seq_d365:aggregation"],
                verdict="accepted")
    rep3 = backfill_from_trials(run, trials + [t7])
    st3 = _states(p)
    assert st3["bucketing"][0] == "implemented"
    assert st3["aggregation"][0] == "implemented"
    assert st3["recency"][0] == "partial"
    assert st3["status"] == "partially_consumed", "recency 仍为 partial"
    assert rep3["status"] == "updated"
    assert len(list(bak_dir.iterdir())) == n_bak + 1


def test_backfill_never_overrides_higher_agent_state(tmp_path):
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path, _spec(reparse=["seq_d365"]))
    p = _inventory(run, {
        "bucketing": {"state": "explicitly_falsified", "state_source": "agent"},
        "window": {"state": "implemented"},              # 历史文件:无来源=人工
        "recency": {"state": "partial", "state_source": "agent"}},
        status="partially_consumed")
    trials = [
        _trial(run, "t1", hints=["info:seq_d365:bucketing"], verdict="accepted"),
        _trial(run, "t2", hints=["info:seq_d365:window"], verdict="rejected"),
        _trial(run, "t3", hints=["info:seq_d365:recency"], verdict="rejected"),
    ]
    rep = backfill_from_trials(run, trials)
    st = _states(p)
    assert st["bucketing"] == ("explicitly_falsified", "agent")
    assert st["window"] == ("implemented", None)
    assert st["recency"] == ("explicitly_falsified", "trial_backfill"), \
        "人工 partial 可被更强的 trial 证据升级"
    skipped = {s["dimension"] for s in rep["skipped_agent_higher"]}
    assert skipped == {"bucketing", "window"}


def test_backfill_prefers_parent_verdict(tmp_path):
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path, _spec(reparse=["seq_d365"]))
    p = _inventory(run)
    t = _trial(run, "t1", hints=["info:seq_d365:window"], verdict="accepted",
               parent_verdict="rejected")
    backfill_from_trials(run, [t])
    assert _states(p)["window"][0] == "explicitly_falsified"


def test_backfill_no_inventory_and_refresh_keeps_backfill(tmp_path):
    from runtime.inventory import backfill_from_trials, refresh_inventory

    spec = _spec(reparse=["seq_d365"])
    run = _run(tmp_path, spec)
    t = _trial(run, "t1", hints=["info:seq_d365:window"], verdict="accepted")
    assert backfill_from_trials(run, [t])["status"] == "no_inventory"
    refresh_inventory(run, spec)        # 建清单(全部 unexplored)
    p = run / "sample-profile" / "information-inventory.yaml"
    rep = backfill_from_trials(run, TrialFactory(run).all_trials())
    assert rep["status"] == "updated"
    refresh_inventory(run, spec)        # finalize 刷新:回填状态与来源保留
    inv = yaml.safe_load(p.read_text(encoding="utf-8"))
    g = next(g for g in inv["sources"][0]["granularity_space"]
             if g["dimension"] == "window")
    assert g["state"] == "implemented"
    assert g["state_source"] == "trial_backfill"
    assert g["state_evidence"] == ["t1"]
    assert inv["sources"][0]["consumption_status"] == "partially_consumed"
