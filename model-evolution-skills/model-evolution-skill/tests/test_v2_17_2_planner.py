"""v2.17.2 planner / 评估侧修复批次(run4-v2.16.0-2.opus5.5 / req-t7-400w 复盘)。

A1 撞 cap 混杂:trial-017(HPO lr-shrink 重训被 8000 树上限截断,best_iter
   7999/8000、cap_extended_to=null)得伪负 Δ −0.00095 → evaluation 附
   cap_confounded,方向台账不记负奖励,报告提示;
A2 冠军撞 cap → planner 提名 lr×2 的 exploit 修正子代(物化命令
   `--from-parent <冠军> --set learning_rate=<2×>`),排在同面 HPO 之前;
A5 "同面"判定用 model/feature_columns.json 的真实列数(旧代码 n_features
   记 700 基础列,实际 1647);
D1 证伪族真正过滤(feat: 前缀归一 + lcb/ucb 判据 + avoid 经验卡);
   清单不再因"被规划"就把粒度标 partial;
C2 capacity-check 的目标可达性;
C4 shadow gap 诊断(gap_sanity_passed,旧键 passed)与确认门分开呈现。

全部离线:临时 run 目录;经验库由 tests/conftest.py 指向不可达端口。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

from runtime.config import MAX_TREES, Config  # noqa: E402
from runtime.directions import DirectionLedger  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402

ATID = "binary-label-200w_1000w-bc5329dc"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _spec(target=None, reparse=None) -> TaskSpec:
    d = {"task_id": "v2172", "task_name": "v2172",
         "model_type": "binary_classification",
         "target": {"name": "label", "positive_definition": "1"},
         "time": {"column": "dt"}, "entity": {"id_column": "uid"},
         "primary_metric": "auc", "business_domains": ["user_operation"],
         "split_policy": {"type": "time_based", "train": "2026-01",
                          "validation": "2026-02", "shadow_test": "2026-03",
                          "final_test": "2026-04"},
         "data": {}, "resource_budget": {"max_trials": 60}}
    if target is not None:
        d["performance_target"] = {"auc": target}
    if reparse:
        d["features"] = {"reparse": [{"column": c, "plan": "p"}
                                     for c in reparse]}
    return TaskSpec(data=d)


def _cfg(min_generations: int = 0) -> Config:
    cfg = Config.load(skill_config=False)
    cfg.raw.setdefault("feature_exploration", {})["min_generations"] = \
        min_generations
    return cfg


def _run(tmp_path, spec: TaskSpec) -> Path:
    run = tmp_path / "run"
    (run / "trials").mkdir(parents=True, exist_ok=True)
    (run / "task-spec.yaml").write_text(
        yaml.safe_dump(spec.data, allow_unicode=True), encoding="utf-8")
    (run / "run-state.json").write_text(json.dumps({
        "abstract_task_id": ATID,
        "abstract_task_id_source": "raw_columns_hash"}), encoding="utf-8")
    return run


def _trial(run: Path, tid: str, *, gen=1, action="explore", parents=None,
           value=0.80, vs_parent=None, verdict="accepted", hints=None,
           modules=("feature",), runtime=3600.0, algo="xgboost",
           hp=None, es_info=None, n_features=700, fc=None, best_params=None,
           complexity=None, extra_ev=None) -> TrialRecord:
    ev = {"verdict": verdict, "trial_id": tid,
          "primary_metric": {"name": "auc", "value": value},
          "delta": {"vs_parent": vs_parent}, "integrity": {"valid": True},
          "selection_correction": {"passed": True}}
    if verdict != "confirmed" and action != "confirm":
        ev["parent_verdict"] = verdict
    if complexity is not None:
        ev["complexity"] = complexity
    ev.update(extra_ev or {})
    data = {"trial_id": tid, "algorithm": algo, "generation": gen,
            "status": "EVALUATED", "search_action": action,
            "parent_trial_ids": list(parents or []),
            "changed_modules": list(modules), "family_hints": hints,
            "data_view": "v1",
            "hypothesis": {"statement": "测试用最小假设陈述"},
            "evaluation": ev, "result": {"runtime_sec": runtime}}
    tdir = run / "trials" / tid
    rec = TrialRecord(data=data, trial_dir=tdir)
    rec.flush()
    (tdir / "evaluation.json").write_text(json.dumps(ev), encoding="utf-8")
    (tdir / "config").mkdir(parents=True, exist_ok=True)
    (tdir / "config" / "trial.yaml").write_text(yaml.safe_dump({
        "trial_id": tid, "algorithm": algo,
        "hyperparameters": dict(hp or {})}), encoding="utf-8")
    m = {"n_features": n_features}
    if es_info is not None:
        m["es_info"] = es_info
    if best_params is not None:
        m["best_params"] = best_params
    (tdir / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    if fc is not None:
        (tdir / "model").mkdir(parents=True, exist_ok=True)
        (tdir / "model" / "feature_columns.json").write_text(
            json.dumps([f"c{i}" for i in range(fc)] if isinstance(fc, int)
                       else list(fc)), encoding="utf-8")
    return rec


CAP_ES = {"cap": 8000, "cap_initial": 8000, "cap_extended_to": None,
          "cap_bound": True, "stopped_by": "cap",
          "stage1_best_iteration": 7999, "stage1_rounds": 8000}
EARLY_ES = {"cap": 8000, "cap_extended_to": None, "cap_bound": False,
            "stopped_by": "early_stop", "stage1_best_iteration": 6314}
#: trial-017 evaluation.complexity 原样(runner 从 es_info 抄的子集)
CAP_CX = {"feature_count": 700, "best_iteration": 7999,
          "n_estimators_used": 8000, "cap": 8000, "cap_bound": True,
          "es_stopped_by": "cap", "cap_extended_to": None,
          "stage1_rounds": 8000}


# ===========================================================================
# A1 撞 cap 混杂
# ===========================================================================

def test_a1_detail_legacy_trial017_shape():
    from runtime.evaluator import cap_confounded_detail

    ev = {"verdict": "rejected", "parent_verdict": "rejected",
          "delta": {"vs_parent": -0.0009527198278900251},
          "complexity": dict(CAP_CX)}
    d = cap_confounded_detail(ev, higher_is_better=True)
    assert d is not None and d["rule"] == "legacy:complexity"
    assert d["cap"] == 8000 == MAX_TREES and d["best_iteration"] == 7999
    # es_info 旧产物同样命中(优先 es_info)
    d2 = cap_confounded_detail({"verdict": "rejected",
                                "delta": {"vs_parent": -0.001}},
                               es_info=dict(CAP_ES))
    assert d2 and d2["rule"] == "legacy:es_info"


def test_a1_detail_new_flag_and_negatives():
    from runtime.evaluator import cap_confounded_detail

    neg = {"verdict": "rejected", "delta": {"vs_parent": -0.0005}}
    # v2.17.2 新旗标:cap 未到 MAX_TREES 也算(余量已用尽)
    d = cap_confounded_detail(neg, es_info={"cap": 6000, "cap_bound": True,
                                            "cap_bound_unresolved": True})
    assert d and d["rule"] == "es_info.cap_bound_unresolved"
    # 不命中:Δ>0 的 rejected(未过门不是负奖励)/ 早停 / 已外扩 / cap 未到硬顶
    assert cap_confounded_detail(
        {"verdict": "rejected", "delta": {"vs_parent": 2e-5},
         "complexity": dict(CAP_CX)}) is None
    assert cap_confounded_detail(neg, es_info=dict(EARLY_ES)) is None
    assert cap_confounded_detail(
        neg, es_info=dict(CAP_ES, cap_extended_to=12000)) is None
    assert cap_confounded_detail(
        neg, es_info=dict(CAP_ES, cap=4000, cap_initial=4000)) is None
    # 根 / confirm 不判
    assert cap_confounded_detail(neg, es_info=dict(CAP_ES),
                                 is_root_trial=True) is None
    assert cap_confounded_detail(neg, es_info=dict(CAP_ES),
                                 confirm_trial=True) is None
    # 越小越好:Δ>0 才是变差
    assert cap_confounded_detail({"verdict": "rejected",
                                  "delta": {"vs_parent": 5.0}},
                                 es_info=dict(CAP_ES),
                                 higher_is_better=False)
    assert cap_confounded_detail({"verdict": "rejected",
                                  "delta": {"vs_parent": -5.0}},
                                 es_info=dict(CAP_ES),
                                 higher_is_better=False) is None


def test_a1_evaluator_attaches_flag_keeps_verdict():
    from runtime.evaluator import Evaluator

    cfg = Config.load(skill_config=False)
    cfg.raw.setdefault("evaluator", {})["bootstrap_samples"] = 200
    rng = np.random.default_rng(7)
    n = 4000
    y = (rng.random(n) < 0.3).astype(int)
    parent = y * 1.0 + rng.normal(0, 0.8, n)
    child = y * 1.0 + rng.normal(0, 1.6, n)       # 显著更差
    out = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent, complexity=dict(CAP_CX))
    assert out["delta"]["vs_parent"] < 0
    assert out["verdict"] == "rejected", "verdict 不改"
    assert out["cap_confounded"] is True
    assert out["cap_confounded_detail"]["rule"] == "legacy:complexity"
    # 未撞 cap 的同一比较不带标记
    out2 = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent,
        complexity=dict(CAP_CX, cap_bound=False, es_stopped_by="early_stop"))
    assert "cap_confounded" not in out2


def test_a1_ledger_no_negative_reward(tmp_path):
    spec = _spec()
    run = _run(tmp_path, spec)
    trials = [
        _trial(run, "t0", gen=0, verdict="accepted", modules=()),
        # trial-017 形态:旧产物,无 cap_confounded 键,靠 complexity 重算
        _trial(run, "t17", gen=4, action="hpo", parents=["t0"],
               vs_parent=-0.00095, verdict="rejected", modules=("model",),
               runtime=17640.0, complexity=dict(CAP_CX)),
        # 只在 metrics.json 的 es_info 里留痕(无 complexity)
        _trial(run, "t18", gen=5, action="hpo", parents=["t0"],
               vs_parent=-0.0002, verdict="rejected", modules=("model",),
               runtime=3600.0, es_info=dict(CAP_ES)),
        # 正常 HPO:早停,负 Δ 照常记
        _trial(run, "t21", gen=6, action="hpo", parents=["t0"],
               vs_parent=-0.00042, verdict="rejected", modules=("model",),
               runtime=5800.0, es_info=dict(EARLY_ES)),
    ]
    st = DirectionLedger(run).refresh(trials, spec)
    arm = st["arms"]["model:hpo"]
    assert arm["n_tried"] == 3 and arm["n_cap_confounded"] == 2
    assert arm["mean_gain"] == pytest.approx(-0.00042), "只剩未混杂的 t21"
    assert arm["cost_hours"] == pytest.approx((17640 + 3600 + 5800) / 3600,
                                              rel=1e-3), "成本照常入账"
    assert "撞 cap" in DirectionLedger(run).render_summary()
    # 证伪判定扣除混杂试次:n_tried_eff=1 → 不判证伪
    from runtime.planner import AutonomousPlanner

    assert "model:hpo" not in AutonomousPlanner(
        run, spec, _cfg())._falsified_families()


def test_a1_report_limitation(tmp_path):
    from runtime.report import _lim_cap_confounded, collect_limitations

    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t0", gen=0, modules=())
    _trial(run, "t17", gen=4, action="hpo", parents=["t0"],
           vs_parent=-0.00095, verdict="rejected", modules=("model",),
           complexity=dict(CAP_CX))
    lines = _lim_cap_confounded(run, spec)
    assert lines and "t17" in lines[0] and "cap_confounded" in lines[0]
    assert "7999" in lines[0]
    assert any("cap_confounded" in x for x in collect_limitations(run, spec))


# ===========================================================================
# A2 冠军撞 cap → lr 修正子代
# ===========================================================================

def _cap_champion_run(tmp_path, *, hp=None, best_params=None):
    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t0", gen=0, value=0.80, modules=(), hp={"learning_rate": 0.05},
           es_info=dict(EARLY_ES), fc=1500)
    _trial(run, "t11", gen=2, value=0.8119, parents=["t0"], vs_parent=0.0001,
           verdict="accepted", hints=["info:seq_d365:window"],
           hp=hp if hp is not None else {"learning_rate": 0.0125,
                                         "max_depth": 6},
           best_params=best_params, es_info=dict(CAP_ES),
           n_features=700, fc=1647)
    return spec, run


def test_a2_cap_bound_champion_nominates_lr_fix(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec, run = _cap_champion_run(tmp_path)
    pl = AutonomousPlanner(run, spec, _cfg())
    champ = pl._parent()
    assert champ.trial_id == "t11"
    st = pl._champion_face_hpo_state(champ)
    assert st["cap_bound"] is True
    assert st["cap_bound_info"]["source"] == "metrics.es_info"

    cands = pl._model_candidates(champ)
    fix = [p for p in cands if p["family"] == "model:cap_bound_lr"]
    assert len(fix) == 1
    fix = fix[0]
    assert fix["action"] == "exploit" and fix["changed_modules"] == ["model"]
    assert fix["recipe"]["set_overrides"] == {"learning_rate": 0.025}
    cmd = fix["materialize_command"]
    for frag in ("--action exploit", "--from-parent t11",
                 "--set learning_rate=0.025", "--changed-modules model"):
        assert frag in cmd, (frag, cmd)
    hpo = [p for p in cands if p["action"] == "hpo"]
    assert hpo and all(p["cap_bound_parent"]["trial_id"] == "t11"
                       for p in hpo)
    same_face = next(p for p in hpo if p["family"] == "model:hpo")
    assert fix["acquisition"]["score"] > max(
        p["acquisition"]["score"] for p in hpo), "排在同面 HPO 之前"

    batch = pl.plan(slots=3)
    assert batch["proposals"][0]["family"] == "model:cap_bound_lr"
    assert same_face["proposal_id"] not in {
        p["proposal_id"] for p in batch["proposals"][:1]}


def test_a2_feature_phase_release_prefers_lr_fix(tmp_path):
    """特征期内(收敛释放一个名额)优先释放 lr 修正子代。"""
    from runtime.planner import AutonomousPlanner

    spec, run = _cap_champion_run(tmp_path)
    (run / "sample-profile").mkdir(exist_ok=True)
    (run / "sample-profile" / "information-inventory.yaml").write_text(
        yaml.safe_dump({"sources": [], "summary": {"unconsumed": 0}}),
        encoding="utf-8")
    pl = AutonomousPlanner(run, spec, _cfg(min_generations=5))
    batch = pl.plan(slots=4)
    gate = batch["feature_phase_gate"]
    assert gate["active"] is True
    rel = gate["released_proposal_id"]
    assert rel and any(p["proposal_id"] == rel
                       and p["family"] == "model:cap_bound_lr"
                       for p in batch["proposals"])


def test_a2_skipped_when_child_already_changed_lr(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec, run = _cap_champion_run(tmp_path)
    _trial(run, "t18", gen=5, action="exploit", parents=["t11"],
           value=0.8122, vs_parent=0.0003, modules=("model",),
           hp={"learning_rate": 0.025, "max_depth": 6},
           es_info=dict(EARLY_ES), fc=1647)
    # t18 成为冠军(不撞 cap)→ 不提名;把 t18 降级后 t11 仍是冠军也不提名
    pl = AutonomousPlanner(run, spec, _cfg())
    assert not [p for p in pl._model_candidates(pl._parent())
                if p["family"] == "model:cap_bound_lr"]
    t18 = pl.by_id["t18"]
    t18.data["evaluation"]["verdict"] = "rejected"
    t18.flush()
    pl = AutonomousPlanner(run, spec, _cfg())
    assert pl._parent().trial_id == "t11"
    assert not [p for p in pl._model_candidates(pl._parent())
                if p["family"] == "model:cap_bound_lr"]
    batch = pl.plan(slots=3)
    sk = batch["cap_bound_correction_skipped"]
    assert sk["skipped"] == "lr_already_changed_by_child"
    assert sk["child"] == "t18"


def test_a2_hpo_champion_lr_from_best_params(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec, run = _cap_champion_run(
        tmp_path, hp={}, best_params={"learning_rate": 0.0125,
                                      "max_depth": 5})
    pl = AutonomousPlanner(run, spec, _cfg())
    fix = [p for p in pl._model_candidates(pl._parent())
           if p["family"] == "model:cap_bound_lr"]
    assert fix and fix[0]["recipe"]["lr_source"] == "metrics.best_params"
    assert fix[0]["recipe"]["set_overrides"] == {"learning_rate": 0.025}


# ===========================================================================
# A5 同面判定用真实模型列
# ===========================================================================

def test_a5_same_face_uses_feature_columns(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t11", gen=2, value=0.812, n_features=700, fc=1647,
           es_info=dict(EARLY_ES))
    # 基础面 HPO:旧代码同样记 n_features=700,实际 1500 列 → 不是同面
    _trial(run, "t12", gen=3, action="hpo", parents=["t11"], value=0.80,
           vs_parent=-0.01, verdict="rejected", modules=("model",),
           n_features=700, fc=1500)
    pl = AutonomousPlanner(run, spec, _cfg())
    champ = pl.by_id["t11"]
    assert pl._model_n_features(champ) == 1647
    assert pl._champion_face_hpo_done(champ) is False
    assert pl._champion_face_hpo_state(champ)["done"] is False
    # 真同面(1647 列)HPO → 判已调参
    _trial(run, "t13", gen=3, action="hpo", parents=["t11"], value=0.811,
           vs_parent=-0.0002, verdict="rejected", modules=("model",),
           n_features=700, fc=1647)
    pl = AutonomousPlanner(run, spec, _cfg())
    assert pl._champion_face_hpo_done(pl.by_id["t11"]) is True
    assert pl._champion_face_hpo_state(pl.by_id["t11"])["last_hpo"] == "t13"


def test_a5_falls_back_to_n_features_without_columns_file(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t11", gen=2, n_features=1887)
    _trial(run, "t12", gen=3, action="hpo", parents=["t11"], vs_parent=0.0,
           verdict="rejected", modules=("model",), n_features=1887)
    pl = AutonomousPlanner(run, spec, _cfg())
    assert pl._model_n_features(pl.by_id["t11"]) == 1887
    assert pl._champion_face_hpo_done(pl.by_id["t11"]) is True


# ===========================================================================
# D1 证伪族过滤 / avoid 经验卡 / 清单自反馈
# ===========================================================================

def _ledger(run: Path, arms: dict) -> None:
    (run / "directions.json").write_text(json.dumps({
        "gain_convention": "direction_aware_gain_higher_is_better",
        "arms": arms}), encoding="utf-8")


def _inventory(run: Path, sources: list[dict]) -> None:
    (run / "sample-profile").mkdir(parents=True, exist_ok=True)
    (run / "sample-profile" / "information-inventory.yaml").write_text(
        yaml.safe_dump({"sources": sources}, allow_unicode=True),
        encoding="utf-8")


#: 本 run 台账原样(directions.json)
SEQGAP_ARM = {"n_tried": 2, "n_accepted": 0, "mean_delta": 3.91e-05,
              "std_delta": 2.588e-05, "lcb_delta": 8.56e-06,
              "mean_gain": 3.91e-05, "std_gain": 2.588e-05,
              "lcb_gain": 8.56e-06, "n_eff": 1.176, "cost_hours": 2.3,
              "delta_per_hour": 3.68e-06, "last_generation": 2}


def test_d1_falsified_rules_and_prefix_normalization(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t0", gen=0, modules=(), value=0.80)
    _ledger(run, {
        "feat:info:seq_gap_d365:recency": dict(SEQGAP_ARM),
        "feat:lcbneg": dict(SEQGAP_ARM, mean_gain=0.0004, lcb_gain=-0.0001,
                            std_gain=0.0005),
        "feat:alive": dict(SEQGAP_ARM, n_accepted=1),
        "feat:hopeful": dict(SEQGAP_ARM, mean_gain=0.0008, lcb_gain=0.0002,
                             std_gain=0.0005),
        # 两次里一次撞 cap 混杂 → 有效试次 1,不判
        "feat:capped": dict(SEQGAP_ARM, mean_gain=-0.001, lcb_gain=-0.001,
                            n_cap_confounded=1),
    })
    _inventory(run, [{"source_type": "reparse", "column": "seq_gap_d365",
                      "consumption_status": "partially_consumed",
                      "granularity_space": [
                          {"dimension": "recency", "hint": "r",
                           "state": "partial"}]}])
    pl = AutonomousPlanner(run, spec, _cfg())
    fal = pl._falsified_families()
    assert fal == {"feat:info:seq_gap_d365:recency", "feat:lcbneg"}
    assert pl._falsified_detail["feat:info:seq_gap_d365:recency"]["rule"] \
        == "ucb_gain<mde"
    assert pl._falsified_detail["feat:lcbneg"]["rule"] == "lcb_gain<0"

    batch = pl.plan(slots=8)
    fams = [p["family"] for p in batch["proposals"]]
    # 信息源候选 info:… 与台账 feat:info:… 归一后同键 → 两侧都被拒
    assert "info:seq_gap_d365:recency" not in fams
    assert "feat:info:seq_gap_d365:recency" not in fams
    rej = [r for r in batch["rejected_candidates"]
           if "family_falsified" in r["errors"]]
    assert len(rej) >= 2
    assert batch["falsified_detail"]["feat:lcbneg"]["rule"] == "lcb_gain<0"


# v2.18.0:test_d1_experience_avoid_cards 已移除 —— 测的是旧 memory/ 全局库的接线(由经验库视图取代),新行为见 tests/test_v2_18_integration.py

def test_d1_inventory_planning_alone_is_not_partial(tmp_path):
    from runtime.inventory import refresh_inventory

    spec = _spec(reparse=["seq_d365"])
    run = _run(tmp_path, spec)
    (run / "planner").mkdir()
    (run / "planner" / "proposal-batch-gen-01.yaml").write_text(
        yaml.safe_dump({"proposals": [{"family": "info:seq_d365:window"},
                                      {"family": "info:seq_d365:recency"}]}),
        encoding="utf-8")
    # 历史文件:planner 派生的 partial(非粘性)
    _inventory(run, [{"source_type": "reparse", "column": "seq_d365",
                      "consumption_status": "partially_consumed",
                      "granularity_space": [
                          {"dimension": "window", "state": "partial",
                           "state_source": "planner"},
                          {"dimension": "bucketing", "state": "implemented",
                           "state_source": "agent"}]}])
    inv = refresh_inventory(run, spec)
    g = {x["dimension"]: (x["state"], x.get("state_source"))
         for x in inv["sources"][0]["granularity_space"]}
    assert g["window"] == ("unexplored", None), "只被规划过不算触达"
    assert g["recency"] == ("unexplored", None)
    assert g["bucketing"] == ("implemented", "agent"), "Agent 裁决仍粘性"
    # trial 证据(已评估、非 invalid)才标 partial
    _trial(run, "t1", hints=["feat:info:seq_d365:recency"],
           verdict="promising", vs_parent=5e-5)
    _trial(run, "t2", hints=["info:seq_d365:window"], verdict="invalid")
    inv = refresh_inventory(run, spec)
    g = {x["dimension"]: (x["state"], x.get("state_source"))
         for x in inv["sources"][0]["granularity_space"]}
    assert g["recency"] == ("partial", "trial_evidence")
    assert g["window"] == ("unexplored", None), "invalid 不算证据"


# ===========================================================================
# C2 目标可达性
# ===========================================================================

#: 本 run capacity-report 的样本曲线/列数曲线(原样数值)
RUN_SC = {"points": [{"rows": 18486, "metric": 0.7749162},
                     {"rows": 36972, "metric": 0.7826595},
                     {"rows": 73944, "metric": 0.7858883},
                     {"rows": 147889, "metric": 0.7917746}],
          "last_step_gain": 0.00588628441385064,
          "last_step_gain_se": 0.0003392005151962585, "still_steep": True}
RUN_FC = {"tail_gain": -0.00012, "saturated": True, "tail_gain_negative": True}
RUN_IR = {"local_ambiguity_index": 0.9607908499099967}


def test_c2_target_reachability_verdicts():
    from runtime.capacity import target_reachability

    tr = target_reachability(RUN_SC, RUN_FC, RUN_IR, target=0.85,
                             metric="auc", best_observed=0.8122422,
                             best_source="best_trial", best_trial="t18")
    assert tr["verdict"] == "not_reachable"
    assert tr["shortfall"] == pytest.approx(0.0377578, abs=1e-6)
    assert tr["extrapolated_ceiling"] == pytest.approx(
        0.8122422 + 0.00588628 + 1.96 * 0.00033920, abs=1e-6)
    assert tr["extrapolated_ceiling"] < 0.85
    assert "局部歧义指数" in tr["rationale"]
    # 列数曲线未饱和 → 行外推不覆盖新列,降为 unlikely
    assert target_reachability(
        RUN_SC, {"saturated": False}, RUN_IR, target=0.85, metric="auc",
        best_observed=0.8122)["verdict"] == "unlikely"
    # 缺口在点估计 headroom 内 → reachable;只在上界内 → unlikely;已达标
    assert target_reachability(RUN_SC, RUN_FC, None, target=0.815,
                               metric="auc", best_observed=0.812
                               )["verdict"] == "reachable"
    assert target_reachability(RUN_SC, RUN_FC, None, target=0.8183,
                               metric="auc", best_observed=0.812
                               )["verdict"] == "unlikely"
    assert target_reachability(RUN_SC, RUN_FC, None, target=0.80,
                               metric="auc", best_observed=0.812
                               )["verdict"] == "reachable"
    # 越小越好:rmse 目标 100,当前 130,曲线末段每翻倍 −2 → 不可达
    rm = target_reachability({"points": [{"metric": 140}, {"metric": 132}],
                              "last_step_gain": 2.0, "last_step_gain_se": 0.5},
                             {"saturated": True}, None, target=100.0,
                             metric="rmse", best_observed=130.0)
    assert rm["verdict"] == "not_reachable"
    assert rm["extrapolated_ceiling"] == pytest.approx(130 - 2.98)
    # 无目标 → 跳过
    sk = target_reachability(RUN_SC, RUN_FC, RUN_IR, target=None,
                             metric="auc", best_observed=0.81)
    assert sk["skipped"] and sk["verdict"] is None


def test_c2_assess_from_run_and_cached_capacity_check(tmp_path):
    import pandas as pd

    from runtime.capacity import assess_target_reachability, run_capacity_check

    spec = _spec(target=0.85)
    run = _run(tmp_path, spec)
    _trial(run, "t11", gen=2, value=0.8119)
    _trial(run, "t18", gen=5, value=0.8122422, parents=["t11"],
           vs_parent=0.0003)
    _trial(run, "tinv", gen=5, value=0.99, verdict="invalid")
    report = {"metric": "auc", "sample_size_curve": RUN_SC,
              "feature_count_curve": RUN_FC, "irreducible_noise": RUN_IR,
              "reference_trial": "t11"}
    tr = assess_target_reachability(run, report, spec)
    assert tr["best_observed_trial"] == "t18", "invalid 不参与"
    assert tr["best_observed"] == pytest.approx(0.8122422)
    assert tr["verdict"] == "not_reachable"
    # capacity-check 缓存命中时(不重拟合)也按最新 trial 重算并落盘
    splits = run / "sample-profile" / "splits-sample"
    splits.mkdir(parents=True)
    pd.DataFrame({"x": [1, 2]}).to_parquet(splits / "train.parquet")
    (splits / "manifest.json").write_text("{}", encoding="utf-8")
    from runtime.capacity import _splits_fingerprint

    cached = dict(report, splits_fingerprint=_splits_fingerprint(splits))
    (run / "sample-profile" / "capacity-report.json").write_text(
        json.dumps(cached), encoding="utf-8")
    out = run_capacity_check(run)
    assert out["target_reachability"]["verdict"] == "not_reachable"
    disk = json.loads((run / "sample-profile" / "capacity-report.json")
                      .read_text(encoding="utf-8"))
    assert disk["target_reachability"]["best_observed_trial"] == "t18"


def test_c2_planner_context_section10_renders():
    from runtime.planner_context import render_planner_context
    from runtime.capacity import target_reachability

    tr = target_reachability(RUN_SC, RUN_FC, RUN_IR, target=0.85,
                             metric="auc", best_observed=0.8122)
    md = render_planner_context({"capacity": {
        "ready": True, "action_ranking": [], "target_reachability": tr}})
    assert "目标可达性" in md and "not_reachable" in md


# ===========================================================================
# C4 shadow gap 诊断 vs 确认门
# ===========================================================================

def test_c4_shadow_gap_sanity_reader():
    from runtime.report import _confirmation_gate_lines, shadow_gap_sanity

    new = {"shadow_test": {"auc": 0.806, "gap_vs_validation": 0.0059,
                           "gap_relative": 0.0073,
                           "gap_sanity_passed": False,
                           "criterion": "gap_sanity"},
           "fresh_confirmation": {"passed": True,
                                  "criterion": "noninferiority_seed_aware_v1"}}
    g = shadow_gap_sanity(new)
    assert g["passed"] is False and g["key"] == "gap_sanity_passed"
    assert g["blocking"] is False and g["criterion"] == "gap_sanity"
    legacy = {"shadow_test": {"auc": 0.806, "gap_vs_validation": 0.0059,
                              "passed": False}}
    gl = shadow_gap_sanity(legacy)
    assert gl["passed"] is False and gl["key"] == "passed(legacy)"
    assert gl["criterion"] == "gap_sanity"
    assert shadow_gap_sanity({"verdict": "accepted"}) is None
    lines = _confirmation_gate_lines(new, "t20")
    assert len(lines) == 2
    assert "fresh_confirmation.passed=True" in lines[0] and "交付门" in lines[0]
    assert "gap_sanity_passed=False" in lines[1] and "非阻断" in lines[1]
