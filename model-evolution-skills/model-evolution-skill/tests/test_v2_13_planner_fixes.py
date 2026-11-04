"""回归测试:v2.13 planner 四项修复(D1~D4)。

事故来源(docs/MODELEVO_SELF_IMPROVEMENT_REVIEW_AND_ROADMAP_20260919.md
F-P2/P3/P4/P5,run-jt-draw-t7 / example-seq-v3-v2.12.2.9 实测):

D1 同一 proposal 同时出现在 proposals[] 与 rejected_candidates[]
    ``_direction_candidates`` 对每个 UCB arm ``for _ in range(min(n, 2))``
    生成两个字段完全相同的 proposal → 语义 hash 与 proposal_id 相同 →
    第二个被 plan() 判 duplicate_semantics,但 proposal_id 与第一个
    一样——proposal-batch-gen-02 里同一 prop-id 既在 proposals 又在
    rejected。修法:每 arm 只产 1 个候选 + plan() 末尾断言性去重。

D2 跨代重复提名已证伪方向
    候选生成不查方向台账负面结果,且去重身份含 parent_ids——冠军换代
    后同一 family 身份变化即"新候选"(gen-03 重提 3 个 gen-01 已判负的
    family)。修法:``_falsified_families()``(n_tried>=2 且
    n_accepted==0 且 mean_delta<=0)命中 explore/exploit 候选时以
    ``family_falsified`` 进 rejected;confirm/hpo/model 族不受影响。

D3 acquisition 量纲错配
    expected_gain/uncertainty 来自方向台账(主指标量纲,~0.001),
    information_gain/novelty 是手设常数(0.3~2.0),直接相加后排序
    实际由常数决定,方向后验没有影响。修法:两项除以 MDE 再进
    score;potential_gain_upper_bound 保持原量纲(evolution 停止
    判定按主指标量纲与 MDE 直接比较)。

D4 capacity 闸门永久锁(v2.16.0 起改为特征期门)
    changed_modules 含 model/loss 或 action=hpo 的候选,只要
    capacity-report 的 action_ranking 含 add_rows/new_information 就
    一律拒绝——而这两个动作在固定样本源下结构性不可兑现,HPO 是实测
    唯一正期望动作。v2.13 修法为分期(特征期硬拒、期满 advisory);
    v2.16.0 把闸门与 capacity-report 解耦:特征期
    (n_done < feature_exploration.min_generations)内以
    ``feature_phase_model_locked`` 拒绝,期满后不拦,与报告是否存在、
    报告说什么都无关;batch.feature_phase_gate 记录状态。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from runtime.config import Config  # noqa: E402
from runtime.planner import (FALSIFIED_MIN_TRIED,  # noqa: E402
                             AutonomousPlanner)
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialFactory  # noqa: E402


def _spec() -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v213-planner", "task_name": "v213-planner",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
        "resource_budget": {"max_trials": 30},
    })


def _mk_trial(fac: TrialFactory, generation: int = 0,
              modules: tuple = ()):
    """建一个最小可用的 TrialRecord(与既有 planner 测试同款桩)。"""
    return fac.create(
        algorithm="lightgbm",
        hypothesis={"statement": "测试用最小假设陈述",
                    "rationale": "r", "falsification": "f"},
        execution={"action_description": ["a"], "code_entry": "train.py",
                   "command": "python train.py",
                   "changed_modules": list(modules)},
        search_action="explore", generation=generation)


def _write_directions(run_dir: Path, arms: dict) -> None:
    """落盘方向台账 stub(结构与 DirectionLedger.refresh 输出一致)。"""
    (run_dir / "directions.json").write_text(
        json.dumps({"updated_at": "stub", "arms": arms}), encoding="utf-8")


def _arm(n_tried: int, n_accepted: int, mean_delta: float,
         std_delta: float = 0.0005, cost_hours: float = 1.0,
         dph: float | None = None) -> dict:
    return {
        "n_tried": n_tried, "n_accepted": n_accepted,
        "mean_delta": mean_delta, "std_delta": std_delta,
        "cost_hours": cost_hours,
        "delta_per_hour": (dph if dph is not None
                           else max(mean_delta, 0.0) / max(cost_hours, 0.05)),
        "accept_rate": round(n_accepted / max(n_tried, 1), 3),
        "last_generation": 2,
    }


def _baseline_trial(fac: TrialFactory) -> None:
    """gen-00 baseline:root_accepted,可作 confirmation/方向候选父代。"""
    t = _mk_trial(fac, 0)
    t.data["evaluation"] = {
        "verdict": "accepted", "trial_id": t.trial_id,
        "primary_metric": {"name": "auc", "value": 0.75},
        "root_baseline_only": True, "integrity": {"valid": True},
    }
    t.flush()


def _capacity_report(run_dir: Path) -> None:
    """capacity-check 产物 stub:action_ranking 优先 add_rows/new_information。"""
    (run_dir / "sample-profile").mkdir(parents=True, exist_ok=True)
    (run_dir / "sample-profile" / "capacity-report.json").write_text(
        json.dumps({"verdict": {"action_ranking": [
            {"action": "add_rows", "priority": 1},
            {"action": "new_information", "priority": 1},
            {"action": "model_hpo", "priority": 2}]}}), encoding="utf-8")


# --------------------------------------------------------------------------
# D1: 同一 proposal 不再同时出现在 proposals[] 与 rejected_candidates[]
# --------------------------------------------------------------------------
def test_d1_no_duplicate_proposal_ids(tmp_path: Path):
    """某 arm 分配 >=2 槽时,同 family 的 exploit 候选只出现一次,
    且 proposals 与 rejected 的 proposal_id 集合不相交。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # 单一健康 arm 吃满全部 UCB 槽(旧实现会给它 min(n,2)=2 个重复候选)
    _write_directions(run_dir, {
        "feat:grow": _arm(1, 1, 0.004, std_delta=0.001,
                          cost_hours=0.5, dph=0.008),
    })
    cfg = Config.load(skill_config=False)
    fac = TrialFactory(run_dir, spec=_spec(), config=cfg)
    _baseline_trial(fac)
    planner = AutonomousPlanner(run_dir, _spec(), cfg)

    batch = planner.plan(slots=6)
    prop_ids = {p["proposal_id"] for p in batch["proposals"]}
    rej_ids = {r["proposal_id"] for r in batch["rejected_candidates"]}
    assert prop_ids, "feat:grow 的 exploit 候选必须进入批次"
    assert not (prop_ids & rej_ids), (
        f"proposals 与 rejected 出现重叠 id: {prop_ids & rej_ids}")
    # 同一 family 的 exploit 候选只出现一次(旧实现因 min(n,2) 生成两份)
    grow = [p for p in batch["proposals"] if p["family"] == "feat:grow"]
    assert len(grow) == 1, f"feat:grow 出现 {len(grow)} 次"
    assert grow[0]["action"] == "exploit"
    # 重复候选不再被生成 → rejected 里也不该有 duplicate_semantics 的
    # 同 id 记录
    dup_grow = [r for r in batch["rejected_candidates"]
                if "duplicate_semantics" in r["errors"]]
    assert all(r["proposal_id"] not in prop_ids for r in dup_grow)


# --------------------------------------------------------------------------
# D2: 已证伪 family 不再被提名
# --------------------------------------------------------------------------
def test_d2_falsified_family_rejected(tmp_path: Path):
    """feat:dead(n_tried=3, n_accepted=0, mean_delta<0)不在 proposals,
    出现在 rejected 且 errors 含 family_falsified;batch 记录
    falsified_families;confirm 候选不受影响。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_directions(run_dir, {
        # 健康 arm:有 accepted、正均值 → 允许继续提名
        "feat:alive": _arm(2, 1, 0.002, cost_hours=1.0, dph=0.002),
        # 已证伪 arm:3 次全负、零 accepted
        "feat:dead": _arm(3, 0, -0.0005, std_delta=0.0004,
                          cost_hours=1.5, dph=0.0),
    })
    cfg = Config.load(skill_config=False)
    # v2.19.0:confirm.policy 默认 final_only(中途不提名 confirm);本用例
    # 验证"证伪规则不误伤 confirm",按旧的每代提名口径运行
    cfg.raw.setdefault("confirm", {})["policy"] = "per_generation"
    fac = TrialFactory(run_dir, spec=_spec(), config=cfg)
    _baseline_trial(fac)
    planner = AutonomousPlanner(run_dir, _spec(), cfg)

    batch = planner.plan(slots=8)
    families = [p["family"] for p in batch["proposals"]]
    assert "feat:dead" not in families
    assert "feat:alive" in families, "健康方向仍应被提名"
    assert batch["falsified_families"] == ["feat:dead"]
    # dead 的 exploit 候选被生成过(UCB 给了槽)且以 family_falsified 拒
    dead_rej = [r for r in batch["rejected_candidates"]
                if "family_falsified" in r["errors"]]
    assert dead_rej, "feat:dead 候选应以 family_falsified 进 rejected"
    # confirm 候选不受证伪规则影响(root baseline 可独立确认)
    confirms = [p for p in batch["proposals"] if p["action"] == "confirm"]
    assert confirms, "confirm 候选不应被 family_falsified 误伤"
    assert all("family_falsified" not in (r["errors"])
               for r in batch["rejected_candidates"]
               if r["proposal_id"] in
               {p["proposal_id"] for p in confirms})


def test_d2_falsified_rule_thresholds(tmp_path: Path):
    """阈值口径:n_tried < FALSIFIED_MIN_TRIED、或有 accepted、或
    mean_delta > 0 的 arm 都不判证伪。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_directions(run_dir, {
        "feat:fresh": _arm(1, 0, -0.001),          # 只试 1 次 → 不判
        "feat:redeemed": _arm(2, 1, -0.001),       # 有 accepted → 不判
        "feat:hopeful": _arm(2, 0, 0.0005),        # 正均值 → 不判
        "feat:flat": _arm(2, 0, 0.0),              # 2 次零收益零接受 → 判
    })
    cfg = Config.load(skill_config=False)
    planner = AutonomousPlanner(run_dir, _spec(), cfg)
    falsified = planner._falsified_families()
    assert falsified == {"feat:flat"}
    assert FALSIFIED_MIN_TRIED == 2


# --------------------------------------------------------------------------
# D3: acquisition 量纲归一(MDE)
# --------------------------------------------------------------------------
def test_d3_acquisition_mde_normalization(tmp_path: Path):
    """expected_gain=0.002(=2×MDE)的候选 score 必须高于
    information_gain=1.2 的常数候选;expected_gain/uncertainty 的
    归一化值落盘;potential_gain_upper_bound 原样未归一化。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = Config.load(skill_config=False)
    mde = float(cfg.get("evaluator.minimum_meaningful_improvement", 0.001))
    assert mde == pytest.approx(0.001)
    planner = AutonomousPlanner(run_dir, _spec(), cfg)

    measured = planner._base(
        action="exploit", family="feat:measured",
        hypothesis="h", rationale="r", falsification="f",
        expected_gain=0.002, uncertainty=0.0, information_gain=0.0,
        novelty=0.0, cost_hours=0.5,
        potential_gain_upper_bound=0.003)
    constant = planner._base(
        action="explore", family="feat:unconsumed",
        hypothesis="h", rationale="r", falsification="f",
        expected_gain=0.0, uncertainty=0.0, information_gain=1.2,
        novelty=0.0, cost_hours=0.5,
        potential_gain_upper_bound=0.002)

    a, b = measured["acquisition"], constant["acquisition"]
    # 方向后验(+0.002 = 2 MDE)必须赢过 1.2 的信息常数
    assert a["score"] > b["score"], (
        f"measured={a['score']} 应高于 constant={b['score']}")
    # 归一化中间量落盘,原始字段保持主指标量纲
    assert a["expected_gain_mde"] == pytest.approx(0.002 / mde)
    assert a["uncertainty_mde"] == pytest.approx(0.0)
    assert a["expected_gain"] == pytest.approx(0.002)
    # upper bound 不归一化(停止判定按主指标量纲与 MDE 直接比较)
    assert a["potential_gain_upper_bound"] == pytest.approx(0.003)
    assert b["potential_gain_upper_bound"] == pytest.approx(0.002)
    # 手算复核:score = eg_mde + 0.5*u_mde + ig + nov - cost_penalty
    assert a["score"] == pytest.approx(
        0.002 / mde + 0.5 * 0.0 + 0.0 + 0.0 - 0.5 * 0.15)


# --------------------------------------------------------------------------
# D4: capacity 闸门分期(特征期硬拒 → 期满 advisory)
# --------------------------------------------------------------------------
def _spec_with_inventory(tmp_path: Path) -> TaskSpec:
    """带 reparse 列的 spec + 一个 unconsumed 信息源清单。

    收敛释放通道(_exploration_converged 分支 a)要求 unconsumed==0;
    清单里有未消费源 → 不收敛 → 特征期内闸门没有放行通道,
    才能验证"硬拒"本体行为(而不是收敛释放的旁路)。
    """
    spec = TaskSpec(data={
        "task_id": "v213-planner", "task_name": "v213-planner",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "features": {"reparse": ["seq_d365"]},
        "data": {},
        "resource_budget": {"max_trials": 30},
    })
    return spec


def _write_inventory(run_dir: Path) -> None:
    (run_dir / "sample-profile").mkdir(parents=True, exist_ok=True)
    (run_dir / "sample-profile" / "information-inventory.yaml").write_text(
        yaml.safe_dump({
            "sources": [{
                "source_type": "reparse", "column": "seq_d365",
                "consumption_status": "unconsumed",
                "granularity_space": [
                    {"dimension": "window", "state": "unexplored"}]}],
            "summary": {"n_sources": 1, "unconsumed": 1,
                        "partially_consumed": 0, "fully_consumed": 0}},
            allow_unicode=True), encoding="utf-8")


def test_d4_feature_phase_locks_model_without_capacity_report(
        tmp_path: Path):
    """(a) 只有 gen-00 baseline trial 时(n_done=0 < min_generations=2),
    hpo 候选被 feature_phase_model_locked 硬拒——**不需要**
    capacity-report.json 存在(v2.16.0 前报告缺失时门完全敞开)。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_inventory(run_dir)
    assert not (run_dir / "sample-profile" / "capacity-report.json").exists()
    cfg = Config.load(skill_config=False)
    assert int(cfg.get("feature_exploration.min_generations", 2)) == 2
    spec = _spec_with_inventory(tmp_path)
    fac = TrialFactory(run_dir, spec=spec, config=cfg)
    _baseline_trial(fac)
    planner = AutonomousPlanner(run_dir, spec, cfg)
    # 信息源未清账 → 无收敛释放通道 → 硬拒是唯一行为
    assert planner._exploration_converged() == (False, "")

    assert planner._feature_generations_done() == 0
    batch = planner.plan(slots=4)
    hpo = [p for p in batch["proposals"]
           if p["action"] == "hpo" or
           set(p.get("changed_modules") or []) & {"model", "loss"}]
    assert not hpo, "特征期内 model/hpo 候选必须被硬拒"
    blocked = [r for r in batch["rejected_candidates"]
               if "feature_phase_model_locked" in r["errors"]]
    assert blocked, "硬拒必须以 feature_phase_model_locked 落盘"
    gate = batch["feature_phase_gate"]
    assert gate["active"] is True
    assert gate["feature_generations_done"] == 0
    assert gate["min_generations"] == 2
    assert sorted(r["proposal_id"] for r in blocked) == \
        gate["blocked_proposal_ids"]
    assert "capacity_gate" not in batch


def test_d4_feature_phase_ends_regardless_of_capacity_report(
        tmp_path: Path):
    """(b) gen-01、gen-02 各一个已评估 trial 后(n_done=2 >= 2),
    即便 capacity-report 判 add_rows/new_information 优先,hpo 候选也按
    acquisition 正常入选、不带任何闸门留痕。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _capacity_report(run_dir)
    _write_inventory(run_dir)
    cfg = Config.load(skill_config=False)
    spec = _spec_with_inventory(tmp_path)
    fac = TrialFactory(run_dir, spec=spec, config=cfg)
    _baseline_trial(fac)
    for g in (1, 2):
        t = _mk_trial(fac, g, modules=("feature",))
        t.data["evaluation"] = {"verdict": "rejected",
                                "trial_id": t.trial_id}
        t.flush()

    planner = AutonomousPlanner(run_dir, spec, cfg)
    assert planner._feature_generations_done() == 2
    batch = planner.plan(slots=4)
    hpo = [p for p in batch["proposals"] if p["action"] == "hpo"]
    assert hpo, "特征期满后 hpo 候选应按 acquisition 正常竞争入选"
    for p in hpo:
        assert "capacity_advisory" not in p
        assert "feature_phase_release" not in p
    assert not any(
        "feature_phase_model_locked" in r["errors"]
        for r in batch["rejected_candidates"]), (
        "特征期满后不再产生拒绝记录")
    gate = batch["feature_phase_gate"]
    assert gate["active"] is False
    assert gate["feature_generations_done"] == 2
    assert gate["blocked_proposal_ids"] == []
    assert gate["released_proposal_id"] is None


def test_d4_min_generations_zero_disables_gate(tmp_path: Path):
    """feature_exploration.min_generations=0 关闭特征期门。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = Config.load(skill_config=False)
    cfg.raw.setdefault("feature_exploration", {})["min_generations"] = 0
    fac = TrialFactory(run_dir, spec=_spec(), config=cfg)
    _baseline_trial(fac)
    planner = AutonomousPlanner(run_dir, _spec(), cfg)
    stub = {"action": "hpo", "family": "model:hpo", "parent_ids": [],
            "algorithm": "lightgbm", "data_view": "v1",
            "changed_modules": ["model"], "hypothesis": "h",
            "falsification": "f", "proposal_id": "prop-x"}
    assert "feature_phase_model_locked" not in planner._validate(dict(stub))


def test_d4_validate_release_id(tmp_path: Path):
    """特征期内无释放名额的调用返回门错误码;带释放名额的候选放行并
    留痕;confirm 不算模型方向(与 test_v212dot2dot2 场景对齐)。"""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = Config.load(skill_config=False)
    fac = TrialFactory(run_dir, spec=_spec(), config=cfg)
    _baseline_trial(fac)
    planner = AutonomousPlanner(run_dir, _spec(), cfg)

    stub = {"action": "hpo", "family": "model:hpo", "parent_ids": [],
            "algorithm": "lightgbm", "data_view": "v1",
            "changed_modules": ["model"], "hypothesis": "h",
            "falsification": "f", "proposal_id": "prop-x"}
    assert "feature_phase_model_locked" in planner._validate(dict(stub))
    released = dict(stub)
    errs = planner._validate(released, release_id="prop-x")
    assert "feature_phase_model_locked" not in errs
    rel = released.get("feature_phase_release") or {}
    assert rel.get("granted") is True and rel.get("reason")
    confirm = dict(stub, action="confirm", proposal_id="prop-c")
    assert "feature_phase_model_locked" not in planner._validate(confirm)
