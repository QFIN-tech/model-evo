"""v2.18.4 修复批次单元验证:req-t7 400w 全流程复盘(4M × 1493 二分类)。

来源:`run4-v2.18.0-3.ds-flash`(req_cnt_t7 二分类,400w 行)从 init 到交付的
完整一轮,以及同源 `run4-v2.18.0` / `run4-v2.18.0-1` 的产物对照。每条对应
一处实测缺陷:

A1 报告 §2「总样本量」取的是 profile **分析面**(下采样后的 train split),
   实测 4,000,000 行的样本被写成 739,444;
A2 报告 §13 `multi_seed_std: —` —— final-model/evaluation.json 里根本没有
   stability 键,而冠军 trial 有三个 seed 实测的 1.75e-4;
A3 信息源清单把**跨数据视图**、**被切片门拦下**的 rejected 臂回填成
   explicitly_falsified,把两个真实未探的粒度永久标成"已证伪";
A4 方向台账按整面 group_importance 的 top 族归因,于是"摘 seqh_ 的消融臂"
   被记成 feat:seqc2、真正加 seqh_ 的臂被记成 info: 源;
A5 leaderboard / 报告对 7 个 rejected trial 全无原因(交付校验要求"含原因");
A6 diagnose 对跨视图不可比臂报 medium 异常并要求"重评"(重评只会重复同一结论);
A7 经验库榜单 rank-1/rank-2 是同一训练配方(模型工件 sha 全同),两 trial
   各烧 69 min × 2 卡;
A8 planner-context 只显示 se_ratio=4.07,与 recommend_v1=false 的结论相反。

运行:python -m pytest tests/test_v2_18_4_fixes.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SKILL  # noqa: E402

sys.path.insert(0, str(SKILL))

from runtime.config import Config  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def _spec(tmp_path: Path) -> TaskSpec:
    data = {
        "task_id": "t", "task_name": "t",
        "objective_description": "t",
        "model_type": "binary_classification",
        "business_domains": ["user_operation"],
        "target": {"name": "y", "positive_definition": "y > 0",
                   "observation_window": "past",
                   "performance_window": "T7"},
        "time": {"column": "pday"},
        "entity": {"id_column": "uid"},
        "data": {"data_loader": "data-loader.py"},
        "primary_metric": "auc",
        "secondary_metrics": ["ks"],
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "features": {"reparse": [{"column": "seq_d365",
                                  "reason": "form_unusable",
                                  "plan": "x"}]},
    }
    p = tmp_path / "task-spec.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return TaskSpec.load(p)


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "trials").mkdir(parents=True, exist_ok=True)
    return run


def _write_trial(run: Path, tid: str, *, view="v1", verdict="rejected",
                 reason="相对父代无提升", vs_parent=-1e-5,
                 incomparable=False, parents=("t-parent",),
                 generation=1) -> dict:
    d = {
        "trial_id": tid, "generation": generation, "search_action": "explore",
        "data_view": view, "changed_modules": ["feature"],
        "parent_trial_ids": list(parents),
        "evaluation": {
            "verdict": verdict, "reason": reason,
            "delta": {"vs_parent": vs_parent,
                      **({"baseline_incomparable": True}
                         if incomparable else {})},
            "integrity": {"valid": True},
        },
        "result": {"runtime_sec": 60.0},
    }
    tdir = run / "trials" / tid
    (tdir / "config").mkdir(parents=True, exist_ok=True)
    (tdir / "trial.yaml").write_text(
        yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    (tdir / "evaluation.json").write_text(json.dumps(d["evaluation"]),
                                          encoding="utf-8")
    return d


class _T:
    """backfill_from_trials 接受的最小 trial 协议。"""

    def __init__(self, data: dict):
        self.data = data
        self.trial_id = data["trial_id"]
        self.generation = data.get("generation")


def _inventory(run: Path, dims_state: dict | None = None) -> Path:
    gs = []
    for d in ("window", "aggregation", "bucketing"):
        g = {"dimension": d, "hint": "h", "state": "unexplored"}
        if dims_state and d in dims_state:
            g.update(dims_state[d])
        gs.append(g)
    p = run / "sample-profile" / "information-inventory.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"sources": [{
        "source_type": "reparse", "column": "seq_d365",
        "consumption_status": "unconsumed",
        "granularity_space": gs}]}, allow_unicode=True), encoding="utf-8")
    return p


def _states(p: Path) -> dict:
    src = yaml.safe_load(p.read_text(encoding="utf-8"))["sources"][0]
    return {g["dimension"]: (g["state"], g.get("state_source"))
            for g in src["granularity_space"]} | {
        "_status": src["consumption_status"]}


# ---------------------------------------------------------------------------
# A3 信息源清单:只认方向证据
# ---------------------------------------------------------------------------
def test_cross_view_rejection_is_not_falsification(tmp_path):
    """跨视图臂(reason=无父代可比分数, baseline_incomparable)不得证伪粒度。

    实测:gen-01 三个 data_view=sample 的臂全部 rejected,旧实现在
    seq_gap_d365:bucketing 与 seq_d365:aggregation 上写下
    explicitly_falsified,而这两项其实从未被真正检验过。
    """
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path)
    p = _inventory(run)
    data = _write_trial(run, "t1", view="sample", verdict="rejected",
                        reason="无父代可比分数", vs_parent=None,
                        incomparable=True)
    rep = backfill_from_trials(run, [_T(data)], generation=1)
    assert rep["changes"] == [], "跨视图臂不得产生回填"
    assert _states(p)["window"] == ("unexplored", None)


def test_slice_gate_rejection_is_not_falsification(tmp_path):
    """切片门拦下的 rejected 是质量门,不是"该粒度无信息"。"""
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path)
    p = _inventory(run)
    data = _write_trial(run, "t1", verdict="rejected",
                        reason="稳定性或切片/时序检查未通过",
                        vs_parent=+4.9e-5)
    backfill_from_trials(run, [_T(data)], generation=1)
    assert _states(p)["window"] == ("unexplored", None)


def test_true_directional_rejection_still_falsifies(tmp_path):
    """真正的方向性负结论仍要证伪(不能因为修 A3 把口径放宽成永不证伪)。"""
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path)
    p = _inventory(run)
    data = _write_trial(run, "t1", verdict="rejected",
                        reason="相对父代无提升", vs_parent=-2.9e-5)
    data["family_hints"] = ["info:seq_d365:window"]
    (run / "trials" / "t1" / "trial.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    backfill_from_trials(run, [_T(data)], generation=1)
    assert _states(p)["window"] == ("explicitly_falsified", "trial_backfill")


def test_repair_removes_legacy_false_falsification(tmp_path):
    """旧版写下的假证伪必须被复核撤销,且 consumption_status 双向回退。

    回填"只升不降" ⇒ 判据修好后,存量假证伪永远不会自己消失。这里模拟
    run4 现场:seq_d365:aggregation 挂着 state_source=trial_backfill +
    两条 sample 视图证据。
    """
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path)
    p = _inventory(run, {"aggregation": {
        "state": "explicitly_falsified",
        "state_source": "trial_backfill",
        "state_evidence": ["t1", "t2"],
        "state_verdict": "rejected"}})
    t1 = _write_trial(run, "t1", view="sample", reason="无父代可比分数",
                      vs_parent=None, incomparable=True)
    t2 = _write_trial(run, "t2", view="v1", reason="稳定性或切片/时序检查未通过",
                      vs_parent=-2.5e-5)
    rep = backfill_from_trials(run, [_T(t1), _T(t2)])
    assert rep["status"] == "updated"
    assert rep["demotions"], "应记录降级动作"
    st = _states(p)
    assert st["aggregation"] == ("unexplored", None), \
        "两条证据都不成立 ⇒ 回 unexplored 且清掉 state_source"
    assert st["_status"] == "unconsumed", "源状态必须双向回算"


def test_repair_keeps_valid_backfilled_evidence(tmp_path):
    """复核只撤无效证据,有效证据与其状态保留。"""
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path)
    p = _inventory(run, {"window": {
        "state": "explicitly_falsified",
        "state_source": "trial_backfill",
        "state_evidence": ["t-bad", "t-good"]}})
    bad = _write_trial(run, "t-bad", view="sample", reason="无父代可比分数",
                       vs_parent=None, incomparable=True)
    good = _write_trial(run, "t-good", verdict="rejected",
                        reason="相对父代无提升", vs_parent=-1e-5)
    backfill_from_trials(run, [_T(bad), _T(good)])
    st = _states(p)
    assert st["window"][0] == "explicitly_falsified"


def test_repair_restores_older_generation_evidence(tmp_path):
    """撤掉假证伪后,被它覆盖的往代证据必须恢复(生产按代调用)。

    state_evidence 只记最强状态的 trial:gen-1 的 partial 被 gen-2 的
    切片门臂覆盖成证伪后,列表里只剩后者。生产以 generation=gid 调用,
    第 2 步不扫往代 —— 复核若不全量重算,往代 partial 永久丢失。
    """
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path)
    p = _inventory(run, {"window": {
        "state": "explicitly_falsified",
        "state_source": "trial_backfill",
        "state_evidence": ["t-bad"]}})
    good = _write_trial(run, "t-good", verdict="unconfirmed", reason="x",
                        vs_parent=2e-5, generation=1)
    good["family_hints"] = ["info:seq_d365:window"]
    bad = _write_trial(run, "t-bad", verdict="rejected",
                       reason="稳定性或切片/时序检查未通过",
                       vs_parent=-2.4e-5, generation=2)
    bad["family_hints"] = ["info:seq_d365:window"]
    rep = backfill_from_trials(run, [_T(good), _T(bad)], generation=3)
    assert rep["demotions"]
    st = _states(p)
    assert st["window"] == ("partial", "trial_backfill")
    src = yaml.safe_load(p.read_text(encoding="utf-8"))["sources"][0]
    g = next(x for x in src["granularity_space"]
             if x["dimension"] == "window")
    assert g["state_evidence"] == ["t-good"]
    assert st["_status"] == "partially_consumed"


def test_repair_never_touches_agent_or_derived_states(tmp_path):
    """人工裁决与 planner 派生态不参与复核。"""
    from runtime.inventory import backfill_from_trials

    run = _run(tmp_path)
    p = _inventory(run, {
        "window": {"state": "explicitly_falsified", "state_source": "agent",
                   "state_evidence": ["t-bad"]},
        "bucketing": {"state": "partial", "state_source": "trial_evidence"},
    })
    bad = _write_trial(run, "t-bad", view="sample", reason="无父代可比分数",
                       vs_parent=None, incomparable=True)
    rep = backfill_from_trials(run, [_T(bad)])
    st = _states(p)
    assert st["window"] == ("explicitly_falsified", "agent")
    assert st["bucketing"] == ("partial", "trial_evidence")
    assert not rep.get("demotions")


# ---------------------------------------------------------------------------
# A4 方向归因:按列集差集,不按整面 group_importance
# ---------------------------------------------------------------------------
def _trial_with_cols(run: Path, tid: str, cols: list[str],
                     hints=None, parents=()) -> dict:
    d = _write_trial(run, tid, verdict="accepted", reason="ok",
                     vs_parent=7e-5, parents=parents)
    d["family_hints"] = hints
    (run / "trials" / tid / "trial.yaml").write_text(
        yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    (run / "trials" / tid / "model").mkdir(parents=True, exist_ok=True)
    (run / "trials" / tid / "model" / "feature_columns.json").write_text(
        json.dumps(cols), encoding="utf-8")
    return d


def test_family_uses_column_diff_not_top_gain_family(tmp_path):
    """加块臂的族 = 新增列前缀;整面 top 族(seqc2)不得抢走归因。

    实测:trial-008 加了 seqh_ 24 列,榜上 group_importance 的 top 族仍是
    seqc2(113 列),旧口径把它的 +7.2e-5 记到 seqc2 名下。
    """
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    base = ["seqc2_a", "seqc2_b", "dm_x", "dm_y"]
    _trial_with_cols(run, "p", base)
    child = _trial_with_cols(
        run, "c", base + [f"seqh_{i}" for i in range(6)], parents=("p",))
    gi = run / "trials" / "c" / "feature" / "group_importance.csv"
    gi.parent.mkdir(parents=True, exist_ok=True)
    gi.write_text("family,n_cols,delta_mean,delta_std,rel_delta\n"
                  "seqc2,113,0.0527,0.0002,0.06\n"
                  "seqh,24,0.0001,0.0,0.001\n", encoding="utf-8")
    assert family_of_trial(child, run) == "feat:seqh"


def test_ablation_arm_attributes_removed_block(tmp_path):
    """摘块臂归因到被摘掉的块,且单独记 ablate:<族>。

    摘块臂的 vs_parent Δ 与"加该块"符号相反:若记成 feat:seqh,摘掉
    seqh_ 后变好(Δ>0)会被台账当成 seqh 的正收益。
    """
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    _trial_with_cols(run, "p", ["seqc2_a", "seqc2_b"] +
                     [f"seqh_{i}" for i in range(5)])
    child = _trial_with_cols(run, "c", ["seqc2_a", "seqc2_b"],
                             hints=["seqh_"], parents=("p",))
    assert family_of_trial(child, run) == "ablate:seqh"


def _gi_dominated_by_seqc2(run: Path, tid: str) -> None:
    gi = run / "trials" / tid / "feature" / "group_importance.csv"
    gi.parent.mkdir(parents=True, exist_ok=True)
    gi.write_text("family,n_cols,delta_mean,delta_std,rel_delta\n"
                  "seqc2,113,0.0527,0.0002,0.06\n"
                  "ratio,1,0.0001,0.0,0.001\n", encoding="utf-8")


def test_hint_beats_group_importance_when_parent_lacks_columns(tmp_path):
    """差集给不出结论时先回 hint,不得落到整面 top 族(旧根因)。"""
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    (run / "trials" / "p").mkdir(parents=True)          # 父代无列清单
    child = _trial_with_cols(run, "c", ["seqc2_a", "ratio_x"],
                             hints=["ratio_"], parents=("p",))
    _gi_dominated_by_seqc2(run, "c")
    assert family_of_trial(child, run) == "feat:ratio"


def test_hint_beats_group_importance_for_single_new_column(tmp_path):
    """只新增 1 列(差集不足 2 列,不下结论)⇒ 用 hint。"""
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    _trial_with_cols(run, "p", ["seqc2_a", "seqc2_b"])
    child = _trial_with_cols(run, "c", ["seqc2_a", "seqc2_b", "ratio_x"],
                             hints=["ratio_"], parents=("p",))
    _gi_dominated_by_seqc2(run, "c")
    assert family_of_trial(child, run) == "feat:ratio"


def test_hints_used_only_when_no_parent_columns(tmp_path):
    """父代缺 feature_columns.json 时退回 hint / gain-top,不猜整面。"""
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    child = _trial_with_cols(run, "c", ["seqc2_a", "seqh_1"],
                             hints=["seqh_"], parents=("p",))
    assert family_of_trial(child, run) == "feat:seqh"


# ---------------------------------------------------------------------------
# A7 配方语义指纹
# ---------------------------------------------------------------------------
def test_semantic_fingerprint_ignores_reorder_and_slot_label():
    """语句顺序、注释、槽位标签不同 ⇒ 同一指纹(实测 rank-1/rank-2)。"""
    from runtime.recipe_fp import recipe_semantic_fingerprint as fp

    a = '''import os
_G2_SLOT = "A"
_G2_N_BLOCK_COLS = 842

def f():
    return 1

X = 2
'''
    b = '''import os
X = 2
_G2_SLOT = "A, v1 reconfirm"

def f():
    return 1

_G2_N_BLOCK_COLS = 842
'''
    assert fp(a) == fp(b)


def test_semantic_fingerprint_ignores_seed_and_location():
    """换 seed 复跑 / 换 run 渲染不改变配方(与经验库 recipe_family 同口径)。"""
    from runtime.recipe_fp import recipe_semantic_fingerprint as fp

    a = "SEED = 1\nRUN_DIR = pathlib.Path('/r/a')\nSKILL_ROOT: str = '/s/a'\nX = 1\n"
    b = "X = 1\nSEED = 2\nRUN_DIR = pathlib.Path('/r/b')\nSKILL_ROOT: str = '/s/b'\n"
    assert fp(a) == fp(b)


def test_warm_start_dedup_refills_slot():
    """去重跳过的 rank 不占 warm 名额,顺延到下一名。"""
    src = (SKILL / "scripts" / "_mvo" / "cmd_startup.py").read_text(encoding="utf-8")
    assert "enumerate(_view.leaderboard, 1)" in src
    assert "_rank - len(_warm_skipped_dup) > _warm_limit" in src


def test_semantic_fingerprint_detects_real_change():
    from runtime.recipe_fp import recipe_semantic_fingerprint as fp

    a = "X = 2\n"
    b = "X = 3\n"
    assert fp(a) != fp(b)


def test_semantic_fingerprint_unparsable():
    from runtime.recipe_fp import recipe_semantic_fingerprint as fp

    assert fp("def broken(:\n") == "unparsable"


# ---------------------------------------------------------------------------
# A5 leaderboard / 报告带原因;A2 报告回落冠军 trial 的证据
# ---------------------------------------------------------------------------
def _lb_run(tmp_path: Path) -> Path:
    run = _run(tmp_path)
    for tid, verdict, reason in (
            ("t1", "confirmed", "ok"),
            ("t2", "rejected", "稳定性或切片/时序检查未通过"),
            ("t3", "rejected", "无父代可比分数")):
        _write_trial(run, tid, verdict=verdict, reason=reason,
                     vs_parent=1e-4 if verdict == "confirmed" else -3e-5)
    (run / "run-state.json").write_text(
        json.dumps({"baseline_id": "t1"}), encoding="utf-8")
    return run


def _build_lb(run: Path, spec: TaskSpec) -> None:
    """报告读 leaderboard.json,测试里必须先落表。"""
    from runtime.leaderboard import Leaderboard
    from runtime.trial import TrialRecord

    trials = [TrialRecord.load(d) for d in sorted((run / "trials").iterdir())]
    Leaderboard(run, spec, Config.load()).build(trials=trials)


def test_leaderboard_carries_verdict_reason(tmp_path):
    from runtime.leaderboard import LEADERBOARD_COLUMNS, Leaderboard

    assert "verdict_reason" in LEADERBOARD_COLUMNS
    run = _lb_run(tmp_path)
    spec = _spec(tmp_path)
    from runtime.trial import TrialRecord

    trials = [TrialRecord.load(run / "trials" / tid)
              for tid in ("t1", "t2", "t3")]
    Leaderboard(run, spec, Config.load()).build(trials=trials)
    rows = json.loads((run / "leaderboard" / "leaderboard.json")
                      .read_text(encoding="utf-8"))["rows"]
    by_id = {r["trial_id"]: r for r in rows}
    assert "切片" in (by_id["t2"]["verdict_reason"] or "")
    assert "无父代可比" in (by_id["t3"]["verdict_reason"] or "")


def test_report_states_rejection_reasons(tmp_path):
    from runtime.report import ReportBuilder

    run = _lb_run(tmp_path)
    spec = _spec(tmp_path)
    _build_lb(run, spec)
    (run / "sample-profile").mkdir(exist_ok=True)
    (run / "sample-profile" / "sample-profile.json").write_text(
        json.dumps({"n_rows": 4_000_000, "n_rows_profiled": 1_000_000,
                    "sampled_from": 3_432_427,
                    "sample_summary": {"total_rows": 739_444,
                                       "label_rate": 0.2509,
                                       "unique_entities": 658_014}}),
        encoding="utf-8")
    md = ReportBuilder(run, spec, Config.load()).build().read_text(
        encoding="utf-8")
    assert "未通过 trial 与原因" in md
    assert "稳定性或切片/时序检查未通过" in md


def test_report_total_rows_uses_full_frame_not_analysis_sample(tmp_path):
    """A1:§2 总样本量必须是全量 4,000,000,不是分析面 739,444。"""
    from runtime.report import ReportBuilder

    run = _lb_run(tmp_path)
    spec = _spec(tmp_path)
    _build_lb(run, spec)
    (run / "sample-profile").mkdir(exist_ok=True)
    (run / "sample-profile" / "sample-profile.json").write_text(
        json.dumps({"n_rows": 4_000_000, "n_rows_profiled": 739_444,
                    "sampled_from": None,
                    "sample_summary": {"total_rows": 739_444,
                                       "label_rate": 0.2509,
                                       "unique_entities": 658_014}}),
        encoding="utf-8")
    md = ReportBuilder(run, spec, Config.load()).build().read_text(
        encoding="utf-8")
    assert "- 总样本量: 4000000" in md
    assert "- 总样本量: 739444" not in md
    assert "分析面(profile 统计口径)" in md


def test_report_stability_falls_back_to_champion_trial(tmp_path):
    """A2:final-model/evaluation.json 无 stability 时回落冠军 trial。"""
    from runtime.report import ReportBuilder

    run = _lb_run(tmp_path)
    spec = _spec(tmp_path)
    _build_lb(run, spec)
    (run / "final-model").mkdir(parents=True, exist_ok=True)
    (run / "final-model" / "evaluation.json").write_text(json.dumps({
        "final_test": {"auc": 0.8}, "best_trial": "t1",
        "search_champion": "t1"}), encoding="utf-8")
    champ_ev = json.loads((run / "trials" / "t1" / "evaluation.json")
                          .read_text(encoding="utf-8"))
    champ_ev["stability"] = {"multi_seed_std": 1.75e-4,
                             "bootstrap_ci": [0.81, 0.814],
                             "passed": True}
    champ_ev["slice_evaluation"] = {"worst_slice": "2026-05-20",
                                    "worst_slice_delta": -1.9e-4,
                                    "passed": False}
    (run / "trials" / "t1" / "evaluation.json").write_text(
        json.dumps(champ_ev), encoding="utf-8")
    md = ReportBuilder(run, spec, Config.load()).build().read_text(
        encoding="utf-8")
    assert "multi_seed_std: 0.000175" in md
    assert "trials/t1/evaluation.json" in md
    assert "最差切片: 2026-05-20" in md


# ---------------------------------------------------------------------------
# A8 planner-context 噪声预检口径
# ---------------------------------------------------------------------------
def test_planner_context_shows_basis_not_ratio(tmp_path):
    from runtime.planner_context import render_planner_context

    ctx = {"spec_summary": {}, "sample_noise_precheck": {
        "recommend_v1": False, "se_ratio": 4.0706,
        "se_sample": 0.0030554, "se_full": 0.0007506,
        "se_sample_paired_est": 0.001066, "mde": 0.001,
        "basis": "paired_estimate",
        "reason": "抽样面 SE=0.001066(paired_estimate)≤ 2×MDE=0.002"}}
    text = render_planner_context(ctx)
    assert "判据用量" in text
    assert "paired_estimate" in text
    assert "不是**判据**" in text or "该比值**不是**判据" in text
    assert "仍有分辨力" in text


def test_planner_context_noisy_but_expensive_is_not_called_resolvable():
    """recommend_v1=false 也可能是"噪声大但全量太贵",判读不得写成"有分辨力"。"""
    from runtime.planner_context import render_planner_context

    ctx = {"spec_summary": {}, "sample_noise_precheck": {
        "recommend_v1": False, "se_ratio": 4.0, "se_sample": 0.004,
        "se_full": 0.001, "se_sample_paired_est": 0.003, "mde": 0.001,
        "basis": "paired_estimate", "reason": "…全量运行 60 min ≥ 阈值"}}
    text = render_planner_context(ctx)
    assert "仍有分辨力" not in text
    assert "**无**分辨力" in text


def test_planner_context_missing_mde_not_rendered_as_zero():
    from runtime.planner_context import render_planner_context

    ctx = {"spec_summary": {}, "sample_noise_precheck": {
        "recommend_v1": False, "se_ratio": 1.2, "se_sample": 0.001,
        "se_full": 0.0008, "mde": None, "basis": "metric_bootstrap_se",
        "reason": "MDE 不可解析,不做建议"}}
    text = render_planner_context(ctx)
    assert "2×MDE=—" in text and "2×MDE=0" not in text
    assert "仍有分辨力" not in text


# ---------------------------------------------------------------------------
# A6 diagnose:跨视图不可比属协议内预期
# ---------------------------------------------------------------------------
def test_diagnose_marks_cross_view_incomparable_as_info(tmp_path):
    from _mvo.cmd_inspect import _diag_trial_anomalies

    run = _run(tmp_path)
    d = _write_trial(run, "t1", view="sample", reason="无父代可比分数",
                     vs_parent=None, incomparable=True)
    from runtime.trial import TrialRecord

    tr = TrialRecord.load(run / "trials" / "t1")
    anomalies = _diag_trial_anomalies(tr, _spec(tmp_path),
                                      {"harness_epoch": 0}, None)
    codes = {a["code"]: a["severity"] for a in anomalies}
    assert codes.get("cross_view_incomparable") == "info"
    assert "parent_scores_unavailable" not in codes


def test_diagnose_keeps_medium_for_same_view_missing_parent(tmp_path):
    from _mvo.cmd_inspect import _diag_trial_anomalies
    from runtime.trial import TrialRecord

    run = _run(tmp_path)
    _write_trial(run, "t1", view="v1", reason="无父代可比分数",
                 vs_parent=None)
    tr = TrialRecord.load(run / "trials" / "t1")
    anomalies = _diag_trial_anomalies(tr, _spec(tmp_path),
                                      {"harness_epoch": 0}, None)
    assert any(a["code"] == "parent_scores_unavailable"
               and a["severity"] == "medium" for a in anomalies)


def test_seed_replicate_gets_its_own_arm(tmp_path):
    """列集与父代完全相同的换 seed 复测 → replicate:seed,不冒充特征族。

    实测:gen-02 的消融双臂(trial-011/012)列集分别等于父代
    trial-008/trial-002,旧台账把它们记成 feat:seqc2 与 feat:seqh,
    两个方向都得到了并不存在的证据。
    """
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    cols = ["seqc2_a", "seqc2_b", "dm_x"]
    _trial_with_cols(run, "p", cols)
    rep = _trial_with_cols(run, "c", cols, parents=("p",))
    assert family_of_trial(rep, run) == "replicate:seed"


def _code(run: Path, tid: str, text: str) -> None:
    d = run / "trials" / tid / "code"
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.py").write_text(text, encoding="utf-8")


def test_seed_replicate_confirmed_by_recipe_fingerprint(tmp_path):
    """列集与配方指纹都同父代 ⇒ 复测,即使 hint 声称了某个族(trial-012)。"""
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    cols = ["seqc2_a", "seqc2_b", "seqh_1", "seqh_2"]
    _trial_with_cols(run, "p", cols)
    _code(run, "p", "# parent\nA = 1\nB = 2\n")
    rep = _trial_with_cols(run, "c", cols, hints=["seqh_"], parents=("p",))
    _code(run, "c", "B = 2\n\nA = 1  # reordered\n")
    assert family_of_trial(rep, run) == "replicate:seed"


def test_same_columns_changed_recipe_is_a_feature_direction(tmp_path):
    """列名不变但配方变了(改分桶边界 / 修特征计算)⇒ 真实特征方向。"""
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    cols = ["seqc2_a", "seqc2_b", "seqh_1", "seqh_2"]
    _trial_with_cols(run, "p", cols)
    _code(run, "p", "BINS = [0, 7, 30]\n")
    child = _trial_with_cols(run, "c", cols, hints=["seqh_"], parents=("p",))
    _code(run, "c", "BINS = [0, 3, 14, 30]\n")
    assert family_of_trial(child, run) == "feat:seqh"


def test_same_columns_without_recipe_trusts_declared_hint(tmp_path):
    """配方指纹不可得时:声明了 hint 就不当复测处理。"""
    from runtime.directions import family_of_trial

    run = _run(tmp_path)
    cols = ["seqc2_a", "seqc2_b", "seqh_1", "seqh_2"]
    _trial_with_cols(run, "p", cols)
    child = _trial_with_cols(run, "c", cols, hints=["seqh_"], parents=("p",))
    assert family_of_trial(child, run) == "feat:seqh"


def test_planner_skips_replicate_arm():
    """replicate:seed 只作噪声标尺,不生成 exploit 提案。"""
    src = (SKILL / "runtime" / "planner.py").read_text(encoding="utf-8")
    assert 'startswith(("replicate:", "ablate:"))' in src


def test_report_section9_reason_when_task_experience_is_stale(tmp_path):
    """§9 走 task-experience 分支时也要带原因。

    实测:收尾写入的 task-experience.yaml 只有 failure_reason(方向性拒绝
    恒为 null),报告 §9 于是只有"trial 名 + 假设",7 条 rejected 全部
    看不出原因。修复前只有 _auto_directions 分支带原因。
    """
    from runtime.report import ReportBuilder

    run = _lb_run(tmp_path)
    spec = _spec(tmp_path)
    _build_lb(run, spec)
    (run / "experience").mkdir(exist_ok=True)
    (run / "experience" / "task-experience.yaml").write_text(yaml.safe_dump({
        "task_id": "t", "successful_directions": [],
        "failed_directions": [
            {"trial_id": "t2", "hypothesis": "切片门臂",
             "failure_reason": None, "verdict": "rejected"},
            {"trial_id": "t3", "hypothesis": "跨视图臂",
             "failure_reason": None, "verdict": "rejected"}],
        "diminishing_return_directions": []}, allow_unicode=True),
        encoding="utf-8")
    md = ReportBuilder(run, spec, Config.load()).build().read_text(
        encoding="utf-8")
    assert "原因:稳定性或切片/时序检查未通过" in md
    assert "原因:无父代可比分数" in md


def test_task_experience_records_verdict_reason(tmp_path):
    """自动压缩 task-experience 时把 evaluation.reason 一并落盘。"""
    from runtime.memory import TaskExperience
    from runtime.trial import TrialRecord

    run = _run(tmp_path)
    spec = _spec(tmp_path)
    _write_trial(run, "t1", verdict="rejected",
                 reason="稳定性或切片/时序检查未通过")
    exp = TaskExperience(run).auto_summarize(
        [TrialRecord.load(run / "trials" / "t1")], spec)
    fd = exp["failed_directions"][0]
    assert fd["reason"] == "稳定性或切片/时序检查未通过"
