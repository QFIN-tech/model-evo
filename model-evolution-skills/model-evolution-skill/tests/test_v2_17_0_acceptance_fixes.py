"""v2.17.0 端到端验收缺陷 D1–D9 的回归测试(WP6)。

验收 run:合成零膨胀重尾金额回归(amt_t7,rmse),每期 n≈2223、各期独立同分布。

- D1 误差归因把目标列当特征(high_loss_profile 第一位 = amt_t7,z=+14.4);
- D2 `--set` 快通道对 baseline 父代不可用(创建成功、run 时 lint 规则 11 拒跑);
- D3 空代与代际 delta 口径(gen-3 零 trial,best_delta 却记 +6.30 并触发停止);
- D4 undelivered_gain 把交付位的同配方 trial(confirm 父代)列为更优候选;
- D5 confirm 不带 --hypothesis 抛 traceback;
- D6 回归脚手架不遵从 --algorithm,lint 规则 10 文案自相矛盾;
- D7 hash 未变时 --verify-against 被静默忽略;
- D8 分期水位告警不考虑噪声,报告整体/分期混写;
- D9 profile 把连续浮点特征报成 id_leakage。

全部离线合成数据,不读真实 run。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "assets"))

import lint_trial as L  # noqa: E402
from runtime.config import Config  # noqa: E402
from runtime.logging_utils import dump_yaml, load_json, load_yaml  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402


def _spec() -> TaskSpec:
    return TaskSpec(data={
        "task_id": "wp6", "task_name": "wp6", "model_type": "regression",
        "target": {"name": "amt_t7"}, "time": {"column": "pday"},
        "entity": {"id_column": "user_no"}, "primary_metric": "rmse",
        "resource_budget": {"max_generations": 8, "max_trials": 50},
        "data": {},
    })


# ===========================================================================
# D1 误差归因只用特征面
# ===========================================================================

def test_d1_error_analysis_excludes_label_plane_columns():
    from runtime.error_analysis import (build_error_analysis,
                                        non_feature_columns)

    rng = np.random.default_rng(0)
    n = 2000
    y = np.where(rng.random(n) < 0.8, 0.0, rng.gamma(1.5, 2000.0, n))
    f1 = rng.normal(size=n) + (y > 0) * 0.5
    X = pd.DataFrame({
        "f1": f1, "f2": rng.normal(size=n),
        # 标签面 join 进评估帧的列(runner._evaluate_trial 口径)
        "amt_t7": y, "pday": rng.integers(0, 4, n).astype(float),
        "user_no": np.arange(n, dtype=float), "__mvo_sample_key": np.arange(n),
        "amt_t7_lab": y,          # join 冲突后缀列
    })
    score = np.full(n, float(y.mean()))
    excl = non_feature_columns(_spec(), ["amt_t7", "pday", "user_no",
                                         "__mvo_sample_key"])
    assert {"amt_t7", "pday", "user_no", "amt_t7_lab"} <= excl
    ea = build_error_analysis(y, score, None, X, model_type="regression",
                              exclude_columns=excl)
    feats = [r["feature"] for r in ea["high_loss_profile"]]
    assert feats and set(feats) <= {"f1", "f2"}, feats
    # 不传排除集时(旧调用方)目标会登顶 —— 复现原缺陷形态
    ea_old = build_error_analysis(y, score, None, X, model_type="regression")
    assert ea_old["high_loss_profile"][0]["feature"] in ("amt_t7",
                                                         "amt_t7_lab")


def test_d1_runner_passes_label_columns_to_error_analysis():
    src = (SKILL / "runtime" / "runner.py").read_text(encoding="utf-8")
    i = src.index("ea = build_error_analysis(")
    seg = src[i:i + 600]
    assert "exclude_columns=non_feature_columns(" in seg
    assert "_va_lab.columns" in seg


# ===========================================================================
# D2 --set 快通道对 baseline 父代
# ===========================================================================

def _baseline_code(tmp: Path, algorithm: str) -> str:
    from runtime.baseline import BaselineRunner

    run = tmp / f"bl-{algorithm}"
    (run / "trials" / "t").mkdir(parents=True, exist_ok=True)
    cfg = Config.load(skill_config=False)
    p = BaselineRunner(_spec(), cfg, run).write_baseline_code(
        run / "trials" / "t", algorithm, ["f1", "f2"])
    return Path(p).read_text(encoding="utf-8")


class _FakeTrial:
    def __init__(self, d: Path):
        self.trial_dir = str(d)
        self.trial_id = d.name
        self.data = {"execution": {}}

    def flush(self):
        pass


def _child(run: Path, tid: str, code: str, hp: dict) -> Path:
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "code" / "train.py").write_text(code, encoding="utf-8")
    dump_yaml(d / "config" / "trial.yaml", {"hyperparameters": hp})
    return d


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_d2_baseline_parent_gets_hp_consumer_injected(tmp_path):
    pytest.importorskip("xgboost")
    from _mvo import render as R
    from _mvo.templates import BASELINE_HP_BEGIN

    code = _baseline_code(tmp_path, "xgboost")
    st = R._baseline_lineage_status(code)
    assert st == {"baseline": True, "injectable": True,
                  "algorithm": "xgboost"}
    run = tmp_path / "run"
    d = _child(run, "trial-006-g03-exploit-xgboost-1", code, {"max_depth": 2})
    # 注入前:lint 规则 11 拒(复现 run6.out)
    assert L.check_hyperparameters_consumption(run, d, code)
    notes = R._ensure_hyperparameters_consumed(run, _FakeTrial(d), "trial-001",
                                               "exploit")
    new = (d / "code" / "train.py").read_text(encoding="utf-8")
    assert notes and BASELINE_HP_BEGIN in new
    assert L.check_hyperparameters_consumption(run, d, new) == []
    # 幂等:再次调用不重复注入
    assert R._ensure_hyperparameters_consumed(run, _FakeTrial(d), "trial-001",
                                              "exploit") == []
    assert new.count(BASELINE_HP_BEGIN) == 1
    # 消费点真实生效:主轮与 seed 变体轮都经全局名 build_estimator
    mod = _load_module(d / "code" / "train.py", "wp6_d2_child")
    m = mod.build_estimator(1, 10)
    assert m.get_params()["max_depth"] == 2
    assert m.get_params()["n_estimators"] == mod.N_ESTIMATORS_CAP
    # 父代冻结代码未被改动(只改子代副本)
    assert BASELINE_HP_BEGIN not in code


def test_d2_segment_falls_back_without_load_config(tmp_path):
    """父代模板无 load_config 时注入段直接读 TRIAL_DIR/config/trial.yaml。"""
    pytest.importorskip("xgboost")
    from _mvo import render as R

    code = _baseline_code(tmp_path, "lightgbm")
    code = code.replace("def load_config() -> dict:", "def _unused_load() -> dict:")
    run = tmp_path / "run"
    d = _child(run, "trial-011-g03-exploit-lightgbm-1", code,
               {"num_leaves": 7})
    R._ensure_hyperparameters_consumed(run, _FakeTrial(d), "trial-001",
                                       "exploit")
    mod = _load_module(d / "code" / "train.py", "wp6_d2_nolc")
    assert "_MVO_BASE_BUILD_ESTIMATOR" in vars(mod)
    assert mod._mvo_cfg_hp() == {"num_leaves": 7}


def test_d2_infeasible_parent_rejected_at_creation(tmp_path):
    from _mvo import render as R

    run = tmp_path / "run"
    # mlp baseline:包装器不支持 set_params → 不可注入
    mlp = _baseline_code(tmp_path, "mlp")
    assert R._baseline_lineage_status(mlp)["injectable"] is False
    d = _child(run, "trial-007-g03-exploit-mlp-1", mlp, {"lr": 0.01})
    with pytest.raises(SystemExit) as ei:
        R._ensure_hyperparameters_consumed(run, _FakeTrial(d), "trial-001",
                                           "exploit")
    msg = str(ei.value)
    assert "创建期拒绝" in msg and "--allow-code-edit" in msg
    # 非 baseline/hurdle 父代同样拒绝;--allow-code-edit 时降级为告警+留痕
    d2 = _child(run, "trial-008-g03-exploit-x-1", "SEED = 1\n", {"lr": 0.03})
    with pytest.raises(SystemExit):
        R._ensure_hyperparameters_consumed(run, _FakeTrial(d2), "x", "exploit")
    t2 = _FakeTrial(d2)
    assert R._ensure_hyperparameters_consumed(
        run, t2, "x", "exploit", allow_code_edit=True) == []
    assert t2.data["execution"]["hyperparameters_unconsumed"] == ["lr"]


def test_d2_prefixed_keys_rejected_for_single_estimator_baseline(tmp_path):
    from _mvo import render as R

    code = _baseline_code(tmp_path, "lightgbm")
    run = tmp_path / "run"
    d = _child(run, "trial-009-g03-exploit-lightgbm-1", code,
               {"reg__max_depth": 3})
    with pytest.raises(SystemExit) as ei:
        R._ensure_hyperparameters_consumed(run, _FakeTrial(d), "trial-001",
                                           "exploit")
    assert "max_depth=3" in str(ei.value)


def test_d2_new_trial_rollback_on_rejection_is_wired():
    """创建期 SystemExit → cmd_new_trial 回滚(不留 trial)后再抛出(exit 1)。"""
    src = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    i = src.index("_hp_notes = _ensure_hyperparameters_consumed(")
    seg = src[i:i + 1200]
    assert "except SystemExit:" in seg and "_rollback_trial(trial)" in seg


def test_d2_diagnose_reports_unconsumed_hyperparameters(tmp_path):
    from _mvo.cmd_inspect import _diag_hp_unconsumed, _diag_trial_anomalies

    run = tmp_path / "run"
    d = _child(run, "trial-010-g03-exploit-xgboost-1", "SEED = 1\n",
               {"max_depth": 2})
    msg = _diag_hp_unconsumed(d)
    assert msg and "规则 11" in msg
    rec = TrialRecord(data={"trial_id": d.name, "status": "PROPOSED",
                            "generation": 3, "search_action": "exploit"},
                      trial_dir=d)
    found = _diag_trial_anomalies(rec, _spec(), {}, None)
    assert any(f["code"] == "hyperparameters_unconsumed"
               and f["severity"] == "high" for f in found), found
    # 有消费点后不再报
    (d / "code" / "train.py").write_text(
        'HP = {}.get("hyperparameters")\n', encoding="utf-8")
    assert _diag_hp_unconsumed(d) is None


# ===========================================================================
# D3 空代与代际 delta 口径
# ===========================================================================

def _d3_run(tmp: Path, *, skill_version: str | None,
            base_metric: float = 100.0) -> Path:
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    rs = {"baseline_id": "t-base"}
    if skill_version:
        rs["skill_version"] = skill_version
    (run / "run-state.json").write_text(json.dumps(rs), encoding="utf-8")
    _d3_trial(run, "t-base", 0, base_metric, "accepted",
              extra={"root_baseline_only": True})
    return run


def _d3_trial(run: Path, tid: str, gen: int, metric: float, verdict: str, *,
              action: str = "explore", parents=None, extra=None):
    d = run / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    ev = {"primary_metric": {"name": "rmse", "value": metric},
          "verdict": verdict, "integrity": {"valid": True}, "delta": {}}
    if verdict == "confirmed":
        ev["fresh_confirmation"] = {"passed": True}
    ev.update(extra or {})
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "xgboost", "generation": gen,
        "status": "EVALUATED", "search_action": action,
        "parent_trial_ids": parents or [], "data_view": "v1",
        "evaluation": ev}, trial_dir=d)
    rec.flush()
    return rec


def _engine(run: Path):
    from runtime.evolution import EvolutionEngine

    return EvolutionEngine(run, _spec(), Config.load(skill_config=False))


def test_d3_champion_delta_and_empty_generation(tmp_path):
    run = _d3_run(tmp_path, skill_version="2.17.0")
    # gen-1:冠军 100 → 90(confirm 形态的已确认 trial,避开代级校正干扰)
    _d3_trial(run, "t-g1", 1, 90.0, "confirmed", action="confirm",
              parents=["t-base"])
    _d3_trial(run, "t-g1b", 1, 85.0, "rejected", parents=["t-base"])
    out1 = _engine(run).finalize_generation(1, ["t-g1", "t-g1b"],
                                            baseline_metric=100.0)
    assert out1["generation_delta"] == pytest.approx(10.0)
    assert out1["generation_delta_protocol"] == "champion_vs_champion"
    # gen-2:冠军没换(上一代最优 85 是 rejected trial)→ 恰为 0
    _d3_trial(run, "t-g2", 2, 95.0, "rejected", parents=["t-g1"])
    out2 = _engine(run).finalize_generation(2, ["t-g2"],
                                            baseline_metric=100.0)
    assert out2["generation_delta"] == 0.0
    # gen-3:空代 → None、告警、不触发停止(旧实现:≥3 代即单口径护栏停止)
    out3 = _engine(run).finalize_generation(3, [], baseline_metric=100.0)
    assert out3["generation_delta"] is None
    assert out3["empty_generation"] is True
    assert "空代" in out3["empty_generation_warning"]
    assert out3["stop_report"] is None
    st = load_json(run / "evolution-state.json")
    assert st["generation_best_deltas"] == [pytest.approx(10.0), 0.0, None]
    assert st["empty_generations"] == [3]
    hist = st["champion_metric_history"]
    assert [h["generation"] for h in hist] == [1, 2, 3]
    assert hist[0]["champion_before_source"] == "baseline"
    assert hist[1]["champion_before"] == 90.0
    assert hist[2]["champion_metric"] == 90.0 and hist[2]["delta"] is None
    gy = load_yaml(run / "generations" / "gen-03" / "generation.yaml")
    assert gy["empty_generation"] is True
    assert gy["summary"]["best_delta_vs_previous_generation"] is None
    # 幂等:重复 finalize 空代不追加、不改口径
    out3b = _engine(run).finalize_generation(3, [], baseline_metric=100.0)
    assert out3b["generation_delta"] is None
    assert load_json(run / "evolution-state.json")[
        "generation_best_deltas"] == [pytest.approx(10.0), 0.0, None]
    # gen-4 非空:空代不计入"≥3 代"——此时非空代恰 3 个,单口径护栏才生效
    _d3_trial(run, "t-g4", 4, 97.0, "rejected", parents=["t-g1"])
    out4 = _engine(run).finalize_generation(4, ["t-g4"],
                                            baseline_metric=100.0)
    assert out4["generation_delta"] == 0.0
    assert out4["stop_report"] is not None
    assert "single_data_view_insufficient" in out4["stop_report"][
        "stop_reason"]
    assert out4["stop_report"]["recent_generations"]["best_delta"][-2] is None


def test_d3_legacy_run_keeps_old_delta_protocol(tmp_path):
    run = _d3_run(tmp_path, skill_version=None)
    _d3_trial(run, "t-g1", 1, 90.0, "confirmed", action="confirm",
              parents=["t-base"])
    _d3_trial(run, "t-g1b", 1, 85.0, "rejected", parents=["t-base"])
    _engine(run).finalize_generation(1, ["t-g1", "t-g1b"],
                                     baseline_metric=100.0)
    _d3_trial(run, "t-g2", 2, 95.0, "rejected", parents=["t-g1"])
    out2 = _engine(run).finalize_generation(2, ["t-g2"],
                                            baseline_metric=100.0)
    # 旧口径:本代冠军 90 − 上一代最优 85(rmse 越小越好)= −5
    assert out2["generation_delta"] == pytest.approx(-5.0)
    assert out2["generation_delta_protocol"] == \
        "legacy_best_vs_prev_generation_best"
    assert "champion_metric_history" not in load_json(
        run / "evolution-state.json")
    # 空代语义对旧 run 同样生效(没有测量就没有 delta)
    out3 = _engine(run).finalize_generation(3, [], baseline_metric=100.0)
    assert out3["generation_delta"] is None and out3["empty_generation"]


def test_d3_stop_condition_none_and_empty_generation():
    from runtime.generation import StopCondition

    sc = StopCondition(_spec(), Config.load(skill_config=False))
    base = dict(n_trials_started=5, n_generations=4, wall_time_hours=0.1,
                trials_accepted=0, best_metric=90.0, baseline_metric=100.0,
                noise=1.0, unexplored_high_potential=[],
                data_views_tried=["v1"])
    # None 不计入"≥3 代":两个非空代 + 一个空代 → 单口径护栏不触发
    assert sc.check(generation_best_deltas=[10.0, 0.0, None], **base) is None
    assert sc.check(generation_best_deltas=[10.0, 0.0, 0.0], **base) \
        is not None
    # 空代:只保留硬预算原因,其余压制并留痕
    rep = sc.check(generation_best_deltas=[10.0, 0.0, 0.0],
                   empty_generation=True, **base)
    assert rep is None
    b2 = dict(base, n_trials_started=50)
    rep2 = sc.check(generation_best_deltas=[10.0, 0.0, 0.0],
                    empty_generation=True, **b2)
    assert rep2["stop_reason"] == "budget_max_trials"
    assert "single_data_view_insufficient" in rep2[
        "suppressed_by_empty_generation"]
    # ROI:None 代与 compute 小时同步剔除,不错位
    rep3 = sc.check(generation_best_deltas=[10.0, None, 0.1, 0.1],
                    generation_compute_hours=[1.0, 0.0, 1.0, 1.0],
                    **dict(base, data_views_tried=["v1", "v2"]))
    assert rep3 is not None and "roi_diminishing_return" in rep3[
        "stop_reason"]


def test_d3_reason_detail_matches_implementation():
    from runtime.generation import REASON_DETAIL

    roi = REASON_DETAIL["roi_diminishing_return"]
    assert "本代开始前的冠军" in roi and "冠军没换时恰为 0" in roi
    assert "空代" in roi and "None" in roi
    assert "上一代最优非 screening" in roi and "旧 run" in roi
    assert "冠军没换(候选没晋级)时为 0 或负" not in roi
    assert "非空代" in REASON_DETAIL["convergence_no_accepted"]
    assert "非空代" in REASON_DETAIL["single_data_view_insufficient"]


# ===========================================================================
# D4 undelivered_gain 排除交付位同配方
# ===========================================================================

def _d4_trials(tmp: Path):
    run = _d3_run(tmp, skill_version="2.17.0", base_metric=2129.0)
    _d3_trial(run, "t-003", 1, 2085.35, "accepted", parents=["t-base"])
    _d3_trial(run, "t-004", 2, 2091.65, "accepted", parents=["t-003"],
              action="exploit")
    _d3_trial(run, "t-005", 2, 2097.70, "confirmed", parents=["t-003"],
              action="confirm")
    # confirm 的 confirm(同配方链)与另一配方
    _d3_trial(run, "t-006", 3, 2090.00, "confirmed", parents=["t-005"],
              action="confirm")
    from runtime.trial import TrialFactory

    return run, TrialFactory(run).all_trials()


def test_d4_undelivered_gain_excludes_same_recipe(tmp_path):
    from runtime.evolution import compute_undelivered_gain

    run, trials = _d4_trials(tmp_path)
    ug = compute_undelivered_gain(
        trials, anchor_id="t-005", anchor_source="delivery_trial",
        spec=_spec(), config=Config.load(skill_config=False), run_dir=run,
        reference="anchor")
    assert ug["anchor_recipe"] == "t-003"
    assert ug["excluded_same_recipe"] == ["t-003", "t-006"]
    assert ug["best_trial"] == "t-004"          # 唯一的不同配方更优候选
    assert all(c["trial_id"] not in ("t-003", "t-006")
               for c in ug["top_candidates"])
    assert ug["gain"] == pytest.approx(2097.70 - 2091.65)


def test_d4_stale_stop_report_corrected_on_read(tmp_path):
    from runtime.evolution import exclude_same_recipe_from_stop
    from runtime.report import _latest_stop_report, _lim_undelivered

    run, trials = _d4_trials(tmp_path)
    stale = {
        "stop_reason": "single_data_view_insufficient",
        "remaining_high_potential_directions": [
            "confirm_or_consolidate:t-003", "x"],
        "undelivered_gain": {
            "anchor_id": "t-005", "reference": "anchor", "mde": 1.7,
            "best_trial": "t-003", "delta": -12.35, "gain": 12.35,
            "gain_in_mde": 7.26, "meets_mde": True,
            "source": "metric_difference",
            "top_candidates": [
                {"trial_id": "t-003", "delta": -12.35, "gain": 12.35},
                {"trial_id": "t-004", "delta": -6.05, "gain": 6.05}]}}
    fixed = exclude_same_recipe_from_stop(stale, trials)
    ug = fixed["undelivered_gain"]
    assert ug["best_trial"] == "t-004" and ug["excluded_same_recipe"] == [
        "t-003"]
    assert ug["gain_in_mde"] == pytest.approx(6.05 / 1.7)
    assert fixed["remaining_high_potential_directions"] == ["x"]
    assert stale["undelivered_gain"]["best_trial"] == "t-003", "不改入参"
    # 报告 §14/§15 读取同一修正口径
    got = _latest_stop_report(run, {"stopped": stale})
    assert got["undelivered_gain"]["best_trial"] == "t-004"
    lim = " ".join(_lim_undelivered(got, {}, _spec(), "t-005"))
    assert "t-003" not in lim and "t-004" in lim
    # 只有同配方候选时:不再报"未交付更优候选"
    only_same = json.loads(json.dumps(stale))
    only_same["undelivered_gain"]["top_candidates"] = only_same[
        "undelivered_gain"]["top_candidates"][:1]
    got2 = exclude_same_recipe_from_stop(only_same, trials)
    assert got2["undelivered_gain"]["best_trial"] is None
    assert got2["undelivered_gain"]["meets_mde"] is False
    assert _lim_undelivered(got2, {}, _spec(), "t-005") == []


def test_d4_report_section15_uses_corrected_stop():
    src = (SKILL / "runtime" / "report.py").read_text(encoding="utf-8")
    i = src.index('"## 15. stop reason"')
    assert "_correct_stop_same_recipe(" in src[i:i + 400]


# ===========================================================================
# D5 confirm 缺省 --hypothesis
# ===========================================================================

def test_d5_auto_hypothesis_for_confirm(tmp_path):
    from _mvo.cmd_trial import _auto_hypothesis
    from runtime.trial import TrialFactory

    h = _auto_hypothesis("confirm", "trial-003-g01-explore-xgboost-2059")
    assert h["statement"].startswith(
        "确认 trial-003-g01-explore-xgboost-2059 配方在新 seed 与 shadow 上"
        "非劣且保真")
    assert h["falsification"] and h["rationale"]
    for act in ("explore", "exploit", "crossover", "hpo", "data_view"):
        assert _auto_hypothesis(act, "p") is None
    assert _auto_hypothesis("confirm", None) is None
    # 自动假设能通过 TrialFactory 的占位/长度校验
    f = TrialFactory(tmp_path / "run")
    t = f.create(algorithm="xgboost", hypothesis=h,
                 execution={"changed_modules": []}, search_action="confirm",
                 parent_trial_ids=["p"], generation=2)
    assert Path(t.trial_dir).exists()
    with pytest.raises(ValueError):
        f.create(algorithm="xgboost",
                 hypothesis={"statement": "(待填写:可证伪假设陈述)"},
                 execution={}, search_action="explore", generation=2)


def test_d5_cli_catches_create_valueerror():
    src = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    i = src.index("trial = factory.create(")
    seg = src[i - 40:i + 1100]
    assert "except ValueError as exc:" in seg and "return 1" in seg
    j = src.index("_auto = _auto_hypothesis(")
    assert "return 1" in src[j:j + 900]


# ===========================================================================
# D6 回归脚手架遵从 --algorithm;规则 10 文案一致
# ===========================================================================

def _render_scaffold(model_type: str) -> str:
    from _mvo.templates import TRIAL_BODY
    from runtime.templates import TRIAL_TEMPLATE

    return TRIAL_TEMPLATE.format(
        trial_id="trial-002-g01-explore-xgboost-1", hypothesis="h" * 8,
        actions="a", run_dir="/tmp/x", target="amt_t7", model_type=model_type,
        treatment="", seed=20260814, skill_root=str(SKILL), body=TRIAL_BODY)


def test_d6_regression_scaffold_uses_algorithm_and_passes_rule10():
    from _mvo.render import _inject_hpo_algo_at
    from runtime.templates import narrow_model_type_dispatch

    code = _render_scaffold("regression")
    # 未注入算法:脚手架标记仍在 → 规则 10 拒(与修正后的文案一致)
    errs = L.check_scaffold_fit_and_score(code)
    assert errs and "DummyRegressor 占位" in errs[0]
    assert "regression/uplift 分支已有可执行实现" not in errs[0]
    out = _inject_hpo_algo_at(code, "xgboost")
    out, _ = narrow_model_type_dispatch(out, "regression")
    assert "_HPO_ALGO = 'xgboost'" in out
    assert L.check_scaffold_fit_and_score(out) == []
    # 算法化分支排在 DummyRegressor 分支之前(真实回归器接管)
    i_alg = out.index('elif globals().get("_HPO_ALGO"):')
    i_dummy = out.index("DummyRegressor(strategy=")
    assert i_alg < i_dummy


def test_d6_cmd_new_trial_injects_algorithm_for_regression_scaffold():
    src = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    i = src.index("v2.17.0(D6):回归脚手架遵从 --algorithm")
    seg = src[i:i + 2400]
    assert 'str(spec.model_type) == "regression"' in seg
    assert "_inject_hpo_algo_at(code, _scaffold_algo)" in seg


def test_d6_rule10_doc_and_template_text_consistent():
    from _mvo.templates import TRIAL_BODY

    doc = L.check_scaffold_fit_and_score.__doc__
    assert "回归/uplift 分支是可执行默认实现,不拦" not in doc
    assert "对回归/uplift 脚手架同样" in doc
    assert "回归/uplift 分支已给可执行实现" not in TRIAL_BODY
    assert "回归可执行默认实现" not in TRIAL_BODY


# ===========================================================================
# D7 hash 未变时 --verify-against 仍执行
# ===========================================================================

def _d7_run(tmp: Path) -> Path:
    from runtime.harness import protocol_stamp

    run = tmp / "run"
    run.mkdir(parents=True)
    stamp = protocol_stamp(SKILL / "runtime")
    st = dict(stamp, harness_change_log=[{"at": "x"}], harness_epoch=0)
    (run / "run-state.json").write_text(json.dumps(st), encoding="utf-8")
    return run


def _d7_args(run: Path, verify=None):
    return types.SimpleNamespace(
        run=str(run), spec=None, reason="wp6", verify_against=verify,
        allow_measurement_impact=False, measurement_no_score_impact=False)


@pytest.mark.parametrize("consistent,rc", [(True, 0), (False, 1)])
def test_d7_verify_against_runs_when_hash_unchanged(tmp_path, monkeypatch,
                                                    capsys, consistent, rc):
    import runtime.harness as H
    from _mvo import cmd_inspect as CI

    run = _d7_run(tmp_path)
    calls = []

    def _fake(run_dir, spec, cfg, tid):
        calls.append(tid)
        return {"trial": tid, "consistent": consistent,
                "compared_fields": ["primary_metric"],
                "diffs": [] if consistent else [
                    {"field": "primary_metric", "stored": 1.0,
                     "recomputed": 2.0}]}

    monkeypatch.setattr(H, "verify_trial_evaluation", _fake)
    monkeypatch.setattr(CI, "_load_spec", lambda args: None)
    assert CI.cmd_acknowledge_harness_change(_d7_args(run, "t-a,t-b")) == rc
    out = capsys.readouterr().out
    assert calls == ["t-a", "t-b"]
    assert "verify-against" in out and "不登记变更" in out
    assert ("一致" if consistent else "不一致") in out
    log = json.loads((run / "run-state.json").read_text())[
        "harness_change_log"]
    assert log == [{"at": "x"}], "hash 未变不登记变更"
    # 不带 --verify-against:行为不变
    assert CI.cmd_acknowledge_harness_change(_d7_args(run)) == 0
    assert "无需登记" in capsys.readouterr().out


# ===========================================================================
# D8 水位告警的噪声判据
# ===========================================================================

def _heavy_tail(rng, n):
    return np.where(rng.random(n) < 0.8, 0.0, rng.lognormal(6.5, 1.2, n))


def test_d8_level_stats_se_and_noise_threshold():
    from runtime.regression_diagnostics import (LEVEL_RATIO_WARN,
                                                LEVEL_RATIO_Z, level_ratio_se,
                                                level_stats)

    assert LEVEL_RATIO_Z == 2.0
    rng = np.random.default_rng(1)
    n = 2223
    y = _heavy_tail(rng, n)
    pred = np.full(n, float(y.mean()) * 1.15)       # +15%,但期均值噪声更大
    st = level_stats(y, pred)
    se = st["level_ratio_se"]
    assert se == pytest.approx(level_ratio_se(y, pred))
    assert se > 0.075, se                           # 重尾:SE 远大于 5%
    assert abs(st["level_ratio"] - 1) > LEVEL_RATIO_WARN
    assert st["level_ratio_threshold"] == pytest.approx(
        max(LEVEL_RATIO_WARN, 2.0 * se))
    assert st["level_shift_warning"] is False and st["level_within_noise"]
    # 低噪声目标 + 同样 +15% → 告警
    y2 = rng.normal(100.0, 5.0, n)
    st2 = level_stats(y2, y2 * 1.15)
    assert st2["level_shift_warning"] is True
    assert st2["level_ratio_threshold"] == LEVEL_RATIO_WARN


def test_d8_level_by_period_iid_heavy_tail_no_false_alarm():
    from runtime.regression_diagnostics import (level_by_period,
                                                regression_diagnostics)

    rng = np.random.default_rng(7)
    n = 2223
    y = np.concatenate([_heavy_tail(rng, n) for _ in range(4)])
    per = np.repeat(["2026-07-01", "2026-07-08", "2026-07-15",
                     "2026-07-22"], n)
    pred = np.full(y.size, float(y.mean()))         # 各期同水位(iid)
    rows = level_by_period(y, pred, per)
    assert len(rows) == 4 and all("level_ratio_se" in r for r in rows)
    ratios = [r["level_ratio"] for r in rows]
    assert max(abs(r - 1) for r in ratios) < max(2.2 * r["level_ratio_se"]
                                                 for r in rows)
    assert not any(r["level_shift_warning"] for r in rows), ratios
    # 真实的期水位下移(×0.5)仍能检出,且整体/分期分开标记
    y_shift = y.copy()
    y_shift[3 * n:] *= 0.5
    d = regression_diagnostics(y_shift, pred, periods=per)
    assert d["period_level_shift"] is True
    assert d["period_level_shift_periods"] == ["2026-07-22"]
    assert "overall_level_shift" in d


def test_d8_delivery_diag_and_report_separate_overall_period():
    from _mvo.cmd_deliver import (_delivery_regression_diagnostics,
                                  _level_hint_lines)
    from runtime.report import _lim_level_shift

    rng = np.random.default_rng(3)
    y = rng.normal(100.0, 5.0, 800)
    per = np.repeat(["p1", "p2", "p3", "p4"], 200)
    pred = y.copy()
    pred[600:] *= 1.6                               # 只有 p4 偏离
    out = _delivery_regression_diagnostics({"final_test": {
        "y": y, "periods": per, "models": {"delivery": pred}}})
    assert out["level_ratio_z"] == 2.0
    assert out["period_level_shift_splits"] == ["final_test"]
    assert out["overall_level_shift_splits"] == ["final_test"]  # 整体 +15%
    hint = _level_hint_lines(out)
    assert len(hint) == 1 and "整体水位偏离" in hint[0] \
        and "分期水位偏离" in hint[0] and "p4=1.600" in hint[0]
    # 只有分期偏离、整体在阈内
    pred2 = y.copy()
    pred2[600:] *= 1.3
    pred2[:600] *= 0.97
    out2 = _delivery_regression_diagnostics({"final_test": {
        "y": y, "periods": per, "models": {"delivery": pred2}}})
    assert out2["overall_level_shift_splits"] == []
    assert out2["period_level_shift_splits"] == ["final_test"]
    h2 = _level_hint_lines(out2)
    assert "整体水位偏离" not in h2[0] and "整体水位在阈内" in h2[0]
    lims = _lim_level_shift({"regression_diagnostics": out2}, {})
    assert len(lims) == 1 and lims[0].startswith("交付期分期水位偏离(final_test)")
    lims1 = _lim_level_shift({"regression_diagnostics": out}, {})
    assert [x.split(":")[0] for x in lims1] == [
        "交付期水位偏差(final_test)", "交付期分期水位偏离(final_test)"]


def test_d8_report_section12_separates_overall_and_period():
    src = (SKILL / "runtime" / "report.py").read_text(encoding="utf-8")
    i = src.index("v2.17.0(D8):整体与分期告警分开表述")
    seg = src[i:i + 3500]
    assert "· 整体" in seg and "· 分期" in seg
    assert "period_level_shift\")" not in seg     # 不再把分期并进整体告警


# ===========================================================================
# D9 id_leakage 只报整数/字符串型标识符列
# ===========================================================================

def test_d9_identifier_heuristic():
    from runtime.profile import SampleProfile as P

    rng = np.random.default_rng(0)
    n = 3000
    assert P._looks_like_identifier("f00", pd.Series(rng.normal(size=n))) \
        is False
    assert P._looks_like_identifier(
        "amt", pd.Series(rng.gamma(2.0, 100.0, n))) is False
    assert P._looks_like_identifier(
        "user_no", pd.Series(np.arange(n))) is True
    assert P._looks_like_identifier(
        "x", pd.Series(np.arange(10 ** 7, 10 ** 7 + n))) is True
    assert P._looks_like_identifier(
        "uid", pd.Series(np.arange(n, dtype=float))) is True  # NaN 致 float
    assert P._looks_like_identifier(
        "k", pd.Series([f"{i:016x}" for i in range(n)])) is True
    assert P._looks_like_identifier(
        "price_s", pd.Series([f"{v:.3f}" for v in rng.random(n) * 9])) \
        is False
    seq = pd.Series([",".join(map(str, rng.integers(0, 5, rng.integers(
        1, 30)))) for _ in range(n)])
    assert P._looks_like_identifier("seq_evt", seq) is False
    assert P._looks_like_identifier(
        "memo", pd.Series([f"hello world {i}" for i in range(n)])) is False


def test_d9_quality_record_and_leakage_use_id_like():
    from runtime.profile import SampleProfile as P

    rng = np.random.default_rng(0)
    n = 3000
    prof = object.__new__(P)
    prof.df = pd.DataFrame({"f00": rng.normal(size=n),
                            "user_id": np.arange(n)})
    r_float = prof._quality_one_col("f00")
    r_id = prof._quality_one_col("user_id")
    assert r_float["high_cardinality"] is True and r_float["id_like"] is False
    assert r_id["high_cardinality"] is True and r_id["id_like"] is True
    src = (SKILL / "runtime" / "profile.py").read_text(encoding="utf-8")
    i = src.index("# 3) ID-like 特征")
    assert 'feat_summary.get("id_like_features")' in src[i:i + 800]
