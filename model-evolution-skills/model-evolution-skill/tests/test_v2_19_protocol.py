"""v2.19.0 回归测试:P0-C 裁决口径小修 + 停止信号(新 run 口径)。

事故/统计来源(docs/MODELEVO_GOAL_REFLECTION_AND_ROADMAP_20261101.md §4.4):
- 锚点在首次 confirm 前一直是弱 baseline:ds-flash trial-009 Δparent=0.0000、
  p=0.577,vs_anchor=+0.0114、p=1 → accepted(BH 形同虚设);
- 切片门乘整体 SE,零效应误拒率 29~43%;
- 配对 SE 不含 seed 方差;
- 冷启动 run gen-04 起 15 个 trial |Δ|≤5e-4 仍不停。

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

from runtime.ceiling import (ceiling_summary, gap_to_ref,  # noqa: E402
                             lift_capture, load_ceiling, write_ceiling)
from runtime.config import Config, _deep_merge  # noqa: E402
from runtime.evaluator import Evaluator  # noqa: E402
from runtime.evolution import (_anchor_from_state, anchor_mode,  # noqa: E402
                               resolve_promotion_anchor)
from runtime.generation import StopCondition  # noqa: E402
from runtime.protocol_gates import (V2_17_0_RUN_CONFIG,  # noqa: E402
                                    V2_19_0_RUN_CONFIG)
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402


def _spec(metric: str = "auc", model_type: str = "binary_classification",
          max_generations: int = 8) -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v2190", "task_name": "v2190", "model_type": model_type,
        "target": {"name": "y"}, "time": {"column": "dt"},
        "entity": {"id_column": "uid"}, "primary_metric": metric,
        "resource_budget": {"max_generations": max_generations},
        "data": {},
    })


def _cfg(**evaluator) -> Config:
    cfg = Config.load(skill_config=False)
    cfg.raw["evaluator"].update(evaluator)
    return cfg


def _new_run_cfg() -> Config:
    cfg = Config.load(skill_config=False)
    merged = _deep_merge(_deep_merge(cfg.raw, V2_17_0_RUN_CONFIG),
                         V2_19_0_RUN_CONFIG)
    for sec, vals in merged.items():
        if isinstance(vals, dict) and isinstance(cfg.raw.get(sec), dict):
            cfg.raw[sec].update(vals)
        else:
            cfg.raw[sec] = vals
    return cfg


def _trial(run: Path, tid: str, gen: int, *, metric: float,
           verdict: str = "accepted", action: str = "explore",
           parents: list[str] | None = None, view: str = "v1",
           extra: dict | None = None) -> TrialRecord:
    d = run / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    ev = {"primary_metric": {"name": "auc", "value": metric},
          "verdict": verdict, "integrity": {"valid": True},
          "stability": {"passed": True}, "delta": {}}
    rec = TrialRecord(data={
        "trial_id": tid, "algorithm": "xgboost", "generation": gen,
        "status": "EVALUATED", "search_action": action,
        "parent_trial_ids": parents or [], "data_view": view,
        "evaluation": ev, **(extra or {}),
        "status_history": [{"state": "EVALUATED",
                            "at": "2026-11-01T00:00:00+08:00"}],
    }, trial_dir=d)
    rec.flush()
    return rec


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "run-state.json").write_text(json.dumps(
        {"baseline_id": "t-001", "skill_version": "2.19.0"}),
        encoding="utf-8")
    (run / "evolution-state.json").write_text(json.dumps({}),
                                              encoding="utf-8")
    return run


# ---------------------------------------------------------------------------
# 1. 配置门
# ---------------------------------------------------------------------------

def test_defaults_keep_legacy_and_new_run_config_enables():
    old = Config.load(skill_config=False)
    assert anchor_mode(old) == "delivery"
    assert old.get("evaluator.slice_se_scaling") == "none"
    assert old.get("evaluator.seed_variance_in_se") is False
    assert old.get("evaluator.plateau_stop_generations") is None
    # skill 级 global-config 也不得启用新口径(在途 run 不换尺)
    g = Config.load()
    assert anchor_mode(g) == "delivery"
    assert not g.get("evaluator.seed_variance_in_se")
    new = _new_run_cfg()
    assert anchor_mode(new) == "incumbent"
    assert new.get("evaluator.promotion_reference") == "anchor"
    assert new.get("evaluator.slice_se_scaling") == "sqrt_k"
    assert new.get("evaluator.seed_variance_in_se") is True
    assert float(new.get("evaluator.anchor_parent_guard_p")) == 0.80
    assert int(new.get("evaluator.plateau_stop_generations")) == 2


def test_cmd_init_writes_v219_keys(tmp_path):
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
    assert ev["anchor_mode"] == "incumbent"
    assert ev["slice_se_scaling"] == "sqrt_k"
    assert ev["seed_variance_in_se"] is True
    assert ev["promotion_reference"] == "anchor"


# ---------------------------------------------------------------------------
# 2. 锚点:现任冠军
# ---------------------------------------------------------------------------

def test_incumbent_anchor_picks_warm_start_over_weak_baseline(tmp_path):
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.8009)                      # baseline
    _trial(run, "t-002", 0, metric=0.8123, action="warm_start")
    _trial(run, "t-003", 0, metric=0.8120, action="warm_start")
    new = _new_run_cfg()
    a = resolve_promotion_anchor(run, 1, config=new)
    assert a["anchor_id"] == "t-002" and a["source"] == "incumbent"
    # 旧口径:无确认 → baseline
    legacy = resolve_promotion_anchor(run, 1, config=_cfg())
    assert legacy["anchor_id"] == "t-001" and legacy["source"] == "baseline"


def test_incumbent_anchor_rules(tmp_path):
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.80)
    _trial(run, "t-002", 1, metric=0.83, verdict="promising")   # 不合格
    _trial(run, "t-003", 1, metric=0.84, view="sample")         # 跨视图
    _trial(run, "t-004", 1, metric=0.81)
    _trial(run, "t-005", 2, metric=0.82)                        # 同代不看
    _trial(run, "t-006", 1, metric=0.85, extra={
        "screening": {"screen_only": True}, "screen_only": True})
    new = _new_run_cfg()
    a = _anchor_from_state(run, {}, config=new, before_generation=2)
    assert a["anchor_id"] in ("t-004", "t-006")
    # screening trial 不得当锚点
    from runtime.trial import trial_is_screening
    if trial_is_screening(TrialRecord.load(run / "trials" / "t-006")):
        assert a["anchor_id"] == "t-004"
    # 回归方向:越小越好
    run2 = _run(tmp_path / "x")
    for tid, v in (("t-001", 12.0), ("t-002", 10.0), ("t-003", 11.0)):
        rec = _trial(run2, tid, 0, metric=v)
        rec.data["evaluation"]["primary_metric"]["name"] = "rmse"
        rec.flush()
    a2 = _anchor_from_state(run2, {}, config=new, before_generation=1)
    assert a2["anchor_id"] == "t-002"


def test_incumbent_falls_back_when_no_eligible(tmp_path):
    run = _run(tmp_path)
    _trial(run, "t-001", 0, metric=0.80, verdict="rejected")
    a = _anchor_from_state(run, {}, config=_new_run_cfg(),
                           before_generation=1)
    assert a["source"] == "baseline"


# ---------------------------------------------------------------------------
# 3. 切片门 √k
# ---------------------------------------------------------------------------

def _slice_pack(rng, n=4000, k=4, eps=0.0):
    y = rng.integers(0, 2, n)
    # 零效应:父子共享信号,各自独立同分布的"训练随机性"噪声
    base = rng.normal(size=n) + y * 0.8
    parent = base + rng.normal(scale=0.3, size=n)
    child = base + rng.normal(scale=0.3, size=n) + eps * y
    idx = np.array_split(np.arange(n), k)
    slices = {f"s{i}": {"y_true": y[ix], "score": child[ix]}
              for i, ix in enumerate(idx)}
    sparent = {f"s{i}": parent[ix] for i, ix in enumerate(idx)}
    return y, child, parent, slices, sparent


def test_slice_threshold_scales_with_sqrt_k():
    rng = np.random.default_rng(0)
    y, child, parent, slices, sparent = _slice_pack(rng)
    base_cfg = _cfg(bootstrap_samples=300)
    ev_old = Evaluator(_spec(), base_cfg).evaluate(
        y, child, parent_score=parent, slices=slices,
        slice_parent_scores=sparent)
    new_cfg = _cfg(bootstrap_samples=300, slice_se_scaling="sqrt_k")
    ev_new = Evaluator(_spec(), new_cfg).evaluate(
        y, child, parent_score=parent, slices=slices,
        slice_parent_scores=sparent)
    m_old = ev_old["slice_evaluation"]["threshold_multiplier"]
    m_new = ev_new["slice_evaluation"]["threshold_multiplier"]
    assert ev_new["slice_evaluation"]["se_scaling"] == pytest.approx(2.0)
    assert m_new == pytest.approx(m_old * 2.0, rel=1e-3)


def test_slice_false_reject_rate_drops_under_null():
    """零效应(子代 = 父代 + 噪声)下 √k 口径的切片误拒率显著低于旧口径。"""
    rej_old = rej_new = 0
    n_rep = 30
    for i in range(n_rep):
        rng = np.random.default_rng(100 + i)
        y, child, parent, slices, sparent = _slice_pack(rng, k=4)
        for cfg, bucket in ((_cfg(bootstrap_samples=200), "old"),
                            (_cfg(bootstrap_samples=200,
                                  slice_se_scaling="sqrt_k"), "new")):
            ev = Evaluator(_spec(), cfg).evaluate(
                y, child, parent_score=parent, slices=slices,
                slice_parent_scores=sparent)
            if ev["slice_evaluation"].get("passed") is False:
                if bucket == "old":
                    rej_old += 1
                else:
                    rej_new += 1
    assert rej_new <= rej_old
    assert rej_new / n_rep <= 0.15


# ---------------------------------------------------------------------------
# 4. SE 计入 seed 方差
# ---------------------------------------------------------------------------

def test_seed_variance_inflates_se_and_shrinks_p():
    rng = np.random.default_rng(1)
    n = 20000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y * 0.8
    child = parent + 0.02 * y + rng.normal(scale=0.05, size=n)
    off = Evaluator(_spec(), _cfg(bootstrap_samples=400)).evaluate(
        y, child, parent_score=parent, seed_sd_context={})
    on = Evaluator(_spec(), _cfg(bootstrap_samples=400,
                                 seed_variance_in_se=True)).evaluate(
        y, child, parent_score=parent, seed_sd_context={})
    pb_off = off["delta"]["vs_parent_paired_bootstrap"]
    pb_on = on["delta"]["vs_parent_paired_bootstrap"]
    assert "sigma_seed" not in pb_off
    # floor_ratio=1 ⇒ σ_seed ≥ SE ⇒ se_eff ≥ √2·SE
    assert pb_on["se"] >= pb_on["se_bootstrap"] * np.sqrt(2) - 1e-12
    assert pb_on["p_better"] <= pb_off["p_better"] + 1e-9
    assert on["noise"] == pytest.approx(pb_on["se"])
    assert pb_on["p_basis"] == "normal_approx_se_eff"


def test_seed_variance_uses_replicate_sd_and_override():
    rng = np.random.default_rng(2)
    n = 5000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y
    child = parent + rng.normal(scale=0.1, size=n)
    cfg = _cfg(bootstrap_samples=200, seed_variance_in_se=True)
    big = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent,
        seed_sd_context={"replicate_sd": 0.05, "replicate_df": 3})
    pb = big["delta"]["vs_parent_paired_bootstrap"]
    assert pb["sigma_seed"] == pytest.approx(0.05)
    assert pb["sigma_seed_source"] == "run_replicates"
    ov = Evaluator(_spec(), cfg).evaluate(
        y, child, parent_score=parent,
        seed_sd_context={"sigma_seed_override": pb["sigma_seed"]})
    assert ov["delta"]["vs_parent_paired_bootstrap"]["se"] == \
        pytest.approx(pb["se"])


def test_seed_variance_skipped_without_context_or_for_confirm():
    rng = np.random.default_rng(3)
    n = 3000
    y = rng.integers(0, 2, n)
    parent = rng.normal(size=n) + y
    child = parent + rng.normal(scale=0.1, size=n)
    cfg = _cfg(bootstrap_samples=200, seed_variance_in_se=True)
    ev = Evaluator(_spec(), cfg).evaluate(y, child, parent_score=parent)
    assert "sigma_seed" not in ev["delta"]["vs_parent_paired_bootstrap"]


# ---------------------------------------------------------------------------
# 5. 停止信号
# ---------------------------------------------------------------------------

def _check(stop: StopCondition, deltas, **kw):
    base = dict(n_trials_started=10, n_generations=len(deltas),
                wall_time_hours=1.0, trials_accepted=1, best_metric=0.81,
                baseline_metric=0.80, generation_best_deltas=deltas,
                noise=0.0001, unexplored_high_potential=[])
    base.update(kw)
    return stop.check(**base)


def test_plateau_stop_new_runs_only():
    deltas = [0.011, 0.0002, 0.0001]
    assert _check(StopCondition(_spec(), _cfg()), deltas) is None
    rep = _check(StopCondition(_spec(), _new_run_cfg()), deltas)
    assert rep is not None and "plateau_below_half_mde" in rep["stop_reason"]
    # 最近一代有 ≥0.5×MDE 的增益 → 不停
    assert _check(StopCondition(_spec(), _new_run_cfg()),
                  [0.011, 0.0001, 0.0008]) is None


def test_ceiling_stop_only_on_trial_surface(tmp_path):
    stop = StopCondition(_spec(), _cfg())
    base_surface = {"gap_to_ref": -0.002, "feature_surface": "base",
                    "ceiling_estimate": 0.80}
    rep = _check(stop, [0.01], ceiling=base_surface)
    assert rep is None
    trial_surface = {"gap_to_ref": 0.0004, "feature_surface": "trial:t-9",
                     "ceiling_estimate": 0.8104}
    rep = _check(stop, [0.01], ceiling=trial_surface)
    assert rep is not None
    assert "ceiling_gap_within_mde" in rep["stop_reason"]
    assert rep["ceiling"]["feature_surface"] == "trial:t-9"
    far = dict(trial_surface, gap_to_ref=0.004)
    assert _check(stop, [0.01], ceiling=far) is None


# ---------------------------------------------------------------------------
# 6. ceiling 契约
# ---------------------------------------------------------------------------

def test_ceiling_contract_roundtrip(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError):
        write_ceiling(run, {"primary_metric": "auc"})
    write_ceiling(run, {"primary_metric": "auc", "higher_is_better": True,
                        "ceiling_estimate": 0.812, "best_reference": "strong",
                        "references": {"strong": {"value": 0.812}},
                        "feature_surface": "base"})
    c = load_ceiling(run)
    assert c["schema"] == 1 and c["surface"] == "validation"
    assert gap_to_ref(c, 0.810) == pytest.approx(0.002)
    s = ceiling_summary(run, 0.8115, mde=0.001)
    assert s["within_mde"] is True and s["gap_in_mde"] == pytest.approx(0.5)
    assert lift_capture(base=0.80, achieved=0.81, reference=0.82) == \
        pytest.approx(0.5)
    assert lift_capture(base=2.0, achieved=1.5, reference=1.0,
                        higher_is_better=False) == pytest.approx(0.5)
    assert lift_capture(base=0.8, achieved=0.81, reference=0.79) is None
    assert load_ceiling(tmp_path / "none") is None


# ---------------------------------------------------------------------------
# 7. leak_scan 哨兵尊重泄漏裁决(benchmark creditcard)
# ---------------------------------------------------------------------------

def _runner_stub(cfg, kept):
    from runtime.runner import TrialRunner

    class _Spec:  # 与 TaskSpec 一致:两者都是 property
        @property
        def kept_ambiguous_columns(self):
            return list(kept)

        @property
        def leak_confirmation_note(self):
            return "PCA 分量,复核后合法保留"

    r = TrialRunner.__new__(TrialRunner)
    r.config, r.spec = cfg, _Spec()
    return r


def test_leak_scan_waives_adjudicated_and_weaker_derived():
    r = _runner_stub(_new_run_cfg(), ["V14", "V10"])
    ev: dict = {}
    ls = {"suspects": [{"feature": "V14", "train_auc": 0.9534},
                       {"feature": "V10", "train_auc": 0.9310},
                       {"feature": "vi_x_V14_V10", "train_auc": 0.9600},
                       {"feature": "label_echo", "train_auc": 0.9990}]}
    out = r._waive_adjudicated_suspects(ls, ev)
    assert [x["feature"] for x in out["suspects"]] == ["label_echo"]
    waived = {x["feature"] for x in ev["leakage_suspects_waived"]["suspects"]}
    assert waived == {"V14", "V10", "vi_x_V14_V10"}


def test_leak_scan_no_adjudication_no_waiver():
    r = _runner_stub(_new_run_cfg(), [])
    ev: dict = {}
    ls = {"suspects": [{"feature": "V14", "train_auc": 0.95}]}
    assert r._waive_adjudicated_suspects(ls, ev) == ls and not ev
    # 旧 run 默认关闭
    assert not Config.load(skill_config=False).get(
        "evaluator.leak_scan_respect_adjudication")


def test_leak_scan_waiver_with_real_taskspec():
    """用真实 TaskSpec(property 接口)跑一遍,防测试桩与实现接口不一致。"""
    spec = TaskSpec(data={
        "task_id": "x", "task_name": "x", "model_type": "binary_classification",
        "target": {"name": "y"}, "primary_metric": "auc", "data": {},
        "features": {"leak_confirmation": {"kept_ambiguous": ["V14"],
                                           "note": "复核合法"}}})
    from runtime.runner import TrialRunner

    r = TrialRunner.__new__(TrialRunner)
    r.config, r.spec = _new_run_cfg(), spec
    ev: dict = {}
    out = r._waive_adjudicated_suspects(
        {"suspects": [{"feature": "V14", "train_auc": 0.95}]}, ev)
    assert out["suspects"] == []
    assert ev["leakage_suspects_waived"]["note"] == "复核合法"
