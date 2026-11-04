from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.benchmark import evaluate_benchmark
from runtime.config import Config
from runtime.data import (SAMPLE_KEY_COLUMN, load_identities, load_labels,
                          load_split_frames, write_splits,
                          write_view_splits)
from runtime.evaluator import FinalTestGate
from runtime.evolution import EvolutionEngine
from runtime.integrity import freeze_trial_source, verify_trial_source
from runtime.leaderboard import Leaderboard
from runtime.metrics import align_score_by_keys, paired_bootstrap_delta
from runtime.planner import AutonomousPlanner
from runtime.profile import SampleProfile
from runtime.spec import TaskSpec
from runtime.trial import TrialRecord
from runtime.trial import confirmation_parent_eligibility
from scripts._mvo.cmd_deliver import (_finalize_best_iteration,
                                      _finalize_es_fit_kwargs)


def _spec() -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v21", "task_name": "v21",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
    })


def test_final_labels_not_materialized_and_profile_blind(tmp_path: Path):
    df = pd.DataFrame({
        "dt": [f"2026-0{m}-01" for m in range(1, 5) for _ in range(4)],
        "uid": list(range(16)), "x": np.arange(16),
        "label": [0, 1, 0, 1] * 4,
    })
    out = tmp_path / "splits"
    manifest = write_splits(df, _spec(), out)
    assert not (out / "labels" / "final_test.parquet").exists()
    assert manifest["splits"]["final_test"]["labels_materialized"] is False
    prof = SampleProfile(df, _spec(), tmp_path / "profile")
    assert set(prof.df["dt"].str[:7]) == {"2026-01", "2026-02", "2026-03"}
    assert set(prof.supervised_df["dt"].str[:7]) == {"2026-01"}


def test_data_view_keeps_final_rows_without_materializing_labels(
        tmp_path: Path):
    df = pd.DataFrame({
        "dt": [f"2026-0{m}-01" for m in range(1, 5) for _ in range(4)],
        "uid": list(range(16)), "x": np.arange(16, dtype=float),
        "label": [0, 1, 0, 1] * 4,
    })
    run = tmp_path / "run"
    base = run / "sample-profile" / "splits"
    write_splits(df, _spec(), base)
    frames = load_split_frames(run)
    labels = load_labels(run)
    identities = load_identities(run)
    parts = []
    for name in ("train", "validation", "shadow_test", "final_test"):
        cols = [frames[name].reset_index(drop=True)]
        if name in labels:
            cols.append(labels[name].reset_index(drop=True))
        else:
            stub = pd.DataFrame({
                "label": np.full(len(frames[name]), np.nan),
                "uid": identities[name][SAMPLE_KEY_COLUMN].astype(str),
            })
            cols.append(stub)
        cols.append(identities[name].reset_index(drop=True))
        part = pd.concat(cols, axis=1)
        part = part.loc[:, ~part.columns.duplicated()].copy()
        parts.append(part)
    view = pd.concat(parts, ignore_index=True)
    view["x_squared"] = view["x"] ** 2

    manifest = write_view_splits(view, _spec(), run, "v2")
    out = run / "sample-profile" / "splits-v2"
    assert manifest["splits"]["final_test"]["rows"] == 4
    assert not (out / "labels" / "final_test.parquet").exists()
    final_identity = pd.read_parquet(out / "identity/final_test.parquet")
    assert list(final_identity.columns) == ["dt", SAMPLE_KEY_COLUMN]
    assert len(final_identity) == 4


def test_final_gate_never_refunds(tmp_path: Path):
    gate = FinalTestGate(tmp_path)
    gate.acquire("test")
    gate.release_last("failure after acquire")
    state = json.loads((tmp_path / "final-test-gate.json").read_text())
    assert state["count"] == 1
    assert state["invalidated"] is True
    try:
        gate.acquire("retry")
    except PermissionError:
        pass
    else:
        raise AssertionError("consumed final gate must not be reusable")


def test_pairing_uses_both_key_sides_and_keeps_treatment_order():
    keys = np.asarray(["a", "b", "c"])
    old_keys = np.asarray(["c", "a", "b"])
    old = np.asarray([0.9, 0.1, 0.6])
    assert np.allclose(align_score_by_keys(old, keys, old_keys),
                       [0.1, 0.6, 0.9])
    # uplift 入口只应重排 old score，不应按 old_keys 重排当前 treatment。
    out = paired_bootstrap_delta(
        np.asarray([0, 1, 1, 0, 1, 0]),
        np.asarray([0.1, 0.8, 0.7, 0.2, 0.9, 0.3]),
        np.asarray([0.3, 0.1, 0.8, 0.9, 0.2, 0.7]),
        "qini", treatment=np.asarray([0, 1, 0, 1, 1, 0]),
        keys=np.asarray(["a", "b", "c", "d", "e", "f"]),
        old_keys=np.asarray(["f", "a", "b", "e", "d", "c"]),
        n_boot=50)
    assert "delta_mean" in out


def test_source_freeze_detects_mutation(tmp_path: Path):
    trial = tmp_path / "trial"
    (trial / "code").mkdir(parents=True)
    (trial / "config").mkdir()
    (trial / "code" / "train.py").write_text("print('a')\n")
    (trial / "config" / "trial.yaml").write_text("seed: 1\n")
    frozen = freeze_trial_source(trial)
    assert verify_trial_source(trial)["combined_sha256"] == \
        frozen["combined_sha256"]
    (trial / "code" / "train.py").write_text("print('b')\n")
    try:
        verify_trial_source(trial)
    except PermissionError:
        pass
    else:
        raise AssertionError("mutated source must fail verification")


def test_finalize_early_stopping_preserves_library_iteration_semantics():
    LightGBMModel = type(
        "LightGBMModel", (),
        {"__module__": "lightgbm.sklearn", "best_iteration_": 17})
    XGBoostModel = type(
        "XGBoostModel", (),
        {"__module__": "xgboost.sklearn", "best_iteration": 0})
    assert _finalize_best_iteration(LightGBMModel()) == 17
    # XGBoost reports a zero-based index; zero means one selected tree.
    assert _finalize_best_iteration(XGBoostModel()) == 1

    class PlainEstimator:
        def fit(self, X, y):
            return self

    assert _finalize_es_fit_kwargs(
        PlainEstimator(), [1], [0], 50) == {}


def _trial(tmp_path: Path, tid: str, verdict: str, metric: float,
           *, fresh: bool = False) -> TrialRecord:
    d = tmp_path / tid
    d.mkdir()
    return TrialRecord(data={
        "trial_id": tid, "algorithm": "lightgbm", "generation": 1,
        "status": "EVALUATED", "parent_trial_ids": [],
        "evaluation": {"primary_metric": {"name": "auc", "value": metric},
                       "verdict": verdict, "integrity": {"valid": True},
                       "fresh_confirmation": {"passed": fresh}},
    }, trial_dir=d)


def test_champion_selection_fails_closed_and_delivery_needs_confirmation(
        tmp_path: Path):
    config = Config.load(skill_config=False)
    lb = Leaderboard(tmp_path, _spec(), config)
    rejected = _trial(tmp_path, "r", "rejected", 0.99)
    assert lb.best_trial([rejected]) is None
    accepted = _trial(tmp_path, "a", "accepted", 0.80)
    unverified = _trial(tmp_path, "u", "confirmed", 0.81)
    confirmed = _trial(tmp_path, "c", "confirmed", 0.79, fresh=True)
    confirmed.data["evaluation"]["shadow_test"] = {"auc": 0.999}
    assert lb.best_trial([accepted, confirmed]).trial_id == "a"
    assert lb.best_trial(
        [accepted, confirmed], shadow_recheck=True).trial_id == "a"
    assert lb.best_trial([accepted, unverified], delivery=True) is None
    # v2.14.0:交付位必须绑定搜索冠军配方。搜索冠军是 u(0.81),c 是另一
    # 个配方的已确认 trial 且落后 0.02 ≫ 容差 → 不得静默回落到它
    assert lb.best_trial(
        [accepted, unverified, confirmed], delivery=True) is None
    assert lb.last_delivery_resolution["status"] == "diverged"
    # c 是 u 的 confirm 子代(同配方)→ 交付位绑定成立
    confirmed.data["search_action"] = "confirm"
    confirmed.data["parent_trial_ids"] = ["u"]
    assert lb.best_trial(
        [accepted, unverified, confirmed], delivery=True).trial_id == "c"
    assert lb.last_delivery_resolution["status"] == "bound"


def test_generation_fdr_downgrades_uncorrected_accept(tmp_path: Path):
    engine = EvolutionEngine(tmp_path, _spec(), Config.load(skill_config=False))
    first = _trial(tmp_path, "fdr-a", "accepted", 0.81)
    second = _trial(tmp_path, "fdr-b", "accepted", 0.82)
    first.data["evaluation"]["delta"] = {
        "vs_parent_paired_bootstrap": {"p_better": 0.96}}
    second.data["evaluation"]["delta"] = {
        "vs_parent_paired_bootstrap": {"p_better": 0.80}}
    report = engine._apply_generation_fdr(1, [first, second])
    assert report["family_size"] == 2
    assert first.data["evaluation"]["verdict"] == "promising"
    assert second.data["evaluation"]["verdict"] == "promising"


def test_confirmation_requires_selection_correction(tmp_path: Path):
    parent = _trial(tmp_path, "parent", "accepted", 0.81)
    ok, reason = confirmation_parent_eligibility(parent)
    assert ok is False
    assert reason == "selection_correction_not_passed"
    parent.data["evaluation"]["selection_correction"] = {"passed": True}
    assert confirmation_parent_eligibility(parent)[0] is True

    baseline = _trial(tmp_path, "baseline", "accepted", 0.78)
    baseline.data["evaluation"]["root_baseline_only"] = True
    assert confirmation_parent_eligibility(baseline)[0] is True


def test_autonomous_planner_and_benchmark_gate(tmp_path: Path):
    run = tmp_path / "run"
    (run / "trials").mkdir(parents=True)
    batch = AutonomousPlanner(
        run, _spec(), Config.load(skill_config=False)).plan(slots=2)
    assert batch["proposals"]
    assert "final_test" not in json.dumps(batch["proposals"]).lower()
    assert (run / "planner" / "latest.json").exists()
    first = batch["proposals"][0]
    command = shlex.split(first["materialize_command"])
    assert Path(command[1]).is_absolute()
    assert "--planner-proposal-id" in command
    assert "--planner-semantic-hash" in command
    # 已物化的 proposal hash 必须阻止下一轮重复推荐同一语义。
    rec_dir = run / "trials" / "trial-existing"
    TrialRecord(data={
        "trial_id": "trial-existing", "algorithm": first["algorithm"],
        "generation": first["generation"], "status": "PROPOSED",
        "search_action": first["action"],
        "parent_trial_ids": first["parent_ids"],
        "changed_modules": first["changed_modules"],
        "family_hints": [first["family"]],
        "planner_semantic_hash": first["semantic_hash"],
        "hypothesis": {"statement": first["hypothesis"]},
    }, trial_dir=rec_dir).flush()
    next_batch = AutonomousPlanner(
        run, _spec(), Config.load(skill_config=False)).plan(slots=2)
    assert first["semantic_hash"] not in {
        p["semantic_hash"] for p in next_batch["proposals"]}

    task = {
        "task_id": "t", "model_type": "binary_classification",
        "direction": "maximize",
        "baseline_score": 0.70, "achieved_score": 0.79,
        "reference_score": 0.80, "compute_hours": 1.0,
        "reference_compute_hours": 1.0, "compute_budget_hours": 1.0,
        "mde": 0.02, "search_seeds": 3, "delivered": True,
        "reproducible": True, "reference_preregistered": True,
        "independent_test": True,
        "leakage_detected": False, "invalid_champion": False,
        "final_test_touches": 1,
    }
    one = evaluate_benchmark({"suite_id": "s", "protocol": {
        "fixed_compute_budget": True}, "tasks": [task]})
    assert one["quality_gate"]["passed"] is False
    model_types = ["binary_classification", "regression", "uplift"] * 2
    tasks = [{**task, "task_id": f"t{i}", "model_type": model_types[i]}
             for i in range(6)]
    report = evaluate_benchmark({"suite_id": "s", "protocol": {
        "fixed_compute_budget": True}, "tasks": tasks})
    assert np.isclose(report["aggregate"]["median_lift_capture"], 0.9)
    assert np.isclose(report["aggregate"]["p10_lift_capture"], 0.9)
    assert report["quality_gate"]["passed"] is True
