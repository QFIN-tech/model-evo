"""v2.19.0 回归测试:P0-A 证明体系(强参照 / ceiling / benchmark 报告 / 驱动)。

全部离线合成数据,CPU 小数据;不读真实 run,不联网。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.benchmark import build_suite_report, collect_run_scores  # noqa: E402
from runtime.ceiling import load_ceiling, write_ceiling  # noqa: E402
from runtime.reference import (FeatureSurface, ReferenceError_,  # noqa: E402
                               govern_frames, resolve_task, run_strong)
from runtime.spec import TaskSpec  # noqa: E402


def _spec(model_type="binary_classification", metric="auc") -> TaskSpec:
    return TaskSpec(data={
        "task_id": "b", "task_name": "b", "model_type": model_type,
        "target": {"name": "y"}, "time": {"column": "dt"},
        "entity": {"id_column": "uid"}, "primary_metric": metric,
        "data": {}})


def _surface(n=1500, seed=0, reg=False) -> FeatureSurface:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in
                                                       range(6)])
    X["cat"] = rng.choice(["a", "b", "c"], size=n)
    lin = X["f0"] * 1.2 - X["f1"] + (X["cat"] == "a") * 0.8
    y = (lin + rng.normal(scale=0.8, size=n)).to_numpy()
    if not reg:
        y = (y > 0).astype(int)
    k = int(n * 0.7)
    X_tr, X_va, gov = govern_frames(X.iloc[:k].reset_index(drop=True),
                                    X.iloc[k:].reset_index(drop=True))
    return FeatureSurface(X_tr, y[:k], X_va, y[k:], "base", {"gov": gov})


def test_resolve_task_supported_and_rejected():
    t = resolve_task(_spec())
    assert t["higher_is_better"] is True
    assert resolve_task(_spec("regression", "rmse"))["higher_is_better"] is False
    with pytest.raises(ReferenceError_):
        resolve_task(_spec("uplift", "qini"))


def test_govern_frames_numeric_only():
    s = _surface(300)
    assert all(str(dt) == "category" or np.issubdtype(dt, np.number)
               for dt in s.X_tr.dtypes)
    assert list(s.X_tr.columns) == list(s.X_va.columns)


@pytest.mark.parametrize("reg", [False, True])
def test_run_strong_cpu_small(reg):
    s = _surface(reg=reg)
    task = resolve_task(_spec("regression", "rmse") if reg else _spec())
    out = run_strong(task, s, budget_sec=120, n_seeds=2, n_jobs=2,
                     leases_dir=None, algos=("xgboost",), log=lambda m: None)
    ref = out["strong_xgboost"]
    assert ref["status"] == "ok"
    assert ref["n_seeds"] == 2 and len(ref["seed_values"]) == 2
    if not reg:
        assert 0.6 < ref["value"] <= 1.0


def _fake_run(root: Path, name: str, *, base: float, champ: float | None,
              ref: float | None, metric="auc", hib=True) -> Path:
    run = root / name
    (run / "trials").mkdir(parents=True)
    (run / "run-state.json").write_text(json.dumps(
        {"baseline_id": "t-001", "phase": "EVOLVING"}), encoding="utf-8")

    def _t(tid, v, verdict):
        d = run / "trials" / tid
        d.mkdir()
        (d / "evaluation.json").write_text(json.dumps({
            "primary_metric": {"name": metric, "value": v},
            "verdict": verdict, "resource": {"runtime_sec": 60}}),
            encoding="utf-8")
    _t("t-001", base, "accepted")
    if champ is not None:
        _t("t-002", champ, "accepted")
    if ref is not None:
        write_ceiling(run, {"primary_metric": metric, "higher_is_better": hib,
                            "ceiling_estimate": ref,
                            "best_reference": "strong_xgboost",
                            "references": {"strong_xgboost": {"value": ref}},
                            "feature_surface": "base"})
    return run


def test_collect_scores_baseline_only_is_no_search(tmp_path):
    run = _fake_run(tmp_path, "a", base=0.80, champ=None, ref=0.82)
    sc = collect_run_scores(run, "auc", True)
    assert sc["S_base"] == 0.80 and sc["S_mvo"] is None


def test_suite_report_lift_and_gap(tmp_path):
    _fake_run(tmp_path, "a", base=0.80, champ=0.81, ref=0.82)
    _fake_run(tmp_path, "b", base=10.0, champ=9.5, ref=9.0, metric="rmse",
              hib=False)
    _fake_run(tmp_path, "c", base=0.70, champ=None, ref=0.71)
    suite = {"suite_id": "t", "datasets": [
        {"name": "a", "primary_metric": "auc", "tiers": ["fast"],
         "model_type": "binary_classification"},
        {"name": "b", "primary_metric": "rmse", "tiers": ["fast"],
         "model_type": "regression"},
        {"name": "c", "primary_metric": "auc", "tiers": ["fast"],
         "model_type": "binary_classification"},
        {"name": "d", "primary_metric": "auc", "tiers": ["fast"],
         "model_type": "binary_classification"}]}
    rep = build_suite_report(suite, tmp_path, tier="fast",
                             out_dir=tmp_path / "out")
    rows = {r["task_id"]: r for r in rep["tasks"]}
    assert rows["a"]["status"] == "ok"
    assert rows["a"]["lift_capture"] == pytest.approx(0.5)
    assert rows["a"]["gap_to_ref"] == pytest.approx(0.01)
    assert rows["b"]["lift_capture"] == pytest.approx(0.5)
    assert rows["c"]["status"] == "no_search"
    assert rows["d"]["status"] == "missing_run"
    assert Path(rep["output_path"]).exists()


def test_suite_yaml_shape():
    suite = yaml.safe_load((SKILL / "assets" / "benchmark-suite.yaml")
                           .read_text(encoding="utf-8"))
    fast = [d for d in suite["datasets"] if "fast" in d.get("tiers", [])]
    full = [d for d in suite["datasets"] if "full" in d.get("tiers", [])]
    assert len(fast) == 3 and 6 <= len(full) <= 8
    types = {d["model_type"] for d in full}
    assert {"binary_classification", "regression"} <= types
    for d in suite["datasets"]:
        assert d["target"] and d["primary_metric"]
        # OpenML 来源要有 data id;本地来源要声明 builder(如 Home Credit)
        assert d.get("openml_id") or (d.get("source") == "local"
                                      and d.get("builder"))
    large = [d for d in suite["datasets"] if "large" in d.get("tiers", [])]
    assert any(d["name"] == "home_credit" for d in large)


def test_ceiling_out_dir_rejects_inside_run(tmp_path):
    from runtime.reference import compute_ceiling

    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ReferenceError_):
        compute_ceiling(run, _spec(), out_dir=run / "x")
    assert load_ceiling(run) is None


def test_run_agent_dry_run(tmp_path, capsys):
    import importlib.util

    p = SKILL / "scripts" / "benchmark" / "run_agent.py"
    spec = importlib.util.spec_from_file_location("bench_run_agent", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = tmp_path / "credit_g"
    run.mkdir()
    (run / "benchmark-task.json").write_text(json.dumps({
        "dataset": {"name": "credit_g", "primary_metric": "auc",
                    "model_type": "binary_classification"},
        "target_name": "class"}), encoding="utf-8")
    rc = mod.main(["--runs-root", str(tmp_path), "--model", "m-x",
                   "--only", "credit_g", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0 and "--model m-x" in out and "无人值守" in out
    assert not (run / ".claude").exists()  # dry-run 零写入


def test_single_model_reference_and_report_basis(tmp_path):
    from runtime.reference import _single_model_reference

    task = {"primary_metric": "auc", "higher_is_better": True,
            "model_type": "binary_classification"}
    refs = {"strong_xgboost": {"status": "ok", "value": 0.83,
                               "single_seed_value": 0.82},
            "autogluon_best_quality": {"status": "ok", "value": 0.85,
                                       "single_model_value": 0.825},
            "strong_lightgbm": {"status": "failed"}}
    name, v = _single_model_reference(task, refs)
    assert name == "autogluon_best_quality:single_model" and v == 0.825
    run = _fake_run(tmp_path, "a", base=0.80, champ=0.81, ref=0.85)
    c = json.loads((run / "ceiling" / "ceiling-estimate.json").read_text())
    c.update(single_model_estimate=0.82, ensemble_premium=0.03)
    (run / "ceiling" / "ceiling-estimate.json").write_text(json.dumps(c))
    suite = {"suite_id": "t", "datasets": [
        {"name": "a", "primary_metric": "auc", "tiers": ["fast"],
         "model_type": "binary_classification"}]}
    row = build_suite_report(suite, tmp_path, tier="fast",
                             out_dir=tmp_path / "o")["tasks"][0]
    assert row["S_ref_basis"] == "single_model" and row["S_ref"] == 0.82
    assert row["lift_capture"] == pytest.approx(0.5)
    assert row["lift_capture_ensemble"] == pytest.approx(0.2)
    assert row["gap_to_ensemble_ref"] == pytest.approx(0.04)


def test_refengines_unavailable_and_weights(tmp_path, monkeypatch):
    from runtime import refengines as R

    task = resolve_task(_spec())
    s = _surface(300)
    out = R.run_flaml(task, s, budget_sec=5, work_dir=tmp_path / "f",
                      python=str(tmp_path / "nope"))
    # 显式路径不存在时回落环境变量/缺省路径;都不存在才 unavailable
    monkeypatch.setenv(R.REFS_PYTHON_ENV, str(tmp_path / "nope2"))
    monkeypatch.setattr(R, "DEFAULT_REFS_PYTHON", str(tmp_path / "nope3"))
    out = R.run_flaml(task, s, budget_sec=5, work_dir=tmp_path / "f",
                      python=str(tmp_path / "nope"))
    assert out["status"] == "unavailable"
    out = R.run_tabpfn(task, s, budget_sec=5, work_dir=tmp_path / "t",
                       python=str(tmp_path / "nope"))
    assert out["status"] == "unavailable"
    d = tmp_path / "w"
    d.mkdir()
    (d / "tabpfn-v2-classifier.ckpt").write_bytes(b"x")
    monkeypatch.setenv(R.TABPFN_MODEL_DIR_ENV, str(d))
    assert R._tabpfn_weights(True).endswith("tabpfn-v2-classifier.ckpt")
    assert R._tabpfn_weights(False) is None


@pytest.mark.skipif(not Path("/data/oceanus_share/j-wangyaoxuan-jk/"
                             "modelevo-202611/venvs/refs/bin/python").exists(),
                    reason="参照 venv 不存在")
def test_flaml_engine_smoke(tmp_path):
    from runtime.refengines import run_flaml

    task = resolve_task(_spec())
    out = run_flaml(task, _surface(800), budget_sec=10,
                    work_dir=tmp_path / "f", n_jobs=2, log=lambda m: None)
    assert out["status"] == "ok", out
    assert out["single_model_value"] == out["value"] and out["value"] > 0.6


def test_s_mvo_uses_delivery_recipe_and_base_surface_exceeded(tmp_path):
    run = _fake_run(tmp_path, "a", base=0.75, champ=0.79, ref=0.756)
    (run / "final-model").mkdir()
    # final-model 的 validation 是确认子代(换 seed)的分数;交付配方是 t-002
    (run / "final-model" / "evaluation.json").write_text(json.dumps({
        "validation": {"auc": 0.788},
        "delivery_resolution": {"delivery_recipe": "t-002"}}))
    sc = collect_run_scores(run, "auc", True)
    assert sc["S_mvo"] == pytest.approx(0.79) and sc["delivered"] is True
    suite = {"suite_id": "t", "datasets": [
        {"name": "a", "primary_metric": "auc", "tiers": ["large"],
         "model_type": "binary_classification"}]}
    row = build_suite_report(suite, tmp_path, tier="large",
                             out_dir=tmp_path / "o")["tasks"][0]
    assert row["ref_exceeded_on_base_surface"] is True
    assert row["lift_capture"] is None
    assert row["status"] == "ref_below_mvo_base_surface"


def test_check_alignment_rules():
    from runtime.benchmark import _check_alignment

    truth = np.array(["a", "b", "c"])
    idx = {k: i for i, k in enumerate(truth)}
    order, err = _check_alignment("x", np.array(["c", "a", "b"]), truth, idx)
    assert err is None and list(np.array(["c", "a", "b"])[order]) == list(truth)
    assert _check_alignment("x", np.array(["a", "b"]), truth, idx)[1]
    assert _check_alignment("x", np.array(["a", "a", "b"]), truth, idx)[1]
    assert _check_alignment("x", np.array(["a", "b", "z"]), truth, idx)[1]


def test_govern_frames_holdout_uses_train_tables():
    s = _surface(200)
    X = pd.DataFrame({"n": [1.0, 2.0, 3.0], "c": ["a", "b", "zz"]})
    tr, va, gov, ho = govern_frames(X.iloc[:2].reset_index(drop=True),
                                    X.iloc[:2].reset_index(drop=True),
                                    X.iloc[2:].reset_index(drop=True))
    # 未见类别 → 缺失;类别表只来自 train
    assert list(tr["c"].cat.categories) == list(ho["c"].cat.categories)
    assert ho["c"].isna().all()
    del s


def test_aligned_report_end_to_end(tmp_path):
    """合成 run:框架/参照在同一 shadow 面上评分,并能识别不对齐的参照。"""
    from runtime.benchmark import aligned_holdout_report

    run = tmp_path / "ds"
    sp = run / "sample-profile" / "splits"
    (sp / "labels").mkdir(parents=True)
    (sp / "identity").mkdir()
    rng = np.random.default_rng(0)
    n = 400
    keys = np.array([f"k{i}" for i in range(n)])
    y = rng.integers(0, 2, n)
    pd.DataFrame({"y": y}).to_parquet(sp / "labels" / "shadow_test.parquet")
    pd.DataFrame({"__mvo_sample_key": keys}).to_parquet(
        sp / "identity" / "shadow_test.parquet")
    (run / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "ds", "task_name": "ds",
        "model_type": "binary_classification", "target": {"name": "y"},
        "primary_metric": "auc", "data": {}}))
    (run / "run-state.json").write_text(json.dumps({"baseline_id": "t-001"}))
    (run / "final-model").mkdir()
    (run / "final-model" / "evaluation.json").write_text(json.dumps(
        {"delivery_resolution": {"delivery_recipe": "t-002"}}))

    def _w(path, score, k=keys):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"score": score, "__mvo_sample_key": k}).to_parquet(path)
    perm = rng.permutation(n)  # 行序打乱:必须按键对齐
    _w(run / "trials" / "t-001" / "shadow_test_score.parquet",
       (y + rng.normal(0, 1.5, n))[perm], keys[perm])
    _w(run / "trials" / "t-002" / "shadow_test_score.parquet",
       y + rng.normal(0, 1.0, n))
    _w(run / "final-model" / "shadow_test_score.parquet",
       y + rng.normal(0, 1.0, n))
    hd = run / "ceiling" / "holdout"
    _w(hd / "strong_xgboost.parquet", y + rng.normal(0, 0.3, n))
    _w(hd / "flaml.parquet", y[:-1] + 0.0, keys[:-1])  # 少一行 → 不对齐
    write_ceiling(run, {"primary_metric": "auc", "higher_is_better": True,
                        "ceiling_estimate": 0.9, "references": {},
                        "feature_surface": "base",
                        "holdout": {"split": "shadow_test", "files": {
                            "strong_xgboost": {"file": str(hd / "strong_xgboost.parquet"),
                                               "kind": "ensemble"},
                            "flaml": {"file": str(hd / "flaml.parquet"),
                                      "kind": "single_model"}}}})
    suite = {"suite_id": "t", "datasets": [{"name": "ds", "tiers": ["fast"]}]}
    rep = aligned_holdout_report(suite, tmp_path, tier="fast",
                                 out_dir=tmp_path / "o", n_boot=200)
    d = rep["datasets"][0]
    assert d["status"] == "ok_with_missing", d.get("error")
    assert any("FLAML" in e and "行数" in e for e in d["alignment_errors"])
    names = [e["name"] for e in d["entrants"] if e.get("rank")]
    assert names[0] == "strong XGB(5-seed bag)"  # 噪声最小 → 第 1
    xgb = next(e for e in d["entrants"] if e["name"].startswith("strong XGB"))
    assert xgb["vs_framework"] == "better" and xgb["delta_ci95"][0] > 0
    deliv = next(e for e in d["entrants"] if "重训" in e["name"])
    assert deliv.get("rank") is None  # train+val 重训不参与排名


def test_aligned_markdown_shows_row_confidence():
    from runtime.benchmark import render_aligned_markdown

    md = render_aligned_markdown({"split": "shadow_test", "summary": {}, "datasets": [
        {"task_id": "small", "metric": "auc", "n_rows": 999, "status": "ok"},
        {"task_id": "mid", "metric": "auc", "n_rows": 1000, "status": "ok"},
        {"task_id": "big", "metric": "auc", "n_rows": 10000, "status": "ok"},
        {"task_id": "none", "status": "no_holdout"},
    ]})
    assert "| small | auc | 999 | 低 | ok |" in md
    assert "| mid | auc | 1000 | 中 | ok |" in md
    assert "| big | auc | 10000 | 高 | ok |" in md
    assert "| none | None | — | — | no_holdout |" in md


def test_aligned_pairs_by_training_scope(tmp_path):
    """外部参与者按 run.json 的 trained_on 分组配对:train+validation 对交付模型、
    不进名次;train-only 对交付配方、进名次。"""
    from runtime.benchmark import (ALIGNED_DELIVERED, ALIGNED_FRAMEWORK,
                                   aligned_holdout_report)

    run = tmp_path / "ds"
    sp = run / "sample-profile" / "splits"
    (sp / "labels").mkdir(parents=True)
    (sp / "identity").mkdir()
    rng = np.random.default_rng(1)
    n = 400
    keys = np.array([f"k{i}" for i in range(n)])
    y = rng.integers(0, 2, n)
    pd.DataFrame({"y": y}).to_parquet(sp / "labels" / "shadow_test.parquet")
    pd.DataFrame({"__mvo_sample_key": keys}).to_parquet(
        sp / "identity" / "shadow_test.parquet")
    (run / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "ds", "task_name": "ds",
        "model_type": "binary_classification", "target": {"name": "y"},
        "primary_metric": "auc", "data": {}}))
    (run / "run-state.json").write_text(json.dumps({"baseline_id": "t-001"}))
    (run / "final-model").mkdir()
    (run / "final-model" / "evaluation.json").write_text(json.dumps(
        {"delivery_resolution": {"delivery_recipe": "t-001"}}))

    def _w(path, score):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"score": score, "__mvo_sample_key": keys}).to_parquet(path)
    _w(run / "trials" / "t-001" / "shadow_test_score.parquet", y + rng.normal(0, 2.0, n))
    _w(run / "final-model" / "shadow_test_score.parquet", y + rng.normal(0, 0.2, n))
    ext = tmp_path / "ext"
    _w(ext / "ds" / "shadow_score.parquet", y + rng.normal(0, 1.0, n))
    (ext / "ds" / "run.json").write_text(json.dumps({"trained_on": "train+validation"}))
    raw = tmp_path / "raw"
    _w(raw / "ds" / "shadow_score.parquet", y + rng.normal(0, 1.0, n))
    suite = {"suite_id": "t", "datasets": [{"name": "ds", "tiers": ["fast"]}]}
    rep = aligned_holdout_report(suite, tmp_path, tier="fast", out_dir=tmp_path / "o",
                                 n_boot=200, extra_participants={"aide": ext, "raw": raw})
    d = rep["datasets"][0]
    ent = {e["name"]: e for e in d["entrants"]}
    # train+val 参与者:对交付模型(噪声 0.2 远好于 1.0)→ 框架胜,不进名次
    assert ent["aide"]["vs_reference"] == ALIGNED_DELIVERED
    assert ent["aide"]["vs_framework"] == "worse" and ent["aide"].get("rank") is None
    # train-only 参与者:对交付配方(噪声 2.0 差于 1.0)→ 参与者胜,进名次
    assert ent["raw"]["vs_reference"] == ALIGNED_FRAMEWORK
    assert ent["raw"]["vs_framework"] == "better" and ent["raw"].get("rank")
    assert rep["summary"]["framework_vs"]["aide"]["reference"] == ALIGNED_DELIVERED
