"""v2.19.0 P1「计划由数据驱动」:画像规则表 / 去硬编码 / 经验生成候选 /
HPO 降权 / confirm 只做一次 / planner-context 天花板节。

全部离线:合成 profile / inventory / trial 目录与 library-view.json;经验库
客户端由 tests/conftest.py 指向不可达端口。序列列刻意不用 seq* 命名,验证
判定只看形态与 trial 产物。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

from runtime import planner as planner_mod  # noqa: E402
from runtime import profile_rules as PR  # noqa: E402
from runtime.config import Config  # noqa: E402
from runtime.planner import AutonomousPlanner  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402

ATID = "binary-label-200w_1000w-bc5329dc"
SEQ_COL = "colA"          # 事件序列列(故意不叫 seq*)
CAT_COL = "tier"          # 低基数类别串(reparse 源,但不是序列)
BASE_COLS = [f"f{i}" for i in range(40)]


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _spec(reparse=(SEQ_COL, CAT_COL)) -> TaskSpec:
    d = {"task_id": "v219", "task_name": "v219",
         "model_type": "binary_classification",
         "target": {"name": "label", "positive_definition": "1"},
         "time": {"column": "dt"}, "entity": {"id_column": "uid"},
         "primary_metric": "auc", "business_domains": ["user_operation"],
         "split_policy": {"type": "time_based", "train": "2026-01",
                          "validation": "2026-02", "shadow_test": "2026-03",
                          "final_test": "2026-04"},
         "data": {}, "resource_budget": {"max_trials": 60}}
    if reparse:
        d["features"] = {"reparse": [{"column": c, "plan": "p"}
                                     for c in reparse]}
    return TaskSpec(data=d)


def _cfg(**overrides) -> Config:
    cfg = Config.load(skill_config=False)
    cfg.raw.setdefault("feature_exploration", {})["min_generations"] = 0
    for dotted, val in overrides.items():
        node = cfg.raw
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def _run(tmp_path, spec: TaskSpec | None = None) -> Path:
    run = tmp_path / "run"
    (run / "trials").mkdir(parents=True, exist_ok=True)
    (run / "task-spec.yaml").write_text(
        yaml.safe_dump((spec or _spec()).data, allow_unicode=True),
        encoding="utf-8")
    (run / "run-state.json").write_text(json.dumps({
        "abstract_task_id": ATID}), encoding="utf-8")
    return run


def _seq_value(i: int) -> str:
    # 变长事件序列:词表 6 个事件,长度 2~40(长度 CV 大、token 复用高)
    ev = ["LOGIN", "VIEW", "CLICK", "APPLY", "PAY", "EXIT"]
    n = 2 + (i * 7) % 39
    return ",".join(ev[(i + k) % 6] for k in range(n))


def _profile(run: Path, *, n_train=15000, pos_rate=0.03, drift_cols=6,
             high_card=True, missing_cols=4, splits=True) -> None:
    sp = run / "sample-profile"
    (sp / "splits" / "reparse").mkdir(parents=True, exist_ok=True)
    (sp / "sample-profile.json").write_text(json.dumps({
        "n_rows": n_train, "n_features": len(BASE_COLS),
        "sample_summary": {"label_rate": pos_rate},
        "splits_summary": {"splits": [
            {"split": "train", "rows": n_train, "positive_rate": pos_rate}]},
    }), encoding="utf-8")
    rows = []
    for i, c in enumerate(BASE_COLS):
        psi = 0.6 if i < drift_cols else 0.01
        rows.append({"feature": c, "dtype": "float64", "n_unique": 1000,
                     "unique_ratio": 0.07, "high_cardinality": False,
                     "id_like": False, "iv": 0.01,
                     "psi_train_vs_validation": psi,
                     "psi_max_adjacent_months": psi / 2})
    if high_card:
        rows.append({"feature": "city", "dtype": "object", "n_unique": 300,
                     "unique_ratio": 0.02, "high_cardinality": False,
                     "id_like": False, "iv": 0.05,
                     "psi_train_vs_validation": 0.0,
                     "psi_max_adjacent_months": 0.0})
    pd.DataFrame(rows).to_csv(sp / "feature-quality.csv", index=False)
    woe = []
    for i, c in enumerate(BASE_COLS[:8]):
        woe.append({"feature": c, "bin": "[-inf,1)", "count": 900,
                    "woe": 0.01, "iv": 0.0001})
        nan_woe = 0.8 if i < missing_cols else 0.02
        woe.append({"feature": c, "bin": "__nan__", "count": 100,
                    "woe": nan_woe, "iv": 0.05})
    pd.DataFrame(woe).to_csv(sp / "feature-woe.csv", index=False)
    if splits:
        n = 400
        pd.DataFrame({SEQ_COL: [_seq_value(i) for i in range(n)],
                      CAT_COL: [["A", "B", "C"][i % 3] for i in range(n)],
                      "dt": ["2026-01-01"] * n,
                      "uid": [f"u{i}" for i in range(n)]}).to_parquet(
            sp / "splits" / "reparse" / "train.parquet", index=False)
        pd.DataFrame({c: [0.0] * 10 for c in BASE_COLS}).to_parquet(
            sp / "splits" / "train.parquet", index=False)


def _trial(run: Path, tid: str, *, gen=1, action="explore", parents=None,
           value=0.80, vs_parent=None, verdict="accepted", hints=None,
           modules=("feature",), algo="xgboost", code="", fc=None,
           topics=None, vs_baseline=None, extra=None) -> TrialRecord:
    ev = {"verdict": verdict, "trial_id": tid,
          "primary_metric": {"name": "auc", "value": value},
          "delta": {"vs_parent": vs_parent, "vs_baseline": vs_baseline},
          "integrity": {"valid": True},
          "selection_correction": {"passed": True}}
    data = {"trial_id": tid, "algorithm": algo, "generation": gen,
            "status": "EVALUATED", "search_action": action,
            "parent_trial_ids": list(parents or []),
            "changed_modules": list(modules), "family_hints": hints,
            "data_view": "v1",
            "hypothesis": {"statement": "测试用最小假设陈述"},
            "evaluation": ev, "result": {"runtime_sec": 600}}
    if topics:
        data["topics"] = list(topics)
    data.update(extra or {})
    tdir = run / "trials" / tid
    rec = TrialRecord(data=data, trial_dir=tdir)
    rec.flush()
    (tdir / "evaluation.json").write_text(json.dumps(ev), encoding="utf-8")
    (tdir / "code").mkdir(parents=True, exist_ok=True)
    (tdir / "code" / "train.py").write_text(code or "def main():\n    pass\n",
                                            encoding="utf-8")
    (tdir / "config").mkdir(parents=True, exist_ok=True)
    # 配置里列全部 reparse 源(与真实 run 一致:配置不构成"引用")
    (tdir / "config" / "trial.yaml").write_text(yaml.safe_dump({
        "trial_id": tid, "algorithm": algo,
        "features": {"reparse": [SEQ_COL, CAT_COL]}}), encoding="utf-8")
    (tdir / "metrics.json").write_text(json.dumps({"n_features": 40}),
                                       encoding="utf-8")
    if fc is not None:
        (tdir / "model").mkdir(parents=True, exist_ok=True)
        (tdir / "model" / "feature_columns.json").write_text(
            json.dumps(list(fc)), encoding="utf-8")
    return rec


def _planner(run: Path, cfg: Config | None = None,
             spec: TaskSpec | None = None) -> AutonomousPlanner:
    return AutonomousPlanner(run, spec or _spec(), cfg or _cfg())


# ---------------------------------------------------------------------------
# 1. 画像规则表
# ---------------------------------------------------------------------------

def test_profile_rules_hits_by_shape(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    res = PR.evaluate(run, _spec())
    rules = {h["rule"]: h for h in res["hits"]}
    assert set(rules) >= {"sequence_column", "high_cardinality_categorical",
                          "class_imbalance", "strong_drift", "small_sample",
                          "informative_missing"}
    assert "large_sample" not in rules
    seq = [h for h in res["hits"] if h["rule"] == "sequence_column"]
    # 只认形态:colA 命中、低基数类别串 tier 不命中
    assert [h["evidence"]["column"] for h in seq] == [SEQ_COL]
    ev = seq[0]["evidence"]
    assert ev["kind"] == "variable_length_sequence"
    assert ev["delim_density"] >= PR.SEQ_DELIM_MIN
    assert ev["token_reuse"] > PR.TEXT_SHAPE_VOCAB_REUSE_MIN
    # 每条建议带 prior / cost / family
    for h in res["hits"]:
        for sg in h["suggestions"]:
            assert sg["prior_gain_mde"] == PR.PRIOR_GAIN_MDE[sg["key"]]
            assert sg["cost_hours"] == PR.COST_HOURS[sg["key"]]
            assert sg["family"].startswith("profile:")
    assert rules["class_imbalance"]["evidence"]["positive_rate"] == 0.03
    assert rules["strong_drift"]["evidence"]["n_psi_gt"] == 6
    assert rules["informative_missing"]["evidence"]["n_columns"] == 4
    # 形态缓存落盘到 planner/(下一次 CLI 调用不重扫)
    assert (run / "planner" / "profile-shapes.json").exists()


def test_profile_rules_large_sample_and_quiet_data(tmp_path):
    run = _run(tmp_path)
    _profile(run, n_train=2_000_000, pos_rate=0.25, drift_cols=0,
             high_card=False, missing_cols=0)
    res = PR.evaluate(run, _spec(reparse=(CAT_COL,)))
    rules = [h["rule"] for h in res["hits"]]
    assert rules == ["large_sample"]
    # 大样本建议只做提示,不物化
    assert res["hits"][0]["suggestions"][0]["materialize"] is False


def test_profile_rule_candidates_materialized(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    pl = _planner(run)
    cands = pl._profile_rule_candidates(None)
    by_fam = {p["family"]: p for p in cands}
    assert f"profile:seq_ops:{SEQ_COL}" in by_fam
    assert "profile:class_weight" in by_fam
    seqp = by_fam[f"profile:seq_ops:{SEQ_COL}"]
    # expected_gain = prior × MDE(不再是 0)
    assert seqp["acquisition"]["expected_gain_mde"] == pytest.approx(
        PR.PRIOR_GAIN_MDE["seq_ops"])
    assert by_fam["profile:class_weight"]["changed_modules"] == ["sample"]
    assert seqp["recipe"]["type"] == "profile_rule"
    assert seqp["profile_rule"]["evidence"]["column"] == SEQ_COL
    batch = pl.plan(slots=4)
    assert batch["candidate_sources"].get("profile_rule", 0) >= 5
    assert any(r["rule"] == "sequence_column" for r in batch["profile_rules"])


def test_profile_rule_posterior_replaces_prior(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _trial(run, "trial-002", gen=1, parents=["trial-001"], value=0.7995,
           vs_parent=-0.0005, verdict="rejected",
           hints=["profile:class_weight"], modules=("sample",),
           fc=BASE_COLS)
    pl = _planner(run)
    cands = {p["family"]: p for p in pl._profile_rule_candidates(
        pl._parent())}
    a = cands["profile:class_weight"]["acquisition"]
    assert a["expected_gain"] == pytest.approx(-0.0005)
    assert "后验" in a["prior_note"]
    # 未评估过的 family 仍按先验
    assert cands["profile:drift_column_drop"]["acquisition"][
        "expected_gain_mde"] == pytest.approx(
            PR.PRIOR_GAIN_MDE["drift_column_drop"])


def test_profile_rules_disabled_by_config(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    pl = _planner(run, _cfg(**{"planner.profile_rules": False}))
    assert pl._profile_rule_candidates(None) == []


# ---------------------------------------------------------------------------
# 2. 去硬编码
# ---------------------------------------------------------------------------

def test_prefix_table_removed():
    assert not hasattr(planner_mod, "_REPARSE_PREFIX_FAMILIES")
    assert not hasattr(planner_mod, "_reparse_prefix_families")


def test_inventory_sequence_dimensions_follow_shape(tmp_path):
    from runtime.inventory import (NON_SEQUENCE_GRANULARITY_DIMENSIONS,
                                   SEQUENCE_GRANULARITY_DIMENSIONS,
                                   build_inventory)

    run = _run(tmp_path)
    _profile(run)
    # 关键词兜底会把名字含 seq 的列判成序列;形态证据在时必须以形态为准
    sp = run / "sample-profile" / "splits" / "reparse" / "train.parquet"
    df = pd.read_parquet(sp)
    df["seq_flag"] = ["Y", "N"] * (len(df) // 2)
    df.to_parquet(sp, index=False)
    spec = _spec(reparse=(SEQ_COL, CAT_COL, "seq_flag"))
    inv = build_inventory(run, spec)
    dims = {s["column"]: [g["dimension"] for g in s["granularity_space"]]
            for s in inv["sources"] if s["source_type"] == "reparse"}
    seq_dims = [d["dimension"] for d in SEQUENCE_GRANULARITY_DIMENSIONS]
    non_dims = [d["dimension"] for d in NON_SEQUENCE_GRANULARITY_DIMENSIONS]
    assert dims[SEQ_COL] == seq_dims
    assert dims[CAT_COL] == non_dims
    assert dims["seq_flag"] == non_dims
    ev = {s["column"]: s["evidence"] for s in inv["sources"]
          if s["source_type"] == "reparse"}
    assert ev[SEQ_COL]["sequence_basis"] == "shape"


def test_inventory_keyword_fallback_without_shapes(tmp_path):
    from runtime.inventory import build_inventory

    run = _run(tmp_path)
    inv = build_inventory(run, _spec(reparse=("event_log",)))
    src = inv["sources"][0]
    assert src["evidence"]["sequence_basis"] == "name_keyword_fallback"
    assert src["evidence"]["is_sequence"] is True


def test_source_consumed_by_trial_artifacts(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    new_cols = ["zz_tok_cnt_7", "zz_tok_cnt_30", "zz_last_gap"]
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _trial(run, "trial-002", gen=1, parents=["trial-001"], value=0.805,
           vs_parent=0.005, fc=BASE_COLS + new_cols,
           code=f"raw = df['{SEQ_COL}'].str.split(',')\n")
    # 冠军复制父代代码(同样引用源列)再加一列 —— 引入者仍是 trial-002
    _trial(run, "trial-003", gen=2, parents=["trial-002"], value=0.807,
           vs_parent=0.002, fc=BASE_COLS + new_cols + ["f_ratio"],
           code=f"raw = df['{SEQ_COL}'].str.split(',')\nx = 1\n")
    pl = _planner(run)
    assert pl._parent().trial_id == "trial-003"
    assert pl._source_already_consumed_by_champion(SEQ_COL, "window")
    ev = pl._consumption_evidence(SEQ_COL)
    assert ev["introduced_by"] == "trial-002"
    assert ev["n_new_cols_in_champion"] == 3
    # 配置里列着但代码没引用的源 → 未消费
    assert not pl._source_already_consumed_by_champion(CAT_COL, "window")


def test_source_not_consumed_when_new_cols_dropped(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _trial(run, "trial-002", gen=1, parents=["trial-001"], value=0.805,
           vs_parent=0.005, fc=BASE_COLS + ["zz_a", "zz_b"],
           code=f"raw = df[\"{SEQ_COL}\"]\n")
    # 冠军摘掉了这批列(列集回到基础面)
    _trial(run, "trial-003", gen=2, parents=["trial-002"], value=0.806,
           vs_parent=0.001, fc=BASE_COLS,
           code=f"raw = df[\"{SEQ_COL}\"]\n")
    pl = _planner(run)
    assert not pl._source_already_consumed_by_champion(SEQ_COL, "window")


def test_root_trial_consumption_uses_base_face(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    # 无父代:新增列相对 splits/train.parquet 基础面判定
    _trial(run, "trial-001", gen=0, value=0.80,
           fc=BASE_COLS + ["zz_tok"], code=f"c = '{SEQ_COL}'\n")
    pl = _planner(run)
    assert pl._source_already_consumed_by_champion(SEQ_COL, "window")


def test_consumed_sequence_halves_profile_prior(tmp_path):
    run = _run(tmp_path)
    _profile(run)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _trial(run, "trial-002", gen=1, parents=["trial-001"], value=0.805,
           vs_parent=0.005, fc=BASE_COLS + ["zz_a"],
           code=f"raw = df['{SEQ_COL}']\n")
    pl = _planner(run)
    p = {c["family"]: c for c in pl._profile_rule_candidates(pl._parent())}[
        f"profile:seq_ops:{SEQ_COL}"]
    assert p["acquisition"]["expected_gain_mde"] == pytest.approx(
        0.5 * PR.PRIOR_GAIN_MDE["seq_ops"])
    assert p["acquisition"]["information_gain"] == 0.0


# ---------------------------------------------------------------------------
# 3. 经验生成候选
# ---------------------------------------------------------------------------

def _principle(pid, *, direction=1, topics=("feature.seq.window_count",),
               category="feature_engineering", objects=(), atid=ATID,
               metric="auc"):
    return {"principle_id": pid, "version": 1, "direction": direction,
            "topics": list(topics), "objects": list(objects),
            "category": category, "claim": f"{pid} 的结论陈述",
            "scope": {"model_type": "binary_classification",
                      "primary_metric": metric, "target_transform": "raw"},
            "evidence": {"abstract_task_ids": [atid]},
            "status": "active", "confidence": "medium"}


def _view(run: Path, principles, stats, outcomes=()):
    topics = sorted({t for p in principles for t in p["topics"]})
    data = {"schema": "modelevo.run-view/1", "watermark": 10, "stale": False,
            "atid": ATID, "namespace": "test",
            "scope": {"model_type": "binary_classification",
                      "primary_metric": "auc", "target_transform": "raw"},
            "principles": list(principles), "stats": stats,
            "topics": [{"code": t, "aliases": []} for t in topics],
            "aliases": {}, "fingerprints": [], "dossiers": {},
            "leaderboard": [], "baselines": {},
            "trial_outcomes": list(outcomes)}
    (run / "experience").mkdir(parents=True, exist_ok=True)
    (run / "experience" / "library-view.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_experience_candidates_from_positive_principles(tmp_path):
    run = _run(tmp_path)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    # 本 run 已失败的主题
    _trial(run, "trial-002", gen=1, parents=["trial-001"], value=0.799,
           vs_parent=-0.001, verdict="rejected",
           topics=["feature.cross.ratio"], fc=BASE_COLS)
    _view(run, [
        _principle("p-good", objects=["evt_"]),
        _principle("p-mixed", topics=("feature.seq.recency",)),
        _principle("p-failed-topic", topics=("feature.cross.ratio",)),
        _principle("p-avoid", direction=-1),
        _principle("p-eng", category="engineering",
                   topics=("engineering.delivery",)),
        _principle("p-other-metric", topics=("feature.seq.bigram",),
                   metric="ks"),
    ], stats={"p-good": {"posterior": 0.7, "confidence": "medium"},
              "p-mixed": {"posterior": 0.6, "confidence": "mixed"},
              "p-failed-topic": {"posterior": 0.7, "confidence": "medium"},
              "p-other-metric": {"posterior": 0.7, "confidence": "medium"}})
    pl = _planner(run)
    cands = pl._experience_candidates(pl._parent())
    assert [c["experience_basis"]["principle_id"] for c in cands] == ["p-good"]
    c = cands[0]
    assert c["family"] == "exp:p-good"
    assert "--basis p-good:follows" in c["materialize_command"]
    assert "--topics feature.seq.window_count" in c["materialize_command"]
    # prior = 刻度 × 后验 × L0 折扣(1.0)
    assert c["acquisition"]["expected_gain_mde"] == pytest.approx(
        planner_mod.EXPERIENCE_PRIOR_SCALE_MDE * 0.7)
    skipped = {s["principle_id"]: s["reason"]
               for s in pl._experience_skipped}
    assert "mixed" in skipped["p-mixed"]
    assert "失败" in skipped["p-failed-topic"]
    assert "p-other-metric" in skipped     # 口径不一致(metric_family)
    batch = pl.plan(slots=4)
    assert batch["candidate_sources"].get("experience") == 1


def test_experience_candidates_empty_when_library_disabled(tmp_path):
    run = _run(tmp_path)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _view(run, [_principle("p-good")],
          stats={"p-good": {"posterior": 0.7, "confidence": "medium"}})
    pl = _planner(run, _cfg(**{"library.enabled": False}))
    assert pl._experience_candidates(pl._parent()) == []


def test_experience_candidate_not_renominated_once_used(tmp_path):
    run = _run(tmp_path)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS,
           extra={"experience_ids": ["p-good"]})
    _view(run, [_principle("p-good")],
          stats={"p-good": {"posterior": 0.7, "confidence": "medium"}})
    pl = _planner(run)
    assert pl._experience_candidates(pl._parent()) == []


# ---------------------------------------------------------------------------
# 4. HPO 降权
# ---------------------------------------------------------------------------

def _hpo_outcome(i, delta):
    return {"run_id": f"r{i}", "trial_id": f"t-hpo-{i}", "atid": ATID,
            "action": "hpo", "higher_is_better": True,
            "delta_parent": delta}


def test_hpo_not_nominated_on_negative_history(tmp_path):
    run = _run(tmp_path)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _view(run, [], stats={}, outcomes=[_hpo_outcome(i, -0.0004)
                                       for i in range(4)])
    pl = _planner(run)
    fams = [p["family"] for p in pl._model_candidates(pl._parent())]
    assert "model:hpo" not in fams
    assert "不提名" in pl._hpo_skipped["reason"]
    batch = pl.plan(slots=4)
    assert batch["hpo_not_nominated"]["history"]["n_library"] == 4


def test_hpo_posterior_on_positive_history(tmp_path):
    run = _run(tmp_path)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _view(run, [], stats={}, outcomes=[_hpo_outcome(1, 0.0015)])
    pl = _planner(run)
    hpo = [p for p in pl._model_candidates(pl._parent())
           if p["family"] == "model:hpo"][0]
    mde = pl._mde()
    expect = (0.0015 + planner_mod.HPO_PRIOR_K0 * 0.75 * mde) / (
        1 + planner_mod.HPO_PRIOR_K0)
    assert hpo["acquisition"]["expected_gain"] == pytest.approx(expect)


def test_hpo_large_sample_without_positive_history(tmp_path):
    run = _run(tmp_path)
    _profile(run, n_train=2_000_000, pos_rate=0.25)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    pl = _planner(run)
    fams = [p["family"] for p in pl._model_candidates(pl._parent())]
    assert "model:hpo" not in fams
    assert "大样本" in pl._hpo_skipped["reason"]


def test_hpo_prior_without_history_small_data(tmp_path):
    run = _run(tmp_path)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    pl = _planner(run)
    hpo = [p for p in pl._model_candidates(pl._parent())
           if p["family"] == "model:hpo"][0]
    assert hpo["acquisition"]["expected_gain_mde"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 5. confirm 只做一次
# ---------------------------------------------------------------------------

def _confirm_run(tmp_path) -> Path:
    run = _run(tmp_path)
    _trial(run, "trial-001", gen=0, value=0.80, fc=BASE_COLS)
    _trial(run, "trial-002", gen=1, parents=["trial-001"], value=0.806,
           vs_parent=0.006, vs_baseline=0.006, fc=BASE_COLS + ["x1"])
    return run


def _confirms(batch):
    return [p for p in batch["proposals"] if p["action"] == "confirm"]


def test_confirm_final_only_not_nominated_midrun(tmp_path):
    run = _confirm_run(tmp_path)
    batch = _planner(run).plan(slots=4)
    assert batch["confirm_policy"]["policy"] == "final_only"
    assert batch["confirm_policy"]["nominate"] is False
    assert not _confirms(batch)


def test_confirm_final_only_after_stop_suggestion(tmp_path):
    run = _confirm_run(tmp_path)
    (run / "evolution-state.json").write_text(json.dumps({
        "stopped": {"stop_reason": "plateau_below_half_mde"}}),
        encoding="utf-8")
    batch = _planner(run).plan(slots=2)
    assert "stop_suggested" in batch["confirm_policy"]["triggers"]
    conf = _confirms(batch)
    assert len(conf) == 1 and conf[0]["parent_ids"] == ["trial-002"]
    assert conf[0]["confirm_nomination"]["role"] == "search_champion_recipe"
    # 保底入批:排在第一位
    assert batch["proposals"][0]["action"] == "confirm"


def test_confirm_final_only_last_generation_budget(tmp_path):
    run = _confirm_run(tmp_path)
    spec = _spec()
    spec.data["resource_budget"] = {"max_trials": 4}
    batch = _planner(run, spec=spec).plan(slots=4)
    assert "remaining_trials_le_one_generation" in \
        batch["confirm_policy"]["triggers"]
    assert _confirms(batch)


def test_confirm_per_generation_keeps_old_behaviour(tmp_path):
    run = _confirm_run(tmp_path)
    batch = _planner(run, _cfg(**{"confirm.policy": "per_generation"})
                     ).plan(slots=4)
    assert batch["confirm_policy"]["policy"] == "per_generation"
    assert batch["candidate_sources"].get("confirm") == 1


def test_confirm_undelivered_gain_without_high_value(tmp_path):
    run = _confirm_run(tmp_path)
    pl = _planner(run)
    # 无更多高分候选(others 为空)+ 冠军未交付且相对 baseline 增益 6×MDE
    d = pl._confirm_policy_decision([], requested_slots=4, remaining=50)
    assert "undelivered_gain_ge_mde_no_high_value" in d["triggers"]
    # 有高分候选时不触发
    hi = pl._base(action="explore", family="feat:x", hypothesis="h",
                  rationale="r", falsification="f",
                  expected_gain=2 * pl._mde())
    d2 = pl._confirm_policy_decision([hi], requested_slots=4, remaining=50)
    assert d2["nominate"] is False


# ---------------------------------------------------------------------------
# 6. planner-context:画像规则 + 天花板节
# ---------------------------------------------------------------------------

def test_planner_context_ceiling_and_profile_rules(tmp_path):
    from runtime.ceiling import write_ceiling
    from runtime.planner_context import (build_planner_context,
                                         render_planner_context)

    run = _run(tmp_path)
    _profile(run)
    cfg = _cfg()
    txt = render_planner_context(build_planner_context(run, _spec(), cfg))
    assert "## 13. 画像规则命中" in txt and "sequence_column" in txt
    assert "modelevo.py ceiling --run" in txt
    write_ceiling(run, {"primary_metric": "auc", "higher_is_better": True,
                        "ceiling_estimate": 0.81, "feature_surface": "base",
                        "references": {"gbdt": {"value": 0.81}}})
    txt = render_planner_context(build_planner_context(run, _spec(), cfg))
    assert "feature_surface=`base`" in txt
    assert "未加宽特征面上的模型侧上限" in txt
