"""v2.18.0 中心化经验库(M3):证据关联、证据加权、检索、run 视图与固定水位。

docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md §7 §9 §17。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import json

import pytest

from runtime.library import ArtifactStore, LibraryStore, LocalRepository
from runtime.library import evidence as EV
from runtime.library import retrieval as RT
from runtime.library.curation import _topic_resolver

A = "binary-y-200w_1000w-bc5329dc"
A_OTHER_TASK = "reg-amt-200w_1000w-bc5329dc"
A_OTHER_BASE = "binary-z-200w_1000w-deadbeef"
BIN = {"model_type": "binary_classification", "primary_metric": "auc", "target_transform": "raw"}
NS = "t"
TOPICS = {
    "feature": {"code": "feature", "aliases": []},
    "feature.seq": {"code": "feature.seq", "aliases": ["seq_"]},
    "feature.seq.gap": {"code": "feature.seq.gap", "aliases": ["seq_gap_d365", "seqg_"]},
    "feature.seq.window_count": {"code": "feature.seq.window_count", "aliases": ["seqd_"]},
    "feature.base.interaction": {"code": "feature.base.interaction", "aliases": ["xint_"]},
    "hpo.search": {"code": "hpo.search", "aliases": []},
    "eval.confirm": {"code": "eval.confirm", "aliases": []},
}
RES = _topic_resolver(TOPICS)


def _p(pid, direction, topics, *, atids=(A,), objects=(), category="feature_engineering",
       scope=BIN, claim=None):
    return {"principle_id": pid, "version": 1, "direction": direction, "topics": list(topics),
            "objects": list(objects), "category": category, "scope": scope,
            "claim": claim or f"{pid} claim", "condition": {},
            "evidence": {"abstract_task_ids": list(atids)}, "limitations": []}


def _t(tid, *, hints=(), lo=None, hi=None, verdict="accepted", atid=A, metric="auc",
       topics=None, **kw):
    return {"run_id": "r@1", "trial_id": tid, "atid": atid, "primary_metric": metric,
            "higher_is_better": metric != "rmse", "family_hints": list(hints),
            "delta_parent": None if lo is None else (lo + hi) / 2,
            "delta_parent_ci_low": lo, "delta_parent_ci_high": hi, "verdict": verdict,
            "topics": topics, "changed_modules": [], **kw}


# ---------------------------------------------------------------- 主题推断

def test_infer_trial_topics():
    tp, obj = EV.infer_trial_topics(_t("x", hints=["info:seq_gap_d365:window"]), RES)
    assert tp == ["feature.seq.gap"] and obj == ["seq_gap_d365"]
    tp, _ = EV.infer_trial_topics(_t("x", hints=["info:dm_pboc_cnt:derived_interaction"]), RES)
    assert tp == ["feature.base.interaction"]
    # 联合面:新增族在末尾,倒序读;去掉被子主题覆盖的父主题
    tp, obj = EV.infer_trial_topics(_t("x", hints=["seq_", "seqd_", "xint_"]), RES)
    assert tp[0] == "feature.base.interaction" and "feature.seq" not in tp
    assert "xint_" in obj
    tp, _ = EV.infer_trial_topics(_t("x", action="hpo"), RES)
    assert tp == ["hpo.search"]
    tp, _ = EV.infer_trial_topics(_t("x", topics=["seqg_"]), RES)
    assert tp == ["feature.seq.gap"]


def test_relation_semantics():
    sig_pos = _t("a", lo=0.001, hi=0.003)
    sig_neg = _t("b", lo=-0.003, hi=-0.001)
    flat = _t("c", lo=-0.001, hi=0.001)
    assert [EV.relation_for(1, x) for x in (sig_pos, sig_neg, flat)] == \
        ["support", "refute", "neutral"]
    # direction=0 有中性带:CI 跨 0 的 trial 什么也没证明,既非支持也非反驳
    # (旧实现记 support,把无分辨力的 trial 当成"支持无增益")
    assert [EV.relation_for(0, x) for x in (sig_pos, sig_neg, flat)] == \
        ["refute", "support", "neutral"]
    assert [EV.relation_for(-1, x) for x in (sig_pos, sig_neg, flat)] == \
        ["refute", "support", "neutral"]
    # 越小越好的指标:rmse 下降(Δ<0)是增益
    rm = _t("d", lo=-0.003, hi=-0.001, metric="rmse")
    assert EV.relation_for(1, rm) == "support"
    assert EV.relation_for(1, _t("e", verdict="promising")) is None


# ---------------------------------------------------------------- 关联

def test_build_links_rules():
    ps = [_p("gap0", 0, ["feature.seq.gap"]),
          _p("seq1", 1, ["feature.seq"]),                       # 父主题:被动证据不匹配
          _p("conf", -1, ["eval.confirm"], category="evaluation"),
          _p("gapreg", 0, ["feature.seq.gap"],
             scope={**BIN, "model_type": "regression", "primary_metric": "rmse"}),
          _p("objs", -1, ["feature.seq.gap"], objects=["seqg_"])]
    t = _t("t1", hints=["info:seq_gap_d365:window"], lo=-0.003, hi=-0.0005)
    links = EV.build_links([t], ps, RES, lambda a: a)
    got = {(lk["principle_id"], lk["binding"], lk["relation"]) for lk in links}
    assert ("gap0", "passive", "support") in got
    # direction=0 的 trial 若 CI 跨 0,记中性(不再算 support)
    flat = EV.build_links([_t("t2", hints=["info:seq_gap_d365:window"],
                              lo=-0.001, hi=0.0005)], ps, RES, lambda a: a)
    assert {(lk["principle_id"], lk["relation"]) for lk in flat} == \
        {("gap0", "neutral"), ("objs", "neutral")}
    assert not {x[0] for x in got} & {"seq1", "conf", "gapreg"}
    w = {lk["principle_id"]: lk["weight"] for lk in links}
    assert w["objs"] == pytest.approx(w["gap0"] / 2)          # 同类动作、不同对象:减半
    # 显式依据:任何类别都记,且优先于被动
    t2 = _t("t2", hints=["info:seq_gap_d365:window"], lo=0.001, hi=0.002,
            extra={"basis": {"novel": False, "refs": [{"id": "conf", "relation": "tests"},
                                                      {"id": "gap0", "relation": "contradicts"}]}})
    links = EV.build_links([t2], ps, RES, lambda a: a)
    exp = {lk["principle_id"]: lk for lk in links if lk["binding"] == "explicit"}
    assert set(exp) == {"conf", "gap0"} and exp["gap0"]["relation"] == "refute"
    assert exp["gap0"]["declared"] == "contradicts"
    assert not [lk for lk in links if lk["binding"] == "passive" and lk["principle_id"] == "gap0"]


def test_base_relation_and_weights():
    canon = lambda a: {"alias-x": A}.get(a, a)  # noqa: E731
    p = _p("x", 1, ["feature.seq.gap"])
    assert EV.base_relation(A, p, canon) == "same_task"
    assert EV.base_relation("alias-x", p, canon) == "same_task"
    assert EV.base_relation(A_OTHER_TASK, p, canon) == "same_base"
    assert EV.base_relation(A_OTHER_BASE, p, canon) == "cross_base"
    assert EV.base_relation(A, _p("y", 1, [], atids=()), canon) == "unknown"


def test_aggregate_confidence_labels():
    mk = lambda rel, n, b="passive": [{"principle_id": "p", "run_id": f"r{i}", "trial_id": "t",  # noqa: E731
                                       "binding": b, "relation": rel, "base_relation": "same_task",
                                       "weight": 1.0, "trial_at": None} for i in range(n)]
    assert EV.aggregate(mk("support", 4))["p"]["confidence"] == "high"
    assert EV.aggregate(mk("refute", 3))["p"]["confidence"] == "mixed"
    assert EV.aggregate(mk("refute", 3, "explicit"))["p"]["confidence"] == "contradicted"
    assert EV.aggregate(mk("neutral", 3))["p"]["confidence"] == "low"


# ---------------------------------------------------------------- 检索

def _view(principles, **kw):
    return {"atid": A, "scope": BIN, "topics": list(TOPICS.values()), "aliases": {},
            "fingerprints": [{"atid": A, "data_base": "200w_1000w-bc5329dc",
                              "sample_scale": "200w_1000w"},
                             {"atid": A_OTHER_BASE, "data_base": "200w_1000w-deadbeef",
                              "sample_scale": "200w_1000w"}],
            "principles": principles, "stats": kw.get("stats", {}),
            "trial_outcomes": kw.get("trials", [])}


def test_search_scoring_and_filters():
    ps = [_p("same", 0, ["feature.seq.gap"]),
          _p("parent", 1, ["feature.seq"]),
          _p("othertask", 0, ["feature.seq.gap"], atids=(A_OTHER_TASK,)),
          _p("wrongscope", 0, ["feature.seq.gap"], scope={**BIN, "primary_metric": "ks"}),
          _p("x1", 0, ["feature.seq.gap"], atids=(A_OTHER_BASE,), objects=["seqg_"]),
          _p("x2", 1, ["feature.seq.gap"], atids=(A_OTHER_BASE,), objects=["seqg_"]),
          _p("x3", -1, ["feature.seq.gap"], atids=(A_OTHER_BASE,), objects=["seqg_"]),
          _p("anchor", 1, ["feature.seq.window_count"], objects=["seqg_"])]
    v = _view(ps)
    cards = RT.search(v, topics=["seqg_"], k=10)
    ids = [c["principle_id"] for c in cards]
    assert ids[0] == "same"
    assert ids.index("othertask") < ids.index("parent")         # 同底座精确 > 同任务父主题
    # v2.18.1:指标同族(ks↔auc)不再硬过滤,进 L2 放宽档,排在全部精确档之后
    by = {c["principle_id"]: c for c in cards}
    assert by["wrongscope"]["tier"] == "L2" and by["wrongscope"]["scope_match"] == "metric_family"
    strict = [c for c in cards if c["tier"] in ("L0", "L1")]
    assert ids.index("wrongscope") >= len(strict)
    cross = [c for c in cards if c["base_relation"] == "cross_base"]
    assert len(cross) == 2 and all(c["objects"] == [] for c in cross)
    assert {c["label"] for c in cards} >= {"同任务", "同数据·跨任务", "跨底座"}
    # 跨底座相似度不足:非冷启动时不出;冷启动(本任务没有任何精确 / 相关档)时作 L3 兜底
    v2 = _view([_p("far", 0, ["feature.seq.gap"], atids=(A_OTHER_BASE,)),
                _p("mine", 0, ["feature.seq.gap"])])
    assert [c["principle_id"] for c in RT.search(v2, topics=["feature.seq.gap"])] == ["mine"]
    v2b = _view([_p("far", 0, ["feature.seq.gap"], atids=(A_OTHER_BASE,))])
    got = RT.search(v2b, topics=["feature.seq.gap"])
    assert [c["tier"] for c in got] == ["L3"] and "冷启动" in "".join(got[0]["why"])
    # 口径完全不同(回归 rmse)的原则任何档都不出
    v2c = _view([_p("reg", 0, ["feature.seq.gap"],
                    scope={"model_type": "regression", "primary_metric": "rmse",
                           "target_transform": "raw"})])
    assert RT.search(v2c, topics=["feature.seq.gap"]) == []
    # 无主题命中时靠假设文本重叠
    v3 = _view([_p("txt", 1, ["hpo.search"], claim="容量上限 与 学习率 耦合 联合调整")])
    assert RT.search(v3, topics=["feature.seq.gap"]) == []
    assert RT.search(v3, hypothesis="容量上限与学习率耦合需要联合调整")[0]["principle_id"] == "txt"


def test_direction_table_and_failed():
    trials = [_t("t1", hints=["info:seq_gap_d365:window"], lo=0.001, hi=0.003),
              _t("t2", hints=["xint_"], lo=-0.002, hi=-0.001, verdict="rejected",
                 failure_reason="no gain"),
              _t("t3", hints=["xint_"], lo=-0.004, hi=-0.002, verdict="rejected")]
    v = _view([], trials=trials)
    dt = RT.direction_table(v)
    assert dt["effective"][0]["trial_id"] == "t1"
    assert dt["failed"][0]["trial_id"] == "t3" and dt["failed"][0]["n_trials"] == 2
    assert RT.failed_for_topics(v, ["feature.base.interaction"])[0]["trial_id"] == "t3"
    assert RT.failed_for_topics(v, ["feature.seq.gap"]) == []


# ---------------------------------------------------------------- 仓库:提交 / 合并后自动关联

def _repo(tmp_path) -> LocalRepository:
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _seed_principles(repo):
    repo.init_topics(NS, [{"code": "feature.seq", "definition": "d", "aliases": ["seq_"]},
                          {"code": "feature.seq.gap", "definition": "d",
                           "aliases": ["seq_gap_d365"]}], reason="v1")
    sid = repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_principles",
                                "run": {"run_id": "legacy:x", "atid": A},
                                "discoveries": [{"local_id": "d1", "payload": {"claim": "c"}}]},
                               namespace=NS)["snapshot_id"]
    ctx = repo.merge_context(NS, sid)
    return repo.merge_apply(NS, {
        "schema": "modelevo.merge-decision/1", "snapshot_id": sid,
        "lease_token": ctx["lease"]["token"],
        "actions": [{"local_id": "d1", "action": "add", "reason": "r", "principle": {
            "principle_id": "p-gap", "claim": "间隔族无独立增量", "direction": 0,
            "topics": ["feature.seq.gap"], "scope": BIN,
            "evidence": {"abstract_task_ids": [A]}}}]})


def test_submit_and_merge_create_evidence(tmp_path):
    repo = _repo(tmp_path)
    # CI 整体为负 → 对 direction=0 的原则是"显著无增益",记 support(旧数据用跨 0 的
    # CI 也能记 support,是 v2.18.0 补丁前的语义)
    trial = {"run_id": "r@1", "trial_id": "t1", "family_hints": ["info:seq_gap_d365:window"],
             "primary_metric": "auc", "verdict": "rejected", "delta_parent_ci_low": -0.003,
             "delta_parent_ci_high": -0.0005}
    rec = repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_evidence",
                                "run": {"run_id": "r@1", "atid": A},
                                "trial_outcomes": [trial]}, namespace=NS)
    assert rec["evidence"]["links"] == 0                         # 还没有原则
    merged = _seed_principles(repo)
    assert merged["evidence"]["links"] == 1                      # 合并后回补历史 trial
    ev = repo.evidence(NS, "p-gap")
    assert ev["stats"]["w_support"] == 0.5 and ev["links"][0]["binding"] == "passive"
    rec2 = repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "run",
                                 "run": {"run_id": "r@2", "atid": A},
                                 "trial_outcomes": [{**trial, "run_id": "r@2"}]}, namespace=NS)
    assert rec2["evidence"]["links"] == 1
    assert repo.relink(NS)["links"] == 0                         # 幂等


def test_run_view_contents(tmp_path):
    repo = _repo(tmp_path)
    _seed_principles(repo)
    repo.add_fingerprint_alias("binary-y-200w_1000w-7580b665", A, reason="漂移")
    rid = repo.put_bundle({"train.py": b"x = 1\n"})
    repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "baseline",
                          "run": {"run_id": "r@1", "atid": A},
                          "recipes": [{"recipe_id": rid, "kind": "baseline",
                                       "algorithm": "xgboost"}],
                          "governance": [{"subject_type": "recipe", "subject_id": rid,
                                          "action": "pin_baseline"}]}, namespace=NS)
    v = repo.run_view(NS, "binary-y-200w_1000w-7580b665")
    assert v["atid"] == A and v["data_base"] == "200w_1000w-bc5329dc"
    assert [p["principle_id"] for p in v["principles"]] == ["p-gap"]
    assert v["baselines"]["xgboost"]["recipe_id"] == rid
    w = v["watermark"]
    repo.govern(NS, [{"subject_type": "recipe", "subject_id": rid, "action": "exclude"}])
    assert repo.run_view(NS, A, at=w)["watermark"] == w


# ---------------------------------------------------------------- run 侧视图

def test_library_view_pin_and_fallback(tmp_path, monkeypatch):
    from runtime import library_view as LV

    repo = _repo(tmp_path / "lib")
    _seed_principles(repo)
    rid = repo.put_bundle({"train.py": b"import pathlib\nRUN_DIR = pathlib.Path("
                                       b"'{{MODELEVO_RUN_DIR}}')\nSKILL_ROOT = '{{MODELEVO_SKILL_ROOT}}'\n"})
    repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_recipes",
                          "run": {"run_id": "x", "atid": A},
                          "recipes": [{"recipe_id": rid, "kind": "top_trial", "algorithm": "xgboost"}],
                          "measurements": [{"recipe_id": rid, "kind": "legacy", "gain": 0.01}]},
                         namespace=NS)
    monkeypatch.setenv("MODELEVO_LIBRARY_NS", NS)
    monkeypatch.setenv("MODELEVO_LIBRARY_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(LV, "client", lambda settings, **kw: repo)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run-state.json").write_text("{}")
    spec = {"model_type": "binary_classification", "primary_metric": "auc",
            "target": {"name": "y"}}
    v1 = LV.ensure(run, spec=spec)
    w = v1.watermark
    assert v1.pinned and not v1.stale and v1.atid is None and len(v1.principles) == 1
    repo.govern(NS, [{"subject_type": "recipe", "subject_id": rid, "action": "exclude"}])
    v2 = LV.ensure(run, atid=A, spec=spec)                       # 同一水位补取任务数据
    assert v2.watermark == w and v2.atid == A and len(v2.leaderboard) == 1
    errs = v2.materialize(rid, run / "t", run_dir=run, skill_root="/skill")
    assert errs == [] and f"pathlib.Path('{run}')" in (run / "t" / "train.py").read_text()
    st = json.loads((run / "run-state.json").read_text())["library"]
    assert st["watermark"] == w and st["source"] == "server"
    assert LV.load(run).search(topics=["seq_gap_d365"])[0]["principle_id"] == "p-gap"
    # 服务不可达:用本机缓存(stale);无缓存:空视图
    from runtime.library import LibraryUnavailable

    def down(settings, **kw):
        class R:
            def run_view(self, *a, **k):
                raise LibraryUnavailable("down")
        return R()
    monkeypatch.setattr(LV, "client", down)
    run2 = tmp_path / "run2"
    run2.mkdir()
    (run2 / "run-state.json").write_text("{}")
    v3 = LV.ensure(run2, atid=A, spec=spec)
    assert v3.stale and v3.watermark == w and v3.materialize(rid, run2 / "t", run_dir=run2,
                                                             skill_root="/s") == []
    monkeypatch.setenv("MODELEVO_LIBRARY_CACHE", str(tmp_path / "empty"))
    run3 = tmp_path / "run3"
    run3.mkdir()
    (run3 / "run-state.json").write_text("{}")
    v4 = LV.ensure(run3, atid=A)
    assert v4.watermark is None and v4.principles == [] and not v4.pinned


def test_parse_basis():
    sys.path.insert(0, str(SKILL / "scripts"))
    from _mvo.cmd_trial import _parse_basis

    assert _parse_basis(None, []) == (None, None)
    assert _parse_basis("novel", [])[0] == {"novel": True, "refs": []}
    b, e = _parse_basis("p-a:tests,p-b", ["p-c"])
    assert e is None and b["refs"] == [{"id": "p-a", "relation": "tests"},
                                       {"id": "p-b", "relation": "follows"},
                                       {"id": "p-c", "relation": "follows"}]
    assert _parse_basis("p-a:bogus", [])[1]
    assert _parse_basis("novel,p-a", [])[1]
    assert _parse_basis("novel", ["p-a"])[1]
