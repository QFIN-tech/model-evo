"""v2.18.0 补丁批次回归:迁库完整性、离线快照、指纹注入、证据中性带、别名归一、
原则排除、配方归属、召回对称归一、推翻口径、租约与快照终态。

每一项对应一次实测缺陷(报告 §07 F1–F15),用来保证修好之后不再回来。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import json

import pytest

from runtime.library import (ArtifactStore, LibraryStore, LocalRepository)
from runtime.library import curation, evidence as EV, retrieval as RT
from runtime.library import schema as S
from runtime.library.curation import LeaseBusy
from runtime.library.snapshot import snapshot_id

ATID = "binary-y-200w_1000w-bc5329dc"
ATID2 = "reg-amt-200w_1000w-bc5329dc"
NS = "t"
BIN = {"model_type": "binary_classification", "primary_metric": "auc",
       "target_transform": "raw"}


def _repo(tmp_path, name="lib") -> LocalRepository:
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/{name}.db"),
                           ArtifactStore(f"file://{tmp_path}/{name}-cas"))


def _topics(repo, ns=NS):
    curation.init_topics(repo.store, ns, [
        {"code": "feature.seq", "definition": "序列", "aliases": ["seq_"]},
        {"code": "feature.seq.gap", "definition": "间隔", "aliases": ["seq_gap_d365"]},
        {"code": "feature.seq.window_count", "definition": "计数", "aliases": []},
    ], reason="v1", actor={})


def _run_snap(repo, atid, local_id="d1", claim="c", topics=("feature.seq",), ns=NS,
              run_id="r@1", extra=None):
    payload = {"schema": "modelevo.snapshot/1", "kind": "run",
               "run": {"run_id": run_id, "atid": atid},
               "discoveries": [{"local_id": local_id,
                                "payload": {"claim": claim, "topics": list(topics),
                                            **(extra or {})}}]}
    return repo.submit_snapshot(payload, namespace=ns)["snapshot_id"]


def _merge(repo, sid, actions, ns=NS, **kw):
    ctx = repo.merge_context(ns, sid)
    return repo.merge_apply(ns, {
        "schema": "modelevo.merge-decision/1", "snapshot_id": sid,
        "lease_token": ctx["lease"]["token"], "actions": actions, **kw})


def _p(claim="c", direction=1, topics=("feature.seq",), scope=None, **kw):
    return {"claim": claim, "direction": direction, "topics": list(topics),
            "scope": scope or BIN, **kw}


# ---------------------------------------------------------------- F1 迁库完整性

def test_dump_covers_every_table_except_leases():
    from runtime.library.dump import SKIP_TABLES, TABLES

    names = {t.name for t in S.metadata.sorted_tables} - set(SKIP_TABLES)
    assert {t.name for t in TABLES} == names
    assert "leases" not in {t.name for t in TABLES}


def test_dump_load_roundtrip_keeps_topics_dossiers_changelog_evidence(tmp_path):
    """旧实现漏了 topics / dossiers / changelog_entries / evidence_links:
    迁库后无法合并(词表为空)、丢失审计与全部证据加权,而且不报错。"""
    from runtime.library.dump import dump, load

    src = _repo(tmp_path, "a")
    _topics(src)
    rid = src.put_bundle({"train.py": b"x = 1\n"})
    src.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "run",
                         "run": {"run_id": "r@1", "atid": ATID},
                         "trial_outcomes": [{"run_id": "r@1", "trial_id": "t1",
                                             "atid": ATID, "primary_metric": "auc",
                                             "higher_is_better": True,
                                             "delta_parent_ci_low": -0.003,
                                             "delta_parent_ci_high": -0.0005}]},
                        namespace=NS)
    sid = _run_snap(src, ATID)
    _merge(src, sid, [{"local_id": "d1", "action": "add", "reason": "r",
                       "principle": _p()}])
    dump(src.store, src.artifacts, tmp_path / "dump")

    dst = _repo(tmp_path, "b")
    load(dst.store, dst.artifacts, tmp_path / "dump")
    assert len(dst.topics(NS)) == len(src.topics(NS))
    assert [p["principle_id"] for p in dst.principles(NS)] == \
        [p["principle_id"] for p in src.principles(NS)]
    assert len(dst.changelog(NS)) == len(src.changelog(NS))
    assert dst.dossier(NS, "task", ATID) is None
    pid = dst.principles(NS)[0]["principle_id"]
    assert len(dst.evidence(NS, pid)["links"]) == len(src.evidence(NS, pid)["links"])
    assert dst.evidence(NS, pid)["stats"] == src.evidence(NS, pid)["stats"]


def test_load_rejects_partial_manifest(tmp_path):
    """清单里有的表没导全时必须报错,不能静默少行。"""
    from runtime.library.dump import dump, load

    src = _repo(tmp_path, "a")
    _topics(src)
    dump(src.store, src.artifacts, tmp_path / "dump")
    (tmp_path / "dump" / "tables" / "topics.jsonl").write_text("", encoding="utf-8")
    dst = _repo(tmp_path, "b")
    with pytest.raises(ValueError, match="行数"):
        load(dst.store, dst.artifacts, tmp_path / "dump")


# ---------------------------------------------------------------- F2 离线快照

def test_snapshot_build_never_touches_network(tmp_path):
    """构建阶段不得访问服务:服务不可达时收尾仍须能产出带配方的快照。"""
    from _mvo.library_snapshot import build_run_snapshot

    run = tmp_path / "run"
    (run / "trials" / "t1" / "code").mkdir(parents=True)
    (run / "trials" / "t1" / "config").mkdir(parents=True)
    (run / "run-state.json").write_text(json.dumps({"abstract_task_id": ATID}))
    (run / "trials" / "t1" / "trial.yaml").write_text(
        "trial_id: t1\nalgorithm: xgboost\nsearch_action: explore\n")
    (run / "trials" / "t1" / "config" / "trial.yaml").write_text("trial_id: t1\n")
    (run / "trials" / "t1" / "code" / "train.py").write_text(
        f"import pathlib\nRUN_DIR = pathlib.Path('{run}')\n")
    (run / "trials" / "t1" / "evaluation.json").write_text(json.dumps(
        {"verdict": "accepted", "primary_metric": {"name": "auc", "value": 0.8}}))

    class Boom:
        def __getattr__(self, name):
            raise AssertionError(f"构建阶段不得调用 repo.{name}")

    staged: dict = {}
    payload, rep = build_run_snapshot(run, Boom(), stage=staged)
    assert rep["recipes"] == 1 and set(staged) == {payload["recipes"][0]["recipe_id"]}


def test_queue_carries_recipes_and_submit_pending_uploads_first(tmp_path, monkeypatch):
    """outbox 必须连配方一起落盘;补交时先传产物再提交快照。"""
    from _mvo import library_snapshot as LS
    from runtime.library import HttpRepository, LibraryUnavailable

    run = tmp_path / "run"
    (run / "trials" / "t1" / "code").mkdir(parents=True)
    (run / "trials" / "t1" / "config").mkdir(parents=True)
    (run / "run-state.json").write_text(json.dumps({"abstract_task_id": ATID}))
    (run / "trials" / "t1" / "trial.yaml").write_text(
        "trial_id: t1\nalgorithm: xgboost\n")
    (run / "trials" / "t1" / "config" / "trial.yaml").write_text("trial_id: t1\n")
    (run / "trials" / "t1" / "code" / "train.py").write_text("x = 1\n")
    (run / "trials" / "t1" / "evaluation.json").write_text(json.dumps(
        {"verdict": "accepted", "primary_metric": {"name": "auc", "value": 0.8}}))

    staged: dict = {}
    payload, _rep = LS.build_run_snapshot(run, None, stage=staged)
    down = HttpRepository("http://127.0.0.1:9", timeout=1)
    res = LS.submit_or_queue(run, down, payload, namespace=NS, staged=staged)
    assert res["status"] == "queued" and res["queued_recipes"] == 1
    rid = next(iter(staged))
    sid = snapshot_id(payload)
    assert (run / "experience" / "outbox" / "cas" / sid[:12] / rid / "train.py").exists()

    repo = _repo(tmp_path, "lib")
    sent = LS.submit_pending(run, repo)
    assert len(sent) == 1 and sent[0]["uploaded_recipes"] == 1
    assert repo.has_artifact(rid)


# ---------------------------------------------------------------- F3 指纹注入

def test_merge_injects_snapshot_atid_into_evidence(tmp_path):
    """原则必须自带可核验指纹,否则 same_task 权重(检索首档 / 避免清单 / 方向表)
    对新知识永久失效。指纹由程序写,不依赖 LLM 在决议里填。"""
    repo = _repo(tmp_path)
    _topics(repo)
    sid = _run_snap(repo, ATID)
    _merge(repo, sid, [{"local_id": "d1", "action": "add", "reason": "r",
                        "principle": _p(evidence={"task_ids": ["req-t7"]})}])
    p = repo.principles(NS)[0]
    assert p["evidence"]["task_ids"] == ["req-t7"]                  # 原有字段保留
    assert p["evidence"]["abstract_task_ids"] == [ATID]             # 指纹补上
    assert EV.base_relation(ATID, p, lambda a: a) == "same_task"


def test_merge_keeps_existing_atids_and_dedupes(tmp_path):
    repo = _repo(tmp_path)
    _topics(repo)
    sid = _run_snap(repo, ATID)
    _merge(repo, sid, [{"local_id": "d1", "action": "add", "reason": "r",
                        "principle": _p(evidence={
                            "abstract_task_ids": [ATID, ATID2]})}])
    assert repo.principles(NS)[0]["evidence"]["abstract_task_ids"] == [ATID, ATID2]


# ---------------------------------------------------------------- F4 中性带

def test_direction_zero_has_neutral_band():
    flat = {"delta_parent_ci_low": -0.001, "delta_parent_ci_high": 0.001,
            "higher_is_better": True}
    assert EV.relation_for(0, flat) == "neutral"
    assert EV.relation_for(0, {"delta_parent_ci_low": -0.003,
                               "delta_parent_ci_high": -0.001,
                               "higher_is_better": True}) == "support"
    assert EV.relation_for(0, {"delta_parent_ci_low": 0.001,
                               "delta_parent_ci_high": 0.003,
                               "higher_is_better": True}) == "refute"


def test_aggregate_recomputes_relation_for_stored_links():
    """已入库关联沿用旧判定;汇总时按 CI 重算,存量立即纠偏,关联行不动。"""
    stale = [{"principle_id": "p", "run_id": "r1", "trial_id": "t1", "binding": "passive",
              "relation": "support", "base_relation": "same_task", "weight": 1.0,
              "ci_low": -0.001, "ci_high": 0.001, "trial_at": None}]
    assert EV.aggregate(stale)["p"]["w_support"] == 1.0                      # 旧口径
    fixed = EV.aggregate(stale, directions={"p": 0})["p"]
    assert fixed["w_support"] == 0.0 and fixed["w_neutral"] == 1.0           # 新口径
    # direction=+1 的原则不受影响
    assert EV.aggregate(stale, directions={"p": 1})["p"]["w_neutral"] == 1.0


# ---------------------------------------------------------------- F6 别名归一

def test_search_resolves_alias_on_principle_side():
    alias = [{"code": "feature.seq.window_count", "status": "active", "aliases": []},
             {"code": "feature.seq.count", "status": "alias_of",
              "alias_of": "feature.seq.window_count"}]
    view = {"principles": [{"principle_id": "p1", "version": 1, "direction": 1,
                            "topics": ["feature.seq.count"], "objects": [],
                            "claim": "x", "scope": BIN, "evidence": {}}],
            "topics": alias, "stats": {}, "atid": ATID, "scope": BIN,
            "aliases": {}, "fingerprints": [], "trial_outcomes": []}
    assert [c["principle_id"] for c in
            RT.search(view, topics=["feature.seq.window_count"], k=3)] == ["p1"]


def test_passive_evidence_matches_aliased_topic():
    topics = {"feature.seq.count": {"code": "feature.seq.count", "aliases": [],
                                    "status": "alias_of",
                                    "alias_of": "feature.seq.window_count"},
              "feature.seq.window_count": {"code": "feature.seq.window_count",
                                           "aliases": [], "status": "active"}}
    resolver = curation._topic_resolver(topics)
    p = {"principle_id": "p1", "version": 1, "direction": 1,
         "topics": ["feature.seq.count"], "objects": [], "category": "feature_engineering",
         "scope": BIN, "claim": "c", "evidence": {"abstract_task_ids": [ATID]}}
    trial = {"run_id": "r@1", "trial_id": "t1", "atid": ATID, "primary_metric": "auc",
             "higher_is_better": True, "topics": ["feature.seq.window_count"],
             "family_hints": [], "changed_modules": [], "delta_parent": 0.002,
             "delta_parent_ci_low": 0.001, "delta_parent_ci_high": 0.003,
             "verdict": "accepted"}
    links = EV.build_links([trial], [p], resolver, lambda a: a)
    assert [(lk["principle_id"], lk["relation"]) for lk in links] == [("p1", "support")]


# ---------------------------------------------------------------- F7 原则排除

def test_principle_exclude_hides_and_restore_brings_back(tmp_path):
    repo = _repo(tmp_path)
    _topics(repo)
    sid = _run_snap(repo, ATID)
    _merge(repo, sid, [{"local_id": "d1", "action": "add", "reason": "r",
                        "principle": _p()}])
    pid = repo.principles(NS)[0]["principle_id"]
    repo.govern(NS, [{"subject_type": "principle", "subject_id": pid,
                      "action": "exclude", "reason": "结论有误"}])
    assert repo.principles(NS) == []
    st = {p["principle_id"]: p for p in repo.principles(NS, include_inactive=True)}
    assert st[pid]["status"] == "excluded" and st[pid]["excluded_reason"] == "结论有误"
    repo.govern(NS, [{"subject_type": "principle", "subject_id": pid,
                      "action": "restore", "reason": "复核后恢复"}])
    assert [p["principle_id"] for p in repo.principles(NS)] == [pid]


# ---------------------------------------------------------------- F8 配方归属

def test_same_recipe_in_second_task_gets_its_own_baseline(tmp_path):
    """recipe_id 是内容哈希,同一 baseline 被两个任务复用时,第二行插不进去 ——
    旧实现让第二个任务永远固化不了 baseline、榜单缺行,而且没有报错。"""
    from runtime.library.snapshot import SNAPSHOT_SCHEMA

    repo = _repo(tmp_path)
    rid = repo.put_bundle({"train.py": b"x = 1\n"})
    for atid in (ATID, ATID2):
        repo.submit_snapshot({
            "schema": SNAPSHOT_SCHEMA, "kind": "baseline",
            "run": {"run_id": f"r@{atid[-8:]}", "atid": atid},
            "recipes": [{"recipe_id": rid, "atid": atid, "kind": "baseline",
                         "algorithm": "xgboost", "is_hpo": False}],
            "measurements": [{"recipe_id": rid, "kind": "baseline", "value": 0.8}]},
            namespace=NS)
    assert {r["atid"] for r in repo.recipe_atids(NS, recipe_id=rid)} == {ATID, ATID2}
    assert repo.frozen_baseline(NS, ATID, "xgboost")["recipe_id"] == rid
    assert repo.frozen_baseline(NS, ATID2, "xgboost")["recipe_id"] == rid
    assert repo.recipes(NS, atid=ATID2, kind="baseline")[0]["atid"] == ATID2
    # 不相干的任务看不到
    assert repo.recipes(NS, atid="binary-zz-200w_1000w-deadbeef") == []


# ---------------------------------------------------------------- F9 召回

def test_lexical_score_is_symmetric_not_min_normalized():
    from runtime.library.curation import lexical_score, tokens

    short = tokens("窗口计数有效")
    long_ = tokens("事件×窗口计数网格在联合面上无独立增量,已饱和")
    assert lexical_score(short, long_) < 0.3        # 旧 overlap 会给 0.6
    assert lexical_score(long_, long_) == 1.0
    assert lexical_score(short, long_) == lexical_score(long_, short)


def test_merge_recall_prefers_shared_topic_over_lexical_noise(tmp_path):
    repo = _repo(tmp_path)
    _topics(repo)
    sid = _run_snap(repo, ATID, local_id="base", claim="计数网格无独立增量",
                    topics=["feature.seq.window_count"])
    _merge(repo, sid, [{"local_id": "base", "action": "add", "reason": "r",
                        "principle": _p(claim="事件×窗口计数网格在联合面上无独立增量,已饱和",
                                        direction=0,
                                        topics=["feature.seq.window_count"])}])
    # 新候选写得短,但与已有原则同主题:必须被召回
    sid2 = _run_snap(repo, ATID, local_id="new", claim="窗口计数有效",
                     topics=["feature.seq.window_count"], run_id="r@2")
    ctx = repo.merge_context(NS, sid2)
    related = ctx["discoveries"][0]["related"]
    assert related and related[0] == repo.principles(NS)[0]["principle_id"]


def test_merge_recall_skips_incompatible_scope(tmp_path):
    repo = _repo(tmp_path)
    _topics(repo)
    sid = _run_snap(repo, ATID, topics=["feature.seq"])
    _merge(repo, sid, [{"local_id": "d1", "action": "add", "reason": "r",
                        "principle": _p(topics=["feature.seq"])}])
    # 候选写成回归口径:口径不同本来就不允许 merge_into,不应进召回
    sid2 = _run_snap(repo, ATID, topics=["feature.seq"], run_id="r@2",
                     extra={"scope": {"model_type": "regression",
                                      "primary_metric": "rmse",
                                      "target_transform": "raw"}})
    ctx = repo.merge_context(NS, sid2)
    assert ctx["discoveries"][0]["related"] == []


# ---------------------------------------------------------------- F12 推翻口径

def test_overturn_requires_full_scope_match(tmp_path):
    repo = _repo(tmp_path)
    _topics(repo)
    sid = _run_snap(repo, ATID)
    _merge(repo, sid, [{"local_id": "d1", "action": "add", "reason": "r",
                        "principle": _p(claim="计数网格无增益", direction=0,
                                        topics=["feature.seq"])}])
    pid = repo.principles(NS)[0]["principle_id"]
    repo.submit_snapshot({
        "schema": "modelevo.snapshot/1", "kind": "run",
        "run": {"run_id": "r@2", "atid": ATID},
        "trial_outcomes": [{"run_id": "r@2", "trial_id": "t1", "atid": ATID,
                            "primary_metric": "auc", "higher_is_better": True,
                            "delta_parent_ci_low": 0.002,
                            "delta_parent_ci_high": 0.006}]}, namespace=NS)
    sid2 = _run_snap(repo, ATID, local_id="d2", run_id="r@3", topics=["feature.seq"])
    act = {"local_id": "d2", "action": "supersede", "target": pid, "reason": "新证据",
           "principle": _p(claim="计数网格有效", topics=["feature.seq"]),
           "evidence_trials": [{"run_id": "r@2", "trial_id": "t1"}]}
    # 指标相同但 target_transform 不同(log1p vs raw):不得作为推翻依据
    bad = {**act, "principle": _p(claim="计数网格有效", topics=["feature.seq"],
                                  scope={**BIN, "target_transform": "log1p"})}
    ctx = repo.merge_context(NS, sid2)
    decision = {"schema": "modelevo.merge-decision/1", "snapshot_id": sid2,
                "lease_token": ctx["lease"]["token"], "actions": [bad]}
    with pytest.raises(Exception, match="口径"):
        repo.merge_apply(NS, decision)
