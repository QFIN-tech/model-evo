"""v2.18.2 第 1 期:业务域受控词表、证据汇总的可选口径、检索回放评测。

docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md §22。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import pytest

from runtime.library import ArtifactStore, LibraryStore, LocalRepository
from runtime.library import curation as CU
from runtime.library import domains as DOM
from runtime.library import evidence as EV
from runtime.library import replay as RP
from runtime.library import task_profile as TP

NS = "t"
A = "binary-y-200w_1000w-bc5329dc"
BIN = {"model_type": "binary_classification", "primary_metric": "auc", "target_transform": "raw"}


# ---------------------------------------------------------------- 业务域词表

def test_vocabulary_is_well_formed():
    voc = DOM.vocabulary()
    assert voc, "词表不能为空"
    seen: dict[str, str] = {}
    for code, d in voc.items():
        assert code.count(".") <= 1, f"{code}: 最多两层"
        if d["parent"]:
            assert d["parent"] in voc, f"{code}: 父节点 {d['parent']} 不在词表中"
        # 每个写法只能指向一个词,否则归一结果取决于文件顺序
        for name in [code, *d["aliases"]]:
            k = DOM._key(name)
            assert seen.get(k, code) == code, f"{name!r} 同时是 {seen[k]} 与 {code} 的写法"
            seen[k] = code


def test_resolve_aliases_labels_and_unknown():
    assert DOM.resolve("credit_loan") == ("credit.loan", True)
    assert DOM.resolve("Consumer-Credit") == ("credit", True)
    assert DOM.resolve("user_ops") == ("user_operation", True)
    assert DOM.resolve("获客") == ("credit.acquisition", True)       # 中文名的一半
    assert DOM.resolve("Space Mining") == ("space_mining", False)
    assert DOM.normalize(["credit_loan", "loan", "", "user ops"]) == ["credit.loan",
                                                                    "user_operation"]
    assert DOM.unknown(["credit", "space mining"]) == ["space_mining"]


def test_hierarchical_similarity():
    assert DOM.similarity(["user_ops"], ["user_operation", "credit_loan"]) == 1.0
    assert DOM.similarity(["credit"], ["loan_drawdown"]) == DOM.LINEAGE
    assert DOM.similarity(["credit.drawdown"], ["credit_loan"]) == DOM.SIBLING
    assert DOM.similarity(["marketing"], ["credit"]) == 0.0
    assert DOM.similarity(["foo"], ["foo"]) == 1.0                   # 未知写法仍可精确匹配
    assert DOM.similarity([], ["credit"]) is None
    # 无层级关系时与 v2.18.1 的重合系数一致
    assert DOM.similarity(["a_x"], ["a_x", "b_y", "c_z"]) == 1.0


def test_task_similarity_uses_vocabulary():
    a = TP.build_profile(spec_data={"business_domains": ["user_ops"], "target": {"name": "y"}})
    assert a["business_domains"] == ["user_operation"]            # 写入画像时归一
    # 存量画像里的原文(未归一)也按词表比较
    _sim, dims = TP.similarity({"business_domains": ["user_operation"]},
                               {"business_domains": ["User-Operations"]})
    assert dims["domain"] == 1.0


def test_merge_normalizes_condition_domains():
    errors: list = []
    p = CU._norm_principle({"claim": "c", "direction": 1, "topics": ["feature.seq"],
                            "scope": BIN,
                            "condition": {"business_domains": ["credit_loan", "space mining"],
                                          "data_modalities": ["event_sequence"]}},
                           {"feature.seq": "feature.seq"}, errors, "a1", set())
    assert errors == []
    assert p["condition"] == {"business_domains": ["credit.loan", "space_mining"],
                              "data_modalities": ["event_sequence"]}


def test_spec_rejects_non_list_business_domains():
    from runtime.spec import TaskSpec

    spec = TaskSpec.__new__(TaskSpec)
    spec.data = {"task_id": "x", "task_name": "x", "model_type": "binary_classification",
                 "target": {"name": "y", "positive_definition": "p"}, "primary_metric": "auc",
                 "split_policy": {"type": "random"}, "business_domains": "credit"}
    assert any("business_domains" in e for e in spec.validate())
    spec.data["business_domains"] = ["credit", "随便写"]
    assert not any("business_domains" in e for e in spec.validate())


# ---------------------------------------------------------------- 证据汇总口径

def _lk(pid, run, trial, rel, w=0.5, binding="passive", topics=("a",), lo=None, hi=None):
    return {"principle_id": pid, "run_id": run, "trial_id": trial, "binding": binding,
            "relation": rel, "base_relation": "same_task", "weight": w, "trial_at": None,
            "match": {"topics": list(topics)}, "ci_low": lo, "ci_high": hi}


def test_aggregate_default_is_trial_level():
    links = [_lk("p", "r1", f"t{i}", "support") for i in range(4)] + \
        [_lk("p", "r2", "t9", "refute")]
    st = EV.aggregate(links)["p"]
    assert st["w_support"] == 2.0 and st["w_refute"] == 0.5        # 每个 trial 一票
    assert EV.MULTI_FACTOR_POWER == 0.0


def test_aggregate_run_dedup_caps_each_run():
    links = [_lk("p", "r1", f"t{i}", "support") for i in range(3)] + \
        [_lk("p", "r1", "t3", "refute"), _lk("p", "r2", "t9", "refute"),
         {**_lk("p", "r3", "x", "support"), "base_relation": "cross_base"},
         {**_lk("p", "r4", "y", "refute"), "base_relation": "cross_base"},
         {**_lk("p", "r4", "z", "refute"), "base_relation": "cross_base"}]
    st = EV.aggregate(links, run_dedup=True)["p"]
    # r1:最强一条 0.5,按 3:1 分摊 → 支持 0.375、反驳 0.125;r2 反驳 0.5;r3 支持 0.5;
    # r4 两条反驳合计封顶 0.5
    assert st["w_support"] == pytest.approx(0.875)
    assert st["w_refute"] == pytest.approx(1.125)
    assert st["n_links"] == 8 and st["n_runs"] == 4
    # 迁移战绩按 run 计:r3 支持一次、r4 反驳一次
    assert st["transfer"] == pytest.approx((1 + 1) / (1 + 1 + 2))


def test_aggregate_multi_factor_only_passive():
    links = [_lk("p", "r1", "t1", "support", topics=("a", "b", "c", "d")),
             _lk("p", "r2", "t2", "support", w=1.0, binding="explicit", topics=("a", "b"))]
    st = EV.aggregate(links, multi_factor=1.0)["p"]
    assert st["w_support"] == pytest.approx(0.5 / 4 + 1.0)


# ---------------------------------------------------------------- 回放评测(纯函数)

def _p(pid, direction, topics, atids=(A,), source=None):
    return {"principle_id": pid, "version": 1, "direction": direction, "topics": list(topics),
            "objects": [], "category": "feature_engineering", "scope": BIN, "claim": pid,
            "condition": {}, "evidence": {"abstract_task_ids": list(atids)},
            "limitations": [], "source_snapshot_id": source}


def _trial(run, tid, topics):
    return {"run_id": run, "trial_id": tid, "atid": A, "primary_metric": "auc",
            "higher_is_better": True, "topics": list(topics), "hypothesis": "h"}


def _view(principles):
    return {"atid": None, "topics": [{"code": c, "aliases": []} for c in
                                     ("feature", "feature.seq", "feature.seq.gap")],
            "aliases": {}, "fingerprints": [{"atid": A}],
            "fingerprint_profiles": {A: {"business_domains": ["credit"],
                                         "sample_scale": "200w_1000w"}},
            "principles": principles, "stats": {}}


def test_replay_labels_metrics_and_leave_one_run_out():
    good, bad = _p("p-good", 1, ["feature.seq.gap"]), _p("p-bad", -1, ["feature.seq.gap"])
    trials = [_trial("r1", "t1", ["feature.seq.gap"]), _trial("r2", "t2", ["feature.seq.gap"]),
              _trial("r3", "t3", ["feature.seq"])]        # t3 没有标签,跳过
    links = []
    for run, tid in (("r1", "t1"), ("r2", "t2")):
        links += [_lk("p-good", run, tid, "support", lo=0.001, hi=0.003),
                  _lk("p-bad", run, tid, "refute", lo=0.001, hi=0.003)]
    out = RP.replay(_view([good, bad]), trials, links, k=8)
    assert out["n_queries"] == 2 and out["skipped"] == {"no_label": 1}
    # 留一 run 后另一个 run 的证据仍让 p-good 排在 p-bad 前
    assert out["recall"] == 1.0 and out["mrr"] == 1.0 and out["ndcg"] == 1.0
    assert out["bad_share"] == 0.5
    assert out["evidence_auc"] == 1.0 and out["n_evidence_pairs"] == [2, 2]
    assert set(out["tier_mix"]) == {"L0"}
    cold = RP.replay(_view([good, bad]), trials, links, k=8, mode="new_task")
    assert "L0" not in cold["tier_mix"] and cold["n_queries"] == 2


def test_replay_excludes_principles_written_by_the_same_run():
    own = _p("p-own", 1, ["feature.seq.gap"], source="snap-r1")
    trials = [_trial("r1", "t1", ["feature.seq.gap"])]
    links = [_lk("p-own", "r1", "t1", "support", lo=0.001, hi=0.002)]
    out = RP.replay(_view([own]), trials, links, run_sources={"r1": {"snap-r1"}})
    assert out["n_queries"] == 1 and out["recall"] == 0.0
    with pytest.raises(ValueError):
        RP.replay(_view([own]), trials, links, mode="bogus")


# ---------------------------------------------------------------- 仓库与服务

def _repo(tmp_path) -> LocalRepository:
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _seed(repo):
    repo.init_topics(NS, [{"code": "feature.seq", "definition": "d", "aliases": ["seq_"]},
                          {"code": "feature.seq.gap", "definition": "d",
                           "aliases": ["seq_gap_d365"]}], reason="v1")
    sid = repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_principles",
                                "run": {"run_id": "legacy:x", "atid": A},
                                "discoveries": [{"local_id": "d1", "payload": {"claim": "c"}}]},
                               namespace=NS)["snapshot_id"]
    ctx = repo.merge_context(NS, sid)
    repo.merge_apply(NS, {
        "schema": "modelevo.merge-decision/1", "snapshot_id": sid,
        "lease_token": ctx["lease"]["token"],
        "actions": [{"local_id": "d1", "action": "add", "reason": "r", "principle": {
            "principle_id": "p-gap", "claim": "间隔族加宽有效", "direction": 1,
            "topics": ["feature.seq.gap"], "scope": BIN,
            "condition": {"business_domains": ["consumer_credit"]},
            "evidence": {"abstract_task_ids": [A]}}}]})
    for run in ("r@1", "r@2"):
        repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "run",
                              "run": {"run_id": run, "atid": A},
                              "trial_outcomes": [{
                                  "run_id": run, "trial_id": "t1", "atid": A,
                                  "family_hints": ["info:seq_gap_d365:window"],
                                  "primary_metric": "auc", "higher_is_better": True,
                                  "verdict": "accepted", "delta_parent_ci_low": 0.001,
                                  "delta_parent_ci_high": 0.003}]}, namespace=NS)


def test_repository_replay_end_to_end(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    assert repo.principles(NS)[0]["condition"]["business_domains"] == ["credit"]
    out = repo.replay(NS, mode="same_task")
    assert out["n_queries"] == 2 and out["recall"] == 1.0
    assert out["evidence"] == {"run_dedup": False, "multi_factor": 0.0}
    alt = repo.replay(NS, run_dedup=True, multi_factor=0.5)
    assert alt["evidence"] == {"run_dedup": True, "multi_factor": 0.5}


def test_replay_over_http(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from runtime.library import HttpRepository
    from service.app import create_app

    repo = _repo(tmp_path)
    _seed(repo)
    client = TestClient(create_app(repo))

    def transport(method, url, headers, body, timeout):
        r = client.request(method, url.split("testserver", 1)[1], headers=headers,
                           content=body)
        return r.status_code, dict(r.headers), r.content

    http = HttpRepository("http://testserver", transport=transport)
    got = http.replay(NS, mode="new_task", run_dedup=True, multi_factor=0.5)
    assert got["mode"] == "new_task" and got["n_queries"] == 2
    assert got["evidence"] == {"run_dedup": True, "multi_factor": 0.5}
