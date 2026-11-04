"""v2.18.0 中心化经验库(M2):快照合并、主题词表、卷宗、撤销、更新日志、run 快照。

docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md §5 §6 §12 M2;
合并协议 references/experience-library-merge.md。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import json

import pytest
import yaml

from runtime.library import (ArtifactStore, HttpRepository, LibraryError,
                             LibraryStore, LibraryUnavailable,
                             LibraryValidationError, LocalRepository)
from runtime.library import curation
from runtime.library import schema as S
from runtime.library.curation import LeaseBusy

ATID = "binary-y-200w_1000w-bc5329dc"
NS = "t"
BIN = {"model_type": "binary_classification", "primary_metric": "auc"}


def _repo(tmp_path) -> LocalRepository:
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _submit(repo, ids, run_id="legacy:x", claims=None) -> str:
    claims = claims or {}
    return repo.submit_snapshot({
        "schema": "modelevo.snapshot/1", "kind": "legacy_principles",
        "run": {"run_id": run_id, "atid": ATID},
        "discoveries": [{"local_id": i, "payload": {"claim": claims.get(i, f"claim {i}")}}
                        for i in ids]}, namespace=NS)["snapshot_id"]


def _p(claim, direction=1, topics=("feature.seq",), scope=None, **kw):
    return {"claim": claim, "direction": direction, "topics": list(topics),
            "scope": scope or BIN, **kw}


def _decision(ctx, actions, **kw):
    return {"schema": "modelevo.merge-decision/1",
            "snapshot_id": ctx["snapshot"]["snapshot_id"],
            "lease_token": ctx["lease"]["token"], "actions": actions, **kw}


SEQ_TOPIC = [{"code": "feature.seq", "definition": "事件序列特征", "aliases": ["seq_"]}]


def _seed(repo):
    """合并一条 +1 原则与一条 0 原则,返回 {方向: principle_id}。"""
    sid = _submit(repo, ["a", "b", "c"])
    ctx = repo.merge_context(NS, sid)
    rec = repo.merge_apply(NS, _decision(ctx, [
        {"local_id": "a", "action": "add", "reason": "r", "principle": _p("漏斗特征有效", 1)},
        {"local_id": "b", "action": "add", "reason": "r",
         "principle": _p("计数网格无独立增量", 0, objects=["seqp_cnt_"])},
        {"local_id": "c", "action": "drop", "reason": "流水账"}],
        topics_added=SEQ_TOPIC,
        dossiers=[{"scope_type": "task", "scope_id": ATID,
                   "sections": {"ceiling": "0.812", "effective": "漏斗"}}]))
    ps = {p["direction"]: p["principle_id"] for p in repo.principles(NS)}
    return sid, rec, ps


# ---------------------------------------------------------------- 主流程

def test_merge_happy_path(tmp_path):
    repo = _repo(tmp_path)
    assert repo.store.schema_revision() == S.SCHEMA_REVISION
    sid, rec, ps = _seed(repo)
    assert rec["counts"] == {"principle_versions": 2, "topics": 1, "governance": 0,
                             "dossiers": 1, "dropped": 1}
    rows = {p["principle_id"]: p for p in repo.principles(NS)}
    assert len(rows) == 2
    p0 = rows[ps[0]]
    assert p0["category"] == "feature_engineering" and p0["topics"] == ["feature.seq"]
    assert p0["scope"]["target_transform"] == "raw" and p0["source_snapshot_id"] == sid
    assert repo.list_snapshots(NS)[0]["state"] == "merged"
    assert repo.merge_pending(NS) == []
    assert repo.dossier(NS, "task", ATID)["sections"]["ceiling"] == "0.812"
    assert "feature.seq" in {t["code"] for t in repo.topics(NS)}
    kinds = [(c["entry_type"], c["action"]) for c in repo.changelog(NS)]
    assert kinds.count(("merge", "add")) == 2 and ("merge", "drop") in kinds
    assert ("topic", "add") in kinds and ("dossier", "add") in kinds
    # 租约已释放:立即可以再取上下文
    sid2 = _submit(repo, ["d"], run_id="legacy:y", claims={"d": "计数网格 无独立增量 联合面"})
    ctx2 = repo.merge_context(NS, sid2)
    assert ps[0] in ctx2["discoveries"][0]["related"]
    assert {p["principle_id"] for p in ctx2["principles"]} >= {ps[0]}


def test_merge_into_creates_new_version(tmp_path):
    repo = _repo(tmp_path)
    _sid, _rec, ps = _seed(repo)
    ctx = repo.merge_context(NS, _submit(repo, ["m"], run_id="legacy:y"))
    repo.merge_apply(NS, _decision(ctx, [{
        "local_id": "m", "action": "merge_into", "target": ps[1], "reason": "复现",
        "principle": _p("漏斗特征有效,两次复现", 1,
                        effect={"n_replications": 2})}]))
    cur = {p["principle_id"]: p for p in repo.principles(NS)}[ps[1]]
    assert cur["version"] == 2 and cur["effect"] == {"n_replications": 2}


def test_lease_busy_and_expiry(tmp_path):
    repo = _repo(tmp_path)
    sid = _submit(repo, ["a"])
    repo.merge_context(NS, sid)
    with pytest.raises(LeaseBusy):
        repo.merge_context(NS, sid)
    # 另一个命名空间不受影响
    other = repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_principles",
                                  "run": {"run_id": "x"}, "discoveries": [
                                      {"local_id": "z", "payload": {"claim": "z"}}]},
                                 namespace="other")["snapshot_id"]
    repo.merge_context("other", other)
    # 错误 token 释放不了;过期后可重新申请,旧 token 失效
    assert not curation.release_lease(repo.store, "merge:t", "wrong-token")
    with repo.store._write() as conn:
        conn.execute(curation.S.leases.update().values(expires_at=0))
    ctx_new = repo.merge_context(NS, sid)
    assert ctx_new["lease"]["token"]
    with pytest.raises(LibraryValidationError, match="租约"):
        repo.merge_apply(NS, {**_decision(ctx_new, [
            {"local_id": "a", "action": "drop", "reason": "r"}]), "lease_token": "stale"})


def test_merge_abort_marks_failed(tmp_path):
    repo = _repo(tmp_path)
    sid = _submit(repo, ["a"])
    ctx = repo.merge_context(NS, sid)
    repo.merge_abort(NS, sid, lease_token=ctx["lease"]["token"], reason="放弃")
    assert repo.merge_pending(NS)[0]["state"] == "merge_failed"
    assert repo.merge_context(NS, sid)["lease"]["token"]       # 租约已释放


# ---------------------------------------------------------------- 校验规则

def _errors(repo, actions, **kw) -> list[str]:
    sid = _submit(repo, [a["local_id"] for a in actions if "local_id" in a] or ["x"],
                  run_id=f"legacy:{len(repo.list_snapshots(NS))}")
    ctx = repo.merge_context(NS, sid)
    try:
        repo.merge_apply(NS, _decision(ctx, actions, **kw))
    except LibraryValidationError as exc:
        return exc.errors
    finally:
        curation.release_lease(repo.store, f"merge:{NS}", ctx["lease"]["token"])
    return []


def test_validation_rules(tmp_path):
    repo = _repo(tmp_path)
    _sid, _rec, ps = _seed(repo)
    E = lambda acts, **kw: " | ".join(_errors(repo, acts, **kw))  # noqa: E731

    assert "不在词表" in E([{"local_id": "x", "action": "add", "reason": "r",
                          "principle": _p("c", topics=["feature.nope"])}])
    assert "reason 必填" in E([{"local_id": "x", "action": "drop"}])
    assert "direction" in E([{"local_id": "x", "action": "add", "reason": "r",
                              "principle": _p("c", direction=2)}])
    assert "target_transform" in E([{"local_id": "x", "action": "add", "reason": "r",
                                     "principle": _p("c", scope={"model_type": "regression",
                                                                 "primary_metric": "rmse"})}])
    assert "口径不同" in E([{"local_id": "x", "action": "merge_into", "target": ps[1],
                          "reason": "r", "principle": _p("c", scope={
                              "model_type": "binary", "primary_metric": "ks"})}])
    assert "不存在" in E([{"local_id": "x", "action": "drop_duplicate", "target": "p-nope",
                         "reason": "r"}])
    msg = E([{"local_id": "x", "action": "add", "reason": "r",
              "principle": _p("c", topics=["feature.seqs"])}],
            topics_added=[{"code": "feature.seqs", "definition": "d"}])
    assert "过于相近" in msg
    assert "父节点" in E([{"local_id": "x", "action": "add", "reason": "r",
                         "principle": _p("c", topics=["model.gbm.xgb"])}],
                        topics_added=[{"code": "model.gbm.xgb", "definition": "d"}])
    assert "没有被任何原则使用" in E([{"local_id": "x", "action": "drop", "reason": "r"}],
                                  topics_added=[{"code": "hpo.lr", "definition": "d"}])
    assert "BAD" not in E([{"local_id": "x", "action": "drop", "reason": "r"}])
    long = {"ceiling": "x" * 3001}
    assert "3000" in E([{"local_id": "x", "action": "drop", "reason": "r"}],
                       dossiers=[{"scope_type": "task", "scope_id": ATID, "sections": long}])
    assert "split_by_scope" in E([{"local_id": "x", "action": "split_by_scope", "reason": "r",
                                   "principles": [_p("c")]}])
    # 未覆盖的 discovery
    sid = _submit(repo, ["u", "v"], run_id="legacy:cover")
    ctx = repo.merge_context(NS, sid)
    dry = repo.merge_apply(NS, {**_decision(ctx, [{"local_id": u, "action": "drop", "reason": "r"}
                                                  for u in ("u", "v")]), "dry_run": True})
    assert dry["dry_run"] and repo.list_snapshots(NS)[-1]["state"] == "submitted"
    with pytest.raises(LibraryValidationError, match="没有对应动作"):
        repo.merge_apply(NS, _decision(ctx, [{"local_id": "u", "action": "drop", "reason": "r"}]))


def test_overturn_requires_evidence(tmp_path):
    repo = _repo(tmp_path)
    _sid, _rec, ps = _seed(repo)
    repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "run",
                          "run": {"run_id": "r@1", "atid": ATID},
                          "trial_outcomes": [
                              {"run_id": "r@1", "trial_id": "weak", "primary_metric": "auc",
                               "higher_is_better": True, "delta_parent_ci_low": -0.001,
                               "delta_parent_ci_high": 0.003},
                              {"run_id": "r@1", "trial_id": "strong", "primary_metric": "auc",
                               "higher_is_better": True, "delta_parent_ci_low": 0.002,
                               "delta_parent_ci_high": 0.006}]}, namespace=NS)
    act = {"local_id": "x", "action": "supersede", "target": ps[0], "reason": "新证据",
           "principle": _p("计数网格有效", 1)}
    assert any("证据" in e for e in _errors(repo, [act]))
    assert any("证据" in e for e in _errors(repo, [{**act, "evidence_trials": [
        {"run_id": "r@1", "trial_id": "weak"}]}]))
    assert _errors(repo, [{**act, "evidence_trials": [
        {"run_id": "r@1", "trial_id": "strong"}]}]) == []
    st = {p["principle_id"]: p["status"] for p in repo.principles(NS, include_inactive=True)}
    assert st[ps[0]] == "superseded"


def test_contest_and_split(tmp_path):
    repo = _repo(tmp_path)
    _sid, _rec, ps = _seed(repo)
    ctx = repo.merge_context(NS, _submit(repo, ["c1", "s1"], run_id="legacy:y"))
    repo.merge_apply(NS, _decision(ctx, [
        {"local_id": "c1", "action": "contest", "target": ps[0], "reason": "反例",
         "principle": _p("计数网格有效", 1)},
        {"local_id": "s1", "action": "split_by_scope", "target": ps[1], "reason": "口径不同",
         "principles": [_p("auc 上有效", 1), _p("ks 上无效", 0, scope={
             "model_type": "binary", "primary_metric": "ks"})]}]))
    allp = {p["principle_id"]: p for p in repo.principles(NS, include_inactive=True)}
    assert allp[ps[1]]["status"] == "superseded" and len(allp[ps[1]]["superseded_by"]) == 2
    assert allp[ps[0]]["status"] == "active" and len(allp[ps[0]]["contested_with"]) == 1


def test_topic_alias_resolves(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    ctx = repo.merge_context(NS, _submit(repo, ["a2"], run_id="legacy:y"))
    repo.merge_apply(NS, _decision(ctx, [
        {"local_id": "a2", "action": "add", "reason": "r",
         "principle": _p("c", topics=["feature.seq"])}],
        topics_added=[{"code": "feature.event", "definition": "事件特征"}],
        topics_aliased=[{"code": "feature.seq", "into": "feature.event", "reason": "同义"}]))
    # 已并入的旧词、旧别名都解析到新词
    ctx = repo.merge_context(NS, _submit(repo, ["a3"], run_id="legacy:z"))
    repo.merge_apply(NS, _decision(ctx, [{"local_id": "a3", "action": "add", "reason": "r",
                                           "principle": _p("d", topics=["seq_"])}]))
    newest = [p for p in repo.principles(NS) if p["claim"] == "d"][0]
    assert newest["topics"] == ["feature.event"]


# ---------------------------------------------------------------- 撤销

def test_revert_hides_merge_and_allows_remerge(tmp_path):
    repo = _repo(tmp_path)
    sid, _rec, ps = _seed(repo)
    ctx = repo.merge_context(NS, _submit(repo, ["m"], run_id="legacy:y"))
    sid2 = ctx["snapshot"]["snapshot_id"]
    repo.merge_apply(NS, _decision(ctx, [{"local_id": "m", "action": "merge_into",
                                           "target": ps[1], "reason": "r",
                                           "principle": _p("v2 内容", 1)}],
                                   dossiers=[{"scope_type": "task", "scope_id": ATID,
                                              "sections": {"ceiling": "v2"}}]))
    w_before = repo.watermark()
    repo.revert_merge(NS, sid2, reason="合并有误")
    cur = {p["principle_id"]: p for p in repo.principles(NS)}
    assert cur[ps[1]]["version"] == 1 and cur[ps[1]]["claim"] == "漏斗特征有效"
    assert repo.dossier(NS, "task", ATID)["sections"]["ceiling"] == "0.812"
    # 按历史水位仍能读到撤销前的状态
    assert {p["principle_id"]: p for p in repo.principles(NS, at=w_before)}[ps[1]]["version"] == 2
    with pytest.raises(LibraryValidationError):
        repo.revert_merge(NS, sid2, reason="重复撤销")
    # 撤销后可重新合并;版本号不复用
    ctx = repo.merge_context(NS, sid2)
    repo.merge_apply(NS, _decision(ctx, [{"local_id": "m", "action": "merge_into",
                                           "target": ps[1], "reason": "r",
                                           "principle": _p("v3 内容", 1)}]))
    assert {p["principle_id"]: p for p in repo.principles(NS)}[ps[1]]["version"] == 3
    log = repo.changelog(NS)
    assert any(r["reverted"] for r in log) and any(r["action"] == "revert" for r in log)


def test_changelog_render_and_files(tmp_path):
    from service.server import write_changelogs

    repo = _repo(tmp_path)
    _seed(repo)
    text = curation.render_changelog(repo.changelog(NS), NS)
    assert "# 经验库更新日志 · t" in text and "**新增**" in text and "**丢弃**" in text
    paths = write_changelogs(repo, tmp_path / "CHANGELOG")
    assert [p.name for p in paths] == ["t.md"]


# ---------------------------------------------------------------- 服务

def _http(repo) -> HttpRepository:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from service.app import create_app

    client = TestClient(create_app(repo))

    def transport(method, url, headers, body, timeout):
        r = client.request(method, url.split("testserver", 1)[1], headers=headers,
                           content=body)
        return r.status_code, dict(r.headers), r.content

    return HttpRepository("http://testserver", transport=transport,
                          actor={"user": "u", "host": "h", "team": "", "run": "",
                                 "model": "opus-test"})


def test_merge_over_http(tmp_path):
    local = _repo(tmp_path)
    http = _http(local)
    sid = _submit(local, ["a"])
    assert http.merge_pending(NS)[0]["snapshot_id"] == sid
    ctx = http.merge_context(NS, sid)
    with pytest.raises(LibraryError) as e:
        http.merge_context(NS, sid)
    assert e.value.status == 409
    with pytest.raises(LibraryError) as e:
        http.merge_apply(NS, _decision(ctx, [{"local_id": "a", "action": "add", "reason": "r",
                                               "principle": _p("c", topics=["nope.x"])}]))
    assert e.value.status == 422 and e.value.errors
    rec = http.merge_apply(NS, _decision(ctx, [{"local_id": "a", "action": "add", "reason": "r",
                                                 "principle": _p("c")}],
                                         topics_added=SEQ_TOPIC))
    assert rec["counts"]["principle_versions"] == 1
    assert http.principles(NS)[0]["claim"] == "c"
    assert any(t["code"] == "feature.seq" for t in http.topics(NS))
    assert http.dossier(NS, "task", ATID) is None
    assert http.changelog(NS)[0]["actor"]["model"] == "opus-test"
    assert http.revert_merge(NS, sid, reason="x") > rec["commit_seq"]
    assert http.principles(NS) == []


# ---------------------------------------------------------------- run 快照

def _run_fixture(root: pathlib.Path) -> pathlib.Path:
    run = root / "run-demo"
    (run / "experience").mkdir(parents=True)
    (run / "run-state.json").write_text(json.dumps(
        {"abstract_task_id": ATID, "skill_version": "2.18.0", "status": "FINAL_EVALUATED"}))
    (run / "task-spec.yaml").write_text("task_id: demo\n")
    (run / "experience" / "exported-global-experiences.yaml").write_text(yaml.safe_dump([
        {"claim": "漏斗特征有效", "category": "feature_engineering", "action_type": "add",
         "outcome": "positive", "evidence": "trial-002 +0.003"}], allow_unicode=True))
    (run / "experience" / "dossier-draft.md").write_text("天花板 0.81")
    (run / "experience-retrieval-telemetry.jsonl").write_text(json.dumps({"shown_ids": ["x"]}) + "\n")
    for tid, verdict in (("trial-001", "accepted"), ("trial-002", "accepted"),
                         ("trial-003", "rejected")):
        t = run / "trials" / tid
        (t / "code").mkdir(parents=True)
        (t / "config").mkdir()
        (t / "code" / "train.py").write_text(
            f"import pathlib\nRUN_DIR = pathlib.Path('{run.resolve()}')\nx = '{tid}'\n")
        (t / "config" / "trial.yaml").write_text(yaml.safe_dump(
            {"trial_id": tid, "hyperparameters": {"n_estimators": 300}}))
        (t / "trial.yaml").write_text(yaml.safe_dump(
            {"trial_id": tid, "algorithm": "xgboost", "status": "EVALUATED",
             "search_action": "explore"}))
        (t / "evaluation.json").write_text(json.dumps({
            "verdict": verdict, "primary_metric": {"name": "auc", "value": 0.8},
            "delta": {"vs_baseline": 0.01, "vs_anchor_paired_bootstrap": {
                "delta_ci": [0.005, 0.015], "p_positive": 1.0}}}))
    return run


def test_run_snapshot_build_submit_and_outbox(tmp_path):
    from _mvo.library_snapshot import build_run_snapshot, submit_or_queue, submit_pending

    repo = _repo(tmp_path / "lib")
    run = _run_fixture(tmp_path)
    staged: dict = {}
    payload, rep = build_run_snapshot(run, repo, model="opus-test", stage=staged)
    assert rep["discoveries"] == 1 and rep["trial_outcomes"] == 3 and rep["recipes"] == 2
    assert payload["run"]["model"] == "opus-test" and payload["dossier_draft"] == "天花板 0.81"
    # v2.18.0 补丁:构建阶段不碰网络,配方只暂存在 stage 里(bundle id 本地算)
    assert not repo.has_artifact(payload["recipes"][0]["recipe_id"])
    rid = payload["recipes"][0]["recipe_id"]
    assert set(staged) == {r["recipe_id"] for r in payload["recipes"]}

    # 服务不可达:快照**和配方**一起进 outbox(Q26),不再抛异常
    down = HttpRepository("http://127.0.0.1:9", timeout=1)
    res = submit_or_queue(run, down, payload, namespace=NS, staged=staged)
    assert res["status"] == "queued" and res["queued_recipes"] == 2
    assert (run / "experience" / "snapshot.json").exists()
    assert (run / "experience" / "outbox" / "cas" / rep["snapshot_id"][:12]
            / rid / "train.py").exists()
    with pytest.raises(LibraryUnavailable):
        submit_pending(run, down)

    # 恢复后补交:先补产物再提交,否则服务端会因 bundle 不存在而 422
    sent = submit_pending(run, repo)
    assert len(sent) == 1 and not sent[0]["receipt"]["duplicate"]
    assert sent[0]["uploaded_recipes"] == 2
    assert submit_pending(run, repo) == []
    files = repo.get_bundle(rid)
    assert b"{{MODELEVO_RUN_DIR}}" in files["train.py"] and "task-spec.yaml" in files
    lb = repo.leaderboard(NS, ATID)
    assert len(lb) == 2 and lb[0]["measurement"]["ci_low"] == 0.005
    assert repo.merge_pending(NS)[0]["kind"] == "run"


def test_init_topics_only_once(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(LibraryValidationError, match="父节点"):
        repo.init_topics(NS, [{"code": "feature.seq.window", "definition": "d"}], reason="v1")
    seq = repo.init_topics(NS, [
        {"code": "feature.seq.window_count", "definition": "计数", "aliases": ["seqp_cnt_"]},
        {"code": "feature.seq", "definition": "序列"},
        {"code": "hpo.lr", "definition": "学习率"}], reason="v1")
    assert seq == repo.watermark()
    codes = {t["code"] for t in repo.topics(NS)}
    assert {"feature.seq", "feature.seq.window_count", "hpo.lr"} <= codes
    with pytest.raises(LibraryValidationError, match="已初始化"):
        repo.init_topics(NS, [{"code": "model.gbm", "definition": "d"}], reason="v2")
    # 初版词可直接在合并里使用(别名也行)
    ctx = repo.merge_context(NS, _submit(repo, ["a"]))
    repo.merge_apply(NS, _decision(ctx, [{"local_id": "a", "action": "add", "reason": "r",
                                           "principle": _p("c", topics=["seqp_cnt_"])}]))
    assert repo.principles(NS)[0]["topics"] == ["feature.seq.window_count"]
