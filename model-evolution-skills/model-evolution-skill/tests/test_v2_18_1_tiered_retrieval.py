"""v2.18.1:多维任务画像、分级检索(L0–L4)、口径放宽、检索反馈落表、合并复核按主题、
租约续约、token 认证、指纹别名建议。

docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md §21。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import pytest

from runtime.library import ArtifactStore, HttpRepository, LibraryStore, LocalRepository
from runtime.library import curation as CU
from runtime.library import evidence as EV
from runtime.library import retrieval as RT
from runtime.library import task_profile as TP

NS = "t"
A = "binary-y-200w_1000w-bc5329dc"
A_NEAR = "binary-y-200w_1000w-11111111"       # 同目标,列集小改
A_REL = "binary-z-200w_1000w-22222222"        # 同业务,不同目标
A_FAR = "binary-w-lt10w-33333333"             # 无关任务
BIN = {"model_type": "binary_classification", "primary_metric": "auc", "target_transform": "raw"}
KS = {**BIN, "primary_metric": "ks"}
REG = {"model_type": "regression", "primary_metric": "rmse", "target_transform": "raw"}
TOPICS = [{"code": c, "aliases": []} for c in
          ("feature", "feature.seq", "feature.seq.gap", "eval", "eval.cv", "engineering",
           "engineering.oom")]
COLS = [f"seq_gap_d{i}_cnt" for i in range(60)] + [f"base_x{i}_amt" for i in range(40)]


def _p(pid, direction, topics, *, atids=(A,), category="feature_engineering", scope=BIN,
       objects=(), domains=()):
    return {"principle_id": pid, "version": 1, "direction": direction, "topics": list(topics),
            "objects": list(objects), "category": category, "scope": scope,
            "claim": f"{pid} claim", "condition": {"business_domains": list(domains)},
            "evidence": {"abstract_task_ids": list(atids)}, "limitations": []}


def _view(principles, *, atid=A, scope=BIN, profiles=None, mine=None, stats=None):
    return {"atid": atid, "scope": scope, "topics": TOPICS, "aliases": {},
            "fingerprints": [{"atid": a} for a in (A, A_NEAR, A_REL, A_FAR)],
            "fingerprint_profiles": profiles or {}, "task_profile": mine or {},
            "principles": principles, "stats": stats or {}, "trial_outcomes": []}


def _prof(cols, target, domains=("credit",), scale="200w_1000w"):
    return TP.build_profile(spec_data={"model_type": "binary_classification",
                                       "target": {"name": target},
                                       "business_domains": list(domains),
                                       "primary_metric": "auc"},
                            columns=cols, sample_scale=scale)


# ---------------------------------------------------------------- 画像

def test_profile_dimensions_and_similarity():
    p = _prof(COLS, "loan_amt_t7_log1p")
    assert p["target_base"] == "loan_amt_t7" and p["metric_family"] == "rank"
    assert p["n_columns"] == 100 and len(p["col_minhash"]) == TP.MINHASH_K
    assert "seq_gap_" in p["col_families"]
    near = _prof(COLS + ["new_col_a", "new_col_b"], "loan_amt_t7")
    sim, dims = TP.similarity(p, near)
    assert dims["target"] == 1.0 and dims["columns"] >= 0.85 and sim >= 0.9
    far = _prof([f"other_{i}" for i in range(50)], "churn", domains=("marketing",),
                scale="lt10w")
    sim2, dims2 = TP.similarity(p, far)
    assert dims2["columns"] < 0.2 and dims2["target"] == 0.0 and sim2 < 0.15
    assert TP.similarity({}, {}) == (0.0, {})
    assert TP.metric_family("gini") == TP.metric_family("ks") == "rank"


def test_minhash_estimates_jaccard():
    a = [f"c{i}" for i in range(200)]
    b = [f"c{i}" for i in range(100, 300)]          # 真 Jaccard = 100/300
    j = TP.minhash_jaccard(TP.minhash(a), TP.minhash(b))
    assert abs(j - 1 / 3) < 0.15
    assert TP.minhash([]) is None


# ---------------------------------------------------------------- 口径

def test_scope_level():
    assert EV.scope_level(BIN, BIN) == "exact"
    assert EV.scope_level(BIN, {**BIN, "primary_metric": "*"}) == "exact"
    assert EV.scope_matches(BIN, {**BIN, "primary_metric": "*"})
    assert EV.scope_level(BIN, KS) == "metric_family"
    assert EV.scope_level(BIN, {**BIN, "primary_metric": "logloss"}) is None
    assert EV.scope_level(BIN, REG) is None
    assert EV.scope_level(BIN, REG, "engineering") == "agnostic"
    assert EV.scope_level(BIN, {**BIN, "primary_metric": "logloss"}, "evaluation") == "agnostic"
    assert EV.scope_level(BIN, REG, "evaluation") is None      # 评估类仍要求任务类型一致


def test_norm_scope_accepts_wildcard_metric():
    errs = []
    out = CU._norm_scope({"model_type": "regression", "primary_metric": "*"}, errs, "x")
    assert errs == [] and out["primary_metric"] == "*"
    assert CU._scope_compatible({"primary_metric": "*"}, {"primary_metric": "auc"})


# ---------------------------------------------------------------- 分级检索

def test_cascade_cold_start_reaches_near_and_related_tasks():
    """v2.18.0 实测:列集一变 / 新任务时,带指纹原则全部不可见。现在按画像进 L1 / L2。"""
    mine = _prof(COLS + ["new_a"], "y")
    profiles = {A: _prof(COLS, "y"), A_REL: _prof(COLS[:60] + ["z1", "z2"], "z"),
                A_FAR: _prof([f"o{i}" for i in range(30)], "w", domains=("ads",),
                             scale="lt10w")}
    ps = [_p("near", 1, ["feature.seq.gap"], atids=(A,)),
          _p("rel", 1, ["feature.seq.gap"], atids=(A_REL,)),
          _p("far", 1, ["feature.seq.gap"], atids=(A_FAR,))]
    v = _view(ps, atid=A_NEAR, profiles=profiles, mine=mine)
    cards = RT.search(v, topics=["feature.seq.gap"])
    tiers = {c["principle_id"]: c["tier"] for c in cards}
    assert tiers["near"] == "L1" and tiers["rel"] == "L2"
    assert [c["principle_id"] for c in cards][:2] == ["near", "rel"]
    near = next(c for c in cards if c["principle_id"] == "near")
    assert near["task_similarity"] >= 0.75 and any("同目标" in w for w in near["why"])
    # 有精确档时,无通用性证据的低相似跨业务原则不出
    assert "far" not in tiers


def test_cascade_relaxes_metric_and_adds_agnostic_methods():
    ps = [_p("same", 1, ["feature.seq.gap"], scope=BIN),
          _p("ks", 1, ["feature.seq.gap"], scope=KS),
          _p("oom", -1, ["engineering.oom"], category="engineering", scope=REG),
          _p("cv", 0, ["eval.cv"], category="evaluation",
             scope={**BIN, "primary_metric": "logloss"})]
    v = _view(ps, scope=KS)        # 本任务主指标 ks
    got = RT.search(v, topics=["feature.seq.gap", "engineering.oom", "eval.cv"])
    by = {c["principle_id"]: c for c in got}
    assert by["ks"]["tier"] == "L0"                       # ks 原则对 ks 任务是精确
    assert by["same"]["tier"] == "L2" and by["same"]["scope_match"] == "metric_family"
    assert by["oom"]["tier"] == by["cv"]["tier"] == "L4"
    order = [c["tier"] for c in got]
    assert order == sorted(order)                         # 精确档在前,放宽档依次补位
    assert "放宽档" in RT.render_card(by["same"])


def test_cascade_fills_relaxed_only_when_strict_insufficient():
    ps = [_p(f"s{i}", 1, ["feature.seq.gap"]) for i in range(3)] + \
         [_p("ks", 1, ["feature.seq.gap"], scope=KS)]
    v = _view(ps)
    assert [c["tier"] for c in RT.search(v, topics=["feature.seq.gap"], k=3)] == ["L0"] * 3
    assert RT.search(v, topics=["feature.seq.gap"], k=4)[-1]["principle_id"] == "ks"


# ---------------------------------------------------------------- 库:画像与反馈

def _repo(tmp_path) -> LocalRepository:
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _run_snap(run_id, *, profile=None, logs=(), trials=(), tag=""):
    return {"schema": "modelevo.snapshot/1", "kind": "run",
            "run": {"run_id": run_id, "atid": A, "task_profile": profile or {}, "tag": tag},
            "trial_outcomes": list(trials), "retrieval": {"logs": list(logs)}}


def _trial(tid, refs=(), novel=False, delta=0.002, verdict="accepted"):
    basis = {"novel": True, "refs": []} if novel else \
        {"novel": False, "refs": [{"id": r, "relation": rel} for r, rel in refs]}
    return {"run_id": "r@1", "trial_id": tid, "atid": A, "primary_metric": "auc",
            "higher_is_better": True, "delta_parent": delta, "verdict": verdict,
            "basis": basis, "experience_ids": [r for r, _ in refs], "generation": 1}


def test_profiles_are_stored_deduped_and_in_run_view(tmp_path):
    repo = _repo(tmp_path)
    prof = _prof(COLS, "y")
    r1 = repo.submit_snapshot(_run_snap("r@1", profile=prof), namespace=NS)
    assert r1["counts"]["task_profile"] == 1
    r2 = repo.submit_snapshot(_run_snap("r@1", profile=prof, tag="2"), namespace=NS)
    assert r2["counts"]["task_profile"] == 0                    # 内容相同不重复写
    assert repo.fingerprint_profiles()[A]["target_base"] == "y"
    assert repo.run_view(NS, A)["fingerprint_profiles"][A]["n_columns"] == 100


def test_feedback_tables_written_on_submit_and_backfill_idempotent(tmp_path):
    repo = _repo(tmp_path)
    logs = [{"kind": "search", "watermark": 3, "topics": ["feature.seq.gap"],
             "shown_ids": ["p1", "p2"]},
            {"kind": "planner-context", "watermark": 3, "shown_ids": ["p3"]}]
    trials = [_trial("t1", [("p1", "follows")]), _trial("t2", [("p9", "tests")], delta=-0.001,
                                                        verdict="rejected"),
              _trial("t3", novel=True)]
    rec = repo.submit_snapshot(_run_snap("r@1", logs=logs, trials=trials), namespace=NS)
    assert rec["counts"]["retrieval_logs"] == 2 and rec["counts"]["applications"] == 2
    # 同一 run 再提交一次(内容不同):已有的 (原则, run, trial) 不重复插入,也不报错
    rec2 = repo.submit_snapshot(_run_snap("r@1", logs=logs, trials=trials, tag="2"),
                                namespace=NS)
    assert rec2["counts"]["applications"] == 0
    assert rec2["counts"]["retrieval_logs"] == 0            # 遥测已入库的条目不再重复写
    assert repo.feedback_backfill(NS)["applications"] == 0
    st = repo.feedback_stats(NS)
    assert st["n_shown"] == 3 and st["n_retrievals"] == 2   # 3 张卡,不是 6 张
    assert st["n_trials"] == 3                              # 重新导出的快照不重复计 trial
    assert st["n_applications"] == 2
    assert st["cited_not_shown"] == 1                        # p9 被引用但从未展示
    assert st["trials"]["novel"] == 1 and st["trials"]["with_basis"] == 2
    p1 = next(r for r in st["principles"] if r["principle_id"] == "p1")
    assert p1["cited"] == 1 and p1["mean_gain_when_cited"] == 0.002


def test_backfill_covers_snapshots_submitted_before_v2181(tmp_path):
    repo = _repo(tmp_path)
    repo.submit_snapshot(_run_snap("r@1", logs=[{"kind": "search", "shown_ids": ["p1"]}],
                                   trials=[_trial("t1", [("p1", "follows")])]), namespace=NS)
    from sqlalchemy import delete

    from runtime.library import schema as S
    with repo.store._write() as conn:                       # 模拟 v2.18.0 时代:两表为空
        conn.execute(delete(S.retrieval_logs))
        conn.execute(delete(S.applications))
    got = repo.feedback_backfill(NS)
    assert got["retrieval_logs"] == 1 and got["applications"] == 1
    assert repo.feedback_backfill(NS)["commit_seq"] is None   # 幂等


def test_suggest_aliases_uses_profiles(tmp_path):
    repo = _repo(tmp_path)
    for atid, cols in ((A, COLS), (A_NEAR, COLS + ["new_a"]),
                       (A_REL, [f"z{i}" for i in range(80)])):
        snap = _run_snap(f"{atid}@1", profile=_prof(cols, TP.profile_from_atid(atid)
                                                    ["target_name"]))
        snap["run"]["atid"] = atid
        repo.submit_snapshot(snap, namespace=NS)
    sug = repo.suggest_aliases()
    assert len(sug) == 1 and sug[0]["alias_atid"] == A_NEAR and sug[0]["canonical_atid"] == A
    repo.add_fingerprint_alias(A_NEAR, A, reason="ok")
    assert repo.suggest_aliases() == []


# ---------------------------------------------------------------- 合并复核

def test_post_merge_review_groups_by_topic_and_respects_contest(monkeypatch):
    cur = [_p("a", 1, ["feature.seq.gap"]), _p("b", 0, ["hpo.search"]),
           _p("c", -1, ["feature.seq.gap"]), _p("d", 0, ["feature.seq.gap"])]
    cur[3]["contested_with"] = ["a"]
    monkeypatch.setattr(CU, "principles_current", lambda store, ns: cur)
    # 新写入 b(另一主题):不报
    assert CU._post_merge_review(None, NS, {"principle_versions": [cur[1]]}) == []
    # 新写入 a:与 c 同主题相反 → 报;与 d 已登记 contest → 豁免
    flags = CU._post_merge_review(None, NS, {"principle_versions": [cur[0]]})
    assert len(flags) == 1 and flags[0]["negative_principles"] == ["c"]
    # 本次决议里登记了 a↔c 的 contest → 不报
    plan = {"principle_versions": [cur[0]],
            "governance": [{"action": "contest", "subject_id": "c", "payload": {"with": ["a"]}}]}
    assert CU._post_merge_review(None, NS, plan) == []


# ---------------------------------------------------------------- 服务:续约与认证

def _http(repo, tokens=None, token=None):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from service.app import create_app

    client = TestClient(create_app(repo, tokens=tokens))

    def transport(method, url, headers, body, timeout):
        r = client.request(method, url.split("testserver", 1)[1], headers=headers,
                           content=body)
        return r.status_code, dict(r.headers), r.content

    return HttpRepository("http://testserver", transport=transport, token=token,
                          actor={"user": "self-claimed", "host": "h", "team": "x",
                                 "run": "r", "model": "m"})


def _legacy_snapshot(repo):
    repo.init_topics(NS, [{"code": "feature.seq", "definition": "d", "aliases": []}],
                     reason="v1")
    return repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_principles",
                                 "run": {"run_id": "legacy:x", "atid": A},
                                 "discoveries": [{"local_id": "d1", "payload": {"claim": "c"}}]},
                                namespace=NS)["snapshot_id"]


def test_renew_lease_over_http(tmp_path):
    repo = _repo(tmp_path)
    http = _http(repo)
    sid = _legacy_snapshot(repo)
    ctx = http.merge_context(NS, sid)
    lease = http.renew_lease(NS, ctx["lease"]["token"], extra_ttl=60)
    assert lease["expires_at"] == pytest.approx(ctx["lease"]["expires_at"] + 60)
    from runtime.library import LibraryError
    with pytest.raises(LibraryError):
        http.renew_lease(NS, "wrong-token")


def test_token_auth(tmp_path):
    from runtime.library import LibraryError
    from runtime.library.identity import token_hash

    repo = _repo(tmp_path)
    tokens = {token_hash("rw"): {"user": "alice", "team": "risk", "write": True},
              token_hash("ro"): {"user": "bob", "team": "risk", "write": False}}
    assert _http(repo, tokens).health()["schema_revision"]          # health 免认证
    with pytest.raises(LibraryError) as e:
        _http(repo, tokens).watermark()
    assert e.value.status == 401
    with pytest.raises(LibraryError) as e:
        _http(repo, tokens, token="ro").init_topics(NS, [], reason="x")
    assert e.value.status == 403
    assert _http(repo, tokens, token="ro").watermark() >= 0
    _http(repo, tokens, token="rw").submit_snapshot(_run_snap("r@1"), namespace=NS)
    actor = repo.commits()[-1]["actor"]
    assert actor["user"] == "alice" and actor["team"] == "risk" and actor["auth"] == "token"


def test_load_tokens(tmp_path):
    from runtime.library.identity import load_tokens, token_hash

    f = tmp_path / "tokens.yaml"
    f.write_text(f"tokens:\n  - sha256: {token_hash('a')}\n    user: u1\n"
                 "  - token: b\n    user: u2\n    write: false\n", encoding="utf-8")
    t = load_tokens(f)
    assert t[token_hash("a")]["write"] and not t[token_hash("b")]["write"]
    f.write_text("tokens: [{token: c}]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tokens(f)


def test_server_refuses_open_bind_without_tokens(tmp_path):
    from service import server

    assert server.main(["--root", str(tmp_path), "--host", "0.0.0.0"]) == 2


def test_put_task_profile_and_object_family_overlap(tmp_path):
    repo = _repo(tmp_path)
    prof = _prof(COLS, "y")
    seq = repo.put_task_profile(A, prof)
    assert seq and repo.put_task_profile(A, prof) is None        # 内容未变不提交
    assert repo.fingerprint_profiles()[A]["n_columns"] == 100
    # 原则 objects 推出的列族:单独作 objects 维度(重合系数),不冒充真实列重合
    _sim, dims = TP.similarity({"object_families": ["seq_gap_"]},
                               {"object_families": ["seq_gap_", "base_x_"]})
    assert "columns" not in dims and dims["objects"] == 1.0
    _sim, dims2 = TP.similarity({"object_families": ["seq_gap_"]},
                                {"object_families": ["base_x_", "seqp_"]})
    assert dims2["objects"] == 0.0
    # 业务域用重合系数:存量任务的业务域是多条原则的并集,新任务只有一个也算"同业务"
    _s, d2 = TP.similarity({"business_domains": ["credit"]},
                           {"business_domains": ["credit", "a", "b", "c"]})
    assert d2["domain"] == 1.0


def test_suggest_aliases_compares_canonical_only(tmp_path):
    repo = _repo(tmp_path)
    for atid in (A, A_NEAR):
        repo.put_task_profile(atid, _prof(COLS, "y"))
    third = "binary-y-200w_1000w-44444444"
    repo.put_task_profile(third, _prof(COLS, "y"))
    repo.add_fingerprint_alias(A_NEAR, A, reason="已确认")
    sug = repo.suggest_aliases()
    # A_NEAR 已是 A 的别名:只剩 third ↔ A 一条,且列数相同时先登记的 A 作规范
    assert [(s["alias_atid"], s["canonical_atid"]) for s in sug] == [(third, A)]


def test_prefill_telemetry_is_not_counted_as_missed(tmp_path, monkeypatch):
    """planner 预填的依据来自程序自己的检索:写 kind=planner-prefill,不算漏召回。"""
    repo = _repo(tmp_path)
    logs = [{"kind": "planner-prefill", "shown_ids": ["p1"]},
            {"kind": "search", "shown_ids": ["p2"]}]
    repo.submit_snapshot(_run_snap("r@1", logs=logs,
                                   trials=[_trial("t1", [("p1", "follows")]),
                                           _trial("t2", [("p3", "tests")])]), namespace=NS)
    st = repo.feedback_stats(NS)
    assert st["cited_not_shown"] == 1                        # 只有 p3 是真的没展示(p1 已预填)
    assert st["n_shown"] == 1                                # 预填不计入主动曝光


def test_health_exemption_is_exact_path(tmp_path):
    from runtime.library import LibraryError
    from runtime.library.identity import token_hash

    repo = _repo(tmp_path)
    tokens = {token_hash("rw"): {"user": "u", "team": "", "write": True}}
    assert _http(repo, tokens).health()["schema_revision"]
    with pytest.raises(LibraryError) as e:                   # 以 /health 结尾的其它路由仍需 token
        _http(repo, tokens).get_snapshot(NS, "health")
    assert e.value.status == 401


def test_backfill_dedupes_duplicate_trial_rows_in_batch(tmp_path):
    """同一 (run, trial) 出现在两份快照里(历史重复导入)时,补写不能撞唯一键,
    同一依据只记一次。"""
    repo = _repo(tmp_path)
    t = _trial("t1", [("p1", "follows")])
    repo.submit_snapshot(_run_snap("r@1", trials=[t]), namespace=NS)
    repo.submit_snapshot(_run_snap("r@1", trials=[{**t, "verdict": "rejected"}], tag="2"),
                         namespace=NS)
    from sqlalchemy import delete

    from runtime.library import schema as S
    with repo.store._write() as conn:                          # 模拟 v2.18.1 之前:表为空
        conn.execute(delete(S.applications))
    assert repo.feedback_backfill(NS)["applications"] == 1     # 修复前:撞唯一键 500
    assert repo.feedback_backfill(NS)["commit_seq"] is None
    got = repo.feedback_stats(NS)["principles"][0]
    assert got["cited"] == 1
