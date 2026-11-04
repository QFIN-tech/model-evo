"""v2.18.0 中心化经验库(M1):存储核心、配方规范化、种子导入、服务。

docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md §12 M1 验收项。
服务相关用例需要 fastapi(装在服务解释器里);PostgreSQL 用例需要
``MODELEVO_TEST_PG_URL``(Q37:无 PG 时跳过)。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import datetime as dt
import json
import os

import pytest
import yaml

from runtime.library import (ArtifactError, ArtifactStore, HttpRepository,
                             LibraryError, LibraryStore, LibraryUnavailable,
                             LibraryValidationError, LocalRepository)
from runtime.library import schema as S
from runtime.library.recipes import (RUN_DIR_TOKEN, SKILL_ROOT_TOKEN,
                                     canonicalize, check_python, render)
from runtime.library.snapshot import parse_atid, run_id_for

ATID = "binary-y-200w_1000w-bc5329dc"
NS = "test"


def _repo(tmp_path, url=None) -> LocalRepository:
    return LocalRepository(LibraryStore(url or f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _snap(**kw) -> dict:
    p = {"schema": "modelevo.snapshot/1", "kind": "run",
         "run": {"run_id": "run-a@00000000", "atid": ATID}}
    p.update(kw)
    return p


def _recipe(repo, code: str = "x = 1\n", **kw) -> dict:
    rid = repo.put_bundle({"train.py": code.encode()})
    return {"recipe_id": rid, "kind": "top_trial", "algorithm": "xgboost", **kw}


# ---------------------------------------------------------------- schema / 迁移

def test_migration_matches_schema(tmp_path):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    repo = _repo(tmp_path)
    assert repo.store.schema_revision() == S.SCHEMA_REVISION
    with repo.store.engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(
            conn, opts={"compare_type": True}), S.metadata)
    assert diff == []


# ---------------------------------------------------------------- 提交与水位

def test_snapshot_idempotent_and_watermark(tmp_path):
    repo = _repo(tmp_path)
    assert repo.watermark() == 0
    p = _snap(trial_outcomes=[{"run_id": "run-a@00000000", "trial_id": "t1",
                               "verdict": "accepted", "delta_parent": 0.002}])
    r1 = repo.submit_snapshot(p, namespace=NS)
    r2 = repo.submit_snapshot(p, namespace=NS)
    assert not r1["duplicate"] and r2["duplicate"]
    assert r1["snapshot_id"] == r2["snapshot_id"] and repo.watermark() == 1
    # 另一个命名空间是独立的
    assert not repo.submit_snapshot(p, namespace="other")["duplicate"]
    assert len(repo.list_snapshots(NS)) == 1
    assert repo.list_snapshots(NS)[0]["state"] == "submitted"
    assert repo.commits()[0]["actor"]["host"]


def test_reads_are_pinned_to_watermark(tmp_path):
    repo = _repo(tmp_path)
    t = {"run_id": "run-a@00000000", "trial_id": "t1", "verdict": "promising"}
    repo.submit_snapshot(_snap(trial_outcomes=[t]), namespace=NS)
    w1 = repo.watermark()
    repo.submit_snapshot(_snap(trial_outcomes=[{**t, "verdict": "accepted"}],
                               extra={"v": 2}), namespace=NS)
    assert [x["verdict"] for x in repo.trial_outcomes(NS)] == ["accepted"]
    assert [x["verdict"] for x in repo.trial_outcomes(NS, at=w1)] == ["promising"]
    # 缺 atid 的 trial 继承 run.atid,按 atid 能查到
    assert len(repo.trial_outcomes(NS, atid=ATID)) == 1


def test_validation_rejects_without_commit(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(LibraryValidationError) as e:
        repo.submit_snapshot({"schema": "x", "kind": "nope", "run": {}}, namespace=NS)
    assert any("schema" in m for m in e.value.errors)
    with pytest.raises(LibraryValidationError):
        repo.submit_snapshot(_snap(measurements=[
            {"recipe_id": "a" * 64, "kind": "export"}]), namespace=NS)
    with pytest.raises(LibraryValidationError):    # bundle 未上传
        repo.submit_snapshot(_snap(recipes=[{"recipe_id": "b" * 64,
                                             "kind": "top_trial"}]), namespace=NS)
    assert repo.watermark() == 0


def test_nan_is_stored_as_null(tmp_path):
    repo = _repo(tmp_path)
    repo.submit_snapshot(_snap(trial_outcomes=[{
        "run_id": "r@1", "trial_id": "t", "metric_value": float("nan"),
        "custom": {"x": float("inf")}}]), namespace=NS)
    row = repo.trial_outcomes(NS)[0]
    assert row["metric_value"] is None and row["extra"] == {"custom": {"x": None}}


# ---------------------------------------------------------------- 榜单与治理

def test_leaderboard_filters_and_governance(tmp_path):
    repo = _repo(tmp_path)
    good = _recipe(repo, "a = 1\n", n_trees=500)
    hpo = _recipe(repo, "b = 1\n", is_hpo=True)
    big = _recipe(repo, "c = 1\n", n_trees=9000)
    low = _recipe(repo, "d = 1\n")
    meas = [{"recipe_id": good["recipe_id"], "kind": "export", "gain": 0.01},
            {"recipe_id": hpo["recipe_id"], "kind": "export", "gain": 0.05},
            {"recipe_id": big["recipe_id"], "kind": "export", "gain": 0.04},
            {"recipe_id": low["recipe_id"], "kind": "export", "gain": 0.002}]
    repo.submit_snapshot(_snap(recipes=[good, hpo, big, low], measurements=meas),
                         namespace=NS)
    lb = repo.leaderboard(NS, ATID)
    assert [r["recipe_id"] for r in lb] == [good["recipe_id"], low["recipe_id"]]
    assert [r["rank"] for r in lb] == [1, 2]
    # 复测追加一行测量,榜单取最新
    repo.submit_snapshot(_snap(measurements=[
        {"recipe_id": low["recipe_id"], "kind": "warm_remeasure", "gain": 0.03}],
        recipes=[low]), namespace=NS)
    assert repo.leaderboard(NS, ATID)[0]["recipe_id"] == low["recipe_id"]
    repo.govern(NS, [{"subject_type": "recipe", "subject_id": low["recipe_id"],
                      "action": "exclude", "reason": "t"}])
    assert [r["recipe_id"] for r in repo.leaderboard(NS, ATID)] == [good["recipe_id"]]
    repo.govern(NS, [{"subject_type": "recipe", "subject_id": low["recipe_id"],
                      "action": "restore", "reason": "t"}])
    assert len(repo.leaderboard(NS, ATID)) == 2
    with pytest.raises(LibraryValidationError):
        repo.govern(NS, [{"subject_type": "bogus", "subject_id": "x", "action": "y"}])


def test_lower_is_better_gain(tmp_path):
    repo = _repo(tmp_path)
    a, b = _recipe(repo, "a = 1\n"), _recipe(repo, "b = 2\n")
    repo.submit_snapshot(_snap(recipes=[a, b], measurements=[
        {"recipe_id": a["recipe_id"], "kind": "export", "delta_baseline": -100.0,
         "higher_is_better": False},
        {"recipe_id": b["recipe_id"], "kind": "export", "delta_baseline": -250.0,
         "higher_is_better": False}]), namespace=NS)
    lb = repo.leaderboard(NS, ATID)
    assert lb[0]["recipe_id"] == b["recipe_id"] and lb[0]["measurement"]["gain"] == 250.0


def test_frozen_baseline_prefers_pin(tmp_path):
    repo = _repo(tmp_path)
    b1 = _recipe(repo, "b1 = 1\n", kind="baseline")
    b2 = _recipe(repo, "b2 = 1\n", kind="baseline")
    repo.submit_snapshot(_snap(recipes=[b1, b2], governance=[{
        "subject_type": "recipe", "subject_id": b1["recipe_id"],
        "action": "pin_baseline"}]), namespace=NS)
    assert repo.frozen_baseline(NS, ATID, "xgboost")["recipe_id"] == b1["recipe_id"]
    assert repo.frozen_baseline(NS, ATID, "lightgbm") is None


def test_fingerprint_alias(tmp_path):
    repo = _repo(tmp_path)
    alias = "binary-y-unknown-unverified"
    repo.submit_snapshot(_snap(run={"run_id": "old@1", "atid": alias},
                               trial_outcomes=[{"run_id": "old@1", "trial_id": "t"}]),
                         namespace=NS)
    assert repo.trial_outcomes(NS, atid=ATID) == []
    repo.add_fingerprint_alias(alias, ATID, reason="迁移降级条目归位")
    assert repo.resolve_atid(alias) == ATID
    assert len(repo.trial_outcomes(NS, atid=ATID)) == 1
    with pytest.raises(LibraryValidationError):
        repo.add_fingerprint_alias(ATID, alias, reason="成环")
    repo.add_fingerprint_alias(alias, alias, reason="撤销")
    assert repo.trial_outcomes(NS, atid=ATID) == []
    fp = {f["atid"]: f for f in repo.fingerprints()}
    assert fp[ATID]["data_base"] == "200w_1000w-bc5329dc"
    assert fp[alias]["data_base"] is None


def test_parse_atid_and_run_id():
    p = parse_atid("reg-loan_amt_t7_log1p-200w_1000w-bc5329dc")
    assert (p["model_type"], p["target_name"], p["data_base"]) == \
        ("reg", "loan_amt_t7_log1p", "200w_1000w-bc5329dc")
    assert parse_atid("nodash") is None
    a, b = run_id_for("/x/run1/run-req"), run_id_for("/x/run2/run-req")
    assert a.startswith("run-req@") and a != b


# ---------------------------------------------------------------- 配方与产物

def test_canonicalize_render_roundtrip():
    run = "/data/s/run4-v2.16.0-2"
    files = {"train.py": (f"import pathlib\nRUN_DIR = pathlib.Path('{run}')\n"
                          "SKILL_ROOT = '/data/c/skill'\n"
                          f"OTHER = '{run}.opus5.5/x'\n").encode(),
             "trial.yaml": f"evaluation_path: {run}/trials/t1\n".encode(),
             "task-spec.yaml": b"path: ../data/train_400w-part-*.parquet\n"}
    canon, info = canonicalize(files)
    text = canon["train.py"].decode()
    assert f"pathlib.Path('{RUN_DIR_TOKEN}')" in text
    assert f"SKILL_ROOT = '{SKILL_ROOT_TOKEN}'" in text
    assert f"'{run}.opus5.5/x'" in text            # 前缀相同的另一个 run 不被吃掉
    assert canon["trial.yaml"].decode().startswith(f"evaluation_path: {RUN_DIR_TOKEN}/")
    assert info["residual_abs_paths"] == [f"{run}.opus5.5/x"]   # 相对路径不算
    # 同一配方在不同机器上哈希一致
    other = {k: v.replace(b"/data/s/", b"/mnt/z/").replace(b"/data/c/", b"/opt/")
             for k, v in files.items()}
    other["train.py"] = other["train.py"].replace(b"/mnt/z/run4-v2.16.0-2.opus5.5",
                                                  f"{run}.opus5.5".encode())
    assert canonicalize(other)[0] == canon
    out = render(canon, run_dir="/new/run", skill_root="/new/skill")
    assert "RUN_DIR = pathlib.Path('/new/run')" in out["train.py"].decode()
    assert check_python(out) == []
    assert check_python(canon) == ["train.py: 仍含未渲染的占位符"]
    assert check_python({"a.py": b"def (:\n"})[0].startswith("a.py:1")


def test_artifact_store_verifies_hash(tmp_path):
    store = ArtifactStore(f"file://{tmp_path}/cas")
    h = store.put_bytes(b"hello")
    assert store.put_bytes(b"hello") == h and store.get_bytes(h) == b"hello"
    with pytest.raises(ArtifactError):
        store.put_bytes(b"x", expected="0" * 64)
    path = pathlib.Path(store._path(h))
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactError):
        store.get_bytes(h)
    assert list(store.iter_hashes()) == [h]


def test_bundle_materialize_refuses_bad_code(tmp_path):
    repo = _repo(tmp_path)
    rid = repo.put_bundle({"train.py": b"def broken(:\n", "sub/a.yaml": b"k: 1\n"})
    dest = tmp_path / "out"
    assert repo.materialize(rid, dest, run_dir="/r", skill_root="/s")
    assert not dest.exists()
    with pytest.raises(ArtifactError):
        repo.put_bundle({"../escape.py": b"x"})


# ---------------------------------------------------------------- 导出导入

def test_dump_load_roundtrip(tmp_path):
    from runtime.library.dump import dump, load

    src = _repo(tmp_path / "a")
    r = _recipe(src, "z = 1\n")
    src.submit_snapshot(_snap(recipes=[r], measurements=[
        {"recipe_id": r["recipe_id"], "kind": "export", "gain": 0.1}],
        trial_outcomes=[{"run_id": "r@1", "trial_id": "t"}]), namespace=NS)
    src.add_fingerprint_alias("binary-y-unknown-unverified", ATID, reason="x")
    dump(src.store, src.artifacts, tmp_path / "dump")
    dst = _repo(tmp_path / "b")
    load(dst.store, dst.artifacts, tmp_path / "dump")
    assert dst.commits() == src.commits()
    assert dst.leaderboard(NS, ATID) == src.leaderboard(NS, ATID)
    assert dst.get_bundle(r["recipe_id"]) == {"train.py": b"z = 1\n"}
    assert dst.govern(NS, [{"subject_type": "recipe", "subject_id": r["recipe_id"],
                            "action": "exclude"}]) == src.watermark() + 1
    with pytest.raises(ValueError):                 # 目标非空
        load(dst.store, dst.artifacts, tmp_path / "dump")


# ---------------------------------------------------------------- 种子导入

def _legacy_fixture(root: pathlib.Path) -> dict:
    mem = root / "memory"
    (mem / "global").mkdir(parents=True)
    rows = [
        {"experience_id": "exp-202609-aaaa", "claim": "seq 计数无增益", "category":
         "feature_engineering", "outcome": "negative", "condition": {},
         "evidence": {"task_ids": ["run-x"], "abstract_task_ids": [ATID]}},
        {"experience_id": "exp-202610-bbbb", "claim": "lr 0.0125 更好", "category": "hpo",
         "outcome": "positive", "condition": {}, "evidence": {"task_ids": ["run-y"]},
         "created_at": "2026-10-01T00:00:00+08:00"},
        {"experience_id": f"top-trial-{ATID}-rank1", "category": "top_trial_baseline",
         "claim": "idx"},
    ]
    (mem / "global" / "principles.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    extra = root / "extra.jsonl"
    extra.write_text(json.dumps({**rows[0], "experience_id": "dup-by-claim"},
                                ensure_ascii=False) + "\n" +
                     json.dumps({"experience_id": "exp-x", "claim": "新的", "category": "hpo",
                                 "outcome": "neutral", "condition": {},
                                 "evidence": {"task_ids": ["run-z"]}},
                                ensure_ascii=False) + "\n", encoding="utf-8")
    run = root / "runs" / "run-src"
    adir = mem / "top-trials" / ATID
    code = f"import pathlib\nRUN_DIR = pathlib.Path('{run}')\nSKILL_ROOT = '/s'\n"
    for name, tid, extra_files in (("rank1", "trial-002", {"hypothesis.md": "h"}),
                                   ("rank2", "trial-001", {}),
                                   ("baseline/xgboost", "trial-001", {})):
        d = adir / name
        d.mkdir(parents=True)
        (d / "train.py").write_text(code + ("y = 2\n" if name == "rank1" else ""))
        (d / "trial.yaml").write_text(yaml.safe_dump(
            {"trial_id": tid, "algorithm": "xgboost",
             "hyperparameters": {"n_estimators": 800}}))
        (d / "evaluation.json").write_text(json.dumps(
            {"primary_metric": {"name": "auc", "value": 0.8}}))
        for k, v in extra_files.items():
            (d / k).write_text(v)
    (adir / "baseline/xgboost/frozen.yaml").write_text(yaml.safe_dump(
        {"source_run": str(run), "source_trial": "trial-001", "algorithm": "xgboost"}))
    (adir / "leaderboard.yaml").write_text(yaml.safe_dump({
        "metric_name": "auc", "higher_is_better": True, "rows": [
            {"trial_id": "trial-002", "metric": 0.81, "delta_vs_baseline": 0.01,
             "improvement_vs_baseline": 0.01, "source_run": str(run)},
            {"trial_id": "trial-001", "metric": 0.80, "delta_vs_baseline": 0.0,
             "improvement_vs_baseline": 0.0, "source_run": str(run)}]}))
    (mem / "top-trials" / f"{ATID}.bak-20261022").mkdir()
    (run / "trials" / "trial-001").mkdir(parents=True)
    (run / "run-state.json").write_text(json.dumps(
        {"abstract_task_id": ATID, "created_at": "2026-10-20T00:00:00+08:00"}))
    (run / "trials" / "trial-001" / "trial.yaml").write_text(yaml.safe_dump({
        "trial_id": "trial-001", "generation": 0, "search_action": "explore",
        "algorithm": "xgboost", "status": "EVALUATED", "family_hints": ["info:seq:window"],
        "hypothesis": {"statement": "baseline"}}))
    (run / "trials" / "trial-001" / "evaluation.json").write_text(json.dumps({
        "verdict": "accepted", "primary_metric": {"name": "auc", "value": 0.8},
        "delta": {"vs_parent": 0.001, "vs_parent_paired_bootstrap": {
            "delta_ci": [0.0005, 0.0015], "p_positive": 0.99}, "vs_baseline": 0.0}}))
    smoke = root / "runs" / "smoke-run"
    (smoke / "trials" / "t").mkdir(parents=True)
    (smoke / "run-state.json").write_text("{}")
    (smoke / "trials" / "t" / "trial.yaml").write_text(yaml.safe_dump({"trial_id": "t"}))
    return {"memory": mem, "extra": extra, "runs": root / "runs"}


def test_legacy_import_idempotent(tmp_path):
    from runtime.library.legacy_import import import_legacy

    fx = _legacy_fixture(tmp_path / "src")
    repo = _repo(tmp_path / "lib")
    kw = dict(namespace="prod", principle_sources=[
        ("main", fx["memory"] / "global" / "principles.jsonl"), ("extra", fx["extra"])],
        top_trials_dir=fx["memory"] / "top-trials", run_roots=[fx["runs"]])
    r1 = import_legacy(repo, **kw)
    assert r1["errors"] == []
    assert r1["summary"]["discoveries"] == 3            # top-trial 索引与按 claim 重复的被剔除
    assert r1["principle_sources"]["extra"]["duplicates"] == 1
    # baseline 与 rank2 内容相同 → 只登记一次,kind=baseline
    assert r1["top_trials"][ATID]["same_content_as_baseline"] == ["trial-001"]
    lb = repo.leaderboard("prod", ATID)
    assert [r["source_trial_id"] for r in lb] == ["trial-002"]
    assert lb[0]["n_trees"] == 800 and lb[0]["runtime_api_version"] == "2.16"
    base = repo.frozen_baseline("prod", ATID, "xgboost")
    assert base["source_trial_id"] == "trial-001"
    files = repo.get_bundle(base["recipe_id"])
    assert "frozen.yaml" not in files and RUN_DIR_TOKEN.encode() in files["train.py"]
    t = repo.trial_outcomes("prod", atid=ATID)
    assert len(t) == 1 and t[0]["delta_parent_ci_low"] == 0.0005 and \
        t[0]["hypothesis"] == "baseline" and t[0]["family_hints"] == ["info:seq:window"]
    assert len(repo.list_snapshots("test")) == 1        # smoke run 进 test 命名空间
    w = repo.watermark()
    r2 = import_legacy(repo, **kw)
    assert r2["principles_already_imported"] == 3 and repo.watermark() == w
    assert all(s["receipt"]["duplicate"] for s in r2["snapshots"])


def test_legacy_import_dry_run_needs_no_repo(tmp_path):
    from runtime.library.legacy_import import import_legacy

    fx = _legacy_fixture(tmp_path / "src")
    r = import_legacy(None, namespace="prod", dry_run=True,
                      principle_sources=[("main", fx["memory"] / "global" / "principles.jsonl")],
                      top_trials_dir=fx["memory"] / "top-trials", run_roots=[fx["runs"]])
    assert r["summary"]["snapshots"] == 5 and "receipt" not in r["snapshots"][0]


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
                          actor={"user": "u", "host": "h", "team": "t",
                                 "run": "r", "model": "m"})


def test_service_matches_local(tmp_path):
    local = _repo(tmp_path)
    http = _http(local)
    assert http.health()["schema_revision"] == S.SCHEMA_REVISION
    r = _recipe(http, "q = 1\n", kind="baseline")
    rec = http.submit_snapshot(_snap(recipes=[r], measurements=[
        {"recipe_id": r["recipe_id"], "kind": "baseline", "value": 0.8}],
        trial_outcomes=[{"run_id": "r@1", "trial_id": "t"}]), namespace=NS)
    assert not rec["duplicate"]
    assert http.commits()[0]["actor"]["model"] == "m"
    assert http.trial_outcomes(NS, atid=ATID) == local.trial_outcomes(NS, atid=ATID)
    assert http.frozen_baseline(NS, ATID, "xgboost")["recipe_id"] == r["recipe_id"]
    assert http.frozen_baseline(NS, ATID, "lightgbm") is None
    assert http.get_snapshot(NS, "0" * 64) is None
    assert http.get_snapshot(NS, rec["snapshot_id"])["n_trial_outcomes"] == 1
    assert http.measurements(NS, recipe_ids=[r["recipe_id"]])[0]["value"] == 0.8
    assert http.set_snapshot_state(NS, rec["snapshot_id"], "merged") > rec["commit_seq"]
    assert http.list_snapshots(NS, state="merged")[0]["snapshot_id"] == rec["snapshot_id"]
    http.add_fingerprint_alias("binary-y-unknown-unverified", ATID, reason="x")
    assert http.resolve_atid("binary-y-unknown-unverified") == ATID
    assert http.governance(NS) == []
    with pytest.raises(LibraryError) as e:
        http.submit_snapshot({"schema": "bad"}, namespace=NS)
    assert e.value.status == 422 and e.value.errors
    with pytest.raises(ArtifactError):
        http.get_artifact("f" * 64)


def test_http_unavailable_raises():
    repo = HttpRepository("http://127.0.0.1:9", timeout=2)
    with pytest.raises(LibraryUnavailable):
        repo.health()


def test_sqlite_backup_and_prune(tmp_path):
    from service.server import backup_sqlite

    _repo(tmp_path)
    bdir = tmp_path / "backups"
    assert backup_sqlite(str(tmp_path / "lib.db"), bdir, 30, today=dt.date(2026, 10, 1))
    assert backup_sqlite(str(tmp_path / "lib.db"), bdir, 30,
                         today=dt.date(2026, 10, 1)) is None      # 当天已备份
    backup_sqlite(str(tmp_path / "lib.db"), bdir, 30, today=dt.date(2026, 11, 15))
    assert sorted(p.name for p in bdir.iterdir()) == ["library-20261115.db"]


# ---------------------------------------------------------------- PostgreSQL(Q37)

@pytest.mark.skipif(not os.environ.get("MODELEVO_TEST_PG_URL"),
                    reason="需要 MODELEVO_TEST_PG_URL(空库)")
def test_postgres_backend(tmp_path):
    url = os.environ["MODELEVO_TEST_PG_URL"]
    repo = _repo(tmp_path, url=url)
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    with repo.store.engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(
            conn, opts={"compare_type": True}), S.metadata) == []
    w0 = repo.watermark()
    r = _recipe(repo, "pg = 1\n")
    rec = repo.submit_snapshot(_snap(recipes=[r], measurements=[
        {"recipe_id": r["recipe_id"], "kind": "export", "gain": float("nan"),
         "extra_field": float("inf")}],
        trial_outcomes=[{"run_id": "r@1", "trial_id": "t", "x": {"y": float("nan")}}]),
        namespace=f"pg-{os.getpid()}")
    assert rec["commit_seq"] == w0 + 1
    ns = f"pg-{os.getpid()}"
    assert repo.leaderboard(ns, ATID)[0]["measurement"]["gain"] is None
    assert repo.trial_outcomes(ns)[0]["extra"] == {"x": {"y": None}}
    assert repo.submit_snapshot(_snap(recipes=[r], measurements=[
        {"recipe_id": r["recipe_id"], "kind": "export", "gain": float("nan"),
         "extra_field": float("inf")}],
        trial_outcomes=[{"run_id": "r@1", "trial_id": "t", "x": {"y": float("nan")}}]),
        namespace=ns)["duplicate"]


# ---------------------------------------------------------------- 榜单去重 / 契约版本

def _bundle_recipe(repo, *, run_dir="{{MODELEVO_RUN_DIR}}", seed=1, depth=6, **kw) -> dict:
    code = (f"SEED = {seed}\nRUN_DIR = pathlib.Path('{run_dir}')\n"
            "def fit_and_score(): pass\n").encode()
    cfg = yaml.safe_dump({"algorithm": "xgboost", "hyperparameters": {"max_depth": depth},
                          "seed": seed, "trial_id": kw.get("source_trial_id")}).encode()
    rid = repo.put_bundle({"train.py": code, "trial.yaml": cfg})
    return {"recipe_id": rid, "kind": "top_trial", "algorithm": "xgboost", **kw}


def test_leaderboard_one_rank_per_recipe(tmp_path):
    """同一配方只占一个名次:同一 trial 的两份副本(一份 RUN_DIR 写死别的 run)、
    confirm 子代(只换 seed)都并入同一组;超参不同的是不同配方。组内保留规范化
    干净的一份,其次增益高者。"""
    repo = _repo(tmp_path)
    run = {"source_run_id": "run-a@00000000"}
    parent = _bundle_recipe(repo, seed=1, source_trial_id="trial-031", **run)
    clean = _bundle_recipe(repo, seed=2, source_trial_id="trial-032", **run,
                           canonicalization={"residual_abs_paths": []})
    dirty = _bundle_recipe(repo, seed=2, run_dir="/data/other/run", source_trial_id="trial-032",
                           **run, canonicalization={"residual_abs_paths": ["/data/other/run"]})
    other = _bundle_recipe(repo, seed=3, depth=8, source_trial_id="trial-013", **run)
    gains = ((parent, 0.0366), (clean, 0.0367), (dirty, 0.0367), (other, 0.0300))
    repo.submit_snapshot(_snap(recipes=[r for r, _g in gains], measurements=[
        {"recipe_id": r["recipe_id"], "kind": "legacy", "gain": g} for r, g in gains]),
        namespace=NS)
    lb = repo.leaderboard(NS, ATID)
    assert [(r["recipe_id"], r["family_size"]) for r in lb] == [
        (clean["recipe_id"], 3), (other["recipe_id"], 1)]
    assert [r["rank"] for r in lb] == [1, 2]
    # 取不到 bundle 时按来源 trial 去重(不同 run 的同名 trial 是两个配方)
    lb2 = repo.store.leaderboard(NS, ATID)
    assert {r["source_trial_id"] for r in lb2} == {"trial-031", "trial-032", "trial-013"}


def test_leaderboard_merges_reordered_recipe(tmp_path):
    """v2.18.4:语句顺序 / 注释排版不同、训练行为相同的两份配方只占一个名次。

    实测 req-t7 榜单 rank-1/rank-2 逐字节差 435 行,模型工件 sha256 全同,两份都被
    热启动。按行去 SEED/RUN_DIR 的旧口径把它们当成两个配方。
    """
    repo = _repo(tmp_path)
    cfg = yaml.safe_dump({"algorithm": "xgboost",
                          "hyperparameters": {"max_depth": 6}}).encode()
    a = ("# rank-1\nSEED = 1\nRUN_DIR = pathlib.Path('{{MODELEVO_RUN_DIR}}')\n"
         "FEATS = ['a', 'b']\nDEPTH = 6\n\ndef fit_and_score():\n    return FEATS\n")
    b = ("DEPTH = 6  # 注释排版不同\nRUN_DIR = pathlib.Path('/data/other/run')\n"
         "def fit_and_score():\n    return FEATS\nFEATS = ['a', 'b']\nSEED = 7\n")
    c = a.replace("DEPTH = 6", "DEPTH = 8")
    recs = []
    for i, code in enumerate((a, b, c)):
        rid = repo.put_bundle({"train.py": code.encode(), "trial.yaml": cfg})
        recs.append({"recipe_id": rid, "kind": "top_trial", "algorithm": "xgboost",
                     "source_run_id": f"run-{i}@00000000",
                     "source_trial_id": f"trial-00{i}"})
    gains = (0.0367, 0.0366, 0.0300)
    repo.submit_snapshot(_snap(recipes=recs, measurements=[
        {"recipe_id": r["recipe_id"], "kind": "legacy", "gain": g}
        for r, g in zip(recs, gains)]), namespace=NS)
    lb = repo.leaderboard(NS, ATID)
    assert [(r["recipe_id"], r["family_size"]) for r in lb] == [
        (recs[0]["recipe_id"], 2), (recs[2]["recipe_id"], 1)]


def test_leaderboard_runtime_api_version_filter(tmp_path):
    repo = _repo(tmp_path)
    cur = _recipe(repo, "a = 1\n", runtime_api_version="2.16")
    new = _recipe(repo, "b = 1\n", runtime_api_version="2.20")
    repo.submit_snapshot(_snap(recipes=[cur, new], measurements=[
        {"recipe_id": cur["recipe_id"], "kind": "export", "gain": 0.01},
        {"recipe_id": new["recipe_id"], "kind": "export", "gain": 0.02}]), namespace=NS)
    assert len(repo.leaderboard(NS, ATID)) == 2                         # 不给版本:不过滤
    view = repo.run_view(NS, ATID, runtime_api_version="2.16")
    assert [r["recipe_id"] for r in view["leaderboard"]] == [cur["recipe_id"]]
    assert len(repo.run_view(NS, ATID, runtime_api_version="2.20")["leaderboard"]) == 2


def test_canonicalize_rewritten_run_dir_line():
    """配方被后续 run 热启动后 RUN_DIR 已改写成那个 run,再按来源 run 目录导出:
    两个路径都要换成占位符,不留残留绝对路径。"""
    src = "/data/share/run-src"
    code = ("RUN_DIR = pathlib.Path('/data/share/run-other/run-x')\n"
            f"SKILL_ROOT = '/data/skill'\nA = '{src}/sample-profile'\n").encode()
    out, info = canonicalize({"train.py": code}, run_dir=src)
    text = out["train.py"].decode()
    assert f"RUN_DIR = pathlib.Path('{RUN_DIR_TOKEN}')" in text
    assert f"A = '{RUN_DIR_TOKEN}/sample-profile'" in text
    assert f"SKILL_ROOT = '{SKILL_ROOT_TOKEN}'" in text
    assert info["residual_abs_paths"] == []
