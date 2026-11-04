"""v2.18.1:存量指纹修复(library fix-fingerprints)。

覆盖:按 run 目录重推 atid、splits 列集反推原生面、漂移指纹优先登记别名、
原地修正 trial 行并留痕、修复后 relink、幂等。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import hashlib
import json

import pytest

from runtime.library import ArtifactStore, LibraryStore, LocalRepository
from runtime.library.hygiene import legacy_fingerprint as LF

NS = "t"
COLS = [f"c{i}" for i in range(30)]


def _h(cols):
    return hashlib.sha256("|".join(sorted(cols)).encode()).hexdigest()[:8]


def _write_parquet(path, cols, rows=250_000):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pq.write_table(pa.table({c: [0] * rows for c in cols}), str(path))


def _run_dir(tmp_path, name, *, cols=COLS, rows=250_000, target="y", mt="binary_classification",
             splits_cols=None, raw_hash=None, atid=None):
    r = tmp_path / name
    (r / "data").mkdir(parents=True)
    (r / "sample-profile").mkdir()
    if cols:
        _write_parquet(r / "data" / "train_sample.parquet", cols, rows)
    prof = {"n_rows": rows}
    if raw_hash:
        prof.update({"raw_columns_hash": raw_hash, "raw_columns_count": len(cols)})
    if splits_cols:
        (r / "sample-profile" / "splits").mkdir()
        _write_parquet(r / "sample-profile" / "splits" / "train.parquet", splits_cols, rows)
    (r / "sample-profile" / "sample-profile.json").write_text(json.dumps(prof))
    (r / "task-spec.yaml").write_text(
        f"model_type: {mt}\nprimary_metric: auc\ntarget:\n  name: {target}\n")
    (r / "run-state.json").write_text(json.dumps({"abstract_task_id": atid}))
    return r


def _repo(tmp_path):
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _snapshot(repo, run_id, atid, trials=()):
    return repo.submit_snapshot({
        "schema": "modelevo.snapshot/1", "kind": "run",
        "run": {"run_id": run_id, "atid": atid},
        "trial_outcomes": [{"run_id": run_id, "trial_id": t, "atid": atid}
                           for t in trials]}, namespace=NS)


def test_derive_atid_from_run_dir(tmp_path):
    r = _run_dir(tmp_path, "r1")
    d = LF.derive_atid(r)
    assert d["atid"] == f"binary-y-10w_50w-{_h(COLS)}"
    assert d["evidence"]["columns_source"].endswith("train_sample.parquet")
    # run-state 里已有可信 atid 时以它为准
    r2 = _run_dir(tmp_path, "r2", atid="binary-y-10w_50w-deadbeef")
    assert LF.derive_atid(r2)["atid"] == "binary-y-10w_50w-deadbeef"
    # 量级与列集都拿不到:不出 atid,只报告问题
    r3 = tmp_path / "r3"
    (r3 / "sample-profile").mkdir(parents=True)
    (r3 / "task-spec.yaml").write_text("model_type: binary_classification\ntarget:\n  name: y\n")
    (r3 / "run-state.json").write_text("{}")
    d3 = LF.derive_atid(r3)
    assert d3["atid"] is None and {"量级不可得(n_rows 缺失)", "列集不可得"} <= set(d3["problems"])


def test_splits_inferred_from_known_mapping(tmp_path):
    """只有 splits 列集的 run:靠"已知 run 的 splits↔原生面"对照推出原生 hash。"""
    native = COLS
    splits = COLS[:25]
    _run_dir(tmp_path, "known", cols=native, splits_cols=splits)
    only = _run_dir(tmp_path, "sp", cols=[], splits_cols=splits)     # 没有 data/ 分片
    derived = {k: LF.derive_atid(v) for k, v in
               {"known": tmp_path / "known", "sp": only}.items()}
    assert derived["known"]["evidence"]["col_hash8"] == _h(native)
    assert derived["sp"]["atid"] is None            # 单独看推不出来
    LF._infer_from_splits(derived)
    assert derived["sp"]["atid"] == f"binary-y-10w_50w-{_h(native)}"
    assert "splits 列集" in derived["sp"]["evidence"]["columns_source"]


def test_apply_retags_in_place_and_alias(tmp_path):
    repo = _repo(tmp_path)
    right = f"binary-y-10w_50w-{_h(COLS)}"
    drifted = "binary-y-10w_50w-11111111"
    _run_dir(tmp_path, "run-a", cols=COLS)                       # 推出 right
    _run_dir(tmp_path, "run-b", cols=COLS)
    # 库里:run-a 的 trial 写成了漂移指纹;另有一个 run 的 trial 完全没有 atid
    _snapshot(repo, "run-a@1", drifted, trials=["t1", "t2"])
    _snapshot(repo, "run-b@1", None, trials=["t3"])
    p = repo.fix_fingerprints(NS, run_dirs={"run-a@1": str(tmp_path / "run-a"),
                                            "run-b@1": ""})
    assert p["applied"] is False
    kinds = [(a["action"], a.get("to_atid") or a.get("canonical_atid")) for a in p["actions"]]
    # run-a 的漂移指纹:同目标同量级、只差列哈希 → 登记别名,不重打 trial
    assert ("alias", right) in kinds
    out = repo.fix_fingerprints(NS, apply=True,
                                run_dirs={"run-a@1": str(tmp_path / "run-a"),
                                          "run-b@1": str(tmp_path / "run-b")})
    assert out["applied"] and out["aliased"] == 1
    assert repo.resolve_atid(drifted) == right
    # 漂移指纹的 trial 不重打(t1/t2 仍写 drifted),但解析后落到规范指纹
    got = {t["trial_id"]: repo.resolve_atid(t["atid"]) for t in repo.store.trial_outcomes(NS)}
    assert got["t1"] == got["t2"] == right
    assert len(repo.store.trial_outcomes(NS, atid=right)) == 3
    # run-b 的 trial(原本 atid=None)被重打,旧值留痕
    t3 = [t for t in repo.store.trial_outcomes(NS) if t["trial_id"] == "t3"][0]
    assert t3["atid"] == right and t3["extra"]["atid_fix"]["from"] is None
    log = repo.changelog(NS)
    assert any(e["action"] == "retag_atid" for e in log)
    assert any(e["action"] == "retag_atid_row" for e in log)
    # 幂等:再跑没有动作
    assert repo.fix_fingerprints(NS, run_dirs={"run-a@1": str(tmp_path / "run-a"),
                                               "run-b@1": str(tmp_path / "run-b")})["actions"] == []


def test_apply_retags_when_fingerprint_not_centrally_wrong(tmp_path):
    """只有部分 run 与该指纹不一致时,不能整体改判指纹 —— 只重打这些 run 的 trial。"""
    repo = _repo(tmp_path)
    wrong = "binary-y-10w_50w-22222222"
    right = f"binary-y-10w_50w-{_h(COLS)}"
    _run_dir(tmp_path, "run-a", cols=COLS)
    _snapshot(repo, "run-a@1", wrong, trials=["t1"])
    _snapshot(repo, "run-other@1", wrong, trials=["t9"])        # 另一个 run 也用该指纹
    p = repo.fix_fingerprints(NS, run_dirs={"run-a@1": str(tmp_path / "run-a")})
    assert [a["action"] for a in p["actions"]] == ["retag"]
    repo.fix_fingerprints(NS, apply=True, run_dirs={"run-a@1": str(tmp_path / "run-a")})
    got = {t["trial_id"]: t["atid"] for t in repo.store.trial_outcomes(NS)}
    assert got["t1"] == right and got["t9"] == wrong           # 另一个 run 不动
