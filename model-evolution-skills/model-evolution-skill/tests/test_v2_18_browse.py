"""v2.18.0 经验库明文镜像:渲染内容、原子发布、写入后防抖触发、导出命令。

镜像只读、可随时重建,权威数据仍在数据库 + 内容寻址存储(开发文档 §20)。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import argparse
import json
import os
import stat

import pytest
import yaml

from runtime.library import ArtifactStore, LibraryStore, LocalRepository
from runtime.library import browse
from runtime.library.recipes import RUN_DIR_TOKEN

ATID = "binary-y-200w_1000w-bc5329dc"
NS = "test"


def _repo(tmp_path) -> LocalRepository:
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _seed(repo, *, run="run-a@00000000", trial="trial-003", depth=6, gain=0.01, seq_kind="run"):
    code = (f"SEED = 1\nRUN_DIR = pathlib.Path('{RUN_DIR_TOKEN}')\n"
            "def fit_and_score(): pass\n").encode()
    cfg = yaml.safe_dump({"algorithm": "xgboost", "hyperparameters": {"max_depth": depth}}).encode()
    rid = repo.put_bundle({"train.py": code, "trial.yaml": cfg, "hypothesis.md": b"seq widen\n",
                           "code/deep_seq.py": b"def f(): pass\n"})
    repo.submit_snapshot({
        "schema": "modelevo.snapshot/1", "kind": seq_kind, "run": {"run_id": run, "atid": ATID},
        "recipes": [{"recipe_id": rid, "kind": "top_trial", "algorithm": "xgboost",
                     "source_run_id": run, "source_trial_id": trial,
                     "runtime_api_version": "2.16"}],
        "measurements": [{"recipe_id": rid, "kind": "export", "run_id": run, "trial_id": trial,
                          "value": 0.81, "gain": gain}],
        "trial_outcomes": [{"run_id": run, "trial_id": trial, "atid": ATID, "generation": 1,
                            "action": "explore", "algorithm": "xgboost", "verdict": "accepted",
                            "primary_metric": "auc", "higher_is_better": True,
                            "metric_value": 0.81, "delta_parent": gain,
                            "delta_parent_ci_low": gain / 2, "delta_parent_ci_high": gain * 2,
                            "hypothesis": "seq 加宽", "basis": {"novel": True, "refs": []}}],
    }, namespace=NS)
    return rid


# ---------------------------------------------------------------- 渲染与发布

def test_render_contents(tmp_path):
    repo = _repo(tmp_path)
    rid = _seed(repo)
    info = browse.render(repo, NS, tmp_path / "out")
    out = tmp_path / "out"
    assert info["tasks"] == 1 and info["recipes"] == 1 and info["watermark"] == repo.watermark()
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert f"**水位** {repo.watermark()}" in readme and ATID in readme and "只读" in readme
    tdir = out / "tasks" / ATID
    lb = (tdir / "leaderboard.md").read_text(encoding="utf-8")
    assert rid[:12] in lb and "trial-003" in lb
    rdir = next((tdir / "recipes").iterdir())
    assert rdir.name.startswith("01-trial-003-")
    assert (rdir / "train.py").read_text(encoding="utf-8").startswith("SEED = 1")
    assert (rdir / "code" / "deep_seq.py").exists() and (rdir / "hypothesis.md").exists()
    trials = (tdir / "trials.md").read_text(encoding="utf-8")
    assert "trial-003" in trials and "novel" in trials and "有效方向" in trials
    for f in ("CHANGELOG.md", "topics.md", "principles/INDEX.md", f"tasks/{ATID}/dossier.md"):
        assert (out / f).exists(), f


def test_publish_atomic_readonly_and_reuse(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    root = tmp_path / "browse"
    a = browse.publish(repo, NS, root)
    link = root / NS
    assert link.is_symlink() and (link / "README.md").exists()
    assert a["recipes_reused"] == 0
    readme_mode = (link / "README.md").stat().st_mode
    assert not readme_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)    # 只读
    first_target = link.resolve()
    _seed(repo, run="run-b@11111111", trial="trial-009", depth=8, gain=0.02)
    b = browse.publish(repo, NS, root)
    assert link.resolve() != first_target and b["recipes"] == 2 and b["recipes_reused"] == 1
    lb = (link / "tasks" / ATID / "leaderboard.md").read_text(encoding="utf-8")
    assert lb.index("trial-009") < lb.index("trial-003")                      # 增益高的在前
    for _ in range(3):
        browse.publish(repo, NS, root)
    assert len(list((root / ".gen").glob(f"{NS}-*"))) == 2                    # 只保留最近两版


def test_publish_refuses_foreign_directory(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    root = tmp_path / "browse"
    (root / NS).mkdir(parents=True)
    (root / NS / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(RuntimeError):
        browse.publish(repo, NS, root)
    assert (root / NS / "notes.txt").exists()


def test_export_dir_replaces_mirror_only(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    out = tmp_path / "mirror"
    browse.export_dir(repo, NS, out)
    assert (out / browse.MARKER).exists()
    browse.export_dir(repo, NS, out)                                          # 再导一次:整体替换
    other = tmp_path / "other"
    other.mkdir()
    (other / "x").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError):
        browse.export_dir(repo, NS, other)


# ---------------------------------------------------------------- 写入后触发

def test_publisher_publishes_only_changed_namespaces(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    pub = browse.BrowsePublisher(repo, tmp_path / "browse")
    assert list(pub.changed_namespaces()) == [NS]
    assert [i["namespace"] for i in pub.run_once()] == [NS]
    assert pub.changed_namespaces() == {} and pub.run_once() == []
    _seed(repo, run="run-b@11111111", trial="trial-009", depth=8)
    assert list(pub.changed_namespaces()) == [NS]
    pub.run_once()
    os.unlink(tmp_path / "browse" / NS)                                       # 镜像被删:补生成
    assert list(pub.changed_namespaces()) == [NS]
    pub.run_once()
    assert (tmp_path / "browse" / NS / "README.md").exists()


def test_service_write_triggers_publisher(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from service.app import create_app

    repo = _repo(tmp_path)
    calls = []
    client = TestClient(create_app(repo, on_write=lambda: calls.append(1)))
    assert client.get("/v1/health").status_code == 200 and calls == []
    r = client.post("/v1/snapshots", json={"namespace": NS, "payload": {"schema": "bad"}})
    assert r.status_code == 422 and calls == []                               # 失败的写不触发
    r = client.post("/v1/snapshots", json={"namespace": NS, "payload": {
        "schema": "modelevo.snapshot/1", "kind": "run",
        "run": {"run_id": "r@1", "atid": ATID},
        "trial_outcomes": [{"run_id": "r@1", "trial_id": "t"}]}})
    assert r.status_code == 200 and calls == [1]


# ---------------------------------------------------------------- 导出命令

def test_recipe_export_command(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(SKILL / "scripts"))
    from _mvo import cmd_library

    repo = _repo(tmp_path)
    rid = _seed(repo)
    monkeypatch.setattr(cmd_library, "_client", lambda s, **kw: repo)
    args = argparse.Namespace(uri=None, namespace=NS, root=None, recipe_id=rid[:12],
                              out=str(tmp_path / "plain"), run=None)
    assert cmd_library.cmd_library_recipe_export(args) == 0
    assert RUN_DIR_TOKEN in (tmp_path / "plain" / "train.py").read_text(encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    args.out, args.run = str(tmp_path / "runnable"), str(run)
    assert cmd_library.cmd_library_recipe_export(args) == 0
    code = (tmp_path / "runnable" / "train.py").read_text(encoding="utf-8")
    assert f"RUN_DIR = pathlib.Path('{run.resolve()}')" in code
    assert cmd_library.cmd_library_recipe_export(args) == 1                   # 目标非空不覆盖
    args.recipe_id, args.out = "ffff", str(tmp_path / "x")
    assert cmd_library.cmd_library_recipe_export(args) == 1                   # 前缀不存在
    capsys.readouterr()
    lb_args = argparse.Namespace(uri=None, namespace=NS, root=None, atid=ATID, json=True)
    assert cmd_library.cmd_library_leaderboard(lb_args) == 0
    assert json.loads(capsys.readouterr().out)[0]["recipe_id"] == rid


def test_changelog_paged_without_splitting_commits(monkeypatch):
    rows = [{"commit_seq": s, "i": i} for s in range(1, 8) for i in range(3)]   # 7 个提交 × 3 条

    class Repo:
        def changelog(self, namespace, *, since=0, limit=1000):
            return [r for r in rows if r["commit_seq"] > since][:limit]

    monkeypatch.setattr(browse, "_CHANGELOG_PAGE", 4)
    assert browse._all_changelog(Repo(), NS) == rows
