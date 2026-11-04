"""v2.18.2 第 2 期:与 run 解耦的异步整理(consolidation.py)。

docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md §23。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import pytest

from runtime.library import ArtifactStore, LibraryStore, LibraryValidationError, LocalRepository
from runtime.library import browse
from runtime.library import consolidation as CO
from runtime.library import curation as CU
from runtime.library import store as ST

NS = "t"
A = "binary-y-200w_1000w-bc5329dc"
B = "binary-z-200w_1000w-bc5329dc"          # 同底座另一个目标(第二个任务)
BIN = {"model_type": "binary_classification", "primary_metric": "auc", "target_transform": "raw"}
G = "feature.seq.gap"


def _repo(tmp_path) -> LocalRepository:
    return LocalRepository(LibraryStore(f"sqlite:///{tmp_path}/lib.db"),
                           ArtifactStore(f"file://{tmp_path}/cas"))


def _merge(repo, run_id, atid, principles):
    sid = repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_principles",
                                "run": {"run_id": run_id, "atid": atid},
                                "discoveries": [{"local_id": p["principle_id"],
                                                 "payload": {"claim": p["claim"]}}
                                                for p in principles]},
                               namespace=NS)["snapshot_id"]
    ctx = repo.merge_context(NS, sid)
    return repo.merge_apply(NS, {
        "schema": "modelevo.merge-decision/1", "snapshot_id": sid,
        "lease_token": ctx["lease"]["token"],
        "actions": [{"local_id": p["principle_id"], "action": "add", "reason": "seed",
                     "principle": p} for p in principles]})


def _pr(pid, direction, claim, topics=(G,), scope=BIN):
    return {"principle_id": pid, "claim": claim, "direction": direction,
            "topics": list(topics), "scope": scope}


def _refuting_runs(repo, n=3):
    """n 个 run 的 trial 在 G 主题上增益显著为负:对 +1 原则是反驳。"""
    for i in range(n):
        run = f"neg@{i}"
        repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "run",
                              "run": {"run_id": run, "atid": A},
                              "trial_outcomes": [{
                                  "run_id": run, "trial_id": "t1", "atid": A,
                                  "topics": [G], "primary_metric": "auc",
                                  "higher_is_better": True, "verdict": "rejected",
                                  "hypothesis": "继续加宽间隔族",
                                  "delta_parent_ci_low": -0.003,
                                  "delta_parent_ci_high": -0.001}]}, namespace=NS)


def _seed(repo):
    repo.init_topics(NS, [{"code": "feature.seq", "definition": "d", "aliases": []},
                          {"code": G, "definition": "d", "aliases": []},
                          {"code": "hpo.search", "definition": "d", "aliases": []}],
                     reason="v1")
    _merge(repo, "legacy:a", A, [
        _pr("p-wide", 1, "间隔族 seq_gap 加宽到 d365 窗口带来稳定的 AUC 增益"),
        _pr("p-wide-dup", 1, "间隔族 seq_gap 加宽到 d365 窗口带来稳定 AUC 增益,两次复现"),
        _pr("p-old", 0, "两段式训练下间隔族无增益"),
        _pr("p-hpo", 1, "HPO 首轮有效", topics=("hpo.search",))])
    _merge(repo, "legacy:b", B, [_pr("p-wide-b", 1, "另一个目标上间隔族加宽同样有效")])


def _open(repo):
    plan = repo.curation_plan(NS)
    grp = next(g for g in plan["groups"] if g["group"] == G)
    rec = repo.curation_open(NS, G)
    return grp, rec, repo.curation_context(NS, rec["snapshot_id"])


def _decision(ctx, actions, generalized=()):
    return {"schema": CO.DECISION_SCHEMA, "snapshot_id": ctx["snapshot_id"],
            "lease_token": ctx["lease"]["token"], "actions": actions,
            "generalized": list(generalized)}


def _full_actions(ctx, **override):
    acts = {pid: {"principle_id": pid, "action": "keep", "reason": "ok"}
            for pid in ctx["principle_ids"]}
    acts.update(override)
    return list(acts.values())


# ---------------------------------------------------------------- 计划与打开

def test_plan_triggers_and_open_is_idempotent(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    grp, rec, ctx = _open(repo)
    kinds = {t["kind"] for t in grp["triggers"]}
    assert {"generalizable", "near_duplicate"} <= kinds
    assert grp["last_curated_seq"] is None and grp["due"]
    # hpo.search 组只有一条原则,没有触发条件
    assert all(g["group"] != "hpo.search" for g in repo.curation_plan(NS)["groups"])
    assert rec["state"] == "submitted" and set(ctx["principle_ids"]) == {
        "p-wide", "p-wide-dup", "p-old", "p-wide-b"}
    assert ctx["lease"]["token"] and tuple(ctx["rules"]["actions"]) == CO.ACTIONS
    # 内容没变:同一快照(判重),不会产生第二个整理快照
    assert repo.curation_open(NS, G)["duplicate"] is True
    # 整理快照不进 run 合并队列
    assert all(s["kind"] != "curation" for s in repo.merge_pending(NS))
    # 租约被占:run 合并与另一次整理都要等
    with pytest.raises(LibraryValidationError):
        repo.curation_context(NS, rec["snapshot_id"])


def test_context_has_decisive_evidence(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    _refuting_runs(repo)
    _grp, _rec, ctx = _open(repo)
    wide = next(p for p in ctx["principles"] if p["principle"]["principle_id"] == "p-wide")
    assert wide["n_decisive_evidence"] == 3
    assert wide["decisive_evidence"][0]["relation"] == "refute"
    assert wide["decisive_evidence"][0]["hypothesis"] == "继续加宽间隔族"
    assert wide["stats"]["confidence"] in ("mixed", "low")


# ---------------------------------------------------------------- 落库

def test_apply_full_decision(tmp_path, monkeypatch):
    # 每次提交时间不同,才能区分首次提交 / 最近更新
    tick = iter(range(10**6))
    monkeypatch.setattr(ST, "now_iso", lambda: f"2026-10-30T10:{next(tick):02d}:00+08:00")
    repo = _repo(tmp_path)
    _seed(repo)
    _grp, rec, ctx = _open(repo)
    merged_claim = "间隔族 seq_gap 加宽到 d365 窗口带来稳定的 AUC 增益(两次复现)"
    dec = _decision(ctx, _full_actions(
        ctx,
        **{"p-wide": {"principle_id": "p-wide", "action": "revise", "reason": "吸收重复条目",
                      "principle": {**_pr("p-wide", 1, merged_claim),
                                    "evidence": {"abstract_task_ids": []}}},
           "p-wide-dup": {"principle_id": "p-wide-dup", "action": "merge_into",
                          "target": "p-wide", "reason": "同一结论"},
           "p-old": {"principle_id": "p-old", "action": "retire", "retire_kind": "obsolete",
                     "reason": "两段式训练已删除"}}),
        generalized=[{"derived_from": ["p-wide", "p-wide-b"], "reason": "两个目标上都成立",
                      "principle": _pr(None, 1, "同底座不同目标上,间隔族加宽都有增益",
                                       scope={**BIN, "primary_metric": "*"})}])
    dry = repo.curation_apply(NS, {**dec, "dry_run": True})
    assert dry["dry_run"] and dry["counts"]["generalize"] == 1
    out = repo.curation_apply(NS, dec)
    assert out["counts"] == {"principle_versions": 2, "governance": 2, "keep": 1, "revise": 1,
                             "merge_into": 1, "retire": 1, "generalize": 1}
    cur = {p["principle_id"]: p for p in repo.principles(NS, include_inactive=True)}
    assert cur["p-wide"]["version"] == 2 and cur["p-wide"]["claim"] == merged_claim
    when = {c["seq"]: c["created_at"] for c in repo.store.commits(limit=1000)}
    assert cur["p-wide"]["updated_at"] == when[cur["p-wide"]["commit_seq"]]
    assert cur["p-wide"]["first_committed_at"] < cur["p-wide"]["updated_at"]
    assert cur["p-old"]["first_committed_at"] == cur["p-old"]["updated_at"]
    browse.render(repo, NS, tmp_path / "browse")
    idx = (tmp_path / "browse" / "principles" / "INDEX.md").read_text(encoding="utf-8")
    assert "| 首次提交 | 最近更新 |" in idx
    assert cur["p-wide"]["updated_at"][:16].replace("T", " ") in idx
    # 任务指纹由服务端保留(LLM 给了空列表也不会丢)
    assert cur["p-wide"]["evidence"]["abstract_task_ids"] == [A]
    assert cur["p-wide-dup"]["status"] == "superseded" and \
        cur["p-wide-dup"]["superseded_by"] == ["p-wide"]
    assert cur["p-old"]["status"] == "retired"
    gen = [p for p in cur.values() if (p.get("evidence") or {}).get("derived_from")]
    assert len(gen) == 1 and gen[0]["evidence"]["derived_from"] == ["p-wide", "p-wide-b"]
    assert sorted(gen[0]["evidence"]["abstract_task_ids"]) == sorted([A, B])
    assert repo.store.get_snapshot(NS, rec["snapshot_id"])["state"] == "merged"
    log = repo.changelog(NS)
    assert {"keep", "revise+absorb", "merge_into", "retire", "generalize"} <= \
        {r["action"] for r in log}
    # 刚整理过、没有新变化:不再列出
    assert all(g["group"] != G for g in repo.curation_plan(NS)["groups"])
    # 租约已释放:run 合并可以继续
    _merge(repo, "legacy:c", A, [_pr("p-new", -1, "间隔族与漏斗同时加入时有害")])
    plan = repo.curation_plan(NS)
    assert any(g["group"] == G and g["changes"]["new_versions"] >= 1 for g in plan["groups"])


def test_revert_curation(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    _grp, rec, ctx = _open(repo)
    repo.curation_apply(NS, _decision(ctx, _full_actions(ctx, **{
        "p-old": {"principle_id": "p-old", "action": "retire", "retire_kind": "obsolete",
                  "reason": "过时"}})))
    assert "p-old" not in {p["principle_id"] for p in repo.principles(NS)}
    repo.revert_merge(NS, rec["snapshot_id"], reason="误判")
    assert "p-old" in {p["principle_id"] for p in repo.principles(NS)}
    # 撤销后该组重新进入计划,且同一内容的快照可以再整理
    assert any(g["group"] == G for g in repo.curation_plan(NS)["groups"])
    assert repo.curation_open(NS, G)["state"] == "reverted"


# ---------------------------------------------------------------- 校验

@pytest.mark.parametrize("override,generalized,needle", [
    ({"p-old": None}, (), "没有对应动作"),
    ({"p-wide-dup": {"action": "merge_into", "target": "p-wide", "reason": "r"}}, (),
     "必须 revise"),
    ({"p-old": {"action": "merge_into", "target": "p-wide", "reason": "r"},
      "p-wide": {"action": "revise", "reason": "r",
                 "principle": _pr("p-wide", 1, "x")}}, (), "同口径、同方向"),
    ({"p-wide": {"action": "retire", "retire_kind": "refuted", "reason": "r"}}, (),
     "证据不足"),
    ({"p-wide": {"action": "revise", "reason": "r",
                 "principle": _pr("p-wide", 1, "x", scope={**BIN, "primary_metric": "ks"})}},
     (), "不得改口径"),
    ({"p-old": {"action": "revise", "reason": "r", "principle": _pr("p-old", 1, "其实有效")}},
     (), "推翻"),
    ({"p-wide": {"action": "keep", "reason": "r", "contest_with": ["p-old"]},
      "p-old": {"action": "retire", "retire_kind": "obsolete", "reason": "r"}}, (), "下线"),
    ({}, [{"derived_from": ["p-wide", "p-wide-dup"], "reason": "r",
           "principle": _pr(None, 1, "x")}], "只来自 1 个任务"),
    ({}, [{"derived_from": ["p-wide", "p-wide-b"], "reason": "r",
           "principle": _pr(None, 1, "x", scope={**BIN, "primary_metric": "ks"})}],
     "不覆盖"),
    ({}, [{"derived_from": ["p-wide", "p-old"], "reason": "r",
           "principle": _pr(None, 1, "x")}], "方向"),
])
def test_validation_rules(tmp_path, override, generalized, needle):
    repo = _repo(tmp_path)
    _seed(repo)
    _grp, _rec, ctx = _open(repo)
    acts = {a["principle_id"]: a for a in _full_actions(ctx)}
    for pid, a in override.items():
        if a is None:
            acts.pop(pid)
        else:
            acts[pid] = {"principle_id": pid, **a}
    with pytest.raises(LibraryValidationError) as ei:
        repo.curation_apply(NS, _decision(ctx, list(acts.values()), generalized))
    assert any(needle in e for e in ei.value.errors), ei.value.errors


def test_retire_refuted_allowed_with_evidence(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    _refuting_runs(repo)
    _grp, _rec, ctx = _open(repo)
    out = repo.curation_apply(NS, _decision(ctx, _full_actions(ctx, **{
        "p-wide-b": {"principle_id": "p-wide-b", "action": "keep", "reason": "另一任务无反证"},
        "p-wide": {"principle_id": "p-wide", "action": "retire", "retire_kind": "refuted",
                   "reason": "三个 run 显著为负"},
        "p-wide-dup": {"principle_id": "p-wide-dup", "action": "retire",
                       "retire_kind": "refuted", "reason": "同上"}})))
    assert out["counts"]["retire"] == 2


def test_stale_principle_rejects_decision(tmp_path):
    repo = _repo(tmp_path)
    _seed(repo)
    _grp, rec, ctx = _open(repo)
    dec = _decision(ctx, _full_actions(ctx))
    # 打开整理后别处修改了组内原则(这里模拟:放弃整理、合并一个新版本,再提交旧决议)
    repo.merge_abort(NS, rec["snapshot_id"], lease_token=ctx["lease"]["token"], reason="x")
    sid = repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "legacy_principles",
                                "run": {"run_id": "legacy:z", "atid": A},
                                "discoveries": [{"local_id": "d", "payload": {"claim": "c"}}]},
                               namespace=NS)["snapshot_id"]
    mctx = repo.merge_context(NS, sid)
    repo.merge_apply(NS, {"schema": "modelevo.merge-decision/1", "snapshot_id": sid,
                          "lease_token": mctx["lease"]["token"],
                          "actions": [{"local_id": "d", "action": "merge_into",
                                       "target": "p-old", "reason": "r",
                                       "principle": _pr("p-old", 0, "两段式下无增益(补充)")}]})
    ctx2 = repo.curation_context(NS, rec["snapshot_id"])
    with pytest.raises(LibraryValidationError) as ei:
        repo.curation_apply(NS, {**dec, "lease_token": ctx2["lease"]["token"]})
    assert any("已被修改" in e for e in ei.value.errors)


# ---------------------------------------------------------------- HTTP 与 worker

def test_curation_over_http(tmp_path):
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

    http = HttpRepository("http://testserver", transport=transport,
                          actor={"user": "u", "host": "h", "model": "m-x"})
    assert any(g["group"] == G for g in http.curation_plan(NS)["groups"])
    rec = http.curation_open(NS, G)
    ctx = http.curation_context(NS, rec["snapshot_id"])
    out = http.curation_apply(NS, _decision(ctx, _full_actions(ctx)))
    assert out["counts"]["keep"] == 4
    commit = [c for c in repo.commits(0, 1000) if c["kind"] == "curate"][-1]
    assert commit["actor"]["model"] == "m-x"


def test_worker_aborts_when_model_does_not_finish(tmp_path, monkeypatch):
    sys.path.insert(0, str(SKILL / "scripts"))
    import curate_worker as W

    calls = []

    def fake_library(*args, model):
        calls.append(args[0])
        if args[0] == "curate-next":
            return {"status": "opened", "group": G, "snapshot_id": "s1", "lease_token": "tk",
                    "principles": 2, "triggers": ["generalizable"]}
        return {"status": "merge_failed"}

    class P:
        stdout = "no json"

    monkeypatch.setattr(W, "_library", fake_library)
    monkeypatch.setattr(W, "_snapshot_state", lambda sid: "submitted")
    monkeypatch.setattr(W.subprocess, "run", lambda *a, **k: P())
    assert W.main(["--model", "m-x", "--max-groups", "1", "--work-dir", str(tmp_path)]) == 0
    assert calls == ["curate-next", "merge-abort"]
    assert '"state": "merge_failed"' in (tmp_path / "worker-log.jsonl").read_text()
    with pytest.raises(SystemExit):
        W.main(["--max-groups", "1"])                       # --model 必填


def test_scheduler_next_run_and_lifecycle(tmp_path):
    import datetime as dt
    import json as _json
    import subprocess

    sys.path.insert(0, str(SKILL / "scripts"))
    import curate_scheduler as SC

    sat = SC.WEEKDAYS.index("sat")
    thu = dt.datetime(2026, 10, 29, 13, 0)                     # 周四
    assert SC.next_run(thu, sat, 2, 0) == dt.datetime(2026, 10, 31, 2, 0)
    # 正好在点上或已过:顺延一周
    assert SC.next_run(dt.datetime(2026, 10, 31, 2, 0), sat, 2, 0) == \
        dt.datetime(2026, 11, 7, 2, 0)
    assert SC.next_run(dt.datetime(2026, 10, 31, 1, 59), sat, 2, 0) == \
        dt.datetime(2026, 10, 31, 2, 0)

    state = tmp_path / "cur"
    script = str(SKILL / "scripts" / "curate_scheduler.py")
    run = lambda *a: _json.loads(subprocess.run(  # noqa: E731
        [sys.executable, script, *a, "--state-dir", str(state)] if a[0] in ("status", "stop")
        else [sys.executable, script, *a, "--state-dir", str(state)],
        capture_output=True, text=True, check=True).stdout)
    up = run("start", "--model", "m-x", "--weekday", "sat", "--at", "02:00")
    try:
        assert up["started"] and up["next_run"].endswith("02:00")
        import time
        for _ in range(50):
            if (state / "scheduler.json").exists():
                break
            time.sleep(0.1)
        st = run("status")
        assert st["running"] and st["model"] == "m-x" and st["weekday_name"] == "sat"
        # 重复 start 拒绝
        again = subprocess.run([sys.executable, script, "start", "--model", "m-x",
                                "--state-dir", str(state)], capture_output=True, text=True)
        assert again.returncode == 1 and "already running" in again.stdout
    finally:
        assert run("stop")["stopped"]
    for _ in range(50):
        if not run("status")["running"]:
            break
        time.sleep(0.1)
    assert not run("status")["running"]
    assert "scheduler down" in (state / "scheduler.log").read_text()


def test_relink_after_curation_does_not_make_group_due_again(tmp_path):
    """10-29 首轮:整理提了上位原则 → relink 给它补了历史证据 → 组立刻再次到期。"""
    repo = _repo(tmp_path)
    _seed(repo)
    _refuting_runs(repo)
    _grp, _rec, ctx = _open(repo)
    out = repo.curation_apply(NS, _decision(ctx, _full_actions(ctx), generalized=[
        {"derived_from": ["p-wide", "p-wide-b"], "reason": "两个目标上都成立",
         "principle": _pr(None, 1, "同底座不同目标上间隔族加宽都有增益",
                          scope={**BIN, "primary_metric": "*"})}]))
    assert out["evidence"]["links"] >= 3              # relink 确实给上位原则补了证据
    assert all(g["group"] != G for g in repo.curation_plan(NS)["groups"])
    _refuting_runs(repo, n=2)                          # 新 run 带来的新证据才算变化
    for i in range(2):
        repo.submit_snapshot({"schema": "modelevo.snapshot/1", "kind": "run",
                              "run": {"run_id": f"neg2@{i}", "atid": A},
                              "trial_outcomes": [{
                                  "run_id": f"neg2@{i}", "trial_id": "t1", "atid": A,
                                  "topics": [G], "primary_metric": "auc",
                                  "higher_is_better": True, "verdict": "rejected",
                                  "delta_parent_ci_low": -0.003,
                                  "delta_parent_ci_high": -0.001}]}, namespace=NS)
    assert any(g["group"] == G for g in repo.curation_plan(NS)["groups"])


def test_worker_releases_lease_for_reopened_failed_group(tmp_path, monkeypatch):
    """重开的失败组快照状态本来就是 merge_failed,模型没完成时也必须 merge-abort。"""
    sys.path.insert(0, str(SKILL / "scripts"))
    import curate_worker as W

    calls = []

    def fake_library(*args, model):
        calls.append(args[0])
        return {"status": "opened", "group": G, "snapshot_id": "s1", "lease_token": "tk",
                "principles": 2, "triggers": []} if args[0] == "curate-next" else {}

    class P:
        stdout = "x"

    monkeypatch.setattr(W, "_library", fake_library)
    monkeypatch.setattr(W, "_snapshot_state", lambda sid: "merge_failed")
    monkeypatch.setattr(W.subprocess, "run", lambda *a, **k: P())
    W.main(["--model", "m", "--max-groups", "1", "--work-dir", str(tmp_path)])
    assert calls == ["curate-next", "merge-abort"]


def test_worker_allows_relative_and_absolute_command_forms():
    sys.path.insert(0, str(SKILL / "scripts"))
    import curate_worker as W

    rules = W._allowed_commands()
    for form in (f"Bash({sys.executable} scripts/modelevo.py library curate-apply:*)",
                 f"Bash({sys.executable} {W.MODELEVO[1]} library curate-apply:*)",
                 "Bash(python scripts/modelevo.py library merge-abort:*)",
                 "Bash(python3 scripts/modelevo.py library merge-renew:*)"):
        assert form in rules
    assert not any("curate-open" in r or "revert" in r for r in rules)   # 只放行这三个子命令
