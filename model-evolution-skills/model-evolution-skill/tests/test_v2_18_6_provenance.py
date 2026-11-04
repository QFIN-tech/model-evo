"""v2.18.6:经验库来源字段 —— 快照 / actor 的模型名校验;依据来源(程序预填 vs Agent)。

背景(2026-10-30 现状报告 §7):正式库 5 个 run 快照里 2 个的 ``run.model`` 是 trial id;
plan 物化命令预填的 ``<id>:tests`` 被原样沿用时,被当成 Agent 的显式引用(权重 1.0)。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import pytest

from runtime.library import evidence as EV
from runtime.library import feedback as FB
from runtime.library.identity import current_actor, resolve_model, valid_model

from test_v2_18_integration import (_load_cli, _make_run, _new_trial_args, _pr, _trial_data,
                                    _write_view)
from test_v2_18_retrieval import RES, _p, _t
from test_v2_18_1_tiered_retrieval import NS, _repo, _run_snap, _trial


# ---------------------------------------------------------------- 模型名

@pytest.mark.parametrize("name", ["claude-opus-5-5", "deepseek-flash", "glm-5.3",
                                  "claude-opus-5-5[1m]", "us.anthropic.claude-sonnet-5-5"])
def test_valid_model_names(name):
    assert valid_model(name)


@pytest.mark.parametrize("name", ["trial-029-g07-confirm-xgboost-828", "run4-v2.18.0@abc",
                                  "prop-1234", "/path/to/model", "two words", "", "x" * 80])
def test_invalid_model_names(name):
    assert not valid_model(name)


def test_resolve_model_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setenv("MODELEVO_MODEL", "claude-opus-5-5")
    assert resolve_model("trial-009-g04-confirm-xgboost-641") == \
        ("claude-opus-5-5", "trial-009-g04-confirm-xgboost-641")
    assert resolve_model("glm-5.3") == ("glm-5.3", None)
    monkeypatch.delenv("MODELEVO_MODEL")
    assert resolve_model("trial-1") == ("", "trial-1")
    assert current_actor(model="trial-1")["model"] == ""


def test_snapshot_rejects_trial_id_as_model(tmp_path, monkeypatch):
    from _mvo.library_snapshot import build_run_snapshot

    monkeypatch.delenv("MODELEVO_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    run = _make_run(tmp_path)
    payload, report = build_run_snapshot(run, None, model="trial-029-g07-confirm-xgboost-828",
                                         upload=False)
    assert payload["run"]["model"] == ""
    assert any("不是模型名" in str(w) for w in report["warnings"])
    payload, report = build_run_snapshot(run, None, model="deepseek-flash", upload=False)
    assert payload["run"]["model"] == "deepseek-flash"
    assert not any("不是模型名" in str(w) for w in report["warnings"])


# ---------------------------------------------------------------- 依据来源

def test_new_trial_marks_prefilled_basis(tmp_path):
    run = _make_run(tmp_path)
    _write_view(run, [_pr("p-gap", 0, ["feature.seq.gap"]), _pr("p-win", 1, ["feature.seq"])])
    cli = _load_cli()
    # 原样沿用预填 → prefilled
    assert cli.cmd_new_trial(_new_trial_args(run, basis="p-gap:tests",
                                             basis_prefill="p-gap:tests")) == 0
    # 改了关系 / 换了原则 / 没有预填 → agent
    assert cli.cmd_new_trial(_new_trial_args(run, basis="p-gap:follows",
                                             basis_prefill="p-gap:tests", generation=2)) == 0
    assert cli.cmd_new_trial(_new_trial_args(run, basis="p-win:extends,p-gap:tests",
                                             basis_prefill="p-gap:tests", generation=3)) == 0
    assert cli.cmd_new_trial(_new_trial_args(run, basis="p-gap:tests", generation=4)) == 0
    # novel 不记来源
    assert cli.cmd_new_trial(_new_trial_args(run, basis="novel",
                                             basis_prefill="p-gap:tests", generation=5)) == 0
    got = {t["generation"]: t["basis"].get("source") for _d, t in _trial_data(run)}
    assert got == {1: "prefilled", 2: "agent", 3: "agent", 4: "agent", 5: None}


def test_prefilled_basis_links_with_passive_weight():
    ps = [_p("conf", -1, ["eval.confirm"], category="evaluation")]
    refs = [{"id": "conf", "relation": "tests"}]
    agent = _t("a", lo=0.001, hi=0.002, extra={"basis": {"novel": False, "refs": refs,
                                                           "source": "agent"}})
    pre = _t("b", lo=0.001, hi=0.002, extra={"basis": {"novel": False, "refs": refs,
                                                         "source": "prefilled"}})
    (la,), (lb,) = (EV.build_links([x], ps, RES, lambda a: a) for x in (agent, pre))
    assert (la["binding"], lb["binding"]) == ("explicit", "prefilled")
    assert lb["weight"] == pytest.approx(la["weight"] * EV.BINDING_W["passive"])
    st = EV.aggregate([la, lb])["conf"]
    assert st["n_explicit"] == 1 and st["n_links"] == 2


def test_feedback_separates_prefilled_citations(tmp_path):
    repo = _repo(tmp_path)
    logs = [{"kind": "search", "watermark": 3, "topics": ["feature.seq.gap"],
             "shown_ids": ["p1", "p2"]}]
    t_pre = _trial("t2", [("p1", "tests")])
    t_pre["basis"]["source"] = "prefilled"
    trials = [_trial("t1", [("p1", "follows")]), t_pre, _trial("t3", novel=True)]
    assert [r["binding"] for r in FB.application_rows(trials)
            if r["trial_id"] == "t2"] == ["basis_prefilled"]
    repo.submit_snapshot(_run_snap("r@1", logs=logs, trials=trials), namespace=NS)
    st = repo.feedback_stats(NS)
    assert st["n_applications"] == 1 and st["n_applications_prefilled"] == 1
    assert st["cite_rate"] == 0.5
    assert st["trials"]["with_basis"] == 1 and st["trials"]["prefilled_kept"] == 1
    p1 = next(r for r in st["principles"] if r["principle_id"] == "p1")
    assert (p1["cited"], p1["prefilled_kept"]) == (1, 1)
