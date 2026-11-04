"""回归测试:v2.13.2 screening-only trial(宽面初筛)不得参与任何裁决。

背景:`feature_exploration.super_trial` 自 v10 起就是配置键,但 runtime
**零引用**(纯摆设)。语义是审计文档三级漏斗的第一级 —— 一次性加宽全部
候选特征族,产出 `group_importance` 的族级置换 delta 作**方向先验**,
回答"gen-02 该拆哪几族、哪些族可以判负"。

宽面混合了全部族,其 `vs_parent` 不可归因到任何单一方向。因此它必须
"只产证据、不参与裁决"。本文件逐处锁定这个语义 —— 漏一处就等于没修,
而且**漏了是静默的**(宽 trial 混进冠军或污染方向台账不会有任何报错)。

标记形态:`trial.yaml.screen_only = true`(经 `new-trial --screen-only`),
判据函数 `runtime.trial.trial_is_screening`。

**不新增 search_action 取值**的理由:SEARCH_ACTIONS 被方向台账、
leaderboard 模型族覆盖、report 分方向聚合、Archive novelty 分档、
templates confirm 分支等 7 处按动作分组消费,新动作值会污染全部分组。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import yaml

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _trial(tid, *, gen=1, action="explore", verdict="accepted",
           value=0.80, screen=False, parents=("t0",), modules=("feature",),
           view="v1", se=0.0033, p=0.99, run_dir=None):
    from runtime.trial import TrialRecord
    d = {
        "trial_id": tid, "algorithm": "lightgbm",
        "parent_trial_ids": list(parents), "generation": gen,
        "search_action": action, "changed_modules": list(modules),
        "data_view": view, "status": "EVALUATED",
        "evaluation": {
            "primary_metric": {"name": "auc", "value": value},
            "verdict": verdict,
            "delta": {"vs_parent": value - 0.78,
                      "vs_parent_paired_bootstrap": {
                          "se": se, "p_positive": p, "noise": se}},
            "stability": {"passed": True},
            "integrity": {"valid": True, "errors": []},
        },
        "result": {"runtime_sec": 100},
    }
    if screen:
        d["screen_only"] = True
    tdir = None
    if run_dir is not None:
        tdir = Path(run_dir) / "trials" / tid
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "evaluation.json").write_text(
            json.dumps(d["evaluation"]), encoding="utf-8")
        # finalize_generation 内部走 trial_factory.all_trials() 重新加载,
        # 因此 trial.yaml 必须真落盘 —— 否则测的是"传进去的列表",而
        # 被测代码读的是磁盘,断言会假通过。
        (tdir / "trial.yaml").write_text(
            yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return TrialRecord(data=d, trial_dir=tdir)


def _baseline(run_dir=None):
    from runtime.trial import TrialRecord
    d = {
        "trial_id": "t0", "algorithm": "lightgbm",
        "parent_trial_ids": [], "generation": 0, "status": "EVALUATED",
        "search_action": "explore", "changed_modules": [],
        "data_view": "v1",
        "evaluation": {"primary_metric": {"name": "auc", "value": 0.78},
                       "verdict": "accepted",
                       "stability": {"passed": True},
                       "integrity": {"valid": True, "errors": []}},
        "result": {"runtime_sec": 100}}
    tdir = None
    if run_dir is not None:
        tdir = Path(run_dir) / "trials" / "t0"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "evaluation.json").write_text(
            json.dumps(d["evaluation"]), encoding="utf-8")
        (tdir / "trial.yaml").write_text(
            yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return TrialRecord(data=d, trial_dir=tdir)


def _run_dir(td: Path):
    from runtime.config import Config
    from runtime.spec import TaskSpec
    (td / "trials").mkdir(parents=True, exist_ok=True)
    (td / "leaderboard").mkdir(exist_ok=True)
    (td / "run-state.json").write_text(
        json.dumps({"baseline_id": "t0", "harness_epoch": 0}),
        encoding="utf-8")
    spec = TaskSpec(data={
        "task_id": "v213screen", "task_name": "v213screen",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
    })
    return td, spec, Config.load(skill_config=False)


# ---------------------------------------------------------------------------
# 1. 判据函数
# ---------------------------------------------------------------------------

def test_predicate():
    print("== trial_is_screening 判据 ==")
    from runtime.trial import trial_is_screening
    check("screen_only=true → True",
          trial_is_screening(_trial("a", screen=True)) is True)
    check("无标记 → False",
          trial_is_screening(_trial("b")) is False)
    check("screen_only=false → False",
          trial_is_screening(_trial("c", screen=False)) is False)
    from runtime.trial import TrialRecord
    t = TrialRecord(data={"trial_id": "d", "screen_only": 0})
    check("falsy 值 → False", trial_is_screening(t) is False)


# ---------------------------------------------------------------------------
# 2. 五处裁决通道逐处断言
# ---------------------------------------------------------------------------

def test_archive_excludes_screening():
    print("== Archive 不收录 screening ==")
    from runtime.leaderboard import CandidateArchive
    with tempfile.TemporaryDirectory() as tds:
        run, spec, config = _run_dir(Path(tds))
        arch = CandidateArchive(run, spec, config)
        arch.update([_baseline(),
                     _trial("normal", value=0.79),
                     _trial("screen", value=0.99, screen=True)])
        ids = {c["trial_id"] for c in arch.state["candidates"]}
        check("普通 accepted 进 Archive", "normal" in ids, str(ids))
        check("screening 不进 Archive(宽面 0.99 不得占槽)",
              "screen" not in ids, str(ids))


def test_best_trial_excludes_screening():
    print("== best_trial 不选 screening ==")
    from runtime.leaderboard import Leaderboard
    with tempfile.TemporaryDirectory() as tds:
        run, spec, config = _run_dir(Path(tds))
        lb = Leaderboard(run, spec, config)
        trials = [_baseline(), _trial("normal", value=0.79),
                  _trial("screen", value=0.99, screen=True)]
        best = lb.best_trial(trials)
        check("冠军不是分数最高的 screening(0.99)",
              best is not None and best.trial_id == "normal",
              best.trial_id if best else "None")


def test_leaderboard_lists_screening_with_flag():
    print("== leaderboard 表内可见且带 screen_only 列 ==")
    from runtime.leaderboard import LEADERBOARD_COLUMNS, Leaderboard
    with tempfile.TemporaryDirectory() as tds:
        run, spec, config = _run_dir(Path(tds))
        lb = Leaderboard(run, spec, config)
        lb.build([_baseline(), _trial("screen", value=0.99, screen=True)])
        check("LEADERBOARD_COLUMNS 含 screen_only",
              "screen_only" in LEADERBOARD_COLUMNS)
        rows = json.loads((run / "leaderboard" / "leaderboard.json")
                          .read_text(encoding="utf-8"))["rows"]
        by = {r["trial_id"]: r for r in rows}
        check("screening 仍在表里(可审计)", "screen" in by, str(list(by)))
        check("screening 行带 screen_only 标记",
              by.get("screen", {}).get("screen_only") is True,
              str(by.get("screen", {}).get("screen_only")))


def test_direction_ledger_excludes_screening():
    print("== 方向台账不入账 screening ==")
    from runtime.directions import DirectionLedger
    with tempfile.TemporaryDirectory() as tds:
        run, spec, config = _run_dir(Path(tds))
        ledger = DirectionLedger(run)
        ledger.refresh([_trial("normal", value=0.79, modules=("feature",)),
                        _trial("screen", value=0.99, screen=True,
                               modules=("feature",))], spec)
        state = json.loads((run / "directions.json")
                           .read_text(encoding="utf-8"))
        tried = sum(int(a.get("n_tried") or 0)
                    for a in (state.get("arms") or {}).values())
        check("screening 不计入任何 arm 的 n_tried",
              tried == 1, f"n_tried 合计={tried}(期望 1)")


def test_fdr_and_accepted_counts_exclude_screening():
    print("== FDR / accepted 计数排除 screening ==")
    from runtime.evolution import EvolutionEngine
    with tempfile.TemporaryDirectory() as tds:
        run, spec, config = _run_dir(Path(tds))
        engine = EvolutionEngine(run, spec, config)
        trials = [_baseline(run),
                  _trial("normal", value=0.79, gen=1, run_dir=run),
                  _trial("screen", value=0.99, gen=1, screen=True,
                         run_dir=run)]
        engine._apply_generation_fdr(1, trials, charge_alpha=True)
        led = (engine.state.get("alpha_ledger") or [])
        fam = [e.get("family_size") for e in led if e.get("generation") == 1]
        check("BH 候选集不含 screening(family_size=1,非 2)",
              fam == [1], f"family_size={fam}")

        engine.state["finalized_generations"] = [1]
        engine.finalize_generation(1, ["normal", "screen"],
                                   reflection={"key_findings": []})
        check("accepted_in_last_generation 不计 screening",
              engine.state.get("accepted_in_last_generation") == 1,
              str(engine.state.get("accepted_in_last_generation")))
        # trials_accepted 是全史累计,含 gen-00 baseline(既有语义:
        # 根 trial 的 accepted 计 1)。这里 2 = baseline + normal;
        # 若 screening 未被排除则是 3。
        check("trials_accepted 不计 screening(2=baseline+normal,非 3)",
              engine.state.get("trials_accepted") == 2,
              str(engine.state.get("trials_accepted")))


def test_confirm_parent_rejects_screening():
    print("== screening 不能当 confirm 父代 ==")
    from runtime.trial import confirmation_parent_eligibility
    ok, why = confirmation_parent_eligibility(_trial("s", screen=True))
    check("screening 父代被拒", ok is False and why ==
          "parent_is_screening_only", f"{ok}/{why}")
    # 普通 accepted 需要 selection_correction.passed(非根 trial 的硬门),
    # 这里补上以验证"screening 门没有误伤普通路径"。
    normal = _trial("n")
    normal.data["evaluation"]["selection_correction"] = {"passed": True}
    ok2, why2 = confirmation_parent_eligibility(normal)
    check("普通 accepted 父代仍可(未被误伤)", ok2 is True, why2)


# ---------------------------------------------------------------------------
# 3. Planner 侧:super_trial 默认关 → 不产出 screening 候选
# ---------------------------------------------------------------------------

def test_planner_super_trial_gated_by_config():
    print("== Planner super_trial 候选受配置门控 ==")
    from runtime.config import Config
    from runtime.planner import AutonomousPlanner
    from runtime.spec import TaskSpec
    with tempfile.TemporaryDirectory() as tds:
        run = Path(tds)
        (run / "trials").mkdir(parents=True, exist_ok=True)
        (run / "planner").mkdir(exist_ok=True)
        spec = TaskSpec(data={
            "task_id": "v213sup", "task_name": "v213sup",
            "model_type": "binary_classification",
            "target": {"name": "label", "positive_definition": "1"},
            "time": {"column": "dt"}, "entity": {"id_column": "uid"},
            "primary_metric": "auc",
            "split_policy": {"type": "time_based", "train": "2026-01",
                             "validation": "2026-02",
                             "shadow_test": "2026-03",
                             "final_test": "2026-04"},
            "data": {},
        })
        off = Config.load(skill_config=False)
        check("默认 super_trial=false",
              off.get("feature_exploration.super_trial") is False)
        cands_off = AutonomousPlanner(run, spec, off)._super_trial_candidate(
            None)
        check("默认不产出 screening 候选", cands_off == [], str(cands_off))

        on = Config.load(skill_config=False)
        on._data["feature_exploration"]["super_trial"] = True
        cands_on = AutonomousPlanner(run, spec, on)._super_trial_candidate(
            None)
        check("开启后产出 1 个候选", len(cands_on) == 1, str(len(cands_on)))
        if cands_on:
            p = cands_on[0]
            check("候选 recipe 带 screen_only",
                  (p.get("recipe") or {}).get("screen_only") is True)
            check("物化命令带 --screen-only",
                  "--screen-only" in p["materialize_command"],
                  p["materialize_command"])
            check("heavy 标记(宽面是大活)",
                  p["expected_resources"].get("heavy") is True)


# ---------------------------------------------------------------------------
# 4. CLI:--screen-only 落 trial.yaml
# ---------------------------------------------------------------------------

def test_cli_writes_screen_only_flag():
    print("== new-trial --screen-only 落盘 ==")
    import importlib.util
    spec_ = importlib.util.spec_from_file_location(
        "modelevo_cli_screen", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(cli)
    from runtime.config import Config
    from runtime.trial import TrialFactory
    cli.TrialFactory, cli.Config = TrialFactory, Config

    with tempfile.TemporaryDirectory() as tds:
        run = Path(tds)
        (run / "trials").mkdir(parents=True)
        (run / "sample-profile" / "splits").mkdir(parents=True)
        import pandas as pd
        pd.DataFrame({"f1": [0.1], "f2": [1]}).to_parquet(
            run / "sample-profile" / "splits" / "train.parquet")
        (run / "task-spec.yaml").write_text(yaml.safe_dump({
            "task_id": "v213cli", "task_name": "v213cli",
            "model_type": "binary_classification",
            "target": {"name": "label", "positive_definition": "y=1"},
            "primary_metric": "auc",
            "split_policy": {"type": "time_based", "train": "2025-01",
                             "validation": "2025-02",
                             "final_test": "2025-03"},
        }), encoding="utf-8")
        (run / "run-state.json").write_text(json.dumps({
            "phase": "EVOLVING", "harness_epoch": 0}), encoding="utf-8")

        class _A:
            pass
        a = _A()
        for k, v in {
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "algorithm": "lightgbm", "from_parent": None,
            "hpo_parent": None, "set": None,
            "hypothesis": "宽面初筛,建立族级方向先验",
            "rationale": "r", "falsification": "f", "actions": "widen_all",
            "action": "explore", "search_space": None,
            "budget_configs": None, "force_hpo_budget": False,
            "parents": None, "changed_modules": "feature",
            "override_feature_phase": None, "family_hints": "screen_",
            "generation": 1, "data_view": "v1", "heavy": True,
            "features": None, "override_beam": None, "screen_only": True,
            "planner_proposal_id": None, "planner_semantic_hash": None,
            "experience_ids": None, "allow_code_edit": False,
            "hpo_sample_rows": None,
        }.items():
            setattr(a, k, v)
        a.command = "new-trial"
        rc = cli.cmd_new_trial(a)
        check("创建成功", rc == 0, f"rc={rc}")
        tdirs = sorted((run / "trials").iterdir())
        check("恰好 1 个 trial", len(tdirs) == 1, str(len(tdirs)))
        if tdirs:
            d = yaml.safe_load((tdirs[0] / "trial.yaml")
                               .read_text(encoding="utf-8"))
            check("trial.yaml.screen_only=True",
                  d.get("screen_only") is True, str(d.get("screen_only")))
            check("search_action 仍为 explore(未新增动作值)",
                  d.get("search_action") == "explore",
                  str(d.get("search_action")))


if __name__ == "__main__":
    test_predicate()
    test_archive_excludes_screening()
    test_best_trial_excludes_screening()
    test_leaderboard_lists_screening_with_flag()
    test_direction_ledger_excludes_screening()
    test_fdr_and_accepted_counts_exclude_screening()
    test_confirm_parent_rejects_screening()
    test_planner_super_trial_gated_by_config()
    test_cli_writes_screen_only_flag()
    print(f"\nscreen-only:{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
