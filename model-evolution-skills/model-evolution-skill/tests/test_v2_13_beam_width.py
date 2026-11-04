"""回归测试:v2.13.2 特征 bundle 宽度接线 + gen-01 动作硬门。

背景:`feature_exploration.bundle_beam_width` 自 v10 起就是配置键,
但 runtime **零引用**(逐处 grep 确认:runtime/ scripts/ assets/*.py
里只有 config.py 与 global-config.yaml 两处定义)。gen-01 的实际宽度
由调用方手传的 `plan --slots N` 决定,配置完全不参与。

**为什么门放在 new-trial 而不是 Planner**:实测
`example-seq-v3-v2.12.2.4` 的 gen-01 共 15 个 trial,**0 个**带
`planner_proposal_id` —— Agent 全部手建,Planner 侧任何约束都咬不到。
Planner 侧只做记账(批次里落 `feature_beam`),真正生效的是创建口。

同时锁定 gen-01 动作门:`example-seq-v3-v2.12.2.4` 的 gen-01 有一个
Agent 手建的 exploit "prune 对照臂"(trial-007/trial-016),净收益
+0.00022(promising),代价是一个槽位 —— 首代无父代证据可加码。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
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


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "modelevo_cli_beam", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from runtime.config import Config
    from runtime.trial import TrialFactory
    cli.TrialFactory, cli.Config = TrialFactory, Config
    return cli


def _make_run(td: Path) -> Path:
    (td / "trials").mkdir(parents=True)
    (td / "leaderboard").mkdir()
    sp = td / "sample-profile" / "splits"
    sp.mkdir(parents=True)
    pd.DataFrame({"f1": [0.1, 0.2], "f2": [1, 2]}).to_parquet(
        sp / "train.parquet")
    pd.DataFrame({"f1": [0.4], "f2": [4]}).to_parquet(
        sp / "validation.parquet")
    (td / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "v213beam", "task_name": "v213beam",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "y=1"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2025-01",
                         "validation": "2025-02",
                         "final_test": "2025-03"},
    }), encoding="utf-8")
    (td / "run-state.json").write_text(json.dumps({
        "phase": "EVOLVING", "harness_epoch": 0}), encoding="utf-8")
    return td


def _args(**over):
    class _A:
        pass
    a = _A()
    base = {
        "spec": None, "run": None, "algorithm": "lightgbm",
        "from_parent": None, "hpo_parent": None, "set": None,
        "hypothesis": "单测假设陈述,长度足够",
        "rationale": "r", "falsification": "f", "actions": "a",
        "action": "explore", "search_space": None,
        "budget_configs": None, "force_hpo_budget": False,
        "parents": None, "changed_modules": "feature",
        "override_feature_phase": None, "family_hints": None,
        "generation": None, "data_view": "v1", "heavy": False,
        "features": None, "override_beam": None, "screen_only": False,
        "planner_proposal_id": None, "planner_semantic_hash": None,
        "experience_ids": None, "allow_code_edit": False,
        "hpo_sample_rows": None,
    }
    base.update(over)
    for k, v in base.items():
        setattr(a, k, v)
    a.command = "new-trial"
    return a


def _count(run: Path) -> int:
    return len(list((run / "trials").iterdir()))


# ---------------------------------------------------------------------------
# 1. gen-01 动作硬门
# ---------------------------------------------------------------------------

def test_gen1_blocks_exploit_and_crossover():
    print("== gen-01 禁止 exploit / crossover ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        spec, rd = str(run / "task-spec.yaml"), str(run)

        for action in ("exploit", "crossover"):
            n0 = _count(run)
            rc = cli.cmd_new_trial(_args(spec=spec, run=rd, action=action,
                                         generation=1,
                                         changed_modules="feature"))
            check(f"gen-01 --action {action} 被拒(rc=1)", rc == 1, f"rc={rc}")
            check(f"gen-01 {action} 未落盘孤儿目录",
                  _count(run) == n0, f"{n0}→{_count(run)}")

        # explore / data_view 不受影响
        rc = cli.cmd_new_trial(_args(spec=spec, run=rd, action="explore",
                                     generation=1,
                                     changed_modules="feature"))
        check("gen-01 --action explore 仍可创建", rc == 0, f"rc={rc}")


def test_gen2_allows_exploit():
    print("== gen-02 起 exploit 不被误伤 ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        spec, rd = str(run / "task-spec.yaml"), str(run)
        rc = cli.cmd_new_trial(_args(spec=spec, run=rd, action="exploit",
                                     generation=2,
                                     changed_modules="feature"))
        check("gen-02 --action exploit 可创建(门只拦首代)",
              rc == 0, f"rc={rc}")


# ---------------------------------------------------------------------------
# 2. bundle_beam_width 槽位门
# ---------------------------------------------------------------------------

def test_beam_width_caps_feature_trials():
    print("== bundle_beam_width 槽位门 ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        spec, rd = str(run / "task-spec.yaml"), str(run)
        # 默认宽度 4:建满 4 个后第 5 个被拒
        rcs = []
        for i in range(5):
            rcs.append(cli.cmd_new_trial(_args(
                spec=spec, run=rd, action="explore", generation=1,
                changed_modules="feature",
                hypothesis=f"第 {i} 个 bundle 假设陈述")))
        check("前 4 个创建成功(默认宽度 4)",
              rcs[:4] == [0, 0, 0, 0], str(rcs))
        check("第 5 个被槽位门拒绝", rcs[4] == 1, str(rcs))
        check("实际落盘 4 个", _count(run) == 4, str(_count(run)))

        # --override-beam 放行并留痕
        rc = cli.cmd_new_trial(_args(
            spec=spec, run=rd, action="explore", generation=1,
            changed_modules="feature", override_beam="消融对照臂,需超宽",
            hypothesis="超宽放行的第 5 个 bundle"))
        check("--override-beam 放行(rc=0)", rc == 0, f"rc={rc}")
        check("落盘 5 个", _count(run) == 5, str(_count(run)))
        tdirs = sorted((run / "trials").iterdir())
        d = yaml.safe_load((tdirs[-1] / "trial.yaml")
                           .read_text(encoding="utf-8"))
        check("beam_override 留痕理由",
              (d.get("beam_override") or {}).get("reason")
              == "消融对照臂,需超宽",
              str(d.get("beam_override")))


def test_beam_ignores_model_and_confirm_trials():
    print("== 槽位门不误伤 model/confirm ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        spec, rd = str(run / "task-spec.yaml"), str(run)
        for i in range(4):
            cli.cmd_new_trial(_args(spec=spec, run=rd, action="explore",
                                    generation=2, changed_modules="feature",
                                    hypothesis=f"bundle {i} 假设陈述"))
        n0 = _count(run)
        # model 方向不受特征槽位门约束(它走 capacity 闸门那条线)
        sp = run / "sp.yaml"
        sp.write_text(yaml.safe_dump({
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2,
                              "log": True},
            "num_leaves": {"type": "int", "low": 15, "high": 63},
        }), encoding="utf-8")
        rc = cli.cmd_new_trial(_args(spec=spec, run=rd, action="hpo",
                                     generation=2, changed_modules="model",
                                     search_space=str(sp),
                                     algorithm="lightgbm",
                                     hypothesis="模型方向 HPO 假设",
                                     # 本 run 仍在特征期内;特征期门另有
                                     # 专测,这里只验 beam 门不约束 HPO
                                     override_feature_phase="单测"))
        check("特征槽满后 model/HPO 仍可创建(不受 beam 门约束)",
              rc == 0, f"rc={rc}")
        check("确实落盘", _count(run) == n0 + 1, str(_count(run)))


def test_beam_screening_counts_toward_width():
    print("== screening 计入槽位(总数 = bundle_beam_width)==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        spec, rd = str(run / "task-spec.yaml"), str(run)
        rc0 = cli.cmd_new_trial(_args(spec=spec, run=rd, action="explore",
                                      generation=1, changed_modules="feature",
                                      screen_only=True, heavy=True,
                                      hypothesis="宽面初筛 screening 假设"))
        check("screening trial 创建成功", rc0 == 0, f"rc={rc0}")
        rcs = [cli.cmd_new_trial(_args(
            spec=spec, run=rd, action="explore", generation=1,
            changed_modules="feature",
            hypothesis=f"bundle {i} 假设陈述")) for i in range(3)]
        check("screening + 3 bundle = 4 满额(前 3 个成功)",
              rcs == [0, 0, 0], str(rcs))
        rc_last = cli.cmd_new_trial(_args(
            spec=spec, run=rd, action="explore", generation=1,
            changed_modules="feature", hypothesis="第 5 个应被拒"))
        check("总数达 4 后第 5 个被拒(screening 计入)",
              rc_last == 1, f"rc={rc_last}")


def test_beam_is_per_generation():
    print("== 槽位按代计数(下一代重置)==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        spec, rd = str(run / "task-spec.yaml"), str(run)
        for i in range(4):
            cli.cmd_new_trial(_args(spec=spec, run=rd, action="explore",
                                    generation=1, changed_modules="feature",
                                    hypothesis=f"g1 bundle {i} 假设"))
        rc = cli.cmd_new_trial(_args(spec=spec, run=rd, action="explore",
                                     generation=2, changed_modules="feature",
                                     hypothesis="gen-02 的新 bundle 假设"))
        check("gen-02 槽位独立,可继续创建", rc == 0, f"rc={rc}")


# ---------------------------------------------------------------------------
# 3. Planner 侧记账
# ---------------------------------------------------------------------------

def test_planner_batch_records_feature_beam():
    print("== Planner 批次落 feature_beam 记账 ==")
    from runtime.config import Config
    from runtime.planner import AutonomousPlanner
    from runtime.spec import TaskSpec
    with tempfile.TemporaryDirectory() as tds:
        run = Path(tds)
        (run / "trials").mkdir(parents=True, exist_ok=True)
        spec = TaskSpec(data={
            "task_id": "v213beam2", "task_name": "v213beam2",
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
        batch = AutonomousPlanner(run, spec,
                                  Config.load(skill_config=False)).plan(
            slots=4)
        fb = batch.get("feature_beam")
        check("批次含 feature_beam 记账", isinstance(fb, dict), str(fb))
        if isinstance(fb, dict):
            check("bundle_beam_width 如实落盘(4)",
                  fb.get("bundle_beam_width") == 4, str(fb))
            check("super_trial_enabled 如实落盘(false)",
                  fb.get("super_trial_enabled") is False, str(fb))
            check("feature_slots_planned 字段存在",
                  "feature_slots_planned" in fb, str(fb))


# ---------------------------------------------------------------------------
# 4. finalize 的 underfilled 对账
# ---------------------------------------------------------------------------

def test_finalize_records_underfilled():
    print("== finalize 落 feature_beam.underfilled(只记不拒)==")
    from runtime.config import Config
    from runtime.evolution import EvolutionEngine
    from runtime.spec import TaskSpec
    from runtime.trial import TrialRecord
    with tempfile.TemporaryDirectory() as tds:
        run = Path(tds)
        (run / "trials").mkdir(parents=True, exist_ok=True)
        (run / "leaderboard").mkdir(exist_ok=True)
        (run / "run-state.json").write_text(
            json.dumps({"baseline_id": "t0"}), encoding="utf-8")
        spec = TaskSpec(data={
            "task_id": "v213beam3", "task_name": "v213beam3",
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

        def _mk(tid, gen, action="explore", modules=("feature",)):
            d = {"trial_id": tid, "algorithm": "lightgbm",
                 "parent_trial_ids": [], "generation": gen,
                 "search_action": action,
                 "changed_modules": list(modules), "data_view": "v1",
                 "status": "EVALUATED",
                 "evaluation": {"primary_metric": {"name": "auc",
                                                   "value": 0.79},
                                "verdict": "rejected",
                                "stability": {"passed": True},
                                "integrity": {"valid": True, "errors": []}},
                 "result": {"runtime_sec": 10}}
            tdir = run / "trials" / tid
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / "trial.yaml").write_text(
                yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
            (tdir / "evaluation.json").write_text(
                json.dumps(d["evaluation"]), encoding="utf-8")
            return TrialRecord(data=d, trial_dir=tdir)

        _mk("t0", 0, action="explore", modules=())
        _mk("t1", 1)
        engine = EvolutionEngine(run, spec, Config.load(skill_config=False))
        engine.finalize_generation(1, ["t1"],
                                   reflection={"key_findings": []})
        gy = yaml.safe_load((run / "generations" / "gen-01"
                             / "generation.yaml").read_text(encoding="utf-8"))
        fb = gy.get("feature_beam")
        check("generation.yaml 含 feature_beam", isinstance(fb, dict),
              str(fb))
        if isinstance(fb, dict):
            check("feature_trials 如实计数(1)",
                  fb.get("feature_trials") == 1, str(fb))
            check("underfilled = 4-1 = 3(只记不拒)",
                  fb.get("underfilled") == 3, str(fb))
            check("未抛错、run 未卡死(「只记不拒」的核心)",
                  gy.get("generation_id") == "gen-01")


if __name__ == "__main__":
    test_gen1_blocks_exploit_and_crossover()
    test_gen2_allows_exploit()
    test_beam_width_caps_feature_trials()
    test_beam_ignores_model_and_confirm_trials()
    test_beam_screening_counts_toward_width()
    test_beam_is_per_generation()
    test_planner_batch_records_feature_beam()
    test_finalize_records_underfilled()
    print(f"\nbeam/gen1:{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)
