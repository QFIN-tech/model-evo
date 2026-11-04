"""回归测试:v2.16.0 特征期门(Planner / new-trial 同一判定)。

背景见 runtime/feature_phase.py 与 runtime/planner.py 的头部注释:此前
「先特征、后模型」由 capacity 闸门兼职,而它的两个触发信号(add_rows /
new_information)在抽样面上近乎常量(本机 44 份去重报告局部歧义指数
0.946–0.961、阈值 0.10,44/44 触发),门的实际输出不携带信息;同时两层
口径不一致:Planner 期满降 advisory,new-trial 不看代数仍要 override;
只写 --action hpo 不带 --changed-modules 可绕过 new-trial 门;报告不存在
时整条规则又完全不生效。

本文件锁定解耦后的三条不变量:
1. 门只看代数,与 capacity-report.json 是否存在/说什么无关;
2. Planner 与 new-trial 判定一致(期满两层都放行,特征期内两层都拦);
3. 放行通道两条且都留痕:Planner 释放名额、人工 override。
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


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "modelevo_cli_fp", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
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
        "task_id": "v216fp", "task_name": "v216fp",
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


def _write_search_space(run: Path) -> str:
    sp = run / "sp.yaml"
    sp.write_text(yaml.safe_dump({
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2,
                          "log": True},
        "num_leaves": {"type": "int", "low": 15, "high": 63},
    }), encoding="utf-8")
    return str(sp)


def _evaluated_trial(run: Path, generation: int, verdict: str) -> None:
    """落一个带 evaluation 的 trial(代数口径只看这个)。"""
    d = run / "trials" / f"trial-{900 + generation:03d}-g{generation:02d}-x"
    d.mkdir(parents=True)
    (d / "trial.yaml").write_text(yaml.safe_dump({
        "trial_id": d.name, "generation": generation,
        "status": "EVALUATED", "search_action": "explore",
        "changed_modules": ["feature"],
        "evaluation": {"verdict": verdict, "trial_id": d.name},
    }, allow_unicode=True), encoding="utf-8")


def _add_capacity_report(run: Path) -> None:
    """即使报告说 add_rows/new_information 优先,也不应影响门。"""
    (run / "sample-profile" / "capacity-report.json").write_text(
        json.dumps({"verdict": {"action_ranking": [
            {"action": "add_rows", "priority": 1},
            {"action": "new_information", "priority": 1},
            {"action": "model_hpo", "priority": 2}]}}), encoding="utf-8")


def _last_hpo_override(run: Path) -> dict | None:
    """返回最后一个 hpo trial 的 feature_phase_override(无则 None)。"""
    dirs = sorted((run / "trials").iterdir())
    for d in reversed(dirs):
        p = d / "trial.yaml"
        if not p.exists():
            continue
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if data.get("search_action") == "hpo":
            return data.get("feature_phase_override")
    return None


# ---------------------------------------------------------------------------
# 1. 特征期内:拦模型方向,且与 capacity 报告无关
# ---------------------------------------------------------------------------

def _new_hpo(cli, run: Path, **over) -> int:
    kw = dict(spec=str(run / "task-spec.yaml"), run=str(run),
              action="hpo", generation=1, changed_modules="model",
              search_space=_write_search_space(run),
              hypothesis="模型方向 HPO 假设")
    kw.update(over)
    return cli.cmd_new_trial(_args(**kw))


def test_locked_without_capacity_report():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")  # gen-00 baseline 不计
        assert not (run / "sample-profile" / "capacity-report.json").exists()
        n0 = len(list((run / "trials").iterdir()))
        assert _new_hpo(cli, run) == 1
        assert len(list((run / "trials").iterdir())) == n0, "不落孤儿目录"


def test_locked_even_if_capacity_report_says_model_first():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")
        (run / "sample-profile" / "capacity-report.json").write_text(
            json.dumps({"verdict": {"action_ranking": [
                {"action": "model_hpo", "priority": 2}]}}),
            encoding="utf-8")
        assert _new_hpo(cli, run) == 1, "门只看代数,不看报告"


def test_hpo_without_changed_modules_also_locked():
    """旧实现 `if cap_path.exists() and changed_modules` 可被绕过。"""
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")
        _add_capacity_report(run)
        assert _new_hpo(cli, run, changed_modules=None) == 1


def test_non_model_directions_unaffected():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")
        _add_capacity_report(run)
        rc = cli.cmd_new_trial(_args(
            spec=str(run / "task-spec.yaml"), run=str(run),
            action="explore", generation=1, changed_modules="feature",
            hypothesis="特征方向假设陈述"))
        assert rc == 0


# ---------------------------------------------------------------------------
# 2. 两条放行通道都留痕
# ---------------------------------------------------------------------------

def test_manual_override_marks_trial():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")
        assert _new_hpo(cli, run,
                        override_feature_phase="特征面已证伪,模型轴先行") == 0
        ovr = _last_hpo_override(run) or {}
        assert ovr.get("source") == "manual", ovr
        assert ovr.get("reason") == "特征面已证伪,模型轴先行", ovr


def test_override_capacity_flag_is_alias():
    """旧 CLI 旗标 --override-capacity 仍可用(Agent 习惯/历史文档)。"""
    cli = _load_cli()
    ns = cli.build_parser().parse_args(
        ["new-trial", "--run", "x", "--override-capacity", "理由"])
    assert ns.override_feature_phase == "理由"


def test_planner_release_accepted_by_new_trial():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")
        (run / "planner").mkdir()
        (run / "planner" / "latest.json").write_text(json.dumps({
            "feature_phase_gate": {
                "released_proposal_id": "prop-1",
                "release_reason": "特征侧已收敛(...)",
            }}), encoding="utf-8")
        assert _new_hpo(cli, run, planner_proposal_id="prop-1") == 0
        ovr = _last_hpo_override(run) or {}
        assert ovr.get("source") == "planner_feature_phase_release", ovr
        assert ovr.get("proposal_id") == "prop-1", ovr
        # 不是释放名额的 proposal 仍拒
        assert _new_hpo(cli, run, planner_proposal_id="prop-2",
                        hypothesis="模型方向 HPO 假设 2") == 1


# ---------------------------------------------------------------------------
# 3. 特征期满:与 Planner 一致地放行
# ---------------------------------------------------------------------------

def test_open_after_min_generations():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")
        for g in (1, 2):
            _evaluated_trial(run, g, "rejected")
        _add_capacity_report(run)  # 报告仍判 add_rows/new_information 优先
        assert _new_hpo(cli, run, generation=3) == 0, "期满后无需 override"
        assert _last_hpo_override(run) is None, "无放行就不留痕"


def test_min_generations_zero_disables_gate():
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as tds:
        run = _make_run(Path(tds))
        _evaluated_trial(run, 0, "accepted")
        (run / "modelevo-config.yaml").write_text(yaml.safe_dump({
            "feature_exploration": {"min_generations": 0}}),
            encoding="utf-8")
        assert _new_hpo(cli, run) == 0
