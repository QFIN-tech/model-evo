"""v2.18.0 中心化经验库(M3):运行时接线 —— 依据必填、planner 避免过滤与预填、
profile 泄漏对账、planner-context 第 7 节、代内依据汇总。

替代已移除的旧 memory/ 接线测试(test_v2_13_experience_loop t2–t5、
test_v2_15_8_memory_loop、test_v2_15_9_memory_isolation、test_v2_16_3 / v2_17_0 /
v2_17_2 中的对应用例)。经验库视图直接写入 ``<run>/experience/library-view.json``。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

import importlib.util
import json
from pathlib import Path

import yaml

from test_v2_17_2_planner import ATID, _cfg, _inventory, _run, _spec, _trial

# ---------------------------------------------------------------- 公共底座
# (原在 test_v2_13_experience_loop.py,该文件随旧 memory/ 全局库删除)

def _make_run(tmp_path: Path) -> Path:
    """最小 run 工作区(参考 tests/test_v29_fixes.py 的 _make_run)。"""
    run = tmp_path / "run-v213"
    (run / "trials").mkdir(parents=True)
    (run / "leaderboard").mkdir()
    (run / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "v213-exp-loop",
        "task_name": "经验回路回归",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "y==1"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based",
                         "train": "2026-01~2026-01",
                         "validation": "2026-02~2026-02",
                         "final_test": "2026-04~2026-04"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
    }), encoding="utf-8")
    (run / "run-state.json").write_text(json.dumps({
        "phase": "EVOLVING", "harness_epoch": 0,
    }), encoding="utf-8")
    return run


def _load_cli():
    """装载 scripts/modelevo.py(v2.11.9 拆分后 _mvo 由入口引导)。"""
    spec_m = importlib.util.spec_from_file_location(
        "mcli_v213", SKILL / "scripts" / "modelevo.py")
    m = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(m)
    return m


class _Args:
    """argparse.Namespace 等价物(模拟 CLI parse 后的 args)。"""

    def __init__(self, **kw):
        self.command = "new-trial"
        for k, v in kw.items():
            setattr(self, k, v)


def _new_trial_args(run: Path, **over) -> _Args:
    base = {
        "spec": str(run / "task-spec.yaml"), "run": str(run),
        "algorithm": None, "from_parent": None, "hpo_parent": None,
        "set": None, "hypothesis": "多窗口行为趋势特征有增量",
        "rationale": "r", "falsification": "f", "actions": "构造窗口特征",
        "action": "explore", "search_space": None,
        "budget_configs": None, "force_hpo_budget": False,
        "parents": None, "changed_modules": "feature",
        "override_feature_phase": None, "family_hints": None,
        "planner_proposal_id": None, "planner_semantic_hash": None,
        "generation": 1, "data_view": "v1", "heavy": False,
        "features": None, "experience_ids": None,
        "allow_code_edit": False, "hpo_sample_rows": None,
    }
    base.update(over)
    return _Args(**base)


BIN = {"model_type": "binary_classification", "primary_metric": "auc", "target_transform": "raw"}
TOPICS = [{"code": "feature.seq", "aliases": ["seq_"], "status": "active"},
          {"code": "feature.seq.gap", "aliases": ["seq_gap_d365"], "status": "active"},
          {"code": "feature.seq.window_count", "aliases": ["seqd_"], "status": "active"},
          {"code": "feature.base.interaction", "aliases": ["xint_"], "status": "active"}]


def _pr(pid, direction, topics, *, objects=(), category="feature_engineering", atid=ATID,
        claim=None, scope=BIN):
    return {"principle_id": pid, "version": 1, "direction": direction, "topics": list(topics),
            "objects": list(objects), "category": category, "scope": scope,
            "claim": claim or f"{pid} 结论", "condition": {}, "limitations": [],
            "evidence": {"abstract_task_ids": [atid]}}


def _write_view(run: Path, principles, *, atid=ATID, stats=None, dossiers=None,
                leaderboard=None, trials=None, watermark=42):
    data = {"schema": "modelevo.run-view/1", "watermark": watermark, "stale": False,
            "atid": atid, "requested_atid": atid, "namespace": "t", "scope": BIN,
            "principles": principles, "stats": stats or {}, "topics": TOPICS, "aliases": {},
            "fingerprints": [], "dossiers": dossiers or {}, "leaderboard": leaderboard or [],
            "baselines": {}, "trial_outcomes": trials or []}
    (run / "experience").mkdir(parents=True, exist_ok=True)
    (run / "experience" / "library-view.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- new-trial --basis

def _trial_data(run: Path, exclude=()):
    out = []
    for d in sorted((run / "trials").iterdir()):
        if (d / "trial.yaml").exists() and d.name not in exclude:
            out.append((d, yaml.safe_load((d / "trial.yaml").read_text(encoding="utf-8"))))
    return out


def test_new_trial_basis_required_when_pinned(tmp_path):
    run = _make_run(tmp_path)
    _write_view(run, [_pr("p-gap", 0, ["feature.seq.gap"])])
    cli = _load_cli()
    assert cli.cmd_new_trial(_new_trial_args(run)) == 1                       # 缺依据
    assert cli.cmd_new_trial(_new_trial_args(run, basis="p-ghost:tests")) == 1  # id 不存在
    assert cli.cmd_new_trial(_new_trial_args(run, basis="p-gap:bogus")) == 1    # 关系非法
    assert _trial_data(run) == []                                              # 失败全部回滚
    assert cli.cmd_new_trial(_new_trial_args(
        run, basis="p-gap:contradicts", topics="seq_gap_d365,foo.bar")) == 0
    (d, t), = _trial_data(run)
    assert t["basis"] == {"novel": False, "refs": [{"id": "p-gap", "relation": "contradicts"}],
                          "source": "agent"}
    assert t["experience_ids"] == ["p-gap"]
    assert t["topics"] == ["feature.seq.gap", "foo.bar"]                       # 词表外原文保留
    assert cli.cmd_new_trial(_new_trial_args(run, basis="novel", generation=2)) == 0
    # --from-parent 子代未给依据时继承父代
    (d / "code" / "train.py").write_text("import pandas\nHP = {}.get(\"hyperparameters\")\n",
                                         encoding="utf-8")
    (d / "config" / "trial.yaml").write_text(yaml.safe_dump(
        {"trial_id": d.name, "hyperparameters": {}, "data_view": "v1"}), encoding="utf-8")
    before = {x.name for x, _ in _trial_data(run)}
    assert cli.cmd_new_trial(_new_trial_args(run, from_parent=d.name, set=["lr=0.03"],
                                             generation=3)) == 0
    (_c, child), = [(x, y) for x, y in _trial_data(run) if x.name not in before]
    assert child["basis"]["refs"] == t["basis"]["refs"]
    assert child["basis"]["inherited_from"] == d.name


def test_new_trial_basis_optional_and_unverified_without_view(tmp_path):
    run = _make_run(tmp_path)                       # 未固定经验库视图(旧 run / 服务不可达)
    cli = _load_cli()
    assert cli.cmd_new_trial(_new_trial_args(run)) == 0
    assert cli.cmd_new_trial(_new_trial_args(run, experience_ids="p-any", generation=2)) == 0
    data = [t for _d, t in _trial_data(run)]
    assert not data[0].get("basis")
    assert data[1]["basis"]["unverified"] is True and data[1]["experience_ids"] == ["p-any"]


# ---------------------------------------------------------------- planner

def test_planner_avoid_filter_and_basis_prefill(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    _write_view(run, [
        # 没有具体对象的避免原则按主主题拦:拦所有 info:<列>:derived_interaction 候选
        _pr("p-di", -1, ["feature.base.interaction"]),
        # 架构类原则的对象是源列,不拦特征候选
        _pr("p-arch", -1, ["model.deep.architecture"], objects=["seq_d365"],
            category="architecture"),
        # 无增益原则不拦;其他任务的原则不拦
        _pr("p-neutral", 0, ["feature.seq.gap"], objects=["seq_gap_d365"]),
        _pr("p-other", -1, ["feature.seq.window_count"], objects=["seq_d365"],
            atid="binary-label-200w_1000w-deadbeef"),
    ])
    _trial(run, "t0", gen=0, modules=(), value=0.80)
    _inventory(run, [
        {"source_type": "base_face_family", "column": "offline_inner_x",
         "family": "offline", "consumption_status": "partially_consumed"},
        {"source_type": "reparse", "column": "seq_d365",
         "consumption_status": "partially_consumed",
         "granularity_space": [{"dimension": "window", "hint": "w", "state": "unexplored"}]},
        {"source_type": "reparse", "column": "seq_gap_d365",
         "consumption_status": "partially_consumed",
         "granularity_space": [{"dimension": "recency", "hint": "r", "state": "unexplored"}]},
    ])
    batch = AutonomousPlanner(run, spec, _cfg()).plan(slots=8)
    fams = {p["family"]: p for p in batch["proposals"]}
    assert "info:offline_inner_x:derived_interaction" not in fams
    assert {"info:seq_d365:window", "info:seq_gap_d365:recency"} <= set(fams)
    assert [(r["family"], r["experience_ids"]) for r in batch["experience_avoid_rejected"]] == [
        ("info:offline_inner_x:derived_interaction", ["p-di"])]
    # 物化命令:主题 + 预填依据(命中同主题原则 → tests,否则 novel)
    gap_cmd = fams["info:seq_gap_d365:recency"]["materialize_command"]
    assert "--topics feature.seq.gap" in gap_cmd and "--basis p-neutral:tests" in gap_cmd
    assert "--basis novel" in fams["info:seq_d365:window"]["materialize_command"]
    # v2.18.6:预填值同时带上,供 new-trial 判断是否原样沿用
    assert "--basis-prefill p-neutral:tests" in gap_cmd
    assert "--basis-prefill" not in fams["info:seq_d365:window"]["materialize_command"]


def test_planner_without_view_blocks_nothing(tmp_path):
    from runtime.planner import AutonomousPlanner

    spec = _spec()
    run = _run(tmp_path, spec)
    _trial(run, "t0", gen=0, modules=(), value=0.80)
    _inventory(run, [{"source_type": "base_face_family", "column": "offline_inner_x",
                      "family": "offline", "consumption_status": "partially_consumed"}])
    batch = AutonomousPlanner(run, spec, _cfg()).plan(slots=8)
    assert batch["experience_avoid_rejected"] == []
    assert all("--basis novel" in p["materialize_command"] for p in batch["proposals"]
               if p["action"] != "confirm")


# ---------------------------------------------------------------- profile 泄漏对账

def test_profile_cross_check_uses_positive_principles(tmp_path):
    from runtime.profile import SampleProfile, _experience_supports_column

    assert _experience_supports_column({"objects": ["seqd_"]}, "seqd_cnt_7")
    assert _experience_supports_column({"objects": ["seq_d365"]}, "seq_d365")
    assert not _experience_supports_column({"objects": ["xint_"]}, "seq_d365")
    run = tmp_path / "run"
    run.mkdir()
    (run / "run-state.json").write_text("{}")
    _write_view(run, [
        _pr("avoid-seq", -1, ["feature.seq"], objects=["seq"], claim="seq 通道边际递减"),
        _pr("eng-seq", 1, ["engineering.data_semantics"], objects=["seq_d365"],
            category="engineering"),
        _pr("reg-seq", 1, ["feature.seq"], objects=["seq_d365"],
            scope={**BIN, "model_type": "regression", "primary_metric": "rmse"}),
        _pr("pos-seq", 1, ["feature.seq"], objects=["seq_d365"], claim="seq reparse 加宽带来增益"),
    ])

    class _Self:
        out_dir = run / "sample-profile"

    props = [{"column": "seq_d365", "verdict": "banned-certain", "evidence": "id-like"},
             {"column": "uid_hash", "verdict": "banned-certain", "evidence": "id-like"}]
    hits = SampleProfile._cross_check_experience(_Self(), props, _spec())
    assert [h["experience_id"] for h in hits] == ["pos-seq"]
    assert props[0]["verdict"] == "ambiguous" and props[0]["reason"] == "experience_conflict"
    assert props[1]["verdict"] == "banned-certain"
    # 视图不可用 / self 缺失:静默无命中,不让 profile 失败
    assert SampleProfile._cross_check_experience(None, [dict(props[1])], _spec()) == []


# ---------------------------------------------------------------- planner-context 第 7 节

def test_planner_context_section7(tmp_path):
    from runtime.planner_context import build_planner_context, render_planner_context

    spec = _spec()
    run = _run(tmp_path, spec)
    _write_view(run, [_pr("p-avoid", -1, ["feature.base.interaction"], claim="跨族交互无增益"),
                      _pr("p-other-task", -1, ["feature.base.interaction"],
                          atid="binary-other-200w_1000w-bc5329dc")],
                dossiers={"task": {"sections": {"ceiling": "AUC 约 0.81", "effective": "seq 加宽"}},
                          "data_base": None},
                leaderboard=[{"rank": 1, "recipe_id": "a" * 64, "source_trial_id": "trial-013",
                              "algorithm": "xgboost",
                              "measurement": {"value": 0.8124, "gain": 0.0115, "kind": "legacy",
                                              "confirmed": False}}],
                trials=[{"run_id": "r@1", "trial_id": "t1", "atid": ATID, "primary_metric": "auc",
                         "higher_is_better": True, "family_hints": ["xint_"], "verdict": "rejected",
                         "delta_parent": -0.001, "delta_parent_ci_low": -0.002,
                         "delta_parent_ci_high": -0.0005}])
    ctx = build_planner_context(run, spec, _cfg())
    ex = ctx["experience"]
    assert ex["watermark"] == 42 and [a["principle_id"] for a in ex["avoid"]] == ["p-avoid"]
    assert ex["direction_table"]["failed"][0]["trial_id"] == "t1"
    md = render_planner_context(ctx)
    for s in ("7.0 任务卷宗", "AUC 约 0.81", "7.2 同任务方向表", "7.3 同任务避免清单",
              "`p-avoid`", "7.4 Warm Baselines", "trial-013", "experience search", "--basis"):
        assert s in md, s
    tel = (run / "experience-retrieval-telemetry.jsonl").read_text(encoding="utf-8")
    assert '"kind": "planner-context"' in tel


def test_planner_context_without_library(tmp_path):
    from runtime.planner_context import build_planner_context, render_planner_context

    spec = _spec()
    run = _run(tmp_path, spec)
    md = render_planner_context(build_planner_context(run, spec, _cfg()))
    assert "经验库不可用" in md


# ---------------------------------------------------------------- 代内依据汇总

def test_basis_summary(tmp_path):
    sys.path.insert(0, str(SKILL / "scripts"))
    from _mvo.cmd_trial import _basis_summary

    class T:
        def __init__(self, gen, basis, action="explore"):
            self.data = {"generation": gen, "basis": basis, "search_action": action}

    trials = [T(1, {"novel": False, "refs": [{"id": "p-a", "relation": "tests"}]}),
              T(1, {"novel": True, "refs": []}), T(1, None), T(1, None, action="warm_start"),
              T(0, {"novel": False, "refs": [{"id": "p-b", "relation": "follows"}]})]
    (tmp_path / "experience-retrieval-telemetry.jsonl").write_text(
        json.dumps({"shown_ids": ["p-a", "p-b", "p-c"]}) + "\n", encoding="utf-8")
    s = _basis_summary(tmp_path, 1, trials)
    assert (s["trials"], s["cited"], s["novel"], s["missing"]) == (3, 1, 1, 1)
    assert s["cited_principles"] == {"p-a": 1}
    assert s["shown_never_cited"] == 1 and s["shown_never_cited_ids"] == ["p-c"]


# ---------------------------------------------------------------- M4 切换

def test_diagnose_flags_runs_created_before_switch(tmp_path):
    sys.path.insert(0, str(SKILL / "scripts"))
    from _mvo.cmd_inspect import _diag_library

    run = tmp_path / "run"
    run.mkdir()

    def codes(state):
        (run / "run-state.json").write_text(json.dumps(state), encoding="utf-8")
        return [f["code"] for f in _diag_library(run)]

    assert codes({"skill_version": "2.17.4", "phase": "EVOLVING"}) == [
        "run_predates_experience_library"]
    assert codes({"skill_version": "2.17.4", "phase": "FINAL_EVALUATED"}) == []
    assert codes({"skill_version": "2.18.0", "phase": "EVOLVING"}) == []
    assert codes({"phase": "EVOLVING"}) == []            # v2.15.8 前未记版本的旧 run 不判


def test_main_session_guard_allows_experience_not_library():
    sys.path.insert(0, str(SKILL / "assets"))
    import guard_main_session as guard

    assert "experience" in guard.ALLOWED_SUBCOMMANDS
    assert "library" not in guard.ALLOWED_SUBCOMMANDS


def test_import_legacy_requires_existing_memory_dir(tmp_path, capsys):
    import argparse
    sys.path.insert(0, str(SKILL / "scripts"))
    from _mvo.cmd_library import cmd_library_import_legacy

    args = argparse.Namespace(memory=str(tmp_path / "nope"), extra_principles=None, dry_run=True,
                              runs_root=None, test_namespace=None, resolve_atid=False,
                              report=None, uri=None, namespace=None, root=None)
    assert cmd_library_import_legacy(args) == 1
    assert "global/principles.jsonl" in capsys.readouterr().out


# ---------------------------------------------------------------- 配方完整性

def test_recipe_bundle_carries_full_trial(tmp_path):
    sys.path.insert(0, str(SKILL / "scripts"))
    from _mvo.cmd_startup import _copy_recipe_code_modules
    from _mvo.library_snapshot import _recipe_files

    run = tmp_path / "run"
    td = run / "trials" / "t1"
    for sub in ("code/__pycache__", "config", "views", "sample-profile/splits-v2"):
        (td / sub if not sub.startswith(("views", "sample")) else run / sub).mkdir(
            parents=True, exist_ok=True)
    (td / "code" / "train.py").write_text("from deep_seq import f\n", encoding="utf-8")
    (td / "code" / "deep_seq.py").write_text("def f(): pass\n", encoding="utf-8")
    (td / "code" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (td / "config" / "trial.yaml").write_text("data_view: v2\n", encoding="utf-8")
    (td / "config" / "search-space.yaml").write_text("lr: [0.01, 0.1]\n", encoding="utf-8")
    (run / "views" / "v2.py").write_text("def build_view(df): return df\n", encoding="utf-8")
    (run / "sample-profile" / "splits-v2" / "manifest.json").write_text("{}", encoding="utf-8")
    (run / "env-manifest.json").write_text(json.dumps({
        "packages": [{"name": "xgboost", "import_ok": True, "version": "2.1.4"},
                     {"name": "torch", "import_ok": False, "version": None}],
        "interpreter": "/usr/bin/python3"}), encoding="utf-8")
    (run / "run-state.json").write_text(json.dumps({"skill_version": "2.18.0",
                                                    "harness_hash": "h"}), encoding="utf-8")
    files = _recipe_files(run, td)
    assert set(files) == {"train.py", "trial.yaml", "code/deep_seq.py",
                          "config/search-space.yaml", "environment.json",
                          "data_view/v2.py", "data_view/manifest.json"}
    env = json.loads(files["environment.json"])
    assert env["packages"] == {"xgboost": "2.1.4"} and env["skill_version"] == "2.18.0"
    # 热启动 / baseline 复用把其余代码模块拷进新 trial
    bundle = tmp_path / "bundle"
    for name, data in files.items():
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        (bundle / name).write_bytes(data)
    new_trial = tmp_path / "new-trial"
    assert _copy_recipe_code_modules(bundle, new_trial) == ["deep_seq.py"]
    assert (new_trial / "code" / "deep_seq.py").read_text(encoding="utf-8") == "def f(): pass\n"
