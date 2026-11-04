"""回归测试:v2.15.8 在途收口(HPO 嫁接 / ROI 停止 / 熔断 / wall 余量 /
has_secondary / baseline cap 遥测 / 父代冻结 / 版本与配置)。

见 docs/MODELEVO_V2_15_8_MEMORY_LOOP_FIX_20261013.md §B。
"""
from __future__ import annotations

import ast
import json
import logging
import re
import sys
from pathlib import Path

import pytest
import yaml

SKILL = Path(__file__).resolve().parents[1]
for _p in (str(SKILL), str(SKILL / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime import __version__, version_at_least  # noqa: E402
from runtime.config import Config  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402


def _src(rel: str) -> str:
    return (SKILL / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# B2 HPO 嫁接搬运 _clamp_trees + _MAX_TREES
# --------------------------------------------------------------------------

def _baseline_parent(tmp_path) -> str:
    """baseline 血缘父代(refactor8 trial-011 的父代就是这一形态:前置段
    没有 _MAX_TREES / _clamp_trees)。"""
    from runtime.baseline import BaselineRunner

    spec = TaskSpec(data={"task_id": "x", "task_name": "x",
                          "model_type": "binary_classification",
                          "target": {"name": "y"}, "primary_metric": "auc",
                          "id_column": "uid", "time_column": "dt"})
    br = BaselineRunner(spec, Config.load(skill_config=False), tmp_path)
    pdir = tmp_path / "parent"
    (pdir / "code").mkdir(parents=True)
    code = Path(br.write_baseline_code(pdir, "xgboost", ["a", "b"])).read_text(
        encoding="utf-8")
    assert "def _clamp_trees(" not in code, "前提:baseline 父代不含 _clamp_trees"
    return code


def test_graft_carries_clamp_trees(tmp_path):
    """refactor8 trial-011:嫁接后的 train.py 调 _clamp_trees 却无定义。"""
    import modelevo as M

    child = tmp_path / "child"
    (child / "code").mkdir(parents=True)
    (child / "code" / "train.py").write_text(_baseline_parent(tmp_path),
                                             encoding="utf-8")

    class FakeTrial:
        trial_dir = str(child)
        trial_id = "trial-900-g09-hpo-xgboost-9999"
        data = {"algorithm": "xgboost"}

    logging.disable(logging.CRITICAL)
    try:
        M._graft_hpo_section(FakeTrial())
        g = (child / "code" / "train.py").read_text(encoding="utf-8")
        ast.parse(g)
        assert "_clamp_trees(" in g
        assert g.count("def _clamp_trees(") == 1
        assert len(re.findall(r"(?m)^_MAX_TREES\s*=", g)) == 1
        M._graft_hpo_section(FakeTrial())       # 幂等
        g2 = (child / "code" / "train.py").read_text(encoding="utf-8")
        assert g2.count("def _clamp_trees(") == 1
        assert len(re.findall(r"(?m)^_MAX_TREES\s*=", g2)) == 1
    finally:
        logging.disable(logging.NOTSET)


# --------------------------------------------------------------------------
# B4 ROI 停止接线
# --------------------------------------------------------------------------

def test_evolution_passes_generation_compute_hours():
    tree = ast.parse(_src("runtime/evolution.py"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "check"
             and isinstance(n.func.value, ast.Attribute)
             and n.func.value.attr == "stop"]
    assert calls, "找不到 self.stop.check 调用"
    kws = {k.arg for k in calls[0].keywords}
    assert "generation_compute_hours" in kws
    src = _src("runtime/evolution.py")
    assert 'version_at_least(_rs.get("skill_version"), "2.15.8")' in src
    assert 'finalized_generations' in src.split(
        "generation_compute_hours: list[float] | None = None")[1][:1500]


def test_roi_stop_fires_on_aligned_hours():
    from runtime.generation import StopCondition

    spec = TaskSpec(data={"task_id": "x", "task_name": "x",
                          "model_type": "binary_classification",
                          "target": {"name": "y"}, "primary_metric": "auc"})
    sc = StopCondition(spec, Config.load(skill_config=False))
    base = dict(n_trials_started=10, n_generations=4, wall_time_hours=1.0,
                trials_accepted=1, best_metric=0.8, baseline_metric=0.79,
                noise=0.0003, unexplored_high_potential=["x"],
                generation_best_deltas=[0.006, 0.0001, 0.0001])
    rep = sc.check(**base, generation_compute_hours=[1.0, 5.0, 5.0])
    assert rep is not None and "roi_diminishing_return" in str(rep)
    rep2 = sc.check(**base)
    assert rep2 is None or "roi_diminishing_return" not in str(rep2)


# --------------------------------------------------------------------------
# B5 / B6 / B9 调度、熔断、父代冻结
# --------------------------------------------------------------------------

def test_cmd_run_generation_passes_wall_remaining():
    src = _src("scripts/_mvo/cmd_trial.py")
    assert "wall_remaining_sec=_wall_remaining_sec" in src
    assert 'version_at_least(_rs.get("skill_version"), "2.15.8")' in src


def _bare_runner(tmp_path):
    from runtime.runner import TrialRunner
    from runtime.scheduler import FailureCircuitBreaker

    r = object.__new__(TrialRunner)
    r.run_dir = tmp_path
    r.breaker = FailureCircuitBreaker(Config.load(skill_config=False))
    r.breaker_open = False
    r.breaker_reason = None
    return r


def test_breaker_delegates_consecutive(tmp_path):
    r = _bare_runner(tmp_path)
    for _ in range(r.breaker.consecutive_limit):
        r.breaker.record("code_error")
    trip = r._check_breaker()
    assert trip and trip["kind"] == "consecutive_same_reason"
    assert r.breaker_open and r.breaker_reason == "code_error"
    assert json.loads((tmp_path / "breaker-state.json").read_text())["open"]


def test_breaker_generation_failure_rate(tmp_path):
    r = _bare_runner(tmp_path)
    assert r._check_breaker(1, 4) is None
    trip = r._check_breaker(3, 4)
    assert trip and trip["kind"] == "generation_failure_rate"
    assert r.breaker_reason.startswith("generation_failure_rate")


def test_generation_breaker_warn_only_for_old_runs(tmp_path):
    from runtime.gen_executor import GenerationExecutor

    r = _bare_runner(tmp_path)
    (tmp_path / "run-state.json").write_text("{}")
    ex = object.__new__(GenerationExecutor)
    ex.runner = r
    summary = {"n_failed": 3, "n_succeeded": 1}
    ex._generation_breaker(summary, tmp_path)
    assert summary["breaker"]["warn_only"] and not r.breaker_open
    (tmp_path / "run-state.json").write_text(
        json.dumps({"skill_version": "2.15.8"}))
    summary2 = {"n_failed": 3, "n_succeeded": 1}
    ex._generation_breaker(summary2, tmp_path)
    assert summary2["breaker"]["tripped"] and r.breaker_open


def test_from_parent_verifies_parent_source():
    src = _src("scripts/_mvo/cmd_trial.py")
    seg = src.split("v2.15.8:父代源码冻结前置断言")[1][:1600]
    assert "verify_trial_source(_parent_dir)" in seg
    assert 'version_at_least(_rs.get("skill_version"), "2.15.8")' in seg
    assert "return 1" in seg


# --------------------------------------------------------------------------
# B7 inventory has_secondary 同源判据
# --------------------------------------------------------------------------

def _inv_run(tmp_path, prefixes):
    pd = pytest.importorskip("pandas")
    run = tmp_path / "run"
    (run / "sample-profile").mkdir(parents=True)
    pd.DataFrame({"feature": ["offline_amt_t30"], "iv": [0.5]}).to_csv(
        run / "sample-profile" / "feature-quality.csv", index=False)
    fi = run / "trials" / "trial-001" / "feature"
    fi.mkdir(parents=True)
    pd.DataFrame({"feature": [f"{p}_x" for p in prefixes],
                  "importance": [1.0] * len(prefixes)}).to_csv(
        fi / "feature_importance.csv", index=False)
    return run


@pytest.mark.parametrize("prefixes,status", [
    (["xint", "seqd"], "partially_consumed"),     # 别族前缀不是本族二次加工
    (["offline", "offlined"], "fully_consumed"),   # 同源派生族
])
def test_has_secondary_same_source(tmp_path, prefixes, status):
    from runtime.inventory import build_inventory

    run = _inv_run(tmp_path, prefixes)
    spec = TaskSpec(data={"task_id": "x", "task_name": "x",
                          "model_type": "binary_classification",
                          "target": {"name": "y"}, "primary_metric": "auc"})
    inv = build_inventory(run, spec)
    src = [s for s in inv["sources"] if s.get("family") == "offline"]
    assert src and src[0]["consumption_status"] == status


# --------------------------------------------------------------------------
# B8 baseline cap_bound 遥测
# --------------------------------------------------------------------------

def test_baseline_template_writes_cap_bound(tmp_path):
    from runtime.baseline import BaselineRunner

    spec = TaskSpec(data={"task_id": "x", "task_name": "x",
                          "model_type": "binary_classification",
                          "target": {"name": "y"}, "primary_metric": "auc",
                          "id_column": "uid", "time_column": "dt"})
    br = BaselineRunner(spec, Config.load(skill_config=False), tmp_path)
    tdir = tmp_path / "trial"
    (tdir / "code").mkdir(parents=True)
    path = br.write_baseline_code(tdir, "xgboost", ["a", "b"])
    code = Path(path).read_text(encoding="utf-8")
    ast.parse(code)
    assert 'metrics["es_info"]' in code and '"cap_bound": bool(_bound)' in code


# --------------------------------------------------------------------------
# B3 / B10 / B11 / B12 版本、配置、标记
# --------------------------------------------------------------------------

def test_version_unified():
    assert __version__ == "2.20.0"
    assert "(v2.20.0)" in _src("SKILL.md")
    # 代码里不应出现尚未发布的下一版本号
    for rel in ("runtime", "scripts", "assets"):
        for f in (SKILL / rel).rglob("*.py"):
            assert not re.search(r"\bv2\.21\b", f.read_text(encoding="utf-8")), f


def test_version_at_least():
    assert version_at_least("2.15.8", "2.15.8")
    assert version_at_least("2.16.0", "2.15.8")
    assert not version_at_least("2.15.7", "2.15.8")
    assert not version_at_least(None, "2.15.8")


def test_init_stamps_skill_version():
    assert "skill_version=_skill_version" in _src("scripts/_mvo/cmd_startup.py")


def test_require_confirmation_default_true():
    cfg = yaml.safe_load(_src("assets/global-config.yaml"))
    node = cfg.get("modelevo", cfg)
    assert node["evaluator"]["require_confirmation"] is True


def test_dead_memory_params_removed():
    gc = _src("assets/global-config.yaml")
    assert "embedding_backend" not in gc
    assert "warm_start_ranks" not in _src("scripts/_mvo/cmd_startup.py")
    # v2.18.0:memory 节随旧全局库整体移除(经验库配置在 library 节)
    assert Config.load(skill_config=False).get("memory.retrieve_top_k") is None
    assert "\n  memory:" not in gc


def test_warm_trial_records_binding():
    src = _src("scripts/_mvo/cmd_startup.py")
    assert '_wt.data["experience_bindings"]' in src
    assert '"source_trial_id": _src_tid' in src
    assert 'from .context import _SKILL_ROOT' not in _src("runtime/evolution.py")
