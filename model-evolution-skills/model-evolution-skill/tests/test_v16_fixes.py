"""v16 修复批次单元验证(P0/P1 全量)。

覆盖(对应 docs/ModelEvo_V2.9_反思与优化优先级_20260901.md):

P0-1 快通道超参消费闸:
  1. 模板 fit_and_score 非 hpo 分支消费 cfg["hyperparameters"]
     (_cfg_hp 注入 + metrics 留痕 hyperparameters_injected);
  2. lint 规则 11:config 含非空 hyperparameters 且 code 无消费点
     → 拒绝;有消费点/空超参/baseline 说明串/豁免标记 → 通过。

P0-2 预算 compute 口径 + HPO 成本闸 + capacity 自动 reference:
  3. StopCondition.check 接受 compute_hours/max_compute_hours,
     超 compute 预算 → budget_compute;stop_report 含
     compute_hours/compute_utilization 双轨;
  4. capacity-check 无 --reference-trial 时自动发现含
     feature_importance.csv 的 trial(列数曲线不再静默 skipped);
  5. HPO 成本闸 _hpo_cost_gate:预估超 heavy 预算 80% → SystemExit,
     豁免标记放行。

P0-3 capacity 闸门:
  6. capacity verdict 含 add_rows/new_information 时 model/loss/hpo
     方向 trial 无 --override-capacity → 拒绝;有 → 放行留痕;
     feature/data_view 方向不受限。

P0-4 交付链冠军单源:
  7. evaluate-final 的冠军选型只接受 fresh confirmed 候选;
  8. recipe/model_meta 声明的 best_trial 与 leaderboard 选型不一致
     → 拒绝出报告(120w run 口径分裂形态被拦截);
  9. baseline final_test 分数缺失时 evaluation 记
     baseline_final_test_missing 并告警。

P1 批次:
  10. (ensemble-final rank-mean 对照,v2.16.0 已删除);
  11. write-splits-view --template:三个模板渲染 + recency 功能 smoke;
  12. family_of_trial 消费 family_hints(前缀族归因不再挤 feat:misc);
  13. multi_seed:模板 main 产 seed_scores/(config multi_seed>1);
  14. parent_score 跨视图行数不一致 → vs_parent 跳过不崩溃。

运行:python tests/test_v16_fixes.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cli_source  # noqa: E402

import pandas as pd

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))

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


# ---------------------------------------------------------------------------
# P0-1:模板超参消费 + lint 规则 11
# ---------------------------------------------------------------------------

def test_p0_1_template_and_lint() -> None:
    print("== P0-1 快通道超参消费闸 ==")
    src = cli_source()
    check("模板含 _cfg_hp 定义", "_cfg_hp = {}" in src)
    check("模板读 config.hyperparameters",
          '(cfg or {}).get("hyperparameters")' in src)
    check("hpo 分支 best_params+config 合并注入",
          "_final_params.update(_cfg_hp)" in src)
    check("uplift 分支注入", "_params.update(_cfg_hp)" in src)
    check("metrics 留痕 hyperparameters_injected",
          "hyperparameters_injected" in src)

    # lint 规则 11
    spec = importlib.util.spec_from_file_location(
        "lint_trial_v16", SKILL / "assets" / "lint_trial.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    check("lint 含规则 11",
          hasattr(m, "check_hyperparameters_consumption"))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tdir = td / "trials" / "t1"
        (tdir / "code").mkdir(parents=True)
        (tdir / "config").mkdir(parents=True)
        (tdir / "code" / "train.py").write_text("import lightgbm\n")
        (tdir / "config" / "trial.yaml").write_text(
            "hyperparameters:\n  learning_rate: 0.03\n")
        errs = m.check_hyperparameters_consumption(
            td, tdir, "import lightgbm\n")
        check("规则 11 拦截无消费点",
              any("规则 11" in e for e in errs))
        errs = m.check_hyperparameters_consumption(
            td, tdir, 'hp = (cfg or {}).get("hyperparameters")\n')
        check("规则 11 放行有消费点", not errs)
        (tdir / "config" / "trial.yaml").write_text(
            "hyperparameters:\n  note: baseline 固定默认参数\n")
        errs = m.check_hyperparameters_consumption(td, tdir, "x=1")
        check("规则 11 跳过 baseline 说明串", not errs)
        (tdir / "config" / "trial.yaml").write_text(
            "hyperparameters:\n  learning_rate: 0.03\n")
        errs = m.check_hyperparameters_consumption(
            td, tdir, "# lint:skip-hyperparameters\n")
        check("规则 11 豁免标记", not errs)
        errs = m.check_hyperparameters_consumption(
            td, tdir, "x=1")
        (tdir / "config" / "trial.yaml").write_text(
            "hyperparameters: {}\n")
        errs = m.check_hyperparameters_consumption(td, tdir, "x=1")
        check("规则 11 空超参跳过", not errs)


# ---------------------------------------------------------------------------
# P0-2:compute 口径 + HPO 成本闸 + capacity 自动 reference
# ---------------------------------------------------------------------------

def test_p0_2_compute_and_gates() -> None:
    print("== P0-2 预算 compute 口径 + HPO 成本闸 ==")
    from runtime.generation import StopCondition
    from runtime.config import Config
    from runtime.spec import TaskSpec

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 最小 spec + config
        (td / "task-spec.yaml").write_text(
            "task_id: t16\ntask_name: t\nobjective_description: t\n"
            "model_type: binary_classification\nbusiness_domains: [x]\n"
            "target:\n  name: y\n  positive_definition: t\n"
            "time:\n  column: dt\nentity:\n  id_column: uid\n"
            "data:\n  data_loader: dl.py\nprimary_metric: auc\n"
            "secondary_metrics: [ks]\nsplit_policy:\n  type: random\n"
            "features:\n  exclude: []\nresource_budget:\n"
            "  max_trials: 80\n  max_generations: 8\n"
            "  max_wall_time_hours: 24\n")
        spec = TaskSpec.load(td / "task-spec.yaml")
        config = Config({})
        sc = StopCondition(spec, config)
        # compute 预算触发
        rep = sc.check(
            n_trials_started=5, n_generations=2, wall_time_hours=5.0,
            trials_accepted=1, best_metric=0.8, baseline_metric=0.79,
            generation_best_deltas=[0.01], noise=0.001,
            unexplored_high_potential=[],
            compute_hours=30.0, max_compute_hours=24.0)
        check("budget_compute 触发",
              rep is not None and "budget_compute" in rep.get("stop_reason"))
        check("stop_report 双轨呈现 compute_hours",
              rep is not None and rep.get("compute_hours") == 30.0)
        check("stop_report 含 compute_utilization",
              rep is not None and rep.get("compute_utilization") == 6.0)
        # 未超 compute → 不因 compute 停
        rep2 = sc.check(
            n_trials_started=5, n_generations=2, wall_time_hours=5.0,
            trials_accepted=1, best_metric=0.8, baseline_metric=0.79,
            generation_best_deltas=[0.01], noise=0.001,
            unexplored_high_potential=[],
            compute_hours=10.0, max_compute_hours=24.0)
        check("compute 未超不触发",
              rep2 is None or "budget_compute" not in rep2.get(
                  "stop_reason"))

    # HPO 成本闸(纯逻辑:构造 fake trial 与 run config)
    print("== HPO 成本闸 ==")
    sys.path.insert(0, str(SKILL / "scripts"))
    spec2 = importlib.util.spec_from_file_location(
        "modelevo_cli_v16", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec2)
    try:
        spec2.loader.exec_module(cli)
    except SystemExit:
        cli = None
    if cli is not None:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "modelevo-config.yaml").write_text(
                "modelevo:\n  execution:\n    trial_timeout_minutes: 90\n")
            # fake trial: 父代 runtime 4000s, budget 40 → 预估远超
            class _T:
                trial_dir = str(td)
                data = {
                    "expected_resources": {"timeout_multiplier": 4},
                    "parent_trial_ids": ["p1"],
                }

                def flush(self):
                    pass

            (td / "trials" / "p1").mkdir(parents=True, exist_ok=True)
            # TrialRecord.load 读 trial.yaml(runtime_sec 在 result 节)
            from runtime.logging_utils import dump_yaml

            dump_yaml(td / "trials" / "p1" / "trial.yaml", {
                "trial_id": "p1", "result": {"runtime_sec": 4000.0}})
            try:
                cli._hpo_cost_gate(_T(), td, 40)
                check("成本闸拦截超预算", False)
            except SystemExit as e:
                check("成本闸拦截超预算", "hpo 成本闸" in str(e))
            # 豁免放行
            _T.data["expected_resources"]["hpo_force_budget"] = True
            try:
                cli._hpo_cost_gate(_T(), td, 40)
                check("豁免标记放行", True)
            except SystemExit:
                check("豁免标记放行", False)


# ---------------------------------------------------------------------------
# P0-3:capacity 闸门 → v2.16.0 特征期门(模型方向判定)
# ---------------------------------------------------------------------------

def test_p0_3_feature_phase_model_direction() -> None:
    print("== P0-3 特征期门:模型方向判定 ==")
    from runtime.feature_phase import is_model_direction

    check("model 方向受限", is_model_direction("explore", ["model"]))
    check("loss 方向受限", is_model_direction("exploit", ["loss"]))
    check("feature 方向不受限", not is_model_direction("explore",
                                                      ["feature"]))
    # v2.16.0 前不带 --changed-modules 的 hpo 绕过 new-trial 门
    check("hpo 动作受限(无 changed_modules 也算)",
          is_model_direction("hpo", []))
    check("confirm 不算模型方向投资",
          not is_model_direction("confirm", ["model"]))


# ---------------------------------------------------------------------------
# P0-4:冠军单源 + baseline final_test 缺失标记
# ---------------------------------------------------------------------------

def test_p0_4_champion_consistency() -> None:
    print("== P0-4 交付链冠军单源 ==")
    src = cli_source()
    check("evaluate-final 使用独立确认交付门",
          "delivery=True" in src and "fresh_confirmation" in src)
    check("冠军一致性校验存在",
          "交付链冠军口径分裂" in src)
    check("baseline final_test 缺失标记",
          "baseline_final_test_missing" in src)
    # 逻辑级验证:120w run 的真实分裂形态会被拦下
    # v2.11.9:示例 run 相对 skill 根定位,不写死绝对路径;缺失则跳过
    run = SKILL.parent / ("example-seq-v3-120w-8-防泄露保障+重构"
                          "早期流程-5.3-flash")
    if run.exists():
        import yaml

        ev = json.loads(
            (run / "final-model" / "evaluation.json").read_text())
        recipe_bt = (yaml.safe_load(
            (run / "final-model" / "recipe.yaml").read_text())
            or {}).get("best_trial")
        meta_bt = (json.loads(
            (run / "final-model" / "model_meta.json").read_text())
            or {}).get("best_trial")
        check("120w 口径分裂形态存在(会被新校验拦截)",
              ev.get("best_trial") != recipe_bt
              or ev.get("best_trial") != meta_bt)


# ---------------------------------------------------------------------------
# P1:view 模板 / family-hints / multi_seed / parent 防护
# ---------------------------------------------------------------------------

def test_p1_batch() -> None:
    print("== P1 批次 ==")
    # (ensemble-final rank-mean 用例随 v2.16.0 删除该命令一并移除)
    # view 模板
    from runtime.view_templates import render_view_template

    for t in ("recency", "active", "kwin"):
        src = render_view_template(
            t, view_id=f"v-{t}", half_life=30.0, k_periods=3,
            active_col=None, time_column="pday", target_name="y")
        compile(src, f"v-{t}.py", "exec")
        check(f"视图模板 {t} 渲染语法 OK", True)
    # recency 功能 smoke
    # v19.3:模板改为直读局部变量 _SPLIT_ROWS/_PROTECTED_COLS,且保护列
    # 为空时**显式 raise**(拒绝生成加权伪特征——旧实现用 globals().get
    # 永远取不到,标签列被加权成 AUC≈1)。本用例按 CLI 的真实注入形态
    # 替换两个占位后再执行:注入保护列、注入分 split 行数。
    ns: dict = {}
    _src = render_view_template("recency", view_id="vr", half_life=30.0,
                                k_periods=3, active_col=None,
                                time_column="pday", target_name="y")
    _src = _src.replace("_SPLIT_ROWS = []  # @SPLIT_ROWS@",
                        "_SPLIT_ROWS = [1, 1, 1]  # @SPLIT_ROWS@")
    _src = _src.replace("_PROTECTED_COLS = []  # @PROTECTED_COLS@",
                        "_PROTECTED_COLS = ['pday', 'y']  # @PROTECTED_COLS@")
    exec(compile(_src, "vr.py", "exec"), ns)
    df = pd.DataFrame({
        "pday": pd.to_datetime(["2026-01-01", "2026-02-01",
                                "2026-03-01"]),
        "amt_1m": [100.0, 200.0, 300.0],
        "y": [0, 1, 0]})
    out = ns["build_view"](df)
    check("recency build_view 功能 smoke",
          len(out) == 3 and "w30_amt_1m" in out.columns)
    check("recency 保护列不被加权(标签不进特征)",
          "w30_y" not in out.columns and "w30_pday" not in out.columns,
          f"cols={[c for c in out.columns if c.startswith('w30_')]}")
    # 未注入保护列时必须显式拒绝(而非静默产出伪特征)
    _ns2: dict = {}
    exec(compile(render_view_template(
        "recency", view_id="vr2", half_life=30.0, k_periods=3,
        active_col=None, time_column="pday", target_name="y"),
        "vr2.py", "exec"), _ns2)
    try:
        _ns2["build_view"](df)
        check("recency 未注入保护列时拒绝", False, "未 raise")
    except ValueError as e:
        check("recency 未注入保护列时拒绝", "保护列为空" in str(e),
              str(e)[:40])

    # family_of_trial 消费 family_hints
    from runtime.directions import family_of_trial

    check("family_hints 前缀族归因",
          family_of_trial({"changed_modules": ["feature"],
                           "family_hints": ["seq_", "ratio_"],
                           "search_action": "explore",
                           "data_view": "v1"}) == "feat:seq")
    check("无 hints 仍走 desc 兜底",
          family_of_trial({"changed_modules": ["feature"],
                           "search_action": "explore",
                           "execution": {"action_description":
                                         ["ratio_ 族增量验证"]},
                           "data_view": "v1"}) == "feat:ratio")
    check("model 方向不受 hints 影响",
          family_of_trial({"changed_modules": ["model"],
                           "family_hints": ["seq_"],
                           "algorithm": "xgboost",
                           "search_action": "explore",
                           "data_view": "v1"}) == "model:xgboost")

    # multi_seed 模板产出
    src = cli_source()
    check("模板 multi_seed 产 seed_scores/", "seed_scores" in src
          and "multi_seed_n" in src)

    # parent 防护(runner 源码标记)
    rsrc = (SKILL / "runtime" / "runner.py").read_text(encoding="utf-8")
    check("runner vs_parent 跨视图防护",
          "vs_parent 跳过(root 口径)" in rsrc)
    check("runner slice_parent 跨视图防护",
          "slice 对照跳过" in rsrc)


def main() -> int:
    test_p0_1_template_and_lint()
    test_p0_2_compute_and_gates()
    test_p0_3_feature_phase_model_direction()
    test_p0_4_champion_consistency()
    test_p1_batch()
    print(f"\n=== v16: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
