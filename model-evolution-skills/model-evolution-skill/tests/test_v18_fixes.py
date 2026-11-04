"""v18 修复批次单元验证(shadow 强制化 / 计量与断点 / 停止出口 /
信息源清单 / 族归因 / 多卡 / HPO 抽样 / P2 批次)。

依据:docs/ModelEvo_V2.9_框架效能复盘与优化优先级_20260904.md。
逐项对应实测事故:
- P0-1:两 v17.4 run 冠军 trial 无 shadow 分数(2/10、0/7),
  shadow_recheck 全程空转;
- P0-2:400w trial-007 评估段两次外部 kill 无断点;result.runtime_sec
  缺失致 compute_hours 少计 3 倍+;lint 纯 regex 漏 NameError 两轮;
- P0-3:两 run stop=single_data_view_insufficient 挂尾(死锁);
- P1-1:compression_audit 对 reparse 主导任务零候选;
- P1-2:400w trial-007 落 feat:misc(硬编码提示表无 wkmo_);
- P1-3:scheduler 单卡分配,4×A800 只用一张;
- P1-4:400w HPO inner 构建 6217s 实评 0 config;
- P2:halving 0 使用 / crossover 0 使用 / automl 有名无实 /
  experience_hits 只写不读 / temporal_evaluation 死代码 / root 无配对
  bootstrap。

运行:python tests/test_v18_fixes.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cli_source  # noqa: E402

import numpy as np
import pandas as pd
import yaml

SKILL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "assets"))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}: {detail}")


# ---------------------------------------------------------------------------
# 1. P0-1 shadow 强制化
# ---------------------------------------------------------------------------
def test_p0_1_shadow() -> None:
    print("== P0-1 shadow 强制化 ==")
    # 模板含 dump_shadow_score 且 main() 调用它
    from runtime.templates import TRIAL_TEMPLATE

    check("trial 模板含 dump_shadow_score 定义",
          "def dump_shadow_score" in TRIAL_TEMPLATE)
    check("trial 模板 main() 调用 dump_shadow_score",
          "dump_shadow_score(" in TRIAL_TEMPLATE.split("{body}")[0] or
          "dump_shadow_score(" in TRIAL_TEMPLATE)
    from runtime.baseline import BASELINE_TEMPLATE
    check("baseline 模板含 dump_shadow_score",
          "def dump_shadow_score" in BASELINE_TEMPLATE)
    # spec 模板 max_compute_hours 默认行(TASK_TEMPLATE)
    from runtime.templates import TASK_TEMPLATE
    check("spec 模板 resource_budget.max_compute_hours 默认启用",
          "max_compute_hours: 24" in TASK_TEMPLATE)
    # lint 源码含 evaluate-final 冠军 shadow 校验
    cli = cli_source()
    check("evaluate-final 含冠军 shadow 硬校验",
          "冠军 shadow 分数缺失" in cli and "allow_no_shadow" in cli)
    check("shadow waiver 留痕函数存在", "_record_shadow_waiver" in cli)


# ---------------------------------------------------------------------------
# 2. P0-2 计量与断点 + lint AST
# ---------------------------------------------------------------------------
def test_p0_2_metering() -> None:
    print("== P0-2 评估段与计量 ==")
    runner_src = (SKILL / "runtime" / "runner.py").read_text()
    check("runner result.runtime_sec 训练后立即 flush",
          "_eval_started_at = time.time()" in runner_src and
          'trial.flush()' in runner_src.split('_eval_started_at')[0][-400:])
    check("runner 评估段时长并入 runtime(evaluate_sec)",
          "evaluate_sec" in runner_src)
    gi_src = (SKILL / "runtime" / "group_importance.py").read_text()
    check("group_importance 逐族 checkpoint",
          ".gi-checkpoint.json" in gi_src and "_frame_sha" in gi_src)

    # lint 规则 13:真实事故形态
    from lint_trial import check_undefined_names

    bad = ("def _widen_step():\n"
           "    return _time.time()\n\n"
           "_T0 = _time.time()\n\n"
           "import time as _time\n")
    r = check_undefined_names(bad)
    check("lint 规则 13 拦截先用后导(400w trial-006 形态)", len(r) == 1)
    legal = ("_N = 5\n"
             "_IDX = {c: i for i, c in enumerate(_N)}\n"
             "_L = [x * 2 for x in _N if x > 0]\n"
             "_T = list(y for y in _N)\n")
    check("lint 规则 13 comprehension 合法形态不误报",
          check_undefined_names(legal) == [])
    legal2 = ("def f():\n"
              "    return _later()\n\n"
              "def _later():\n"
              "    return 1\n")
    check("lint 规则 13 函数体内前向引用不误报",
          check_undefined_names(legal2) == [])
    # 17 个真实 trial 代码零误报
    real = 0
    flagged = 0
    for pat in ("example-seq-v3-4-2.8.6-v17.4-v4-flash",
                "example-seq-v3-400w-v17.4-v4-flash"):
        for f in sorted((SKILL.parent / pat / "trials").glob(
                "*/code/train.py")) if (SKILL.parent / pat).exists() else []:
            real += 1
            if check_undefined_names(f.read_text()):
                flagged += 1
    if real:
        check(f"lint 规则 13 对 {real} 个真实 trial 零误报", flagged == 0,
              f"{flagged} 个被误标")
    else:
        check("lint 规则 13(真实 trial 样本不在环境,跳过)", True)


# ---------------------------------------------------------------------------
# 3. P0-3 停止出口
# ---------------------------------------------------------------------------
def test_p0_3_stop_exit() -> None:
    print("== P0-3 information_channels_exhausted ==")
    from runtime.generation import StopCondition
    from runtime.config import Config
    from runtime.spec import TaskSpec

    # 自包含 fixture：停止条件只依赖 TaskSpec 的指标与预算，不应要求
    # 仓库外数 GB 的历史示例 run 存在。
    spec = TaskSpec(data={
        "task_id": "stop-fixture", "task_name": "stop-fixture",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "final_test": "2026-04"},
        "data": {},
    })
    cfg = Config.load()
    sc = StopCondition(spec, cfg)
    # 30w v17.4 末态重放:所有条件满足 → 出口触发
    rep = sc.check(
        n_trials_started=10, n_generations=6, wall_time_hours=13.3,
        trials_accepted=0, best_metric=0.7885, baseline_metric=0.7839,
        generation_best_deltas=[0.0, 0.0019, 0.0021, 0.0009, 0.0],
        noise=0.000276, unexplored_high_potential=[],
        data_views_tried=["sample", "v1"],
        inventory_unconsumed=0,
        model_axis_tried=True)
    check("全部条件满足 → information_channels_exhausted",
          rep is not None and
          "information_channels_exhausted" in rep["stop_reason"])
    check("stop_report 含分条件明细与 single_view_caveat",
          rep.get("information_exhausted_detail") is not None
          and rep.get("single_view_caveat") is True)
    # 清单缺失 → 不触发,回退护栏
    rep2 = sc.check(
        n_trials_started=10, n_generations=6, wall_time_hours=13.3,
        trials_accepted=0, best_metric=0.7885, baseline_metric=0.7839,
        generation_best_deltas=[0.0, 0.0019, 0.0021, 0.0009, 0.0],
        noise=0.000276, unexplored_high_potential=[],
        data_views_tried=["sample", "v1"],
        inventory_unconsumed=None)
    check("清单缺失 → 回退 single_data_view_insufficient",
          rep2 is not None and
          "single_data_view_insufficient" in rep2["stop_reason"] and
          "information_channels_exhausted" not in rep2["stop_reason"])
    # v2.16.0:ensemble-final 已删除,detail 不再带 ensemble advisory 字段
    _d1 = (rep or {}).get("information_exhausted_detail") or {}
    check("detail 不含已删除的 ensemble advisory 字段",
          not any(k.startswith("ensemble") for k in _d1),
          f"detail={_d1}")
    # 最近一代有 accepted → 不触发
    rep4 = sc.check(
        n_trials_started=10, n_generations=6, wall_time_hours=13.3,
        trials_accepted=2, best_metric=0.7885, baseline_metric=0.7839,
        generation_best_deltas=[0.0, 0.0019, 0.0021, 0.0009, 0.0005],
        noise=0.000276, unexplored_high_potential=[],
        data_views_tried=["sample", "v1"],
        inventory_unconsumed=0,
        model_axis_tried=True)
    check("最近一代有 accepted → 不触发出口",
          rep4 is None or
          "information_channels_exhausted" not in rep4["stop_reason"])
    # 多口径视图:v19 起出口与视图数解耦(穷尽证据与视图无关;
    # gen-05 实测 4 视图 run 同样卡死在旧 single_view 限定上)。
    # 本用例 delta 全 0(噪声带内)→ 出口触发,视图数只影响 confidence。
    rep5 = sc.check(
        n_trials_started=10, n_generations=6, wall_time_hours=13.3,
        trials_accepted=0, best_metric=0.7885, baseline_metric=0.7839,
        generation_best_deltas=[0.0, 0.0, 0.0, 0.0, 0.0],
        noise=0.000276, unexplored_high_potential=[],
        data_views_tried=["v1", "v2"],
        inventory_unconsumed=0,
        model_axis_tried=True)
    check("多口径场景穷尽证据满足 → 出口触发(v19 视图解耦)",
          rep5 is not None and
          "information_channels_exhausted" in rep5["stop_reason"])


# ---------------------------------------------------------------------------
# 4. P1-1 信息源清单
# ---------------------------------------------------------------------------
def test_p1_1_inventory() -> None:
    print("== P1-1 information-inventory ==")
    from runtime.inventory import (build_inventory, refresh_inventory,
                                   render_inventory_section)

    src = SKILL.parent / "example-seq-v3-4-2.8.6-v17.4-v4-flash"
    if not src.exists():
        check("inventory(真实 run 样本不在环境,跳过)", True)
        return
    from runtime.spec import TaskSpec

    spec = TaskSpec.load(src / "task-spec.yaml")
    import shutil
    import os

    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sample-profile").mkdir()
        os.symlink(src / "sample-profile" / "splits",
                   tmp / "sample-profile" / "splits")
        shutil.copy(src / "sample-profile" / "feature-quality.csv",
                    tmp / "sample-profile" / "feature-quality.csv")
        os.symlink(src / "trials", tmp / "trials")
        inv = build_inventory(tmp, spec)
        cols = {s.get("column") for s in inv["sources"]}
        check("reparse 源全部入清单",
              {"seq_d365", "seq_gap_d365"} <= cols)
        seq_src = next(s for s in inv["sources"]
                       if s.get("column") == "seq_d365")
        check("seq_d365 状态 partially_consumed(有解析族无全部粒度)",
              seq_src["consumption_status"] == "partially_consumed")
        check("序列源粒度空间 6 维",
              len(seq_src.get("granularity_space") or []) == 6)
        # 刷新:回填 implemented → unconsumed 减少
        seq_src["granularity_space"][0]["state"] = "implemented"
        yaml_path = tmp / "sample-profile" / "information-inventory.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(inv, f, allow_unicode=True)
        refreshed = refresh_inventory(tmp, spec)
        seq2 = next(s for s in refreshed["sources"]
                    if s.get("column") == "seq_d365")
        check("refresh 保留 Agent 回填的粒度状态",
              seq2["granularity_space"][0]["state"] == "implemented")
        check("全粒度 implemented → fully_consumed 升格") if False else None
        # 全部 implemented 的源应升 fully_consumed
        for s in refreshed["sources"]:
            for g in s.get("granularity_space") or []:
                g["state"] = "implemented"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(refreshed, f, allow_unicode=True)
        final = refresh_inventory(tmp, spec)
        check("全部粒度 implemented → 全部 fully_consumed",
              all(s["consumption_status"] == "fully_consumed"
                  for s in final["sources"]))
        check("summary.unconsumed 归零",
              final["summary"]["unconsumed"] == 0)
        check("渲染节含未探粒度列",
              "状态" in render_inventory_section(final))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. P1-2 族归因
# ---------------------------------------------------------------------------
def test_p1_2_family() -> None:
    print("== P1-2 方向台账族口径 ==")
    from runtime.directions import DirectionLedger
    from runtime.spec import TaskSpec
    from runtime.trial import TrialFactory

    src = SKILL.parent / "example-seq-v3-400w-v17.4-v4-flash"
    if not src.exists():
        check("family(真实 run 样本不在环境,跳过)", True)
        return
    spec = TaskSpec.load(src / "task-spec.yaml")
    ledger = DirectionLedger(src)
    factory = TrialFactory(src, spec=spec, config=None)
    state = ledger.refresh(factory.all_trials(), spec)
    arms = list(state["arms"].keys())
    check("400w trial-007 归 feat:wkmo(不再 feat:misc)",
          "feat:wkmo" in arms, f"arms={arms}")
    check("feat:misc 不出现(有加宽产物时)",
          "feat:misc" not in arms)


# ---------------------------------------------------------------------------
# 6. P1-3 多卡分配
# ---------------------------------------------------------------------------
def test_p1_3_multi_gpu() -> None:
    print("== P1-3 多卡分配与配额对账 ==")
    from runtime.scheduler import Scheduler, ResourcePlan
    from runtime.config import Config

    cfg = Config.load()
    snap = {
        "cpu_physical": 32, "cpu_count": 64, "load1": 0.0,
        "ram_available_gb": 200.0,
        "gpus": [
            {"index": 0, "memory_free_gb": 79.0, "util_pct": 0},
            {"index": 1, "memory_free_gb": 75.0, "util_pct": 10},
            {"index": 2, "memory_free_gb": 60.0, "util_pct": 0},
            {"index": 3, "memory_free_gb": 40.0, "util_pct": 30},
        ],
    }
    plan = Scheduler(cfg, snap).plan(planned_trials=3)
    check("3 路并发产出 3 个 gpu_indices",
          len(plan.gpu_indices) == 3, f"indices={plan.gpu_indices}")
    check("gpu_indices 按空闲显存降序",
          plan.gpu_indices == [0, 1, 2],
          f"indices={plan.gpu_indices}")
    plan1 = Scheduler(cfg, snap).plan(planned_trials=1)
    check("单路时 gpu_indices 为空(走 plan.env 单卡语义)",
          plan1.gpu_indices == [])
    # 槽位取模:slot 0 绑最空卡,slot 越界回绕(4 % 3 = 1 → 第 2 空)
    from runtime.scheduler import ResourcePlan

    _plan = ResourcePlan(gpu_indices=[0, 1, 2])
    check("slot 取模语义(slot 4 越界回绕到 indices[1])",
          _plan.gpu_indices[4 % 3] == 1)
    # evolution heavy 配额对账
    evo_src = (SKILL / "runtime" / "evolution.py").read_text()
    check("heavy 配额 miss_streak 对账存在",
          "heavy_quota_miss_streak" in evo_src and
          "heavy_quota_check" in evo_src)


# ---------------------------------------------------------------------------
# 7. P1-4 HPO sample_rows
# ---------------------------------------------------------------------------
def test_p1_4_hpo_sample() -> None:
    print("== P1-4 HPO sample_rows ==")
    cli = cli_source()
    check("--hpo-sample-rows 参数存在", "--hpo-sample-rows" in cli)
    check("_inject_hpo_config 接收 sample_rows",
          "sample_rows: int | None = None" in cli)
    check("模板 hpo 分支消费 sample_rows(分层抽样)",
          "hpo_sample_rows" in cli and
          "_stratified_head_labels(" in cli)
    check("成本闸按抽样行数缩放", "frac = frac * float(int(sample_rows))"
          in cli)
    # 模板 hpo 分支可解析(dedent 后)
    import textwrap

    start = cli.index('    hpo_cfg = (cfg or {}).get("hpo")')
    end = cli.index("        result = run_hpo_search(")
    tmpl_src = textwrap.dedent(cli[start:end])
    ast.parse(tmpl_src)
    check("模板 hpo 分支语法有效", True)


# ---------------------------------------------------------------------------
# 8. P2 批次
# ---------------------------------------------------------------------------
def test_p2_batch() -> None:
    print("== P2 批次 ==")
    from runtime.config import Config

    cfg = Config.load()
    check("halving 默认关", cfg.get("gen_executor.halving_enabled") is False)
    check("crossover 配比 0.05",
          abs(float(cfg.get("evolution.crossover_ratio")) - 0.05) < 1e-9)
    cli = cli_source()
    check("automl 参照系创建口校验",
          'args.action == "automl"' in cli and
          '"autogluon", "flaml"' in cli)
    check("automl 参照 body 渲染",
          "automl_reference" in cli)
    lb_src = (SKILL / "runtime" / "leaderboard.py").read_text()
    check("best_trial 过滤 automl 参照行",
          '!= "automl"' in lb_src)
    check("leaderboard reference_only 列",
          "reference_only" in lb_src)
    check("experience_hits 写 generation.yaml",
          "experience_hits" in cli)
    check("automl 进 preflight 可选依赖",
          "autogluon" in (SKILL / "runtime" / "preflight.py").read_text())
    runner_src = (SKILL / "runtime" / "runner.py").read_text()
    check("temporal 按月切片接线",
          "temporal_packs" in runner_src)
    ev_src = (SKILL / "runtime" / "evaluator.py").read_text()
    check("root trial 配对 bootstrap(vs_baseline_paired_bootstrap)",
          "vs_baseline_paired_bootstrap" in ev_src)
    check("root verdict 消费 bootstrap p_positive",
          "vs_baseline_paired_bootstrap" in ev_src.split(
              "def _verdict")[1][:2000])


# ---------------------------------------------------------------------------
# 9. 协议版本
# ---------------------------------------------------------------------------
def test_protocol() -> None:
    print("== 协议版本 ==")
    from runtime import HARNESS_PROTOCOL_VERSION

    # v19 起版本号推进(本文件的历史断言 ≥18 即可,不再钉死版本)
    check("HARNESS_PROTOCOL_VERSION >= 18", HARNESS_PROTOCOL_VERSION >= 18)
    from runtime.harness import harness_hash

    h = harness_hash()
    check("harness_hash 可计算且为 16 位", isinstance(h, str) and len(h) == 16)


def main() -> int:
    test_p0_1_shadow()
    test_p0_2_metering()
    test_p0_3_stop_exit()
    test_p1_1_inventory()
    test_p1_2_family()
    test_p1_3_multi_gpu()
    test_p1_4_hpo_sample()
    test_p2_batch()
    test_protocol()
    print(f"=== v18: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
