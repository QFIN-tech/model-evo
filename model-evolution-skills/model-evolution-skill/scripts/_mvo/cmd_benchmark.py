"""v2.19.0 证明体系命令:ceiling / benchmark(evaluate | prepare | report)。

- ``ceiling --run <run>``:强参照 + 天花板估计(runtime.reference),结果写
  ``<run>/ceiling/ceiling-estimate.json``;``--out-dir`` 时对 run 只读、一切写入
  落在 out-dir(用于测量在跑/已归档的真实 run);
- ``benchmark --results <yaml>``(兼容旧用法,等价 ``benchmark evaluate``):
  固定预算结果文件 → 发布质量门;
- ``benchmark prepare``:下载套件数据并生成 run 模板(scripts/benchmark/prepare.py);
- ``benchmark report --suite <yaml> --runs-root <dir>``:汇总 S_base/S_mvo/S_ref。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .context import _SKILL_ROOT, _load_spec, _run_dir_from, logger

#: 套件默认路径
DEFAULT_SUITE = Path(_SKILL_ROOT) / "assets" / "benchmark-suite.yaml"


def cmd_ceiling(args) -> int:
    from runtime.reference import ReferenceError_, compute_ceiling

    run_dir = _run_dir_from(args)
    spec = _load_spec(args)
    engines = tuple(e.strip() for e in (args.engines or "strong").split(",")
                    if e.strip())
    # v2.19.0:AutoGluon 预算与框架 run 的实际 compute 时长对齐(等预算参照)
    ag_minutes = args.autogluon_budget_minutes
    if args.match_run_compute:
        from runtime.benchmark import collect_run_scores

        sc = collect_run_scores(run_dir, spec.primary_metric, True)
        hours = float(sc.get("compute_hours") or 0.0)
        if hours > 0:
            ag_minutes = max(float(args.budget_minutes), hours * 60.0)
            print(f"[ceiling] AutoGluon 预算对齐 run compute:{hours:.2f} h → "
                  f"{ag_minutes:.0f} min", flush=True)
    try:
        payload = compute_ceiling(
            run_dir, spec, engines=engines,
            budget_minutes=float(args.budget_minutes),
            out_dir=args.out_dir, max_rows=args.max_rows,
            from_trial=args.from_trial, n_seeds=int(args.n_seeds),
            n_jobs=args.n_jobs, autogluon_python=args.autogluon_python,
            lgbm=not args.no_lightgbm,
            autogluon_budget_minutes=ag_minutes,
            refs_python=args.refs_python,
            holdout=args.holdout,
            log=lambda m: print(m, flush=True))
    except ReferenceError_ as exc:
        logger.error("ceiling 失败: %s", exc)
        print(json.dumps({"status": "error", "error": str(exc)},
                         ensure_ascii=False))
        return 2
    brief = {k: payload.get(k) for k in (
        "primary_metric", "ceiling_estimate", "best_reference",
        "single_model_estimate", "single_model_reference", "ensemble_premium",
        "baseline_trial", "baseline_value", "baseline_gap_to_ref",
        "feature_surface", "path")}
    brief["references"] = {
        k: {kk: v.get(kk) for kk in ("status", "value", "single_seed_value",
                                     "n_seeds", "seconds", "reason")
            if v.get(kk) is not None}
        for k, v in (payload.get("references") or {}).items()}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    return 0


def _prepare_module():
    """scripts/benchmark/prepare.py(按路径加载,不依赖 scripts 是否在包路径上)。"""
    import importlib.util

    p = Path(_SKILL_ROOT) / "scripts" / "benchmark" / "prepare.py"
    spec = importlib.util.spec_from_file_location("mvo_benchmark_prepare", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mvo_benchmark_prepare"] = mod
    spec.loader.exec_module(mod)
    return mod


def cmd_benchmark(args) -> int:
    action = getattr(args, "action", None) or "evaluate"
    if action == "evaluate":
        if not getattr(args, "results", None):
            logger.error("benchmark evaluate 需要 --results <yaml|json>")
            return 2
        from runtime.benchmark import run_benchmark

        report = run_benchmark(Path(args.results),
                               Path(args.output) if args.output else None)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["quality_gate"]["passed"] else 2
    suite_path = Path(getattr(args, "suite", None) or DEFAULT_SUITE)
    if action == "prepare":
        if not args.runs_root or not args.datasets_root:
            logger.error("benchmark prepare 需要 --datasets-root 与 --runs-root")
            return 2
        prep = _prepare_module()
        res = prep.prepare_suite(
            suite_path, datasets_root=args.datasets_root,
            runs_root=args.runs_root, tier=args.tier or "fast",
            only=[x for x in (args.only or "").split(",") if x] or None,
            force=bool(args.force))
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0 if all(r.get("status") != "error" for r in res) else 1
    if action == "report":
        if not args.runs_root:
            logger.error("benchmark report 需要 --runs-root")
            return 2
        from runtime.benchmark import build_suite_report
        from runtime.logging_utils import load_yaml

        suite = load_yaml(suite_path) or {}
        rep = build_suite_report(
            suite, Path(args.runs_root), tier=args.tier,
            out_dir=Path(args.output).parent if args.output else None)
        print(json.dumps({"output_path": rep["output_path"],
                          "summary": rep["summary"],
                          "quality_gate": rep["quality_gate"]},
                         ensure_ascii=False, indent=2))
        return 0
    if action == "aligned":
        if not args.runs_root:
            logger.error("benchmark aligned 需要 --runs-root")
            return 2
        from runtime.benchmark import aligned_holdout_report
        from runtime.logging_utils import load_yaml

        suite = load_yaml(suite_path) or {}
        rep = aligned_holdout_report(
            suite, Path(args.runs_root), split=args.split, tier=args.tier,
            out_dir=Path(args.output).parent if args.output else None,
            extra_ceilings=dict(x.split("=", 1) for x in (args.extra_ceiling or [])),
            extra_participants=dict(x.split("=", 1)
                                    for x in (args.extra_participant or [])))
        print(json.dumps({"output_path": rep["output_path"],
                          "summary": rep["summary"]},
                         ensure_ascii=False, indent=2))
        return 0 if rep["summary"]["n_misaligned"] == 0 else 3
    logger.error("未知 benchmark 动作: %s", action)
    return 2


def add_ceiling_arguments(sp) -> None:
    """``ceiling`` 子命令参数(modelevo.py 只做注册)。"""
    sp.add_argument("--run")
    sp.add_argument("--spec")
    sp.add_argument("--budget-minutes", type=float, default=45.0,
                    help="每个 engine 的预算(分钟,默认 45)")
    sp.add_argument("--engines", default="strong",
                    help="逗号分隔:strong,autogluon,flaml,tabpfn(默认 strong)")
    sp.add_argument("--out-dir", default=None,
                    help="只读模式:结果写 <out-dir>/ceiling/,不写 run 目录")
    sp.add_argument("--max-rows", type=int, default=None,
                    help="train 分层抽样上限(大数据用)")
    sp.add_argument("--from-trial", default=None,
                    help="用该 trial 的特征面(第五族 trial_feature_blocks)")
    sp.add_argument("--n-seeds", type=int, default=5,
                    help="strong engine 的 seed bagging 数(默认 5)")
    sp.add_argument("--n-jobs", type=int, default=None)
    sp.add_argument("--no-lightgbm", action="store_true",
                    help="strong engine 只跑 xgboost")
    sp.add_argument("--autogluon-budget-minutes", type=float, default=None,
                    help="AutoGluon 单独预算(分钟);缺省同 --budget-minutes")
    sp.add_argument("--match-run-compute", action="store_true",
                    help="AutoGluon 预算取 max(--budget-minutes, 本 run 已用 "
                         "trial compute 时长):与框架等预算对比")
    sp.add_argument("--holdout", default=None,
                    choices=("shadow_test",),
                    help="对齐测试面:参照在该面上打分并按样本键落盘(不读标签),"
                         "供 benchmark aligned 统一评分")
    sp.add_argument("--refs-python", default=None,
                    help="FLAML/TabPFN venv 解释器(缺省读 MODELEVO_REFS_PYTHON "
                         "或内置路径)")
    sp.add_argument("--autogluon-python", default=None,
                    help="AutoGluon venv 解释器(缺省读 "
                         "MODELEVO_AUTOGLUON_PYTHON 或内置路径)")
    sp.set_defaults(func=cmd_ceiling)


def add_benchmark_arguments(sp) -> None:
    """``benchmark`` 子命令参数;不带动作 = 旧用法(evaluate --results)。"""
    sp.add_argument("action", nargs="?", default=None,
                    choices=("evaluate", "prepare", "report", "aligned"),
                    help="evaluate(缺省,兼容旧 --results)| prepare | report | "
                         "aligned(对齐测试面统一评分)")
    sp.add_argument("--extra-ceiling", action="append", default=None,
                    help="[aligned] 额外参照 标签=根目录(读 <根>/<数据集>/ceiling/),"
                         "可多次;例如 raw=<dir> 放原始特征面参照")
    sp.add_argument("--split", default="shadow_test", choices=("shadow_test",),
                    help="[aligned] 对齐测试面(缺省 shadow_test;final_test 只经 "
                         "gate 读取一次,不用于参照对比)")
    sp.add_argument("--extra-participant", action="append", default=None,
                    help="[aligned] 额外参与者 标签=<根目录>,可多次;目录布局 "
                         "<根>/<数据集>/shadow_score.parquet(列 score 与 "
                         "__mvo_sample_key)。用于 AIDE / 无框架 Agent 等对照基线")
    sp.add_argument("--results", help="[evaluate] benchmark 结果 YAML/JSON")
    sp.add_argument("--output", help="[evaluate/report] 报告 JSON 路径")
    sp.add_argument("--suite", default=None,
                    help="[prepare/report] 套件 YAML(缺省 assets/benchmark-suite.yaml)")
    sp.add_argument("--runs-root", default=None,
                    help="[prepare/report] run 目录根(<root>/<数据集名>)")
    sp.add_argument("--datasets-root", default=None,
                    help="[prepare] 数据下载目录")
    sp.add_argument("--tier", default=None, choices=("fast", "full", "large"),
                    help="套件档位(prepare 缺省 fast;report 缺省全部)")
    sp.add_argument("--only", default=None, help="[prepare] 逗号分隔数据集名")
    sp.add_argument("--force", action="store_true",
                    help="[prepare] 覆盖已存在的 run 模板")
    sp.set_defaults(func=cmd_benchmark)
