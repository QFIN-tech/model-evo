# -*- coding: utf-8 -*-
"""model-tuning 二级入口: 读 baseline + feature-analysis 产出 → 按规则筛选特征 → 落新 run。

复用 model-training 的产物管线(run_layout / write_*_stage),
新 run 落在 `<output_dir>/new-models/{algo}-feat-v{N}/`,
config.json runtime 含 baseline_run/selection(rules/thresholds/dropped/kept)。

跟 run_tuning 的差异: 不调超参,只缩小特征集; 用 baseline 的 used_params 直接重训。
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts + model-training/scripts

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

PRODUCED_BY = "skills/model-tuning"


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """递归合并 overrides 到 base;同 key 且双方均 dict 则递归,否则 overrides 覆盖。"""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _copy_baseline_yaml_with_overrides(
    layout,
    baseline_run_dir: Path,
    overrides: Dict[str, Any],
    writer,
    produced_by: str,
) -> None:
    """把 baseline 的 config/train_config.yaml 复制到新 run 的 config/,并应用 overrides。

    baseline 无 yaml 时打 warning 跳过。实现同 run_tuning,独立保留以避免跨模块依赖。
    """
    src = baseline_run_dir / "config" / "train_config.yaml"
    if not src.exists():
        print(f"[select_features] baseline 无 train_config.yaml ({src}), 跳过 yaml 复制")
        return

    try:
        import yaml
    except ImportError:
        print("[select_features] PyYAML 未安装, 跳过 yaml 覆写, 直接复制 baseline yaml")
        writer(layout, str(src), produced_by=produced_by)
        return

    data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    _deep_update(data, overrides)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(data, tmp, allow_unicode=True, sort_keys=False)
        tmp_path = tmp.name
    try:
        writer(layout, tmp_path, produced_by=produced_by)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _resolve_output_dir(baseline_run: Path, cli_output_dir: Optional[str]) -> str:
    """决定本次 run 的 output_dir。优先 CLI;否则从 baseline_run.parent.parent 推。"""
    if cli_output_dir:
        return cli_output_dir
    return str(baseline_run.parent.parent)


def _resolve_version(cli_version: Optional[str]) -> Optional[str]:
    """决定本次 run 的 version(形如 v1 / v2 / custom-tag)。

    CLI 传入则用 CLI;否则返回 None,由 RunLayout.create 自动调 next_version 自增。
    suffix 固定为 "-feat",由调用方传给 RunLayout.create。

    非法值(带 algo/suffix 前缀, 如 `xgb-v1` / `feat-v1` / `tuned`)会在此提前拦截,
    避免产出 `xgb-xgb-v1` / `xgb-feat-feat-v1` 等重复前缀目录。
    """
    from stages.layout import validate_version_label
    validate_version_label(cli_version)
    if cli_version:
        return cli_version
    return None


def _confirm(prompt: str, default_yes: bool = True) -> bool:
    """交互式确认;非 TTY 时按 default_yes 处理。"""
    if not sys.stdin.isatty():
        return default_yes
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default_yes
    if not ans:
        return default_yes
    return ans in ("y", "yes")


def _print_selection_summary(snap, result) -> None:
    """打印 baseline 特征数 + 各规则剔除数 + 最终保留数,供用户决策。"""
    print("\n========== baseline ==========")
    print(f"  run        = {snap.run_name}")
    print(f"  algo       = {snap.algo}")
    print(f"  n_features = {len(snap.features)}")

    print("\n========== 规则启用 + 阈值 ==========")
    for rule, enabled in result.rules_enabled.items():
        flag = "ON " if enabled else "off"
        key_map = {"high_psi": "psi", "low_iv": "iv", "high_missing": "missing"}
        thr = result.thresholds.get(key_map.get(rule, ""), None)
        print(f"  [{flag}] {rule:14s} threshold={thr}")

    print("\n========== 剔除明细 ==========")
    for rule, feats in result.dropped_by_rule.items():
        head = feats[:5]
        more = "" if len(feats) <= 5 else f" ...(共 {len(feats)})"
        sample = ", ".join(head) if head else "(无)"
        print(f"  {rule:14s} drop {len(feats):4d}: {sample}{more}")

    n_drop = len(result.dropped_features)
    n_keep = len(result.kept_features)
    print(f"\n  合计剔除 {n_drop} 个; 保留 {n_keep} 个特征。\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """端到端特征筛选入口。"""
    from load_baseline import load as load_baseline
    from selection_rules import select as run_select

    from stages.layout import RunLayout, write_config_snapshot, write_train_config_yaml
    from stages.logs import tee_stdout, process_tee, finalize_logs_stage
    from data_clean import clean_object_features
    from trainer_dispatch import dispatch_train
    from stages.eval_data import assemble as assemble_eval_data
    from stages.features import write_features_stage
    from stages.model import write_model_stage
    from invoke.evaluation import invoke_evaluation_stage
    from invoke.comparison import invoke_comparison_stage
    from stages.predictions import write_predictions_stage
    from stages.explainability import write_explainability_stage
    from stages.run_summary import write_run_summary_stage

    baseline_run = Path(args.baseline_run).resolve()
    snap = load_baseline(str(baseline_run))
    print(f"[select_features] baseline = {snap.run_name} (algo={snap.algo})")

    # 算法直通: 按 baseline.algo 走对应路径 (xgb/dnn/lr), 不强制切到 xgb
    target_algo = snap.algo

    # 应用筛选规则
    result = run_select(
        baseline_features=list(snap.features),
        analysis_dir=args.analysis_dir,
        enable_psi=not args.no_psi,
        enable_iv=not args.no_iv,
        enable_missing=not args.no_missing,
        psi_threshold=args.psi_threshold,
        iv_threshold=args.iv_threshold,
        missing_threshold=args.missing_threshold,
    )

    _print_selection_summary(snap, result)

    if not result.kept_features:
        raise SystemExit("[select_features] 所有特征均被剔除,无法重训;放宽阈值后重试")

    if not args.auto_apply:
        if not _confirm("应用此筛选结果进行重训?", default_yes=True):
            raise SystemExit("[select_features] 用户选择不应用筛选,退出")

    output_dir = _resolve_output_dir(baseline_run, args.output_dir)
    version = _resolve_version(args.version or args.label)
    layout = RunLayout.create(output_dir=output_dir, algo=target_algo, suffix="-feat", version=version)
    print(f"[select_features] run_dir = {layout.run_dir}")

    # 训练参数装配: 用 baseline 的 used_params(不做调参)
    model_cfg = (snap.cfg.get("model") or {})
    target = model_cfg.get("label_col", "label")
    features = list(result.kept_features)
    id_cols = list(model_cfg.get("id_cols") or [])
    final_params = dict(snap.used_params or {})

    # 进程级日志: 从 run_dir 创建到完成回执的全部 stdout/stderr 落 logs/select_features.log
    # (tee_stdout 只覆盖训练核心阶段, process_tee 额外捕获 run_dir print / import 警告 / 完成回执等)
    process_log = layout.logs_dir / "select_features.log"
    with process_tee(process_log):
        with tee_stdout(layout):
            cleaned_train = clean_object_features(snap.train_path, features)
            cleaned_test = clean_object_features(snap.test_path, features)
            cleaned_oot = clean_object_features(snap.oot_path, features)
            common = dict(
                train_path=cleaned_train, test_path=cleaned_test, oot_path=cleaned_oot,
                target=target, features=features,
            )

            predictor, metrics, used_params, train_info = dispatch_train(
                target_algo, common, params_override=final_params,
            )

            # train_info 字段随 algo 不同 (xgb: best_iteration; dnn: best_epoch/total_epochs/
            # early_stopped/best_val_auc; lr: n_iter/converged), 整体 merge 进 runtime 透传
            # metrics 是 Dict[str, BinMetrics] (dataclass), 转 dict 保证 JSON 可序列化
            metrics_payload = {k: v.to_dict() for k, v in metrics.items()} if metrics else {}
            runtime_extra = {
                "version": layout.version,
                "suffix": layout.suffix,
                "n_features": len(features),
                "metrics": metrics_payload,
                "baseline_run": snap.run_name,
                "baseline_metrics": snap.metrics,
                "selection": result.as_dict(),
                "analysis_dir": str(Path(args.analysis_dir).resolve()),
                "final_params": final_params,
            }
            runtime_extra.update(train_info or {})
            write_config_snapshot(
                layout, snap.cfg, snap.data_dir,
                extra=runtime_extra,
                produced_by=PRODUCED_BY,
            )

            # train_config.yaml: 复制 baseline 的入参 yaml(若有), 用筛选后 features 覆写 model.features
            # baseline 无 yaml → 跳过, warning
            _copy_baseline_yaml_with_overrides(
                layout=layout,
                baseline_run_dir=snap.run_dir,
                overrides={"model": {"features": features}},
                writer=write_train_config_yaml,
                produced_by=PRODUCED_BY,
            )

            write_features_stage(
                layout, features=features,
                upstream_source=f"baseline:{snap.run_name} | selection:{args.analysis_dir}",
                dropped=list(result.dropped_features),
                dropped_by_rule=result.dropped_by_rule,
                produced_by=PRODUCED_BY,
            )
            model_path = write_model_stage(
                layout, predictor=predictor,
                used_params=used_params, train_info=train_info,
                produced_by=PRODUCED_BY,
            )
            eval_data = assemble_eval_data(
                predictor=predictor,
                train_path=cleaned_train, test_path=cleaned_test, oot_path=cleaned_oot,
                target=target, features=features,
                metrics=metrics,
            )
            predictions_parquet = write_predictions_stage(
                layout, data=eval_data, id_cols=id_cols, produced_by=PRODUCED_BY,
            )
            invoke_evaluation_stage(
                layout, model_cfg=model_cfg, used_params=used_params,
                produced_by=PRODUCED_BY,
            )
            invoke_comparison_stage(
                layout, baseline_eval_dir=str(snap.run_dir / "evaluation"),
                produced_by=PRODUCED_BY,
            )
            # 会话级横向对比聚合: 扫 new-models/ + model-recommend/, 刷新 model-comparison/
            from invoke.session_aggregate import invoke_session_aggregate
            invoke_session_aggregate(output_dir, produced_by=PRODUCED_BY)
            write_explainability_stage(
                layout, predictor=predictor, data=eval_data, produced_by=PRODUCED_BY,
            )
            finalize_logs_stage(layout, produced_by=PRODUCED_BY, extra_logs=[process_log])
            write_run_summary_stage(layout, produced_by=PRODUCED_BY)

        print(f"[select_features] 完成: run_dir={layout.run_dir}")

    return {
        "run_dir": str(layout.run_dir),
        "model_path": str(model_path),
        "evaluation_report": str(layout.evaluation_dir / f"{layout.run_name}_oot_eval.md"),
        "predictions_parquet": str(predictions_parquet),
        "baseline_run": snap.run_name,
        "version": layout.version,
        "label": layout.version,
        "n_kept": len(result.kept_features),
        "n_dropped": len(result.dropped_features),
    }


def main() -> None:
    """命令行入口。"""
    from selection_rules import (
        DEFAULT_PSI_THRESHOLD, DEFAULT_IV_THRESHOLD, DEFAULT_MISSING_THRESHOLD,
    )

    parser = argparse.ArgumentParser(
        description="model-tuning 特征筛选: 基于 feature-analysis csv 缩小特征集并重训"
    )
    parser.add_argument("--baseline_run", required=True,
                        help="model-training 产出的 baseline run 目录")
    parser.add_argument("--analysis_dir", required=True,
                        help="feature-analysis 输出目录(含 stats.csv/iv_table.csv/psi_table.csv)")
    parser.add_argument("--label", default=None,
                        help="新 run 的版本标识(别名, 形如 v1 / v2);留空则自动自增")
    parser.add_argument("--version", default=None,
                        help="同 --label, 显式版本号; 二者都传时 --version 优先")
    parser.add_argument("--output_dir", default=None,
                        help="输出根目录;默认从 baseline_run 推断(session_dir)")
    parser.add_argument("--auto-apply", dest="auto_apply", action="store_true",
                        help="跳过交互式确认,自动应用筛选结果")
    # 规则开关
    parser.add_argument("--no-psi", action="store_true", help="关闭高 PSI 剔除规则")
    parser.add_argument("--no-iv", action="store_true", help="关闭低 IV 剔除规则")
    parser.add_argument("--no-missing", action="store_true", help="关闭高缺失率剔除规则")
    # 阈值
    parser.add_argument("--psi_threshold", type=float, default=DEFAULT_PSI_THRESHOLD,
                        help=f"PSI 剔除阈值(默认 {DEFAULT_PSI_THRESHOLD})")
    parser.add_argument("--iv_threshold", type=float, default=DEFAULT_IV_THRESHOLD,
                        help=f"IV 最低阈值(默认 {DEFAULT_IV_THRESHOLD})")
    parser.add_argument("--missing_threshold", type=float,
                        default=DEFAULT_MISSING_THRESHOLD,
                        help=f"missing_rate 上限(默认 {DEFAULT_MISSING_THRESHOLD})")
    args = parser.parse_args()

    res = run(args)
    print(f"[select_features] 完成: {res}")


if __name__ == "__main__":
    main()
