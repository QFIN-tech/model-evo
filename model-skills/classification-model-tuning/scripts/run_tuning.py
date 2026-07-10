# -*- coding: utf-8 -*-
"""model-tuning 编排器: baseline run → 规则诊断 / Optuna 搜索 → 落新 run。

复用 model-training 的产物管线(run_layout / write_*_stage),
新 run 落在 `<output_dir>/new-models/{algo}-tuned-v{N}/`,
config.json runtime 含 baseline_run/diagnosis/method/recommended_params/baseline_metrics。
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts + model-training/scripts

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

PRODUCED_BY = "skills/model-tuning"


def _copy_baseline_yaml_with_overrides(
    layout,
    baseline_run_dir: Path,
    overrides: Dict[str, Any],
    writer,
    produced_by: str,
) -> None:
    """把 baseline 的 config/train_config.yaml 复制到新 run 的 config/,并应用 overrides。

    机制: 读 baseline yaml → 用 overrides 递归合并 → 写临时 yaml → 调 writer 复制到新 run。
    baseline 无 yaml 时打 warning 跳过(支持未生成 yaml 的 baseline)。

    Args:
        layout: 新 run 的 RunLayout
        baseline_run_dir: baseline run 目录(读 config/train_config.yaml)
        overrides: 递归合并到 yaml 顶层 dict 的覆盖项(如 {"model": {"params": {...}}})
        writer: write_train_config_yaml 函数(传 layout + 临时 yaml 路径)
        produced_by: 来源标识
    """
    src = baseline_run_dir / "config" / "train_config.yaml"
    if not src.exists():
        print(f"[run_tuning] baseline 无 train_config.yaml ({src}), 跳过 yaml 复制")
        return

    try:
        import yaml
    except ImportError:
        print("[run_tuning] PyYAML 未安装, 跳过 yaml 覆写, 直接复制 baseline yaml")
        writer(layout, str(src), produced_by=produced_by)
        return

    data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    _deep_update(data, overrides)

    # 写临时 yaml 再调 writer, 复用 writer 的 manifest 落盘逻辑
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as tmp:
        yaml.safe_dump(data, tmp, allow_unicode=True, sort_keys=False)
        tmp_path = tmp.name
    try:
        writer(layout, tmp_path, produced_by=produced_by)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """递归合并 overrides 到 base;同 key 且双方均 dict 则递归,否则 overrides 覆盖。"""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v



def _resolve_output_dir(baseline_run: Path, cli_output_dir: Optional[str]) -> str:
    """决定本次 tuning run 的 output_dir。

    优先级: CLI --output_dir > 从 baseline_run 推断(baseline_run.parent.parent = session_dir)。
    baseline 目录结构: <session>/new-models/<run_name>/,parent.parent 即 session。
    """
    if cli_output_dir:
        return cli_output_dir
    # baseline_run 内部约定: <output_dir>/new-models/<run_name>
    inferred = baseline_run.parent.parent
    return str(inferred)


def _resolve_version(cli_version: Optional[str]) -> Optional[str]:
    """决定本次 run 的 version(形如 v1 / v2 / custom-tag)。

    CLI 传入则用 CLI;否则返回 None,由 RunLayout.create 自动调 next_version 自增。
    suffix 固定为 "-tuned",由调用方传给 RunLayout.create。

    非法值(带 algo/suffix 前缀, 如 `xgb-v1` / `tuned-v1` / `feat`)会在此提前拦截,
    避免产出 `xgb-xgb-v1` / `xgb-tuned-tuned-v1` 等重复前缀目录。
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


def _print_diagnosis_summary(snap, diagnosis, recommended_params: Dict[str, Any]) -> None:
    """打印 baseline 指标 + 诊断 + 推荐参数差异,供用户决策。"""
    print("\n========== baseline 指标 ==========")
    for split, m in snap.metrics.items():
        auc = m.get("auc"); ks = m.get("ks")
        print(f"  {split:5s}: AUC={auc:.4f} KS={ks:.4f}"
              if auc is not None and ks is not None
              else f"  {split:5s}: {m}")
    if snap.new_psi is not None:
        flag = "  [PSI_WARN]" if snap.new_psi > 0.10 else ""
        print(f"  PSI(train→oot) = {snap.new_psi:.4f}{flag}")
    if snap.best_iteration is not None:
        # algo-aware: xgb=n_estimators / dnn=epochs / lr=max_iter
        # train_info 含 algo 专属字段 (dnn: total_epochs/early_stopped/best_val_auc;
        # lr: converged), 顺手多打一点上下文
        ti = snap.train_info or {}
        if snap.algo == "dnn":
            total = ti.get("total_epochs") or snap.used_params.get("epochs")
            stopped = ti.get("early_stopped")
            best_val_auc = ti.get("best_val_auc")
            print(f"  best_epoch = {snap.best_iteration} / total_epochs={total}"
                  f" / early_stopped={stopped}"
                  + (f" / best_val_auc={best_val_auc:.4f}" if best_val_auc is not None else ""))
        elif snap.algo == "lr":
            converged = ti.get("converged")
            max_iter = snap.used_params.get("max_iter")
            print(f"  n_iter = {snap.best_iteration} / max_iter={max_iter}"
                  f" / converged={converged}")
        else:
            total = snap.used_params.get("n_estimators")
            print(f"  best_iteration = {snap.best_iteration} / n_estimators={total}")

    print("\n========== 诊断 ==========")
    print(f"  status = {diagnosis.status}")
    for r in diagnosis.reasons:
        print(f"  - {r}")

    print("\n========== 推荐参数(diff) ==========")
    diffs = []
    for k in sorted(set(snap.used_params) | set(recommended_params)):
        b = snap.used_params.get(k); r = recommended_params.get(k)
        if b != r:
            diffs.append((k, b, r))
    if not diffs:
        print("  (推荐参数与 baseline 完全一致,无需调优)")
    else:
        for k, b, r in diffs:
            print(f"  {k}: {b} -> {r}")
    print()


def _default_params_for_algo(algo: str) -> Dict[str, Any]:
    """兜底: baseline 没有 used_params 时返回该 algo 的默认超参。

    正常路径不会触发(baseline 的 used_params 一定存在); 仅作防御性兜底。
    """
    algo = (algo or "xgb").lower()
    if algo == "dnn":
        from trainers.train_dnn import DNN_PARAMS
        return dict(DNN_PARAMS)
    if algo == "lr":
        from trainers.train_lr import LR_PARAMS
        return dict(LR_PARAMS)
    from trainers.tune_train import TUNED_PARAMS
    return dict(TUNED_PARAMS)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """端到端调优入口。

    Args:
        args: argparse 解析结果

    Returns:
        摘要 dict: {run_dir, model_path, evaluation_report, predictions_parquet, baseline_run, status}
    """
    # 延迟 import 加速 --help
    from load_baseline import load as load_baseline
    from diagnose import diagnose
    from recommend_params import recommend

    from stages.layout import RunLayout, write_config_snapshot, write_train_config_yaml
    from stages.logs import tee_stdout, process_tee, finalize_logs_stage
    from data_clean import clean_object_features
    from trainers.tune_train import train_with_params
    from trainers.train_dnn import train_dnn_model
    from trainers.train_lr import train_lr_model
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
    print(f"[run_tuning] baseline = {snap.run_name} (algo={snap.algo})")

    # 算法直通: 按 baseline.algo 走对应路径 (xgb/dnn/lr), 不强制切到 xgb
    target_algo = snap.algo
    if snap.used_params:
        baseline_params = dict(snap.used_params)
    else:
        # baseline 没有 used_params (理论上不会发生): 退到该 algo 的默认 _PARAMS
        baseline_params = _default_params_for_algo(target_algo)

    # 诊断 (algo-aware: underconverged 规则按 algo 分流)
    diagnosis = diagnose(
        metrics=snap.metrics, used_params=baseline_params,
        best_iteration=snap.best_iteration, new_psi=snap.new_psi,
        algo=target_algo,
    )

    # 默认推荐参数(规则);若走 optuna 后续会覆盖
    recommended = recommend(baseline_params, diagnosis, algo=target_algo)

    _print_diagnosis_summary(snap, diagnosis, recommended)

    # well_fit 时若无 --auto-apply 直接退出
    if diagnosis.status == "well_fit" and not args.auto_apply:
        if not _confirm("baseline 指标已合理,仍要继续调优重训吗?", default_yes=False):
            print("[run_tuning] 用户选择退出(指标合理)")
            return {"baseline_run": str(baseline_run), "status": "skipped_well_fit"}

    if not args.auto_apply:
        if not _confirm("应用此推荐进行调优重训?", default_yes=True):
            raise SystemExit("[run_tuning] 用户选择不应用推荐,退出")

    # 决议 output_dir / version
    output_dir = _resolve_output_dir(baseline_run, args.output_dir)
    version = _resolve_version(args.version or args.label)

    # 建本次 run 的 layout
    layout = RunLayout.create(output_dir=output_dir, algo=target_algo, suffix="-tuned", version=version)
    print(f"[run_tuning] run_dir = {layout.run_dir}")

    # 训练参数装配(复用 baseline 的 cfg.model)
    model_cfg = (snap.cfg.get("model") or {})
    target = model_cfg.get("label_col", "label")
    features = list(snap.features)
    id_cols = list(model_cfg.get("id_cols") or [])

    # 进程级日志: 从 run_dir 创建到完成回执的全部 stdout/stderr 落 logs/run_tuning.log
    process_log = layout.logs_dir / "run_tuning.log"
    with process_tee(process_log):
        with tee_stdout(layout):
            cleaned_train = clean_object_features(snap.train_path, features)
            cleaned_test = clean_object_features(snap.test_path, features)
            cleaned_oot = clean_object_features(snap.oot_path, features)
            common = dict(
                train_path=cleaned_train, test_path=cleaned_test, oot_path=cleaned_oot,
                target=target, features=features,
            )

            trials_log = None
            if args.method == "optuna":
                from search_optuna import search as optuna_search
                # 按 algo dispatch train_fn: xgb 用 train_with_params 原样;
                # dnn/lr 返回 (predictor, metrics, info), 用 lambda 丢 info
                if target_algo == "dnn":
                    _train_fn = lambda params, **kw: train_dnn_model(params=params, **kw)[:2]
                elif target_algo == "lr":
                    _train_fn = lambda params, **kw: train_lr_model(params=params, **kw)[:2]
                else:
                    _train_fn = train_with_params
                print(f"[run_tuning] 启动 Optuna 搜索 (algo={target_algo}, "
                      f"n_trials={args.n_trials}, ratio=±30% baseline)")
                best_params, trials_log = optuna_search(
                    baseline_params=baseline_params,
                    train_fn=_train_fn, train_common=common,
                    n_trials=args.n_trials,
                    algo=target_algo,
                )
                print(f"[run_tuning] Optuna best_params = {best_params}")
                final_params = best_params
            else:
                final_params = recommended
                print(f"[run_tuning] 规则推荐参数 = {final_params}")

            # 用 final_params 跑正式训练
            predictor, metrics, used_params, train_info = dispatch_train(
                target_algo, common, params_override=final_params,
            )

            # config.json 快照(含 baseline 关联)
            # metrics 是 Dict[str, BinMetrics] (dataclass), 转 dict 保证 JSON 可序列化
            # train_info 字段随 algo 不同 (xgb: best_iteration; dnn: best_epoch/total_epochs/
            # early_stopped/best_val_auc; lr: n_iter/converged), 整体 merge 进 runtime 透传
            metrics_payload = {k: v.to_dict() for k, v in metrics.items()} if metrics else {}
            runtime_extra = {
                "version": layout.version,
                "suffix": layout.suffix,
                "n_features": len(features),
                "metrics": metrics_payload,
                "baseline_run": snap.run_name,
                "baseline_metrics": snap.metrics,
                "diagnosis": diagnosis.as_dict(),
                "method": args.method,
                "recommended_params": recommended,
                "final_params": final_params,
                "trials_log": trials_log,
            }
            runtime_extra.update(train_info or {})
            write_config_snapshot(
                layout, snap.cfg, snap.data_dir,
                extra=runtime_extra,
                produced_by=PRODUCED_BY,
            )

            # train_config.yaml: 复制 baseline 的入参 yaml(若有),用 final_params 覆写 model.params
            # baseline 无 yaml 时跳过, warning
            _copy_baseline_yaml_with_overrides(
                layout=layout,
                baseline_run_dir=snap.run_dir,
                overrides={"model": {"params": final_params}},
                produced_by=PRODUCED_BY,
                writer=write_train_config_yaml,
            )

            write_features_stage(
                layout, features=features,
                upstream_source=f"baseline:{snap.run_name}",
                dropped=[], produced_by=PRODUCED_BY,
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

        print(f"[run_tuning] 完成: run_dir={layout.run_dir}")

    return {
        "run_dir": str(layout.run_dir),
        "model_path": str(model_path),
        "evaluation_report": str(layout.evaluation_dir / f"{layout.run_name}_oot_eval.md"),
        "predictions_parquet": str(predictions_parquet),
        "baseline_run": snap.run_name,
        "status": diagnosis.status,
        "version": layout.version,
        "label": layout.version,
    }


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="model-tuning 调优编排")
    parser.add_argument("--baseline_run", required=True,
                        help="model-training 产出的 baseline run 目录")
    parser.add_argument("--method", choices=("rule", "optuna"), default="rule",
                        help="调优方法: rule(规则推荐)|optuna(贝叶斯搜索)")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Optuna 搜索次数(仅 --method optuna)")
    parser.add_argument("--label", default=None,
                        help="新 run 的版本标识(别名, 形如 v1 / v2);留空则自动自增")
    parser.add_argument("--version", default=None,
                        help="同 --label, 显式版本号; 二者都传时 --version 优先")
    parser.add_argument("--output_dir", default=None,
                        help="输出根目录;默认从 baseline_run 推断(session_dir)")
    parser.add_argument("--auto-apply", dest="auto_apply", action="store_true",
                        help="跳过交互式确认,自动应用推荐参数")
    args = parser.parse_args()

    res = run(args)
    print(f"[run_tuning] 完成: {res}")


if __name__ == "__main__":
    main()
