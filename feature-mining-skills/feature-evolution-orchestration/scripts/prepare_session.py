# -*- coding: utf-8 -*-
"""prepare_session.py: 进化 session 初始化(幂等五步走)。

用法:
    python prepare_session.py --config <session_dir>/session_config.yaml \
        --session-dir <session_dir> [--force]

五步(每步落产物 + .done 标记, 重跑跳过已完成步, --force 全量重建):
  1. 契约校验   样本快照落 data/sample.parquet
  2. 时序切分   data/splits/{train,test,oot}.parquet + _split_manifest.json
  3. 特征画像   profile/feature-profile.csv + data-desc.md
  4. baseline   固定超参 XGB 训练 + 三档预测/评估
  5. 台账初始化 evolution/state.json + ledger.jsonl(champion=baseline)

退出码: 0 成功 / 2 配置校验失败 / 3 数据契约失败 / 4 运行时失败 / 5 安全红线命中
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

import pandas as pd
from config_io import check_sensitive, load_config

from evo_core import paths, state_io
from evo_prep import baseline_xgb, profiler, sample_contract, temporal_split

PRODUCED_BY = "feature-mining-skills/feature-evolution-orchestration"


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def step1_contract(cfg: dict, session_dir: Path) -> tuple:
    """契约校验 + 样本快照。返回 (df, contract, feature_cols)。"""
    contract = sample_contract.parse_sample_contract(cfg)
    df = sample_contract.load_sample(contract["path"])
    feature_cols = sample_contract.validate_sample_df(df, contract)

    data_dir = paths.data_dir(session_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    snap = paths.sample_parquet(session_dir)
    df.to_parquet(snap, index=False)
    # 元数据文件快照(可选)
    if contract["feature_metadata"]:
        meta_src = Path(contract["feature_metadata"])
        if meta_src.exists():
            shutil.copy2(meta_src, paths.feature_metadata_csv(session_dir))
    return df, contract, feature_cols


def step2_split(df, contract: dict, cfg: dict, session_dir: Path) -> dict:
    """时序切分 + 切分清单落盘。"""
    splits = temporal_split.temporal_split(df, contract["dt_col"], cfg.get("split") or {})
    sdir = paths.splits_dir(session_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    for name, sub in splits.items():
        sub.to_parquet(paths.split_parquet(session_dir, name), index=False)
    manifest_info = temporal_split.split_manifest(splits, contract["dt_col"], contract["label_col"])
    state_io.write_json(
        paths.split_manifest_json(session_dir),
        {"schema_version": 1, "produced_by": PRODUCED_BY, "dt_col": contract["dt_col"], "splits": manifest_info},
    )
    return splits


def step3_profile(df, contract: dict, feature_cols: list, session_dir: Path) -> None:
    """特征画像落盘。"""
    profile_df = profiler.build_feature_profile(df, feature_cols)
    pdir = paths.profile_dir(session_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(paths.feature_profile_csv(session_dir), index=False, encoding="utf-8-sig")
    metadata = profiler.load_feature_metadata(contract.get("feature_metadata"))
    desc = profiler.build_data_desc_md(profile_df, metadata, text_cols=contract.get("text_cols"))
    paths.data_desc_md(session_dir).write_text(desc, encoding="utf-8")
    state_io.write_manifest(
        pdir,
        PRODUCED_BY,
        ["feature-profile.csv", "data-desc.md"],
        overview={"n_features": len(feature_cols), "n_text_cols": len(contract.get("text_cols") or [])},
    )


def step4_baseline(splits: dict, contract: dict, feature_cols: list, cfg: dict, session_dir: Path, mining_name: str) -> dict:
    """baseline 训练评估落盘, 返回各档 eval_bundle。"""
    base_cfg = cfg.get("baseline") or {}
    xgb_params = base_cfg.get("params") or None
    # 冻结 champion 为指定特征子集(饱和场景复现既有强模型口径).
    # 支持 baseline.feature_cols(inline 列表) 或 baseline.feature_cols_file(txt 路径, 每行一个).
    # 未配则 None -> run_baseline 走原逻辑(select_numeric_features(全部候选)).
    baseline_feature_cols = base_cfg.get("feature_cols")
    fc_file = base_cfg.get("feature_cols_file")
    if not baseline_feature_cols and fc_file:
        from pathlib import Path as _P
        p = _P(fc_file)
        if p.exists():
            baseline_feature_cols = [
                ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
    bundles = baseline_xgb.run_baseline(
        session_dir, splits, contract, feature_cols, mining_name,
        xgb_params=xgb_params, baseline_feature_cols=baseline_feature_cols,
    )
    bdir = paths.baseline_dir(session_dir)
    state_io.write_manifest(
        bdir,
        PRODUCED_BY,
        [
            "model/model.json",
            "model/model_meta.json",
            "predictions/train_predictions.parquet",
            "predictions/test_predictions.parquet",
            "predictions/oot_predictions.parquet",
            "evaluation/baseline_train_eval.json",
            "evaluation/baseline_test_eval.json",
            "evaluation/baseline_oot_eval.json",
        ],
        overview={
            "oot_auc": bundles["oot"]["auc"],
            "oot_ks": bundles["oot"]["ks"],
            "test_auc": bundles["test"]["auc"],
        },
    )
    return bundles


def step5_init_state(cfg: dict, session_dir: Path, mining_name: str, bundles: dict) -> None:
    """台账初始化: state.json(champion=baseline) + 空 ledger.jsonl。"""
    snapshot = {k: v for k, v in cfg.items() if not k.startswith("_")}
    state = state_io.init_state(session_dir, mining_name, snapshot)
    state["baseline_oot_auc"] = bundles["oot"]["auc"]
    state["baseline_oot_ks"] = bundles["oot"]["ks"]
    state["champion_oot_auc"] = bundles["oot"]["auc"]
    state["champion_oot_ks"] = bundles["oot"]["ks"]
    state_io.save_state(session_dir, state)
    ledger = paths.ledger_jsonl(session_dir)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.touch(exist_ok=True)
    paths.accepted_dir(session_dir).mkdir(parents=True, exist_ok=True)
    paths.feedback_dir(session_dir).mkdir(parents=True, exist_ok=True)
    paths.rounds_dir(session_dir).mkdir(parents=True, exist_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="进化 session 初始化(契约/切分/画像/baseline/台账)")
    parser.add_argument("--config", required=True, help="session_config.yaml 路径")
    parser.add_argument("--session-dir", required=True, help="session 根目录(runs/{ts}-{mining_name})")
    parser.add_argument("--force", action="store_true", help="全量重建(忽略各步 .done 标记)")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        cfg = load_config(args.config)
    except Exception as e:
        return _exit(2, "配置读取失败: %s" % e)

    mining_name = str(cfg.get("name") or session_dir.name)
    try:
        check_sensitive(mining_name)
    except ValueError as e:
        return _exit(5, str(e))

    try:
        # step 1: 契约(快照是基础, --force 或缺失时重跑)
        if args.force or not state_io.is_done(paths.data_dir(session_dir)):
            print("[step1] 样本契约校验 + 快照 ...")
            df, contract, feature_cols = step1_contract(cfg, session_dir)
            state_io.mark_done(paths.data_dir(session_dir))
        else:
            print("[step1] 已完成, 跳过(--force 可重建)")
            contract = sample_contract.parse_sample_contract(cfg)
            df = sample_contract.load_sample(contract["path"])
            feature_cols = sample_contract.validate_sample_df(df, contract)
    except sample_contract.ContractError as e:
        return _exit(3, "数据契约失败: %s" % e)
    except ValueError as e:
        if "红线" in str(e):
            return _exit(5, str(e))
        return _exit(2, "配置校验失败: %s" % e)
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "step1 运行时失败: %s" % e)

    try:
        if args.force or not state_io.is_done(paths.splits_dir(session_dir)):
            print("[step2] 时序切分 ...")
            splits = step2_split(df, contract, cfg, session_dir)
            state_io.mark_done(paths.splits_dir(session_dir))
            for name, sub in splits.items():
                print("        %s: %d 行, label率 %.4f" % (name, len(sub), sub[contract["label_col"]].mean()))
        else:
            print("[step2] 已完成, 跳过")
            splits = {s: pd.read_parquet(paths.split_parquet(session_dir, s)) for s in paths.SPLITS}
    except sample_contract.ContractError as e:
        return _exit(3, "数据契约失败: %s" % e)
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "step2 运行时失败: %s" % e)

    try:
        if args.force or not state_io.is_done(paths.profile_dir(session_dir)):
            print("[step3] 特征画像 ...")
            step3_profile(df, contract, feature_cols, session_dir)
            state_io.mark_done(paths.profile_dir(session_dir))
        else:
            print("[step3] 已完成, 跳过")
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "step3 运行时失败: %s" % e)

    try:
        if args.force or not state_io.is_done(paths.baseline_dir(session_dir)):
            print("[step4] baseline XGB 训练评估 ...")
            bundles = step4_baseline(splits, contract, feature_cols, cfg, session_dir, mining_name)
            state_io.mark_done(paths.baseline_dir(session_dir))
            print(
                "        baseline: test_auc=%s oot_auc=%s oot_ks=%s"
                % (bundles["test"]["auc"], bundles["oot"]["auc"], bundles["oot"]["ks"])
            )
        else:
            print("[step4] 已完成, 跳过")
            bundles = None  # step5 需要时从 baseline_eval_json 回填
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "step4 运行时失败: %s" % e)

    try:
        if args.force or not paths.state_json(session_dir).exists():
            print("[step5] 台账初始化 ...")
            if bundles is None:
                bundles = {
                    s: state_io.read_json(paths.baseline_eval_json(session_dir, s))["metrics"]
                    for s in paths.SPLITS
                }
            step5_init_state(cfg, session_dir, mining_name, bundles)
        else:
            print("[step5] 已完成, 跳过")
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "step5 运行时失败: %s" % e)

    # ---- step6: Stage 0 可挖性诊断(可选, evolution.run_stage0_probe=true 时跑) ----
    evo_cfg = cfg.get("evolution") or {}
    run_probe = evo_cfg.get("run_stage0_probe", True)
    if run_probe:
        print("[step6] Stage 0 可挖性诊断 ...")
        # probe_roi.py 在 evaluation skill 的 scripts/ 下; _bootstrap 已注入其路径
        # gates.stage0_probe_sample_rows: 大数据量降本(默认 probe_roi 用 5万行)
        gates_cfg = cfg.get("gates") or {}
        probe_rows = gates_cfg.get("stage0_probe_sample_rows")
        rc = _run_probe_roi(args.session_dir, probe_rows=probe_rows)
        if rc == 0:
            print("[step6] Stage 0 诊断完成, 见 profile/roi_report.md")
        else:
            print("[step6] ⚠️ Stage 0 诊断失败(退出码 %d), 进化循环可继续但建议检查" % rc)

    state_io.write_manifest(
        session_dir,
        PRODUCED_BY,
        ["data/", "profile/", "baseline/", "evolution/state.json"],
        overview={"mining_name": mining_name},
    )
    print("prepare_session 完成: %s" % session_dir)
    return 0


def _run_probe_roi(session_dir: str, probe_rows=None) -> int:
    """调用 evaluation skill 的 probe_roi.py 做 Stage 0 可挖性诊断.

    probe_rows: 显式传 --sample-rows 降本(大数据量宽表 Spearman 耗时高).
    """
    import subprocess
    eval_scripts = str(Path(__file__).resolve().parent.parent.parent
                       / "feature-evolution-evaluation" / "scripts")
    cmd = [sys.executable, os.path.join(eval_scripts, "probe_roi.py"),
           "--session-dir", str(session_dir)]
    if probe_rows:
        cmd += ["--sample-rows", str(int(probe_rows))]
    probe = subprocess.run(cmd, capture_output=True, text=True)
    if probe.stdout:
        for ln in probe.stdout.strip().splitlines():
            print("  " + ln)
    if probe.returncode != 0 and probe.stderr:
        for ln in probe.stderr.strip().splitlines()[-5:]:
            print("  " + ln, file=sys.stderr)
    return probe.returncode


if __name__ == "__main__":
    sys.exit(main())
