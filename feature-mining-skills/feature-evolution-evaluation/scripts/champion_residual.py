# -*- coding: utf-8 -*-
"""champion_residual.py - champion 残差反推"模型缺什么" (数据驱动候选生成依据).

从 champion 预测错的样本(假阴/假阳)出发, 看这些样本在哪些未用特征上取值异常
--这才是"模型还缺什么信号"的数据驱动答案, 比纯看 case-batch 猜交互可靠.

逻辑:
  1. champion 预测全量 train, 取 top 错分样本(假阴: y=1 但 score 低; 假阳: y=0 但 score 高)
  2. 对每个未用特征, 算错分子集 vs 正确子集的分布差异(KS 或均值差标准化)
  3. 差异大的未用特征 = champion 没用上但能区分错分样本的信号 = 候选原料优先

输出: profile/champion_residual.md(Top 差异特征 + 假阴/假阳方向)

成本: 需 baseline 预测(train 全量). 若 baseline 已有预测缓存则复用, 否则在此训一次固定超参 XGB.
大数据量时在 train 子集上算.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evo_core import paths, state_io, xgb_utils
from evo_prep import sample_contract

RESIDUAL_SAMPLE_ROWS = 50000
TOP_N = 30


def _exit(code, msg):
    print(msg, file=sys.stderr)
    return code


def _ks_two_sample(a, b):
    """两样本 KS(错分 vs 正确子集分布差异)."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    from scipy.stats import ks_2samp
    return float(ks_2samp(a, b).statistic)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="champion 残差反推候选原料")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--sample-rows", type=int, default=RESIDUAL_SAMPLE_ROWS)
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        state = state_io.load_state(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    try:
        contract = sample_contract.parse_sample_contract(state.get("config_snapshot") or {})
        cfg = state.get("config_snapshot") or {}
        all_feature_cols = sample_contract.validate_sample_df(
            pd.read_parquet(paths.sample_parquet(session_dir)), contract)
        champ_cols = list(state_io.read_json(paths.baseline_model_dir(session_dir) / "model_meta.json")["feature_cols"])
        champ_cols = champ_cols + list(state.get("champion_fids") or [])
        label_col = contract["label_col"]

        train = pd.read_parquet(paths.split_parquet(session_dir, "train"))
        n = len(train)
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n, size=min(n, args.sample_rows), replace=False)) if n > args.sample_rows else np.arange(n)
        train_s = train.iloc[idx].reset_index(drop=True)
        y = train_s[label_col].to_numpy().astype(int)

        # champion 预测(复用 baseline 预测缓存或重训一次)
        pred_path = paths.baseline_dir(session_dir) / "predictions" / "train_predictions.parquet"
        if pred_path.exists():
            # 按行号对齐(缓存是全量, 取 idx)
            full_pred = pd.read_parquet(pred_path)["score"].to_numpy()
            scores = full_pred[idx]
        else:
            Xc = xgb_utils.as_numeric_frame(train_s, champ_cols)
            model = xgb_utils.train_xgb(Xc, y)
            scores = xgb_utils.predict_scores(model, Xc)

        # 错分掩码: 假阴(y=1 score 低) / 假阳(y=0 score 高)
        # 阈值: 取 score 中位数附近, 假阴 = y=1 且 score<median, 假阳 = y=0 且 score>median
        med = np.median(scores)
        fn_mask = (y == 1) & (scores < med)   # 假阴
        fp_mask = (y == 0) & (scores > med)   # 假阳
        correct_mask = ~fn_mask & ~fp_mask
        print("[residual] 样本 %d, 假阴 %d, 假阳 %d, 正确 %d" % (len(y), fn_mask.sum(), fp_mask.sum(), correct_mask.sum()))

        unused = [c for c in all_feature_cols if c not in set(champ_cols)]
        results = []
        for i, feat in enumerate(unused):
            try:
                col = pd.to_numeric(train_s[feat], errors="coerce").to_numpy().astype(float)
            except Exception:
                continue
            ks_fn = _ks_two_sample(col[fn_mask], col[correct_mask])     # 假阴 vs 正确
            ks_fp = _ks_two_sample(col[fp_mask], col[correct_mask])      # 假阳 vs 正确
            if np.isnan(ks_fn) and np.isnan(ks_fp):
                continue
            results.append({"feat": feat, "ks_fn": ks_fn, "ks_fp": ks_fp,
                             "max_ks": max(np.nan_to_num(ks_fn), np.nan_to_num(ks_fp))})
            if (i + 1) % 200 == 0:
                print("  残差扫描 %d/%d" % (i + 1, len(unused)), flush=True)

        results.sort(key=lambda r: -r["max_ks"])
        # 方向: 错分子集均值 vs 正确子集均值
        lines = ["# champion 残差反推 — 候选原料建议\n",
                 "> champion 预测错的样本(假阴/假阳)在哪些未用特征上分布异常 = 模型缺的信号.\n",
                 "## Top %d 差异未用特征(候选原料优先)\n" % TOP_N,
                 "| feature | 假阴KS | 假阳KS | 方向(错分均值-正确均值) |", "|---|---|---|---|"]
        for r in results[:TOP_N]:
            feat = r["feat"]
            try:
                col = pd.to_numeric(train_s[feat], errors="coerce").to_numpy().astype(float)
                fn_mean = np.nanmean(col[fn_mask]); cor_mean = np.nanmean(col[correct_mask])
                fp_mean = np.nanmean(col[fp_mask])
                diff_fn = fn_mean - cor_mean; diff_fp = fp_mean - cor_mean
                direction = "假阴%s%.2f 假阳%s%.2f" % ("↑" if diff_fn > 0 else "↓", abs(diff_fn),
                                                      "↑" if diff_fp > 0 else "↓", abs(diff_fp))
            except Exception:
                direction = "-"
            lines.append("| %s | %.4f | %.4f | %s |" % (feat, r["ks_fn"], r["ks_fp"], direction))
        lines += ["",
                  "## 候选设计建议",
                  "- 优先用假阴/假阳 KS 大的特征作交互原料(这些信号 champion 没用上但能区分错分样本)",
                  "- 方向↑=错分子集该特征均值更高, ↓=更低; 据此设计交互(如: 假阴↑特征 × 假阳↓特征 的差分)",
                  "- 这些特征已通过'残差异常'筛选, 大概率低冗余(champion 没用上区分错分的信号)"]

        pdir = paths.profile_dir(session_dir)
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "champion_residual.md").write_text("\n".join(lines), encoding="utf-8")
        # 也落个 txt 供候选生成引用
        (pdir / "residual_materials.txt").write_text(
            "\n".join(r["feat"] for r in results[:TOP_N * 2]), encoding="utf-8")
        print("[residual] 写 profile/champion_residual.md + residual_materials.txt")
        return 0
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "残差反推失败: %s" % e)


if __name__ == "__main__":
    sys.exit(main())
