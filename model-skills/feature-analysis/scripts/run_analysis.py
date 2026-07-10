# -*- coding: utf-8 -*-
"""feature-analysis 编排入口: 读样本 -> 内部切 train/test/oot -> 跑 3 类分析 -> 渲染产物。

用法:
    python feature-analysis/scripts/run_analysis.py \
        --config <session_dir>/sample-features/feature-analysis/feature_config.yaml \
        --data_path <sample.parquet 路径> \
        --output_dir <session_dir>/sample-features/feature-analysis/analysis

配置文件落 session 内 (从 feature-analysis/config/feature_config.example.yaml 复制),
不落 skill 自身 config/ 目录, 保持 session 自包含。
--data_path 可指向 feature-matching 产出的 sample.parquet, 也可指向用户指定的任意路径。

主交付 report.md;同目录另落:
  - feature-profile.csv: 基础统计语义化合并表(同 stats.csv 内容)
  - feature-quality.csv: IV + PSI 按 feature merge 的单变量质量表
  - report.xlsx:        多 sheet 报告(profile/quality/overview)
  - _manifest.json:     产物清单
保留细分 csv: stats.csv / iv_table.csv / psi_table.csv,
供下游 model-tuning 直接读列。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts

# 注入 feature-matching/scripts 以复用 load_feature_list
_FM_SCRIPTS = Path(__file__).resolve().parent.parent / ".." / "feature-matching" / "scripts"
if str(_FM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_FM_SCRIPTS))

import pandas as pd

from gen_feature_list import load_feature_list  # noqa: E402  feature-matching 公共特征加载
from validate_config import load_config, validate_config, cross_validate_features
from feature_stats import compute_basic_stats
from feature_iv import compute_iv_table, build_woe_table
from feature_psi import compute_psi_table
from render_report import render_report


PRODUCED_BY = "skills/feature-analysis"
MANIFEST_SCHEMA_VERSION = 1


def _validate_label(df: pd.DataFrame, label_col: str) -> None:
    """前置校验 label 列: 非全 NaN / 取值 ⊆ {0,1} / 正样本率 ∈ (0,1)。

    不通过直接 raise ValueError, 避免下游被动接一堆 NaN 排查。NaN 行允许存在,
    但会打 warning(下游 compute_iv_for_feature 会按 isin([0,1]) 过滤)。
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col={label_col!r} 不在样本列中")
    s = df[label_col]
    n = len(s)
    if n == 0:
        raise ValueError(f"label_col={label_col!r}: 样本量为 0")
    n_nan = int(s.isna().sum())
    if n_nan == n:
        raise ValueError(f"label_col={label_col!r}: 全部为 NaN")
    if n_nan > 0:
        print(f"[label-check] {label_col} 含 {n_nan}/{n} NaN, 下游会过滤(IV/AUC 不参与)")

    non_null = s.dropna()
    # 允许 int 与 float 的 0/1; 拒绝其他取值
    uniq = set(non_null.unique().tolist())
    allowed = {0, 1, 0.0, 1.0, True, False}
    bad = uniq - allowed
    if bad:
        raise ValueError(
            f"label_col={label_col!r}: 检测到非 0/1 取值 {sorted(str(v) for v in bad)[:5]},"
            f" 本 skill 仅支持二分类(0/1)标签"
        )

    pos = int((non_null.astype(float) == 1).sum())
    neg = int((non_null.astype(float) == 0).sum())
    if pos == 0:
        raise ValueError(f"label_col={label_col!r}: 正样本数为 0, 无法算 IV/AUC")
    if neg == 0:
        raise ValueError(f"label_col={label_col!r}: 负样本数为 0, 无法算 IV/AUC")
    pos_rate = pos / (pos + neg)
    if pos_rate < 0.001 or pos_rate > 0.999:
        print(
            f"[label-check] {label_col} 正样本率 {pos_rate:.4%} 极端 "
            f"(pos={pos}, neg={neg}), IV/AUC 可能不稳定"
        )


def _range_to_query(time_col: str, rng) -> str:
    """把 [起, 止] 两个 8 位 YYYYMMDD 转成 pandas query 表达式(闭区间)。

    与 classification-model-training 的 _range_to_filter 同口径, 避免跨 skill import。
    pday 可能是 string(YYYYMMDD) 也可能是 int, 统一转 int 比较, 由调用方保证 dt_col 列已转 int。
    """
    return f"{time_col} >= {int(rng[0])} and {time_col} <= {int(rng[1])}"


def _pos_rate(df: pd.DataFrame, label_col: str) -> Optional[float]:
    """算正样本率(无数据时返回 None)。"""
    if len(df) == 0 or label_col not in df.columns:
        return None
    s = df[label_col].dropna()
    if len(s) == 0:
        return None
    return float((s.astype(float) == 1).sum()) / len(s)


def _split_sample_to_three(
    df: pd.DataFrame, split_cfg: dict, dt_col: str, label_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """按 model.split 三档 pday 区间切 df, 返回 (train, test, oot, split_report)。

    用 pandas query 直接切 (与 model_training 的 _range_to_query 同口径),
    不依赖 engines._xgb._dataset.prepare_splits, 避免跨 skill import。

    Args:
        df: 全量样本 (含 train+test+oot 时段)
        split_cfg: cfg["model"]["split"] (已通过 validate_split_ranges 校验)
        dt_col: 时间列名(默认 pday)
        label_col: 标签列名(用于算 pos_rate 写入 split_report)

    Returns:
        (train_df, test_df, oot_df, split_report_dict)
        split_report_dict 字段对齐 model_training 的 SplitReport.to_dict():
          split_strategy / oot_boundary / sample_counts / pos_rates / time_col_used
    """
    train_q = _range_to_query(dt_col, split_cfg["train_range"])
    test_q = _range_to_query(dt_col, split_cfg["test_range"])
    oot_q = _range_to_query(dt_col, split_cfg["oot_range"])

    print(
        f"[feature-analysis] 内部切分: dt_col={dt_col} "
        f"train=[{split_cfg['train_range'][0]},{split_cfg['train_range'][1]}] "
        f"test=[{split_cfg['test_range'][0]},{split_cfg['test_range'][1]}] "
        f"oot=[{split_cfg['oot_range'][0]},{split_cfg['oot_range'][1]}]"
    )

    train_df = df.query(train_q).reset_index(drop=True)
    test_df = df.query(test_q).reset_index(drop=True)
    oot_df = df.query(oot_q).reset_index(drop=True)

    report = {
        "split_strategy": "explicit",
        "oot_boundary": f"{dt_col} >= {split_cfg['oot_range'][0]}",
        "sample_counts": {
            "train": int(len(train_df)),
            "val": int(len(test_df)),
            "oot": int(len(oot_df)),
        },
        "pos_rates": {
            "train": _pos_rate(train_df, label_col),
            "val": _pos_rate(test_df, label_col),
            "oot": _pos_rate(oot_df, label_col),
        },
        "time_col_used": dt_col,
    }
    return train_df, test_df, oot_df, report


def _build_feature_quality(iv_df: pd.DataFrame, psi_df: pd.DataFrame) -> pd.DataFrame:
    """把 IV / PSI 按 feature merge 成单变量质量一站表, 列: feature/iv/auc/n_bins_effective/psi/psi_warn。"""
    iv_cols = [c for c in ("feature", "iv", "auc", "n_bins_effective") if c in iv_df.columns]
    base = iv_df[iv_cols].copy() if iv_cols else pd.DataFrame(columns=["feature"])
    if not psi_df.empty and "feature" in psi_df.columns:
        psi_view = psi_df.rename(columns={"warn": "psi_warn"})
        keep = [c for c in ("feature", "psi", "psi_warn") if c in psi_view.columns]
        merged = base.merge(psi_view[keep], on="feature", how="outer")
    else:
        merged = base.copy()
        merged["psi"] = pd.NA
        merged["psi_warn"] = pd.NA
    if "iv" in merged.columns:
        merged = merged.sort_values("iv", ascending=False, na_position="last").reset_index(drop=True)
    return merged


def _write_excel_report(
    xlsx_path: Path,
    profile_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    overview: dict,
    woe_df: pd.DataFrame = None,
) -> bool:
    """落多 sheet xlsx;缺 openpyxl 时返回 False 不抛错。"""
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            pd.DataFrame([overview]).to_excel(w, sheet_name="overview", index=False)
            profile_df.to_excel(w, sheet_name="feature_profile", index=False)
            quality_df.to_excel(w, sheet_name="feature_quality", index=False)
            if woe_df is not None and not woe_df.empty:
                woe_df.to_excel(w, sheet_name="woe", index=False)
        return True
    except ImportError:
        print("[feature-analysis] openpyxl 未安装, 跳过 report.xlsx")
        return False


def _write_manifest(
    out_dir: Path, files: List[str], overview: dict
) -> None:
    """落 _manifest.json: schema_version / produced_by / files / overview。"""
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "produced_by": PRODUCED_BY,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(files),
        "overview": overview,
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_analysis(
    config_path: str,
    data_path: str,
    output_dir: str,
    feature_list_source: Optional[str] = None,
    cross_validate_csv: Optional[str] = None,
) -> str:
    """端到端跑特征分析, 返回报告路径 (report.md)。

    Args:
        config_path: feature_config.yaml 路径
        data_path: sample.parquet 路径 (通常 feature-matching 产出; 也支持用户指定任意路径)
        output_dir: 报告输出目录
        feature_list_source: 可选, 覆盖 yaml 内 feature_list_source 的特征清单文件
        cross_validate_csv: 可选, feature-matching 产出的 feature-list.csv, 用于交叉校验
    """
    if not data_path:
        raise ValueError("必须传 --data_path 指向 sample.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    cfg = load_config(config_path)
    model = cfg.get("model") or {}
    analysis = cfg.get("analysis") or {}

    # 把 yaml 所在目录透传给 gen_feature_list.load_feature_list,
    # 让相对路径 feature_list_source 能按 yaml 目录解析(与 validate_common 对齐)。
    if cfg.get("_config_dir"):
        os.environ["_CONFIG_DIR"] = cfg["_config_dir"]

    # ---- 特征清单解析: CLI > yaml.feature_list_source > yaml.features ----
    features: List[str] = []
    if feature_list_source or model.get("feature_list_source"):
        source = feature_list_source or model.get("feature_list_source")
        features = load_feature_list(source)
    elif model.get("features"):
        features = list(model["features"])
    else:
        raise ValueError(
            "未指定特征清单。请通过 --feature_list_source 参数、yaml model.feature_list_source "
            "或 yaml model.features 至少指定一种特征来源。"
        )

    # ---- 交叉校验: 用户特征 vs 数据中实际存在的特征 ----
    # feature-list.csv 推断 base 目录用 data_path 同目录
    if not cross_validate_csv:
        infer_base = os.path.dirname(os.path.abspath(data_path))
        inferred = os.path.join(infer_base, "feature-list.csv")
        if os.path.exists(inferred):
            cross_validate_csv = inferred

    if cross_validate_csv:
        valid_features, missing_features = cross_validate_features(features, cross_validate_csv)
        if missing_features:
            print(
                "[cross_validate] %d 个特征不在数据中, 已自动排除: %s"
                % (len(missing_features), ", ".join(missing_features[:20]))
            )
            if len(missing_features) > 20:
                print("  ... 共 %d 个, 仅显示前 20 个" % len(missing_features))
        if not valid_features:
            raise ValueError(
                "交叉校验后无有效特征(全部 %d 个特征均不在数据 feature-list.csv 中)" % len(features)
            )
        print(
            "[cross_validate] %d/%d 特征通过校验(数据中 %d/%d 特征不存在)"
            % (len(valid_features), len(features), len(missing_features), len(features))
        )
        features = valid_features

    # 写回 cfg, 满足 validate_common 对 model.features 非空的校验
    model["features"] = features
    validate_config(cfg)
    label_col = model.get("label_col", "label")
    dt_col = model.get("dt_col", "pday")

    # ---- 数据加载 + 内部切分 ----
    df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    _validate_label(df, label_col)

    # dt_col 统一转 int (pday 可能是 string YYYYMMDD), 否则 query '>=' 会触发 str vs int TypeError
    if dt_col in df.columns and df[dt_col].dtype == object:
        df[dt_col] = df[dt_col].astype(str).str.replace(r"\D", "", regex=True).astype("int64")

    split_cfg = model.get("split")
    if not split_cfg:
        raise ValueError(
            "model.split 未配置: feature-analysis 现在内部切分, "
            "必须配 train_range / test_range / oot_range 三档 pday 区间"
        )
    train_df, test_df, oot_df, split_meta = _split_sample_to_three(
        df, split_cfg, dt_col, label_col
    )

    # 切分产物落 <session_dir>/sample-features/splits/
    # output_dir 形如 <session_dir>/sample-features/feature-analysis/analysis,
    # parent.parent 即 <session_dir>/sample-features, 与 feature-matching/sample.parquet 同级
    splits_dir = Path(output_dir).parent.parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    test_df.to_parquet(splits_dir / "test.parquet", index=False)
    oot_df.to_parquet(splits_dir / "oot.parquet", index=False)

    iv_bins = (analysis.get("iv") or {}).get("n_bins", 10)
    psi_cfg = analysis.get("psi") or {}
    psi_bins = psi_cfg.get("n_bins", 10)
    psi_warn = psi_cfg.get("warn_threshold", 0.10)

    stats_df = compute_basic_stats(df, features)
    iv_df = compute_iv_table(df, features, label_col=label_col, n_bins=iv_bins)
    woe_df = build_woe_table(df, features, label_col=label_col, n_bins=iv_bins)
    if len(oot_df) > 0 and len(train_df) > 0:
        psi_df = compute_psi_table(
            train_df, oot_df, features, n_bins=psi_bins, warn_threshold=psi_warn
        )
    else:
        psi_df = pd.DataFrame(columns=["feature", "psi", "warn"])

    md = render_report(
        cfg=cfg,
        features=features,
        n_total=len(df),
        n_train=len(train_df),
        n_oot=len(oot_df),
        stats_df=stats_df,
        iv_df=iv_df,
        psi_df=psi_df,
        woe_df=woe_df,
        n_test=len(test_df),
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 主交付: report.md
    report_path = out_dir / "report.md"
    report_path.write_text(md, encoding="utf-8")

    # 语义化合并 csv
    profile_df = stats_df.copy()
    profile_df.to_csv(out_dir / "feature-profile.csv", index=False)
    quality_df = _build_feature_quality(iv_df, psi_df)
    quality_df.to_csv(out_dir / "feature-quality.csv", index=False)

    # 细分 csv (下游 model-tuning 消费)
    stats_df.to_csv(out_dir / "stats.csv", index=False)
    iv_df.to_csv(out_dir / "iv_table.csv", index=False)
    psi_df.to_csv(out_dir / "psi_table.csv", index=False)
    woe_df.to_csv(out_dir / "woe_table.csv", index=False)

    # 多 sheet xlsx 报告
    overview = {
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_oot": int(len(oot_df)),
        "n_features": len(features),
        "n_psi_warn": int(psi_df.get("warn", pd.Series(dtype=bool)).sum()) if not psi_df.empty else 0,
        "psi_warn_threshold": psi_warn,
    }
    if split_meta is not None:
        overview["split_strategy"] = split_meta.get("split_strategy")
        overview["oot_boundary"] = split_meta.get("oot_boundary")
        overview["sample_counts"] = split_meta.get("sample_counts")
        overview["pos_rates"] = split_meta.get("pos_rates")
        overview["time_col_used"] = split_meta.get("time_col_used")
    xlsx_path = out_dir / "report.xlsx"
    xlsx_ok = _write_excel_report(xlsx_path, profile_df, quality_df, overview, woe_df=woe_df)

    files = [
        "report.md", "feature-profile.csv", "feature-quality.csv",
        "stats.csv", "iv_table.csv", "psi_table.csv", "woe_table.csv",
    ]
    if xlsx_ok:
        files.append("report.xlsx")
    _write_manifest(out_dir, files, overview)

    return str(report_path)


def main() -> None:
    p = argparse.ArgumentParser(description="feature-analysis: 跑特征分析, 出 markdown + csv + xlsx 报告")
    p.add_argument("--config", required=True, help="feature_config.yaml 路径")
    p.add_argument(
        "--data_path",
        required=True,
        help="sample.parquet 路径 (通常 feature-matching 产出; 也支持用户指定任意路径)",
    )
    p.add_argument("--output_dir", required=True, help="报告输出目录")
    p.add_argument(
        "--feature_list_source",
        default=None,
        help="特征清单文件(.txt 按行 / .csv 取 feature_name 列), 覆盖 yaml feature_list_source",
    )
    p.add_argument(
        "--cross_validate_csv",
        default=None,
        help="feature-matching 产出的 feature-list.csv, 用于交叉校验(不传则自动推断)",
    )
    args = p.parse_args()

    path = run_analysis(
        args.config, args.data_path, args.output_dir,
        feature_list_source=args.feature_list_source,
        cross_validate_csv=args.cross_validate_csv,
    )
    print(f"[feature-analysis] 报告已生成: {path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
