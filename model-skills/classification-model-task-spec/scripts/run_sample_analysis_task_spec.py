# -*- coding: utf-8 -*-
"""样本分析 + 切分一体化脚本。

输入: feature-matching 拉的全量样本 parquet (含 id_col / label_col / time_col 三列及可选补充字段),
      + 显式 Train/Test/OOT 三档 pday 区间。id_col 默认 user_no, 可由 --id-cols override
      (分析侧取单列作主 ID, 与 fetch 侧 --id-cols 多列命名对齐)。
输出: report.md / report.xlsx / _manifest.json / _split_manifest.json /
      train.parquet / test.parquet / oot.parquet, 全部落在 --output-dir。

切分逻辑从 feature-matching/scripts/split_sample.py 复制核心函数, 独立可执行, 不依赖外部 skill。

用法:
    python run_sample_analysis_task_spec.py \
        --sample .../sample.parquet \
        --train-range 20260401,20260510 \
        --test-range  20260511,20260520 \
        --oot-range   20260521,20260530 \
        --model-name call_complaint \
        --timestamp 20260629-161231 \
        --output-dir .../data-profile/ \
        [--dt-col pday] [--label-col label] [--id-cols user_no] [--sample-table your_db.xxx]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


# ---------- 区间解析与切分 (复制自 split_sample.py) ----------

def parse_range(text: str) -> Tuple[str, str]:
    """解析 '起,止' 闭区间 (YYYYMMDD)。"""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("区间须为两元素 起,止, 如 20260312,20260430, 当前 %r" % text)
    start, end = parts
    for d in (start, end):
        if not (d.isdigit() and len(d) == 8):
            raise ValueError("区间日期须为 8 位 YYYYMMDD, 当前 %r" % d)
    if start > end:
        raise ValueError("区间起始 %s 不应大于结束 %s" % (start, end))
    return start, end


def validate_ranges(
    train: Tuple[str, str], test: Tuple[str, str], oot: Tuple[str, str]
) -> None:
    """校验三档区间时序递增, 允许相邻, 重叠/逆序报错。"""
    ordered = [("train", train), ("test", test), ("oot", oot)]
    for (n1, r1), (n2, r2) in zip(ordered, ordered[1:]):
        if r1[1] >= r2[0]:
            raise ValueError(
                "%s 区间 [%s,%s] 与 %s 区间 [%s,%s] 重叠或逆序, 要求 train<test<oot(允许相邻)"
                % (n1, r1[0], r1[1], n2, r2[0], r2[1])
            )


def classify_by_ranges(pday, ranges: Dict[str, Tuple[str, str]]):
    """按显式 pday 区间把单个 pday 归到 train/test/oot, 区间外返回 None。"""
    p = str(pday)
    for name in ("train", "test", "oot"):
        start, end = ranges[name]
        if start <= p <= end:
            return name
    return None


def split_dataframe_by_ranges(df, time_col: str, ranges: Dict[str, Tuple[str, str]]):
    """按显式 pday 区间把 DataFrame 切成 train/test/oot 三份, 区间外行计入 dropped。"""
    if time_col not in df.columns:
        raise ValueError("时间列 %r 不在样本列中, 无法按时间切分" % time_col)
    split_series = df[time_col].map(lambda p: classify_by_ranges(p, ranges))
    train_df = df[split_series == "train"]
    test_df = df[split_series == "test"]
    oot_df = df[split_series == "oot"]
    dropped = int(split_series.isna().sum())
    return train_df, test_df, oot_df, dropped


# ---------- 输入校验 ----------

def validate_input(df, args) -> None:
    """校验列存在 / label 取值 / pday 8 位数字。"""
    for col in (args.dt_col, args.label_col, args._id_col_primary):
        if col not in df.columns:
            raise SystemExit("样本缺必要列: %s" % col)

    bad_labels = sorted(set(df[args.label_col].dropna().unique()) - {0, 1})
    if bad_labels:
        raise SystemExit("label 列存在非法取值 %s, 仅允许 0/1" % bad_labels)

    pday_str = df[args.dt_col].astype(str)
    bad_pday = sorted(set(pday_str[pday_str.str.len() != 8]) - {"nan"})
    if bad_pday:
        raise SystemExit("pday 列存在非 8 位 YYYYMMDD 值: %s" % bad_pday[:5])

    ranges = {
        "train": parse_range(args.train_range),
        "test": parse_range(args.test_range),
        "oot": parse_range(args.oot_range),
    }
    validate_ranges(ranges["train"], ranges["test"], ranges["oot"])


# ---------- 统计计算 ----------

def _fmt_ratio(pos: int, neg: int) -> str:
    if neg == 0:
        return "1 : 0" if pos > 0 else "NA"
    g = _gcd(pos, neg)
    return "1 : %.2f" % (neg / max(pos, 1))


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def compute_overall(df, args) -> dict:
    """全量样本统计。"""
    total = int(len(df))
    label_col = args.label_col
    time_col = args.dt_col
    id_col = args._id_col_primary

    valid = df[label_col].isin([0, 1])
    nan_label = int((~valid).sum())
    if nan_label:
        print("[run_sample_analysis_task_spec] [警告] %d 行 label 为 NaN 或非法, 已从统计中剔除" % nan_label, file=sys.stderr)
    pos = int((df.loc[valid, label_col] == 1).sum())
    neg = int((df.loc[valid, label_col] == 0).sum())
    rate = round(pos / total, 6) if total else 0.0

    pday_vals = sorted(df[time_col].astype(str).unique().tolist())
    pday_min = pday_vals[0] if pday_vals else None
    pday_max = pday_vals[-1] if pday_vals else None

    overall = {
        "total_samples": total,
        "positive_samples": pos,
        "negative_samples": neg,
        "positive_rate": rate,
        "positive_negative_ratio": _fmt_ratio(pos, neg),
        "pday_range": [pday_min, pday_max],
        "pday_unique_count": len(pday_vals),
        "pday_values": pday_vals,
        "null_counts": {c: int(df[c].isna().sum()) for c in [id_col, label_col, time_col]},
        "user_unique": int(df[id_col].nunique()),
        "dup_user_pday": int(df.duplicated(subset=[id_col, time_col]).sum()),
    }

    return overall


def segment_by_time(df, args) -> List[dict]:
    """按时间分段统计。pday 唯一值 <= 10 时按 pday 切; 否则按周/月聚合。"""
    time_col = args.dt_col
    label_col = args.label_col
    pday_vals = sorted(df[time_col].astype(str).unique().tolist())

    if len(pday_vals) <= 10:
        segments = []
        for pday in pday_vals:
            sub = df[df[time_col].astype(str) == pday]
            n = int(len(sub))
            pos = int((sub[label_col] == 1).sum())
            neg = int((sub[label_col] == 0).sum())
            segments.append({
                "pday": pday,
                "samples": n,
                "positive": pos,
                "positive_rate": round(pos / n, 6) if n else 0.0,
                "pos_neg_ratio": _fmt_ratio(pos, neg),
            })
        return segments

    # pday > 10 时按月聚合
    df_dt = df.copy()
    df_dt["_month"] = df_dt[time_col].astype(str).str[:6]
    segments = []
    for month in sorted(df_dt["_month"].unique().tolist()):
        sub = df_dt[df_dt["_month"] == month]
        n = int(len(sub))
        pos = int((sub[label_col] == 1).sum())
        neg = int((sub[label_col] == 0).sum())
        segments.append({
            "pday": month,
            "samples": n,
            "positive": pos,
            "positive_rate": round(pos / n, 6) if n else 0.0,
            "pos_neg_ratio": _fmt_ratio(pos, neg),
        })
    return segments


def judge_stability(segments: List[dict]) -> dict:
    """稳定性判定: <1pp 稳定 / 1-3pp 轻微 / >=3pp 显著。"""
    if not segments:
        return {"positive_rate_max": 0, "positive_rate_min": 0, "volatility_pp": 0,
                "std_pp": 0, "judgment": "稳定", "note": "无时间段数据"}

    rates = [s["positive_rate"] for s in segments]
    rmax = max(rates)
    rmin = min(rates)
    vol_pp = round((rmax - rmin) * 100, 2)
    mean = sum(rates) / len(rates)
    std_pp = round(((sum((r - mean) ** 2 for r in rates) / len(rates)) ** 0.5) * 100, 2)

    if vol_pp < 1:
        judgment = "稳定"
    elif vol_pp < 3:
        judgment = "轻微波动"
    else:
        judgment = "显著波动"

    min_seg = next(s for s in segments if s["positive_rate"] == rmin)
    max_seg = next(s for s in segments if s["positive_rate"] == rmax)
    min_label = min_seg.get("pday", min_seg.get("pday_segment", "NA"))
    max_label = max_seg.get("pday", max_seg.get("pday_segment", "NA"))
    note = "正样本率最低: %s (%.2f%%), 最高: %s (%.2f%%)" % (
        min_label, rmin * 100, max_label, rmax * 100)

    return {
        "positive_rate_max": rmax,
        "positive_rate_min": rmin,
        "volatility_pp": vol_pp,
        "std_pp": std_pp,
        "judgment": judgment,
        "note": note,
    }


def judge_sufficiency(overall: dict) -> dict:
    """样本充足度判定。"""
    pos = overall["positive_samples"]
    total = overall["total_samples"]
    if pos >= 10000 and total >= 100000:
        judgment = "充足"
    elif pos >= 500 and total >= 50000:
        judgment = "基本可用"
    else:
        judgment = "不足，建议补充样本"
    return {
        "judgment": judgment,
        "positive_meets_10k": pos >= 10000,
        "positive_count": pos,
        "total_meets_50k": total >= 50000,
        "total_count": total,
    }


def compute_split_stats(df, args, train_df, test_df, oot_df, dropped: int) -> dict:
    """切分后统计。"""
    total = int(len(df))
    ranges = {
        "train": list(parse_range(args.train_range)),
        "test": list(parse_range(args.test_range)),
        "oot": list(parse_range(args.oot_range)),
    }

    def _stat(sub):
        n = int(len(sub))
        pos = int((sub[args.label_col] == 1).sum()) if n else 0
        if args.dt_col in sub.columns and n:
            pday_min = str(sub[args.dt_col].min())
            pday_max = str(sub[args.dt_col].max())
        else:
            pday_min = pday_max = None
        return {
            "rows": n,
            "positive": pos,
            "positive_rate": round(pos / n, 6) if n else 0.0,
            "pday_range": [pday_min, pday_max],
        }

    splits = {"train": _stat(train_df), "test": _stat(test_df), "oot": _stat(oot_df)}
    rates = [splits[k]["positive_rate"] for k in ("train", "test", "oot")]
    cross_diff = round((max(rates) - min(rates)) * 100, 2) if rates else 0.0

    return {
        "method": "time_explicit",
        "ranges": ranges,
        "dropped_rows": dropped,
        "actual_ratios": {
            "train": round(len(train_df) / total, 6) if total else 0,
            "test": round(len(test_df) / total, 6) if total else 0,
            "oot": round(len(oot_df) / total, 6) if total else 0,
        },
        "splits": splits,
        "cross_split_pos_rate_diff_pp": cross_diff,
        "note": "显式 pday 区间切分, 三档正样本率差异 %.2fpp" % cross_diff,
    }


# ---------- 输出 ----------

def write_splits(train_df, test_df, oot_df, output_dir: str) -> None:
    for name, sub in (("train", train_df), ("test", test_df), ("oot", oot_df)):
        sub.to_parquet(os.path.join(output_dir, "%s.parquet" % name), index=False)


def write_report_md(overall, segments, stability, sufficiency, split_stats, args, output_dir: str) -> None:
    rates = [s["positive_rate"] for s in segments]
    n_seg = len(segments)

    def _pos_pct(x):
        return "%.2f%%" % (x * 100)

    seg_rows = "\n".join(
        "| %s | %s | %s | %s | %s |" % (
            s["pday"], f"{s['samples']:,}", f"{s['positive']:,}",
            _pos_pct(s["positive_rate"]), s["pos_neg_ratio"]
        ) for s in segments
    )

    sp = split_stats["splits"]
    tr, te, oo = sp["train"], sp["test"], sp["oot"]
    actual = split_stats["actual_ratios"]
    cross_diff = split_stats["cross_split_pos_rate_diff_pp"]
    id_col = args._id_col_primary
    time_col = args.dt_col

    if args.mode == "local_file":
        source_label = args.local_parquet_path or args.sample or "未提供"
        source_line = "> 数据位置: %s" % source_label
    else:
        source_line = "> 样本表: %s" % (args.sample_table or "未提供")

    content = f"""# 样本分析报告

> 模型名称: {args.model_name}
> Session: {args.timestamp}
{source_line}

## 一、样本概览

| 指标 | 值 |
|------|-----|
| 总样本量 | {overall['total_samples']:,} |
| 正样本数 | {overall['positive_samples']:,} |
| 负样本数 | {overall['negative_samples']:,} |
| 正样本率 | {overall['positive_rate']:.2%} |
| 正负比 | {overall['positive_negative_ratio']} |
| pday 范围 | {overall['pday_range'][0]} ~ {overall['pday_range'][1]} |
| 时间段数 | {n_seg} |
| 用户数（{id_col} 去重） | {overall.get('user_unique', 'NA')} |
| {id_col} + {time_col} 重复 | {overall.get('dup_user_pday', 'NA')} |

## 二、分时间段标签分布

| 时间段 | 样本量 | 正样本数 | 正样本率 | 正负比 |
|--------|--------|----------|----------|--------|
{seg_rows}

## 三、标签稳定性

- 正样本率波动幅度（max - min） = {_pos_pct(stability['positive_rate_max'])} - {_pos_pct(stability['positive_rate_min'])} = **{stability['volatility_pp']:.2f}pp**
- 正样本率标准差 = {stability['std_pp']:.2f}pp

**判定：{stability['judgment']}**

{stability['note']}

## 四、综合判定与建模建议

### 4.1 样本可行性判定

| 维度 | 结果 |
|------|------|
| 标签分布 | {'✓' if 0.01 <= overall['positive_rate'] <= 0.99 else '⚠'} 正样本率 {overall['positive_rate']:.2%} |
| 正负比 | {'✓' if overall['positive_rate'] >= 0.01 else '⚠'} {overall['positive_negative_ratio']} |
| 时间稳定性 | {'✓' if stability['judgment'] == '稳定' else '⚠'} {stability['judgment']}（{stability['volatility_pp']:.2f}pp） |
| 样本量充足度 | {'✓' if sufficiency['judgment'] == '充足' else ('⚠' if sufficiency['judgment'] == '基本可用' else '✗')} {sufficiency['judgment']} |

### 4.2 建模建议

- {'无需重采样' if overall['positive_rate'] >= 0.05 else '建议重采样或调 scale_pos_weight'}：正样本率 {overall['positive_rate']:.2%}
- {'关注 Train→OOT 漂移' if stability['judgment'] != '稳定' else '标签稳定'}：波动幅度 {stability['volatility_pp']:.2f}pp，建模时关注 Train→OOT PSI
- 样本量：{sufficiency['judgment']}

## 五、Train/Test/OOT 切分

| 集合 | 样本量 | 正样本数 | 正样本率 | pday 范围 | 占比 |
|------|--------|----------|----------|-----------|------|
| Train | {tr['rows']:,} | {tr['positive']:,} | {tr['positive_rate']:.2%} | {tr['pday_range'][0]} ~ {tr['pday_range'][1]} | {actual['train']:.0%} |
| Test  | {te['rows']:,} | {te['positive']:,} | {te['positive_rate']:.2%} | {te['pday_range'][0]} ~ {te['pday_range'][1]} | {actual['test']:.0%} |
| OOT   | {oo['rows']:,} | {oo['positive']:,} | {oo['positive_rate']:.2%} | {oo['pday_range'][0]} ~ {oo['pday_range'][1]} | {actual['oot']:.0%} |

> 切分方式：显式区间（time_explicit）
> 三档区间：Train [{split_stats['ranges']['train'][0]}, {split_stats['ranges']['train'][1]}] / Test [{split_stats['ranges']['test'][0]}, {split_stats['ranges']['test'][1]}] / OOT [{split_stats['ranges']['oot'][0]}, {split_stats['ranges']['oot'][1]}]
> 区间外丢弃行数：{split_stats['dropped_rows']}
> 实际占比 {actual['train']:.0%}:{actual['test']:.0%}:{actual['oot']:.0%}
>
> 三档正样本率分别为 {tr['positive_rate']:.2%} / {te['positive_rate']:.2%} / {oo['positive_rate']:.2%}，跨档差异 ≤ {cross_diff:.2f}pp

## 六、下一步

1. ⏳ **等用户确认切分结果**
2. 之后 → 编排器调用 `classification-model-recommend` 检索历史模型
3. 之后 → `feature-matching`（拼接模式）拉特征宽表
4. 最后 → 建模决策（询问是否进入 model-development）
"""
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(content)


def write_report_xlsx(overall, segments, split_stats, args, output_dir: str) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[run_sample_analysis_task_spec] [警告] 缺 openpyxl, 跳过 report.xlsx 产出", file=sys.stderr)
        return

    xlsx_path = os.path.join(output_dir, "report.xlsx")
    overview_df = pd.DataFrame([{
        "总样本量": overall["total_samples"],
        "正样本数": overall["positive_samples"],
        "负样本数": overall["negative_samples"],
        "正样本率": overall["positive_rate"],
        "正负比": overall["positive_negative_ratio"],
        "pday范围": "%s ~ %s" % (overall["pday_range"][0], overall["pday_range"][1]),
        "pday数": overall["pday_unique_count"],
    }])
    seg_df = pd.DataFrame(segments).rename(columns={
        "pday": "时间段", "samples": "样本量", "positive": "正样本数",
        "positive_rate": "正样本率", "pos_neg_ratio": "正负比",
    })
    sp = split_stats["splits"]
    split_df = pd.DataFrame([
        {"集合": "Train", "样本量": sp["train"]["rows"], "正样本数": sp["train"]["positive"],
         "正样本率": sp["train"]["positive_rate"],
         "pday范围": "%s ~ %s" % (sp["train"]["pday_range"][0], sp["train"]["pday_range"][1])},
        {"集合": "Test", "样本量": sp["test"]["rows"], "正样本数": sp["test"]["positive"],
         "正样本率": sp["test"]["positive_rate"],
         "pday范围": "%s ~ %s" % (sp["test"]["pday_range"][0], sp["test"]["pday_range"][1])},
        {"集合": "OOT", "样本量": sp["oot"]["rows"], "正样本数": sp["oot"]["positive"],
         "正样本率": sp["oot"]["positive_rate"],
         "pday范围": "%s ~ %s" % (sp["oot"]["pday_range"][0], sp["oot"]["pday_range"][1])},
    ])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        overview_df.to_excel(w, sheet_name="overview", index=False)
        seg_df.to_excel(w, sheet_name="time_segments", index=False)
        split_df.to_excel(w, sheet_name="split_stats", index=False)


def write_manifest(overall, segments, stability, sufficiency, split_stats, args, output_dir: str) -> None:
    manifest = {
        "produced_by": "classification-skills/classification-model-task-spec",
        "schema_version": 1,
        "model_name": args.model_name,
        "timestamp": args.timestamp,
        "sample_table": args.sample_table,
        "sample_file": os.path.abspath(args.sample),
        "sample_summary": overall,
        "time_segments": segments,
        "stability": stability,
        "sample_sufficiency": sufficiency,
        "split": split_stats,
        "split_manifest": "_split_manifest.json",
        "split_files": {
            "train": "train.parquet",
            "test": "test.parquet",
            "oot": "oot.parquet",
        },
        "user_confirmed": False,
    }
    with open(os.path.join(output_dir, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    split_manifest = {
        "produced_by": "classification-skills/classification-model-task-spec",
        "schema_version": 1,
        "model_name": args.model_name,
        "timestamp": args.timestamp,
        "method": split_stats["method"],
        "ranges": split_stats["ranges"],
        "dropped_rows": split_stats["dropped_rows"],
        "actual_ratios": split_stats["actual_ratios"],
        "splits": split_stats["splits"],
        "cross_split_pos_rate_diff_pp": split_stats["cross_split_pos_rate_diff_pp"],
        "split_files": {
            "train": "train.parquet",
            "test": "test.parquet",
            "oot": "oot.parquet",
        },
    }
    with open(os.path.join(output_dir, "_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)


def print_summary(overall, stability, sufficiency, split_stats) -> None:
    sp = split_stats["splits"]
    tr, te, oo = sp["train"], sp["test"], sp["oot"]
    actual = split_stats["actual_ratios"]
    print()
    print("样本分析 + 切分完成")
    print()
    print("样本概览:")
    print("- 总样本: %s，正样本率 %.2f%%" % (f"{overall['total_samples']:,}", overall["positive_rate"] * 100))
    print("- pday 范围: %s ~ %s" % (overall["pday_range"][0], overall["pday_range"][1]))
    print("- 标签稳定性: %s（%.2fpp）" % (stability["judgment"], stability["volatility_pp"]))
    print("- 样本充足度: %s" % sufficiency["judgment"])
    print()
    print("Train/Test/OOT 切分:")
    print("| 集合 | 样本量 | 正样本率 | pday 范围 |")
    print("|------|--------|----------|-----------|")
    print("| Train | %s | %.2f%% | %s ~ %s |" % (f"{tr['rows']:,}", tr["positive_rate"] * 100, tr["pday_range"][0], tr["pday_range"][1]))
    print("| Test  | %s | %.2f%% | %s ~ %s |" % (f"{te['rows']:,}", te["positive_rate"] * 100, te["pday_range"][0], te["pday_range"][1]))
    print("| OOT   | %s | %.2f%% | %s ~ %s |" % (f"{oo['rows']:,}", oo["positive_rate"] * 100, oo["pday_range"][0], oo["pday_range"][1]))
    print()
    print("切分比例 %d:%d:%d" % (round(actual["train"] * 100), round(actual["test"] * 100), round(actual["oot"] * 100)))
    if split_stats["dropped_rows"]:
        print("[警告] %d 行 pday 不在三档区间内, 已丢弃" % split_stats["dropped_rows"])


# ---------- 主入口 ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="classification-model-task-spec 样本分析 + 切分一体化")
    parser.add_argument("--sample", required=True, help="feature-matching 拉的全量样本 parquet 路径")
    parser.add_argument("--train-range", required=True, help="Train pday 闭区间, 如 20260401,20260510")
    parser.add_argument("--test-range", required=True, help="Test pday 闭区间, 如 20260511,20260520")
    parser.add_argument("--oot-range", required=True, help="OOT pday 闭区间, 如 20260521,20260530")
    parser.add_argument("--model-name", required=True, help="模型英文简称")
    parser.add_argument("--timestamp", required=True, help="session 时间戳 YYYYMMDD-HHMMSS")
    parser.add_argument("--output-dir", required=True, help="输出目录 (通常是 data-profile/)")
    parser.add_argument("--dt-col", default="pday", help="时间列名, 默认 pday")
    parser.add_argument("--time-col", default=argparse.SUPPRESS, dest="time_col_deprecated",
                        help="[已弃用 alias] 用 --dt-col; 传入时覆盖 --dt-col")
    parser.add_argument("--label-col", default="label", help="标签列名, 默认 label")
    parser.add_argument("--id-cols", default="user_no", help="用户唯一标识列名(逗号分隔多列时取首列作主 ID), 默认 user_no")
    parser.add_argument("--id-col", default=argparse.SUPPRESS, dest="id_col_deprecated",
                        help="[已弃用 alias] 用 --id-cols; 传入时覆盖 --id-cols (单列)")
    parser.add_argument("--sample-table", default=None, help="样本源表名 (写入 manifest 用, 可选)")
    parser.add_argument("--mode", choices=["spark", "local_file"], default="spark",
                        help="数据模式: spark=报告头展示表名, local_file=展示本地数据位置")
    parser.add_argument("--local-parquet-path", default=None,
                        help="local_file 模式下的本地数据路径 (parquet/csv), 用于报告头展示")
    args = parser.parse_args()

    # 兼容老参数: --time-col / --id-col 仍可传入 (已弃用, 推荐用 --dt-col / --id-cols)
    if hasattr(args, "time_col_deprecated") and args.time_col_deprecated:
        args.dt_col = args.time_col_deprecated
    if hasattr(args, "id_col_deprecated") and args.id_col_deprecated:
        args.id_cols = args.id_col_deprecated
    # 分析侧只取单列作主 ID
    args._id_col_primary = args.id_cols.split(",")[0].strip() if args.id_cols else "user_no"

    if not os.path.exists(args.sample):
        raise SystemExit("样本文件不存在: %s" % args.sample)

    os.makedirs(args.output_dir, exist_ok=True)

    print("[run_sample_analysis_task_spec] 读取样本: %s" % args.sample)
    df = pd.read_parquet(args.sample)
    print("[run_sample_analysis_task_spec] 样本: %d 行 x %d 列" % (len(df), df.shape[1]))

    validate_input(df, args)

    overall = compute_overall(df, args)
    segments = segment_by_time(df, args)
    stability = judge_stability(segments)
    sufficiency = judge_sufficiency(overall)

    ranges = {
        "train": parse_range(args.train_range),
        "test": parse_range(args.test_range),
        "oot": parse_range(args.oot_range),
    }
    train_df, test_df, oot_df, dropped = split_dataframe_by_ranges(df, args.dt_col, ranges)
    if dropped:
        print("[run_sample_analysis_task_spec] [警告] %d 行落在三档区间外, 已丢弃" % dropped, file=sys.stderr)

    split_stats = compute_split_stats(df, args, train_df, test_df, oot_df, dropped)

    write_splits(train_df, test_df, oot_df, args.output_dir)
    write_report_md(overall, segments, stability, sufficiency, split_stats, args, args.output_dir)
    write_report_xlsx(overall, segments, split_stats, args, args.output_dir)
    write_manifest(overall, segments, stability, sufficiency, split_stats, args, args.output_dir)
    print_summary(overall, stability, sufficiency, split_stats)


if __name__ == "__main__":
    main()
