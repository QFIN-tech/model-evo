# -*- coding: utf-8 -*-
"""把 recommend 取数产出的 sample.parquet(样本表⋈模型表 JOIN 结果)按时间(pday)切分成 train/test/oot 三份。

独立可调起的本地 pandas 脚本, 两种切分方式(显式区间优先):
  1. 显式 pday 区间(首选): --train-range/--test-range/--oot-range 各给闭区间 [起,止],
     区间外的行丢弃并告警; 由用户/上游 task-spec 文档给定, 口径可控。
  2. 比例: --ratios train,test,oot, 沿 pday 升序累计行占比切分, oot 取时间最末段,
     **同一 pday 不跨切分**(整天归一档)。

切分入参来源优先级: classification-model-task-spec 文档 > 其他上游 md > 交互询问。
安全: 仅打印聚合统计, 不输出用户级明细到日志。

用法:
    # 显式区间(首选)
    python split_sample.py --input <sample.parquet> \
        --train-range 20260312,20260430 \
        --test-range  20260501,20260516 \
        --oot-range   20260517,20260524 \
        [--time-col pday] [--label-col label] [--output_dir <同 input 目录>]

    # 比例
    python split_sample.py --input <sample.parquet> --ratios 0.6,0.2,0.2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple


def parse_ratios(text: str) -> Tuple[float, float, float]:
    """解析 "train,test,oot" 三元组比例并校验求和约等于 1。

    Args:
        text: 形如 "0.6,0.2,0.2" 的字符串

    Returns:
        (train_ratio, test_ratio, oot_ratio)

    Raises:
        ValueError: 非三元素 / 含非正数 / 求和偏离 1 超过 1e-6
    """
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--ratios 须为三元素 train,test,oot, 如 0.6,0.2,0.2")
    try:
        r = tuple(float(p) for p in parts)
    except ValueError:
        raise ValueError("--ratios 三个值须为数字, 如 0.6,0.2,0.2")
    if any(v <= 0 for v in r):
        raise ValueError("--ratios 三个值须均为正数")
    if abs(sum(r) - 1.0) > 1e-6:
        raise ValueError("--ratios 三个值之和须为 1.0, 当前为 %.6f" % sum(r))
    return r  # type: ignore[return-value]


def parse_range(text: str) -> Tuple[str, str]:
    """解析 "起,止" 闭区间(YYYYMMDD)。

    Args:
        text: 形如 "20260312,20260430" 的字符串

    Returns:
        (start, end) 两个 8 位日期字符串

    Raises:
        ValueError: 非两元素 / 非 8 位数字 / 起 > 止
    """
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
    """校验 train/test/oot 三个 pday 区间: 时序递增, 允许相邻, 重叠/逆序报错。

    间隔逻辑: 前档结束日的次日可以等于后档开始日(允许相邻), 仅当前档结束日
    >= 后档开始日时才视为重叠或逆序。

    Args:
        train/test/oot: 各 (起, 止) 闭区间

    Raises:
        ValueError: 区间重叠或时序逆序(要求 train < test < oot, 允许相邻)
    """
    ordered = [("train", train), ("test", test), ("oot", oot)]
    for (n1, r1), (n2, r2) in zip(ordered, ordered[1:]):
        # 前档结束日必须早于后档开始日: 允许相邻(前档结束日次日 = 后档开始日),
        # 仅当前档结束日 >= 后档开始日时视为重叠或逆序
        if r1[1] >= r2[0]:
            raise ValueError(
                "%s 区间 [%s,%s] 与 %s 区间 [%s,%s] 重叠或逆序, 要求 train<test<oot(允许相邻, 间隔≥1天)"
                % (n1, r1[0], r1[1], n2, r2[0], r2[1])
            )


def classify_by_ranges(pday: object, ranges: Dict[str, Tuple[str, str]]):
    """按显式 pday 区间把单个 pday 归到 train/test/oot, 区间外返回 None。

    Args:
        pday: 待归档的日期值(转 str 比较)
        ranges: {"train": (起,止), "test": (...), "oot": (...)}

    Returns:
        "train"|"test"|"oot"|None
    """
    p = str(pday)
    for name in ("train", "test", "oot"):
        start, end = ranges[name]
        if start <= p <= end:
            return name
    return None


def assign_splits(
    day_counts: Sequence[Tuple[object, int]],
    ratios: Tuple[float, float, float],
) -> Dict[object, str]:
    """按时间升序的 (pday, 行数) 序列, 依比例把每个 pday 归到 train/test/oot。

    规则: 沿时间升序累计行占比, 以"该天之前的累计占比"落在哪个区间决定该天归属,
    保证 train 全部早于 test 早于 oot, 且整天不跨档。

    Args:
        day_counts: [(pday, 行数), ...] 须已按 pday 升序
        ratios: (train_ratio, test_ratio, oot_ratio), 求和=1

    Returns:
        {pday: "train"|"test"|"oot"}
    """
    train_r, test_r, _ = ratios
    total = sum(n for _, n in day_counts)
    if total <= 0:
        return {}
    cum_before = 0
    mapping: Dict[object, str] = {}
    for day, n in day_counts:
        frac_before = cum_before / total
        if frac_before < train_r:
            mapping[day] = "train"
        elif frac_before < train_r + test_r:
            mapping[day] = "test"
        else:
            mapping[day] = "oot"
        cum_before += n
    return mapping


def _summarize(df, label_col: str, time_col: str) -> dict:
    """统计一个切分子集的行数 / 正样本率 / 时间边界(聚合量, 不含明细)。"""
    n = int(len(df))
    summary = {"rows": n}
    if label_col in df.columns and n > 0:
        pos = int((df[label_col] == 1).sum())
        summary["pos"] = pos
        summary["pos_rate"] = round(pos / n, 6)
    if time_col in df.columns and n > 0:
        summary["%s_min" % time_col] = str(df[time_col].min())
        summary["%s_max" % time_col] = str(df[time_col].max())
    return summary


def split_dataframe(df, time_col: str, ratios: Tuple[float, float, float]):
    """按 time_col 把 DataFrame 切成 train/test/oot 三份(整天不跨档)。

    Args:
        df: 待切分 DataFrame
        time_col: 时间列名(如 pday)
        ratios: (train, test, oot) 比例

    Returns:
        (train_df, test_df, oot_df, mapping)
    """
    if time_col not in df.columns:
        raise ValueError("时间列 %r 不在样本列中, 无法按时间切分" % time_col)
    day_counts = sorted(df.groupby(time_col).size().items(), key=lambda kv: kv[0])
    mapping = assign_splits(day_counts, ratios)
    split_series = df[time_col].map(mapping)
    train_df = df[split_series == "train"]
    test_df = df[split_series == "test"]
    oot_df = df[split_series == "oot"]
    return train_df, test_df, oot_df, mapping


def split_dataframe_by_ranges(df, time_col: str, ranges: Dict[str, Tuple[str, str]]):
    """按显式 pday 区间把 DataFrame 切成 train/test/oot 三份, 区间外行丢弃。

    Args:
        df: 待切分 DataFrame
        time_col: 时间列名(如 pday)
        ranges: {"train": (起,止), "test": (...), "oot": (...)}

    Returns:
        (train_df, test_df, oot_df, dropped) — dropped 为落在所有区间外的行数
    """
    if time_col not in df.columns:
        raise ValueError("时间列 %r 不在样本列中, 无法按时间切分" % time_col)
    split_series = df[time_col].map(lambda p: classify_by_ranges(p, ranges))
    train_df = df[split_series == "train"]
    test_df = df[split_series == "test"]
    oot_df = df[split_series == "oot"]
    dropped = int(split_series.isna().sum())
    return train_df, test_df, oot_df, dropped


def _write_splits(train_df, test_df, oot_df, out_dir: str) -> dict:
    """把三档 DataFrame 落 parquet, 返回 {name: path}。"""
    out_paths = {}
    for name, sub in (("train", train_df), ("test", test_df), ("oot", oot_df)):
        p = os.path.join(out_dir, "%s.parquet" % name)
        sub.to_parquet(p, index=False)
        out_paths[name] = p
    return out_paths


def _print_and_dump_manifest(manifest, out_paths, out_dir, time_col):
    """统一打印各档摘要并落 _split_manifest.json。"""
    manifest_path = os.path.join(out_dir, "_split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    for name in ("train", "test", "oot"):
        s = manifest["splits"][name]
        print("[split_sample] %-5s -> %s | %d 行, 正样本率 %s, %s [%s ~ %s]"
              % (name, out_paths[name], s.get("rows", 0),
                 s.get("pos_rate", "NA"), time_col,
                 s.get("%s_min" % time_col, "NA"),
                 s.get("%s_max" % time_col, "NA")))
    print("[split_sample] manifest: %s" % manifest_path)
    print("[split_sample] 完成")


def main() -> None:
    """命令行入口: 读 sample.parquet -> 按 pday 区间或比例切分 -> 落三份 parquet + manifest。"""
    parser = argparse.ArgumentParser(description="recommend 评估数据切分(本地 pandas, 按时间)")
    parser.add_argument("--input", required=True, help="sample.parquet 路径")
    parser.add_argument("--ratios", default=None, help="比例模式: train,test,oot 如 0.6,0.2,0.2(与 *-range 互斥)")
    parser.add_argument("--train-range", default=None, help="显式模式: train 的 pday 闭区间, 如 20260312,20260430")
    parser.add_argument("--test-range", default=None, help="显式模式: test 的 pday 闭区间, 如 20260501,20260516")
    parser.add_argument("--oot-range", default=None, help="显式模式: oot 的 pday 闭区间, 如 20260517,20260524")
    parser.add_argument("--time-col", default="pday", help="时间切分列, 默认 pday")
    parser.add_argument("--label-col", default="label", help="标签列(仅用于统计正样本率), 默认 label")
    parser.add_argument("--output_dir", default=None, help="输出目录, 默认与 input 同目录")
    args = parser.parse_args()

    import pandas as pd

    any_range = any([args.train_range, args.test_range, args.oot_range])
    if any_range and args.ratios:
        raise SystemExit("--ratios 与 --*-range 互斥, 二选一")
    if not any_range and not args.ratios:
        raise SystemExit("须指定切分方式: --train-range/--test-range/--oot-range(优先) 或 --ratios")

    in_path = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(in_path)
    os.makedirs(out_dir, exist_ok=True)

    print("[split_sample] 读取: %s" % in_path)
    df = pd.read_parquet(in_path)
    total = len(df)

    if any_range:
        # 显式 pday 区间模式(首选)
        if not (args.train_range and args.test_range and args.oot_range):
            raise SystemExit("显式区间模式须同时给 --train-range / --test-range / --oot-range 三档")
        ranges = {
            "train": parse_range(args.train_range),
            "test": parse_range(args.test_range),
            "oot": parse_range(args.oot_range),
        }
        validate_ranges(ranges["train"], ranges["test"], ranges["oot"])
        print("[split_sample] 样本: %d 行 x %d 列, 按 %r 显式区间切分 train%s test%s oot%s"
              % (total, df.shape[1], args.time_col,
                 ranges["train"], ranges["test"], ranges["oot"]))
        train_df, test_df, oot_df, dropped = split_dataframe_by_ranges(df, args.time_col, ranges)
        if dropped:
            print("[split_sample] [告警] %d 行落在三档区间外, 已丢弃" % dropped)
        manifest = {
            "produced_by": "classification-skills/classification-model-recommend",
            "schema_version": 1,
            "source": in_path,
            "time_col": args.time_col,
            "label_col": args.label_col,
            "strategy": "time_explicit",
            "ranges": {k: {"start": v[0], "end": v[1]} for k, v in ranges.items()},
            "dropped_rows": dropped,
            "actual_ratios": {
                "train": round(len(train_df) / total, 6) if total else 0,
                "test": round(len(test_df) / total, 6) if total else 0,
                "oot": round(len(oot_df) / total, 6) if total else 0,
            },
            "splits": {
                "train": _summarize(train_df, args.label_col, args.time_col),
                "test": _summarize(test_df, args.label_col, args.time_col),
                "oot": _summarize(oot_df, args.label_col, args.time_col),
            },
        }
    else:
        # 比例模式
        ratios = parse_ratios(args.ratios)
        print("[split_sample] 样本: %d 行 x %d 列, 按 %r 比例切分 train/test/oot=%s"
              % (total, df.shape[1], args.time_col, ":".join("%.2f" % r for r in ratios)))
        train_df, test_df, oot_df, _ = split_dataframe(df, args.time_col, ratios)
        manifest = {
            "produced_by": "classification-skills/classification-model-recommend",
            "schema_version": 1,
            "source": in_path,
            "time_col": args.time_col,
            "label_col": args.label_col,
            "strategy": "time_ratio",
            "target_ratios": {"train": ratios[0], "test": ratios[1], "oot": ratios[2]},
            "actual_ratios": {
                "train": round(len(train_df) / total, 6) if total else 0,
                "test": round(len(test_df) / total, 6) if total else 0,
                "oot": round(len(oot_df) / total, 6) if total else 0,
            },
            "splits": {
                "train": _summarize(train_df, args.label_col, args.time_col),
                "test": _summarize(test_df, args.label_col, args.time_col),
                "oot": _summarize(oot_df, args.label_col, args.time_col),
            },
        }

    out_paths = _write_splits(train_df, test_df, oot_df, out_dir)
    _print_and_dump_manifest(manifest, out_paths, out_dir, args.time_col)


if __name__ == "__main__":
    main()
