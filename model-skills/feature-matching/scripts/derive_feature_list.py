# -*- coding: utf-8 -*-
"""从已落本地的 sample.parquet / sample.csv 派生特征清单 feature-list.csv。

用于「全量模式」: 拼接取数 features 留空时, 特征列在集群运行时才确定,
脚本生成阶段拿不到。故取数回本地后, 读 sample 的 schema(仅读列名,
不全量加载), 排除 id/dt/label/base 等非特征列, 把剩余列写成 feature-list.csv,
供下游 feature-analysis / model-training 直接作为候选特征起点。

输入按扩展名判断: .parquet 走 pyarrow schema, .csv 走 pandas 读首行列名
(读 header 后立即丢弃数据, 不全量加载)。

可选 --filter <清单文件>: 传时只输出「清单指定 且 在 sample schema 中存在」
的交集特征(按清单顺序), 不再落全量派生清单。设计意图: 取数仍拉全量列保信息完整,
但下游消费的特征清单以指定清单为准, 避免派生清单与清单冲突。
清单里不在 sample 中的特征会被丢弃并打 warn。
清单文件 .csv 取 feature_name 列, 其他后缀按行读(跳过空行与 # 注释)。

用法:
    # 全量派生(无 --filter), parquet 输入
    python derive_feature_list.py \\
        --input  <local sample.parquet> \\
        --output <local feature-list.csv> \\
        --exclude user_no,pday,label

    # 同上, csv 输入
    python derive_feature_list.py \\
        --input  <local sample.csv> \\
        --output <local feature-list.csv> \\
        --exclude user_no,pday,label

    # 按清单过滤(推荐, 取数全量但下游用 feature-knowledge 登记的清单)
    python derive_feature_list.py \\
        --input  <local sample.parquet> \\
        --output <local feature-list.csv> \\
        --exclude user_no,pday,label \\
        --filter model-knowledge/assets/feature-knowledge/feature-list/<your-feature-list>.csv
"""

from __future__ import annotations

import argparse
from typing import List, Optional


def read_parquet_columns(path: str) -> List[str]:
    """只读 parquet 的 schema 列名(不加载数据), 接受单文件与目录(多 part)。

    Args:
        path: parquet 文件或目录路径

    Returns:
        列名列表(原始顺序)
    """
    import pyarrow.parquet as pq

    # ParquetDataset 接受单文件与目录, 仅读 schema 廉价
    schema = pq.ParquetDataset(path).schema
    # pyarrow >=8 ParquetDataset.schema 是 pyarrow.Schema, .names 给列名
    names = list(getattr(schema, "names", None) or schema)
    return [str(n) for n in names]


def read_sample_columns(path: str) -> List[str]:
    """按扩展名读 sample 的列名: .parquet 走 pyarrow schema, .csv 走 pandas 读首行。

    parquet 仅读 schema(不加载数据); csv 用 pandas 读 header 后立即丢弃数据帧
    (nrows=0, 仅取列名), 不全量加载。

    Args:
        path: sample 文件路径(.parquet 或 .csv)

    Returns:
        列名列表(原始顺序)

    Raises:
        ValueError: 扩展名既非 .parquet 也非 .csv
    """
    p = str(path).lower()
    if p.endswith(".parquet"):
        return read_parquet_columns(path)
    if p.endswith(".csv"):
        import pandas as pd
        # nrows=0 仅读 header 取列名, 不全量加载
        df = pd.read_csv(path, nrows=0)
        return [str(c) for c in df.columns]
    raise ValueError(
        f"不支持的 sample 文件格式: {path} (仅支持 .parquet / .csv)"
    )


def derive_features(all_cols: List[str], exclude: List[str]) -> List[str]:
    """从全部列中排除非特征列, 返回特征列(保序去重)。

    Args:
        all_cols: parquet 全部列名
        exclude: 需排除的列(id/dt/label/base 等)

    Returns:
        特征列名列表
    """
    excl = {c for c in exclude if c}
    features: List[str] = []
    seen = set()
    for c in all_cols:
        if c in excl or c in seen:
            continue
        seen.add(c)
        features.append(c)
    return features


def filter_by_list(
    derived: List[str], allow_list: List[str]
) -> tuple:
    """按 allow_list 过滤 derived 特征, 返回 (交集, allow_list 中不在 derived 里的特征)。

    交集顺序按 allow_list(以 allow_list 为准, 保序去重);
    derived 里不在 allow_list 中的特征直接丢弃(不报错, 因 sample 全量列本就含非入模特征)。

    Args:
        derived: derive_features 已派生出的全量特征列
        allow_list: 过滤清单指定的入模特征(允许清单)

    Returns:
        (kept, missing_in_sample):
          - kept: allow_list ∩ derived, 按 allow_list 顺序
          - missing_in_sample: allow_list 中不在 derived 里的特征(供上游打 warn)
    """
    derived_set = set(derived)
    seen: set = set()
    kept: List[str] = []
    missing: List[str] = []
    for f in allow_list:
        if f in seen:
            continue
        seen.add(f)
        if f in derived_set:
            kept.append(f)
        else:
            missing.append(f)
    return kept, missing


def _load_allow_list(path: str) -> List[str]:
    """读过滤清单(.csv 取 feature_name 列, 其他后缀按行读, 跳过空行与 # 注释行, 保序去重)。

    复用 gen_feature_list 的解析逻辑; 独立实现避免循环 import。
    """
    import csv as _csv
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"过滤清单文件不存在: {p}")
    features: List[str] = []
    seen: set = set()
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames or "feature_name" not in reader.fieldnames:
                raise ValueError(
                    f"过滤清单 CSV 必须包含 feature_name 列, 实际列: {reader.fieldnames}"
                )
            for row in reader:
                name = (row.get("feature_name") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                features.append(name)
    else:
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                name = raw.strip()
                if not name or name.startswith("#"):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                features.append(name)
    if not features:
        raise ValueError(f"过滤清单为空: {p}")
    return features


def main() -> None:
    """命令行入口: 读 sample schema -> 排除非特征列 -> 写 feature-list.csv。"""
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_feature_list import write_feature_list_csv

    parser = argparse.ArgumentParser(description="从 sample.parquet / sample.csv 派生 feature-list.csv")
    parser.add_argument("--input", required=True, help="本地 sample.parquet / sample.csv 路径")
    parser.add_argument("--output", required=True, help="输出 feature-list.csv 路径")
    parser.add_argument("--exclude", default="", help="逗号分隔的非特征列(id/dt/label/base)")
    parser.add_argument(
        "--filter",
        default=None,
        help="可选, 过滤清单路径(.csv 取 feature_name 列 / .txt 按行); 传时只输出该清单与"
             " sample schema 的交集(按清单顺序), 不传则输出全量派生清单",
    )
    args = parser.parse_args()

    exclude = [c.strip() for c in args.exclude.split(",") if c.strip()]
    all_cols = read_sample_columns(args.input)
    derived = derive_features(all_cols, exclude)

    if args.filter:
        # repo 根解析相对路径(与 gen_feature_list.load_feature_list 一致)
        from pathlib import Path
        filter_path = Path(args.filter)
        if not filter_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            filter_path = (repo_root / filter_path).resolve()
        allow_list = _load_allow_list(str(filter_path))
        kept, missing_in_sample = filter_by_list(derived, allow_list)
        features = kept
        if missing_in_sample:
            preview = ", ".join(missing_in_sample[:20])
            print(
                "[derive_feature_list] [WARN] %d/%d 个清单特征不在 sample 中, 已丢弃: %s"
                % (len(missing_in_sample), len(allow_list), preview)
            )
            if len(missing_in_sample) > 20:
                print("  ... 共 %d 个, 仅显示前 20 个" % len(missing_in_sample))
        print(
            "[derive_feature_list] 全部 %d 列, 排除 %d 列, 派生 %d 特征;"
            " 按 %s 过滤后保留 %d (清单 %d, 丢弃 %d 不在 sample) -> %s"
            % (len(all_cols), len(exclude), len(derived), filter_path,
               len(features), len(allow_list), len(missing_in_sample), args.output)
        )
    else:
        features = derived
        print("[derive_feature_list] 全部 %d 列, 排除 %d 列, 写出 %d 个特征 -> %s"
              % (len(all_cols), len(exclude), len(features), args.output))

    write_feature_list_csv(features, args.output)


if __name__ == "__main__":
    main()
