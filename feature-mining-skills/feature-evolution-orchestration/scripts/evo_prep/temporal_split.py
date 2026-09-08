# -*- coding: utf-8 -*-
"""时序切分: 按 dt_col 升序切 train/test/oot, 禁止随机切分(硬规则)。

两种输入(二选一):
  ratio:  [0.7, 0.2, 0.1] —— 按 dt 升序排序后按行数比例切
  ranges: {train: [起,止], test: [起,止], oot: [起,止]} —— 显式日期区间(8 位 YYYYMMDD)

显式区间校验规则与 _modelevo-shared/config_io.validate_split_ranges 一致:
三档齐全、起<=止、时序递增(允许相邻, 前档结束日须早于后档开始日)。
"""
from __future__ import annotations

import pandas as pd

from .sample_contract import ContractError, normalize_dt


def _parse_range(name: str, value) -> tuple:
    """规整单档区间为 (起, 止) 两个 8 位日期字符串。"""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        raise ContractError("split.ranges.%s 须为 [起, 止] 列表或 '起,止' 字符串" % name)
    if len(parts) != 2:
        raise ContractError("split.ranges.%s 须为两元素 [起, 止], 当前 %r" % (name, value))
    start, end = parts
    for d in (start, end):
        if not (d.isdigit() and len(d) == 8):
            raise ContractError("split.ranges.%s 日期须为 8 位 YYYYMMDD, 当前 %r" % (name, d))
    if start > end:
        raise ContractError("split.ranges.%s 起始 %s 不应大于结束 %s" % (name, start, end))
    return start, end


def validate_ranges(ranges_cfg: dict) -> dict:
    """校验显式三档区间(时序递增, 允许相邻)。"""
    if not isinstance(ranges_cfg, dict):
        raise ContractError("split.ranges 须为 {train: [起,止], test: [...], oot: [...]} 字典")
    ranges = {}
    for name in ("train", "test", "oot"):
        if not ranges_cfg.get(name):
            raise ContractError("split.ranges 须三档齐全, 缺 %s" % name)
        ranges[name] = _parse_range(name, ranges_cfg[name])
    ordered = [ranges["train"], ranges["test"], ranges["oot"]]
    for (n1, r1), (n2, r2) in zip(
        [("train", ordered[0]), ("test", ordered[1])],
        [("test", ordered[1]), ("oot", ordered[2])],
    ):
        if r1[1] >= r2[0]:
            raise ContractError(
                "split.ranges.%s [%s,%s] 与 %s [%s,%s] 重叠或逆序, "
                "要求 train<test<oot(允许相邻, 间隔>=1天)" % (n1, r1[0], r1[1], n2, r2[0], r2[1])
            )
    return ranges


def temporal_split(df: pd.DataFrame, dt_col: str, split_cfg: dict) -> dict:
    """执行时序切分, 返回 {split: DataFrame}(索引已重置)。

    Args:
        df: 全量样本
        dt_col: 时间列
        split_cfg: yaml split 段, 含 ratio 或 ranges 之一

    Raises:
        ContractError: ratio/ranges 都缺或都填 / 某档为空
    """
    has_ratio = bool(split_cfg.get("ratio"))
    has_ranges = bool(split_cfg.get("ranges"))
    if has_ratio == has_ranges:
        raise ContractError("split.ratio 与 split.ranges 必须二选一(禁止随机切分)")

    df = df.copy()
    df["_dt"] = normalize_dt(df[dt_col])
    if df["_dt"].isna().any():
        bad = int(df["_dt"].isna().sum())
        raise ContractError("dt_col=%s 有 %d 行无法解析为日期, 无法时序切分" % (dt_col, bad))
    df = df.sort_values("_dt", kind="mergesort").reset_index(drop=True)  # 稳定排序

    if has_ratio:
        ratio = [float(x) for x in split_cfg["ratio"]]
        if len(ratio) != 3 or any(x <= 0 for x in ratio):
            raise ContractError("split.ratio 须为 3 个正数 [train, test, oot], 当前 %r" % (ratio,))
        total = sum(ratio)
        n = len(df)
        n_train = int(round(n * ratio[0] / total))
        n_test = int(round(n * ratio[1] / total))
        cuts = {"train": df.iloc[:n_train], "test": df.iloc[n_train:n_train + n_test], "oot": df.iloc[n_train + n_test:]}
    else:
        ranges = validate_ranges(split_cfg["ranges"])
        cuts = {}
        for name, (start, end) in ranges.items():
            lo = pd.to_datetime(start, format="%Y%m%d")
            hi = pd.to_datetime(end, format="%Y%m%d") + pd.Timedelta(days=1)
            cuts[name] = df[(df["_dt"] >= lo) & (df["_dt"] < hi)]

    out = {}
    for name, sub in cuts.items():
        sub = sub.drop(columns=["_dt"]).reset_index(drop=True)
        if len(sub) == 0:
            raise ContractError("切分后 %s 档为空, 请调整 split 配置" % name)
        out[name] = sub
    return out


def split_manifest(splits: dict, dt_col: str, label_col: str) -> dict:
    """切分清单: 各档行数/时间区间/label 率(manifest 落盘与报告用)。"""
    info = {}
    for name, sub in splits.items():
        dt = normalize_dt(sub[dt_col])
        info[name] = {
            "rows": int(len(sub)),
            "dt_min": dt.min().strftime("%Y%m%d"),
            "dt_max": dt.max().strftime("%Y%m%d"),
            "label_rate": round(float(sub[label_col].mean()), 6),
        }
    return info
