# -*- coding: utf-8 -*-
"""SplitReport 驱动的三段式切分（train / val / oot）。

与 _xgb._impute / _woe 一致, 把切分状态从 dict 收口到 frozen dataclass,
使下游消费方拿到的是带方法的不可变对象而非裸字典。

设计要点:
- 切分结果 = DatasetSplits(frozen dataclass, 含 train/val/oot + report)
- 切分元信息 = SplitReport(frozen dataclass, to_dict/to_markdown/summary_line)
- 策略选择 = _split_by_strategy 分发器, 替代主函数内 if/elif 链
- 比例校验 = _validate_ratios 同时返回 SplitRatios, 避免下游重复算 (1-oot)*(1-val)
- 查询容错 = _safe_query 在模块级定义, 非内联闭包
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._load import load_table, infer_features

logger = logging.getLogger(__name__)

# ============================================================
# 默认值常量
# ============================================================

DEFAULT_OOT_RATIO = 0.20
DEFAULT_VAL_RATIO = 0.25
DEFAULT_RANDOM_SEED = 42

# 策略枚举(字符串常量, 供 SplitReport.strategy 比对)
_STRATEGY_EXPLICIT = "explicit"
_STRATEGY_TIME = "time"
_STRATEGY_RANDOM = "random"
_VALID_STRATEGIES: FrozenSet[str] = frozenset(
    {_STRATEGY_EXPLICIT, _STRATEGY_TIME, _STRATEGY_RANDOM}
)


# ============================================================
# 不可变数据结构(frozen dataclass 簇)
# ============================================================

@dataclass(frozen=True)
class SplitRatios:
    """切分比例 triplet。train_ratio 已由 (1-oot)*(1-val) 推导, 下游不重复算。"""
    oot: float
    val: float
    train: float

    def to_dict(self) -> Dict[str, float]:
        return {"oot": self.oot, "val": self.val, "train": self.train}


@dataclass(frozen=True)
class SplitCounts:
    """三段样本量。"""
    train: int
    val: int
    oot: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.oot

    def to_dict(self) -> Dict[str, int]:
        return {"train": self.train, "val": self.val, "oot": self.oot}


@dataclass(frozen=True)
class SplitPosRates:
    """三段正样本率。"""
    train: float
    val: float
    oot: float

    def to_dict(self) -> Dict[str, float]:
        return {"train": self.train, "val": self.val, "oot": self.oot}


@dataclass(frozen=True)
class SplitReport:
    """切分元信息。

    frozen + to_dict/to_markdown/summary_line 一站式渲染, 下游不必再读 dict key。
    """
    strategy: str
    oot_boundary: str
    counts: SplitCounts
    pos_rates: SplitPosRates
    ratios: SplitRatios
    time_col_used: Optional[str] = None
    train_filter_used: Optional[str] = None
    oot_filter_used: Optional[str] = None
    random_seed: Optional[int] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy={self.strategy!r} 不在合法集合 {_VALID_STRATEGIES}"
            )

    @staticmethod
    def _pos_rate(df: pd.DataFrame, target: str) -> float:
        if target not in df.columns or len(df) == 0:
            return 0.0
        return round(float((df[target] == 1).mean()), 6)

    @staticmethod
    def strategy_zh(strategy: str) -> str:
        return {
            _STRATEGY_EXPLICIT: "用户显式 filter",
            _STRATEGY_TIME: "按时间字段排序",
            _STRATEGY_RANDOM: "随机切分",
        }.get(strategy, strategy)

    @classmethod
    def from_frames(
        cls,
        *,
        strategy: str,
        train: pd.DataFrame,
        val: pd.DataFrame,
        oot: pd.DataFrame,
        target: str,
        ratios: SplitRatios,
        time_col: Optional[str],
        oot_boundary: str,
        train_filter_used: Optional[str],
        oot_filter_used: Optional[str],
        random_seed: int,
        warnings: List[str],
    ) -> "SplitReport":
        total = max(len(train) + len(val) + len(oot), 1)
        return cls(
            strategy=strategy,
            oot_boundary=oot_boundary,
            counts=SplitCounts(train=len(train), val=len(val), oot=len(oot)),
            pos_rates=SplitPosRates(
                train=cls._pos_rate(train, target),
                val=cls._pos_rate(val, target),
                oot=cls._pos_rate(oot, target),
            ),
            ratios=ratios,
            time_col_used=time_col if strategy in (_STRATEGY_TIME, _STRATEGY_EXPLICIT) else None,
            train_filter_used=train_filter_used,
            oot_filter_used=oot_filter_used,
            random_seed=random_seed,
            warnings=tuple(warnings),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_strategy": self.strategy,
            "ratios": self.ratios.to_dict(),
            "sample_counts": self.counts.to_dict(),
            "pos_rates": self.pos_rates.to_dict(),
            "time_col_used": self.time_col_used,
            "oot_boundary": self.oot_boundary,
            "train_filter_used": self.train_filter_used,
            "oot_filter_used": self.oot_filter_used,
            "random_seed": self.random_seed,
            "warnings": list(self.warnings),
        }

    def summary_line(self) -> str:
        return (
            f"[split] strategy={self.strategy} | "
            f"train={self.counts.train}(pos={self.pos_rates.train:.2%}) "
            f"val={self.counts.val}(pos={self.pos_rates.val:.2%}) "
            f"oot={self.counts.oot}(pos={self.pos_rates.oot:.2%}) | "
            f"oot_boundary={self.oot_boundary}"
        )

    def to_markdown(self) -> str:
        lines = [
            "## 数据切分",
            "",
            "- 训练范式：经典三段式 train / val / oot",
            f"- 切分策略：{self.strategy_zh(self.strategy)}",
            f"- OOT 边界：{self.oot_boundary}",
            (
                f"- 样本量：train={self.counts.train} (pos={self.pos_rates.train:.2%}) | "
                f"val={self.counts.val} (pos={self.pos_rates.val:.2%}) | "
                f"oot={self.counts.oot} (pos={self.pos_rates.oot:.2%})"
            ),
            (
                f"- 比例：{self.ratios.train:.0%}/{self.ratios.val:.0%}/{self.ratios.oot:.0%}，"
                f"随机种子 {self.random_seed if self.random_seed is not None else '-'}"
            ),
        ]
        if self.warnings:
            lines.append("")
            lines.append("> 切分警告：")
            for w in self.warnings:
                lines.append(f"> - {w}")
        lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True)
class DatasetSplits:
    """切分结果(frozen dataclass)。

    字段:
        train/val/oot: 三段 DataFrame
        features: 推断特征(可选, 调用方设 return_features=True 时填充)
        report: SplitReport(可选, 调用方设 return_meta=False 时为 None)
    """
    train: pd.DataFrame
    val: pd.DataFrame
    oot: pd.DataFrame
    features: Optional[List[str]] = None
    report: Optional[SplitReport] = None


# ============================================================
# 主切分入口
# ============================================================

def prepare_splits(
    data_path: str,
    target: str,
    time_col: Optional[str] = "busi_dt",
    train_filter: Optional[str] = None,
    test_filter: Optional[str] = None,
    oot_filter: Optional[str] = None,
    *,
    oot_ratio: float = DEFAULT_OOT_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
    label_filter: Optional[Iterable] = (0, 1),
    return_features: bool = False,
    exclude_cols: Optional[List[str]] = None,
    return_meta: bool = True,
) -> DatasetSplits:
    """加载数据并按经典三段式切分为 train / val / oot。

    切分优先级:
        1. 显式 filter(train_filter 或 oot_filter 任一给出)
        2. time_col 存在
        3. 随机切分

    Args:
        data_path: parquet 或 csv 路径
        target: 标签列名
        time_col: 时间列; 为 None 或不存在时跳过策略 2
        train_filter / oot_filter: pandas query 条件
        test_filter: 给出则并入 train_filter 作为 train_full 的显式约束, val 仍在 train_full 内按 val_ratio 切
        oot_ratio: train_full vs oot 比例, ∈ (0, 1)
        val_ratio: val 在 train_full 内的比例, ∈ (0, 0.5), 禁止为 0
        random_seed: 随机切分种子
        label_filter: 仅保留 target 在该集合内的样本; None 跳过
        return_features: 是否填充 DatasetSplits.features
        exclude_cols: 推断特征时额外排除的列
        return_meta: 是否填充 DatasetSplits.report(默认 True)

    Returns:
        DatasetSplits(train, val, oot, features, report)
    """
    ratios = _validate_ratios(oot_ratio, val_ratio)
    df = load_table(data_path, label_filter=label_filter, target=target)
    full_df = df
    warnings: List[str] = []

    strategy, train_full, oot, oot_boundary, t_filter_used, o_filter_used = (
        _split_by_strategy(
            df, time_col=time_col, train_filter=train_filter,
            oot_filter=oot_filter, test_filter=test_filter,
            oot_ratio=oot_ratio, random_seed=random_seed, warnings=warnings,
        )
    )

    train, val = _split_train_val(
        train_full,
        time_col=time_col,
        val_ratio=val_ratio,
        random_seed=random_seed,
        follow_strategy=strategy,
    )

    _validate_splits(train, val, oot, target, warnings)

    report: Optional[SplitReport] = None
    if return_meta:
        report = SplitReport.from_frames(
            strategy=strategy,
            train=train, val=val, oot=oot,
            target=target, ratios=ratios,
            time_col=time_col,
            oot_boundary=oot_boundary,
            train_filter_used=t_filter_used,
            oot_filter_used=o_filter_used,
            random_seed=random_seed, warnings=warnings,
        )

    features: Optional[List[str]] = None
    if return_features:
        features = infer_features(full_df, target, exclude_cols=exclude_cols)

    if report is not None:
        logger.info(report.summary_line())
    for w in warnings:
        logger.warning(f"[split] {w}")

    return DatasetSplits(train=train, val=val, oot=oot, features=features, report=report)


# ============================================================
# 策略分发器(替代主函数内 if/elif 链)
# ============================================================

def _split_by_strategy(
    df: pd.DataFrame,
    *,
    time_col: Optional[str],
    train_filter: Optional[str],
    oot_filter: Optional[str],
    test_filter: Optional[str],
    oot_ratio: float,
    random_seed: int,
    warnings: List[str],
) -> Tuple[str, pd.DataFrame, pd.DataFrame, str, Optional[str], Optional[str]]:
    """按入参选择切分策略, 返回
    (strategy, train_full, oot, oot_boundary, train_filter_used, oot_filter_used)。
    """
    if train_filter is not None or oot_filter is not None:
        train_full, oot, oot_boundary, t_used, o_used = _split_explicit(
            df, train_filter, oot_filter, test_filter, warnings
        )
        return _STRATEGY_EXPLICIT, train_full, oot, oot_boundary, t_used, o_used

    if time_col and time_col in df.columns:
        train_full, oot, oot_boundary = _split_by_time(df, time_col, oot_ratio)
        return _STRATEGY_TIME, train_full, oot, oot_boundary, None, None

    if time_col and time_col not in df.columns:
        warnings.append(
            f"time_col='{time_col}' 在数据中不存在, 已 fallback 到随机切分"
        )
    train_full, oot, oot_boundary = _split_random(df, oot_ratio, random_seed)
    return _STRATEGY_RANDOM, train_full, oot, oot_boundary, None, None


# ============================================================
# 比例校验 + 返回 SplitRatios
# ============================================================

def _validate_ratios(oot_ratio: float, val_ratio: float) -> SplitRatios:
    """校验 oot/val 比例, 返回 SplitRatios(train 已推导)。"""
    if not (0.0 < oot_ratio < 1.0):
        raise ValueError(f"oot_ratio 必须 ∈ (0, 1), 当前 {oot_ratio}")
    if val_ratio <= 0.0:
        raise ValueError(
            f"val_ratio 必须 > 0(经典三段式禁止关闭 val), 当前 {val_ratio}。"
            "当前仅支持三段式切分(train/test/oot), CV 模式或两段式不支持。"
        )
    if val_ratio >= 0.5:
        raise ValueError(f"val_ratio 必须 < 0.5(防止 train 过小), 当前 {val_ratio}")
    train_ratio = (1 - oot_ratio) * (1 - val_ratio)
    if train_ratio < 0.1:
        raise ValueError(
            f"切分后 train 占比仅 {train_ratio:.2%}(< 10%), "
            f"请减小 oot_ratio({oot_ratio}) 或 val_ratio({val_ratio})"
        )
    return SplitRatios(oot=oot_ratio, val=val_ratio, train=train_ratio)


# ============================================================
# query 表达式类型自适应 + 模块级 _safe_query(非内联闭包)
# ============================================================

def _adapt_query_for_dtypes(df: pd.DataFrame, expr: str) -> str:
    """根据 DataFrame 列实际类型自动修正 query 表达式。

    支持的自动修正:
        1. datetime 列上误用 .str.startswith('YYYYMM') → 日期范围比较
        2. datetime 列上误用 .str.startswith('YYYYMMDD') → 日期范围比较
        3. 整数列上误用 .str.startswith('YYYYMM') → 数值范围比较
        4. 整数列上误用 .str.startswith('YYYYMMDD') → 数值范围比较
        5. col.astype(str).str.startswith(...) → 同上述修正
        6. ~col.str.startswith(...) 取反形式同样支持
    """
    if not expr:
        return expr

    adapted = expr

    # 第一步: 去掉 .astype(str) 冗余写法(df.query 中 str 未定义)
    astype_pattern = r'(~?)\s*(\w+)\.astype\(str\)\.str\.startswith'
    def _remove_astype(m: re.Match) -> str:
        negate = m.group(1)
        col = m.group(2)
        return f"{negate}{col}.str.startswith"
    adapted = re.sub(astype_pattern, _remove_astype, adapted)

    # 第二步: 修正 .str.startswith 用于 datetime / 整数列
    pattern = r'(~?)\s*(\w+)\.str\.startswith\([\'"](\d{6,8})[\'"]\)'
    for match in re.finditer(pattern, adapted):
        negate = match.group(1)
        col = match.group(2)
        prefix = match.group(3)
        if col not in df.columns:
            continue
        is_datetime = pd.api.types.is_datetime64_any_dtype(df[col])
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if not is_datetime and not is_numeric:
            continue

        if len(prefix) == 6:
            year = int(prefix[:4])
            month = int(prefix[4:6])
            if month == 12:
                next_year, next_month = year + 1, 1
            else:
                next_year, next_month = year, month + 1
            if is_datetime:
                start = f"{prefix}01"
                end = f"{next_year:04d}{next_month:02d}01"
            else:
                start = int(f"{prefix}01")
                end = int(f"{next_year:04d}{next_month:02d}01")
        elif len(prefix) == 8:
            if is_datetime:
                start = prefix
                try:
                    dt = datetime.strptime(prefix, "%Y%m%d")
                    end = (dt + timedelta(days=1)).strftime("%Y%m%d")
                except ValueError:
                    continue
            else:
                start = int(prefix)
                end = int(prefix) + 1
        else:
            continue

        if is_datetime:
            if negate:
                replacement = f"(({col} < '{start}') or ({col} >= '{end}'))"
            else:
                replacement = f"(({col} >= '{start}') and ({col} < '{end}'))"
        else:
            if negate:
                replacement = f"(({col} < {start}) or ({col} >= {end}))"
            else:
                replacement = f"(({col} >= {start}) and ({col} < {end}))"

        adapted = adapted.replace(match.group(0), replacement)
        logger.warning(
            f"[split] 自动修正 filter: {match.group(0)!r} → {replacement!r} "
            f"(列 '{col}' 为 {df[col].dtype} 类型)"
        )

    return adapted


def _safe_query(df: pd.DataFrame, expr: str, name: str) -> pd.DataFrame:
    """对 df 跑 query, 先 _adapt_query_for_dtypes 修类型, 失败时带提示抛 ValueError。"""
    adapted_expr = _adapt_query_for_dtypes(df, expr)
    try:
        return df.query(adapted_expr)
    except Exception as e:
        err_msg = str(e)
        hint = ""
        if ".str accessor" in err_msg and "string values" in err_msg:
            hint = (
                "(提示: 该列可能是 datetime 类型而非字符串, "
                "请改用日期范围比较, 例如 'col >= \"20260101\" and col < \"20260201\"')"
            )
        elif ".dt accessor" in err_msg:
            hint = "(提示: 该列可能不是 datetime 类型, 请检查数据类型)"
        raise ValueError(f"{name} 表达式无法解析: {expr!r}, 错误: {e}{hint}") from e


# ============================================================
# 切分策略实现
# ============================================================

def _split_explicit(
    df: pd.DataFrame,
    train_filter: Optional[str],
    oot_filter: Optional[str],
    test_filter: Optional[str],
    warnings: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, str, Optional[str], Optional[str]]:
    """显式 filter 模式。返回 (train_full, oot, oot_boundary, train_filter_used, oot_filter_used)。"""
    # 处理 test_filter: 并入 train_filter 作为 train_full 的显式约束
    train_filter_eff = train_filter
    if test_filter:
        if train_filter:
            train_filter_eff = f"({train_filter}) or ({test_filter})"
        else:
            train_filter_eff = test_filter
        warnings.append(
            "test_filter 已合并进 train_full, 仍按 val_ratio 在 train_full 内切 val"
        )

    if train_filter_eff and oot_filter:
        train_full = _safe_query(df, train_filter_eff, "train_filter").reset_index(drop=False)
        oot = _safe_query(df, oot_filter, "oot_filter").reset_index(drop=False)
        overlap = set(train_full["index"]).intersection(set(oot["index"]))
        if overlap:
            warnings.append(f"train_filter 与 oot_filter 存在 {len(overlap)} 条重叠样本")
        coverage = (len(train_full) + len(oot) - len(overlap)) / max(len(df), 1)
        if coverage < 0.95:
            warnings.append(f"train+oot 覆盖率仅 {coverage:.2%}, 部分样本被丢弃")
        train_full = train_full.drop(columns=["index"]).reset_index(drop=True)
        oot = oot.drop(columns=["index"]).reset_index(drop=True)
        oot_boundary = f"oot_filter={oot_filter}"
    elif train_filter_eff and not oot_filter:
        train_full = _safe_query(df, train_filter_eff, "train_filter").reset_index(drop=True)
        oot = _safe_query(df, f"not ({train_filter_eff})", "oot_filter(auto)").reset_index(drop=True)
        oot_boundary = f"oot=NOT({train_filter_eff})"
    else:  # oot_filter only
        oot = _safe_query(df, oot_filter, "oot_filter").reset_index(drop=True)
        train_full = _safe_query(df, f"not ({oot_filter})", "train_filter(auto)").reset_index(drop=True)
        oot_boundary = f"oot_filter={oot_filter}"

    return train_full, oot, oot_boundary, train_filter_eff, oot_filter


def _split_by_time(
    df: pd.DataFrame,
    time_col: str,
    oot_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """按 time_col 唯一值切分; 同一业务日不会被切散。"""
    unique_times = sorted(df[time_col].unique())
    n = len(unique_times)
    cut = int(n * (1 - oot_ratio))
    cut = max(1, min(cut, n - 1))
    train_times = set(unique_times[:cut])
    oot_times = set(unique_times[cut:])
    train_full = df[df[time_col].isin(train_times)].reset_index(drop=True)
    oot = df[df[time_col].isin(oot_times)].reset_index(drop=True)
    oot_boundary = f"{time_col} >= {unique_times[cut]}"
    return train_full, oot, oot_boundary


def _split_random(
    df: pd.DataFrame,
    oot_ratio: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """随机切分。"""
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(df))
    cut = int(len(df) * (1 - oot_ratio))
    train_idx = perm[:cut]
    oot_idx = perm[cut:]
    train_full = df.iloc[train_idx].reset_index(drop=True)
    oot = df.iloc[oot_idx].reset_index(drop=True)
    oot_boundary = f"random seed={random_seed} (前 {1-oot_ratio:.0%} 行)"
    return train_full, oot, oot_boundary


def _split_train_val(
    train_full: pd.DataFrame,
    time_col: Optional[str],
    val_ratio: float,
    random_seed: int,
    follow_strategy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """train_full → train/val。跟随 OOT 策略:
        - explicit / time: 若 time_col 存在 → val 取末端 val_ratio
        - random / 无 time_col: val 在 train_full 内随机抽 val_ratio
    """
    use_time = (
        follow_strategy in (_STRATEGY_EXPLICIT, _STRATEGY_TIME)
        and time_col
        and time_col in train_full.columns
    )

    if use_time:
        unique_times = sorted(train_full[time_col].unique())
        n = len(unique_times)
        cut = int(n * (1 - val_ratio))
        cut = max(1, min(cut, n - 1))
        train_times = set(unique_times[:cut])
        val_times = set(unique_times[cut:])
        train = train_full[train_full[time_col].isin(train_times)].reset_index(drop=True)
        val = train_full[train_full[time_col].isin(val_times)].reset_index(drop=True)
    else:
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(len(train_full))
        cut = int(len(train_full) * (1 - val_ratio))
        train_idx = perm[:cut]
        val_idx = perm[cut:]
        train = train_full.iloc[train_idx].reset_index(drop=True)
        val = train_full.iloc[val_idx].reset_index(drop=True)

    return train, val


def _validate_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    oot: pd.DataFrame,
    target: str,
    warnings: List[str],
) -> None:
    """切分后校验: 最小样本量 / 正负样本 / val 早停稳定性 / 小数据提示。"""
    for name, sub in (("train", train), ("val", val), ("oot", oot)):
        n = len(sub)
        if n < 10:
            raise ValueError(
                f"{name} 集合仅 {n} 条样本, 过小无法训练/评估; 请增大对应比例或检查 filter"
            )
        if n < 100:
            warnings.append(f"{name} 仅 {n} 条样本, 建议增大该集合")
        if target in sub.columns:
            pos = int((sub[target] == 1).sum())
            neg = int((sub[target] == 0).sum())
            if pos == 0 or neg == 0:
                raise ValueError(
                    f"{name} 集合正/负样本失衡: pos={pos}, neg={neg}; 请调整比例或 filter"
                )

    val_n = len(val)
    val_pos = int((val[target] == 1).sum()) if target in val.columns else 0
    if val_n < 500 or val_pos < 30:
        warnings.append(
            f"val 样本量偏小(n={val_n}, pos={val_pos}), 早停信号可能不稳定, "
            "建议加大 val_ratio 或换数据"
        )

    train_full_n = len(train) + len(val)
    if train_full_n < 10000:
        warnings.append(
            f"train_full 仅 {train_full_n} 条, 小数据场景建议未来切换到 CV 模式(本次未实现)"
        )


# ============================================================
# 报告渲染辅助(支持 dict / SplitReport / None 三种入参)
# ============================================================

def format_split_md(meta: Union[SplitReport, Dict[str, Any], None]) -> str:
    """渲染 Markdown 的"数据切分"小节。

    Args:
        meta: SplitReport / dict / None。dict 入参按
            {sample_counts, pos_rates, oot_boundary} 字段渲染。

    Returns:
        Markdown 文本, meta 为 None 或空 dict 时返回空串。
    """
    if meta is None:
        return ""
    if isinstance(meta, SplitReport):
        return meta.to_markdown()
    if not meta:
        return ""

    # dict 入参路径: 降级到 SplitReport.to_markdown 渲染
    cnt = meta.get("sample_counts", {})
    pos = meta.get("pos_rates", {})
    ratios = meta.get("ratios", {})
    warnings_list = meta.get("warnings", [])
    strategy = meta.get("split_strategy", "")
    lines = [
        "## 数据切分",
        "",
        "- 训练范式: 经典三段式 train / val / oot",
        f"- 切分策略: {SplitReport.strategy_zh(strategy) if strategy else '-'}",
        f"- OOT 边界: {meta.get('oot_boundary', '-')}",
        (
            f"- 样本量: train={cnt.get('train', '-')} (pos={pos.get('train', 0):.2%}) | "
            f"val={cnt.get('val', '-')} (pos={pos.get('val', 0):.2%}) | "
            f"oot={cnt.get('oot', '-')} (pos={pos.get('oot', 0):.2%})"
        ),
        (
            f"- 比例: {ratios.get('train', 0):.0%}/{ratios.get('val', 0):.0%}/{ratios.get('oot', 0):.0%}, "
            f"随机种子 {meta.get('random_seed', '-')}"
        ),
    ]
    if warnings_list:
        lines.append("")
        lines.append("> 切分警告:")
        for w in warnings_list:
            lines.append(f"> - {w}")
    lines.append("")
    return "\n".join(lines)
