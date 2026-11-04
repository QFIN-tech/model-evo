"""HPO trial 运行时(v11):trial 内完整超参搜索,一个 trial 一行 leaderboard。

粒度修复:旧模式一个网格点=一条 leaderboard 行=吃掉 max_trials 一格
+一次完整评估;hpo 动作把搜索压进 trial 内部——
- train 时间序后 20% 切 inner-validation,搜索只看 inner;
- best_params 在 train 全量(含 inner)重训;
- outer validation 只评一次 → noise gate 输入仍是干净的单次比较。

采样器:optuna 在→TPESampler;缺失→同空间随机搜索(纯 numpy,
行为对齐,无新依赖)。搜索空间声明见 config/trial.yaml 的 hpo 节
(new-trial --action hpo --search-space space.yaml 注入)。

v2.15.0(req-t7-rand400w run3 三轮 HPO 复盘回写):同一 inner 目标上
连开 3 个独立 study(84 个 config),每轮从零开始、前 10 点是同一组
seed 固定的随机分位数,轮间"学习"全靠 Agent 手改边界;TPE 引导段
相对随机段的增益仅 +0.00008 / +0.00023 / −0.00004。本版把这些变成
一个 trial 内的确定性机制:
- **跨 trial 热启动**:inner 目标签名(特征列/行数/标签和/容量/指标)
  一致的历史观测直接 add_trial 进 study(`warm_start`);父代参数
  enqueue 为基准点(`enqueue`);
- **采样器**:多变量 TPE(lr×树数、λ×colsample 的耦合可建模)、
  seed 按 trial 派生(不再每个 study 复用同一组随机启动点)、启动点数
  随预算缩放;并行 config(`n_jobs`,constant_liar 防重复);
- **边界自适应**:前 K 名贴边时在**同一 trial 内**外扩该维并继续搜
  (所有已有观测带入新空间),替代"多轮手工追边界";
- **逐 config 遥测与诊断**:best_iteration / cap 撞顶 / 拟合秒数,
  以及 top-k 分差、TPE 相对随机增益、边界命中,落 hpo-study.json;
- **多保真逐级减半**(可选,`fidelities`):低保真筛选、高保真复评;
- **lr 收缩模式**(`plan_lr_mode`):lr 是算力旋钮而非有内部最优的
  超参 —— 固定较高 lr 搜结构/正则,最终重训按比例收缩 lr 并放大 cap。
"""

from __future__ import annotations

import math
import os
import threading
import time
import warnings
import zlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .logging_utils import dump_json, get_logger, load_json

logger = get_logger(__name__)

#: inner-validation 占 train 的比例(时间序尾部)
INNER_FRACTION = 0.2

#: 前缀正样本率相对全体偏差的安全阈值(v17.1;见 prefix_sample_safe)
#: ——超阈即放弃时间前缀,降级分层随机。manifest 的
#: _prefix_label_check 落盘该值供调用方读。
PREFIX_DEV_THRESHOLD = 0.2

#: v2.15.0:lr 收缩模式的默认值(plan_lr_mode)。搜索期固定 lr,最终
#: 重训 lr = SEARCH_LR × SHRINK_FACTOR、cap 按 1/SHRINK_FACTOR 放大。
#: run3 实测 inner 目标与 log(lr) 的 Spearman = −0.89(lr 越低越好且
#: 无内部最优),三轮最优 lr 0.024→0.0136→0.0099 一路顶下界 ——
#: 0.05×0.25=0.0125 正落在这段实测最优带内。
DEFAULT_SEARCH_LR = 0.05
DEFAULT_LR_SHRINK_FACTOR = 0.25
#: auto 模式下 inner 拟合面行数 ≥ 该值才启用收缩(小数据单 config 本就便宜)
DEFAULT_LR_SHRINK_MIN_ROWS = 200_000
#: search 模式下边界外扩时 lr 的硬下界(再低单 config 成本失控)
DEFAULT_LR_FLOOR = 0.005

#: v2.15.0:边界自适应参数。已评 config 数达到预算的这些比例时检查一次;
#: 前 K 名在(对数)归一化坐标上的中位数距某边界 < EPS 即判贴边。
#: 观测数不足 EXPAND_MIN_OBS 时不判(前 K 名几乎全是随机启动点,无意义)。
EXPAND_CHECKPOINTS = (0.35, 0.55, 0.75)
EXPAND_MIN_OBS = 10
EXPAND_TOP_K = 5
EXPAND_EDGE_EPS = 0.05
EXPAND_MAX_PER_PARAM = 2
#: 外扩步长:log 维 ×/÷ 该倍数;线性/整数维扩当前宽度的该比例
EXPAND_LOG_FACTOR = 2.5
EXPAND_LINEAR_FRACTION = 0.25

#: 只对语义已知的参数做外扩(未知参数的合法域无从判断,宁可不扩)。
#: 值 = (硬下界, 硬上界);learning_rate 的下界由 lr_floor 决定。
_PARAM_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "learning_rate": (None, 1.0),
    "eta": (None, 1.0),
    "subsample": (0.05, 1.0),
    "colsample_bytree": (0.05, 1.0),
    "colsample_bylevel": (0.05, 1.0),
    "colsample_bynode": (0.05, 1.0),
    "feature_fraction": (0.05, 1.0),
    "bagging_fraction": (0.05, 1.0),
    "rsm": (0.05, 1.0),
    "max_depth": (1, 16),
    "depth": (1, 16),
    "num_leaves": (2, 4096),
    "min_child_weight": (1e-3, 1e4),
    "min_child_samples": (1, 1e5),
    "min_data_in_leaf": (1, 1e5),
    "reg_lambda": (1e-8, 1e3),
    "reg_alpha": (1e-8, 1e3),
    "lambda_l1": (1e-8, 1e3),
    "lambda_l2": (1e-8, 1e3),
    "l2_leaf_reg": (1e-8, 1e3),
    "gamma": (0.0, 100.0),
    "min_split_gain": (0.0, 100.0),
}

#: v2.14.1(2x 线 P2 #19)多保真逐级减半,v2.15.0 移植:inner 拟合面按行
#: 比例分三档,每档保留前 1/eta 进入下一档。27 个 config ≈ 9 个全量
#: config 的算力。可选(hpo.fidelity: successive_halving),默认关闭 ——
#: 低保真排名在小样本上偏好强正则,收益/风险需按任务权衡。
DEFAULT_FIDELITIES = (1.0 / 9.0, 1.0 / 3.0, 1.0)
DEFAULT_ETA = 3
#: inner 拟合面行数低于此值时不启用多保真(小数据的固定开销占主导,
#: 低档排名也不可靠)
MULTI_FIDELITY_MIN_ROWS = 60000
#: 最低档至少这么多行(太小的子样本对超参排序无判别力)
MULTI_FIDELITY_MIN_SUBSET = 5000

#: inner 目标签名的实现版本:inner 打分口径变化(如 v2.15 起 inner 用
#: 第一段早停模型直接打分、不再全量重拟合)时递增,旧观测自动失配。
INNER_SIGNATURE_VERSION = "v2.15-stage1"

#: 支持的分布类型 → (采样器, optuna suggest 方法名)
_SPACE_TYPES = {
    "float": ("_sample_float", "suggest_float"),
    "int": ("_sample_int", "suggest_int"),
    "categorical": ("_sample_categorical", "suggest_categorical"),
}


def _sample_float(rng: np.random.Generator, spec: dict) -> float:
    lo, hi = float(spec["low"]), float(spec["high"])
    if spec.get("log"):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


def _sample_int(rng: np.random.Generator, spec: dict) -> int:
    return int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))


def _sample_categorical(rng: np.random.Generator, spec: dict) -> Any:
    return spec["choices"][int(rng.integers(0, len(spec["choices"])))]


#: 随机搜索采样函数表(type → 函数对象)。v14.1:原实现直接调用
#: _SPACE_TYPES[typ][0](字符串)导致 TypeError;采样器函数改为经本表解析。
_SAMPLE_FNS = {
    "float": _sample_float,
    "int": _sample_int,
    "categorical": _sample_categorical,
}


def _sample_one(rng: np.random.Generator, space: dict) -> dict:
    out = {}
    for k, spec in space.items():
        typ = str(spec.get("type", "float")).lower()
        if typ not in _SPACE_TYPES:
            raise ValueError(f"搜索空间 {k} 的 type={typ} 不支持"
                             f"(支持 {sorted(_SPACE_TYPES)})")
        # 采样器函数名与 optuna suggest 方法名共用一个表,取第 0 元素
        # 是**字符串**("_sample_float")而非函数对象——需经 getattr 解析。
        # 原实现直接调用字符串导致 TypeError('str' object is not callable)。
        # 修复(v14.1):统一走模块级采样函数表 _SAMPLE_FNS。
        out[k] = _SAMPLE_FNS[typ](rng, spec)
    return out


def _suggest_one(trial_obj: Any, space: dict) -> dict:
    """optuna Trial 侧的 define-by-run 采样。"""
    out = {}
    for k, spec in space.items():
        typ = str(spec.get("type", "float")).lower()
        method = getattr(trial_obj, _SPACE_TYPES[typ][1])
        if typ == "categorical":
            out[k] = method(k, spec["choices"])
        else:
            out[k] = method(
                k, spec["low"], spec["high"],
                log=bool(spec.get("log", False)))
    return out


def _space_distribution(spec: dict) -> Any:
    """空间声明 → optuna distribution(断点续跑回放 create_trial 用)。

    回放必须给出与 define-by-run 采样**逐位一致**的分布对象,否则
    optuna 的 add_trial 会拒绝(参数值不在分布支撑集内),或更糟:
    静默接受但 TPE 先验被污染。suggest_float/suggest_int 在
    log=True 时建的是 Float/IntDistribution(log=True),不是
    UniformDistribution —— 这里必须同口径。
    """
    from optuna.distributions import (CategoricalDistribution,
                                      FloatDistribution,
                                      IntDistribution)

    typ = str(spec.get("type", "float")).lower()
    if typ == "categorical":
        return CategoricalDistribution(choices=list(spec["choices"]))
    if typ == "int":
        return IntDistribution(int(spec["low"]), int(spec["high"]),
                               log=bool(spec.get("log", False)))
    return FloatDistribution(float(spec["low"]), float(spec["high"]),
                             log=bool(spec.get("log", False)))


def inner_split(train: pd.DataFrame, *, fraction: float = INNER_FRACTION,
                time_column: str | None = None) -> tuple[int, int]:
    """时间序尾部切 inner-validation;返回 (inner_end, n_train)。

    train 已按时间排序时直接尾部切;无时间列时按行序(等价口径,
    因为 splits 落盘时已按时间切)。返回边界索引供上层切片。
    """
    n = len(train)
    inner_start = int(n * (1.0 - fraction))
    return inner_start, n


def prefix_sample_safe(n: int, frac: float, *, seed: int = 20260814,
                       ordering: str | None = None,
                       prefix_rate_rel_dev: float | None = None,
                       ) -> np.ndarray:
    """前缀采样防护(v12,P0-1 配套;v17.1 补标签偏差闸)。

    v12 口径:ordering 标注已排序时返回时间序前缀;未排序/未知时
    降级为分层随机抽样(分布保真,不受 loader 原始行序的标签聚集
    影响)。capacity.py 的样本量曲线、模板的 halving quick 轮共用
    本函数。

    v17.1 修复(实测 req-t7 drawdown 事故):**时间排序 ≠ 标签分布
    安全**。上游 loader 按天落表、天内按 label 聚集时,排序后前缀
    的正样本率可偏离全体 +207%(manifest 的 prefix_rate_rel_dev=
    2.07,validation 前缀 100% 正样本)——时间前缀会把这种"天内
    标签有序"直接带进子样本,产出假 AUC。修复:ordering 已排序
    **且** manifest 记录的前缀偏差在阈值内才用时间前缀;否则降级
    分层随机。阈值 0.2 = 前缀正样本率偏离全体 20% 以内(v12 的
    train 实测 0.13,正常数据远低于此)。

    prefix_rate_rel_dev 由调用方从 splits manifest 读(键名与
    _prefix_label_check 落盘一致);缺失时保持 v12 行为(仅看
    ordering),不给旧调用方引入静默变化。
    """
    k = max(1, int(n * frac))
    dev_ok = (prefix_rate_rel_dev is None
              or float(prefix_rate_rel_dev) <= PREFIX_DEV_THRESHOLD)
    if (ordering is not None and ordering.startswith("sorted_by:")
            and dev_ok):
        return np.arange(k)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k, replace=False)



# ---------------------------------------------------------------------------
# v2.15.0:搜索引擎辅助(seed 派生 / 采样器构造 / 空间工具 / 诊断)
# ---------------------------------------------------------------------------

def derive_seed(seed: int, salt: str | None = None) -> int:
    """按 trial 派生采样器 seed(v2.15.0)。

    旧实现所有 HPO trial 都用 cfg.seed=20260814:每个 study 的前 10 个
    随机启动点是**同一组分位数向量**,只是映射进改过的空间(run3 三轮
    的第 7 个点恒为 run2 最优点;第 3 轮最终冠军就是这个随机点)——
    启动段每轮重复探索同一批位置。salt=trial_id 时同一 trial 断点续跑
    仍可复现,不同 trial 之间启动点互不相同。
    """
    s = int(seed) % (2 ** 31 - 1)
    if salt:
        s = (s + zlib.crc32(str(salt).encode("utf-8"))) % (2 ** 31 - 1)
    return int(s)


def _n_startup_for(budget: int) -> int:
    """随机启动点数:预算的 1/4,夹在 [5, 10](小预算全随机)。

    旧实现恒 min(10, budget):16 个 config 的 study 有 10 个是纯随机
    (62.5%),TPE 只引导了 6 个。热启动观测同样计入 optuna 的启动计数,
    历史观测足够多时直接进入引导段。
    """
    budget = max(1, int(budget))
    return int(min(budget, max(5, min(10, budget // 4))))


def _make_tpe(seed: int, n_startup: int, *, multivariate: bool = True,
              constant_liar: bool = False):
    """TPE 采样器构造(版本兼容):多变量 + 可选 constant_liar。"""
    from optuna.samplers import TPESampler

    kw: dict = {"seed": int(seed), "n_startup_trials": int(n_startup)}
    if multivariate:
        kw["multivariate"] = True
    kw["constant_liar"] = bool(constant_liar)
    with warnings.catch_warnings():
        # multivariate/constant_liar 在部分版本标注 ExperimentalWarning
        warnings.simplefilter("ignore")
        try:
            return TPESampler(**kw)
        except TypeError:
            kw.pop("constant_liar", None)
            kw.pop("multivariate", None)
            return TPESampler(**kw)


def _unpack_inner(raw: Any) -> tuple[float, dict]:
    """fit_score_inner 的返回值归一:float 或 {"value": float, ...遥测}。"""
    if isinstance(raw, dict):
        extra = {k: v for k, v in raw.items() if k != "value"
                 and isinstance(v, (int, float, str, bool, type(None)))}
        return float(raw.get("value")), extra
    return float(raw), {}


def _key_one(params: dict) -> tuple:
    """参数逐键等价键(与空间声明顺序无关,浮点按 repr 定死)。"""
    return tuple(sorted((str(k), repr(v)) for k, v in params.items()))


def _spec_bounds(spec: dict) -> tuple[float, float]:
    return float(spec["low"]), float(spec["high"])


def point_in_space(params: dict, space: dict) -> bool:
    """params 是否恰好覆盖 space 的全部键且值落在支撑集内。

    热启动/回放进 optuna 的点必须满足此条:缺键会让多变量 TPE 的交集
    空间塌缩,越界会被 add_trial 拒绝(或更糟——静默污染先验)。
    """
    if not isinstance(params, dict) or set(params) != set(space):
        return False
    for k, spec in space.items():
        v = params[k]
        typ = str(spec.get("type", "float")).lower()
        if typ == "categorical":
            if v not in list(spec.get("choices") or []):
                return False
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        lo, hi = _spec_bounds(spec)
        if not (lo - 1e-12 <= fv <= hi + 1e-12):
            return False
        if typ == "int" and int(round(fv)) != fv:
            return False
    return True


def clip_to_space(params: dict | None, space: dict) -> dict:
    """把一个(可能来自父代/历史的)参数点裁进 space;不在 space 的键丢弃。

    用于 enqueue 基准点:父代参数可能落在搜索空间之外(如默认 lr=0.05
    而空间下界 0.06),裁到最近边界仍是有意义的"基准附近"观测;
    categorical 值不在 choices 时丢弃该键(由采样器补)。
    """
    out: dict = {}
    for k, v in (params or {}).items():
        spec = space.get(k)
        if spec is None:
            continue
        typ = str(spec.get("type", "float")).lower()
        if typ == "categorical":
            if v in list(spec.get("choices") or []):
                out[k] = v
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        lo, hi = _spec_bounds(spec)
        fv = min(max(fv, lo), hi)
        out[k] = int(round(fv)) if typ == "int" else float(fv)
    return out


def _position(v: float, spec: dict) -> float | None:
    """值在空间里的归一化位置 [0,1](log 维按对数刻度)。"""
    typ = str(spec.get("type", "float")).lower()
    if typ == "categorical":
        return None
    lo, hi = _spec_bounds(spec)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if spec.get("log") and lo > 0 and hi > lo and fv > 0:
        return float((math.log(fv) - math.log(lo))
                     / (math.log(hi) - math.log(lo)))
    if hi > lo:
        return float((fv - lo) / (hi - lo))
    return None


def _ranked(obs: list[dict], direction_up: bool) -> list[dict]:
    good = [o for o in obs if isinstance(o.get("inner_metric"), (int, float))
            and np.isfinite(float(o["inner_metric"]))]
    return sorted(good, key=lambda o: float(o["inner_metric"]),
                  reverse=bool(direction_up))


def boundary_hits(observations: list[dict], space: dict, direction_up: bool,
                  *, top_k: int = EXPAND_TOP_K,
                  eps: float = EXPAND_EDGE_EPS) -> dict:
    """前 top_k 名在每个数值维上的位置;中位数贴边(< eps)即命中。

    run3 的边界追逐全靠人眼:R2 最优 colsample 0.565 贴下界 0.55、
    λ=17.9 贴上界 20 → 手工开第 3 轮;R3 的前 5 名又贴住新上界 λ=50
    (位置 0.86~1.00)与新下界 colsample=0.40(0.00~0.10)。这里把"贴边"
    变成机器判据,供同 trial 内自动外扩与事后诊断共用。
    """
    top = _ranked(observations, direction_up)[:max(1, int(top_k))]
    out: dict = {}
    if len(top) < min(3, max(1, int(top_k))):
        return out
    for k, spec in space.items():
        pos = [p for p in (_position(o["params"].get(k), spec)
                           for o in top if k in o.get("params", {}))
               if p is not None]
        if not pos:
            continue
        med = float(np.median(pos))
        side = None
        if med <= eps:
            side = "low"
        elif med >= 1.0 - eps:
            side = "high"
        if side:
            out[k] = {"side": side,
                      "top_k_positions": [round(p, 3) for p in pos]}
    return out


#: v2.16.2:hurdle(分类段 P(y>0) × 条件回归段 E[y|y>0])HPO 的分段键前缀。
#: 无前缀的键两段共用;`clf__`/`reg__` 只作用于对应子模型。
HURDLE_SEGMENT_PREFIXES = ("clf__", "reg__")


def param_base_name(key: str) -> str:
    """去掉 hurdle 分段前缀后的参数名(`clf__max_depth` → `max_depth`)。"""
    k = str(key)
    for p in HURDLE_SEGMENT_PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def expand_space(space: dict, hits: dict, expansions: list[dict], *,
                 lr_floor: float = DEFAULT_LR_FLOOR,
                 max_per_param: int = EXPAND_MAX_PER_PARAM
                 ) -> tuple[dict, list[dict]]:
    """按 boundary_hits 外扩贴边维(只扩语义已知的参数,夹在硬界内)。

    返回 (新空间, 本次变更列表)。外扩只放宽、不收窄 —— 已有观测在新
    空间里仍合法,可整体带入下一段 study。
    """
    new = {k: dict(v) for k, v in space.items()}
    done: dict[str, int] = {}
    for e in expansions or []:
        done[e.get("param")] = done.get(e.get("param"), 0) + 1
    changes: list[dict] = []
    for k, h in (hits or {}).items():
        # v2.16.2:hurdle 两段的分段键(clf__max_depth / reg__reg_lambda)按
        # 去前缀后的参数语义取硬界;键名本身(含前缀)仍是空间里的维度
        kb = param_base_name(k)
        if kb not in _PARAM_LIMITS or done.get(k, 0) >= int(max_per_param):
            continue
        spec = new[k]
        typ = str(spec.get("type", "float")).lower()
        if typ == "categorical":
            continue
        lim_lo, lim_hi = _PARAM_LIMITS[kb]
        if kb in ("learning_rate", "eta"):
            lim_lo = float(lr_floor)
        lo, hi = _spec_bounds(spec)
        side = h.get("side")
        if spec.get("log") and lo > 0:
            cand = lo / EXPAND_LOG_FACTOR if side == "low" \
                else hi * EXPAND_LOG_FACTOR
        else:
            width = max(hi - lo, 1e-12)
            step = width * EXPAND_LINEAR_FRACTION
            if typ == "int":
                step = max(1.0, round(step))
            cand = lo - step if side == "low" else hi + step
        if side == "low" and lim_lo is not None:
            cand = max(cand, float(lim_lo))
        if side == "high" and lim_hi is not None:
            cand = min(cand, float(lim_hi))
        if typ == "int":
            cand = float(int(math.floor(cand)) if side == "low"
                         else int(math.ceil(cand)))
        old = lo if side == "low" else hi
        if (side == "low" and cand >= old) or (side == "high" and cand <= old):
            continue  # 已顶硬界,无从外扩
        if spec.get("log") and cand <= 0:
            continue
        key = "low" if side == "low" else "high"
        spec[key] = int(cand) if typ == "int" else float(f"{cand:.6g}")
        changes.append({"param": k, "side": side, "from": old,
                        "to": spec[key],
                        "top_k_positions": h.get("top_k_positions")})
    return new, changes


def hpo_diagnostics(observations: list[dict], space: dict,
                    direction_up: bool, n_startup: int) -> dict:
    """study 诊断摘要(v2.15.0,事后审计与 planner 消费)。

    - top5_spread:前 5 名分差(远小于 inner 配对噪声时,继续搜多半在
      噪声里挑点);
    - tpe_gain_over_random:本 trial 新评 config 中,引导段最优 − 随机
      启动段最优(run3 三轮依次 +0.00008/+0.00023/−0.00004:TPE 几乎
      没贡献,增益来自手工改边界);
    - boundary_hits:最终空间上仍贴边的维(非空 = 最优区可能仍在界外);
    - cap_bound_configs:inner 早停 patience 未到期就撞 cap 的 config 数。
    """
    ranked = _ranked(observations, direction_up)
    diag: dict = {"n_observations": len(ranked)}
    if len(ranked) >= 5:
        diag["top5_spread"] = round(abs(float(ranked[0]["inner_metric"])
                                        - float(ranked[4]["inner_metric"])), 8)
    own_new = [o for o in observations
               if o.get("origin") in ("startup", "guided", "enqueued")]
    rnd = [o for o in own_new if o.get("origin") == "startup"]
    gui = [o for o in own_new if o.get("origin") == "guided"]
    pick = max if direction_up else min
    if rnd and gui:
        diag["tpe_gain_over_random"] = round(
            (pick(float(o["inner_metric"]) for o in gui)
             - pick(float(o["inner_metric"]) for o in rnd))
            * (1.0 if direction_up else -1.0), 8)
    diag["n_startup_random"] = len(rnd)
    diag["n_guided"] = len(gui)
    diag["boundary_hits"] = boundary_hits(observations, space, direction_up)
    diag["cap_bound_configs"] = sum(1 for o in observations
                                    if o.get("cap_bound") is True)
    if ranked:
        diag["best_origin"] = ranked[0].get("origin")
    return diag


# ---------------------------------------------------------------------------
# v2.15.0:inner 目标签名 + 热启动 + lr 模式 + 并行度(模板与 harness 共用)
# ---------------------------------------------------------------------------

def inner_signature(X_fit, y_fit, X_in, y_in, extra: dict | None = None) -> str:
    """inner 目标的确定性签名:同签名 ⇒ 同一目标函数,观测可跨 trial 复用。

    入键:特征列(有序)、inner 拟合/验证行数与行索引指纹、标签和、
    以及调用方给的口径项(算法、指标、cap、早停轮数、lr 模式、
    INNER_SIGNATURE_VERSION)。run3 的 016/018/021 三轮恰好同面、同
    切分、同 300k 抽样 —— 本可共享 84 个观测,却开了 3 个独立 study。
    """
    import hashlib

    h = hashlib.sha256()
    h.update(INNER_SIGNATURE_VERSION.encode())
    h.update(("|".join(map(str, list(getattr(X_fit, "columns", []))))
              ).encode("utf-8"))
    for frame in (X_fit, X_in):
        h.update(f"|n={len(frame)}".encode())
        idx = getattr(frame, "index", None)
        if idx is not None and len(idx):
            try:
                arr = np.asarray(idx, dtype="int64")
                h.update(arr[:: max(1, len(arr) // 4096)].tobytes())
                h.update(np.asarray([arr[0], arr[-1], arr.sum()],
                                    dtype="int64").tobytes())
            except (TypeError, ValueError):
                h.update(str(list(idx[:64])).encode())
    for yy in (y_fit, y_in):
        try:
            h.update(f"|ysum={float(np.nansum(np.asarray(yy, dtype='float64'))):.6f}"
                     .encode())
        except (TypeError, ValueError):
            pass
    for k in sorted((extra or {})):
        h.update(f"|{k}={extra[k]!r}".encode("utf-8"))
    return h.hexdigest()[:24]


def collect_warm_start(run_dir: Path, *, algorithm: str | None = None,
                       exclude: tuple[str, ...] = ()) -> list[dict]:
    """收集 run 内带 inner 签名的历史 HPO 观测(new-trial 创建期调用)。

    只收 hpo-study.json 记了 `inner_signature` 的 study(v2.15 起才写);
    是否真能复用由 trial 运行期按签名精确比对(load_warm_start),这里
    只按算法预筛,避免把不相干的 study 塞进 config。
    """
    out: list[dict] = []
    tdir = Path(run_dir) / "trials"
    if not tdir.exists():
        return out
    for st_path in sorted(tdir.glob("trial-*/hpo-study.json")):
        tid = st_path.parent.name
        if tid in exclude:
            continue
        try:
            st = load_json(st_path) or {}
        except Exception:  # noqa: BLE001
            continue
        sig = st.get("inner_signature")
        if not sig:
            continue
        if algorithm and st.get("algorithm") and \
                str(st.get("algorithm")).lower() != str(algorithm).lower():
            continue
        # v2.17.2(A1):逐 config 最优轮数随热启动观测带过去(热启动点胜出时
        # lr-shrink 重训仍能按搜索期最优轮数推 cap / lr)
        pts = [{"params": t["params"], "inner_metric": t["inner_metric"],
                **({"best_iteration": t["best_iteration"]}
                   if _as_int(t.get("best_iteration")) else {})}
               for t in (st.get("trials") or [])
               if isinstance(t, dict) and t.get("params")
               and isinstance(t.get("inner_metric"), (int, float))
               and t.get("origin") != "warm"]
        if pts:
            out.append({"source_trial": tid, "signature": sig,
                        "algorithm": st.get("algorithm"),
                        "lr_mode": st.get("lr_mode"), "trials": pts})
    return out


def load_warm_start(path: Path | None, signature: str | None) -> list[dict]:
    """读 config/hpo-warm-start.json,只保留签名与本 trial 完全一致的观测。"""
    if path is None or not signature or not Path(path).exists():
        return []
    try:
        blob = load_json(Path(path)) or {}
    except Exception:  # noqa: BLE001
        return []
    entries = blob.get("sources") if isinstance(blob, dict) else blob
    out: list[dict] = []
    for e in entries or []:
        if not isinstance(e, dict) or e.get("signature") != signature:
            continue
        for t in e.get("trials") or []:
            if isinstance(t, dict) and t.get("params") is not None:
                out.append({"params": t["params"],
                            "inner_metric": t.get("inner_metric"),
                            "source": e.get("source_trial"),
                            **({"best_iteration": t["best_iteration"]}
                               if _as_int(t.get("best_iteration")) else {})})
    return out


#: lr 维的参数名(去掉 hurdle 分段前缀后比较;hurdle_lr 在 hurdle 段已归一)
_LR_PARAM_NAMES = ("learning_rate", "eta")


def plan_lr_mode(space: dict, hpo_cfg: dict | None, n_inner_rows: int) -> dict:
    """lr 模式决策(v2.15.0 / v2.16.3):search(lr 进搜索空间)| shrink(固定 lr 搜)。

    shrink:搜索空间去掉 lr 维,inner 拟合固定 search_lr(树少、单
    config 快数倍);最终重训 lr = search_lr × shrink_factor,cap 按
    1/shrink_factor 放大。auto:inner 拟合面行数 ≥ lr_shrink_min_rows
    且空间里有 lr 维时取 shrink,否则 search。

    v2.16.3(run4 regression,与 D-011 同批):**显式 shrink 不得被静默改成
    search**。旧实现 ``if mode != "shrink" or not lr_key: return search`` 有两个
    形态都会静默回落:
      (a) 空间里没有 lr 维 —— gen-05 trial-022:``--hpo-lr-mode shrink
          --hpo-lr-shrink-factor 0.5``(假设写明"lr 收缩 0.05→0.025 重训"),
          空间只有结构/正则维,运行期判 search:inner 用父代 lr 0.025(比
          search_lr 0.05 慢一倍)、重训不收缩,trial.yaml 仍写 ``lr_mode:
          shrink``,创建期零提示;
      (b) lr 维带 hurdle 分段前缀(``clf__learning_rate``)—— 旧检测只认
          无前缀键,lr_key=None → search,lr 维**留在空间里被搜索**。
    现在:
    - 显式 shrink:恒 shrink。空间里的 lr 维(含 ``clf__``/``reg__`` 前缀)全部
      移出,inner 固定 ``search_lr``、重训 ``search_lr × factor``、cap × 1/factor;
      空间本无 lr 维时同样按此执行(lr_key 取 ``learning_rate``);
    - auto:维持 v2.15 语义(有 lr 维且 inner 行数 ≥ 阈值 → shrink,否则
      search),选择理由写 ``note``;
    - search:lr 维(若有)进搜索空间;
    - 返回值恒带 ``requested_mode`` 与 ``note``(请求 ≠ 生效、lr 维被移除、
      auto 的裁定理由),调用方在创建期与运行期 WARNING,并经
      :func:`record_lr_mode` 写进 trial.yaml / hpo-study.json / metrics。
    """
    cfg = dict(hpo_cfg or {})
    requested = str(cfg.get("lr_mode") or "search").strip().lower()
    mode = requested
    lr_dims = [k for k in space if param_base_name(k) in _LR_PARAM_NAMES]
    lr_key = next((k for k in _LR_PARAM_NAMES if k in space), None)
    min_rows = int(cfg.get("lr_shrink_min_rows") or DEFAULT_LR_SHRINK_MIN_ROWS)
    search_lr = float(cfg.get("search_lr") or DEFAULT_SEARCH_LR)
    note = None
    if mode == "auto":
        mode = "shrink" if (lr_dims and int(n_inner_rows) >= min_rows) \
            else "search"
        if not lr_dims:
            note = ("auto:搜索空间没有 lr 维 → search(lr 固定为父代/配置值,"
                    "重训不收缩)")
        elif mode != "shrink":
            note = (f"auto:inner 拟合面 {int(n_inner_rows)} 行 < 阈值 "
                    f"{min_rows} → search(lr 进搜索空间)")
        else:
            note = (f"auto:inner 拟合面 {int(n_inner_rows)} 行 ≥ 阈值 "
                    f"{min_rows} → shrink(lr 维 {lr_dims} 移出搜索空间,"
                    f"inner 固定 lr={search_lr:g})")
    elif mode not in ("search", "shrink"):
        note = f"未知 lr_mode={requested!r},按 search 处理"
        mode = "search"
    if mode != "shrink":
        return {"mode": "search", "space": space, "lr_key": lr_key,
                "requested_mode": requested, "note": note,
                # 请求 ≠ 生效(auto/未知值 → search)才需要 WARNING
                "warn": requested != "search"}
    factor = float(cfg.get("lr_shrink_factor") or DEFAULT_LR_SHRINK_FACTOR)
    factor = min(max(factor, 0.05), 1.0)
    if requested == "shrink":
        if lr_dims:
            note = (f"显式 lr_mode=shrink:搜索空间的 lr 维 {lr_dims} 已移出"
                    f"(inner 固定 lr={search_lr:g},重训 lr="
                    f"{search_lr * factor:g});如需搜索 lr 请显式 "
                    "--hpo-lr-mode search")
        else:
            note = (f"显式 lr_mode=shrink:搜索空间本无 lr 维 —— inner 固定 "
                    f"lr={search_lr:g}(不再沿用父代 lr),重训 lr="
                    f"{search_lr * factor:g}、cap×{1.0 / factor:g}")
    return {"mode": "shrink", "lr_key": lr_key or "learning_rate",
            "search_lr": search_lr,
            "refit_lr": round(search_lr * factor, 8), "factor": factor,
            "refit_cap_scale": round(1.0 / factor, 6),
            "requested_mode": requested, "note": note,
            # 显式 shrink 一律提示(移除了哪些 lr 维 / 父代 lr 不再沿用);
            # auto → shrink 属设计内裁定,只记 note 不告警
            "warn": requested == "shrink",
            "lr_dims_removed": lr_dims,
            "space": {k: v for k, v in space.items() if k not in lr_dims}}


def record_lr_mode(trial_dir: Path, plan: dict | None) -> None:
    """把运行期**生效的** lr 模式补写进 trial 产物(v2.16.3)。

    观测口径缺口:``config/trial.yaml`` 的 ``hpo.lr_mode`` 只是请求值,
    ``hpo-study.json`` / ``metrics.json`` 只有生效值,两者可能相反且无任何
    提示(gen-05 trial-022)。三处分工:

    - ``config/trial.yaml``:**创建期**写 ``hpo.lr_mode_effective`` /
      ``lr_mode_requested`` / ``lr_mode_note``(``render._hpo_lr_mode_preview``,
      与本函数同源 plan_lr_mode)。运行期**不得**改它 —— config/ 在 run 开始时
      被 source-manifest 冻结,改写即触发 integrity fail-closed;
    - ``hpo-study.json``:本函数补 ``lr_mode_effective`` 等字段(模板随后
      write_hpo_artifacts 也会带上);
    - ``metrics.json``:模板在 ``metrics_extra`` 里写 ``hpo_lr_mode_effective``;
      本函数只在文件已存在时补写(断点续跑等场景)。

    幂等;文件缺失/不可解析时静默跳过(遥测不得让 trial 失败)。
    """
    if not isinstance(plan, dict) or not plan.get("mode"):
        return
    effective = str(plan["mode"])
    fields = {
        "lr_mode_effective": effective,
        "lr_mode_requested": plan.get("requested_mode") or effective,
        "lr_mode_note": plan.get("note"),
        "lr_mode_search_lr": plan.get("search_lr"),
        "lr_mode_refit_lr": plan.get("refit_lr"),
    }
    tdir = Path(trial_dir)
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return  # smoke 轮不落 trial 根目录产物(同 write_hpo_artifacts)
    for name in ("hpo-study.json", "metrics.json"):
        p = tdir / name
        try:
            if not p.exists():
                continue
            blob = load_json(p) or {}
            if not isinstance(blob, dict):
                continue
            if name == "metrics.json":
                blob["hpo_lr_mode_effective"] = effective
                blob.setdefault("hpo_lr_mode_requested",
                                fields["lr_mode_requested"])
                if fields["lr_mode_note"]:
                    blob.setdefault("hpo_lr_mode_note", fields["lr_mode_note"])
            else:
                blob.update({k: v for k, v in fields.items() if v is not None})
            dump_json(p, blob)
        except Exception as e:  # noqa: BLE001
            logger.warning("lr_mode 生效值写回 %s 失败(不阻塞): %s", name, e)


def hpo_parallelism(hpo_cfg: dict | None, algorithm: str | None,
                    gpu: bool) -> int:
    """inner 并行 config 数。auto:GPU 上的 xgboost 取 2,其余 1。

    并行发生在同一进程、同一张已租 GPU 上(线程并发,xgboost 训练释放
    GIL);CPU 训练的库并行会与 n_jobs 线程预算相乘超订,auto 不开。
    显式整数按配置执行(调用方负责线程预算)。
    """
    raw = (hpo_cfg or {}).get("parallel_configs", "auto")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return max(1, int(raw))
    s = str(raw or "auto").strip().lower()
    if s.isdigit():
        return max(1, int(s))
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return 1
    return 2 if (gpu and str(algorithm or "").lower() == "xgboost") else 1


def hpo_fidelity_plan(hpo_cfg: dict | None, n_inner_rows: int):
    """多保真档位(None = 单保真)。仅 fidelity=successive_halving 时启用。"""
    mode = str((hpo_cfg or {}).get("fidelity") or "single").strip().lower()
    if mode not in ("successive_halving", "sh", "multi"):
        return None
    if int(n_inner_rows) < MULTI_FIDELITY_MIN_ROWS:
        return None
    fids = [f for f in DEFAULT_FIDELITIES
            if f >= 1.0 or int(n_inner_rows * f) >= MULTI_FIDELITY_MIN_SUBSET]
    return tuple(fids) if len(fids) > 1 else None


def stratified_random_rows(y, k: int, *, seed: int = 20260814) -> np.ndarray:
    """按标签分层的**随机**行位置(升序),用于多保真低档子样本。

    v2.14.1(2x 线)移植:行序头部在时间序落盘的 train 上等于只取最早
    几天,这里在整个拟合池内均匀随机抽,类比例与全体一致;连续目标
    (唯一值 > 50)退化为普通随机抽样。
    """
    y = np.asarray(y)
    n = len(y)
    k = int(k)
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    vals, inv = np.unique(y, return_inverse=True)
    if len(vals) < 2 or len(vals) > 50:
        return np.sort(rng.choice(n, size=k, replace=False))
    parts = []
    for j in range(len(vals)):
        pos = np.flatnonzero(inv == j)
        take = min(len(pos), max(1, int(round(k * len(pos) / n))))
        parts.append(rng.choice(pos, size=take, replace=False))
    return np.sort(np.concatenate(parts))


def successive_halving_plan(budget: int, n_rungs: int,
                            eta: int = DEFAULT_ETA) -> list[int]:
    """各档评估的 config 数:第 0 档 budget 个,之后每档保留 ceil(n/eta)。"""
    plan = [max(1, int(budget))]
    for _ in range(max(n_rungs, 1) - 1):
        plan.append(max(1, -(-plan[-1] // max(int(eta), 2))))
    return plan


#: 各算法族的"轮数上限"参数名(重训 cap / 交付配方)
_CAP_KEYS = {"lightgbm": "n_estimators", "xgboost": "n_estimators",
             "catboost": "iterations"}


def cap_param_key(algorithm: str | None) -> str:
    return _CAP_KEYS.get(str(algorithm or "").lower(), "n_estimators")


# ---------------------------------------------------------------------------
# v2.17.2(A1):lr-shrink 重训自适应,不再被树数上限静默截断
#
# 事故(run4 req-t7-400w trial-017):search_lr=0.05 搜出的冠军 config 搜索期
# 最优 3019 轮;重训 lr=0.05×0.25=0.0125,需要 ≈3019/0.25≈12000 轮,cap 却被
# 树数铁律钳到 8000 → 重训停在 7999/8000(cap_bound)、cap_extended_to=null,
# 得到伪负 Δ −0.00095(4.9 h 白跑)。现在由搜索期最优轮数推重训需求:
#   need = ceil(best_iter_search / factor × REFIT_ITER_MARGIN);
#   need ≤ MAX_TREES → 维持收缩,cap = min(MAX_TREES, max(旧缩放 cap, need));
#   need > MAX_TREES → 放宽收缩 factor_eff = min(1, best_iter×margin / MAX_TREES),
#     重训 lr = search_lr × factor_eff、cap = MAX_TREES(成本硬顶不变)。
# 搜索期最优轮数不可得(热启动点胜出且无遥测等)时沿用旧口径并留痕。
# ---------------------------------------------------------------------------

#: 重训轮数余量:全量 train(比 inner 拟合池多 ~1/(1−inner_fraction))与更低
#: lr 下最优轮数的外推误差
REFIT_ITER_MARGIN = 1.2


def search_best_iteration(result: dict | None) -> int | None:
    """搜索期**所选 config** 的最优轮数(1-based);不可得返回 None。

    优先 ``result["best_search_iteration"]``(run_hpo_search / 多保真 / 保真
    复评写入);缺失时按搜索原始最优参数(``search_best_params`` 优先于
    ``best_params``)在轨迹里逐键匹配(多保真取最高保真度那条)。
    """
    r = result or {}
    bi = _as_int(r.get("best_search_iteration"))
    if bi:
        return bi
    bp = r.get("search_best_params") or r.get("best_params")
    if not isinstance(bp, dict) or not bp:
        return None
    key = _key_one(bp)
    hits = []
    for t in (r.get("trials") or []) + (r.get("trials_all_fidelities") or []):
        if isinstance(t, dict) and isinstance(t.get("params"), dict) \
                and _key_one(t["params"]) == key \
                and _as_int(t.get("best_iteration")):
            hits.append((float(t.get("fidelity") or 1.0),
                         int(t["best_iteration"])))
    return max(hits)[1] if hits else None


def plan_refit_shrink(search_best_iter, *, search_lr: float, factor: float,
                      cap_now, cap_scale: float | None = None,
                      max_trees: int | None = None,
                      margin: float = REFIT_ITER_MARGIN) -> dict:
    """lr-shrink 重训配方(v2.17.2,A1):返回重训 lr / cap 与留痕字段。

    返回 ``{refit_lr, refit_cap, factor, factor_eff, need_trees,
    search_best_iteration, lr_adjusted, cap_clamped, basis, max_trees}``:
      - ``lr_adjusted``:需求超出树数上限,收缩因子已放宽(重训 lr 提高);
      - ``cap_clamped``:按证据(或旧口径 cap×1/factor)要的轮数超过树数上限;
      - ``basis``:``search_best_iteration`` | ``unavailable``(旧口径)。
    """
    from .config import MAX_TREES

    mt = int(max_trees or MAX_TREES)
    f = min(max(float(factor), 1e-6), 1.0)
    scale = float(cap_scale) if cap_scale else 1.0 / f
    try:
        cap0 = int(float(cap_now))
    except (TypeError, ValueError):
        cap0 = mt
    wanted = int(math.ceil(max(1, cap0) * scale))
    base_cap = max(1, min(wanted, mt))
    bi = _as_int(search_best_iter)
    out = {"search_best_iteration": bi, "search_lr": float(search_lr),
           "factor": f, "factor_eff": f,
           "refit_lr": round(float(search_lr) * f, 8),
           "refit_cap": base_cap, "cap_wanted_legacy": wanted,
           "need_trees": None, "max_trees": mt, "margin": float(margin),
           "lr_adjusted": False, "cap_clamped": bool(wanted > mt),
           "basis": "search_best_iteration" if bi else "unavailable"}
    if not bi:
        return out
    need = int(math.ceil(bi / f * float(margin)))
    out["need_trees"] = need
    if need <= mt:
        out.update(refit_cap=max(1, min(mt, max(base_cap, need))),
                   cap_clamped=False)
        return out
    f_eff = min(1.0, bi * float(margin) / float(mt))
    out.update(factor_eff=round(f_eff, 6),
               refit_lr=round(float(search_lr) * f_eff, 8),
               refit_cap=mt, lr_adjusted=bool(f_eff > f), cap_clamped=True)
    return out


# ---------------------------------------------------------------------------
# v2.17.2(A3):deadline 外推的重训预留
# ---------------------------------------------------------------------------

def hpo_refit_reserve_sec(trial_dir: Path | None) -> float | None:
    """创建期成本外推的重训秒数(trial.yaml expected_resources.hpo_cost_estimate
    .retrain_sec,见 render._hpo_cost_estimate);缺失返回 None。"""
    if trial_dir is None:
        return None
    try:
        from .logging_utils import load_yaml

        p = Path(trial_dir) / "trial.yaml"
        if not p.exists():
            return None
        est = ((load_yaml(p) or {}).get("expected_resources") or {}).get(
            "hpo_cost_estimate") or {}
        v = float(est.get("retrain_sec") or 0)
        return v if v > 0 else None
    except Exception:  # noqa: BLE001  预留是优化项,读失败按无
        return None


def estimate_refit_seconds(fit_entries: list | None, *,
                           static_sec: float | None = None,
                           rows_ratio: float = 1.0,
                           trees_scale: float = 1.0,
                           max_trees: int | None = None) -> float:
    """重训预留秒数 = max(创建期外推 static_sec, 搜索实测外推)。

    搜索实测外推 = 已完成 config 拟合秒数中位数 × 行数比(全量 train / inner
    拟合面)× 轮数比(lr-shrink 为 1/factor;有逐 config 实训轮数时封顶
    max_trees / 实训轮数中位数 —— 重训树数受树数铁律约束)。
    """
    ents = [e for e in (fit_entries or []) if isinstance(e, dict)]
    secs = []
    for e in ents:
        try:
            v = float(e.get("fit_sec"))
        except (TypeError, ValueError):
            continue
        if v > 0:
            secs.append(v)
    dyn = 0.0
    if secs:
        ts = max(float(trees_scale or 1.0), 0.0)
        rounds = [r for r in (_as_int(e.get("rounds")) for e in ents) if r]
        if max_trees and rounds:
            ts = min(ts, float(max_trees) / float(np.median(rounds)))
        dyn = float(np.median(secs)) * max(float(rows_ratio or 1.0), 0.0) * ts
    try:
        st = float(static_sec or 0.0)
    except (TypeError, ValueError):
        st = 0.0
    return round(max(st, dyn), 1)


# ---------------------------------------------------------------------------
# v2.17.2(A4):抽样面搜索的保真复评(opt-in:hpo.fidelity_recheck_top_k)
# ---------------------------------------------------------------------------

def fidelity_recheck(result: dict, evaluate: Callable[[dict], Any], *,
                     top_k: int, metric_direction_up: bool) -> dict:
    """抽样面前 top_k 名 config 在**全量 inner 拟合池**上复评,按复评分选 best。

    evaluate(params) → float 或 {"value", ...遥测}(同 fit_score_inner)。
    返回 ``{top_k, n_rechecked, candidates, best_params, best_full_metric,
    best_iteration, best_changed, sampled_best_params}``;无候选时 n_rechecked=0。
    """
    seen: set = set()
    cands: list[dict] = []
    for t in _ranked([t for t in (result.get("trials") or [])
                      if isinstance(t, dict) and t.get("params")
                      and t.get("inner_metric") is not None],
                     metric_direction_up):
        k = _key_one(t["params"])
        if k in seen:
            continue
        seen.add(k)
        cands.append(t)
        if len(cands) >= max(0, int(top_k)):
            break
    rows: list[dict] = []
    for i, t in enumerate(cands):
        _t1 = time.time()
        v, extra = _unpack_inner(evaluate(dict(t["params"])))
        rows.append({"params": dict(t["params"]), "rank_sampled": i + 1,
                     "sampled_metric": float(t["inner_metric"]),
                     "full_metric": float(v),
                     "best_iteration": _as_int(extra.get("best_iteration")),
                     "rounds": _as_int(extra.get("rounds")),
                     "fit_sec": round(time.time() - _t1, 1)})
    out = {"top_k": int(top_k), "n_rechecked": len(rows), "candidates": rows,
           "sampled_best_params": (dict(cands[0]["params"]) if cands
                                   else None)}
    if not rows:
        return out
    best = sorted(rows, key=lambda r: r["full_metric"],
                  reverse=bool(metric_direction_up))[0]
    out.update(best_params=dict(best["params"]),
               best_full_metric=best["full_metric"],
               best_iteration=best["best_iteration"],
               best_rank_sampled=best["rank_sampled"],
               best_changed=best["rank_sampled"] != 1)
    return out


#: build_gbm 的默认构造参数(基准点 enqueue 用;与 build_gbm 同源)
_ALGO_DEFAULTS = {
    "xgboost": {"learning_rate": 0.05, "max_depth": 6, "subsample": 0.9,
                "colsample_bytree": 0.9, "min_child_weight": 1.0,
                "reg_lambda": 1.0, "reg_alpha": 0.0, "gamma": 0.0},
    "lightgbm": {"learning_rate": 0.05, "num_leaves": 31, "subsample": 0.9,
                 "colsample_bytree": 0.9, "min_child_samples": 20,
                 "reg_lambda": 0.0, "reg_alpha": 0.0},
    "catboost": {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
}


def incumbent_params(algorithm: str | None, parent_best: dict | None = None,
                     parent_hp: dict | None = None) -> dict:
    """父代(基准)配方的参数点:算法默认值 ← 父代 hyperparameters ← best_params。

    enqueue 进 study 作为第一个评估点:搜索永远有"父代配方在 inner 上
    得几分"的参照,HPO 在 inner 上不会输给基准而不自知。
    """
    out = dict(_ALGO_DEFAULTS.get(str(algorithm or "").lower(), {}))
    for src in (parent_hp, parent_best):
        if isinstance(src, dict):
            out.update({k: v for k, v in src.items()
                        if isinstance(v, (int, float, str, bool))})
    return out


# ---------------------------------------------------------------------------
# v2.16.3:撞 cap 的单一语义(D-011)
# ---------------------------------------------------------------------------
#: "贴近 cap"的比例(只作提示,不是 cap_bound)
NEAR_CAP_RATIO = 0.95


def _as_int(v) -> int | None:
    try:
        if v is None or isinstance(v, bool):
            return None
        iv = int(v)
        return iv if iv > 0 else None
    except (TypeError, ValueError):
        return None


def cap_bound_of(*, rounds=None, cap=None, best_iteration=None,
                 es_rounds=None) -> bool | None:
    """**唯一**的撞 cap 判据(v2.16.3,D-011)。

    语义:cap_bound ⇔ 训练被**轮数上限**截断、早停 patience 未到期
    ⇔ 实训轮数 ≥ cap(``stopped_by == "cap"``)。

    - 有实训轮数(``rounds``,早停面曲线长度)时直接比;
    - 没有轮数、但有最优轮数与早停耐心时推断:早停在最优轮之后再等
      ``es_rounds`` 轮才停,实训轮数 = min(cap, best + es_rounds),故
      ``best + es_rounds ≥ cap`` ⇔ 撞 cap;
    - 信息不足返回 None(调用方不得当 False 用,也不得当 True 用)。

    旧口径并存三套:baseline 模板 ``best ≥ 0.95×cap``、模板单段
    ``len(curve) ≥ cap``、hurdle 遥测 ``rounds ≥ cap`` —— 同一 trial 判定
    相反(run4 regression trial-022:``metrics.es_info.cap_bound=true``,
    ``hpo_refit_es_info.cap_bound=false``;clf 实训 7973 < cap 8000,
    早停真实触发)。"贴近 cap"另由 :data:`NEAR_CAP_RATIO` 作提示字段
    ``near_cap``,不再混入 cap_bound。
    """
    c = _as_int(cap)
    if not c:
        return None
    r = _as_int(rounds)
    if r:
        return r >= c
    b = _as_int(best_iteration)
    if b is not None and es_rounds is not None:
        try:
            return b + max(0, int(es_rounds)) >= c
        except (TypeError, ValueError):
            return None
    return None


def _cap_segments(metrics: dict) -> tuple[dict, str]:
    """从 trial metrics 抽取各子模型的 (rounds, cap, best, es_rounds)。

    优先级(同一次最终拟合的遥测,越具体越优先):
      1. ``hpo_refit_es_info``(hurdle HPO 重训:clf_/reg_ 分段);
      2. ``hurdle.fit_stats`` / ``hurdle`` 顶层的 clf_/reg_ rounds(hurdle 血缘);
      3. ``es_info``(模板单段 / 两段式第一段:stage1_rounds);
      4. 顶层 ``best_iteration`` / ``n_estimators_used``(baseline 血缘)。
    """
    m = metrics or {}
    es = m.get("es_info") if isinstance(m.get("es_info"), dict) else {}
    hur = m.get("hurdle") if isinstance(m.get("hurdle"), dict) else {}
    cap_default = _as_int(es.get("cap")) or _as_int(hur.get("cap"))
    es_default = es.get("es_rounds", hur.get("es_rounds"))

    def _two_stage(d: dict, src: str):
        segs = {}
        for name in ("clf", "reg"):
            r = d.get(f"{name}_rounds")
            b = d.get(f"{name}_best_iteration")
            if r is None and b is None:
                continue
            segs[name] = {"rounds": r, "best_iteration": b,
                          "cap": d.get(f"{name}_cap") or d.get("cap")
                          or cap_default,
                          "es_rounds": d.get("es_rounds", es_default)}
        if any(s.get("rounds") for s in segs.values()):
            return segs, src
        return None

    for src, d in (("hpo_refit_es_info", m.get("hpo_refit_es_info")),
                   ("hurdle.fit_stats", hur.get("fit_stats")),
                   ("hurdle", hur)):
        if isinstance(d, dict) and d:
            got = _two_stage(d, src)
            if got:
                return got
    if es and (es.get("stage1_rounds") or es.get("stage1_best_iteration")):
        return ({"model": {"rounds": es.get("stage1_rounds"),
                           "best_iteration": es.get("stage1_best_iteration"),
                           "cap": es.get("cap"),
                           "es_rounds": es.get("es_rounds")}},
                "es_info")
    if hur:
        got = _two_stage(hur, "hurdle")
        if got is None:
            segs = {}
            for name in ("clf", "reg"):
                if hur.get(f"{name}_best_iteration"):
                    segs[name] = {"rounds": None,
                                  "best_iteration": hur[f"{name}_best_iteration"],
                                  "cap": hur.get("cap") or cap_default,
                                  "es_rounds": hur.get("es_rounds", es_default)}
            if segs:
                return segs, "hurdle(best_iteration+es_rounds)"
    return ({"model": {"rounds": m.get("n_estimators_used"),
                       "best_iteration": m.get("best_iteration"),
                       "cap": cap_default, "es_rounds": es_default}},
            "metrics.best_iteration")


def unified_cap_bound(metrics: dict | None) -> dict:
    """trial metrics → 单一语义的撞 cap 结论(两段式任一子模型撞即 True)。

    返回 ``{"cap_bound", "stopped_by", "near_cap", "source", "segments"}``;
    ``cap_bound=None`` 表示遥测不足以判定。runner 的
    ``evaluation.complexity.cap_bound``、``assess_hpo_study`` 与模板写的
    ``metrics.es_info.cap_bound`` 都经由本函数(或其底层
    :func:`cap_bound_of`),不再各自一套。
    """
    segs, src = _cap_segments(metrics or {})
    out_segs: dict = {}
    bounds: list[bool] = []
    near: list[bool] = []
    for name, s in segs.items():
        b = cap_bound_of(rounds=s.get("rounds"), cap=s.get("cap"),
                         best_iteration=s.get("best_iteration"),
                         es_rounds=s.get("es_rounds"))
        cap = _as_int(s.get("cap"))
        bi = _as_int(s.get("best_iteration"))
        nc = bool(cap and bi and bi >= NEAR_CAP_RATIO * cap)
        out_segs[name] = {**s, "cap_bound": b, "near_cap": nc}
        if b is not None:
            bounds.append(b)
        near.append(nc)
    cap_bound = any(bounds) if bounds else None
    # 旧口径兜底:只有一个布尔旗标(无轮数/最优轮数)时如实沿用并标注来源
    if cap_bound is None:
        es = (metrics or {}).get("es_info") or {}
        if isinstance(es, dict) and isinstance(es.get("cap_bound"), bool):
            cap_bound = bool(es["cap_bound"])
            src = f"legacy_flag:{es.get('source') or 'es_info'}"
    return {"cap_bound": cap_bound,
            "stopped_by": (None if cap_bound is None
                           else "cap" if cap_bound else "early_stop"),
            "near_cap": any(near) if near else None,
            "source": src, "segments": out_segs}


def es_cap_telemetry(model, *, cap=None, es_rounds=None, best_iteration=None,
                     rounds=None) -> dict:
    """已拟合模型 → ``metrics.es_info`` 的撞 cap 字段(模板/baseline 共用)。

    复合容器(hurdle 等)读 ``model.stage1_`` 的 clf_/reg_ 分段;单模型读
    ``model.mvo_es_info_``;都没有时用调用方给的 rounds / best_iteration。
    """
    probe: dict = {}
    st = getattr(model, "stage1_", None)
    if isinstance(st, dict) and st:
        probe["hurdle"] = {"fit_stats": dict(st), "cap": cap,
                           "es_rounds": es_rounds}
    mi = getattr(model, "mvo_es_info_", None)
    es = dict(mi) if isinstance(mi, dict) else {}
    es.setdefault("cap", cap)
    es.setdefault("es_rounds", es_rounds)
    probe["es_info"] = es
    probe["best_iteration"] = best_iteration
    probe["n_estimators_used"] = rounds
    u = unified_cap_bound(probe)
    return {"cap": _as_int(cap) or _as_int(es.get("cap")),
            "cap_bound": u["cap_bound"], "stopped_by": u["stopped_by"],
            "near_cap": u["near_cap"], "cap_bound_source": u["source"],
            "cap_bound_rule": "rounds>=cap (best+es_rounds>=cap when rounds unknown)"}


def assess_hpo_study(trial_dir: Path, *, default_budget: int = 40) -> dict:
    """[v2.15.0] 一个已完成 HPO trial 的搜索质量评估(planner 与创建期门共用)。

    complete = 预算达到默认值的 75%、最终空间无贴边维、outer 重训未撞
    cap。不完整的 HPO 不能当作"该特征面已调过参"(v2.13.4 的
    `_champion_face_hpo_done` 只看有没有,run3 R1 只跑 16 个 config 且
    lr 顶在人为抬高的下界上)。
    """
    tdir = Path(trial_dir)
    st = load_json(tdir / "hpo-study.json") if (tdir / "hpo-study.json"
                                                ).exists() else {}
    st = st or {}
    meta = load_json(tdir / "metrics.json") if (tdir / "metrics.json"
                                                ).exists() else {}
    meta = meta or {}
    n_cfg = int(st.get("n_configs_evaluated") or meta.get("hpo_n_configs")
                or 0)
    n_warm = int(((st.get("warm_start") or {}).get("n_points")) or 0)
    diag = st.get("diagnostics") or {}
    hits = diag.get("boundary_hits") or {}
    # v2.16.3:撞 cap 走单一语义(unified_cap_bound;两段式任一段撞即 True)
    cap_bound = bool(unified_cap_bound(meta).get("cap_bound"))
    reasons = []
    if n_cfg + n_warm < 0.75 * max(1, int(default_budget)):
        reasons.append(f"under_budget({n_cfg}+{n_warm}warm<0.75×{default_budget})")
    if hits:
        reasons.append("boundary_hits:" + ",".join(
            f"{k}@{v.get('side')}" for k, v in sorted(hits.items())))
    if cap_bound:
        reasons.append("outer_refit_cap_bound")
    return {"complete": not reasons, "reasons": reasons,
            "n_configs": n_cfg, "n_warm": n_warm,
            "boundary_hits": hits, "cap_bound": cap_bound,
            "lr_mode": st.get("lr_mode") or meta.get("hpo_lr_mode")}


# ---------------------------------------------------------------------------
# 搜索主入口
# ---------------------------------------------------------------------------

def _run_hpo_search_core(
    space: dict,
    budget: int,
    fit_score_inner: Callable[..., Any],
    *,
    metric_direction_up: bool,
    seed: int = 20260814,
    deadline_ts: float | None = None,
    checkpoint_path: Path | None = None,
    metric_name: str | None = None,
    resume: bool = True,
    warm_start: list[dict] | None = None,
    enqueue: list[dict] | None = None,
    signature: str | None = None,
    n_jobs: int = 1,
    expand: dict | bool | None = None,
    seed_salt: str | None = None,
    multivariate: bool = True,
    fidelities=None,
    eta: int = DEFAULT_ETA,
    refit_reserve: dict | float | None = None,
    search_cap_ts: float | None = None,
) -> dict:
    """在 inner-validation 上搜索;返回 best_params/inner_score/轨迹。

    fit_score_inner(params) → float 或 {"value": float, ...遥测}(遥测键
    如 best_iteration/cap_bound/fit_sec 原样记入每条轨迹);多保真模式下
    以 (params, fidelity) 调用。方向感知由调用方统一(metric_direction_up)。

    v2.13:metric_name 透传落盘(inner_metric_name 写进 hpo-study.json)。

    v14(时间预算与断点):
    - deadline_ts:绝对时间戳(runner 经 MODELEVO_DEADLINE_TS 注入,
      已扣除 kill grace)。按已评 config 的实测均速外推,剩余预算
      跑不完时**提前收敛**交出当前最优——优于被 SIGTERM 杀掉后
      一无所有(实测 trial-020/021 各 4 次 rc=-15,39 个已评配置
      全部作废,白跑数小时);
    - checkpoint_path:每评完一个 config 立即落盘,超时被杀后轨迹
      不丢;resume:checkpoint 属于同一(基础)空间/预算/签名时,已评
      config 先回放进 study,只补差额。

    v2.17.2(A3)deadline 外推口径:旧实现 ``per = (now − t0) / 已评数``,
    t0 含建池/回放等准备时间、并行 config 时分母只有 1 —— run4 实测首个
    config 完成即把 budget 40 砍到 2。现在:单 config 墙钟 = 本轮已完成
    config 的 fit_sec **中位数 ÷ 并行数**;至少完成 n_jobs 个 config 后才
    允许截断;外推时预留最终重训时间(``refit_reserve``:秒数,或
    ``{static_sec, rows_ratio, trees_scale, max_trees}`` 交给
    :func:`estimate_refit_seconds`);剩余预算整体装不下时,只要再派发一个
    config 仍能在重训预留前跑完就继续(不再整体收敛)。外推明细写
    ``deadline_estimate``。

    v2.13.3(smoke 预算闸):MODELEVO_SMOKE_RUN=1 时 budget 降为 1,
    并关闭热启动/外扩/并行(smoke 只验通路,1 万行小样本的分数与全量
    不可比)。

    v2.15.0(见模块 docstring):
    - warm_start:签名一致的历史观测(不计入本 trial 的 n_configs,
      参与 TPE 先验与 best 选择);
    - enqueue:基准点(如父代配方),裁进空间后作为首批评估;
    - signature:inner 目标签名,写入 checkpoint/study,续跑时签名
      不一致即判陈旧;
    - n_jobs:并行 config 数(>1 时启用 constant_liar);
    - expand:True 或 {"lr_floor":…,"checkpoints":…}——预算 35%/55%/75%
      时检查前 K 名贴边并外扩,新段 study 带入全部已有观测;
    - seed_salt:按 trial 派生采样器 seed(见 derive_seed);
    - multivariate:多变量 TPE(默认开);
    - fidelities:≥2 档时走多保真逐级减半(_run_successive_halving)。
    """
    _smoke_budget_reduced = os.environ.get("MODELEVO_SMOKE_RUN") == "1"
    if _smoke_budget_reduced:
        budget = 1 if int(budget) > 1 else int(budget)
        resume = False
        warm_start, enqueue, expand, fidelities = None, None, None, None
        n_jobs = 1
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path).with_name(
                "hpo-checkpoint-smoke.json")
    budget = max(1, int(budget))
    n_jobs = max(1, int(n_jobs or 1))
    seed_eff = derive_seed(seed, seed_salt)
    if fidelities and len(fidelities) > 1 and budget >= max(int(eta), 2) ** 2:
        # v2.19.0:多保真路径的搜索墙钟上限并入 deadline(取更早者)
        _sh_deadline = deadline_ts
        if search_cap_ts is not None:
            _sh_deadline = (search_cap_ts if deadline_ts is None
                            else min(deadline_ts, search_cap_ts))
        return _run_successive_halving(
            space, budget, fit_score_inner,
            metric_direction_up=metric_direction_up, seed=seed_eff,
            deadline_ts=_sh_deadline, checkpoint_path=checkpoint_path,
            metric_name=metric_name, resume=bool(resume),
            fidelities=fidelities, eta=eta, enqueue=enqueue,
            signature=signature, n_jobs=n_jobs, multivariate=multivariate,
            refit_reserve=refit_reserve)

    _expand_cfg: dict | None = None
    if expand:
        _expand_cfg = dict(expand) if isinstance(expand, dict) else {}
    base_space = {k: dict(v) for k, v in space.items()}
    space_cur = {k: dict(v) for k, v in space.items()}
    expansions: list[dict] = []
    t0 = time.time()
    lock = threading.Lock()
    own: list[dict] = []          # 本 trial 的评估(回放 + 新拟合)
    _n_new = [0]                  # 本轮真正拟合的 config 数(不含回放)
    _truncated = [False]
    _capped = [False]             # v2.19.0:被 max_search_minutes 截断
    # v2.17.2(A3):deadline 外推只看本轮新拟合 config 的拟合耗时(未取整);
    # 时钟从首个 config 派发开始(建池/回放/热启动等准备时间不计入)
    _fit_new: list[float] = []
    _t_dispatch: list = [None]
    _deadline_est: dict = {}

    def _dump() -> None:
        if checkpoint_path is None:
            return
        dump_json(checkpoint_path, {
            "schema": 2,
            "n_configs_evaluated": len(own),
            "elapsed_sec": round(time.time() - t0, 1),
            "budget": int(budget),
            # base_space:续跑判定"是否同一搜索"的口径;space:外扩后的当前空间
            "base_space": base_space,
            "space": space_cur,
            "expansions": expansions,
            "signature": signature,
            "trials": own,
        })

    # ---- 断点续跑:回放已评 config,只补差额 ---------------------------
    # 回放按参数逐键等价匹配,不依赖 optuna 的 trial 序号;(基础)空间、
    # 预算或签名与 checkpoint 记载不一致时判为陈旧轨迹,整份忽略。
    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            _ck = load_json(Path(checkpoint_path)) or {}
            _ck_budget = int(_ck.get("budget") or 0)
            _ck_base = _ck.get("base_space", _ck.get("space"))
            _ck_sig = _ck.get("signature")
            if (_ck_base is not None and _ck_base != base_space) \
                    or (_ck_budget and _ck_budget > budget) \
                    or (signature and _ck_sig and _ck_sig != signature):
                logger.warning(
                    "hpo checkpoint 陈旧(空间/预算/签名已变更,记载 budget=%s "
                    "当前 budget=%s),忽略轨迹从零搜索", _ck_budget, budget)
            else:
                if _ck.get("base_space") is not None and \
                        isinstance(_ck.get("space"), dict):
                    space_cur = {k: dict(v) for k, v in _ck["space"].items()}
                    expansions = list(_ck.get("expansions") or [])
                _seen: set = set()
                for _t in _ck.get("trials") or []:
                    if not isinstance(_t, dict) or not _t.get("params"):
                        continue
                    _p = _t["params"]
                    _k = _key_one(_p)
                    if _k in _seen:
                        continue
                    if any(_pk not in space_cur for _pk in _p):
                        logger.warning(
                            "hpo checkpoint 中一条轨迹的参数 %s 不在当前空间,"
                            "跳过该条", sorted(set(_p) - set(space_cur)))
                        continue
                    _seen.add(_k)
                    _e = dict(_t)
                    _e["inner_metric"] = float(_t["inner_metric"])
                    _e["origin_first"] = _t.get("origin")
                    _e["origin"] = "replay"
                    own.append(_e)
                if own:
                    logger.warning(
                        "hpo 断点续跑:回放 checkpoint 已评 %d/%d 个 config"
                        "(上次搜索被中断,补跑差额)", len(own), budget)
                    _dump()
        except Exception as _e:  # noqa: BLE001
            logger.warning("hpo checkpoint 读取失败,从零搜索: %s", _e)
            own = []
    _n_replayed = len(own)

    # ---- 热启动:签名一致的历史观测(不计入本 trial 的评估数) ----------
    _own_keys = {_key_one(o["params"]) for o in own}
    warm: list[dict] = []
    for _w in warm_start or []:
        _p = _w.get("params") if isinstance(_w, dict) else None
        try:
            _v = float(_w.get("inner_metric"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not np.isfinite(_v) or not point_in_space(_p, space_cur):
            continue
        _k = _key_one(_p)
        if _k in _own_keys:
            continue
        _own_keys.add(_k)
        warm.append({"params": dict(_p), "inner_metric": _v,
                     "origin": "warm", "source": _w.get("source"),
                     **({"best_iteration": _as_int(_w.get("best_iteration"))}
                        if _as_int(_w.get("best_iteration")) else {})})
    if warm:
        logger.info("hpo 热启动:载入 %d 个签名一致的历史观测(来源 %s)",
                    len(warm), sorted({str(w.get('source')) for w in warm}))

    # ---- 基准点 enqueue(父代配方等) ----------------------------------
    _pending_enqueue: list[dict] = []
    for _q in enqueue or []:
        _c = clip_to_space(_q, space_cur)
        if _c and _key_one(_c) not in _own_keys and \
                all(_key_one(_c) != _key_one(x) for x in _pending_enqueue):
            _pending_enqueue.append(_c)

    def _record(entry: dict, dur: float | None = None) -> None:
        """每评完一个 config 立即落盘(超时被杀后轨迹不丢)。"""
        with lock:
            own.append(entry)
            _n_new[0] += 1
            try:
                _fit_new.append(float(dur if dur is not None
                                      else entry.get("fit_sec") or 0.0))
            except (TypeError, ValueError):
                pass
            _dump()

    def _mark_dispatch() -> None:
        with lock:
            if _t_dispatch[0] is None:
                _t_dispatch[0] = time.time()

    def _refit_reserve_sec() -> float:
        """最终重训的预留秒数(见 estimate_refit_seconds)。"""
        if refit_reserve is None:
            return 0.0
        if isinstance(refit_reserve, (int, float)) \
                and not isinstance(refit_reserve, bool):
            return max(0.0, float(refit_reserve))
        rr = dict(refit_reserve) if isinstance(refit_reserve, dict) else {}
        with lock:
            fresh = [o for o in own if o.get("origin") != "replay"]
        return estimate_refit_seconds(
            fresh, static_sec=rr.get("static_sec"),
            rows_ratio=float(rr.get("rows_ratio") or 1.0),
            trees_scale=float(rr.get("trees_scale") or 1.0),
            max_trees=rr.get("max_trees"))

    def _out_of_budget() -> bool:
        """v2.17.2(A3):中位数单 config 耗时 ÷ 并行数外推 + 重训预留。

        - 本轮完成数 < 并行数时不截断(首个 config 完成时另一路尚在训练,
          单点外推会把并行吞吐低估一半);
        - 剩余预算整体装得下(now + per×剩余 + 重训预留 ≤ deadline)→ 继续;
        - 整体装不下时不再"一判跑不完就整体收敛"(run4:40→2),而是只要
          再派发一个 config(墙钟 ≈ 中位数)仍能在重训预留之前跑完就继续,
          否则才收敛交出当前最优。
        """
        if search_cap_ts is not None and _search_capped():
            return True
        if deadline_ts is None:
            return False
        with lock:
            done = list(_fit_new)
        if len(done) < max(1, n_jobs):
            return False
        remain_n = budget - len(own)
        if remain_n <= 0:
            return True
        med = float(np.median(done))
        per = med / max(1, n_jobs)
        reserve = _refit_reserve_sec()
        now = time.time()
        _deadline_est.update({
            "median_fit_sec": round(med, 1), "n_parallel": int(n_jobs),
            "per_config_wall_sec": round(per, 1),
            "n_fit_this_attempt": len(done), "remaining_configs": remain_n,
            "refit_reserve_sec": round(reserve, 1),
            "setup_sec_excluded": (round(_t_dispatch[0] - t0, 1)
                                   if _t_dispatch[0] else None),
            "seconds_to_deadline": round(deadline_ts - now, 1)})
        if now + per * remain_n + reserve <= deadline_ts:
            return False
        _room = deadline_ts - reserve - now
        _deadline_est["fits_estimate"] = len(own) + max(0, int(_room // per)) \
            if per > 0 else len(own)
        if now + med + reserve <= deadline_ts:
            return False
        logger.warning(
            "hpo 时间预算不足:已评 %d/%d,单 config 拟合中位数 %.0f 秒(并行 %d),"
            "剩余 %.0f 秒扣除重训预留 %.0f 秒后再装不下一个 config —— 提前收敛"
            "交出当前最优(优于被 SIGTERM 杀掉后一无所有)",
            len(own), budget, med, n_jobs, max(deadline_ts - now, 0),
            reserve)
        _truncated[0] = True
        _deadline_est["truncated_at"] = len(own)
        return True

    def _search_capped() -> bool:
        """v2.19.0:搜索墙钟上限(hpo.max_search_minutes)。到点不再派发;
        已有完成 config 时按单 config 拟合中位数预判——派发出去会越过上限
        的也不派发(并行时在跑的 config 照常跑完,不中途杀)。"""
        now = time.time()
        with lock:
            done = list(_fit_new)
            n_own = len(own)
        if n_own >= budget:
            return False
        over = now >= search_cap_ts
        if not over and done:
            over = now + float(np.median(done)) > search_cap_ts
        if over and not _capped[0]:
            _capped[0] = True
            _truncated[0] = True
            logger.warning(
                "hpo 搜索墙钟上限(max_search_minutes)到:已评 %d/%d 个 config,"
                "停止派发,用已完成 config 中最优者重训", n_own, budget)
        return over

    _checks: list[int] = []
    if _expand_cfg is not None:
        _fr = _expand_cfg.get("checkpoints") or EXPAND_CHECKPOINTS
        _checks = sorted({max(1, int(math.ceil(budget * float(f))))
                          for f in _fr if 0 < float(f) < 1})
    _lr_floor = float((_expand_cfg or {}).get("lr_floor") or DEFAULT_LR_FLOOR)
    n_startup = _n_startup_for(budget)

    def _maybe_expand() -> None:
        nonlocal space_cur
        if _expand_cfg is None or len(own) + len(warm) < EXPAND_MIN_OBS:
            return
        hits = boundary_hits(own + warm, space_cur, metric_direction_up)
        if not hits:
            return
        new_space, changes = expand_space(space_cur, hits, expansions,
                                          lr_floor=_lr_floor)
        if changes:
            expansions.extend({**c, "at_config": len(own)} for c in changes)
            space_cur = new_space
            logger.warning("hpo 边界自适应:%s",
                           "; ".join(f"{c['param']} {c['side']} "
                                     f"{c['from']}→{c['to']}" for c in changes))
            _dump()

    _checks_pending = list(_checks)

    def _next_stop() -> int:
        for c in _checks_pending:
            if c > len(own):
                return c
        return budget

    def _expand_if_due() -> None:
        due = [c for c in _checks_pending if c <= len(own)]
        if not due:
            return
        for c in due:
            _checks_pending.remove(c)
        if len(own) < budget:
            _maybe_expand()

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        _dir = "maximize" if metric_direction_up else "minimize"
        phase = 0
        while len(own) < budget:
            if _out_of_budget():
                break
            n_this = min(budget - len(own), _next_stop() - len(own))
            if n_this <= 0:
                break
            study = optuna.create_study(direction=_dir, sampler=_make_tpe(
                seed_eff + phase, n_startup, multivariate=multivariate,
                constant_liar=n_jobs > 1))
            _n_added = 0
            for _o in own + warm:
                if not point_in_space(_o["params"], space_cur):
                    continue
                try:
                    study.add_trial(optuna.trial.create_trial(
                        params=dict(_o["params"]),
                        distributions={k: _space_distribution(space_cur[k])
                                       for k in _o["params"]},
                        values=[float(_o["inner_metric"])]))
                    _n_added += 1
                except Exception as _e:  # noqa: BLE001
                    # 单条失败不让整次搜索退化为从零:跳过它(旧轨迹的用途
                    # 是先验与 best 候选,不是正确性)
                    logger.warning("hpo 回放/热启动单条失败(已跳过): %s", _e)
            _take = _pending_enqueue[:n_this]
            _pending_enqueue = _pending_enqueue[len(_take):]
            for _q in _take:
                try:
                    study.enqueue_trial(_q, user_attrs={"mvo_origin": "enqueued"},
                                        skip_if_exists=True)
                except TypeError:
                    study.enqueue_trial(_q, user_attrs={"mvo_origin": "enqueued"})
            _space_phase = {k: dict(v) for k, v in space_cur.items()}
            _phase_idx = phase

            def _objective(trial_obj, _sp=_space_phase, _ph=_phase_idx):
                _origin = (trial_obj.user_attrs or {}).get("mvo_origin")
                if not _origin:
                    _origin = ("startup" if trial_obj.number < n_startup
                               else "guided")
                params = _suggest_one(trial_obj, _sp)
                _mark_dispatch()
                _t1 = time.time()
                v, extra = _unpack_inner(fit_score_inner(params))
                _dur = time.time() - _t1
                entry = {"params": params, "inner_metric": v,
                         "origin": _origin, "phase": _ph,
                         "fit_sec": round(_dur, 1)}
                entry.update({k: x for k, x in extra.items()
                              if k not in entry})
                _record(entry, _dur)
                return v

            def _stop_on_budget(study_obj, _t):
                if _out_of_budget():
                    study_obj.stop()

            study.optimize(_objective, n_trials=n_this, n_jobs=n_jobs,
                           show_progress_bar=False,
                           callbacks=[_stop_on_budget])
            phase += 1
            if _truncated[0]:
                break
            _expand_if_due()
        sampler_name = "tpe" + ("-mv" if multivariate else "") \
            + (f"+resume{_n_replayed}" if _n_replayed else "") \
            + (f"+warm{len(warm)}" if warm else "") \
            + (f"+par{n_jobs}" if n_jobs > 1 else "")
    except ImportError:
        rng = np.random.default_rng(seed_eff)
        while len(own) < budget:
            if _out_of_budget():
                break
            if _pending_enqueue:
                params = dict(_pending_enqueue.pop(0))
                for k, spec in space_cur.items():
                    if k not in params:
                        params[k] = _SAMPLE_FNS[str(spec.get("type", "float"))
                                                .lower()](rng, spec)
                _origin = "enqueued"
            else:
                params = _sample_one(rng, space_cur)
                _origin = "startup"
            _mark_dispatch()
            _t1 = time.time()
            v, extra = _unpack_inner(fit_score_inner(params))
            _dur = time.time() - _t1
            entry = {"params": params, "inner_metric": v, "origin": _origin,
                     "fit_sec": round(_dur, 1)}
            entry.update({k: x for k, x in extra.items() if k not in entry})
            _record(entry, _dur)
            _expand_if_due()
        sampler_name = "random(optuna 缺失)"

    allobs = own + warm
    ranked = _ranked(allobs, metric_direction_up)
    best = dict(ranked[0]["params"]) if ranked else None
    best_val = float(ranked[0]["inner_metric"]) if ranked else None
    diag = hpo_diagnostics(allobs, space_cur, metric_direction_up, n_startup)
    if _deadline_est:
        _deadline_est["fit_phase_sec"] = (round(time.time() - _t_dispatch[0], 1)
                                          if _t_dispatch[0] else None)
    return {
        "best_params": best,
        "inner_score": best_val,
        "best_origin": ranked[0].get("origin") if ranked else None,
        # v2.17.2(A1):所选 config 的搜索期最优轮数(lr-shrink 重训据此推 cap/lr)
        "best_search_iteration": (_as_int(ranked[0].get("best_iteration"))
                                  if ranked else None),
        "n_configs_evaluated": len(own),
        # 本轮实际新拟合的 config 数(不含断点回放)
        "n_configs_fit_this_attempt": _n_new[0],
        "n_configs_replayed": _n_replayed,
        "budget_requested": int(budget),
        "budget_truncated_by_deadline": bool(
            deadline_ts is not None and len(own) < int(budget)
            and not _capped[0]),
        # v2.19.0:被搜索墙钟上限截断(与 deadline 截断分开留痕)
        "search_capped": bool(_capped[0]),
        # v2.17.2(A3):deadline 外推明细(中位数 ÷ 并行 + 重训预留)
        "deadline_estimate": dict(_deadline_est) if deadline_ts is not None
        else None,
        "smoke_budget_reduced": bool(_smoke_budget_reduced),
        "sampler": sampler_name,
        "seed_effective": int(seed_eff),
        "n_startup_trials": int(n_startup),
        "multivariate": bool(multivariate),
        "parallel": int(n_jobs),
        "search_seconds": round(time.time() - t0, 1),
        "inner_metric_name": (str(metric_name).lower()
                              if metric_name else None),
        "inner_signature": signature,
        "warm_start": {"n_points": len(warm),
                       "sources": sorted({str(w.get("source"))
                                          for w in warm})},
        "n_enqueued": sum(1 for o in own if o.get("origin") == "enqueued"),
        "space_initial": base_space,
        "space_final": space_cur,
        "space_expansions": expansions,
        "diagnostics": diag,
        "trials": own,
    }


# ---------------------------------------------------------------------------
# v2.19.0:HPO 跨卡并行 + 搜索墙钟上限
#
# 实测(8 个 run):HPO 单 trial 6.6~7.5 h,config 只在本 trial 租到的一张卡
# 上跑(xgboost GPU 时 parallel_configs=auto 取 2,同卡两路),其余 7 张卡闲置;
# 8 次 AUC HPO 仅 1 次显著为正。两处改动:
# - 跨卡:搜索开始前若进程尚未初始化 CUDA,经 GPU 租约(<run>/.gpu-leases)
#   再租干净卡,把 CUDA_VISIBLE_DEVICES 扩成"主卡,额外卡…"(主卡仍是 0 号,
#   搜索后的重训 / 打分路径 device=cuda 不变),每个并行 worker 绑一张卡
#   (线程局部序号,build_gbm 据此构造 device=cuda:k)。CUDA 已初始化时扩展
#   无效(xgboost 会告警 "Only 1 GPUs are visible" 并静默落回 0 号卡)——用
#   一次极小探测拟合识别,失败即释放租约、恢复旧行为;
# - 限时:evolution.hpo.max_search_minutes(默认 60)从搜索开始计,与 deadline
#   外推取更早者;到时不再派发,用已完成 config 中最优者重训。
# 只改搜索调度,不碰 inner 切分口径(lint 规则 7:HPO 只看 train 内部切分)。
# ---------------------------------------------------------------------------

_HPO_DEVICE = threading.local()


def hpo_device_ordinal() -> int | None:
    """当前线程绑定的 HPO 卡序号(None = 未绑定,按 device=cuda 即 0 号卡)。"""
    return getattr(_HPO_DEVICE, "ordinal", None)


def _bind_device_ordinals(fn: Callable[..., Any], ordinals: list[int],
                          fallback_jobs: int = 1):
    """包装 fit_score_inner:每次评估从卡池取一张、评完归还(并发数 ≤ 卡数,
    不会阻塞)。线程局部序号由 build_gbm 消费(消费时置 consumed)。

    防护:trial 若不经 build_gbm 构造模型(如 hurdle 父代自带构造器),序号
    不会被消费,N 路会全挤主卡。故首个非 0 号卡的评估先当"验证":验证完成
    前其余非 0 号 worker 等待;验证发现序号未被消费 → 回退为最多
    fallback_jobs 路(旧 parallel_configs 口径)同卡并发。返回 (wrapped, state)。
    """
    import queue

    pool: "queue.Queue[int]" = queue.Queue()
    for o in ordinals:
        pool.put(int(o))
    probe_ord = next((int(o) for o in ordinals if int(o) != 0), None)
    state: dict = {"honored": None}
    verified = threading.Event()
    fb_sem = threading.Semaphore(max(1, int(fallback_jobs)))

    def _fallback(*a, **kw):
        with fb_sem:
            return fn(*a, **kw)

    def _wrapped(*a, **kw):
        if state["honored"] is False:
            return _fallback(*a, **kw)
        o = pool.get()
        if o != 0 and o != probe_ord and not verified.is_set():
            pool.put(o)
            verified.wait()
            if state["honored"] is False:
                return _fallback(*a, **kw)
            o = pool.get()
        prev = getattr(_HPO_DEVICE, "ordinal", None)
        _HPO_DEVICE.ordinal = o
        _HPO_DEVICE.consumed = False
        try:
            return fn(*a, **kw)
        finally:
            consumed = bool(getattr(_HPO_DEVICE, "consumed", False))
            _HPO_DEVICE.ordinal = prev
            pool.put(o)
            if o == probe_ord and state["honored"] is None:
                state["honored"] = consumed
                if not consumed:
                    logger.warning(
                        "hpo 跨卡并行:trial 的模型构造未经 runtime.hpo."
                        "build_gbm(卡序号未被消费)→ 回退同卡 %d 路并发",
                        max(1, int(fallback_jobs)))
                verified.set()

    return _wrapped, state


def _run_dir_of(checkpoint_path: Path | None) -> Path | None:
    """checkpoint 在 <run>/trials/<trial>/ 下 → 反推 run 目录(推不出返回 None)。"""
    if checkpoint_path is None:
        return None
    try:
        p = Path(checkpoint_path).resolve()
        rd = p.parents[2]
        if p.parents[1].name == "trials" and (rd / "trials").is_dir():
            return rd
    except (IndexError, OSError):
        return None
    return None


def _hpo_run_cfg(run_dir: Path | None) -> dict:
    """读 run 配置的 evolution.hpo 节(缺省 DEFAULTS);失败返回 {}。"""
    try:
        from .config import Config

        rc = (Path(run_dir) / "modelevo-config.yaml") if run_dir else None
        cfg = Config.load(run_config=rc if rc and rc.exists() else None)
        return dict(cfg.get("evolution.hpo", {}) or {})
    except Exception:  # noqa: BLE001
        return {}


def _probe_visible_ordinal(ordinal: int) -> bool:
    """极小 xgboost 拟合探测 cuda:<ordinal> 是否真的可见(CUDA 已初始化时
    xgboost 告警 "Only N GPUs are visible" 并落回 0 号卡 → 判不可用)。"""
    try:
        import xgboost as _xgb_probe
    except Exception:  # noqa: BLE001
        return False
    rng = np.random.default_rng(0)
    Xp = rng.random((64, 3))
    yp = (Xp[:, 0] > 0.5).astype(int)
    try:
        with warnings.catch_warnings(record=True) as _w:
            warnings.simplefilter("always")
            _xgb_probe.XGBClassifier(
                n_estimators=2, max_depth=2, device=f"cuda:{int(ordinal)}",
                verbosity=1).fit(Xp, yp)
        for m in _w:
            if "GPUs are visible" in str(m.message) \
                    or "device ordinal" in str(m.message):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def acquire_hpo_gpus(run_dir: Path | None, max_gpus: int, *,
                     probe: Callable[[int], bool] | None = None,
                     occupancy: list | None = None) -> dict:
    """为 HPO 搜索再租额外干净卡并扩展 CUDA_VISIBLE_DEVICES。

    返回 {"ordinals": [0, 1, …], "gpus": [物理卡号…], "reason": …,
    "_leases": [...], "_env_prev": 原 CUDA_VISIBLE_DEVICES};不可用时
    ordinals=[](调用方维持旧行为)。用 release_hpo_gpus 释放。
    """
    rec: dict = {"ordinals": [], "gpus": [], "reason": None}
    own_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    own = [s.strip() for s in own_raw.split(",") if s.strip()]
    if os.environ.get("MODELEVO_GPU") != "1" or len(own) != 1:
        rec["reason"] = "no_gpu_allocated" if not own else (
            "already_multi_visible" if len(own) > 1 else "no_gpu_allocated")
        return rec
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        rec["reason"] = "smoke"
        return rec
    if run_dir is None:
        rec["reason"] = "run_dir_unknown"
        return rec
    if int(max_gpus) < 2:
        rec["reason"] = "max_parallel_gpus<2"
        return rec
    try:
        from .scheduler import (GpuLease, annotate_gpu_occupancy,
                                gpu_candidates_by_occupancy,
                                query_gpu_occupancy)
    except Exception as e:  # noqa: BLE001
        rec["reason"] = f"scheduler_unavailable: {type(e).__name__}"
        return rec
    occ = occupancy if occupancy is not None else query_gpu_occupancy()
    if not occ:
        rec["reason"] = "gpu_query_failed"
        return rec
    clean, _busy = gpu_candidates_by_occupancy(
        annotate_gpu_occupancy(occ, Path(run_dir)))
    cands = [g for g in clean if str(g) not in own]
    leases = []
    for g in cands:
        if len(leases) >= int(max_gpus) - 1:
            break
        lease = GpuLease.acquire(Path(run_dir) / ".gpu-leases", int(g))
        if lease is not None:
            leases.append(lease)
    if not leases:
        rec["reason"] = "no_extra_gpu" if not cands else "gpu_lease_unavailable"
        return rec
    phys = [own[0]] + [str(lease.gpu_index) for lease in leases]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(phys)
    _probe = probe or _probe_visible_ordinal
    if not _probe(len(phys) - 1):
        # 进程内 CUDA 已按单卡初始化 → 扩展无效:恢复旧行为
        os.environ["CUDA_VISIBLE_DEVICES"] = own_raw
        for lease in leases:
            lease.release()
        rec["reason"] = "cuda_already_initialized"
        return rec
    rec.update(ordinals=list(range(len(phys))), gpus=[int(x) for x in phys],
               reason="ok", _leases=leases, _env_prev=own_raw)
    return rec


def release_hpo_gpus(rec: dict | None) -> None:
    """释放额外租约并恢复 CUDA_VISIBLE_DEVICES(进程内已建的 CUDA 上下文
    不受影响:主卡仍是 0 号,重训 / 打分路径不变)。"""
    if not rec:
        return
    for lease in rec.pop("_leases", None) or []:
        try:
            lease.release()
        except Exception:  # noqa: BLE001
            pass
    if "_env_prev" in rec:
        os.environ["CUDA_VISIBLE_DEVICES"] = rec.pop("_env_prev")


def run_hpo_search(
    space: dict,
    budget: int,
    fit_score_inner: Callable[..., Any],
    *,
    metric_direction_up: bool,
    seed: int = 20260814,
    deadline_ts: float | None = None,
    checkpoint_path: Path | None = None,
    metric_name: str | None = None,
    resume: bool = True,
    warm_start: list[dict] | None = None,
    enqueue: list[dict] | None = None,
    signature: str | None = None,
    n_jobs: int = 1,
    expand: dict | bool | None = None,
    seed_salt: str | None = None,
    multivariate: bool = True,
    fidelities=None,
    eta: int = DEFAULT_ETA,
    refit_reserve: dict | float | None = None,
    run_dir: Path | None = None,
    max_search_minutes: float | None = None,
    multi_gpu: str | bool | None = None,
    max_parallel_gpus: int | None = None,
) -> dict:
    """在 inner-validation 上搜索(语义见 _run_hpo_search_core)。

    v2.19.0 新增(均可显式传入;None 时读 run 配置 evolution.hpo.*,run 目录
    缺省由 checkpoint_path 反推,推不出则不限时、不跨卡——与旧行为一致):
    - max_search_minutes:搜索墙钟上限(分钟),与 deadline 取更早者;
      0/None = 不限。被截断时 result.search_capped=True;
    - multi_gpu(auto/off)+ max_parallel_gpus:跨卡并行,见模块段首说明。
      生效时 n_jobs = 卡数(每个 worker 一张卡),result.parallel_gpus 记卡数
      (未生效为 1,无 GPU 为 0),result.multi_gpu 记原因与物理卡号。
    """
    rd = Path(run_dir) if run_dir is not None else _run_dir_of(checkpoint_path)
    hcfg = _hpo_run_cfg(rd) if rd is not None else {}
    if max_search_minutes is None:
        max_search_minutes = hcfg.get("max_search_minutes")
        if os.environ.get("MODELEVO_HPO_MAX_SEARCH_MINUTES"):
            max_search_minutes = os.environ["MODELEVO_HPO_MAX_SEARCH_MINUTES"]
    try:
        cap_min = float(max_search_minutes or 0)
    except (TypeError, ValueError):
        cap_min = 0.0
    t_start = time.time()
    cap_ts = t_start + cap_min * 60.0 if cap_min > 0 else None
    if multi_gpu is None:
        multi_gpu = hcfg.get("multi_gpu", "off" if rd is None else "auto")
    if max_parallel_gpus is None:
        max_parallel_gpus = int(hcfg.get("max_parallel_gpus", 4) or 4)
    mg_on = str(multi_gpu).strip().lower() not in (
        "off", "false", "0", "no", "none")
    gpu_rec: dict = {"ordinals": [], "gpus": [],
                     "reason": "config_off" if not mg_on else None}
    fn, n_eff = fit_score_inner, n_jobs
    _bind_state: dict = {}
    try:
        if mg_on:
            gpu_rec = acquire_hpo_gpus(rd, int(max_parallel_gpus))
        if len(gpu_rec.get("ordinals") or []) > 1:
            fn, _bind_state = _bind_device_ordinals(
                fit_score_inner, gpu_rec["ordinals"],
                fallback_jobs=max(1, int(n_jobs or 1)))
            n_eff = len(gpu_rec["ordinals"])
            logger.info("hpo 跨卡并行:%d 张卡 %s,每卡一个 worker",
                        n_eff, gpu_rec.get("gpus"))
        result = _run_hpo_search_core(
            space, budget, fn, metric_direction_up=metric_direction_up,
            seed=seed, deadline_ts=deadline_ts,
            checkpoint_path=checkpoint_path, metric_name=metric_name,
            resume=resume, warm_start=warm_start, enqueue=enqueue,
            signature=signature, n_jobs=n_eff, expand=expand,
            seed_salt=seed_salt, multivariate=multivariate,
            fidelities=fidelities, eta=eta, refit_reserve=refit_reserve,
            search_cap_ts=cap_ts)
    finally:
        release_hpo_gpus(gpu_rec)
    if "search_capped" not in result:
        # 多保真路径:上限并入 deadline,截断且上限早于 deadline → 记为 capped
        result["search_capped"] = bool(
            cap_ts is not None and result.get("budget_truncated_by_deadline")
            and (deadline_ts is None or cap_ts <= deadline_ts))
    _gpu_on = os.environ.get("MODELEVO_GPU") == "1"
    _n_ord = len(gpu_rec.get("ordinals") or [])
    if _n_ord > 1 and _bind_state.get("honored") is False:
        gpu_rec["reason"] = "ordinal_not_honored"
        _n_ord = 1
    result["parallel_gpus"] = (_n_ord if _n_ord > 1
                               else (1 if _gpu_on else 0))
    result["multi_gpu"] = {"reason": gpu_rec.get("reason"),
                           "gpus": list(gpu_rec.get("gpus") or [])}
    result["max_search_minutes"] = cap_min if cap_min > 0 else None
    return result


def _run_successive_halving(
    space: dict,
    budget: int,
    fit_score_inner: Callable[..., Any],
    *,
    metric_direction_up: bool,
    seed: int,
    deadline_ts: float | None,
    checkpoint_path: Path | None,
    metric_name: str | None,
    resume: bool,
    fidelities,
    eta: int,
    enqueue: list[dict] | None = None,
    signature: str | None = None,
    n_jobs: int = 1,
    multivariate: bool = True,
    refit_reserve: dict | float | None = None,
) -> dict:
    """多保真逐级减半(v2.14.1 2x 线 P2 #19,v2.15.0 移植)。

    第 0 档用 TPE(optuna 缺失时随机)在最低保真度上采 budget 个 config;
    之后每档把上一档前 ceil(n/eta) 名放到更高保真度上重评,最高档
    (保真度 1.0 = 完整 inner 拟合面)的第一名即 best_params。
    fit_score_inner(params, fidelity) 由调用方按 fidelity 取子样本拟合。
    热启动观测是满保真分数,与低档不可混排,本模式下不使用(只 enqueue
    基准点进第 0 档)。断点续跑:checkpoint 按 (params, fidelity) 记录。
    """
    fids = sorted(float(f) for f in fidelities)
    if fids[-1] != 1.0:
        fids.append(1.0)
    plan = successive_halving_plan(budget, len(fids), eta)
    t0 = time.time()
    better = (lambda a, b: a > b) if metric_direction_up \
        else (lambda a, b: a < b)
    log: list[dict] = []
    lock = threading.Lock()
    n_new = [0]
    cost_new = [0.0]
    t_dispatch: list = [None]     # v2.17.2(A3):首个评估派发时刻(准备时间不计)

    def _key(params: dict, f: float) -> tuple:
        return (_key_one(params), repr(float(f)))

    cache: dict = {}
    if resume and checkpoint_path is not None \
            and Path(checkpoint_path).exists():
        try:
            _ck = load_json(Path(checkpoint_path)) or {}
            if (_ck.get("mode") == "successive_halving"
                    and _ck.get("space") == space
                    and [float(x) for x in (_ck.get("fidelities") or [])]
                    == fids
                    and int(_ck.get("budget") or 0) <= int(budget)
                    and not (signature and _ck.get("signature")
                             and _ck.get("signature") != signature)):
                for _t in _ck.get("trials") or []:
                    if isinstance(_t, dict) and _t.get("params"):
                        cache[_key(_t["params"], _t["fidelity"])] = (
                            float(_t["inner_metric"]),
                            {k: v for k, v in _t.items()
                             if k not in ("params", "fidelity",
                                          "inner_metric", "replayed")})
                if cache:
                    logger.warning("hpo 多保真断点续跑:checkpoint 有 %d 次"
                                   "评估可复用", len(cache))
            else:
                logger.warning("hpo checkpoint 与本次多保真搜索不匹配,忽略")
        except Exception as _e:  # noqa: BLE001
            logger.warning("hpo checkpoint 读取失败,从零搜索: %s", _e)

    def _dump() -> None:
        if checkpoint_path is None:
            return
        dump_json(checkpoint_path, {
            "mode": "successive_halving", "fidelities": fids,
            "eta": int(eta), "budget": int(budget), "signature": signature,
            "n_evaluations": len(log),
            "elapsed_sec": round(time.time() - t0, 1),
            "space": space, "trials": log,
        })

    def _eval(params: dict, f: float, origin: str = "sh") -> float:
        k = _key(params, f)
        if k in cache:
            v, extra = cache[k]
            with lock:
                log.append({"params": params, "fidelity": f,
                            "inner_metric": v, "replayed": True, **extra})
            return v
        with lock:
            if t_dispatch[0] is None:
                t_dispatch[0] = time.time()
        _t1 = time.time()
        v, extra = _unpack_inner(fit_score_inner(params, f))
        with lock:
            n_new[0] += 1
            cost_new[0] += f
            cache[k] = (v, extra)
            log.append({"params": params, "fidelity": f, "inner_metric": v,
                        "origin": origin,
                        "fit_sec": round(time.time() - _t1, 1), **extra})
            _dump()
        return v

    def _over(remaining_cost: float) -> bool:
        # v2.17.2(A3):时钟从首个评估派发起算;本轮完成数 < 并行数时不截断;
        # 预留最终重训时间(满保真评估的实测外推 / 创建期外推取大)
        if deadline_ts is None or n_new[0] < max(1, n_jobs) \
                or cost_new[0] <= 0 or t_dispatch[0] is None:
            return False
        per_unit = (time.time() - t_dispatch[0]) / cost_new[0]
        reserve = 0.0
        if isinstance(refit_reserve, (int, float)) \
                and not isinstance(refit_reserve, bool):
            reserve = max(0.0, float(refit_reserve))
        elif isinstance(refit_reserve, dict):
            with lock:
                full_ents = [t for t in log if not t.get("replayed")
                             and float(t.get("fidelity") or 0) >= 1.0]
            reserve = estimate_refit_seconds(
                full_ents, static_sec=refit_reserve.get("static_sec"),
                rows_ratio=float(refit_reserve.get("rows_ratio") or 1.0),
                trees_scale=float(refit_reserve.get("trees_scale") or 1.0),
                max_trees=refit_reserve.get("max_trees"))
        return time.time() + per_unit * remaining_cost + reserve > deadline_ts

    def _later_cost(r: int, n_r: int) -> float:
        """第 r 档之后各档的计划成本(按第 r 档实际 n_r 个重新推算)。"""
        cost, n = 0.0, n_r
        for f in fids[r + 1:]:
            n = max(1, -(-n // max(int(eta), 2)))
            cost += n * f
        return cost

    truncated = False
    rung_scores: list[list[tuple[dict, float]]] = [[]]
    n_startup = _n_startup_for(plan[0])
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize" if metric_direction_up else "minimize",
            sampler=_make_tpe(seed, n_startup, multivariate=multivariate,
                              constant_liar=n_jobs > 1))
        for _q in enqueue or []:
            _c = clip_to_space(_q, space)
            if _c:
                study.enqueue_trial(_c, user_attrs={"mvo_origin": "enqueued"})

        def _objective(trial_obj):
            _origin = (trial_obj.user_attrs or {}).get("mvo_origin") or (
                "startup" if trial_obj.number < n_startup else "guided")
            params = _suggest_one(trial_obj, space)
            v = _eval(params, fids[0], _origin)
            with lock:
                rung_scores[0].append((params, v))
            return v

        def _stop(study_obj, _t):
            done = len(rung_scores[0])
            if done >= plan[0]:
                return
            if _over((plan[0] - done) * fids[0] + _later_cost(0, plan[0])):
                study_obj.stop()

        study.optimize(_objective, n_trials=plan[0], n_jobs=max(1, n_jobs),
                       show_progress_bar=False, callbacks=[_stop])
        sampler_name = "tpe" + ("-mv" if multivariate else "") \
            + "+successive_halving"
    except ImportError:
        rng = np.random.default_rng(seed)
        for i in range(plan[0]):
            if i and _over((plan[0] - i) * fids[0] + _later_cost(0, plan[0])):
                break
            params = _sample_one(rng, space)
            rung_scores[0].append((params, _eval(params, fids[0], "startup")))
        sampler_name = "random+successive_halving(optuna 缺失)"
    if len(rung_scores[0]) < plan[0]:
        truncated = True
        logger.warning("hpo 多保真:时间预算不足,第 0 档只评了 %d/%d 个 "
                       "config", len(rung_scores[0]), plan[0])

    for r in range(1, len(fids)):
        prev = sorted(rung_scores[r - 1], key=lambda t: t[1],
                      reverse=metric_direction_up)
        keep = max(1, -(-len(prev) // max(int(eta), 2)))
        promote = [p for p, _ in prev[:keep]]
        cur: list[tuple[dict, float]] = []
        for i, params in enumerate(promote):
            remaining = (len(promote) - i) * fids[r] + _later_cost(
                r, len(promote))
            if i and _over(remaining):
                truncated = True
                break
            cur.append((params, _eval(params, fids[r], f"rung{r}")))
        if not cur:
            truncated = True
            break
        rung_scores.append(cur)
        if len(cur) < len(promote):
            break

    top = rung_scores[-1]
    best, best_val = top[0]
    for p, v in top[1:]:
        if better(v, best_val):
            best, best_val = p, v
    rungs = [{"fidelity": fids[i], "n": len(rs),
              "best": (max if metric_direction_up else min)(
                  v for _, v in rs)}
             for i, rs in enumerate(rung_scores)]
    n_replayed = sum(1 for t in log if t.get("replayed"))
    full = [t for t in log if float(t.get("fidelity", 0)) >= 1.0]
    _top_f = fids[len(rung_scores) - 1]
    _best_bi = next((_as_int(t.get("best_iteration")) for t in reversed(log)
                     if _key_one(t.get("params") or {}) == _key_one(best)
                     and float(t.get("fidelity", 0)) == float(_top_f)), None)
    return {
        "best_params": best,
        "inner_score": best_val,
        # v2.17.2(A1):最高档所选 config 的最优轮数
        "best_search_iteration": _best_bi,
        "n_configs_evaluated": len(rung_scores[0]),
        "n_configs_fit_this_attempt": n_new[0],
        "n_configs_replayed": n_replayed,
        "budget_requested": int(budget),
        "budget_truncated_by_deadline": bool(truncated),
        "smoke_budget_reduced": False,
        "sampler": sampler_name,
        "seed_effective": int(seed),
        "n_startup_trials": int(n_startup),
        "multivariate": bool(multivariate),
        "parallel": int(n_jobs),
        "search_seconds": round(time.time() - t0, 1),
        "inner_metric_name": (str(metric_name).lower()
                              if metric_name else None),
        "inner_signature": signature,
        "warm_start": {"n_points": 0, "sources": [],
                       "note": "多保真模式不使用满保真热启动观测"},
        "space_initial": space,
        "space_final": space,
        "space_expansions": [],
        "diagnostics": hpo_diagnostics(
            [{**t, "origin": t.get("origin", "guided")} for t in full],
            space, metric_direction_up, n_startup),
        "multi_fidelity": {
            "fidelities": fids,
            "eta": int(eta),
            "plan": plan,
            "rungs": rungs,
            "best_fidelity": rungs[-1]["fidelity"],
            "n_evaluations": len(log),
            # 以完整 inner 拟合为 1 的算力当量(不含回放)
            "cost_full_equiv": round(cost_new[0], 3),
        },
        # 热启动只消费满保真观测(低档分数与满保真不可比)
        "trials": [dict(t) for t in full] or log,
        "trials_all_fidelities": log,
    }


def write_hpo_artifacts(trial_dir: Path, result: dict) -> None:
    """hpo 结果落盘:hpo-study.json(汇总 + 全轨迹)。

    smoke 预检轮不落盘(v2.13.3):smoke 现在只评 1 个 config,其
    hpo-study.json 若写进 trial 根目录,后续全量轮被中断时磁盘上会
    留一份"n_configs_evaluated=1"的伪 study,事后审计无法区分它是
    smoke 残留还是真实搜索只跑了 1 个 config。smoke 的产物一律只进
    smoke/ 子目录(与 validation_score.parquet 同口径)。

    v2.15.0:轨迹上限 200 → 1000(热启动从这里读历史观测,截断会
    丢先验);v2.15 起模板在**最终重训之后**调用本函数,best_params
    写的是重训配方(含 lr 收缩/cap 外扩),search_best_params 保留搜索
    原始最优 —— 交付链(finalize-retrain)与 confirm 都从这里取配方。
    """
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return
    trial_dir = Path(trial_dir)
    payload = {k: v for k, v in result.items()
               if k not in ("trials", "trials_all_fidelities")}
    payload["trials"] = list(result.get("trials") or [])[:1000]
    if result.get("trials_all_fidelities"):
        payload["trials_all_fidelities"] = list(
            result["trials_all_fidelities"])[:2000]
    dump_json(trial_dir / "hpo-study.json", payload)
    # metrics.json 的 hpo 节由 trial 代码合并(本函数只落 study 文件)


# ---------------------------------------------------------------------------
# 算法化 GBM 构造(v2.9,[FIX-2026-09-03])
# ---------------------------------------------------------------------------

#: hpo 模板渲染路径支持的算法族(--algorithm 命名,大小写不敏感)
SUPPORTED_HPO_ALGOS = ("lightgbm", "xgboost", "catboost")

# [FIX-2026-09-10] lightgbm wheel CUDA 编译探测(模块级缓存,
# 与 trial 模板 body 的 _lgbm_cuda_ok 同口径):非 CUDA 编译轮上
# device=cuda 的 fit 期 Fatal 曾致进程死、CPU 回退没接住——注入
# 前先 64 行 x 4 特征小样本试 fit,一切异常按 False 处理。
_LGBM_CUDA: bool | None = None


def _lgbm_cuda_ok() -> bool:
    global _LGBM_CUDA
    if _LGBM_CUDA is not None:
        return _LGBM_CUDA
    try:
        import os as _os_probe
        import lightgbm as _lgb_probe
        import numpy as _np_probe
        _rs = _np_probe.random.RandomState(0)
        _Xp = _rs.rand(64, 4).astype("float32")
        _yp = (_rs.rand(64) > 0.5).astype(int)
        # v2.12.2.9:lightgbm 的 Fatal 由 C++ 层直写 fd 2,Python 的
        # redirect_stderr 拦不住——必须 dup2 到 /dev/null,否则
        # "[LightGBM] [Fatal] CUDA Tree Learner was not enabled in this
        # build." 会漏进 trial 的 stderr.log,看起来像 trial 崩溃。
        _saved_fd = _os_probe.dup(2)
        _null_fd = _os_probe.open(_os_probe.devnull, _os_probe.O_WRONLY)
        _os_probe.dup2(_null_fd, 2)
        _os_probe.close(_null_fd)
        try:
            _lgb_probe.LGBMClassifier(device="cuda", n_estimators=1,
                                      verbosity=-1,
                                      random_state=0).fit(_Xp, _yp)
            _LGBM_CUDA = True
        finally:
            _os_probe.dup2(_saved_fd, 2)
            _os_probe.close(_saved_fd)
    except Exception:
        _LGBM_CUDA = False
        print("lightgbm wheel 非 CUDA 编译,回退 CPU"
              "(PyPI wheel 默认不含 CUDA;这是预期降级,非故障)",
              flush=True)
    return _LGBM_CUDA


def _env_int(name: str, default: int) -> int:
    """环境变量读 int(坏值/缺失回退默认;hpo 构造口径单源)。"""
    import os

    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


#: 交付/baseline 机制的轮数上限与早停轮数(§ v2.13 HPO 机制对齐)。
#: HPO 搜索此前默认 n_estimators/iterations=300 且无早停——搜的是
#: "固定 300 棵"机制,交付/baseline 用的是"cap+早停"机制,两者不是
#: 同一函数族;LR 空间 [0.01,0.15] 在 300 棵下系统性偏向高 LR。
#: 默认与 runtime/baseline.py 的 N_ESTIMATORS_CAP/EARLY_STOP_ROUNDS
#: 同源(2000/50),可用环境变量覆盖。
N_ESTIMATORS_CAP_DEFAULT = 2000
EARLY_STOP_ROUNDS_DEFAULT = 50


def build_gbm(algorithm: str, params: dict, seed: int, model_type: str,
              n_rows: int, gpu: bool, *,
              n_estimators_cap: int | None = None,
              early_stopping_rounds: int | None = None) -> Any:
    """按算法构造统一 fit/predict 接口的 GBM 估计器(hpo 模板渲染路径)。

    背景(drawdown-t7 gen-03 实测):模板四件套的 _build_model 硬编码
    lightgbm,hpo 快通道又要求父代码自带四件套——本 run 的父代全部
    是 baseline 血缘 + 手写 widen,被动建 scaffold + 手改 7 处代码。
    构造逻辑下沉到 runtime 单函数后,--action hpo 无四件套父代时可
    直接按 --algorithm 渲染可执行 trial,不再需要 scaffold 探路。

    v2.13(HPO 机制对齐):三族默认树数改为 n_estimators_cap(默认
    2000,env MODELEVO_N_ESTIMATORS_CAP),早停参数随构造注入——
    xgboost 构造期 early_stopping_rounds,catboost od_type/od_wait/
    use_best_model,lightgbm 只抬 n_estimators(早停靠 fit 期 callbacks,
    见模板 _fit_with_es)。params 显式给的键仍覆盖默认(base.update(p)
    保持在最后)。

    统一契约:返回的估计器满足
    - fit(X, y) 可调用(early stopping 由调用方决定是否启用;
      xgboost/catboost 的构造期早停参数要求 fit 时带 eval_set——
      模板 _fit_with_es 统一提供,纯 fit(X, y) 调用会被调用方的
      回退分支兜住);
    - predict_proba(X)[:, 1](分类)或 predict(X)(回归)。
    params 直接 update 进构造参数(搜索空间键名按模板约定对齐各库)。
    """
    algo = str(algorithm or "lightgbm").strip().lower()
    p = dict(params or {})
    cap = int(n_estimators_cap if n_estimators_cap is not None
              else _env_int("MODELEVO_N_ESTIMATORS_CAP",
                            N_ESTIMATORS_CAP_DEFAULT))
    # [铁律]树数量硬上限:无论 cap 从哪来(env/run config/调用方显式传),
    # 构造前统一钳到 MAX_TREES。params 里显式给的 n_estimators 也在
    # 下方 base.update(p) 之后单独钳制(见每个分支)。
    from .config import clamp_trees as _clamp_trees
    cap = _clamp_trees(cap)
    rounds = int(early_stopping_rounds if early_stopping_rounds is not None
                 else _env_int("MODELEVO_EARLY_STOP_ROUNDS",
                               EARLY_STOP_ROUNDS_DEFAULT))
    # v2.19.0:HPO 跨卡并行时本线程绑定的卡序号(None/0 = 主卡,旧口径)
    _ord = hpo_device_ordinal() if gpu else None
    if algo == "lightgbm":
        import lightgbm as lgb

        base = dict(n_estimators=cap, learning_rate=0.05, num_leaves=31,
                    subsample=0.9, colsample_bytree=0.9, n_jobs=n_jobs_hint(),
                    random_state=seed, verbosity=-1)
        if gpu and _lgbm_cuda_ok():
            base["device"] = "cuda"
            if _ord:
                base["gpu_device_id"] = int(_ord)
                _HPO_DEVICE.consumed = True
        base.update(p)
        # params 可能显式覆盖 n_estimators —— 再次钳制
        for _k in ("n_estimators", "num_iterations", "num_boost_round"):
            if _k in base:
                base[_k] = _clamp_trees(base[_k])
        if model_type == "regression":
            return lgb.LGBMRegressor(**base)
        return lgb.LGBMClassifier(**base)
    if algo == "xgboost":
        import xgboost as xgb

        base = dict(n_estimators=cap, learning_rate=0.05, max_depth=6,
                    subsample=0.9, colsample_bytree=0.9, n_jobs=n_jobs_hint(),
                    random_state=seed, verbosity=0,
                    early_stopping_rounds=rounds)
        if gpu:
            base["device"] = f"cuda:{int(_ord)}" if _ord else "cuda"
            if _ord:
                _HPO_DEVICE.consumed = True
        base.update(p)
        if "n_estimators" in base:
            base["n_estimators"] = _clamp_trees(base["n_estimators"])
        elif "num_boost_round" in base:
            base["num_boost_round"] = _clamp_trees(base["num_boost_round"])
        if model_type == "regression":
            return xgb.XGBRegressor(**base)
        return xgb.XGBClassifier(**base)
    if algo == "catboost":
        import catboost as cb

        base = dict(iterations=cap, learning_rate=0.05, depth=6,
                    thread_count=n_jobs_hint(), random_seed=seed,
                    verbose=False, allow_writing_files=False,
                    od_type="Iter", od_wait=rounds, use_best_model=True)
        if gpu:
            base["task_type"] = "GPU"
            if _ord:
                base["devices"] = str(int(_ord))
                _HPO_DEVICE.consumed = True
        base.update(p)
        if "iterations" in base:
            base["iterations"] = _clamp_trees(base["iterations"])
        if model_type == "regression":
            return cb.CatBoostRegressor(**base)
        return cb.CatBoostClassifier(**base)
    raise ValueError(
        f"hpo 模板渲染路径不支持算法 {algorithm!r}(支持 {SUPPORTED_HPO_ALGOS});"
        "其他算法请用 --from-parent 携四件套父代的手写路径")


def n_jobs_hint() -> int:
    """线程预算:scheduler 注入的 OMP_NUM_THREADS(与模板 n_jobs() 同口径;
    runtime 侧独立实现,避免 trial 代码依赖倒挂)。"""
    import os

    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS", "4") or 4))
    except ValueError:
        return 4


def gbm_fit_safe(model: Any, X_fit, y_fit, **kw) -> Any:
    """统一 fit 包装:GPU 编译轮探测失败时回退 CPU 重训一次。

    lightgbm 无公开编译探测(fit 期 LightGBMError);xgboost/catboost
    由构造侧保证(device 只在 MODELEVO_GPU=1 时设置,fit 失败概率低,
    但同样兜底)。所有库共用同一条安全路径,hpo smoke/全量一致。

    v19(req-t7-drawdown gen-03 实测回写):lightgbm 4.x 中 task_type
    是核心参数 task 的别名,set_params(task_type="CPU") 落到
    task="CPU" → "Unknown task type cpu"(回退制造新异常)。优先
    sanitize 重建:弹出 device/task/task_type 矛盾项重构造干净估计器。
    """
    try:
        return model.fit(X_fit, y_fit, **kw)
    except Exception:
        try:
            _cls = type(model)
            _params = dict(model.get_params())
            for _bad in ("device", "task", "task_type"):
                _params.pop(_bad, None)
            model = _cls(**_params)
            return model.fit(X_fit, y_fit, **kw)
        except Exception:
            pass
        try:
            model.set_params(device="cpu", task_type="CPU")
        except Exception:
            try:
                model.set_params(device="cpu")
            except Exception:
                pass
        try:
            return model.fit(X_fit, y_fit, **kw)
        except Exception:
            kw_cpu = {k: v for k, v in kw.items()
                      if k not in ("eval_set", "callbacks")}
            return model.fit(X_fit, y_fit, **kw_cpu)
