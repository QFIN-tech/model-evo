"""强默认配方(v2.14 / P0-B #6):参数、频次编码与 seed bagging。

09-23 独立参照实验(seq-v3 30w×1488 基础面,不含任何工程特征):
框架 baseline 口径 LGBM(lr 0.05 / leaves 31 / 0.9 / 0.9、早停后不重训)
validation 0.7799;只换成下方常识强参数 +0.0030,早停定轮后在全量 train
重训 +0.0004,5 seed 平均 +0.0008,合计 0.7841(shadow +0.0033)。框架 4.4 h
搜索的冠军只比这个强默认高 +0.0028。强默认因此同时承担三件事:
baseline 锚点(不再高估搜索增益)、特征期的固定模型(用弱模型评特征会
高估"强模型本来就学得到"的特征)、交付的下限。

参数不是在单个任务上调出来的,而是宽表 GBDT 的常识设置;列少/行少时
自动放宽(colsample、min_child),避免在小数据上欠拟合。是否作为全局默认
由 benchmark 快速套件判定(见 runtime/benchmark.py 的 fast suite)。

v2.14.3(req-t7-drawdown-2x 实测回写:2.24M×1493 上 gen-00 两个 baseline
各超时两次、白烧约 6h,见 docs/MODELEVO_V2_14_3_TRAIN_COST_20260928.md):

1. ``resolve_n_seeds``:``baseline.n_seeds: auto`` 按 train 规模取 seed 数。
   同一 run 实测 3 个成员 val AUC 标准差 0.00004、bag 增益 +0.0003(配对
   SE 0.00004),成本 3 倍;30w 行级别 5 seed 增益 +0.0008 —— 大数据搜索
   期 1 个 seed,小数据保持 3 个。
2. ``SeedBag(es_members=1)``:定轮只用首个成员(成员间秩相关 0.997,轮数
   一致),其余成员直接按定好的轮数全量重训。3 seed 配方的拟合次数由
   6 降到 4。缺省 None = 旧行为(全部成员各自早停),旧 trial 代码复现不变。
3. 成员级断点(``MODELEVO_CKPT_DIR``,runner 注入):每个成员训完即落盘,
   重入按指纹跳过已训成员。旧实现超时重试会丢掉已训完的成员从头再来。
4. 预算感知(``MODELEVO_DEADLINE_TS``,runner 注入):训练中按 trial 截止
   时间判断"还训得完吗"——首个成员都训不完时抛 ``BudgetInfeasible``
   (快速失败、不重试,给出建议超时);后续成员训不完时按预算缩减成员数
   并留痕 ``n_seeds_effective_`` / ``budget_reduced_``。
5. 训练进度行:每约 5% 轮次打印 iter/elapsed/速率/到 cap 的 ETA,并写
   ``MODELEVO_PROGRESS_FILE``(runner 在超时/失败时据此给出诊断)。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, clone

#: 强默认的早停轮数与树数上限(低学习率需要更长耐心与更高上限)
STRONG_EARLY_STOP_ROUNDS = 200
STRONG_N_ESTIMATORS_CAP = 5000

#: ``baseline.n_seeds: auto`` 的切换阈值:train 行数 × 特征数(格数)。
#: 30w×1488 ≈ 4.5e8(bag 增益 +0.0008,保留 3 seed);2.24M×1493 ≈ 3.3e9
#: (bag 增益 +0.0003、σ_seed 0.00004,成本 3 倍 → 1 seed)。
AUTO_SEED_CELLS_THRESHOLD = 1_000_000_000
AUTO_SEED_SMALL = 3
AUTO_SEED_LARGE = 1

#: v2.14.3.1 搜索期代理配方(``baseline.search_recipe``)。完整配方 = 强默认
#: lr 0.02 + 早停定轮 + 全量 train 重训;代理 = lr 0.05 单遍早停(不重训)。
#: 400w run2 25% 抽样面实测:两个已知方向(seq 解析 +0.0056、剔除低信息列
#: −0.0014)在两种配方下方向/显著性一致、大小差 ≤12%,训练成本 22%;
#: 代理的配对 SE 约大 15%。交付(finalize-retrain / finalize-baseline-score)
#: 一律升级回完整配方。
SEARCH_PROXY_LEARNING_RATE = 0.05
#: 只对这两族换学习率(catboost 强默认本就是 0.05,不参与代理)
SEARCH_PROXY_ALGOS = ("lightgbm", "xgboost")
FULL_RECIPE = {"mode": "full", "learning_rate": None, "refit": True}


def resolve_search_recipe(setting, n_rows: int = 0, n_features: int = 0, *,
                          threshold_cells: float | None = None,
                          proxy_learning_rate: float | None = None) -> dict:
    """解析 ``baseline.search_recipe``:auto | full | proxy。

    auto 与 ``n_seeds: auto`` 同一规模阈值:train 行数 × 特征数 ≥ 阈值 →
    代理,否则完整(小数据上完整配方本就便宜,没有理由接受代理的噪声)。
    返回 {"mode", "learning_rate", "refit", "reason"};learning_rate=None 表示
    沿用 strong_params 的缺省。
    """
    s = "auto" if setting is None else str(setting).strip().lower()
    if s in ("false", "0", "off", "no"):
        s = "full"
    lr = float(proxy_learning_rate or SEARCH_PROXY_LEARNING_RATE)
    proxy = {"mode": "proxy", "learning_rate": lr, "refit": False}
    if s == "full":
        return {**FULL_RECIPE, "reason": "显式配置 baseline.search_recipe=full"}
    if s == "proxy":
        return {**proxy, "reason": "显式配置 baseline.search_recipe=proxy"}
    if s != "auto":
        raise ValueError(f"baseline.search_recipe 取值非法: {setting!r}"
                         "(应为 auto / full / proxy)")
    thr = float(threshold_cells or AUTO_SEED_CELLS_THRESHOLD)
    cells = float(n_rows or 0) * float(n_features or 0)
    if cells >= thr:
        return {**proxy, "reason": (
            f"auto:train {int(n_rows)}×{int(n_features)} ≈ {cells:.2e} 格 ≥ "
            f"{thr:.0e} → 搜索期代理配方(lr {lr:g} 单遍早停);交付升级为完整配方")}
    return {**FULL_RECIPE, "reason": (
        "auto:train 规模未知,按完整配方" if cells <= 0 else
        f"auto:train {int(n_rows)}×{int(n_features)} ≈ {cells:.2e} 格 < "
        f"{thr:.0e} → 完整配方")}


def apply_search_recipe(params: dict, algorithm: str,
                        recipe: dict | None) -> dict:
    """把代理配方的学习率写进 strong_params 的结果(完整配方原样返回)。"""
    if not recipe or recipe.get("mode") != "proxy" \
            or str(algorithm).lower() not in SEARCH_PROXY_ALGOS:
        return params
    lr = recipe.get("learning_rate")
    return {**params, "learning_rate": float(lr)} if lr else params


def strong_params(algorithm: str, n_rows: int = 0,
                  n_features: int = 0) -> dict:
    """三族 GBDT 的强默认超参(不含 n_estimators/seed/线程/设备)。"""
    algo = str(algorithm or "lightgbm").lower()
    wide = n_features >= 200
    colsample = 0.4 if wide else (0.8 if n_features >= 20 else 1.0)
    min_child = int(min(100, max(20, (n_rows or 0) // 2000)))
    if algo == "lightgbm":
        return {"learning_rate": 0.02, "num_leaves": 63,
                "min_child_samples": min_child, "subsample": 0.8,
                "subsample_freq": 1, "colsample_bytree": colsample,
                "reg_lambda": 5.0}
    if algo == "xgboost":
        return {"learning_rate": 0.02, "max_depth": 7,
                "min_child_weight": 5, "subsample": 0.8,
                "colsample_bytree": colsample, "reg_lambda": 5.0,
                "tree_method": "hist"}
    if algo == "catboost":
        return {"learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 5.0,
                "border_count": 254}
    return {}


def resolve_n_seeds(setting, n_rows: int = 0, n_features: int = 0, *,
                    threshold_cells: float | None = None,
                    small: int | None = None,
                    large: int | None = None) -> tuple[int, str]:
    """解析 ``baseline.n_seeds``:整数原样生效;``auto`` 按 train 规模取值。

    返回 (seed 数, 取值理由)。理由写进渲染代码与 trial 日志,事后可审计
    "为什么这个 run 是 1 个 seed"。行列规模未知(splits 未落盘)时按小数据
    口径,不会因为缺信息而把小数据任务降成单 seed。
    """
    small = int(small or AUTO_SEED_SMALL)
    large = int(large or AUTO_SEED_LARGE)
    s = "auto" if setting is None else str(setting).strip().lower()
    if s not in ("auto", ""):
        try:
            n = int(float(s))
        except ValueError:
            raise ValueError(
                f"baseline.n_seeds 取值非法: {setting!r}(应为正整数或 auto)"
            ) from None
        return max(1, n), f"显式配置 baseline.n_seeds={max(1, n)}"
    thr = float(threshold_cells or AUTO_SEED_CELLS_THRESHOLD)
    cells = float(n_rows or 0) * float(n_features or 0)
    if cells <= 0:
        return small, "auto:train 行列规模未知,按小数据口径"
    if cells >= thr:
        return large, (f"auto:train {int(n_rows)}×{int(n_features)} ≈ "
                       f"{cells:.2e} 格 ≥ {thr:.0e} → {large} seed(大数据上"
                       "seed 方差远小于特征差异,bag 只加成本)")
    return small, (f"auto:train {int(n_rows)}×{int(n_features)} ≈ "
                   f"{cells:.2e} 格 < {thr:.0e} → {small} seed")


def frequency_encoding(train_col, other_col):
    """高基数不可数值化列 → 频次编码(按 train 计数/行数;未见值为 0)。

    旧 baseline 对 >100 基数的非数值列整列剔除,信息直接丢失。频次编码
    只用 train 的分布,不接触标签,不构成泄漏。
    """
    import pandas as pd

    freq = train_col.astype(str).value_counts(normalize=True)
    enc_tr = train_col.astype(str).map(freq).astype("float32")
    enc_ot = other_col.astype(str).map(freq).fillna(0.0).astype("float32")
    enc_tr[train_col.isna()] = np.nan
    enc_ot[other_col.isna().to_numpy()] = np.nan
    return pd.Series(enc_tr.to_numpy(), index=train_col.index), \
        pd.Series(enc_ot.to_numpy(), index=other_col.index)


def _seed_key(est) -> str | None:
    params = est.get_params(deep=False)
    for k in ("random_state", "random_seed", "seed"):
        if k in params:
            return k
    return None


def _rounds_of(est) -> int:
    """估计器声明的轮数上限(n_estimators/iterations);拿不到返回 0。"""
    try:
        params = est.get_params(deep=False)
    except Exception:  # noqa: BLE001
        return 0
    for k in ("n_estimators", "iterations", "num_boost_round", "num_trees"):
        v = params.get(k)
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return 0


# ---------------------------------------------------------------------------
# 预算感知(v2.14.3)
# ---------------------------------------------------------------------------

class BudgetInfeasible(RuntimeError):
    """预算感知快速失败:本 trial 在截止时间内训不完。

    异常文本以 ``[budget_infeasible]`` 开头;runner 的失败分类据此归为
    ``budget_infeasible``——它不在 RETRYABLE_FAILURES 里:同样的计算换
    同样的超时再跑一遍只会再失败一次(v2.14.2 实测:两个 baseline 各
    "超时 → 降配重试 → 再超时",白烧约 6h)。
    """

    MARKER = "budget_infeasible"

    def __init__(self, detail: dict):
        self.detail = dict(detail)
        super().__init__(f"[{self.MARKER}] {self.detail.get('message', '')}")


class _DeadlineReached(Exception):
    """训练回调内部信号:软截止已到或按下界判定已不可能完成。"""


class _Budget:
    """trial 截止时间(runner 注入 MODELEVO_DEADLINE_TS,已扣 kill grace)。

    reserve_sec 预留给训练之后的打分/落盘(runner 按超时的 8% 注入,
    下限 300s)。MODELEVO_BUDGET_AWARE=0 可显式关闭(诊断用)。
    """

    def __init__(self, deadline_ts: float, reserve_sec: float):
        self.deadline_ts = float(deadline_ts)
        self.reserve_sec = float(reserve_sec)

    @classmethod
    def from_env(cls) -> "_Budget | None":
        if os.environ.get("MODELEVO_BUDGET_AWARE", "1") == "0":
            return None
        raw = os.environ.get("MODELEVO_DEADLINE_TS")
        if not raw:
            return None
        try:
            ts = float(raw)
        except ValueError:
            return None
        try:
            reserve = float(os.environ.get("MODELEVO_BUDGET_RESERVE_SEC")
                            or 300)
        except ValueError:
            reserve = 300.0
        return cls(ts, reserve)

    @property
    def soft_deadline(self) -> float:
        return self.deadline_ts - self.reserve_sec

    def remaining(self, now: float | None = None) -> float:
        return self.soft_deadline - (time.time() if now is None else now)


def _write_progress(payload: dict) -> None:
    """进度快照写 MODELEVO_PROGRESS_FILE(原子替换;失败静默)。"""
    path = os.environ.get("MODELEVO_PROGRESS_FILE")
    if not path:
        return
    try:
        p = Path(path)
        tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"at": time.time(), **payload},
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


class _Progress:
    """单个成员的训练进度:定期打印 + 截止判定。

    es_patience 非 None 表示"定轮后还要全量重训"(es_members 模式):
    此时若定轮已用 el 秒仍未早停,全量重训至少还要训 (it-patience) 棵,
    耗时下界 ≈ el×(it-patience)/it;剩余时间连 80% 的下界都放不下就
    提前判不可行,而不是等 SIGTERM。
    """

    def __init__(self, label: str, total: int, budget: _Budget | None,
                 es_patience: int | None = None):
        self.label = label
        self.total = int(total or 0)
        self.every = max(50, self.total // 20) if self.total else 100
        self.budget = budget
        self.es_patience = es_patience
        self.t0 = time.time()
        self.last_it = 0

    def tick(self, it: int) -> None:
        now = time.time()
        el = now - self.t0
        self.last_it = int(it)
        b = self.budget
        if b is not None:
            rem = b.remaining(now)
            if rem <= 0:
                raise _DeadlineReached(
                    f"{self.label}: 已到软截止(iter={it},预留 "
                    f"{b.reserve_sec:.0f}s 给打分落盘)")
            if self.es_patience is not None and el > 60 and it > 0:
                lower = el * max(0, it - self.es_patience) / float(it)
                if rem < 0.8 * lower:
                    raise _DeadlineReached(
                        f"{self.label}: 定轮已训 {it} 轮/{el:.0f}s 仍未早停,"
                        f"随后的全量重训至少还要约 {lower:.0f}s,"
                        f"剩余仅 {rem:.0f}s")
        if it % self.every == 0 or (self.total and it >= self.total):
            rate = el / max(it, 1)
            eta = rate * max(self.total - it, 0) if self.total else -1.0
            print(f"[progress] {self.label} iter={it}/{self.total} "
                  f"elapsed={el:.0f}s rate={rate:.3f}s/it "
                  f"eta_to_cap={eta:.0f}s", flush=True)
            _write_progress({"status": "training", "label": self.label,
                             "iter": int(it), "total": self.total,
                             "elapsed_sec": round(el, 1),
                             "sec_per_iter": round(rate, 4),
                             "eta_to_cap_sec": round(eta, 1)})


def _fit_member(m, X, y, kw: dict, prog: _Progress):
    """拟合单个成员并挂进度回调(xgboost/lightgbm);其余族直接 fit。

    xgboost 的回调是构造参数:拟合后恢复原值,落盘模型不携带回调对象。
    """
    fam = str(type(m).__module__ or "").split(".")[0]
    if fam == "xgboost":
        import xgboost as xgb

        class _XgbProgress(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):  # noqa: D401
                prog.tick(epoch + 1)
                return False

        prev = m.get_params().get("callbacks")
        m.set_params(callbacks=list(prev or []) + [_XgbProgress()])
        try:
            return m.fit(X, y, **kw)
        finally:
            m.set_params(callbacks=prev)
    if fam == "lightgbm":
        def _lgb_progress(env):
            prog.tick(env.iteration + 1)

        # lightgbm 按 callback.order 排序(无该属性直接 AttributeError);
        # 40 = 排在 early_stopping(30)之后,早停轮不再多打一行
        _lgb_progress.order = 40
        kw = dict(kw)
        kw["callbacks"] = list(kw.get("callbacks") or []) + [_lgb_progress]
        return m.fit(X, y, **kw)
    return m.fit(X, y, **kw)


# ---------------------------------------------------------------------------
# 成员级断点(v2.14.3)
# ---------------------------------------------------------------------------

#: 指纹忽略的易变参数:线程/设备/日志/回调。重试即使换了线程或设备,
#: 已训成员依然可复用(同一配方、同一数据)。
_CKPT_VOLATILE = frozenset((
    "n_jobs", "nthread", "thread_count", "num_threads", "device", "devices",
    "task_type", "gpu_id", "verbosity", "verbose", "silent", "callbacks"))


def _frame_digest(h, X) -> None:
    """帧指纹:形状 + 列名 + 均匀抽 ≤4096 行的内容哈希(大帧也是毫秒级)。"""
    import pandas as pd

    n = len(X)
    idx = (np.unique(np.linspace(0, n - 1, num=min(n, 4096))
                     .astype(np.int64)) if n else np.array([], np.int64))
    if isinstance(X, pd.DataFrame):
        h.update(repr((X.shape, [str(c) for c in X.columns])).encode())
        h.update(pd.util.hash_pandas_object(X.iloc[idx], index=False)
                 .to_numpy().tobytes())
    else:
        arr = np.asarray(X)
        h.update(repr((arr.shape, str(arr.dtype))).encode())
        h.update(np.ascontiguousarray(arr[idx]).tobytes())


def _array_digest(h, y) -> None:
    arr = np.ascontiguousarray(np.asarray(y))
    h.update(repr((arr.shape, str(arr.dtype))).encode())
    h.update(arr.tobytes())


class _Checkpointer:
    """SeedBag 成员落盘/复用。runner 注入 MODELEVO_CKPT_DIR 时启用;
    smoke 轮不落盘。指纹覆盖基估计器配方、bag 参数、本轮是定轮还是全量、
    训练帧/标签(与 eval 帧)内容——任何一项变化都不复用。"""

    def __init__(self, root: Path, fp: str):
        self.root = Path(root)
        self.fp = fp

    @classmethod
    def from_env(cls, bag, X, y, eval_set, pass_tag: str):
        root = os.environ.get("MODELEVO_CKPT_DIR")
        if not root or os.environ.get("MODELEVO_SMOKE_RUN") == "1":
            return None
        try:
            h = hashlib.sha256()
            base = bag.base_estimator
            params = {k: repr(v) for k, v in
                      sorted(base.get_params(deep=False).items())
                      if k not in _CKPT_VOLATILE}
            h.update(repr((type(base).__module__, type(base).__name__,
                           params)).encode())
            h.update(repr((int(bag.n_seeds), int(bag.seed_stride),
                           bag.es_rounds, getattr(bag, "es_members", None),
                           pass_tag)).encode())
            _frame_digest(h, X)
            _array_digest(h, y)
            if eval_set is not None:
                X_es, y_es = eval_set[0] if isinstance(eval_set, list) \
                    else eval_set
                _frame_digest(h, X_es)
                _array_digest(h, y_es)
            return cls(Path(root), h.hexdigest()[:20])
        except Exception as exc:  # noqa: BLE001  断点是优化,失败不阻塞训练
            print(f"[warn] seed bag 断点关闭(指纹计算失败: {exc})",
                  flush=True)
            return None

    def _path(self, i: int) -> Path:
        return self.root / f"seedbag-{self.fp}-m{i}.joblib"

    def load(self, i: int):
        p = self._path(i)
        if not p.exists():
            return None
        try:
            import joblib

            obj = joblib.load(p)
            if obj.get("fp") != self.fp:
                return None
            return obj["model"], float(obj.get("seconds") or 0.0)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 断点 {p.name} 读取失败,重训该成员: {exc}",
                  flush=True)
            return None

    def save(self, i: int, model, seconds: float) -> None:
        try:
            import joblib

            self.root.mkdir(parents=True, exist_ok=True)
            p = self._path(i)
            tmp = p.with_name(f"{p.name}.tmp{os.getpid()}")
            joblib.dump({"fp": self.fp, "model": model,
                         "seconds": float(seconds), "member": int(i)}, tmp)
            os.replace(tmp, p)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 断点落盘失败(不影响训练): {exc}", flush=True)


class SeedBag(BaseEstimator):
    """K 个不同 seed 的同配方估计器取平均(sklearn 兼容)。

    - fit(X, y, eval_set=None):给 eval_set 时为"定轮"——es_members=None
      (旧行为)全部成员各自早停,es_members=k 只训前 k 个成员;
      best_iteration_ 取已训成员最佳轮数均值;
    - predict_proba / predict:成员平均;
    - 与交付链兼容:runtime.delivery 的 best_rounds / fitted_rounds /
      prepare_refit / param_fidelity 均识别本类(按成员/基估计器处理)。
    - v2.14.3:成员级断点、预算感知、进度行(见模块 docstring)。拟合后
      可读 n_seeds_effective_(实际成员数)、budget_reduced_、
      resumed_members_、cost_per_tree_row_(秒/(树·行),供重训预算估算)。

    seed bagging 同时降低两件事的噪声:交付模型本身的方差,以及候选比较
    中的训练随机性(σ_seed)——后者是 09-17/09-23 确认门频繁失守的来源。
    """

    def __init__(self, base_estimator=None, n_seeds: int = 3,
                 seed_stride: int = 1000, es_rounds: int | None = None,
                 es_members: int | None = None,
                 cost_hint: float | None = None):
        self.base_estimator = base_estimator
        self.n_seeds = n_seeds
        self.seed_stride = seed_stride
        self.es_rounds = es_rounds
        self.es_members = es_members
        self.cost_hint = cost_hint

    def __setstate__(self, state):
        # v2.14.2 及以前落盘的 bag 没有新参数;补缺省值,否则 get_params /
        # clone(prepare_refit)在旧工件上抛 AttributeError
        state.setdefault("es_members", None)
        state.setdefault("cost_hint", None)
        super().__setstate__(state)

    # ---- 训练 ----
    def fit(self, X, y, eval_set=None, **fit_kwargs):
        from .delivery import es_fit_kwargs, fitted_rounds

        if self.base_estimator is None:
            raise ValueError("SeedBag 需要 base_estimator")
        key = _seed_key(self.base_estimator)
        base_seed = int((self.base_estimator.get_params(deep=False).get(key)
                         if key else 0) or 0)
        es_pass = eval_set is not None and bool(self.es_rounds)
        n_plan = max(1, int(self.n_seeds))
        es_members = getattr(self, "es_members", None)
        n_fit = (min(n_plan, max(1, int(es_members)))
                 if es_pass and es_members else n_plan)
        pass_tag = "es" if es_pass else "fit"
        budget = _Budget.from_env()
        ckpt = _Checkpointer.from_env(self, X, y,
                                      eval_set if es_pass else None, pass_tag)
        n_rows = len(X)
        n_rounds = _rounds_of(self.base_estimator)
        self.members_ = []
        self.member_seconds_ = []
        self.budget_reduced_ = False
        self.resumed_members_ = 0
        for i in range(n_fit):
            label = f"seedbag[{pass_tag}] member {i + 1}/{n_fit}"
            cached = ckpt.load(i) if ckpt is not None else None
            if cached is not None:
                self.members_.append(cached[0])
                self.member_seconds_.append(cached[1])
                self.resumed_members_ += 1
                print(f"[step] seed bag member {i + 1}/{n_fit} resumed from "
                      f"checkpoint(原耗时 {cached[1]:.0f}s)", flush=True)
                continue
            if budget is not None:
                est = self._estimate_member_sec(n_rows, n_rounds)
                rem = budget.remaining()
                if est is not None and est > rem:
                    if not self.members_:
                        raise self._infeasible(budget, label, est, rem)
                    self._reduce(i, n_fit, f"下一成员预计 {est:.0f}s,"
                                           f"剩余 {rem:.0f}s")
                    break
            m = clone(self.base_estimator)
            if key:
                m.set_params(**{key: base_seed + i * int(self.seed_stride)})
            kw = dict(fit_kwargs)
            if es_pass:
                X_es, y_es = eval_set[0] if isinstance(eval_set, list) \
                    else eval_set
                kw.update(es_fit_kwargs(m, X_es, y_es, int(self.es_rounds)))
            prog = _Progress(label, n_rounds, budget,
                             es_patience=(int(self.es_rounds)
                                          if es_pass and es_members else None))
            t0 = time.time()
            try:
                fitted = _fit_member(m, X, y, kw, prog)
            except _DeadlineReached as exc:
                if not self.members_:
                    raise self._infeasible(budget, label, None,
                                           budget.remaining(),
                                           reason=str(exc)) from None
                self._reduce(i, n_fit, str(exc))
                break
            sec = time.time() - t0
            member = fitted if fitted is not None else m
            self.members_.append(member)
            self.member_seconds_.append(sec)
            if ckpt is not None:
                ckpt.save(i, member, sec)
            # 训练期进度(trial stdout 可见;bag 成员串行,长训练时有心跳可读)
            print(f"[step] seed bag member {i + 1}/{n_fit} fitted "
                  f"({sec:.0f}s)", flush=True)
        self.n_seeds_effective_ = len(self.members_)
        rates = []
        for mm, sec in zip(self.members_, self.member_seconds_):
            trees = fitted_rounds(mm)
            if trees and n_rows and sec > 0:
                rates.append(sec / (float(trees) * n_rows))
        self.cost_per_tree_row_ = float(np.mean(rates)) if rates else None
        if hasattr(self.members_[0], "classes_"):
            self.classes_ = self.members_[0].classes_
        return self

    def _estimate_member_sec(self, n_rows: int, n_rounds: int):
        """下一成员的预计耗时:本轮已训成员的均值;首个成员用 cost_hint
        (定轮阶段实测的 秒/(树·行))× 轮数 × 行数;都没有时返回 None。"""
        if self.member_seconds_:
            return float(np.mean(self.member_seconds_))
        hint = getattr(self, "cost_hint", None)
        if hint and n_rounds and n_rows:
            return float(hint) * float(n_rounds) * float(n_rows)
        return None

    def _reduce(self, i: int, n_fit: int, why: str) -> None:
        self.budget_reduced_ = True
        msg = (f"seed bag 按预算缩减:计划 {n_fit} 个成员 → 实际 {i} 个"
               f"({why})。已记入 n_seeds_effective,后续 trial 按实际值对齐")
        print(f"[budget] {msg}", flush=True)
        _write_progress({"status": "budget_reduced", "planned": n_fit,
                         "effective": i, "message": msg})

    def _infeasible(self, budget: _Budget, label: str, est, rem,
                    reason: str | None = None) -> BudgetInfeasible:
        timeout_sec = None
        try:
            timeout_sec = float(os.environ.get("MODELEVO_TIMEOUT_SEC") or 0)
        except ValueError:
            timeout_sec = None
        suggest = None
        if timeout_sec and est is not None:
            # 当前超时 + 缺口,再留 20% 余量
            suggest = int(math.ceil((timeout_sec + max(est - rem, 0.0))
                                    * 1.2 / 60.0))
        parts = [f"{label} 在截止时间内训不完"]
        if reason:
            parts.append(reason)
        if est is not None:
            parts.append(f"预计需要 {est:.0f}s,剩余可用 {rem:.0f}s"
                         f"(已预留 {budget.reserve_sec:.0f}s 给打分落盘)")
        advice = ("调大 task-spec 的 resource_budget.trial_timeout_minutes"
                  + (f"(建议 ≥ {suggest} 分钟)" if suggest else "")
                  + ",或缩小配方(baseline.n_seeds / "
                    "baseline.strong_n_estimators_cap)后新建 trial;"
                    "同样的超时重跑只会再失败一次,因此不自动重试")
        detail = {"status": "budget_infeasible", "label": label,
                  "estimate_sec": None if est is None else round(est, 1),
                  "remaining_sec": round(float(rem), 1),
                  "reserve_sec": budget.reserve_sec,
                  "suggest_timeout_minutes": suggest,
                  "message": ";".join(parts) + "。建议:" + advice}
        _write_progress(detail)
        return BudgetInfeasible(detail)

    # ---- 打分 ----
    def predict_proba(self, X):
        return np.mean([np.asarray(m.predict_proba(X))
                        for m in self.members_], axis=0)

    def predict(self, X):
        if hasattr(self.members_[0], "predict_proba") and \
                hasattr(self, "classes_"):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return np.mean([np.asarray(m.predict(X)) for m in self.members_],
                       axis=0)

    # ---- 诊断属性(与单模型同名,下游 dump/leak_scan 直接可用)----
    @property
    def best_iteration_(self) -> int:
        from .delivery import best_rounds

        vals = [best_rounds(m) for m in getattr(self, "members_", [])]
        vals = [v for v in vals if v]
        return int(round(float(np.mean(vals)))) if vals else 0

    @property
    def feature_importances_(self):
        return np.mean([np.asarray(m.feature_importances_, dtype=float)
                        for m in self.members_], axis=0)

    @property
    def evals_result_(self):
        return getattr(self.members_[0], "evals_result_", {})

    @property
    def n_features_in_(self):
        return getattr(self.members_[0], "n_features_in_", None)


def cap_binding(best_rounds: int | None, cap: int | None,
                ratio: float = 0.95) -> bool:
    """早停轮数撞上 cap(≥ ratio×cap):轮数上限是有效约束,模型可能欠拟合。

    v2.14.3:trial-003(2.24M 行、lr 0.02、cap 2000)best_iteration=2000,
    es_valid logloss 最后 100 轮仍降 0.00013——撞顶却无任何提示。
    """
    try:
        return bool(best_rounds and cap and int(best_rounds)
                    >= int(math.floor(ratio * int(cap))))
    except (TypeError, ValueError):
        return False


def es_curve(model) -> list[float] | None:
    """早停评估序列(LGBM evals_result_ / XGB evals_result());取不到返回 None。"""
    ev = getattr(model, "evals_result_", None)
    if ev is None and hasattr(model, "evals_result"):
        try:
            ev = model.evals_result()
        except Exception:  # noqa: BLE001
            ev = None
    try:
        first = next(iter(ev.values()))
        return [float(v) for v in first[next(iter(first.keys()))]]
    except Exception:  # noqa: BLE001
        return None


def cap_tail_gain(curve, tail_frac: float = 0.2,
                  lower_is_better: bool = True) -> dict | None:
    """早停曲线末段(最后 tail_frac 的轮次)相对全程的改善量(按"迄今最优"计)。"""
    if not curve or len(curve) < 10:
        return None
    arr = np.asarray(curve, dtype=float)
    best = (np.minimum.accumulate(arr) if lower_is_better
            else np.maximum.accumulate(arr))
    start = int(len(arr) * (1 - tail_frac))
    total = abs(best[-1] - arr[0])
    tail = abs(best[-1] - best[start - 1])
    return {"tail_frac": tail_frac, "tail_gain": tail,
            "tail_share": tail / total if total > 0 else 0.0}


def cap_advice(best_rounds: int, cap: int, curve=None) -> str:
    """cap 撞顶时的提示(v2.14.3.1:报实测末段收益,不再一律建议调大 cap/学习率)。

    旧提示"调大 cap(或学习率)后重跑"在 400w 动支数据上被实测否定:
    lr 0.02→0.05 验证集 AUC −0.0015、→0.1 −0.0029;cap 5000→5554 棵只多
    ≈+0.0001。同一配方下候选之间的比较不受 cap 撞顶影响;调大 cap 只抬
    绝对水平,成本与轮数成正比。
    """
    msg = (f"[warn] cap_binding:早停轮数 {best_rounds} 撞上上限 {cap}")
    g = cap_tail_gain(curve)
    if g is not None:
        msg += (f";最后 {g['tail_frac']:.0%} 轮次 es_valid 指标只再改善 "
                f"{g['tail_gain']:.5f}(占全程改善 {g['tail_share']:.1%})")
    return msg + ("。同配方候选之间比较不受影响;调大 cap 只抬绝对水平、成本随"
                  "轮数线性增长,按末段收益决定;不要用提高学习率代替(400w 动支"
                  "数据实测 lr 0.02→0.05 AUC −0.0015)")
