"""分组置换重要性(v11):按特征前缀族打乱列,量 validation 掉幅。

修 v10.2 超级 trial 裁决的共线性偏差:gain 是模型内分配量,与已有列
相关的新族 gain 被稀释(gain≈0 不等于无增量),而 gain 高也不等于有
增量。分组置换量的是"把这一族打乱,validation 掉多少"——正是方向
生死的定义,且只需打分不需重训(几百个族几十秒)。

裁决优先级:置换 delta > gain(两者冲突时以置换为准,见
feature-exploration.md 超级 trial 契约)。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .logging_utils import dump_json, get_logger, load_json, now_iso
from .metrics import compute_metric, metric_direction
from .safe_dtype import safe_to_numeric

logger = get_logger(__name__)

#: 未给预算时的置换线程上限(与 execution.parallel_trials_auto_cap 同量级)
_DEFAULT_THREAD_BUDGET = 24


def resolve_worker_threads(workers: int, budget: int | None = None) -> \
        tuple[int, int]:
    """(worker 数, 每个 worker 的预测线程数),两者之积不超过线程预算。

    v2.13.5(req-t7-rand400w 实测回写):旧实现 4 个 worker 各自用库默认
    线程(= 全部核)并发预测,并发 4~6 个 trial 同时评估时线程数上百倍
    超订;同一批 trial 的 GI 从单独跑的 3 min 拉长到 16~24 min。预算
    优先级:环境变量 MODELEVO_GI_THREADS > 调用方(runner 取本 trial 的
    线程配额)> min(cpu 数, 24)。线程数不改变分数(逐行预测与线程划分
    无关)。与 v2.14.3.1 同名函数同口径。
    """
    env = os.environ.get("MODELEVO_GI_THREADS")
    if env:
        try:
            budget = int(env)
        except ValueError:
            pass
    if not budget or budget < 1:
        budget = min(os.cpu_count() or 8, _DEFAULT_THREAD_BUDGET)
    workers = max(1, min(int(workers), int(budget)))
    return workers, max(1, int(budget) // workers)


def _pin_predict_device_cpu(model: Any, n_threads: int | None = None) -> bool:
    """把 GPU 训练的 xgboost booster 切到 CPU 预测(并设线程数)。

    v2.13.5:trial 落盘的 model.joblib 带 device=cuda:*。harness 评估进程
    (gen-executor 的线程池,同进程并发评估多个 trial)用它对内存里的
    numpy 帧打分时,xgboost 每次都走 "mismatched devices → DMatrix 回退"
    (6 s 级的 DMatrix 构建),且所有 trial 挤在同一张卡上。实测
    (trial-024,20 万行 × 1646 列,2000 棵):GPU 回退 1.96 s/次,
    CPU 24 线程 0.37 s/次,GPU/CPU 分数最大差 2.6e-6。基准分与置换分
    在同一设备上计算,置换 delta 口径自洽。返回是否做了切换。

    v2.17.0:复合容器(hurdle 等,自身无 get_booster)逐个子模型切换。
    """
    if not hasattr(model, "get_booster"):
        _subs = [v for v in (getattr(model, "__dict__", None) or {}).values()
                 if hasattr(v, "get_booster")]
        return any([_pin_predict_device_cpu(v, n_threads) for v in _subs])
    booster = None
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
        except Exception:  # noqa: BLE001  未拟合/非 xgboost
            booster = None
    if booster is None or not hasattr(booster, "set_param"):
        return False
    params = {"device": "cpu"}
    if n_threads:
        params["nthread"] = int(n_threads)
    try:
        booster.set_param(params)
    except Exception:  # noqa: BLE001  切设备是优化,失败照旧
        return False
    return True


def infer_families(columns: list[str]) -> dict[str, list[str]]:
    """按列名前缀推断特征族:首个 '_' 前的段为族名,无前缀归 base。

    与超级 trial 的口径前缀约定(seq_/ratio_/miss_/te_/…)一致;
    base 族 = 不带前缀的基础特征面。
    """
    families: dict[str, list[str]] = {}
    for c in columns:
        head = c.split("_", 1)[0] if "_" in c else "base"
        families.setdefault(head, []).append(c)
    return families


class _FastPredict:
    """numpy 快路径打分(v19.4)。

    快路径(位级等价已实测):xgboost `Booster.inplace_predict`
    (带 sklearn 同款 iteration_range=(0, best_iteration+1))/
    lightgbm `Booster.predict`;两者与 sklearn `predict_proba`
    **bit-identical**(同帧同列实测 max|Δ|=0)。仅当构造期探测
    "快路径基准分 == score_fn 基准分(逐位相等)"时启用;否则
    fast=None,全程回退旧 DataFrame 逐帧路径,行为与 v18 一致。

    回退同时覆盖:非 GBM 模型 / uplift 双打分(score_fn 语义复杂,
    不复制)/ dtype 不可转 float32 / best_iteration 语义异常。
    """

    def __init__(self, score_fn, model, X_valid: pd.DataFrame,
                 n_threads: int | None = None):
        self.score_fn = score_fn
        self.model = model
        self.columns = list(X_valid.columns)
        self.col_index = {c: i for i, c in enumerate(self.columns)}
        self._booster = None
        self._iter_range = None
        self.V = None
        self.ok = False
        # v2.13.5:每个 worker 的预测线程(None = 库默认 = 全部核)
        self.n_threads = int(n_threads) if n_threads else None
        try:
            base = np.asarray(score_fn(X_valid), dtype=float)
        except Exception:  # noqa: BLE001
            return
        try:
            V = np.ascontiguousarray(X_valid.to_numpy(dtype=np.float32))
            fast = self._fast_predict(V)
            if fast is not None and np.array_equal(fast, base):
                self.V = V
                self.ok = True
        except Exception:  # noqa: BLE001
            self.ok = False

    def _fast_predict(self, V: np.ndarray) -> np.ndarray | None:
        m = self.model
        booster = None
        iter_range = None
        if hasattr(m, "get_booster"):          # xgboost sklearn wrapper
            booster = m.get_booster()
            bi = getattr(booster, "best_iteration", None)
            if bi is not None and bi > 0:
                iter_range = (0, bi + 1)
        elif hasattr(m, "booster_"):           # lightgbm sklearn wrapper
            booster = m.booster_
        if booster is None:
            return None
        self._booster = booster
        self._iter_range = iter_range
        return np.asarray(self._raw(V), dtype=float)

    def _raw(self, V: np.ndarray):
        booster = self._booster
        try:
            if self._iter_range is not None:
                return booster.inplace_predict(
                    V, iteration_range=self._iter_range)
            return booster.inplace_predict(V)
        except (TypeError, AttributeError):
            # lightgbm Booster:线程数随调用传入(xgboost 已在加载期设 nthread)
            if self.n_threads:
                try:
                    return booster.predict(V, num_threads=self.n_threads)
                except TypeError:
                    pass
            return booster.predict(V)

    def clone(self) -> "_FastPredict":
        """并行 worker 的独立副本:仅 V(buffer)深拷贝,模型句柄共享
        (xgboost/lightgbm 的 predict 是线程安全的读路径)。"""
        import copy as _copy

        new = _copy.copy(self)
        new.V = self.V.copy()
        return new

    def permuted_score(self, j_idx: np.ndarray,
                       perm: np.ndarray) -> np.ndarray | None:
        """打乱 j_idx 列 → 打分 → 恢复。语义 = 旧实现的
        Xp[c]=Xp[c].values[perm](逐位一致);None = 无快路径,
        调用方回退 DataFrame 路径。"""
        if not self.ok:
            return None
        V = self.V
        j = j_idx
        orig = V[:, j].copy()
        V[:, j] = V[perm[:, None], j]
        try:
            return np.asarray(self._raw(V), dtype=float)
        finally:
            V[:, j] = orig


class _GenericPredict:
    """v2.17.0(#14)通用置换路径:按族置换 → 容器 predict → 主指标。

    适用于没有原生 GBM 快路径的模型(hurdle 容器、任意只有 predict 的
    对象)。旧回退路径每个族×每次置换整帧 `X_valid.copy()`(44.7 万行 ×
    1600 列 ≈ 2.8 GB/次),且复合容器在加载阶段就失败 → GI 1.7 s 空跑。
    这里持一份 float32 numpy 缓冲,原位打乱该族列 → 以零拷贝 DataFrame
    视图调 score_fn → 恢复;语义与旧回退逐位一致(同 perm 数组、同打分
    函数、同 float32 帧)。

    submodels=(p_fn, m_fn) 且容器打分 == p × max(m, 0)(构造期逐位校验)
    时,额外给出按子模型的置换贡献:只把置换帧喂给分类子模型
    (clf-only)或回归子模型(reg-only),另一侧用基准帧分数。
    """

    def __init__(self, score_fn, X_valid: pd.DataFrame, submodels=None):
        self.score_fn = score_fn
        self.columns = list(X_valid.columns)
        self.col_index = {c: i for i, c in enumerate(self.columns)}
        self.ok = False
        self.parts = None
        self.parts_exact = False
        self.p_base = self.m_base = None
        try:
            # 列主序(F-order):与 pandas 单 float32 block 的内存布局一致,
            # 线性模型等 BLAS 路径的求和顺序与 DataFrame 打分逐位相同
            self.V = np.asfortranarray(X_valid.to_numpy(dtype=np.float32))
            base_df = self._frame()
            base = np.asarray(score_fn(base_df), dtype=float)
            ref = np.asarray(score_fn(X_valid), dtype=float)
            self.ok = bool(np.array_equal(base, ref, equal_nan=True))
        except Exception:  # noqa: BLE001
            self.ok = False
            return
        if self.ok and submodels is not None:
            try:
                p_fn, m_fn = submodels
                # 子模型原生 dtype 相乘(xgboost 输出 float32;容器 predict
                # 也是 float32 乘法),与容器打分逐位一致
                p0 = np.asarray(p_fn(base_df))
                m0 = np.asarray(m_fn(base_df))
                comp = np.asarray(p0 * np.clip(m0, 0.0, None), dtype=float)
                if np.allclose(comp, base, rtol=1e-6, atol=1e-9,
                               equal_nan=True):
                    self.parts = (p_fn, m_fn)
                    self.p_base, self.m_base = p0, m0
                    # 逐位相等时主 delta 直接用分解合成(省一次容器 predict)
                    self.parts_exact = bool(np.array_equal(
                        comp, base, equal_nan=True))
            except Exception:  # noqa: BLE001  分解只是附加诊断
                self.parts = None

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.V, columns=self.columns, copy=False)

    def permuted(self, j_idx: np.ndarray, perm: np.ndarray) -> dict:
        """返回 {"full": s, "clf": s_c, "reg": s_r}(后两者仅分解可用时)。"""
        V = self.V
        orig = V[:, j_idx].copy()
        V[:, j_idx] = V[perm[:, None], j_idx]
        try:
            Xp = self._frame()
            if self.parts is not None:
                p_fn, m_fn = self.parts
                p = np.asarray(p_fn(Xp))
                m = np.asarray(m_fn(Xp))
                return {
                    "full": (np.asarray(p * np.clip(m, 0.0, None),
                                        dtype=float) if self.parts_exact
                             else np.asarray(self.score_fn(Xp), dtype=float)),
                    "clf": np.asarray(p * np.clip(self.m_base, 0.0, None),
                                      dtype=float),
                    "reg": np.asarray(self.p_base * np.clip(m, 0.0, None),
                                      dtype=float),
                }
            return {"full": np.asarray(self.score_fn(Xp), dtype=float)}
        finally:
            V[:, j_idx] = orig


def _hurdle_parts(model: Any):
    """hurdle 形态容器的两段打分函数 (p_fn, m_fn);不识别返回 None。

    识别口径(与 run4 regression 手写 HurdleXGB / 模板 hurdle 段一致):
    positive_prob + conditional_mean 方法,或 clf_(predict_proba)+
    reg_(predict)子模型属性。是否真是 p × max(m, 0) 由 _GenericPredict
    构造期逐位校验,不符则不分解。
    """
    if hasattr(model, "positive_prob") and hasattr(model, "conditional_mean"):
        return (lambda X: model.positive_prob(X),
                lambda X: model.conditional_mean(X))
    clf = getattr(model, "clf_", None)
    reg = getattr(model, "reg_", None)
    if clf is not None and reg is not None and \
            hasattr(clf, "predict_proba") and hasattr(reg, "predict"):
        return (lambda X: clf.predict_proba(X)[:, 1],
                lambda X: reg.predict(X))
    return None

def group_permutation_importance(
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    X_valid: pd.DataFrame,
    y: np.ndarray,
    columns_by_family: dict[str, list[str]],
    metric: str,
    *,
    n_repeat: int = 1,
    seed: int = 20260814,
    treatment: np.ndarray | None = None,
    trial_dir: Path | None = None,
    model: Any = None,
    threads: int | None = None,
    submodels: Any = None,
) -> pd.DataFrame:
    """对每个特征族:打乱该族全部列 → 重打分 → 量 metric 掉幅。

    score_fn(X) 必须返回与 len(X) 等长的 score 数组(通常是对已落盘
    model/model.joblib 的 predict 包装)。delta>0(方向感知后)即该族
    扰动伤害表现 → 族承载真实增量;delta≈0 → 与 delta<0(打乱反而
    变好)都判死。n_repeat>1 时报均值与 std,缓解单一置换的偶然性。
    threads:本次置换的总线程预算(见 resolve_worker_threads)。
    submodels:(p_fn, m_fn) —— v2.17.0 hurdle 容器按子模型分解(见
    _GenericPredict;只在无 GBM 快路径时生效)。

    返回 DataFrame(family/n_cols/delta_mean/delta_std/rel_delta),
    按 |delta_mean| 降序;子模型分解可用时附 delta_clf_mean/delta_reg_mean。
    """
    direction = metric_direction(metric)
    rng = np.random.default_rng(seed)
    try:
        workers = int(os.environ.get("MODELEVO_GI_WORKERS", "4"))
    except ValueError:
        workers = 4
    workers, per_worker_threads = resolve_worker_threads(workers, threads)
    # v2.13.5:基准分与置换分必须同一设备——切 CPU(并设线程数)放在
    # 基准分之前;xgboost 的线程数挂在共享 booster 上(worker 共享句柄)
    if model is not None:
        _pin_predict_device_cpu(model, per_worker_threads)
    base_score = np.asarray(score_fn(X_valid), dtype=float)
    base_val = compute_metric(metric, y, base_score, score=base_score,
                              treatment=treatment)
    # v18/P0-2:checkpoint 断点续算。评估段(84 族×5 repeat 在 400w 尺
    # ~1h)两次被外部机制 kill 后 state 残留 RUNNING,重评从零开始——
    # 方向裁决长期缺位。增量结果按族落盘 feature/.gi-checkpoint.json,
    # 重跑只补缺失族;base_score/base_val 由模型与帧决定,帧一致时
    # 复用 checkpoint 里的基准值(哈希校验),帧变了则整份作废。
    ck_path = None
    ck_dir = Path(trial_dir) / "feature" if trial_dir else None
    if ck_dir is not None:
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck_path = ck_dir / ".gi-checkpoint.json"
    done: dict[str, dict] = {}
    ck_base = None
    if ck_path is not None and ck_path.exists():
        try:
            ck = load_json(ck_path)
            if ck.get("frame_sha") == _frame_sha(X_valid) \
                    and ck.get("metric") == metric \
                    and ck.get("n_repeat") == int(n_repeat):
                done = ck.get("families") or {}
                ck_base = ck.get("base")
                logger.info("分组置换断点续算:%d/%d 族已有 checkpoint",
                            len(done), len(columns_by_family))
            else:
                logger.info("分组置换 checkpoint 帧口径不符,整份作废")
        except Exception as exc:  # noqa: BLE001
            logger.warning("分组置换 checkpoint 读取失败(忽略): %s", exc)
    if ck_base is not None and abs(ck_base - base_val) < 1e-12:
        base_val = ck_base
    else:
        # v2.13.5:基准值不符 = 模型/打分设备变了(如旧 checkpoint 在 GPU
        # 回退路径上算的),各族 delta 与本次基准不同源,整份作废
        # (旧实现只丢基准值、仍复用各族结果 → 混口径)
        if done:
            logger.info("分组置换 checkpoint 基准值不符,整份作废")
        done = {}
        ck_base = None
    # ---- v19.4:向量化 + 按族并行(位级等价重写) ----
    # 口径不变性(全部保持):
    #  - rng 抽取序不变:仍按 columns_by_family 迭代序 × n_repeat 逐个
    #    rng.permutation(len(X)),与旧实现随机序列逐位一致 →
    #    delta_mean/delta_std 不变(实测 bit-identical);
    #  - 逐族 checkpoint 语义不变,但收拢主线程串行落盘——
    #    _dump_checkpoint 是整文件 JSON 重写,并发写同一文件会互相
    #    截断,绝不下放 worker;
    #  - 置换语义不变:Xp[c] = Xp[c].values[perm](旧)≡
    #    buf[:, j] = buf[perm[:, None], j](新,同一 perm 数组)。
    # 变化的只有三件事:
    #  1. pandas 整帧 copy + 逐列赋值 → numpy 位置索引;
    #  2. 打分走 numpy 快路径(xgboost inplace_predict / lightgbm
    #     booster.predict,与 sklearn predict_proba 实测 bit-identical;
    #     无快路径时回退逐帧 DataFrame score_fn 旧路径);
    #  3. 按族并行:worker 各持独立 buffer 副本与独立快路径句柄,
    #     异常处理/结果聚合/checkpoint 全在主线程。
    n_rows = len(X_valid)
    work: list[tuple[str, list[str], list[np.ndarray]]] = []
    for fam, cols in columns_by_family.items():
        if fam in done:
            continue
        cols_a = [c for c in cols if c in X_valid.columns]
        if not cols_a:
            continue
        perms = [rng.permutation(n_rows)
                 for _ in range(max(1, int(n_repeat)))]
        work.append((fam, cols_a, perms))
    fast = _FastPredict(score_fn, model, X_valid,
                        n_threads=per_worker_threads) \
        if model is not None else None
    generic = None
    if fast is None or not fast.ok:
        # 无快路径:DataFrame 回退路径里 X_valid.copy() 本身线程安全,
        # 但 score_fn 的模型 predict 未必;保守单线程,行为同 v18
        workers = 1
        # v2.17.0:全数值帧走通用缓冲路径(原位置换,免整帧拷贝;复合容器
        # 可按子模型分解);混合 dtype 帧仍走旧 DataFrame 路径
        try:
            _all_num = all(pd.api.types.is_numeric_dtype(t)
                           for t in X_valid.dtypes)
        except Exception:  # noqa: BLE001
            _all_num = False
        if _all_num and len(X_valid):
            generic = _GenericPredict(score_fn, X_valid, submodels=submodels)
            if not generic.ok:
                generic = None
        logger.info("分组置换无 numpy 快路径,回退%s单线程路径",
                    "通用缓冲(容器 predict)" if generic is not None
                    else " DataFrame ")
    results: dict[str, list[float] | None] = {}
    sub_results: dict[str, dict[str, list[float]]] = {}

    def _metric_delta(s: np.ndarray) -> float | None:
        if not np.isfinite(s).all():
            return None
        v = compute_metric(metric, y, s, score=s, treatment=treatment)
        return float((base_val - v) if direction else (v - base_val))

    def _eval_family(args, f):
        fam, cols_a, perms = args
        deltas: list[float] = []
        try:
            if f is not None and f.ok:
                j_idx = np.asarray(
                    [f.col_index[c] for c in cols_a], dtype=np.intp)
                for perm in perms:
                    s = f.permuted_score(j_idx, perm)
                    if s is None:
                        return fam, None
                    if not np.isfinite(s).all():
                        continue
                    v = compute_metric(metric, y, s, score=s,
                                       treatment=treatment)
                    d = (base_val - v) if direction else (v - base_val)
                    deltas.append(float(d))
            elif generic is not None:
                j_idx = np.asarray(
                    [generic.col_index[c] for c in cols_a], dtype=np.intp)
                subs: dict[str, list[float]] = {}
                for perm in perms:
                    out = generic.permuted(j_idx, perm)
                    d = _metric_delta(out["full"])
                    if d is None:
                        continue
                    deltas.append(d)
                    for part in ("clf", "reg"):
                        if part in out:
                            dp = _metric_delta(out[part])
                            if dp is not None:
                                subs.setdefault(part, []).append(dp)
                if subs:
                    sub_results[fam] = subs
            else:
                # 回退:与 v18 相同的 DataFrame 逐帧路径(慢但通用;
                # 非 GBM / uplift / dtype 异常时走这里)
                for perm in perms:
                    Xp = X_valid.copy()
                    for c in cols_a:
                        Xp[c] = Xp[c].values[perm]
                    s = np.asarray(score_fn(Xp), dtype=float)
                    if not np.isfinite(s).all():
                        continue
                    v = compute_metric(metric, y, s, score=s,
                                       treatment=treatment)
                    d = (base_val - v) if direction else (v - base_val)
                    deltas.append(float(d))
        except Exception as exc:  # noqa: BLE001
            logger.warning("family %s 置换打分失败: %s", fam, exc)
            return fam, None
        return fam, (deltas or None)

    if workers > 1 and len(work) > 1:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        tls = threading.local()
        pool = None
        try:
            pool = ThreadPoolExecutor(max_workers=workers)

            def _job(args):
                if not hasattr(tls, "fast"):
                    tls.fast = fast.clone() if fast is not None else None
                return _eval_family(args, tls.fast)

            for fam, deltas in pool.map(_job, work):
                results[fam] = deltas
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
    else:
        for args in work:
            fam, deltas = _eval_family(args, fast)
            results[fam] = deltas

    rows = []
    for fam, cols_a, _perms in work:
        deltas = results.get(fam)
        if not deltas:
            continue
        dm = float(np.mean(deltas))
        rows.append({
            "family": fam,
            "n_cols": len(cols_a),
            "delta_mean": dm,
            "delta_std": float(np.std(deltas)) if len(deltas) > 1 else 0.0,
            "rel_delta": dm / max(abs(float(base_val)), 1e-9),
        })
        for part, vals in (sub_results.get(fam) or {}).items():
            rows[-1][f"delta_{part}_mean"] = float(np.mean(vals))
        if ck_path is not None:
            done[fam] = dict(rows[-1])
            _dump_checkpoint(ck_path, X_valid, metric, n_repeat,
                             base_val, done)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("delta_mean", ascending=False,
                            key=lambda s: s.abs()).reset_index(drop=True)
    df.attrs["predict_path"] = ("gbm_fast" if fast is not None and fast.ok
                                else "generic_container" if generic is not None
                                else "dataframe_fallback")
    df.attrs["submodel_decomposition"] = bool(
        generic is not None and generic.parts is not None)
    return df


def _frame_sha(X: pd.DataFrame) -> str:
    """评估帧指纹(列名+行数+shape):checkpoint 帧一致性校验。"""
    import hashlib

    h = hashlib.sha256()
    h.update("|".join(map(str, X.columns)).encode())
    h.update(f"|{len(X)}x{X.shape[1]}".encode())
    return h.hexdigest()[:16]


def _dump_checkpoint(path: Path, X: pd.DataFrame, metric: str,
                     n_repeat: int, base_val: float,
                     done: dict) -> None:
    try:
        dump_json(path, {
            "frame_sha": _frame_sha(X),
            "metric": metric,
            "n_repeat": int(n_repeat),
            "base": base_val,
            "families": done,
            "updated_at": now_iso(),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("分组置换 checkpoint 写盘失败: %s", exc)


#: joblib 反序列化期间临时向 __main__ 注入 trial 类的互斥锁(gen-executor
#: 线程池并发评估多个 trial,不同 trial 可能各自定义同名容器类)
_MAIN_INJECT_LOCK = __import__("threading").Lock()


def _load_trial_code_module(trial_dir: Path):
    """以隔离模块名加载 trial 的 code/<entry>.py(只执行顶层定义,不跑 main)。

    v2.17.0(#14):trial 手写的复合容器(HurdleXGB 等)定义在 trial 进程的
    `__main__` 里,joblib 落盘的是 `__main__.HurdleXGB` 引用;harness 评估
    进程的 __main__ 是 modelevo.py → 反序列化抛 "Can't get attribute
    'HurdleXGB' on <module '__main__'>",GI 1.7 s 空跑(run4 gen-03~05 全部
    hurdle trial)。加载期间 sys.path / sys.argv 原样恢复(trial 顶层会把
    渲染期 SKILL_ROOT 插进 sys.path)。
    """
    import importlib.util
    import sys as _sys

    tdir = Path(trial_dir)
    rec: dict = {}
    try:
        import yaml as _yaml

        _ty = tdir / "trial.yaml"
        if _ty.exists():
            rec = _yaml.safe_load(_ty.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        rec = {}
    entry = str((rec.get("execution") or {}).get("code_entry")
                or "train.py")
    path = tdir / "code" / entry
    if not path.exists():
        raise FileNotFoundError(f"trial 代码不存在: {path}")
    mod_name = "modelevo_trial_" + "".join(
        ch if ch.isalnum() else "_" for ch in tdir.name)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    saved_path, saved_argv = list(_sys.path), list(_sys.argv)
    try:
        _sys.argv = [str(path)]
        spec.loader.exec_module(module)
    finally:
        _sys.path[:] = saved_path
        _sys.argv = saved_argv
    return module


def _joblib_load_with_trial_main(joblib_path: Path, trial_dir: Path):
    """反序列化引用 `__main__.<X>` 的工件:临时把 trial 代码模块里定义的
    类/函数注入 __main__(只补缺失名,加载后立即撤销,全程持锁)。"""
    import __main__
    import inspect as _inspect

    import joblib

    module = _load_trial_code_module(trial_dir)
    with _MAIN_INJECT_LOCK:
        injected = []
        for name, obj in vars(module).items():
            if (_inspect.isclass(obj) or _inspect.isfunction(obj)) and \
                    getattr(obj, "__module__", None) == module.__name__ and \
                    not hasattr(__main__, name):
                setattr(__main__, name, obj)
                injected.append(name)
        try:
            return joblib.load(joblib_path)
        finally:
            for name in injected:
                try:
                    delattr(__main__, name)
                except AttributeError:
                    pass


def _load_model_predict(trial_dir: Path) -> tuple[Any, list[str]] | None:
    """加载已落盘模型工件(model/model.joblib + feature_columns.json)。

    返回 (model, feature_columns);模型缺 feature_columns/joblib 时
    返回 None(调用方降级:无分组置换,只有 gain)。

    v2.17.0:工件引用 trial `__main__` 里定义的容器类时,按 trial 代码
    模块重建引用后再加载(见 _joblib_load_with_trial_main)。
    """
    import joblib

    mdir = Path(trial_dir) / "model"
    joblib_path = mdir / "model.joblib"
    cols_path = mdir / "feature_columns.json"
    if not joblib_path.exists() or not cols_path.exists():
        return None
    try:
        try:
            model = joblib.load(joblib_path)
        except (AttributeError, ModuleNotFoundError, ImportError) as exc:
            if "__main__" not in str(exc):
                raise
            logger.info("模型工件引用 trial __main__ 中的定义(%s),"
                        "按 trial 代码模块重建后加载", exc)
            model = _joblib_load_with_trial_main(joblib_path, trial_dir)
        import json

        cols = json.loads(cols_path.read_text(encoding="utf-8"))
        return model, list(cols)
    except Exception as exc:  # noqa: BLE001
        logger.warning("加载模型工件失败 %s: %s", trial_dir, exc)
        return None


def _coerce_numeric(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """置换打分侧 dtype 治理:只做数值化,不做 one-hot。

    feature_columns.json 记的是 prepare_features **之后**的列名,而
    runner 传进来的 X_valid 是原始 validation 帧——列名对得上,dtype
    对不上(parquet 的 decimal 列读回来是 object/Decimal),LightGBM
    直接抛 "pandas dtypes must be int, float or bool"(实测事故:
    p5_can_use_total_amt: object,基准打分崩,异常被 runner 的
    except 吞掉 → evaluation 无 group_importance,协议头号裁决信号
    静默失踪)。one-hot 列名不在 avail 里,不能在这里补——补了就是
    造模型没见过的列。顺带裁到 avail 列:置换每族要 copy 一次全帧,
    1500 列 × 5 repeat 的整帧拷贝没有必要。

    v2.12.2.8:数值化改走 `safe_to_numeric`——`pd.to_numeric` 在含
    非数字值的长 object 列上会段错误(见 safe_dtype 模块 docstring),
    且会把 24 位十六进制键静默解析成数字。不可数值化的列保持原样,
    由调用方的 dtype 检查兜住。
    """
    out = X.loc[:, cols].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(out[c]):
            continue
        conv = safe_to_numeric(out[c])
        if conv is not None:
            out[c] = conv
    return out.fillna(0).astype("float32")


def _load_widened_eval_frame(trial_dir: Path) -> pd.DataFrame | None:
    """读取 trial 自存的模型输入级 validation 帧(model/validation_input.parquet)。

    v15.2 通道:加宽特征 trial(超级 trial)的 transform 只存在于
    trial 代码里,harness 无法重放——runner 传来的基础 splits 帧没有
    加宽列。v14.1 曾以 0 填充补列,但**常数列的置换 delta 结构性为 0**
    (打乱全零列不改变任何分数),全部加宽族必然被判 dead 而 coverage
    报 1.0,伪影不可见(实测 gen-02:seq 族判死,真帧重算 delta=+0.0049)。
    治本 = trial 在训练侧落盘真实评估帧,本通道优先消费它。
    行序必须与 validation split 严格一致(验证帧 = split 的行序变换,
    无 join);文件缺失/行数不符时返回 None(调用方回退并如实降级)。
    """
    p = Path(trial_dir) / "model" / "validation_input.parquet"
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 validation_input.parquet 失败 %s: %s",
                       trial_dir, exc)
        return None


def _resolve_generic_cap(trial_dir: Path, max_rows: int | None) -> int | None:
    """通用置换路径的行数上限:显式参数 > run 配置
    evaluator.group_importance_sample_rows(0/缺失 = 不限)。"""
    if max_rows is not None:
        return int(max_rows) if int(max_rows) > 0 else None
    try:
        from .config import Config

        run_dir = Path(trial_dir).resolve().parent.parent
        cfg = Config.load(run_config=run_dir / "modelevo-config.yaml")
        v = int(cfg.get("evaluator.group_importance_sample_rows", 0) or 0)
        return v if v > 0 else None
    except Exception:  # noqa: BLE001  配置不可读 → 不限
        return None


def run_group_importance(trial_dir: Path, X_valid: pd.DataFrame,
                         y: np.ndarray, metric: str, *,
                         treatment: np.ndarray | None = None,
                         n_repeat: int = 5, seed: int = 20260814,
                         families: dict[str, list[str]] | None = None,
                         threads: int | None = None,
                         row_positions: np.ndarray | None = None,
                         n_full_rows: int | None = None,
                         max_rows: int | None = None) -> \
        dict | None:
    """端到端:加载 trial 的模型 → 分组置换 → 落盘 CSV + JSON。

    供 runner 评估后调用(超级 trial 裁决)与 adhoc 手工复算。

    v12(P1-3)修复:
    - n_repeat 默认 1→5:单次置换的 delta_std 恒为 0,裁决随机性
      不可见;≥5 次才有 std 可言;
    - 特征列对齐:runner 传入的 X_valid 是 one-hot **前**的原始帧,
      而 feature_columns.json 记 one-hot 后的列——旧行为"缺列整段放弃"
      让 28 个 trial 只有 2 个产出 group_importance.csv(其中 1 个还是
      null);改为按可用列交集裁决并显式记录覆盖率;
    - uplift 的 score_fn 用 p1−p0 双次打分(模型特征含 __treatment__
      时;T-learner 双模型工件不支持单 predict 口径,自动跳过)。

    v14:打分前按训练同口径做 dtype 治理(见 _coerce_numeric)——
    名字交集过检但 dtype 是 object 时,LightGBM 在**基准打分**就崩,
    且异常一路传到 runner 的宽 except 被吞成一行 WARNING。

    v14.1(已于 v15.2 撤销):加宽特征 trial 的模型列含训练期新增列
    (seq_/inter_ 等),而 runner 传的 X_valid 是基础 splits 帧——曾以
    0 填充补列。该修复对**基准打分**成立,对**置换重要性**不成立:
    常数列打乱不掉分 → delta≡0 → 加宽族必然 dead(伪影);且 coverage
    在填充后计算报 1.0,伪影在产物里不可见。实测回写(gen-02 t4/t5):
    8 个加宽族全部判 dead,真帧重算 seq_ delta=+0.0049(alive),
    方向生死裁决被系统性污染。v15.2 起改为:
    - 优先消费 trial 自存的真实评估帧(model/validation_input.parquet,
      模板 dump_eval_frame 落盘;见 _load_widened_eval_frame);
    - 无该帧时,缺失列不再静默补 0:全部列缺失的族判 unverifiable
      (产出"无证据"而非"伪死"),部分缺失的族照算但按族标注
      zero_fill_share,delta 方向解读必须打折;
    - coverage 一律按填充前的真实交集计算,填充列数单列
      filled_constant_columns,不再虚报 1.0。

    v2.16.2(run4 regression D-003):runner 在 validation 上分层抽样
    (evaluator.group_importance_sample_rows)后,X_valid / y 是子样本,而
    trial 自存帧是全量 validation(行序 = split 行序)。旧实现比行数不等即
    弃用自存帧、回退基础帧(加宽列 0 填充)→ 加宽族恒 unverifiable。现在
    调用方传入抽样的行位置 `row_positions`(相对全量 validation,升序)与
    全量行数 `n_full_rows`:自存帧行数 == n_full_rows 时按**同一行位置**取
    子集(与 y/treatment 逐行对齐);行数仍对不上才回退并降级。

    v2.17.0(#14 复合容器):
    - 工件引用 trial `__main__` 定义的容器类(hurdle HurdleXGB 等)时按
      trial 代码模块重建后加载(旧实现加载即失败 → GI 空跑);
    - 无原生 GBM 快路径的模型走通用路径(按族置换 → 容器 predict → 主
      指标,见 _GenericPredict),评估行数受 max_rows 约束(缺省读 run 配置
      evaluator.group_importance_sample_rows;runner 已抽样时通常不再触发),
      在 row_positions 子集之上**再**取固定 seed 的升序子集,仍与 y 逐行对齐;
    - hurdle 容器(p × max(m, 0))额外输出按子模型的置换贡献
      delta_clf_mean / delta_reg_mean。
    """
    pack = _load_model_predict(trial_dir)
    if pack is None:
        return None
    model, cols = pack
    eval_frame_source = "splits_base"
    eval_frame_subset = None
    widened = _load_widened_eval_frame(trial_dir)
    if widened is not None and len(widened) != len(X_valid) \
            and row_positions is not None and n_full_rows is not None \
            and len(widened) == int(n_full_rows):
        _pos = np.asarray(row_positions, dtype="int64")
        if len(_pos) == len(X_valid) and (len(_pos) == 0 or (
                int(_pos.min()) >= 0 and int(_pos.max()) < len(widened))):
            widened = widened.iloc[_pos].reset_index(drop=True)
            eval_frame_subset = {"rows": int(len(_pos)),
                                 "of_rows": int(n_full_rows),
                                 "by": "row_positions"}
            logger.info(
                "trial 自存评估帧按抽样行位置取子集 %d/%d 行(与 y 逐行对齐)",
                len(_pos), int(n_full_rows))
    if widened is not None:
        if len(widened) == len(X_valid):
            X_valid = widened
            eval_frame_source = "trial_saved_validation_input"
            logger.info(
                "分组置换使用 trial 自存评估帧(model/"
                "validation_input.parquet,%d 行 %d 列)",
                len(widened), widened.shape[1])
        else:
            # 行数不符 = 帧与 split 行序无对应(quick 轮/smoke 误存等),
            # 置换打分行集与 y 不一致 → 结果不可用。回退基础帧并降级。
            logger.warning(
                "trial 自存评估帧 %d 行 != validation %d 行,弃用,"
                "回退基础 splits 帧(加宽族将按 unverifiable 降级)",
                len(widened), len(X_valid))
    # v2.17.0:无 GBM 快路径(复合容器/任意 predict 对象)→ 通用路径的行数上限
    is_native_gbm = hasattr(model, "get_booster") or hasattr(model, "booster_")
    generic_cap = None
    if not is_native_gbm:
        generic_cap = _resolve_generic_cap(trial_dir, max_rows)
    if generic_cap and len(X_valid) > generic_cap:
        _rng = np.random.default_rng(seed + 17)
        _keep = np.sort(_rng.choice(len(X_valid), size=int(generic_cap),
                                    replace=False))
        X_valid = X_valid.iloc[_keep].reset_index(drop=True)
        y = np.asarray(y)[_keep]
        if treatment is not None:
            treatment = np.asarray(treatment)[_keep]
        eval_frame_subset = {
            "rows": int(len(_keep)),
            "of_rows": int((eval_frame_subset or {}).get("of_rows")
                           or n_full_rows or len(_keep)),
            "by": ("row_positions+generic_cap" if eval_frame_subset
                   else "generic_cap"),
            "cap": int(generic_cap)}
        logger.info("通用置换路径行数上限 %d(evaluator."
                    "group_importance_sample_rows)", int(generic_cap))
    # v15.2:coverage 一律按基础帧真实交集计;缺失列分两类——
    # 在自存帧/基础帧里都找不到的列才允许 0 填充(基准打分需要稠密帧),
    # 但**填充列不参与裁决语义**:全部被填充的族判 unverifiable,
    # 部分填充的族标注 zero_fill_share(delta 方向打折解读)。
    missing_from_frame = [c for c in cols if c not in X_valid.columns]
    filled = set(missing_from_frame)
    coverage = 1.0 - len(filled) / max(len(cols), 1)
    if missing_from_frame:
        X_valid = X_valid.copy()
        for c in missing_from_frame:
            X_valid[c] = 0.0
        logger.warning(
            "分组置换评估帧缺 %d 个模型列(如 %s),以 0 填充仅供基准"
            "打分;全缺失族将判 unverifiable(填充列置换 delta≡0,"
            "不构成方向生死证据)",
            len(filled), missing_from_frame[:3])
    avail = [c for c in cols if c in X_valid.columns]
    if not avail:
        logger.warning("分组置换跳过:validation 与模型特征列无交集"
                       "(模型 %d 列)", len(cols))
        return None
    # v14:按训练同口径把 avail 列数值化(只强转,不 one-hot)
    X_valid = _coerce_numeric(X_valid, avail)

    from .metrics import UPLIFT_METRICS

    is_uplift = metric in UPLIFT_METRICS and "__treatment__" in avail

    def score_fn(X: pd.DataFrame) -> np.ndarray:
        if is_uplift:
            Xt = X[avail].assign(
                __treatment__=np.ones(len(X), dtype="int8"))
            Xc = X[avail].assign(
                __treatment__=np.zeros(len(X), dtype="int8"))
            return (model.predict_proba(Xt[avail])[:, 1]
                    - model.predict_proba(Xc[avail])[:, 1])
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X[avail])[:, 1]
        return model.predict(X[avail])

    fam_map = families or infer_families(avail)
    # v15.2:按族标注零填充份额;全部列被填充的族移出裁决(unverifiable)
    fam_filled_share: dict[str, float] = {}
    unverifiable: list[dict] = []
    judged_map: dict[str, list[str]] = {}
    for fam, fcols in fam_map.items():
        n_filled = sum(1 for c in fcols if c in filled)
        share = n_filled / max(len(fcols), 1)
        fam_filled_share[fam] = round(share, 4)
        if n_filled and n_filled == len(fcols):
            unverifiable.append({
                "family": fam,
                "n_cols": len(fcols),
                "reason": "评估帧全列缺失(0 填充常数列,置换 delta≡0,"
                          "不构成生死证据);需 trial 落盘 "
                          "model/validation_input.parquet 后重评,或 adhoc "
                          "在真实加宽帧上复算",
            })
        else:
            judged_map[fam] = fcols
    import time as _time

    _t0 = _time.perf_counter()
    _parts = None
    if not is_native_gbm and not is_uplift:
        _hp = _hurdle_parts(model)
        if _hp is not None:
            _p_fn, _m_fn = _hp
            _parts = (lambda X: _p_fn(X[avail]), lambda X: _m_fn(X[avail]))
    df = group_permutation_importance(
        score_fn, X_valid, y, judged_map, metric,
        n_repeat=n_repeat, seed=seed, treatment=treatment,
        trial_dir=trial_dir, model=model, threads=threads,
        submodels=_parts)
    _gi_elapsed = round(_time.perf_counter() - _t0, 2)
    _predict_path = df.attrs.get("predict_path")
    _decomp = bool(df.attrs.get("submodel_decomposition"))
    if len(df) and fam_filled_share:
        df["zero_fill_share"] = df["family"].map(fam_filled_share)
    if not len(df) and not unverifiable:
        return None
    out_dir = Path(trial_dir) / "feature"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(df):
        df.to_csv(out_dir / "group_importance.csv", index=False)
    else:
        # 全部族 unverifiable:CSV 无可裁决行,但 JSON 摘要仍落盘
        # (产物必须显式声明"无证据",而不是让下游看到文件缺失)
        df = pd.DataFrame(columns=[
            "family", "n_cols", "delta_mean", "delta_std", "rel_delta"])
    alive = df.loc[df["delta_mean"] > 0, "family"].tolist()
    summary = {
        "metric": metric,
        "n_families": int(len(df)),
        "n_repeat": int(n_repeat),
        "gi_elapsed_sec": _gi_elapsed,
        "n_rows": int(len(X_valid)),
        "feature_coverage": round(float(coverage), 4),
        "filled_constant_columns": int(len(filled)),
        "eval_frame_source": eval_frame_source,
        "eval_frame_subset": eval_frame_subset,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "predict_path": _predict_path,
        "submodel_decomposition": (
            {"available": True, "form": "p*max(m,0)",
             "columns": ["delta_clf_mean", "delta_reg_mean"]}
            if _decomp else {"available": False}),
        "families_alive": alive,
        "families_dead": df.loc[
            (df["delta_mean"] <= 0), "family"].tolist(),
        "families_unverifiable": [u["family"] for u in unverifiable],
        "unverifiable_detail": unverifiable,
        "top": df.head(10).to_dict(orient="records"),
        "note": "delta_mean>0=族承载真实增量(打乱掉分);裁决优先于 gain;"
                "zero_fill_share>0 的族 delta 被结构性压向 0,方向解读"
                "须打折;unverifiable=无证据(非死),不得据此砍方向",
    }
    dump_json(out_dir / "group_importance.json", summary)
    return summary
