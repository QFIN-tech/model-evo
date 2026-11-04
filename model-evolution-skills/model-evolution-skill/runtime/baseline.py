"""Baseline:统一起跑线(§8)。

目标不是聪明,而是稳定、便宜、可复现、低决策成本。
默认 LightGBM quick baseline,资源允许时并行补 XGBoost;
不做大规模 HPO。作为第 0 代标准 Trial 产出(§8.2)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .code_audit import check_generated_code
from .config import Config
from .logging_utils import get_logger, dump_json
from .safe_dtype import safe_to_numeric
from .spec import TaskSpec
from .templates import LEAK_SCAN_SOURCE, _fmt_escape

logger = get_logger(__name__)

BASELINE_TEMPLATE = '''\\
"""ModelEvo baseline trial(自动生成,可按需微调参数)。

约定:
- 读取 sample-profile/splits/ 下 train.parquet 与 validation.parquet;
- 训练 {algorithm} 默认参数模型(GBM 配 eval_set + early stopping);
- 早停专用 es_valid 从 train 尾部按时间切 10%(v12/P0-4):validation
  不再承担早停职责——它同时是主指标/delta/noise gate/榜单排名/最终
  选型的裁决面,早停在上面会引入与搜索方向相关的系统性乐观偏差
  (实测全 run 最大两笔增益恰是早停自由度最大的配置);
- 在 validation 上输出 metrics.json 与 validation_score.parquet(与父代对比用);
- 输出 feature/feature_importance.csv(feature, importance)供反思与经验沉淀;
- learning curve 落盘 learning_curve.csv(iteration, valid_metric),
  best_iteration / n_estimators_used 写入 metrics.json(供反思区分欠拟合/过拟合)。
"""
import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import yaml

#: 数字字面量(与 runtime/safe_dtype 同口径)
_NUM_STR_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

RUN_DIR = pathlib.Path(__file__).resolve().parents[3]
SPLITS = RUN_DIR / "sample-profile" / "splits"
TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent
# v2.13.5:skill 根入 sys.path —— baseline 血缘(`--from-parent` 派生的手写
# trial)可直接 `from runtime.feature_cache import feature_block`(跨 trial
# 特征块缓存 + 并行解析);hpo 嫁接段的 `from runtime.hpo import` 同样依赖
# 它(此前每次嫁接都要注入/手补 bootstrap)。
SKILL_ROOT = ""  # write_baseline_code 渲染后注入(未注入时不动 sys.path)
if SKILL_ROOT and SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)

TARGET = {target!r}
MODEL_TYPE = {model_type!r}
TREATMENT = {treatment!r}
_ALGORITHM = {algorithm!r}
SEED = {seed}
N_ESTIMATORS_CAP = {n_estimators_cap}   # 早停前的轮数上限
EARLY_STOP_ROUNDS = {early_stopping_rounds}
#: 早停验证集占 train 的比例(时间尾部;仅 train_tail 回退模式使用)
ES_FRACTION = {es_fraction}
#: v2.16.0:早停面默认值(validation | train_tail;write_baseline_code 按
#: run 配置 training.early_stop_source 改写,运行期 MODELEVO_ES_SOURCE 优先)
ES_SOURCE = "validation"
#: GPU 加速的最小训练行数(v17.4):小样本(smoke 以下/small-sample)不上卡
_GPU_MIN_ROWS = 2000


def _es_source() -> str:
    """v2.16.0:validation = outer validation 早停、100% train 只训一次;
    train_tail = 旧口径(train 尾部 ES_FRACTION 早停,validation 只评分)。"""
    _v = str(os.environ.get("MODELEVO_ES_SOURCE") or ES_SOURCE or "validation")
    return "train_tail" if _v.strip().lower() == "train_tail" else "validation"


def _gpu_allocated(n_rows: int) -> bool:
    # 唯一 GPU 意图信号 = scheduler/runner 注入的 MODELEVO_GPU
    # (CPU 降配重试时被置 "0",不依赖 CUDA_VISIBLE_DEVICES 空串语义)。
    return os.environ.get("MODELEVO_GPU") == "1" and (n_rows or 0) >= _GPU_MIN_ROWS


def _xgb_cuda_built() -> bool:
    # xgboost 有可靠编译探测;经 importlib 动态加载,避免静态 import
    # xgboost 出现在非 xgboost 算法 trial 里被 lint 规则 2 判成多 GBM 库
    try:
        import importlib as _il
        _m = _il.import_module("xgboost")
        return bool((_m.build_info() or {{}}).get("USE_CUDA"))
    except Exception:
        return False


# [FIX-2026-09-10] lightgbm wheel CUDA 编译探测(模块级缓存):
# 非 CUDA 编译轮上 device=cuda 的 fit 期 Fatal 曾致进程死、CPU
# 回退没接住——注入前先 64 行 x 4 特征小样本试 fit,一切异常按
# False 处理。xgboost 走 _xgb_cuda_built()。
_LGBM_CUDA = None


def _lgbm_cuda_ok() -> bool:
    global _LGBM_CUDA
    if _LGBM_CUDA is not None:
        return _LGBM_CUDA
    try:
        import numpy as _np_probe
        import lightgbm as _lgb_probe
        _rs = _np_probe.random.RandomState(0)
        _Xp = _rs.rand(64, 4).astype("float32")
        _yp = (_rs.rand(64) > 0.5).astype(int)
        # v2.12.2.9:lightgbm 的 Fatal 由 C++ 层直写 fd 2,Python 的
        # redirect_stderr 拦不住(实测 buffer 为空)。必须 dup2 到
        # /dev/null,否则每次 trial 的 stderr.log 里都留一条
        # "[LightGBM] [Fatal] CUDA Tree Learner was not enabled in this
        # build." —— 看起来像 trial 崩溃,实测一次排查即被误导。
        _saved_fd = os.dup(2)
        _null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_null_fd, 2)
        os.close(_null_fd)
        try:
            _lgb_probe.LGBMClassifier(device="cuda", n_estimators=1,
                                      verbosity=-1,
                                      random_state=0).fit(_Xp, _yp)
            _LGBM_CUDA = True
        finally:
            os.dup2(_saved_fd, 2)
            os.close(_saved_fd)
    except Exception:
        _LGBM_CUDA = False
        print("lightgbm wheel 非 CUDA 编译,回退 CPU"
              "(PyPI wheel 默认不含 CUDA;这是预期降级,非故障)",
              flush=True)
    return _LGBM_CUDA


def _lgbm_fit_safe(model, *a, **kw):
    # lightgbm 无公开 GPU 编译探测;device=cuda 在非 GPU 编译轮上于
    # booster 创建期抛 LightGBMError → 去 device 回退 CPU 重训一次
    # (失败发生在建树前,CPU 重训代价≈0,保证"有卡尝试、无卡回退")。
    try:
        return model.fit(*a, **kw)
    except Exception:
        try:
            model.set_params(device="cpu")
            return model.fit(*a, **kw)
        except Exception:
            pass
        raise


def safe_to_numeric(s: pd.Series, *, max_rows=None,
                    min_parse_ratio: float = 0.95):
    """把 s 数值化;不是数值列时返回 None。

    v2.12.2.9 修复:本函数原被 dtype 治理段调用却既未定义也未 import,
    导致 baseline 在 dtype_governance 段 NameError。语义与
    runtime/safe_dtype.safe_to_numeric 一致:
    - 已是数值 dtype 原样返回;
    - 非数值字面量占比超过 (1-min_parse_ratio) 即判非数值列(返回 None),
      避免把十六进制样本键部分解析成数字后当特征;
    - 逐元素解析用显式循环,不用 pandas 递归,长 object 列不会爆栈。
    """
    if pd.api.types.is_numeric_dtype(s):
        return s
    if s.dtype != object and not pd.api.types.is_string_dtype(s):
        return None
    nn = s.dropna()
    if nn.empty:
        return None
    try:
        as_str = nn.astype(str)
    except Exception:  # noqa: BLE001
        return None
    try:
        all_numeric = bool(as_str.str.match(_NUM_STR_RE).all())
    except Exception:  # noqa: BLE001
        return None
    if all_numeric:
        return pd.to_numeric(s, errors="coerce")
    if max_rows is not None and len(s) > max_rows:
        return None
    values = nn.to_numpy(dtype=object)
    parsed = np.full(len(values), np.nan, dtype="float64")
    for i, v in enumerate(values):
        if isinstance(v, (int, float, np.integer, np.floating)):
            parsed[i] = float(v)
            continue
        try:
            parsed[i] = float(v)
        except (TypeError, ValueError):
            continue
    if np.isfinite(parsed).sum() < min_parse_ratio * len(values):
        return None
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[nn.index] = parsed
    return out


def load_config() -> dict:
    # 读取 ../config/trial.yaml(可复现配置:features/seed/超参)
    cfg_path = pathlib.Path(__file__).resolve().parent.parent / "config" / "trial.yaml"
    if not cfg_path.exists():
        return dict()
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or dict()


# TaskSpec 泄漏分类(harness 注入;v13 毒性轴)。banned 的主防御在
# 数据面物理隔离(四平面),此集合做 fit 前终检的名字防御。
BANNED: frozenset[str] = frozenset()
#: 非特征列:目标/时间/ID + TaskSpec 派生排除(v12/P0-8:不再硬编码
#: pday/user_no,由 harness 按 spec 注入 _SPEC_NONFEATURE)
_SPEC_NONFEATURE: set[str] = set()


def exclude_columns(df: pd.DataFrame, cfg=None) -> list[str]:
    # 特征面 = 全列 − 目标/时间/ID − banned/reparse − features.exclude
    # (config 为准)− features.columns 白名单过滤。
    # 特征面单源存 config/trial.yaml,代码运行期读取,不内置 FEATURES
    # 常量(避免双写漂移)。v12/P0-8:非特征列由 _SPEC_NONFEATURE
    # 注入(spec 的 time/id/treatment 列),不再硬编码任务特定列名。
    cfg = cfg if cfg is not None else load_config()
    feats_cfg = cfg.get("features") or dict()
    drop = {{TARGET}} | _SPEC_NONFEATURE | set(BANNED)
    if isinstance(feats_cfg, dict):
        drop |= set(feats_cfg.get("exclude") or [])
    cols = [c for c in df.columns if c not in drop]
    whitelist = (feats_cfg.get("columns")
                 if isinstance(feats_cfg, dict) else None)
    if whitelist:
        cols = [c for c in whitelist if c in cols]
    return cols


def load_labels(split: str) -> pd.DataFrame:
    """v21 标签面访问器:目标 + treatment + 对齐键(四平面隔离)。"""
    path = SPLITS / "labels" / f"{{split}}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"标签面缺失: {{path}}(先跑 profile)")
    return pd.read_parquet(path)


def assert_banned_clean(X: pd.DataFrame) -> None:
    """v13 fit 前终检:特征面不得含 banned 列(零成本,防绕行)。"""
    hit = BANNED & set(X.columns)
    if hit:
        raise ValueError(f"泄漏列渗入特征面: {{sorted(hit)}}")


''' + _fmt_escape(LEAK_SCAN_SOURCE) + '''


def dump_feature_importance(model, columns) -> None:
    """尽力输出 gain 重要性;模型不提供时静默跳过。

    落盘到 TRIAL_DIR/feature/feature_importance.csv。
    """
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "booster_"):
        try:
            imp = model.booster_.feature_importance(importance_type="gain")
        except Exception:
            imp = None
    if imp is None:
        return
    out_dir = TRIAL_DIR / "feature"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"feature": list(columns),
                   "importance": list(imp)}}).sort_values(
        "importance", ascending=False).to_csv(
        out_dir / "feature_importance.csv", index=False)


def _load_treatment_for_metrics():
    # uplift sanity 量:从标签面重读 treatment 列(v21 四平面)
    return load_labels("validation")[TREATMENT].values


def dump_learning_curve(model) -> dict:
    """落盘 learning_curve.csv(iteration, valid_metric),尽力而为。

    LGBM evals_result_ → XGB evals_result() → torch history_(每 epoch loss);
    都不可得(旧式代码/自定义模型)时返回空 dict 不落盘。
    """
    curve = None
    ev = getattr(model, "evals_result_", None)
    if ev is None and hasattr(model, "evals_result"):
        try:
            ev = model.evals_result()
        except Exception:
            ev = None
    if ev:
        # 取第一个 eval 集的任一指标序列(LGBM/XGB 均为 dict of dict)
        first = next(iter(ev.values()))
        key = next(iter(first.keys()))
        curve = list(first[key])
    elif isinstance(getattr(model, "history_", None), list):
        curve = list(model.history_)  # torch _Wrapper:每 epoch 平均训练 loss
    if not curve:
        return {{"learning_curve_rows": 0}}
    pd.DataFrame({{"iteration": np.arange(1, len(curve) + 1),
                   "valid_metric": curve}}).to_csv(
        TRIAL_DIR / "learning_curve.csv", index=False)
    return {{"learning_curve_rows": len(curve)}}


def dump_model(model, columns) -> None:
    """持久化模型工件到 TRIAL_DIR/model/(尽力而为,不阻塞 trial):
    model.joblib + 原生格式(lightgbm .txt / xgboost .json /
    catboost .bin / torch .pt)+ feature_columns.json。
    quick 轮(MODELEVO_QUICK_RATIO 子样本)跳过,防止小样本模型
    覆盖全量模型。"""
    import os
    if os.environ.get("MODELEVO_QUICK_RATIO", "0") not in ("", "0"):
        return
    if os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name:
        print("[guard] dump_model 拒写: 非 runner 执行上下文"
              "(MODELEVO_TRIAL_EXEC 不匹配),trial 工件只读", flush=True)
        return
    try:
        import joblib
        mdir = TRIAL_DIR / "model"
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, mdir / "model.joblib")
        mod = type(model).__module__
        if hasattr(model, "booster_"):
            model.booster_.save_model(str(mdir / "model.txt"))
        elif mod.startswith("xgboost") and hasattr(model, "save_model"):
            model.save_model(str(mdir / "model.json"))
        elif mod.startswith("catboost") and hasattr(model, "save_model"):
            model.save_model(str(mdir / "model.bin"))
        elif hasattr(model, "model") and hasattr(model.model, "state_dict"):
            import torch
            torch.save(model.model.state_dict(), mdir / "model.pt")
        with open(mdir / "feature_columns.json", "w",
                  encoding="utf-8") as fh:
            json.dump(list(columns), fh)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] dump_model skipped: {{_e}}", flush=True)


def dump_eval_frame(X_va) -> None:
    """落盘模型输入级 validation 帧(v15.2;baseline 无 transform,
    基础帧即模型输入——但统一落盘保持与加宽 trial 同契约,分组置换
    优先消费它,dtype 口径与训练打分完全一致)。smoke/quick 轮跳过。"""
    import os
    if os.environ.get("MODELEVO_QUICK_RATIO", "0") not in ("", "0"):
        return
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return
    if os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name:
        print("[guard] dump_eval_frame 拒写: 非 runner 执行上下文"
              "(MODELEVO_TRIAL_EXEC 不匹配),trial 工件只读", flush=True)
        return
    try:
        mdir = TRIAL_DIR / "model"
        mdir.mkdir(parents=True, exist_ok=True)
        X_va.to_parquet(mdir / "validation_input.parquet", index=False)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] dump_eval_frame skipped: {{_e}}", flush=True)


def dump_shadow_score(score) -> None:
    """shadow_test 分数落盘(v18,与 trial 模板同契约)。baseline 的
    shadow 分数在 cmd_baseline 侧另有 backfill 兜底,此处直接产出
    省一次补打分;smoke/quick 轮跳过。"""
    import os
    if os.environ.get("MODELEVO_QUICK_RATIO", "0") not in ("", "0"):
        return
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return
    if os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name:
        print("[guard] dump_shadow_score 拒写: 非 runner 执行上下文"
              "(MODELEVO_TRIAL_EXEC 不匹配),trial 工件只读", flush=True)
        return
    try:
        _frame = pd.DataFrame({{"score": score}})
        # v2.19.1:落盘即带样本键(与 trial 模板同口径;缺键的 shadow 工件
        # 无法被 confirm 父代对照与 benchmark 对齐评分器消费)
        _lab = SPLITS / "labels" / "shadow_test.parquet"
        if _lab.exists() and len(_frame) > 0:
            try:
                _keys = pd.read_parquet(
                    _lab, columns=["__mvo_sample_key"])["__mvo_sample_key"]
                if len(_keys) == len(_frame):
                    _frame["__mvo_sample_key"] = _keys.to_numpy()
                else:
                    print(f"[warn] dump_shadow_score 不加样本键: 分数 {{len(_frame)}} 行"
                          f" != 标签面 {{len(_keys)}} 行", flush=True)
            except Exception as _ke:  # noqa: BLE001
                print(f"[warn] dump_shadow_score 读样本键失败: {{_ke}}", flush=True)
        _frame.to_parquet(TRIAL_DIR / "shadow_test_score.parquet", index=False)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] dump_shadow_score skipped: {{_e}}", flush=True)


def best_iteration_of(model):
    """早停轮次统一口径(+1 转为 1-based);不可得返回 None。"""
    bi = getattr(model, "best_iteration_", None)
    if bi is None and hasattr(model, "best_iteration"):
        try:
            bi = int(model.best_iteration)
        except Exception:
            bi = None
    if bi is None and isinstance(getattr(model, "history_", None), list):
        bi = max(len(model.history_) - 1, 0)  # torch:patience 断点近似
    return (int(bi) + 1) if bi is not None else None


def _es_eval_cap(X_es, y_es):
    """v2.17.3:早停集降采样(只影响"何时停",不影响任何落盘分数)。

    行数上限由 training.early_stop_eval_rows → MODELEVO_ES_EVAL_ROWS 下发,
    0/缺省 = 不降采样(旧行为)。取**标签分层头**(不是时间前缀:天内标签
    聚集时前缀近乎单类)。失败时原样返回,不阻塞训练。
    """
    try:
        _cap = int(os.environ.get("MODELEVO_ES_EVAL_ROWS") or 0)
    except (TypeError, ValueError):
        return X_es, y_es
    if not _cap or len(X_es) <= _cap:
        return X_es, y_es
    try:
        _y = pd.Series(np.asarray(y_es))
        _frac = float(_cap) / float(len(_y))
        _parts = []
        for _v, _grp in _y.groupby(_y):
            _k = max(1, int(round(len(_grp) * _frac)))
            _parts.append(_grp.index[:_k])
        _idx = np.sort(np.concatenate(_parts)).astype(np.int64)
        if len(_idx) < 1000:
            return X_es, y_es
        _xs = X_es.iloc[_idx] if hasattr(X_es, "iloc") else X_es[_idx]
        _ys = y_es.iloc[_idx] if hasattr(y_es, "iloc") else y_es[_idx]
        print(f"[step] es_eval 降采样 {{len(X_es)}} → {{len(_idx)}} 行"
              "(分层;只影响早停位置)", flush=True)
        return _xs, _ys
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] es_eval 降采样跳过: {{type(_e).__name__}}: {{_e}}",
              flush=True)
        return X_es, y_es


def prepare_shadow_features(shadow_df, cat_cols, dropped_cols,
                            feature_columns):
    """shadow 帧按训练同口径的 dtype 治理(one-hot/数值化/列对齐)。

    cat_cols/dropped_cols 来自训练段的治理记录,保证 shadow 特征面
    与 model.feature_columns 逐列一致(v12 backfill 同款口径,前置到
    训练内完成)。"""
    X = shadow_df.copy()
    X = X.drop(columns=[c for c in dropped_cols if c in X.columns])
    _cats = [c for c in cat_cols if c in X.columns]
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            continue
        # v2.12.2.8:safe_to_numeric 替下 pd.to_numeric——后者在含非数字
        # 值的长 object 列上段错误,且会把十六进制键静默解析成数字。
        conv = safe_to_numeric(X[c])
        if conv is not None:
            X[c] = conv.astype("float32")
    if _cats:
        X = pd.get_dummies(X, columns=_cats)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X.fillna(0).astype("float32"), None


def _append_feature_blocks(X, split):
    """v2.13.5:trial 定义了 trial_feature_blocks(split) 时,把特征块右接到
    治理后的数值帧上(行序 = split 行序;quick 子样本按行位置对齐);
    未定义 → 原样返回(baseline 本身不定义,行为不变)。

    契约见 runtime/feature_cache.py:无训练期拟合状态的行内变换(seq 串
    解析等)在这里接入,即自动覆盖训练/shadow 重放/finalize 交付重放,
    同一内核跨 trial 只算一次。"""
    fb = globals().get("trial_feature_blocks")
    if not callable(fb):
        return X
    from runtime.feature_cache import align_rows
    W = fb(split)
    dup = [c for c in W.columns if c in X.columns]
    if dup:
        raise ValueError("特征块列与基础面重名(须带解析前缀): {{}}".format(
            dup[:5]))
    W = align_rows(W, X).astype("float32")
    step("feature blocks appended({{}}): +{{}} cols".format(
        split, W.shape[1]))
    return pd.concat([X, W], axis=1)


def align_predict_device(model) -> None:
    """打分前把 xgboost booster 切回 CPU(v2.12.2.9)。

    背景:xgboost 在 GPU 上训练(device=cuda),但预测路径走普通
    `model.predict_proba(X)`,X 在 CPU 内存上 → xgboost 每次预测都触发
    "Falling back to prediction using DMatrix due to mismatched devices"
    并把数据在设备间搬运。实测每个 trial 的 stderr.log 都有这条警告,
    validation + shadow 两次打分各付一次代价。

    等价性:实测 xgboost 2.1.4(USE_CUDA=True)下,把 booster 的 device
    改回 cpu 前后 predict_proba 的 np.array_equal 为 False,但最大绝对差
    仅 5.96e-08 —— 这是 CPU/GPU 浮点累加顺序差异,不是语义差异。AUC 由
    全量样本的秩统计得到,1e-8 量级的扰动不改变任何一对样本的相对次序。
    且该差异**本来就存在**:现状就是在 CPU 上完成预测的(DMatrix 回退),
    本函数只是把这次回退从"隐式警告"变成"显式声明",不改变落盘分数口径。

    只对 xgboost 生效(lightgbm 的 device 由 _lgbm_fit_safe 在 fit 期
    处理;torch 走 score_validation 自带 device 管理)。

    设备读法:`booster.attributes()` 实测**恒为空 dict**(xgboost 2.1.4),
    真实设备在 `save_config()` 的 `learner.generic_param.device`
    (如 "cuda:0" / "cpu")。用错读法会让本函数静默失效。
    """
    try:
        if not type(model).__module__.startswith("xgboost"):
            return
        booster = model.get_booster() if hasattr(model, "get_booster") else None
        if booster is None:
            return
        cfg = json.loads(booster.save_config())
        dev = str(((cfg.get("learner") or {{}}).get("generic_param") or {{}})
                  .get("device") or "")
        if "cuda" in dev.lower() or "gpu" in dev.lower():
            booster.set_param({{"device": "cpu"}})
            print("[step] xgboost booster 已切回 CPU 打分"
                  "(避免 DMatrix 设备回退;分数口径不变)", flush=True)
    except Exception as _e:  # noqa: BLE001 — 切设备是优化,失败不阻塞打分
        print(f"[warn] align_predict_device skipped: {{_e}}", flush=True)


import time as _time

_T0 = _time.time()


def step(msg: str) -> None:
    """分阶段进度日志(flush 到 stdout,harness 心跳互补)。"""
    print(f"[step] {{msg}} ({{_time.time() - _T0:.1f}}s)", flush=True)


def build_estimator(n_jobs: int, n_rows: int = 0):
{estimator_code}
    raise ValueError(f"unknown algorithm")


def _fit_seed_variant(n_jobs, X_fit, y_fit, eval_set):
    """v2.14.0:multi-seed 变体轮拟合(与主轮同估计器、同早停口径)。

    build_estimator 在调用时读全局 SEED;调用方临时改写 SEED 后调用。
    """
    model = build_estimator(n_jobs, len(X_fit))
    if _ALGORITHM == "lightgbm":
        import lightgbm as _lgb_fit
        _lgbm_fit_safe(model, X_fit, y_fit, eval_set=eval_set,
                       callbacks=[_lgb_fit.early_stopping(
                           EARLY_STOP_ROUNDS, verbose=False)])
    else:
        model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)
    return model


def main() -> None:
    cfg = load_config()
    # 特征面运行期解析(config 单源:features.exclude/columns)
    # 只读需要的列,避免整表(含无关宽列)常驻内存:先读表头解析特征面
    _head = pd.read_parquet(SPLITS / "train.parquet").columns
    feats = exclude_columns(pd.DataFrame(columns=_head), cfg)
    step(f"load_data start: cols={{len(feats)}}")
    # v13:特征面/标签面分产物读取(特征面无目标列;标签从 labels/ 取)
    train = pd.read_parquet(SPLITS / "train.parquet", columns=feats)
    valid = pd.read_parquet(SPLITS / "validation.parquet", columns=feats)
    labels_tr = load_labels("train")
    labels_va = load_labels("validation")
    step(f"load_data done: train={{len(train)}} valid={{len(valid)}}")
    # successive halving quick 轮:harness 经 MODELEVO_QUICK_RATIO 注入
    # 子样本比例。v17.1:前缀改为标签分层头(模板同款 _stratified_head
    # 逻辑)——天内 label 聚集的落盘序上时间前缀近乎单类,quick 轮
    # 排序语义失真;不识别时等价全量。
    _quick = float(os.environ.get("MODELEVO_QUICK_RATIO", "0") or 0)
    if 0 < _quick < 1:
        _y = labels_tr[TARGET]
        _vc = _y.value_counts()
        if len(_vc) == 2 and (_vc > 0).all():
            _k = int(len(train) * _quick)
            _parts = [_y.index[_y == v][:max(1, int(round(_k * c / len(_y))))]
                      for v, c in _vc.items()]
            _sel = _parts[0].append(_parts[1]).sort_values()[:_k]
        else:
            _sel = _y.index[: int(len(train) * _quick)]
        train = train.loc[_sel]
        labels_tr = labels_tr.loc[_sel]
        step(f"quick_ratio={{_quick}}: train subsampled to {{len(train)}}")
    X_tr, y_tr = train[feats], labels_tr[TARGET]
    X_va, y_va = valid[feats], labels_va[TARGET]
    if MODEL_TYPE == "uplift":
        # S-learner:treatment 作为特征进入模型(单一 fit,不违反铁律 12),
        # 打分时同一批样本分别置 w=1/w=0,p1-p0 即 CATE 估计
        _w_tr = labels_tr[TREATMENT].values
        X_tr = X_tr.assign(__treatment__=_w_tr)
        X_va_t = X_va.assign(__treatment__=np.ones(len(X_va), dtype="int8"))
        X_va_c = X_va.assign(__treatment__=np.zeros(len(X_va), dtype="int8"))
    del train, valid
    # v13:fit 前终检(banned 列物理隔离之上的第二道断言)
    assert_banned_clean(X_tr)
    step("dtype_governance start")
    # dtype 治理(历史教训:decimal/字符串金额列读入为 object,若直接
    # get_dummies 会按唯一值膨胀成百万列,OOM。必须先区分:
    # 可数值化的 object 列 -> float32;低基数类别列 -> category)
    cat_cols = []
    dropped_cols = []
    for c in X_tr.columns:
        if pd.api.types.is_numeric_dtype(X_tr[c]):
            continue
        # v2.12.2.8:safe_to_numeric 替下 pd.to_numeric。旧实现有两个
        # 实测缺陷:(1) 含非数字值的长 object 列会段错误(见
        # safe_dtype docstring);(2) 十六进制键列被部分解析成数字后
        # 通过 95% 判据,ID 列被当成数值特征进入训练面。
        conv = safe_to_numeric(X_tr[c])
        if conv is not None:
            X_tr[c] = conv.astype("float32")
            va_conv = safe_to_numeric(X_va[c])
            X_va[c] = (va_conv.astype("float32") if va_conv is not None
                       else np.nan)
        elif X_tr[c].nunique() <= 100:
            cat_cols.append(c)
        else:
            # v10.3:高基数不可数值化列(如行为序列字符串)防御性剔除
            # 并显式告警,不再 raise 把 baseline 卡死——剔除清单进
            # metrics.json,特征探索阶段按需深挖(避免静默吞掉数据事实)
            dropped_cols.append(c)
    if dropped_cols:
        print("[warn] dtype 防护: {{}} 列无法数值化且基数>100,"
              "防御性剔除(特征探索阶段可深挖): {{}}".format(
                  len(dropped_cols), dropped_cols[:10]), flush=True)
        X_tr = X_tr.drop(columns=dropped_cols)
        X_va = X_va.drop(columns=dropped_cols)
        if MODEL_TYPE == "uplift":
            X_va_t = X_va_t.drop(columns=dropped_cols)
            X_va_c = X_va_c.drop(columns=dropped_cols)
    if cat_cols:
        X_tr = pd.get_dummies(X_tr, columns=cat_cols)
        X_va = pd.get_dummies(X_va, columns=cat_cols)
        X_va = X_va.reindex(columns=X_tr.columns, fill_value=0).astype(
            X_tr.dtypes.to_dict())
        if MODEL_TYPE == "uplift":
            # S-learner 的两个打分副本走同一 one-hot 口径
            X_va_t = pd.get_dummies(X_va_t, columns=cat_cols)
            X_va_c = pd.get_dummies(X_va_c, columns=cat_cols)
            X_va_t = X_va_t.reindex(columns=X_tr.columns,
                                    fill_value=0).astype(X_tr.dtypes.to_dict())
            X_va_c = X_va_c.reindex(columns=X_tr.columns,
                                    fill_value=0).astype(X_tr.dtypes.to_dict())
    # NaN 统一填 0(GBM 可原生吃 NaN,但 torch/线性模型必须稠密)
    X_tr = X_tr.fillna(0).astype("float32")
    X_va = X_va.fillna(0).astype("float32")
    if MODEL_TYPE == "uplift":
        X_va_t = X_va_t.fillna(0).astype("float32")
        X_va_c = X_va_c.fillna(0).astype("float32")
    # v2.13.5:特征块钩子(uplift 的双打分副本不在支持范围内)
    _base_cols = list(X_tr.columns)
    if MODEL_TYPE != "uplift":
        X_tr = _append_feature_blocks(X_tr, "train")
        X_va = _append_feature_blocks(X_va, "validation")
    _n_block_cols = X_tr.shape[1] - len(_base_cols)
    # v2.16.0:默认在 outer validation 上早停、100% train 只训一次
    # (training.early_stop_source=validation)。train_tail 回退 = P0-4 口径:
    # 早停专用 es_valid 从 train 尾部按时间切 ES_FRACTION,validation 只评分。
    # uplift 维持 train 尾部(S-learner 特征面含 __treatment__)。
    # 切分在 dtype 治理之后(X_es 与 X_tr 同特征面)。
    _es_cut = int(len(X_tr) * (1.0 - ES_FRACTION))
    X_fit, X_es = X_tr.iloc[:_es_cut], X_tr.iloc[_es_cut:]
    y_fit, y_es = y_tr.iloc[:_es_cut], y_tr.iloc[_es_cut:]
    if _es_source() == "validation" and MODEL_TYPE != "uplift":  # [v2.16.0 es_source]
        X_fit, y_fit, X_es, y_es = X_tr, y_tr, X_va, y_va
    _es_face = (_es_source() if MODEL_TYPE != "uplift" else "train_tail")
    if MODEL_TYPE == "uplift":
        X_es_t = X_es.assign(__treatment__=np.ones(len(X_es), dtype="int8"))
    step(f"es_split: fit={{len(X_fit)}} es_valid={{len(X_es)}} "
         f"(early stopping 面={{_es_face}})")

    # 线程数:优先读 scheduler 注入的 OMP_NUM_THREADS(与本批并发数
    # 反比联动:批越小线程越多,吃满 CPU 预算),--n-jobs 显式覆盖
    n_jobs = int(sys.argv[sys.argv.index("--n-jobs") + 1]) \\
        if "--n-jobs" in sys.argv \\
        else max(1, int(os.environ.get("OMP_NUM_THREADS", "4") or 4))
    model = build_estimator(n_jobs, len(X_fit))
    # early stopping 的 eval_set:validation 模式 = outer validation;
    # train_tail 模式 = train 尾部 es_valid。uplift 用同特征面的处置侧副本。
    # v2.17.3:早停集降采样(只影响"何时停",评分仍在全量 X_va/X_es 上算)。
    # 每轮评估在整个早停集上打分,是单 trial wall 的可观部分;2.55M/450k
    # 实测 full-ES 与 150k-ES 的 best_iter 804 vs 818、全量 AUC 一致到 6 位。
    _X_es_c, _y_es_c = _es_eval_cap(X_es, y_es)
    if MODEL_TYPE == "uplift":
        eval_set = [(X_es_t, y_es)]
    else:
        eval_set = [(_X_es_c, _y_es_c)]
    step(f"train start: algo={algorithm} n_jobs={{n_jobs}} "
         f"n_estimators={{N_ESTIMATORS_CAP}} rows={{len(X_fit)}} "
         f"features={{X_fit.shape[1]}}")
    {fit_code}
    step("train done")
    dump_feature_importance(model, X_tr.columns)
    dump_model(model, X_tr.columns)
    # v15.2:模型输入级 validation 帧落盘(分组置换消费;baseline 无
    # transform,但统一走同一契约)。uplift 打分帧含 __treatment__,
    # 落盘 S-learner 的处置侧副本口径)
    if MODEL_TYPE == "uplift":
        dump_eval_frame(X_va_t)
    else:
        dump_eval_frame(X_va)
    curve_info = dump_learning_curve(model)
    # v2.12.2.9:打分前统一预测设备(xgboost GPU 训练 → CPU 预测的显式切换,
    # 消除 DMatrix 设备回退;等价性证据见 align_predict_device docstring)
    align_predict_device(model)

    # 统一打分口:优先 predict_proba(GBM/sklearn),torch 模型走
    # score_validation()(自定义网卡的 forward sigmoid/sigmoid+max)
    if hasattr(model, "score_validation"):
        if MODEL_TYPE == "uplift":
            # torch baseline 不支持 uplift(配置层已剔除);防御性报错
            raise ValueError("uplift baseline 不支持 mlp score_validation 口径")
        score = model.score_validation(X_va)
    elif MODEL_TYPE == "uplift":
        # S-learner 双次打分:p1(w=1) - p0(w=0) = CATE 估计(可正可负)
        score = model.predict_proba(X_va_t)[:, 1] \\
            - model.predict_proba(X_va_c)[:, 1]
    elif MODEL_TYPE == "binary_classification":
        score = model.predict_proba(X_va)[:, 1]
    elif MODEL_TYPE == "multiclass":
        score = model.predict_proba(X_va).max(axis=1)
    else:
        score = model.predict(X_va)
    pd.DataFrame({{"score": score}}).to_parquet(
        TRIAL_DIR / "validation_score.parquet", index=False)
    # v18:shadow_test 打分落盘(交付链第二把尺子)。shadow 只作
    # sanity 与交付复核,不进反馈环(铁律 3);split 缺失(如 sample
    # 视图未落盘 shadow)时静默跳过,不阻塞 baseline。
    try:
        _sh_path = SPLITS / "shadow_test.parquet"
        # v2.17.3:搜索代(非 confirm)不产 shadow 工件。消费方只有 fresh
        # confirmation(runner 的 shadow_policy=confirmation_only),其余 trial
        # 的 shadow 分数从不进反馈环;447k 行上重放 transform + 特征块 +
        # 全量打分在冷缓存世代实测 557~686 s(g03 热缓存 55 s)。
        # MODELEVO_SHADOW_SCORE=skip 由 runner 按 search_action 下发。
        if (_sh_path.exists()
                and os.environ.get("MODELEVO_SHADOW_SCORE", "1") != "skip"):
            _sh = pd.read_parquet(_sh_path, columns=feats)
            _X_sh, _ = prepare_shadow_features(_sh, cat_cols, dropped_cols,
                                               _base_cols)
            if MODEL_TYPE != "uplift":
                _X_sh = _append_feature_blocks(_X_sh, "shadow_test")
            if hasattr(model, "score_validation"):
                _sh_score = model.score_validation(_X_sh)
            elif MODEL_TYPE == "uplift":
                _sh_score = model.predict_proba(
                    _X_sh.assign(__treatment__=np.ones(
                        len(_X_sh), dtype="int8")))[:, 1] \\
                    - model.predict_proba(
                        _X_sh.assign(__treatment__=np.zeros(
                            len(_X_sh), dtype="int8")))[:, 1]
            elif MODEL_TYPE == "binary_classification":
                _sh_score = model.predict_proba(_X_sh)[:, 1]
            elif MODEL_TYPE == "multiclass":
                _sh_score = model.predict_proba(_X_sh).max(axis=1)
            else:
                _sh_score = model.predict(_X_sh)
            dump_shadow_score(_sh_score)
            step(f"shadow score done: n={{len(_X_sh)}}")
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] shadow 打分跳过(不阻塞 baseline): {{_e}}",
              flush=True)
    step("score & dump done")

    metrics = {{"n_features": len(feats),
                "n_train": len(y_tr), "n_valid": len(y_va),
                "n_es_valid": len(y_es),
                "model_num": 1,   # 单模型 baseline;集成 trial 按实际数声明
                "best_iteration": best_iteration_of(model),
                "n_estimators_used": curve_info.get("learning_curve_rows"),
                "dtype_dropped_columns": dropped_cols,
                **curve_info}}
    if _n_block_cols:
        # v2.13.5:特征块加宽留痕(_hpo_parent_face_risk 据此识别加宽面)
        metrics["widen_stats"] = {{"n_new_cols": int(_n_block_cols),
                                  "source": "trial_feature_blocks"}}
    # v2.15.8:撞 cap 遥测(只留痕,不改裁决)。runner 读 es_info.cap_bound
    # 进 complexity,但 baseline 血缘从不写 —— 400w 上 baseline
    # best_iter=1976 贴着 cap=2000,反思与 Planner 看不到"被轮数截断"
    # v2.16.3(D-011):判据改走 runtime.hpo 的单一语义(实训轮数 ≥ cap;
    # 无轮数时 best + 早停耐心 ≥ cap;复合容器读 stage1_ 分段,任一段撞即
    # True)。旧口径 best ≥ 0.95×cap 与 hurdle 遥测同一 trial 判定相反
    # (run4 regression trial-022),"贴近 cap"另记 near_cap。
    try:
        _bi = metrics.get("best_iteration")
        if _bi and N_ESTIMATORS_CAP:
            _near = None
            try:
                from runtime.hpo import es_cap_telemetry as _ect
                _tel = _ect(model, cap=int(N_ESTIMATORS_CAP),
                            es_rounds=int(EARLY_STOP_ROUNDS or 0),
                            best_iteration=_bi,
                            rounds=metrics.get("n_estimators_used"))
                _bound, _near = _tel.get("cap_bound"), _tel.get("near_cap")
                _src = "unified:" + str(_tel.get("cap_bound_source"))
            except Exception:  # noqa: BLE001  runtime 不可导入:同语义内联版
                _r = metrics.get("n_estimators_used")
                _bound = (int(_r) >= int(N_ESTIMATORS_CAP)) if _r else (
                    int(_bi) + int(EARLY_STOP_ROUNDS or 0)
                    >= int(N_ESTIMATORS_CAP))
                _src = "unified:inline"
            metrics["es_info"] = {{
                "cap": int(N_ESTIMATORS_CAP), "cap_bound": bool(_bound),
                "stopped_by": "cap" if _bound else "early_stop",
                "near_cap": _near,
                "es_source": _es_face, "fit_mode": "single",
                "source": _src}}
    except Exception:  # noqa: BLE001
        pass
    # v13:训练后泄漏哨兵(top-K gain 特征的 train 单变量 AUC);
    # suspects 非空时评估器强制 invalid(leakage_suspect)
    metrics["leak_scan"] = leak_scan(model, X_tr, y_tr)
    if MODEL_TYPE == "binary_classification":
        from sklearn.metrics import roc_auc_score
        metrics["valid_auc"] = float(roc_auc_score(y_va, score))
    elif MODEL_TYPE == "uplift":
        # uplift:score=CATE 估计,harness 以 auuc/qini 裁决;
        # 此处只落 sanity 量(ATE 与处置占比),不算分类口径指标
        w_va = _load_treatment_for_metrics()
        metrics["treatment_rate"] = float(np.mean(w_va))
        metrics["ate_observed"] = float(
            y_va[w_va == 1].mean() - y_va[w_va == 0].mean())
    elif MODEL_TYPE == "regression":
        # 回归:score 即预测值,报 MSE 供快速 sanity(裁决仍由 harness 评估)
        metrics["valid_mse"] = float(np.mean((y_va - score) ** 2))
    # v2.14.0:multi-seed 段(与 trial 模板同契约)。baseline 血缘的后代
    # (快通道 confirm 复制父代代码)此前没有这一段 ⇒ confirm 声明的
    # multi_seed=3 被静默忽略,确认门只能靠 floor 估 seed 方差(run2
    # 实测)。变体轮只写 seed_scores/seed-k.parquet(validation 分数),
    # 不覆写 primary 工件(MODELEVO_SEED_VARIANT 期间不调用任何 dump_*)。
    _ms = int(cfg.get("multi_seed", 1) or 1)
    if _ms <= 1 and str(cfg.get("search_action") or "") == "confirm":
        _ms = 3
    if _ms > 1 and not (0 < _quick < 1) and MODEL_TYPE != "uplift" \\
            and not hasattr(model, "score_validation"):
        _seed0 = SEED
        os.environ["MODELEVO_SEED_VARIANT"] = "1"
        _sp_h = None
        try:
            _sd = TRIAL_DIR / "seed_scores"
            _sd.mkdir(parents=True, exist_ok=True)
            # v2.19.0:seed 变体分卡并行(runtime.seed_parallel)。旧实现在这里
            # 单卡串行(confirm 3 seed 115~230 min;并行实现只在 trial 模板
            # main 里,baseline 血缘从不经过)。条件不满足 / 不可用即串行;
            # worker 失败的变体在第 2 遍串行补齐(与串行同口径、同文件名)。
            try:
                from runtime.seed_parallel import start_seed_workers as _ssw
                _sp_h = _ssw(RUN_DIR, TRIAL_DIR, pathlib.Path(__file__).resolve(),
                             [_k for _k in range(1, _ms) if not (
                                 _sd / f"seed-{{_k}}.parquet").exists()],
                             _seed0, n_jobs, X_fit, y_fit, eval_set, X_va,
                             MODEL_TYPE,
                             # v2.20.1:worker 是独立进程,不会继承本模块的
                             # 模块级状态。当前血缘没有这两个名字(get None),
                             # 传给嫁接版后代时才有值 —— 不传就会让它们落进
                             # _fit_seed_variant 的兜底分支(把 validation 拼进
                             # 训练集 + 丢配方),产出伪 seed 分数。
                             primary_params=globals().get("_PRIMARY_FIT_PARAMS"),
                             primary_xy=globals().get("_PRIMARY_FIT_XY"))
            except Exception as _e_sp:  # noqa: BLE001
                print(f"[warn] seed 变体并行不可用,串行: {{_e_sp}}", flush=True)
            _sp_assigned = set((_sp_h or {{}}).get("assigned") or [])
            for _pass in range(2):
                if _pass == 1:
                    if _sp_h is None:
                        break
                    from runtime.seed_parallel import finish_seed_workers as _fsw
                    metrics["multi_seed_parallel"] = _fsw(_sp_h)
                    _sp_h, _sp_assigned = None, set()
                for _k in range(1, _ms):
                    if _k in _sp_assigned:
                        continue
                    if (_sd / f"seed-{{_k}}.parquet").exists():
                        if _pass == 0:
                            print(f"[resume] seed-{{_k}} 已存在,跳过", flush=True)
                        continue
                    globals()["SEED"] = _seed0 + 1000 * _k
                    _m = _fit_seed_variant(n_jobs, X_fit, y_fit, eval_set)
                    align_predict_device(_m)
                    if MODEL_TYPE == "binary_classification":
                        _sc = _m.predict_proba(X_va)[:, 1]
                    elif MODEL_TYPE == "multiclass":
                        _sc = _m.predict_proba(X_va).max(axis=1)
                    else:
                        _sc = _m.predict(X_va)
                    pd.DataFrame({{"score": _sc}}).to_parquet(
                        _sd / f"seed-{{_k}}.parquet", index=False)
                    step(f"multi-seed variant {{_k}}/{{_ms - 1}} done")
            metrics["multi_seed_n"] = _ms
        except Exception as _e:  # noqa: BLE001
            print(f"[warn] multi_seed 轮失败: {{_e}}", flush=True)
        finally:
            if _sp_h is not None:
                try:
                    from runtime.seed_parallel import finish_seed_workers as _fsw
                    metrics["multi_seed_parallel"] = _fsw(_sp_h, kill=True)
                except Exception:  # noqa: BLE001
                    pass
            globals()["SEED"] = _seed0
            os.environ.pop("MODELEVO_SEED_VARIANT", None)
    with open(TRIAL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
'''

_LGBM_CODE = """\
    import lightgbm as lgb
    _gpu = dict(device="cuda") if (_gpu_allocated(n_rows)
                                   and _lgbm_cuda_ok()) else {}
    if MODEL_TYPE == "regression":
        return lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS_CAP, learning_rate=0.05, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=n_jobs,
            verbosity=-1, **_gpu)
    return lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS_CAP, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=n_jobs,
        verbosity=-1, **_gpu)
"""

_XGB_CODE = """\
    import xgboost as xgb
    _gpu = dict(device="cuda") if (_gpu_allocated(n_rows)
                                   and _xgb_cuda_built()) else {}
    if MODEL_TYPE == "regression":
        return xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS_CAP, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=SEED,
            n_jobs=n_jobs, tree_method="hist",
            early_stopping_rounds=EARLY_STOP_ROUNDS, **_gpu)
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS_CAP, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=SEED,
        n_jobs=n_jobs, tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP_ROUNDS, **_gpu)
"""

# 深度学习(mlp):torch 实现。统一包装成与 sklearn 同口的 estimator:
#   - fit(X, y)         :内部 minibatch、early stopping(patience on loss)
#   - score_validation(X):返回正类概率/回归值(替代 predict_proba 口径)
#   - feature_importances_:数值特征 |W1| 均值近似(供 dump_feature_importance)
# GPU 约定:scheduler 已设 CUDA_VISIBLE_DEVICES 并注入 MODELEVO_GPU;
# torch 以 torch.cuda.is_available() 自动回退 CPU。GBDT(lightgbm/
# xgboost)由模板 _gpu_allocated + device=cuda 自动上卡,无卡/小样本/库
# 不支持时回退 CPU(lightgbm 非 GPU 编译轮在 fit 期由 _lgbm_fit_safe 兜底)。
_MLP_CODE = """\
    import torch
    import torch.nn as nn

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    class _MLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1 if MODEL_TYPE != "multiclass" else 16),
            )

        def forward(self, x):
            return self.net(x)

    class _Wrapper:
        # sklearn 同口包装:fit / score_validation / feature_importances_

        def __init__(self, n_jobs):
            self.n_jobs = n_jobs
            self.device = dev

        def fit(self, X, y):
            self.history_ = []  # 每 epoch 平均训练 loss(learning curve 口径)
            Xt = torch.tensor(X.to_numpy(dtype="float32"),
                              device=self.device)
            y_np = y.to_numpy()
            if MODEL_TYPE == "regression":
                yt = torch.tensor(y_np.astype("float32"),
                                  device=self.device).unsqueeze(1)
                self.criterion = nn.MSELoss()
            elif MODEL_TYPE == "multiclass":
                yt = torch.tensor(y_np.astype("int64"),
                                  device=self.device)
                self.criterion = nn.CrossEntropyLoss()
            else:
                yt = torch.tensor(y_np.astype("float32"),
                                  device=self.device).unsqueeze(1)
                self.criterion = nn.BCEWithLogitsLoss()
            self.model = _MLP(Xt.shape[1]).to(self.device)
            opt = torch.optim.AdamW(self.model.parameters(),
                                    lr=1e-3, weight_decay=1e-5)
            n = Xt.shape[0]
            bs = 4096
            epochs = 50
            best_loss, patience, bad = float("inf"), 3, 0
            self.model.train()
            g = torch.Generator(device=self.device).manual_seed(SEED)
            for ep in range(epochs):
                perm = torch.randperm(n, generator=g, device=self.device)
                tot = 0.0
                for i in range(0, n, bs):
                    idx = perm[i:i + bs]
                    opt.zero_grad()
                    out = self.model(Xt[idx])
                    loss = self.criterion(out, yt[idx])
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach()) * len(idx)
                avg = tot / n
                self.history_.append(avg)
                if avg < best_loss - 1e-5:
                    best_loss, bad = avg, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            # 数值特征重要性近似:首层权重绝对值按输入维平均
            with torch.no_grad():
                w = self.model.net[0].weight.abs().mean(dim=0)
            self.feature_importances_ = w.cpu().numpy()
            return self

        def score_validation(self, X):
            self.model.eval()
            with torch.no_grad():
                out = self.model(torch.tensor(
                    X.to_numpy(dtype="float32"), device=self.device))
            if MODEL_TYPE == "regression":
                return out.squeeze(1).cpu().numpy()
            if MODEL_TYPE == "multiclass":
                return torch.softmax(out, dim=1).max(dim=1).values.cpu().numpy()
            return torch.sigmoid(out).squeeze(1).cpu().numpy()

    return _Wrapper(n_jobs)
"""

_MLP_FIT_CODE = "model.fit(X_tr, y_tr)"

ESTIMATOR_CODE = {"lightgbm": _LGBM_CODE, "xgboost": _XGB_CODE,
                  "mlp": _MLP_CODE}
_LGBM_FIT_CODE = (
    "import lightgbm as _lgb_fit\n"
    "    # lightgbm 经 _lgbm_fit_safe 兜底:非 GPU 编译轮 device=cuda 在\n"
    "    # fit 期抛错 → 去 device 回退 CPU 重训一次(有卡尝试、无卡回退)\n"
    "    _lgbm_fit_safe(model, X_fit, y_fit, eval_set=eval_set,\n"
    "                   callbacks=[_lgb_fit.early_stopping(\n"
    "                       EARLY_STOP_ROUNDS, verbose=False)])")
FIT_CODE = {"lightgbm": _LGBM_FIT_CODE,
            "xgboost": "model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)",
            "mlp": _MLP_FIT_CODE}


class BaselineRunner:
    """生成并执行 gen-00 baseline Trials(经由 Scheduler 执行)。"""

    def __init__(self, spec: TaskSpec, config: Config, run_dir: Path):
        self.spec = spec
        self.config = config
        self.run_dir = Path(run_dir)

    def write_baseline_code(
        self, trial_dir: Path, algorithm: str, feature_cols: list[str],
        seed: int = 20260814, *,
        n_estimators_cap: int | None = None,
        early_stopping_rounds: int | None = None,
    ) -> Path:
        # v12/P0-8:非特征列(time/id/treatment)按 spec 注入
        # _SPEC_NONFEATURE,替换硬编码的 "pday"/"user_no"(换任务时
        # 硬编码列名会让时间列直接进特征面,构成时间穿越泄漏)
        nonfeature = {c for c in (self.spec.time_column,
                                  self.spec.id_column,
                                  self.spec.treatment_column) if c}
        from .config import clamp_trees
        _cap_raw = int(n_estimators_cap if n_estimators_cap is not None
                       else self.config.get("baseline.n_estimators_cap", 2000))
        code = BASELINE_TEMPLATE.format(
            algorithm=algorithm,
            target=self.spec.target_name,
            model_type=self.spec.model_type,
            treatment=self.spec.treatment_column or "",
            seed=seed,
            n_estimators_cap=clamp_trees(_cap_raw),
            early_stopping_rounds=int(
                early_stopping_rounds
                if early_stopping_rounds is not None else
                self.config.get("baseline.early_stopping_rounds", 50)),
            es_fraction=float(self.config.get(
                "baseline.early_stop_validation_fraction", 0.1)),
            estimator_code=ESTIMATOR_CODE[algorithm],
            fit_code=FIT_CODE[algorithm],
        )
        code = code.replace(
            "BANNED: frozenset[str] = frozenset()",
            f"BANNED: frozenset[str] = frozenset({sorted(set(self.spec.banned_columns))!r})")
        code = code.replace(
            'SKILL_ROOT = ""  # write_baseline_code 渲染后注入',
            f"SKILL_ROOT = {str(Path(__file__).resolve().parents[1])!r}"
            "  # v2.13.5 渲染期注入", 1)
        code = code.replace(
            "_SPEC_NONFEATURE: set[str] = set()",
            f"_SPEC_NONFEATURE: set[str] = set({sorted(nonfeature)!r})")
        # v2.16.0:早停面按 run 配置渲染(不加 format 字段,旧调用方不受影响)
        _es_src = str(self.config.get("training.early_stop_source",
                                      "validation") or "validation")
        _es_src = "train_tail" if _es_src.strip().lower() == "train_tail" \
            else "validation"
        code = code.replace('ES_SOURCE = "validation"',
                            f"ES_SOURCE = {_es_src!r}", 1)
        # v2.17.0(run4 regression.2 #7):回归任务按 model_type 渲染单一打分
        # 分支 —— 分类/uplift 的 predict_proba 通用分派不再生成(lint 规则 4
        # 误报的根治;分类任务逐字节不变)
        from .templates import narrow_model_type_dispatch
        code, _narrow_notes = narrow_model_type_dispatch(
            code, str(self.spec.model_type))
        # [FIX-2026-09-18] 渲染后自洽性检查:模板改错(调用了模板体里
        # 不存在也未 import 的名字)不该等到两轮全量训练烧完才发现。
        # 实测事故:本模板的 dtype 治理段调用 safe_to_numeric,而模板体
        # 既未定义它也未 import,两个 gen-00 trial 双双 NameError;失败
        # trial 按协议只留 invalid、不可删除,审计面被永久污染。
        # 这里在写盘前用 AST 扫一遍,把这类缺陷挡在 baseline 起跑之前。
        problems = check_generated_code(code)
        if problems:
            raise RuntimeError(
                "baseline 模板自洽性检查未通过(生成代码含未定义名/"
                "语法错误),拒绝落盘——这是 Harness 缺陷,请修复 "
                "runtime/baseline.py 的 BASELINE_TEMPLATE:\n  - "
                + "\n  - ".join(problems))
        code_dir = Path(trial_dir) / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        path = code_dir / "train.py"
        path.write_text(code, encoding="utf-8")
        # 可复现配置(§10.3):特征面单源存本文件——features.exclude 恒存
        # (TaskSpec 派生);TaskSpec 显式白名单(features.columns)时同步写
        # columns 锁定特征面。baseline 走 resolve_feature_columns 的全列
        # 派生(非白名单)时不写 columns(=全列−排除列),train.py 运行期
        # exclude_columns() 读取,代码不再内置 FEATURES 常量
        from .logging_utils import dump_yaml

        feats_cfg: dict = {"exclude": sorted(set(self.spec.excluded_columns)),
                           "banned": sorted(set(self.spec.banned_columns)),
                           "reparse": sorted(set(self.spec.reparse_columns))}
        if self.spec.feature_columns:
            feats_cfg["columns"] = list(self.spec.feature_columns)
        dump_yaml(Path(trial_dir) / "config" / "trial.yaml", {
            "trial_id": Path(trial_dir).name,
            "algorithm": algorithm,
            "features": feats_cfg,
            "target": self.spec.target_name,
            "model_type": self.spec.model_type,
            "seed": seed,
            "hyperparameters": "baseline 固定默认参数(见 code/train.py)",
            "notes": "baseline 不做 HPO(§8);特征面单源存本文件 features,"
                     "code/train.py 运行期经 exclude_columns() 读取",
        })
        return path

    def _configured_algorithms(self) -> tuple[list[str], bool]:
        """run 配置声明的 baseline 算法(v2.17.0)。

        cmd_baseline 的 v2.15 规则会在内存里把 [lightgbm, xgboost] 收成
        [xgboost](run4 regression.2 #22:配置要求 LGBM+XGB,却只产出 xgboost
        一个 trial)。这里以落盘的 run 配置为准还原;返回 (算法列表, 是否
        撤销了内存收缩)。
        """
        mem = [str(a) for a in (self.config.get("baseline.algorithms")
                                or ["lightgbm"])]
        try:
            from .config import Config

            disk = Config.load(run_config=self.run_dir / "modelevo-config.yaml"
                               ).get("baseline.algorithms")
        except Exception:  # noqa: BLE001
            disk = None
        disk = [str(a) for a in disk] if isinstance(disk, (list, tuple)) else None
        if disk and mem == ["xgboost"] and {"lightgbm", "xgboost"} <= set(disk):
            return disk, True
        return mem, False

    def _warm_covered_algorithms(self) -> list[str]:
        """经验库榜单前 warm_rank_limit 名配方覆盖的算法(只用于留痕)。

        v2.18.0:读本 run 固定水位的经验库视图(旧实现读 memory/top-trials)。"""
        try:
            from . import library_view

            atid = self.spec.resolve_abstract_task_id(self.run_dir)
            if not atid or "pending" in atid:
                return []
            view = library_view.ensure(self.run_dir, atid=atid, spec=self.spec,
                                      config=self.config)
            lim = int(self.config.get("baseline.warm_rank_limit", 2) or 2)
            out = []
            for row in view.leaderboard[:lim]:
                a = str(row.get("algorithm") or "").lower()
                if a and a not in out:
                    out.append(a)
            return out
        except Exception:  # noqa: BLE001
            return []

    def plan_baseline_trials(self, resource_plan,
                              force_all: bool = False) -> list[str]:
        """按配置与资源决定 baseline 算法集合。

        uplift 任务剔除 mlp:torch baseline 的 score_validation() 只有
        单次前向口径,不支持 S-learner 的 p1-p0 双次打分。
        v10.3:force_all=True(cmd_baseline 传入)时不受单并发截断——
        baseline 是串行执行,单并发只影响线程,不该砍掉声明的算法。
        v2.17.0(#22):warm-start 不再"替代"已配置算法 —— 全局库热启动未
        覆盖的已配置算法按常规 baseline 补跑;不想补时显式配置
        `baseline.skip_algorithms: [..]`。锚点(trial-001)算法取
        `baseline.anchor_algorithm`(缺省:配置含 xgboost 时为 xgboost,与
        v2.15+ 的锚点及全局库固化配方保持跨 run 可比)。计划写 stdout 留痕。
        """
        configured, reverted = self._configured_algorithms()
        algorithms, trace = resolve_baseline_algorithms(
            configured,
            model_type=str(self.spec.model_type),
            max_trials=int(self.config.get("baseline.max_trials", 2) or 2),
            skip=self.config.get("baseline.skip_algorithms") or (),
            anchor=self.config.get("baseline.anchor_algorithm"),
            warm_covered=self._warm_covered_algorithms())
        trace["legacy_collapse_reverted"] = reverted
        if not force_all and (resource_plan.n_parallel_trials < 2
                              or int(self.config.get("baseline.max_trials", 2)
                                     or 2) < 2):
            for a in algorithms[1:]:
                trace["skipped"].append({"algorithm": a,
                                         "reason": "单并发且非 force_all"})
            algorithms = algorithms[:1]
            trace["run"] = list(algorithms)
        self.last_plan_trace = trace
        print(format_baseline_plan(trace), flush=True)
        return algorithms


def resolve_baseline_algorithms(configured, *, model_type: str | None = None,
                                max_trials: int = 2, skip=(), anchor=None,
                                warm_covered=()) -> tuple[list[str], dict]:
    """baseline 算法计划(v2.17.0)。返回 (运行顺序, 留痕 dict)。

    - 已配置算法全部按常规 baseline 运行(warm-start 覆盖与否不影响);
    - `skip`(baseline.skip_algorithms)显式排除,留痕原因;
    - uplift 剔除 mlp/nn/torch(S-learner 双次打分不支持);
    - 锚点算法排第一(trial-001 = baseline 锚点);其余保持配置顺序;
    - 超出 max_trials 的截断并留痕(不静默)。
    """
    conf = []
    for a in configured or []:
        a = str(a).strip().lower()
        if a and a not in conf:
            conf.append(a)
    skip_set = {str(s).strip().lower() for s in (skip or ()) if str(s).strip()}
    skipped: list[dict] = []
    algos = []
    for a in conf:
        if a in skip_set:
            skipped.append({"algorithm": a,
                            "reason": "baseline.skip_algorithms 显式配置"})
        elif model_type == "uplift" and a in ("mlp", "nn", "torch"):
            skipped.append({"algorithm": a,
                            "reason": "uplift S-learner 不支持 torch baseline"})
        else:
            algos.append(a)
    anc = str(anchor).strip().lower() if anchor else (
        "xgboost" if "xgboost" in algos else (algos[0] if algos else None))
    if anc in algos:
        algos = [anc] + [a for a in algos if a != anc]
    cap = max(1, int(max_trials or 1))
    for a in algos[cap:]:
        skipped.append({"algorithm": a,
                        "reason": f"超出 baseline.max_trials={cap}"})
    algos = algos[:cap]
    warm = [str(w).lower() for w in (warm_covered or ())]
    trace = {"configured": conf, "run": list(algos),
             "anchor": algos[0] if algos else None,
             "warm_start_covered": warm,
             "backfilled": [a for a in algos if a not in warm],
             "skipped": skipped}
    return algos, trace


def format_baseline_plan(trace: dict) -> str:
    """baseline 算法计划的 stdout 留痕行。"""
    sk = "; ".join(f"{s['algorithm']}({s['reason']})"
                   for s in trace.get("skipped") or []) or "无"
    rev = (";已撤销 v2.15 内存收缩([lightgbm, xgboost]→[xgboost]),按落盘配置补齐"
           if trace.get("legacy_collapse_reverted") else "")
    return (f"[baseline] 算法计划: 配置 {trace.get('configured')} → 运行 "
            f"{trace.get('run')}(锚点 {trace.get('anchor')});warm-start 覆盖 "
            f"{trace.get('warm_start_covered') or []},按常规 baseline 补跑 "
            f"{trace.get('backfilled') or []};跳过: {sk}{rev}")
