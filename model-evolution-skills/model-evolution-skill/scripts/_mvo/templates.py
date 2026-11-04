"""Trial 代码模板(v2.11.9 从 scripts/modelevo.py 抽出)。

这两个常量是**代码生成**的输入:`cmd_new_trial` 把 `TRIAL_BODY` 交给
`runtime.templates.TRIAL_TEMPLATE.format(body=...)` 渲染成 trial 的
`code/train.py`;`_graft_hpo_section` 也从 `TRIAL_BODY` 里抽取 hpo 段
嫁接进父代代码。

抽出的动机(实测):
- 原实现把 811 行模板字面量嵌在 `cmd_new_trial`(1325 行)函数体内,
  文件无法按"命令实现"浏览;
- 嫁接函数此前靠**读自己的源码**做字面量考古(找 `"\\n    body = " +
  "\'\'\'"`、`ast.literal_eval` 切片)才拿到模板值,模板结构一改就
  SystemExit;现在直接 `from .templates import TRIAL_BODY`。
"""
from __future__ import annotations

#: automl 参照系 trial body(v18/P2-3):--action automl 零写码渲染。
#: 外部锚点,不计入交付候选;引擎 autogluon|flaml 缺失时显式报错。
AUTOML_BODY = '''import os


def load_config() -> dict:
    cfg_path = pathlib.Path(__file__).resolve().parent.parent / "config" / "trial.yaml"
    if not cfg_path.exists():
        return dict()
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or dict()


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    """外部 AutoML 参照系(v18/P2-3):回答"成熟 AutoML 在当前特征面
    能到哪"——capacity 归因的第三方证词。不计入交付候选
    (leaderboard reference_only,best_trial 过滤),不进反馈环。"""
    metrics_extra = {"n_features": X_tr.shape[1], "model_num": 1,
                     "automl_reference": True}
    import pathlib
    import yaml
    _engine = str(cfg.get("automl_engine") or os.environ.get(
        "MODELEVO_AUTOML_ENGINE", "autogluon")).lower()
    _time_limit = int(cfg.get("automl_time_limit_s") or 600)
    step(f"automl reference: engine={_engine} time_limit={_time_limit}s")
    if _engine == "autogluon":
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as _e:
            raise RuntimeError(
                "autogluon 未安装(可选依赖,preflight 已声明)。"
                "automl 参照系需要它;安装后再跑本 trial。") from _e
        _tr = X_tr.copy()
        _tr["__label__"] = y_tr.values if hasattr(y_tr, "values") else y_tr
        _pred = TabularPredictor(
            label="__label__", verbosity=0,
            eval_metric="roc_auc" if MODEL_TYPE == "binary_classification"
            else None)
        _pred.fit(_tr, time_limit=_time_limit, seed=seed)
        score = _pred.predict_proba(X_va)[:, 1]
        model = _pred
    elif _engine == "flaml":
        try:
            from flaml import AutoML
        except ImportError as _e:
            raise RuntimeError(
                "flaml 未安装(可选依赖,preflight 已声明)。"
                "automl 参照系需要它;安装后再跑本 trial。") from _e
        _automl = AutoML()
        _est = "lgbm"
        _y = y_tr.values if hasattr(y_tr, "values") else y_tr
        _automl.fit(X_tr, _y, task="classification"
                    if MODEL_TYPE == "binary_classification" else "regression",
                    time_budget=_time_limit, seed=seed, estimator_list=[_est])
        if hasattr(_automl, "predict_proba"):
            score = _automl.predict_proba(X_va)[:, 1]
        else:
            score = _automl.predict(X_va)
        model = _automl
    else:
        raise ValueError(f"未知 automl engine: {_engine}")
    metrics_extra["automl_engine"] = _engine
    metrics_extra["automl_time_limit_s"] = _time_limit
    dump_model(model, X_tr.columns)
    global _last_model
    _last_model = None if os.environ.get("MODELEVO_SMOKE_RUN") == "1" else model
    return score, metrics_extra'''


TRIAL_BODY = '''
import os

#: [铁律]树数量硬上限。任何 trial(含 agent 手改后的)的
#  n_estimators / num_boost_round / iterations 都不得超过此值。
#  本常量与下方 _clamp_trees() 自包含在本文件内——不依赖 runtime import,
#  agent 改写其它逻辑时该上限仍在。改动前先读 references/protocol-digest.md
#  的"树数上限"条目:该上限的例外需要人工授权,不允许 run 内自行绕过。
_MAX_TREES = 8000


def _clamp_trees(value) -> int:
    """把树数量读数钳到 [1, _MAX_TREES];非法值回退 1。"""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return 1
    return max(1, min(n, _MAX_TREES))


def load_config() -> dict:
    # 读取 ../config/trial.yaml(可复现配置;超参/seed 等运行时参数)
    cfg_path = pathlib.Path(__file__).resolve().parent.parent / "config" / "trial.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def pick_gpu() -> None:
    # GPU 优先(v14.2):有卡即用,选空闲显存最大的卡;无卡/探测失败
    # 则不动(回退 CPU)。旧实现 >20GB 硬门槛把小卡全判给 CPU,有卡不用。
    # v17.4:降配重试(runner 把 MODELEVO_GPU 置 "0")时保持 CPU——不再
    # 用 pynvml 把 scheduler 的 CPU 回退顶回一张卡。
    # v2.13.3:runner 已经用**跨进程排他租约**定好设备时(CUDA_VISIBLE_
    # DEVICES 非空),必须原样沿用——本函数是"独立运行/手工调试"时的
    # 兜底探测,自行挑"空闲显存最大"的卡会覆盖租约裁决,让多个并发
    # trial 又回到"各自认定同一张最空卡"的老路(gen-04 三 trial 全落
    # GPU0 的成因)。故只在未指定设备时才探测。
    if os.environ.get("MODELEVO_GPU") == "0":
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
    try:
        import pynvml

        pynvml.nvmlInit()
        best_i, best_free = None, -1.0
        for gi in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(gi)
            free_gb = pynvml.nvmlDeviceGetMemoryInfo(h).free / 2**30
            if free_gb > best_free:
                best_i, best_free = gi, free_gb
        pynvml.nvmlShutdown()
        if best_i is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_i)
    except Exception:
        pass


def n_jobs() -> int:
    """线程预算:scheduler 注入的 OMP_NUM_THREADS(与本批 trial 数
    反比联动——批越小每 trial 线程越多)。estimator 的 n_jobs 参数
    一律取本函数返回值,不要硬编码:否则调度器"并发×线程≈可用核"
    的分配会被代码内固定值(如 n_jobs=12)短路,小批次时 CPU 闲置。"""
    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS", "4") or 4))
    except ValueError:
        return 4


#: GPU 加速的最小训练行数(v17.4):小样本(极端小数据/small-sample)不上卡
_GPU_MIN_ROWS = 2000


def _gpu_allocated(n_rows: int) -> bool:
    # 唯一 GPU 意图信号 = scheduler/runner 注入的 MODELEVO_GPU(降配重试
    # 时置 "0")。不读 CUDA_VISIBLE_DEVICES(它会被 pick_gpu 改写,不代表
    # scheduler 的真实分配)。
    return os.environ.get("MODELEVO_GPU") == "1" and (n_rows or 0) >= _GPU_MIN_ROWS


def _xgb_cuda_built() -> bool:
    # xgboost 有可靠编译探测;经 importlib 动态加载,避免静态 import
    # xgboost 出现在非 xgboost 算法 trial 里被 lint 规则 2 判成多 GBM 库
    try:
        import importlib as _il
        _m = _il.import_module("xgboost")
        return bool((_m.build_info() or {}).get("USE_CUDA"))
    except Exception:
        return False


def _lgbm_fit_safe(model, *a, **kw):
    # lightgbm 无公开 GPU 编译探测;device=cuda 在非 GPU 编译轮上于
    # booster 创建期抛 LightGBMError → 去 device 回退 CPU 重训一次
    # (失败在建树前,CPU 重训代价≈0;保证"有卡尝试、无卡回退")。
    # [FIX-2026-09-03] 算法化:device/task_type 两种回退都尝试,
    # 覆盖 xgboost(device)/catboost(task_type)构造差异。
    # v19(req-t7-drawdown gen-03 实测回写):lightgbm 4.x 中
    # task_type 是核心参数 task 的别名(_ConfigAliases.get("task")=
    # {'task','task_type'}),set_params(task_type="CPU") 落到
    # task="CPU" → C++ 端小写化后报 "Unknown task type cpu"——回退
    # 本身制造新异常,t011 attempt-1 因此烧一轮全量重训。修复:回退
    # 统一 sanitize——从 get_params() 弹出 device/task/task_type 全部
    # 矛盾项后**重构造干净估计器**再 fit(set_params 对别名参数有
    # 残留语义,清场必须重建)。
    try:
        return model.fit(*a, **kw)
    except Exception:
        _clean = None
        try:
            _cls = type(model)
            _params = dict(model.get_params())
            for _bad in ("device", "task", "task_type"):
                _params.pop(_bad, None)
            _clean = _cls(**_params)
            return _clean.fit(*a, **kw)
        except Exception:
            pass
        try:
            model.set_params(device="cpu")
            return model.fit(*a, **kw)
        except Exception:
            pass
        try:
            model.set_params(task_type="CPU")
            return model.fit(*a, **kw)
        except Exception:
            pass
        raise


def dump_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


import time as _time

_T0 = _time.time()

# hpo 动作(v11)的 trial 内搜索实现:inner-validation 拟合与打分。
# _build_model/_hpo_fit_inner/_fit_with_eval/_model_score 四件套由
# Agent 按父代算法改写(默认实现 = LGBM,与 baseline 默认参同源);
# 搜索过程绝不触碰 outer validation 的 y(只有最终重训模型打一次分)。
# v2.13:METRIC_NAME 从 RUN_DIR/task-spec.yaml 读 spec.primary_metric
# (缺 spec/读取失败回退 "auc");inner 目标跟随主指标而非恒 roc_auc,
# 方向由 runtime.metrics.metric_direction 统一判定(此前 _INNER_METRIC_UP
# 恒 True,logloss/rmse 任务的搜索方向是反的)。
def _spec_primary_metric(default="auc"):
    try:
        _spec_path = RUN_DIR / "task-spec.yaml"
        if not _spec_path.exists():
            return default
        import yaml as _yaml_spec
        _d = _yaml_spec.safe_load(
            _spec_path.read_text(encoding="utf-8")) or {}
        _pm = str((_d or {}).get("primary_metric") or "").lower()
        return _pm or default
    except Exception:
        return default


METRIC_NAME = _spec_primary_metric()
# _inner_metric 返回值方向由 metric_direction(METRIC_NAME) 决定
# (logloss/rmse 越小越好),hpo 采样器经 metric_direction_up 感知
_INNER_METRIC_UP = True  # 占位:模块加载后由 metric_direction 覆写
try:
    from runtime.metrics import metric_direction as _md_dir

    _INNER_METRIC_UP = bool(_md_dir(METRIC_NAME))
except Exception:
    _INNER_METRIC_UP = True

# [FIX-2026-09-10] lightgbm wheel CUDA 编译探测(模块级缓存):
# 非 CUDA 编译轮上 device=cuda 的 fit 期 Fatal 曾致进程死、
# CPU 回退没接住——注入前先 64 行 x 4 特征小样本试 fit,
# 一切异常按 False 处理。xgboost 走 _xgb_cuda_built()。
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


def _build_model(params, seed, _n_rows=0):
    # [FIX-2026-09-03] 算法化构造:模板渲染路径(--action hpo 无
    # --from-parent)由 harness 在代码顶部注入 _HPO_ALGO 常量,构造
    # 经 runtime.hpo.build_gbm 按算法分发——四件套不再硬编码 lightgbm
    # (旧实现迫使 baseline 血缘 run 建 scaffold+手改 7 处,gen-03
    # 实测 21 分钟)。--from-parent 手写路径不受影响(无 _HPO_ALGO
    # 时回退 LGBM 默认,Agent 照旧按父代算法改写)。
    if "_HPO_ALGO" in globals() and globals().get("_HPO_ALGO"):
        from runtime.hpo import build_gbm

        # [v2.15.0] 容量口径透传:trial 声明了 N_ESTIMATORS_CAP /
        # EARLY_STOP_ROUNDS(baseline 血缘的模块常量)时必须交给 build_gbm,
        # 否则落回 runtime 默认 2000(未声明时由 runner 按 run 配置下发的
        # MODELEVO_N_ESTIMATORS_CAP 兜底)。实测 req-t7-rand400w:嫁接段
        # 不透传 → cap=8000 的 run 上 HPO 重训被截在 1999 轮、Δ 反号
        # −0.0024;同一缺陷在 run2/run3 命中 3 次,每次靠 Agent 手补。
        _cap = globals().get("N_ESTIMATORS_CAP")
        _esr = globals().get("EARLY_STOP_ROUNDS")
        return build_gbm(globals()["_HPO_ALGO"], params, seed, MODEL_TYPE,
                         _n_rows, _gpu_allocated(_n_rows),
                         n_estimators_cap=int(_cap) if _cap else None,
                         early_stopping_rounds=int(_esr) if _esr else None)
    import lightgbm as _lgb

    base = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                subsample=0.9, colsample_bytree=0.9, n_jobs=n_jobs(),
                random_state=seed, verbosity=-1)
    # v17.4:scheduler 分配了卡且样本量足够 → device=cuda(lightgbm 无
    # 编译探测,非 GPU 轮由 _lgbm_fit_safe 在 fit 期回退 CPU)。放在
    # base.update(params) 之前,config 的 device=cpu 仍可显式压回 CPU。
    if _gpu_allocated(_n_rows) and _lgbm_cuda_ok():
        base["device"] = "cuda"
    base.update(params or {})
    # [铁律]params 可能显式给超限树数 —— 构造前钳到 _MAX_TREES
    for _k in ("n_estimators", "num_iterations", "num_boost_round"):
        if _k in base:
            base[_k] = _clamp_trees(base[_k])
    if MODEL_TYPE == "regression":
        return _lgb.LGBMRegressor(**base)
    return _lgb.LGBMClassifier(**base)


def _inner_metric(y_true, score):
    # v2.13:inner 目标跟随 spec.primary_metric(经 METRIC_NAME),统一走
    # runtime.metrics.compute_metric——旧实现分类恒 roc_auc、回归恒负
    # MSE,spec 声明 logloss/rmse 时"搜的指标"与"裁决指标"不一致。
    from runtime.metrics import compute_metric

    return float(compute_metric(METRIC_NAME, y_true, None, score=score))


def _hpo_fit_inner(params, X_fit, y_fit, X_in, y_in, seed, fidelity=1.0):
    # v2.13:inner 训练集的 fit 同样走 _fit_with_es(cap+早停机制,
    # 与交付/baseline 同函数族;旧实现固定 300 棵全量 fit,LR 空间
    # 在该机制下系统性偏向高 LR)。inner-valid X_in/y_in 只用于评分。
    # v2.15.0:
    # - inner 打分直接用**第一段早停模型**(refit=False)。旧实现每个
    #   config 早停定轮后再在 X_fit 全量重拟合一次,inner 成本约翻倍,
    #   而重拟合对"哪组超参更好"的排序没有增量信息;
    # - 返回 dict 带逐 config 遥测(best_iteration / 是否撞 cap),
    #   run_hpo_search 兼容 float / dict 两种返回;
    # - fidelity < 1(多保真低档)时在 X_fit 内分层随机抽子样本。
    if fidelity is not None and float(fidelity) < 1.0:
        from runtime.hpo import stratified_random_rows

        _pos = stratified_random_rows(
            y_fit, max(1, int(len(y_fit) * float(fidelity))), seed=int(seed))
        X_fit, y_fit = X_fit.iloc[_pos], y_fit.iloc[_pos]
    m = _build_model(params, seed, len(X_fit))
    m = _fit_with_es(m, X_fit, y_fit, refit=False)
    align_predict_device(m)
    if hasattr(m, "predict_proba"):
        s = m.predict_proba(X_in)[:, 1]
    else:
        s = m.predict(X_in)
    _info = dict(getattr(m, "mvo_es_info_", None) or {})
    return {"value": _inner_metric(y_in, s),
            "best_iteration": _info.get("stage1_best_iteration"),
            "rounds": _info.get("stage1_rounds"),
            "cap_bound": _info.get("cap_bound")}


def _fit_with_es(model, X, y, es_fraction=0.1, rounds=None, refit=True,
                 auto_extend=False):
    """cap+早停机制的三族统一 fit(v2.13 HPO 机制对齐)。

    机制对齐的背景:baseline/交付重训用 n_estimators_cap=2000 +
    early_stopping_rounds=50,而 hpo 模板旧实现(`_fit_with_eval` 的
    hasattr 探测)在 LightGBM 上必然走 else 分支——未 fit 的
    LGBMClassifier 访问 best_iteration_ 抛 LGBMNotFittedError
    (AttributeError 子类)→ hasattr 返回 False → 全量 fit、无早停、
    无轮数截断;XGB/CatBoost 分支传了 eval_set 但构造参数里没有
    早停项,同样不早停。HPO 搜的是"固定 N 棵"机制、交付用的是
    "早停"机制,不是同一函数族。

    实现:
    - 族判定用 type(model).__module__ 顶层包名(lightgbm/xgboost/
      catboost),不再 hasattr(best_iteration_ 在未 fit 的 LGBM 上
      必然抛 AttributeError);
    - 只切**传入的 X/y** 尾部 es_fraction 作 es_valid,绝不引用
      outer validation(lint 规则 8);
    - 第一段:在 X_head 上拟合、X_es 早停,定出最优树数;
    - 第二段(refit=True):树数定死后在全量 X/y 重训(catboost 按
      get_params() 重建干净估计器,**返回新对象**,调用方必须用返回值
      重绑定);
    - 拿不到 best_iter(=0/None)或任何异常 → 回退全量 fit(仍不触
      outer validation)。

    v2.15.0(req-t7-rand400w run3 复盘回写):
    - **第一段遥测**挂在返回模型的 `mvo_es_info_` 上:第一段 held-out
      曲线、最优树数(1-based)、实训轮数、cap、是否撞 cap、早停耐心。
      旧实现只留第二段的 evals_result —— 第二段的 eval_set 已在训练
      集里,是**样本内曲线**,天然单调到最后一轮,于是
      `best_iteration == n_estimators_used` 恒成立;run3 的 UD-08 据此
      误判"7 个 trial 全欠拟合",而真正撞 cap 的 021/022 却被反思写成
      "7990<8000 早停真实触发"。撞 cap 的正确判据 = 第一段实训轮数
      达到 cap(早停 patience 没来得及到期);
    - **auto_extend**(outer 重训用):第一段撞 cap 时按
      MODELEVO_CAP_EXTEND_FACTOR(默认 2)放大 cap 重跑第一段,最多
      MODELEVO_CAP_EXTEND_TIMES(默认 1)次;外扩事实写进遥测;
    - 早停耐心随 lr 缩放:max(rounds, ceil(MODELEVO_ES_LR_SCALE / lr)),
      默认 scale=0.5(lr≥0.01 时与旧口径相同;低 lr 下 50 轮的耐心对应的
      "学习距离"过短,易被 es 切片噪声提前叫停);
    - 第二段树数按各库索引口径归一(xgboost/catboost 的 best 是 0-based
      索引,旧实现直接当树数用,少一棵最优树)。
    """
    import math as _math_es

    if rounds is None:
        try:
            rounds = int(os.environ.get("MODELEVO_EARLY_STOP_ROUNDS")
                         or globals().get("EARLY_STOP_ROUNDS") or 50)
        except (TypeError, ValueError):
            rounds = 50
    _fam = str(type(model).__module__ or "").split(".")[0]
    _n = len(X)
    _cut = int(_n * (1.0 - float(es_fraction)))
    if _cut <= 0 or _cut >= _n:
        # 数据太小切不出 es_valid:直接全量 fit(降级但不报错)
        return _lgbm_fit_safe(model, X, y)
    X_head, y_head = X.iloc[:_cut], y.iloc[:_cut]
    X_es, y_es = X.iloc[_cut:], y.iloc[_cut:]
    try:
        rounds = _es_rounds_for(model, rounds)
        _cap0 = _model_cap(model, _fam)
        info = {"es_rounds": int(rounds), "cap": _cap0,
                "cap_initial": _cap0, "cap_extended_to": None}
        _times = (int(os.environ.get("MODELEVO_CAP_EXTEND_TIMES", "1") or 0)
                  if auto_extend else 0)
        _factor = float(os.environ.get("MODELEVO_CAP_EXTEND_FACTOR", "2")
                        or 2)
        while True:
            model = _es_stage1(model, _fam, X_head, y_head, X_es, y_es,
                               rounds)
            _curve = _es_curve_of(model, _fam)
            _best = _es_best_trees(model, _fam)
            _cap = _model_cap(model, _fam)
            _bound = bool(_cap and len(_curve) >= int(_cap))
            info.update({"stage1_best_iteration": _best,
                         "stage1_rounds": len(_curve) or None,
                         "cap": _cap, "cap_bound": _bound,
                         "stopped_by": "cap" if _bound else "early_stop",
                         "stage1_curve": [round(float(v), 6)
                                          for v in _curve]})
            if not (_bound and _times > 0 and _cap):
                break
            _times -= 1
            _new_cap = _clamp_trees(int(_math_es.ceil(int(_cap) * max(_factor, 1.0))))
            if _new_cap <= int(_cap):
                break
            model = _set_cap(model, _fam, _new_cap)
            info["cap_extended_to"] = _new_cap
            print(f"[step] es stage1 撞 cap={_cap}(patience 未到期),"
                  f"cap 外扩到 {_new_cap} 重跑第一段", flush=True)
        # v2.17.2(A1):与 _fit_es_on_validation 同口径(无余量仍撞 cap)
        info["cap_bound_unresolved"] = bool(_bound and int(_cap or 0) >= int(
            globals().get("_MAX_TREES") or 8000))
        if not refit or not _best:
            try:
                model.mvo_es_info_ = info
            except Exception:  # noqa: BLE001
                pass
            if not _best:
                return _lgbm_fit_safe(model, X, y)
            return model
        info["refit_trees"] = int(_best)
        if _fam == "lightgbm":
            import lightgbm as _lgb_es

            model.set_params(n_estimators=int(_best))
            model = _lgbm_fit_safe(model, X, y, eval_set=[(X_es, y_es)],
                                   callbacks=[_lgb_es.early_stopping(
                                       int(rounds), verbose=False)]) or model
        elif _fam == "xgboost":
            model.set_params(n_estimators=int(_best))
            model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
        elif _fam == "catboost":
            _params = dict(model.get_params())
            _params["iterations"] = int(_best)
            model = type(model)(**_params)
            model.fit(X, y, eval_set=(X_es, y_es))
        else:
            model = _lgbm_fit_safe(model, X, y) or model
        try:
            model.mvo_es_info_ = info
        except Exception:  # noqa: BLE001
            pass
        return model
    except Exception:
        try:
            return _lgbm_fit_safe(model, X, y)
        except Exception:
            try:
                model.fit(X, y)
            except Exception:
                # xgboost/catboost 构造期早停参数要求 eval_set:
                # 全量 fit 兜底也要带上 es 切片(仍只用传入的 X/y)
                model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
            return model


def _es_eval_cap(X_es, y_es):
    """v2.17.3:早停集降采样(只影响"何时停",不影响任何评分口径)。

    每轮评估都在整个早停集上打分。req-t7 run4 上早停集是 44.7 万行 ×
    2317 列,每轮评估成本随已建树数累积,是单 trial wall 的可观部分;而
    早停只需要"验证损失曲线的最小值位置",44.7 万行的曲线与 15 万行几乎
    重合(2.55M/450k 实测:best_iter 804 vs 818、全量 AUC 一致到 6 位)。

    **按标签分层 + 固定种子随机抽**(v2.17.3 验证回写):早先实现取"分层头",
    但特征帧按时间排序,头部退化成时间前缀 —— 本 run 的 validation 只有
    3 个 pday,头部只抽中最早一天(149,069 / 931 / 0 行),早停判定面被收窄
    到单一日期。改为在每类内部做确定性随机抽样(seed 固定),三个日期按
    原比例保留。

    行数上限由 training.early_stop_eval_rows → MODELEVO_ES_EVAL_ROWS 下发,
    0/缺省 = 不降采样(旧行为)。返回 (X_es, y_es),不满足条件时原样返回。
    """
    try:
        _cap = int(os.environ.get("MODELEVO_ES_EVAL_ROWS") or 0)
    except (TypeError, ValueError):
        return X_es, y_es
    if not _cap or len(X_es) <= _cap:
        return X_es, y_es
    try:
        import pandas as _pd
        _y = _pd.Series(np.asarray(y_es))
        _frac = float(_cap) / float(len(_y))
        _rng = np.random.default_rng(int(os.environ.get(
            "MODELEVO_ES_EVAL_SEED") or 20260814))
        _parts = []
        for _v, _grp in _y.groupby(_y, sort=True):
            _k = max(1, int(round(len(_grp) * _frac)))
            _idx = np.asarray(_grp.index)
            if _k < len(_idx):
                _idx = np.sort(_rng.choice(_idx, _k, replace=False))
            _parts.append(_idx)
        _idx = np.sort(np.concatenate(_parts)).astype(np.int64)
        if len(_idx) < 1000:
            return X_es, y_es
        _xs = (X_es.iloc[_idx] if hasattr(X_es, "iloc") else X_es[_idx])
        _ys = (y_es.iloc[_idx] if hasattr(y_es, "iloc") else y_es[_idx])
        print(f"[step] es_eval 降采样 {len(X_es)} → {len(_idx)} 行"
              f"(分层随机;只影响早停位置)", flush=True)
        return _xs, _ys
    except Exception as _e:  # noqa: BLE001  降采样失败不阻塞训练
        print(f"[warn] es_eval 降采样跳过: {type(_e).__name__}: {_e}",
              flush=True)
        return X_es, y_es


def _es_stage1(model, fam, X_head, y_head, X_es, y_es, rounds):
    """第一段:X_head 拟合、X_es 早停(各库原生早停机制)。"""
    # v2.17.3:早停集降采样(globals().get:老副本被逐函数搬运时该 helper
    # 可能不存在,缺失即保持旧行为,不让升级后的 _es_stage1 崩)
    _cap_fn = globals().get("_es_eval_cap")
    if _cap_fn is not None:
        X_es, y_es = _cap_fn(X_es, y_es)
    if fam == "lightgbm":
        import lightgbm as _lgb_s1

        return _lgbm_fit_safe(model, X_head, y_head,
                              eval_set=[(X_es, y_es)],
                              callbacks=[_lgb_s1.early_stopping(
                                  int(rounds), verbose=False)]) or model
    if fam == "xgboost":
        try:
            model.set_params(early_stopping_rounds=int(rounds))
        except Exception:  # noqa: BLE001
            pass
        model.fit(X_head, y_head, eval_set=[(X_es, y_es)], verbose=False)
        return model
    if fam == "catboost":
        _p = dict(model.get_params())
        if _p.get("od_wait") != int(rounds):
            _p["od_wait"] = int(rounds)
            _p.setdefault("od_type", "Iter")
            model = type(model)(**_p)
        model.fit(X_head, y_head, eval_set=(X_es, y_es))
        return model
    return _lgbm_fit_safe(model, X_head, y_head) or model


def _es_rounds_for(model, rounds):
    """早停耐心随学习率缩放(v2.15.0;MODELEVO_ES_LR_SCALE=0 关闭)。"""
    try:
        _scale = float(os.environ.get("MODELEVO_ES_LR_SCALE", "0.5") or 0)
    except ValueError:
        _scale = 0.5
    if _scale <= 0:
        return int(rounds)
    try:
        _p = model.get_params()
        _lr = float(_p.get("learning_rate") or _p.get("eta") or 0)
    except Exception:  # noqa: BLE001
        _lr = 0.0
    if _lr <= 0:
        return int(rounds)
    import math as _math_r

    return max(int(rounds), int(_math_r.ceil(_scale / _lr)))


def _model_cap(model, fam):
    """估计器的轮数上限(n_estimators / catboost iterations)。"""
    try:
        _p = model.get_params()
    except Exception:  # noqa: BLE001
        return None
    _v = _p.get("iterations") if fam == "catboost" else _p.get("n_estimators")
    if _v is None and fam == "catboost":
        _v = _p.get("n_estimators") or _p.get("num_boost_round")
    try:
        return int(_v) if _v else None
    except (TypeError, ValueError):
        return None


def _set_cap(model, fam, cap):
    """改写轮数上限;catboost 拟合后禁止 set_params → 按参数重建。

    [铁律]cap 在写入估计器前统一钳到 _MAX_TREES。调用点有 hpo refit
    的 lr-shrink 外扩(×1/factor)与 es stage1 的 cap 外扩(×factor),
    两者都能把 cap 推过上限。
    钳制用本文件的 _clamp_trees(自包含),不 import runtime —— agent
    改写 train.py 后即使 sys.path 里没有 skill root,上限依然生效。
    """
    cap = _clamp_trees(cap)
    if fam == "catboost":
        _p = dict(model.get_params())
        _p["iterations"] = int(cap)
        _p.pop("n_estimators", None)
        return type(model)(**_p)
    try:
        model.set_params(n_estimators=int(cap))
    except Exception:  # noqa: BLE001
        pass
    return model


def _es_curve_of(model, fam):
    """第一段早停集(held-out)上的逐轮指标序列。"""
    try:
        if fam == "lightgbm":
            ev = getattr(model, "evals_result_", None)
        elif fam == "xgboost":
            ev = model.evals_result()
        elif fam == "catboost":
            ev = model.get_evals_result()
        else:
            return []
    except Exception:  # noqa: BLE001
        return []
    if not ev:
        return []
    keys = [k for k in ev if "valid" in str(k).lower()] or list(ev)
    first = ev[keys[0]] or {}
    if not first:
        return []
    return list(first[next(iter(first))])


def _es_best_trees(model, fam):
    """第一段最优树数(1-based;各库索引口径归一)。"""
    try:
        if fam == "lightgbm":
            _b = getattr(model, "best_iteration_", None)
            return int(_b) if _b else None
        if fam == "xgboost":
            _b = getattr(model, "best_iteration", None)
            return int(_b) + 1 if _b is not None else None
        if fam == "catboost":
            _b = model.get_best_iteration()
            return int(_b) + 1 if _b is not None else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _es_source() -> str:
    """v2.16.0:outer 拟合的早停面。

    - validation(默认):在 outer validation 上早停,100% train 只训一次;
    - train_tail(回退):v2.15 两段式(train 尾部早停 + 全量重拟合)。
    runner 按 run 配置 training.early_stop_source 下发 MODELEVO_ES_SOURCE;
    缺 env(smoke/手工运行)时按默认 validation。
    """
    _v = str(os.environ.get("MODELEVO_ES_SOURCE") or "validation")
    return "train_tail" if _v.strip().lower() == "train_tail" else "validation"


#: [v2.16.0] 最近一次 outer 拟合的早停面 (X_va, y_va);multi-seed 变体复用
_LAST_EVAL_XY = None


def _fit_es_on_validation(model, X_tr, y_tr, X_va, y_va, rounds=None):
    """v2.16.0 单段拟合:100% train 只训一次,早停在 outer validation 上。

    - 复用 _es_stage1(各库原生早停),早停耐心沿用 _es_rounds_for;
    - cap 一次性放到 cap × MODELEVO_CAP_EXTEND_FACTOR ^ MODELEVO_CAP_EXTEND_TIMES
      (钳到 _MAX_TREES):早停会先叫停,大 cap 无额外成本,因此不再"撞 cap
      重跑第一段",保证只训一次;
    - 遥测键与两段式同名(stage1_* / cap / cap_bound / stopped_by),下游
      消费者不变;另记 es_source=validation、fit_mode=single,无 refit_trees;
    - v2.17.2(A1):外扩只在剩余余量内进行(cand = min(_MAX_TREES, …),
      cand > 当前 cap 才外扩);最终拟合撞 cap 且 cap 已达 _MAX_TREES(无余量)
      时 cap_bound_unresolved=True(否则 False)—— 评估侧据此把负 Δ 标为
      cap 混杂,而不是当作方向证伪;
    - 预测走最优轮数:xgboost/lightgbm 的 sklearn API 早停后自动用 best,
      catboost 带 eval_set 时默认 use_best_model;
    - 非 GBM 族(sklearn/torch)直接全量 fit;任何异常回退两段式。
    """
    import math as _math_v

    if rounds is None:
        try:
            rounds = int(os.environ.get("MODELEVO_EARLY_STOP_ROUNDS")
                         or globals().get("EARLY_STOP_ROUNDS") or 50)
        except (TypeError, ValueError):
            rounds = 50
    _fam = str(type(model).__module__ or "").split(".")[0]
    if _fam not in ("lightgbm", "xgboost", "catboost"):
        return _lgbm_fit_safe(model, X_tr, y_tr) or model
    try:
        rounds = _es_rounds_for(model, rounds)
        _cap0 = _model_cap(model, _fam)
        _times = int(os.environ.get("MODELEVO_CAP_EXTEND_TIMES", "1") or 0)
        _factor = float(os.environ.get("MODELEVO_CAP_EXTEND_FACTOR", "2")
                        or 2)
        _cap_ext = None
        if _cap0 and _times > 0 and _factor > 1:
            _cand = _clamp_trees(int(_math_v.ceil(
                int(_cap0) * (_factor ** _times))))
            if _cand > int(_cap0):
                model = _set_cap(model, _fam, _cand)
                _cap_ext = _cand
        info = {"es_rounds": int(rounds), "cap_initial": _cap0,
                "cap_extended_to": _cap_ext, "es_source": "validation",
                "fit_mode": "single", "n_es_rows": int(len(X_va)),
                # v2.17.3:早停集可能被降采样,实际参与早停的行数单列
                # (由 MODELEVO_ES_EVAL_ROWS 决定,与 _es_stage1 同口径)
                "n_es_eval_rows": min(
                    int(len(X_va)),
                    int(os.environ.get("MODELEVO_ES_EVAL_ROWS") or 0)
                    or int(len(X_va)))}
        model = _es_stage1(model, _fam, X_tr, y_tr, X_va, y_va, rounds)
        _curve = _es_curve_of(model, _fam)
        _best = _es_best_trees(model, _fam)
        _cap = _model_cap(model, _fam)
        _bound = bool(_cap and len(_curve) >= int(_cap))
        # v2.17.2(A1):撞 cap 且 cap 已达 _MAX_TREES(无外扩余量)= 截断未解除
        info["cap_bound_unresolved"] = bool(_bound and int(_cap or 0) >= int(
            globals().get("_MAX_TREES") or 8000))
        info.update({"stage1_best_iteration": _best,
                     "stage1_rounds": len(_curve) or None,
                     "cap": _cap, "cap_bound": _bound,
                     "stopped_by": "cap" if _bound else "early_stop",
                     "stage1_curve": [round(float(v), 6) for v in _curve]})
        try:
            model.mvo_es_info_ = info
        except Exception:  # noqa: BLE001
            pass
        return model
    except Exception as _e_v:  # noqa: BLE001
        print(f"[warn] validation 早停单段拟合失败,回退两段式: "
              f"{type(_e_v).__name__}: {_e_v}", flush=True)
        return _fit_with_es(model, X_tr, y_tr, auto_extend=True)


def _fit_with_eval(model, X_tr, y_tr, X_va, y_va):
    # v2.16.0:默认在 outer validation 上早停、100% train 只训一次
    # (_fit_es_on_validation);MODELEVO_ES_SOURCE=train_tail 或拿不到
    # y_va(smoke 等)时回退 v2.15 两段式 _fit_with_es(train 尾部早停 +
    # 全量重拟合,outer 重训开启 cap 自动外扩)。
    # 返回拟合后的模型——catboost 重建路径会换对象,调用方必须
    # 用返回值重绑定(model = _fit_with_eval(...))。
    global _LAST_EVAL_XY
    if X_va is not None and y_va is not None:
        _LAST_EVAL_XY = (X_va, y_va)
        if _es_source() == "validation":
            return _fit_es_on_validation(model, X_tr, y_tr, X_va, y_va)
    return _fit_with_es(model, X_tr, y_tr, auto_extend=True)


def align_predict_device(model) -> None:
    """打分前把 xgboost booster 切回 CPU(v2.12.2.9;与 runtime/baseline.py
    的 _BASELINE_TEMPLATE 同源实现,改动需两处同步)。

    背景:xgboost 在 GPU 上训练(device=cuda),但预测走普通
    `predict_proba(X)`,X 在 CPU → 每次触发
    "Falling back to prediction using DMatrix due to mismatched devices"。
    实测 xgboost 2.1.4 下 set_param({"device":"cpu"}) 前后 predict_proba
    最大绝对差 5.96e-08(CPU/GPU 浮点累加顺序差异),不改变任何一对样本
    的相对次序,故 AUC 口径不变。现状预测本就落在 CPU(隐式回退),本函数
    只是把它变成显式声明并省掉数据搬运。

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
        # 自带 import:不依赖调用方作用域里恰好有 json(exec/嫁接场景下
        # 缺失会 NameError 被 except 吞掉 → 静默失效)
        import json as _json_dev
        cfg = _json_dev.loads(booster.save_config())
        dev = str(((cfg.get("learner") or {}).get("generic_param") or {})
                  .get("device") or "")
        if "cuda" in dev.lower() or "gpu" in dev.lower():
            booster.set_param({"device": "cpu"})
    except Exception:  # noqa: BLE001 — 优化项,失败不阻塞打分
        pass


def _model_score(model, X_va, w_va=None):
    align_predict_device(model)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_va)[:, 1]
    return model.predict(X_va)


def _es_info_of(model) -> dict:
    """[v2.15.0] _fit_with_es 挂在模型上的第一段遥测(去掉逐轮曲线)。"""
    info = dict(getattr(model, "mvo_es_info_", None) or {})
    info.pop("stage1_curve", None)
    return info


def dump_learning_curve(model) -> dict:
    """落盘 learning_curve.csv(iteration, valid_metric),尽力而为。

    v2.15.0(覆盖 baseline 血缘同名函数,经嫁接段生效):优先写
    _fit_with_es **第一段的 held-out 早停曲线**。旧实现读最终模型的
    evals_result —— 两段式拟合的第二段 eval_set 已在训练集里,曲线是
    样本内的、天然单调到最后一轮(run3 HPO trial 曲线末端 logloss
    0.393,而 outer validation 是 0.438),据此判"欠拟合/撞 cap"必错。
    无第一段遥测时(单段早停的 baseline 血缘等)沿用旧口径:LGBM
    evals_result_ → XGB evals_result() → torch history_。
    """
    curve, src = None, "final_model_evals"
    _info = getattr(model, "mvo_es_info_", None) or {}
    if _info.get("stage1_curve"):
        # v2.16.0:单段拟合的早停曲线就是 outer validation 曲线
        curve = list(_info["stage1_curve"])
        src = ("es_validation" if _info.get("es_source") == "validation"
               else "es_stage1_heldout")
    else:
        ev = getattr(model, "evals_result_", None)
        if ev is None and hasattr(model, "evals_result"):
            try:
                ev = model.evals_result()
            except Exception:  # noqa: BLE001
                ev = None
        if ev:
            first = next(iter(ev.values()))
            key = next(iter(first.keys()))
            curve = list(first[key])
        elif isinstance(getattr(model, "history_", None), list):
            curve, src = list(model.history_), "torch_history"
    if not curve:
        return {"learning_curve_rows": 0}
    pd.DataFrame({"iteration": np.arange(1, len(curve) + 1),
                  "valid_metric": curve}).to_csv(
        TRIAL_DIR / "learning_curve.csv", index=False)
    return {"learning_curve_rows": len(curve), "learning_curve_source": src}


def best_iteration_of(model):
    """早停轮次统一口径(1-based);不可得返回 None。

    v2.15.0(覆盖 baseline 血缘同名函数):两段式拟合的最终模型上,
    best_iteration 读到的是第二段(样本内)的末轮;优先取第一段遥测的
    最优树数,与 learning_curve.csv(第一段曲线)同口径 —— 二者之差
    = 早停耐心 ⇒ 正常收敛;差为 0 且曲线行数 = cap ⇒ 撞 cap。
    """
    _info = getattr(model, "mvo_es_info_", None) or {}
    if _info.get("stage1_best_iteration"):
        return int(_info["stage1_best_iteration"])
    bi = getattr(model, "best_iteration_", None)
    if bi is None and hasattr(model, "best_iteration"):
        try:
            bi = int(model.best_iteration)
        except Exception:  # noqa: BLE001
            bi = None
    if bi is None and isinstance(getattr(model, "history_", None), list):
        bi = max(len(model.history_) - 1, 0)  # torch:patience 断点近似
    return (int(bi) + 1) if bi is not None else None


#: [v2.15.0] 主轮 fit_and_score 的重训配方与训练数据(multi-seed 变体复用)
_PRIMARY_FIT_PARAMS = None
_PRIMARY_FIT_XY = None


def _fit_seed_variant(n_jobs, X_fit, y_fit, eval_set):
    """multi-seed 变体轮拟合(v2.15.0,覆盖 baseline 血缘同名函数)。

    嫁接 + 改线后的 baseline 血缘 main() 用本函数训 seed 变体。旧版两处
    缺陷(run3 confirm trial-020/022 实测):
    1. 改线段把 eval_set 置 None,而 xgboost 构造期带早停参数 →
       `Must have at least 1 validation dataset for early stopping`,
       被吞成 warn,seed_scores/ 为空,σ_seed 回退 floor;
    2. 变体走 build_estimator 的**默认超参**,主轮却是 HPO 冠军配方 ——
       量到的是另一个模型的 seed 方差。
    现在:与主轮同配方(_PRIMARY_FIT_PARAMS)、同训练数据
    (_PRIMARY_FIT_XY)、同 _fit_with_eval 早停口径,只换 seed(调用方已
    改写全局 SEED)。v2.16.0:早停面复用主轮记下的 _LAST_EVAL_XY
    (validation 模式下即 outer validation),旧版传 None 会让变体退回
    两段式、与主轮口径不一致。
    """
    _xy = globals().get("_PRIMARY_FIT_XY")
    if _xy is None:
        # v2.20.1:旧实现在这里把 eval_set 拼进训练集再继续。v2.16.0 起默认
        # 早停面是 outer validation,那等于把验证集当训练集 —— 产出的分数虚高
        # 却仍被当成 seed 噪声写进 sigma_seed。该分支只可能被"独立进程里的
        # seed worker"触发(主进程 main() 必先经 fit_and_score 完成赋值),
        # 现由 runtime.seed_parallel 显式下发 _PRIMARY_FIT_PARAMS/_PRIMARY_FIT_XY
        # 填补;仍然取不到就说明调用方没有按该契约传参,必须失败而不是降级。
        raise RuntimeError(
            "seed 变体缺少 _PRIMARY_FIT_XY(主轮未赋值):独立进程 worker 必须"
            "携带主轮配方与训练面;禁止把 eval_set 拼进训练集")
    X_all, y_all = _xy
    _pp_glob = globals().get("_PRIMARY_FIT_PARAMS")
    if _pp_glob is None:
        # 配方缺失同样不可静默退回默认超参(会量到另一个模型的 seed 方差)
        raise RuntimeError(
            "seed 变体缺少 _PRIMARY_FIT_PARAMS(主轮配方未赋值)")
    _params = dict(_pp_glob)
    _seed = int(globals().get("SEED", 0) or 0)
    m = _build_model(_params, _seed, len(X_all))
    _ev = globals().get("_LAST_EVAL_XY")
    if _ev is not None:
        return _fit_with_eval(m, X_all, y_all, _ev[0], _ev[1])
    return _fit_with_eval(m, X_all, y_all, None, None)


def step(msg: str) -> None:
    # 分阶段进度日志(flush 到 stdout,与 harness 心跳互补)
    print(f"[step] {msg} ({_time.time() - _T0:.1f}s)", flush=True)


def prepare_features(X_tr: pd.DataFrame, X_va: pd.DataFrame) -> tuple:
    # dtype 治理(历史教训:parquet decimal/字符串金额列读入为 object,
    # 直接 get_dummies 会按唯一值膨胀成百万哑变量列导致 OOM)。
    # 可数值化 object 列 -> float32;低基数(<=100)类别列 -> one-hot;
    # 其余报错,不做静默猜测。
    cat_cols = []
    for c in X_tr.columns:
        if pd.api.types.is_numeric_dtype(X_tr[c]):
            continue
        conv = pd.to_numeric(X_tr[c], errors="coerce")
        if conv.notna().sum() >= 0.95 * max(X_tr[c].notna().sum(), 1):
            X_tr[c] = conv.astype("float32")
            X_va[c] = pd.to_numeric(X_va[c], errors="coerce").astype("float32")
        elif X_tr[c].nunique() <= 100:
            cat_cols.append(c)
        else:
            raise ValueError(
                f"特征 {c} 无法数值化且基数>100,不适合 one-hot,请检查数据")
    if cat_cols:
        X_tr = pd.get_dummies(X_tr, columns=cat_cols)
        X_va = pd.get_dummies(X_va, columns=cat_cols)
        X_va = X_va.reindex(columns=X_tr.columns, fill_value=0).astype(
            X_tr.dtypes.to_dict())
    return X_tr, X_va


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    """唯一训练+打分路径:smoke 预检与全量共用,输出类 bug 在 smoke
    阶段即可暴露。TODO(Agent): 按 hypothesis 改造本函数(特征/模型/
    超参),保持"train 拟合 → 返回 (score, metrics_extra)"契约,
    不要在 main() 里另写一条训练路径绕过 smoke 覆盖。
    y_va 已入参:GBM early stopping(eval_set)/任何需要验证集标签的
    改造直接用,不必再改签名(历史教训:只改函数体加 eval_set 而签名
    无 y_va,曾导致整代 trial NameError 失败)。"""
    metrics_extra = {"n_features": X_tr.shape[1], "model_num": 1}
    # v16/P0-1:config 超参消费点(快通道 --set 静默失效的根治)。
    # 非 hpo 分支的模型构造统一以 _cfg_hp 覆写默认参——快通道把
    # --set 写进 config.hyperparameters 后,模板保证它真的被读
    # (实测 400w run trial-006:--set 覆写从未生效,"假设未验证"
    # 实为"假设从未被注入")。Agent 改造 _build_model 时保持同一
    # 消费口径(base.update(params) 已是注入点);删除本块会让
    # lint 规则 11 拒绝 run。
    _cfg_hp = {}
    _hp_raw = (cfg or {}).get("hyperparameters")
    if isinstance(_hp_raw, dict):
        _cfg_hp = {k: v for k, v in _hp_raw.items()
                   if isinstance(v, (int, float, str, bool, type(None)))}
    if _cfg_hp:
        step(f"config hyperparameters 生效: {_cfg_hp}")
        metrics_extra["hyperparameters_injected"] = _cfg_hp
    hpo_cfg = (cfg or {}).get("hpo")
    if hpo_cfg:
        # hpo 动作(v11):inner-split 搜索,outer 只评一次。
        # _hpo_fit_inner 由模板注入(见 main() 前的定义),在 train
        # 尾部 inner-validation 上按 space 采样训练;搜索完用
        # best_params 在全量 train 重训(v2.16.0:重训默认在 outer
        # validation 上早停定轮,只训一次)。禁止在 outer validation 上
        # 做任何**超参选择**(lint 规则 7)。
        # v18/P1-4:sample_rows 时 inner 搜索在分层抽样子样本上做
        # (大数据可行性;400w 尺 inner 构建 6217s+单 config 900s 的
        # 教训)。抽样子样本按标签分层(防时间前缀单类化),best_params
        # 仍在全量 train 重训,outer 只评一次。
        from runtime.hpo import run_hpo_search, write_hpo_artifacts

        space = hpo_cfg["search_space"]
        budget = int(hpo_cfg.get("budget_configs", 40))
        frac = float(hpo_cfg.get("inner_fraction", 0.2))
        n = len(X_tr)
        _sample_rows = int(hpo_cfg.get("sample_rows") or 0)
        _fit_pool = _fit_pool_y = None
        if _sample_rows > 0 and n > _sample_rows:
            _fit_pool = X_tr.iloc[:int(n * (1.0 - frac))]
            _fit_pool_y = y_tr.iloc[:int(n * (1.0 - frac))] if hasattr(
                y_tr, "iloc") else y_tr[:int(n * (1.0 - frac))]
            try:
                # v2.17.2(A4):拟合池内按标签分层**随机**抽行(固定 seed,确定
                # 性;行位置升序 = 保持时间序,inner 早停仍取尾部)。旧实现
                # 用分层**头部**(_stratified_head_labels):时间序落盘的
                # train 上等于只取最早几周,搜索面与重训面分布错位。
                from runtime.hpo import stratified_random_rows

                _pos = stratified_random_rows(
                    np.asarray(_fit_pool_y), _sample_rows)
                X_fit = _fit_pool.iloc[_pos]
                y_fit = _fit_pool_y.iloc[_pos] if hasattr(
                    _fit_pool_y, "iloc") else _fit_pool_y[_pos]
                metrics_extra["hpo_sample_rows"] = int(_sample_rows)
                metrics_extra["hpo_sample_method"] = "stratified_random"
                metrics_extra["hpo_inner_rows"] = int(len(X_fit))
                step(f"hpo sample: inner 搜索面 {len(X_fit)} 行"
                     f"(分层随机抽样,{_sample_rows} 上限)")
            except Exception as _e:  # noqa: BLE001
                # [FIX-2026-09-17] 失败不再静默:旧实现在这里只打一句
                # [warn] 就退回全量 inner,`hpo_sample_rows` 不写入
                # metrics —— 用户以为限制了采样,实际烧全量,事后从
                # 产物里查不出(实测 drawdown/t7 gen-03 trial-016:
                # inner 1433s,hpo-sample-rows 120000 完全没生效,只有
                # 日志里一行 warn)。现在把失败事实与原因一并写进
                # metrics,让"抽样是否生效"从产物可验证。
                X_fit, y_fit = _fit_pool, _fit_pool_y
                metrics_extra["hpo_sample_rows"] = 0
                metrics_extra["hpo_sample_fallback"] = (
                    f"{type(_e).__name__}: {_e}")
                metrics_extra["hpo_inner_rows"] = int(len(X_fit))
                print(f"[warn] hpo 抽样失败,退回全量 inner"
                      f"({len(X_fit)} 行;已记入 metrics"
                      f".hpo_sample_fallback): {_e}", flush=True)
        else:
            cut = int(n * (1.0 - frac))
            X_fit, y_fit = X_tr.iloc[:cut], y_tr.iloc[:cut] if hasattr(
                y_tr, "iloc") else y_tr[:cut]
            # [FIX-2026-09-17] 未请求抽样时也落 inner 行数,便于事后
            # 对照"实际搜索面"与预估成本(与抽样分支同口径)。
            metrics_extra["hpo_inner_rows"] = int(len(X_fit))
        X_in, y_in = X_tr.iloc[int(n * (1.0 - frac)):], \
            y_tr.iloc[int(n * (1.0 - frac)):] if hasattr(
                y_tr, "iloc") else y_tr[int(n * (1.0 - frac)):]

        # v2.15.0:lr 模式 / inner 签名 / 热启动 / 并行 / 边界自适应 /
        # 多保真(见 runtime/hpo.py 模块 docstring;run3 三轮 HPO 复盘)
        from runtime.hpo import (cap_param_key, hpo_fidelity_plan,
                                 hpo_parallelism, inner_signature,
                                 load_warm_start, plan_lr_mode,
                                 record_lr_mode)

        _algo = str(globals().get("_HPO_ALGO")
                    or (cfg or {}).get("algorithm") or "").lower()
        _cap_now = int(globals().get("N_ESTIMATORS_CAP")
                       or os.environ.get("MODELEVO_N_ESTIMATORS_CAP") or 2000)
        _esr_now = int(globals().get("EARLY_STOP_ROUNDS")
                       or os.environ.get("MODELEVO_EARLY_STOP_ROUNDS") or 50)
        _lrp = plan_lr_mode(space, hpo_cfg, len(X_fit))
        # v2.16.3:生效 lr 模式必须可见。请求值在 config/trial.yaml 的
        # hpo.lr_mode(创建期另写预判 lr_mode_effective;config 已冻结,运行期
        # 不改),这里把运行期裁定写进 hpo-study.json / metrics(record_lr_mode
        # + 下方 metrics_extra),请求 ≠ 生效时打印原因 —— gen-05 trial-022
        # 配置写 shrink、实跑 search 且无任何提示。
        if _lrp.get("note"):
            print(f"[{'warn' if _lrp.get('warn') else 'info'}] hpo lr_mode: "
                  f"{_lrp['note']}", flush=True)
            step(f"hpo lr_mode requested={_lrp.get('requested_mode')} "
                 f"effective={_lrp['mode']}")
        record_lr_mode(TRIAL_DIR, _lrp)
        _fids = hpo_fidelity_plan(hpo_cfg, len(X_fit))
        _sig = inner_signature(X_fit, y_fit, X_in, y_in, extra={
            "algorithm": _algo, "metric": METRIC_NAME,
            "model_type": MODEL_TYPE, "cap": _cap_now,
            "es_rounds": _esr_now, "lr_mode": _lrp["mode"],
            "search_lr": _lrp.get("search_lr")})
        _warm = [] if _fids else load_warm_start(
            TRIAL_DIR / "config" / "hpo-warm-start.json", _sig)
        _par = hpo_parallelism(hpo_cfg, _algo, _gpu_allocated(len(X_fit)))
        _expand = hpo_cfg.get("auto_expand", True)
        if _expand is True:
            _expand = {"lr_floor": float(hpo_cfg.get("lr_floor") or 0.005)}

        def _inner(p, fidelity=1.0):
            pp = dict(p)
            if _lrp["mode"] == "shrink":
                pp[_lrp["lr_key"]] = _lrp["search_lr"]
            if fidelity is not None and float(fidelity) < 1.0:
                return _hpo_fit_inner(pp, X_fit, y_fit, X_in, y_in, seed,
                                      fidelity=fidelity)
            return _hpo_fit_inner(pp, X_fit, y_fit, X_in, y_in, seed)

        # v2.17.2(A3):deadline 外推预留最终重训时间 —— 创建期成本外推的
        # retrain_sec(trial.yaml expected_resources.hpo_cost_estimate)与搜索
        # 实测外推(拟合秒数中位数 × 全量/inner 行数比 × 轮数比)取大
        from runtime.hpo import hpo_refit_reserve_sec

        _refit_reserve = {
            "static_sec": hpo_refit_reserve_sec(TRIAL_DIR),
            "rows_ratio": float(len(X_tr)) / max(1, len(X_fit)),
            "trees_scale": (float(_lrp.get("refit_cap_scale") or 1.0)
                            if _lrp["mode"] == "shrink" else 1.0),
            "max_trees": int(globals().get("_MAX_TREES") or 8000)}
        step(f"hpo search start: configs={budget} inner_rows={len(X_fit)} "
             f"lr_mode={_lrp['mode']} warm={len(_warm)} parallel={_par} "
             f"fidelity={'sh' if _fids else 'single'}")
        result = run_hpo_search(
            _lrp["space"], budget, _inner,
            metric_direction_up=_INNER_METRIC_UP,
            seed=seed,
            # v2.13:inner 指标名跟随 spec.primary_metric,落盘进
            # hpo-study.json(inner_metric_name),事后可审计"搜的
            # 指标"与"裁决指标"是否一致
            metric_name=METRIC_NAME,
            # v14:runner 注入的绝对 deadline(超时前提前收敛交出当前
            # 最优,而不是被 SIGTERM 杀掉后 39 个已评配置全作废)
            deadline_ts=float(os.environ.get("MODELEVO_DEADLINE_TS") or 0)
            or None,
            checkpoint_path=TRIAL_DIR / "hpo-checkpoint.json",
            warm_start=_warm,
            enqueue=list(hpo_cfg.get("enqueue") or []),
            signature=_sig, n_jobs=_par,
            expand=_expand or None,
            seed_salt=TRIAL_DIR.name,
            fidelities=_fids,
            refit_reserve=_refit_reserve)
        step(f"hpo search done: n={result['n_configs_evaluated']} "
             f"inner_best={result['inner_score']} "
             f"expansions={len(result.get('space_expansions') or [])}")
        # v2.17.2(A4):抽样面搜索的保真复评(opt-in,hpo.fidelity_recheck_top_k)。
        # 抽样生效时把抽样面前 K 名放到**全量 inner 拟合池**上复评,按复评分
        # 选 best_params(与重训同分布);搜索已被 deadline 截断时跳过。
        _rk = int(hpo_cfg.get("fidelity_recheck_top_k") or 0)
        if _rk > 0 and _fit_pool is not None \\
                and metrics_extra.get("hpo_sample_rows") \\
                and not result.get("budget_truncated_by_deadline"):
            from runtime.hpo import fidelity_recheck

            def _inner_full(p):
                pp = dict(p)
                if _lrp["mode"] == "shrink":
                    pp[_lrp["lr_key"]] = _lrp["search_lr"]
                return _hpo_fit_inner(pp, _fit_pool, _fit_pool_y, X_in, y_in,
                                      seed)

            step(f"hpo fidelity recheck: top{_rk} on full inner pool "
                 f"({len(_fit_pool)} rows)")
            _rc = fidelity_recheck(result, _inner_full, top_k=_rk,
                                   metric_direction_up=_INNER_METRIC_UP)
            result["fidelity_recheck"] = _rc
            if _rc.get("best_params"):
                result["inner_score_sampled"] = result["inner_score"]
                result["best_params"] = dict(_rc["best_params"])
                result["inner_score"] = _rc.get("best_full_metric")
                result["best_search_iteration"] = _rc.get("best_iteration")
            metrics_extra["hpo_fidelity_recheck"] = {
                k: _rc.get(k) for k in ("top_k", "n_rechecked", "best_changed",
                                        "best_rank_sampled",
                                        "best_full_metric")}
            step(f"hpo fidelity recheck done: n={_rc.get('n_rechecked')} "
                 f"best_changed={_rc.get('best_changed')}")
        if result.get("deadline_estimate"):
            metrics_extra["hpo_deadline_estimate"] = result["deadline_estimate"]
        result["algorithm"] = _algo
        result["lr_mode"] = _lrp["mode"]
        result["lr_mode_requested"] = _lrp.get("requested_mode")
        result["lr_mode_note"] = _lrp.get("note")
        result["refit_pending"] = True
        # 搜索一结束先落一次(重训若崩,搜索结果不丢);重训后覆写
        write_hpo_artifacts(TRIAL_DIR, result)
        metrics_extra.update({
            "hpo_n_configs": result["n_configs_evaluated"],
            "hpo_sampler": result["sampler"],
            "hpo_inner_score": result["inner_score"],
            "hpo_inner_signature": _sig,
            "hpo_lr_mode": _lrp["mode"],
            # v2.16.3:请求值 vs 生效值并列留痕(shrink 被空间 lr 维顶掉的
            # 形态下两者不同;旧实现只有生效值,事后无法判断用户要的是什么)
            "hpo_lr_mode_effective": _lrp["mode"],
            "hpo_lr_mode_requested": _lrp.get("requested_mode"),
            "hpo_lr_mode_note": _lrp.get("note"),
            "hpo_parallel": _par,
            # v2.19.0:跨卡并行卡数 / 是否被搜索墙钟上限截断(runtime.hpo)
            "hpo_parallel_gpus": result.get("parallel_gpus"),
            "hpo_search_capped": bool(result.get("search_capped")),
            "hpo_warm_start_n": len(_warm),
            "hpo_space_expansions": result.get("space_expansions") or [],
            "hpo_diagnostics": result.get("diagnostics") or {},
        })
        # v14:deadline 截断必须在 metrics 里留痕(40 config 与 15 config
        # 是不同的 study,leaderboard 必须能看到搜索被运维约束截断)
        if result.get("budget_truncated_by_deadline"):
            metrics_extra["hpo_budget_truncated_by_deadline"] = True
            metrics_extra["hpo_budget_requested"] = result.get(
                "budget_requested", budget)
        # v2.13.3:断点续跑留痕。"这次真的重算了多少 config"必须可从产物
        # 读出——否则无法判断 checkpoint 回放是否生效(旧实现只写不读,
        # 40 config 全重算,事后与首次运行不可区分)。
        if result.get("n_configs_replayed"):
            metrics_extra["hpo_n_configs_replayed"] = int(
                result["n_configs_replayed"])
            metrics_extra["hpo_n_configs_fit_this_attempt"] = int(
                result.get("n_configs_fit_this_attempt", 0))
        # v2.13.3:smoke 轮 budget 降到 1(config 预算闸)。留痕使"smoke 的
        # hpo 只跑了 1 个 config"与"真实搜索被截断到 1 个"可区分。
        if result.get("smoke_budget_reduced"):
            metrics_extra["hpo_smoke_budget_reduced"] = True
        # hpo 分支:best_params(搜索结果)为底,config 覆写(如收窄
        # 空间的固定维度)叠加其上——两个来源都真正被消费
        # v2.15.0:lr 收缩模式下重训 lr = search_lr × factor、cap 按
        # 1/factor 放大;最终落盘的 best_params = **重训配方**(含 lr 与
        # 生效的 cap),confirm 快通道与 finalize-retrain 都从它取配方。
        _search_best = dict(result["best_params"] or {})
        _final_params = dict(_search_best)
        _cap_key = cap_param_key(_algo)
        _refit_cap = None
        _refit_plan = None
        if _lrp["mode"] == "shrink":
            # v2.17.2(A1):重训 cap / lr 由搜索期所选 config 的最优轮数推出
            # (runtime.hpo.plan_refit_shrink)。need = ceil(best_iter / factor
            # × 1.2) ≤ _MAX_TREES 时维持收缩;超出时放宽收缩因子(重训 lr =
            # search_lr × factor_eff、cap = _MAX_TREES),不再被树数上限静默
            # 截断(run4 trial-017:3019 轮 → lr 0.0125 需 ~14500 轮,cap 钳到
            # 8000 停在 7999/8000,伪负 Δ −0.00095)。
            # [铁律]cap 仍在写入前钳到 _MAX_TREES(历史事故:8000×1/0.25=32000)。
            from runtime.hpo import plan_refit_shrink, search_best_iteration

            _refit_plan = plan_refit_shrink(
                search_best_iteration(result), search_lr=_lrp["search_lr"],
                factor=_lrp["factor"], cap_now=_cap_now,
                cap_scale=_lrp.get("refit_cap_scale"),
                max_trees=int(globals().get("_MAX_TREES") or 8000))
            _final_params[_lrp["lr_key"]] = _refit_plan["refit_lr"]
            _refit_cap = _clamp_trees(_refit_plan["refit_cap"])
            if _refit_plan["lr_adjusted"]:
                print(f"[warn] lr 收缩重训:搜索期最优 "
                      f"{_refit_plan['search_best_iteration']} 轮 / factor "
                      f"{_refit_plan['factor']:g} × 1.2 需 "
                      f"{_refit_plan['need_trees']} 轮 > 树数上限 "
                      f"{_refit_plan['max_trees']} → 收缩因子放宽为 "
                      f"{_refit_plan['factor_eff']:g}(重训 lr="
                      f"{_refit_plan['refit_lr']:g},cap={_refit_cap})",
                      flush=True)
            elif _refit_plan["basis"] == "unavailable" \\
                    and _refit_plan["cap_clamped"]:
                print(f"[warn] lr 收缩重训:搜索期最优轮数不可得,沿用旧口径 "
                      f"cap={_refit_cap}(cap×1/factor="
                      f"{_refit_plan['cap_wanted_legacy']} 已被树数上限钳制,"
                      "重训可能撞 cap;见 metrics.hpo_refit_cap_clamped)",
                      flush=True)
            metrics_extra.update({
                "hpo_refit_lr_adjusted": bool(_refit_plan["lr_adjusted"]),
                "hpo_refit_factor_eff": float(_refit_plan["factor_eff"]),
                "hpo_refit_cap_clamped": bool(_refit_plan["cap_clamped"])})
            # 生效值随重训配方一起写进 hpo-study.json(record_lr_mode 早先写的
            # 是计划值 search_lr × factor)
            result["lr_mode_refit_lr"] = _refit_plan["refit_lr"]
        _final_params.update(_cfg_hp)
        # config 显式 hyperparameters 也可能给超限树数 —— 同点钳制
        if _cap_key in _final_params:
            _final_params[_cap_key] = _clamp_trees(_final_params[_cap_key])
        model = _build_model(_final_params, seed, len(X_tr))
        if _refit_cap and _cap_key not in _cfg_hp:
            model = _set_cap(model, _algo, _refit_cap)
        if hasattr(model, "fit") and w_tr is None:
            # v2.13:catboost 重建路径会换对象,必须重绑定返回值
            model = _fit_with_eval(model, X_tr, y_tr, X_va, y_va)
        else:
            model.fit(X_tr, y_tr)
        _es = _es_info_of(model)
        _eff_cap = _es.get("cap_extended_to") or _refit_cap
        if _eff_cap and _cap_key not in _cfg_hp:
            _final_params[_cap_key] = int(_eff_cap)
        result["search_best_params"] = _search_best
        result["best_params"] = _final_params
        result["refit"] = {"params": _final_params, "es_info": _es,
                           "refit_cap": _refit_cap,
                           # v2.17.2(A1):生效的重训 lr / cap 与推导依据
                           "lr": _final_params.get(_lrp.get("lr_key")
                                                   or "learning_rate"),
                           "cap": _final_params.get(_cap_key),
                           "plan": _refit_plan}
        result.pop("refit_pending", None)
        write_hpo_artifacts(TRIAL_DIR, result)
        metrics_extra["best_params"] = _final_params
        metrics_extra["hpo_search_best_params"] = _search_best
        metrics_extra["es_info"] = _es
        metrics_extra.setdefault("best_iteration",
                                 _es.get("stage1_best_iteration"))
        metrics_extra.setdefault("n_estimators_used",
                                 _es.get("stage1_rounds"))
        globals()["_PRIMARY_FIT_PARAMS"] = dict(_final_params)
        globals()["_PRIMARY_FIT_XY"] = (X_tr, y_tr)
        step(f"final refit done: trees={_es.get('refit_trees') or _es.get('stage1_best_iteration')} "
             f"stage1_rounds={_es.get('stage1_rounds')} cap={_es.get('cap')} "
             f"cap_bound={_es.get('cap_bound')} es_source={_es.get('es_source', 'train_tail')}")
        score = _model_score(model, X_va, w_tr)
    elif MODEL_TYPE == "uplift":
        # T-learner 脚手架(可执行默认实现,Agent 按 hypothesis 改造):
        # 处置/对照各训一个 LGBM,score = p1 - p0(处置侧 − 对照侧概率)。
        # 双模型差分是 uplift 建模语义(T-learner),不属于铁律 12 的
        # ensemble 融合;hypothesis/rationale 中应声明 T-learner 口径。
        import lightgbm as _lgb
        assert w_tr is not None, "uplift 任务缺少 treatment 列"
        feats_t = list(X_tr.columns)  # v13:特征面物理无 treatment 列
        _params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                       subsample=0.9, colsample_bytree=0.9,
                       n_jobs=n_jobs(),
                       random_state=seed, verbosity=-1)
        _params.update(_cfg_hp)  # v16/P0-1:config 超参覆写
        # [铁律]_cfg_hp 可能给超限树数 —— 钳制后再构造
        if "n_estimators" in _params:
            _params["n_estimators"] = _clamp_trees(_params["n_estimators"])
        # v17.4:有卡且样本足够即 device=cuda;非 GPU 编译轮由
        # _lgbm_fit_safe 在 fit 期回退 CPU(处置/对照双子模型同路径)
        if _gpu_allocated(len(X_tr)) and _lgbm_cuda_ok():
            _params["device"] = "cuda"
        _m_t = _lgbm_fit_safe(_lgb.LGBMClassifier(**_params),
                              X_tr.loc[w_tr == 1, feats_t],
                              y_tr[w_tr == 1])
        _m_c = _lgbm_fit_safe(_lgb.LGBMClassifier(**_params),
                              X_tr.loc[w_tr == 0, feats_t],
                              y_tr[w_tr == 0])
        score = (_m_t.predict_proba(X_va[feats_t])[:, 1]
                 - _m_c.predict_proba(X_va[feats_t])[:, 1])
        # T-learner 交付两个子模型;uplift sanity 量与 baseline 对齐
        metrics_extra["model_num"] = 2
        metrics_extra["treatment_rate"] = float(np.mean(
            np.concatenate([w_tr, w_va])))
        metrics_extra["ate_observed"] = float(
            np.mean(y_tr[w_tr == 1]) - np.mean(y_tr[w_tr == 0]))
        # 模型工件:T-learner 双子模型各存一份(dump_model 只收单模型)
        model = _m_t
    elif globals().get("_HPO_ALGO"):
        # [v2.15.0] 算法化构造路径的非 hpo 分支(confirm / --set 参数子代)。
        # 旧版 if 链只有 hpo / uplift / regression:`--action confirm
        # --from-parent <HPO 冠军>` 剥离 hpo 节后,二分类落进脚手架 raise
        # (run2/run3 共命中 5 次,如 run3 trial-017;每次靠 Agent 首次 run
        # 前手补本分支)。配方 = config.hyperparameters(快通道已从父代
        # metrics.best_params 继承重训配方),机制与 hpo 重训同
        # (_fit_with_eval:v2.16.0 起默认 validation 早停单段拟合)。
        model = _build_model(dict(_cfg_hp), seed, len(X_tr))
        if hasattr(model, "fit") and w_tr is None:
            model = _fit_with_eval(model, X_tr, y_tr, X_va, y_va)
        else:
            model.fit(X_tr, y_tr)
        _es = _es_info_of(model)
        metrics_extra["es_info"] = _es
        metrics_extra.setdefault("best_iteration",
                                 _es.get("stage1_best_iteration"))
        metrics_extra.setdefault("n_estimators_used",
                                 _es.get("stage1_rounds"))
        globals()["_PRIMARY_FIT_PARAMS"] = dict(_cfg_hp)
        globals()["_PRIMARY_FIT_XY"] = (X_tr, y_tr)
        step(f"final refit done: trees={_es.get('refit_trees') or _es.get('stage1_best_iteration')} "
             f"stage1_rounds={_es.get('stage1_rounds')} cap={_es.get('cap')} "
             f"cap_bound={_es.get('cap_bound')} es_source={_es.get('es_source', 'train_tail')}")
        score = _model_score(model, X_va, w_tr)
    elif MODEL_TYPE == "regression":
        # 回归占位实现(DummyRegressor 中位数常数预测,score 列 = 预测值)。
        # v2.17.0(D6):new-trial --algorithm xgboost|lightgbm|catboost 渲染
        # 回归脚手架时 harness 注入 _HPO_ALGO,上方算法化分支构造真实回归器,
        # 本分支不可达;未注入时它只是占位 —— 脚手架标记仍在,lint 规则 10
        # 会拒跑,必须按 hypothesis 实现真实回归器
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor(strategy="median")
        model.fit(X_tr, y_tr)
        score = model.predict(X_va)
    else:
        # v12(P2,O5):默认脚手架直接 raise——DummyClassifier 时代 Agent
        # 未改造的 trial 会静默成功并产出 AUC≈0.5,smoke 检查照样通过,
        # 噪声进 leaderboard/档案。显式失败把 fix_code 轮交给 Agent。
        # [FIX-2026-09-01] 错误信息含 lint 标记,把"训练尾才爆"提前为
        # "run 前静态拦截"(lint_trial 规则 10):fit_and_score 仍保留该
        # raise 时 lint 直接拒绝,Agent 必须实现真实路径后才允许 run。
        raise RuntimeError(
            "脚手架默认实现已禁用(v12):Agent 未按 hypothesis 改造 "
            "fit_and_score;请实现真实模型路径后重跑(分类默认 LGBM,"
            "参考 baseline 模板;回归脚手架用 --algorithm xgboost|lightgbm "
            "渲染即走算法化分支)"
            "  # lint:scaffold-fit-and-score")
    # 模型工件持久化(必须保留):Agent 重写本函数时在训练完成后
    # 调用 dump_model(model, X_tr.columns),保证 model/ 目录落盘
    dump_model(model, X_tr.columns)
    # v15.2:模型输入级 validation 帧落盘(分组置换裁决消费)。fit 中
    # 做了特征加宽(transform 新增列)的 trial **必须**保留本调用——
    # 基础 splits 帧没有新列,harness 侧重放不了 transform,缺失时
    # 加宽族判 unverifiable(方向生死无证据)。
    dump_eval_frame(X_va)
    # v13:训练后泄漏哨兵的模型引用(leak_scan 在 main() 里调用;
    # smoke 轮跳过哨兵,与 dump_model 同口径)
    global _last_model
    _last_model = None if os.environ.get("MODELEVO_SMOKE_RUN") == "1" \\
        else model
    return score, metrics_extra


def dump_eval_frame(X_va) -> None:
    """落盘模型输入级 validation 帧到 TRIAL_DIR/model/validation_input.parquet
    (v15.2,加宽特征 trial 必须,尽力而为不阻塞 trial)。

    分组置换重要性(方向生死裁决)在 harness 侧重放不了 trial 的
    transform——基础 splits 帧没有 transform 新增列。旧行为补 0:
    常数列打乱不掉分,delta≡0,全部加宽族被伪判 dead(实测 gen-02
    回写)。本函数把训练时真实用到的 X_va 落盘,harness 直接消费。
    契约:行序与 validation split 严格一致(行内变换,不 join 不重排);
    smoke 轮与 quick 轮(子样本)跳过,小样本帧不得覆盖全量帧。
    无 transform 的 trial 不必调用(基础帧即模型输入,harness 按基础
    帧降级并如实标注 eval_frame_source)。"""
    import os
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return
    if os.environ.get("MODELEVO_QUICK_RATIO", "0") not in ("", "0"):
        return
    # v2.13.4:multi-seed 变体轮不覆写 primary 工件(见 main 的 multi_seed 段)
    if os.environ.get("MODELEVO_SEED_VARIANT") == "1":
        return
    # v2.13.4 写保护:仅 runner 执行本 trial 时写盘(见 dump_model)
    if os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name:
        print("[guard] dump_eval_frame 拒写: 非 runner 执行上下文"
              "(MODELEVO_TRIAL_EXEC 不匹配),trial 工件只读", flush=True)
        return
    try:
        mdir = TRIAL_DIR / "model"
        mdir.mkdir(parents=True, exist_ok=True)
        X_va.to_parquet(mdir / "validation_input.parquet", index=False)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] dump_eval_frame skipped: {_e}", flush=True)


# leak_scan 引用的最近一次训练模型(fit_and_score 写入)
_last_model = None


SMOKE_MAX_ROWS = 10000
SMOKE_MAX_VALID = 5000


# ===========================================================================
# v2.11 加宽缓存(widen_cached):加宽面 = 确定性行内变换,一次落盘多次复用
# ---------------------------------------------------------------------------
# 实测(v2.10.1 run drawdown-req-t7-v2101):超级 trial 的 _widen 逐行
# 解析 246k 行 seq 串 + 七族派生,单次 830~1000s,而 GBM 训练只要
# 48~74s。同一张加宽面被反复重算:smoke 1 次 + 全量 1 次 + confirm 的
# multi_seed 3 次 + shadow 重放,一个 run 合计 13~15 次全量 widen
# ≈ 3.2h,占全部 compute 55%+。widen 是确定性的(transform 只依赖
# 输入帧与冻结常量,分群统计仅 train 拟合),因此可以缓存:
#   - 进程内 memo:同进程的 smoke→全量→multi_seed→shadow 重放直接复用;
#   - 落盘缓存:RUN_DIR/.widen-cache/<key>.parquet,断点续跑与
#     finalize-retrain 跨进程复用(key = face_hash,口径变更即失配)。
# 用法(改造 _widen 时):把"裸 _widen(raw, re, raw, re)"换成
# "widen_cached('tag', raw_tr, re_tr, raw_va, re_va)"——tag 是本 trial
# 内同一口径的稳定标签(如 'main');口径任何一项变化时必须换 tag。
# smoke/quick 轮传 subsample_rows(小样本),与全量缓存隔离不互踩。
# ===========================================================================
_WIDEN_MEMO: dict = {}


def _widen_face_key(tag: str, tr_rows: int, tr_cols,
                    va_rows: int = -1, va_cols=(), re_cols=()) -> str:
    """缓存键:trial 级口径指纹 = tag + **两侧**帧指纹(行数 + 列集)。

    v2.11.6(gen-02 事故回写):旧实现只哈希**第一侧**(raw_tr)帧,
    而 shadow 重放的调用形态是「train 帧当 raw_tr、shadow 帧当 raw_va」
    ——第一侧与训练轮完全相同(同 33862 行/同列集),键碰撞命中训练
    缓存,返回的是 **validation 侧** 6624 行加宽帧。随后对该帧打分却
    与 6578 行 shadow 标签配对 → 交付链第二把尺子静默污染(实测
    drawdown-req-t7-v211:gen-01/gen-02 共 7 个加宽 trial 的
    shadow_test_score.parquet 逐位等于 validation 分数副本,而
    shadow_test.passed 报「评测失败」被当成 split 缺标签的假象)。

    两侧指纹都必须入键:train 侧 (33862, cols) 与 shadow 侧
    (6578, shadow_cols) 天然不同键,必不碰撞。行内容变化(重切分/
    换视图)时行数即不同——key 不匹配 = 缓存失配,重算并覆盖
    (数据面以 splits 为准,物理上不会被旧缓存毒化)。

    v2.12.2.5(drawdown-t7-sv24 实测事故回写):键里**必须含 RUN_DIR**。
    旧实现只有 `TRIAL_DIR.name`,而缓存目录是 run 级的
    `RUN_DIR/.widen-cache`——同一数据面复制出的多个 run 目录里,同名
    trial 名 + 同 tag + 同帧指纹会算出**同一个 key**。实测
    example-seq-v3-v2.12.2.4 从 .3 硬链接继承了前序 run(.2)的 14 组
    缓存(8.2GB):两组 stats 字段形状不同(前序 run 写 `plane`/`desc`/
    `plane_want`,本 run 写 `bundle`),说明格式独立,却共用同一 key
    空间。一旦新 run 复用同名 trial 名并撞 tag,`widen_cached` 会静默
    返回**别的 run 实现的列**(本 run 与 .2 的 J 面分别是 1503 / 1412
    新列,下游 AUC 与归因全部失准且无任何告警)。
    RUN_DIR 入键后跨 run 天然不碰撞,run 内预材料化(adhoc 与 trial 同
    RUN_DIR)仍然命中,不损失缓存收益。
    """
    import hashlib as _hl
    cols_sig = ",".join(map(str, tr_cols))
    va_sig = ",".join(map(str, va_cols))
    re_sig = ",".join(map(str, re_cols))
    _run_sig = str(RUN_DIR.resolve())
    # v2.13.5:实现指纹替代 trial 名(跨 trial 复用);解析源列与数据
    # 视图目录一并入键(同实现、不同解析列/视图不得互相命中)
    return _hl.sha256(
        f"{_run_sig}|{_widen_impl_sig()}|{globals().get('SPLITS', '')}|{tag}"
        f"|{tr_rows}|{cols_sig}"
        f"|{va_rows}|{va_sig}|{re_sig}".encode()
    ).hexdigest()[:16]


def _widen_impl_sig() -> str:
    """v2.13.5:加宽实现指纹 —— 取代 TRIAL_DIR.name 进键。

    旧键含 trial 名,同一份 `_widen` 在每个派生 trial 里都失配重算
    (req-t7-rand400w:seqd 块被逐字相同地解析 9 次)。改为 `_widen` 的
    代码指纹(runtime.feature_cache.code_fingerprint:字节码 + 常量 +
    递归引用的同模块函数/全局常量):同实现跨 trial 命中,实现任何改动
    即失配。runtime 不可导入(极旧工作区)时退回 trial 名,行为同 v2.12。
    """
    try:
        from runtime.feature_cache import code_fingerprint as _cfp
        return "code:" + _cfp(globals()["_widen"])
    except Exception:  # noqa: BLE001
        return "trial:" + TRIAL_DIR.name


def widen_cached(tag: str, raw_tr, re_tr, raw_va, re_va,
                 subsample_rows: int = 0, cache_dir=None):
    """_widen 的缓存包装:命中即返回 (X_tr, X_va, stats, info),
    未命中调用真实 _widen 后写缓存。

    - 进程内 memo 优先(同进程 smoke→全量→multi_seed 复用,零 IO);
    - 落盘缓存次之(parquet ×2 + json stats),跨进程(断点续跑/
      finalize-retrain)复用;
    - MODELEVO_WIDEN_CACHE=0 一键关闭(诊断场景);
    - smoke/quick 轮(subsample_rows>0)只进 memo 不落盘——小样本帧
      不得覆盖全量缓存(与 dump_eval_frame 同纪律);
    - 缓存读写失败一律降级直算(尽力而为,不阻塞 trial);
    - v2.11.6:键含**两侧**帧指纹(raw_va 行数+列集)。旧实现只看
      raw_tr,shadow 重放(把 shadow 帧当 raw_va 传)与训练轮键碰撞,
      命中训练缓存拿到 validation 帧 → shadow 分数与标签错配。
    """
    import os as _os
    # v2.13.5:动态取 trial 定义的 `_widen`。静态引用让每个未定义 _widen 的
    # 模板渲染 trial 都被 lint 规则 19(生成代码自洽性)判"未定义名"拒跑
    # (tests/test_e2e_smoke.py 在 v2.13.3 即以此失败)。
    _widen_fn = globals().get("_widen")
    if not callable(_widen_fn):
        raise NameError(
            "widen_cached 需要 trial 定义 _widen(raw_tr, re_tr, raw_va, re_va)"
            " -> (X_tr, X_va, stats, info);无训练期状态的行内变换请改用 "
            "trial_feature_blocks(见 runtime/feature_cache.py)")
    if str(_os.environ.get("MODELEVO_WIDEN_CACHE", "1")) == "0":
        return _widen_fn(raw_tr, re_tr, raw_va, re_va)
    key = _widen_face_key(
        tag, len(raw_tr), list(raw_tr.columns),
        -1 if raw_va is None else len(raw_va),
        () if raw_va is None else list(raw_va.columns),
        () if re_tr is None else list(getattr(re_tr, "columns", ())))
    hit = _WIDEN_MEMO.get(key)
    if hit is not None:
        step(f"widen cache hit(memo): tag={tag} rows={len(raw_tr)}")
        return hit
    cdir = (pathlib.Path(cache_dir) if cache_dir
            else RUN_DIR / ".widen-cache")
    on_disk_ok = (not subsample_rows)
    ftr = cdir / f"{key}.tr.parquet"
    fva = cdir / f"{key}.va.parquet"
    fst = cdir / f"{key}.stats.json"
    if on_disk_ok and ftr.exists() and fva.exists():
        try:
            X_tr = pd.read_parquet(ftr)
            X_va = pd.read_parquet(fva)
            import json as _json
            stats = _json.loads(fst.read_text(encoding="utf-8"))
            info = {"widen_cache": "disk", "widen_cache_key": key}
            # v2.12.2.5:命中留痕来源 run。旧缓存(键不含 RUN_DIR)或跨 run
            # 硬链接继承的文件在本 run 里没有 writer 记录,一旦被命中就是
            # 「拿到了别的 run 实现的列」;此处显式打印来源标记,让审计
            # 能在日志里发现异常来源,而不是只有一行 'hit(disk)'。
            _src = stats.get("_run_dir")
            if _src is None:
                print("[warn] widen cache hit(disk) 命中无来源标记的旧缓存 "
                      f"(key={key} tag={tag});该文件可能由其它 run 写入,"
                      "列实现未必与本 run 一致 —— 核对 stats 后再采信",
                      flush=True)
            elif str(_src) != str(RUN_DIR):
                print("[warn] widen cache hit(disk) 跨 run 命中: 文件来源 "
                      f"{_src} != 本 run {RUN_DIR}(key={key} tag={tag})",
                      flush=True)
            step(f"widen cache hit(disk): tag={tag} rows={len(raw_tr)} "
                 f"cols={X_tr.shape[1]}")
            out = (X_tr, X_va, stats, info)
            _WIDEN_MEMO[key] = out
            return out
        except Exception as _e:  # noqa: BLE001 缓存损坏 → 直算
            print(f"[warn] widen cache 读取失败,直算: {_e}", flush=True)
    X_tr, X_va, stats, info = _widen_fn(raw_tr, re_tr, raw_va, re_va)
    _WIDEN_MEMO[key] = (X_tr, X_va, stats, info)
    if on_disk_ok:
        try:
            cdir.mkdir(parents=True, exist_ok=True)
            X_tr.to_parquet(ftr, index=False)
            X_va.to_parquet(fva, index=False)
            import json as _json
            # v2.12.2.5:来源标记入 stats。缓存目录是 run 级的,但同名
            # trial 名 + 同 tag + 同帧指纹在**不同 run 目录间**会算出同一
            # key;写入来源 run 后可让命中侧识别"这不是本 run 算的列"。
            _stats_out = dict(stats) if isinstance(stats, dict) else {}
            _stats_out["_run_dir"] = str(RUN_DIR)
            fst.write_text(_json.dumps(_stats_out, ensure_ascii=False),
                           encoding="utf-8")
            info = dict(info)
            info["widen_cache"] = "written"
            info["widen_cache_key"] = key
            step(f"widen cache written: tag={tag} rows={len(raw_tr)} "
                 f"cols={X_tr.shape[1]}")
        except Exception as _e:  # noqa: BLE001
            print(f"[warn] widen cache 写盘失败(不影响本轮): {_e}",
                  flush=True)
    return X_tr, X_va, stats, info


# ===========================================================================
# v2.13.5 特征块(trial_feature_blocks):跨 trial 内容寻址缓存 + 并行解析
# ---------------------------------------------------------------------------
# 无训练期拟合状态的行内变换(seq 串解析、JSON 展开、计数/近因族等)
# 用本契约,不要手写逐行循环进 main():
#
#     from runtime.feature_cache import feature_block
#
#     def _seq_kernel(b):            # b = {"seq_d365": [...], ...}(python list)
#         return [...]               # (len(b[...]), len(_SEQ_COLS)) 数值矩阵
#
#     def trial_feature_blocks(split: str) -> pd.DataFrame:
#         return feature_block("seqd", _seq_kernel, split=split,
#                              columns=_SEQ_COLS,
#                              sources=["seq_d365", "seq_gap_d365"],
#                              splits_dir=SPLITS, n_jobs=n_jobs())
#
# 定义了 trial_feature_blocks 后,main()/smoke/shadow 重放/finalize-retrain
# 交付重放/final_test 打分全部自动右接同一块(本函数 _append_feature_blocks),
# 同一内核在任意 trial 与交付链之间只算一次(RUN_DIR/.feature-cache/),
# 首次计算按 trial 线程配额多进程并行(实测 44.7 万行 seq 串 200s→21s)。
# 有训练期拟合状态(分群统计、词表选择)的变换仍走 widen_cached。
# ===========================================================================
def _append_feature_blocks(X, split):
    """trial 定义了 trial_feature_blocks(split) 时右接特征块;否则原样返回。"""
    fb = globals().get("trial_feature_blocks")
    if not callable(fb):
        return X
    from runtime.feature_cache import align_rows
    W = fb(split)
    dup = [c for c in W.columns if c in X.columns]
    if dup:
        raise ValueError(f"特征块列与基础面重名(须带解析前缀): {dup[:5]}")
    W = align_rows(W, X).astype("float32")
    step(f"feature blocks appended({split}): +{W.shape[1]} cols")
    return pd.concat([X, W], axis=1)


def _stratified_head_labels(y: pd.Series, k: int) -> pd.Index:
    """v17.1:按标签分层取 k 行(前缀采样的标签偏差防护)。

    背景(drawdown-t7 实测):labels 落盘前按时间排序,但上游 loader
    的天内行序按 label 聚集——validation 前缀 5000 行 ≈ 73%~100%
    正样本(全体 33%)。smoke 的 AUC/形状断言在这种前缀上失去意义,
    甚至恒真(全正样本时断言不触发)。分层头部:两类各按行序取
    ceil(k*rate) 行再合并排序,保持"靠近原序"且类分布与全体一致。
    回归/多类标签退化时回退普通头部。
    """
    if k >= len(y):
        return y.index[:k]
    vc = y.value_counts()
    if len(vc) < 2 or not (vc > 0).all():
        return y.index[:k]
    n = len(y)
    idx_parts = []
    for val, cnt in vc.items():
        take = max(1, int(round(k * cnt / n)))
        idx_parts.append(y.index[y == val][:take])
    sel = idx_parts[0].append(idx_parts[1:])
    sel = sel.sort_values()  # 保持接近原序
    if len(sel) > k:
        sel = sel[:k]
    return sel


def run_smoke(feats, seed, cfg):
    """smoke 预检:train/valid 各取小样本,走完整 fit_and_score +
    产物落盘路径,几十秒内暴露输出类 bug(列名错/NaN/model_num 缺/
    torch 形状不匹配)——这类 bug 若在全量训练尾部才爆,harness 的
    fix_code 轮重跑代价是全量重训。产物只写 smoke/ 子目录,绝不碰
    validation_score.parquet(smoke 分数无评估价值,不得进入反馈环)。
    纯超参 trial(代码路径与父代一致)可设 env MODELEVO_SMOKE=0 跳过。

    v17.1:子样本从"时间序前缀"改为**标签分层行序头**(见
    _stratified_head_labels)——前缀在"天内 label 聚集"的落盘序上
    近乎单类,smoke 的 AUC 语义失效(drawdown-t7 实测 valid 前缀
    100% 正)。smoke 只用于抓机械 bug,不用其 AUC 判断方向。
    """
    os.environ["MODELEVO_SMOKE_RUN"] = "1"  # smoke 轮跳过模型落盘
    train_s = pd.read_parquet(SPLITS / "train.parquet", columns=feats)
    valid_s = pd.read_parquet(SPLITS / "validation.parquet", columns=feats)
    # v21:标签从标签面取(四平面隔离);v17.1:分层头防标签聚集前缀
    labels_tr_full = load_labels("train")
    labels_va_full = load_labels("validation")
    tr_sel = _stratified_head_labels(labels_tr_full[TARGET],
                                     SMOKE_MAX_ROWS)
    va_sel = _stratified_head_labels(labels_va_full[TARGET],
                                     SMOKE_MAX_VALID)
    labels_tr_s = labels_tr_full.loc[tr_sel]
    labels_va_s = labels_va_full.loc[va_sel]
    train_s = train_s.loc[tr_sel]
    valid_s = valid_s.loc[va_sel]
    X_tr, X_va = prepare_features(train_s[feats], valid_s[feats])
    X_tr = _append_feature_blocks(X_tr, "train")
    X_va = _append_feature_blocks(X_va, "validation")
    w_tr = (labels_tr_s[TREATMENT].values
            if TREATMENT and TREATMENT in labels_tr_s else None)
    w_va = (labels_va_s[TREATMENT].values
            if TREATMENT and TREATMENT in labels_va_s else None)
    score, _ = fit_and_score(X_tr, labels_tr_s[TARGET], X_va,
                             labels_va_s[TARGET], seed, cfg,
                             w_tr=w_tr, w_va=w_va)
    assert len(score) == len(valid_s), (
        f"smoke: score 行数 {len(score)} != valid {len(valid_s)}")
    arr = np.asarray(score, dtype="float64")
    assert np.isfinite(arr).all(), "smoke: score 含 NaN/inf"
    smoke_dir = TRIAL_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"score": arr}).to_parquet(
        smoke_dir / "validation_score.parquet", index=False)
    step(f"smoke ok: train={len(train_s)} valid={len(valid_s)} "
         f"score_finite=True")
    os.environ.pop("MODELEVO_SMOKE_RUN", None)


# ===========================================================================
# v2.17.2(D2):confirm seed 变体并行
# ---------------------------------------------------------------------------
# run4 req-t7-400w:confirm 的 3 个 seed 串行(主轮 + 2 个变体)≈3.4 h 拖整代。
# 变体之间互不依赖:主进程在本卡训一个变体,同时拉起子进程 worker(重新执行
# 本脚本,MODELEVO_SEED_WORKER_K=k,只训变体 k)在**额外租到的卡**上训其余
# 变体。每个变体与串行完全同口径(同一 fit_and_score / 同一配方 / seed =
# seed + 1000×k),各写 seed_scores/seed-k.parquet —— 文件名即合并键,合并
# 确定性与串行一致。并行的前提(缺一即串行,原因留痕):
#   - confirm.parallel_seeds != off(run 配置;env MODELEVO_CONFIRM_PARALLEL_SEEDS 优先);
#   - 本 trial 在 GPU 上(MODELEVO_GPU=1),且经 runtime.scheduler 的 GPU 租约
#     (<run>/.gpu-leases,跨进程排他)真能再租到无外来进程的干净卡;
#   - v2.17 内存准入:有效可用 − 本 run 未兑现预留 − 安全余量 ≥ 额外 worker 数 ×
#     单 worker 峰值估计(父代实测峰值 ×1.15 与本进程当前占用 ×1.15 取大),
#     并为额外 worker 落一份准入预留(logs/admission.jsonl)。
# worker 失败 / 缺文件的变体由主进程串行补齐;留痕 metrics.multi_seed_parallel。
# ===========================================================================
_SEED_WORKER_ENV = "MODELEVO_SEED_WORKER_K"


def _confirm_parallel_setting() -> str:
    """confirm.parallel_seeds:auto | off(env 优先,其次 run 配置,缺省 auto)。"""
    _v = os.environ.get("MODELEVO_CONFIRM_PARALLEL_SEEDS")
    if not _v:
        try:
            from runtime.config import Config as _Cfg_ps

            _v = _Cfg_ps.load(run_config=RUN_DIR / "modelevo-config.yaml").get(
                "confirm.parallel_seeds", "auto")
        except Exception:  # noqa: BLE001
            _v = "auto"
    _v = str("auto" if _v is None else _v).strip().lower()
    return "off" if _v in ("off", "false", "0", "no", "serial") else "auto"


def _seed_parallel_plan(n_todo: int) -> dict:
    """confirm seed 变体并行计划(见段首)。返回 mode=parallel|serial、reason、
    slots=[{"gpu": 卡号}](每个额外 worker 一项)、n_parallel;并行时附
    _leases / _ticket,由 _seed_parallel_release 释放。"""
    rec = {"mode": "serial", "reason": None, "slots": [], "n_parallel": 1}
    if n_todo < 2:
        rec["reason"] = "single_variant"
        return rec
    if _confirm_parallel_setting() == "off":
        rec["reason"] = "config_off"
        return rec
    if os.environ.get("MODELEVO_GPU") != "1":
        rec["reason"] = "no_gpu_allocated"
        return rec
    try:
        from runtime.config import Config as _Cfg_ps
        from runtime.scheduler import (PEAK_SAFETY_FACTOR, GpuLease,
                                       MemoryAdmission, annotate_gpu_occupancy,
                                       estimate_trial_peak_gb,
                                       gpu_candidates_by_occupancy,
                                       query_gpu_occupancy)
    except Exception as _e_ps:  # noqa: BLE001
        rec["reason"] = f"runtime_unavailable: {type(_e_ps).__name__}"
        return rec
    _own = {s.strip() for s in os.environ.get(
        "CUDA_VISIBLE_DEVICES", "").split(",") if s.strip()}
    _occ = query_gpu_occupancy()
    if not _occ:
        rec["reason"] = "gpu_query_failed"
        return rec
    _clean, _busy = gpu_candidates_by_occupancy(
        annotate_gpu_occupancy(_occ, RUN_DIR))
    _cands = [g for g in _clean if str(g) not in _own]
    if not _cands:
        rec["reason"] = "no_extra_gpu"
        return rec
    _cfg_run = _Cfg_ps.load(run_config=RUN_DIR / "modelevo-config.yaml")
    try:
        import yaml as _yaml_ps

        _td = _yaml_ps.safe_load((TRIAL_DIR / "trial.yaml").read_text(
            encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        _td = {}
    _est = estimate_trial_peak_gb(RUN_DIR, _td, _cfg_run)
    _per = float(_est.get("estimate_gb") or 0.0)
    try:
        import psutil as _psu_ps

        _pp = _psu_ps.Process(os.getpid())
        _rss = sum(q.memory_info().rss for q in
                   [_pp] + _pp.children(recursive=True)) / float(2 ** 30)
        _per = max(_per, _rss * float(PEAK_SAFETY_FACTOR))
    except Exception:  # noqa: BLE001
        pass
    _adm = MemoryAdmission(RUN_DIR, _cfg_run)
    _ev = _adm.evaluate(_per)
    _head = _ev.get("headroom_gb")
    rec.update(per_worker_gb=round(_per, 2), headroom_gb=_head,
               estimate_source=_est.get("source"), gpu_candidates=_cands)
    if _head is None:
        rec["reason"] = "no_memory_probe"
        return rec
    _n_mem = int(float(_head) // _per) if _per > 0 else 0
    _n_extra = min(int(n_todo) - 1, len(_cands), _n_mem)
    if _n_extra < 1:
        rec["reason"] = "insufficient_memory"
        return rec
    _leases = []
    for _g in _cands:
        if len(_leases) >= _n_extra:
            break
        _l = GpuLease.acquire(RUN_DIR / ".gpu-leases", int(_g))
        if _l is not None:
            _leases.append(_l)
    if not _leases:
        rec["reason"] = "gpu_lease_unavailable"
        return rec
    _ticket = None
    if _adm.enabled:
        try:
            _ticket = _adm.acquire(
                f"{TRIAL_DIR.name}#seed-par",
                {"estimate_gb": _per * len(_leases),
                 "source": "confirm_parallel_seeds",
                 "basis_trial": _est.get("basis_trial")}, max_wait_sec=0)
        except Exception:  # noqa: BLE001
            _ticket = None
        if _ticket is None or not _ticket.admitted:
            for _l in _leases:
                _l.release()
            rec["reason"] = "admission_rejected"
            return rec
    rec.update(mode="parallel", reason="ok",
               slots=[{"gpu": _l.gpu_index} for _l in _leases],
               n_parallel=1 + len(_leases), _leases=_leases, _ticket=_ticket)
    return rec


def _seed_parallel_release(plan: dict) -> None:
    for _l in plan.pop("_leases", None) or []:
        try:
            _l.release()
        except Exception:  # noqa: BLE001
            pass
    _t = plan.pop("_ticket", None)
    if _t is not None:
        try:
            _t.release()
        except Exception:  # noqa: BLE001
            pass


def _seed_worker_run(X_tr, y_tr, X_va, y_va, seed, w_tr=None, w_va=None):
    """worker 模式:只训变体 k(配方取主进程落盘的 cfg),原子写 seed-k.parquet。"""
    import json as _json_sw
    import pathlib as _pl_sw

    _k = int(os.environ[_SEED_WORKER_ENV])
    _cfg_v = _json_sw.loads(_pl_sw.Path(
        os.environ["MODELEVO_SEED_WORKER_CFG"]).read_text(encoding="utf-8"))
    os.environ["MODELEVO_SEED_VARIANT"] = "1"
    _sc, _ = fit_and_score(X_tr, y_tr, X_va, y_va, seed + 1000 * _k, _cfg_v,
                           w_tr=w_tr, w_va=w_va)
    _out = _pl_sw.Path(TRIAL_DIR) / "seed_scores" / f"seed-{_k}.parquet"
    _tmp = _out.with_name(f".{_out.name}.tmp-{os.getpid()}")
    pd.DataFrame({"score": _sc}).to_parquet(_tmp, index=False)
    os.replace(_tmp, _out)
    step(f"seed worker: variant {_k} done → {_out.name}")


def _run_seed_variants_parallel(todo, cfg_v, seed, X_tr, y_tr, X_va, y_va,
                                w_tr, w_va, sd) -> dict:
    """按 _seed_parallel_plan 并行训 seed 变体:主进程训 todo[0],worker 训
    其余(每个额外租约卡一个)。返回留痕(不含租约对象);未覆盖 / 失败的变体
    由调用方串行补齐。"""
    import json as _json_sp
    import pathlib as _pl_sp
    import subprocess as _sp_sp
    import sys as _sys_sp

    plan = _seed_parallel_plan(len(todo))
    rec = {k: v for k, v in plan.items() if not k.startswith("_")}
    if plan["mode"] != "parallel":
        if rec.get("reason") != "single_variant":
            step(f"multi-seed 串行(并行条件不满足: {rec.get('reason')})")
        return rec
    procs = []
    try:
        _cfg_path = _pl_sp.Path(sd) / ".seed-worker-cfg.json"
        _cfg_path.write_text(_json_sp.dumps(cfg_v, ensure_ascii=False,
                                            default=str), encoding="utf-8")
        _script = _pl_sp.Path(__file__).resolve()
        for _k, _slot in zip(todo[1:], plan["slots"]):
            _env = dict(os.environ)
            _env.update({_SEED_WORKER_ENV: str(_k),
                         "MODELEVO_SEED_WORKER_CFG": str(_cfg_path),
                         "MODELEVO_SEED_VARIANT": "1", "MODELEVO_SMOKE": "0"})
            if _slot.get("gpu") is not None:
                _env["CUDA_VISIBLE_DEVICES"] = str(_slot["gpu"])
                _env["MODELEVO_GPU"] = "1"
            _log = open(_pl_sp.Path(sd) / f".seed-worker-{_k}.log", "wb")
            procs.append((_k, _slot, _sp_sp.Popen(
                [_sys_sp.executable, str(_script)], cwd=str(_script.parent),
                env=_env, stdout=_log, stderr=_sp_sp.STDOUT), _log))
        step(f"multi-seed 并行: 主进程训 seed-{todo[0]},worker "
             f"{[(k, s.get('gpu')) for k, s, _, _ in procs]}")
        _own_k = todo[0]
        _sc, _ = fit_and_score(X_tr, y_tr, X_va, y_va, seed + 1000 * _own_k,
                               cfg_v, w_tr=w_tr, w_va=w_va)
        pd.DataFrame({"score": _sc}).to_parquet(
            _pl_sp.Path(sd) / f"seed-{_own_k}.parquet", index=False)
        rec["workers"] = {}
        for _k, _slot, _p, _log in procs:
            _rc = _p.wait()
            rec["workers"][str(_k)] = {
                "gpu": _slot.get("gpu"), "rc": _rc,
                "ok": bool(_rc == 0 and (_pl_sp.Path(sd)
                                         / f"seed-{_k}.parquet").exists())}
    finally:
        for _k, _slot, _p, _log in procs:
            if _p.poll() is None:
                _p.kill()
                _p.wait()
            _log.close()
            # 成功的 worker 日志不留;失败的留给排查(串行补齐照常进行)
            if ((rec.get("workers") or {}).get(str(_k)) or {}).get("ok"):
                try:
                    _pl_sp.Path(_log.name).unlink()
                except OSError:
                    pass
        try:
            (_pl_sp.Path(sd) / ".seed-worker-cfg.json").unlink()
        except OSError:
            pass
        _seed_parallel_release(plan)
    return rec


def main() -> None:
    cfg = load_config()
    seed = int(cfg.get("seed", SEED))
    # v17.4:先定 CUDA 设备再跑 smoke——smoke 与全量共用同一 device 路径
    # (模板模型构造读 MODELEVO_GPU,此处只为让 smoke 里的 CUDA context
    # 与全量一致;scheduler 已注入 MODELEVO_GPU,无卡时 pick_gpu 不动)。
    pick_gpu()
    train = pd.read_parquet(SPLITS / "train.parquet")
    valid = pd.read_parquet(SPLITS / "validation.parquet")
    # v21:标签从标签面取(四平面隔离;特征面无目标列)
    labels_tr = load_labels("train")
    labels_va = load_labels("validation")
    feats = exclude_columns(train, cfg)
    # successive halving quick 轮:harness 经 MODELEVO_QUICK_RATIO 注入
    # 子样本比例。v17.1:从"时间序前缀"改为**标签分层行序头**——
    # 天内 label 聚集的落盘序上前缀近乎单类(drawdown-t7 实测),
    # quick 轮的排序语义会失真;不识别时等价全量。
    _quick = float(os.environ.get("MODELEVO_QUICK_RATIO", "0") or 0)
    if 0 < _quick < 1:
        _sel = _stratified_head_labels(
            labels_tr[TARGET], int(len(train) * _quick))
        train = train.loc[_sel]
        labels_tr = labels_tr.loc[_sel]
    # smoke 预检(quick 轮本身已是子样本,跳过):全量训练前用小样本
    # 走一遍完整代码路径;MODELEVO_SMOKE=0 可显式跳过
    if (not _quick
            and os.environ.get("MODELEVO_SMOKE", "1") != "0"
            and len(train) > SMOKE_MAX_ROWS):
        run_smoke(feats, seed, cfg)
    X_tr, X_va = prepare_features(train[feats], valid[feats])
    _n_base_cols = X_tr.shape[1]
    X_tr = _append_feature_blocks(X_tr, "train")
    X_va = _append_feature_blocks(X_va, "validation")
    y_tr = labels_tr[TARGET]
    y_va = labels_va[TARGET]
    w_tr = (labels_tr[TREATMENT].values
            if TREATMENT and TREATMENT in labels_tr else None)
    w_va = (labels_va[TREATMENT].values
            if TREATMENT and TREATMENT in labels_va else None)
    del train, valid
    # v13:fit 前终检(banned 列任何形态不得进模型;数据面物理隔离
    # 之上的第二道断言,把绕行重建提前到训练前报错)
    assert_banned_clean(X_tr)
    # v2.17.2(D2):confirm seed 变体 worker(主进程 _run_seed_variants_parallel
    # 拉起;数据面与主轮同一路径重建,只训一个 seed 变体后退出)
    if os.environ.get(_SEED_WORKER_ENV):
        _seed_worker_run(X_tr, y_tr, X_va, y_va, seed, w_tr, w_va)
        return
    # TODO(Agent): 按 hypothesis 实现特征/样本/模型改动;
    # 超参优先从 config(trial.yaml)读,代码内只放默认值
    metrics_extra = {"config_used": bool(cfg)}
    # v17.4:GPU 运行留痕(区分 CPU/GPU 口径,反思与协议审计用)
    metrics_extra["gpu_allocated"] = int(
        os.environ.get("MODELEVO_GPU", "0") == "1")
    if _quick:
        metrics_extra["quick_ratio"] = _quick
    if X_tr.shape[1] > _n_base_cols:
        # v2.13.5:特征块加宽留痕(_hpo_parent_face_risk 据此识别加宽面)
        metrics_extra["widen_stats"] = {
            "n_new_cols": int(X_tr.shape[1] - _n_base_cols),
            "source": "trial_feature_blocks"}
    # v2.17.2(A5):n_features = 拟合期模型列数(fit_and_score 写 X_tr.shape[1]),
    # 基础面列数另记 n_features_base(planner 判"同面"用模型列)
    metrics_extra["n_features_base"] = int(_n_base_cols)
    if os.environ.get("MODELEVO_SMOKE", "1") != "0":
        metrics_extra["smoke_passed"] = True
    score, _m = fit_and_score(X_tr, y_tr, X_va, y_va, seed, cfg,
                              w_tr=w_tr, w_va=w_va)
    metrics_extra.update(_m)
    # v13:训练后泄漏哨兵(top-K gain 特征的 train 单变量 AUC);
    # suspects 非空时评估器强制 invalid(leakage_suspect)。fit_and_score
    # 内部把训练好的 model 存进 _last_model 供此处引用
    metrics_extra["leak_scan"] = leak_scan(_last_model, X_tr, y_tr)
    pd.DataFrame({"score": score}).to_parquet(
        TRIAL_DIR / "validation_score.parquet", index=False)
    # shadow_test 预测落盘，供未来独立 confirm 作父代对照。普通搜索
    # 只绑定样本键，不读取目标或计算 shadow 指标。
    # transform 加宽的 trial:shadow 特征面按 validation_input 同源
    # 重建(行内变换与 split 无关),失败降级跳过不阻塞 trial。
    try:
        _sh_path = SPLITS / "shadow_test.parquet"
        # v2.17.3:搜索代(非 confirm)不产 shadow 工件。shadow 分数只在
        # fresh confirmation 里被消费(runner 只对 confirm 打开 shadow,
        # shadow_policy=confirmation_only);在 447k 行上重放加宽 transform
        # + 特征块 + 全量打分,对 g03 实测是 ~55 s 纯浪费,对冷缓存
        # 世代是 557~686 s。MODELEVO_SHADOW_SCORE=skip(由 runner 按
        # search_action 下发)跳过;confirm 仍照常产出。
        if (_sh_path.exists() and _last_model is not None
                and os.environ.get("MODELEVO_SHADOW_SCORE", "1") != "skip"):
            # 注意:模板的加宽 transform 不在本段——本段只做与 validation
            # 同源的基础面重建,**加宽面 trial 必须自行重放 transform
            # 后再调 dump_shadow_score**。模板基座无加宽逻辑,对加宽冠军
            # 会走「列缺失 → 降级跳过」分支(见下方 else),不会产生
            # 错配分数。历史事故(v2.11.6)不在本段,而在 agent 自写
            # 的 shadow 重放段误用 widen_cached 同 tag —— 命中训练缓存
            # 拿到 validation 帧,详见 _widen_face_key 的 docstring。
            _sh = pd.read_parquet(_sh_path)
            # v2.13.1(实测事故回写):trial 定义了 `_bundle_widen` 时
            # **自动重放** transform。此前加宽面 trial 只能靠作者自觉在
            # shadow 段手写重放,而 run-draw-req-t7-v2 gen-01 实测
            # 7/7 加宽臂都漏了这一步 —— 基座只重建基础面,新列不在
            # `_sh_feats` 里,reindex 出全 0 后喂给模型,报
            # `No match for FieldRef.Name(<新列>) in <cat_col>: string`,
            # 被下方 except 静默吞掉 ⇒ **整代无 shadow 工件**,交付链在
            # confirmation 阶段才暴露。自动重放把"作者纪律"降级为
            # "框架保证":只要有 _bundle_widen 就走同口径重放。
            _sh_base = [c for c in X_tr.columns if c in _sh.columns]
            _X_sh = prepare_features(_sh[_sh_base], _sh[_sh_base])[0]
            if "_bundle_widen" in globals():
                _bw = globals()["_bundle_widen"]
                try:
                    _X_sh = _bw(_X_sh, "shadow_test")
                except TypeError:
                    _X_sh = _bw(_X_sh, "shadow_test", None)  # (X, split, y)
            # v2.13.5:特征块同口径重放(命中缓存,不重算)
            _X_sh = _append_feature_blocks(_X_sh, "shadow_test")
            _sh_missing = [c for c in X_tr.columns
                           if c not in _X_sh.columns]
            if not _sh_missing:
                # v2.13.4:NaN 原生打分,与训练/validation/交付 _score_split
                # (v2.10.1/F1)同口径。旧实现 reindex(fill_value=0)+fillna(0)
                # 把缺失编码成 0,是训练-打分口径错位:实测 req-t7-drawdown
                # (shadow 帧 29% 缺失)同配方不同 seed 的分数相关 0.987→0.893、
                # 配对 SE 放大约 3 倍,confirm 的 shadow 等价检验沦为抽签。
                # 仅当模型拒收 NaN(非 GBM 血缘)时降级填 0,并留痕口径。
                _X_sh = _X_sh.reindex(columns=X_tr.columns).astype("float32")

                def _score_sh(_X):
                    if hasattr(_last_model, "predict_proba"):
                        try:
                            return _last_model.predict_proba(_X)[:, 1]
                        except Exception:
                            return _last_model.predict(_X)
                    if hasattr(_last_model, "predict"):
                        return _last_model.predict(_X)
                    return None

                _sh_mode = "nan_native"
                try:
                    _sh_score = _score_sh(_X_sh)
                except Exception:
                    _sh_mode = "fillna0_fallback"
                    _sh_score = _score_sh(_X_sh.fillna(0))
                metrics_extra["shadow_scoring_mode"] = _sh_mode
                if _sh_score is not None:
                    dump_shadow_score(_sh_score)
                    metrics_extra["shadow_scored"] = True
            else:
                print(f"[warn] shadow 打分跳过: transform 重放后仍有 "
                      f"{len(_sh_missing)} 列缺失(示例 "
                      f"{_sh_missing[:3]})—— 检查 _bundle_widen 是否"
                      "真的覆盖了 shadow 路径", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] shadow 打分跳过(不阻塞 trial): {_e}", flush=True)
    # v16/P1-2:multi_seed 稳定性激活——config 声明 multi_seed(n>1)
    # 时,主 seed 之外再训 n-1 个变体,分数落 seed_scores/(runner 的
    # _collect_multi_seed_scores 消费 → evaluator 稳定性门从"恒空转"
    # 变为"跨 seed 方差有界")。v18:config 未声明时,confirm 动作
    # 的 trial 默认 3 seed(交付候选的稳定性证据从建议变默认;
    # explore/exploit 保持单 seed 不放大训练成本)。
    # v19:回退读 hyperparameters.multi_seed——v19 前快通道把 --set
    # multi_seed 并进该节而 main() 只读顶层,静默失效(t011/t012 实测);
    # 键位已在创建期分流,此处回退兼容旧 run 的存量 config。
    _ms = int(cfg.get("multi_seed", 1) or 1)
    if _ms <= 1:
        _ms = int((cfg.get("hyperparameters") or {}).get("multi_seed", 1)
                  or 1)
        if _ms > 1:
            print("[warn] multi_seed 从 hyperparameters 节回退读取"
                  "(v19 前快通道键位错位的存量 config;建议把键移到"
                  "config 顶层)", flush=True)
    if _ms <= 1 and str(cfg.get("search_action")
                        or cfg.get("action") or "") == "confirm":
        _ms = 3
    if _ms > 1 and not _quick:
        # v2.13.4:seed 变体轮只产出 seed_scores/,不得覆写 primary 工件。
        # 旧实现每轮 fit_and_score 都 dump_model/dump_eval_frame,最终
        # model/ 落的是最后一个 seed,而 validation/shadow 分数来自 primary
        # ⇒ 打分模型 ≠ 落盘/完整性绑定模型(实测 req-t7-drawdown trial-016)。
        _primary_model = _last_model
        os.environ["MODELEVO_SEED_VARIANT"] = "1"
        try:
            import pathlib as _pl

            _sd = _pl.Path(TRIAL_DIR) / "seed_scores"
            _sd.mkdir(parents=True, exist_ok=True)
            # v2.15.0:HPO trial 的 seed 变体复用主轮**重训配方**,
            # 不再每个 seed 重跑一遍完整搜索(旧实现 multi_seed=3 的
            # HPO trial = 3 次完整 HPO)
            _cfg_v = cfg
            if (cfg or {}).get("hpo") and isinstance(
                    metrics_extra.get("best_params"), dict):
                _cfg_v = {k: v for k, v in cfg.items() if k != "hpo"}
                _cfg_v["hyperparameters"] = dict(
                    metrics_extra["best_params"])
            _todo = []
            for _k in range(1, _ms):
                # [FIX-2026-09-10] 幂等续跑:该 seed 分数已落盘(上次
                # run 中途被杀)即跳过,已训轮不推倒重训
                if (_sd / f"seed-{_k}.parquet").exists():
                    print(f"[resume] seed-{_k} 已存在,跳过", flush=True)
                    continue
                _todo.append(_k)
            # v2.17.2(D2):条件满足(额外 GPU 租约 + 内存准入)时变体并行;
            # 否则(或 worker 失败)下方串行补齐,seed-k 文件与串行同口径
            if _todo:
                metrics_extra["multi_seed_parallel"] = \\
                    _run_seed_variants_parallel(
                        _todo, _cfg_v, seed, X_tr, y_tr, X_va, y_va,
                        w_tr, w_va, _sd)
            for _k in _todo:
                if (_sd / f"seed-{_k}.parquet").exists():
                    continue
                _sc, _ = fit_and_score(X_tr, y_tr, X_va, y_va,
                                       seed + 1000 * _k, _cfg_v,
                                       w_tr=w_tr, w_va=w_va)
                pd.DataFrame({"score": _sc}).to_parquet(
                    _sd / f"seed-{_k}.parquet", index=False)
            metrics_extra["multi_seed_n"] = _ms
        except Exception as _e:  # noqa: BLE001
            print(f"[warn] multi_seed 轮失败: {_e}", flush=True)
        finally:
            os.environ.pop("MODELEVO_SEED_VARIANT", None)
            globals()["_last_model"] = _primary_model
            metrics_extra["artifact_model"] = "primary_seed"
    dump_json(TRIAL_DIR / "metrics.json", metrics_extra)


if __name__ == "__main__":
    main()
'''


#: [v2.16.2] hurdle 超参段(D-008,run4 regression loan_amt_t7 gen-04 回写)。
#: 由 `render._inject_hurdle_hp_segment` 追加到 hurdle 父代副本文件尾
#: (`if __name__ == "__main__"` 之前),**包住**父代的 hurdle 训练入口而不是
#: 覆盖它:旧 HPO 嫁接把模板版 fit_and_score 追加在文件尾,"后定义覆盖"使
#: inner 搜索与 outer 重训都退化成单个 XGBRegressor(hurdle→单模型的历史
#: Δ≈−0.0076 与 HPO 增益混在一起)。契约(父代必须提供,创建期校验):
#:   - `_hurdle_estimators(n_jobs, n_rows_clf, n_rows_reg) -> (clf, reg)`;
#:   - `_fit_hurdle(n_jobs, X_fit, y_fit, X_es, y_es)` 返回带 predict /
#:     clf_ / reg_ 的容器(score = P(y>0)·max(E[y|y>0], 0));
#:   - `fit_and_score(X_tr, y_tr, X_va, y_va, seed, cfg)` 是 main 的训练入口。
HURDLE_HP_BEGIN = "# ==== v2.16.2 hurdle 超参段 BEGIN"
HURDLE_HP_END = "# ==== v2.16.2 hurdle 超参段 END ===="
HURDLE_HP_SEGMENT = r'''# 作用(父代训练代码一字不改,只在外层加一层):
#   - config.hpo 存在:HPO。inner 搜索在 train 内切(时间序尾部 inner_fraction 为
#     inner 验证面,其余为拟合池;拟合池头部 90% 拟合、尾部 10% 早停),每个 config
#     构造父代 hurdle(_fit_hurdle:分类段 P(y>0) + 条件回归段 E[y|y>0]),按主指标
#     评组合预测 p·μ;outer validation 只在最终重训后评一次(lint 规则 7)。
#     搜索键:无前缀 = 两段共用,clf__<键> / reg__<键> = 只作用于对应段;
#     hurdle_lr / hurdle_cap 是 learning_rate / n_estimators 的别名。
#     重训沿用父代 fit_and_score(v2.16 默认 outer validation 早停,两段各自早停);
#     best_params = 重训配方(含生效的 lr / cap),confirm 快通道与交付复跑都消费它。
#   - 否则 config.hyperparameters 非空:按同一键口径覆写 hurdle 配方(confirm 子代、
#     finalize-retrain 复跑、--set 参数子代、multi-seed 变体轮共用,进程内持续生效)。
#   - 两者皆无:与父代逐字等价。
# 覆写方式:父代 _hurdle_estimators 原样构造后 set_params(seed / 设备 / eval_metric
# 等沿用父代;未知键报错,不静默空转)。
import os as _hhp_os
import threading as _hhp_threading

_fit_and_score_hurdle_parent = fit_and_score
_hurdle_estimators_parent = _hurdle_estimators
#: HPO inner 并行 config 的线程局部覆写;None 时用进程级覆写
_HURDLE_HP_TLS = _hhp_threading.local()
#: 进程级覆写(重训 / confirm / 交付复跑 / seed 变体)
_HURDLE_HP_ACTIVE = {"clf": {}, "reg": {}}
_HURDLE_HP_ALIASES = {"hurdle_lr": "learning_rate", "hurdle_cap": "n_estimators"}
#: 不经超参覆写的键:seed 由 SEED / multi-seed 管,线程与设备由调度管
_HURDLE_HP_FORBIDDEN = frozenset({"random_state", "seed", "n_jobs", "nthread",
                                  "device"})
_HURDLE_HP_INT_KEYS = frozenset({"max_depth", "n_estimators", "max_leaves",
                                 "max_bin", "num_parallel_tree",
                                 "early_stopping_rounds", "num_leaves",
                                 "min_child_samples"})


def _hhp_log(msg):
    _st = globals().get("step")
    if callable(_st):
        _st(msg)
    else:
        print("[step] " + str(msg), flush=True)


def _hhp_max_trees():
    """[铁律]树数上限(模块 _MAX_TREES,缺省 8000)。"""
    try:
        return int(globals().get("_MAX_TREES") or 8000)
    except (TypeError, ValueError):
        return 8000


def _hhp_es_rounds(lr):
    """早停耐心随 lr 缩放:max(MODELEVO_EARLY_STOP_ROUNDS 或 50, ceil(MODELEVO_ES_LR_SCALE/lr)),
    与模板 _es_rounds_for、父代 EARLY_STOP_ROUNDS = max(50, ceil(0.5/lr)) 同口径。"""
    import math as _hhp_math
    try:
        base = int(_hhp_os.environ.get("MODELEVO_EARLY_STOP_ROUNDS") or 50)
    except ValueError:
        base = 50
    try:
        scale = float(_hhp_os.environ.get("MODELEVO_ES_LR_SCALE", "0.5") or 0)
        lr = float(lr)
    except (TypeError, ValueError):
        return base
    if scale <= 0 or lr <= 0:
        return base
    return max(base, int(_hhp_math.ceil(scale / lr)))


def _hhp_norm_key(key):
    k = str(key)
    for p in ("clf__", "reg__"):
        if k.startswith(p):
            return p + _HURDLE_HP_ALIASES.get(k[len(p):], k[len(p):])
    return _HURDLE_HP_ALIASES.get(k, k)


def _hurdle_hp_split(params):
    """超参 dict → (分类段覆写, 条件回归段覆写)。分段键优先于共用键;n_estimators
    钳到树数上限;learning_rate 变化而未显式给 early_stopping_rounds 时按 lr 重算耐心。"""
    shared, segs = {}, {"clf__": {}, "reg__": {}}
    for k, v in (params or {}).items():
        k = _hhp_norm_key(k)
        for p in segs:
            if k.startswith(p):
                segs[p][k[len(p):]] = v
                break
        else:
            shared[k] = v
    out = []
    for p in ("clf__", "reg__"):
        d = dict(shared)
        d.update(segs[p])
        bad = sorted(set(d) & _HURDLE_HP_FORBIDDEN)
        if bad:
            raise ValueError("hurdle 超参不允许覆写 %s(seed/线程/设备由 harness 管)"
                             % bad)
        for k in list(d):
            if k in _HURDLE_HP_INT_KEYS and d[k] is not None:
                d[k] = int(round(float(d[k])))
        if "n_estimators" in d:
            d["n_estimators"] = max(1, min(int(d["n_estimators"]),
                                           _hhp_max_trees()))
        if "learning_rate" in d and "early_stopping_rounds" not in d:
            d["early_stopping_rounds"] = _hhp_es_rounds(d["learning_rate"])
        out.append(d)
    return out[0], out[1]


def _hurdle_estimators(n_jobs, n_rows_clf, n_rows_reg):
    """v2.16.2:父代 _hurdle_estimators 原样构造,再按当前覆写 set_params。"""
    clf, reg = _hurdle_estimators_parent(n_jobs, n_rows_clf, n_rows_reg)
    over = getattr(_HURDLE_HP_TLS, "over", None) or _HURDLE_HP_ACTIVE
    for name, est in (("clf", clf), ("reg", reg)):
        d = over.get(name) or {}
        if not d:
            continue
        known = set(est.get_params())
        unknown = sorted(k for k in d if k not in known)
        if unknown:
            raise ValueError("hurdle %s 段未知超参键 %s(%s 无此参数)"
                             % (name, unknown, type(est).__name__))
        est.set_params(**d)
    return clf, reg


def _hurdle_hp_activate(params, source):
    """进程级覆写生效。两段一致的 lr / cap / 早停耐心同步回模块常量
    (HURDLE_LR / N_ESTIMATORS_CAP / EARLY_STOP_ROUNDS),父代 fit_and_score
    与 main 的遥测(cap_bound / learning_rate / es_rounds)随实际配方。"""
    clf, reg = _hurdle_hp_split(params)
    _HURDLE_HP_ACTIVE["clf"] = dict(clf)
    _HURDLE_HP_ACTIVE["reg"] = dict(reg)
    g = globals()
    for key, const in (("learning_rate", "HURDLE_LR"),
                       ("n_estimators", "N_ESTIMATORS_CAP"),
                       ("early_stopping_rounds", "EARLY_STOP_ROUNDS")):
        if key in clf and key in reg and clf[key] == reg[key] and const in g:
            g[const] = clf[key]
    if clf or reg:
        _hhp_log("hurdle 超参生效(%s): clf=%s reg=%s" % (source, clf, reg))
    return clf, reg


def _hurdle_hp_base(keys):
    """父代配方在给定键上的取值(基准点):父代 _hurdle_estimators 原样构造后读回
    (HURDLE_STRUCT / HURDLE_LR / cap 及父代对两段的任何定制都在内)。"""
    clf, reg = _hurdle_estimators_parent(1, 0, 0)
    pc, pr = clf.get_params(), reg.get_params()
    out = {}
    for k in keys or []:
        nk = _hhp_norm_key(k)
        if nk.startswith("clf__"):
            v = pc.get(nk[5:])
        elif nk.startswith("reg__"):
            v = pr.get(nk[5:])
        else:
            v = pc.get(nk)
            if v != pr.get(nk):
                continue  # 父代两段在共用键上取值不同:不伪造单点(该维由采样补齐)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = v
    return out


def _hhp_seg_tel(est):
    """单段遥测:最优树数(1-based)、实训轮数、cap、是否撞 cap(实训轮数 ≥ cap)。"""
    bi = rounds = cap = None
    try:
        bi = int(est.best_iteration) + 1
    except Exception:  # noqa: BLE001
        bi = None
    try:
        ev = est.evals_result()
        first = next(iter(ev.values()))
        rounds = int(len(next(iter(first.values()))))
    except Exception:  # noqa: BLE001
        rounds = None
    try:
        cap = int(est.get_params().get("n_estimators") or 0) or None
    except Exception:  # noqa: BLE001
        cap = None
    # v2.16.3(D-011):撞 cap 走 runtime.hpo.cap_bound_of 单一语义(实训轮数
    # ≥ cap;无轮数时 best + 早停耐心 ≥ cap);runtime 不可导入时同语义内联
    try:
        from runtime.hpo import cap_bound_of as _hhp_cbo
        _esr = est.get_params().get("early_stopping_rounds")
        bound = bool(_hhp_cbo(rounds=rounds, cap=cap, best_iteration=bi,
                              es_rounds=_esr))
    except Exception:  # noqa: BLE001
        bound = bool(rounds and cap and rounds >= cap)
    return {"best_iteration": bi, "rounds": rounds, "cap": cap,
            "cap_bound": bound}


def _hhp_fit_telemetry(model):
    out = {}
    for name in ("clf", "reg"):
        est = getattr(model, name + "_", None)
        if est is None:
            continue
        for k, v in _hhp_seg_tel(est).items():
            out[name + "_" + k] = v
    bis = [out.get(n + "_best_iteration") for n in ("clf", "reg")
           if out.get(n + "_best_iteration")]
    rds = [out.get(n + "_rounds") for n in ("clf", "reg")
           if out.get(n + "_rounds")]
    out["best_iteration"] = max(bis) if bis else None
    out["rounds"] = max(rds) if rds else None
    out["cap_bound"] = bool(out.get("clf_cap_bound") or out.get("reg_cap_bound"))
    return out


#: 遥测里回读的两段生效参数(父代 metrics 的 struct_params 是模块常量,覆写后会过时)
_HURDLE_HP_REPORT_KEYS = ("learning_rate", "n_estimators", "early_stopping_rounds",
                          "max_depth", "min_child_weight", "subsample",
                          "colsample_bytree", "reg_lambda", "reg_alpha", "gamma")


def _hhp_report(model):
    """{"clf": 覆写, "reg": 覆写, "effective": {段: 生效参数}}(从拟合模型读回)。"""
    rep = {"clf": dict(_HURDLE_HP_ACTIVE["clf"]),
           "reg": dict(_HURDLE_HP_ACTIVE["reg"]), "effective": {}}
    for name in ("clf", "reg"):
        est = getattr(model, name + "_", None)
        if est is None:
            continue
        try:
            pp = est.get_params()
        except Exception:  # noqa: BLE001
            continue
        keys = set(_HURDLE_HP_REPORT_KEYS) | set(rep[name])
        rep["effective"][name] = {
            k: pp[k] for k in sorted(keys)
            if k in pp and isinstance(pp[k], (int, float, str, bool))}
    return rep


def _hhp_metric():
    """inner 指标 = spec.primary_metric(回归缺省 rmse)及其方向。"""
    name = str(globals().get("METRIC_NAME") or "")
    if not name:
        try:
            import pathlib as _hhp_pl
            import yaml as _hhp_yaml
            _rd = globals().get("RUN_DIR")
            _sp = _hhp_pl.Path(str(_rd)) / "task-spec.yaml"
            if _rd and _sp.exists():
                _d = _hhp_yaml.safe_load(_sp.read_text(encoding="utf-8")) or {}
                name = str(_d.get("primary_metric") or "")
        except Exception:  # noqa: BLE001
            name = ""
    name = (name or "rmse").lower()
    try:
        from runtime.metrics import metric_direction
        up = bool(metric_direction(name))
    except Exception:  # noqa: BLE001
        up = name not in ("rmse", "mae", "mape", "logloss", "nll")
    return name, up


def _hhp_n_jobs():
    _nj = globals().get("n_jobs")
    if callable(_nj):
        try:
            return int(_nj())
        except Exception:  # noqa: BLE001
            pass
    try:
        return max(1, int(_hhp_os.environ.get("OMP_NUM_THREADS", "4") or 4))
    except ValueError:
        return 4


def _hurdle_hpo_fit_inner(params, X_fit, y_fit, X_in, y_in, seed,
                          fidelity=1.0, metric=None):
    """HPO inner:拟合池头部拟合、尾部 ES_FRACTION 早停(两段各自早停),构造父代
    hurdle,在 inner 验证面上按主指标评组合预测;不触碰 outer validation。"""
    import numpy as _hhp_np
    from runtime.metrics import compute_metric
    if fidelity is not None and float(fidelity) < 1.0:
        from runtime.hpo import stratified_random_rows
        _pos = stratified_random_rows(
            (_hhp_np.asarray(y_fit, dtype="float64") > 0).astype("int8"),
            max(1, int(len(y_fit) * float(fidelity))), seed=int(seed))
        X_fit, y_fit = X_fit.iloc[_pos], y_fit.iloc[_pos]
    _frac = float(globals().get("ES_FRACTION") or 0.1)
    _cut = int(len(X_fit) * (1.0 - _frac))
    clf_o, reg_o = _hurdle_hp_split(params)
    _HURDLE_HP_TLS.over = {"clf": clf_o, "reg": reg_o}
    try:
        m = _fit_hurdle(_hhp_n_jobs(), X_fit.iloc[:_cut], y_fit.iloc[:_cut],
                        X_fit.iloc[_cut:], y_fit.iloc[_cut:])
    finally:
        _HURDLE_HP_TLS.over = None
    s = m.predict(X_in)
    tel = _hhp_fit_telemetry(m)
    out = {"value": float(compute_metric(metric or _hhp_metric()[0],
                                         _hhp_np.asarray(y_in), None,
                                         score=_hhp_np.asarray(s))),
           "n_inner_fit_rows": int(_cut), "n_inner_es_rows": int(len(X_fit) - _cut)}
    out.update(tel)
    return out


def _hhp_recipe_sig():
    """父代配方指纹(热启动签名用):两段估计器的标量参数(不含 seed/线程/设备)。"""
    import hashlib as _hhp_hashlib
    clf, reg = _hurdle_estimators_parent(1, 0, 0)
    parts = []
    for name, est in (("clf", clf), ("reg", reg)):
        p = {k: v for k, v in est.get_params().items()
             if k not in _HURDLE_HP_FORBIDDEN
             and isinstance(v, (int, float, str, bool, type(None)))}
        parts.append(name + ":" + repr(sorted(p.items())))
    return _hhp_hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _hurdle_hpo_run(X_tr, y_tr, X_va, y_va, seed, hpo_cfg, fixed, down,
                    a, kw):
    """hurdle HPO:inner 搜索(v2.15 引擎:基准点 / 热启动签名 / 边界自适应 /
    lr 收缩 / 并行 / 多保真)→ best_params 重训(父代 fit_and_score)。"""
    import math as _hhp_math
    import numpy as _hhp_np
    import pandas as _hhp_pd
    from runtime.hpo import (hpo_fidelity_plan, hpo_parallelism,
                             inner_signature, load_warm_start, plan_lr_mode,
                             record_lr_mode, run_hpo_search,
                             stratified_random_rows, write_hpo_artifacts)
    if not hasattr(y_tr, "iloc"):
        y_tr = _hhp_pd.Series(_hhp_np.asarray(y_tr), index=X_tr.index)
    mx = {"hpo_topology": "structural:hurdle"}
    space = {_hhp_norm_key(k): dict(v)
             for k, v in (hpo_cfg.get("search_space") or {}).items()}
    fixed = {_hhp_norm_key(k): v for k, v in (fixed or {}).items()}
    if fixed:
        mx["hyperparameters_injected"] = dict(fixed)
    budget = int(hpo_cfg.get("budget_configs", 40))
    frac = float(hpo_cfg.get("inner_fraction", 0.2))
    n = len(X_tr)
    cut = int(n * (1.0 - frac))
    pool_X, pool_y = X_tr.iloc[:cut], y_tr.iloc[:cut]
    X_in, y_in = X_tr.iloc[cut:], y_tr.iloc[cut:]
    _sr = int(hpo_cfg.get("sample_rows") or 0)
    if _sr > 0 and len(pool_X) > _sr:
        # 按 y>0 分层随机抽(两段的训练面同时按比例缩:正例率与全体一致),行序保持时间序
        _pos = stratified_random_rows(
            (_hhp_np.asarray(pool_y, dtype="float64") > 0).astype("int8"), _sr)
        X_fit, y_fit = pool_X.iloc[_pos], pool_y.iloc[_pos]
        mx["hpo_sample_rows"] = int(_sr)
    else:
        X_fit, y_fit = pool_X, pool_y
    mx["hpo_inner_rows"] = int(len(X_fit))
    mx["hpo_inner_valid_rows"] = int(len(X_in))
    metric, up = _hhp_metric()
    algo = str(globals().get("_ALGORITHM") or globals().get("_HPO_ALGO")
               or "xgboost").lower()
    _bc, _br = _hurdle_estimators_parent(1, 0, 0)
    base_cap = int(_bc.get_params().get("n_estimators")
                   or globals().get("N_ESTIMATORS_CAP") or 2000)
    base_es = _bc.get_params().get("early_stopping_rounds")
    lrp = plan_lr_mode(space, hpo_cfg, len(X_fit))
    # v2.16.3:hurdle 血缘同样记录生效 lr 模式(hpo-study / metrics)并提示
    if lrp.get("note"):
        _hhp_log("hpo lr_mode: %s" % lrp["note"])
    record_lr_mode(TRIAL_DIR, lrp)
    fids = hpo_fidelity_plan(hpo_cfg, len(X_fit))
    sig = inner_signature(X_fit, y_fit, X_in, y_in, extra={
        "topology": "structural:hurdle", "algorithm": algo, "metric": metric,
        "model_type": "regression", "cap": base_cap, "es_rounds": base_es,
        "lr_mode": lrp["mode"], "search_lr": lrp.get("search_lr"),
        "recipe": _hhp_recipe_sig(), "fixed": sorted(fixed.items())})
    warm = [] if fids else load_warm_start(
        TRIAL_DIR / "config" / "hpo-warm-start.json", sig)
    _ga = globals().get("_gpu_allocated")
    gpu = bool(callable(_ga) and _ga(len(X_fit)))
    par = hpo_parallelism(hpo_cfg, algo, gpu)
    expand = hpo_cfg.get("auto_expand", True)
    if expand is True:
        expand = {"lr_floor": float(hpo_cfg.get("lr_floor") or 0.005)}
    # 基准点:父代配方(当前构造读回)+ 显式 --set;再接创建期给的(如父代 HPO 冠军配方)
    enq = []
    base = _hurdle_hp_base(list(lrp["space"]))
    base.update({k: v for k, v in fixed.items() if k in lrp["space"]})
    if base:
        enq.append(base)
    for q in hpo_cfg.get("enqueue") or []:
        if isinstance(q, dict) and q:
            enq.append({_hhp_norm_key(k): v for k, v in q.items()})

    def _inner(p, fidelity=1.0):
        pp = dict(p)
        if lrp["mode"] == "shrink":
            pp[lrp["lr_key"]] = lrp["search_lr"]
        pp.update(fixed)
        if fidelity is not None and float(fidelity) < 1.0:
            return _hurdle_hpo_fit_inner(pp, X_fit, y_fit, X_in, y_in, seed,
                                         fidelity=fidelity, metric=metric)
        return _hurdle_hpo_fit_inner(pp, X_fit, y_fit, X_in, y_in, seed,
                                     metric=metric)

    _hhp_log("hurdle hpo search start: configs=%d inner_rows=%d inner_eval_rows=%d "
             "lr_mode=%s warm=%d parallel=%d metric=%s keys=%s"
             % (budget, len(X_fit), len(X_in), lrp["mode"], len(warm), par,
                metric, sorted(lrp["space"])))
    # v2.17.2(A3):deadline 外推预留最终重训时间(同单模型口径)
    from runtime.hpo import hpo_refit_reserve_sec
    _reserve = {"static_sec": hpo_refit_reserve_sec(TRIAL_DIR),
                "rows_ratio": float(n) / max(1, len(X_fit)),
                "trees_scale": (float(lrp.get("refit_cap_scale") or 1.0)
                                if lrp["mode"] == "shrink" else 1.0),
                "max_trees": _hhp_max_trees()}
    result = run_hpo_search(
        lrp["space"], budget, _inner, metric_direction_up=up, seed=seed,
        metric_name=metric,
        deadline_ts=float(_hhp_os.environ.get("MODELEVO_DEADLINE_TS") or 0)
        or None,
        checkpoint_path=TRIAL_DIR / "hpo-checkpoint.json", warm_start=warm,
        enqueue=enq, signature=sig, n_jobs=par, expand=expand or None,
        seed_salt=TRIAL_DIR.name, fidelities=fids, refit_reserve=_reserve)
    _hhp_log("hurdle hpo search done: n=%s inner_best=%s expansions=%d"
             % (result.get("n_configs_evaluated"), result.get("inner_score"),
                len(result.get("space_expansions") or [])))
    result.update({"algorithm": algo, "lr_mode": lrp["mode"],
                   "lr_mode_requested": lrp.get("requested_mode"),
                   "lr_mode_note": lrp.get("note"),
                   "topology": "structural:hurdle", "refit_pending": True})
    write_hpo_artifacts(TRIAL_DIR, result)   # 搜索结果先落盘(重训崩溃不丢)
    search_best = dict(result.get("best_params") or {})
    final = dict(search_best)
    refit_cap = None
    refit_plan = None
    if lrp["mode"] == "shrink":
        # v2.17.2(A1):重训 cap / lr 由搜索期所选 config 的最优轮数推出(两段
        # 取较大者;runtime.hpo.plan_refit_shrink,与单模型同口径)。需求超出
        # 树数上限时放宽收缩因子(重训 lr 提高、cap = 上限),不再静默截断。
        from runtime.hpo import plan_refit_shrink, search_best_iteration
        refit_plan = plan_refit_shrink(
            search_best_iteration(result), search_lr=lrp["search_lr"],
            factor=lrp["factor"], cap_now=base_cap,
            cap_scale=lrp.get("refit_cap_scale"), max_trees=_hhp_max_trees())
        final[lrp["lr_key"]] = refit_plan["refit_lr"]
        want = int(refit_plan["cap_wanted_legacy"])
        refit_cap = max(1, min(int(refit_plan["refit_cap"]), _hhp_max_trees()))
        mx["hpo_refit_lr_adjusted"] = bool(refit_plan["lr_adjusted"])
        mx["hpo_refit_factor_eff"] = float(refit_plan["factor_eff"])
        mx["hpo_refit_cap_clamped"] = bool(refit_plan["cap_clamped"])
        mx["hpo_refit_plan"] = dict(refit_plan)
        if refit_plan["lr_adjusted"]:
            print("[warn] lr 收缩重训:搜索期最优 %s 轮 / factor %g × 1.2 需 %s 轮 > "
                  "树数上限 %d → 收缩因子放宽为 %g(重训 lr=%s,cap=%d)"
                  % (refit_plan["search_best_iteration"], refit_plan["factor"],
                     refit_plan["need_trees"], refit_plan["max_trees"],
                     refit_plan["factor_eff"], refit_plan["refit_lr"],
                     refit_cap), flush=True)
        elif refit_plan["basis"] == "unavailable" and refit_cap < want:
            try:
                _base_lr = float(_bc.get_params().get("learning_rate") or 0)
            except (TypeError, ValueError):
                _base_lr = 0.0
            # 旧口径(搜索期最优轮数不可得):cap 被钳制时告警
            print("[warn] lr 收缩重训 cap %d 被树数上限钳到 %d(搜索期最优轮数不可得,"
                  "沿用旧口径),重训 lr=%s(父代 %s):两段可能撞 cap(见 "
                  "hpo_refit_es_info.*_cap_bound)"
                  % (want, refit_cap, refit_plan["refit_lr"], _base_lr), flush=True)
        if not any(_hhp_norm_key(k).endswith("n_estimators") for k in final):
            final["n_estimators"] = refit_cap
    final.update(fixed)
    _hurdle_hp_activate(final, "hpo_refit")
    score, pmx = _fit_and_score_hurdle_parent(X_tr, y_tr, X_va, y_va, seed,
                                              down, *a, **kw)
    model = globals().get("_last_model")
    tel = _hhp_fit_telemetry(model) if model is not None else {}
    # v2.17.2(A1):任一段撞 cap 且该段 cap 已达树数上限 = 截断未解除
    _mt = _hhp_max_trees()
    tel["cap_bound_unresolved"] = bool(any(
        tel.get(n + "_cap_bound") and int(tel.get(n + "_cap") or 0) >= _mt
        for n in ("clf", "reg")))
    # best_params = 重训配方:两段一致的 lr / cap 显式写出(confirm / 交付复跑自包含)
    try:
        pc, pr = model.clf_.get_params(), model.reg_.get_params()
        for key in ("learning_rate", "n_estimators"):
            if not any(_hhp_norm_key(k).split("__")[-1] == key for k in final) \
                    and pc.get(key) == pr.get(key) and pc.get(key) is not None:
                final[key] = pc.get(key)
    except Exception:  # noqa: BLE001
        pass
    result["search_best_params"] = search_best
    result["best_params"] = final
    result["refit"] = {"params": final, "es_info": tel, "refit_cap": refit_cap,
                       "lr": final.get(lrp.get("lr_key") or "learning_rate"),
                       "plan": refit_plan,
                       "segments": {"clf": dict(_HURDLE_HP_ACTIVE["clf"]),
                                    "reg": dict(_HURDLE_HP_ACTIVE["reg"])}}
    if refit_plan is not None:
        result["lr_mode_refit_lr"] = refit_plan["refit_lr"]
    result.pop("refit_pending", None)
    write_hpo_artifacts(TRIAL_DIR, result)
    mx.update({
        "best_params": final, "hpo_search_best_params": search_best,
        "hpo_n_configs": result.get("n_configs_evaluated"),
        "hpo_sampler": result.get("sampler"),
        "hpo_inner_score": result.get("inner_score"),
        "hpo_inner_signature": sig, "hpo_lr_mode": lrp["mode"],
        # v2.16.3:请求 vs 生效并列(hurdle 血缘同口径)
        "hpo_lr_mode_effective": lrp["mode"],
        "hpo_lr_mode_requested": lrp.get("requested_mode"),
        "hpo_lr_mode_note": lrp.get("note"),
        "hpo_parallel": par, "hpo_warm_start_n": len(warm),
        "hpo_space_expansions": result.get("space_expansions") or [],
        "hpo_diagnostics": result.get("diagnostics") or {},
        "hpo_refit_es_info": tel})
    if result.get("budget_truncated_by_deadline"):
        mx["hpo_budget_truncated_by_deadline"] = True
        mx["hpo_budget_requested"] = result.get("budget_requested", budget)
    if result.get("n_configs_replayed"):
        mx["hpo_n_configs_replayed"] = int(result["n_configs_replayed"])
        mx["hpo_n_configs_fit_this_attempt"] = int(
            result.get("n_configs_fit_this_attempt", 0))
    if result.get("smoke_budget_reduced"):
        mx["hpo_smoke_budget_reduced"] = True
    out = dict(pmx or {})
    out.update(mx)
    # v2.17.2(A1):评估侧契约 metrics.es_info.cap_bound_unresolved(bool)。父代
    # 不写 es_info(手写 hurdle)时只在 True 时补建,避免"未知"被读成 cap_bound=False
    if isinstance(out.get("es_info"), dict):
        _es_out = dict(out["es_info"])
        _es_out["cap_bound_unresolved"] = bool(
            tel.get("cap_bound_unresolved")
            or _es_out.get("cap_bound_unresolved"))
        out["es_info"] = _es_out
    elif tel.get("cap_bound_unresolved"):
        out["es_info"] = {"cap_bound": True, "cap_bound_unresolved": True,
                          "stopped_by": "cap",
                          "source": "hurdle_hpo_refit_segments"}
    if model is not None:
        out["hurdle_hp"] = _hhp_report(model)
    _hhp_log("hurdle hpo refit done: clf_best=%s reg_best=%s clf_cap_bound=%s "
             "reg_cap_bound=%s" % (tel.get("clf_best_iteration"),
                                   tel.get("reg_best_iteration"),
                                   tel.get("clf_cap_bound"),
                                   tel.get("reg_cap_bound")))
    return score, out


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None, *args,
                  **kwargs):
    """v2.16.2 hurdle 超参入口:HPO / config 超参覆写 / 原样委托(见段首说明)。"""
    _cfg = dict(cfg or {})
    _hp_raw = _cfg.get("hyperparameters")
    _hp = {k: v for k, v in (_hp_raw.items() if isinstance(_hp_raw, dict) else ())
           if isinstance(v, (int, float, str, bool))}
    _hpo = _cfg.get("hpo")
    if not (isinstance(_hpo, dict) and _hpo) and not _hp:
        _HURDLE_HP_ACTIVE["clf"], _HURDLE_HP_ACTIVE["reg"] = {}, {}
        return _fit_and_score_hurdle_parent(X_tr, y_tr, X_va, y_va, seed, cfg,
                                            *args, **kwargs)
    # 向父代传的 cfg:去掉 hpo 节与已在本段消费的 hyperparameters(父代自带的
    # 消费点,如 gen-04 的 _apply_hurdle_hp,不重复消费、不因共用键报错)
    _down = {k: v for k, v in _cfg.items() if k not in ("hpo", "hyperparameters")}
    _down["hyperparameters"] = {}
    if isinstance(_hpo, dict) and _hpo:
        return _hurdle_hpo_run(X_tr, y_tr, X_va, y_va, seed, _hpo, _hp, _down,
                               args, kwargs)
    _hurdle_hp_activate(_hp, "config.hyperparameters")
    score, mx = _fit_and_score_hurdle_parent(X_tr, y_tr, X_va, y_va, seed,
                                             _down, *args, **kwargs)
    mx = dict(mx or {})
    mx["hyperparameters_injected"] = dict(_hp)
    _mdl = globals().get("_last_model")
    mx["hurdle_hp"] = _hhp_report(_mdl) if _mdl is not None else {
        "clf": dict(_HURDLE_HP_ACTIVE["clf"]),
        "reg": dict(_HURDLE_HP_ACTIVE["reg"])}
    return score, mx'''


#: [v2.17.0] 官方 hurdle trial 模板入口(run4 regression.2 #15)。
#: `new-trial --hurdle-template` 渲染:TRIAL_BODY 的脚手架 fit_and_score 换成
#: 本段(v2.16.2 hurdle 超参段契约的三个入口 `_hurdle_estimators` /
#: `_fit_hurdle` / `fit_and_score`,容器 = runtime.hurdle.HurdleRegressor),
#: 渲染后再注入 HURDLE_HP_SEGMENT(HPO / --set 分段键 / confirm 与交付复跑
#: 同一消费口径)。容器可 sklearn.clone(交付 frozen_rounds 直接可用)、
#: 不暴露 predict_proba(lint 规则 4 / 交付打分一律 predict),两段拟合经
#: 子模型缓存(只改一段的子代,另一段命中缓存)。
HURDLE_TEMPLATE_BEGIN = "# ==== v2.17.0 官方 hurdle 段 BEGIN"
HURDLE_TEMPLATE_END = "# ==== v2.17.0 官方 hurdle 段 END ===="
HURDLE_TRIAL_ENTRIES = r'''# ==== v2.17.0 官方 hurdle 段 BEGIN(new-trial --hurdle-template 渲染;structural:hurdle)====
# score = P(y>0|X) · max(E[y|y>0,X], 0)(原值尺度的条件均值,无 Jensen 偏差)。
# 拓扑:两个子模型各自只吃同一份特征,固定公式 p·μ 组合 —— 分类概率**不**
# 进入回归段的特征/目标(那是级联,须用户批准)。容器 = runtime.hurdle.
# HurdleRegressor:可 sklearn.clone、不暴露 predict_proba、两段各自早停、
# 两段拟合经子模型缓存(RUN_DIR/.submodel-cache;只改一段的子代另一段命中)。
# 超参:改下方常量,或 --set clf__<键>=…/reg__<键>=…/<共用键>=…(hurdle 超参段
# 消费;hurdle_lr / hurdle_cap 为 lr / cap 别名);HPO 用 --action hpo
# --from-parent <本 trial>(inner 搜索与重训都构造本 hurdle)。
from runtime.hurdle import HurdleRegressor  # noqa: E402
import runtime.hurdle as _hurdle_rt  # noqa: E402

_hurdle_rt.set_progress_logger(step)

_ALGORITHM = "xgboost"
HURDLE_LR = 0.05
N_ESTIMATORS_CAP = 2000
EARLY_STOP_ROUNDS = 50
ES_FRACTION = 0.1
#: 两段结构 / 正则参数(seed、线程、设备由 harness 管,不在此处)
HURDLE_CLF_PARAMS = dict(max_depth=6, min_child_weight=1.0, subsample=0.9,
                         colsample_bytree=0.9, reg_lambda=1.0)
HURDLE_REG_PARAMS = dict(max_depth=6, min_child_weight=1.0, subsample=0.9,
                         colsample_bytree=0.9, reg_lambda=1.0)
#: 本轮 seed(fit_and_score 入参;multi-seed 变体轮按入参换 seed)
_HURDLE_SEED = None


def _hurdle_estimators(n_jobs, n_rows_clf, n_rows_reg):
    """两段估计器原型 (clf, reg)(v2.16.2 hurdle 超参段契约)。"""
    import xgboost as xgb
    _seed = int(_HURDLE_SEED if _HURDLE_SEED is not None else SEED)
    _gpu_c = dict(device="cuda") if (_gpu_allocated(n_rows_clf)
                                     and _xgb_cuda_built()) else {}
    _gpu_r = dict(device="cuda") if (_gpu_allocated(n_rows_reg)
                                     and _xgb_cuda_built()) else {}
    clf = xgb.XGBClassifier(
        n_estimators=_clamp_trees(N_ESTIMATORS_CAP), learning_rate=HURDLE_LR,
        **HURDLE_CLF_PARAMS, random_state=_seed, n_jobs=n_jobs,
        tree_method="hist", eval_metric="logloss",
        early_stopping_rounds=EARLY_STOP_ROUNDS, **_gpu_c)
    reg = xgb.XGBRegressor(
        n_estimators=_clamp_trees(N_ESTIMATORS_CAP), learning_rate=HURDLE_LR,
        **HURDLE_REG_PARAMS, random_state=_seed, n_jobs=n_jobs,
        tree_method="hist", early_stopping_rounds=EARLY_STOP_ROUNDS, **_gpu_r)
    return clf, reg


def _fit_hurdle(n_jobs, X_fit, y_fit, X_es, y_es):
    """拟合官方 hurdle 容器(两段各自早停:分类段 1[y_es>0],回归段 es 面正例)。"""
    _y = np.asarray(y_fit, dtype="float64")
    clf, reg = _hurdle_estimators(n_jobs, len(X_fit), int((_y > 0).sum()))
    model = HurdleRegressor(clf=clf, reg=reg, threshold=0.0,
                            submodel_cache=True, cache_dir=str(RUN_DIR))
    _es = [(X_es, y_es)] if X_es is not None and len(X_es) else None
    model.fit(X_fit, y_fit, eval_set=_es)
    return model


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    """hurdle 唯一训练与打分入口(模板契约;finalize-retrain 复跑本函数)。

    早停面:validation(默认)→ 100% 传入 train 拟合、outer validation 两段各自
    早停;train_tail 或拿不到 y_va → 传入帧最新尾部 ES_FRACTION 早停。
    返回 (score, metrics_extra);_last_model = 容器。
    """
    global _last_model, _HURDLE_SEED
    _HURDLE_SEED = int(seed) if seed is not None else None
    if _es_source() == "validation" and y_va is not None:
        X_fit, y_fit, X_es, y_es = X_tr, y_tr, X_va, y_va
        _face = "validation"
    else:
        _cut = int(len(X_tr) * (1.0 - ES_FRACTION))
        X_fit, X_es = X_tr.iloc[:_cut], X_tr.iloc[_cut:]
        y_fit, y_es = y_tr.iloc[:_cut], y_tr.iloc[_cut:]
        _face = "train_tail"
    step(f"hurdle es face={_face}: fit={len(X_fit)} es={len(X_es)}")
    model = _fit_hurdle(n_jobs(), X_fit, y_fit, X_es, y_es)
    dump_model(model, X_tr.columns)
    dump_eval_frame(X_va)
    p_va = model.positive_prob(X_va)
    m_va = np.maximum(model.conditional_mean(X_va), 0.0)
    score = model.predict(X_va)
    tel = model.segment_telemetry()
    _cap = int(_clamp_trees(N_ESTIMATORS_CAP))
    _bound = bool(tel.get("clf_cap_bound") or tel.get("reg_cap_bound"))
    _bis = [b for b in (tel.get("clf_best_iteration"),
                        tel.get("reg_best_iteration")) if b]
    metrics_extra = {
        "n_features": X_tr.shape[1],
        "model_num": 2,   # hurdle 两个子模型(分类 × 条件回归),单一打分链
        "hurdle": dict(tel, es_face=_face,
                       train_pos_rate=float(np.mean(np.asarray(y_fit) > 0)),
                       n_reg_fit_rows=int(np.sum(np.asarray(y_fit) > 0)),
                       valid_p_mean=float(np.mean(p_va)),
                       valid_m_mean=float(np.mean(m_va)),
                       valid_score_mean=float(np.mean(score)),
                       container="runtime.hurdle.HurdleRegressor"),
        "es_info": {"cap": _cap, "cap_bound": _bound,
                    "stopped_by": "cap" if _bound else "early_stop",
                    "es_source": _face, "fit_mode": "single",
                    "source": "hurdle_segments",
                    # v2.17.2(A1):撞 cap 且 cap 已达树数上限 = 截断未解除
                    "cap_bound_unresolved": bool(_bound and _cap >= int(
                        globals().get("_MAX_TREES") or 8000))},
        "best_iteration": max(_bis) if _bis else None,
        "submodel_cache": model.submodel_cache_summary(),
    }
    _last_model = None if os.environ.get("MODELEVO_SMOKE_RUN") == "1" \
        else model
    return score, metrics_extra
# ==== v2.17.0 官方 hurdle 段 END ====
'''


def hurdle_trial_body() -> str:
    """TRIAL_BODY 的脚手架 fit_and_score 换成官方 hurdle 段(其余段落原样)。"""
    i = TRIAL_BODY.find("\ndef fit_and_score(")
    j = TRIAL_BODY.find("\n\n\ndef dump_eval_frame(", i)
    if i < 0 or j < 0:
        raise SystemExit("hurdle 模板渲染失败:TRIAL_BODY 锚点未找到(模板结构变更,"
                         "请同步 hurdle_trial_body)")
    return TRIAL_BODY[:i] + "\n" + HURDLE_TRIAL_ENTRIES.rstrip("\n") \
        + TRIAL_BODY[j:]


# ---------------------------------------------------------------------------
# v2.17.0(D2):baseline 血缘父代的 --set 消费点
# ---------------------------------------------------------------------------
# baseline 模板(runtime/baseline.py)的估计器由模块级
# `build_estimator(n_jobs, n_rows)` 构造,超参编译在代码里、不读
# config.hyperparameters → `--from-parent <baseline> --set k=v` 创建成功
# 却在 run 前被 lint 规则 11 拒跑。本段在子代副本 main 守卫前**包住**
# build_estimator:原样构造父代估计器,再以 config.hyperparameters 经
# set_params 覆写(与模板 _cfg_hp 同口径:只收标量;n_estimators 钳到
# MAX_TREES)。主轮与 multi-seed 变体轮(_fit_seed_variant)都经全局名
# build_estimator 取模型,故两处同时生效;父代冻结代码一字不改。
BASELINE_HP_BEGIN = "# ==== v2.17.0 baseline 超参消费段 BEGIN"
BASELINE_HP_END = "# ==== v2.17.0 baseline 超参消费段 END ===="
#: 可注入的 baseline 算法(估计器支持 sklearn set_params;mlp 包装器不支持)
BASELINE_HP_ALGOS = ("xgboost", "lightgbm")
BASELINE_HP_SEGMENT = r'''_MVO_BASE_BUILD_ESTIMATOR = build_estimator
#: 本段读到并已生效的 config 超参(metrics/日志留痕用)
_MVO_CFG_HP_APPLIED: dict = {}


def _mvo_cfg_hp() -> dict:
    """config.hyperparameters 的标量子集(与模板 _cfg_hp 同口径)。

    父代模板版本不一(旧 baseline 未必有 load_config):优先用它,
    否则直接读 TRIAL_DIR/config/trial.yaml。
    """
    _cfg_reader = globals().get("load_config")
    try:
        _cfg = (_cfg_reader() if callable(_cfg_reader) else
                (yaml.safe_load((TRIAL_DIR / "config" / "trial.yaml")
                                .read_text(encoding="utf-8")) or dict()))
    except Exception:  # noqa: BLE001 配置读失败 = 无覆写,不阻断训练
        return dict()
    _hp_raw = (_cfg or dict()).get("hyperparameters")
    if not isinstance(_hp_raw, dict):
        return dict()
    return {k: v for k, v in _hp_raw.items()
            if isinstance(v, (int, float, str, bool, type(None)))}


def build_estimator(n_jobs: int, n_rows: int = 0):  # noqa: F811 包住父代构造
    model = _MVO_BASE_BUILD_ESTIMATOR(n_jobs, n_rows)
    _cfg_hp = _mvo_cfg_hp()
    if not _cfg_hp:
        return model
    if "n_estimators" in _cfg_hp:
        try:
            _cfg_hp["n_estimators"] = max(1, min(int(float(
                _cfg_hp["n_estimators"])), 8000))
        except (TypeError, ValueError):
            _cfg_hp["n_estimators"] = 1
    if not hasattr(model, "set_params"):
        raise RuntimeError(
            "config.hyperparameters 非空但 baseline 估计器不支持 set_params"
            f"({type(model).__name__}),--set 无法生效")
    model.set_params(**_cfg_hp)
    if _cfg_hp != _MVO_CFG_HP_APPLIED:
        _MVO_CFG_HP_APPLIED.clear()
        _MVO_CFG_HP_APPLIED.update(_cfg_hp)
        print(f"[step] config hyperparameters 生效: {_cfg_hp}", flush=True)
    return model
'''
