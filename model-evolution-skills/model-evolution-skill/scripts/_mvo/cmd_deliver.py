"""交付链命令:finalize-retrain / finalize-baseline-score / evaluate-final / report / leaderboard / 盲测回滚

v2.11.9 从单文件 `scripts/modelevo.py` 拆出(拆分动机见
`_mvo/__init__.py`)。**本文件是机械搬运,未改语义** —— 便于对照原
实现审阅。新增命令请按职责放进对应模块,不要回填到 `modelevo.py`
(那里只留 argparse 与分发)。
"""
from __future__ import annotations

from runtime.confirmation import divergence_summary, recipe_root  # noqa: E402
from runtime.evaluator import FinalTestGate  # noqa: E402
from runtime.leaderboard import Leaderboard  # noqa: E402
from pathlib import Path
from runtime.report import ReportBuilder  # noqa: E402
from runtime.trial import (  # noqa: E402
    TrialFactory, trial_currently_failed, trial_is_screening)

from runtime.logging_utils import (dump_json, dump_yaml, get_logger,  # noqa: E402
                                   load_json, load_yaml, locked_state_write,
                                   now_iso)
import inspect
import json
import os
import sys
from .context import _load_config, _load_spec, _resolve_splits_dir, _run_dir_from, _update_run_state, logger


def _delivery_divergence_check(lb, trials, champion_id: str) -> tuple[bool, dict]:
    """v2.14.0:以 champion_id 作交付位时,是否与搜索冠军配方分叉超出容差。

    返回 (ok, info)。ok=False ⇒ 调用方必须拒绝,除非用户显式
    ``--ack-divergence`` 裁决(留痕)。与 Leaderboard.delivery_resolution
    同源:搜索冠军、配方根、容差都从那里取。
    """
    res = lb.delivery_resolution(trials)
    by_id = {t.trial_id: t for t in trials}
    info: dict = {"resolution": res, "champion": champion_id}
    champ = by_id.get(champion_id)
    if champ is None or res.get("search_champion") is None:
        return True, info
    rec = recipe_root(champ, by_id)
    info["champion_recipe"] = rec
    if rec == res.get("search_recipe"):
        info["bound"] = True
        return True, info
    value = ((champ.data.get("evaluation") or {}).get("primary_metric")
             or {}).get("value")
    if value is None:
        return True, info
    gap = lb._gap(float(res["search_metric"]), float(value))
    info.update(bound=False, gap=gap, tolerance=res.get("tolerance"))
    return gap <= float(res.get("tolerance") or 0.0), info


def _record_divergence_ack(run_dir: Path, champion_id: str, info: dict,
                           reason: str, command: str) -> None:
    """交付位分叉的人工裁决留痕(run-state.delivery_divergence_acks)。"""
    try:
        st_path = run_dir / "run-state.json"
        with locked_state_write(st_path):
            st = load_json(st_path) if st_path.exists() else {}
            st.setdefault("delivery_divergence_acks", []).append({
                "champion": champion_id, "command": command,
                "search_champion": (info.get("resolution") or {}).get(
                    "search_champion"),
                "gap": info.get("gap"), "tolerance": info.get("tolerance"),
                "reason": reason, "at": now_iso()})
            dump_json(st_path, st)
    except Exception:  # noqa: BLE001
        pass


def _write_divergence_note(run_dir: Path, res: dict) -> Path | None:
    """final-model/DELIVERY-DIVERGENCE.md:交付位空缺/分叉的人读说明。

    放在 final-model/ 顶层而不是 unconfirmed/ 里 —— 旧的已确认交付包
    若还在 final-model/model/,读者第一眼就要知道它不是搜索冠军。
    """
    try:
        fm = run_dir / "final-model"
        fm.mkdir(parents=True, exist_ok=True)
        stale = res.get("stale_candidates") or []
        lines = [
            "# 交付位分叉(不是交付冠军的已确认模型不得静默交付)", "",
            f"- 状态: `{res.get('status')}`",
            f"- 搜索冠军: `{res.get('search_champion')}`(配方 "
            f"`{res.get('search_recipe')}`,{res.get('search_metric')})",
            f"- 容差: {res.get('tolerance')}(默认 MDE,且不低于 2×noise)",
            f"- 说明: {divergence_summary(res)}", "",
            "## 已确认但不属于冠军配方的 trial", "",
            "| trial | 配方 | 指标 | 落后冠军 |", "|---|---|---|---|",
            *[f"| {s['trial_id']} | {s['recipe']} | {s['metric']} | "
              f"{s['gap']:+.6f} |" for s in stale],
            "", "## 处置(需用户裁决,不走默认分支)", "",
            "1. 对搜索冠军配方补一次 confirm(`new-trial --from-parent "
            f"{res.get('search_recipe')} --action confirm`),通过后重跑"
            " finalize-retrain;",
            "2. 接受未确认交付:使用 `final-model/unconfirmed/`(无样本外估计);",
            "3. 明知落后仍交付某个已确认 trial:`finalize-retrain --champion "
            "<id> --ack-divergence \"<理由>\"`,evaluate-final 同样需要 "
            "`--ack-divergence`(留痕于 run-state.delivery_divergence_acks)。",
        ]
        p = fm / "DELIVERY-DIVERGENCE.md"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p
    except Exception:  # noqa: BLE001
        return None


def _confirmation_required(config) -> bool:
    """v2.15.3:evaluator.require_confirmation(默认 true)。

    false 时交付不再要求 fresh_confirmation.passed:无已确认交付位时直接
    以 accepted 搜索冠军重训并盲测,model_meta.delivery_status 记
    ``confirmation_waived``,留痕 run-state.shadow_waivers。只放开确认门;
    完整性、重训证明、分叉守卫与 final_test 单次 gate 照旧。
    读取放在 scripts 侧而非 runtime DEFAULTS:不改 harness_hash。
    """
    return bool(config.get("evaluator.require_confirmation", True))


def _record_shadow_waiver(run_dir: Path, champion_id: str,
                          kind: str) -> None:
    """冠军 shadow 缺口的留痕(v18):审计链可见,不静默。"""
    try:
        st_path = run_dir / "run-state.json"
        with locked_state_write(st_path):
            st = load_json(st_path) if st_path.exists() else {}
            waivers = st.setdefault("shadow_waivers", [])
            waivers.append({"champion": champion_id, "kind": kind,
                            "at": now_iso()})
            dump_json(st_path, st)
    except Exception:  # noqa: BLE001
        pass


def _finalize_load_module(path: Path, name: str):
    """加载 trial code/train.py 为模块(与 runner 同款 cwd 无关加载)。

    v2.10.1/P3:加载后显式注册 sys.modules[name]。joblib.dump(model)
    序列化时按 type(model).__module__ 反查类定义;若模块只经
    importlib 加载而未注册进 sys.modules,pickle 的 save_global 会以
    "import of module failed" 拒绝闭包内或模块顶层的自定义 estimator
    子类(实测冠军 trial-004 的 _VerboseSafeLGBM/_AbsorbsVerboseLGBM)。
    注册后 pickle 经 sys.modules 定位到同名模块,即可按
    <module>.<ClassName> 找到并反序列化。
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _finalize_threads() -> str:
    """finalize 的缺省线程数(v2.13.5)。旧实现固定 16 —— 128 核机器上
    baseline LightGBM(CPU)train-only 重训 41 min、只用 1/8 的核。现取
    MODELEVO_FINALIZE_THREADS > min(64, 核数-8)(下限 16);finalize-all
    并行两路时由调用方预置 OMP_NUM_THREADS 对半分(setdefault 不覆写)。"""
    env = os.environ.get("MODELEVO_FINALIZE_THREADS")
    if env and env.strip().isdigit():
        return env.strip()
    return str(max(16, min(64, (os.cpu_count() or 24) - 8)))


def _clear_early_stopping(model) -> str | None:
    """按 estimator 自己的键关闭早停,返回实际使用的键。

    与 :func:`_freeze_rounds` 同一类问题:catboost 把 ``od_wait`` /
    ``early_stopping_rounds``(以及 ``od_type``)当同义参数。模板/HPO 构造时
    写的是 ``od_wait=50``,重拟合前无条件 ``set_params(early_stopping_rounds
    =None)`` 会让两个键同时存在,``fit`` 抛
    ``CatBoostError: only one of the parameters od_wait, early_stopping_rounds
    should be initialized``(实测 2026-11-03:creditcard / amazon 的 catboost
    冠军在轮数冲突修好后紧接着撞上这一条)。

    只设该实例 ``get_params()`` 里已有的键;catboost 的 ``od_wait=None``
    显式关闭早停,语义与 ``early_stopping_rounds=None`` 一致。复合容器等
    没有早停语义的对象返回 None,调用方保持原行为(忽略异常)。
    """
    try:
        current = dict(model.get_params() or {})
    except Exception:  # noqa: BLE001
        current = {}
    used = None
    for key in ("od_wait", "early_stopping_rounds"):
        if key in current and current.get(key) is not None:
            try:
                model.set_params(**{key: None})
                used = key
                break
            except Exception:  # noqa: BLE001
                continue
    # catboost 的 use_best_model 依赖早停验证集:训练集全量重拟合时没有
    # eval_set,留着 True 会抛 "To employ param {'use_best_model': True}
    # provide non-empty 'eval_set'"。实测 2026-11-03 creditcard。它只接受
    # 布尔(null 会解析失败),所以显式置 False。
    if current.get("use_best_model"):
        try:
            model.set_params(use_best_model=False)
            used = used or "use_best_model"
        except Exception:  # noqa: BLE001
            pass
    if used:
        return used
    # 没有早停语义的 estimator 保持旧行为(多数会抛,被调用方吞掉)
    try:
        model.set_params(early_stopping_rounds=None)
    except Exception:  # noqa: BLE001
        pass
    return None


def _freeze_rounds(model, n: int) -> str | None:
    """按 estimator **自己的**轮数键冻结轮数,返回实际使用的键。

    catboost 把 ``iterations`` / ``n_estimators`` / ``num_boost_round`` /
    ``num_trees`` 当同义参数:构造时给了 ``iterations``(HPO best_params),
    再 ``set_params(n_estimators=...)`` 两个键就同时存在,``fit`` 直接抛
    ``CatBoostError: only one of the parameters iterations, n_estimators, ...``。
    实测 2026-11-02 冷启动批次:creditcard / amazon_employee_access 的
    catboost HPO 冠军在 finalize-retrain 崩于此(rc=2),三行 traceback 都停在
    ``model.fit(X_all, y_all)``,交付链中断、run 留在 STOPPED。

    旧实现无条件 ``set_params(n_estimators=...)``;本次改为先看该实例
    ``get_params()`` 里已经有哪个键,只设那一个。返回 None 表示该 estimator
    没有轮数语义(复合容器等),调用方保持原行为。
    """
    try:
        current = dict(model.get_params() or {})
    except Exception:  # noqa: BLE001  非 sklearn 协议对象
        current = {}
    # 只认 GBDT 的**树数**键。``max_iter`` 虽有少数 sklearn 估计器使用,
    # 但它是收敛上限(LogisticRegression / MLP),语义不是"最优树数",
    # 旧实现也从未冻结它 —— 保持原行为。
    for key in ("iterations", "n_estimators", "num_boost_round",
                "num_iterations"):
        if key in current:
            try:
                model.set_params(**{key: int(n)})
                return key
            except Exception:  # noqa: BLE001  键存在但不可写
                continue
    return None


def _finalize_build_estimator(champ, n_jobs: int, n_rows: int,
                              hp: dict | None):
    """冠军模块的 estimator 工厂适配(v2.11.3 API 偏斜修复)。

    finalize-retrain 期望冠军导出 `build_estimator(n_jobs, n_rows, hp=...)`,
    但 `runtime/templates.py` 渲染的 trial 代码只导出
    `_build_model(params, seed, _n_rows=0)` —— 两处符号不同名,CLI 对本仓库
    任何模板血缘的冠军都必然 AttributeError,交付链在写 model_meta 前中断
    (实测 drawdown-req-t7-v211;同族 example-seq-v3-4-2.8.6-v20 曾用
    logs/finalize/retrain_champion.py 降级绕过,该变通未回写 harness ⇒
    本次复发)。此处按血缘回落:`build_estimator` 优先,缺失走
    `_build_model`,并把 n_jobs 并入 params(模板 `_build_model` 的 base
    n_jobs 取自 `n_jobs()`,会被 `base.update(params)` 覆写,与 CLI 原有的
    线程分配口径一致)。与同函数内 v2.10.1/F2 对 prepare_features 返回形态
    的既有兼容处理同款式;两处符号都缺时显式报错,不做静默猜测。

    v2.13.2(实测 example-seq-v3-v2.13.0.1 finalize 复发):手写加宽血缘
    (gen-01 超级 trial)的 `_build_model` 是模板路径的四件套占位(无
    `_HPO_ALGO` 注入时 raise RuntimeError),真实训练走 `_build_xgb`
    (参数签名 `(params, seed, n_rows=0)`)。此时按 `_build_xgb` 回落,
    与 `_build_model` 同签名口径,避免交付链被模板占位符号炸断。
    """
    fn = getattr(champ, "build_estimator", None)
    if callable(fn):
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        model = fn(n_jobs, n_rows, hp=hp or None) if "hp" in params \
            else fn(n_jobs, n_rows)
        # baseline 血缘的 build_estimator 只有两个位置参数；config HPO/
        # --set 覆写仍须作用到重训模型，而不是因 API 形态不同被丢弃。
        if hp and "hp" not in params and hasattr(model, "set_params"):
            model.set_params(**hp)
        return model
    fn = getattr(champ, "_build_model", None)
    if not callable(fn):
        raise SystemExit(
            "冠军模块既无 build_estimator 也无 _build_model,无法重训:"
            f" {getattr(champ, '__file__', '?')}")
    params = dict(hp or {})
    params.setdefault("n_jobs", n_jobs)
    if not getattr(champ, "_HPO_ALGO", None) and \
            getattr(champ, "_build_xgb", None) and \
            callable(getattr(champ, "_build_xgb")):
        # 手写加宽血缘:模板占位 _build_model 在无 _HPO_ALGO 时必然 raise,
        # 真实训练口径 = _build_xgb(params, seed, n_rows)(trial-008 家族)。
        fn = getattr(champ, "_build_xgb")
    return fn(params, int(getattr(champ, "SEED", 20260814) or 20260814),
              n_rows)


#: 第三方模型库根模块;不在其中的模块名(搜索期的 `__main__`、交付沙箱的
#: `champ_train` 等)视为 trial 代码自定义的模型容器
_MODEL_LIBRARY_ROOTS = frozenset({
    "xgboost", "lightgbm", "catboost", "sklearn", "torch", "keras",
    "tensorflow"})


def _model_family(model) -> str:
    """模型所属库(xgboost/lightgbm/catboost/sklearn...),用于算法族一致性校验。

    v2.16.3:trial 代码自定义的复合容器(如两段式 HurdleXGB)按"类名 + 子模型
    所属库"取族。容器类在搜索期定义于 `__main__`,交付沙箱从配方副本以
    `champ_train` 加载,按模块名比较会把同一配方误判为换族而拒绝落盘
    (run4 regression D-013)。
    """
    root = str(type(model).__module__ or "").split(".")[0]
    if root in _MODEL_LIBRARY_ROOTS:
        return root
    parts = sorted({
        _model_family(v) for v in getattr(model, "__dict__", {}).values()
        if str(type(v).__module__ or "").split(".")[0] in _MODEL_LIBRARY_ROOTS})
    return f"custom:{type(model).__qualname__}[{','.join(parts)}]"


def _fitted_rounds(model) -> int | None:
    """已拟合 GBM 的实际轮数(与各库 best_iteration 的索引口径无关)。"""
    fam = _model_family(model)
    try:
        if fam == "xgboost":
            return int(model.get_booster().num_boosted_rounds())
        if fam == "lightgbm":
            return int(model.booster_.current_iteration())
        if fam == "catboost":
            return int(model.tree_count_)
    except Exception:  # noqa: BLE001
        return None
    return None


def _reproduce_champion_fit(champ, X_tr, y_tr, X_va, y_va, seed, cfg,
                            champ_dir, spec, tol: float = 5e-4):
    """用冠军**自己的** fit_and_score 在 train 上复现搜索期模型(v2.13.4)。

    背景(实测 req-t7-drawdown):`_finalize_build_estimator` 按符号名猜
    estimator —— 冠军在 fit_and_score 里内联构造 XGBClassifier、未导出
    build_estimator/_build_xgb 时,回落到模板 HPO 占位 `_build_model`,无
    `_HPO_ALGO` 即默认 LightGBM + 默认超参。交付的"重训模型"与冠军换了
    算法,model_meta 却按目录名写 algorithm=xgboost。确认门通过的正常交付
    路径同样命中。

    做法:训练代码直接来自冠军模块,算法/超参/早停口径由构造保证;再把
    复现出的 validation 分数与冠军落盘分数比对,由验证兜底。trial 工件
    写入被隔离到沙箱(旧模板的 dump_* 会写 TRIAL_DIR;新模板在无
    MODELEVO_TRIAL_EXEC 时本就拒写)。

    返回 (train-only 复现模型 或 None, 复现报告 dict)。
    """
    import shutil
    import tempfile

    import numpy as np
    import pandas as pd

    from runtime.metrics import compute_metric

    _cfg = dict(cfg or {})
    _cfg.pop("multi_seed", None)
    _orig_td = getattr(champ, "TRIAL_DIR", None)
    _sandbox = Path(tempfile.mkdtemp(prefix="mvo-finalize-sandbox-"))
    try:
        champ.TRIAL_DIR = _sandbox
        champ._last_model = None
        score, _ = champ.fit_and_score(X_tr, y_tr, X_va, y_va, seed, _cfg)
        m_ref = getattr(champ, "_last_model", None)
    finally:
        if _orig_td is not None:
            champ.TRIAL_DIR = _orig_td
        shutil.rmtree(_sandbox, ignore_errors=True)
    report = {"builder": "champion_fit_and_score", "passed": False,
              "tolerance": tol}
    if m_ref is None:
        report["reason"] = "fit_and_score 未设置 _last_model"
        return None, report
    ref_path = champ_dir / "validation_score.parquet"
    if not ref_path.exists():
        report["reason"] = "冠军无 validation_score.parquet,无法校验复现"
        return m_ref, report
    ref = pd.read_parquet(ref_path)["score"].to_numpy(dtype="float64")
    cur = np.asarray(score, dtype="float64")
    if len(ref) != len(cur):
        report["reason"] = f"复现分数行数 {len(cur)} != 冠军 {len(ref)}"
        return m_ref, report
    yv = np.asarray(y_va)
    m_trial = float(compute_metric(spec.primary_metric, yv, ref, score=ref))
    m_repro = float(compute_metric(spec.primary_metric, yv, cur, score=cur))
    report.update({
        "metric": spec.primary_metric,
        "validation_metric_trial": m_trial,
        "validation_metric_repro": m_repro,
        "validation_max_abs_score_diff": float(np.max(np.abs(ref - cur))),
        "passed": bool(abs(m_repro - m_trial) <= tol),
    })
    return m_ref, report


# ==================== v2.15.9 交付重训复跑配方 ====================
# 动机(req-t7 run4 regression,2026-10-21 实测):旧交付链在 train 上复现
# 冠军(约 40 min)后 `sklearn.clone` 出副本、冻结 train-only 早停轮数,再
# 在 train+validation 重拟合。这把"交付模型必须是可 clone 的单一 GBM"当
# 成隐含契约 —— lint/trial/confirm 都不检查,首次暴露在交付最后一步;两段式
# hurdle 容器在此直接 TypeError,而它恰是搜索冠军。
# 新口径:复制冠军 code+config 到交付目录,以 train+validation 为训练集
# **重跑冠军自己的配方**,早停落在 train+validation 的最新尾部(模板
# `_fit_with_es` 与手写 `_fit_hurdle_refit_full` 都按传入 X_tr 的尾部
# ES_FRACTION 切)。v2.16 起配方为单段拟合(头部拟合、尾部早停、只训一次,
# 不再全量重拟合;v2.16.3 起 model_meta.retrain_scope 按遥测如实描述)。
# 代码字节不变 ⇒ 配方可审计;不依赖
# clone ⇒ 复合容器可用。
REFIT_MODES = ("rerun_recipe", "frozen_rounds")


def _trial_es_source(trial_dir: Path) -> str:
    """v2.16.0:trial 实际采用的早停面(validation | train_tail)。

    以 metrics.json 的 es_info.es_source 为准(模板单段拟合与 baseline 模板
    都会写);缺字段 = v2.16.0 之前的 trial(两段式 / train 尾部早停)。
    """
    try:
        m = json.loads((Path(trial_dir) / "metrics.json")
                       .read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return "train_tail"
    es = m.get("es_info") or ((m.get("refit") or {}).get("es_info")) or {}
    return "validation" if str(es.get("es_source") or "") == "validation" \
        else "train_tail"


def _resolve_refit_for_champion(args, refit_mode: str, champ_es: str) -> str:
    """v2.16.0:validation 早停的冠军交付一律 frozen_rounds。

    交付训练集 = train+validation,validation 已进训练集,不能再当早停集;
    rerun_recipe 让冠军代码在 X_va 上早停 = 样本内早停。冠军早停面是
    validation 时,配置级 rerun_recipe 自动改为 frozen_rounds;CLI 显式
    --refit-mode rerun_recipe 直接拒绝。
    """
    if champ_es != "validation":
        return refit_mode
    if getattr(args, "refit_mode", None) == "rerun_recipe":
        raise SystemExit(
            "冠军在 outer validation 上早停(v2.16.0 单段拟合):rerun_recipe "
            "会把 validation 同时当训练集与早停集(样本内早停)。去掉 "
            "--refit-mode 以使用 frozen_rounds")
    return "frozen_rounds"


def _validation_es_rounds(model) -> int | None:
    """v2.16.0:单段早停模型的最优树数(1-based)。

    单段拟合的模型里还留着早停耐心那段树(xgboost num_boosted_rounds =
    best + 耐心),`_fitted_rounds` 会多算;优先取模板遥测,其次各库 best。
    """
    info = getattr(model, "mvo_es_info_", None) or {}
    if info.get("stage1_best_iteration"):
        return int(info["stage1_best_iteration"])
    fam = _model_family(model)
    if fam == "xgboost":
        bi = getattr(model, "best_iteration", None)
        if bi is not None:
            return int(bi) + 1
    if fam == "lightgbm":
        bi = getattr(model, "best_iteration_", None)
        if bi:
            return int(bi)
    return _fitted_rounds(model)


# ==================== v2.17.0 交付 refit 口径留痕 ====================
# 事故(run4 regression.2):冠军 hurdle 容器 es_source=validation → 自动切
# frozen_rounds → train 上完整复现 10.4 min → sklearn.clone 处 TypeError →
# 自动回落 train_tail 口径复跑配方。回落只在 finalize-retrain.log 里留了一行,
# model_meta 只写 refit_mode=rerun_recipe,generated_by 还是硬编码的 v2.15.9。
# v2.17.0:复现之前先用冠军 train-only 模型做 clone 探测;不可 clone 且冠军
# 早停面是 validation 时直接走回落路径(行为不变,不白跑复现),回落原因、
# 探测结果写进 model_meta / recipe / stdout 摘要。
def _skill_version() -> str:
    """交付产物的生成版本(读 runtime.__version__,不再硬编码)。"""
    try:
        from runtime import __version__
        return str(__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


# ==================== v2.19.0 finalize 复用冠军冻结模型 ====================
# 实测(8 个 run):frozen_rounds 交付先在 train 上完整复现冠军约 50 min,只为
# 拿到一个"与冠军一致"的 train-only 模型来 clone + 读最优轮数 —— 而冠军 trial
# 自己落盘的 model/model.joblib 就是那个模型。复用前必须证明一致性:把冻结
# 模型在**交付管线重建出的** validation 帧上重打分,与冠军落盘的
# validation_score.parquet 逐位比对(容差默认 1e-9)。这一步同时证明 ①冻结
# 模型 = 冠军搜索期模型、②交付特征管线(含 train 拟合的变换)与冠军一致
# (validation 特征依赖 train 侧拟合状态);比完整复现快两个数量级。
# 证明通过 → 跳过复现,后续 clone / 冻结轮数 / train+validation 重训口径
# 完全不变(轮数仍由 _validation_es_rounds(冻结模型) 取,与旧路径对复现模型
# 取值同一函数);证明不通过或工件缺失 → 退回完整复现(旧路径)。
# --full-reproduction 强制旧路径。留痕 model_meta.reproduction_mode。
REUSE_SCORE_ATOL = 1e-9


def _trial_style_score(model, X, model_type: str):
    """与 trial 落盘 validation_score 同口径的打分(binary 正类概率、
    multiclass 最大类概率、其余 predict)。xgboost 先把 booster 切到 CPU
    预测(与 trial 的 align_predict_device 同口径;只改 booster,不改
    sklearn 层 device 参数,clone 不受影响)。"""
    try:
        if _model_family(model) == "xgboost":
            model.get_booster().set_param({"device": "cpu"})
    except Exception:  # noqa: BLE001
        pass
    if model_type == "binary_classification":
        return model.predict_proba(X)[:, 1]
    if model_type == "multiclass":
        return model.predict_proba(X).max(axis=1)
    return model.predict(X)


def _reuse_candidate_dirs(run_dir: Path, champ_id: str, trials) -> list:
    """复用候选:冠军自身 → 其 confirm 子代(同配方复制代码,主 seed 模型)。"""
    out = [Path(run_dir) / "trials" / champ_id]
    for t in trials or []:
        try:
            if t.trial_id != champ_id \
                    and champ_id in (t.parent_ids or []) \
                    and str(t.data.get("search_action") or "") == "confirm":
                out.append(Path(run_dir) / "trials" / t.trial_id)
        except Exception:  # noqa: BLE001
            continue
    return out


def _frozen_model_reuse(champ_dirs, champ_module, X_va, spec, *,
                        atol: float = REUSE_SCORE_ATOL,
                        loader=None) -> tuple:
    """尝试复用冻结模型代替 train 上的完整复现(见段首)。

    返回 (模型 或 None, 报告 dict)。报告 passed=True 时调用方跳过复现。
    """
    import numpy as np
    import pandas as pd

    mt = str(getattr(spec, "model_type", "") or "")
    report: dict = {"mode": "frozen_model_rescore", "passed": False,
                    "atol": atol, "attempts": []}
    if mt == "uplift":
        report["reason"] = "uplift 不支持复用(双打分口径)"
        return None, report
    _load = loader or _load_champion_model
    for d in champ_dirs:
        d = Path(d)
        att: dict = {"trial_dir": d.name, "passed": False}
        report["attempts"].append(att)
        ref_path = d / "validation_score.parquet"
        if not ref_path.exists():
            att["reason"] = "缺 validation_score.parquet"
            continue
        model, err = _load(d, champ_module)
        if model is None:
            att["reason"] = f"冻结模型不可加载:{err}"
            continue
        if hasattr(model, "score_validation") \
                or not callable(getattr(model, "predict", None)):
            att["reason"] = "非 predict 协议模型(torch 等),不复用"
            continue
        try:
            ref = pd.read_parquet(ref_path)["score"].to_numpy(dtype="float64")
            cur = np.asarray(_trial_style_score(model, X_va, mt),
                             dtype="float64")
        except Exception as exc:  # noqa: BLE001
            att["reason"] = f"重打分失败:{type(exc).__name__}: {exc}"
            continue
        if len(ref) != len(cur):
            att["reason"] = f"行数 {len(cur)} != 落盘 {len(ref)}"
            continue
        same_nan = bool(np.array_equal(np.isnan(ref), np.isnan(cur)))
        diff = np.abs(np.nan_to_num(ref) - np.nan_to_num(cur))
        mx = float(diff.max()) if len(diff) else 0.0
        att.update({"max_abs_score_diff": mx,
                    "n_exact_equal": int(np.sum(ref == cur)),
                    "n_rows": int(len(ref))})
        if not same_nan or mx > atol:
            att["reason"] = "冻结模型重打分与落盘 validation 分数不一致"
            continue
        att["passed"] = True
        try:
            _m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            att["trial_best_iteration"] = (_m or {}).get("best_iteration")
        except (OSError, ValueError):
            pass
        report.update({"passed": True, "source_trial": d.name,
                       "max_abs_score_diff": mx,
                       "builder": "frozen_model_reuse"})
        return model, report
    report["reason"] = "无可复用的冻结模型,退回 train 上完整复现"
    return None, report


def _align_reused_runtime_params(model, champ_module, n_rows: int) -> dict:
    """复用冻结模型 clone 后,把**运行环境类**参数对齐到旧路径(冠军代码在
    交付进程里新建估计器)会取的值:线程数(OMP_NUM_THREADS)与 xgboost
    device(冠军 _gpu_allocated(n_rows))。配方参数(轮数/lr/深度/seed…)
    不动。返回实际改动的参数。"""
    changed: dict = {}
    try:
        params = model.get_params(deep=False)
    except Exception:  # noqa: BLE001
        return changed
    try:
        _nj = max(1, int(os.environ.get("OMP_NUM_THREADS", "0") or 0))
    except ValueError:
        _nj = 0
    if _nj and "n_jobs" in params and params.get("n_jobs") != _nj:
        changed["n_jobs"] = _nj
    if _model_family(model) == "xgboost" and "device" in params:
        _ga = getattr(champ_module, "_gpu_allocated", None)
        if callable(_ga):
            try:
                want = "cuda" if _ga(int(n_rows)) else "cpu"
            except Exception:  # noqa: BLE001
                want = None
            cur = str(params.get("device") or "cpu")
            if want and cur.split(":")[0] != want:
                changed["device"] = want
    if changed:
        try:
            model.set_params(**changed)
        except Exception:  # noqa: BLE001
            return {}
    return changed


def _load_champion_model(champ_dir: Path, champ_module=None):
    """以最小代价反序列化冠军 train-only 搜索模型(clone 探测用,v2.17.0)。

    trial 以脚本执行,`model/model.joblib` 里的自定义容器按
    `__main__.<Class>` pickle;交付进程的 `__main__` 是 modelevo.py,直接
    joblib.load 必然 AttributeError。加载期间把 `sys.modules['__main__']`
    临时指向已加载的冠军模块(交付配方副本,与冠军源码逐字节一致),加载完
    立即恢复。返回 (模型 或 None, 失败原因 或 None)。
    """
    path = Path(champ_dir) / "model" / "model.joblib"
    if not path.exists():
        return None, "冠军无 model/model.joblib"
    import joblib

    _main_prev = sys.modules.get("__main__")
    try:
        if champ_module is not None:
            sys.modules["__main__"] = champ_module
        return joblib.load(path), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if _main_prev is not None:
            sys.modules["__main__"] = _main_prev


def _clone_probe(model) -> dict:
    """对模型对象做 sklearn.clone 探测;返回 {status: ok|error, error}。"""
    from sklearn.base import clone

    try:
        clone(model)
    except Exception as exc:  # noqa: BLE001  TypeError 为主,其余同样不可 clone
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ok", "error": None}


def _probe_champion_clone(champ_dir: Path, champ_module=None) -> dict:
    """frozen_rounds 前置的冠军 clone 探测(v2.17.0)。

    status:ok(可 clone)/ error(clone 抛错,error 为异常文本)/
    unavailable(取不到模型对象,调用方退回"复现后再 clone"的旧顺序)。
    """
    rel = "model/model.joblib"
    model, err = _load_champion_model(champ_dir, champ_module)
    if model is None:
        return {"status": "unavailable", "error": err, "source": rel,
                "model_class": None}
    if not callable(getattr(model, "predict", None)):
        return {"status": "unavailable", "source": rel,
                "error": f"model.joblib 不是可 predict 的模型({type(model).__name__})",
                "model_class": f"{type(model).__module__}.{type(model).__name__}"}
    out = _clone_probe(model)
    out.update({"source": rel,
                "model_class": f"{type(model).__module__}.{type(model).__name__}"})
    return out


def _composite_frozen_params(model, champ_es: str) -> dict:
    """sklearn 协议复合容器(如 v2.17 官方 HurdleRegressor)的冻结轮数参数。

    约定(sklearn 惯例):段名 ``<name>`` 来自构造参数里的子估计器原型,或
    ``get_params(deep=True)`` 的 ``<name>__*`` 嵌套键(HurdleRegressor 在
    clf/reg=None 时按 clf_params/reg_params 构造段);拟合后的子估计器挂在
    ``<name>_``。逐段取最优树数(validation 早停取 best,不含早停耐心段;
    train_tail 取实际轮数),返回 set_params 用的嵌套键
    ``<name>__n_estimators`` 与 ``<name>__early_stopping_rounds=None``;
    容器自身的 ``early_stopping_rounds`` 同样关闭。单模型或取不到子估计器
    时返回 {}(调用方沿用单模型口径)。
    """
    try:
        params = model.get_params(deep=False)
    except Exception:  # noqa: BLE001
        return {}
    try:
        deep = model.get_params(deep=True)
    except Exception:  # noqa: BLE001
        deep = dict(params)
    names = [k for k, v in params.items()
             if v is not None and hasattr(v, "get_params")]
    names += sorted({k.split("__", 1)[0] for k in deep if "__" in k}
                    - set(names))
    out: dict = {}
    for name in names:
        fitted = getattr(model, f"{name}_", None)
        if fitted is None or not hasattr(fitted, "get_params"):
            continue
        rounds = (_validation_es_rounds(fitted) if champ_es == "validation"
                  else _fitted_rounds(fitted))
        try:
            sub = fitted.get_params(deep=False)
        except Exception:  # noqa: BLE001
            sub = {}
        if rounds and ("n_estimators" in sub
                       or f"{name}__n_estimators" in deep):
            out[f"{name}__n_estimators"] = int(rounds)
        if "early_stopping_rounds" in sub \
                or f"{name}__early_stopping_rounds" in deep:
            out[f"{name}__early_stopping_rounds"] = None
    if out and "early_stopping_rounds" in params:
        out["early_stopping_rounds"] = None
    return out


def _fit_baseline_train_tail(bmodel, base, btr, y_b) -> int:
    """train_tail 口径的 baseline 对照拟合:train 尾部早停定轮 → train-only
    全量重拟合(v2.16.0 前的唯一口径,原样抽出)。返回轮数。"""
    es_frac = float(getattr(base, "ES_FRACTION", 0.1) or 0.1)
    bcut = int(len(btr) * (1.0 - es_frac))
    _base_es_kw = _finalize_es_fit_kwargs(
        bmodel, btr.iloc[bcut:], y_b.iloc[bcut:],
        int(getattr(base, "EARLY_STOP_ROUNDS", 50) or 50))
    if "eval_set" in _base_es_kw:
        bmodel.fit(btr.iloc[:bcut], y_b.iloc[:bcut], **_base_es_kw)
    else:
        bmodel.fit(btr, y_b, **_base_es_kw)
    bbest = _finalize_best_iteration(bmodel)
    _baseline_n_est = bbest or int(
        getattr(bmodel, "n_estimators", None)
        or getattr(base, "N_ESTIMATORS_CAP", 2000))
    if bbest:
        _freeze_rounds(bmodel, bbest)
    # [FIX-2026-10-12] 第二次"train-only 全量重拟合"前必须清掉构造期
    # 早停参数 —— 冠军重训路径早已这么做(cmd_deliver.py 的
    # `model.set_params(early_stopping_rounds=None)`),基线路径漏了,
    # 于是 xgboost 基线在这里抛
    # `ValueError: Must have at least 1 validation dataset for early
    # stopping`(实测 req-t7-rand400w run5:gen-00 三个 trial 全是 xgboost,
    # `finalize-baseline-score` rc=2 崩在 ~50 min 处,交付报告因此失去
    # 样本外口径的 baseline 对照,只剩 validation 口径)。
    # 历史 run 未暴露该缺陷:它们的 gen-00 基线是 lightgbm,其 fit 在无
    # eval_set 时忽略 callbacks。
    # 放在 set_params(n_estimators=...) 之后、fit 之前:`_finalize_
    # best_iteration` 已在上一段带 eval_set 的拟合里定下轮数,本段是
    # 纯重拟合,不需要也无法提供早停验证集。
    _clear_early_stopping(bmodel)     # 按 estimator 自己的键(catboost: od_wait)
    bmodel.fit(btr, y_b)
    return _baseline_n_est


def _delivery_predict(model, X, spec):
    """交付打分按任务类型分派(v2.15.9)。

    旧实现按 duck typing 优先 predict_proba:回归任务的复合容器只要暴露
    predict_proba(如两段式的正类概率),交付分数就会静默变成概率 —— 与
    lint 规则 4 对 trial 的要求自相矛盾。regression 一律 predict;
    分类取正类概率;其余(uplift 等)沿用旧口径。
    """
    mt = str(getattr(spec, "model_type", "") or "")
    if mt == "regression" or not hasattr(model, "predict_proba"):
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def _metric_panel(spec, y, score, *, treatment=None) -> dict:
    """主指标 + secondary_metrics 一次算齐(v2.15.9;单指标失败记 None)。"""
    from runtime.metrics import compute_metric

    _secondary = ((getattr(spec, "data", None) or {}).get("secondary_metrics")
                  or [])
    names = [spec.primary_metric] + [
        m for m in _secondary if m != spec.primary_metric]
    out: dict = {}
    for name in names:
        try:
            out[name] = float(compute_metric(name, y, score, score=score,
                                             treatment=treatment))
        except Exception:  # noqa: BLE001  次指标不适用于本任务时不阻塞
            if name == spec.primary_metric:
                raise
            out[name] = None
    return out


def _load_aligned_score(path: Path, keys):
    """读带 sidecar 的预测产物并按样本键对齐(完整性校验失败即抛)。"""
    import pandas as pd

    from runtime.data import SAMPLE_KEY_COLUMN
    from runtime.integrity import verify_prediction_artifact
    from runtime.metrics import align_score_by_keys

    verify_prediction_artifact(path)
    df = pd.read_parquet(path)
    return align_score_by_keys(df["score"].values, keys,
                               df[SAMPLE_KEY_COLUMN].values)


def _raw_scale_report(spec, y_log, score_log, *, out_dir: Path, n_bins: int,
                      split: str, best_trial=None) -> dict:
    """log1p 目标的原值尺度(expm1)分桶报表 + 整体指标(v2.15.9)。

    分桶仍按 log 预测排序(expm1 单调,桶划分与官方口径一致);桶内统计
    改用原值:y_true=expm1(y)、y_pred=expm1(clip(score,0))。复用
    runtime.segment_report 的回归列集(预测/真实均值、Bias、MAE、RMSE、
    中位数、最大/最小、label 求和/占比/累计),落 out_dir/segment-report.*。
    """
    import numpy as np
    from scipy.stats import spearmanr

    from runtime.segment_report import build_segment_report

    y_raw = np.expm1(np.asarray(y_log, dtype=float))
    clipped = np.clip(np.asarray(score_log, dtype=float), 0.0, None)
    p_raw = np.expm1(clipped)
    rep = build_segment_report(
        y_raw, np.asarray(score_log, dtype=float), spec, n_bins=n_bins,
        y_pred=p_raw, out_dir=out_dir,
        extra_summary={"split": split, "best_trial": best_trial,
                       "scale": "raw(expm1)"},
        label=f"{split} 原值尺度(expm1)")
    d = p_raw - y_raw
    total = float(y_raw.sum())
    return {
        "n": int(len(y_raw)),
        "n_negative_pred_clipped": int((np.asarray(score_log) < 0).sum()),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "mae": float(np.mean(np.abs(d))),
        "mean_bias": float(d.mean()),
        "pred_mean": float(p_raw.mean()),
        "true_mean": float(y_raw.mean()),
        "sum_ratio": float(p_raw.sum() / total) if total else None,
        "spearman": float(spearmanr(p_raw, y_raw).correlation),
        "segment_report": (str(Path("segment-raw") / split)
                           if rep is not None else None),
    }


# ==================== v2.17.0 回归交付诊断(#20)====================
# 事故(run4 regression.2):final_test 交付模型预测均值 ≈2350,真实 ≈1965
# (+20%,7 月水位下移);shadow 预检与 evaluate-final 都没有水位信号。
# 下列 helper 只产出统计量(均值/比值/分期汇总),绝不返回逐行数组 ——
# final_test 标签只在 evaluate-final 的 gate 内同一次内存读取中使用。
def _period_labels(values):
    """identity 时间坐标 → 期标签(与 y 同长的 str 数组;取不到返回 None)。

    ≤24 个不同取值按原值(日期规范为 YYYY-MM-DD);更多时按月,只剩 1 个月
    时按 ISO 周。整数日期(20260707)按 %Y%m%d 解析。
    """
    import numpy as np
    import pandas as pd

    if values is None:
        return None
    s = pd.Series(np.asarray(values)).reset_index(drop=True)
    if not len(s):
        return None
    dt = None
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = s
        elif pd.api.types.is_integer_dtype(s):
            if s.between(19000101, 21001231).all():
                dt = pd.to_datetime(s.astype(str), format="%Y%m%d",
                                    errors="coerce")
        elif not pd.api.types.is_numeric_dtype(s):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(s, errors="coerce")
    except Exception:  # noqa: BLE001  解析失败按原值分期
        dt = None
    if dt is None or dt.notna().mean() < 0.99:
        return s.astype(str).to_numpy()
    if dt.nunique() <= 24:
        return dt.dt.strftime("%Y-%m-%d").fillna("NA").to_numpy()
    month = dt.dt.strftime("%Y-%m").fillna("NA")
    if month.nunique() >= 2:
        return month.to_numpy()
    return dt.dt.strftime("%G-W%V").fillna("NA").to_numpy()


def _delivery_regression_diagnostics(faces: dict, *,
                                     period_column: str | None = None) -> dict:
    """交付期回归诊断:各评估面 × 各模型的整体水位与分期水位(v2.17.0)。

    ``faces`` = {split: {"y": 标签, "periods": 期标签或 None,
    "models": {"delivery": 预测, "baseline": 预测, "search_model": 预测}}}。
    交付模型额外给出 vs 基线的 rmse/mae 背离(``vs_baseline_divergence``)。
    只返回统计量;任一模型缺失/行数不符即跳过该模型。
    """
    from runtime.regression_diagnostics import (LEVEL_RATIO_WARN,
                                                LEVEL_RATIO_Z,
                                                regression_diagnostics)

    out: dict = {"scale": "label", "level_ratio_warn": LEVEL_RATIO_WARN,
                 # v2.17.0(D8):告警判据 |ratio−1| > max(warn, z·SE)
                 "level_ratio_z": LEVEL_RATIO_Z,
                 "period_column": period_column}
    flagged: list[str] = []
    overall_flagged: list[str] = []
    period_flagged: list[str] = []
    for split, face in (faces or {}).items():
        y = face.get("y")
        if y is None or not len(y):
            continue
        n = len(y)
        per = face.get("periods")
        per = per if per is not None and len(per) == n else None
        models = {k: v for k, v in (face.get("models") or {}).items()
                  if v is not None and len(v) == n}
        if not models:
            continue
        base = models.get("baseline")
        sd: dict = {}
        for name, pred in models.items():
            d = regression_diagnostics(
                y, pred, periods=per,
                parent_pred=base if name == "delivery" else None)
            if "vs_parent_divergence" in d:
                d["vs_baseline_divergence"] = d.pop("vs_parent_divergence")
            sd[name] = d
        out[split] = sd
        dl = sd.get("delivery") or {}
        _ov = bool((dl.get("level") or {}).get("level_shift_warning"))
        _pv = bool(dl.get("period_level_shift"))
        if _ov:
            overall_flagged.append(split)
        if _pv:
            period_flagged.append(split)
        if _ov or _pv:
            flagged.append(split)
    out["level_shift_warning"] = bool(flagged)
    out["level_shift_splits"] = flagged
    # v2.17.0(D8):整体水位与分期水位告警分开(报告 §12 分述)
    out["overall_level_shift_splits"] = overall_flagged
    out["period_level_shift_splits"] = period_flagged
    _adv: list[str] = []
    if overall_flagged:
        _adv.append(
            f"整体水位:交付模型预测均值偏离真实均值超过 max(±{LEVEL_RATIO_WARN:.0%}, "
            f"{LEVEL_RATIO_Z:g}·SE)({'/'.join(overall_flagged)}):排序可用、"
            "绝对金额需校准(按最新期水位比缩放或做 isotonic/线性校准后再用于金额口径)")
    if period_flagged:
        _adv.append(
            f"分期水位:部分期的水位比超出噪声带({'/'.join(period_flagged)}"
            "):检查对应期的业务水位变化;整体未告警时属局部偏离")
    out["advice"] = ";".join(_adv) if _adv else None
    return out


def _level_hint_lines(diag: dict) -> list[str]:
    """回归水位偏离的 stdout 提示行(无告警时为空;整体/分期分开表述)。"""
    lines: list[str] = []
    _ov = set((diag or {}).get("overall_level_shift_splits") or [])
    _pv = set((diag or {}).get("period_level_shift_splits") or [])
    if not _ov and not _pv:
        # 旧产物(无分述字段):按 level_shift_splits 推断
        for split in (diag or {}).get("level_shift_splits") or []:
            dl = ((diag.get(split) or {}).get("delivery") or {})
            if (dl.get("level") or {}).get("level_shift_warning"):
                _ov.add(split)
            if dl.get("period_level_shift"):
                _pv.add(split)
    for split in (diag or {}).get("level_shift_splits") or []:
        dl = ((diag.get(split) or {}).get("delivery") or {})
        lv = dl.get("level") or {}
        ratio = lv.get("level_ratio")
        se = lv.get("level_ratio_se")
        _desc = (f"预测均值 {lv.get('pred_mean', float('nan')):.6g}"
                 f" vs 实际 {lv.get('y_mean', float('nan')):.6g}"
                 + (f"(水位比 {ratio:.3f}" if isinstance(ratio, float) else "")
                 + (f",SE {se:.3f}" if isinstance(se, float) else "")
                 + (")" if isinstance(ratio, float) else ""))
        parts: list[str] = []
        if split in _ov:
            parts.append(f"整体水位偏离:交付模型{_desc} —— 排序可用、绝对金额需"
                         "校准")
        bad = [r for r in dl.get("level_by_period") or []
               if r.get("level_shift_warning")]
        if split in _pv and bad:
            parts.append(
                "分期水位偏离(逐期独立判据 max(固定阈, z·SE)):" + ", ".join(
                    f"{r.get('period')}={r.get('level_ratio'):.3f}"
                    + (f"±{r.get('level_ratio_se'):.3f}"
                       if isinstance(r.get("level_ratio_se"), float) else "")
                    for r in bad[:6] if isinstance(r.get("level_ratio"), float))
                + ("(整体水位在阈内,属局部偏离;偏离期的绝对金额需校准)"
                   if split not in _ov else ""))
        if parts:
            lines.append(f"[回归水位] {split}: " + ";".join(parts)
                         + "(明细见 final-model/evaluation.json "
                         "regression_diagnostics)")
    return lines


def _refit_mode(args, config) -> str:
    """交付重训口径:CLI --refit-mode > config.finalize.refit_mode > 默认 rerun_recipe。

    读取放在 scripts 侧(同 `_confirmation_required`):不改 harness_hash。
    """
    mode = getattr(args, "refit_mode", None) or config.get(
        "finalize.refit_mode", "rerun_recipe")
    mode = str(mode or "rerun_recipe")
    if mode not in REFIT_MODES:
        raise SystemExit(
            f"未知 refit_mode={mode!r},可选 {list(REFIT_MODES)}")
    return mode


def _copy_delivery_recipe(champ_dir: Path, dst: Path) -> tuple[Path, dict]:
    """把冠军 trial 的 code/ + config/ 字节级复制到交付配方目录(v2.15.9)。

    复制后逐文件比对 sha256:副本是交付口径的唯一执行源,必须与冠军
    trial 源码逐字节一致 —— 否则审计链断开。dst 与 `trials/<id>/` 同深度
    (`<run>/final-model/<name>/`),旧模板以 `__file__.parents[3]` 推导
    RUN_DIR 的 trial 代码在副本里同样解析到 run 根。
    返回 (副本目录, {相对路径: sha256})。
    """
    import hashlib
    import shutil

    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    shas: dict[str, str] = {}
    for sub in ("code", "config"):
        src = champ_dir / sub
        if not src.exists():
            continue
        tgt = dst / sub
        shutil.copytree(src, tgt)
        for path in sorted(tgt.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            rel = f"{sub}/{path.relative_to(tgt).as_posix()}"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            shas[rel] = digest
            origin = champ_dir / rel
            if origin.exists() and hashlib.sha256(
                    origin.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"交付配方副本与冠军源码不一致:{rel}")
    if not (dst / "code" / "train.py").exists():
        raise RuntimeError(f"交付配方副本缺 code/train.py:{dst}")
    return dst, shas


def _es_holdout_window(run_dir: Path, splits: Path, n_train: int, n_val: int,
                       es_frac: float) -> dict:
    """train+validation 尾部早停面的 pday 窗口(v2.15.9,取不到则 null)。

    split 产物按 pday 单调落盘(finalize 实测),故尾部行 = 时间上最新
    的一批样本。这里只读 identity 面的 pday 列,不产生任何训练信息。
    """
    out: dict = {"es_fraction": float(es_frac), "n_rows": None,
                 "first_pday": None, "last_pday": None, "source": None}
    try:
        import pandas as pd

        frames = []
        for split in ("train", "validation"):
            path = splits / "identity" / f"{split}.parquet"
            if not path.exists():
                return out
            frames.append(pd.read_parquet(path, columns=["pday"]))
        pday = pd.concat(frames, ignore_index=True)["pday"]
        if len(pday) != n_train + n_val:
            return out
        monotonic = bool(pday.is_monotonic_increasing)
        cut = int(len(pday) * (1.0 - es_frac))
        tail = pday.iloc[cut:]
        out.update({
            "n_rows": int(len(tail)),
            "first_pday": str(pd.Timestamp(tail.iloc[0]).date()),
            "last_pday": str(pd.Timestamp(tail.iloc[-1]).date()),
            "source": "identity/train+validation tail (monotonic by pday)"
            if monotonic else "identity/train+validation tail (NOT monotonic)",
        })
    except Exception:  # noqa: BLE001  遥测失败不阻塞交付
        pass
    return out


def _strip_hpo_cfg(cfg: dict, champ_dir: Path) -> tuple[dict, str | None]:
    """HPO 血缘的交付 cfg(v2.15.9):剥离 hpo 节、注入 best_params。

    cfg 含 hpo 节时调用 fit_and_score 会在交付期重跑整轮 inner 搜索并写
    hpo 产物;交付要的是**搜索产出的配方**(best_params = 重训配方:lr 收缩
    后的 lr、生效的 cap)。回落次序与 frozen 路径 hp_inj 一致:
    hpo-study.json → metrics.json;cfg 里人工 --set 的 hyperparameters 优先。
    非 HPO cfg 原样返回(浅拷贝)。
    """
    _cfg = dict(cfg or {})
    if not _cfg.get("hpo"):
        return _cfg, None
    for src in (champ_dir / "hpo-study.json", champ_dir / "metrics.json"):
        if not src.exists():
            continue
        try:
            blob = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        bp = (blob or {}).get("best_params") or {}
        if isinstance(bp, dict) and bp:
            best = {k: v for k, v in bp.items()
                    if isinstance(v, (int, float, str, bool))}
            _cfg.pop("hpo", None)
            _cfg["hyperparameters"] = {
                **best, **(_cfg.get("hyperparameters") or {})}
            return _cfg, src.name
    raise RuntimeError(
        "HPO 冠军缺 hpo-study.json/metrics.json 的 best_params,"
        "无法在交付期复跑配方(拒绝在交付时重跑搜索)")


def _rerun_champion_recipe(champ, X_all, y_all, X_va, y_va, seed, cfg,
                           champ_dir: Path) -> tuple[object, dict]:
    """在 train+validation 上重跑冠军自己的 fit_and_score(v2.15.9)。

    与 `_reproduce_champion_fit` 共用沙箱纪律(TRIAL_DIR 指向临时目录、
    清空 `_last_model`、pop multi_seed);区别是训练集为 train+validation,
    且**不**与冠军 validation 分数比对 —— 该证明由特征面一致性校验
    (`_check_feature_face`)承担。cfg 须已经过 `_strip_hpo_cfg`。
    v2.16.0:复跑期间强制 MODELEVO_ES_SOURCE=train_tail —— validation 已并入
    训练集,不能再当早停面(v2.16 冠军代码缺 env 时默认 validation 早停,
    hurdle 等复合容器的 metrics 不带 es_info.es_source,会被当作 train_tail
    冠军走到这里;全局库单段配方在 train_tail 下为 train+validation 尾部早停、
    只训一次)。
    返回 (交付模型, 遥测 dict)。
    """
    import os
    import shutil
    import tempfile

    _cfg = dict(cfg or {})
    _cfg.pop("multi_seed", None)
    if _cfg.get("hpo"):
        raise RuntimeError("rerun_recipe 收到未剥离 hpo 节的 cfg(会重跑搜索)")
    if not callable(getattr(champ, "fit_and_score", None)):
        raise RuntimeError("冠军模块无 fit_and_score,不能 rerun_recipe")

    _orig_td = getattr(champ, "TRIAL_DIR", None)
    _sandbox = Path(tempfile.mkdtemp(prefix="mvo-delivery-rerun-"))
    _es_prev = os.environ.get("MODELEVO_ES_SOURCE")
    os.environ["MODELEVO_ES_SOURCE"] = "train_tail"
    try:
        champ.TRIAL_DIR = _sandbox
        champ._last_model = None
        score, metrics_extra = champ.fit_and_score(
            X_all, y_all, X_va, y_va, seed, _cfg)
        model = getattr(champ, "_last_model", None)
    finally:
        if _es_prev is None:
            os.environ.pop("MODELEVO_ES_SOURCE", None)
        else:
            os.environ["MODELEVO_ES_SOURCE"] = _es_prev
        if _orig_td is not None:
            champ.TRIAL_DIR = _orig_td
        shutil.rmtree(_sandbox, ignore_errors=True)
    if model is None:
        raise RuntimeError("fit_and_score 未设置 _last_model,无法取得交付模型")

    report: dict = {
        "n_fit_rows": int(len(X_all)),
        "rounds": _fitted_rounds(model),
        "score_rows": int(len(score)),
        "metrics_keys": sorted((metrics_extra or {}).keys()),
    }
    # 复合容器:取 model 自带的阶段遥测(如 hurdle 的 stage1_ 子模型轮数)
    try:
        st = getattr(model, "stage1_", None)
        if isinstance(st, dict):
            report["stage1"] = {k: st.get(k) for k in (
                "clf_best_iteration", "reg_best_iteration",
                "refit_clf_trees", "refit_reg_trees",
                "n_fit_rows", "n_es_rows", "es_source",
                "fit_mode") if k in st}
    except Exception:  # noqa: BLE001
        pass
    # v2.16.3:单模型血缘的拟合遥测(mvo_es_info_ / metrics.es_info)也取
    # fit_mode / refit_trees —— retrain_scope 按**实际发生**的拟合方式描述
    try:
        es = dict(getattr(model, "mvo_es_info_", None)
                  or (metrics_extra or {}).get("es_info") or {})
        if es:
            report["es_info"] = {k: es.get(k) for k in (
                "es_source", "fit_mode", "refit_trees",
                "stage1_best_iteration", "stage1_rounds") if k in es}
    except Exception:  # noqa: BLE001
        pass
    report["full_refit"] = _rerun_full_refit(report)
    return model, report


def _rerun_full_refit(report: dict | None) -> bool | None:
    """rerun_recipe 复跑是否发生了"定轮后在全量 train+validation 上重拟合"。

    v2.16.3(D-014 表内第 2 条):v2.16 起冠军配方都是**单段拟合**(头部
    1−ES_FRACTION 拟合、最新尾部 ES_FRACTION 早停、只训一次,遥测
    ``fit_mode=single`` 且无 ``refit_trees``);只有 v2.15 两段式旧代码
    (``_fit_with_es(refit=True)`` / 旧 ``_fit_hurdle_refit_full``)才在早停
    定轮后全量重拟合(遥测带 ``refit_trees`` / ``refit_*_trees``)。

    返回 True = 有全量重拟合证据;False = 有单段证据;None = 无遥测可判。
    调用方已知拟合方式时可直接给 ``report["full_refit"]``(估计器适配器
    路径:头部早停定轮 → 新建估计器在 train+validation 全量重拟合)。
    """
    if not isinstance(report, dict):
        return None
    if isinstance(report.get("full_refit_known"), bool):
        return report["full_refit_known"]
    for sect in ("stage1", "es_info"):
        st = report.get(sect) or {}
        if not isinstance(st, dict) or not st:
            continue
        if any(st.get(k) for k in ("refit_trees", "refit_clf_trees",
                                   "refit_reg_trees")):
            return True
        if str(st.get("fit_mode") or "") == "single":
            return False
    return None


def _retrain_scope_text(rerun: bool, champ_es: str, es_frac: float,
                        rerun_report: dict | None = None) -> str:
    """model_meta / recipe 的 ``retrain_scope``(v2.16.3:如实描述拟合方式)。

    旧文案恒写 ``...; full refit``,而 v2.16 单段配方的 rerun 只训一次:
    头部 ``1−es_frac`` 拟合、最新尾部 ``es_frac`` 早停,**没有**全量重拟合
    (实测 run4 regression 交付 model_meta:rerun.stage1.n_fit_rows=2,686,509
    = 2,985,010×0.9,模型只训了一次)。现在按 :func:`_rerun_full_refit`
    的证据写:有 refit 证据才写 full refit;单段或无遥测(v2.16 起的默认
    口径)写 no full refit,并在后者注明"由遥测缺失推定"。
    """
    if rerun:
        head = f"{1.0 - float(es_frac):g}"
        tail = f"{float(es_frac):g}"
        refit = _rerun_full_refit(rerun_report)
        if refit is True:
            return (f"train+validation; early-stop on latest tail {tail} by "
                    f"date (fit on earliest {head}); then full refit on "
                    "train+validation at early-stopped rounds")
        suffix = "no full refit" if refit is False else \
            "no full refit (single-stage recipe; no refit telemetry)"
        return (f"train+validation; fit on earliest {head} of train+validation "
                f"by date; early-stop on latest tail {tail}; {suffix}")
    if champ_es == "validation":
        return ("train+validation; frozen rounds from outer-validation "
                "early stop (train-only reproduction)")
    return "train+validation; early-stop head=train tail"


#: v2.17.2(D3):各 GBDT 实现的轮数上限键(final_recipe 以交付冻结值覆盖)
_ROUND_CAP_KEYS = ("n_estimators", "iterations", "num_boost_round",
                   "num_iterations", "max_iter")


def final_recipe_summary(meta: dict, champ_dir: Path | None = None) -> dict:
    """冠军交付配方摘要 → task-experience.yaml 的 ``final_recipe``(v2.16.3)。

    单源取自 ``model_meta.json``(finalize-retrain 落盘的交付证明);超参
    优先 ``metrics.best_params``(HPO 重训配方)→ ``config.hyperparameters``
    → model_meta.hyperparameters;两段式(hurdle)另取 ``hurdle_hp.effective``
    的两段生效参数。不含时间戳以外的易变字段 —— 同一交付重复调用结果相同。
    """
    meta = dict(meta or {})
    hp: dict = {}
    segments = None
    src = "model_meta.hyperparameters"
    if champ_dir is not None:
        cdir = Path(champ_dir)
        try:
            m = json.loads((cdir / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = {}
        bp = (m or {}).get("best_params")
        if isinstance(bp, dict) and bp:
            hp, src = dict(bp), "metrics.best_params"
        else:
            try:
                cfg = load_yaml(cdir / "config" / "trial.yaml") or {}
            except Exception:  # noqa: BLE001
                cfg = {}
            if isinstance(cfg.get("hyperparameters"), dict) and \
                    cfg["hyperparameters"]:
                hp, src = dict(cfg["hyperparameters"]), \
                    "config.hyperparameters"
        eff = ((m or {}).get("hurdle_hp") or {}).get("effective")
        if isinstance(eff, dict) and eff:
            segments = eff
    if not hp:
        hp = {k: v for k, v in (meta.get("hyperparameters") or {}).items()
              if v is not None}
    hp = {k: v for k, v in hp.items()
          if isinstance(v, (int, float, str, bool)) or v is None}
    # v2.17.2(D3):轮数(cap 键)以交付模型实际冻结值为准。config 里的
    # n_estimators 是搜索期上限(8000),交付 frozen_rounds 实际只训 5802 轮
    # (run4 opus5.5 final_recipe 记成 8000);model_meta.hyperparameters 是
    # 交付模型的生效参数,覆盖 cap 键并留痕来源。
    _meta_hp = meta.get("hyperparameters") or {}
    _rounds_key = next((k for k in _ROUND_CAP_KEYS
                        if isinstance(_meta_hp.get(k), (int, float))
                        and not isinstance(_meta_hp.get(k), bool)), None)
    frozen_rounds = None
    configured_rounds = None
    if _rounds_key is not None:
        frozen_rounds = int(_meta_hp[_rounds_key])
        configured_rounds = hp.get(_rounds_key)
        if hp.get(_rounds_key) != frozen_rounds:
            hp[_rounds_key] = frozen_rounds
            if src != "model_meta.hyperparameters":
                src = f"{src}+model_meta.hyperparameters[{_rounds_key}]"
    out = {
        "champion_trial_id": meta.get("best_trial"),
        "algorithm": meta.get("algorithm"),
        "estimator_class": meta.get("estimator_class"),
        "refit_mode": meta.get("refit_mode"),
        "retrain_scope": meta.get("retrain_scope"),
        "n_features": meta.get("n_features"),
        "hyperparameters": hp,
        "hyperparameters_source": src,
        "n_train_rows": meta.get("n_train_rows"),
        "n_validation_rows": meta.get("n_validation_rows"),
        "n_train_validation_rows": meta.get("n_train_validation_rows"),
        "model_sha256": meta.get("final_model_sha256"),
        "delivery_status": meta.get("delivery_status"),
        "data_view": meta.get("data_view"),
        "seed": meta.get("seed"),
        "source": "final-model/model_meta.json",
    }
    if frozen_rounds is not None:
        out["frozen_rounds"] = frozen_rounds
        out["rounds_key"] = _rounds_key
        if configured_rounds is not None and configured_rounds != frozen_rounds:
            out["configured_rounds_cap"] = configured_rounds
    _bi = meta.get("early_stopping_best_iteration")
    if isinstance(_bi, (int, float)) and not isinstance(_bi, bool):
        out["best_iteration"] = int(_bi)
    if segments:
        out["segments"] = segments
    return out


def _record_final_recipe(run_dir: Path, meta: dict,
                         champ_dir: Path | None) -> None:
    """交付阶段把配方摘要写进 task-experience(失败只告警,不阻塞交付)。

    v2.17.2(D3):record_final_recipe 按 (model_sha256, refit_mode) 幂等;
    v2.17.1 及更早写下的同一交付(无 frozen_rounds、轮数记成配置上限)
    在此按新口径覆写一次,之后照旧幂等。
    """
    try:
        from runtime.memory import TaskExperience

        te = TaskExperience(run_dir)
        summ = final_recipe_summary(meta, champ_dir)
        prev = (te.load() or {}).get("final_recipe")
        if isinstance(prev, dict) and "frozen_rounds" in summ \
                and prev.get("model_sha256") == summ.get("model_sha256") \
                and prev.get("frozen_rounds") != summ.get("frozen_rounds"):
            data = te.load() or {}
            data["final_recipe"] = dict(summ, recorded_at=now_iso())
            te.save(data)
            return
        te.record_final_recipe(summ)
    except Exception as exc:  # noqa: BLE001
        logger.warning("task-experience.final_recipe 写入失败(不阻塞): %s", exc)


def _check_feature_face(X_va, cols, champ_dir: Path,
                        atol: float = 1e-6) -> dict:
    """交付特征面 vs 冠军真实 validation 输入的一致性校验(v2.15.9)。

    替代旧口径"在 train 上重训一遍比对分数"的 40 min 复现:模型代码
    一致性由交付配方副本的源码 sha 保证,这里只需证明交付管线重建出的
    validation 特征帧与冠军训练时喂给模型的帧逐列等价(列序 + 数值)。
    冠军无 validation_input.parquet 时 passed=None,调用方退回完整复现。
    """
    import numpy as np
    import pandas as pd

    ref_path = champ_dir / "model" / "validation_input.parquet"
    report: dict = {"passed": None, "reference": None,
                    "n_columns": int(len(cols))}
    if not ref_path.exists():
        report["reason"] = "冠军无 model/validation_input.parquet,退回完整复现"
        return report
    try:
        ref = pd.read_parquet(ref_path)
    except Exception as exc:  # noqa: BLE001
        report["reason"] = f"冠军 validation_input 不可读:{exc}"
        return report
    report["reference"] = str(ref_path.relative_to(champ_dir.parent.parent))
    report["n_rows"] = int(len(ref))
    if list(ref.columns) != list(cols):
        ref_cols, cur_cols = list(ref.columns), list(cols)
        only_ref = [c for c in ref_cols if c not in set(cur_cols)][:5]
        only_cur = [c for c in cur_cols if c not in set(ref_cols)][:5]
        order_ok = ref_cols == cur_cols
        report.update({
            "passed": False,
            "reason": "列集合或列序与冠军不一致",
            "order_ok": order_ok,
            "missing_in_delivery": only_ref,
            "extra_in_delivery": only_cur,
        })
        return report
    if len(ref) != len(X_va):
        report.update({"passed": False,
                       "reason": f"行数 {len(X_va)} != 冠军 {len(ref)}"})
        return report
    ref = ref.astype("float32", copy=False)
    cur = X_va.astype("float32", copy=False)
    n_bad = 0
    bad_cols: list[str] = []
    max_abs = 0.0
    for col in cols:
        a = np.asarray(ref[col].values, dtype="float64")
        b = np.asarray(cur[col].values, dtype="float64")
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        diff = np.abs(np.where(np.isnan(a), 0.0, a)
                      - np.where(np.isnan(b), 0.0, b))
        col_max = float(np.nanmax(diff)) if len(diff) else 0.0
        max_abs = max(max_abs, col_max)
        if not same_nan or col_max > atol:
            n_bad += 1
            if len(bad_cols) < 10:
                bad_cols.append(col)
    report.update({
        "passed": n_bad == 0,
        "mismatched_columns": n_bad,
        "mismatched_examples": bad_cols,
        "max_abs_diff": max_abs,
        "atol": atol,
    })
    if n_bad:
        report["reason"] = "交付特征帧与冠军 validation 输入数值不一致"
    return report


def _shadow_precheck(run_dir: Path, fm: Path, champ_dir: Path, spec,
                     n_boot: int = 2000) -> dict:
    """gate 前的 shadow 预检(v2.15.9):交付模型 vs 冠军 train-only 模型的
    shadow 主指标配对对照。shadow 标签已在盘上,不消耗 final_test。

    交付模型显著更差(Δ 的 95% CI 两端都不优于 0)时调用方必须要求
    `--ack-shadow-regression` 才放行,否则拒绝进入盲测。
    """
    import numpy as np
    import pandas as pd

    from runtime.data import SAMPLE_KEY_COLUMN
    from runtime.metrics import (align_score_by_keys, compute_metric,
                                 metric_direction, paired_bootstrap_delta)

    out: dict = {"available": False}
    new_p = fm / "shadow_test_score.parquet"
    old_p = champ_dir / "shadow_test_score.parquet"
    lab_p = run_dir / "sample-profile" / "splits" / "labels" \
        / "shadow_test.parquet"
    if not (new_p.exists() and old_p.exists() and lab_p.exists()):
        out["reason"] = "缺交付/冠军 shadow 分数或 shadow 标签"
        return out
    try:
        lab = pd.read_parquet(lab_p)
        new = pd.read_parquet(new_p)
        old = pd.read_parquet(old_p)
        keys = lab[SAMPLE_KEY_COLUMN].astype(str).values
        y = lab[spec.target_name].values
        sc_new = align_score_by_keys(new["score"].values, keys,
                                     new[SAMPLE_KEY_COLUMN].astype(str).values)
        sc_old = align_score_by_keys(old["score"].values, keys,
                                     old[SAMPLE_KEY_COLUMN].astype(str).values)
        stat = paired_bootstrap_delta(
            y, sc_new, sc_old, spec.primary_metric, n_boot=n_boot)
        metric_new = float(compute_metric(spec.primary_metric, y, sc_new,
                                          score=sc_new))
        metric_old = float(compute_metric(spec.primary_metric, y, sc_old,
                                          score=sc_old))
        higher_better = metric_direction(spec.primary_metric)
        lo, hi = stat.get("delta_ci") or (float("nan"), float("nan"))
        # 方向感知:Δ_dir = Δ(越大越好) / −Δ(越小越好);显著更差 ⇔ Δ_dir
        # 的 CI 上界 < 0 ⇔ 越大越好时 hi < 0、越小越好时 lo > 0。
        worse = bool(np.isfinite(lo) and np.isfinite(hi)
                     and ((hi < 0.0) if higher_better else (lo > 0.0)))
        out.update({
            "available": True,
            "metric": spec.primary_metric,
            "delivery_model": metric_new,
            "search_model": metric_old,
            "delta_mean": stat.get("delta_mean"),
            "delta_ci": [float(lo), float(hi)],
            "worse_than_search_model": worse,
        })
        if str(getattr(spec, "model_type", "") or "") == "regression":
            # v2.17.0(#20):回归任务的水位(pred_mean / y_mean / level_ratio)。
            # 只诊断,不进放行判定;shadow 标签在盘上,不涉及 final_test。
            try:
                from runtime.regression_diagnostics import level_stats

                _lv = {"delivery_model": level_stats(y, sc_new),
                       "search_model": level_stats(y, sc_old)}
                out["level_stats"] = _lv
                out["level_shift_warning"] = bool(
                    _lv["delivery_model"].get("level_shift_warning"))
            except Exception as exc:  # noqa: BLE001  诊断失败不影响预检
                out["level_stats_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001  预检不阻塞,但如实留痕
        out["reason"] = f"shadow 预检失败:{exc}"
    return out


def _finalize_best_iteration(model) -> int:
    """Return the fitted tree count using each library's index convention.

    LightGBM exposes a one-based tree count, while XGBoost and CatBoost expose
    a zero-based best *index*.  Keeping this normalization here prevents the
    final train+validation refit from silently dropping the last selected
    tree (or treating XGBoost's valid ``best_iteration == 0`` as "missing").
    """
    module = str(type(model).__module__ or "").lower()
    for attr in ("best_iteration_", "best_iteration"):
        try:
            raw = getattr(model, attr)
        except (AttributeError, TypeError, ValueError):
            continue
        if raw is None:
            continue
        value = int(raw)
        if module.startswith("xgboost"):
            return value + 1
        if value > 0:
            return value
    if module.startswith("catboost") and hasattr(model, "get_best_iteration"):
        try:
            value = int(model.get_best_iteration())
            return value + 1 if value >= 0 else 0
        except (TypeError, ValueError):
            pass
    return 0


def _finalize_es_fit_kwargs(model, X_eval, y_eval,
                            early_stopping_rounds: int) -> dict:
    """Build version-safe early-stopping kwargs for the champion estimator."""
    try:
        params = inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        params = {}
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepts = lambda name: name in params or accepts_kwargs
    kwargs: dict = {}
    if accepts("eval_set"):
        kwargs["eval_set"] = [(X_eval, y_eval)]
    if accepts("verbose"):
        kwargs["verbose"] = False
    module = str(type(model).__module__ or "").lower()
    if module.startswith("lightgbm") and accepts("callbacks"):
        import lightgbm as lgb

        kwargs["callbacks"] = [lgb.early_stopping(
            int(early_stopping_rounds), verbose=False)]
    return kwargs


def _widen_self_contained(base_df, wide_df, ratio: float = 0.9) -> bool:
    """加宽重放输出的自含性判定(v2.11.8)。

    背景:第三族加宽符号(`_widen`/`_widen_pruned`/`widen_cached`)的**返回值
    契约不统一** —— 有的血缘只返回"新增列"(需 concat 回基础面),有的血缘
    (gen-01 超级 trial 血缘)返回的是**完整模型输入帧**(基础面 + 类别展开 +
    新增列,即 `加列不替换` 的成品)。

    对后者无条件 concat 会产出重名列帧:实测 `req-t7-drawdown` 冠军
    trial-004 的基础面 1493 列与加宽输出 2614 列交集 1491 ⇒
    `concat` 得 4107 列含 1491 个重名,下游 `prepare_features` 的
    `pd.to_numeric(X_tr[c])` 拿到 DataFrame 直接
    `TypeError: arg must be a list, tuple, 1-d array, or Series`,
    交付链在打分前中断(final-model/ 为空)。

    判据用重叠率而非严格 `issubset`:基础面的 object 列会在加宽帧内被
    one-hot 展开替换(本例 `prime_cust_mob6`/`loan_flag_v2` 两列不在加宽
    输出中),严格子集判定会漏判。
    """
    try:
        base_cols = set(base_df.columns)
        if not base_cols:
            return False
        overlap = len(base_cols & set(wide_df.columns))
        return overlap >= ratio * len(base_cols)
    except Exception:  # noqa: BLE001  判定失败时退回旧的 concat 行为
        return False


def _assert_no_duplicate_columns(df, ctx: str) -> None:
    """加宽重放后的重名列硬断言(v2.11.8)。

    重名列会让下游按列名取数得到 DataFrame(而非 Series),在 `prepare_features`
    里炸出与根因无关的 `TypeError`。在拼接点直接报出可诊断错误,把这种缺陷
    从"下游随机崩溃"提前为"拼接点明确报错"。
    """
    try:
        dup = df.columns[df.columns.duplicated()].tolist()
    except Exception:  # noqa: BLE001
        return
    if dup:
        raise RuntimeError(
            f"{ctx}:加宽重放后出现 {len(dup)} 个重名列(前 10:{dup[:10]})"
            " —— 加宽输出已自含基础面时不得再次 concat(见 "
            "_widen_self_contained)")


def _best_unconfirmed_candidate(trials, spec, lb=None):
    """确认门未过时的最强合法候选。

    v2.14.0:给定 ``lb`` 时取**搜索冠军**(与交付位解析同源:收缩分、
    基线同视图、非 screening、最新 epoch)。旧口径按原始主指标裸排序,
    与冠军选型不同源,也不排除 screening trial。解析不可用时退回旧口径:
    verdict ∈ confirmed/accepted、完整性有效、无失败、主指标可用。
    """
    if lb is not None:
        try:
            champ = lb.best_trial(trials)
        except Exception:  # noqa: BLE001
            champ = None
        if champ is not None:
            return champ
    rev = spec.primary_metric
    rows = [t for t in trials
            if (t.data.get("evaluation") or {}).get("verdict")
            in ("confirmed", "accepted")
            and not trial_is_screening(t)
            and (t.data.get("evaluation") or {}).get("integrity", {})
            .get("valid", True) is not False
            and not trial_currently_failed(t)
            and (t.data.get("evaluation") or {})
            .get("primary_metric", {}).get("value") is not None]
    if not rows:
        return None
    higher = True
    try:
        from runtime.report import _metric_higher_better
        higher = _metric_higher_better(rev)
    except Exception:  # noqa: BLE001
        pass
    rows.sort(
        key=lambda t: t.data["evaluation"]["primary_metric"]["value"],
        reverse=higher)
    return rows[0]


def _file_sha256(path) -> str | None:
    """文件 sha256;不存在/不可读返回 None。"""
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


#: final-model/unconfirmed/ 的证据拷贝白名单。v2.13.4:**不含任何模型
#: 二进制**。trial 目录里的模型是 train-only 搜索工件,旧白名单把它拷进
#: final-model/ 下,读者自然当成交付模型(实测 req-t7-drawdown:交付物
#: 实为 train-only 模型,用户一句「交付不经过重训吗?」才暴露)。
#: 未确认路径的模型由 finalize-retrain 同口径重训后写入 unconfirmed/model/。
_UNCONFIRMED_EVIDENCE = (
    "feature/feature_importance.csv", "feature/group_importance.csv",
    "segment-report.md", "segment-report.xlsx",
    "evaluation.json", "metrics.json",
    "config/trial.yaml", "code/train.py")


def _confirm_attempt_lines(run_dir, pick, spec) -> list[str]:
    """v2.14.0:该候选配方的全部 confirm 尝试(UNCONFIRMED.md 用)。

    未确认候选通常是搜索冠军本身(accepted),它自己没有
    fresh_confirmation —— 确认证据在它的 confirm 子代上。旧说明只打印
    候选自身的 fresh_confirmation(恒为 None),读者看不到"试过几次、
    差在哪"。
    """
    try:
        trials = TrialFactory(run_dir, spec=spec).all_trials()
    except Exception:  # noqa: BLE001
        return []
    by_id = {t.trial_id: t for t in trials}
    root = recipe_root(pick, by_id)
    rows = []
    for t in trials:
        if str(t.data.get("search_action") or "") != "confirm" \
                or recipe_root(t, by_id) != root:
            continue
        ev = t.data.get("evaluation") or {}
        fc = ev.get("fresh_confirmation") or {}
        val = (fc.get("validation") or {})
        rows.append(
            f"| {t.trial_id} | {ev.get('verdict')} | "
            f"{val.get('delta_used', (ev.get('delta') or {}).get('vs_parent'))}"
            f" | {fc.get('delta_vs_parent')} | {fc.get('margin_lower') or fc.get('equivalence_margin')}"
            f" | {fc.get('criterion') or 'legacy_two_sided'} | {fc.get('passed')} |")
    if not rows:
        return ["", "## 该配方的 confirm 尝试", "",
                f"配方 `{root}` 尚无 confirm trial。"]
    return ["", "## 该配方的 confirm 尝试", "",
            "| trial | verdict | val Δ | shadow Δ | 下界 margin | 判据 | passed |",
            "|---|---|---|---|---|---|---|", *rows,
            "", "判据 `legacy_two_sided` = v2.14.0 之前的双侧等价 |Δ|≤2·SE"
            "(不含 seed 方差,可复现配方单次过门率仅 35~45%);如需按新判据"
            "重判,对该 confirm 重新评估或新建 confirm。"]


def _write_unconfirmed_evidence(run_dir, spec, pick, *, retrained: bool,
                                step, resolution: dict | None = None) -> Path:
    """把未确认候选的**证据**(非模型)与说明写入 final-model/unconfirmed/。

    不写任何会被 evaluate-final / 交付校验当成冠军产物的位置
    (final-model/model/、final-model/recipe.yaml、final-model/model_meta.json)。
    retrained=True 时说明同目录 model/ 下是 train+validation 重训模型。
    """
    import json as _json
    import shutil

    rev = spec.primary_metric
    ev = pick.data.get("evaluation") or {}
    src = run_dir / "trials" / pick.trial_id
    out = run_dir / "final-model" / "unconfirmed"
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in _UNCONFIRMED_EVIDENCE:
        s = src / rel
        if s.exists():
            d = out / "evidence" / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            copied.append(rel)
    status = ("unconfirmed_retrained" if retrained
              else "unconfirmed_snapshot")
    meta = {
        "status": status,
        "trial_id": pick.trial_id,
        "verdict": ev.get("verdict"),
        "primary_metric": rev,
        "value": ev.get("primary_metric", {}).get("value"),
        "fresh_confirmation": ev.get("fresh_confirmation"),
        "slice_evaluation": ev.get("slice_evaluation"),
        "temporal_evaluation": ev.get("temporal_evaluation"),
        "integrity_valid": (ev.get("integrity") or {}).get("valid"),
        "retrained_model_dir": "model/" if retrained else None,
        "note": ("未通过独立确认的最优合法候选。它不是交付冠军,不得直接上线;"
                 "交付门要求绑定搜索冠军配方的 fresh_confirmation.passed=true,"
                 "本 run 未满足。"),
        "delivery_resolution": resolution,
        "evidence_copied": copied,
    }
    (out / "snapshot_meta.json").write_text(
        _json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    fc = ev.get("fresh_confirmation") or {}
    if retrained:
        model_sec = (
            "## 模型\n\n"
            "`model/` 下是用该候选的冻结代码与配置,按 `finalize-retrain` "
            "**同一口径**(model_meta.json 的 refit_mode/retrain_scope;默认 "
            "rerun_recipe:train+validation 头部拟合、最新尾部早停,单段配方只训"
            "一次,以 retrain_scope 为准)"
            "重训的模型;`model_meta.json` 记录训练行数与哈希。\n\n"
            "- validation 已并入训练集:**没有样本外效果估计**;"
            "`evidence/evaluation.json` 的指标是 train-only 搜索模型在 "
            "validation 上的读数,仅供参考。\n"
            "- 未对 final_test / shadow_test 打分,盲测机会未消耗。\n")
    else:
        model_sec = (
            "## 模型\n\n"
            "本目录**不含模型二进制**(`--snapshot-only` 或候选无法重训)。"
            "trial 目录里的模型是 train-only 搜索工件,不得作为交付物;"
            "需要模型时去掉 `--snapshot-only` 重跑 `finalize-retrain`。\n")
    (out / "UNCONFIRMED.md").write_text(
        "# 未确认候选(不是交付冠军)\n\n"
        f"- 候选: `{pick.trial_id}`\n"
        f"- verdict: `{ev.get('verdict')}`\n"
        f"- {rev}: {ev.get('primary_metric', {}).get('value')}\n"
        f"- fresh_confirmation.passed: `{fc.get('passed')}`"
        f"(delta_vs_parent={fc.get('delta_vs_parent')}, "
        f"equivalence_margin={fc.get('equivalence_margin')})\n"
        f"- integrity.valid: `{(ev.get('integrity') or {}).get('valid')}`\n"
        + (f"- 交付位解析: `{(resolution or {}).get('status')}` —— "
           f"{divergence_summary(resolution)}\n" if resolution else "")
        + "\n## 为什么不是冠军\n\n"
        "交付门要求**绑定搜索冠军配方**的 `fresh_confirmation.passed=true`"
        "(独立新 seed 重训;validation 与 shadow 均非劣且保真,margin 含 "
        "seed 方差)。本 run 的搜索冠军配方未满足,因此未写 "
        "`final-model/model/`、`recipe.yaml`、`model_meta.json`,未对 "
        "final_test 打分。\n\n"
        + model_sec
        + "\n".join(_confirm_attempt_lines(run_dir, pick, spec)) + "\n"
        "\n## 证据\n\n"
        + "\n".join(f"- `evidence/{c}`" for c in copied) + "\n",
        encoding="utf-8")
    return out


def _stage_unconfirmed_snapshot(run_dir, spec, trials, lb, *,
                                logger, step) -> int:
    """无已确认冠军、且不(能)重训时的**证据快照**(不改交付判定)。

    v2.12.2.5 起确认门未过时落 final-model/unconfirmed/,避免交付层完全
    空白。v2.13.4 起:① 正常情况下 finalize-retrain 对最强合法候选执行
    **同口径未确认重训**(见 cmd_finalize_retrain),本函数只在
    `--snapshot-only` 或无合法候选时兜底;② 快照**不再拷贝 trial 的
    train-only 模型二进制**(见 _UNCONFIRMED_EVIDENCE)。

    返回 1(保持"未成功交付"的退出语义)。
    """
    pick = _best_unconfirmed_candidate(trials, spec, lb)
    if pick is None:
        logger.error(
            "冠军 %s 不满足交付门:必须存在、verdict 为 confirmed、"
            "通过独立 shadow fresh_confirmation、完整性有效且无 "
            "failure_reason；--champion 不能绕过该门", "(无合法候选)")
        return 1
    _write_unconfirmed_evidence(
        run_dir, spec, pick, retrained=False, step=step,
        resolution=getattr(lb, "last_delivery_resolution", None))
    ev = pick.data.get("evaluation") or {}
    step(f"证据快照已落盘: {pick.trial_id} → final-model/unconfirmed/"
         "(不含模型二进制)")
    logger.error(
        "冠军 %s 不满足交付门(verdict=%s, fresh_confirmation.passed=%s):"
        "未生成冠军产物;仅落盘证据快照到 final-model/unconfirmed/。"
        "需要未确认重训模型时去掉 --snapshot-only 重跑 finalize-retrain。"
        "--champion 不能绕过交付门。",
        pick.trial_id, ev.get("verdict"),
        (ev.get("fresh_confirmation") or {}).get("passed"))
    return 1


def cmd_finalize_retrain(args) -> int:
    """冠军代码+config 在 train+validation 重训 → final-model/ 全套产物。

    v19.4/P1-2:移植自收尾期实测脚本(原为每个 run 手写 ~200 行,
    平均烧 40 min Agent 时间且口径易漂移)。契约:
    - 冠军默认 = leaderboard.best_trial 确定性选型;--champion 覆写
      (与 evaluate-final --best-trial 同款人工入口,留痕 model_meta);
    - 冠军 trial 的 code/ + config/ 是唯一口径源:同 exclude_columns
      特征面、同 transform、同 dtype 治理;
    - v2.15.9 交付重训口径(`--refit-mode` / config `finalize.refit_mode`):
      * rerun_recipe(默认):冠军 code/config 字节级复制到
        final-model/delivery-recipe/(逐文件 sha 核对)并从副本加载;特征
        面一致性校验(交付 validation 帧 vs 冠军 model/validation_input
        .parquet;缺该工件或 --full-reproduction 时退回 train 上完整复现)
        通过后,以 train+validation 调冠军自己的 fit_and_score —— 早停
        落在 train+validation 的最新尾部 ES_FRACTION(v2.16 单段配方只训一次,
        不再全量重拟合;retrain_scope 按遥测如实记录);返回
        的 `_last_model` 直接交付(不 clone,复合容器可用);
      * frozen_rounds(兼容 v2.13.4):train 上复现 → 冻结 train 尾部定出的
        轮数 → sklearn.clone → train+validation 重拟合;模型不可 clone 时
        报错并给出 remedy,**不**静默回落估计器适配器;
      * v2.17.0:frozen_rounds 先对冠军 train-only 模型做 clone 探测(复现
        之前);不可 clone 且冠军早停面=validation → 自动回落 train_tail
        口径复跑配方(不复现),train_tail 冠军 → 立即报错;
        model_meta 记 refit_mode_requested / refit_mode_effective /
        refit_mode_fallback_reason / clone_probe,stdout 打印
        FINALIZE_RETRAIN_SUMMARY 单行 JSON;可 clone 的 sklearn 协议复合
        容器逐子估计器冻结轮数(frozen_rounds_params);
    - 交付模型打分按任务类型分派(regression 一律 predict);shadow_test
      打分后立即与冠军 train-only shadow 分数做配对预检(model_meta
      .shadow_precheck),显著更差时 evaluate-final 需 --ack-shadow-regression;
    - final_test 标签不读(铁律 3):只打分落盘
      final-model/final_test_score.parquet / shadow_test_score.parquet;
    - 产物:model/{model.joblib,model.json,feature_columns.json,
      validation_input.parquet,feature_importance.csv} + recipe.yaml +
      model_meta.json(best_trial 与选型一致,v16 冠军单源)。
    幂等:重跑覆写 final-model/(gate 由 evaluate-final 守护)。

    v2.13.4 **未确认重训**:无 fresh confirmed 冠军、但存在合法候选
    (accepted/confirmed、完整性有效、无失败)时,走**完全相同的重训代码**,
    只改三处:产物落 final-model/unconfirmed/(绝不写交付位)、跳过
    confirm 专属的 seed 校验与冠军单源复核、不对 final_test/shadow 打分;
    退出码仍为 1。目的:确认门是否通过都不改变「交付目录里的模型必是
    train+validation 重训产物」这一不变量,Agent 不再有手写重训的理由
    (实测 req-t7-drawdown:门被拦 → 手写脚本口径漂移且两次覆写冻结工件)。
    `--snapshot-only` 退回只落证据、不含模型二进制。
    """
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    # v2.13.5:可 --detach(长时重训不随 agent 会话退出而死;wait --job
    # finalize-retrain 追踪)
    from .cmd_trial import _maybe_detach

    _rc = _maybe_detach(args, run_dir, "finalize-retrain")
    if _rc is not None:
        return _rc
    config = _load_config(run_dir)
    _require_confirm = _confirmation_required(config)
    refit_mode = _refit_mode(args, config)
    _full_repro = bool(getattr(args, "full_reproduction", False))

    import time

    import joblib
    import numpy as np
    import pandas as pd
    import yaml

    from runtime.leaderboard import Leaderboard
    from runtime.trial import TrialFactory

    t0 = time.time()

    def step(msg: str) -> None:
        logger.info("[retrain] %s (%.1fs)", msg, time.time() - t0)

    factory = TrialFactory(run_dir, spec=spec, config=config)
    trials = factory.all_trials()
    lb = Leaderboard(run_dir, spec)
    lb.build(trials)
    snapshot_only = bool(getattr(args, "snapshot_only", False))
    unconfirmed_mode = False
    champ_id = getattr(args, "champion", None)
    _ack = getattr(args, "ack_divergence", None)
    _res: dict = {}
    _div_info: dict = {}
    if champ_id:
        step(f"冠军覆写: {champ_id}(人工入口,留痕 model_meta)")
    else:
        # v2.11.7:与 evaluate-final 的 gate 口径对齐。gate 走
        # 交付选型统一走 delivery=True：只接受 fresh confirmed 候选。
        # shadow 只在 confirm 中形成等价性证据，不再作 top-N 复选，
        # 避免它退化成第二个反复使用的选择集。
        # v2.14.0:delivery=True 只接受绑定搜索冠军配方的确认(或差距
        # 在容差内的陈旧确认);分叉时不再静默回落到陈旧确认。
        best = lb.best_trial(trials, delivery=True)
        _res = getattr(lb, "last_delivery_resolution", None) or {}
        if best is None:
            if _res.get("status") == "diverged":
                _note = _write_divergence_note(run_dir, _res)
                logger.error(
                    "%s。按 v2.14.0 不回落到陈旧确认,改对搜索冠军执行未确认"
                    "重训;说明见 %s。确要交付某个已确认 trial:"
                    "finalize-retrain --champion <id> --ack-divergence "
                    "\"<理由>\"(须用户裁决)",
                    divergence_summary(_res), _note)
            # v2.15.3:require_confirmation=false ⇒ 以 accepted 搜索冠军
            # 直接走交付位重训(不进 unconfirmed/)。
            _cand = _best_unconfirmed_candidate(trials, spec, lb)
            if (not _require_confirm and _cand is not None
                    and not snapshot_only):
                champ_id = _cand.trial_id
                logger.warning(
                    "evaluator.require_confirmation=false:无 fresh confirmed "
                    "冠军,以搜索冠军 %s 直接交付重训(delivery_status="
                    "confirmation_waived,无独立 shadow 确认)", champ_id)
            # v2.12.2.6:无冠军分支必须可达降级路径(旧实现直接 return 1,
            # 降级产物永不生成)。v2.13.4:有合法候选时改为**未确认重训**,
            # 仅 --snapshot-only 或无合法候选才退回证据快照。
            elif _cand is None or snapshot_only:
                logger.error("leaderboard 无 fresh confirmed 交付冠军")
                return _stage_unconfirmed_snapshot(run_dir, spec, trials, lb,
                                                   logger=logger, step=step)
            else:
                champ_id = _cand.trial_id
                unconfirmed_mode = True
                logger.warning(
                    "无 fresh confirmed 冠军:以搜索冠军 %s 执行**未确认重训**"
                    "(train+validation 同口径),产物落 final-model/unconfirmed/,"
                    "不写交付位、不打 final_test", champ_id)
        else:
            champ_id = best.trial_id
            if _res.get("status") == "stale_within_tolerance":
                logger.warning("%s", divergence_summary(_res))
    champ_dir = run_dir / "trials" / champ_id
    # v2.16.0:冠军若在 outer validation 上早停,交付改走 frozen_rounds
    _champ_es = _trial_es_source(champ_dir)
    _mode_before = refit_mode
    refit_mode = _resolve_refit_for_champion(args, refit_mode, _champ_es)
    if refit_mode != _mode_before:
        step(f"冠军早停面=validation:refit_mode {_mode_before} → {refit_mode}"
             "(validation 并入训练集后不能再当早停集)")
    # 人工覆写只改变候选选择，不能绕过客观裁决与完整性门。
    _champ_trial = next((t for t in trials if t.trial_id == champ_id), None)
    _champ_ev = (_champ_trial.data.get("evaluation") or {}) \
        if _champ_trial is not None else {}
    _unconfirmed = (
        _champ_trial is None or _champ_ev.get("verdict") != "confirmed" or
        (_champ_ev.get("fresh_confirmation") or {}).get("passed") is not True
    )
    # v2.15.3:确认门关闭时 accepted 冠军视同可交付(只放开确认,不放开裁决)
    # v2.20.0:先验豁免 —— 冠军配方未确认但"换 seed 后增益仍在"
    # (delivery_resolution.status=waived_by_prior),视同可交付;与
    # require_confirmation=false 走同一条路径(正常交付位、final_test 照常
    # 打分),只少一份独立 seed 复现证明,delivery_status 另记以便审计。
    _prior_waived = (_res.get("status") == "waived_by_prior")
    _confirm_waived = ((not _require_confirm and _unconfirmed
                        and _champ_ev.get("verdict") == "accepted")
                       or _prior_waived)
    if _confirm_waived:
        _unconfirmed = False
    _broken = (
        _champ_trial is None
        or (_champ_ev.get("integrity") or {}).get("valid", True) is False
        or trial_currently_failed(_champ_trial))
    if _broken:
        # 完整性失效/失败的候选不可重训;退回证据快照(取其余最强合法候选)
        return _stage_unconfirmed_snapshot(run_dir, spec, trials, lb,
                                           logger=logger, step=step)
    # v2.16.0 拓扑门(交付侧):冠军若是未获人工批准的级联/融合(例如门上线
    # 前跑出的存量 trial),拒绝重训交付——搜索期的门只管新 trial
    from runtime.topology import evaluate_gate as _topo_gate

    _champ_topo = _topo_gate(run_dir, _champ_trial, spec.model_type)
    if not _champ_topo.get("allowed", True):
        logger.error(
            "交付冠军 %s 的模型拓扑 %s 未获人工批准(需要:%s)——拒绝重训交付。"
            "由用户在自己的终端执行:%s;或改选单模型冠军(--champion)",
            champ_id, _champ_topo.get("kind"),
            ", ".join(_champ_topo.get("needs") or []),
            _champ_topo.get("approve_command"))
        return 1
    _champ_trial.data["model_topology"] = _champ_topo
    if _unconfirmed and not unconfirmed_mode:
        # 显式 --champion 指向未确认候选:同样走未确认重训,绝不进交付位
        if snapshot_only or _champ_ev.get("verdict") not in (
                "confirmed", "accepted"):
            return _stage_unconfirmed_snapshot(run_dir, spec, trials, lb,
                                               logger=logger, step=step)
        unconfirmed_mode = True
        logger.warning(
            "--champion %s 未通过 fresh confirmation:执行未确认重训,"
            "产物落 final-model/unconfirmed/,不写交付位", champ_id)
    if getattr(args, "champion", None) and not _unconfirmed:
        # v2.14.0:显式指定的已确认 trial 若与搜索冠军配方分叉且超出
        # 容差,必须有用户裁决(--ack-divergence)才能写交付位 —— 这正是
        # run2 静默发生的事(trial-011 落后冠军 0.00157 > MDE)。
        _ok, _div_info = _delivery_divergence_check(lb, trials, champ_id)
        _res = _div_info.get("resolution") or {}
        if not _ok:
            if not _ack:
                _write_divergence_note(run_dir, _res)
                logger.error(
                    "--champion %s 与搜索冠军 %s 配方分叉:落后 %+.6f > 容差 "
                    "%.6f。写交付位需用户裁决:加 --ack-divergence \"<理由>\""
                    "(留痕);或改为确认搜索冠军配方", champ_id,
                    _res.get("search_champion"), _div_info.get("gap"),
                    _div_info.get("tolerance"))
                return 1
            _record_divergence_ack(run_dir, champ_id, _div_info, str(_ack),
                                   "finalize-retrain")
            logger.warning("交付位分叉已由 --ack-divergence 裁决放行(留痕):"
                           "%s", _ack)
    elif not _res:
        try:
            _res = lb.delivery_resolution(trials)
        except Exception:  # noqa: BLE001
            _res = {}
    try:
        from runtime.integrity import (verify_confirmation_seed,
                                       verify_trial_source)

        _champ_source_manifest = verify_trial_source(champ_dir)
        if not unconfirmed_mode and not _confirm_waived:
            # confirm 专属:新 seed 必须不同于父代。未确认候选不是 confirm
            # trial,其父代同 seed 属正常,不校验。
            verify_confirmation_seed(
                champ_dir, run_dir / "trials" / _champ_trial.parent_ids[0])
    except Exception as exc:  # noqa: BLE001
        logger.error("冠军源码完整性校验失败:%s", exc)
        return 1
    if not (champ_dir / "code" / "train.py").exists():
        logger.error("冠军 trial 缺 code/train.py: %s", champ_dir)
        return 1

    fm = run_dir / "final-model"
    if unconfirmed_mode:
        # 未确认重训只写 unconfirmed/,绝不碰 final-model/model/、
        # final-model/recipe.yaml、final-model/model_meta.json(交付位)
        fm = fm / "unconfirmed"
    fm.mkdir(parents=True, exist_ok=True)
    (fm / "model").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", _finalize_threads())
    os.environ["MODELEVO_GPU"] = "1"
    # v2.15.9:交付执行源 = 冠军 code/config 的字节级副本(sha 逐文件核对)。
    # 副本与 trials/<id>/ 同深度,未确认重训用独立目录,绝不写交付位。
    _recipe_dir, _recipe_shas = _copy_delivery_recipe(
        champ_dir, run_dir / "final-model" / (
            "unconfirmed-recipe" if unconfirmed_mode else "delivery-recipe"))
    step(f"delivery recipe copied: {_recipe_dir.relative_to(run_dir)} "
         f"({len(_recipe_shas)} files, refit_mode={refit_mode})")
    champ = _finalize_load_module(_recipe_dir / "code" / "train.py",
                                  "champ_train")
    if hasattr(champ, "pick_gpu"):
        try:
            champ.pick_gpu()
        except Exception:  # noqa: BLE001  无卡环境照跑
            pass
    cfg = yaml.safe_load(
        (_recipe_dir / "config" / "trial.yaml").read_text()) or {}
    seed = int(cfg.get("seed", getattr(champ, "SEED", 20260814)))
    step(f"champion={champ_id} seed={seed}")

    # ---------- v2.17.0 clone 探测(#5:先检查再复现)----------
    # frozen_rounds = train 上复现 → clone → train+validation 冻结轮数重训。
    # 旧序在复现(run4 实测 10.4 min)之后才 clone,复合容器(手写 hurdle)
    # 必然 TypeError 后才回落。现于特征构造/复现之前对冠军 train-only 模型
    # 做 clone 探测:不可 clone 且冠军早停面=validation → 直接走 train_tail
    # 口径复跑配方(回落行为不变,只是不白跑复现);train_tail 冠军显式
    # frozen_rounds → 立即报错(旧行为,只是提前)。探测取不到模型对象时
    # (unavailable)退回"复现后再 clone"的旧顺序。全部留痕 model_meta。
    _refit_requested = refit_mode
    _refit_fallback_reason = None
    _clone_probe_rep: dict = {"status": "not_required", "error": None}
    _clone_needed = (refit_mode == "frozen_rounds"
                     and callable(getattr(champ, "fit_and_score", None))
                     and (not (cfg or {}).get("hpo")
                          or _champ_es == "validation"))
    if _clone_needed:
        _clone_probe_rep = _probe_champion_clone(champ_dir, champ)
        step(f"clone probe: {_clone_probe_rep.get('status')} "
             f"({_clone_probe_rep.get('model_class')}) "
             f"{_clone_probe_rep.get('error') or ''}".rstrip())
        if _clone_probe_rep.get("status") == "error":
            if _champ_es != "validation":
                logger.error(
                    "frozen_rounds 需要可 sklearn.clone 的冠军模型(clone 探测:"
                    "%s)。改用 --refit-mode rerun_recipe(复跑冠军配方,"
                    "不依赖 clone)", _clone_probe_rep.get("error"))
                return 1
            _refit_fallback_reason = (
                "clone_probe_failed: 冠军模型不可 sklearn.clone("
                f"{_clone_probe_rep.get('error')});早停面=validation,"
                "改为 train_tail 口径复跑冠军配方(train+validation 最新尾部"
                "早停后全量重拟合),未执行 train 上复现")
            refit_mode = "rerun_recipe"
            logger.warning("交付 refit 口径回落 frozen_rounds → rerun_recipe:%s",
                           _refit_fallback_reason)

    # ---------- 特征面(冠军同口径) ----------
    splits = _resolve_splits_dir(run_dir, cfg)
    # v2.10.1/P4:盲测打分回退目录 = 基础 v1 splits(final_test 唯一
    # 在基础面;见 _score_split 内的回退)。与 _resolve_splits_dir 同
    # 款兜底,保证 view 目录存在性判断与读取一致。
    _view_name = str(cfg.get("data_view") or "v1")
    _base_splits = run_dir / "sample-profile" / "splits"
    splits_base = _base_splits if _base_splits.exists() else splits
    header = pd.read_parquet(splits / "train.parquet").columns
    # v2.10.1/F3:解析域 = 基础帧列 ∪ config 声明的白名单列。旧实现只用
    # 基础 header 解析,而严格白名单(--features @file)按定义包含
    # transform 派生列(不在基础帧里),求交后派生列被静默丢弃
    # (实测:冠军 2017 列 → 交付只解析出 1493 列,528 个加宽列全丢,
    # 交付模型与冠军不是同一个特征面)。
    _declared = ((cfg.get("features") or {}).get("columns")
                 or (cfg.get("features") or {}).get("include") or [])
    _domain = list(dict.fromkeys(
        [c for c in header if isinstance(c, str)]
        + [c for c in _declared if isinstance(c, str)]))
    feats = champ.exclude_columns(pd.DataFrame(columns=_domain), cfg)
    step(f"features resolved: {len(feats)} cols "
         f"(base_header={len(header)} declared={len(_declared)})")
    # v2.17.2:严格白名单(features.columns)按定义包含 transform 派生列
    # (本 run 为 trial_feature_blocks 的 635 个块列),它们不在基础 parquet
    # 中。read_parquet 只能取基础帧列,派生列由各 transform 分支在治理后
    # 右接;直接传 feats 会 ArrowInvalid(实测 run4-v2.16.0-2.deepseek4.1flash
    # 冠军 013/015,base_header=1493 declared=1644)。训练期同一逻辑天然
    # 成立:champion 的 exclude_columns 是在 df.columns 上求交的。
    _base_set = set(c for c in header if isinstance(c, str))
    feats_base = [c for c in feats if c in _base_set]
    if len(feats_base) != len(feats):
        step(f"features base-read face: {len(feats_base)} cols "
             f"(派生列 {len(feats) - len(feats_base)} 个由 transform 分支右接)")

    lab_tr = champ.load_labels("train")
    lab_va = champ.load_labels("validation")
    y_tr_raw = lab_tr[spec.target_name].to_numpy()

    # ---------- transform:冠军 trial 的 build 通道按需执行 ----------
    # 手写加宽 trial 用 build_wide(reparse 词表);无该符号的 trial
    # (模板/baseline 血缘)直接读基础帧列。两条路都归一到
    # prepare_features 的 dtype 治理。
    # v2.10.1/P7:冠军为**手写加宽**(transform 在 fit_and_score 内的
    # _build_widen_columns 里做,无 build_wide)——finalize 重训也必须
    # 重放该 transform,否则交付模型退化成纯基础 1494 列(加宽列
    # 全部丢失,模型与冠军不是同一个特征面)。本 run 冠军已带
    # 列集合冻结(widen_cols_frozen.json,首次 run 时落盘):直接调用
    # 冠军的 _build_widen_columns 对 train+validation 各造一次加宽列
    # (冻结分支按列名物化,跨轮稳定,不重新跑 train 内选择器),再
    # 右接基础面;y 传 train 标签(冻结分支不读取 y 的预测语义,
    # 只保证函数签名与选择器分支兼容)。未实现冻结的旧手写冠军
    # 会走选择器分支(等价抽样面,列集合随机种子漂移——如实降级)。
    # v2.13.5 第五族(标准契约 trial_feature_blocks):特征块由
    # runtime.feature_cache 内容寻址缓存 —— 冠军 trial 算过的块在这里直接
    # 命中,不再像第三族那样把整张 reparse 串表读进内存再逐行重放
    # (req-t7-rand400w 实测交付重放 train+val 24 min + final 11 min +
    # shadow 7.5 min)。与 trial main 同序:基础面先治理,块后右接。
    # 定义了本契约的冠军只走本族(其余族符号即使存在也视为其内部实现)。
    _fb_mode = callable(getattr(champ, "trial_feature_blocks", None))
    _fb_cols: list = []
    if _fb_mode:
        X_tr = pd.read_parquet(splits / "train.parquet", columns=feats_base)
        X_va = pd.read_parquet(splits / "validation.parquet",
                               columns=feats_base)
        step("feature blocks mode: 基础面读入,块在 dtype 治理后右接")
    elif hasattr(champ, "build_wide"):
        V = None
        if hasattr(champ, "_build_vocab"):
            toks = []
            tr_rep = champ.reparse_frame(
                "train", sorted(getattr(champ, "REPARSE_SOURCES")))
            cols_rep = [c for c in tr_rep.columns
                        if str(tr_rep[c].dtype) == "object"]
            for c in cols_rep:
                s = tr_rep[c].to_numpy()
                toks.append(pd.unique(np.concatenate(
                    [np.asarray(x.split(","), dtype=object)
                     for x in s[np.arange(0, len(tr_rep), 5)]])))
            for sp in ("validation", "shadow_test", "final_test"):
                try:
                    sv = champ.reparse_frame(sp, sorted(
                        getattr(champ, "REPARSE_SOURCES")))
                    for c in cols_rep:
                        s = sv[c].to_numpy()
                        toks.append(pd.unique(np.concatenate(
                            [np.asarray(x.split(","), dtype=object)
                             for x in s])))
                except Exception:  # noqa: BLE001  split 缺失则词表少一路
                    continue
            V = champ._build_vocab(np.concatenate(toks))
            step(f"vocab built: tokens={len(V.get('voc', []))}")
            del tr_rep
        step("build_wide train start")
        X_tr = champ.build_wide("train", feats, V, y_tr=y_tr_raw)
        step("build_wide validation start")
        X_va = champ.build_wide("validation", feats, V)
    elif hasattr(champ, "_build_widen_columns"):
        # P7:手写加宽冠军的重放路径(train+validation 都加宽,右接)
        _X0_tr = pd.read_parquet(splits / "train.parquet", columns=feats_base)
        _X0_va = pd.read_parquet(splits / "validation.parquet",
                                 columns=feats_base)
        _X0_tr, _X0_va = champ.prepare_features(_X0_tr, _X0_va)
        _XW_tr, _XW_va, _wa = champ._build_widen_columns(
            _X0_tr, _X0_va, y_tr_raw, seed)
        step(f"widen replay: base={_X0_tr.shape[1]} -> "
             f"total={_X0_tr.shape[1] + _XW_tr.shape[1]} "
             f"(+{_XW_tr.shape[1]}) {_wa}")
        X_tr = pd.concat([_X0_tr, _XW_tr], axis=1)
        X_va = pd.concat([_X0_va, _XW_va], axis=1)
    elif (hasattr(champ, "_widen_pruned")
            or hasattr(champ, "_widen")):
        # v2.11.4:第三族加宽符号(_widen/_widen_pruned/widen_cached,
        # 见冠军 code:1672/1717/1740)。与冠军 main() 的
        # `_widen_pruned(...) -> concat -> prepare_features` 顺序逐字对齐:
        # 加宽在**未治理的原始基础帧**上做(seq 解析与 dtype 无关),
        # concat 回基础面(加列不替换),再统一走下方 prepare_features。
        _X0_tr = pd.read_parquet(splits / "train.parquet", columns=feats_base)
        _X0_va = pd.read_parquet(splits / "validation.parquet",
                                 columns=feats_base)
        _rep_cols = sorted(getattr(champ, "REPARSE_SOURCES", []) or [])
        _re_tr = champ.reparse_frame("train", _rep_cols)
        _re_va = champ.reparse_frame("validation", _rep_cols)
        _wp = getattr(champ, "_widen_pruned", None)
        if callable(_wp):
            _XW_tr, _XW_va = _wp("finalize_retrain", _X0_tr, _re_tr,
                                 _X0_va, _re_va)[:2]
        else:
            _XW_tr, _XW_va = champ._widen(_X0_tr, _re_tr,
                                          _X0_va, _re_va)[:2]
        _sc = _widen_self_contained(_X0_tr, _XW_tr)
        step(f"widen replay(third-family): base={_X0_tr.shape[1]} "
             f"+{_XW_tr.shape[1]} self_contained={int(_sc)}")
        if _sc:
            # v2.11.8:加宽输出已自含基础面 ⇒ 直接用,禁止二次 concat
            #(否则重名列帧在下游 prepare_features 炸出 TypeError)
            X_tr, X_va = _XW_tr, _XW_va
        else:
            X_tr = pd.concat([_X0_tr, _XW_tr], axis=1)
            X_va = pd.concat([_X0_va, _XW_va], axis=1)
        _assert_no_duplicate_columns(X_tr, "finalize-retrain/train")
        _assert_no_duplicate_columns(X_va, "finalize-retrain/validation")
        del _XW_tr, _XW_va, _re_tr, _re_va
    elif (hasattr(champ, "widen_face")
            and hasattr(champ, "fit_widen_state")
            and hasattr(champ, "_apply_widen")):
        # v2.11.7:第四族加宽符号。前三族:build_wide(手写词表)/
        # _build_widen_columns(手写加宽)/ _widen|_widen_pruned(缓存族)。
        # 本 run 冠军 trial-014(gen-01 超级 trial 血缘)三者皆在、前三族
        # 皆不在 ⇒ 四族两两不匹配 ⇒ 落 else 只读基础帧 ⇒ 交付面从 2637
        # 退化到 1493(v2.11.4 已为第三族修过同款,本处是它的续集)。
        # 与冠军 main() 时序对齐:
        #   prepare_features(train[feats]) → widen_face("main", ...)
        # widen_face 自行取 reparse 并在 train 侧拟合统计量。
        _X0_tr = pd.read_parquet(splits / "train.parquet", columns=feats_base)
        _X0_va = pd.read_parquet(splits / "validation.parquet",
                                 columns=feats_base)
        _X0_tr, _X0_va = champ.prepare_features(_X0_tr, _X0_va)[:2]
        _XW_tr, _XW_va, _wst, _wstate = champ.widen_face(
            "finalize_retrain", "train", "validation",
            _X0_tr, _X0_va, y_tr_raw)
        step(f"widen replay(fourth-family): base={_X0_tr.shape[1]} "
             f"-> total={_XW_tr.shape[1]} "
             f"(+{_XW_tr.shape[1] - _X0_tr.shape[1]}) {_wst}")
        # 交付打分必须复用同一份 train 侧 state:打分帧自行拟合统计量
        # = 用本帧分布做变换 = 泄漏口径。挂在冠军模块上跨函数传递,
        # 由 _score_split 读取(见下方 _FINALIZE_WSTATE 消费点)。
        try:
            champ._FINALIZE_WSTATE = _wstate
        except Exception:  # noqa: BLE001  模块只读时打分侧退回重拟合
            pass
        X_tr, X_va = _XW_tr, _XW_va
        del _X0_tr, _X0_va, _XW_tr, _XW_va
    else:
        X_tr = pd.read_parquet(splits / "train.parquet", columns=feats_base)
        X_va = pd.read_parquet(splits / "validation.parquet",
                               columns=feats_base)
    y_tr = lab_tr[spec.target_name].reset_index(drop=True)
    y_va = lab_va[spec.target_name].reset_index(drop=True)
    del lab_tr, lab_va
    # v2.10.1/F2:prepare_features 的返回形态随 trial 血缘而变 ——
    # 模板血缘返回 (X_tr, X_va)(dtype 治理在其内部完成,cat_cols/
    # dropped 不对外),手写血缘可能返回 4 元组。旧实现硬解 4 元组,
    # 模板血缘的冠军在交付期必然 ValueError 崩溃(实测 drawdown-req-t7-v20)。
    if hasattr(champ, "prepare_features"):
        _pf = champ.prepare_features(X_tr, X_va)
    else:
        # baseline 渲染代码把 dtype 治理内联在 main()，只导出
        # prepare_shadow_features；确认 Trial 复制该血缘后同样没有
        # prepare_features。交付重训必须复刻 baseline 的确定性治理，
        # 不能因缺少一个模板符号而让所有 baseline/confirm 冠军崩溃。
        cat_cols, dropped = [], []
        X_tr, X_va = X_tr.copy(), X_va.copy()
        for _c in list(X_tr.columns):
            if pd.api.types.is_numeric_dtype(X_tr[_c]):
                continue
            _conv = pd.to_numeric(X_tr[_c], errors="coerce")
            if _conv.notna().sum() >= 0.95 * max(
                    X_tr[_c].notna().sum(), 1):
                X_tr[_c] = _conv.astype("float32")
                X_va[_c] = pd.to_numeric(
                    X_va[_c], errors="coerce").astype("float32")
            elif X_tr[_c].nunique() <= 100:
                cat_cols.append(_c)
            else:
                dropped.append(_c)
        if dropped:
            X_tr = X_tr.drop(columns=dropped)
            X_va = X_va.drop(columns=[c for c in dropped
                                      if c in X_va.columns])
        if cat_cols:
            X_tr = pd.get_dummies(X_tr, columns=cat_cols)
            X_va = pd.get_dummies(X_va, columns=[
                c for c in cat_cols if c in X_va.columns])
            X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
        _pf = (X_tr.fillna(0).astype("float32"),
               X_va.fillna(0).astype("float32"), cat_cols, dropped)
    if isinstance(_pf, (tuple, list)) and len(_pf) == 4:
        X_tr, X_va, cat_cols, dropped = _pf
    elif isinstance(_pf, (tuple, list)) and len(_pf) == 2:
        X_tr, X_va = _pf
        # 2 元组血缘:cat_cols/dropped 由 trial 内部持有(module 常量或
        # _govern 闭包)。交付侧只需在 _score_split 复用同一 _govern。
        cat_cols = list(getattr(champ, "CAT_COLS", []) or [])
        dropped = list(getattr(champ, "DROPPED", []) or [])
    else:
        logger.error(
            "prepare_features 返回形态无法识别:%s(期望 2 或 4 元组)",
            type(_pf).__name__)
        return 1
    step(f"dtype governance done: cols={X_tr.shape[1]} "
         f"onehot={len(cat_cols)} dropped={len(dropped)}")
    if _fb_mode:
        from runtime.feature_cache import align_rows as _fb_align

        _W_tr = champ.trial_feature_blocks("train")
        _W_va = champ.trial_feature_blocks("validation")
        _fb_cols = list(_W_tr.columns)
        X_tr = pd.concat([X_tr, _fb_align(_W_tr, X_tr).astype("float32")],
                         axis=1)
        X_va = pd.concat([X_va, _fb_align(_W_va, X_va).astype("float32")],
                         axis=1)
        del _W_tr, _W_va
        _assert_no_duplicate_columns(X_tr, "finalize-retrain/train")
        _assert_no_duplicate_columns(X_va, "finalize-retrain/validation")
        step(f"feature blocks replay: +{len(_fb_cols)} cols "
             f"-> {X_tr.shape[1]}")
    cols = list(X_tr.columns)
    # v2.13.4 重训证明:分项行数写入 model_meta,evaluate-final 据此校验
    # 「训练集 = train + validation」而不是 train-only 搜索模型。
    _n_tr_rows, _n_va_rows = int(len(X_tr)), int(len(X_va))
    X_all = pd.concat([X_tr, X_va], axis=0, ignore_index=True)
    y_all = pd.concat([y_tr, y_va], axis=0,
                      ignore_index=True).reset_index(drop=True)
    step(f"train+validation combined {X_all.shape}")

    # ---------- 交付重训(v2.15.9:refit_mode 分派)----------
    # rerun_recipe(默认):train+validation 上重跑冠军配方,早停落在
    #   train+validation 的最新尾部 ES_FRACTION;v2.16 单段配方只训一次(头部
    #   1-ES_FRACTION 拟合),v2.15 两段式旧代码才全量重拟合(retrain_scope 如实记);
    # frozen_rounds(兼容):train 尾部早停定轮 → clone → train+validation 重拟合。
    es_frac = float(getattr(champ, "ES_FRACTION", 0.1) or 0.1)
    _rerun = refit_mode == "rerun_recipe"
    _es_X, _es_y = (X_all, y_all) if _rerun else (X_tr, y_tr)
    es_cut = int(len(_es_X) * (1.0 - es_frac))
    _es_window = _es_holdout_window(run_dir, splits, _n_tr_rows, _n_va_rows,
                                    es_frac) if _rerun else None
    n_jobs = champ.n_jobs() if hasattr(champ, "n_jobs") else 16
    hp_inj = (cfg.get("hyperparameters") or {}) \
        if isinstance(cfg.get("hyperparameters"), dict) else {}
    # v2.11.7:hpo 血缘冠军的 best_params **不在** config.hyperparameters
    # (该节为空 {}),而在 trials/<id>/hpo-study.json 与 metrics.json。
    # 旧实现只读 config ⇒ 交付模型回落默认轮数(实测 n_estimators=2000,
    # 冠军实为 570 轮 / lr 0.011 / colsample 0.589)。
    # 回落次序:config(人工 --set 覆写优先) → hpo-study.json(搜索产物,
    # 权威) → metrics.json(best_params 落盘副本)。
    if not hp_inj:
        for _hp_src in (champ_dir / "hpo-study.json",
                        champ_dir / "metrics.json"):
            if not _hp_src.exists():
                continue
            try:
                _blob = json.loads(_hp_src.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            _bp = (_blob or {}).get("best_params") or {}
            if isinstance(_bp, dict) and _bp:
                hp_inj = {k: v for k, v in _bp.items()
                          if isinstance(v, (int, float, str, bool))}
                step(f"hyperparameters from {_hp_src.name}: {hp_inj}")
                break
    # v2.13.4:非 HPO 冠军优先走**冠军自己的训练代码**(构造保证同一配方),
    # 并校验复现分数与冠军一致(验证兜底)。HPO 冠军的 _build_model 带
    # _HPO_ALGO,沿用下方适配器 + best_params 路径。
    model = None
    _repro = None
    _face = None
    _rerun_report = None
    n_est = None
    best_it = None
    _retrain_builder = "estimator_adapter"
    _frozen_sub: dict = {}
    _has_fs = callable(getattr(champ, "fit_and_score", None))
    # v2.19.0:复现方式留痕(reused_frozen_model / full_reproduction /
    # full_reproduction_forced / none)
    _reproduction_mode = "none"
    if _rerun and _has_fs:
        # (1) 交付面证明:特征帧与冠军真实 validation 输入逐列一致;冠军无
        #     该工件或 --full-reproduction 时退回 train 上的完整复现。
        try:
            _cfg_dl, _hp_src_name = _strip_hpo_cfg(cfg, champ_dir)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
        if _hp_src_name:
            step(f"hpo 节已剥离,重训配方取自 {_hp_src_name}")
        _face = _check_feature_face(X_va, cols, champ_dir)
        step(f"feature face check: passed={_face.get('passed')} "
             f"mismatched={_face.get('mismatched_columns')}")
        if _face.get("passed") is False:
            logger.error(
                "交付特征面与冠军 validation 输入不一致:%s —— 交付管线"
                "未忠实重放冠军特征口径,拒绝重训", _face)
            return 1
        if _face.get("passed") is None or _full_repro:
            _m_ref, _repro = _reproduce_champion_fit(
                champ, X_tr, y_tr, X_va, y_va, seed, _cfg_dl, champ_dir,
                spec)
            if not (_repro or {}).get("passed"):
                logger.error("冠军复现校验失败:%s —— 拒绝产出重训模型",
                             _repro)
                return 1
            step(f"champion reproduced on train: {_repro}")
            _reproduction_mode = ("full_reproduction_forced" if _full_repro
                                  else "full_reproduction")
            del _m_ref
        # 证明完成后释放 train-only 帧(交付训练只用 X_all;峰值内存约 -1/3)
        X_tr = X_tr.iloc[:0]
        # (2) 复跑配方:train+validation 进 fit_and_score,冠军代码自行在
        #     X_all 头部拟合、尾部早停(单段配方只训一次;旧两段式代码会再
        #     全量重拟合);返回模型直接交付,不 clone。
        try:
            model, _rerun_report = _rerun_champion_recipe(
                champ, X_all, y_all, X_va, y_va, seed, _cfg_dl, champ_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rerun_recipe 失败:%s —— 不回落估计器适配器(会静默换模型);"
                "修复冠军 fit_and_score 契约,或 --refit-mode frozen_rounds",
                exc)
            return 1
        _retrain_builder = "champion_fit_and_score_rerun"
        n_est = (_rerun_report or {}).get("rounds")
        best_it = n_est
        step(f"retrained on train+validation (rerun recipe): "
             f"{type(model).__name__} es_tail={_es_window}")
    elif not _rerun and _has_fs and (not (cfg or {}).get("hpo")
                                     or _champ_es == "validation"):
        # v2.16.0:validation 早停的 HPO 冠军也走冻结轮数(hpo 节剥离、
        # best_params 注入,与 rerun 路径同一 _strip_hpo_cfg)
        _cfg_fr = cfg
        if (cfg or {}).get("hpo"):
            try:
                _cfg_fr, _hp_src_name = _strip_hpo_cfg(cfg, champ_dir)
            except RuntimeError as exc:
                logger.error("%s", exc)
                return 1
            step(f"hpo 节已剥离,冻结轮数配方取自 {_hp_src_name}")
        # v2.19.0:先尝试复用冠军(或其 confirm 子代)的冻结模型(validation
        # 重打分逐位一致即视为已证明),不再在 train 上完整复现;
        # --full-reproduction 强制旧路径
        _m_ref = None
        _reuse_rep = None
        if not _full_repro:
            _m_ref, _reuse_rep = _frozen_model_reuse(
                _reuse_candidate_dirs(run_dir, champ_id, trials), champ,
                X_va, spec)
            step(f"frozen model reuse: passed={_reuse_rep.get('passed')} "
                 f"source={_reuse_rep.get('source_trial')} "
                 f"{_reuse_rep.get('reason') or ''}".rstrip())
            if _m_ref is not None:
                _repro = _reuse_rep
                _reproduction_mode = "reused_frozen_model"
        if _m_ref is None:
            try:
                _m_ref, _repro = _reproduce_champion_fit(
                    champ, X_tr, y_tr, X_va, y_va, seed, _cfg_fr, champ_dir,
                    spec)
                _reproduction_mode = ("full_reproduction_forced" if _full_repro
                                      else "full_reproduction")
                if _reuse_rep is not None and isinstance(_repro, dict):
                    _repro["reuse_attempt"] = _reuse_rep
            except Exception as _exc:  # noqa: BLE001
                logger.warning("冠军 fit_and_score 复现失败(%s),回退 estimator "
                               "适配器(算法族校验仍会兜底)", _exc)
                _m_ref = None
        if _m_ref is not None:
            if not (_repro or {}).get("passed"):
                logger.error(
                    "冠军复现校验失败:%s —— 交付重训的特征面/模型与冠军"
                    "不一致,拒绝产出重训模型", _repro)
                return 1
            step(f"champion {'reused (frozen model)' if _reproduction_mode == 'reused_frozen_model' else 'reproduced on train'}: {_repro}")
            from sklearn.base import clone as _sk_clone

            # v2.16.0:单段早停模型含早停耐心那段树,轮数取最优树数
            n_est = (_validation_es_rounds(_m_ref)
                     if _champ_es == "validation"
                     else _fitted_rounds(_m_ref)) or int(
                getattr(_m_ref, "n_estimators", 0) or 0) or None
            best_it = n_est
            _composite_rerun = False
            try:
                model = _sk_clone(_m_ref)
            except TypeError as exc:
                if _champ_es != "validation":
                    # v2.15.9:复合容器(如两段式 hurdle)不是 sklearn estimator。
                    # 禁止静默回落适配器 —— trial 工件以 __main__ pickle 时族
                    # 校验失效,回落等于悄悄交付另一个模型。
                    logger.error(
                        "frozen_rounds 需要可 sklearn.clone 的冠军模型:%s。"
                        "改用 --refit-mode rerun_recipe(复跑冠军配方,"
                        "不依赖 clone)", exc)
                    return 1
                # v2.16.0:validation 早停的复合容器(hurdle 等)不可 clone ——
                # 以 train_tail 口径复跑冠军配方:早停落在 train+validation 的
                # 最新尾部(validation 已并入训练集,不能再拿它早停),返回模型
                # 直接交付(v2.15.9 rerun_recipe 同一实现)
                # v2.17.0:到这里说明前置 clone 探测拿不到模型对象(unavailable);
                # 回落同样写入 model_meta.refit_mode_fallback_reason。
                _refit_fallback_reason = (
                    f"clone_failed_after_reproduction: {type(exc).__name__}: "
                    f"{exc};早停面=validation,改为 train_tail 口径复跑冠军配方"
                    "(train+validation 最新尾部早停后全量重拟合)")
                logger.warning("交付 refit 口径回落 frozen_rounds → "
                               "rerun_recipe:%s", _refit_fallback_reason)
                step(f"冠军不可 clone({exc});validation 早停冠军改为 "
                     "train_tail 口径复跑配方(train+validation 尾部早停)")
                _es_prev = os.environ.get("MODELEVO_ES_SOURCE")
                os.environ["MODELEVO_ES_SOURCE"] = "train_tail"
                try:
                    model, _rerun_report = _rerun_champion_recipe(
                        champ, X_all, y_all, X_va, y_va, seed, _cfg_fr,
                        champ_dir)
                except Exception as exc2:  # noqa: BLE001
                    logger.error("复合容器冠军复跑配方失败:%s —— 不回落估计器"
                                 "适配器(会静默换模型)", exc2)
                    return 1
                finally:
                    if _es_prev is None:
                        os.environ.pop("MODELEVO_ES_SOURCE", None)
                    else:
                        os.environ["MODELEVO_ES_SOURCE"] = _es_prev
                _composite_rerun = True
                refit_mode, _rerun = "rerun_recipe", True
                _es_window = _es_holdout_window(run_dir, splits, _n_tr_rows,
                                                _n_va_rows, es_frac)
                n_est = (_rerun_report or {}).get("rounds")
                best_it = n_est
                _retrain_builder = "champion_fit_and_score_rerun"
                step(f"retrained on train+validation (rerun recipe, "
                     f"train_tail ES): {type(model).__name__} "
                     f"es_tail={_es_window}")
        if _m_ref is not None and not _composite_rerun:
            if n_est:
                _freeze_rounds(model, n_est)
            _clear_early_stopping(model)
            # v2.17.0:可 clone 的复合容器(官方 HurdleRegressor 等 sklearn
            # 协议容器)逐子估计器冻结最优轮数、关闭早停
            _frozen_sub = _composite_frozen_params(_m_ref, _champ_es)
            if _frozen_sub:
                try:
                    model.set_params(**_frozen_sub)
                except Exception as _exc_fz:  # noqa: BLE001
                    logger.warning("复合容器冻结轮数 set_params 失败(%s):%s",
                                   _exc_fz, _frozen_sub)
                step(f"composite frozen rounds: {_frozen_sub}")
            if _reproduction_mode == "reused_frozen_model":
                _rt = _align_reused_runtime_params(model, champ, len(X_tr))
                if _rt:
                    step(f"reused model runtime params aligned: {_rt}")
                    if isinstance(_repro, dict):
                        _repro["runtime_params_aligned"] = _rt
            model.fit(X_all, y_all)
            _retrain_builder = "champion_fit_and_score"
            step(f"retrained on train+validation (champion code): "
                 f"{type(model).__name__} rounds={n_est}")
    if model is None:
        _retrain_builder = ("estimator_adapter_rerun" if _rerun
                            else "estimator_adapter")
        m1 = _finalize_build_estimator(
            champ, n_jobs, es_cut, hp_inj or None)
        step(f"train start(es): rows={es_cut} features={_es_X.shape[1]} "
             f"es_on={'train+validation tail' if _rerun else ('outer validation' if _champ_es == 'validation' else 'train tail')}")
        # v2.11.7:lightgbm 4.x 的 sklearn API 已移除 `fit(verbose=)`
        # (实测 4.6.0;3.x 接受该参数),xgb/catboost 的 fit 仍接受 ⇒
        # **只有 lightgbm 血缘的冠军**在此必然抛
        #   TypeError: fit() got an unexpected keyword argument 'verbose'
        # 交付链在写 model_meta 前中断(实测 run-ddreq-t7-v2:冠军
        # trial-011 崩于 scripts/modelevo.py:3723,而同族的
        # 旧 baseline 兜底虽能跑通，却会把 eval_set 一并丢掉，使
        # best_iteration=0 并退回上限轮数。现统一按 fit 签名注入
        # eval_set/verbose，并为 LightGBM 4.x 显式注入 early_stopping
        # callback；冠军与 baseline 共用同一版本安全语义。
        _es_rounds = int(getattr(
            champ, "EARLY_STOP_ROUNDS",
            config.get("baseline.early_stopping_rounds", 50)) or 50)
        if _champ_es == "validation" and not _rerun:
            # v2.16.0:与冠军同口径 —— 全量 train 拟合、outer validation 早停
            # (一次拟合定轮数),随后在 train+validation 上冻结轮数重拟合
            _es_kw = _finalize_es_fit_kwargs(m1, X_va, y_va, _es_rounds)
            m1.fit(X_tr, y_tr, **_es_kw)
        else:
            _es_kw = _finalize_es_fit_kwargs(
                m1, _es_X.iloc[es_cut:], _es_y.iloc[es_cut:], _es_rounds)
            if "eval_set" in _es_kw:
                m1.fit(_es_X.iloc[:es_cut], _es_y.iloc[:es_cut], **_es_kw)
            else:
                # 非 GBM estimator(例如内建早停的 MLP)不接受 eval_set;让它在
                # 完整训练面上按自己的契约拟合,禁止把未知 kwargs 强塞进 fit。
                m1.fit(_es_X, _es_y, **_es_kw)
        best_it = _finalize_best_iteration(m1)
        step(f"train(es) done: best_iteration={best_it}")
        _configured_n_est = (
            hp_inj.get("n_estimators")
            or getattr(m1, "n_estimators", None)
            or getattr(champ, "N_ESTIMATORS_CAP", None)
            or config.get("baseline.n_estimators_cap", 2000))
        from runtime.config import clamp_trees
        n_est = clamp_trees(best_it or int(_configured_n_est))
        model = _finalize_build_estimator(
            champ, n_jobs, len(X_all), hp_inj or None)
        _rounds_key = _freeze_rounds(model, n_est)
        _clear_early_stopping(model)
        model.fit(X_all, y_all)
        step(f"retrained on train+validation: {_rounds_key or 'n_estimators'}={n_est}")

    # v2.13.4:算法族一致性(任何构造路径都兜底)。重训模型必须与冠军
    # trial 的模型同一库,否则交付的不是冠军配方。
    _champ_model_p = champ_dir / "model" / "model.joblib"
    if _champ_model_p.exists():
        try:
            _fam_ref = _model_family(joblib.load(_champ_model_p))
        except Exception:  # noqa: BLE001  无法反序列化则不校验族
            _fam_ref = None
        if _fam_ref and _fam_ref != _model_family(model):
            logger.error(
                "重训模型算法族 %s ≠ 冠军模型 %s(builder=%s)——交付的不是"
                "冠军配方,拒绝落盘", _model_family(model), _fam_ref,
                _retrain_builder)
            return 1
    # ---------- 工件 ----------
    mdir = fm / "model"
    # v2.13.4:model/ 整目录归本命令所有,落盘前清空上一轮残留(换算法族时
    # 旧原生格式文件会与新模型并存,例如 xgboost 的 model.json 旁边留着
    # 上一轮 lightgbm 的 model.txt)。放在模型训练与校验通过之后,失败不丢旧产物。
    import shutil as _shutil_md

    _shutil_md.rmtree(mdir, ignore_errors=True)
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, mdir / "model.joblib")
    # v2.11.7:原生格式按血缘分派。旧实现对**所有**算法硬编码
    # `save_model("model.json")`,而 lightgbm 的原生文本格式要求 .txt
    # (模板 dump_model 即按血缘分派:lgb→model.txt / xgb→model.json /
    # catboost→model.bin)。lightgbm 血缘的冠军在此必然保存失败,却又被
    # `except Exception: pass` **静默吞掉** ⇒ final-model/model/ 缺原生
    # 格式(实测 run-ddreq-t7-v2:trial-011 自己的 model/ 有 model.txt
    # 2.3MB,交付产物里没有),而"模型工件齐备(joblib+原生格式+
    # feature_columns.json)"是交付校验项。与模板同口径分派,并把失败
    # 降级为 WARNING —— 静默是这处漏检的真正原因。
    try:
        _mod = type(model).__module__ or ""
        if _mod.startswith("lightgbm") and hasattr(model, "booster_"):
            model.booster_.save_model(str(mdir / "model.txt"))
        elif _mod.startswith("xgboost") and hasattr(model, "save_model"):
            model.save_model(str(mdir / "model.json"))
        elif _mod.startswith("catboost") and hasattr(model, "save_model"):
            model.save_model(str(mdir / "model.bin"))
        elif hasattr(model, "save_model"):
            model.save_model(str(mdir / "model.json"))
        elif _model_family(model).startswith("custom:"):
            # v2.16.3:自定义复合容器(两段式 hurdle 等)逐个子模型存原生格式
            for _attr, _sub in vars(model).items():
                _smod = type(_sub).__module__ or ""
                _tag = _attr.strip("_")
                if _smod.startswith("lightgbm") and hasattr(_sub, "booster_"):
                    _sub.booster_.save_model(str(mdir / f"model.{_tag}.txt"))
                elif _smod.startswith("xgboost") and hasattr(_sub, "save_model"):
                    _sub.save_model(str(mdir / f"model.{_tag}.json"))
                elif _smod.startswith("catboost") and hasattr(_sub, "save_model"):
                    _sub.save_model(str(mdir / f"model.{_tag}.bin"))
    except Exception as _exc:  # noqa: BLE001  非 GBM 血缘无原生格式
        logger.warning("原生格式落盘失败(joblib 仍可用): %s", _exc)
    (mdir / "feature_columns.json").write_text(
        json.dumps(cols), encoding="utf-8")
    X_va.to_parquet(mdir / "validation_input.parquet", index=False)
    try:
        pd.DataFrame({
            "feature": cols,
            "importance": list(model.feature_importances_),
        }).sort_values("importance", ascending=False).to_csv(
            mdir / "feature_importance.csv", index=False)
    except Exception:  # noqa: BLE001
        pass

    missing_levels: dict[str, list[str]] = {}

    def _score_split(split: str, out_name: str) -> None:
        if hasattr(champ, "build_wide") and "V" in dir():
            X = champ.build_wide(split, feats, V)
        else:
            # v2.10.1/P4:交付打分特征面回退到基础 splits。冠军训练用
            # data_view=sample 的 splits-sample(train/validation 在
            # 抽样面重写),但抽样面**不含 final_test**(抽样面按协议只
            # 覆盖 train/validation;final_test 唯一、原样留 v1 基础
            # splits)。finalize 的 _score_split 若仍读 sample 视图目录,
            # final_test/shadow 打分必然 FileNotFoundError(本 run 实测)。
            # 训练面(train+validation)仍在抽样视图;盲测打分必须回到
            # 基础 splits(final_test 是干净面,v1 基础 splits 唯一持有)。
            # 特征列集合两个视图完全一致(抽样按行,不改列)。
            _view_splits = splits
            try:
                X = pd.read_parquet(_view_splits / f"{split}.parquet",
                                    columns=feats_base)
            except FileNotFoundError:
                try:
                    X = pd.read_parquet(splits_base / f"{split}.parquet",
                                        columns=feats_base)
                except FileNotFoundError:
                    # v2.10.1/P6:split 本身不存在(本 run spec 无
                    # shadow 段——6/2/0/1 切分,split_policy 不声明
                    # shadow_test,基础面与抽样面都没有该 split)。
                    # 交付链按协议降级:无 shadow 的 run 不产
                    # shadow_test_score.parquet,evaluate-final 的
                    # shadow 硬校验据此跳过(shadow split 不存在)。
                    logger.warning(
                        "split %s 不存在(视图与基础面都没有),"
                        "跳过该打分——交付链按无 shadow 协议降级",
                        split)
                    return
        gov = getattr(champ, "_govern", None)
        if gov is not None:
            X = gov(X, cat_cols, dropped)
        # v2.10.1/P7:手写加宽冠军(无 build_wide)的盲测打分帧必须
        # 重放 _build_widen_columns 加宽(与训练面同口径)——否则
        # 下方 reindex 到 1780 列时加宽列全部缺失,显式报错退出。
        # 冻结列集合物化跨轮稳定,不重跑 train 内选择器;y 传占位
        # (冻结分支不读 y 的预测语义)。prepare_features 已在上方
        # 对基础帧做 dtype 治理,加宽在治理后的数值面上做。
        if _fb_mode:
            pass  # 第五族:块在下方治理之后右接(见 _fb_mode 分支)
        elif (hasattr(champ, "_build_widen_columns")
                and not hasattr(champ, "build_wide")):
            _Xw, _, _wa = champ._build_widen_columns(
                X, X, np.zeros(len(X)), seed)
            X = pd.concat([X, _Xw], axis=1)
        elif (hasattr(champ, "widen_face")
                and hasattr(champ, "_apply_widen")):
            # v2.11.7:第四族(widen_face)的打分帧重放。统计量优先复用
            # 训练侧挂载的 state(_FINALIZE_WSTATE,由 train 帧拟合),
            # **绝不**在本帧自身上重新拟合 —— 对 final_test/shadow 那样
            # 做等于用本帧分布做变换,是泄漏口径。state 缺失时退回 train
            # 帧拟合(仍不引入本帧信息,只是更慢)。
            _Xb = champ.prepare_features(X, X)[0]
            _wst_obj = getattr(champ, "_FINALIZE_WSTATE", None)
            if _wst_obj is None:
                _rc2 = sorted(
                    getattr(champ, "REPARSE_SOURCES", []) or [])
                _btr = pd.read_parquet(splits / "train.parquet",
                                       columns=feats_base)
                _btr = champ.prepare_features(_btr, _btr)[0]
                _ytr2 = None
                try:
                    _ytr2 = champ.load_labels("train")[
                        spec.target_name].reset_index(drop=True)
                except Exception:  # noqa: BLE001  无标签按无 y 口径
                    _ytr2 = None
                _wst_obj = champ.fit_widen_state(
                    _btr, _ytr2, champ.reparse_frame("train", _rc2))
                del _btr
                logger.warning(
                    "打分帧加宽 state 缺失,退回 train 帧重拟合(口径一致"
                    "但更慢);确认训练段已挂载 _FINALIZE_WSTATE")
            _rc_cur = sorted(getattr(champ, "REPARSE_SOURCES", []) or [])
            # v2.11.7:reparse 源取数必须与**特征面 split 同源**。冠军
            # 血缘的 SPLITS 可能指向抽样视图(splits-sample),而协议规定
            # final_test 不抽样、不落盘 ⇒ 索要
            # splits-sample/reparse/final_test.parquet 必然 FileNotFoundError
            # (实测 run-ddreq-t7-v2:模型工件已按 2637 列落盘,却在打分前
            # 崩,连带 recipe/model_meta 未回填 ⇒ 冠军单源不一致)。
            # load_labels 同款缺陷已由 v2.11.6 处理过一次,此处是其续集。
            # 处置:优先用冠军自己的解析源;失败时回退到基础面 v1 的
            # splits/reparse(抽样按行不改列,行序由对齐键保证一致),
            # 并把 train 侧结果缓存进进程内,供后续 split 复用。
            try:
                _re_cur = champ.reparse_frame(split, _rc_cur)
                _recache = getattr(champ, "_FINALIZE_REPARSE_CACHE", None)
                if _recache is None:
                    _recache = {}
                    try:
                        champ._FINALIZE_REPARSE_CACHE = _recache
                    except Exception:  # noqa: BLE001  模块只读则不缓存
                        _recache = None
                if _recache is not None:
                    _recache[split] = _re_cur
            except Exception as _re_exc:  # noqa: BLE001
                _recache = getattr(champ, "_FINALIZE_REPARSE_CACHE", None)
                _re_cur = (_recache or {}).get(split)
                if _re_cur is None:
                    _re_path = splits_base / "reparse" / f"{split}.parquet"
                    if not _re_path.exists():
                        raise
                    _re_cur = pd.read_parquet(_re_path)
                    _miss2 = [c for c in _rc_cur
                              if c not in _re_cur.columns]
                    if _miss2:
                        raise
                    _re_cur = _re_cur[_rc_cur]
                    logger.warning(
                        "冠军 reparse_frame(%s) 不可用(%s);回退基础面 "
                        "splits/reparse 同源同列取数(抽样按行不改列,行序"
                        "由对齐键保证)", split, type(_re_exc).__name__)
            X = champ._apply_widen(_Xb, _re_cur, _wst_obj)
            step(f"{split} widen replay(fourth-family): -> {X.shape[1]} cols")
        elif (hasattr(champ, "_widen_pruned")
                or hasattr(champ, "_widen")):
            # v2.11.4:第三族加宽符号的打分帧重放。必须发生在下方
            # prepare_features 之前 —— 与冠军 main() 的
            # `widen -> concat -> prepare_features` 顺序一致。
            _rep_cols = sorted(getattr(champ, "REPARSE_SOURCES", []) or [])
            _re = champ.reparse_frame(split, _rep_cols)
            _wp = getattr(champ, "_widen_pruned", None)
            if callable(_wp):
                _Xw = _wp("finalize_" + split, X, _re, X, _re)[0]
            else:
                _Xw = champ._widen(X, _re, X, _re)[0]
            # v2.11.8:同训练路径 —— 加宽输出自含基础面时不得二次 concat
            #(v2.11.4 假设"仅新增列",超级 trial 血缘返回的是完整帧;
            # 无条件 concat 会造出重名列,下方 prepare_features 直接崩)
            _sc = _widen_self_contained(X, _Xw)
            if _sc:
                X = _Xw
            else:
                X = pd.concat([X, _Xw], axis=1)
            _assert_no_duplicate_columns(X, f"finalize-score/{split}")
            step(f"{split} widen replay: -> {X.shape[1]} cols "
                 f"self_contained={int(_sc)}")
        # v2.10.1/P5:手写血缘冠军(无 build_wide,transform/dtype 治理
        # 在 prepare_features 内做)的打分帧必须复用同一治理——train
        # 路径经 prepare_features 把 object 列(prime_cust_mob6 业务序码、
        # loan_flag_v2、Decimal 金额)转成数值;盲测打分帧若不治理,
        # 在下方 reindex().astype('float32') 会抛 ValueError
        # ("could not convert string to float",本 run final_test 实测)。
        # 与冠军 main() 对 shadow 打分同款调用 prepare_features(X, X)[0]
        # (prepare_features 按 train 拟合序码/one-hot 词表,打分帧行内
        # 同源变换,不接触标签)。
        if (hasattr(champ, "prepare_features")
                and not hasattr(champ, "build_wide")):
            try:
                X = champ.prepare_features(X, X)[0]
            except Exception as _gov_e:  # noqa: BLE001
                logger.warning("prepare_features 治理打分帧失败(%s),"
                               "按原列集继续(缺失列检查会兜底)",
                               _gov_e)
        elif hasattr(champ, "prepare_shadow_features"):
            # baseline/confirm 血缘的训练治理内联在 main()，打分治理由
            # 这个导出函数承载；必须传入训练时识别出的类别/剔除列与
            # 最终列集合，才能保持 one-hot 词表和列序完全一致。
            X = champ.prepare_shadow_features(
                X, cat_cols, dropped,
                [c for c in cols if c not in set(_fb_cols)])[0]
        if _fb_mode:
            from runtime.feature_cache import align_rows as _fb_align2

            X = pd.concat([X, _fb_align2(champ.trial_feature_blocks(split),
                                         X).astype("float32")], axis=1)
            _assert_no_duplicate_columns(X, f"finalize-score/{split}")
            step(f"{split} feature blocks replay: -> {X.shape[1]} cols")
        # v2.10.1/F1:此处原为 .reindex(..., fill_value=0).fillna(0)。
        # 冠军为 xgboost/lightgbm 等 NaN 原生缺失路由时,训练与评测路径
        # 保留 NaN;交付打分若 fillna(0),缺失信息被抹成 0,打分口径与
        # 训练/选型口径分裂。实测(drawdown-req-t7-v20,2017 列、缺失
        # 单元格 22.9%):final_test 0.7842 → 0.7706,
        # improvement_vs_baseline_final_test 由 +0.0098 翻转为 -0.0038
        # (符号相反,交付结论整体反转)。缺列同样不再静默填 0 —— 那会
        # 在缺失语义之外再引入一整列的伪造常量,改为显式报错。
        # v2.11.4:缺列分两类处置。
        #  (a) **基础面列缺失** = 特征面口径漂移/transform 未重放 →
        #      保留 v2.10.1/F1 的硬报错(当年无差别 fill 0 使
        #      final_test 0.7842→0.7706、improvement 符号翻转)。
        #  (b) **派生展开列缺级** = one-hot/序码在"本打分批"未出现的
        #      稀有档(train/validation 齐全,单快照 final_test 缺席)
        #      → 补 0 是 one-hot 的正确编码,不是伪造常量;补 0 后
        #      该档系数在本批不参与,与训练语义一致。计入诊断供复核。
        _feats_set = set(feats)
        missing = [c for c in cols if c not in X.columns]
        _miss_base = [c for c in missing if c in _feats_set]
        _miss_derived = [c for c in missing if c not in _feats_set]
        if _miss_base:
            logger.error(
                "交付打分特征面缺基础列 %d/%d(前 10):%s —— 冠军 "
                "transform 未在交付路径重放,或特征面口径漂移;"
                "交付产物不可信",
                len(_miss_base), len(cols), _miss_base[:10])
            raise RuntimeError(
                f"delivery scoring frame missing base columns: "
                f"{_miss_base[:10]}")
        if _miss_derived:
            logger.warning(
                "%s: 派生展开列缺级 %d/%d(前 10):%s —— 该档在本批"
                "未出现,按 one-hot 语义补 0",
                split, len(_miss_derived), len(cols), _miss_derived[:10])
            missing_levels.setdefault(split, _miss_derived)
            for _c in _miss_derived:
                X[_c] = 0.0
        X = X.reindex(columns=cols).astype("float32")
        sc = _delivery_predict(model, X, spec)
        # 交付预测与 trial 预测使用同一身份协议。data_view=sample 时
        # final_test 只存在于基础 v1，按行数选择实际打分面的 identity；
        # 无键或键数不符一律拒绝，绝不让最终评测退化成位置配对。
        from runtime.data import SAMPLE_KEY_COLUMN
        from runtime.integrity import bind_prediction_artifact

        _identity = None
        _identity_root = None
        for _root in dict.fromkeys((splits, splits_base)):
            _ip = Path(_root) / "identity" / f"{split}.parquet"
            if not _ip.exists():
                continue
            _candidate = pd.read_parquet(_ip)
            if len(_candidate) == len(sc) and \
                    SAMPLE_KEY_COLUMN in _candidate.columns:
                _identity, _identity_root = _candidate, Path(_root)
                break
        if _identity is None or _identity_root is None:
            raise RuntimeError(
                f"{split} 缺与预测等长的 identity 样本键，拒绝落无身份预测")
        _score_path = fm / out_name
        pd.DataFrame({"score": sc, SAMPLE_KEY_COLUMN:
                      _identity[SAMPLE_KEY_COLUMN].astype(str).values})\
            .to_parquet(_score_path, index=False)
        bind_prediction_artifact(
            _score_path, _identity,
            source_manifest=_champ_source_manifest,
            split_manifest_path=_identity_root / "manifest.json",
            data_view=(str(cfg.get("data_view") or "v1")
                       if _identity_root == splits else "v1"),
            model_paths=[p for p in mdir.rglob("*") if p.is_file()])
        step(f"{split} scored: {X.shape} -> {out_name}")

    if unconfirmed_mode:
        # 未确认重训不产出任何可评测的盲测/shadow 预测:盲测机会与确认
        # 证据都只属于通过交付门的冠军。
        step("未确认重训:跳过 final_test/shadow_test 打分")
    else:
        _score_split("final_test", "final_test_score.parquet")
        _score_split("shadow_test", "shadow_test_score.parquet")

    # ---------- recipe / model_meta(冠军单源) ----------
    _trial_model_sha = _file_sha256(champ_dir / "model" / "model.joblib")
    _final_model_sha = _file_sha256(mdir / "model.joblib")
    if _final_model_sha is None:
        logger.error("重训模型未落盘:%s", mdir / "model.joblib")
        return 1
    if _trial_model_sha is not None and _final_model_sha == _trial_model_sha:
        logger.error("重训产物与冠军 trial 的 train-only 搜索模型哈希相同"
                     "——未真正重训,拒绝写入重训证明")
        return 1
    _delivery_status = ("unconfirmed" if unconfirmed_mode
                        else "confirmation_waived_by_prior" if _prior_waived
                        else "confirmation_waived" if _confirm_waived
                        else "confirmed")
    if _confirm_waived:
        _record_shadow_waiver(run_dir, champ_id,
                              "confirmation_waived_by_prior"
                              if _prior_waived else "confirmation_waived")
    # v2.16.3:按实际拟合方式描述(旧文案恒写 full refit,与单段配方不符)。
    # 估计器适配器路径(无 fit_and_score 的 baseline 血缘)确实是"头部早停
    # 定轮 → 新建估计器在 train+validation 全量重拟合"。
    _scope_evidence = _rerun_report
    if _retrain_builder == "estimator_adapter_rerun":
        _scope_evidence = {"full_refit_known": True}
    _retrain_scope = _retrain_scope_text(_rerun, _champ_es, es_frac,
                                         _scope_evidence)
    _shadow_pre = None if unconfirmed_mode else _shadow_precheck(
        run_dir, fm, champ_dir, spec)
    if _shadow_pre and _shadow_pre.get("worse_than_search_model"):
        logger.warning(
            "shadow 预检:交付模型显著劣于冠军 train-only 模型(%s)。"
            "evaluate-final 将要求 --ack-shadow-regression 才放行", _shadow_pre)
    elif _shadow_pre and _shadow_pre.get("available"):
        step(f"shadow precheck: delivery={_shadow_pre['delivery_model']:.6f} "
             f"search={_shadow_pre['search_model']:.6f} "
             f"ci={_shadow_pre['delta_ci']}")
    if _shadow_pre and _shadow_pre.get("level_stats"):
        _lv_d = _shadow_pre["level_stats"].get("delivery_model") or {}
        _lv_s = _shadow_pre["level_stats"].get("search_model") or {}
        step(f"shadow level: delivery pred_mean={_lv_d.get('pred_mean')} "
             f"y_mean={_lv_d.get('y_mean')} ratio={_lv_d.get('level_ratio')}"
             f";search ratio={_lv_s.get('level_ratio')}")
        if _shadow_pre.get("level_shift_warning"):
            logger.warning(
                "shadow 水位:交付模型 pred_mean/y_mean=%s 偏离超过阈值 —— "
                "排序可用、绝对金额需校准(只诊断,不阻塞 evaluate-final)",
                _lv_d.get("level_ratio"))
    _rel_recipe = _recipe_dir.relative_to(run_dir).as_posix()
    _gen_version = _skill_version()
    meta = {
        "best_trial": champ_id,
        "delivery_status": _delivery_status,
        "missing_derived_levels": missing_levels or None,
        "algorithm": (champ_dir.name.split("-")[3]
                      if len(champ_dir.name.split("-")) > 3 else ""),
        "data_view": cfg.get("data_view", "v1"),
        "seed": seed,
        "n_features": len(cols),
        "n_train_rows": _n_tr_rows,
        "n_validation_rows": _n_va_rows,
        "n_train_validation_rows": int(len(X_all)),
        "refit_mode": refit_mode,
        # v2.17.0(#5):refit 口径全链留痕。configured = CLI/config;
        # requested = 按冠军早停面解析后的口径(validation 早停 ⇒
        # frozen_rounds);effective = 实际执行的口径(= refit_mode)。
        "refit_mode_configured": _mode_before,
        "refit_mode_requested": _refit_requested,
        "refit_mode_effective": refit_mode,
        "refit_mode_fallback": bool(_refit_fallback_reason),
        "refit_mode_fallback_reason": _refit_fallback_reason,
        "clone_probe": _clone_probe_rep,
        "frozen_rounds_params": _frozen_sub or None,
        "es_source": _champ_es,
        "retrain_scope": _retrain_scope,
        "es_holdout": _es_window,
        "trial_model_sha256": _trial_model_sha,
        "final_model_sha256": _final_model_sha,
        # v2.15.9:轮数取不到(复合容器)时如实记 null,不再写 0
        "hyperparameters": {**(hp_inj or {}),
                            "n_estimators": (int(n_est) if n_est else None)},
        "early_stopping_best_iteration": (int(best_it) if best_it else None),
        "estimator_class": f"{type(model).__module__}.{type(model).__name__}",
        "retrain_builder": _retrain_builder,
        "reproduction": _repro,
        "reproduction_mode": _reproduction_mode,
        "face_check": _face,
        "rerun": _rerun_report,
        "shadow_precheck": _shadow_pre,
        "lineage": " > ".join([champ_id]),
        "code_source": [f"{_rel_recipe}/code/train.py"],
        "config_source": f"{_rel_recipe}/config/trial.yaml",
        "delivery_recipe_dir": _rel_recipe,
        "delivery_recipe_sha256": _recipe_shas,
        "champion_source": f"trials/{champ_id}",
        # v2.14.0:交付位与搜索冠军的关系必须写进交付物本身(run2 的
        # model_meta 只写 best_trial=trial-011,读者无从知道它落后冠军)
        "search_champion": _res.get("search_champion"),
        "recipe": recipe_root(_champ_trial,
                              {t.trial_id: t for t in trials}),
        "delivery_resolution": {k: _res.get(k) for k in (
            "status", "search_champion", "search_recipe", "delivery_trial",
            "gap", "tolerance")},
        # v2.20.0:先验豁免的证据(判据版本、z、σ_seed 来源)随交付物落盘;
        # 报告与审计据此区分"已确认交付"与"按先验豁免交付"
        "prior_waiver": (_res.get("prior_waiver")
                         if _prior_waived else None),
        "divergence_ack": (str(_ack) if _ack and _div_info
                           and _div_info.get("bound") is False else None),
        "generated_by": f"modelevo.py finalize-retrain (v{_gen_version})",
        "skill_version": _gen_version,
    }
    (fm / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (mdir / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    recipe = {
        "best_trial": champ_id,
        "algorithm": meta["algorithm"],
        "data_view": meta["data_view"],
        "features": {"base_columns": len(feats),
                     "widened_columns": len(cols),
                     "total_columns": len(cols)},
        "hyperparameters": meta["hyperparameters"],
        "seed": seed,
        "training_rows": int(len(X_all)),
        "n_train_rows": _n_tr_rows,
        "n_validation_rows": _n_va_rows,
        "refit_mode": refit_mode,
        "refit_mode_requested": _refit_requested,
        "refit_mode_effective": refit_mode,
        "refit_mode_fallback_reason": _refit_fallback_reason,
        "es_source": _champ_es,
        "retrain_scope": _retrain_scope,
        "es_holdout": _es_window,
        "delivery_recipe_dir": _rel_recipe,
        "delivery_status": _delivery_status,
        "lineage": meta["lineage"],
        "search_champion": meta["search_champion"],
        "delivery_resolution_status": (meta["delivery_resolution"]
                                       or {}).get("status"),
        # v2.16.0:spec.ensemble_enabled 已废弃;交付配方记录冠军的拓扑裁决
        "model_topology": {
            k: (_champ_trial.data.get("model_topology") or {}).get(k)
            for k in ("kind", "decision", "approvals", "needs")},
        "model_num": 1,
    }
    (fm / "recipe.yaml").write_text(
        yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    step("recipe/model_meta dumped")
    # v2.17.0(#5):单行 JSON 摘要(stdout),refit 口径与回落原因一眼可见 ——
    # run4 的回落只在 finalize-retrain.log 里有一行 INFO。
    _sp_lv = ((_shadow_pre or {}).get("level_stats") or {}).get(
        "delivery_model") or {}
    _summary = {
        "best_trial": champ_id,
        "delivery_status": _delivery_status,
        "model_meta": str((fm / "model_meta.json").relative_to(run_dir)),
        "refit_mode_requested": _refit_requested,
        "refit_mode_effective": refit_mode,
        "refit_mode_fallback": bool(_refit_fallback_reason),
        "refit_mode_fallback_reason": _refit_fallback_reason,
        "clone_probe": _clone_probe_rep.get("status"),
        "retrain_builder": _retrain_builder,
        "shadow_precheck_worse": (_shadow_pre or {}).get(
            "worse_than_search_model"),
        "shadow_level_ratio": _sp_lv.get("level_ratio"),
        "generated_by": meta["generated_by"],
    }
    print("FINALIZE_RETRAIN_SUMMARY\t"
          + json.dumps(_summary, ensure_ascii=False, default=str))
    if _refit_fallback_reason:
        print(f"[refit] 交付重训口径回落:{_refit_requested} → {refit_mode}"
              f"(原因:{_refit_fallback_reason})")
    if unconfirmed_mode:
        _write_unconfirmed_evidence(run_dir, spec, _champ_trial,
                                    retrained=True, step=step,
                                    resolution=_res)
        logger.error(
            "未确认重训完成(候选=%s,train %d + validation %d 行)→ "
            "final-model/unconfirmed/。确认门未通过,**未交付**:"
            "final-model/model/ 未写、final_test 未打分;说明见 "
            "final-model/unconfirmed/UNCONFIRMED.md",
            champ_id, _n_tr_rows, _n_va_rows)
        return 1
    # v2.11:冠军单源校验**前置到 retrain 完成时刻**。旧序只由
    # evaluate-final 在盲测 gate 前校验,而 recipe/model_meta 的
    # best_trial 就出自本命令(本命令自己写 champ_id,校验恒过)——
    # 实际的口径分裂来自"retrain 用了旧选型,之后 leaderboard 变了"
    # (reflection 回写新 verdict / backfill 调整 / 用户改 --champion)。
    # v2.10.1 run 实测:recipe 声明 trial-004,evaluate-final 选型
    # trial-002,首跑失败=一轮 Agent 修复往返 + 5~10min 空耗。此处
    # retrain 完成后立即对"当前 leaderboard 确定性选型"复核,不一致
    # 即 fail-fast(盲测机会不消耗,且错误在产生它的时刻就暴露)。
    if not getattr(args, "champion", None):
        # v2.11.7:同上——复核口径必须与初选、与 gate 三者同源,
        # 否则本复核会在自己的选型上误报「口径分裂」。
        _best_now = lb.best_trial(trials, delivery=True)
        _cur = _best_now.trial_id if _best_now is not None else None
        if _cur and _cur != champ_id:
            logger.error(
                "冠军单源校验失败(retrain 期末复核):本命令以 %s 重训,"
                "但当前 leaderboard 确定性选型已是 %s——选型在 retrain "
                "期间发生了变化(reflection 回写/backfill/数据变更)。"
                "请先确认选型口径,再重新 finalize-retrain;"
                "确要交付 %s 时用 --champion %s 显式指定(留痕)。",
                champ_id, _cur, champ_id, champ_id)
            return 1
    logger.info("RETRAIN_ALL_DONE(final-model/ 全套产物已落盘,冠军=%s)",
                champ_id)
    # v2.16.3:task-experience.final_recipe(此前全链路无人写,恒 null)
    _record_final_recipe(run_dir, meta, champ_dir)
    return 0


def cmd_finalize_baseline_score(args) -> int:
    """baseline 配方 train-only 重训 → 对 final_test 打分(不进反馈环)。

    产物 final-model/baseline_final_test_score.parquet:"提升了多少"的
    样本外口径分母。baseline 不参与冠军选择。
    幂等:重跑覆写。
    """
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    from .cmd_trial import _maybe_detach

    _rc = _maybe_detach(args, run_dir, "finalize-baseline-score")
    if _rc is not None:
        return _rc

    import time

    import pandas as pd
    import yaml

    t0 = time.time()

    def step(msg: str) -> None:
        logger.info("[baseline-final] %s (%.1fs)", msg, time.time() - t0)

    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    base_id = getattr(args, "baseline", None) or st.get("baseline_id")
    if not base_id:
        logger.error("无法确定 baseline trial(run-state 无 baseline_id;"
                     "用 --baseline <id> 显式指定)")
        return 1
    base_dir = run_dir / "trials" / base_id
    if not (base_dir / "code" / "train.py").exists():
        logger.error("baseline trial 缺 code/train.py: %s", base_dir)
        return 1
    try:
        from runtime.integrity import verify_trial_source

        _base_source_manifest = verify_trial_source(base_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("baseline 源码完整性校验失败:%s", exc)
        return 1

    os.environ["MODELEVO_GPU"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", _finalize_threads())
    base = _finalize_load_module(base_dir / "code" / "train.py",
                                 "base_train")
    cfg = yaml.safe_load(
        (base_dir / "config" / "trial.yaml").read_text()) or {}
    seed = int(cfg.get("seed", getattr(base, "SEED", 20260814)))
    splits = _resolve_splits_dir(run_dir, cfg)
    header = pd.read_parquet(splits / "train.parquet").columns
    feats = base.exclude_columns(pd.DataFrame(columns=header), cfg)
    _base_set = set(c for c in header if isinstance(c, str))
    feats_base = [c for c in feats if c in _base_set]
    step(f"baseline feats={len(feats)} (trial={base_id})")
    btr = pd.read_parquet(splits / "train.parquet", columns=feats_base)
    bft = pd.read_parquet(splits / "final_test.parquet", columns=feats_base)
    y_b = base.load_labels("train")[spec.target_name]
    # v2.16.0:baseline trial 在 outer validation 上早停时,对照模型同口径:
    # train-only 单段拟合、validation 早停(不再 train 尾部早停 + 重拟合)
    _base_es = _trial_es_source(base_dir)
    bva = y_bva = None
    if _base_es == "validation":
        bva = pd.read_parquet(splits / "validation.parquet", columns=feats_base)
        y_bva = base.load_labels("validation")[spec.target_name]

    # dtype 治理(模板 main() 同口径)
    cat_cols: list[str] = []
    dropped: list[str] = []
    for c in list(btr.columns):
        if pd.api.types.is_numeric_dtype(btr[c]):
            continue
        conv = pd.to_numeric(btr[c], errors="coerce")
        if conv.notna().sum() >= 0.95 * max(btr[c].notna().sum(), 1):
            btr[c] = conv.astype("float32")
            bft[c] = pd.to_numeric(bft[c], errors="coerce").astype("float32")
            if bva is not None:
                bva[c] = pd.to_numeric(bva[c], errors="coerce").astype(
                    "float32")
        elif btr[c].nunique() <= 100:
            cat_cols.append(c)
        else:
            dropped.append(c)
    if dropped:
        btr = btr.drop(columns=dropped)
        bft = bft.drop(columns=dropped)
        if bva is not None:
            bva = bva.drop(columns=dropped)
    if cat_cols:
        btr = pd.get_dummies(btr, columns=cat_cols)
        bft = pd.get_dummies(bft, columns=cat_cols)
        if bva is not None:
            bva = pd.get_dummies(bva, columns=cat_cols)
    bft = bft.reindex(columns=btr.columns, fill_value=0)
    btr = btr.fillna(0).astype("float32")
    bft = bft.fillna(0).astype("float32")
    if bva is not None:
        bva = bva.reindex(columns=btr.columns, fill_value=0).fillna(0).astype(
            "float32")
    step(f"face prepared {btr.shape}/{bft.shape} "
         f"onehot={len(cat_cols)} dropped={len(dropped)} es_source={_base_es}")

    bmodel = _finalize_build_estimator(base, 16, len(btr), None)
    if _base_es == "validation":
        _base_es_kw = _finalize_es_fit_kwargs(
            bmodel, bva, y_bva,
            int(getattr(base, "EARLY_STOP_ROUNDS", 50) or 50))
        bmodel.fit(btr, y_b, **_base_es_kw)
        _baseline_n_est = _validation_es_rounds(bmodel) or int(
            getattr(bmodel, "n_estimators", None)
            or getattr(base, "N_ESTIMATORS_CAP", 2000))
        step(f"baseline trained (train only, single fit, validation ES): "
             f"best_rounds={_baseline_n_est} seed={seed}")
    else:
        _baseline_n_est = _fit_baseline_train_tail(bmodel, base, btr, y_b)
        step(f"baseline retrained (train only): "
             f"n_estimators={_baseline_n_est} seed={seed}")
    bsc = _delivery_predict(bmodel, bft, spec)
    fm = run_dir / "final-model"
    fm.mkdir(parents=True, exist_ok=True)
    from runtime.data import SAMPLE_KEY_COLUMN
    from runtime.integrity import bind_prediction_artifact

    _identity_p = splits / "identity" / "final_test.parquet"
    if not _identity_p.exists():
        logger.error("baseline final_test identity 缺失:%s", _identity_p)
        return 1
    _identity = pd.read_parquet(_identity_p)
    if len(_identity) != len(bsc):
        logger.error("baseline final_test identity 行数 %d != score %d",
                     len(_identity), len(bsc))
        return 1
    _bl_score_path = fm / "baseline_final_test_score.parquet"
    pd.DataFrame({"score": bsc, SAMPLE_KEY_COLUMN:
                  _identity[SAMPLE_KEY_COLUMN].astype(str).values}).to_parquet(
                      _bl_score_path, index=False)
    bind_prediction_artifact(
        _bl_score_path, _identity,
        source_manifest=_base_source_manifest,
        split_manifest_path=splits / "manifest.json",
        data_view=str(cfg.get("data_view") or "v1"), model_paths=[])
    step(f"baseline final_test scored n={len(bsc)}")
    return 0


def _verify_retrain_provenance(run_dir, champion_id,
                               allow_waived: bool = False) -> str | None:
    """交付模型的重训证明(v2.13.4);通过返回 None,否则返回失败原因。

    交付位(final-model/)里的模型必须是 finalize-retrain 在 train+validation
    上的重训产物,而不是 trial 目录里的 train-only 搜索模型。旧版只核对
    recipe/model_meta 的 best_trial 字段,且文件缺失时 `continue` 跳过——
    交付目录里放一个手工拷来的 trial 模型也能过(实测 req-t7-drawdown)。
    """
    fm = run_dir / "final-model"
    meta_p = fm / "model_meta.json"
    if not meta_p.exists():
        return ("final-model/model_meta.json 缺失——交付模型未经 "
                "finalize-retrain 重训")
    try:
        meta = load_json(meta_p) or {}
    except Exception as exc:  # noqa: BLE001
        return f"final-model/model_meta.json 不可解析: {exc}"
    # v2.20.0:confirmation_waived_by_prior 与 confirmation_waived 同属"未经
    # confirm 的交付",校验侧一并放行(审计信息在 model_meta.prior_waiver)
    _ok_status = ({"confirmed", "confirmation_waived",
                   "confirmation_waived_by_prior"} if allow_waived
                  else {"confirmed"})
    if meta.get("delivery_status") not in _ok_status:
        return (f"delivery_status={meta.get('delivery_status')!r},须为 "
                f"{sorted(_ok_status)}(由 v2.13.4+ finalize-retrain 对交付"
                "冠军写入;旧版产物或切换 require_confirmation 后请重跑 "
                "finalize-retrain)")
    n_tr = meta.get("n_train_rows")
    n_va = meta.get("n_validation_rows")
    n_all = meta.get("n_train_validation_rows")
    if not all(isinstance(v, int) for v in (n_tr, n_va, n_all)):
        return ("model_meta 缺 n_train_rows/n_validation_rows/"
                "n_train_validation_rows")
    if n_tr <= 0 or n_va <= 0 or n_tr + n_va != n_all:
        return (f"训练行数不符合 train+validation 重训口径(train={n_tr}, "
                f"validation={n_va}, total={n_all})")
    actual = _file_sha256(fm / "model" / "model.joblib")
    if actual is None:
        return "final-model/model/model.joblib 缺失"
    if meta.get("final_model_sha256") != actual:
        return ("final-model/model/model.joblib 与 model_meta.final_model_sha256 "
                "不一致(重训后被替换)")
    _trial_sha_now = (_file_sha256(run_dir / "trials" / champion_id / "model"
                                   / "model.joblib")
                      if champion_id else None)
    if actual in {meta.get("trial_model_sha256"), _trial_sha_now} - {None}:
        return ("final-model/model/model.joblib 与冠军 trial 的 train-only "
                "搜索模型哈希相同——交付的是未重训模型")
    return None


def cmd_evaluate_final(args) -> int:
    import time as _ef_time

    _ef_t0 = _ef_time.time()  # v2.17.2(B5):收尾计时 → logs/evaluate-final.json
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    config = _load_config(run_dir)
    _require_confirm = _confirmation_required(config)

    import pandas as pd

    fs = run_dir / "final-model" / "final_test_score.parquet"
    if not fs.exists():
        logger.error("缺少 %s;请先完成最终模型打分"
                     "(用最佳 Trial 代码 train+validation 重训后对 final_test 打分)",
                     fs)
        return 1
    # ---- P0-8:前置校验全部通过后才 acquire gate(旧序先落盘计数、后检查
    # leaderboard 为空退出,一次失败就永久烧掉唯一盲测机会)----
    from runtime.data import (SAMPLE_KEY_COLUMN, load_identities,
                              load_split_frames)
    from runtime.integrity import verify_prediction_artifact

    frames = load_split_frames(run_dir)
    ft = frames.get("final_test")
    if ft is None:
        logger.error("缺少 final_test split")
        return 1
    lb = load_json(run_dir / "leaderboard" / "leaderboard.json") \
        if (run_dir / "leaderboard" / "leaderboard.json").exists() else {}
    rows = lb.get("rows") or []
    if not rows:
        logger.error("leaderboard 为空,无法确定最佳 Trial")
        return 1
    try:
        _score_meta = verify_prediction_artifact(fs)
        _score_df = pd.read_parquet(fs)
    except Exception as exc:  # noqa: BLE001
        logger.error("final_test 预测完整性校验失败:%s", exc)
        return 1
    score = _score_df["score"].values
    if len(score) != len(ft):
        logger.error("final_test_score 行数 %d != final_test split %d 行",
                     len(score), len(ft))
        return 1
    _identities = load_identities(run_dir, splits=("final_test",))
    _final_identity = _identities.get("final_test")
    if _final_identity is None or not _score_df[
            SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True).equals(
                _final_identity[SAMPLE_KEY_COLUMN].astype(str).reset_index(
                    drop=True)):
        logger.error("final_test 预测样本键与权威 identity 面不一致")
        return 1

    # ---- v19(P0 修复,req-t7-drawdown 实测事故回写):冠军选型/一致
    # 性校验/shadow 硬校验全部提到 gate.acquire **之前**。v18/P0-8 只把
    # 当时已知的校验前置,后加的 v16/P0-4 冠军一致性(v19 前位于
    # acquire 之后的 2745-2751)与 v18/P0-1 shadow 硬校验落在 acquire
    # 之后——实测一次"选型不一致"退出就永久烧掉唯一盲测机会(count=1
    # 而 evaluation.json 未落盘)。原则:gate.acquire 是读 final_test
    # 前的最后一步,它之前的一切校验失败都不消耗盲测次数。
    champion_id = getattr(args, "best_trial", None) or None
    # v2.20.0:_res_ef 在两条选型路径上都要可读(--best-trial 路径不解析
    # delivery_resolution,先验豁免按"未豁免"处理)
    _res_ef: dict = {}
    if champion_id:
        _trials_dir = run_dir / "trials" / champion_id
        if not _trials_dir.is_dir():
            logger.error("--best-trial %s 不存在(trials/ 下无该目录)",
                         champion_id)
            return 1
        _override_trial = next((t for t in TrialFactory(
            run_dir, spec=spec, config=config).all_trials()
            if t.trial_id == champion_id), None)
        _override_ev = (_override_trial.data.get("evaluation") or {}) \
            if _override_trial is not None else {}
        if _override_trial is None or _override_ev.get("verdict") != \
                "confirmed" or \
                (_override_ev.get("fresh_confirmation") or {}).get(
                    "passed") is not True or \
                (_override_ev.get("integrity") or {}).get(
                    "valid", True) is False or \
                trial_currently_failed(_override_trial):
            logger.error(
                "--best-trial %s 不满足交付门:必须 confirmed、"
                "通过独立 shadow fresh_confirmation、完整性有效且无失败；"
                "显式指定不能绕过客观裁决", champion_id)
            return 1
        _record_shadow_waiver(run_dir, champion_id,
                              "best_trial_override")
        logger.warning(
            "冠军由 --best-trial 显式指定为 %s(跳过确定性选型;"
            "已留痕 run-state.shadow_waivers)", champion_id)
    else:
        try:
            _lb = Leaderboard(run_dir, spec, config)
            _bt = _lb.best_trial(
                TrialFactory(run_dir, spec=spec, config=config).all_trials(),
                delivery=True)
            _res_ef = getattr(_lb, "last_delivery_resolution", None) or {}
            champion_id = _bt.trial_id if _bt else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("best_trial 选型失败: %s", exc)
            champion_id = None
        if not champion_id and not _require_confirm:
            # v2.15.3:与 finalize-retrain 同源取 accepted 搜索冠军
            try:
                _all_w = TrialFactory(run_dir, spec=spec,
                                      config=config).all_trials()
                _cw = _best_unconfirmed_candidate(
                    _all_w, spec, Leaderboard(run_dir, spec, config))
                champion_id = _cw.trial_id if _cw else None
            except Exception as exc:  # noqa: BLE001
                logger.warning("确认门关闭时的冠军选型失败: %s", exc)
            if champion_id:
                logger.warning(
                    "evaluator.require_confirmation=false:以搜索冠军 %s "
                    "盲测(无独立 shadow 确认)", champion_id)
        if not champion_id:
            if _res_ef.get("status") == "diverged":
                _write_divergence_note(run_dir, _res_ef)
            logger.error("无法确定交付冠军:%s。可用 --best-trial 显式指定"
                         "(与搜索冠军分叉时还需 --ack-divergence)",
                         divergence_summary(_res_ef)
                         if _res_ef else "leaderboard 选型为空")
            return 1
    # v2.14.0 交付位分叉守卫(前置于 acquire;两条选型路径都过):交付
    # 模型与搜索冠军配方不同且落后超过容差时,唯一一次盲测不得静默花在
    # 它身上(run2:gate 花在落后冠军 0.00157 的 trial-011 上,冠军配方
    # 从此无样本外估计)。用户显式裁决 --ack-divergence 才放行(留痕)。
    _all_trials_ef = TrialFactory(run_dir, spec=spec, config=config).all_trials()
    _div_ok, _div_ef = _delivery_divergence_check(
        Leaderboard(run_dir, spec, config), _all_trials_ef, champion_id)
    if not _div_ok:
        _ack_ef = getattr(args, "ack_divergence", None)
        if not _ack_ef:
            _write_divergence_note(run_dir, _div_ef.get("resolution") or {})
            logger.error(
                "交付位分叉:%s 落后搜索冠军 %s %+.6f > 容差 %.6f。盲测只有"
                "一次,不得静默花在非冠军配方上;确认冠军配方后重跑,或经用户"
                "裁决加 --ack-divergence \"<理由>\"(本失败发生在 gate 之前,"
                "final_test 未被消耗)", champion_id,
                (_div_ef.get("resolution") or {}).get("search_champion"),
                _div_ef.get("gap"), _div_ef.get("tolerance"))
            return 1
        _record_divergence_ack(run_dir, champion_id, _div_ef, str(_ack_ef),
                               "evaluate-final")
        logger.warning("交付位分叉已由 --ack-divergence 裁决放行(留痕):%s",
                       _ack_ef)
    _chosen_trial = next((t for t in _all_trials_ef
                          if t.trial_id == champion_id), None)
    _ce = ((_chosen_trial.data.get("evaluation") or {})
           if _chosen_trial is not None else {})
    # v2.15.3:确认门关闭且冠军未经 confirm ⇒ 不要求 confirm 血缘/新 seed
    # v2.20.0:先验豁免(_res_ef.status=waived_by_prior)同属"未经 confirm 的
    # 交付",血缘校验同样放行;审计信息随 model_meta.prior_waiver 落盘。
    _ef_prior_waived = (_res_ef.get("status") == "waived_by_prior")
    _ef_waived = (((not _require_confirm and _chosen_trial is not None
                   and (_ce.get("fresh_confirmation") or {}).get("passed")
                   is not True))
                  or _ef_prior_waived)
    if _chosen_trial is None or (
            not _ef_waived and len(_chosen_trial.parent_ids) != 1):
        logger.error("冠军 confirmation lineage 无效:必须且只能有一个父 Trial")
        return 1
    # v16/P0-4 冠军单源校验(前置版):retrain 产物的 recipe/model_meta
    # 声明 best_trial 与选型/显式指定不一致 = 交付链口径分裂,拒绝——
    # 在 acquire 之前失败,盲测机会不消耗。
    for _prod, _key in (("final-model/recipe.yaml", "best_trial"),
                        ("final-model/model_meta.json", "best_trial"),
                        ("final-model/model/model_meta.json",
                         "best_trial")):
        _pp = run_dir / _prod
        if not _pp.exists():
            continue
        try:
            if _pp.suffix == ".json":
                _declared = (load_json(_pp) or {}).get(_key)
            else:
                from runtime.logging_utils import load_yaml as _ly

                _declared = (_ly(_pp) or {}).get(_key)
        except Exception:  # noqa: BLE001
            continue
        if _declared and champion_id and _declared != champion_id:
            logger.error(
                "交付链冠军口径分裂: %s 声明 best_trial=%s,"
                "冠军选型=%s——重训产物与选型脱节,请用冠军代码重训、"
                "修正产物,或 --best-trial 显式对齐(留痕)后重试"
                "(本失败发生在盲测 gate 之前,final_test 未被消耗)",
                _prod, _declared, champion_id)
            return 1
    # v2.13.4 重训证明(前置于 acquire):交付模型必须是 train+validation
    # 重训产物,且不是 trial 的 train-only 搜索模型。
    _prov_err = _verify_retrain_provenance(run_dir, champion_id,
                                           allow_waived=_ef_waived)
    if _prov_err:
        logger.error("交付模型重训证明校验失败:%s(本失败发生在盲测 gate "
                     "之前,final_test 未被消耗)", _prov_err)
        return 1
    # v2.15.9 shadow 预检门(前置于 acquire):交付模型(train+validation
    # 重训)在 shadow 上显著劣于冠军 train-only 搜索模型 ⇒ 交付重训口径
    # 可疑(如早停定轮异常),须用户裁决才消耗唯一一次盲测。
    try:
        _meta_pre = load_json(run_dir / "final-model" / "model_meta.json") \
            or {}
    except Exception:  # noqa: BLE001
        _meta_pre = {}
    _sp = _meta_pre.get("shadow_precheck") or {}
    if _sp.get("worse_than_search_model"):
        _ack_sh = getattr(args, "ack_shadow_regression", None)
        if not _ack_sh:
            logger.error(
                "shadow 预检未通过:交付模型 %s=%s 显著劣于冠军 train-only "
                "模型 %s(Δ CI %s)。确认交付需用户裁决:加 "
                "--ack-shadow-regression \"<理由>\"(本失败发生在盲测 gate "
                "之前,final_test 未被消耗)", _sp.get("metric"),
                _sp.get("delivery_model"), _sp.get("search_model"),
                _sp.get("delta_ci"))
            return 1
        _record_shadow_waiver(run_dir, champion_id,
                              f"shadow_regression_ack: {_ack_sh}")
        logger.warning("shadow 预检回退已由 --ack-shadow-regression 放行"
                       "(留痕):%s", _ack_sh)
    _inv = str(getattr(args, "segment_inverse_transform", None) or "none")
    if _inv != "none" and spec.model_type != "regression":
        logger.warning("--segment-inverse-transform 仅适用于 regression,"
                       "当前 %s,忽略", spec.model_type)
        _inv = "none"
    # v18/P0-1 冠军 shadow 分数硬校验(前置版)
    if getattr(args, "allow_no_shadow", False):
        logger.warning(
            "--allow-no-shadow 已弃用且不会绕过交付门；必须完成独立 "
            "shadow fresh_confirmation")
        _record_shadow_waiver(run_dir, champion_id, "deprecated_flag_ignored")
    if champion_id:
        _champ_dir = run_dir / "trials" / champion_id
        _has_sh_score = (_champ_dir / "shadow_test_score.parquet").exists() \
            or (run_dir / "final-model" / "shadow_test_score.parquet").exists()
        if not _has_sh_score and _ef_waived:
            # v2.15.3:确认门关闭,shadow 只属于 confirm 证据,缺失不阻塞
            logger.warning(
                "require_confirmation=false:冠军 %s 无 shadow 分数,"
                "跳过 shadow 硬校验(留痕)", champion_id)
            _record_shadow_waiver(run_dir, champion_id, "confirmation_waived")
        elif not _has_sh_score:
            _sh_split = run_dir / "sample-profile" / "splits" \
                / "shadow_test.parquet"
            if _sh_split.exists():
                logger.error(
                    "冠军 shadow 分数缺失(v18 硬校验):trial %s 无 "
                    "shadow_test_score.parquet——交付置信的第二把尺子空转。"
                    "处置:创建 confirm trial，在独立 shadow 上完成确认后重试；"
                    "该门不接受人工豁免",
                    champion_id)
                _record_shadow_waiver(run_dir, champion_id, "blocked")
                return 1
            else:
                logger.warning(
                    "shadow split 不存在(sample 视图未落盘 shadow),"
                    "冠军 shadow 校验跳过(留痕)")
                _record_shadow_waiver(run_dir, champion_id, "no_split")

    try:
        from runtime.integrity import (verify_confirmation_seed,
                                       verify_trial_source)

        _champ_manifest = verify_trial_source(
            run_dir / "trials" / champion_id)
        if not _ef_waived:
            verify_confirmation_seed(
                run_dir / "trials" / champion_id,
                run_dir / "trials" / _chosen_trial.parent_ids[0])
        if _score_meta.get("source_manifest_sha256") != \
                _champ_manifest.get("combined_sha256"):
            raise PermissionError("final_test 预测并非由当前冠军冻结源码生成")
    except Exception as exc:  # noqa: BLE001
        logger.error("冠军源码完整性校验失败:%s", exc)
        return 1

    gate = FinalTestGate(run_dir)

    def _fail_after_acquire(msg: str, *a) -> int:
        # acquire 后标签可能已进入内存，任何失败都只能标记终测失效，
        # 不能恢复额度。要重评必须生成新的独立 holdout / 新 run。
        gate.release_last(f"evaluate_final_aborted: {msg % a if a else msg}")
        logger.error(msg, *a)
        return 1

    gate.acquire(purpose="final_model_evaluation")
    print("final_test 盲测开始(唯一一次)")

    # final_test 标签从不物化到搜索工作区。只有持有一次性 gate 的本
    # finalizer 进程临时从原始数据重建标签，使用后随进程内存释放。
    try:
        from runtime.data import load_final_test_labels

        _lab = load_final_test_labels(spec, run_dir)
    except Exception as exc:  # noqa: BLE001
        return _fail_after_acquire("final_test 临时标签加载失败:%s", exc)
    if len(_lab) != len(ft):
        return _fail_after_acquire(
            "final_test 标签行数 %d != 特征行数 %d", len(_lab), len(ft))
    if not _lab[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True).equals(
            _score_df[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True)):
        return _fail_after_acquire(
            "final_test 标签样本键与预测样本键不一致，拒绝位置配对")
    y = _lab[spec.target_name].values
    # v2.17.0(#20):回归交付诊断的模型分数(只在内存里,与标签同生命周期)
    _bl_final_sc = None
    _sh_scores: dict = {}
    ft_treatment = None
    tcol = spec.treatment_column
    if tcol is not None:
        # v13:treatment 与目标同在标签面(旧实现读特征面,uplift 任务
        # 必走 return 1 且发生在 gate.acquire 之后——盲测机会被烧掉)
        if tcol not in _lab.columns:
            return _fail_after_acquire(
                "final_test 标签面缺少 treatment 列 %s", tcol)
        ft_treatment = _lab[tcol].values
    evaluation = {"final_test": _metric_panel(
        spec, y, score, treatment=ft_treatment)}
    # final_test 分段排序性(与 Trial 评测同口径,默认 10 段)
    from runtime.metrics import segment_ranking

    seg = segment_ranking(
        y, score, n_bins=int(config.get("evaluator.segment_bins", 10)),
        treatment=ft_treatment)
    if seg.get("bin_means") is not None:
        # v2.15.9:段均值应随 score 递增(与指标方向无关),见
        # runtime.metrics.segment_passed
        from runtime.metrics import segment_passed

        seg["passed"] = segment_passed(
            seg, config.get("evaluator.segment_monotonic_threshold", 0.8))
    evaluation["segment_evaluation"] = seg
    # ---- final_test 分桶报表(xlsx + md,与 Trial 评测同口径)----
    from runtime.segment_report import build_segment_report

    try:
        seg_rep = build_segment_report(
            y, score, spec,
            treatment=ft_treatment,
            n_bins=int(config.get("evaluator.segment_bins", 10)),
            y_pred=score if spec.model_type == "regression" else None,
            out_dir=run_dir / "final-model",
            extra_summary={"split": "final_test",
                           "best_trial": evaluation.get("best_trial")},
            label="final_test")
        if seg_rep is not None:
            evaluation["segment_report"] = {
                "xlsx": "segment-report.xlsx",
                "md": "segment-report.md",
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("final_test 分桶报表产出失败: %s", exc)
    # validation 指标:从 leaderboard 取(P0-3:冠军行来自收缩分选型,
    # 不再是裸 argmax 的第一行;evaluate-final 语义 = 报告选中模型
    # 的 validation 口径分数。v14.2:冠军走 Leaderboard.best_trial
    # ——verdict∈{accepted,confirmed} 过滤 + 收缩分 + 最新 epoch 池,
    # 与 finalize/停止判定同一口径;旧实现裸取 rows[0](表内 argmax),
    # 可能选中 rejected 行,盲测报告与裁决面脱节)
    # v19:冠军选型已前置到 gate.acquire 之前(见上),champion_id
    # 与 _shadow_note 在那里产生;此处只消费。
    _shadow_note = locals().get("_shadow_note", "")
    # v19:冠军选型与一致性/shadow 校验已全部前置到 gate.acquire 之前
    # (见上方);此处只做行选取,不再重复校验。
    best_row = None
    for row in rows:
        if row.get("trial_id") == (champion_id or rows[0].get("trial_id")):
            best_row = row
            break
    best = best_row or rows[0]
    metric_col = spec.primary_metric  # 行内指标列名即实际指标名

    def _metric_of(row):
        return row.get(metric_col, row.get("primary_metric"))  # 兼容旧行

    evaluation["best_trial"] = best.get("trial_id")
    # v2.14.0:交付模型与搜索冠军的关系随盲测结果落盘(report 消费)
    _res_out = _div_ef.get("resolution") or {}
    evaluation["search_champion"] = _res_out.get("search_champion")
    evaluation["delivery_resolution"] = {
        "status": _res_out.get("status"),
        "search_champion": _res_out.get("search_champion"),
        "search_recipe": _res_out.get("search_recipe"),
        "delivery_recipe": _div_ef.get("champion_recipe"),
        "gap_vs_search_champion": _div_ef.get("gap", _res_out.get("gap")),
        "tolerance": _res_out.get("tolerance"),
        "divergence_acknowledged": bool(
            not _div_ok and getattr(args, "ack_divergence", None)),
    }
    if _shadow_note:
        evaluation["best_trial_note"] = _shadow_note
    elif getattr(args, "best_trial", None):
        evaluation["best_trial_note"] = (
            "冠军由 --best-trial 显式指定(业务拍板口径,留痕 "
            "run-state.shadow_waivers;确定性选型被跳过)")
    evaluation["validation"] = {spec.primary_metric: _metric_of(best)}
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    bid = st.get("baseline_id")
    base_val = None
    for row in rows:
        if row.get("trial_id") == bid:
            base_val = _metric_of(row)
    if base_val is not None:
        from runtime.metrics import metric_direction

        delta = (_metric_of(best) or 0) - base_val
        evaluation["improvement_vs_baseline"] = delta if metric_direction(
            spec.primary_metric) else -delta
        # P0-8:baseline 也上 final_test——"提升了多少"必须是样本外口径;
        # baseline 的 final_test 分数存在时给出 final_test 口径的改善
        bl_fs = run_dir / "final-model" / "baseline_final_test_score.parquet"
        if bl_fs.exists() and ft is not None:
            try:
                from runtime.integrity import verify_prediction_artifact
                from runtime.metrics import align_score_by_keys

                verify_prediction_artifact(bl_fs)
                _bl_df = pd.read_parquet(bl_fs)
                bl_score = align_score_by_keys(
                    _bl_df["score"].values,
                    _score_df[SAMPLE_KEY_COLUMN].values,
                    _bl_df[SAMPLE_KEY_COLUMN].values)
                if len(bl_score) == len(y):
                    _bl_final_sc = bl_score
                    _bl_panel = _metric_panel(spec, y, bl_score,
                                              treatment=ft_treatment)
                    bl_final = _bl_panel[spec.primary_metric]
                    evaluation["baseline_final_test"] = _bl_panel
                    d_final = (evaluation["final_test"][spec.primary_metric]
                               - bl_final)
                    evaluation["improvement_vs_baseline_final_test"] = \
                        d_final if metric_direction(spec.primary_metric) \
                        else -d_final
            except Exception as exc:  # noqa: BLE001
                logger.warning("baseline final_test 分数读取失败: %s", exc)
        else:
            # v16/P0-4:baseline 未在 final_test 打分 → improvement 只有
            # validation 口径,"提升了多少"无样本外证据(实测 120w run)
            logger.warning(
                "baseline_final_test_score.parquet 缺失:交付报告的 "
                "improvement_vs_baseline 只有 validation 口径——"
                "收尾流程要求先用 baseline 代码对 final_test 打分"
                "(baseline 不参与选择,打分不构成反馈环)")
            evaluation["baseline_final_test_missing"] = True
    # v18/P0-1 → v2.15.9:shadow 指标进 evaluation。shadow 不参与选择,
    # 只补齐交付报表的第二把尺子。v2.15.9 起区分三个模型:
    #   shadow_test               = 交付模型(final-model,train+validation 重训);
    #   shadow_test_search_model  = 冠军 trial 的 train-only 搜索模型(旧口径);
    #   baseline_shadow_test      = 基线 trial 的 train-only 分数(与基线
    #                               final_test 同口径)。
    # 旧实现优先读 trial 分数,交付模型的 shadow 表现从未被报告。
    _sh_lab = run_dir / "sample-profile" / "splits" / "labels" \
        / "shadow_test.parquet"
    _sh_split = run_dir / "sample-profile" / "splits" / "shadow_test.parquet"
    _sh_y = _sh_sc = _sh_treat = _sh_keys = None
    if _sh_lab.exists() and _sh_split.exists() and champion_id:
        try:
            _sh_lab_df = pd.read_parquet(_sh_lab)
            _sh_y = _sh_lab_df[spec.target_name].values
            if spec.treatment_column and \
                    spec.treatment_column in _sh_lab_df.columns:
                _sh_treat = _sh_lab_df[spec.treatment_column].values
            _sh_keys = _sh_lab_df[SAMPLE_KEY_COLUMN].values
            _base_id = (load_json(run_dir / "run-state.json") or {}).get(
                "baseline_id") if (run_dir / "run-state.json").exists() \
                else None
            for _key, _path in (
                    ("shadow_test",
                     run_dir / "final-model" / "shadow_test_score.parquet"),
                    ("shadow_test_search_model",
                     run_dir / "trials" / champion_id
                     / "shadow_test_score.parquet"),
                    ("baseline_shadow_test",
                     run_dir / "trials" / str(_base_id or "")
                     / "shadow_test_score.parquet")):
                if not _path.is_file():
                    continue
                try:
                    _sc = _load_aligned_score(_path, _sh_keys)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s 分数读取失败(不阻塞): %s", _key, exc)
                    continue
                if _sc is None or len(_sc) != len(_sh_y):
                    continue
                evaluation[_key] = _metric_panel(spec, _sh_y, _sc,
                                                 treatment=_sh_treat)
                _sh_scores[_key] = _sc
                if _key == "shadow_test":
                    _sh_sc = _sc
            if "shadow_test" not in evaluation and \
                    "shadow_test_search_model" in evaluation:
                # 交付模型未在 shadow 打分(旧产物):退回旧口径并标注
                evaluation["shadow_test"] = evaluation[
                    "shadow_test_search_model"]
                evaluation["shadow_test_source"] = "search_model_fallback"
            elif "shadow_test" in evaluation:
                evaluation["shadow_test_source"] = "delivery_model"
            if "shadow_test" in evaluation and \
                    "baseline_shadow_test" in evaluation:
                from runtime.metrics import metric_direction

                _d_sh = (evaluation["shadow_test"][spec.primary_metric]
                         - evaluation["baseline_shadow_test"][
                             spec.primary_metric])
                evaluation["improvement_vs_baseline_shadow_test"] = \
                    _d_sh if metric_direction(spec.primary_metric) \
                    else -_d_sh
        except Exception as exc:  # noqa: BLE001
            logger.warning("shadow 指标计算失败(不阻塞): %s", exc)
    # v2.17.0(#20):回归交付诊断 —— final_test / shadow 上交付模型与基线
    # (shadow 另含冠军 train-only 搜索模型)的整体水位与按期水位,交付模型
    # 附 vs 基线的 rmse/mae 背离。时间期取 identity 面的时间坐标(无标签面);
    # final_test 标签只在本次 gate 内的内存里使用,evaluation.json 只落统计量。
    if spec.model_type == "regression":
        try:
            _tc = spec.time_column
            _ft_per = None
            if _tc and _final_identity is not None \
                    and _tc in _final_identity.columns:
                _ft_per = _period_labels(_final_identity[_tc].values)
            _faces = {"final_test": {
                "y": y, "periods": _ft_per,
                "models": {"delivery": score, "baseline": _bl_final_sc}}}
            if _sh_y is not None and _sh_scores:
                _sh_per = None
                if _tc and _sh_keys is not None:
                    _sh_id = load_identities(
                        run_dir, splits=("shadow_test",)).get("shadow_test")
                    if _sh_id is not None and _tc in _sh_id.columns:
                        _sh_map = dict(zip(
                            _sh_id[SAMPLE_KEY_COLUMN].astype(str),
                            _sh_id[_tc]))
                        _sh_per = _period_labels(
                            pd.Series(_sh_keys).astype(str).map(_sh_map)
                            .values)
                _faces["shadow_test"] = {
                    "y": _sh_y, "periods": _sh_per,
                    "models": {
                        "delivery": _sh_scores.get("shadow_test"),
                        "search_model": _sh_scores.get(
                            "shadow_test_search_model"),
                        "baseline": _sh_scores.get("baseline_shadow_test")}}
            evaluation["regression_diagnostics"] = \
                _delivery_regression_diagnostics(_faces, period_column=_tc)
        except Exception as exc:  # noqa: BLE001  诊断不阻塞盲测落盘
            logger.warning("回归交付诊断失败(不阻塞): %s", exc)
            evaluation["regression_diagnostics"] = {
                "error": f"{type(exc).__name__}: {exc}"}
    # v2.15.9:原值尺度分桶报表(仅 regression,--segment-inverse-transform)。
    # final_test 标签只在本进程内存里存在一次,事后无法补算 —— 故在此同批产出。
    if _inv == "expm1":
        _raw = {}
        for _split, _yy, _ss in (("final_test", y, score),
                                 ("shadow_test", _sh_y, _sh_sc)):
            if _yy is None or _ss is None:
                continue
            try:
                _raw[_split] = _raw_scale_report(
                    spec, _yy, _ss,
                    out_dir=run_dir / "final-model" / "segment-raw" / _split,
                    n_bins=int(config.get("evaluator.segment_bins", 10)),
                    split=_split, best_trial=evaluation.get("best_trial"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s 原值分桶报表失败(不阻塞): %s", _split, exc)
        if _raw:
            evaluation["raw_scale"] = {
                "inverse_transform": "expm1",
                "note": ("原值 = expm1(log1p 预测),负预测截断为 0;log1p 尺度"
                         "的条件均值经 expm1 回算接近几何口径(Jensen),原值"
                         "总额比显著 < 1 属口径而非模型偏置;仅供业务解读,"
                         "不参与裁决"),
                **_raw}
    try:
        dump_json(run_dir / "final-model" / "evaluation.json", evaluation)
        _update_run_state(run_dir, status="FINAL_EVALUATED")
    except Exception as exc:  # noqa: BLE001
        # v19:指标算完但落盘失败 → 事务补偿(计数回滚+留痕),不让
        # 唯一盲测被"算了但没存下来"白白消耗
        return _fail_after_acquire("evaluation.json 落盘失败: %s", exc)
    # v2.17.2(B5):evaluate-final 自身用时落 logs/evaluate-final.json
    # (compute_hours_summary 读取,计入 finalize_hours;失败不阻塞)
    try:
        from runtime.trial import record_timing_log

        _ef_sec = _ef_time.time() - _ef_t0
        record_timing_log(run_dir / "logs" / "evaluate-final.json", {
            "at": now_iso(), "wall_sec": round(_ef_sec, 1),
            "wall_min": round(_ef_sec / 60.0, 2),
            "best_trial": evaluation.get("best_trial")})
    except Exception as exc:  # noqa: BLE001
        logger.warning("evaluate-final 计时写入失败(不阻塞): %s", exc)
    # v2.16.3:evaluate-final 再写一次 final_recipe(幂等;覆盖 finalize-retrain
    # 由旧版本执行、从未写过 final_recipe 的 run)。单源 = model_meta.json
    try:
        _mm_p = run_dir / "final-model" / "model_meta.json"
        if _mm_p.exists():
            _mm = load_json(_mm_p) or {}
            _cid = _mm.get("best_trial")
            _record_final_recipe(
                run_dir, _mm,
                (run_dir / "trials" / str(_cid)) if _cid else None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("final_recipe 回填失败(不阻塞): %s", exc)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    # v2.17.0(#20):水位偏离提示(排序可用、绝对金额需校准)
    for _hint in _level_hint_lines(
            evaluation.get("regression_diagnostics") or {}):
        logger.warning("%s", _hint)
        print(_hint)
    return 0


def cmd_report(args) -> int:
    """生成交付报告:report/report.md(+ v2.16.1 默认 report/report.html)。

    HTML 为训练历程可视化(冠军路径/进化地图/全部尝试/最终效果/过程/裁决,
    左侧目录),由 `_mvo.report_html` 从落盘产物确定性生成,可选读取
    report/journey-narrative.yaml 的说明文字。HTML 任一环节失败只告警,
    不影响 report.md 与退出码;`--no-html` 关闭。
    """
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    config = _load_config(run_dir)
    path = ReportBuilder(run_dir, spec, config).build()
    print(f"报告: {path}")
    if getattr(args, "no_html", False):
        return 0
    try:
        from .report_html import build_journey_html

        html = build_journey_html(run_dir, spec, config)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n\n---\n\n可视化训练历程报告(冠军路径、进化地图、"
                         "全部尝试与最终效果):[report.html](report.html)\n")
        except OSError:
            pass
        print(f"可视化报告: {html}")
    except Exception as exc:  # noqa: BLE001  HTML 是附加产物,不阻塞交付报告
        logger.warning("report.html 生成失败(report.md 不受影响): %s", exc)
    return 0


def cmd_leaderboard(args) -> int:
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    factory = TrialFactory(run_dir)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    df = Leaderboard(run_dir, spec).build(factory.all_trials(),
                                          baseline_id=st.get("baseline_id"))
    print(df.to_string(index=False))
    return 0


def cmd_release_final_test(args) -> int:
    """关闭旧版“回滚终测”旁路；只留痕并将该 holdout 标为失效。"""
    run_dir = _run_dir_from(args)
    gate = FinalTestGate(run_dir)
    if not gate.state.get("touches"):
        logger.error("final-test-gate.json 无 touch 记录,无可标记的终测")
        return 1
    _count_before = gate.state.get("count", 0)
    gate.release_last(f"cli_release: {args.reason}")
    _rec = {"kind": "final_test_invalidation", "at": now_iso(),
            "reason": args.reason, "count_before": _count_before,
            "count_after": gate.state.get("count", 0)}
    try:
        _st = load_json(run_dir / "run-state.json") or {}
        _st.setdefault("final_test_invalidations", []).append(_rec)
        dump_json(run_dir / "run-state.json", _st)
    except Exception as exc:  # noqa: BLE001  gate 本身已经完成留痕
        logger.warning("run-state 失效留痕失败: %s", exc)
    logger.error(
        "final_test 已消费且不能回滚:count 仍为 %s。已记录失效理由:%s。"
        "请创建新的独立 holdout/新 run 后再评测。",
        gate.state.get("count", 0), args.reason)
    return 1
