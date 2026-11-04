"""v2.17.2 回归测试:HPO / 重训 / confirm 批次(run4 req-t7-400w 复盘)。

  A1. lr-shrink 重训不再被树数上限静默截断:由搜索期所选 config 的最优轮数推
      need = ceil(best_iter / factor × 1.2);超 MAX_TREES 时放宽收缩因子
      (trial-017:3019 轮、factor 0.25、MAX 8000 → factor_eff≈0.4529、
      重训 lr≈0.02264、cap 8000);``es_info.cap_bound_unresolved``(撞 cap 且无
      外扩余量)语义;旧副本升级幂等。
  A3. HPO deadline 外推:中位数拟合秒数 ÷ 并行数、完成数 ≥ 并行数才截断、
      重训预留;剩余预算整体装不下时不再"一判即收敛"(40→2)。
  A4. 单保真抽样改分层**随机**行(非前缀、保分层、确定性);opt-in 保真复评。
  A5. 旧 main 的 metrics.n_features 以拟合期模型列数为准,基础列数另存
      n_features_base。
  D2. confirm seed 变体并行(主进程 + worker 子进程)与串行逐位一致;无额外
      GPU / 内存不足 / 关闭时回退串行。
全部离线(CPU、小数据)。
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

for _p in (str(SKILL), str(SKILL / "scripts"), str(SKILL / "assets")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime import hpo as H  # noqa: E402

xgb = pytest.importorskip("xgboost")
optuna = pytest.importorskip("optuna")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("MODELEVO_SMOKE_RUN", "MODELEVO_SEED_VARIANT",
              "MODELEVO_DEADLINE_TS", "MODELEVO_TRIAL_EXEC",
              "MODELEVO_ES_SOURCE", "MODELEVO_CONFIRM_PARALLEL_SEEDS",
              "MODELEVO_SEED_WORKER_K", "MODELEVO_SEED_WORKER_CFG",
              "MODELEVO_N_ESTIMATORS_CAP", "MODELEVO_EARLY_STOP_ROUNDS",
              "MODELEVO_QUICK_RATIO", "CUDA_VISIBLE_DEVICES",
              "MODELEVO_MEMORY_ADMISSION"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODELEVO_GPU", "0")
    monkeypatch.setenv("MODELEVO_SMOKE", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")


# ---------------------------------------------------------------------------
# 夹具:极简二分类 run + 模板渲染 trial(_HPO_ALGO=xgboost 的算法化分支)
# ---------------------------------------------------------------------------

def _mk_run(tmp: Path, n_tr=3000, n_va=1000, seed=0) -> Path:
    run = tmp / "run"
    sp = run / "sample-profile" / "splits"
    (sp / "labels").mkdir(parents=True, exist_ok=True)
    (run / "trials").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_tr), ("validation", n_va)):
        X = rng.normal(size=(n, 6)).astype("float32")
        logit = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] * X[:, 3]
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]).to_parquet(
            sp / f"{split}.parquet")
        pd.DataFrame({"y": y}).to_parquet(sp / "labels" / f"{split}.parquet")
    (run / "task-spec.yaml").write_text(
        "model_type: binary_classification\nprimary_metric: auc\n",
        encoding="utf-8")
    return run


def _render_trial(run: Path, tid: str, cfg: dict) -> Path:
    from _mvo.render import _inject_hpo_algo_at
    from _mvo.templates import TRIAL_BODY
    from runtime.templates import TRIAL_TEMPLATE

    code = TRIAL_TEMPLATE.format(
        trial_id=tid, hypothesis="h", actions="a", run_dir=str(run),
        target="y", model_type="binary_classification", treatment="",
        seed=20260814, skill_root=str(SKILL), body=TRIAL_BODY)
    code = _inject_hpo_algo_at(code, "xgboost")
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "code" / "train.py").write_text(code, encoding="utf-8")
    (d / "config" / "trial.yaml").write_text(yaml.safe_dump(cfg),
                                             encoding="utf-8")
    return d


_LOADED = [0]


def _load(path: Path):
    _LOADED[0] += 1
    nm = f"mvo_v2172_mod_{_LOADED[0]}"
    spec = importlib.util.spec_from_file_location(nm, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nm] = mod
    spec.loader.exec_module(mod)
    return mod


def _xy(run: Path):
    sp = run / "sample-profile" / "splits"
    X_tr = pd.read_parquet(sp / "train.parquet")
    X_va = pd.read_parquet(sp / "validation.parquet")
    y_tr = pd.read_parquet(sp / "labels" / "train.parquet")["y"]
    y_va = pd.read_parquet(sp / "labels" / "validation.parquet")["y"]
    return X_tr, y_tr, X_va, y_va


# ===========================================================================
# A1. lr-shrink 重训自适应
# ===========================================================================

def test_plan_refit_shrink_trial017_shape():
    """trial-017:搜索期最优 3019 轮、factor 0.25、MAX 8000 → 放宽收缩。"""
    p = H.plan_refit_shrink(3019, search_lr=0.05, factor=0.25, cap_now=8000,
                            cap_scale=4.0, max_trees=8000)
    assert p["need_trees"] == 14492                     # ceil(3019/0.25×1.2)
    assert p["factor_eff"] == pytest.approx(0.4529, abs=1e-4)
    assert p["refit_lr"] == pytest.approx(0.02264, abs=1e-5)
    assert p["refit_cap"] == 8000
    assert p["lr_adjusted"] is True and p["cap_clamped"] is True
    assert p["basis"] == "search_best_iteration"
    # 需求在上限内:维持收缩,cap = max(旧缩放 cap, need) ≤ MAX
    q = H.plan_refit_shrink(1000, search_lr=0.05, factor=0.25, cap_now=1000,
                            max_trees=8000)
    assert q["refit_lr"] == pytest.approx(0.0125) and not q["lr_adjusted"]
    assert q["refit_cap"] == 4800 and q["cap_clamped"] is False
    # 搜索期最优轮数不可得:旧口径 + 钳制留痕
    r = H.plan_refit_shrink(None, search_lr=0.05, factor=0.25, cap_now=8000)
    assert r["basis"] == "unavailable" and r["refit_cap"] == 8000
    assert r["refit_lr"] == pytest.approx(0.0125)
    assert r["cap_clamped"] is True and r["lr_adjusted"] is False
    # 极端:搜索本身已贴近上限 → factor_eff 封顶 1(重训 lr = search_lr)
    s = H.plan_refit_shrink(7900, search_lr=0.05, factor=0.25, cap_now=8000,
                            max_trees=8000)
    assert s["factor_eff"] == 1.0 and s["refit_lr"] == pytest.approx(0.05)


def test_search_best_iteration_sources(tmp_path):
    space = {"max_depth": {"type": "int", "low": 2, "high": 6}}

    def f(p):
        return {"value": -abs(p["max_depth"] - 4), "best_iteration":
                100 * p["max_depth"], "rounds": 100 * p["max_depth"] + 50}

    r = H.run_hpo_search(space, 6, f, metric_direction_up=True, seed=3,
                         enqueue=[{"max_depth": 4}])
    assert r["best_params"] == {"max_depth": 4}
    assert r["best_search_iteration"] == 400
    assert H.search_best_iteration(r) == 400
    # 缺 best_search_iteration 时按参数在轨迹里匹配
    r2 = {k: v for k, v in r.items() if k != "best_search_iteration"}
    r2["search_best_params"] = {"max_depth": 4}
    r2["best_params"] = {"max_depth": 4, "learning_rate": 0.01}
    assert H.search_best_iteration(r2) == 400
    assert H.search_best_iteration({"best_params": {"max_depth": 9},
                                    "trials": r["trials"]}) is None
    # 热启动观测带 best_iteration:热启动点胜出时仍可得
    st = tmp_path / "run" / "trials" / "trial-001" / "hpo-study.json"
    st.parent.mkdir(parents=True)
    st.write_text(json.dumps({"inner_signature": "S", "algorithm": "xgboost",
                              "trials": [{"params": {"max_depth": 5},
                                          "inner_metric": 9.0,
                                          "best_iteration": 777}]}))
    srcs = H.collect_warm_start(tmp_path / "run", algorithm="xgboost")
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"sources": srcs}))
    warm = H.load_warm_start(ws, "S")
    assert warm[0]["best_iteration"] == 777
    r3 = H.run_hpo_search(space, 2, f, metric_direction_up=True, seed=3,
                          warm_start=warm)
    assert r3["best_origin"] == "warm" and r3["best_search_iteration"] == 777


def test_template_hpo_refit_relaxes_shrink_and_records(tmp_path):
    """模板 hpo 分支:重训 lr / cap 由搜索期最优轮数推出并留痕(小 MAX_TREES)。"""
    run = _mk_run(tmp_path)
    d = _render_trial(run, "trial-005-g02-hpo-xgboost-1", {"hyperparameters": {}})
    m = _load(d / "code" / "train.py")
    m._MAX_TREES = 60
    m.N_ESTIMATORS_CAP = 60
    m.EARLY_STOP_ROUNDS = 10
    X_tr, y_tr, X_va, y_va = _xy(run)
    cfg = {"hyperparameters": {}, "seed": 7, "search_action": "hpo",
           "hpo": {"search_space": {"max_depth": {"type": "int", "low": 2,
                                                  "high": 4}},
                   "budget_configs": 3, "inner_fraction": 0.2,
                   "lr_mode": "shrink", "search_lr": 0.05,
                   "lr_shrink_factor": 0.25, "parallel_configs": 1,
                   "auto_expand": False, "enqueue": []}}
    _, mx = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, cfg)
    st = json.loads((d / "hpo-study.json").read_text(encoding="utf-8"))
    bi = H.search_best_iteration(st)
    assert bi and bi == st["best_search_iteration"]
    plan = H.plan_refit_shrink(bi, search_lr=0.05, factor=0.25, cap_now=60,
                               cap_scale=4.0, max_trees=60)
    assert plan["lr_adjusted"], plan          # 0.05 搜出的轮数 ×4.8 > 60
    assert mx["hpo_refit_lr_adjusted"] is True
    assert mx["hpo_refit_factor_eff"] == pytest.approx(plan["factor_eff"])
    assert mx["hpo_refit_cap_clamped"] is True
    assert mx["best_params"]["learning_rate"] == pytest.approx(plan["refit_lr"])
    assert mx["best_params"]["learning_rate"] > 0.0125      # 不再死板 ×0.25
    assert mx["best_params"]["n_estimators"] <= 60
    assert st["refit"]["lr"] == pytest.approx(plan["refit_lr"])
    assert st["refit"]["plan"]["factor_eff"] == pytest.approx(plan["factor_eff"])
    assert st["lr_mode_refit_lr"] == pytest.approx(plan["refit_lr"])
    es = mx["es_info"]
    assert isinstance(es["cap_bound_unresolved"], bool)
    # cap 已达(补丁后的)上限 60:撞 cap 即未解除
    assert es["cap_bound_unresolved"] == bool(es["cap_bound"])


def _es_mod(tmp_path, max_trees):
    run = _mk_run(tmp_path / f"r{max_trees}", n_tr=1500, n_va=500)
    d = _render_trial(run, "trial-001-g01-explore-xgboost-1",
                      {"hyperparameters": {}})
    m = _load(d / "code" / "train.py")
    m._MAX_TREES = max_trees
    return m, _xy(run)


def test_fit_es_on_validation_headroom_and_unresolved(tmp_path):
    m, (X, y, Xv, yv) = _es_mod(tmp_path, 40)

    def slow(cap):
        return xgb.XGBClassifier(n_estimators=cap, learning_rate=0.005,
                                 max_depth=3, random_state=0, n_jobs=2)

    # 余量内外扩:cap 30 → min(40, 60) = 40;仍撞 cap 且已达上限 → 未解除
    a = m._fit_es_on_validation(slow(30), X, y, Xv, yv, rounds=10)
    ia = a.mvo_es_info_
    assert ia["cap_initial"] == 30 and ia["cap_extended_to"] == 40
    assert ia["cap"] == 40 and ia["cap_bound"] is True
    assert ia["cap_bound_unresolved"] is True
    # 已在上限:不外扩(cand 不大于当前 cap),撞 cap → 未解除
    b = m._fit_es_on_validation(slow(40), X, y, Xv, yv, rounds=10)
    assert b.mvo_es_info_["cap_extended_to"] is None
    assert b.mvo_es_info_["cap_bound_unresolved"] is True
    # 早停先到:不撞 cap
    fast = xgb.XGBClassifier(n_estimators=40, learning_rate=0.5, max_depth=3,
                             random_state=0, n_jobs=2)
    c = m._fit_es_on_validation(fast, X, y, Xv, yv, rounds=5)
    assert c.mvo_es_info_["cap_bound"] is False
    assert c.mvo_es_info_["cap_bound_unresolved"] is False
    # 余量仍在(cap 外扩到 40 < 上限 8000)时撞 cap:不算"未解除"
    m2, (X2, y2, Xv2, yv2) = _es_mod(tmp_path, 8000)
    e = m2._fit_es_on_validation(slow(20), X2, y2, Xv2, yv2, rounds=10)
    assert e.mvo_es_info_["cap_extended_to"] == 40
    assert e.mvo_es_info_["cap_bound"] is True
    assert e.mvo_es_info_["cap_bound_unresolved"] is False
    # 全局库硬化版同口径
    from _mvo.render import _HARD_FIT_ES_ON_VALIDATION
    assert 'info["cap_bound_unresolved"]' in _HARD_FIT_ES_ON_VALIDATION


#: v2.17.1 的 _fit_es_on_validation(--from-parent 复制来的旧代码形态)
_OLD_FIT_ES_ON_VALIDATION = '''def _fit_es_on_validation(model, X_tr, y_tr, X_va, y_va, rounds=None):
    """v2.16.0 单段拟合(v2.17.1 及以前的副本:es_info 无未解除标记)。"""
    import math as _math_v

    if rounds is None:
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
                "fit_mode": "single", "n_es_rows": int(len(X_va))}
        model = _es_stage1(model, _fam, X_tr, y_tr, X_va, y_va, rounds)
        _curve = _es_curve_of(model, _fam)
        _best = _es_best_trees(model, _fam)
        _cap = _model_cap(model, _fam)
        _bound = bool(_cap and len(_curve) >= int(_cap))
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
        raise
'''

_OLD_HURDLE_ES = '''def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    _cap, _bound, _face = 8000, True, "validation"
    metrics_extra = {
        "es_info": {"cap": _cap, "cap_bound": _bound,
                    "stopped_by": "cap" if _bound else "early_stop",
                    "es_source": _face, "fit_mode": "single",
                    "source": "hurdle_segments"},
    }
    return None, metrics_extra
'''


def test_upgrade_cap_unresolved_old_copy_idempotent(tmp_path):
    from _mvo.render import (_apply_es_source_upgrade, _template_func_sources,
                             _upgrade_cap_unresolved_code)

    src = _template_func_sources()
    helpers = "\n\n".join(src[n] for n in (
        "_clamp_trees", "_es_stage1", "_es_rounds_for", "_model_cap",
        "_set_cap", "_es_curve_of", "_es_best_trees"))
    # 旧副本里常见两份(v2.16 升级段 + 嫁接段),两份都要补
    old = (_OLD_FIT_ES_ON_VALIDATION + "\n\nX = 1\n\n\n"
           + _OLD_FIT_ES_ON_VALIDATION + "\n\n" + _OLD_HURDLE_ES)
    out, notes = _upgrade_cap_unresolved_code(old)
    assert any("×2" in n for n in notes), notes
    assert any("hurdle" in n for n in notes), notes
    assert out.count('info["cap_bound_unresolved"]') == 2
    assert _upgrade_cap_unresolved_code(out) == (out, [])          # 幂等
    # 补丁后的旧 helper 真能产出该键(exec 现行依赖 + 升级后的旧函数)
    ns: dict = {"__builtins__": __builtins__}
    exec("import os\n_MAX_TREES = 40\n"
         "def _lgbm_fit_safe(m, *a, **k):\n    return m.fit(*a, **k)\n"
         + helpers + "\n\n" + out, ns)
    X, y = pd.DataFrame(np.random.default_rng(0).normal(size=(800, 4))), \
        pd.Series(np.random.default_rng(1).integers(0, 2, 800))
    mdl = xgb.XGBClassifier(n_estimators=40, learning_rate=0.001,
                            max_depth=2, random_state=0, n_jobs=1)
    mdl = ns["_fit_es_on_validation"](mdl, X.iloc[:600], y.iloc[:600],
                                      X.iloc[600:], y.iloc[600:], rounds=200)
    assert mdl.mvo_es_info_["cap_bound"] is True
    assert mdl.mvo_es_info_["cap_bound_unresolved"] is True
    _, hm = ns["fit_and_score"](None, None, None)
    assert hm["es_info"]["cap_bound_unresolved"] is True
    # --from-parent 副本入口:第一次补齐,第二次无事可做
    tp = tmp_path / "train.py"
    tp.write_text("import os\n\n" + _OLD_FIT_ES_ON_VALIDATION
                  + '\n\nif __name__ == "__main__":\n    pass\n',
                  encoding="utf-8")
    n1 = _apply_es_source_upgrade(tp)
    assert any("cap_bound_unresolved" in n for n in n1), n1
    assert _apply_es_source_upgrade(tp) == []


def test_hurdle_hpo_refit_uses_search_best_iteration(tmp_path):
    """hurdle 超参段:lr-shrink 重训同口径(放宽收缩 + 留痕 + es_info 契约)。"""
    from test_v2_16_2_fixes import (_data, _load as _load162, _mk_run as
                                    _mk_run162, _mk_trial_dir, _parent_code)
    from _mvo import render as R

    run = _mk_run162(tmp_path)
    d = _mk_trial_dir(run, "trial-030-g05-hpo-xgboost-1", _parent_code())
    R._inject_hurdle_hp_segment(d / "code" / "train.py")
    m = _load162(d / "code" / "train.py")
    m._MAX_TREES = 120               # 父代 cap 120;×1/0.25 的需求必超上限
    X_tr, y_tr, X_va, y_va = _data()
    cfg = {"hyperparameters": {}, "seed": 7, "search_action": "hpo",
           "hpo": {"search_space": {"max_depth": {"type": "int", "low": 2,
                                                  "high": 3}},
                   "budget_configs": 2, "inner_fraction": 0.2,
                   "lr_mode": "shrink", "search_lr": 0.1,
                   "lr_shrink_factor": 0.25, "parallel_configs": 1,
                   "topology": "structural:hurdle", "enqueue": []}}
    _, mx = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, cfg)
    st = json.loads((d / "hpo-study.json").read_text(encoding="utf-8"))
    bi = H.search_best_iteration(st)
    plan = H.plan_refit_shrink(bi, search_lr=0.1, factor=0.25, cap_now=120,
                               cap_scale=4.0, max_trees=120)
    assert mx["hpo_refit_plan"]["search_best_iteration"] == bi
    assert mx["hpo_refit_lr_adjusted"] is plan["lr_adjusted"]
    assert isinstance(mx["hpo_refit_cap_clamped"], bool)      # 旧版是 dict
    assert mx["best_params"]["learning_rate"] == pytest.approx(plan["refit_lr"])
    assert st["refit"]["plan"]["factor_eff"] == pytest.approx(plan["factor_eff"])
    assert "cap_bound_unresolved" in st["refit"]["es_info"]


# ===========================================================================
# A3. HPO deadline 估算
# ===========================================================================

_SP1 = {"x": {"type": "float", "low": 0.0, "high": 1.0}}


def _sleeper(sec):
    def f(p):
        time.sleep(sec)
        return {"value": -abs(p["x"] - 0.3), "best_iteration": 10}
    return f


def test_deadline_parallel_does_not_collapse_budget():
    """2 并行、budget 整体装不下:旧实现首个 config 完成即收敛(40→2),现在
    按中位数 ÷ 并行外推,装不下时只要再跑一个仍赶得上就继续。"""
    r = H.run_hpo_search(_SP1, 16, _sleeper(0.25), metric_direction_up=True,
                         seed=1, n_jobs=2, deadline_ts=time.time() + 1.6)
    assert r["budget_truncated_by_deadline"] is True
    assert 6 <= r["n_configs_evaluated"] < 16, r["n_configs_evaluated"]
    est = r["deadline_estimate"]
    assert est["n_parallel"] == 2
    assert est["per_config_wall_sec"] == pytest.approx(
        est["median_fit_sec"] / 2, abs=0.06)
    assert est["truncated_at"] >= 6
    # 装得下时跑满预算
    r2 = H.run_hpo_search(_SP1, 6, _sleeper(0.05), metric_direction_up=True,
                          seed=1, n_jobs=2, deadline_ts=time.time() + 30)
    assert r2["n_configs_evaluated"] == 6
    assert r2["budget_truncated_by_deadline"] is False


def test_deadline_reserves_refit_time():
    no_res = H.run_hpo_search(_SP1, 20, _sleeper(0.05),
                              metric_direction_up=True, seed=2,
                              deadline_ts=time.time() + 0.7)
    with_res = H.run_hpo_search(_SP1, 20, _sleeper(0.05),
                                metric_direction_up=True, seed=2,
                                deadline_ts=time.time() + 0.7,
                                refit_reserve=0.55)
    assert no_res["n_configs_evaluated"] >= 7
    assert with_res["n_configs_evaluated"] <= 4
    assert with_res["deadline_estimate"]["refit_reserve_sec"] == \
        pytest.approx(0.55, abs=0.06)              # 留痕保留 1 位小数


def test_deadline_needs_n_parallel_completions_before_truncating():
    """deadline 已过:单路第 1 个完成即截断;2 并行至少等 2 个完成才截断。"""
    r1 = H.run_hpo_search(_SP1, 10, _sleeper(0.05), metric_direction_up=True,
                          seed=3, n_jobs=1, deadline_ts=time.time() - 1)
    assert r1["n_configs_evaluated"] == 1
    r2 = H.run_hpo_search(_SP1, 10, _sleeper(0.05), metric_direction_up=True,
                          seed=3, n_jobs=2, deadline_ts=time.time() - 1)
    assert 2 <= r2["n_configs_evaluated"] <= 4
    assert r2["deadline_estimate"]["n_fit_this_attempt"] >= 2


def test_refit_reserve_estimate_and_trial_yaml(tmp_path):
    ents = [{"fit_sec": 1400, "rounds": 3000}, {"fit_sec": 1500,
                                                "rounds": 3100}]
    # 中位数 1450 × 行数比 1.25 × min(1/0.25, 8000/3050)
    v = H.estimate_refit_seconds(ents, rows_ratio=1.25, trees_scale=4.0,
                                 max_trees=8000)
    assert v == pytest.approx(1450 * 1.25 * 8000 / 3050, rel=1e-3)
    assert H.estimate_refit_seconds(ents, static_sec=99999) == 99999
    assert H.estimate_refit_seconds([], static_sec=None) == 0.0
    (tmp_path / "trial.yaml").write_text(yaml.safe_dump({
        "expected_resources": {"hpo_cost_estimate": {"retrain_sec": 812.5}}}))
    assert H.hpo_refit_reserve_sec(tmp_path) == pytest.approx(812.5)
    assert H.hpo_refit_reserve_sec(tmp_path / "nope") is None


# ===========================================================================
# A4. 抽样保真
# ===========================================================================

def test_stratified_random_rows_not_prefix_and_deterministic():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(3000, int), np.zeros(7000, int)]      # 时间序头部全正
    rng.shuffle(y[5000:])
    a = H.stratified_random_rows(y, 1000)
    b = H.stratified_random_rows(y, 1000)
    assert np.array_equal(a, b)                              # 确定性
    assert list(a) == sorted(a)                              # 保持时间序
    assert not np.array_equal(a, np.arange(len(a)))          # 不是前缀
    assert abs(y[a].mean() - y.mean()) < 0.01                # 分层
    assert a.max() > 9000                                    # 覆盖到尾部


def _hpo_cfg_sample(extra=None):
    cfg = {"hyperparameters": {}, "seed": 7, "search_action": "hpo",
           "hpo": {"search_space": {"max_depth": {"type": "int", "low": 2,
                                                  "high": 4}},
                   "budget_configs": 3, "inner_fraction": 0.2,
                   "sample_rows": 600, "lr_mode": "search",
                   "parallel_configs": 1, "auto_expand": False,
                   "enqueue": []}}
    cfg["hpo"].update(extra or {})
    return cfg


def test_template_hpo_sampling_is_stratified_random(tmp_path):
    run = _mk_run(tmp_path)
    d = _render_trial(run, "trial-006-g02-hpo-xgboost-1", {"hyperparameters": {}})
    m = _load(d / "code" / "train.py")
    m.N_ESTIMATORS_CAP = 40
    X_tr, y_tr, X_va, y_va = _xy(run)
    seen = []
    orig = m._hpo_fit_inner

    def spy(params, X_fit, y_fit, X_in, y_in, seed, fidelity=1.0):
        seen.append((list(X_fit.index), float(np.mean(y_fit))))
        return orig(params, X_fit, y_fit, X_in, y_in, seed)

    m._hpo_fit_inner = spy
    _, mx = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, _hpo_cfg_sample())
    pool = int(len(X_tr) * 0.8)
    idx, rate = seen[0]
    assert mx["hpo_sample_method"] == "stratified_random"
    assert mx["hpo_sample_rows"] == 600 and abs(len(idx) - 600) <= 2
    assert idx != list(range(len(idx)))                       # 不是头部
    assert idx == sorted(idx) and max(idx) < pool             # 只在拟合池内
    assert max(idx) > pool * 0.9                              # 覆盖池尾部
    assert abs(rate - float(y_tr.iloc[:pool].mean())) < 0.02  # 分层
    assert all(s[0] == idx for s in seen)                     # 各 config 同一面
    # 确定性:重跑同一面
    seen.clear()
    m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, _hpo_cfg_sample())
    assert seen[0][0] == idx
    # 默认不复评
    assert "hpo_fidelity_recheck" not in mx


def test_template_hpo_fidelity_recheck_opt_in(tmp_path):
    run = _mk_run(tmp_path)
    d = _render_trial(run, "trial-007-g02-hpo-xgboost-1", {"hyperparameters": {}})
    m = _load(d / "code" / "train.py")
    m.N_ESTIMATORS_CAP = 40
    X_tr, y_tr, X_va, y_va = _xy(run)
    sizes = []
    orig = m._hpo_fit_inner

    def spy(params, X_fit, y_fit, X_in, y_in, seed, fidelity=1.0):
        sizes.append(len(X_fit))
        return orig(params, X_fit, y_fit, X_in, y_in, seed)

    m._hpo_fit_inner = spy
    _, mx = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7,
                            _hpo_cfg_sample({"fidelity_recheck_top_k": 2}))
    pool = int(len(X_tr) * 0.8)
    assert sizes.count(pool) == 2                     # 前 2 名在全量池复评
    rc = mx["hpo_fidelity_recheck"]
    assert rc["top_k"] == 2 and rc["n_rechecked"] == 2
    st = json.loads((d / "hpo-study.json").read_text(encoding="utf-8"))
    fr = st["fidelity_recheck"]
    assert len(fr["candidates"]) == 2
    assert st["search_best_params"] == fr["best_params"]
    assert st["inner_score"] == pytest.approx(fr["best_full_metric"])
    assert "inner_score_sampled" in st


def test_fidelity_recheck_pure_and_config_default():
    res = {"trials": [{"params": {"a": 1}, "inner_metric": 0.9},
                      {"params": {"a": 2}, "inner_metric": 0.8},
                      {"params": {"a": 1}, "inner_metric": 0.9},
                      {"params": {"a": 3}, "inner_metric": 0.7}]}
    out = H.fidelity_recheck(res, lambda p: {"value": 1.0 / p["a"] if p["a"]
                                             != 1 else 0.1,
                                             "best_iteration": 10 * p["a"]},
                             top_k=2, metric_direction_up=True)
    assert out["n_rechecked"] == 2                         # 去重后前 2 名
    assert out["best_params"] == {"a": 2} and out["best_changed"] is True
    assert out["best_iteration"] == 20
    from runtime.config import Config
    c = Config.load(skill_config=False)
    assert c.get("evolution.hpo.fidelity_recheck_top_k") == 0
    assert c.get("confirm.parallel_seeds") == "auto"


# ===========================================================================
# A5. n_features = 拟合期模型列数
# ===========================================================================

_OLD_MAIN = '''import json
import pathlib

import pandas as pd

TRIAL_DIR = pathlib.Path(__TD__)
BASE = pd.DataFrame({f"b{i}": range(5) for i in range(6)})
BLOCKS = pd.DataFrame({f"w{i}": range(5) for i in range(4)})


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    return [0.5] * len(X_va), {"n_features": X_tr.shape[1], "model_num": 1}


def main():
    feats = list(BASE.columns)
    X_tr = pd.concat([BASE, BLOCKS], axis=1)
    __TRAIN__
    metrics = {"n_features": len(feats),
               "n_train": len(X_tr)}
    __MERGE__
    with open(TRIAL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
'''


def _old_main(td: Path, rewired: bool) -> str:
    return (_OLD_MAIN.replace("__TD__", repr(str(td)))
            .replace("__TRAIN__", "score, _fs_metrics = fit_and_score("
                     "X_tr, None, X_tr, None)" if rewired else "pass")
            .replace("__MERGE__", "for _k, _v in (_fs_metrics or {}).items():"
                     "\n        metrics.setdefault(_k, _v)" if rewired
                     else "pass"))


def _run_main(code: str) -> dict:
    ns: dict = {"__name__": "mvo_v2172_nf"}
    exec(compile(code, "train.py", "exec"), ns)
    ns["main"]()
    return json.loads((Path(ns["TRIAL_DIR"]) / "metrics.json").read_text())


@pytest.mark.parametrize("rewired", [True, False])
def test_upgrade_n_features_old_main(tmp_path, rewired):
    from _mvo.render import _apply_es_source_upgrade, _upgrade_n_features_code

    code = _old_main(tmp_path, rewired)
    assert _run_main(code)["n_features"] == 6           # 旧行为:基础列数胜出
    out, notes = _upgrade_n_features_code(code)
    assert notes and "n_features_base" in notes[0]
    mt = _run_main(out)
    assert mt["n_features"] == 10 and mt["n_features_base"] == 6
    assert _upgrade_n_features_code(out) == (out, [])    # 幂等
    tp = tmp_path / "train.py"
    tp.write_text(code, encoding="utf-8")
    assert any("n_features" in n for n in _apply_es_source_upgrade(tp))
    assert _apply_es_source_upgrade(tp) == []
    # 新模板 main 不受影响(只补 n_features_base)
    from _mvo.templates import TRIAL_BODY
    assert _upgrade_n_features_code(TRIAL_BODY)[1] == []
    assert 'metrics_extra["n_features_base"] = int(_n_base_cols)' in TRIAL_BODY


def test_rewire_injects_n_features_block():
    from _mvo.render import _NF_UPGRADE_MARK, _rewire_main_to_fit_and_score

    code = (
        "import json\nSEED = 20260814\n"
        "def main():\n"
        "    cfg = {}\n    X_tr = y_tr = X_va = y_va = None\n"
        "    feats = []\n"
        "    model = build_estimator(1)\n"
        "    pd.DataFrame().to_parquet(TRIAL_DIR / 'validation_score.parquet')\n"
        "    metrics = {'n_features': len(feats)}\n"
        "    with open(TRIAL_DIR / 'metrics.json', 'w') as fh:\n"
        "        json.dump(metrics, fh)\n")
    out, notes = _rewire_main_to_fit_and_score(code)
    assert out is not None and _NF_UPGRADE_MARK in out
    assert out.index("metrics.setdefault(_k, _v)") < out.index(_NF_UPGRADE_MARK)
    assert any("n_features" in n for n in notes)


# ===========================================================================
# D2. confirm seed 变体并行
# ===========================================================================

_CONFIRM_CFG = {"hyperparameters": {"max_depth": 3, "n_estimators": 40,
                                    "learning_rate": 0.1},
                "search_action": "confirm", "seed": 11}


def test_confirm_seed_variants_parallel_matches_serial(tmp_path, monkeypatch):
    run = _mk_run(tmp_path, n_tr=2000, n_va=800)
    d_ser = _render_trial(run, "trial-010-g03-confirm-xgboost-1", _CONFIRM_CFG)
    d_par = _render_trial(run, "trial-011-g03-confirm-xgboost-2", _CONFIRM_CFG)
    monkeypatch.setenv("MODELEVO_CONFIRM_PARALLEL_SEEDS", "off")
    _load(d_ser / "code" / "train.py").main()
    ms = json.loads((d_ser / "metrics.json").read_text(encoding="utf-8"))
    assert ms["multi_seed_n"] == 3
    assert ms["multi_seed_parallel"]["mode"] == "serial"
    assert ms["multi_seed_parallel"]["reason"] == "config_off"
    assert ms["n_features"] == 6 and ms["n_features_base"] == 6
    monkeypatch.delenv("MODELEVO_CONFIRM_PARALLEL_SEEDS")
    m_par = _load(d_par / "code" / "train.py")
    # 测试机无 GPU:直接给出"可并行"计划(CPU worker),验证执行与合并口径
    monkeypatch.setattr(m_par, "_seed_parallel_plan", lambda n: {
        "mode": "parallel", "reason": "ok", "n_parallel": n,
        "slots": [{"gpu": None}] * (n - 1)})
    m_par.main()
    mp = json.loads((d_par / "metrics.json").read_text(encoding="utf-8"))
    par = mp["multi_seed_parallel"]
    assert par["mode"] == "parallel"
    assert par["workers"] == {"2": {"gpu": None, "rc": 0, "ok": True}}, par
    assert mp["multi_seed_n"] == 3
    for k in (1, 2):
        a = pd.read_parquet(d_ser / "seed_scores" / f"seed-{k}.parquet")
        b = pd.read_parquet(d_par / "seed_scores" / f"seed-{k}.parquet")
        assert np.array_equal(a["score"].to_numpy(), b["score"].to_numpy()), k
    assert not np.array_equal(
        pd.read_parquet(d_par / "seed_scores" / "seed-1.parquet")["score"],
        pd.read_parquet(d_par / "seed_scores" / "seed-2.parquet")["score"])
    # 主轮分数与工件不受变体轮影响
    assert np.array_equal(
        pd.read_parquet(d_ser / "validation_score.parquet")["score"],
        pd.read_parquet(d_par / "validation_score.parquet")["score"])
    assert not list((d_par / "seed_scores").glob(".seed-*.tmp-*"))


def _card(i, free=70.0, procs=()):
    return {"index": i, "uuid": f"u{i}", "memory_total_gb": 80.0,
            "memory_used_gb": 80.0 - free, "memory_free_gb": free,
            "util_pct": 0, "procs": list(procs), "procs_known": True}


def test_seed_parallel_plan_fallbacks_and_lease(tmp_path, monkeypatch):
    import runtime.scheduler as S

    run = _mk_run(tmp_path, n_tr=200, n_va=100)
    d = _render_trial(run, "trial-012-g03-confirm-xgboost-3", _CONFIRM_CFG)
    m = _load(d / "code" / "train.py")
    assert m._seed_parallel_plan(1)["reason"] == "single_variant"
    monkeypatch.setenv("MODELEVO_CONFIRM_PARALLEL_SEEDS", "off")
    assert m._seed_parallel_plan(2)["reason"] == "config_off"
    monkeypatch.delenv("MODELEVO_CONFIRM_PARALLEL_SEEDS")
    (run / "modelevo-config.yaml").write_text(
        "modelevo:\n  confirm:\n    parallel_seeds: 'off'\n", encoding="utf-8")
    assert m._seed_parallel_plan(2)["reason"] == "config_off"   # run 配置
    (run / "modelevo-config.yaml").unlink()
    assert m._seed_parallel_plan(2)["reason"] == "no_gpu_allocated"
    monkeypatch.setenv("MODELEVO_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(S, "query_gpu_occupancy", lambda **kw: None)
    assert m._seed_parallel_plan(2)["reason"] == "gpu_query_failed"
    # 只有本卡 / 其余卡被外来进程占用 → 无额外卡
    monkeypatch.setattr(S, "query_gpu_occupancy", lambda **kw: [
        _card(0), _card(1, procs=[{"pid": 999999991, "used_gb": 9.0}])])
    assert m._seed_parallel_plan(2)["reason"] == "no_extra_gpu"
    # 有干净卡但内存不足
    monkeypatch.setattr(S, "query_gpu_occupancy",
                        lambda **kw: [_card(0), _card(1)])
    monkeypatch.setattr(S, "estimate_trial_peak_gb",
                        lambda *a, **k: {"estimate_gb": 50.0,
                                         "source": "test"})
    ev = {"ok": True, "decision": "admitted", "available_gb": 500.0,
          "outstanding_gb": 0.0, "n_reserved": 0, "need_gb": 50.0,
          "margin_gb": 60.0}
    monkeypatch.setattr(S.MemoryAdmission, "evaluate",
                        lambda self, est: dict(ev, headroom_gb=30.0))
    p = m._seed_parallel_plan(3)
    assert p["mode"] == "serial" and p["reason"] == "insufficient_memory"
    assert p["per_worker_gb"] >= 50.0
    # 卡与内存都满足 → 并行;额外卡经 GPU 租约(本 run .gpu-leases)排他持有
    monkeypatch.setattr(S.MemoryAdmission, "evaluate",
                        lambda self, est: dict(ev, headroom_gb=120.0))
    p = m._seed_parallel_plan(3)
    assert p["mode"] == "parallel" and p["slots"] == [{"gpu": 1}]
    assert p["n_parallel"] == 2                  # 卡数封顶(内存够 2 个 worker)
    lease = run / ".gpu-leases" / "gpu-1.lock"
    assert lease.exists()
    # 额外 worker 的内存准入预留(logs/admission.jsonl 同源)
    resv = run / S.ADMISSION_DIR
    assert p["_ticket"] is not None and len(list(resv.glob("*.json"))) == 1
    # 已租走 → 同 run 的另一次计划租不到同一张卡
    assert m._seed_parallel_plan(3)["reason"] == "gpu_lease_unavailable"
    m._seed_parallel_release(p)
    assert not lease.exists() and not list(resv.glob("*.json"))
    # 释放后可再租
    p2 = m._seed_parallel_plan(3)
    assert p2["mode"] == "parallel"
    m._seed_parallel_release(p2)


def test_template_main_runs_parallel_helper_only_for_multi_seed():
    from _mvo.templates import TRIAL_BODY

    i = TRIAL_BODY.index("def main(")
    body = TRIAL_BODY[i:]
    assert "_run_seed_variants_parallel(" in body
    assert body.index("_run_seed_variants_parallel(") < body.index(
        "for _k in _todo:")
    # worker 分支在主轮 fit_and_score 之前(worker 不训主轮、不写 metrics)
    assert body.index("_seed_worker_run(") < body.index(
        "score, _m = fit_and_score(")
    assert re.search(r"if os\.environ\.get\(_SEED_WORKER_ENV\):", body)
