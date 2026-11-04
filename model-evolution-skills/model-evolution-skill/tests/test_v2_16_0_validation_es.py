"""v2.16.0 回归测试:outer 拟合在 validation 上早停、100% train 只训一次。

动机与口径见 docs/MODELEVO_V2_16_0_VALIDATION_ES_20261022.md。本文件锁定
(全部离线、无 CLI):

  A. 模板单段拟合:三族只 fit 一次、早停面 = X_va、预测按最优轮数、遥测键;
     env 分派与 train_tail 回退;seed 变体复用主轮早停面;
  B. 旧代码升级:_fit_with_eval 段插入、baseline 血缘 main 守卫、幂等、可编译;
  C. lint 规则 8 按早停面口径生效;
  D. baseline 模板渲染与 ES_SOURCE 改写;
  E. 交付:refit 模式解析、拒绝 rerun_recipe、最优树数、es_source 读取。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "assets"))

xgb = pytest.importorskip("xgboost")
lgb = pytest.importorskip("lightgbm")

from test_v2_15_0_hpo import _clf, _exec_segment  # noqa: E402


def _ns(extra: str = "") -> dict:
    # _fit_with_es 起到 _es_info_of 前:含两段式原语、_es_source、
    # _fit_es_on_validation 与 _fit_with_eval(均取自现行模板原文)
    return _exec_segment("_fit_with_es", "_es_info_of", extra)


def _split(n=4000):
    X, y = _clf(n, 0)
    k = int(n * 0.75)
    return X.iloc[:k], y.iloc[:k], X.iloc[k:], y.iloc[k:]


def _count_fits(model):
    calls = {"n": 0, "eval_rows": []}
    orig = type(model).fit

    def fit(self, X, y, *a, **kw):
        calls["n"] += 1
        es = kw.get("eval_set")
        if es:
            first = es[0] if isinstance(es, list) else es
            calls["eval_rows"].append(len(first[0]))
        return orig(self, X, y, *a, **kw)

    return calls, orig, fit


# --------------------------------------------------------------------------
# A. 模板单段拟合
# --------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost"])
def test_single_fit_on_validation(family, monkeypatch):
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    ns = _ns()
    Xtr, ytr, Xva, yva = _split()
    if family == "xgboost":
        m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1,
                              max_depth=3, early_stopping_rounds=30,
                              random_state=0, n_jobs=2)
    elif family == "lightgbm":
        m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.1,
                               num_leaves=15, random_state=0, verbose=-1)
    else:
        cb = pytest.importorskip("catboost")
        m = cb.CatBoostClassifier(iterations=2000, learning_rate=0.1,
                                  depth=4, random_seed=0, verbose=0,
                                  allow_writing_files=False)
    calls, orig, fit = _count_fits(m)
    monkeypatch.setattr(type(m), "fit", fit)
    out = ns["_fit_with_eval"](m, Xtr, ytr, Xva, yva)
    monkeypatch.setattr(type(m), "fit", orig)
    info = out.mvo_es_info_
    assert calls["n"] == 1                       # 只训一次
    assert calls["eval_rows"] == [len(Xva)]      # 早停面 = outer validation
    assert info["es_source"] == "validation" and info["fit_mode"] == "single"
    assert "refit_trees" not in info
    assert info["stopped_by"] == "early_stop" and info["cap_bound"] is False
    # cap 一次性外扩(默认 ×2),早停先叫停
    assert info["cap_initial"] == 2000 and info["cap"] == 4000
    assert info["stage1_rounds"] == len(info["stage1_curve"])
    assert 0 < info["stage1_best_iteration"] <= info["stage1_rounds"]
    assert ns["_LAST_EVAL_XY"][0] is Xva


def test_xgboost_predicts_with_best_iteration(monkeypatch):
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    ns = _ns()
    Xtr, ytr, Xva, yva = _split()
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1, max_depth=3,
                          early_stopping_rounds=30, random_state=0, n_jobs=2)
    out = ns["_fit_with_eval"](m, Xtr, ytr, Xva, yva)
    best = out.mvo_es_info_["stage1_best_iteration"]
    # 模型里还留着耐心那段树,但预测只用前 best 棵
    assert out.get_booster().num_boosted_rounds() > best
    p_auto = out.predict_proba(Xva)[:, 1]
    p_best = out.predict_proba(Xva, iteration_range=(0, best))[:, 1]
    np.testing.assert_allclose(p_auto, p_best)


def test_train_tail_fallback_and_missing_labels(monkeypatch):
    ns = _ns()
    Xtr, ytr, Xva, yva = _split()
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "train_tail")
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1, max_depth=3,
                          early_stopping_rounds=30, random_state=0, n_jobs=2)
    out = ns["_fit_with_eval"](m, Xtr, ytr, Xva, yva)
    assert "es_source" not in out.mvo_es_info_          # 两段式遥测
    assert out.mvo_es_info_["refit_trees"] == \
        out.mvo_es_info_["stage1_best_iteration"]
    monkeypatch.delenv("MODELEVO_ES_SOURCE")
    m2 = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1, max_depth=3,
                           early_stopping_rounds=30, random_state=0, n_jobs=2)
    out2 = ns["_fit_with_eval"](m2, Xtr, ytr, None, None)   # smoke 等无 y_va
    assert "refit_trees" in out2.mvo_es_info_


def test_es_source_resolution(monkeypatch):
    ns = _ns()
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    assert ns["_es_source"]() == "validation"
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "TRAIN_TAIL")
    assert ns["_es_source"]() == "train_tail"
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "bogus")
    assert ns["_es_source"]() == "validation"


def test_seed_variant_reuses_validation_face(monkeypatch):
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    extra = ("_HPO_ALGO = 'xgboost'\nN_ESTIMATORS_CAP = 500\n"
             "EARLY_STOP_ROUNDS = 20\nSEED = 0\n")
    ns = _exec_segment("_build_model", "_inner_metric", extra)
    ns2 = _exec_segment("_fit_with_es", "step", extra)
    ns2["_build_model"] = ns["_build_model"]
    Xtr, ytr, Xva, yva = _split()
    ns2["_PRIMARY_FIT_PARAMS"] = {"max_depth": 3, "learning_rate": 0.1}
    ns2["_PRIMARY_FIT_XY"] = (Xtr, ytr)
    ns2["_LAST_EVAL_XY"] = (Xva, yva)
    m = ns2["_fit_seed_variant"](2, Xtr, ytr, None)
    assert m.mvo_es_info_["es_source"] == "validation"
    assert m.mvo_es_info_["n_es_rows"] == len(Xva)


def test_learning_curve_source_validation(tmp_path):
    ns = _exec_segment("_es_info_of", "_fit_seed_variant",
                       f"TRIAL_DIR = pathlib.Path({str(tmp_path)!r})\n")

    class M:
        mvo_es_info_ = {"stage1_curve": [0.5, 0.4, 0.45],
                        "stage1_best_iteration": 2, "es_source": "validation"}

    out = ns["dump_learning_curve"](M())
    assert out == {"learning_curve_rows": 3,
                   "learning_curve_source": "es_validation"}
    assert ns["best_iteration_of"](M()) == 2


# --------------------------------------------------------------------------
# B. 旧代码升级
# --------------------------------------------------------------------------

_OLD_LINEAGE = '''
import os
MODEL_TYPE = "binary_classification"
_MAX_TREES = 8000


def _clamp_trees(v):
    return max(1, min(int(v), _MAX_TREES))


def step(msg):
    print(msg)


def _build_model(params, seed, n):
    return None


def _fit_with_es(model, X, y, es_fraction=0.1, rounds=None, refit=True,
                 auto_extend=False):
    return model


def _fit_with_eval(model, X_tr, y_tr, X_va, y_va):
    return _fit_with_es(model, X_tr, y_tr, auto_extend=True)


def main():
    X_tr = y_tr = X_va = y_va = None
    _es_cut = 10
    X_fit, X_es = X_tr, X_tr
    y_fit, y_es = y_tr.iloc[:_es_cut], y_tr.iloc[_es_cut:]
    return X_fit, X_es


if __name__ == "__main__":
    main()
'''


def test_upgrade_old_lineage_code_inserts_block_and_guard():
    from _mvo.render import (_ES_MAIN_GUARD, _ES_UPGRADE_BEGIN,
                             _upgrade_es_source_code)

    new, notes = _upgrade_es_source_code(_OLD_LINEAGE)
    assert len(notes) == 2
    compile(new, "upgraded", "exec")
    assert new.count(_ES_UPGRADE_BEGIN) == 1 and _ES_MAIN_GUARD in new
    # 新段在 __main__ 之前,重定义生效(后定义覆盖)
    blk = new.index(_ES_UPGRADE_BEGIN)
    assert blk < new.rindex('if __name__ == "__main__"')
    assert new.index("def _fit_es_on_validation(") > blk
    assert new.rindex("def _fit_with_eval(") > blk
    assert new.rindex("def _fit_seed_variant(") > blk     # 有 _build_model
    # 缺的依赖补齐,已有的不重复
    assert "def _es_stage1(" in new and "def _es_rounds_for(" in new
    assert new.count("def _fit_with_es(") == 1
    # 守卫紧跟锚点行、缩进一致,且要求拿得到 y_va
    anchor = "    y_fit, y_es = y_tr.iloc[:_es_cut], y_tr.iloc[_es_cut:]\n"
    i = new.index(anchor) + len(anchor)
    assert new[i:].startswith('    if (_es_source() == "validation"')
    assert 'locals().get("y_va") is not None' in new[i:i + 200]
    # 幂等
    again, notes2 = _upgrade_es_source_code(new)
    assert again == new and notes2 == []


def test_upgrade_guards_every_anchor():
    from _mvo.render import _ES_MAIN_GUARD, _upgrade_es_source_code

    anchor = "y_fit, y_es = y_tr.iloc[:_es_cut], y_tr.iloc[_es_cut:]"
    code = ("MODEL_TYPE = 'regression'\n\ndef step(m):\n    pass\n\n"
            "def fit_and_score(X_tr, y_tr, X_va, y_va=None):\n"
            f"    _es_cut = 3\n    X_fit, X_es = X_tr, X_tr\n    {anchor}\n"
            "    return X_fit, X_es\n\n"
            "def main():\n    X_tr = y_tr = X_va = y_va = None\n"
            f"    _es_cut = 3\n    X_fit, X_es = X_tr, X_tr\n    {anchor}\n"
            "    return X_fit\n")
    new, notes = _upgrade_es_source_code(code)
    assert new.count(_ES_MAIN_GUARD) == 2
    assert any("×2" in n for n in notes)
    compile(new, "x", "exec")
    assert _upgrade_es_source_code(new) == (new, [])
    # smoke 等拿不到 y_va 时守卫不生效(仍按 train 尾部)
    ns: dict = {}
    exec(new, ns)
    import pandas as _pd
    X = _pd.DataFrame({"a": range(10)})
    y = _pd.Series(range(10))
    ns["fit_and_score"](X, y, X.iloc[:4], None)      # 不抛
    Xf, Xe = ns["fit_and_score"](X, y, X.iloc[:4], y.iloc[:4])
    assert len(Xf) == 10 and len(Xe) == 4


_HURDLE_CODE = '''
import numpy as np
import xgboost as xgb
MODEL_TYPE = "regression"
N_ESTIMATORS_CAP = 400
ES_FRACTION = 0.1
CALLS = []


def step(m):
    pass


class HurdleXGB:
    def __init__(self, clf, reg, stage1=None):
        self.clf_, self.reg_, self.stage1_ = clf, reg, stage1


def _stage1_rounds(m):
    return int(m.get_booster().num_boosted_rounds())


def _fit_hurdle(n_jobs, X_fit, y_fit, X_es, y_es):
    CALLS.append(("fit_hurdle", len(X_fit), len(X_es)))
    kw = dict(n_estimators=N_ESTIMATORS_CAP, learning_rate=0.1, max_depth=3,
              early_stopping_rounds=20, n_jobs=2)
    clf, reg = xgb.XGBClassifier(**kw), xgb.XGBRegressor(**kw)
    pf, pe = np.asarray(y_fit) > 0, np.asarray(y_es) > 0
    clf.fit(X_fit, pf.astype(int), eval_set=[(X_es, pe.astype(int))],
            verbose=False)
    reg.fit(X_fit[pf], y_fit[pf], eval_set=[(X_es[pe], y_es[pe])],
            verbose=False)
    return HurdleXGB(clf, reg)


def _fit_hurdle_refit_full(n_jobs, X_tr, y_tr):
    CALLS.append(("refit_full", len(X_tr)))
    _cut = int(len(X_tr) * (1.0 - ES_FRACTION))
    s1 = _fit_hurdle(n_jobs, X_tr.iloc[:_cut], y_tr.iloc[:_cut],
                     X_tr.iloc[_cut:], y_tr.iloc[_cut:])
    s1.stage1_ = {"clf_best_iteration": 1, "reg_best_iteration": 1}
    return s1


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    model = _fit_hurdle_refit_full(2, X_tr, y_tr)
    return model, dict(model.stage1_)


if __name__ == "__main__":
    pass
'''


def test_upgrade_hurdle_two_stage_to_single_fit(monkeypatch):
    from _mvo.render import _ES_HURDLE_BEGIN, _upgrade_es_source_code

    new, notes = _upgrade_es_source_code(_HURDLE_CODE)
    assert "hurdle 两段式改为 validation 单段早停" in notes
    assert new.count(_ES_HURDLE_BEGIN) == 1
    assert _upgrade_es_source_code(new) == (new, [])
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 6)), columns=list("abcdef"))
    y = pd.Series(np.clip(X["a"] + 0.5 * X["b"] + rng.normal(size=3000),
                          0, None))
    Xtr, ytr, Xva, yva = X.iloc[:2400], y.iloc[:2400], X.iloc[2400:], \
        y.iloc[2400:]
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    ns: dict = {}
    exec(new, ns)
    model, st = ns["fit_and_score"](Xtr, ytr, Xva, yva)
    # 单段:100% train 拟合、outer validation 早停,不再"尾部早停 → 重训"
    assert ns["CALLS"] == [("fit_hurdle", 2400, 600)]
    assert st["es_source"] == "validation" and st["fit_mode"] == "single"
    assert st["clf_best_iteration"] == model.clf_.best_iteration + 1
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "train_tail")
    ns2: dict = {}
    exec(new, ns2)
    ns2["fit_and_score"](Xtr, ytr, Xva, yva)
    assert ns2["CALLS"] == [("refit_full", 2400), ("fit_hurdle", 2160, 240)]


def test_reuse_and_export_paths_wire_upgrade():
    st = (SKILL / "scripts" / "_mvo" / "cmd_startup.py").read_text(
        encoding="utf-8")
    body = st.split("def _apply_frozen_baseline(")[1].split("\ndef ")[0]
    assert "_apply_es_source_upgrade(" in body          # 固化 baseline 复用
    assert st.count("_apply_es_source_upgrade(") >= 2   # + warm-start
    # v2.18.0:导出不再就地硬化库内配方 —— 经验库配方入库前已统一为单段口径,
    # 记 runtime_api_version;复用路径仍对本 run 的副本兜底硬化(上两条断言)


def test_upgrade_skips_new_template_code():
    from _mvo.render import _upgrade_es_source_code
    from _mvo.templates import TRIAL_BODY

    out, notes = _upgrade_es_source_code(TRIAL_BODY)
    assert out == TRIAL_BODY and notes == []


def test_upgrade_baseline_lineage_only_guard():
    from _mvo.render import _upgrade_es_source_code

    code = _OLD_LINEAGE.replace(
        "def _fit_with_eval(model, X_tr, y_tr, X_va, y_va):\n"
        "    return _fit_with_es(model, X_tr, y_tr, auto_extend=True)\n", "")
    new, notes = _upgrade_es_source_code(code)
    compile(new, "upgraded", "exec")
    assert "def _es_source(" in new
    assert "def _fit_es_on_validation(" not in new


def test_apply_upgrade_on_file(tmp_path):
    from _mvo.render import _apply_es_source_upgrade

    p = tmp_path / "train.py"
    p.write_text(_OLD_LINEAGE, encoding="utf-8")
    assert _apply_es_source_upgrade(p)
    assert _apply_es_source_upgrade(p) == []


# --------------------------------------------------------------------------
# C. lint 规则 8 按口径生效
# --------------------------------------------------------------------------

def test_lint_rule8_follows_es_source(tmp_path, monkeypatch):
    import lint_trial as L

    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    assert L.resolve_es_source(run) == "validation"
    (run / "modelevo-config.yaml").write_text(
        "modelevo:\n  training:\n    early_stop_source: train_tail\n",
        encoding="utf-8")
    assert L.resolve_es_source(run) == "train_tail"
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "validation")
    assert L.resolve_es_source(run) == "validation"
    # 规则函数本身语义不变(train_tail 下仍拦)
    assert L.check_eval_set_leak("eval_set=[(X_va, y_va)]")


# --------------------------------------------------------------------------
# D. baseline 模板
# --------------------------------------------------------------------------

def test_baseline_template_validation_es():
    from runtime.baseline import BASELINE_TEMPLATE, ESTIMATOR_CODE, FIT_CODE

    for algo in ESTIMATOR_CODE:
        code = BASELINE_TEMPLATE.format(
            algorithm=algo, target="y", model_type="binary_classification",
            treatment="", seed=1, n_estimators_cap=2000,
            early_stopping_rounds=50, es_fraction=0.1,
            estimator_code=ESTIMATOR_CODE[algo], fit_code=FIT_CODE[algo])
        compile(code, algo, "exec")
        assert 'ES_SOURCE = "validation"' in code
        assert "X_fit, y_fit, X_es, y_es = X_tr, y_tr, X_va, y_va" in code
        assert '"es_source": _es_face' in code


def test_config_default_es_source():
    from runtime.config import DEFAULTS

    assert DEFAULTS["training"]["early_stop_source"] == "validation"


# --------------------------------------------------------------------------
# E. 交付
# --------------------------------------------------------------------------

def test_trial_es_source_and_refit_resolution(tmp_path):
    from types import SimpleNamespace

    from _mvo.cmd_deliver import _resolve_refit_for_champion, _trial_es_source

    t = tmp_path / "t"
    t.mkdir()
    assert _trial_es_source(t) == "train_tail"            # 无 metrics
    (t / "metrics.json").write_text(json.dumps(
        {"es_info": {"es_source": "validation"}}))
    assert _trial_es_source(t) == "validation"
    (t / "metrics.json").write_text(json.dumps(
        {"refit": {"es_info": {"es_source": "validation"}}}))
    assert _trial_es_source(t) == "validation"

    a = SimpleNamespace(refit_mode=None)
    assert _resolve_refit_for_champion(a, "rerun_recipe",
                                       "validation") == "frozen_rounds"
    assert _resolve_refit_for_champion(a, "rerun_recipe",
                                       "train_tail") == "rerun_recipe"
    with pytest.raises(SystemExit):
        _resolve_refit_for_champion(SimpleNamespace(refit_mode="rerun_recipe"),
                                    "rerun_recipe", "validation")


def test_validation_es_rounds_uses_best_not_boosted(monkeypatch):
    from _mvo.cmd_deliver import _fitted_rounds, _validation_es_rounds

    Xtr, ytr, Xva, yva = _split()
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1, max_depth=3,
                          early_stopping_rounds=30, random_state=0, n_jobs=2)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    assert _validation_es_rounds(m) == m.best_iteration + 1
    assert _fitted_rounds(m) == _validation_es_rounds(m) + 30
    m.mvo_es_info_ = {"stage1_best_iteration": 7}
    assert _validation_es_rounds(m) == 7
    lm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.1,
                            num_leaves=15, verbose=-1)
    lm.fit(Xtr, ytr, eval_set=[(Xva, yva)],
           callbacks=[lgb.early_stopping(30, verbose=False)])
    assert _validation_es_rounds(lm) == lm.best_iteration_


def test_retrain_body_wires_es_source():
    src = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    body = src.split("def cmd_finalize_retrain(")[1].split(
        "\ndef cmd_finalize_baseline_score(")[0]
    assert "_resolve_refit_for_champion(args, refit_mode, _champ_es)" in body
    assert "_validation_es_rounds(_m_ref)" in body
    assert '"es_source": _champ_es' in body
    base = src.split("def cmd_finalize_baseline_score(")[1].split("\ndef ")[0]
    assert "_trial_es_source(base_dir)" in base
    assert "_fit_baseline_train_tail(" in base
    # 复合容器(hurdle 等不可 clone)的 validation 冠军:train_tail 口径复跑
    assert "_composite_rerun = True" in body
    assert 'os.environ["MODELEVO_ES_SOURCE"] = "train_tail"' in body
