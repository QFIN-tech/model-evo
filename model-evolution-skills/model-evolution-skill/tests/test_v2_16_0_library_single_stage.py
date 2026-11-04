"""v2.16.0 全局库配方单段硬化:任何口径都只训一次,两段式实现与回退删除。

动机与口径见 docs/MODELEVO_V2_16_0_VALIDATION_ES_20261022.md 末节。本文件锁定
(全部离线、无 CLI):

  A. 模板血缘(_fit_with_es / _fit_es_on_validation / _fit_with_eval):
     validation / train_tail / 无 y_va 三种口径都只 fit 一次;refit=True 不生效;
     单段异常直接抛出;旧定义删除;幂等;run 内嫁接回两段式模板段后再硬化;
  B. hurdle 全量重训血缘:两种口径只 fit 一次,v215 别名与回退删除,
     multi-seed 变体轮在 validation 守卫下不再因行数判据抛错;
  C. 库升级 / 复用路径接线;
  D. 交付 rerun_recipe 期间强制 train_tail(validation 已并入训练集);
  E. DL 5 折 OOF 特征块整列删除(段删除、注册表 / 块列数 / 槽位标签同步,
     形态不符不改,残留 K 折拟合可报告)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

xgb = pytest.importorskip("xgboost")

from test_v2_15_0_hpo import _PRELUDE, _body, _clf  # noqa: E402
from test_v2_16_0_validation_es import _OLD_LINEAGE  # noqa: E402


def _template_module() -> str:
    """模板前置段 + `_fit_with_es`…`_es_info_of` 段,拼成可独立 exec 的模块。"""
    src = _body()
    k0 = src.index("\n_MAX_TREES = ")
    k1 = src.index("\ndef load_config(", k0)
    i = src.index("def _fit_with_es")
    j = src.index("def _es_info_of", i)
    return (_PRELUDE + src[k0:k1] + "\n" + src[i:j]
            + '\n\nif __name__ == "__main__":\n    pass\n')


def _exec(code: str) -> dict:
    ns: dict = {"__builtins__": __builtins__, "__name__": "lib_recipe"}
    exec(code, ns)
    return ns


def _split(n=4000):
    X, y = _clf(n, 0)
    k = int(n * 0.75)
    return X.iloc[:k], y.iloc[:k], X.iloc[k:], y.iloc[k:]


def _xgb():
    return xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1,
                             max_depth=3, early_stopping_rounds=30,
                             random_state=0, n_jobs=2)


@pytest.fixture
def fits(monkeypatch):
    """记录每次 XGBClassifier.fit 的 (训练行数, 早停面行数)。"""
    calls: list = []
    orig = xgb.XGBClassifier.fit

    def fit(self, X, y, *a, **kw):
        es = kw.get("eval_set")
        calls.append((len(X), len(es[0][0]) if es else None))
        return orig(self, X, y, *a, **kw)

    monkeypatch.setattr(xgb.XGBClassifier, "fit", fit)
    return calls


# --------------------------------------------------------------------------
# A. 模板血缘
# --------------------------------------------------------------------------

def test_template_lineage_hardened_trains_once(fits, monkeypatch):
    from _mvo.render import (_residual_two_stage, _upgrade_es_source_code,
                             harden_library_es_code)

    raw = _template_module()
    assert "_fit_with_es" in _residual_two_stage(raw)     # 模板仍是两段式
    code, notes = harden_library_es_code(raw)
    assert notes and _residual_two_stage(code) == []
    assert "early_stopping_rounds=None" not in code
    assert harden_library_es_code(code) == (code, [])       # 幂等
    assert _upgrade_es_source_code(code) == (code, [])      # 不再追加段
    ns = _exec(code)
    Xtr, ytr, Xva, yva = _split()
    head = int(len(Xtr) * 0.9)

    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    info = ns["_fit_with_eval"](_xgb(), Xtr, ytr, Xva, yva).mvo_es_info_
    assert fits == [(len(Xtr), len(Xva))]
    assert info["es_source"] == "validation" and info["fit_mode"] == "single"
    assert "refit_trees" not in info

    fits.clear()
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "train_tail")
    info = ns["_fit_with_eval"](_xgb(), Xtr, ytr, Xva, yva).mvo_es_info_
    assert fits == [(head, len(Xtr) - head)]                # 无全量重训
    assert info["es_source"] == "train_tail" and "refit_trees" not in info
    assert info["cap"] == 2 * info["cap_initial"]           # cap 一次性外扩

    fits.clear()
    monkeypatch.delenv("MODELEVO_ES_SOURCE")
    ns["_fit_with_eval"](_xgb(), Xtr, ytr, None, None)      # smoke 等无 y_va
    assert fits == [(head, len(Xtr) - head)]

    fits.clear()
    info = ns["_fit_with_es"](_xgb(), Xtr, ytr, refit=True).mvo_es_info_
    assert fits == [(head, len(Xtr) - head)]                # refit 不生效
    assert info["cap"] == info["cap_initial"]               # inner 不外扩


def test_hardened_fit_raises_instead_of_fallback(monkeypatch):
    from _mvo.render import harden_library_es_code

    ns = _exec(harden_library_es_code(_template_module())[0])
    Xtr, ytr, Xva, yva = _split(800)

    def boom(self, *a, **kw):
        raise ValueError("boom")

    monkeypatch.setattr(xgb.XGBClassifier, "fit", boom)
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    with pytest.raises(ValueError, match="boom"):
        ns["_fit_with_eval"](_xgb(), Xtr, ytr, Xva, yva)
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "train_tail")
    with pytest.raises(ValueError, match="boom"):
        ns["_fit_with_eval"](_xgb(), Xtr, ytr, Xva, yva)


def test_old_lineage_dedup_and_regraft(fits, monkeypatch):
    """v2.16 前代码:先补单段段再硬化,被覆盖的旧定义删掉;run 内嫁接回
    两段式模板段(后定义覆盖)后再导出,仍被硬化回单段。"""
    from _mvo.render import (_residual_two_stage, _top_level_defs,
                             harden_library_es_code)

    code, notes = harden_library_es_code(_OLD_LINEAGE)
    compile(code, "hardened", "exec")
    defs = _top_level_defs(code)
    for name in ("_fit_with_es", "_fit_with_eval", "_fit_es_on_validation",
                 "_es_source"):
        assert len(defs[name]) == 1, name
    assert any("删除被覆盖的旧定义" in n for n in notes)
    # 模拟 --from-parent 嫁接:模板两段式段追加在 __main__ 之前
    src = _body()
    seg = src[src.index("def _fit_with_es"):src.index("def _es_info_of")]
    mi = code.rindex('\nif __name__ == "__main__"')
    regraft = code[:mi] + "\n\n" + seg + code[mi:]
    assert _residual_two_stage(regraft)
    again, notes2 = harden_library_es_code(regraft)
    assert notes2 and _residual_two_stage(again) == []
    defs2 = _top_level_defs(again)
    assert all(len(defs2[n]) == 1 for n in ("_fit_with_es", "_fit_with_eval",
                                            "_fit_es_on_validation"))
    assert harden_library_es_code(again) == (again, [])


# --------------------------------------------------------------------------
# B. hurdle 全量重训血缘
# --------------------------------------------------------------------------

_HURDLE_REFIT_CODE = '''
import numpy as np
import xgboost as xgb
MODEL_TYPE = "regression"
N_ESTIMATORS_CAP = 400
ES_FRACTION = 0.1
REFIT_FULL_TRAIN = True
REFIT_ROUND_SCALE = 1.0
CALLS = []


def step(m):
    pass


def _fit_seed_variant(n_jobs, X_fit, y_fit, eval_set):
    X_es, y_es = eval_set[0]
    if REFIT_FULL_TRAIN:
        if _FULL_TRAIN is None or len(_FULL_TRAIN[0]) != len(X_fit) + len(X_es):
            raise RuntimeError("全量重训变体轮缺少主轮登记的完整 train")
        return _fit_hurdle_refit_full(n_jobs, _FULL_TRAIN[0], _FULL_TRAIN[1])
    return _fit_hurdle(n_jobs, X_fit, y_fit, X_es, y_es)


class HurdleXGB:
    def __init__(self, clf, reg, stage1=None):
        self.clf_, self.reg_, self.stage1_ = clf, reg, stage1


def _stage1_rounds(m):
    return int(m.get_booster().num_boosted_rounds())


def _kw():
    return dict(n_estimators=N_ESTIMATORS_CAP, learning_rate=0.1, max_depth=3,
                early_stopping_rounds=20, n_jobs=2)


def _fit_hurdle(n_jobs, X_fit, y_fit, X_es, y_es):
    CALLS.append(("fit_hurdle", len(X_fit), len(X_es)))
    clf, reg = xgb.XGBClassifier(**_kw()), xgb.XGBRegressor(**_kw())
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
    clf, reg = xgb.XGBClassifier(**_kw()), xgb.XGBRegressor(**_kw())
    clf.set_params(n_estimators=s1.clf_.best_iteration + 1,
                   early_stopping_rounds=None)
    reg.set_params(n_estimators=s1.reg_.best_iteration + 1,
                   early_stopping_rounds=None)
    pos = np.asarray(y_tr) > 0
    clf.fit(X_tr, pos.astype(int), verbose=False)
    reg.fit(X_tr[pos], y_tr[pos], verbose=False)
    return HurdleXGB(clf, reg, stage1={"refit_reg_rows": int(pos.sum())})


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    global _FULL_TRAIN
    _FULL_TRAIN = (X_tr, y_tr)
    model = _fit_hurdle_refit_full(2, X_tr, y_tr)
    _st = dict(model.stage1_ or {})
    return model, {"hurdle": {
        "n_reg_fit_rows": int(_st.get("refit_reg_rows") or 0),
        "refit_full_train": bool(REFIT_FULL_TRAIN),
        "two_stage": _st,
    }}


_FULL_TRAIN = None


if __name__ == "__main__":
    pass
'''


def _reg_split():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 6)), columns=list("abcdef"))
    y = pd.Series(np.clip(X["a"] + 0.5 * X["b"] + rng.normal(size=3000),
                          0, None))
    return X.iloc[:2400], y.iloc[:2400], X.iloc[2400:], y.iloc[2400:]


def test_hurdle_refit_lineage_trains_once(monkeypatch):
    from _mvo.render import (_ES_HURDLE_BEGIN, _residual_two_stage,
                             harden_library_es_code)

    assert _residual_two_stage(_HURDLE_REFIT_CODE) == [
        "_fit_hurdle_refit_full"]
    code, notes = harden_library_es_code(_HURDLE_REFIT_CODE)
    assert code.count(_ES_HURDLE_BEGIN) == 1
    assert "_fit_hurdle_refit_full_v215" not in code
    assert code.count("\ndef _fit_hurdle_refit_full(") == 1
    assert _residual_two_stage(code) == []
    assert harden_library_es_code(code) == (code, [])
    Xtr, ytr, Xva, yva = _reg_split()

    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    ns = _exec(code)
    model, mx = ns["fit_and_score"](Xtr, ytr, Xva, yva)
    assert ns["CALLS"] == [("fit_hurdle", 2400, 600)]
    st = mx["hurdle"]["fit_stats"]                  # two_stage 键已改名
    assert "two_stage" not in mx["hurdle"]
    assert st["es_source"] == "validation" and st["fit_mode"] == "single"
    assert mx["hurdle"]["n_reg_fit_rows"] == st["n_reg_fit_rows"] > 0
    # multi-seed:v2.16 守卫下 main 的 X_fit=全量 train、eval_set=validation
    ns["CALLS"].clear()
    v = ns["_fit_seed_variant"](2, Xtr, ytr, [(Xva, yva)])
    assert ns["CALLS"] == [("fit_hurdle", 2400, 600)]
    assert v.stage1_["es_source"] == "validation"

    monkeypatch.setenv("MODELEVO_ES_SOURCE", "train_tail")
    ns2 = _exec(code)
    _, mx2 = ns2["fit_and_score"](Xtr, ytr, Xva, yva)
    assert ns2["CALLS"] == [("fit_hurdle", 2160, 240)]     # 无第二段重训
    assert mx2["hurdle"]["fit_stats"]["es_source"] == "train_tail"


# --------------------------------------------------------------------------
# C. 库升级 / 复用接线
# --------------------------------------------------------------------------

def test_apply_upgrade_harden_flag_and_reuse_wiring(tmp_path):
    from _mvo.render import _apply_es_source_upgrade, _residual_two_stage

    p = tmp_path / "train.py"
    p.write_text(_HURDLE_REFIT_CODE, encoding="utf-8")
    assert _apply_es_source_upgrade(p, harden=True)
    assert _residual_two_stage(p.read_text(encoding="utf-8")) == []
    assert _apply_es_source_upgrade(p, harden=True) == []
    st = (SKILL / "scripts" / "_mvo" / "cmd_startup.py").read_text(
        encoding="utf-8")
    body = st.split("def _apply_frozen_baseline(")[1].split("\ndef ")[0]
    assert "harden=True" in body                        # 固化 baseline 复用
    assert st.count("harden=True") >= 2                 # + warm-start


# --------------------------------------------------------------------------
# D. 交付
# --------------------------------------------------------------------------

def test_rerun_recipe_forces_train_tail_es(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from _mvo.cmd_deliver import _rerun_champion_recipe

    seen = {}
    champ = SimpleNamespace(TRIAL_DIR=tmp_path)

    def fit_and_score(X, y, X_va, y_va, seed, cfg):
        seen["es"] = os.environ.get("MODELEVO_ES_SOURCE")
        champ._last_model = xgb.XGBClassifier(n_estimators=3).fit(X, y)
        return np.zeros(len(X_va)), {}

    champ.fit_and_score = fit_and_score
    Xtr, ytr, Xva, yva = _split(400)
    monkeypatch.setenv("MODELEVO_ES_SOURCE", "validation")
    _rerun_champion_recipe(champ, Xtr, ytr, Xva, yva, 0, {}, tmp_path)
    assert seen["es"] == "train_tail"
    assert os.environ["MODELEVO_ES_SOURCE"] == "validation"   # 复原
    monkeypatch.delenv("MODELEVO_ES_SOURCE")
    _rerun_champion_recipe(champ, Xtr, ytr, Xva, yva, 0, {}, tmp_path)
    assert seen["es"] == "train_tail"
    assert "MODELEVO_ES_SOURCE" not in os.environ
    assert champ.TRIAL_DIR == tmp_path


# --------------------------------------------------------------------------
# E. DL 5 折 OOF 特征块整列删除
# --------------------------------------------------------------------------

_DL_RECIPE = '''
import numpy as np
import pandas as pd
CALLS = []


def _seqd_block(split):
    return pd.DataFrame({"seqd_a": [1.0, 2.0], "seqd_b": [3.0, 4.0]})


def _bbint_block(split):
    return pd.DataFrame({"bbint_a": [5.0, 6.0]})


# ================== gen-01 DL 段(深度模型公共件) ==================
import time as _dl_time
_DL_MEMO = {}


def _dl_oof(groups, n_folds=5):
    from sklearn.model_selection import GroupKFold
    CALLS.append("dl_oof")
    _ = GroupKFold(n_splits=n_folds)
    return np.zeros(len(groups))


class _DLData:
    pass
# ================== end gen-01 DL 段 ==================

# ------------------------------------------------------- 臂 B/D
_P1_PARTS = (("hier", "hier_logit", {}, 1),)


def _p1_logit_block(split):
    return pd.DataFrame({"hier_logit": _dl_oof([0, 1])})


# ------------------------------------------------------- T3:seqtab_logit
_STB_COL = "seqtab_logit"


def _widen(raw_tr, re_tr, raw_va, re_va):
    return None


def _seqtab_block(split):
    return pd.DataFrame({_STB_COL: _dl_oof([0, 1])})

# ------------------------------------------------------- gen-02 槽位配置
_G2_SLOT = "S2+seqtab+hier"
_G2_BLOCKS = ("seqd", "seqtab",
              "bbint", "p1")
_G2_N_BLOCK_COLS = 5


def trial_feature_blocks(split):
    fns = {"seqd": _seqd_block, "seqtab": _seqtab_block,
           "bbint": _bbint_block, "p1": _p1_logit_block}
    W = pd.concat([fns[k](split) for k in _G2_BLOCKS], axis=1)
    if W.shape[1] != _G2_N_BLOCK_COLS:
        raise ValueError("块列数 {} != 预期 {}".format(W.shape[1],
                                                 _G2_N_BLOCK_COLS))
    return W


if __name__ == "__main__":
    pass
'''


def test_strip_dl_oof_blocks_drops_columns_and_code():
    from _mvo.render import _residual_kfold_oof, harden_library_es_code

    before = _exec(_DL_RECIPE)
    assert before["trial_feature_blocks"]("validation").shape == (2, 5)
    assert before["CALLS"] == ["dl_oof", "dl_oof"]
    assert _residual_kfold_oof(_DL_RECIPE) == ["_dl_oof"]
    code, notes = harden_library_es_code(_DL_RECIPE)
    assert any("DL 5 折 OOF" in n and "'p1', 'seqtab'" in n for n in notes)
    for tok in ("_dl_", "_DL_", "_STB_", "_P1_", "_p1_", "GroupKFold",
                "def _widen(", "_seqtab_block", "class _DLData"):
        assert tok not in code, tok
    assert _residual_kfold_oof(code) == []
    ns = _exec(code)
    W = ns["trial_feature_blocks"]("validation")
    assert list(W.columns) == ["seqd_a", "seqd_b", "bbint_a"]
    assert ns["CALLS"] == []                              # 不再训 DL
    assert ns["_G2_BLOCKS"] == ("seqd", "bbint")
    assert ns["_G2_N_BLOCK_COLS"] == 3 and ns["_G2_SLOT"] == "S2"
    assert "\n\n\n\n" not in code                         # 接缝不留大段空行
    assert harden_library_es_code(code) == (code, [])     # 幂等


def test_strip_dl_oof_blocks_assign_style_registry():
    from _mvo.render import _strip_dl_oof_blocks

    code = (_DL_RECIPE
            .replace('fns = {"seqd": _seqd_block, "seqtab": _seqtab_block,\n'
                     '           "bbint": _bbint_block, "p1": _p1_logit_block}',
                     'fns = {"seqtab": _seqtab_block, "seqd": _seqd_block,\n'
                     '           "bbint": _bbint_block}\n'
                     '    fns["p1"] = _p1_logit_block')
            .replace('_G2_SLOT = "S2+seqtab+hier"', '_G2_SLOT = "S2+seqtab"'))
    assert 'fns["p1"] = _p1_logit_block' in code
    out, notes = _strip_dl_oof_blocks(code)
    assert notes
    ns = _exec(out)
    assert ns["trial_feature_blocks"]("x").shape == (2, 3)
    assert 'fns["p1"]' not in out and ns["_G2_SLOT"] == "S2"


@pytest.mark.parametrize("mutate", [
    # 段外(trial_feature_blocks 以外)还引用 DL 实现
    lambda c: c.replace("def trial_feature_blocks(split):\n",
                        "def warm():\n    return _DL_MEMO\n\n\n"
                        "def trial_feature_blocks(split):\n"),
    # P1 多臂:列数推不准
    lambda c: c.replace('_P1_PARTS = (("hier", "hier_logit", {}, 1),)',
                        '_P1_PARTS = (("hier", "hier_logit", {}, 1),'
                        ' ("daymat", "daymat_logit", {}, 2))'),
    # 未知块键
    lambda c: c.replace('"p1": _p1_logit_block}', '"p9": _p1_logit_block}')
              .replace('"bbint", "p1")', '"bbint", "p9")'),
])
def test_strip_dl_oof_blocks_leaves_unknown_shapes(mutate):
    from _mvo.render import _residual_kfold_oof, _strip_dl_oof_blocks

    code = mutate(_DL_RECIPE)
    assert code != _DL_RECIPE
    assert _strip_dl_oof_blocks(code) == (code, [])
    assert _residual_kfold_oof(code) == ["_dl_oof"]      # 留给报告


