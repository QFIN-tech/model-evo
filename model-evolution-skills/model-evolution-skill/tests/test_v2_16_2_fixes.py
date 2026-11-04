"""v2.16.2 回归测试:run4 regression(loan_amt_t7)D-003/D-004/D-006/D-008 修复。

口径见 docs/MODELEVO_V2_16_2_FIXES_20261023.md。全部离线、无 CLI 子进程
(创建期导入探针除外):

  A. D-008 hurdle HPO:父代识别与契约、超参段注入(inner 与重训都构造父代
     hurdle、分段键、lr 收缩、基准点、遥测)、config 超参消费(confirm /
     交付 / seed 变体同配方)、`_inject_hpo_config` 走 hurdle 路径、嫁接后
     结构检查与 lint 规则 23、成本闸(两段成本 / sample_rows / 并行)、
     边界外扩识别分段键、--set 消费检查、CLI 旋钮;
  B. D-004 leak_scan:回归 |Spearman|、分类口径不变、模板渲染、副本升级;
  C. D-003 group_importance:自存评估帧按抽样行位置取子集;
  D. D-006 回归 multi-seed 指标、证据对账、confirm 的 multi_seed 写实际值、
     FAILED 重入归档旧 result、wait 只认当前 attempt。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
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

import lint_trial as L  # noqa: E402
from _mvo import render as R  # noqa: E402
from runtime.logging_utils import dump_yaml, load_yaml  # noqa: E402

# ---------------------------------------------------------------------------
# 夹具:与 run4 regression trial-008 同契约的极简 hurdle 父代
# ---------------------------------------------------------------------------

_OLD_LEAK_SCAN = '''def leak_scan(model, X_tr: pd.DataFrame, y_tr, top_k: int = 50) -> dict:
    """v13 训练后哨兵(旧版:只有单变量 AUC)。"""
    out = {"n_scanned": 0, "suspects": []}
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            return out
        order = np.argsort(-np.asarray(imp))[:top_k]
        cols = list(X_tr.columns)
        y_arr = np.asarray(y_tr).astype(int)
        from sklearn.metrics import roc_auc_score
        suspects = []
        for i in order:
            c = cols[i]
            s = pd.to_numeric(X_tr[c]).fillna(0).values
            if np.unique(s).size < 2:
                continue
            a = roc_auc_score(y_arr, s)
            a = max(a, 1 - a)
            if a > 0.9:
                suspects.append({"feature": c, "train_auc": round(a, 4)})
        out = {"n_scanned": int(len(order)), "suspects": suspects}
    except Exception as _e:  # noqa: BLE001
        out["error"] = str(_e)[:120]
    return out
'''

HURDLE_PARENT = '''import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent
RUN_DIR = TRIAL_DIR.parent.parent
SKILL_ROOT = __SKILL__
if SKILL_ROOT and SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)
TARGET = "y"
MODEL_TYPE = "regression"
_ALGORITHM = "xgboost"
SEED = 7
HURDLE_LR = 0.1
HURDLE_STRUCT = dict(max_depth=3, min_child_weight=2.0, subsample=0.9,
                     reg_lambda=1.0)
N_ESTIMATORS_CAP = 120
EARLY_STOP_ROUNDS = 20
ES_FRACTION = 0.1


def step(msg):
    print("[step] " + str(msg), flush=True)


def n_jobs():
    return 2


def _gpu_allocated(n_rows):
    return False


def load_config():
    p = TRIAL_DIR / "config" / "trial.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) \\
        if p.exists() else {}


__LEAK__

def _hurdle_estimators(n_jobs, n_rows_clf, n_rows_reg):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS_CAP, learning_rate=HURDLE_LR,
        random_state=SEED, n_jobs=n_jobs, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=EARLY_STOP_ROUNDS,
        **HURDLE_STRUCT)
    reg = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS_CAP, learning_rate=HURDLE_LR,
        random_state=SEED, n_jobs=n_jobs, tree_method="hist",
        early_stopping_rounds=EARLY_STOP_ROUNDS, **HURDLE_STRUCT)
    return clf, reg


class HurdleXGB:
    def __init__(self, clf, reg):
        self.clf_ = clf
        self.reg_ = reg

    def positive_prob(self, X):
        _proba = self.clf_.predict_proba(X)
        return _proba[:, 1]

    def predict(self, X):
        return self.positive_prob(X) * np.clip(self.reg_.predict(X), 0.0, None)

    @property
    def feature_importances_(self):
        return 0.5 * (np.asarray(self.clf_.feature_importances_)
                      + np.asarray(self.reg_.feature_importances_))


FIT_LOG = []


def _fit_hurdle(n_jobs, X_fit, y_fit, X_es, y_es):
    FIT_LOG.append({"n_fit": len(X_fit), "fit_max": int(X_fit.index.max()),
                    "es_min": int(X_es.index.min()),
                    "es_max": int(X_es.index.max())})
    y_fit = np.asarray(y_fit, dtype="float64")
    y_es = np.asarray(y_es, dtype="float64")
    pf, pe = y_fit > 0, y_es > 0
    clf, reg = _hurdle_estimators(n_jobs, len(X_fit), int(pf.sum()))
    clf.fit(X_fit, pf.astype("int8"), eval_set=[(X_es, pe.astype("int8"))],
            verbose=False)
    reg.fit(X_fit[pf], y_fit[pf], eval_set=[(X_es[pe], y_es[pe])],
            verbose=False)
    return HurdleXGB(clf, reg)


_last_model = None


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    global _last_model
    model = _fit_hurdle(n_jobs(), X_tr, y_tr, X_va, y_va)
    _last_model = model
    return model.predict(X_va), {"model_num": 2,
                                 "hurdle": {"cap": int(N_ESTIMATORS_CAP)}}


def main():
    cfg = load_config()
    score, _m = fit_and_score(None, None, None, None, SEED, cfg)
    metrics = {"leak_scan": leak_scan(_last_model, None, None)}
    pd.DataFrame({"score": score}).to_parquet(
        TRIAL_DIR / "validation_score.parquet", index=False)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
'''


def _parent_code() -> str:
    return (HURDLE_PARENT.replace("__SKILL__", repr(str(SKILL)))
            .replace("__LEAK__", _OLD_LEAK_SCAN))


def _mk_trial_dir(run: Path, tid: str, code: str, cfg: dict | None = None
                  ) -> Path:
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "code" / "train.py").write_text(code, encoding="utf-8")
    dump_yaml(d / "config" / "trial.yaml", cfg or {"hyperparameters": {}})
    return d


def _mk_run(tmp: Path) -> Path:
    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    (run / "task-spec.yaml").write_text(
        "model_type: regression\nprimary_metric: rmse\n", encoding="utf-8")
    (run / "modelevo-config.yaml").write_text(
        "modelevo:\n  execution:\n    trial_timeout_minutes: 90\n"
        "  evolution:\n    hpo:\n      inner_fraction: 0.2\n",
        encoding="utf-8")
    return run


_LOADED = 0


def _load(path: Path):
    global _LOADED
    _LOADED += 1
    spec = importlib.util.spec_from_file_location(f"mvo_v2162_{_LOADED}",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data(n_tr=4000, n_va=1500, seed=0):
    rng = np.random.default_rng(seed)

    def mk(n, off):
        X = pd.DataFrame(rng.normal(size=(n, 6)).astype("float32"),
                         columns=[f"f{i}" for i in range(6)],
                         index=np.arange(off, off + n))
        p = 1.0 / (1.0 + np.exp(-(1.5 * X.f0 - 1.5)))
        pos = rng.random(n) < p
        y = np.where(pos, np.clip(3 + X.f1 + rng.normal(scale=.4, size=n),
                                  .1, None), 0.0)
        return X, pd.Series(y, index=X.index)

    X_tr, y_tr = mk(n_tr, 0)
    X_va, y_va = mk(n_va, 1_000_000)   # outer validation 的行索引与 train 不相交
    return X_tr, y_tr, X_va, y_va


class _FakeTrial:
    def __init__(self, d: Path, parents=("trial-008-g02-confirm-xgboost-1",),
                 algo="xgboost"):
        self.trial_dir = str(d)
        self.trial_id = d.name
        self.data = {"algorithm": algo, "parent_trial_ids": list(parents),
                     "execution": {}, "expected_resources": {}}

    def flush(self):
        pass


_SPACE = {"max_depth": {"type": "int", "low": 2, "high": 6},
          "clf__min_child_weight": {"type": "float", "low": .5, "high": 20.0,
                                    "log": True},
          "reg__reg_lambda": {"type": "float", "low": .001, "high": 20.0,
                              "log": True},
          "learning_rate": {"type": "float", "low": .01, "high": .3,
                            "log": True}}


# ===========================================================================
# A. D-008 hurdle HPO
# ===========================================================================

def test_hurdle_lineage_status_and_contract():
    st = R._hurdle_lineage_status(_parent_code())
    assert st["hurdle"] and st["entries_ok"], st
    assert st["source"] == "topology:structural:hurdle"
    broken = _parent_code().replace("def _hurdle_estimators(",
                                    "def _hurdle_estimators_x(")
    st2 = R._hurdle_lineage_status(broken)
    assert st2["hurdle"] and not st2["entries_ok"]
    assert "_hurdle_estimators" in st2["missing"]
    msg = R._hurdle_reject_message("trial-008", st2, "hurdle HPO 拒绝创建")
    assert "_hurdle_estimators" in msg and "--set clf__max_depth" in msg
    # 普通单模型代码不是 hurdle
    assert not R._hurdle_lineage_status(
        "def fit_and_score(X, y, Xv, yv=None, seed=0, cfg=None):\n"
        "    return None, {}\n")["hurdle"]


def test_hurdle_hp_split_semantics(tmp_path):
    run = _mk_run(tmp_path)
    d = _mk_trial_dir(run, "trial-010-g03-hpo-xgboost-1", _parent_code())
    R._inject_hurdle_hp_segment(d / "code" / "train.py")
    m = _load(d / "code" / "train.py")
    clf, reg = m._hurdle_hp_split({
        "max_depth": 5, "clf__max_depth": 7, "reg__reg_lambda": 3.0,
        "hurdle_lr": 0.0125, "hurdle_cap": 99999})
    assert clf["max_depth"] == 7 and reg["max_depth"] == 5   # 分段键优先
    assert "reg_lambda" not in clf and reg["reg_lambda"] == 3.0
    assert clf["learning_rate"] == reg["learning_rate"] == 0.0125   # 别名
    assert clf["n_estimators"] == reg["n_estimators"] == 8000     # 树数铁律
    assert clf["early_stopping_rounds"] == 50                     # max(50,40)
    assert m._hurdle_hp_split({"learning_rate": 0.002})[0][
        "early_stopping_rounds"] == 250                            # ceil(.5/lr)
    with pytest.raises(ValueError):
        m._hurdle_hp_split({"random_state": 1})
    # 注入幂等:二次注入整段替换,不叠加
    R._inject_hurdle_hp_segment(d / "code" / "train.py")
    code = (d / "code" / "train.py").read_text(encoding="utf-8")
    assert code.count("v2.16.2 hurdle 超参段 BEGIN") == 1
    compile(code, "train.py", "exec")


def test_hurdle_hpo_inner_and_refit_construct_parent_hurdle(tmp_path):
    run = _mk_run(tmp_path)
    d = _mk_trial_dir(run, "trial-010-g03-hpo-xgboost-1", _parent_code())
    R._inject_hurdle_hp_segment(d / "code" / "train.py")
    m = _load(d / "code" / "train.py")
    X_tr, y_tr, X_va, y_va = _data()
    cfg = {"hyperparameters": {}, "seed": 7, "search_action": "hpo",
           "hpo": {"search_space": _SPACE, "budget_configs": 5,
                   "inner_fraction": 0.2, "sample_rows": 2000,
                   "lr_mode": "shrink", "search_lr": 0.2,
                   "lr_shrink_factor": 0.5,
                   # v2.18.4:并行 2 时 config 完成序不定 → TPE 提案序不定 →
                   # 选中的配方不定 → 重训 cap 随之在 240/276/279 间漂移,
                   # 全量跑(机器负载高)时确定性断言会随机失败。本用例断言
                   # 的是"重训 cap = 父代轮数 × 1/factor"这条口径,不是并行
                   # 度,改为串行以获得可复现的搜索轨迹。
                   "parallel_configs": 1,
                   "topology": "structural:hurdle", "enqueue": []}}
    score, mx = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, cfg)
    assert len(score) == len(X_va)
    # inner 搜索只在 train 内切(拟合池头部拟合、尾部早停,inner 验证面另评);
    # 只有最后一次(重训)以 outer validation 为早停面
    inner, refit = m.FIT_LOG[:-1], m.FIT_LOG[-1]
    assert len(inner) == 5
    assert all(r["es_max"] < len(X_tr) for r in inner)
    assert all(r["n_fit"] == int(2000 * 0.9) for r in inner)   # sample_rows
    assert refit["es_min"] >= 1_000_000 and refit["n_fit"] == len(X_tr)
    st = json.loads((d / "hpo-study.json").read_text(encoding="utf-8"))
    assert st["topology"] == "structural:hurdle"
    assert st["inner_metric_name"] == "rmse" and st["lr_mode"] == "shrink"
    # 基准点 = 父代配方(从父代 _hurdle_estimators 读回);并行 2 时完成序不定
    enq = [t for t in st["trials"] if t["origin"] == "enqueued"]
    assert len(enq) == 1, [t["origin"] for t in st["trials"]]
    t0 = enq[0]
    assert t0["params"] == {"max_depth": 3, "clf__min_child_weight": 2.0,
                            "reg__reg_lambda": 1.0}
    for k in ("clf_best_iteration", "reg_best_iteration", "clf_cap_bound",
              "reg_cap_bound", "cap_bound"):
        assert k in t0, k                      # 两段各自的遥测
    # lr 收缩:inner 固定 search_lr,重训 lr = search_lr × factor、cap × 1/factor
    bp = mx["best_params"]
    assert bp["learning_rate"] == pytest.approx(0.1)
    assert bp["n_estimators"] == 240
    assert st["best_params"] == bp and st["refit"]["refit_cap"] == 240
    mdl = m._last_model
    pc, pr = mdl.clf_.get_params(), mdl.reg_.get_params()
    assert pc["learning_rate"] == pr["learning_rate"] == pytest.approx(0.1)
    assert pc["max_depth"] == pr["max_depth"] == bp["max_depth"]
    assert pc["min_child_weight"] == pytest.approx(bp["clf__min_child_weight"])
    assert pr["min_child_weight"] == 2.0       # 分类段键不作用于回归段
    assert pr["reg_lambda"] == pytest.approx(bp["reg__reg_lambda"])
    assert pc["reg_lambda"] == 1.0
    assert m.HURDLE_LR == pytest.approx(0.1) and m.N_ESTIMATORS_CAP == 240
    for k in ("hpo_n_configs", "hpo_inner_signature", "hpo_refit_es_info",
              "hpo_search_best_params", "hurdle_hp", "hurdle"):
        assert k in mx, k
    assert set(mx["hurdle_hp"]["effective"]) == {"clf", "reg"}


def test_hurdle_hp_consumer_confirm_delivery_and_seed_variant(tmp_path):
    run = _mk_run(tmp_path)
    d = _mk_trial_dir(run, "trial-011-g04-confirm-xgboost-1", _parent_code())
    R._inject_hurdle_hp_segment(d / "code" / "train.py")
    m = _load(d / "code" / "train.py")
    base = _load(_mk_trial_dir(run, "p", _parent_code()) / "code" / "train.py")
    X_tr, y_tr, X_va, y_va = _data(seed=3)
    # 无 hpo、无 hyperparameters:与父代逐字等价
    s0, _ = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, {"hyperparameters": {}})
    sb, _ = base.fit_and_score(X_tr, y_tr, X_va, y_va, 7, {})
    assert np.array_equal(s0, sb)
    hp = {"max_depth": 4, "clf__min_child_weight": 5.0, "learning_rate": 0.05,
          "n_estimators": 300}
    s1, mx = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7,
                             {"hyperparameters": hp})
    assert mx["hyperparameters_injected"] == hp
    pc = m._last_model.clf_.get_params()
    assert (pc["max_depth"], pc["min_child_weight"], pc["learning_rate"],
            pc["n_estimators"]) == (4, 5.0, 0.05, 300)
    assert not np.array_equal(s1, s0)
    # 同配方复现(confirm 快通道 / 交付复跑都是同一入口)
    s2, _ = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, {"hyperparameters": hp})
    assert np.array_equal(s1, s2)
    # multi-seed 变体轮(父代血缘直接调 _fit_hurdle):配方沿用,只换 seed
    m.SEED = 1007
    mv = m._fit_hurdle(2, X_tr, y_tr, X_va, y_va)
    pv = mv.clf_.get_params()
    assert (pv["max_depth"], pv["learning_rate"], pv["random_state"]) == \
        (4, 0.05, 1007)
    # 未知键报错,不静默空转
    with pytest.raises(ValueError):
        m.fit_and_score(X_tr, y_tr, X_va, y_va, 7,
                        {"hyperparameters": {"clf__no_such_param": 1}})


def test_inject_hpo_config_takes_hurdle_path(tmp_path):
    run = _mk_run(tmp_path)
    d = _mk_trial_dir(run, "trial-012-g03-hpo-xgboost-1", _parent_code())
    sp = tmp_path / "space.yaml"
    dump_yaml(sp, _SPACE)
    t = _FakeTrial(d)
    R._inject_hpo_config(run, t, str(sp), 40, sample_rows=400000,
                         hpo_opts={"lr_mode": "shrink",
                                   "lr_shrink_factor": 0.5})
    code = (d / "code" / "train.py").read_text(encoding="utf-8")
    assert "v2.16.2 hurdle 超参段 BEGIN" in code
    assert "v19 hpo 嫁接段 BEGIN" not in code          # 不嫁接模板 hpo 段
    assert not R._has_module_level_algo(code)          # 不注入 _HPO_ALGO
    cfg = load_yaml(d / "config" / "trial.yaml")
    assert cfg["hpo"]["topology"] == "structural:hurdle"
    assert cfg["hpo"]["enqueue"] == []                 # 基准点运行期读回
    assert cfg["hpo"]["sample_rows"] == 400000
    assert cfg["hpo"]["lr_shrink_factor"] == 0.5
    assert t.data["execution"]["hpo_topology"] == "structural:hurdle"
    # lint:规则 4 / 7 / 11 / 13 / 14 / 19 / 23 全过
    errs = (L.check_regression_score_semantics(run, code)
            + L.check_hpo_selection_leak(run, d, code)
            + L.check_hyperparameters_consumption(run, d, code)
            + L.check_undefined_names(code)
            + L.check_generated_code_consistency(code)
            + L.check_function_scope_shadowing(code)
            + L.check_structural_override(code))
    assert errs == [], errs


def test_inject_hpo_config_rejects_hurdle_without_entries(tmp_path):
    run = _mk_run(tmp_path)
    broken = _parent_code().replace("def _fit_hurdle(", "def _fit_hurdle2(")
    d = _mk_trial_dir(run, "trial-013-g03-hpo-xgboost-1", broken)
    sp = tmp_path / "space.yaml"
    dump_yaml(sp, _SPACE)
    with pytest.raises(SystemExit) as ei:
        R._inject_hpo_config(run, _FakeTrial(d), str(sp), 40)
    assert "_fit_hurdle" in str(ei.value) and "替代路径" in str(ei.value)


def test_generic_graft_structural_guard(tmp_path, monkeypatch):
    """hurdle 识别失效时,模板整段嫁接覆盖父代入口 → 创建期拒绝(旧:WARNING)。"""
    run = _mk_run(tmp_path)
    d = _mk_trial_dir(run, "trial-014-g03-hpo-xgboost-1", _parent_code())
    sp = tmp_path / "space.yaml"
    dump_yaml(sp, _SPACE)
    monkeypatch.setattr(R, "_hurdle_lineage_status",
                        lambda code: {"hurdle": False, "entries_ok": False,
                                      "missing": [], "source": None,
                                      "main_calls_fit_and_score": True})
    with pytest.raises(SystemExit) as ei:
        R._inject_hpo_config(run, _FakeTrial(d), str(sp), 40)
    assert "覆盖了父代 structural 训练入口" in str(ei.value)


def test_lint_rule23_structural_override():
    code = _parent_code()
    assert L.check_structural_override(code) == []
    graft = code.replace('\nif __name__ == "__main__":', '''
def _build_model(params, seed, n=0):
    import xgboost as xgb
    return xgb.XGBRegressor(**params)


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    global _last_model
    m = _build_model({}, seed)
    m.fit(X_tr, y_tr)
    _last_model = m
    return m.predict(X_va), {}


if __name__ == "__main__":''')
    f = L.structural_override_findings(graft)
    assert f and "_fit_hurdle" in f[0]["entries"] and "HurdleXGB" in \
        f[0]["entries"]
    errs = L.check_structural_override(graft)
    assert errs and errs[0].startswith("规则 23 违反")
    assert L.check_structural_override(
        graft + "\n# lint:allow-structural-override\n") == []
    # 包装委托(v2.16 段 / hurdle 超参段同款)不误报
    wrap = code.replace('\nif __name__ == "__main__":', '''
_fit_and_score_v215 = fit_and_score


def fit_and_score(X_tr, y_tr, X_va, y_va=None, *args, **kwargs):
    return _fit_and_score_v215(X_tr, y_tr, X_va, y_va, *args, **kwargs)


if __name__ == "__main__":''')
    assert L.structural_override_findings(wrap) == []
    assert "check_structural_override(code)" in (
        SKILL / "assets" / "lint_trial.py").read_text(encoding="utf-8")


def _cost_fixture(run: Path, *, topology: str | None, sample_rows=400000,
                  parallel="auto"):
    p = run / "trials" / "p1"
    p.mkdir(parents=True, exist_ok=True)
    dump_yaml(p / "trial.yaml", {
        "trial_id": "p1", "environment": {"gpu_index": 3},
        "result": {"runtime_sec": 5303.37, "evaluate_sec": 2243.7}})
    (p / "metrics.json").write_text(json.dumps(
        {"multi_seed_n": 3, "n_train": 2538086}), encoding="utf-8")
    (p / "stdout.log").write_text(
        "[step] load_data start: cols=1493 (5.7s)\n"
        "[step] hurdle clf fit start: rows=2538086 pos_rate=0.1240 (202.6s)\n"
        "[step] hurdle clf fit done: best_iter=6144 (1079.3s)\n"
        "[step] hurdle reg fit start: rows=314763 es_rows=64126 (1079.3s)\n"
        "[step] hurdle reg fit done: best_iter=4151 (1226.2s)\n"
        "[step] hurdle clf fit start: rows=2538086 pos_rate=0.1240 (1290.5s)\n",
        encoding="utf-8")
    c = run / "trials" / "child"
    (c / "config").mkdir(parents=True, exist_ok=True)
    hpo = {"budget_configs": 40, "parallel_configs": parallel}
    if topology:
        hpo["topology"] = topology
    if sample_rows:
        hpo["sample_rows"] = sample_rows
    dump_yaml(c / "config" / "trial.yaml", {"hpo": hpo})
    t = _FakeTrial(c, parents=("p1",))
    t.data["expected_resources"] = {"timeout_multiplier": 4}
    return t


def test_cost_gate_hurdle_two_segments_sample_rows_parallel(tmp_path,
                                                             monkeypatch):
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    run = _mk_run(tmp_path)
    t = _cost_fixture(run, topology="structural:hurdle")
    est = R._hpo_cost_estimate(t, run, 40)
    # D-008 实测:旧口径 796 min 被拒;新口径单次拟合 = (5303−2244)/3 s,
    # 两段实测 877+147 s × inner 行数比 400000/2538086,GPU xgboost 并行 2
    assert est["single_fit_sec"] == pytest.approx(1019.9, abs=0.1)
    assert est["segments_sec_parent"] == {"clf": 876.7, "reg": 146.9}
    assert est["inner_rows_ratio"] == pytest.approx(0.1576, abs=1e-4)
    assert est["parallel"] == 2
    assert est["total_min"] == pytest.approx(70.8, abs=0.5)
    assert est["fits"] is True
    R._hpo_cost_gate(t, run, 40)                      # 不拒绝
    assert t.data["expected_resources"]["hpo_cost_estimate"]["fits"]
    # 串行 + 不抽样:两段全量 inner 成本 → 拒绝,并给出降级选项
    t2 = _cost_fixture(run, topology="structural:hurdle", sample_rows=None,
                       parallel=1)
    with pytest.raises(SystemExit) as ei:
        R._hpo_cost_gate(t2, run, 40)
    assert "hpo 成本闸" in str(ei.value) and "--hpo-sample-rows" in str(ei.value)
    t2.data["expected_resources"]["hpo_force_budget"] = True
    R._hpo_cost_gate(t2, run, 40)                     # 豁免放行且留痕
    assert t2.data["expected_resources"]["hpo_cost_estimate"]["fits"] is False


def test_cost_gate_single_model_reads_sample_rows_from_config(tmp_path,
                                                              monkeypatch):
    """旧实现读 trial.data["config"](不存在)→ --hpo-sample-rows 从未生效。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    run = _mk_run(tmp_path)
    est = R._hpo_cost_estimate(_cost_fixture(run, topology=None,
                                             parallel=1), run, 40)
    assert est["topology"] == "single"
    assert est["per_config_sec"] == pytest.approx(
        1019.9 * 0.2 * 400000 / 2538086, abs=0.06)
    est2 = R._hpo_cost_estimate(_cost_fixture(run, topology=None,
                                              sample_rows=None, parallel=1),
                                run, 40)
    assert est2["per_config_sec"] == pytest.approx(1019.9 * 0.2, abs=0.06)


def test_expand_space_handles_segment_prefix():
    from runtime.hpo import expand_space, param_base_name

    assert param_base_name("clf__max_depth") == "max_depth"
    assert param_base_name("reg__reg_lambda") == "reg_lambda"
    assert param_base_name("max_depth") == "max_depth"
    space = {"clf__max_depth": {"type": "int", "low": 3, "high": 16},
             "reg__subsample": {"type": "float", "low": 0.6, "high": 1.0},
             "clf__unknown": {"type": "float", "low": 0.0, "high": 1.0}}
    hits = {"clf__max_depth": {"side": "low"},
            "reg__subsample": {"side": "low"},
            "clf__unknown": {"side": "high"}}
    new, ch = expand_space(space, hits, [])
    got = {c["param"]: c["to"] for c in ch}
    assert got["clf__max_depth"] == 1 or got["clf__max_depth"] < 3
    assert got["reg__subsample"] == pytest.approx(0.5)
    assert "clf__unknown" not in got                    # 语义未知不外扩
    assert new["clf__max_depth"]["high"] == 16          # 已顶硬界侧不动


def test_set_consumption_injects_hurdle_consumer_or_warns(tmp_path):
    run = _mk_run(tmp_path)
    d = _mk_trial_dir(run, "trial-015-g04-exploit-xgboost-1", _parent_code(),
                      {"hyperparameters": {"clf__max_depth": 8}})
    t = _FakeTrial(d)
    notes = R._ensure_hyperparameters_consumed(run, t, "trial-008", "exploit")
    assert notes and "v2.16.2 hurdle 超参段 BEGIN" in (
        d / "code" / "train.py").read_text(encoding="utf-8")
    assert L.check_hyperparameters_consumption(
        run, d, (d / "code" / "train.py").read_text(encoding="utf-8")) == []
    # 非 hurdle、无消费点:v2.17.0(D2)起缺省创建期拒绝;--allow-code-edit
    # 时明确提示并留痕(run 前规则 11 仍拒跑)
    d2 = _mk_trial_dir(run, "trial-016-g04-exploit-xgboost-1",
                       "SEED = 1\n", {"hyperparameters": {"lr": 0.03}})
    t2 = _FakeTrial(d2)
    with pytest.raises(SystemExit):
        R._ensure_hyperparameters_consumed(run, t2, "x", "exploit")
    assert R._ensure_hyperparameters_consumed(
        run, t2, "x", "exploit", allow_code_edit=True) == []
    assert t2.data["execution"]["hyperparameters_unconsumed"] == ["lr"]
    # hpo 动作不在此检查;已有消费点不动
    assert R._ensure_hyperparameters_consumed(run, t2, "x", "hpo") == []


def test_hpo_cli_lr_knobs():
    from _mvo.cmd_trial import _hpo_opts_from_args

    o = _hpo_opts_from_args(types.SimpleNamespace(
        hpo_lr_mode="shrink", hpo_parallel="2", hpo_fidelity=None,
        hpo_no_expand=False, hpo_no_warm_start=False, hpo_search_lr=0.05,
        hpo_lr_shrink_factor=0.5))
    assert o == {"lr_mode": "shrink", "parallel_configs": 2,
                 "search_lr": 0.05, "lr_shrink_factor": 0.5}
    src = (SKILL / "scripts" / "modelevo.py").read_text(encoding="utf-8")
    assert "--hpo-search-lr" in src and "--hpo-lr-shrink-factor" in src


def test_hurdle_lr_cap_advice(tmp_path):
    run = _mk_run(tmp_path)
    p = run / "trials" / "p1"
    p.mkdir(parents=True)
    (p / "metrics.json").write_text(json.dumps(
        {"hurdle": {"clf_best_iteration": 6145, "reg_best_iteration": 4152}}),
        encoding="utf-8")
    t = _FakeTrial(run / "trials" / "c", parents=("p1",))
    code = "HURDLE_LR = 0.025\n"
    adv = R._hurdle_lr_cap_advice(run, t, code, {
        "lr_mode": "shrink", "search_space": {"learning_rate": {}},
        "search_lr": 0.05, "lr_shrink_factor": 0.25})
    assert adv and "--hpo-lr-shrink-factor 0.5" in adv
    assert R._hurdle_lr_cap_advice(run, t, code, {
        "lr_mode": "shrink", "search_space": {"learning_rate": {}},
        "search_lr": 0.05, "lr_shrink_factor": 0.5}) is None


# ===========================================================================
# B. D-004 leak_scan
# ===========================================================================

class _Imp:
    def __init__(self, n):
        self.feature_importances_ = np.linspace(1.0, 0.1, n)


def _leak_ns(model_type: str) -> dict:
    from runtime.templates import LEAK_SCAN_SOURCE

    ns = {"np": np, "pd": pd, "MODEL_TYPE": model_type}
    exec(LEAK_SCAN_SOURCE, ns)  # noqa: S102
    return ns


def test_leak_scan_regression_uses_abs_spearman():
    rng = np.random.default_rng(0)
    n = 5000
    y = np.where(rng.random(n) < 0.15, rng.gamma(3, 1, n), 0.0)
    X = pd.DataFrame({"leak": (np.log1p(y) * 3 - 1).astype("float32"),
                      "noise": rng.normal(size=n).astype("float32"),
                      "weak": (y + rng.normal(scale=3, size=n)).astype(
                          "float32")})
    out = _leak_ns("regression")["leak_scan"](_Imp(3), X, pd.Series(y))
    assert "error" not in out and out["n_scanned"] == 3
    assert out["statistic"] == "abs_spearman" and out["threshold"] == 0.95
    assert [s["feature"] for s in out["suspects"]] == ["leak"]
    assert out["suspects"][0]["train_abs_spearman"] == pytest.approx(1.0)
    # 未声明 MODEL_TYPE 时按目标取值数判回归
    out2 = _leak_ns("")["leak_scan"](_Imp(3), X, pd.Series(y))
    assert out2.get("statistic") == "abs_spearman"


def test_leak_scan_classification_unchanged():
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.3).astype(int)
    X = pd.DataFrame({"leak": (y * 5 + rng.normal(scale=.1, size=n)),
                      "noise": rng.normal(size=n)})
    new = _leak_ns("binary_classification")["leak_scan"](_Imp(2), X, y)
    ns_old = {"np": np, "pd": pd}
    exec(_OLD_LEAK_SCAN, ns_old)  # noqa: S102
    old = ns_old["leak_scan"](_Imp(2), X, y)
    assert new == old and new["suspects"][0]["feature"] == "leak"
    assert set(new) == {"n_scanned", "suspects"}       # 分类输出键不变


def test_templates_render_single_source_leak_scan():
    from runtime.baseline import BASELINE_TEMPLATE, ESTIMATOR_CODE, FIT_CODE
    from runtime.code_audit import check_generated_code
    from runtime.templates import LEAK_SCAN_SOURCE, TRIAL_TEMPLATE

    code = BASELINE_TEMPLATE.format(
        algorithm="xgboost", target="y", model_type="regression",
        treatment="", seed=0, n_estimators_cap=2000,
        early_stopping_rounds=50, es_fraction=0.1,
        estimator_code=ESTIMATOR_CODE["xgboost"],
        fit_code=FIT_CODE["xgboost"])
    assert LEAK_SCAN_SOURCE in code
    assert check_generated_code(code) == []
    assert "abs_spearman" in TRIAL_TEMPLATE
    assert "{{" not in LEAK_SCAN_SOURCE


def test_upgrade_leak_scan_in_copied_code(tmp_path):
    from runtime.templates import LEAK_SCAN_SOURCE

    code = _parent_code()
    out, notes = R._upgrade_leak_scan_code(code)
    assert notes and LEAK_SCAN_SOURCE in out and _OLD_LEAK_SCAN not in out
    compile(out, "x", "exec")
    assert R._upgrade_leak_scan_code(out) == (out, [])   # 幂等
    assert R._upgrade_leak_scan_code("x = 1\n") == ("x = 1\n", [])
    # 副本升级入口(--from-parent / warm-start / 固化 baseline)接线
    p = tmp_path / "train.py"
    p.write_text(code, encoding="utf-8")
    notes2 = R._apply_es_source_upgrade(p)
    assert any("leak_scan" in n for n in notes2)
    assert "abs_spearman" in p.read_text(encoding="utf-8")


# ===========================================================================
# C. D-003 group_importance 子样本对齐
# ===========================================================================

def _gi_trial(tmp: Path, n=3000):
    import joblib
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    base = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    wide = rng.normal(size=n)
    y = 3.0 * wide + 0.2 * base["a"].to_numpy() + rng.normal(scale=.1, size=n)
    full = base.assign(wide_x=wide)
    d = tmp / "trial"
    (d / "model").mkdir(parents=True)
    m = LinearRegression().fit(full, y)
    joblib.dump(m, d / "model" / "model.joblib")
    (d / "model" / "feature_columns.json").write_text(
        json.dumps(list(full.columns)), encoding="utf-8")
    full.to_parquet(d / "model" / "validation_input.parquet", index=False)
    return d, base, y


def test_gi_subsets_saved_frame_by_row_positions(tmp_path):
    from runtime.group_importance import run_group_importance

    d, base, y = _gi_trial(tmp_path)
    keep = np.sort(np.random.default_rng(1).choice(len(y), 1200,
                                                   replace=False))
    out = run_group_importance(d, base.iloc[keep], y[keep], "rmse",
                               n_repeat=3, row_positions=keep,
                               n_full_rows=len(y))
    assert out["eval_frame_source"] == "trial_saved_validation_input"
    assert out["eval_frame_subset"] == {"rows": 1200, "of_rows": len(y),
                                        "by": "row_positions"}
    assert out["filled_constant_columns"] == 0
    assert "wide" in out["families_alive"]
    assert out["families_unverifiable"] == []
    top = {r["family"]: r for r in out["top"]}
    assert top["wide"]["delta_mean"] > 1.0              # 行对齐:打乱掉分显著
    # 旧行为(不传行位置):行数不符 → 回退基础帧,加宽族 unverifiable
    d2, base2, y2 = _gi_trial(tmp_path / "old")
    out2 = run_group_importance(d2, base2.iloc[keep], y2[keep], "rmse",
                                n_repeat=3)
    assert out2["eval_frame_source"] == "splits_base"
    assert "wide" in out2["families_unverifiable"]


def test_runner_passes_gi_sample_positions(monkeypatch):
    import runtime.group_importance as gi
    from runtime.runner import TrialRunner

    seen = {}

    def fake(trial_dir, X_valid, y, metric, **kw):
        seen.update(kw, n=len(X_valid), y=len(y))
        return {"ok": True}

    monkeypatch.setattr(gi, "run_group_importance", fake)
    r = TrialRunner.__new__(TrialRunner)
    r.config = types.SimpleNamespace(get=lambda k, d=None: {
        "evaluator.group_importance_sample_rows": 500}.get(k, d))
    r.spec = types.SimpleNamespace(primary_metric="rmse")
    valid = pd.DataFrame({"a": np.arange(2000)})
    y = np.where(np.arange(2000) % 7 == 0, 1.0, 0.0)
    tr = types.SimpleNamespace(trial_dir="/tmp/x", trial_id="t")
    assert r._run_group_importance(tr, valid, y, np.zeros(2000)) == {"ok": True}
    pos = seen["row_positions"]
    assert seen["n_full_rows"] == 2000 and len(pos) == seen["n"] == seen["y"]
    assert np.all(np.diff(pos) > 0)
    assert len(seen["treatment"]) == len(pos)          # treatment 同步取子集


# ===========================================================================
# D. D-006
# ===========================================================================

def _reg_spec():
    from runtime.spec import TaskSpec

    return TaskSpec(data={
        "task_id": "v2162", "task_name": "v2162", "model_type": "regression",
        "target": {"name": "y", "value_definition": "x"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "rmse",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {}})


def test_regression_seed_metrics_use_variant_predictions():
    from runtime.config import Config
    from runtime.evaluator import Evaluator

    rng = np.random.default_rng(0)
    n = 4000
    y = np.where(rng.random(n) < .2, rng.gamma(3, 1, n), 0.0)
    child = y * .8 + rng.normal(scale=.5, size=n)
    parent = child + rng.normal(scale=.05, size=n)
    seeds = [child + rng.normal(scale=.3, size=n) for _ in range(2)]
    ev = Evaluator(_reg_spec(), Config.load(skill_config=False))
    out = ev.evaluate(y, child, parent_score=parent, confirm_trial=True,
                      multi_seed_scores=seeds)
    cv = out["confirmation_validation"]
    sm = cv["seed_metrics"]
    assert len(sm) == 3 and len(set(round(v, 9) for v in sm)) == 3, sm
    want = [float(np.sqrt(np.mean((y - s) ** 2))) for s in seeds]
    assert sm[1:] == pytest.approx(want)
    assert out["stability"]["multi_seed_std"] > 0
    assert cv["sigma_seed_estimate"] > 1e-6


def test_multi_seed_evidence_reconciles_code_executed(tmp_path):
    from runtime.runner import TrialRunner

    d = tmp_path / "t"
    (d / "config").mkdir(parents=True)
    (d / "seed_scores").mkdir()
    dump_yaml(d / "config" / "trial.yaml", {"multi_seed": 1})
    for k in (1, 2):
        pd.DataFrame({"score": [0.0]}).to_parquet(
            d / "seed_scores" / f"seed-{k}.parquet")
    (d / "metrics.json").write_text(json.dumps({"multi_seed_n": 3}),
                                    encoding="utf-8")
    r = TrialRunner.__new__(TrialRunner)
    ev = r._multi_seed_evidence(types.SimpleNamespace(trial_dir=str(d)))
    assert ev["status"] == "complete" and ev["declared"] == 3
    assert ev["declared_config"] == 1 and ev["found"] == 2


def test_confirm_writes_effective_multi_seed():
    from _mvo.cmd_trial import _confirm_effective_multi_seed

    assert _confirm_effective_multi_seed(1, "... seed_scores ...") == 3
    assert _confirm_effective_multi_seed(None, "... seed_scores ...") == 3
    assert _confirm_effective_multi_seed(5, "... seed_scores ...") == 5
    assert _confirm_effective_multi_seed(0, "no seed segment") == 1
    src = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    # confirm 脚手架确实调用它写 config.multi_seed
    assert '_confirm_cfg["multi_seed"] = _confirm_effective_multi_seed(' in src


def test_failed_reentry_archives_previous_result(tmp_path):
    from runtime.runner import TrialRunner
    from runtime.scheduler import AttemptPolicy

    class _T:
        trial_dir = str(tmp_path)
        data = {"status": "FAILED", "failure_reason": "code_error",
                "result": {"status": "failed", "returncode": -9,
                           "runtime_sec": 241.2},
                "status_history": [{"state": "FAILED",
                                    "at": "2026-10-23T04:02:35+08:00"}]}

        def flush(self):
            pass

    r = TrialRunner.__new__(TrialRunner)
    r.attempt_policy = AttemptPolicy(max_retries=1, max_fix_rounds=2)
    t = _T()
    assert r._reenter_failed_trial(t) == "code_error"
    assert "result" not in t.data and "failure_reason" not in t.data
    att = t.data["attempts"][-1]
    assert att["previous_result"]["returncode"] == -9
    assert att["failure_reason"] == "code_error"
    assert t.data["status"] == "CODE_WRITTEN"


def test_wait_only_counts_current_attempt(tmp_path):
    from _mvo.context import _trial_reached_terminal

    run = tmp_path
    (run / "logs").mkdir()
    tdir = run / "trials" / "t1"
    tdir.mkdir(parents=True)

    def _t(status, hist):
        return types.SimpleNamespace(
            trial_id="t1", trial_dir=str(tdir), status=status,
            data={"status_history": [{"state": s, "at": a} for s, a in hist]})

    old = "2020-01-01T00:00:00+08:00"
    # 重入后 RUNNING:历史里的上一轮 FAILED 不再让 wait 立即返回
    t_run = _t("RUNNING", [("RUNNING", old), ("FAILED", old),
                           ("CODE_WRITTEN", old), ("RUNNING", old)])
    assert not _trial_reached_terminal(t_run)
    assert not _trial_reached_terminal(t_run, run)
    # detach 刚返回、worker 尚未改状态:pid 文件晚于上一轮终态且进程存活 → 等
    t_failed = _t("FAILED", [("RUNNING", old), ("FAILED", old)])
    pidf = run / "logs" / "detached-run-t1.pid"
    pidf.write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert not _trial_reached_terminal(t_failed, run)
    assert _trial_reached_terminal(t_failed)           # 旧调用方口径不变
    # worker 已退出(pid 不存活)→ 终态
    pidf.write_text("999999999\n", encoding="utf-8")
    assert _trial_reached_terminal(t_failed, run)
    # pid 文件早于最近一次终态(旧 attempt 的残留)→ 不误判
    pidf.write_text(f"{os.getpid()}\n", encoding="utf-8")
    os.utime(pidf, (1_000_000_000, 1_000_000_000))
    t_new = _t("FAILED", [("FAILED", "2026-10-23T05:56:02+08:00")])
    assert _trial_reached_terminal(t_new, run)
    # 活 runner 锁 → 当前 attempt 在路上
    (tdir / ".runner.lock").write_text(f"{os.getpid()}\n", encoding="utf-8")
    assert not _trial_reached_terminal(t_new, run)
    (tdir / ".runner.lock").unlink()
    # force_rerun 回退态(无活进程)沿用旧语义:以历史为准立即返回
    t_cw = _t("CODE_WRITTEN", [("EVALUATED", old), ("CODE_WRITTEN", old)])
    assert _trial_reached_terminal(t_cw, run)
