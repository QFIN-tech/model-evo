"""v2.15.0 回归测试:HPO 全链路修复(req-t7-rand400w run3 三轮 HPO 复盘回写)。

复盘结论(详见 docs/MODELEVO_V2_15_0_HPO_20261003.md):
  - 同一 inner 目标连开 3 个独立 study(84 config),每轮从零、seed 固定,
    TPE 引导段相对随机段增益 +0.00008/+0.00023/−0.00004,轮间学习靠手改边界;
  - 嫁接段 `_build_model` 不透传 N_ESTIMATORS_CAP(3 次命中),且"后定义
    覆盖"的追加式嫁接让父代修好的版本被覆盖,每代复发;
  - `_fit_with_es` 第二段 eval_set 在训练集内 → learning_curve 是样本内曲线,
    `best_iteration == n_estimators_used` 恒成立,撞 cap 判据失效;
  - confirm 二分类分支落脚手架 raise、multi-seed eval_set=None 且用默认超参;
  - 方向台账 `feat:feat:` 双前缀堵住 diminishing_return 停止出口;
  - 预注册"Δ≤MDE 即证伪"未被执行,第三轮仍开;confirm 打在当代即被超越的配方上。

本文件锁定(全部离线、无 CLI,<30s):
  A. 搜索引擎:seed 派生、启动点数、热启动、enqueue、边界外扩、续跑、签名陈旧、
     并行、多保真、诊断、lr 模式、并行度、签名、热启动收集/加载、质量评估;
  B. 模板:_build_model 透传 cap、_fit_with_es 第一段遥测/撞 cap/外扩/树数口径/
     refit=False/耐心缩放、dump_learning_curve 与 best_iteration_of 口径;
  C. 嫁接/升级:幂等替换、confirm 代码升级、lint 标记中和、改线 eval_set;
  D. 方向台账键归一;lint 规则 10 兜底;
  E. 创建期两道门与 planner 质量感知。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

from runtime import hpo as H  # noqa: E402

optuna = pytest.importorskip("optuna")
xgb = pytest.importorskip("xgboost")

SPACE = {
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.15, "log": True},
    "reg_lambda": {"type": "float", "low": 0.001, "high": 20.0, "log": True},
    "colsample_bytree": {"type": "float", "low": 0.55, "high": 1.0},
    "max_depth": {"type": "int", "low": 3, "high": 10},
}


def _obj_outside(p):
    """最优点在空间外(lr↓ λ↑ colsample↓,run3 的实测形态)。"""
    return -(math.log(p["learning_rate"] / 0.008) ** 2 * 0.01
             + math.log(p["reg_lambda"] / 40) ** 2 * 0.002
             + (p["colsample_bytree"] - 0.45) ** 2
             + (p["max_depth"] - 8) ** 2 * 0.001)


@pytest.fixture(autouse=True)
def _no_smoke(monkeypatch):
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)


# --------------------------------------------------------------------------
# A. 搜索引擎
# --------------------------------------------------------------------------

def test_derive_seed_varies_by_trial_and_is_stable():
    a = H.derive_seed(20260814, "trial-016")
    assert a == H.derive_seed(20260814, "trial-016")
    assert a != H.derive_seed(20260814, "trial-018")
    assert H.derive_seed(20260814) == 20260814


def test_n_startup_scales_with_budget():
    assert H._n_startup_for(16) == 5      # 旧实现 10/16 是纯随机
    assert H._n_startup_for(36) == 9
    assert H._n_startup_for(40) == 10
    assert H._n_startup_for(3) == 3


def test_point_in_space_and_clip():
    p = {"learning_rate": 0.05, "reg_lambda": 1.0,
         "colsample_bytree": 0.9, "max_depth": 6}
    assert H.point_in_space(p, SPACE)
    assert not H.point_in_space({**p, "max_depth": 11}, SPACE)
    assert not H.point_in_space({k: v for k, v in p.items()
                                 if k != "max_depth"}, SPACE)
    c = H.clip_to_space({"learning_rate": 0.001, "max_depth": 12.7,
                         "n_estimators": 8000}, SPACE)
    assert c == {"learning_rate": 0.02, "max_depth": 10}


def test_boundary_hits_and_expand_respect_limits():
    obs = [{"params": {"learning_rate": 0.0201, "reg_lambda": 19.0,
                       "colsample_bytree": 0.56, "max_depth": 6},
            "inner_metric": 1.0 - i * 0.01} for i in range(6)]
    hits = H.boundary_hits(obs, SPACE, True)
    assert hits["learning_rate"]["side"] == "low"
    assert hits["reg_lambda"]["side"] == "high"
    assert hits["colsample_bytree"]["side"] == "low"
    assert "max_depth" not in hits
    new, ch = H.expand_space(SPACE, hits, [], lr_floor=0.01)
    assert new["learning_rate"]["low"] == pytest.approx(0.01)  # 2.5 倍受 floor 截断
    assert new["reg_lambda"]["high"] == pytest.approx(50.0)
    assert new["colsample_bytree"]["low"] == pytest.approx(0.55 - 0.45 * 0.25)
    assert {c["param"] for c in ch} == {"learning_rate", "reg_lambda",
                                        "colsample_bytree"}
    # 已达单参上限次数 → 不再扩
    new2, ch2 = H.expand_space(new, hits, ch + ch, lr_floor=0.01)
    assert ch2 == []


def test_expand_ignores_unknown_params():
    sp = {"weird": {"type": "float", "low": 0.0, "high": 1.0}}
    obs = [{"params": {"weird": 0.0}, "inner_metric": 1 - i * .1}
           for i in range(5)]
    hits = H.boundary_hits(obs, sp, True)
    assert "weird" in hits
    assert H.expand_space(sp, hits, [])[1] == []


def test_run_expands_space_within_one_trial(tmp_path):
    r = H.run_hpo_search(SPACE, 40, _obj_outside, metric_direction_up=True,
                         seed=1, expand=True, seed_salt="t-exp",
                         checkpoint_path=tmp_path / "ck.json")
    assert r["n_configs_evaluated"] == 40
    assert r["space_expansions"], "最优在界外时必须在同 trial 内外扩"
    assert r["space_final"] != r["space_initial"]
    ck = json.loads((tmp_path / "ck.json").read_text())
    assert ck["schema"] == 2 and ck["base_space"] == SPACE
    # 续跑:基础空间一致 → 回放全部、沿用外扩后的空间
    r2 = H.run_hpo_search(SPACE, 40, _obj_outside, metric_direction_up=True,
                          seed=1, expand=True, seed_salt="t-exp",
                          checkpoint_path=tmp_path / "ck.json")
    assert r2["n_configs_replayed"] == 40
    assert r2["n_configs_fit_this_attempt"] == 0
    assert r2["space_final"] == r["space_final"]


def test_signature_mismatch_makes_checkpoint_stale(tmp_path):
    ck = tmp_path / "ck.json"
    H.run_hpo_search(SPACE, 6, _obj_outside, metric_direction_up=True,
                     seed=1, checkpoint_path=ck, signature="A")
    calls = []

    def f(p):
        calls.append(p)
        return _obj_outside(p)

    r = H.run_hpo_search(SPACE, 6, f, metric_direction_up=True, seed=1,
                         checkpoint_path=ck, signature="B")
    assert r["n_configs_replayed"] == 0 and len(calls) == 6


def test_warm_start_feeds_prior_and_best():
    warm = [{"params": {"learning_rate": 0.0201, "reg_lambda": 19.9,
                        "colsample_bytree": 0.551, "max_depth": 8},
             "inner_metric": 0.0, "source": "trial-x"}]
    warm += [{"params": {"learning_rate": 0.1, "reg_lambda": 0.01,
                         "colsample_bytree": 0.9, "max_depth": 3},
              "inner_metric": -1.0 - i, "source": "trial-x"}
             for i in range(12)]
    warm[2]["params"] = {**warm[2]["params"], "max_depth": 4}  # 去重键不同
    r = H.run_hpo_search(SPACE, 4, lambda p: -5.0, metric_direction_up=True,
                         seed=1, warm_start=warm)
    assert r["warm_start"]["n_points"] >= 2
    assert r["n_configs_evaluated"] == 4        # 热启动不计入本 trial 评估数
    assert r["inner_score"] == 0.0 and r["best_origin"] == "warm"
    # 越界/缺键的热启动点被丢弃
    bad = [{"params": {"learning_rate": 0.5}, "inner_metric": 9.0}]
    r2 = H.run_hpo_search(SPACE, 3, lambda p: -5.0, metric_direction_up=True,
                          seed=1, warm_start=bad)
    assert r2["warm_start"]["n_points"] == 0


def test_enqueue_incumbent_evaluated_first():
    seen = []

    def f(p):
        seen.append(dict(p))
        return 0.0

    inc = H.incumbent_params("xgboost", None, None)
    r = H.run_hpo_search(SPACE, 5, f, metric_direction_up=True, seed=1,
                         enqueue=[inc])
    assert seen[0]["learning_rate"] == pytest.approx(0.05)
    assert seen[0]["max_depth"] == 6
    assert r["trials"][0]["origin"] == "enqueued" and r["n_enqueued"] == 1


def test_parallel_jobs_complete_budget():
    r = H.run_hpo_search(SPACE, 10, _obj_outside, metric_direction_up=True,
                         seed=1, n_jobs=2, seed_salt="par")
    assert r["n_configs_evaluated"] == 10 and r["parallel"] == 2
    assert "+par2" in r["sampler"]


def test_dict_return_telemetry_recorded():
    r = H.run_hpo_search(SPACE, 4, lambda p: {"value": 0.1,
                                              "best_iteration": 77,
                                              "cap_bound": True},
                         metric_direction_up=True, seed=1)
    assert all(t["best_iteration"] == 77 for t in r["trials"])
    assert r["diagnostics"]["cap_bound_configs"] == 4


def test_successive_halving_mode():
    calls = []

    def g(p, fid=1.0):
        calls.append(fid)
        return _obj_outside(p) - (1 - fid) * 0.01

    r = H.run_hpo_search(SPACE, 27, g, metric_direction_up=True, seed=1,
                         fidelities=H.DEFAULT_FIDELITIES)
    mf = r["multi_fidelity"]
    assert mf["plan"] == [27, 9, 3] and mf["best_fidelity"] == 1.0
    assert calls.count(1.0) == 3
    assert all(float(t["fidelity"]) == 1.0 for t in r["trials"])


def test_smoke_disables_engine_extras(monkeypatch):
    monkeypatch.setenv("MODELEVO_SMOKE_RUN", "1")
    r = H.run_hpo_search(SPACE, 40, _obj_outside, metric_direction_up=True,
                         seed=1, n_jobs=4, expand=True,
                         warm_start=[{"params": {}, "inner_metric": 1}],
                         enqueue=[{"learning_rate": 0.05}])
    assert r["n_configs_evaluated"] == 1 and r["parallel"] == 1
    assert r["warm_start"]["n_points"] == 0 and r["space_expansions"] == []


def test_plan_lr_mode():
    s = H.plan_lr_mode(SPACE, {"lr_mode": "auto"}, 300_000)
    assert s["mode"] == "shrink" and "learning_rate" not in s["space"]
    assert s["refit_lr"] == pytest.approx(0.0125)
    assert s["refit_cap_scale"] == pytest.approx(4.0)
    assert H.plan_lr_mode(SPACE, {"lr_mode": "auto"}, 50_000)["mode"] == \
        "search"
    assert H.plan_lr_mode(SPACE, {}, 10 ** 7)["mode"] == "search"
    nolr = {k: v for k, v in SPACE.items() if k != "learning_rate"}
    # v2.16.3:显式 shrink 不再因"空间无 lr 维"静默回落 search(gen-05
    # trial-022 事故);仍收缩重训,lr_key 取 learning_rate
    p = H.plan_lr_mode(nolr, {"lr_mode": "shrink"}, 10 ** 7)
    assert p["mode"] == "shrink" and p["lr_key"] == "learning_rate"
    assert p["space"] == nolr and p["note"]


def test_hpo_parallelism_and_fidelity_plan():
    assert H.hpo_parallelism({}, "xgboost", True) == 2
    assert H.hpo_parallelism({}, "lightgbm", True) == 1
    assert H.hpo_parallelism({"parallel_configs": 3}, "lightgbm", False) == 3
    assert H.hpo_fidelity_plan({}, 10 ** 6) is None
    assert H.hpo_fidelity_plan({"fidelity": "successive_halving"},
                               300_000) == H.DEFAULT_FIDELITIES
    assert H.hpo_fidelity_plan({"fidelity": "successive_halving"},
                               1000) is None


def _frames(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = pd.Series((rng.random(n) > 0.6).astype(int))
    return X, y


def test_inner_signature_stable_and_sensitive():
    X, y = _frames()
    s1 = H.inner_signature(X.iloc[:400], y.iloc[:400], X.iloc[400:],
                           y.iloc[400:], {"cap": 8000})
    assert s1 == H.inner_signature(X.iloc[:400], y.iloc[:400], X.iloc[400:],
                                   y.iloc[400:], {"cap": 8000})
    # cap 不同(trial-011 cap=2000 vs 016 cap=8000)→ 不同目标
    assert s1 != H.inner_signature(X.iloc[:400], y.iloc[:400], X.iloc[400:],
                                   y.iloc[400:], {"cap": 2000})
    assert s1 != H.inner_signature(X.iloc[:399], y.iloc[:399], X.iloc[399:],
                                   y.iloc[399:], {"cap": 8000})


def test_collect_and_load_warm_start_roundtrip(tmp_path):
    tdir = tmp_path / "trials" / "trial-007-g03-xgboost-1"
    tdir.mkdir(parents=True)
    (tdir / "hpo-study.json").write_text(json.dumps({
        "inner_signature": "SIG", "algorithm": "xgboost",
        "trials": [{"params": {"a": 1}, "inner_metric": 0.5,
                    "origin": "guided"},
                   {"params": {"a": 2}, "inner_metric": 0.6,
                    "origin": "warm"}]}))
    old = tmp_path / "trials" / "trial-001-g01-xgboost-1"
    old.mkdir(parents=True)
    (old / "hpo-study.json").write_text(json.dumps({"trials": [
        {"params": {"a": 3}, "inner_metric": 0.9}]}))   # 无签名 → 不收
    srcs = H.collect_warm_start(tmp_path, algorithm="xgboost")
    assert len(srcs) == 1 and len(srcs[0]["trials"]) == 1  # warm 来源不重复收
    f = tmp_path / "ws.json"
    f.write_text(json.dumps({"sources": srcs}))
    assert len(H.load_warm_start(f, "SIG")) == 1
    assert H.load_warm_start(f, "OTHER") == []


def test_assess_hpo_study(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "hpo-study.json").write_text(json.dumps({
        "n_configs_evaluated": 16,
        "diagnostics": {"boundary_hits": {"learning_rate": {"side": "low"}}}}))
    (d / "metrics.json").write_text(json.dumps({"es_info": {
        "cap_bound": True}}))
    q = H.assess_hpo_study(d, default_budget=40)
    assert q["complete"] is False
    assert any("under_budget" in r for r in q["reasons"])
    assert any("boundary_hits" in r for r in q["reasons"])
    assert "outer_refit_cap_bound" in q["reasons"]


# --------------------------------------------------------------------------
# B. 模板(exec 渲染源码段,测真正跑在 trial 里的实现)
# --------------------------------------------------------------------------

def _body() -> str:
    from _mvo.templates import TRIAL_BODY
    return TRIAL_BODY


_PRELUDE = (
    "import os\nimport json\nimport pathlib\nimport numpy as np\n"
    "import pandas as pd\nRUN_DIR = pathlib.Path('/tmp/_v215_no_run')\n"
    "METRIC_NAME = 'auc'\n_INNER_METRIC_UP = True\n"
    "MODEL_TYPE = 'binary_classification'\n"
    "def _lgbm_fit_safe(model, *a, **kw):\n    return model.fit(*a, **kw)\n"
    "def _gpu_allocated(n):\n    return False\n"
)


def _exec_segment(start: str, end: str, extra: str = "") -> dict:
    src = _body()
    i = src.index(f"def {start}")
    j = src.index(f"def {end}", i)
    # v2.15.8:段内会调用前置段的 _clamp_trees(读 _MAX_TREES)。直接从
    # 模板前置段取原文,而不是在 _PRELUDE 里手抄一份 —— 手抄版缺了它时,
    # _fit_with_es 的宽 except 会把 NameError 吞成"退回全量 fit",
    # cap 外扩测试拿不到 mvo_es_info_(与嫁接路径 NameError 同根)。
    k0 = src.index("\n_MAX_TREES = ")
    k1 = src.index("\ndef load_config(", k0)
    ns: dict = {"__builtins__": __builtins__}
    exec(_PRELUDE + src[k0:k1] + "\n" + extra + src[i:j], ns)
    return ns


def _clf(n=3000, seed=0):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=n, n_features=10, n_informative=6,
                               random_state=seed)
    return (pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]),
            pd.Series(y))


def test_build_model_passes_trial_cap(monkeypatch):
    monkeypatch.delenv("MODELEVO_N_ESTIMATORS_CAP", raising=False)
    ns = _exec_segment("_build_model", "_inner_metric",
                       "_HPO_ALGO = 'xgboost'\nN_ESTIMATORS_CAP = 8000\n"
                       "EARLY_STOP_ROUNDS = 77\n")
    m = ns["_build_model"]({}, 0, 100)
    assert m.get_params()["n_estimators"] == 8000
    assert m.get_params()["early_stopping_rounds"] == 77
    # 无模块常量时走 runner 下发的环境变量
    ns2 = _exec_segment("_build_model", "_inner_metric",
                        "_HPO_ALGO = 'xgboost'\n")
    monkeypatch.setenv("MODELEVO_N_ESTIMATORS_CAP", "3333")
    assert ns2["_build_model"]({}, 0, 100).get_params()["n_estimators"] == 3333


def _fit_ns():
    return _exec_segment("_fit_with_es", "_fit_with_eval",
                         "def _fit_with_eval(m, a, b, c, d):\n"
                         "    return _fit_with_es(m, a, b, auto_extend=True)\n")


def test_fit_with_es_stage1_telemetry_and_tree_count():
    ns = _fit_ns()
    X, y = _clf()
    m = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.1, max_depth=3,
                          early_stopping_rounds=30, random_state=0, n_jobs=2)
    m = ns["_fit_with_es"](m, X, y)
    info = m.mvo_es_info_
    assert info["stopped_by"] == "early_stop" and info["cap_bound"] is False
    assert info["stage1_rounds"] == len(info["stage1_curve"])
    # 正常早停:实训轮数 − 最优树数 = 耐心
    assert info["stage1_rounds"] - info["stage1_best_iteration"] == \
        info["es_rounds"]
    # 第二段树数 = 第一段最优树数(1-based;旧实现少一棵)
    assert m.get_params()["n_estimators"] == info["stage1_best_iteration"]
    assert info["refit_trees"] == info["stage1_best_iteration"]


def test_fit_with_es_detects_cap_and_auto_extends(monkeypatch):
    ns = _fit_ns()
    X, y = _clf()
    m = xgb.XGBClassifier(n_estimators=20, learning_rate=0.01, max_depth=3,
                          early_stopping_rounds=10, random_state=0, n_jobs=2)
    m1 = ns["_fit_with_es"](m, X, y, refit=False)
    assert m1.mvo_es_info_["cap_bound"] is True
    assert m1.mvo_es_info_["stopped_by"] == "cap"
    m2 = xgb.XGBClassifier(n_estimators=20, learning_rate=0.01, max_depth=3,
                           early_stopping_rounds=10, random_state=0, n_jobs=2)
    m2 = ns["_fit_with_es"](m2, X, y, auto_extend=True)
    assert m2.mvo_es_info_["cap_extended_to"] == 40
    assert m2.mvo_es_info_["cap_initial"] == 20


def test_fit_with_es_refit_false_skips_stage2():
    ns = _fit_ns()
    X, y = _clf()
    m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=3,
                          early_stopping_rounds=20, random_state=0, n_jobs=2)
    m = ns["_fit_with_es"](m, X, y, refit=False)
    assert m.get_params()["n_estimators"] == 500    # 未改写 = 未进第二段
    assert "refit_trees" not in m.mvo_es_info_


def test_es_patience_scales_with_low_lr(monkeypatch):
    ns = _fit_ns()
    m = xgb.XGBClassifier(learning_rate=0.005)
    assert ns["_es_rounds_for"](m, 50) == 100       # ceil(0.5/0.005)
    assert ns["_es_rounds_for"](xgb.XGBClassifier(learning_rate=0.05),
                                50) == 50
    monkeypatch.setenv("MODELEVO_ES_LR_SCALE", "0")
    assert ns["_es_rounds_for"](m, 50) == 50


def test_learning_curve_uses_stage1_heldout(tmp_path):
    ns = _exec_segment("_es_info_of", "_fit_seed_variant",
                       f"TRIAL_DIR = pathlib.Path({str(tmp_path)!r})\n")

    class M:
        mvo_es_info_ = {"stage1_curve": [0.5, 0.4, 0.45],
                        "stage1_best_iteration": 2}

        def evals_result(self):
            return {"validation_0": {"logloss": [9, 8, 7, 6]}}

    out = ns["dump_learning_curve"](M())
    assert out == {"learning_curve_rows": 3,
                   "learning_curve_source": "es_stage1_heldout"}
    assert ns["best_iteration_of"](M()) == 2
    assert "stage1_curve" not in ns["_es_info_of"](M())


# --------------------------------------------------------------------------
# C. 嫁接幂等 / confirm 代码升级 / lint 标记 / 改线
# --------------------------------------------------------------------------

def test_strip_previous_graft_blocks():
    from _mvo.render import _strip_previous_graft_blocks

    block = ("# v19.3 嫁接后自动裁剪: x\n# ==== v19 hpo 嫁接段 BEGIN(...)\n"
             "def _build_model(p, s, n=0):\n    return 1\n"
             "# ==== v19 hpo 嫁接段 END ====\n")
    code = "SEED = 20260814\n" + block + block + "def main():\n    pass\n"
    new = "def _build_model(p, s, n=0):\n    return 2\n"
    out, n = _strip_previous_graft_blocks(code, new)
    assert n == 2 and "嫁接段 BEGIN" not in out and "def main" in out
    # 旧段独有符号 → 放弃移除(退回追加语义)
    code2 = code.replace("return 1\n", "return 1\ndef _only_old():\n"
                         "    return 0\n", 1)
    out2, n2 = _strip_previous_graft_blocks(code2, new)
    assert n2 == 0 and out2 == code2


def test_upgrade_old_hpo_code_for_confirm():
    from _mvo.render import _upgrade_hpo_lineage_code

    old = ("SEED = 20260814\n_HPO_ALGO = 'xgboost'\n"
           "def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,"
           " w_tr=None, w_va=None):\n"
           "    _cfg_hp = {}\n"
           "    if cfg.get('hpo'):\n        pass\n"
           '    elif MODEL_TYPE == "regression":\n        pass\n'
           "    else:\n        raise RuntimeError('x'  # lint:scaffold-"
           "fit-and-score\n                           )\n"
           "def main():\n    pass\n\n\nif __name__ == \"__main__\":\n"
           "    main()\n")
    out, notes = _upgrade_hpo_lineage_code(old)
    assert len(notes) == 2
    import ast
    ast.parse(out)
    assert 'elif globals().get("_HPO_ALGO"):' in out
    assert "def _fit_seed_variant" in out
    assert "lint:scaffold-fit-and-score" not in out
    assert out.index("def _fit_seed_variant") < out.index("if __name__")
    assert _upgrade_hpo_lineage_code(out)[1] == []          # 幂等
    # 非 HPO 血缘不动
    assert _upgrade_hpo_lineage_code("x = 1\n") == ("x = 1\n", [])


def test_template_scaffold_marker_neutralized_on_algo_injection():
    from _mvo.render import _inject_hpo_algo_at
    from runtime.templates import TRIAL_TEMPLATE

    code = TRIAL_TEMPLATE.format(
        trial_id="t", hypothesis="h", actions="a", run_dir="/tmp/x",
        target="y", model_type="binary_classification", treatment="",
        seed=20260814, skill_root=str(SKILL), body=_body())
    assert "lint:scaffold-fit-and-score" in code
    out = _inject_hpo_algo_at(code, "xgboost")
    assert "lint:scaffold-fit-and-score" not in out
    import lint_trial as L
    assert L.check_scaffold_fit_and_score(out) == []
    # 无 _HPO_ALGO 的探索脚手架仍被规则 10 拦
    assert L.check_scaffold_fit_and_score(code)


def test_rewire_carries_es_slice_into_eval_set():
    from _mvo.render import _rewire_main_to_fit_and_score

    code = (
        "SEED = 20260814\n"
        "def main():\n"
        "    cfg = {}\n    X_tr = y_tr = X_va = y_va = None\n"
        "    X_es, y_es = 1, 2\n"
        "    model = build_estimator(1)\n"
        "    eval_set = [(X_es, y_es)]\n"
        "    pd.DataFrame().to_parquet(TRIAL_DIR / 'validation_score.parquet')\n"
        "    f(eval_set)\n")
    out, _ = _rewire_main_to_fit_and_score(code)
    assert "eval_set = [(X_es, y_es)]" in out and "eval_set = None" not in out


# --------------------------------------------------------------------------
# D. 方向台账键归一
# --------------------------------------------------------------------------

def test_family_hint_double_prefix_normalized():
    from runtime.directions import family_of_trial

    base = {"search_action": "exploit", "changed_modules": ["feature"]}
    assert family_of_trial({**base, "family_hints": ["feat:seqw"]}) == \
        "feat:seqw"
    assert family_of_trial({**base, "family_hints": ["seqw_"]}) == "feat:seqw"
    assert family_of_trial({**base, "family_hints": ["feat:feat:seqw"]}) == \
        "feat:seqw"


# --------------------------------------------------------------------------
# E. 创建期两道门 + planner 质量感知
# --------------------------------------------------------------------------

def _spec():
    from runtime.spec import TaskSpec

    return TaskSpec(data={
        "task_id": "v215", "task_name": "v215",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {}, "resource_budget": {"max_trials": 30}})


def _mk(fac, action, parents=None, gen=1, delta=None, nfeat=10,
        verdict="accepted", status=None, algo="xgboost"):
    t = fac.create(algorithm=algo,
                   hypothesis={"statement": "测试用最小假设陈述",
                               "rationale": "r", "falsification": "f"},
                   execution={"action_description": ["a"],
                              "code_entry": "train.py",
                              "command": "python train.py",
                              "changed_modules": ["model"]},
                   search_action=action, parent_trial_ids=parents,
                   generation=gen)
    if delta is not None or verdict:
        t.data["evaluation"] = {
            "verdict": verdict, "trial_id": t.trial_id,
            "primary_metric": {"name": "auc", "value": 0.8 + (delta or 0)},
            "delta": {"vs_parent": delta}, "integrity": {"valid": True},
            "selection_correction": {"passed": True}}
    if status:
        t.data["status"] = status
    t.flush()
    (Path(t.trial_dir) / "metrics.json").write_text(
        json.dumps({"n_features": nfeat}))
    return t


def test_hpo_diminishing_gate_follows_preregistered_mde(tmp_path):
    from runtime.config import Config
    from runtime.trial import TrialFactory
    from _mvo.cmd_trial import _hpo_diminishing_check

    run = tmp_path / "run"
    run.mkdir()
    cfg = Config.load(skill_config=False)
    fac = TrialFactory(run, spec=_spec(), config=cfg)
    base = _mk(fac, "explore", gen=0, delta=None)
    r1 = _mk(fac, "hpo", [base.trial_id], 1, delta=0.0011)
    r2 = _mk(fac, "hpo", [r1.trial_id], 2, delta=0.00084)
    conf = _mk(fac, "confirm", [r2.trial_id], 3, delta=0.00001)
    assert _hpo_diminishing_check(run, base.trial_id, cfg) is None
    assert _hpo_diminishing_check(run, r1.trial_id, cfg) is None   # 0.0011≥MDE
    d = _hpo_diminishing_check(run, r2.trial_id, cfg)               # run3 R3
    assert d and d["last_hpo"] == r2.trial_id and d["delta_vs_parent"] < 0.001
    d2 = _hpo_diminishing_check(run, conf.trial_id, cfg)            # confirm 的配方
    assert d2 and d2["last_hpo"] == r2.trial_id


def test_planner_hpo_state_quality_aware(tmp_path):
    from runtime.config import Config
    from runtime.planner import AutonomousPlanner
    from runtime.trial import TrialFactory

    run = tmp_path / "run"
    run.mkdir()
    cfg = Config.load(skill_config=False)
    fac = TrialFactory(run, spec=_spec(), config=cfg)
    base = _mk(fac, "explore", gen=0, delta=None)
    champ = _mk(fac, "hpo", [base.trial_id], 1, delta=0.0011)
    (Path(champ.trial_dir) / "hpo-study.json").write_text(json.dumps({
        "n_configs_evaluated": 16,
        "diagnostics": {"boundary_hits": {"learning_rate": {"side": "low"}}}}))
    pl = AutonomousPlanner(run, _spec(), cfg)
    st = pl._champion_face_hpo_state(pl.by_id[champ.trial_id])
    # 冠军本身是 HPO → done;16 config 且顶边界 → 不完整;Δ≥MDE → 非递减
    assert st["done"] and not st["complete"] and not st["diminishing"]
    assert any("under_budget" in r for r in st["reasons"])
    cands = [p for p in pl._model_candidates(pl.by_id[champ.trial_id])
             if p["family"] == "model:hpo"]
    assert cands and cands[0]["acquisition"]["expected_gain"] == \
        pytest.approx(min(0.00075, 0.5 * 0.0011))
    # 同算法 HPO 的物化命令走 --from-parent(加宽面安全)
    assert "--from-parent" in cands[0]["materialize_command"]
    assert "--hpo-parent" not in cands[0]["materialize_command"]


def test_inflight_hpo_children(tmp_path):
    from runtime.config import Config
    from runtime.trial import TrialFactory
    from _mvo.cmd_trial import _inflight_hpo_children

    run = tmp_path / "run"
    run.mkdir()
    cfg = Config.load(skill_config=False)
    fac = TrialFactory(run, spec=_spec(), config=cfg)
    base = _mk(fac, "hpo", None, 1, delta=0.002)
    child = _mk(fac, "hpo", [base.trial_id], 2, delta=None, verdict=None)
    assert child.status == "PROPOSED"
    assert _inflight_hpo_children(fac, base.trial_id) == [child.trial_id]
