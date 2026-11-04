"""v2.17.0 WP3:交付重训口径留痕、回归交付诊断与报告「模型局限」自动填充。

事故来源:run4 regression.2(run-loan-amt-t7-reg,原值金额 rmse 回归)。

1. #5 refit 口径:冠军 hurdle 容器 es_source=validation → frozen_rounds →
   train 上复现 10.4 min → sklearn.clone TypeError → 自动回落 train_tail 口径
   复跑配方;回落只在 log 里有一行,model_meta 只记 refit_mode=rerun_recipe,
   generated_by 硬编码 v2.15.9。
2. #20 回归诊断:final_test 交付预测均值 ≈2350 vs 实际 ≈1965(+20%),shadow 预检
   与 evaluate-final 都没有水位信号。
3. #21 报告 §14「模型局限」显示"未记录"。

全部离线、合成数据、临时 run 目录;finalize-retrain / evaluate-final 走真实
命令函数,只把选型/源码冻结/拓扑门/final_test 标签加载替换为夹具。
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

from runtime import __version__ as SKILL_VERSION  # noqa: E402
from runtime.config import Config  # noqa: E402
from runtime.data import SAMPLE_KEY_COLUMN  # noqa: E402
from runtime.integrity import bind_prediction_artifact  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from _mvo import cmd_deliver  # noqa: E402

_SRC = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
    encoding="utf-8")
CHAMP = "trial-015-g04-confirm-xgboost-353"
BASE = "trial-001-g00-explore-xgboost-5"
PARENT = "trial-013-g03-explore-xgboost-254"
SIZES = {"train": 1200, "validation": 300, "shadow_test": 300,
         "final_test": 300}
FINAL_LEVEL = 0.7          # final_test 真实水位下移 30%(交付模型必然高估)

# 冠军 trial 代码:KIND=plain → run4 同款手写容器(无 get_params,不可 clone);
# KIND=sk → sklearn 协议 hurdle(与 v2.17 官方 HurdleRegressor 同协议,可 clone)。
CHAMP_CODE = r'''
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

RUN_DIR = Path(__file__).resolve().parents[3]
SPLITS = RUN_DIR / "sample-profile" / "splits"
TRIAL_DIR = Path(__file__).resolve().parents[1]
SEED = 7
ES_FRACTION = 0.1
KIND = "__KIND__"
_last_model = None


def load_labels(split):
    return pd.read_parquet(SPLITS / "labels" / f"{split}.parquet")


def exclude_columns(df, cfg):
    return [c for c in df.columns if str(c).startswith("f")]


def prepare_features(X_tr, X_va):
    return X_tr.astype("float32"), X_va.astype("float32")


class HurdleXGB:
    """手写复合容器(run4 同款):无 get_params,不可 sklearn.clone。"""

    def __init__(self, coef, icpt):
        self.coef_, self.icpt_ = coef, icpt

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.icpt_


class HurdleReg(BaseEstimator, RegressorMixin):
    """sklearn 协议 hurdle:构造参数 clf/reg 为原型,拟合后子模型挂 clf_/reg_。"""

    def __init__(self, clf=None, reg=None):
        self.clf = clf
        self.reg = reg

    def fit(self, X, y, eval_set=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        pos = y > 0
        self.clf_, self.reg_ = clone(self.clf), clone(self.reg)
        kc, kr = {}, {}
        if eval_set is not None:
            Xe = np.asarray(eval_set[0], dtype=float)
            ye = np.asarray(eval_set[1], dtype=float)
            kc = {"eval_set": [(Xe, (ye > 0).astype(int))], "verbose": False}
            kr = {"eval_set": [(Xe[ye > 0], ye[ye > 0])], "verbose": False}
        self.clf_.fit(X, pos.astype(int), **kc)
        self.reg_.fit(X[pos], y[pos], **kr)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.clf_.predict_proba(X)[:, 1] * np.clip(
            self.reg_.predict(X), 0, None)


def _log(n):
    p = os.environ.get("MVO_TEST_CALL_LOG")
    if p:
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"n": int(n), "es": os.environ.get(
                "MODELEVO_ES_SOURCE", "validation")}) + "\n")


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    global _last_model
    _log(len(X_tr))
    if os.environ.get("MODELEVO_ES_SOURCE", "validation") == "validation" \
            and y_va is not None:
        X_fit, y_fit, X_es, y_es = X_tr, y_tr, X_va, y_va
    else:
        cut = int(len(X_tr) * (1 - ES_FRACTION))
        X_fit, y_fit = X_tr.iloc[:cut], y_tr.iloc[:cut]
        X_es, y_es = X_tr.iloc[cut:], y_tr.iloc[cut:]
    if KIND == "sk":
        import xgboost as xgb

        kw = dict(n_estimators=300, learning_rate=0.3, max_depth=2, n_jobs=1,
                  random_state=SEED, tree_method="hist",
                  early_stopping_rounds=10)
        model = HurdleReg(xgb.XGBClassifier(**kw), xgb.XGBRegressor(**kw))
        model.fit(X_fit, y_fit, eval_set=(X_es, y_es))
    elif KIND == "official":
        from runtime.hurdle import HurdleRegressor

        kw = dict(n_estimators=300, learning_rate=0.3, max_depth=2, n_jobs=1,
                  random_state=SEED, early_stopping_rounds=10)
        model = HurdleRegressor(clf_params=dict(kw), reg_params=dict(kw),
                                submodel_cache=False)
        model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)])
    else:
        A = np.asarray(X_fit, dtype=float)
        coef, *_ = np.linalg.lstsq(np.c_[A, np.ones(len(A))],
                                   np.asarray(y_fit, dtype=float), rcond=None)
        model = HurdleXGB(coef[:-1], float(coef[-1]))
    _last_model = model
    return model.predict(X_va), {"model_num": 2}
'''


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def _spec(model_type="regression", metric="rmse") -> TaskSpec:
    return TaskSpec(data={
        "task_id": "wp3-reg", "task_name": "WP3 合成回归",
        "model_type": model_type, "primary_metric": metric,
        "secondary_metrics": ["mae"] if model_type == "regression" else [],
        "target": {"name": "y"}, "time": {"column": "pday"},
        "split_policy": {"type": "time_based"}})


def _make_data(run: pathlib.Path) -> dict:
    """splits 特征面 / labels(不含 final_test)/ identity;返回 final_test 标签(只在内存)。"""
    rng = np.random.default_rng(11)
    base = run / "sample-profile" / "splits"
    (base / "labels").mkdir(parents=True)
    (base / "identity").mkdir()
    start = {"train": "2026-01-01", "validation": "2026-05-01",
             "shadow_test": "2026-06-01", "final_test": "2026-07-01"}
    final_labels = None
    for split, n in SIZES.items():
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
        pos = rng.random(n) < 1 / (1 + np.exp(-X["f0"].values))
        amt = np.exp(7.0 + 0.4 * X["f1"].values + rng.normal(0, 0.3, n))
        y = np.where(pos, amt, 0.0) * (FINAL_LEVEL if split == "final_test"
                                       else 1.0)
        days = pd.date_range(start[split], periods=4, freq="7D")
        pday = np.repeat(days.strftime("%Y-%m-%d").values, n // 4)
        keys = np.array([f"k-{split}-{i:05d}" for i in range(n)])
        X.to_parquet(base / f"{split}.parquet", index=False)
        pd.DataFrame({"pday": pday, SAMPLE_KEY_COLUMN: keys}).to_parquet(
            base / "identity" / f"{split}.parquet", index=False)
        lab = pd.DataFrame({"y": y, "pday": pday, SAMPLE_KEY_COLUMN: keys})
        if split == "final_test":
            final_labels = lab          # 铁律:final_test 标签不落盘
        else:
            lab.to_parquet(base / "labels" / f"{split}.parquet", index=False)
    return {"final_labels": final_labels}


def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dump_as_main(model, path: pathlib.Path) -> None:
    """按 trial 脚本执行的真实形态落盘:类引用为 `__main__.<Class>`。"""
    cls = type(model)
    main = sys.modules["__main__"]
    old = cls.__module__
    cls.__module__ = "__main__"
    setattr(main, cls.__name__, cls)
    try:
        joblib.dump(model, path)
    finally:
        cls.__module__ = old
        delattr(main, cls.__name__)


def _bind(path, scores, identity, run):
    pd.DataFrame({"score": scores, SAMPLE_KEY_COLUMN:
                  identity[SAMPLE_KEY_COLUMN].values}).to_parquet(
                      path, index=False)
    bind_prediction_artifact(
        path, identity, source_manifest={"combined_sha256": "src-sha"},
        split_manifest_path=run / "sample-profile" / "splits" / "manifest.json",
        data_view="v1")


def _make_run(tmp_path, *, kind="plain", es_source="validation",
              with_model=True) -> tuple[pathlib.Path, dict]:
    run = tmp_path / "run"
    info = _make_data(run)
    splits = run / "sample-profile" / "splits"
    td = run / "trials" / CHAMP
    for sub in ("code", "config", "model"):
        (td / sub).mkdir(parents=True)
    (td / "code" / "train.py").write_text(
        CHAMP_CODE.replace("__KIND__", kind), encoding="utf-8")
    (td / "config" / "trial.yaml").write_text("seed: 7\n", encoding="utf-8")
    (td / "metrics.json").write_text(json.dumps({"es_info": {
        "es_source": es_source, "cap": 300, "cap_bound": False}}))
    # 冠军 train-only 搜索产物(与 finalize 同一读数口径)
    mod = _load_module(td / "code" / "train.py", f"wp3_fixture_{kind}")
    feats = ["f0", "f1", "f2"]
    X_tr, X_va = mod.prepare_features(
        pd.read_parquet(splits / "train.parquet", columns=feats),
        pd.read_parquet(splits / "validation.parquet", columns=feats))
    y_tr = mod.load_labels("train")["y"].reset_index(drop=True)
    y_va = mod.load_labels("validation")["y"].reset_index(drop=True)
    sc_va, _ = mod.fit_and_score(X_tr, y_tr, X_va, y_va, 7, {"seed": 7})
    model = mod._last_model
    pd.DataFrame({"score": sc_va}).to_parquet(td / "validation_score.parquet")
    X_va.to_parquet(td / "model" / "validation_input.parquet", index=False)
    if with_model:
        if kind == "official":      # 类定义在 runtime 包内,不依赖 __main__
            joblib.dump(model, td / "model" / "model.joblib")
        else:
            _dump_as_main(model, td / "model" / "model.joblib")
    ident = {s: pd.read_parquet(splits / "identity" / f"{s}.parquet")
             for s in SIZES}
    X_sh = pd.read_parquet(splits / "shadow_test.parquet").astype("float32")
    _bind(td / "shadow_test_score.parquet", model.predict(X_sh),
          ident["shadow_test"], run)
    # 基线:常数均值(train-only),shadow + final_test 分数
    y_mean = float(y_tr.mean())
    bd = run / "trials" / BASE
    bd.mkdir(parents=True)
    rng = np.random.default_rng(5)
    _bind(bd / "shadow_test_score.parquet",
          y_mean + rng.normal(0, 1, SIZES["shadow_test"]),
          ident["shadow_test"], run)
    (run / "final-model").mkdir()
    _bind(run / "final-model" / "baseline_final_test_score.parquet",
          y_mean + rng.normal(0, 1, SIZES["final_test"]),
          ident["final_test"], run)
    (run / "run-state.json").write_text(json.dumps({"baseline_id": BASE}))
    (run / "leaderboard").mkdir()
    (run / "leaderboard" / "leaderboard.json").write_text(json.dumps({
        "primary_metric": "rmse", "baseline_id": BASE, "rows": [
            {"trial_id": CHAMP, "rmse": 500.0, "verdict": "confirmed"},
            {"trial_id": BASE, "rmse": 520.0, "verdict": "accepted"}]}))
    info.update(run=run, champ_dir=td, n_tr=SIZES["train"],
                n_va=SIZES["validation"])
    return run, info


class _FakeTrial:
    def __init__(self, tid=CHAMP, parent=PARENT):
        self.trial_id = tid
        self.parent_ids = [parent]
        self.data = {"evaluation": {
            "verdict": "confirmed", "fresh_confirmation": {"passed": True},
            "integrity": {"valid": True}}}


def _install_fakes(monkeypatch, run, info, spec=None):
    import runtime.data as rdata
    import runtime.integrity as rint
    import runtime.leaderboard as rlb
    import runtime.topology as rtopo
    import runtime.trial as rtr

    champ = _FakeTrial()
    res = {"status": "bound", "search_champion": CHAMP,
           "search_recipe": PARENT, "delivery_trial": CHAMP, "gap": 0.0,
           "tolerance": 1.0}

    class FakeLB:
        def __init__(self, *a, **k):
            self.last_delivery_resolution = dict(res)

        def build(self, *a, **k):
            return None

        def best_trial(self, trials, delivery=False):
            return champ

        def delivery_resolution(self, trials):
            return dict(res)

    class FakeFactory:
        def __init__(self, *a, **k):
            pass

        def all_trials(self):
            return [champ]

    for mod in (rlb, cmd_deliver):
        monkeypatch.setattr(mod, "Leaderboard", FakeLB)
    for mod in (rtr, cmd_deliver):
        monkeypatch.setattr(mod, "TrialFactory", FakeFactory)
    monkeypatch.setattr(cmd_deliver, "trial_currently_failed", lambda t: False)
    monkeypatch.setattr(cmd_deliver, "_load_spec",
                        lambda args: spec or _spec())
    monkeypatch.setattr(cmd_deliver, "_delivery_divergence_check",
                        lambda lb, trials, cid: (True, {
                            "resolution": dict(res), "bound": True,
                            "champion_recipe": PARENT}))
    monkeypatch.setattr(rtopo, "evaluate_gate",
                        lambda *a, **k: {"allowed": True, "kind": "single"})
    monkeypatch.setattr(rint, "verify_trial_source",
                        lambda d: {"combined_sha256": "src-sha"})
    monkeypatch.setattr(rint, "verify_confirmation_seed",
                        lambda *a, **k: None)
    monkeypatch.setattr(rdata, "load_final_test_labels",
                        lambda spec, run_dir: info["final_labels"].copy())
    monkeypatch.setenv("MODELEVO_GPU", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.delenv("MODELEVO_ES_SOURCE", raising=False)
    monkeypatch.setenv("MVO_TEST_CALL_LOG", str(run / "calls.jsonl"))


def _retrain_args(run, **kw):
    base = dict(run=str(run), spec=None, champion=None, snapshot_only=False,
                ack_divergence=None, refit_mode=None, full_reproduction=False,
                detach=False, no_detach=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _calls(run) -> list[dict]:
    p = run / "calls.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _meta(run) -> dict:
    return json.loads((run / "final-model" / "model_meta.json").read_text(
        encoding="utf-8"))


def _summary(out: str) -> dict:
    line = next(x for x in out.splitlines()
                if x.startswith("FINALIZE_RETRAIN_SUMMARY\t"))
    return json.loads(line.split("\t", 1)[1])


# ---------------------------------------------------------------------------
# 1. #5 refit 口径:先探测再复现,回落留痕
# ---------------------------------------------------------------------------
def test_clone_probe_failure_falls_back_without_reproduction(
        tmp_path, monkeypatch, capsys):
    run, info = _make_run(tmp_path, kind="plain")
    _install_fakes(monkeypatch, run, info)
    rc = cmd_deliver.cmd_finalize_retrain(_retrain_args(run))
    assert rc == 0
    calls = _calls(run)
    # 只调用一次冠军 fit_and_score:train+validation 复跑(train_tail 早停),
    # 没有 train 上的完整复现(run4 白跑 10.4 min 的那一步)
    assert calls == [{"n": info["n_tr"] + info["n_va"], "es": "train_tail"}], calls
    meta = _meta(run)
    assert meta["refit_mode_configured"] == "rerun_recipe"
    assert meta["refit_mode_requested"] == "frozen_rounds"
    assert meta["refit_mode_effective"] == "rerun_recipe" == meta["refit_mode"]
    assert meta["refit_mode_fallback"] is True
    assert meta["refit_mode_fallback_reason"].startswith("clone_probe_failed")
    assert "get_params" in meta["refit_mode_fallback_reason"]
    assert meta["clone_probe"]["status"] == "error"
    assert meta["clone_probe"]["model_class"].endswith(".HurdleXGB")
    assert meta["reproduction"] is None and meta["face_check"]["passed"] is True
    assert meta["retrain_builder"] == "champion_fit_and_score_rerun"
    assert meta["es_holdout"]["first_pday"], "回落后早停面 = train+validation 尾部"
    # generated_by 读 runtime.__version__,不再硬编码
    assert meta["generated_by"] == \
        f"modelevo.py finalize-retrain (v{SKILL_VERSION})"
    assert meta["skill_version"] == SKILL_VERSION
    recipe = yaml.safe_load((run / "final-model" / "recipe.yaml").read_text(
        encoding="utf-8"))
    assert recipe["refit_mode_requested"] == "frozen_rounds"
    assert recipe["refit_mode_effective"] == "rerun_recipe"
    assert recipe["refit_mode_fallback_reason"].startswith("clone_probe_failed")
    # stdout 摘要同样体现
    out = capsys.readouterr().out
    s = _summary(out)
    assert s["refit_mode_requested"] == "frozen_rounds"
    assert s["refit_mode_effective"] == "rerun_recipe"
    assert s["refit_mode_fallback"] is True and s["clone_probe"] == "error"
    assert s["generated_by"] == meta["generated_by"]
    assert "[refit] 交付重训口径回落" in out
    # 交付模型在 shadow 上的水位(回归):交付 vs 冠军 train-only 搜索模型
    lv = meta["shadow_precheck"]["level_stats"]
    for k in ("delivery_model", "search_model"):
        assert {"pred_mean", "y_mean", "level_ratio", "n"} <= set(lv[k])
        assert lv[k]["n"] == SIZES["shadow_test"]
    assert lv["delivery_model"]["y_mean"] == pytest.approx(
        lv["search_model"]["y_mean"])


def test_probe_loads_main_pickles_only_via_champion_module(tmp_path):
    from _mvo.cmd_deliver import _load_champion_model, _probe_champion_clone

    run, info = _make_run(tmp_path, kind="plain")
    td = info["champ_dir"]
    m, err = _load_champion_model(td, None)
    assert m is None and "HurdleXGB" in err, "trial pickle 引用 __main__"
    mod = _load_module(td / "code" / "train.py", "wp3_probe_mod")
    main_before = sys.modules["__main__"]
    rep = _probe_champion_clone(td, mod)
    assert sys.modules["__main__"] is main_before, "__main__ 必须恢复"
    assert rep["status"] == "error" and rep["source"] == "model/model.joblib"
    (td / "model" / "model.joblib").unlink()
    assert _probe_champion_clone(td, mod)["status"] == "unavailable"


def test_clonable_composite_uses_frozen_rounds(tmp_path, monkeypatch, capsys):
    pytest.importorskip("xgboost")
    run, info = _make_run(tmp_path, kind="sk")
    _install_fakes(monkeypatch, run, info)
    rc = cmd_deliver.cmd_finalize_retrain(_retrain_args(run))
    assert rc == 0
    meta = _meta(run)
    assert meta["clone_probe"]["status"] == "ok"
    assert meta["refit_mode_requested"] == meta["refit_mode_effective"] \
        == "frozen_rounds"
    assert meta["refit_mode_fallback"] is False
    assert meta["refit_mode_fallback_reason"] is None
    assert meta["retrain_builder"] == "champion_fit_and_score"
    assert meta["reproduction"]["passed"] is True
    fz = meta["frozen_rounds_params"]
    assert fz["clf__early_stopping_rounds"] is None
    assert fz["reg__early_stopping_rounds"] is None
    assert 1 <= fz["clf__n_estimators"] <= 300
    assert 1 <= fz["reg__n_estimators"] <= 300
    # v2.19.0:冻结模型 validation 重打分逐位一致 → 复用,不再在 train 上
    # 复现(冠军代码零调用);交付拟合仍走 clone + 冻结轮数
    assert meta["reproduction_mode"] == "reused_frozen_model"
    assert _calls(run) == []
    model = joblib.load(run / "final-model" / "model" / "model.joblib")
    assert model.clf_.get_booster().num_boosted_rounds() == \
        fz["clf__n_estimators"], "train+validation 上按冻结轮数重训(无早停)"
    assert model.clf.get_params()["early_stopping_rounds"] is None
    assert _summary(capsys.readouterr().out)["clone_probe"] == "ok"


def test_train_tail_champion_frozen_rounds_fails_fast(tmp_path, monkeypatch):
    run, info = _make_run(tmp_path, kind="plain", es_source="train_tail")
    _install_fakes(monkeypatch, run, info)
    rc = cmd_deliver.cmd_finalize_retrain(
        _retrain_args(run, refit_mode="frozen_rounds"))
    assert rc == 1
    assert _calls(run) == [], "clone 探测失败须在任何复现/重训之前报错"
    assert not (run / "final-model" / "model_meta.json").exists()


def test_probe_unavailable_keeps_old_order_but_records_fallback(
        tmp_path, monkeypatch):
    run, info = _make_run(tmp_path, kind="plain", with_model=False)
    _install_fakes(monkeypatch, run, info)
    assert cmd_deliver.cmd_finalize_retrain(_retrain_args(run)) == 0
    calls = _calls(run)
    assert calls[0] == {"n": info["n_tr"], "es": "validation"}, "旧序:先复现"
    assert calls[-1] == {"n": info["n_tr"] + info["n_va"], "es": "train_tail"}
    meta = _meta(run)
    assert meta["clone_probe"]["status"] == "unavailable"
    assert meta["refit_mode_effective"] == "rerun_recipe"
    assert meta["refit_mode_fallback_reason"].startswith(
        "clone_failed_after_reproduction")


def test_official_hurdle_regressor_is_clonable_if_present():
    """WP4 的 runtime.hurdle.HurdleRegressor 落地后:可 clone ⇒ frozen_rounds 可用。"""
    hurdle = pytest.importorskip("runtime.hurdle")
    cls = getattr(hurdle, "HurdleRegressor", None)
    if cls is None:
        pytest.skip("runtime.hurdle 无 HurdleRegressor")
    try:
        est = cls()
    except TypeError:
        pytest.skip("HurdleRegressor 无缺省构造")
    from _mvo.cmd_deliver import _clone_probe

    assert _clone_probe(est)["status"] == "ok"


def test_official_hurdle_regressor_frozen_rounds_end_to_end(
        tmp_path, monkeypatch):
    """官方容器(clf/reg=None,按 clf_params/reg_params 构造段)的冻结轮数重训。"""
    pytest.importorskip("xgboost")
    hurdle = pytest.importorskip("runtime.hurdle")
    if not hasattr(hurdle, "HurdleRegressor"):
        pytest.skip("runtime.hurdle 无 HurdleRegressor")
    run, info = _make_run(tmp_path, kind="official")
    _install_fakes(monkeypatch, run, info)
    assert cmd_deliver.cmd_finalize_retrain(_retrain_args(run)) == 0
    meta = _meta(run)
    assert meta["clone_probe"]["status"] == "ok"
    assert meta["refit_mode_effective"] == "frozen_rounds"
    assert meta["refit_mode_fallback_reason"] is None
    assert meta["estimator_class"] == "runtime.hurdle.HurdleRegressor"
    fz = meta["frozen_rounds_params"]
    assert set(fz) >= {"clf__n_estimators", "reg__n_estimators"}
    # v2.19.0:复用冻结模型,不在 train 上复现(--full-reproduction 见
    # tests/test_v2_19_speed.py 的两路等价性用例)
    assert _calls(run) == []
    model = joblib.load(run / "final-model" / "model" / "model.joblib")
    for seg in ("clf", "reg"):
        booster = getattr(model, f"{seg}_").get_booster()
        assert booster.num_boosted_rounds() == fz[f"{seg}__n_estimators"] < 300, \
            f"{seg} 段须按冻结轮数重训,不得训到 cap"


def test_composite_frozen_params_protocol():
    from sklearn.base import BaseEstimator

    from _mvo.cmd_deliver import _composite_frozen_params

    class Sub(BaseEstimator):
        def __init__(self, n_estimators=100, early_stopping_rounds=5):
            self.n_estimators = n_estimators
            self.early_stopping_rounds = early_stopping_rounds

        def fit(self, X, y):
            return self

    class Box(BaseEstimator):
        def __init__(self, a=None, b=None, early_stopping_rounds=3):
            self.a, self.b = a, b
            self.early_stopping_rounds = early_stopping_rounds

    box = Box(Sub(), Sub())
    fa, fb = Sub(), Sub()
    fa.mvo_es_info_ = {"stage1_best_iteration": 17}
    fb.mvo_es_info_ = {"stage1_best_iteration": 9}
    box.a_, box.b_ = fa, fb
    out = _composite_frozen_params(box, "validation")
    assert out == {"a__n_estimators": 17, "a__early_stopping_rounds": None,
                   "b__n_estimators": 9, "b__early_stopping_rounds": None,
                   "early_stopping_rounds": None}
    box.set_params(**out)           # 嵌套键合法
    assert box.a.n_estimators == 17 and box.b.early_stopping_rounds is None
    assert _composite_frozen_params(Sub(), "validation") == {}, "单模型不产出"
    assert _composite_frozen_params(object(), "validation") == {}


def test_source_contracts():
    body = _SRC.split("def cmd_finalize_retrain(")[1].split(
        "\ndef cmd_finalize_baseline_score(")[0]
    assert body.index("_probe_champion_clone(") < body.index(
        "_reproduce_champion_fit("), "clone 探测必须在复现之前"
    assert body.index("_probe_champion_clone(") < body.index(
        "feats = champ.exclude_columns("), "探测在特征构造之前(最小代价)"
    assert "(v2.15.9)\"" not in _SRC and "_skill_version()" in body
    ev = _SRC.split("def cmd_evaluate_final(")[1].split("\ndef ")[0]
    i_gate = ev.index('gate.acquire(purpose="final_model_evaluation")')
    assert ev.index("_delivery_regression_diagnostics(") > i_gate
    assert 'if spec.model_type == "regression":' in ev, "只对回归生效"


# ---------------------------------------------------------------------------
# 2. #20 回归交付诊断
# ---------------------------------------------------------------------------
def test_evaluate_final_regression_diagnostics_without_label_leak(
        tmp_path, monkeypatch, capsys):
    run, info = _make_run(tmp_path, kind="plain")
    _install_fakes(monkeypatch, run, info)
    assert cmd_deliver.cmd_finalize_retrain(_retrain_args(run)) == 0
    capsys.readouterr()
    before = {p: p.stat().st_mtime_ns for p in run.rglob("*") if p.is_file()}
    args = types.SimpleNamespace(
        run=str(run), spec=None, best_trial=None, ack_divergence=None,
        ack_shadow_regression="test", segment_inverse_transform=None,
        allow_no_shadow=False)
    assert cmd_deliver.cmd_evaluate_final(args) == 0
    out = capsys.readouterr().out
    ev = json.loads((run / "final-model" / "evaluation.json").read_text(
        encoding="utf-8"))
    rd = ev["regression_diagnostics"]
    assert rd["period_column"] == "pday"
    ft = rd["final_test"]
    lv = ft["delivery"]["level"]
    assert lv["n"] == SIZES["final_test"]
    assert lv["level_ratio"] > 1.1 and lv["level_shift_warning"] is True
    assert [r["period"] for r in ft["delivery"]["level_by_period"]] == [
        "2026-07-01", "2026-07-08", "2026-07-15", "2026-07-22"]
    assert "level" in ft["baseline"] and "vs_baseline_divergence" in ft[
        "delivery"]
    sh = rd["shadow_test"]
    assert {"delivery", "search_model", "baseline"} <= set(sh)
    assert len(sh["delivery"]["level_by_period"]) == 4
    assert rd["level_shift_warning"] is True and "final_test" in rd[
        "level_shift_splits"]
    assert "排序可用、绝对金额需校准" in rd["advice"]
    assert "排序可用、绝对金额需校准" in out and "[回归水位] final_test" in out
    # ---- 标签零落盘:新文件只有评测汇总产物,任何 parquet 都不是新写的;
    # evaluation.json 里没有逐行数组,也没有 final_test 标签原值
    after = {p: p.stat().st_mtime_ns for p in run.rglob("*") if p.is_file()}
    changed = {str(p.relative_to(run)) for p in after
               if p not in before or after[p] != before[p]}
    assert changed <= {"final-model/evaluation.json",
                       "final-model/segment-report.xlsx",
                       "final-model/segment-report.md",
                       "final-test-gate.json", "run-state.json",
                       "run-state.json.lock",
                       # v2.17.2(B5):evaluate-final 计时(无标签内容)
                       "logs/evaluate-final.json"}, changed
    assert not any(p.suffix == ".parquet" for p in map(pathlib.Path, changed))
    assert not (run / "sample-profile" / "splits" / "labels"
                / "final_test.parquet").exists()

    def _max_list(o):
        if isinstance(o, dict):
            return max([_max_list(v) for v in o.values()] or [0])
        if isinstance(o, list):
            return max([len(o)] + [_max_list(v) for v in o])
        return 0

    assert _max_list(rd) <= 4, "诊断只含分期统计量,不含逐行数组"
    text = (run / "final-model" / "evaluation.json").read_text(encoding="utf-8")
    ys = info["final_labels"]["y"].values
    for v in ys[ys > 0][:20]:
        assert repr(float(v)) not in text


def test_shadow_precheck_level_stats_regression_only(tmp_path):
    from _mvo.cmd_deliver import _shadow_precheck

    run = tmp_path / "r"
    lab_dir = run / "sample-profile" / "splits" / "labels"
    lab_dir.mkdir(parents=True)
    cd = run / "trials" / "t"
    cd.mkdir(parents=True)
    fm = run / "final-model"
    fm.mkdir()
    rng = np.random.default_rng(1)
    n = 2000
    keys = np.array([f"k{i}" for i in range(n)])
    y = rng.gamma(2.0, 100.0, size=n)
    pd.DataFrame({"y": y, SAMPLE_KEY_COLUMN: keys}).to_parquet(
        lab_dir / "shadow_test.parquet")
    pd.DataFrame({"score": y * 1.25 + rng.normal(0, 5, n),
                  SAMPLE_KEY_COLUMN: keys}).to_parquet(
                      fm / "shadow_test_score.parquet")
    pd.DataFrame({"score": y + rng.normal(0, 5, n),
                  SAMPLE_KEY_COLUMN: keys}).to_parquet(
                      cd / "shadow_test_score.parquet")
    spec = types.SimpleNamespace(primary_metric="rmse", target_name="y",
                                 model_type="regression")
    rep = _shadow_precheck(run, fm, cd, spec, n_boot=200)
    d, s = rep["level_stats"]["delivery_model"], rep["level_stats"][
        "search_model"]
    assert d["level_ratio"] == pytest.approx(1.25, abs=0.02)
    assert d["level_shift_warning"] is True and rep["level_shift_warning"]
    assert s["level_ratio"] == pytest.approx(1.0, abs=0.02)
    assert d["y_mean"] == pytest.approx(float(y.mean()))
    spec_c = types.SimpleNamespace(primary_metric="rmse", target_name="y",
                                   model_type="binary_classification")
    assert "level_stats" not in _shadow_precheck(run, fm, cd, spec_c,
                                                 n_boot=50)


def test_period_labels_and_diag_helper():
    from _mvo.cmd_deliver import (_delivery_regression_diagnostics,
                                  _level_hint_lines, _period_labels)

    days = pd.date_range("2026-07-01", periods=4, freq="7D").strftime(
        "%Y-%m-%d")
    assert list(_period_labels(np.repeat(days.values, 2))[:3]) == [
        "2026-07-01", "2026-07-01", "2026-07-08"]
    long = pd.date_range("2026-01-01", periods=90, freq="D")
    assert sorted(set(_period_labels(long.values))) == [
        "2026-01", "2026-02", "2026-03"]
    assert list(_period_labels(np.array([20260701, 20260708]))) == [
        "2026-07-01", "2026-07-08"]
    assert list(_period_labels(np.array(["A", "B"]))) == ["A", "B"]
    assert _period_labels(None) is None

    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 50.0, 400)
    per = np.repeat(["p1", "p2"], 200)
    out = _delivery_regression_diagnostics({
        "final_test": {"y": y, "periods": per,
                       "models": {"delivery": y * 1.02, "baseline": None}},
        "shadow_test": {"y": None}}, period_column="pday")
    assert set(out["final_test"]) == {"delivery"}, "缺基线只出交付模型"
    assert "vs_baseline_divergence" not in out["final_test"]["delivery"]
    assert out["level_shift_warning"] is False and out["advice"] is None
    assert _level_hint_lines(out) == []
    y2 = y.copy()
    y2[200:] *= 0.5                           # 第二期水位下移
    out2 = _delivery_regression_diagnostics({"final_test": {
        "y": y2, "periods": per, "models": {"delivery": y, "baseline": y2}}})
    assert out2["final_test"]["delivery"]["period_level_shift"] is True
    hint = _level_hint_lines(out2)
    assert len(hint) == 1 and "p2=2.000" in hint[0] and "绝对金额需校准" in hint[0]


# ---------------------------------------------------------------------------
# 3. #21 报告「模型局限」
# ---------------------------------------------------------------------------
def _lim_run(tmp_path, *, legacy_meta=False) -> pathlib.Path:
    run = tmp_path / "lim"
    (run / "final-model").mkdir(parents=True)
    (run / "leaderboard").mkdir()
    ids = {"deliv": CHAMP, "p1": "trial-025-g11-exploit-xgboost-1712",
           "p2": "trial-020-g06-crossover-xgboost-620",
           "worse": "trial-017-g04-explore-xgboost-354",
           "s1": "trial-030-g12-explore-xgboost-1"}
    (run / "leaderboard" / "leaderboard.json").write_text(json.dumps({
        "baseline_id": BASE,
        "delivery_resolution": {"search_champion": CHAMP,
                                "delivery_trial": CHAMP},
        "rows": [
            {"trial_id": ids["p1"], "rmse": 12623.49, "verdict": "promising",
             "data_view": "v1"},
            {"trial_id": ids["p2"], "rmse": 12632.22, "verdict": "promising",
             "data_view": "v1"},
            {"trial_id": CHAMP, "rmse": 12681.56, "verdict": "confirmed",
             "data_view": "v1"},
            {"trial_id": ids["worse"], "rmse": 12682.46,
             "verdict": "promising", "data_view": "v1"},
            {"trial_id": ids["s1"], "rmse": 12900.0, "verdict": "rejected",
             "data_view": "sample"}]}))
    (run / "final-model" / "evaluation.json").write_text(json.dumps({
        "best_trial": CHAMP,
        "regression_diagnostics": {"final_test": {
            "delivery": {"level": {"n": 567573, "y_mean": 1965.3,
                                   "pred_mean": 2350.1,
                                   "level_ratio": 1.1958,
                                   "level_shift_warning": True},
                         "level_by_period": [
                             {"period": "2026-07-07", "level_ratio": 1.18,
                              "level_shift_warning": True}],
                         "vs_baseline_divergence": {
                             "diverged": True, "primary": "rmse",
                             "secondary": "mae", "delta_primary": -162.6,
                             "delta_secondary": 12.0}},
            "baseline": {"level": {"level_ratio": 1.15}}}}}))
    meta = ({"best_trial": CHAMP, "refit_mode": "rerun_recipe",
             "es_source": "validation"} if legacy_meta else
            {"best_trial": CHAMP, "refit_mode": "rerun_recipe",
             "refit_mode_requested": "frozen_rounds",
             "refit_mode_effective": "rerun_recipe",
             "refit_mode_fallback_reason": "clone_probe_failed: TypeError",
             "clone_probe": {"status": "error", "error": "TypeError: x"}})
    (run / "final-model" / "model_meta.json").write_text(json.dumps(meta))
    td = run / "trials" / CHAMP
    td.mkdir(parents=True)
    (td / "metrics.json").write_text(json.dumps({
        "es_info": {"cap": 2000, "cap_bound": True, "stopped_by": "cap",
                    "es_source": "validation"},
        "hurdle": {"clf_best_iteration": 1999, "reg_best_iteration": 787,
                   "clf_cap_bound": True, "reg_cap_bound": False}}))
    for tid, div in (("trial-022-g08-exploit-xgboost-902", True),
                     ("trial-023-g09-exploit-xgboost-1212", True),
                     ("trial-024-g10-exploit-xgboost-1420", False)):
        d = run / "trials" / tid
        d.mkdir(parents=True)
        (d / "trial.yaml").write_text(yaml.safe_dump({
            "trial_id": tid, "evaluation": {"regression_diagnostics": {
                "vs_parent_divergence": {"diverged": div, "primary": "rmse",
                                         "secondary": "mae"}}}}))
    (run / "evolution-state.json").write_text(json.dumps({"stopped": {
        "stop_reason": "roi_diminishing_return + single_data_view_insufficient",
        "data_views_tried": ["v1"], "non_sample_data_views": ["v1"]}}))
    (run / "run-state.json").write_text(json.dumps({
        "baseline_id": BASE, "harness_change_log": [
            {"at": "2026-10-23T00:18:49+08:00", "changed": ["config.py"],
             "measurement_impact": "no-score-impact"},
            {"at": "2026-10-23T08:55:22+08:00",
             "changed": ["__init__.py", "feature_cache.py"],
             "measurement_impact": "no-score-impact"}]}))
    return run


def test_collect_limitations_all_sources(tmp_path):
    from runtime.report import collect_limitations

    run = _lim_run(tmp_path)
    lims = collect_limitations(run, _spec())
    blob = "\n".join(lims)
    # (a) 水位
    assert "交付期水位偏差(final_test)" in blob and "1.196" in blob
    assert "排序可用、绝对金额需校准" in blob and "基线水位比 1.150" in blob
    # (b) 撞 cap(整体 + hurdle 子模型)
    assert "撞树数上限" in blob and "hurdle.clf 子模型最优轮数 1999" in blob
    assert "hurdle.reg" not in blob
    # (c) 未交付候选:只列比交付位更优的 promising,注明证据性质
    assert "仅 validation 证据、含选择偏差" in blob
    assert "trial-025-g11-exploit-xgboost-1712" in blob
    assert "trial-020-g06-crossover-xgboost-620" in blob
    assert "trial-017-g04-explore-xgboost-354" not in blob
    # (d) 单视图 (e) 抽样面
    assert "只试过单一数据视图(v1)" in blob
    assert "1 个 trial 在抽样面(data_view=sample)上裁决" in blob
    # (f) 背离:trial 级 2 个 + 交付 vs 基线
    assert "2 个 trial 的 rmse 与 mae 相对父代方向相反" in blob
    assert "trial-024" not in blob.split("相对父代方向相反")[1].split("——")[0]
    assert "交付模型 vs 基线(final_test)" in blob
    # (g) harness 变更 (h) refit 回落
    assert "运行中登记了 2 次 harness 变更" in blob
    assert "交付重训口径回落:请求 frozen_rounds,实际 rerun_recipe" in blob


def _ug(**kw) -> dict:
    """WP1 runtime.evolution.compute_undelivered_gain 的输出形态。"""
    out = {"anchor_id": CHAMP, "anchor_source": "delivery_resolution",
           "reference": "parent", "mde": 10.3, "best_trial": None,
           "delta": None, "gain": None, "se": None, "z": None,
           "p_better": None, "source": None, "gain_in_mde": None,
           "meets_mde": False, "top_candidates": []}
    out.update(kw)
    return out


def test_collect_limitations_undelivered_gain_and_legacy_meta(tmp_path):
    from runtime.report import collect_limitations

    run = _lim_run(tmp_path, legacy_meta=True)
    st = json.loads((run / "evolution-state.json").read_text())

    def _blob(ug):
        st["stopped"]["undelivered_gain"] = ug
        (run / "evolution-state.json").write_text(json.dumps(st))
        return "\n".join(collect_limitations(run, _spec()))

    blob = _blob(_ug(
        best_trial="trial-025-g11-exploit-xgboost-1712", delta=-58.07,
        gain=58.07, se=14.6, z=3.98, gain_in_mde=5.64, meets_mde=True,
        source="vs_anchor", action_hint="以 anchor 协议重开或手工 confirm",
        top_candidates=[
            {"trial_id": "trial-025-g11-exploit-xgboost-1712", "gain": 58.07},
            {"trial_id": "trial-020-g06-crossover-xgboost-620", "gain": 49.3},
            {"trial_id": "trial-099", "gain": -3.0}]))
    assert "stop_report.undelivered_gain" in blob
    assert f"trial-025-g11-exploit-xgboost-1712 相对交付锚点 {CHAMP}" in blob
    assert "z=3.98" in blob and "5.6×MDE" in blob
    assert "其余候选 trial-020-g06-crossover-xgboost-620" in blob
    assert "trial-099" not in blob and "手工 confirm" in blob
    assert "leaderboard 中" not in blob, "有 undelivered_gain 时不再回退 leaderboard"
    assert "由旧版 model_meta 推断" in blob
    # runtime 已算出"锚点之上无更优 trial" → 不报(c),也不回退 leaderboard
    blob = _blob(_ug())
    assert "未交付的更优候选" not in blob
    blob = _blob(_ug(best_trial="trial-x", delta=3.0, gain=-3.0))
    assert "未交付的更优候选" not in blob
    # 锚点缺失(无法计算)→ 回退 leaderboard 的 promising 列表
    blob = _blob(_ug(anchor_id=None, note="锚点缺失"))
    assert "leaderboard 中 2 个 promising" in blob


def test_report_section14_fill_and_default(tmp_path):
    from runtime.report import ReportBuilder

    cfg = Config.load(skill_config=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    md = ReportBuilder(empty, _spec(), cfg).build().read_text(encoding="utf-8")
    sec = md.split("## 14. 模型局限")[1].split("## 15.")[0]
    assert sec.strip() == "- 未记录", "全部来源缺失时才显示未记录"

    run = _lim_run(tmp_path)
    st = json.loads((run / "evolution-state.json").read_text())
    st["stopped"].update(
        advisory=True, blocks=[],
        reason_detail={"roi_diminishing_return": "代级 ROI 低于历史最佳"},
        undelivered_gain=_ug(best_trial="trial-025-g11-exploit-xgboost-1712",
                             delta=-58.0, gain=58.0))
    (run / "evolution-state.json").write_text(json.dumps(st))
    md = ReportBuilder(run, _spec(), cfg).build().read_text(encoding="utf-8")
    sec = md.split("## 14. 模型局限")[1].split("## 15.")[0]
    assert "未记录" not in sec
    for needle in ("交付期水位偏差", "撞树数上限", "未交付的更优候选",
                   "只试过单一数据视图", "抽样面", "主/次指标背离",
                   "harness 变更", "交付重训口径回落"):
        assert needle in sec, needle
    s12 = md.split("## 12.")[1].split("## 13.")[0]
    assert "口径回落: frozen_rounds → rerun_recipe" in s12
    assert "回归水位(final_test)" in s12 and "绝对金额需校准" in s12
    s15 = md.split("## 15. stop reason")[1].split("## 16.")[0]
    assert "advisory" in s15 and "停止原因明细" in s15
    assert "`roi_diminishing_return`: 代级 ROI 低于历史最佳" in s15
    assert "未交付增益" in s15 and "trial-025-g11-exploit-xgboost-1712" in s15


def test_collect_limitations_tolerates_broken_sources(tmp_path):
    from runtime.report import collect_limitations

    run = _lim_run(tmp_path)
    (run / "trials" / CHAMP / "metrics.json").write_text("{not json")
    lims = collect_limitations(run, _spec())
    assert any("交付期水位偏差" in x for x in lims), "单来源坏掉不影响其余"
    assert not any("撞树数上限" in x for x in lims)


def test_journey_html_limitations_and_effective_mde(tmp_path):
    from _mvo.report_html import collect_journey_data

    run = _lim_run(tmp_path)
    # MDE:relative × |baseline 主指标| 优于绝对值(量纲感知)
    bd = run / "trials" / BASE
    bd.mkdir(parents=True)
    (bd / "evaluation.json").write_text(json.dumps(
        {"primary_metric": {"value": 12932.47}}))
    cfg = types.SimpleNamespace(get=lambda k, d=None: {
        "evaluator.minimum_meaningful_improvement": 0.001,
        "evaluator.minimum_meaningful_improvement_relative": 0.0008}.get(k, d))
    data = collect_journey_data(run, _spec(), cfg)
    assert data["mde"] == pytest.approx(0.0008 * 12932.47)
    assert data["limitations"] and any(
        "交付期水位偏差" in x for x in data["limitations"])
    assert any(n.startswith("模型局限 · ") for n in data["notes"])
    assert data["final"]["refit_mode_fallback_reason"].startswith(
        "clone_probe_failed")
    assert data["final"]["regression_diagnostics"]["final_test"]
