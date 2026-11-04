"""v2.15.9 交付重训复跑配方回归测试。

事故来源:req-t7 run4 regression(2026-10-21)。搜索冠军是两段式 hurdle 容器
`HurdleXGB`(非 sklearn estimator),旧 finalize-retrain 在 train 上复现 40 min
后于 `sklearn.clone` 处 TypeError。说明见
`docs/MODELEVO_V2_15_9_DELIVERY_RERUN_20261021.md`。

锁定的行为:

1. rerun_recipe:以 train+validation 调冠军 fit_and_score,早停落在传入帧的
   最新尾部;返回模型直接交付,不 clone(复合容器可用);
2. HPO 冠军先剥 hpo 节、注入 best_params,交付期不重跑搜索;
3. 特征面一致性校验:一致通过、列序/数值失配 fail-fast、缺工件 → None;
4. frozen_rounds 下 clone 失败报错且不回落估计器适配器;
5. 回归任务交付打分不走 predict_proba;
6. shadow 预检方向感知;evaluate-final 在 gate 前执行;
7. 原值尺度分桶报表;交付配方副本 sha 校验;CLI 参数。
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

_SRC = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
    encoding="utf-8")


def _retrain_body() -> str:
    return _SRC.split("def cmd_finalize_retrain(")[1].split(
        "\ndef cmd_finalize_baseline_score(")[0]


def _eval_body() -> str:
    return _SRC.split("def cmd_evaluate_final(")[1].split("\ndef ")[0]


class _Hurdle:
    """非 sklearn 复合容器:无 get_params / fit;回归分只走 predict。"""

    def __init__(self, a, b, stage1):
        self.a, self.b, self.stage1_ = a, b, stage1

    def predict(self, X):
        return np.asarray(X.iloc[:, 0]) * self.a + self.b


def _fake_regression_champion(tmp_path: pathlib.Path, *, n_tr=800, n_va=200):
    """手写回归冠军:fit_and_score 在传入 X_tr 尾部 10% '早停'(记录切点)。"""
    tdir = tmp_path / "trials" / "trial-031-g08-exploit-xgboost-1"
    (tdir / "model").mkdir(parents=True)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f0": rng.normal(size=n_tr + n_va),
                      "f1": rng.normal(size=n_tr + n_va)})
    y = pd.Series(2.0 * X["f0"].values + rng.normal(scale=0.1,
                                                    size=len(X)))
    X_tr, X_va = X.iloc[:n_tr].reset_index(drop=True), \
        X.iloc[n_tr:].reset_index(drop=True)
    y_tr, y_va = y.iloc[:n_tr].reset_index(drop=True), \
        y.iloc[n_tr:].reset_index(drop=True)

    champ = types.ModuleType("fake_champ_rerun")
    champ.TRIAL_DIR = tdir
    champ.ES_FRACTION = 0.1
    champ._last_model = None
    champ.calls = []

    def fit_and_score(X_tr_, y_tr_, X_va_, y_va_=None, seed=0, cfg=None):
        cut = int(len(X_tr_) * (1 - champ.ES_FRACTION))
        champ.calls.append({"n": len(X_tr_), "cut": cut,
                            "cfg": dict(cfg or {})})
        # TRIAL_DIR 写入:沙箱必须接住
        (pathlib.Path(champ.TRIAL_DIR) / "touched.txt").write_text("x")
        a = float(np.polyfit(X_tr_["f0"].iloc[:cut], y_tr_.iloc[:cut], 1)[0])
        model = _Hurdle(a, 0.0, {"clf_best_iteration": cut,
                                 "reg_best_iteration": cut // 2,
                                 "n_fit_rows": cut,
                                 "n_es_rows": len(X_tr_) - cut})
        champ._last_model = model
        return model.predict(X_va_), {"model_num": 2}

    champ.fit_and_score = fit_and_score
    spec = types.SimpleNamespace(primary_metric="rmse", model_type="regression",
                                 target_name="y",
                                 data={"secondary_metrics": ["mae"]})
    return champ, tdir, (X_tr, y_tr, X_va, y_va), spec


# ---------------------------------------------------------------------------
# 1. rerun_recipe:train+validation 复跑,尾部早停,不 clone
# ---------------------------------------------------------------------------
def test_rerun_recipe_trains_on_train_plus_validation(tmp_path):
    from _mvo.cmd_deliver import _rerun_champion_recipe

    champ, tdir, (X_tr, y_tr, X_va, y_va), _ = _fake_regression_champion(
        tmp_path)
    X_all = pd.concat([X_tr, X_va], ignore_index=True)
    y_all = pd.concat([y_tr, y_va], ignore_index=True)
    model, rep = _rerun_champion_recipe(
        champ, X_all, y_all, X_va, y_va, 7, {"multi_seed": 3}, tdir)
    call = champ.calls[-1]
    assert call["n"] == len(X_tr) + len(X_va)
    # 尾部 10% 早停面 = 最新的 100 行,全部落在 validation 段
    assert len(X_all) - call["cut"] == 100
    assert call["cut"] >= len(X_tr)
    assert "multi_seed" not in call["cfg"]
    assert isinstance(model, _Hurdle), "复合容器应被原样交付(不 clone)"
    assert rep["n_fit_rows"] == len(X_all)
    assert rep["stage1"]["n_es_rows"] == 100
    assert champ.TRIAL_DIR == tdir, "TRIAL_DIR 未恢复"
    assert not (tdir / "touched.txt").exists(), "复跑写入落到了真实 trial 目录"


def test_rerun_recipe_rejects_missing_last_model(tmp_path):
    from _mvo.cmd_deliver import _rerun_champion_recipe

    champ, tdir, (X_tr, y_tr, X_va, y_va), _ = _fake_regression_champion(
        tmp_path)
    champ.fit_and_score = lambda *a, **k: (np.zeros(len(X_va)), {})
    with pytest.raises(RuntimeError, match="_last_model"):
        _rerun_champion_recipe(champ, X_tr, y_tr, X_va, y_va, 1, {}, tdir)


def test_rerun_recipe_refuses_unstripped_hpo(tmp_path):
    from _mvo.cmd_deliver import _rerun_champion_recipe

    champ, tdir, (X_tr, y_tr, X_va, y_va), _ = _fake_regression_champion(
        tmp_path)
    with pytest.raises(RuntimeError, match="hpo"):
        _rerun_champion_recipe(champ, X_tr, y_tr, X_va, y_va, 1,
                               {"hpo": {"budget": 40}}, tdir)
    assert champ.calls == [], "未剥离 hpo 节时不得调用 fit_and_score"


# ---------------------------------------------------------------------------
# 2. HPO 剥离 + best_params 注入
# ---------------------------------------------------------------------------
def test_strip_hpo_cfg_injects_best_params(tmp_path):
    from _mvo.cmd_deliver import _strip_hpo_cfg

    tdir = tmp_path / "trial"
    tdir.mkdir()
    (tdir / "hpo-study.json").write_text(json.dumps(
        {"best_params": {"learning_rate": 0.0125, "n_estimators": 8000,
                         "max_depth": 6, "bad": [1, 2]}}))
    cfg = {"hpo": {"budget": 40}, "hyperparameters": {"max_depth": 7},
           "seed": 3}
    out, src = _strip_hpo_cfg(cfg, tdir)
    assert src == "hpo-study.json"
    assert "hpo" not in out and "hpo" in cfg, "不得原地修改冠军 cfg"
    hp = out["hyperparameters"]
    assert hp["learning_rate"] == 0.0125 and hp["n_estimators"] == 8000
    assert hp["max_depth"] == 7, "人工 --set 覆写应优先于 best_params"
    assert "bad" not in hp


def test_strip_hpo_cfg_noop_and_missing(tmp_path):
    from _mvo.cmd_deliver import _strip_hpo_cfg

    out, src = _strip_hpo_cfg({"seed": 1}, tmp_path)
    assert out == {"seed": 1} and src is None
    with pytest.raises(RuntimeError, match="best_params"):
        _strip_hpo_cfg({"hpo": {"budget": 4}}, tmp_path)


# ---------------------------------------------------------------------------
# 3. 特征面一致性校验
# ---------------------------------------------------------------------------
def _face_fixture(tmp_path, ref: pd.DataFrame):
    tdir = tmp_path / "trials" / "trial-x"
    (tdir / "model").mkdir(parents=True)
    ref.to_parquet(tdir / "model" / "validation_input.parquet", index=False)
    return tdir


def test_feature_face_check_passes_with_nan(tmp_path):
    from _mvo.cmd_deliver import _check_feature_face

    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [0.5, 0.25, np.nan]},
                     dtype="float32")
    tdir = _face_fixture(tmp_path, X)
    rep = _check_feature_face(X.copy(), list(X.columns), tdir)
    assert rep["passed"] is True, rep
    assert rep["mismatched_columns"] == 0


def test_feature_face_check_detects_order_value_and_nan_drift(tmp_path):
    from _mvo.cmd_deliver import _check_feature_face

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, np.nan, 1.0]},
                     dtype="float32")
    tdir = _face_fixture(tmp_path, X)
    swapped = X[["b", "a"]]
    assert _check_feature_face(swapped, ["b", "a"], tdir)["passed"] is False
    drift = X.copy()
    drift.loc[1, "a"] = 2.5
    rep = _check_feature_face(drift, list(X.columns), tdir)
    assert rep["passed"] is False and "a" in rep["mismatched_examples"]
    nan_drift = X.copy()
    nan_drift.loc[1, "b"] = 0.0          # NaN 被填 0:数值差为 0 但缺失语义变了
    rep = _check_feature_face(nan_drift, list(X.columns), tdir)
    assert rep["passed"] is False and "b" in rep["mismatched_examples"]


def test_feature_face_check_missing_artifact_returns_none(tmp_path):
    from _mvo.cmd_deliver import _check_feature_face

    tdir = tmp_path / "trials" / "trial-y"
    (tdir / "model").mkdir(parents=True)
    rep = _check_feature_face(pd.DataFrame({"a": [1.0]}), ["a"], tdir)
    assert rep["passed"] is None and "退回完整复现" in rep["reason"]


# ---------------------------------------------------------------------------
# 4. finalize-retrain 分派契约(源码级)
# ---------------------------------------------------------------------------
def test_finalize_retrain_dispatch_contract():
    body = _retrain_body()
    # rerun 路径:特征面证明 → 复跑 → 不 clone
    rerun = body.split("if _rerun and _has_fs:")[1].split(
        "elif not _rerun")[0]
    assert "_check_feature_face(" in rerun
    assert "_rerun_champion_recipe(" in rerun
    assert "_strip_hpo_cfg(" in rerun
    assert "_sk_clone" not in rerun and "clone(" not in rerun
    assert 'retrain_builder = "champion_fit_and_score_rerun"' in rerun
    # rerun 失败不回落适配器
    assert "return 1" in rerun.split("_rerun_champion_recipe(")[1]
    # frozen 路径:clone 失败报错 + remedy,不回落
    frozen = body.split("elif not _rerun")[1].split("if model is None:")[0]
    assert "except TypeError" in frozen and "rerun_recipe" in frozen
    assert "return 1" in frozen.split("except TypeError")[1]
    # 适配器 rerun 模式早停面基于 X_all
    assert "_es_X, _es_y = (X_all, y_all) if _rerun else (X_tr, y_tr)" in body
    assert "es_cut = int(len(_es_X) * (1.0 - es_frac))" in body
    # 交付配方副本 + 从副本加载
    assert "_copy_delivery_recipe(" in body
    assert '_finalize_load_module(_recipe_dir / "code" / "train.py"' in body
    # model_meta 新字段
    for key in ('"refit_mode"', '"face_check"', '"es_holdout"',
                '"shadow_precheck"', '"delivery_recipe_sha256"',
                '"delivery_recipe_dir"', '"rerun"'):
        assert key in body, key


def test_copy_delivery_recipe_verifies_bytes(tmp_path):
    from _mvo.cmd_deliver import _copy_delivery_recipe

    tdir = tmp_path / "trials" / "trial-a"
    (tdir / "code").mkdir(parents=True)
    (tdir / "config").mkdir()
    (tdir / "code" / "train.py").write_text("X = 1\n")
    (tdir / "config" / "trial.yaml").write_text("seed: 1\n")
    dst, shas = _copy_delivery_recipe(tdir, tmp_path / "final-model" / "dr")
    assert (dst / "code" / "train.py").read_text() == "X = 1\n"
    assert set(shas) == {"code/train.py", "config/trial.yaml"}
    # 副本与 trials/<id>/ 同深度(旧模板 parents[3] 推导 RUN_DIR 不漂移)
    assert len(dst.relative_to(tmp_path).parts) == len(
        tdir.relative_to(tmp_path).parts)


def test_refit_mode_resolution():
    from _mvo.cmd_deliver import _refit_mode

    cfg = types.SimpleNamespace(get=lambda k, d=None: d)
    assert _refit_mode(types.SimpleNamespace(refit_mode=None), cfg) \
        == "rerun_recipe"
    assert _refit_mode(types.SimpleNamespace(refit_mode="frozen_rounds"),
                       cfg) == "frozen_rounds"
    cfg2 = types.SimpleNamespace(
        get=lambda k, d=None: "frozen_rounds"
        if k == "finalize.refit_mode" else d)
    assert _refit_mode(types.SimpleNamespace(), cfg2) == "frozen_rounds"
    with pytest.raises(SystemExit):
        _refit_mode(types.SimpleNamespace(refit_mode="bogus"), cfg)


# ---------------------------------------------------------------------------
# 5. 打分按任务类型分派
# ---------------------------------------------------------------------------
def test_delivery_predict_regression_never_uses_proba():
    from _mvo.cmd_deliver import _delivery_predict

    class Both:
        def predict(self, X):
            return np.full(len(X), 7.0)

        def predict_proba(self, X):
            raise AssertionError("回归不得调用 predict_proba")

    X = pd.DataFrame({"a": [1.0, 2.0]})
    reg = types.SimpleNamespace(model_type="regression")
    assert list(_delivery_predict(Both(), X, reg)) == [7.0, 7.0]

    class Clf:
        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))

    binary = types.SimpleNamespace(model_type="binary_classification")
    assert list(_delivery_predict(Clf(), X, binary)) == [0.7, 0.7]
    assert "predict_proba(X)[:, 1] \\" not in _SRC, "残留旧 duck-typing 分派"


# ---------------------------------------------------------------------------
# 6. shadow 预检
# ---------------------------------------------------------------------------
def _shadow_fixture(tmp_path, delivery_noise: float, search_noise: float):
    from runtime.data import SAMPLE_KEY_COLUMN

    run = tmp_path / "run"
    lab_dir = run / "sample-profile" / "splits" / "labels"
    lab_dir.mkdir(parents=True)
    champ_dir = run / "trials" / "trial-c"
    champ_dir.mkdir(parents=True)
    fm = run / "final-model"
    fm.mkdir()
    rng = np.random.default_rng(3)
    n = 3000
    keys = np.array([f"k{i}" for i in range(n)])
    y = rng.normal(size=n)
    lab = pd.DataFrame({"y": y, SAMPLE_KEY_COLUMN: keys})
    lab.to_parquet(lab_dir / "shadow_test.parquet", index=False)
    perm = rng.permutation(n)                      # 键序与标签不同,须按键对齐
    for path, noise in ((fm / "shadow_test_score.parquet", delivery_noise),
                        (champ_dir / "shadow_test_score.parquet",
                         search_noise)):
        sc = y + rng.normal(scale=noise, size=n)
        pd.DataFrame({"score": sc[perm], SAMPLE_KEY_COLUMN: keys[perm]}
                     ).to_parquet(path, index=False)
    spec = types.SimpleNamespace(primary_metric="rmse", target_name="y",
                                 model_type="regression")
    return run, fm, champ_dir, spec


def test_shadow_precheck_flags_significant_regression(tmp_path):
    from _mvo.cmd_deliver import _shadow_precheck

    run, fm, cd, spec = _shadow_fixture(tmp_path, 1.0, 0.3)
    rep = _shadow_precheck(run, fm, cd, spec, n_boot=300)
    assert rep["available"] is True, rep
    assert rep["delivery_model"] > rep["search_model"]
    assert rep["worse_than_search_model"] is True


def test_shadow_precheck_passes_when_delivery_better(tmp_path):
    from _mvo.cmd_deliver import _shadow_precheck

    run, fm, cd, spec = _shadow_fixture(tmp_path, 0.3, 1.0)
    rep = _shadow_precheck(run, fm, cd, spec, n_boot=300)
    assert rep["available"] is True, rep
    assert rep["worse_than_search_model"] is False


def test_shadow_precheck_missing_inputs(tmp_path):
    from _mvo.cmd_deliver import _shadow_precheck

    spec = types.SimpleNamespace(primary_metric="rmse", target_name="y")
    rep = _shadow_precheck(tmp_path, tmp_path, tmp_path, spec)
    assert rep["available"] is False


def test_evaluate_final_prechecks_before_gate():
    body = _eval_body()
    i_gate = body.index('gate.acquire(purpose="final_model_evaluation")')
    i_shadow = body.index('_sp.get("worse_than_search_model")')
    assert i_shadow < i_gate, "shadow 预检必须在 acquire gate 之前"
    assert "ack_shadow_regression" in body[:i_gate]
    # shadow 三方 + 交付模型优先
    assert '"shadow_test_search_model"' in body
    assert '"baseline_shadow_test"' in body
    assert body.index('("shadow_test",') < body.index(
        '("shadow_test_search_model",'), "交付模型 shadow 应优先"
    assert "improvement_vs_baseline_shadow_test" in body
    # 原值分桶只在 gate 之后、用内存标签
    assert body.index("_raw_scale_report(") > i_gate


# ---------------------------------------------------------------------------
# 7. 原值尺度分桶 + 指标面板 + CLI
# ---------------------------------------------------------------------------
def test_raw_scale_report_expm1(tmp_path):
    from _mvo.cmd_deliver import _raw_scale_report

    rng = np.random.default_rng(5)
    n = 2000
    y_raw = np.where(rng.random(n) < 0.85, 0.0,
                     rng.lognormal(8, 1, size=n))
    y_log = np.log1p(y_raw)
    s_log = y_log * 0.5 + rng.normal(scale=0.3, size=n) - 0.1
    spec = types.SimpleNamespace(model_type="regression", primary_metric="rmse",
                                 target_name="y", treatment_column=None,
                                 task_id="reg-test",
                                 data={"secondary_metrics": ["mae", "r2"]})
    out = _raw_scale_report(spec, y_log, s_log, out_dir=tmp_path / "raw",
                            n_bins=10, split="final_test")
    assert out["n"] == n
    assert out["n_negative_pred_clipped"] == int((s_log < 0).sum())
    assert out["true_mean"] == pytest.approx(y_raw.mean())
    p_raw = np.expm1(np.clip(s_log, 0, None))
    assert out["rmse"] == pytest.approx(np.sqrt(np.mean((p_raw - y_raw) ** 2)))
    assert (tmp_path / "raw" / "segment-report.md").exists()
    md = (tmp_path / "raw" / "segment-report.md").read_text(encoding="utf-8")
    assert "原值尺度" in md


def test_metric_panel_includes_secondary():
    from _mvo.cmd_deliver import _metric_panel

    spec = types.SimpleNamespace(
        primary_metric="rmse",
        data={"secondary_metrics": ["mae", "r2", "auc"]})
    y = np.array([0.0, 1.0, 2.0, 3.0])
    s = np.array([0.1, 0.9, 2.2, 2.8])
    out = _metric_panel(spec, y, s)
    assert set(out) >= {"rmse", "mae", "r2"}
    assert out["rmse"] == pytest.approx(np.sqrt(np.mean((y - s) ** 2)))


def test_cli_flags_parsed():
    sys.path.insert(0, str(SKILL / "scripts"))
    import modelevo as cli

    p = cli.build_parser()
    ns = p.parse_args(["finalize-retrain", "--run", "x",
                       "--refit-mode", "frozen_rounds", "--full-reproduction"])
    assert ns.refit_mode == "frozen_rounds" and ns.full_reproduction is True
    ns = p.parse_args(["finalize-all", "--run", "x"])
    assert ns.refit_mode is None
    ns = p.parse_args(["evaluate-final", "--run", "x",
                       "--segment-inverse-transform", "expm1",
                       "--ack-shadow-regression", "why"])
    assert ns.segment_inverse_transform == "expm1"
    assert ns.ack_shadow_regression == "why"
    speed = (SKILL / "scripts" / "_mvo" / "cmd_speed.py").read_text(
        encoding="utf-8")
    assert '"--refit-mode"' in speed and '"--full-reproduction"' in speed
