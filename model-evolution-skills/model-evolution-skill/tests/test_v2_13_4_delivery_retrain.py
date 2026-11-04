"""v2.13.4 交付链回归测试:确保交付的是冠军配方的重训模型。

事故来源:run `req-t7-drawdown`(2026-09-23 实测)。计划与证据见
`docs/MODELEVO_DELIVERY_RETRAIN_FIX_PLAN_20260923.md`。

锁定的缺陷与修复:

1. 确认门未过时 finalize-retrain 只拷贝 trial 的 train-only 模型、不重训
   → 对最强合法候选同口径重训到 final-model/unconfirmed/,证据白名单不含
   任何模型二进制。
2. finalize-retrain 按符号名猜 estimator,内联构造 XGBoost 的冠军被重训成
   LightGBM + 默认超参 → 非 HPO 冠军用冠军自己的 fit_and_score 复现并校验,
   任何路径加算法族一致性校验。
3. evaluate-final 只核对 best_trial 字段 → 前置校验重训证明(行数、哈希、
   delivery_status)。
4. 外部脚本加载冻结 train.py 调 fit_and_score 会覆写冻结工件 → runner 注入
   MODELEVO_TRIAL_EXEC,dump_* 不匹配即拒写。
5. 模板 shadow 打分 fillna(0) 与训练口径错位;multi-seed 轮覆写 primary 工件
   → NaN 原生打分(拒收 NaN 才降级并留痕);seed 变体轮不写工件。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402


def _render_trial(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    from runtime.templates import TRIAL_TEMPLATE
    from _mvo.templates import TRIAL_BODY

    tdir = tmp_path / "trials" / "trial-999-g09-xgboost-1"
    (tdir / "code").mkdir(parents=True)
    code = TRIAL_TEMPLATE.format(
        trial_id=tdir.name, hypothesis="h", actions="a",
        run_dir=str(tmp_path), target="y",
        model_type="binary_classification", treatment="", seed=1,
        skill_root=str(SKILL), body=TRIAL_BODY)
    (tdir / "code" / "train.py").write_text(code, encoding="utf-8")
    return tdir, code


def _load(path: pathlib.Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 4/5. 模板:写保护、shadow 口径、multi-seed 工件
# ---------------------------------------------------------------------------
def test_template_renders_with_shadow_and_multiseed_fixes(tmp_path):
    import ast

    _, code = _render_trial(tmp_path)
    ast.parse(code)
    assert "_X_sh = _X_sh.fillna(0)" not in code, "shadow 仍按 fillna(0) 打分"
    assert "shadow_scoring_mode" in code and "fillna0_fallback" in code
    assert 'os.environ["MODELEVO_SEED_VARIANT"] = "1"' in code
    assert 'os.environ.pop("MODELEVO_SEED_VARIANT", None)' in code
    assert 'metrics_extra["artifact_model"] = "primary_seed"' in code
    # dump_model / dump_eval_frame / dump_shadow_score 各一道写保护
    assert code.count(
        'os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name') == 3


def test_dump_functions_refuse_outside_runner(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    tdir, _ = _render_trial(tmp_path)
    m = _load(tdir / "code" / "train.py", "t999_guard")
    X = pd.DataFrame({"a": [0., 1, 0, 1], "b": [1., 0, 1, 0]})
    mdl = LogisticRegression().fit(X, [0, 1, 0, 1])
    for k in ("MODELEVO_TRIAL_EXEC", "MODELEVO_SEED_VARIANT",
              "MODELEVO_SMOKE_RUN", "MODELEVO_QUICK_RATIO"):
        monkeypatch.delenv(k, raising=False)

    m.dump_model(mdl, X.columns)
    m.dump_eval_frame(X)
    m.dump_shadow_score(np.array([0.1, 0.9, 0.2, 0.8]))
    assert not (tdir / "model").exists(), "无执行凭证时写入了 trial 工件"
    assert not (tdir / "shadow_test_score.parquet").exists()

    monkeypatch.setenv("MODELEVO_TRIAL_EXEC", "trial-000-someone-else")
    m.dump_model(mdl, X.columns)
    assert not (tdir / "model").exists(), "凭证指向别的 trial 时写入了工件"

    monkeypatch.setenv("MODELEVO_TRIAL_EXEC", tdir.name)
    monkeypatch.setenv("MODELEVO_SEED_VARIANT", "1")
    m.dump_model(mdl, X.columns)
    m.dump_eval_frame(X)
    assert not (tdir / "model").exists(), "seed 变体轮覆写了 primary 工件"

    monkeypatch.delenv("MODELEVO_SEED_VARIANT")
    m.dump_model(mdl, X.columns)
    m.dump_eval_frame(X)
    m.dump_shadow_score(np.array([0.1, 0.9, 0.2, 0.8]))
    names = sorted(p.name for p in (tdir / "model").iterdir())
    assert {"model.joblib", "feature_columns.json",
            "validation_input.parquet"} <= set(names)
    assert (tdir / "shadow_test_score.parquet").exists()


def test_runner_injects_trial_exec_and_baseline_guarded():
    runner = (SKILL / "runtime" / "runner.py").read_text(encoding="utf-8")
    assert 'env_extra["MODELEVO_TRIAL_EXEC"] = Path(tdir).resolve().name' \
        in runner
    baseline = (SKILL / "runtime" / "baseline.py").read_text(encoding="utf-8")
    assert baseline.count(
        'os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name') == 3


# ---------------------------------------------------------------------------
# 1. 未确认路径:重训而不是拷贝 train-only 模型
# ---------------------------------------------------------------------------
def test_unconfirmed_evidence_has_no_model_binaries():
    from _mvo.cmd_deliver import _UNCONFIRMED_EVIDENCE

    joined = " ".join(_UNCONFIRMED_EVIDENCE)
    for bad in ("model.joblib", "model.json", "model.txt", "model.bin",
                "model.pt", "validation_input.parquet", "recipe",
                "model_meta"):
        assert bad not in joined, f"证据白名单含 {bad}"


def test_finalize_retrain_unconfirmed_branch_contract():
    src = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    body = src.split("def cmd_finalize_retrain(")[1].split(
        "\ndef cmd_finalize_baseline_score(")[0]
    # 无交付冠军 → 有合法候选则未确认重训,不是直接快照
    # (v2.14.0:候选取搜索冠军,与交付位解析同源 → 传 lb)
    assert "_best_unconfirmed_candidate(trials, spec, lb)" in body
    assert "unconfirmed_mode = True" in body
    # 只写 unconfirmed/,不打盲测,退出码 1
    assert 'fm = fm / "unconfirmed"' in body
    assert "if unconfirmed_mode:" in body.split(
        '_score_split("final_test"')[0].rsplit("def _score_split", 1)[1]
    assert "retrained=True" in body
    # 未确认候选不是 confirm trial,不做 confirm 专属 seed 校验
    # (v2.15.3 起豁免交付的搜索冠军同理:_confirm_waived 一并跳过)
    assert ("if not unconfirmed_mode:\n" in body
            or "if not unconfirmed_mode and not _confirm_waived:\n" in body)
    _seed_guard = body.split("verify_confirmation_seed(")[0].rsplit("if ", 1)[1]
    assert _seed_guard.startswith("not unconfirmed_mode")
    # 冠军配方:冠军代码复现 + 算法族兜底
    assert "_reproduce_champion_fit(" in body
    assert "_model_family(model)" in body
    # 重训证明写入 model_meta
    for key in ('"n_train_rows"', '"n_validation_rows"',
                '"delivery_status"', '"trial_model_sha256"',
                '"final_model_sha256"', '"retrain_builder"',
                '"reproduction"'):
        assert key in body, key


def test_snapshot_only_flag_parsed():
    sys.path.insert(0, str(SKILL / "scripts"))
    import modelevo as cli

    ns = cli.build_parser().parse_args(
        ["finalize-retrain", "--run", "x", "--snapshot-only"])
    assert ns.snapshot_only is True


# ---------------------------------------------------------------------------
# 2. 冠军配方复现
# ---------------------------------------------------------------------------
def _fake_champion(tmp_path: pathlib.Path, *, drift: float = 0.0):
    """最小冠军模块:fit_and_score 内联构造模型、写 TRIAL_DIR、设 _last_model。"""
    from sklearn.linear_model import LogisticRegression

    tdir = tmp_path / "trials" / "trial-001-g01-logreg-1"
    (tdir / "model").mkdir(parents=True)
    champ = types.ModuleType("fake_champ")
    champ.TRIAL_DIR = tdir
    champ._last_model = None

    def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                      w_tr=None, w_va=None):
        mdl = LogisticRegression(random_state=seed).fit(X_tr, y_tr)
        (champ.TRIAL_DIR / "model").mkdir(parents=True, exist_ok=True)
        (champ.TRIAL_DIR / "model" / "touched.txt").write_text("x")
        champ._last_model = mdl
        return mdl.predict_proba(X_va)[:, 1] + drift, {"model_num": 1}

    champ.fit_and_score = fit_and_score
    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    y_tr = pd.Series((X_tr["a"] + rng.normal(size=200) > 0).astype(int))
    X_va = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
    y_va = pd.Series((X_va["a"] + rng.normal(size=80) > 0).astype(int))
    ref = LogisticRegression(random_state=7).fit(X_tr, y_tr)
    pd.DataFrame({"score": ref.predict_proba(X_va)[:, 1]}).to_parquet(
        tdir / "validation_score.parquet", index=False)
    spec = types.SimpleNamespace(primary_metric="auc")
    return champ, tdir, (X_tr, y_tr, X_va, y_va), spec


def test_reproduce_champion_fit_sandboxes_and_verifies(tmp_path):
    from _mvo.cmd_deliver import _reproduce_champion_fit

    champ, tdir, (X_tr, y_tr, X_va, y_va), spec = _fake_champion(tmp_path)
    m_ref, rep = _reproduce_champion_fit(
        champ, X_tr, y_tr, X_va, y_va, 7, {"multi_seed": 3}, tdir, spec)
    assert m_ref is not None
    assert rep["passed"] is True, rep
    assert rep["validation_max_abs_score_diff"] < 1e-12
    assert champ.TRIAL_DIR == tdir, "TRIAL_DIR 未恢复"
    assert not (tdir / "model" / "touched.txt").exists(), \
        "复现期写入落到了真实 trial 目录"


def test_reproduce_champion_fit_rejects_drift(tmp_path):
    from _mvo.cmd_deliver import _reproduce_champion_fit

    # 分数整体平移不改变 AUC —— 用非单调扰动制造指标漂移
    champ, tdir, (X_tr, y_tr, X_va, y_va), spec = _fake_champion(tmp_path)
    orig = champ.fit_and_score

    def drifting(*a, **k):
        s, meta = orig(*a, **k)
        return np.random.default_rng(1).permutation(s), meta

    champ.fit_and_score = drifting
    _, rep = _reproduce_champion_fit(
        champ, X_tr, y_tr, X_va, y_va, 7, {}, tdir, spec)
    assert rep["passed"] is False, rep


def test_fitted_rounds_and_family():
    xgb = pytest.importorskip("xgboost")
    from _mvo.cmd_deliver import _fitted_rounds, _model_family

    X = np.random.default_rng(0).normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=17, max_depth=2, verbosity=0).fit(X, y)
    assert _fitted_rounds(m) == 17
    assert _model_family(m) == "xgboost"


# ---------------------------------------------------------------------------
# 3. evaluate-final 前置重训证明
# ---------------------------------------------------------------------------
def _provenance_run(tmp_path, *, meta_overrides=None, same_as_trial=False,
                    write_meta=True):
    import joblib
    from sklearn.linear_model import LogisticRegression

    from _mvo.cmd_deliver import _file_sha256

    run = tmp_path / "run"
    champ = "trial-016-g04-xgboost-1"
    (run / "trials" / champ / "model").mkdir(parents=True)
    (run / "final-model" / "model").mkdir(parents=True)
    X = np.array([[0.], [1.], [0.], [1.]])
    trial_m = LogisticRegression().fit(X, [0, 1, 0, 1])
    joblib.dump(trial_m, run / "trials" / champ / "model" / "model.joblib")
    final_p = run / "final-model" / "model" / "model.joblib"
    if same_as_trial:
        final_p.write_bytes(
            (run / "trials" / champ / "model" / "model.joblib").read_bytes())
    else:
        joblib.dump(LogisticRegression(C=0.5).fit(X, [0, 1, 0, 1]), final_p)
    meta = {"best_trial": champ, "delivery_status": "confirmed",
            "n_train_rows": 100, "n_validation_rows": 20,
            "n_train_validation_rows": 120,
            "trial_model_sha256": _file_sha256(
                run / "trials" / champ / "model" / "model.joblib"),
            "final_model_sha256": _file_sha256(final_p)}
    meta.update(meta_overrides or {})
    if write_meta:
        (run / "final-model" / "model_meta.json").write_text(
            json.dumps(meta), encoding="utf-8")
    return run, champ


def test_provenance_accepts_genuine_retrain(tmp_path):
    from _mvo.cmd_deliver import _verify_retrain_provenance

    run, champ = _provenance_run(tmp_path)
    assert _verify_retrain_provenance(run, champ) is None


@pytest.mark.parametrize("kwargs,needle", [
    ({"write_meta": False}, "缺失"),
    ({"meta_overrides": {"delivery_status": "unconfirmed"}}, "delivery_status"),
    ({"meta_overrides": {"n_validation_rows": 0,
                         "n_train_validation_rows": 100}}, "行数"),
    ({"meta_overrides": {"n_train_validation_rows": 999}}, "行数"),
    ({"meta_overrides": {"n_train_rows": None}}, "缺 n_train_rows"),
    ({"meta_overrides": {"final_model_sha256": "0" * 64}}, "不一致"),
    ({"same_as_trial": True}, "未重训"),
])
def test_provenance_rejects(tmp_path, kwargs, needle):
    from _mvo.cmd_deliver import _file_sha256, _verify_retrain_provenance

    run, champ = _provenance_run(tmp_path, **kwargs)
    if kwargs.get("same_as_trial"):
        # 拷贝来的 trial 模型:连 final_model_sha256 都按实际文件写,仍须被识破
        mp = run / "final-model" / "model_meta.json"
        meta = json.loads(mp.read_text())
        meta["final_model_sha256"] = _file_sha256(
            run / "final-model" / "model" / "model.joblib")
        mp.write_text(json.dumps(meta))
    err = _verify_retrain_provenance(run, champ)
    assert err is not None and needle in err, err


def test_evaluate_final_checks_provenance_before_gate():
    src = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    body = src.split("def cmd_evaluate_final(")[1]
    # v2.15.3 起带 allow_waived= 关键字(豁免交付的重训证明),前缀不变
    i_prov = body.index("_verify_retrain_provenance(run_dir, champion_id")
    i_gate = body.index('gate.acquire(purpose="final_model_evaluation")')
    assert i_prov < i_gate, "重训证明必须在 acquire gate 之前校验"
