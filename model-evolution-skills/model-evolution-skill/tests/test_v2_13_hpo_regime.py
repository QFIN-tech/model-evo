"""v2.13 回归测试:HPO 训练机制对齐(cap+早停)与 inner 指标口径。

三个已核实缺陷(见 docs/MODELEVO_SELF_IMPROVEMENT_REVIEW_AND_ROADMAP):
  1. ``_fit_with_eval`` 的 ``hasattr(model, "best_iteration_")`` 探测在
     未 fit 的 LGBMClassifier 上抛 LGBMNotFittedError(AttributeError
     子类)→ hasattr 返回 False → LightGBM 走 else 分支:全量 fit、无
     早停、无轮数截断;XGB/CatBoost 分支传了 eval_set 但构造参数里
     没有早停项,同样不早停。
  2. ``runtime.hpo.build_gbm`` 三族默认 ``n_estimators/iterations=300``
     且无早停参数,而 baseline/交付重训用 cap=2000 + early_stopping
     _rounds=50——HPO 搜的"固定 300 棵"机制与交付用的"早停"机制不是
     同一函数族;LR 空间 [0.01,0.15] 在 300 棵下系统性偏向高 LR。
  3. ``METRIC_NAME`` 硬编码 "auc"、``_inner_metric`` 分类恒 roc_auc,
     HPO inner 目标不跟随 spec.primary_metric;方向恒 maximize,
     logloss/rmse 任务的搜索方向是反的。

本文件锁定:
  A. build_gbm 默认 cap=2000(env 可覆盖)、xgb early_stopping_rounds、
     catboost od_wait/use_best_model;显式 params 仍覆盖默认;
  B. 模板 ``_fit_with_es`` 的三族早停行为(exec 渲染源码段,与
     test_v2_12_2_9_gbm_device 的 ``_load_body_helper`` 同法);
  C. ``_inner_metric`` 走 ``runtime.metrics.compute_metric(METRIC_NAME)``、
     方向跟随 ``metric_direction``;
  D. 渲染的 hpo Trial 代码可过 ``assets/lint_trial.py`` 全规则
     (规则 8 的 eval_set 变量名检查在此显式验证)。
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL, ASSETS, trial_body  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

lgb = pytest.importorskip("lightgbm")
xgb = pytest.importorskip("xgboost")
cb = pytest.importorskip("catboost")


# --------------------------------------------------------------------------
# 公共底座
# --------------------------------------------------------------------------

def _make_frame(n_samples=4000, n_features=10, seed=0):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               random_state=seed)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="y")
    return X, y


def _exec_body_fn(name: str, next_name: str, extra_prelude: str = ""):
    """从 TRIAL_BODY 字符串里 exec 出单个函数(v2_12_2_9 同款手法)。

    ``_fit_with_es`` 定义在 ``_mvo.templates.TRIAL_BODY`` 字符串里
    (渲染进 trial 的 code/train.py),模块级不可见——必须 exec 源码
    段才能测到真正跑在 trial 里的那份实现。
    """
    src = trial_body()
    i = src.index(f"def {name}")
    j = src.index(f"def {next_name}", i)
    prelude = (
        "import os\n"
        "import json\n"
        "import pathlib\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "RUN_DIR = pathlib.Path('/tmp/_v213_no_run')\n"
        "METRIC_NAME = 'auc'\n"
        "_INNER_METRIC_UP = True\n"
        "def _lgbm_fit_safe(model, *a, **kw):\n"
        "    return model.fit(*a, **kw)\n"
    )
    ns: dict = {"__builtins__": __builtins__}
    exec(prelude + extra_prelude + src[i:j], ns)
    return ns[name]


# --------------------------------------------------------------------------
# A. build_gbm:cap + 早停参数随构造注入
# --------------------------------------------------------------------------

def test_build_gbm_default_cap_and_es_params(monkeypatch):
    from runtime.hpo import build_gbm

    monkeypatch.delenv("MODELEVO_N_ESTIMATORS_CAP", raising=False)
    monkeypatch.delenv("MODELEVO_EARLY_STOP_ROUNDS", raising=False)

    m = build_gbm("lightgbm", {}, 0, "binary_classification", 0, False)
    assert m.n_estimators == 2000
    assert m.n_estimators == 2000  # cap,非固定 300

    mx = build_gbm("xgboost", {}, 0, "binary_classification", 0, False)
    assert mx.get_params()["early_stopping_rounds"] == 50
    assert mx.get_params()["n_estimators"] == 2000

    mc = build_gbm("catboost", {}, 0, "binary_classification", 0, False)
    assert mc.get_params()["od_wait"] == 50
    assert mc.get_params()["od_type"] == "Iter"
    assert mc.get_params()["use_best_model"] is True
    assert mc.get_params()["iterations"] == 2000


def test_build_gbm_env_overrides_and_params_win(monkeypatch):
    from runtime.hpo import build_gbm

    monkeypatch.setenv("MODELEVO_N_ESTIMATORS_CAP", "1500")
    monkeypatch.setenv("MODELEVO_EARLY_STOP_ROUNDS", "20")
    m = build_gbm("lightgbm", {}, 0, "binary_classification", 0, False)
    assert m.n_estimators == 1500
    mx = build_gbm("xgboost", {}, 0, "binary_classification", 0, False)
    assert mx.get_params()["early_stopping_rounds"] == 20
    mc = build_gbm("catboost", {}, 0, "binary_classification", 0, False)
    assert mc.get_params()["od_wait"] == 20

    # 显式 params 仍覆盖默认(base.update(p) 保持在最后)
    m2 = build_gbm("lightgbm", {"n_estimators": 77}, 0,
                   "binary_classification", 0, False)
    assert m2.n_estimators == 77
    mx2 = build_gbm("xgboost", {"n_estimators": 66, "early_stopping_rounds": 9},
                    0, "binary_classification", 0, False)
    assert mx2.get_params()["n_estimators"] == 66
    assert mx2.get_params()["early_stopping_rounds"] == 9
    mc2 = build_gbm("catboost", {"iterations": 55}, 0,
                    "binary_classification", 0, False)
    assert mc2.get_params()["iterations"] == 55


# --------------------------------------------------------------------------
# B. _fit_with_es:三族早停行为(渲染源码段 exec)
# --------------------------------------------------------------------------

def test_fit_with_es_lightgbm_early_stops_and_truncates():
    fit_es = _exec_body_fn("_fit_with_es", "_fit_with_eval")
    X, y = _make_frame()
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.1,
                               random_state=0, n_jobs=4, verbosity=-1)
    out = fit_es(model, X, y)
    assert out is model or out is not None
    # 早停生效:轮数被截断(< 2000),best_iteration_ 非空
    assert model.n_estimators < 2000, model.n_estimators
    assert model.best_iteration_ and int(model.best_iteration_) > 0
    assert hasattr(model, "predict_proba")


def test_fit_with_es_xgboost_early_stops():
    fit_es = _exec_body_fn("_fit_with_es", "_fit_with_eval")
    X, y = _make_frame()
    model = xgb.XGBClassifier(n_estimators=2000, max_depth=6,
                              learning_rate=0.1, early_stopping_rounds=50,
                              random_state=0, n_jobs=4)
    out = fit_es(model, X, y)
    assert out is not None
    assert model.n_estimators < 2000, model.n_estimators
    assert getattr(model, "best_iteration", None) is not None


def test_fit_with_es_catboost_rebuilds_and_truncates():
    fit_es = _exec_body_fn("_fit_with_es", "_fit_with_eval")
    X, y = _make_frame()
    model = cb.CatBoostClassifier(iterations=2000, depth=6, learning_rate=0.1,
                                  od_type="Iter", od_wait=50,
                                  use_best_model=True, verbose=False,
                                  allow_writing_files=False, random_seed=0)
    out = fit_es(model, X, y)
    # catboost 重建路径会换对象(fitted 模型禁止 set_params)
    assert out is not None
    assert out.tree_count_ < 2000, out.tree_count_
    assert out.get_best_iteration() is not None


def test_fit_with_es_does_not_touch_outer_validation_names():
    """规则 8:两段式 _fit_with_es 及其 helper 的 eval_set 只能引用传入 X/y
    切出的 es 切片(HPO inner 与 train_tail 回退口径依赖它)。v2.16.0 的
    validation 单段拟合(_es_source 起)不在本段,见 test_v2_16_0_validation_es。"""
    src = trial_body()
    i = src.index("def _fit_with_es")
    j = src.index("def _es_source", i)
    seg = src[i:j]
    # 渲染产物过 lint 规则 8(check_eval_set_leak 的正则)
    import lint_trial as L  # noqa: E402  assets/ 在 sys.path

    assert L.check_eval_set_leak(seg) == []
    # 变量名语义:X_es/y_es 只来自传入 X/y 的尾部切片
    assert "X_es, y_es = X.iloc[_cut:], y.iloc[_cut:]" in seg
    assert "X_va" not in seg and "y_va" not in seg


# --------------------------------------------------------------------------
# C. inner 指标口径:compute_metric(METRIC_NAME) + metric_direction
# --------------------------------------------------------------------------

def _exec_inner_metric(metric_name: str):
    src = trial_body()
    i = src.index("def _inner_metric")
    j = src.index("def _hpo_fit_inner", i)
    prelude = (
        "import sys\n"
        f"sys.path.insert(0, {str(SKILL)!r})\n"
        f"METRIC_NAME = {metric_name!r}\n"
    )
    ns: dict = {"__builtins__": __builtins__}
    exec(prelude + src[i:j], ns)
    return ns["_inner_metric"]


def test_inner_metric_follows_metric_name_and_direction():
    from runtime.metrics import compute_metric, metric_direction

    rng = np.random.default_rng(0)
    y = (rng.random(500) > 0.5).astype(int)
    s = rng.random(500)

    inner = _exec_inner_metric("logloss")
    v = inner(y, s)
    assert v == pytest.approx(compute_metric("logloss", y, None, score=s))
    # logloss 方向 = 越小越好
    assert metric_direction("logloss") is False

    inner_auc = _exec_inner_metric("auc")
    v_auc = inner_auc(y, s)
    assert v_auc == pytest.approx(compute_metric("auc", y, None, score=s))
    assert metric_direction("auc") is True


def test_inner_metric_direction_wired_into_template_direction_up():
    """模板里 _INNER_METRIC_UP 必须来自 metric_direction(METRIC_NAME)。"""
    src = trial_body()
    i = src.index("METRIC_NAME = _spec_primary_metric()")
    j = src.index("_LGBM_CUDA", i)
    seg = src[i:j]
    assert "metric_direction" in seg, seg
    assert "_INNER_METRIC_UP = bool(_md_dir(METRIC_NAME))" in seg


def test_spec_primary_metric_reads_task_spec(tmp_path, monkeypatch):
    # exec 时的 RUN_DIR 需指向 tmp_path:重新 exec 一次可控 RUN_DIR
    src = trial_body()
    i = src.index("def _spec_primary_metric")
    j = src.index("def _inner_metric", i)
    prelude = (
        "import pathlib\n"
        "import yaml\n"
        f"RUN_DIR = pathlib.Path({str(tmp_path)!r})\n"
    )
    ns: dict = {"__builtins__": __builtins__}
    exec(prelude + src[i:j], ns)
    fn = ns["_spec_primary_metric"]

    import yaml
    (tmp_path / "task-spec.yaml").write_text(
        yaml.safe_dump({"primary_metric": "logloss"}), encoding="utf-8")
    assert fn() == "logloss"
    # 缺 spec → 回退 auc
    (tmp_path / "task-spec.yaml").unlink()
    assert fn() == "auc"
    # 坏 spec → 回退 auc
    (tmp_path / "task-spec.yaml").write_text(
        "primary_metric: [broken", encoding="utf-8")
    assert fn() == "auc"


# --------------------------------------------------------------------------
# D. run_hpo_search 的 inner_metric_name 落盘
# --------------------------------------------------------------------------

def test_run_hpo_search_records_inner_metric_name(tmp_path):
    from runtime.hpo import run_hpo_search, write_hpo_artifacts

    space = {"learning_rate": {"type": "float", "low": 0.01, "high": 0.02,
                               "log": True}}
    result = run_hpo_search(
        space, 2, lambda p: 1.0, metric_direction_up=False, seed=0,
        metric_name="logloss")
    assert result["inner_metric_name"] == "logloss"
    write_hpo_artifacts(tmp_path, result)
    import json
    study = json.loads((tmp_path / "hpo-study.json").read_text())
    assert study["inner_metric_name"] == "logloss"
    # 未传 metric_name → None(旧行为兼容)
    result2 = run_hpo_search(space, 1, lambda p: 1.0,
                             metric_direction_up=True, seed=0)
    assert result2["inner_metric_name"] is None


# --------------------------------------------------------------------------
# E. 渲染 hpo Trial 代码过 assets/lint_trial.py 全规则
# --------------------------------------------------------------------------

def _render_hpo_trial(run_dir: Path, trial_dir: Path,
                      model_type: str = "binary_classification",
                      primary_metric: str = "logloss") -> str:
    from runtime.templates import TRIAL_TEMPLATE

    import yaml
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "v213", "task_name": "v2.13 hpo regime",
        "model_type": model_type,
        "target": {"name": "label", "positive_definition": "y=1"},
        "primary_metric": primary_metric,
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
    }), encoding="utf-8")
    (run_dir / "run-state.json").write_text(
        '{"phase": "EVOLVING", "harness_epoch": 0, "harness_hash": "x"}',
        encoding="utf-8")
    code = TRIAL_TEMPLATE.format(
        trial_id=trial_dir.name, hypothesis="hpo 机制对齐回归",
        actions="hpo", run_dir=str(run_dir), target="label",
        model_type=model_type, treatment="", seed=20260814,
        skill_root=str(SKILL), body=trial_body())
    code = code.replace("SEED = 20260814",
                        "SEED = 20260814\n_HPO_ALGO = 'lightgbm'", 1)
    (trial_dir / "code").mkdir(parents=True, exist_ok=True)
    (trial_dir / "code" / "train.py").write_text(code, encoding="utf-8")
    # config 带 hpo 节(规则 7/11 的检测输入)
    (trial_dir / "config").mkdir(exist_ok=True)
    (trial_dir / "config" / "trial.yaml").write_text(yaml.safe_dump({
        "trial_id": trial_dir.name, "algorithm": "lightgbm",
        "features": {"exclude": [], "banned": [], "reparse": []},
        "target": "label", "model_type": model_type, "seed": 20260814,
        "hyperparameters": {}, "search_action": "hpo", "data_view": "v1",
        "hpo": {"search_space": {"learning_rate": {
            "type": "float", "low": 0.01, "high": 0.15, "log": True}},
            "budget_configs": 5, "inner_split": "time_tail",
            "sampler": "tpe_or_random"},
    }), encoding="utf-8")
    return code


def test_rendered_hpo_trial_passes_lint(tmp_path):
    run_dir = tmp_path / "run"
    trial_dir = run_dir / "trials" / "trial-001-g00-lightgbm-0001"
    code = _render_hpo_trial(run_dir, trial_dir)
    # 渲染产物可编译(v17.1 回归口径)
    compile(code, "train.py", "exec")

    r = subprocess.run(
        [sys.executable, str(ASSETS / "lint_trial.py"),
         str(run_dir), str(trial_dir)],
        capture_output=True, text=True, timeout=300)
    # 规则 10(脚手架 raise)与规则 19 的 widen_cached 警告是
    # 模板固有行为(HEAD 同款;Agent 改造后才消除),此处只断言
    # 本批次改动相关的规则不新增违规:8(eval_set 泄漏)、
    # 12(runtime import bootstrap)、13(未定义名)、2(双库)。
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("[FAIL]")]
    allowed = ("规则 10 违反", "widen_cached")
    new_fails = [f for f in fails
                 if not any(a in f for a in allowed)]
    assert not new_fails, "\n".join(new_fails)

    # 规则 8 专项:eval_set 只引用 es 切片变量
    import lint_trial as L  # noqa: E402

    assert L.check_eval_set_leak(code) == []


def test_rendered_hpo_trial_metric_follows_spec(tmp_path):
    """METRIC_NAME 渲染后从 RUN_DIR/task-spec.yaml 读出 logloss。"""
    run_dir = tmp_path / "run"
    trial_dir = run_dir / "trials" / "trial-001-g00-lightgbm-0001"
    code = _render_hpo_trial(run_dir, trial_dir, primary_metric="logloss")
    # 模块级求值验证:渲染产物在真实 run 目录下 import 后
    # METRIC_NAME 应为 logloss(_INNER_METRIC_UP=False)
    probe = textwrap.dedent(f"""
        import importlib.util, pathlib, sys
        p = pathlib.Path({str(trial_dir / 'code' / 'train.py')!r})
        spec = importlib.util.spec_from_file_location('probe_v213', p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass  # 脚手架 raise 不影响模块级常量(在 def 之前求值)
        print('METRIC_NAME=' + str(mod.METRIC_NAME))
        print('UP=' + str(mod._INNER_METRIC_UP))
    """)
    r = subprocess.run([sys.executable, "-c", probe],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-400:]
    assert "METRIC_NAME=logloss" in r.stdout, r.stdout
    assert "UP=False" in r.stdout, r.stdout
