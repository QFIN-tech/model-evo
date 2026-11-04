"""v2.17.0 WP4 回归测试:Trial 创建、模板、lint、guard、官方 hurdle 容器、子模型缓存。

口径(run4 regression.2,loan_amt_t7 原值回归)。全部离线、合成小数据:

  A. #7 lint 规则 4:回归 baseline / 模板 / HPO 嫁接 / 副本升级按 model_type
     渲染单一打分分支(分类不变);规则 4 改为污点分析(只在概率流入 score 写出
     路径时拒绝,不可达分支不报,hurdle p·μ 不报);
  B. #22 --changed-modules 严格枚举;
  C. #22 guard:引号内的批准命令文本不算调用,真实调用/写账本仍拒;
  D. baseline 补齐已配置算法(warm-start / v2.15 收缩不再吞掉 lightgbm);
  E. #16 --add-blocks:字节级复制 + 指纹一致 + 拒绝差异;
  F. #15 HurdleRegressor:clone / joblib 新进程 / 无 predict_proba / 分段 set_params;
  G. #10 子模型缓存:命中后 booster 字节一致、开关、非确定性不缓存;
  H. hurdle 模板:new-trial --hurdle-template → topology / lint / HPO / 超参段 / 缓存;
  I. #5-4 交付契约自检;J. MDE 走 effective_mde。
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

for _p in (str(SKILL), str(SKILL / "scripts"), str(SKILL / "assets")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

xgb = pytest.importorskip("xgboost")

import lint_trial as L  # noqa: E402
from _mvo import render as R  # noqa: E402
from runtime.logging_utils import dump_yaml, load_yaml  # noqa: E402

_OLD_RULE4 = re.compile(r"predict_proba[^\n]*\[[^\]]*[:,]\s*1[^\]]*\]")
_AP = "approve" + "-topology"          # 拼接:避免本文件被 guard 当成调用
_LEDGER_FN = "append" + "_ledger"


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def _mk_run(tmp: Path, model_type: str = "regression") -> Path:
    run = tmp / "run"
    (run / "trials").mkdir(parents=True, exist_ok=True)
    (run / "leaderboard").mkdir(exist_ok=True)
    sp = run / "sample-profile" / "splits"
    (sp / "reparse").mkdir(parents=True, exist_ok=True)
    (sp / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, n in (("train", 60), ("validation", 30)):
        pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)}
                     ).to_parquet(sp / f"{split}.parquet")
        pd.DataFrame({"seq": [",".join(f"E{(i * j) % 5}" for j in range(i % 7))
                              for i in range(n)]}
                     ).to_parquet(sp / "reparse" / f"{split}.parquet")
    metric = "rmse" if model_type == "regression" else "auc"
    (run / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "wp4", "task_name": "wp4", "model_type": model_type,
        "target": {"name": "y", "positive_definition": "y>0"},
        "primary_metric": metric,
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "split_policy": {"type": "time_based", "train": "2025-01",
                         "validation": "2025-02", "final_test": "2025-03"},
    }), encoding="utf-8")
    (run / "run-state.json").write_text(json.dumps(
        {"phase": "EVOLVING", "harness_epoch": 0}), encoding="utf-8")
    return run


def _cli():
    spec = importlib.util.spec_from_file_location(
        "modelevo_cli_wp4", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def _args(**over):
    class _A:
        pass
    a = _A()
    base = {
        "spec": None, "run": None, "algorithm": "xgboost",
        "from_parent": None, "hpo_parent": None, "set": None,
        "hypothesis": "单测假设陈述,长度足够", "rationale": "r",
        "falsification": "f", "actions": "a", "action": "explore",
        "search_space": None, "budget_configs": None,
        "force_hpo_budget": False, "parents": None,
        "changed_modules": "feature", "override_feature_phase": None,
        "family_hints": None, "generation": 1, "data_view": "v1",
        "heavy": False, "features": None, "override_beam": None,
        "screen_only": False, "planner_proposal_id": None,
        "planner_semantic_hash": None, "experience_ids": None,
        "allow_code_edit": False, "hpo_sample_rows": None, "topology": None,
        "hurdle_template": False, "add_blocks": None,
    }
    base.update(over)
    for k, v in base.items():
        setattr(a, k, v)
    a.command = "new-trial"
    return a


def _new_trial(run: Path, **over) -> tuple[int, Path | None]:
    cli = _cli()
    before = {d.name for d in (run / "trials").iterdir()}
    rc = cli.cmd_new_trial(_args(spec=str(run / "task-spec.yaml"),
                                 run=str(run), **over))
    after = [d for d in (run / "trials").iterdir() if d.name not in before]
    return rc, (after[0] if after else None)


_LOADED = [0]


def _load(path: Path, name: str | None = None):
    _LOADED[0] += 1
    nm = name or f"wp4_mod_{_LOADED[0]}"
    spec = importlib.util.spec_from_file_location(nm, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nm] = mod
    spec.loader.exec_module(mod)
    return mod


def _hurdle_data(n_tr=3000, n_va=1200, seed=0):
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
    X_va, y_va = mk(n_va, 1_000_000)
    return X_tr, y_tr, X_va, y_va


def _segments(seed=7, subsample=0.9, n_jobs=2):
    clf = xgb.XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=3,
                            subsample=subsample, random_state=seed,
                            n_jobs=n_jobs, tree_method="hist",
                            eval_metric="logloss", early_stopping_rounds=15)
    reg = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, max_depth=3,
                           random_state=seed, n_jobs=n_jobs,
                           tree_method="hist", early_stopping_rounds=15)
    return clf, reg


def _render_baseline(model_type: str, algo: str = "xgboost") -> str:
    from runtime.baseline import BASELINE_TEMPLATE, ESTIMATOR_CODE, FIT_CODE

    return BASELINE_TEMPLATE.format(
        algorithm=algo, target="y", model_type=model_type, treatment="",
        seed=20260814, n_estimators_cap=50, early_stopping_rounds=5,
        es_fraction=0.1, estimator_code=ESTIMATOR_CODE[algo],
        fit_code=FIT_CODE[algo])


def _write_baseline(run: Path, model_type: str, algo: str, tid: str) -> Path:
    from runtime.baseline import BaselineRunner
    from runtime.config import Config
    from runtime.spec import TaskSpec

    spec = TaskSpec.load(run / "task-spec.yaml")
    br = BaselineRunner(spec, Config.load(skill_config=False), run)
    tdir = run / "trials" / tid
    (tdir / "code").mkdir(parents=True, exist_ok=True)
    br.write_baseline_code(tdir, algo, [])
    return tdir


# ===========================================================================
# A. #7 规则 4 与单分支渲染
# ===========================================================================
@pytest.mark.parametrize("algo", ["xgboost", "lightgbm", "mlp"])
def test_regression_baseline_single_branch_passes_lint(tmp_path, algo):
    run = _mk_run(tmp_path)
    tdir = _write_baseline(run, "regression", algo,
                           f"trial-001-g00-explore-{algo}-1")
    code = (tdir / "code" / "train.py").read_text(encoding="utf-8")
    assert not _OLD_RULE4.search(code)          # 旧正则口径也过(在途 run 的 lint 副本)
    assert "单分支渲染" in code
    errs = L.lint_trial(run, tdir, False)
    assert not [e for e in errs if "规则 4" in e], errs
    assert L.regression_score_findings(code) == []


def test_classification_baseline_unchanged(tmp_path):
    from runtime.templates import narrow_model_type_dispatch

    run = _mk_run(tmp_path, "binary_classification")
    tdir = _write_baseline(run, "binary_classification", "xgboost",
                           "trial-001-g00-explore-xgboost-1")
    code = (tdir / "code" / "train.py").read_text(encoding="utf-8")
    assert "score = model.predict_proba(X_va)[:, 1]" in code
    assert "单分支渲染" not in code
    raw = _render_baseline("binary_classification")
    assert narrow_model_type_dispatch(raw) == (raw, [])
    assert narrow_model_type_dispatch(raw, "uplift") == (raw, [])


def test_narrowing_rules_and_idempotence():
    from runtime.templates import narrow_model_type_dispatch

    src = (
        'MODEL_TYPE = "regression"\n'
        "def f(model, X, Xt, Xc):\n"
        '    if hasattr(model, "score_validation"):\n'
        '        if MODEL_TYPE == "uplift":\n'
        '            raise ValueError("x")\n'
        "        score = model.score_validation(X)\n"
        '    elif MODEL_TYPE == "uplift":\n'
        "        score = model.predict_proba(Xt)[:, 1] \\\n"
        "            - model.predict_proba(Xc)[:, 1]\n"
        '    elif MODEL_TYPE == "binary_classification":\n'
        "        score = model.predict_proba(X)[:, 1]\n"
        "    else:\n"
        "        score = model.predict(X)\n"
        '    if MODEL_TYPE == "uplift":\n'
        "        dump(Xt)\n"
        "    return score\n"
        "def g(m, X):\n"
        '    if hasattr(m, "predict_proba"):\n'
        "        return m.predict_proba(X)[:, 1]\n"
        "    return m.predict(X)\n")
    out, notes = narrow_model_type_dispatch(src)
    assert notes and "predict_proba" not in out
    assert '        if MODEL_TYPE == "uplift":\n            raise' in out  # 非打分分派原样
    assert 'if MODEL_TYPE == "uplift":\n        dump(Xt)' in out
    assert "    else:\n        score = model.predict(X)" in out
    assert narrow_model_type_dispatch(out) == (out, [])
    # 自定义容器暴露 predict_proba 时,hasattr 分派不折叠(不改容器语义)
    custom = src + "class C:\n    def predict_proba(self, X):\n        return X\n"
    out2, _ = narrow_model_type_dispatch(custom)
    assert 'if hasattr(m, "predict_proba"):' in out2


def test_rule4_taint_semantics(tmp_path):
    run = _mk_run(tmp_path)
    bad = ('MODEL_TYPE = "regression"\nimport pandas as pd\n'
           "def fit_and_score(X, y, Xv, yv=None, seed=0, cfg=None):\n"
           "    s = clf.predict_proba(Xv)\n"
           "    return s[:, 1], {}\n"
           "def main():\n"
           "    score, _ = fit_and_score(1, 2, 3)\n"
           '    pd.DataFrame({"score": score}).to_parquet("v.parquet")\n')
    errs = L.check_regression_score_semantics(run, bad)
    assert errs and "规则 4" in errs[0] and "第 5 行" in errs[0]
    sig = ('MODEL_TYPE = "regression"\nimport torch\n'
           "def main():\n"
           "    score = torch.sigmoid(net(X)).squeeze(1).numpy()\n"
           "    dump_shadow_score(score)\n")
    assert L.check_regression_score_semantics(run, sig)
    # 不可达分支(分类/uplift)与 hasattr(predict_proba) 分派不报
    dead = ('MODEL_TYPE = "regression"\nimport pandas as pd\n'
            "def main():\n"
            '    if MODEL_TYPE == "binary_classification":\n'
            "        score = m.predict_proba(X)[:, 1]\n"
            '    elif hasattr(m, "predict_proba"):\n'
            "        score = m.predict_proba(X)[:, 1]\n"
            "    else:\n"
            "        score = m.predict(X)\n"
            '    pd.DataFrame({"score": score}).to_parquet("v.parquet")\n')
    assert L.check_regression_score_semantics(run, dead) == []
    assert _OLD_RULE4.search(dead)                 # 旧正则会误报的形态
    # hurdle:p × max(μ,0) 不是概率口径
    hurdle = ('MODEL_TYPE = "regression"\nimport numpy as np, pandas as pd\n'
              "class H:\n"
              "    def positive_prob(self, X):\n"
              "        pr = self.clf_.predict_proba(X)\n"
              "        return pr[:, 1]\n"
              "    def predict(self, X):\n"
              "        return self.positive_prob(X) * np.clip("
              "self.reg_.predict(X), 0, None)\n"
              "def main():\n"
              "    score = H().predict(X)\n"
              '    pd.DataFrame({"score": score}).to_parquet("v.parquet")\n'
              "    p = H().positive_prob(X)\n"
              '    pd.DataFrame({"score": p * mu}).to_parquet("w.parquet")\n')
    assert L.check_regression_score_semantics(run, hurdle) == []
    # 分类任务不适用规则 4
    run_c = _mk_run(tmp_path / "c", "binary_classification")
    assert L.check_regression_score_semantics(run_c, bad) == []


def test_template_and_hpo_render_paths_pass_rule4(tmp_path):
    run = _mk_run(tmp_path)
    rc, tdir = _new_trial(run, action="explore")
    assert rc == 0 and tdir is not None
    code = (tdir / "code" / "train.py").read_text(encoding="utf-8")
    assert not _OLD_RULE4.search(code)
    assert L.check_regression_score_semantics(run, code) == []
    sp = tmp_path / "space.yaml"
    dump_yaml(sp, {"max_depth": {"type": "int", "low": 2, "high": 6}})
    rc, tdir2 = _new_trial(run, action="hpo", search_space=str(sp),
                           changed_modules="model",
                           override_feature_phase="单测")
    assert rc == 0 and tdir2 is not None
    code2 = (tdir2 / "code" / "train.py").read_text(encoding="utf-8")
    assert not _OLD_RULE4.search(code2)
    assert L.check_regression_score_semantics(run, code2) == []


def test_hpo_graft_on_regression_baseline_passes_rule4(tmp_path):
    run = _mk_run(tmp_path)
    parent = _write_baseline(run, "regression", "xgboost",
                             "trial-001-g00-explore-xgboost-1")
    child = run / "trials" / "trial-002-g01-hpo-xgboost-1"
    (child / "code").mkdir(parents=True)
    # 旧(未折叠)的 baseline 血缘代码作父代副本:嫁接 + 副本升级后同样无概率分派
    old = (_render_baseline("regression")
           .replace('SKILL_ROOT = ""', f"SKILL_ROOT = {str(SKILL)!r}", 1))
    assert _OLD_RULE4.search(old)
    (child / "code" / "train.py").write_text(old, encoding="utf-8")
    R._apply_es_source_upgrade(child / "code" / "train.py", harden=True)
    assert not _OLD_RULE4.search(
        (child / "code" / "train.py").read_text(encoding="utf-8"))

    class FakeTrial:
        trial_dir = str(child)
        trial_id = child.name
        data = {"algorithm": "xgboost"}

        def flush(self):
            pass

    logging.disable(logging.CRITICAL)
    try:
        R._graft_hpo_section(FakeTrial())
    finally:
        logging.disable(logging.NOTSET)
    code = (child / "code" / "train.py").read_text(encoding="utf-8")
    assert "v19 hpo 嫁接段 BEGIN" in code
    assert not _OLD_RULE4.search(code)
    assert L.check_regression_score_semantics(run, code) == []
    assert parent.exists()


# ===========================================================================
# B. #22 --changed-modules 严格枚举
# ===========================================================================
def test_changed_modules_rejects_unknown(tmp_path):
    from _mvo.context import logger as _lg

    run = _mk_run(tmp_path)
    seen: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    h = _H()
    _lg.addHandler(h)
    try:
        rc, tdir = _new_trial(run, changed_modules="feature,sample_weight")
    finally:
        _lg.removeHandler(h)
    assert rc == 1 and tdir is None
    msg = " ".join(seen)
    assert "sample_weight" in msg and "feature, sample, model, loss, data" in msg
    rc, tdir = _new_trial(run, changed_modules="Feature, sample")
    assert rc == 0 and tdir is not None


# ===========================================================================
# C. #22 guard:引号参数文本不算调用
# ===========================================================================
def _guard(payload: dict, env: dict | None = None) -> str:
    e = dict(os.environ, CLAUDE_PROJECT_DIR="/x", **(env or {}))
    p = subprocess.run([sys.executable, str(SKILL / "assets" /
                                            "guard_main_session.py")],
                       input=json.dumps(payload), capture_output=True,
                       text=True, env=e)
    return "deny" if '"deny"' in p.stdout else "allow"


def _bash(cmd: str) -> dict:
    return {"agent_id": "a", "tool_name": "Bash", "tool_input": {"command": cmd}}


_MV = "python /s/scripts/modelevo.py"


@pytest.mark.parametrize("cmd", [
    f'{_MV} new-trial --run /r --rationale "待用户执行 {_MV} {_AP} --trial t"'
    ' --hypothesis "h"',
    f"python3 /s/scripts/modelevo.py new-trial --hypothesis '先 {_AP} 再说'",
    f'{_MV} new-trial --rationale="见 approvals/model-topology.jsonl 与 {_AP}"',
    f"{_MV} {_AP} --run /r --list",
    "cat /r/approvals/model-topology.jsonl",
])
def test_guard_allows_quoted_text(cmd):
    assert _guard(_bash(cmd)) == "allow"


@pytest.mark.parametrize("cmd", [
    f"{_MV} {_AP} --run /r --trial t --reason x",
    f'{_MV} new-trial --rationale "x" ; {_MV} {_AP} --trial t',
    f'{_MV} new-trial --rationale "$({_MV} {_AP} --trial t)"',
    f"{_MV} new-trial --rationale {_AP}",
    f"bash -c '{_MV} {_AP} --trial t'",
    f"python -c 'from runtime.topology import {_LEDGER_FN}'",
    f"python3 - <<'EOF'\nimport os\nos.execvp('python', ['python', "
    f"'scripts/modelevo.py', '{_AP}', '--trial', 't'])\nEOF",
    "echo {} >> /r/approvals/model-topology.jsonl",
    f'python other.py --msg "{_AP}"',
    f'{_MV} "{_AP}" --trial t',
])
def test_guard_still_denies_real_calls(cmd):
    assert _guard(_bash(cmd)) == "deny"
    # 无豁免:MODELEVO_ALLOW_MAIN_EXEC 不放行
    assert _guard(_bash(cmd), {"MODELEVO_ALLOW_MAIN_EXEC": "1"}) == "deny"


def test_guard_write_ledger_still_denied():
    assert _guard({"agent_id": "a", "tool_name": "Write", "tool_input": {
        "file_path": "/r/approvals/model-topology.jsonl"}}) == "deny"


# ===========================================================================
# D. baseline 补齐已配置算法
# ===========================================================================
def test_baseline_backfills_configured_algorithms(tmp_path, capsys):
    from runtime.baseline import BaselineRunner, resolve_baseline_algorithms
    from runtime.config import Config
    from runtime.spec import TaskSpec

    run = _mk_run(tmp_path)
    (run / "modelevo-config.yaml").write_text(yaml.safe_dump({"modelevo": {
        "baseline": {"algorithms": ["lightgbm", "xgboost"], "max_trials": 2}}}),
        encoding="utf-8")
    cfg = Config.load(run_config=run / "modelevo-config.yaml")
    # cmd_baseline 的 v2.15 规则在内存里收缩成 [xgboost]
    cfg._data["baseline"]["algorithms"] = ["xgboost"]
    br = BaselineRunner(TaskSpec.load(run / "task-spec.yaml"), cfg, run)

    class _Plan:
        n_parallel_trials = 1

    algos = br.plan_baseline_trials(_Plan(), force_all=True)
    assert algos == ["xgboost", "lightgbm"]          # 锚点 xgboost 在前
    out = capsys.readouterr().out
    assert "[baseline] 算法计划" in out and "lightgbm" in out
    assert br.last_plan_trace["legacy_collapse_reverted"] is True
    # 显式不补:skip_algorithms + stdout 留痕
    a2, tr = resolve_baseline_algorithms(["lightgbm", "xgboost"],
                                         skip=["lightgbm"],
                                         warm_covered=["xgboost"])
    assert a2 == ["xgboost"]
    assert tr["skipped"][0]["algorithm"] == "lightgbm"
    assert "显式配置" in tr["skipped"][0]["reason"]
    a3, tr3 = resolve_baseline_algorithms(["lightgbm", "xgboost"],
                                          warm_covered=["xgboost"])
    assert a3 == ["xgboost", "lightgbm"] and tr3["backfilled"] == ["lightgbm"]
    a4, _ = resolve_baseline_algorithms(["lightgbm", "xgboost", "mlp"],
                                        model_type="uplift", max_trials=3)
    assert a4 == ["xgboost", "lightgbm"]


# ===========================================================================
# E. #16 --add-blocks
# ===========================================================================
_SRC_BLOCK_TRIAL = '''import os
import pathlib
import sys

import numpy as np
import pandas as pd

RUN_DIR = pathlib.Path(__file__).resolve().parents[3]
SPLITS = RUN_DIR / "sample-profile" / "splits"
SKILL_ROOT = __SKILL__
if SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)
from runtime.feature_cache import feature_block  # noqa: E402

# 事件词表(块内核依赖的模块级常量)
_SEQX_HEADS = ("E0", "E1", "E2", "E3")
_SEQX_W = {"E0": 1.0, "E1": 2.0, "E2": 3.0, "E3": 4.0}
_SEQX_COLS = [f"seqx_{h}" for h in _SEQX_HEADS] + ["seqx_len"]


def n_jobs():
    return 1


def _seqx_row(s):
    toks = [t for t in (s or "").split(",") if t]
    out = [float(sum(_SEQX_W[h] for t in toks if t == h)) for h in _SEQX_HEADS]
    return out + [float(len(toks))]


def _seqx_kernel(b):
    return [_seqx_row(s) for s in b["seq"]]


def _seqx_block(split):
    return feature_block("seqx_v1", _seqx_kernel, split=split,
                         columns=_SEQX_COLS, sources=["seq"],
                         splits_dir=SPLITS, n_jobs=n_jobs())


def trial_feature_blocks(split):
    return _seqx_block(split)


def main():
    pass


if __name__ == "__main__":
    main()
'''


def _mk_source_trial(run: Path, tid="trial-001-g00-explore-xgboost-1",
                     code: str | None = None) -> Path:
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "code" / "train.py").write_text(
        (code or _SRC_BLOCK_TRIAL).replace("__SKILL__", repr(str(SKILL))),
        encoding="utf-8")
    return d


def test_add_blocks_copies_segment_with_matching_fingerprint(tmp_path):
    run = _mk_run(tmp_path)
    src = _mk_source_trial(run)
    rc, tdir = _new_trial(run, add_blocks=f"{src.name}:seqx_v1")
    assert rc == 0 and tdir is not None
    code = (tdir / "code" / "train.py").read_text(encoding="utf-8")
    assert R.ADD_BLOCKS_BEGIN in code and R.ADD_BLOCKS_END in code
    src_code = (src / "code" / "train.py").read_text(encoding="utf-8")
    for seg in ("def _seqx_row(s):", "def _seqx_kernel(b):",
                "_SEQX_W = {", "# 事件词表(块内核依赖的模块级常量)\n_SEQX_HEADS"):
        assert seg in src_code and seg in code, seg     # 字节级复制(含注释)
    tr = load_yaml(tdir / "trial.yaml")
    ab = tr["added_blocks"]
    assert len(ab) == 1 and ab[0]["match"] is True
    assert ab[0]["feature_block_name"] == "seqx_v1" and ab[0]["n_columns"] == 5
    # 新/源 trial 的块:同键(缓存命中)、同值;trial_feature_blocks 已注册
    m_src = _load(src / "code" / "train.py")
    m_new = _load(tdir / "code" / "train.py")
    W_src = m_src._seqx_block("train")
    W_new = m_new.trial_feature_blocks("train")
    assert list(W_new.columns) == list(W_src.columns)
    assert np.array_equal(W_new.to_numpy(), W_src.to_numpy())
    from runtime.feature_cache import code_fingerprint
    assert code_fingerprint(m_src._seqx_kernel) == \
        code_fingerprint(m_new._seqx_kernel)
    assert len(list((run / ".feature-cache").glob("seqx_v1-*"))) == 1
    errs = L.lint_trial(run, tdir, False)
    assert not [e for e in errs if "规则 19" in e or "规则 13" in e], errs
    # 再追加同一块 → 拒绝(重名列)
    with pytest.raises(SystemExit):
        R.assemble_add_blocks(code, [{"trial": src.name, "block": "seqx_v1",
                                      "code": src_code}])


def test_add_blocks_rejects_fingerprint_mismatch(tmp_path):
    run = _mk_run(tmp_path)
    src = _mk_source_trial(run)
    # 目标 trial 已有同名常量但取值不同 → 内核指纹 ≠ 源 → 拒绝并给差异
    dst_code = _SRC_BLOCK_TRIAL.replace(
        '_SEQX_W = {"E0": 1.0, "E1": 2.0, "E2": 3.0, "E3": 4.0}',
        '_SEQX_W = {"E0": 1.0, "E1": 2.0, "E2": 3.0, "E3": 9.0}').replace(
        "def trial_feature_blocks(split):\n    return _seqx_block(split)\n", ""
    ).replace("def _seqx_block(split):", "def _other_block(split):").replace(
        "def _seqx_kernel(b):", "def _other_kernel(b):")
    dst = _mk_source_trial(run, "trial-002-g01-explore-xgboost-1", dst_code)

    class T:
        trial_dir = str(dst)
        trial_id = dst.name

    with pytest.raises(SystemExit) as ei:
        R.apply_add_blocks(run, T(), f"{src.name}:_seqx_block")
    msg = str(ei.value)
    assert "指纹校验失败" in msg and "code_fingerprint" in msg
    assert "_SEQX_W" in msg and "9.0" in msg            # 冲突定义的差异
    with pytest.raises(SystemExit):
        R.parse_add_blocks("no-colon")
    with pytest.raises(SystemExit):
        R.resolve_block_wrapper(_SRC_BLOCK_TRIAL, "nope")


# ===========================================================================
# F. #15 HurdleRegressor
# ===========================================================================
def test_hurdle_regressor_protocol(tmp_path):
    from sklearn.base import clone, is_regressor

    from runtime.hurdle import HurdleRegressor

    clf, reg = _segments()
    h = HurdleRegressor(clf=clf, reg=reg, submodel_cache=False)
    assert is_regressor(h) and not hasattr(h, "predict_proba")
    X_tr, y_tr, X_va, y_va = _hurdle_data()
    h.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    assert not hasattr(h, "predict_proba")
    s = h.predict(X_va)
    p, mu = h.positive_prob(X_va), h.conditional_mean(X_va)
    assert np.allclose(s, p * np.maximum(mu, 0)) and (s >= 0).all()
    tel = h.segment_telemetry()
    # 两段各自早停(各自的 best_iteration / 实训轮数)
    assert tel["clf_best_iteration"] and tel["reg_best_iteration"]
    assert tel["clf_rounds"] < 120 and tel["reg_rounds"] < 120
    assert h.reg_.evals_result()["validation_0"]["rmse"]
    # clone:未拟合、参数等价
    h2 = clone(h)
    assert not hasattr(h2, "clf_")
    assert h2.get_params()["clf__max_depth"] == 3
    # 分段 set_params:clf__/reg__ 只作用一段,无前缀两段共用,别名
    h2.set_params(reg__max_depth=5, clf__min_child_weight=4.0, hurdle_lr=0.05,
                  hurdle_cap=99)
    assert h2.reg.max_depth == 5 and h2.clf.max_depth == 3
    assert h2.clf.min_child_weight == 4.0 and h2.reg.min_child_weight is None
    assert h2.clf.learning_rate == h2.reg.learning_rate == 0.05
    assert h2.clf.n_estimators == h2.reg.n_estimators == 99
    assert h.reg.max_depth == 3                           # 原对象不受影响
    with pytest.raises(ValueError):
        h2.set_params(no_such_param=1)
    # 无原型:按 *_params 构造,分段键写进 *_params
    h3 = HurdleRegressor(clf_params={"n_estimators": 20},
                         reg_params={"n_estimators": 30}, submodel_cache=False)
    h3.set_params(reg__max_depth=2)
    assert h3.reg_params == {"n_estimators": 30, "max_depth": 2}
    assert clone(h3).reg_params == h3.reg_params
    # 冻结轮数重训(交付 frozen_rounds:无 eval_set,早停自动关闭)
    h4 = clone(h).set_params(clf__n_estimators=10, reg__n_estimators=12)
    h4.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
    assert h4.clf_.get_booster().num_boosted_rounds() == 10
    assert h4.reg_.get_booster().num_boosted_rounds() == 12


def test_hurdle_regressor_joblib_new_process(tmp_path):
    import joblib

    from runtime.hurdle import HurdleRegressor

    X_tr, y_tr, X_va, y_va = _hurdle_data(seed=1)
    h = HurdleRegressor(*_segments(), submodel_cache=False).fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)])
    joblib.dump(h, tmp_path / "m.joblib")
    X_va.to_parquet(tmp_path / "xv.parquet")
    code = ("import sys, joblib, numpy as np, pandas as pd\n"
            f"sys.path.insert(0, {str(SKILL)!r})\n"
            f"m = joblib.load({str(tmp_path / 'm.joblib')!r})\n"
            f"X = pd.read_parquet({str(tmp_path / 'xv.parquet')!r})\n"
            f"np.save({str(tmp_path / 's.npy')!r}, m.predict(X))\n"
            "print(type(m).__module__, hasattr(m, 'predict_proba'))\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.split() == ["runtime.hurdle", "False"]
    assert np.array_equal(np.load(tmp_path / "s.npy"), h.predict(X_va))


def test_topology_recognizes_official_hurdle():
    from runtime.topology import _needs, analyze_code

    code = ("from runtime.hurdle import HurdleRegressor\n"
            "import xgboost as xgb\n"
            "def _fit_hurdle(X, y, Xe, ye):\n"
            "    m = HurdleRegressor(clf=xgb.XGBClassifier(), "
            "reg=xgb.XGBRegressor())\n"
            "    m.fit(X, y, eval_set=[(Xe, ye)])\n"
            "    return m\n"
            "def fit_and_score(X, y, Xv, yv=None, seed=0, cfg=None):\n"
            "    m = _fit_hurdle(X, y, Xv, yv)\n"
            "    return m.predict(Xv), {}\n")
    an = analyze_code(code)
    assert an["kind"] == "structural" and an["structural_forms"] == ["hurdle"]
    assert "structural:hurdle" in an["signature"]
    assert _needs(an, "regression") == []                  # 回归白名单免批准
    assert _needs(an, "binary_classification") == ["structural:hurdle"]


# ===========================================================================
# G. #10 子模型缓存
# ===========================================================================
def test_submodel_cache_hit_is_byte_identical(tmp_path, monkeypatch):
    from sklearn.base import clone

    from runtime import submodel_cache as SC

    monkeypatch.delenv("MODELEVO_SUBMODEL_CACHE", raising=False)
    X_tr, y_tr, X_va, y_va = _hurdle_data(seed=2)
    yc, yvc = (y_tr > 0).astype("int8"), (y_va > 0).astype("int8")
    clf, _ = _segments()
    fresh = clone(clf).fit(X_tr, yc, eval_set=[(X_va, yvc)], verbose=False)
    m1, i1 = SC.fit_or_load("clf", clf, X_tr, yc, eval_set=[(X_va, yvc)],
                            fit_kwargs={"verbose": False}, run_dir=tmp_path)
    assert i1["status"] == "miss" and i1["key"]
    # 不同线程数(不影响结果的参数)→ 同键命中;booster 逐字节一致
    clf8 = clone(clf).set_params(n_jobs=1)
    m2, i2 = SC.fit_or_load("clf", clf8, X_tr, yc, eval_set=[(X_va, yvc)],
                            fit_kwargs={"verbose": False}, run_dir=tmp_path)
    assert i2["status"] == "hit" and i2["key"] == i1["key"]
    assert SC.booster_bytes(m2) == SC.booster_bytes(m1) == SC.booster_bytes(fresh)
    assert np.array_equal(m2.predict_proba(X_va), fresh.predict_proba(X_va))
    assert m2.best_iteration == fresh.best_iteration
    assert m2.evals_result()["validation_0"]["logloss"]   # 遥测随缓存恢复
    assert clf.get_params()["n_jobs"] == 2                # 原型不被修改
    # 数据 / 参数 / seed 变化 → 新键
    X2 = X_tr.copy()
    X2.iloc[0, 0] += 1.0
    _, i3 = SC.fit_or_load("clf", clf, X2, yc, eval_set=[(X_va, yvc)],
                           run_dir=tmp_path)
    _, i4 = SC.fit_or_load("clf", clone(clf).set_params(random_state=8), X_tr,
                           yc, eval_set=[(X_va, yvc)], run_dir=tmp_path)
    assert i3["status"] == i4["status"] == "miss"
    assert len({i1["key"], i3["key"], i4["key"]}) == 3
    # 损坏的缓存:sha256 不符 → 重训覆盖
    d = Path(i1["path"])
    (d / "model.ubj").write_bytes(b"corrupt")
    m5, i5 = SC.fit_or_load("clf", clf, X_tr, yc, eval_set=[(X_va, yvc)],
                            run_dir=tmp_path)
    assert i5["status"] == "miss" and "缓存读取失败" in i5.get("reason", "")
    assert SC.booster_bytes(m5) == SC.booster_bytes(fresh)


def test_submodel_cache_switch_and_nondeterministic(tmp_path, monkeypatch):
    from runtime import submodel_cache as SC

    X_tr, y_tr, _, _ = _hurdle_data(n_tr=800, n_va=10, seed=3)
    reg = xgb.XGBRegressor(n_estimators=20, subsample=0.8, random_state=None,
                           tree_method="hist")
    _, info = SC.fit_or_load("reg", reg, X_tr, y_tr, run_dir=tmp_path)
    assert info["status"] == "uncacheable" and "seed" in info["reason"]
    assert not (tmp_path / SC.CACHE_DIRNAME).exists()
    lgb = pytest.importorskip("lightgbm")
    _, info = SC.fit_or_load("reg", lgb.LGBMRegressor(n_estimators=5,
                                                      verbosity=-1),
                             X_tr, y_tr, run_dir=tmp_path)
    assert info["status"] == "uncacheable" and "deterministic" in info["reason"]
    monkeypatch.setenv("MODELEVO_SUBMODEL_CACHE", "0")
    det = xgb.XGBRegressor(n_estimators=10, random_state=1, tree_method="hist")
    _, info = SC.fit_or_load("reg", det, X_tr, y_tr, run_dir=tmp_path)
    assert info["status"] == "disabled"
    assert not (tmp_path / SC.CACHE_DIRNAME).exists()
    monkeypatch.setenv("MODELEVO_SUBMODEL_CACHE", "1")
    _, info = SC.fit_or_load("reg", det, X_tr, y_tr, run_dir=tmp_path)
    assert info["status"] == "miss"
    _, info = SC.fit_or_load("reg", det, X_tr, y_tr, run_dir=tmp_path)
    assert info["status"] == "hit"


def test_hurdle_only_changed_segment_refits(tmp_path, monkeypatch):
    from sklearn.base import clone

    from runtime.hurdle import HurdleRegressor
    from runtime.submodel_cache import booster_bytes

    monkeypatch.delenv("MODELEVO_SUBMODEL_CACHE", raising=False)
    X_tr, y_tr, X_va, y_va = _hurdle_data(seed=4)
    h1 = HurdleRegressor(*_segments(), cache_dir=str(tmp_path)).fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)])
    s1 = h1.submodel_cache_summary()
    assert s1["misses"] == 2 and s1["hits"] == 0
    h2 = clone(h1).set_params(reg__max_depth=4).fit(
        X_tr, y_tr, eval_set=[(X_va, y_va)])
    s2 = h2.submodel_cache_summary()
    assert s2["segments"]["clf"]["status"] == "hit"
    assert s2["segments"]["reg"]["status"] == "miss"
    assert booster_bytes(h2.clf_) == booster_bytes(h1.clf_)
    assert booster_bytes(h2.reg_) != booster_bytes(h1.reg_)


# ===========================================================================
# H. hurdle 模板(new-trial --hurdle-template)
# ===========================================================================
def test_hurdle_template_trial_end_to_end(tmp_path, monkeypatch):
    from runtime.topology import analyze_code

    monkeypatch.delenv("MODELEVO_SUBMODEL_CACHE", raising=False)
    run = _mk_run(tmp_path)
    rc, tdir = _new_trial(run, hurdle_template=True, changed_modules="model",
                          override_feature_phase="单测")
    assert rc == 0 and tdir is not None
    tr = load_yaml(tdir / "trial.yaml")
    assert tr["declared_topology"] == "structural:hurdle"
    code = (tdir / "code" / "train.py").read_text(encoding="utf-8")
    assert "from runtime.hurdle import HurdleRegressor" in code
    assert "v2.16.2 hurdle 超参段 BEGIN" in code
    an = analyze_code(code)
    assert an["kind"] == "structural" and an["structural_forms"] == ["hurdle"]
    st = R._hurdle_lineage_status(code)
    assert st["hurdle"] and st["entries_ok"]
    assert L.lint_trial(run, tdir, False) == []
    dv = tr["deliverability"]
    assert dv["deliverable"] and dv["clone_ok"] and dv["rerun_recipe_ok"]
    assert dv["preferred_refit"] == "frozen_rounds" and not dv["issues"]
    # 分类任务 / --from-parent 同用 → 拒绝
    run_c = _mk_run(tmp_path / "c", "binary_classification")
    assert _new_trial(run_c, hurdle_template=True)[0] == 1
    assert _new_trial(run, hurdle_template=True,
                      from_parent=tdir.name)[0] == 1
    # 训练:分段超参覆写只改回归段 → 分类段命中缓存;metrics 带 submodel_cache
    m = _load(tdir / "code" / "train.py")
    m.N_ESTIMATORS_CAP = 80
    X_tr, y_tr, X_va, y_va = _hurdle_data(seed=5)
    s0, mx0 = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, {"hyperparameters": {}})
    assert mx0["submodel_cache"]["misses"] == 2
    assert mx0["model_num"] == 2 and mx0["hurdle"]["clf_best_iteration"]
    s1, mx1 = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7,
                              {"hyperparameters": {"reg__max_depth": 3}})
    assert mx1["hyperparameters_injected"] == {"reg__max_depth": 3}
    seg = mx1["submodel_cache"]["segments"]
    assert seg["clf"]["status"] == "hit" and seg["reg"]["status"] == "miss"
    assert m._last_model.reg_.get_params()["max_depth"] == 3
    # HPO(hurdle 超参段):inner 与重训都构造官方容器
    cfg = {"hyperparameters": {}, "seed": 7, "search_action": "hpo",
           "hpo": {"search_space": {"reg__reg_lambda": {
               "type": "float", "low": .01, "high": 10.0, "log": True}},
               "budget_configs": 3, "inner_fraction": 0.2,
               "lr_mode": "search", "parallel_configs": 1,
               "topology": "structural:hurdle", "enqueue": []}}
    s2, mx2 = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, cfg)
    assert len(s2) == len(X_va) and mx2["hpo_n_configs"] == 3
    assert "reg__reg_lambda" in mx2["best_params"]
    assert type(m._last_model).__name__ == "HurdleRegressor"
    st = json.loads((tdir / "hpo-study.json").read_text(encoding="utf-8"))
    assert st["topology"] == "structural:hurdle"


def test_hurdle_template_hpo_child_from_parent(tmp_path):
    run = _mk_run(tmp_path)
    rc, tdir = _new_trial(run, hurdle_template=True, changed_modules="model",
                          override_feature_phase="单测")
    assert rc == 0
    child = run / "trials" / "trial-009-g02-hpo-xgboost-1"
    (child / "code").mkdir(parents=True)
    (child / "config").mkdir()
    (child / "code" / "train.py").write_text(
        (tdir / "code" / "train.py").read_text(encoding="utf-8"),
        encoding="utf-8")
    dump_yaml(child / "config" / "trial.yaml", {"hyperparameters": {}})
    sp = tmp_path / "space.yaml"
    dump_yaml(sp, {"clf__max_depth": {"type": "int", "low": 2, "high": 6}})

    class T:
        trial_dir = str(child)
        trial_id = child.name
        data = {"algorithm": "xgboost", "parent_trial_ids": [tdir.name],
                "execution": {}, "expected_resources": {}}

        def flush(self):
            pass

    R._inject_hpo_config(run, T(), str(sp), 5)
    code = (child / "code" / "train.py").read_text(encoding="utf-8")
    assert code.count("v2.16.2 hurdle 超参段 BEGIN") == 1   # 幂等替换
    assert "v19 hpo 嫁接段 BEGIN" not in code
    assert T.data["execution"]["hpo_topology"] == "structural:hurdle"
    assert L.check_structural_override(code) == []
    assert L.check_regression_score_semantics(run, code) == []


# ===========================================================================
# I. #5-4 交付契约自检
# ===========================================================================
_HANDWRITTEN_HURDLE = '''import numpy as np
import pandas as pd
MODEL_TYPE = "regression"
_last_model = None


class HurdleXGB:
    def __init__(self, clf, reg):
        self.clf_, self.reg_ = clf, reg

    def predict(self, X):
        return self.clf_.predict_proba(X)[:, 1] * np.clip(self.reg_.predict(X), 0, None)

    def predict_proba(self, X):
        return self.clf_.predict_proba(X)


def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    global _last_model
    _last_model = HurdleXGB(None, None)
    return None, {}


def main():
    fit_and_score(None, None, None)


if __name__ == "__main__":
    main()
'''


def test_deliverability_probe_flags_handwritten_container(tmp_path):
    d = tmp_path / "t"
    (d / "code").mkdir(parents=True)
    (d / "code" / "train.py").write_text(_HANDWRITTEN_HURDLE, encoding="utf-8")
    dv = R.deliverability_probe(d, "regression")
    assert dv["clone_ok"] is False and "HurdleXGB" in dv["clone_detail"]
    assert any("predict_proba" in i for i in dv["issues"])
    assert dv["rerun_recipe_ok"] is False and dv["deliverable"] is False
    ok = _HANDWRITTEN_HURDLE.replace(
        "    def predict_proba(self, X):\n        return self.clf_.predict_proba(X)\n",
        "")
    (d / "code" / "train.py").write_text(ok, encoding="utf-8")
    dv2 = R.deliverability_probe(d, "regression")
    assert dv2["clone_ok"] is False and dv2["rerun_recipe_ok"] is True
    assert dv2["deliverable"] and dv2["preferred_refit"] == "rerun_recipe"


# ===========================================================================
# J. MDE 走 effective_mde
# ===========================================================================
def test_hpo_diminishing_uses_effective_mde(tmp_path):
    from _mvo.cmd_trial import _hpo_diminishing_check
    from runtime.config import Config

    run = _mk_run(tmp_path)
    (run / "run-state.json").write_text(json.dumps(
        {"baseline_id": "trial-001-g00-explore-xgboost-1"}), encoding="utf-8")
    b = run / "trials" / "trial-001-g00-explore-xgboost-1"
    b.mkdir(parents=True)
    (b / "evaluation.json").write_text(json.dumps(
        {"primary_metric": {"value": 12932.0}}), encoding="utf-8")
    h = run / "trials" / "trial-005-g03-hpo-xgboost-1"
    h.mkdir(parents=True)
    dump_yaml(h / "trial.yaml", {
        "trial_id": h.name, "search_action": "hpo", "generation": 3,
        "parent_trial_ids": [], "status": "EVALUATED",
        "evaluation": {"delta": {"vs_parent": 5.0}}})
    cfg = Config({"evaluator": {
        "minimum_meaningful_improvement": 0.001,
        "minimum_meaningful_improvement_relative": 0.0008}})
    ev = _hpo_diminishing_check(run, h.name, cfg)
    assert ev and ev["mde"] == pytest.approx(0.0008 * 12932.0)
    cfg_abs = Config({"evaluator": {"minimum_meaningful_improvement": 0.001}})
    assert _hpo_diminishing_check(run, h.name, cfg_abs) is None


def test_hurdle_template_hpo_and_from_parent_add_blocks(tmp_path):
    """--hurdle-template --action hpo:hurdle 口径(不注入 _HPO_ALGO);其子代
    --from-parent --add-blocks(无 --set)创建成功、指纹一致、拓扑继承。"""
    run = _mk_run(tmp_path)
    sp = tmp_path / "space.yaml"
    dump_yaml(sp, {"reg__max_depth": {"type": "int", "low": 2, "high": 5}})
    rc, tdir = _new_trial(run, hurdle_template=True, action="hpo",
                          search_space=str(sp), changed_modules="model",
                          override_feature_phase="单测")
    assert rc == 0 and tdir is not None
    code = (tdir / "code" / "train.py").read_text(encoding="utf-8")
    assert not R._has_module_level_algo(code)
    assert code.count("v2.16.2 hurdle 超参段 BEGIN") == 1
    cfg = load_yaml(tdir / "config" / "trial.yaml")
    assert cfg["hpo"]["topology"] == "structural:hurdle"
    assert L.lint_trial(run, tdir, False) == []
    src = _mk_source_trial(run, "trial-050-g00-explore-xgboost-1")
    rc2, child = _new_trial(run, from_parent=tdir.name, generation=None,
                            add_blocks=f"{src.name}:seqx_v1")
    assert rc2 == 0 and child is not None
    tr = load_yaml(child / "trial.yaml")
    assert tr["added_blocks"][0]["match"] is True
    assert tr["declared_topology"] == "structural:hurdle"
    assert tr["deliverability"]["deliverable"] is True
    # 没有 --set / --features / --add-blocks 的快通道仍拒绝
    assert _new_trial(run, from_parent=tdir.name, generation=None)[0] == 1
