"""v2.13.5 提速修复回归测试(req-t7-rand400w 23.6h 耗时拆解回写)。

覆盖:
1. 快速 bootstrap AUC 与 sklearn 逐次口径等价(含同分、单类重采样);
2. 分组置换线程预算 / xgboost 切 CPU;
3. 特征块缓存:代码指纹跨模块稳定、改常量即失配、并行 == 串行、
   跨进程磁盘命中、align_rows 行位置对齐;
4. lint 规则 7(hpo 死嫁接)/ 21(reparse 未走缓存)/ 22(无分数落盘);
5. hpo 嫁接改线:baseline 模板 main → fit_and_score;
6. from-parent 按 data_view 改写 SPLITS;
7. baseline 模板 trial_feature_blocks 钩子与 SKILL_ROOT 注入。
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(SKILL / "scripts"))


# ---------------------------------------------------------------------------
# 1. 快速 bootstrap AUC
# ---------------------------------------------------------------------------
def test_bootstrap_auc_fast_equals_sklearn(monkeypatch):
    from runtime import metrics as M

    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    # 刻意制造大量同分(四舍五入到 2 位)
    s_new = np.round(rng.random(n) + 0.4 * y, 2)
    s_old = np.round(rng.random(n) + 0.3 * y, 2)
    monkeypatch.setenv("MODELEVO_BOOTSTRAP_FAST_AUC", "1")
    fast = M.paired_bootstrap_delta(y, s_new, s_old, "auc", n_boot=150)
    ci_fast = M.bootstrap_ci(y, s_new, "auc", n_boot=150)
    monkeypatch.setenv("MODELEVO_BOOTSTRAP_FAST_AUC", "0")
    slow = M.paired_bootstrap_delta(y, s_new, s_old, "auc", n_boot=150)
    ci_slow = M.bootstrap_ci(y, s_new, "auc", n_boot=150)
    for k in ("delta_mean", "se", "p_positive"):
        assert abs(fast[k] - slow[k]) < 1e-12, (k, fast[k], slow[k])
    assert np.allclose(ci_fast, ci_slow, atol=1e-12)


def test_bootstrap_auc_single_class_resample_is_nan():
    from runtime.metrics import _BootstrapAUC

    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    f = _BootstrapAUC(y, s)
    assert f.ok
    counts = np.array([2, 2, 0, 0])  # 只抽到负类
    assert np.isnan(f.at(counts))
    assert f.at(np.array([1, 1, 1, 1])) == pytest.approx(1.0)


def test_bootstrap_auc_nonfinite_falls_back():
    from runtime.metrics import _BootstrapAUC

    f = _BootstrapAUC(np.array([0, 1, 0, 1]),
                      np.array([0.1, np.nan, 0.2, 0.3]))
    assert not f.ok


# ---------------------------------------------------------------------------
# 2. 分组置换线程预算 / 切 CPU
# ---------------------------------------------------------------------------
def test_gi_worker_threads_budget(monkeypatch):
    from runtime.group_importance import resolve_worker_threads

    monkeypatch.delenv("MODELEVO_GI_THREADS", raising=False)
    assert resolve_worker_threads(4, 24) == (4, 6)
    assert resolve_worker_threads(4, 2) == (2, 1)
    monkeypatch.setenv("MODELEVO_GI_THREADS", "8")
    assert resolve_worker_threads(4, 24) == (4, 2)


def test_gi_pin_cpu_xgboost():
    xgb = pytest.importorskip("xgboost")
    from runtime.group_importance import _pin_predict_device_cpu

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)).astype("float32")
    y = (X[:, 0] > 0).astype(int)
    m = xgb.XGBClassifier(n_estimators=5, max_depth=2)
    m.fit(X, y)
    assert _pin_predict_device_cpu(m, 2)
    cfg = json.loads(m.get_booster().save_config())
    assert cfg["learner"]["generic_param"]["device"] == "cpu"
    assert not _pin_predict_device_cpu(object(), 2)


# ---------------------------------------------------------------------------
# 3. 特征块缓存
# ---------------------------------------------------------------------------
_KERNEL_SRC = textwrap.dedent('''
    K = {k}
    HEADS = ("A", "B")

    def _row(tok):
        parts = tok.split(",") if tok else []
        return [sum(1 for p in parts if p == h) * K for h in HEADS]

    def kernel(b):
        return [_row(t) for t in b["seq"]]
''')


def _load_kernel(tmp: Path, name: str, k: int):
    p = tmp / f"{name}.py"
    p.write_text(_KERNEL_SRC.format(k=k), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_reparse(run: Path, n: int = 5000) -> Path:
    splits = run / "sample-profile" / "splits"
    (splits / "reparse").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    seq = [",".join(rng.choice(["A", "B", "C"], size=rng.integers(0, 6)))
           for _ in range(n)]
    pd.DataFrame({"seq": seq}).to_parquet(
        splits / "reparse" / "validation.parquet", index=False)
    return splits


def test_code_fingerprint_stable_across_modules_and_sensitive(tmp_path):
    from runtime.feature_cache import code_fingerprint

    a = _load_kernel(tmp_path, "trial_a", 1)
    b = _load_kernel(tmp_path, "trial_b", 1)   # 同源码、不同模块
    c = _load_kernel(tmp_path, "trial_c", 2)   # 改一个常量
    assert code_fingerprint(a.kernel) == code_fingerprint(b.kernel)
    assert code_fingerprint(a.kernel) != code_fingerprint(c.kernel)


def test_code_fingerprint_sees_object_state_and_frames():
    from runtime.feature_cache import code_fingerprint

    class Cfg:
        def __init__(self, k):
            self.k = k

    def make(cfg, frame):
        def kernel(b):
            return [[x * cfg.k + frame["w"].sum()] for x in b["v"]]
        return kernel

    df1 = pd.DataFrame({"w": [1.0, 2.0]})
    df2 = pd.DataFrame({"w": [1.0, 3.0]})
    assert code_fingerprint(make(Cfg(2), df1)) == \
        code_fingerprint(make(Cfg(2), df1.copy()))
    assert code_fingerprint(make(Cfg(2), df1)) != \
        code_fingerprint(make(Cfg(3), df1))
    assert code_fingerprint(make(Cfg(2), df1)) != \
        code_fingerprint(make(Cfg(2), df2))


# 内核经辅助函数间接引用 import 期运行态值(_T0)与运行中填充的 memo dict,
# 二级辅助函数 _tok 与常量 SCALE 各带一个可改的位置。
_VOLATILE_SRC = textwrap.dedent('''
    import time as _time
    import numpy as np

    _T0 = _time.time()
    _MEMO = {{}}
    SCALE = {scale}

    def _elapsed():
        return _time.time() - _T0

    def _tok(t):
        return t.split("{sep}") if t else []

    def _cnt(t):
        if t not in _MEMO:
            _MEMO[t] = len(_tok(t)) * SCALE
        _elapsed()
        return _MEMO[t]

    def kernel(b):
        return [[_cnt(t), float(np.float32(1))] for t in b["seq"]]
''')


def test_code_fingerprint_cross_process_stable_and_deep_sensitive(tmp_path):
    from runtime.feature_cache import code_fingerprint

    def _write(name, scale=1, sep=","):
        p = tmp_path / f"{name}.py"
        p.write_text(_VOLATILE_SRC.format(scale=scale, sep=sep),
                     encoding="utf-8")
        return p

    probe = textwrap.dedent('''
        import sys, importlib.util
        sys.path.insert(0, sys.argv[1])
        from runtime.feature_cache import code_fingerprint
        spec = importlib.util.spec_from_file_location(sys.argv[3], sys.argv[2])
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.kernel({"seq": ["a,b", "", "c"]})       # 填充 _MEMO
        print(code_fingerprint(m.kernel))
    ''')
    fps = []
    for name in ("proc_a", "proc_b"):
        r = subprocess.run([sys.executable, "-c", probe, str(SKILL),
                            str(_write(name)), name],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        fps.append(r.stdout.strip())
    assert fps[0] == fps[1]

    def _fp(name, **kw):
        p = _write(name, **kw)
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return code_fingerprint(mod.kernel)

    base = _fp("deep_base")
    assert base == fps[0]
    assert _fp("deep_scale", scale=2) != base     # 常量经一级辅助函数引用
    assert _fp("deep_sep", sep=";") != base       # 二级辅助函数改动


def test_feature_block_parallel_equals_serial_and_caches(tmp_path):
    from runtime import feature_cache as FC

    run = tmp_path / "run"
    splits = _make_reparse(run)
    mod = _load_kernel(tmp_path, "trial_k", 1)
    cols = ["seqx_A", "seqx_B"]
    FC._MEMO.clear()
    ser = FC.feature_block("seqx", mod.kernel, split="validation",
                           columns=cols, sources=["seq"], splits_dir=splits,
                           n_jobs=1, batch_rows=700, cache=False, log=None)
    par = FC.feature_block("seqx", mod.kernel, split="validation",
                           columns=cols, sources=["seq"], splits_dir=splits,
                           n_jobs=3, batch_rows=700, log=None)
    assert ser.equals(par)
    inv = FC.cache_inventory(run)
    assert len(inv) == 1 and inv[0]["n_rows"] == 5000
    # 另一个进程(模拟另一个 trial / finalize)同源内核 → 磁盘命中
    other = tmp_path / "trial_other.py"
    other.write_text(_KERNEL_SRC.format(k=1), encoding="utf-8")
    probe = textwrap.dedent(f'''
        import sys, importlib.util
        sys.path.insert(0, {str(SKILL)!r})
        from runtime import feature_cache as FC
        spec = importlib.util.spec_from_file_location("zzz", {str(other)!r})
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        logs = []
        df = FC.feature_block("seqx", m.kernel, split="validation",
                              columns={cols!r}, sources=["seq"],
                              splits_dir={str(splits)!r}, n_jobs=2,
                              log=logs.append)
        print("HIT" if any("hit(disk)" in x for x in logs) else "MISS",
              df.shape)
    ''')
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, timeout=120)
    assert "HIT" in r.stdout, r.stdout + r.stderr


def test_feature_block_changed_kernel_misses(tmp_path):
    from runtime import feature_cache as FC

    run = tmp_path / "run"
    splits = _make_reparse(run, 500)
    k1 = _load_kernel(tmp_path, "t1", 1)
    k2 = _load_kernel(tmp_path, "t2", 3)
    cols = ["seqy_A", "seqy_B"]
    FC._MEMO.clear()
    a = FC.feature_block("seqy", k1.kernel, split="validation", columns=cols,
                         sources=["seq"], splits_dir=splits, n_jobs=1,
                         log=None)
    b = FC.feature_block("seqy", k2.kernel, split="validation", columns=cols,
                         sources=["seq"], splits_dir=splits, n_jobs=1,
                         log=None)
    assert np.allclose(b.values, a.values * 3)
    assert len(FC.cache_inventory(run)) == 2


def test_feature_block_shape_contract(tmp_path):
    from runtime import feature_cache as FC

    splits = _make_reparse(tmp_path / "run", 50)
    with pytest.raises(ValueError):
        FC.feature_block("bad", lambda b: [[1.0]] * len(b["seq"]),
                         split="validation", columns=["x_a", "x_b"],
                         sources=["seq"], splits_dir=splits, n_jobs=1,
                         cache=False, log=None)


def test_align_rows_positional():
    from runtime.feature_cache import align_rows

    blk = pd.DataFrame({"a": np.arange(10, dtype="float32")})
    X = pd.DataFrame({"x": [0, 0, 0]}, index=[2, 5, 7])
    out = align_rows(blk, X)
    assert list(out["a"]) == [2.0, 5.0, 7.0] and list(out.index) == [2, 5, 7]
    with pytest.raises(ValueError):
        align_rows(blk, pd.DataFrame({"x": [0]}, index=["k"]))


# ---------------------------------------------------------------------------
# 4. lint 规则
# ---------------------------------------------------------------------------
def _lint():
    spec = importlib.util.spec_from_file_location(
        "lint_v2135", SKILL / "assets" / "lint_trial.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_lint_rule21_reparse_without_cache(tmp_path, monkeypatch):
    lt = _lint()
    run = tmp_path / "run"
    rp = run / "sample-profile" / "splits" / "reparse"
    rp.mkdir(parents=True)
    pd.DataFrame({"seq": ["a"] * 10}).to_parquet(rp / "train.parquet")
    code = textwrap.dedent('''
        REPARSE_DIR = SPLITS / "reparse"
        def reparse_frame(split, columns):
            return pd.read_parquet(REPARSE_DIR / f"{split}.parquet")
        def _widen_seq(split):
            return reparse_frame(split, ["seq"])
    ''')
    monkeypatch.setattr(lt, "_FEATURE_CACHE_MIN_ROWS", 5)
    assert lt.check_reparse_feature_cache(run, code)
    assert not lt.check_reparse_feature_cache(
        run, code + "\nX = feature_block('s', k, split='train')\n")
    # 只有模板自带的 reparse_frame 定义(未调用)不算消费
    only_def = "\n".join(code.splitlines()[:4])
    assert not lt.check_reparse_feature_cache(run, only_def)
    monkeypatch.setattr(lt, "_FEATURE_CACHE_MIN_ROWS", 10_000)
    assert not lt.check_reparse_feature_cache(run, code)


def test_lint_rule22_and_rule7_dead_graft(tmp_path):
    lt = _lint()
    assert lt.check_validation_score_output("x = 1")
    assert not lt.check_validation_score_output(
        'pd.DataFrame().to_parquet(TRIAL_DIR / "validation_score.parquet")')
    tdir = tmp_path / "trial-001-g01-xgboost-1"
    (tdir / "config").mkdir(parents=True)
    (tdir / "config" / "trial.yaml").write_text(
        "hpo:\n  search_space: {}\n", encoding="utf-8")
    dead = textwrap.dedent('''
        def fit_and_score(*a):
            run_hpo_search()
        def main():
            model = build_estimator(1)
            model.fit(1, 2)
    ''')
    errs = lt.check_hpo_selection_leak(tmp_path, tdir, dead)
    assert any("不调用 fit_and_score" in e for e in errs), errs
    live = dead.replace("model = build_estimator(1)",
                        "score, m = fit_and_score(1)")
    assert not any("不调用 fit_and_score" in e
                   for e in lt.check_hpo_selection_leak(tmp_path, tdir, live))


# ---------------------------------------------------------------------------
# 5. hpo 嫁接改线
# ---------------------------------------------------------------------------
def _render_baseline(algo: str = "xgboost") -> str:
    from runtime.baseline import BASELINE_TEMPLATE, ESTIMATOR_CODE, FIT_CODE

    return BASELINE_TEMPLATE.format(
        algorithm=algo, target="y", model_type="binary_classification",
        treatment="", seed=20260814, n_estimators_cap=10,
        early_stopping_rounds=5, es_fraction=0.1,
        estimator_code=ESTIMATOR_CODE[algo], fit_code=FIT_CODE[algo])


def test_rewire_main_to_fit_and_score_on_baseline_template():
    import ast

    from _mvo.render import _main_calls_name, _rewire_main_to_fit_and_score

    code = _render_baseline()
    out, notes = _rewire_main_to_fit_and_score(code)
    assert out is not None and notes
    tree = ast.parse(out)
    assert _main_calls_name(tree, "fit_and_score")
    assert "validation_score.parquet" in out
    assert "setdefault(_k, _v)" in out          # hpo 摘要并入 metrics
    assert "build_estimator(n_jobs" not in out.split("def main")[1]
    # 已调用 fit_and_score → 不重复改线
    assert _rewire_main_to_fit_and_score(out) == (None, [])


def test_rewire_refuses_unrecognizable_main():
    from _mvo.render import _rewire_main_to_fit_and_score

    code = "def main():\n    m = make()\n    m.fit(1)\n"
    with pytest.raises(SystemExit):
        _rewire_main_to_fit_and_score(code)


# ---------------------------------------------------------------------------
# 6. from-parent 按 data_view 改写 SPLITS
# ---------------------------------------------------------------------------
def test_retarget_splits_to_view(tmp_path):
    from _mvo.cmd_trial import _retarget_splits_to_view

    class _T:
        trial_id = "trial-009"
        trial_dir = tmp_path / "run" / "trials" / "trial-009"

    run = tmp_path / "run"
    (_T.trial_dir / "code").mkdir(parents=True)
    (run / "sample-profile" / "splits-sample").mkdir(parents=True)
    p = _T.trial_dir / "code" / "train.py"
    p.write_text('RUN_DIR = X\nSPLITS = RUN_DIR / "sample-profile" / "splits"\n',
                 encoding="utf-8")
    _retarget_splits_to_view(run, _T, "sample")
    assert "splits-sample" in p.read_text()
    _retarget_splits_to_view(run, _T, "v1")
    assert 'SPLITS = RUN_DIR / "sample-profile" / "splits"' in p.read_text()


# ---------------------------------------------------------------------------
# 7. baseline 模板:SKILL_ROOT 注入 + 特征块钩子
# ---------------------------------------------------------------------------
def test_baseline_template_feature_block_hook_and_skill_root():
    from runtime.code_audit import check_generated_code

    code = _render_baseline("lightgbm")
    assert 'SKILL_ROOT = ""' in code           # 渲染后由 write_baseline_code 注入
    assert "def _append_feature_blocks(X, split):" in code
    assert '_append_feature_blocks(X_tr, "train")' in code
    assert '_append_feature_blocks(_X_sh, "shadow_test")' in code
    assert check_generated_code(code) == []


def test_trial_body_feature_block_hook_and_widen_key():
    from _mvo.templates import TRIAL_BODY

    assert "def _append_feature_blocks(X, split):" in TRIAL_BODY
    assert '_append_feature_blocks(X_tr, "train")' in TRIAL_BODY
    assert '_append_feature_blocks(_X_sh, "shadow_test")' in TRIAL_BODY
    seg = TRIAL_BODY[TRIAL_BODY.find("def _widen_face_key"):
                     TRIAL_BODY.find("def widen_cached(")]
    # 键不再含 trial 名(跨 trial 复用),实现指纹进键
    assert "_widen_impl_sig()" in seg and "TRIAL_DIR.name}|{tag}" not in seg
    ns: dict = {}
    import pathlib as _pl
    ns.update(RUN_DIR=_pl.Path("/tmp/_rA"), TRIAL_DIR=_pl.Path("/tmp/t1"))
    exec("import hashlib\n" + seg, ns)

    def _widen(a, b, c, d):
        return a
    ns["_widen"] = _widen
    k1 = ns["_widen_face_key"]("main", 10, ["a"], 5, ["a"])
    ns["TRIAL_DIR"] = _pl.Path("/tmp/t2")   # 换 trial,同实现 → 同键
    assert ns["_widen_face_key"]("main", 10, ["a"], 5, ["a"]) == k1


def test_config_defaults_v2135():
    from runtime.config import DEFAULTS

    ev = DEFAULTS["evaluator"]
    assert ev["group_importance_sample_rows"] == 200000
    # v2.15.0:默认由 all 改为 auto(只改模型的 trial 继承父代族级置换,
    # 其余 trial 仍逐个重算 = all 的行为),见 tests/test_v2_15_0_hpo.py
    assert ev["group_importance_policy"] == "skip_rejected"
    assert DEFAULTS["baseline"]["parallel"] is True
