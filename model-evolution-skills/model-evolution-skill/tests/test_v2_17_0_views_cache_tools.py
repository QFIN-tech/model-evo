"""v2.17.0 WP5b 回归测试:流式数据视图 / 特征缓存键稳定 / 复合容器 GI /
prescreen / acknowledge --verify-against / 抽样面噪声预检。

全部离线、小合成数据(run4 regression.2 的事故形态缩微复现)。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.data import (SAMPLE_KEY_COLUMN, SPLIT_NAMES,  # noqa: E402
                          view_splits_dir, write_splits)
from runtime.logging_utils import dump_json, dump_yaml, load_json  # noqa: E402


# ---------------------------------------------------------------------------
# 合成 run
# ---------------------------------------------------------------------------
def _spec_dict(model_type: str = "regression", metric: str = "rmse") -> dict:
    return {
        "task_id": "wp5b", "task_name": "wp5b", "model_type": model_type,
        "target": {"name": "y", "positive_definition": "y>0"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": metric,
        "split_policy": {"type": "time_based", "train": "2026-01~2026-03",
                         "validation": "2026-04", "shadow_test": "2026-05",
                         "final_test": "2026-06"},
        "features": {"reparse": [{"column": "seq", "plan": "seq_ 解析"}],
                     "banned": [{"column": "leak", "reason": "future_window"}]},
    }


def _raw_frame(per_month: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    for m in range(1, 7):
        for i in range(per_month):
            day = 1 + (i % 28)
            rows.append((f"2026-0{m}-{day:02d}", f"u{uid:06d}"))
            uid += 1
    df = pd.DataFrame(rows, columns=["dt", "uid"])
    n = len(df)
    df["x_f"] = rng.normal(size=n)
    x_i = pd.Series(rng.integers(0, 50, size=n), dtype="float64")
    # int 列只在 validation(4 月)有缺失:全量拼接时整列 float64,
    # 而 train 单独读是 int64 —— 流式必须对齐到全量 dtype
    x_i[(df["dt"].str[:7] == "2026-04") & (rng.random(n) < 0.1)] = np.nan
    df["x_i"] = x_i
    df["x_s"] = rng.choice(["a", "bb", "ccc"], size=n)
    df["seq"] = [",".join(str(v) for v in rng.integers(0, 9, size=3))
                 for _ in range(n)]
    df["leak"] = rng.normal(size=n)
    df["y"] = np.where(rng.random(n) < 0.7,
                       np.clip(100 + 60 * df["x_f"]
                               + rng.normal(0, 10, size=n), 0, None),
                       0.0)
    return df


def _make_run(tmp: Path, per_month: int = 400, model_type="regression",
              metric="rmse") -> tuple[Path, "object"]:
    from runtime.spec import TaskSpec

    run = tmp / "run"
    run.mkdir(parents=True, exist_ok=True)
    spec_d = _spec_dict(model_type, metric)
    dump_yaml(run / "task-spec.yaml", spec_d)
    spec = TaskSpec.load(run / "task-spec.yaml")
    df = _raw_frame(per_month)
    if model_type == "binary_classification":
        df["y"] = (df["y"] > 0).astype(int)
    write_splits(df, spec, run / "sample-profile" / "splits")
    # train 的 x_i 落成 int64(无缺失),validation 是 float64(有缺失)
    base = run / "sample-profile" / "splits"
    tr = pd.read_parquet(base / "train.parquet")
    tr["x_i"] = tr["x_i"].astype("int64")
    tr.to_parquet(base / "train.parquet", index=False)
    # manifest sha 跟着刷新(模拟 profile 一次写成)
    import hashlib

    man = load_json(base / "manifest.json")
    man["splits"]["train"]["sha256"] = hashlib.sha256(
        (base / "train.parquet").read_bytes()).hexdigest()[:16]
    dump_json(base / "manifest.json", man)
    return run, spec


_ROW_LOCAL_VIEW = '''
import numpy as np
import pandas as pd


def build_view(df):
    out = df.copy()
    out["amt_ratio"] = df["x_f"] / (df["x_i"].fillna(0) + 1.0)
    out["amt_len"] = df["x_s"].str.len().astype("int64")
    out["amt_pos"] = np.where(df["x_f"] > 0, df["x_i"], np.nan)
    return out
'''

_NON_LOCAL_VIEW = '''
import pandas as pd


def build_view(df):
    out = df.copy()
    t = pd.to_datetime(df["dt"])
    out["rec_days"] = (t.max() - t).dt.days.astype(float)   # 依赖块内全局 max
    return out
'''


def _cli(argv: list[str]) -> int:
    import modelevo

    return modelevo.main(argv)


def _read_dir(d: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(d.rglob("*.parquet")):
        out[str(p.relative_to(d))] = pd.read_parquet(p)
    return out


# ---------------------------------------------------------------------------
# #12 流式视图
# ---------------------------------------------------------------------------
def test_stream_view_bitwise_equals_full_mode(tmp_path):
    run, _ = _make_run(tmp_path)
    src = run / "views" / "vamt.py"
    src.parent.mkdir()
    src.write_text(_ROW_LOCAL_VIEW, encoding="utf-8")
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vs",
                 "--source", str(src), "--chunk-rows", "1000"]) == 0
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vf",
                 "--source", str(src), "--no-stream"]) == 0
    ds, df_ = view_splits_dir(run, "vs"), view_splits_dir(run, "vf")
    a, b = _read_dir(ds), _read_dir(df_)
    assert sorted(a) == sorted(b) and len(a) >= 12
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], check_exact=True,
                                      obj=k)
    # 行序 / 样本键与 v1 逐位一致
    v1 = view_splits_dir(run)
    for n in SPLIT_NAMES:
        k1 = pd.read_parquet(v1 / "identity" / f"{n}.parquet")
        ks = pd.read_parquet(ds / "identity" / f"{n}.parquet")
        assert k1[SAMPLE_KEY_COLUMN].equals(ks[SAMPLE_KEY_COLUMN])
    ms, mf = load_json(ds / "manifest.json"), load_json(df_ / "manifest.json")
    assert list(ms) == list(mf)            # manifest 字段集合/顺序不变
    for k in ("n_features", "feature_sha256", "has_reparse_dir",
              "source_script_sha256"):
        assert ms[k] == mf[k]
    for n in ms["splits"]:
        es, ef = dict(ms["splits"][n]), dict(mf["splits"][n])
        es.pop("sha256"), ef.pop("sha256")
        assert es == ef
    bi = load_json(ds / "build-info.json")
    assert bi["mode"] == "stream" and bi["self_check"]["passed"] is True
    assert bi["n_chunks"] > len(SPLIT_NAMES)   # train 1200 行被切成 2 块
    assert load_json(df_ / "build-info.json")["mode"] == "full"


def test_recency_template_is_row_local_and_streams(tmp_path):
    """recency 模板的逐 split 参考日改为按全局行位置查表 → 流式自检通过,
    与全量模式逐位一致(旧实现按块内 max 取参考日,只能全量构建)。"""
    run, _ = _make_run(tmp_path)
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vrec",
                 "--template", "recency", "--chunk-rows", "1000"]) == 0
    src = run / "views" / "vrec.py"
    assert "@SPLIT_REF_DATES@" in src.read_text()
    assert "_SPLIT_REF_DATES = []" not in src.read_text()
    bi = load_json(view_splits_dir(run, "vrec") / "build-info.json")
    assert bi["mode"] == "stream" and bi["self_check"]["passed"] is True
    assert _cli(["write-splits-view", "--run", str(run), "--view-id",
                 "vrec_full", "--source", str(src), "--no-stream"]) == 0
    a = _read_dir(view_splits_dir(run, "vrec"))
    b = _read_dir(view_splits_dir(run, "vrec_full"))
    assert sorted(a) == sorted(b)
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], check_exact=True, obj=k)
    tr = a["train.parquet"]
    assert "w30_x_f" in tr.columns and "w30_y" not in tr.columns


def test_stream_view_memory_cap_shrinks_chunks(tmp_path):
    from runtime.data import write_view_splits_streaming
    from runtime.spec import TaskSpec

    run, spec = _make_run(tmp_path)
    ns: dict = {}
    exec(_ROW_LOCAL_VIEW, ns)
    m = write_view_splits_streaming(ns["build_view"], spec, run, "vcap",
                                    max_mem_gb=1e-4)
    bi = load_json(view_splits_dir(run, "vcap") / "build-info.json")
    assert bi["max_mem_gb"] == 1e-4
    assert bi["chunk_rows_final"] == 1000          # 上限换算后取下限 1000 行
    assert bi["n_chunks"] == 5                     # train 1200 → 2 块,其余各 1
    assert m["splits"]["train"]["rows"] == 1200
    assert TaskSpec  # noqa: B018


def test_stream_view_rejects_non_row_local(tmp_path):
    from runtime.data import ViewNotRowLocalError, write_view_splits_streaming

    run, spec = _make_run(tmp_path)
    ns: dict = {}
    exec(_NON_LOCAL_VIEW, ns)
    with pytest.raises(ViewNotRowLocalError):
        write_view_splits_streaming(ns["build_view"], spec, run, "vbad")
    assert not (view_splits_dir(run, "vbad") / "train.parquet").exists()
    # CLI:小数据 → 估算全量内存在预算内,回退全量模式(留痕 build-info)
    src = run / "views" / "vbad.py"
    src.parent.mkdir(exist_ok=True)
    src.write_text(_NON_LOCAL_VIEW, encoding="utf-8")
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vbad",
                 "--source", str(src)]) == 0
    assert load_json(view_splits_dir(run, "vbad") / "build-info.json")[
        "mode"] == "full"
    # 预算极小 → 拒绝回退
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vbad2",
                 "--source", str(src), "--max-mem-gb", "1e-9"]) == 1


# ---------------------------------------------------------------------------
# #12 --from-feature-block
# ---------------------------------------------------------------------------
_TRIAL_CODE = '''
import os
import pathlib
import sys

import numpy as np
import pandas as pd

RUN_DIR = pathlib.Path(__file__).resolve().parents[3]
SPLITS = RUN_DIR / "sample-profile" / "splits"
TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent
SKILL_ROOT = __SKILL__
if SKILL_ROOT and SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)
SEED = 20260814
ES_FRACTION = 0.1
from runtime.feature_cache import feature_block  # noqa: E402

_AMT_SOURCES = ["x_f", "x_i"]
_AMT_COLS = ["amt_a", "amt_b"]
_AMT_SCALE = __SCALE__


def _amt_kernel(b):
    xf = np.asarray(b["x_f"], dtype="float64")
    xi = np.asarray(b["x_i"], dtype="float64")
    return np.column_stack([xf * _AMT_SCALE,
                            np.where(xi > 10, xf / (xi + 1), np.nan)]
                           ).astype("float32")


def _amt_block(split):
    return feature_block("amt_v1", _amt_kernel, split=split,
                         columns=_AMT_COLS, sources=_AMT_SOURCES,
                         plane="features", batch_format="numpy",
                         splits_dir=SPLITS, n_jobs=1, log=None)


def trial_feature_blocks(split):
    return _amt_block(split)
'''


def _trial_code(scale: float = 2.0, extra: str = "") -> str:
    return (_TRIAL_CODE.replace("__SKILL__", repr(str(SKILL)))
            .replace("__SCALE__", repr(scale)) + extra)


def _mk_trial(run: Path, tid: str, code: str, *, status="EVALUATED",
              parents=(), view="v1", extra_rec: dict | None = None) -> Path:
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "code" / "train.py").write_text(code, encoding="utf-8")
    dump_yaml(d / "config" / "trial.yaml",
              {"trial_id": tid, "seed": 20260814, "multi_seed": 1,
               "search_action": "explore", "hyperparameters": {}})
    rec = {"trial_id": tid, "status": status, "data_view": view,
           "parent_trial_ids": list(parents), "generation": 1,
           "search_action": "explore",
           "execution": {"code_entry": "train.py",
                         "command": "python train.py --config "
                                    "../config/trial.yaml"},
           "status_history": [{"state": status}]}
    rec.update(extra_rec or {})
    dump_yaml(d / "trial.yaml", rec)
    return d


def test_from_feature_block_materializes_cached_block(tmp_path):
    import importlib.util

    from runtime import feature_cache as FC

    run, spec = _make_run(tmp_path)
    tdir = _mk_trial(run, "trial-005-g02-explore-xgboost-1", _trial_code())
    # trial 先跑过一次:块进缓存
    m_spec = importlib.util.spec_from_file_location(
        "wp5b_t005", tdir / "code" / "train.py")
    mod = importlib.util.module_from_spec(m_spec)
    m_spec.loader.exec_module(mod)
    FC._MEMO.clear()
    blocks = {n: mod._amt_block(n) for n in SPLIT_NAMES}
    FC._MEMO.clear()
    assert len(FC.cache_inventory(run)) == 4
    assert _cli(["write-splits-view", "--run", str(run), "--view-id",
                 "v2amt", "--from-feature-block",
                 "trial-005-g02-explore-xgboost-1:amt_v1"]) == 0
    vdir = view_splits_dir(run, "v2amt")
    v1 = view_splits_dir(run)
    for n in SPLIT_NAMES:
        face = pd.read_parquet(vdir / f"{n}.parquet")
        base = pd.read_parquet(v1 / f"{n}.parquet")
        pd.testing.assert_frame_equal(face[list(base.columns)], base,
                                      check_exact=True)
        got = face[["amt_a", "amt_b"]].to_numpy(np.float32)
        assert np.array_equal(got, blocks[n].to_numpy(np.float32),
                              equal_nan=True)
        for sub in ("identity", "labels", "reparse"):
            p = v1 / sub / f"{n}.parquet"
            if p.exists():
                assert (vdir / sub / f"{n}.parquet").read_bytes() == \
                    p.read_bytes()
    assert not (vdir / "labels" / "final_test.parquet").exists()
    bi = load_json(vdir / "build-info.json")
    assert bi["mode"] == "from_feature_block"
    assert bi["value_check"]["mismatches"] == 0
    assert bi["value_check"]["rows"] == 2400
    assert bi["spot_recompute"] and all(s["equal"]
                                        for s in bi["spot_recompute"])
    assert {e["how"] for e in bi["blocks"][0]["per_split"].values()} == \
        {"disk"}                                   # 全部命中缓存,未重算
    man = load_json(vdir / "manifest.json")
    m1 = load_json(v1 / "manifest.json")
    assert man["source_script"].endswith("code/train.py")
    assert man["n_features"] == len(pd.read_parquet(
        v1 / "train.parquet").columns) + 2
    assert set(man["splits"]) == set(m1["splits"])


def test_from_feature_block_recomputes_and_rejects_clash(tmp_path):
    from runtime import feature_cache as FC

    run, _ = _make_run(tmp_path)
    _mk_trial(run, "trial-006", _trial_code(scale=3.0))
    FC._MEMO.clear()
    # 无缓存 → 按内核重算并材料化
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vx",
                 "--from-feature-block", "trial-006:_amt_block"]) == 0
    face = pd.read_parquet(view_splits_dir(run, "vx") / "train.parquet")
    assert np.allclose(face["amt_a"], face["x_f"] * 3.0, rtol=1e-6)
    # 同名列再材料化到含该列的父视图 → 拒绝(视图只增列)
    _mk_trial(run, "trial-007", _trial_code().replace(
        '_AMT_COLS = ["amt_a", "amt_b"]', '_AMT_COLS = ["x_f", "amt_b"]'))
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vy",
                 "--from-feature-block", "trial-007:amt_v1"]) == 1
    # 块名不存在 → 报错并列出可用块函数
    assert _cli(["write-splits-view", "--run", str(run), "--view-id", "vz",
                 "--from-feature-block", "trial-006:nope_v9"]) == 1


# ---------------------------------------------------------------------------
# #13 特征缓存键稳定
# ---------------------------------------------------------------------------
def _fake_module(name: str, src: str):
    import types

    mod = types.ModuleType(name)
    mod.__dict__["__name__"] = name
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    sys.modules[name] = mod
    return mod


def test_cache_key_ignores_harness_module_changes(monkeypatch):
    from runtime import feature_cache as FC
    import runtime

    kernel_src = textwrap.dedent('''
        import numpy as np
        import runtime
        from {helper_mod} import helper

        def kernel(b):
            return [[helper(v), 1.0] for v in b["x"]]
    ''')

    def fp(helper_mod: str, helper_body: str) -> str:
        _fake_module(helper_mod, f"def helper(v):\n    return {helper_body}\n")
        k = _fake_module("wp5b_kernel_mod",
                         kernel_src.format(helper_mod=helper_mod))
        return FC.code_fingerprint(k.kernel)

    try:
        # harness 模块(runtime.*)的实现变化 → 键不变
        a = fp("runtime.zz_wp5b_helper", "v * 2")
        b = fp("runtime.zz_wp5b_helper", "v * 2 + 0  # changed")
        assert a == b
        monkeypatch.setattr(runtime, "__version__", "9.9.9")
        assert fp("runtime.zz_wp5b_helper", "v * 3") == a
        # 非 harness 辅助模块(trial/用户代码)照旧按身份记账 —— 但 trial 内核
        # 同模块的辅助函数改动必须失配
        k1 = _fake_module("wp5b_k1", "S = 2\ndef h(v):\n    return v * S\n"
                          "def kernel(b):\n    return [[h(v)] for v in b['x']]\n")
        f1 = FC.code_fingerprint(k1.kernel)
        k2 = _fake_module("wp5b_k1", "S = 3\ndef h(v):\n    return v * S\n"
                          "def kernel(b):\n    return [[h(v)] for v in b['x']]\n")
        assert FC.code_fingerprint(k2.kernel) != f1
    finally:
        for m in ("runtime.zz_wp5b_helper", "wp5b_kernel_mod", "wp5b_k1"):
            sys.modules.pop(m, None)


def test_fingerprint_probe_golden_guards_algorithm():
    from runtime import feature_cache as FC

    ver = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if ver not in FC._PROBE_GOLDEN:
        pytest.skip(f"无 Python {ver} 的探针金标准")
    # 改了 _fp_* 算法导致摘要变化 → 必须递增 CACHE_SCHEMA_VERSION 并更新金标准
    assert FC.fingerprint_probe_digest() == FC._PROBE_GOLDEN[ver]
    assert FC.CACHE_SCHEMA_VERSION == 2


def _load_trial_module(path: Path, name: str):
    import importlib.util

    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_cache_hits_across_rewrite_and_misses_on_rowset_or_schema(
        tmp_path, monkeypatch):
    from runtime import feature_cache as FC

    run, _ = _make_run(tmp_path)
    tdir = _mk_trial(run, "trial-010", _trial_code())
    mod = _load_trial_module(tdir / "code" / "train.py", "wp5b_t010")
    FC._MEMO.clear()
    logs: list[str] = []

    def _blk(split="validation"):
        return FC.feature_block("amt_v1", mod._amt_kernel, split=split,
                                columns=mod._AMT_COLS,
                                sources=mod._AMT_SOURCES, plane="features",
                                batch_format="numpy", splits_dir=mod.SPLITS,
                                n_jobs=1, log=logs.append)

    a = _blk()
    FC._MEMO.clear()
    # 内容不变的重新落盘(mtime 变)→ 仍命中(旧实现按 mtime 入键会失配)
    p = run / "sample-profile" / "splits" / "validation.parquet"
    data = p.read_bytes()
    os.utime(p, ns=(p.stat().st_atime_ns, p.stat().st_mtime_ns + 10**9))
    p.write_bytes(data)
    logs.clear()
    b = _blk()
    assert any("hit(disk)" in x for x in logs), logs
    assert a.equals(b)
    # 行集不同(删一行)→ 键不同 → 重算
    FC._MEMO.clear()
    pd.read_parquet(p).iloc[:-1].to_parquet(p, index=False)
    logs.clear()
    c = _blk()
    assert len(c) == len(a) - 1
    assert not any("hit(" in x for x in logs)
    # CACHE_SCHEMA_VERSION 递增 → 全部失效(失效预判同步)
    FC._MEMO.clear()
    before = FC.predict_invalidation(run)
    assert before["will_invalidate"] is False
    monkeypatch.setattr(FC, "CACHE_SCHEMA_VERSION", 3)
    pred = FC.predict_invalidation(run, ["feature_cache.py"])
    assert pred["will_invalidate"] is True
    assert pred["stale_by_reason"].get("schema_version_changed", 0) >= 2
    logs.clear()
    _blk()
    assert not any("hit(" in x for x in logs)


def test_legacy_cache_entry_is_silent_miss(tmp_path):
    from runtime import feature_cache as FC

    run, _ = _make_run(tmp_path)
    tdir = _mk_trial(run, "trial-011", _trial_code())
    mod = _load_trial_module(tdir / "code" / "train.py", "wp5b_t011")
    FC._MEMO.clear()
    df = FC.feature_block("amt_v1", mod._amt_kernel, split="validation",
                          columns=mod._AMT_COLS, sources=mod._AMT_SOURCES,
                          plane="features", batch_format="numpy",
                          splits_dir=mod.SPLITS, n_jobs=1, log=None)
    # 伪造一个 v1 旧格式条目(无 cache_schema_version)
    legacy = run / FC.CACHE_DIRNAME / "amt_v1-0123456789abcdef0123"
    legacy.mkdir(parents=True)
    df.to_parquet(legacy / "validation.parquet", index=False)
    (legacy / "validation.meta.json").write_text(json.dumps(
        {"key": "0123456789abcdef0123", "name": "amt_v1",
         "split": "validation", "n_rows": len(df), "format": 1,
         "source": {"path": "x"}}), encoding="utf-8")
    pred = FC.predict_invalidation(run)
    # 同 (块名, split) 已按新键重算 → 旧条目被取代,不计入失效
    assert pred["stale_by_reason"] == {"legacy_superseded": 1}
    assert pred["will_invalidate"] is False
    assert pred["n_valid"] == 1
    # 尚未重算的旧格式块 → 升级后首次使用必重算,预判为失效
    other = run / FC.CACHE_DIRNAME / "seqd_k0r-0123456789abcdef0123"
    other.mkdir()
    (other / "train.meta.json").write_text(json.dumps(
        {"key": "x", "name": "seqd_k0r", "split": "train", "format": 1}),
        encoding="utf-8")
    pred = FC.predict_invalidation(run, ["__init__.py", "feature_cache.py"])
    assert pred["stale_by_reason"]["legacy_not_recomputed"] == 1
    assert pred["will_invalidate"] is True


# ---------------------------------------------------------------------------
# #14 复合容器 group_importance
# ---------------------------------------------------------------------------
_HURDLE_TRIAL = '''
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent


class HurdleC:
    """只有 predict 的 hurdle 容器(无 feature_importances_ / get_booster)。"""

    def __init__(self, clf, reg):
        self.clf_ = clf
        self.reg_ = reg

    def positive_prob(self, X):
        return self.clf_.predict_proba(X)[:, 1]

    def conditional_mean(self, X):
        return self.reg_.predict(X)

    def predict(self, X):
        return self.positive_prob(X) * np.clip(self.conditional_mean(X),
                                               0.0, None)


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"pa_1": rng.normal(size=n), "pa_2": rng.normal(size=n),
                      "mu_1": rng.normal(size=n), "noise_1": rng.normal(size=n)}
                     ).astype("float32")
    pos = rng.random(n) < 1 / (1 + np.exp(-2.5 * X["pa_1"]))
    y = np.where(pos, 50 + 20 * X["mu_1"] + rng.normal(size=n), 0.0)
    return X, y


def main():
    X, y = make_data(3000, 0)
    clf = LogisticRegression().fit(X, (y > 0).astype(int))
    reg = LinearRegression().fit(X[y > 0], y[y > 0])
    model = HurdleC(clf, reg)
    mdir = TRIAL_DIR / "model"
    mdir.mkdir(exist_ok=True)
    joblib.dump(model, mdir / "model.joblib")   # 引用 __main__.HurdleC
    (mdir / "feature_columns.json").write_text(json.dumps(list(X.columns)))
    Xv, yv = make_data(1200, 1)
    Xv.to_parquet(mdir / "validation_input.parquet", index=False)
    pd.DataFrame({"y": yv}).to_parquet(mdir / "_yv.parquet", index=False)


if __name__ == "__main__":
    main()
'''


def _hurdle_trial(tmp_path) -> Path:
    run = tmp_path / "run"
    tdir = _mk_trial(run, "trial-015-g04-confirm-xgboost-1", _HURDLE_TRIAL)
    r = subprocess.run([sys.executable, "train.py"], cwd=tdir / "code",
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return tdir


def test_group_importance_composite_container(tmp_path):
    from runtime.group_importance import run_group_importance

    tdir = _hurdle_trial(tmp_path)
    with open(tdir / "model" / "model.joblib", "rb") as f:
        assert b"__main__" in f.read()          # 事故形态:类在 trial 的 __main__
    Xv = pd.read_parquet(tdir / "model" / "validation_input.parquet")
    yv = pd.read_parquet(tdir / "model" / "_yv.parquet")["y"].to_numpy()
    out = run_group_importance(tdir, Xv, yv, "rmse", n_repeat=2,
                               max_rows=0)
    assert out is not None, "复合容器 GI 不应空跑"
    assert out["predict_path"] == "generic_container"
    assert out["model_class"].endswith("HurdleC")
    fams = {r["family"]: r for r in out["top"]}
    assert set(fams) == {"pa", "mu", "noise"}
    assert fams["pa"]["delta_mean"] > 0 and fams["mu"]["delta_mean"] > 0
    assert abs(fams["noise"]["delta_mean"]) < fams["pa"]["delta_mean"]
    # hurdle 子模型分解:pa 主要通过分类子模型、mu 主要通过回归子模型起作用
    assert out["submodel_decomposition"]["available"] is True
    assert fams["pa"]["delta_clf_mean"] > fams["pa"]["delta_reg_mean"]
    assert fams["mu"]["delta_reg_mean"] > fams["mu"]["delta_clf_mean"]
    assert "HurdleC" not in vars(sys.modules["__main__"])   # 注入已撤销


def test_group_importance_composite_row_positions_and_cap(tmp_path):
    """v2.16.2 的"自存帧按行位置取子集"在复合容器上同样成立;通用路径受
    group_importance_sample_rows 约束(在行位置子集之上再抽)。"""
    from runtime.group_importance import run_group_importance

    tdir = _hurdle_trial(tmp_path)
    full = pd.read_parquet(tdir / "model" / "validation_input.parquet")
    yv = pd.read_parquet(tdir / "model" / "_yv.parquet")["y"].to_numpy()
    pos = np.arange(0, len(full), 2)                   # runner 抽样的行位置
    base = full[["pa_1"]].iloc[pos].reset_index(drop=True)   # 基础帧缺加宽列
    out = run_group_importance(tdir, base, yv[pos], "rmse", n_repeat=1,
                               row_positions=pos, n_full_rows=len(full),
                               max_rows=0)
    assert out["eval_frame_source"] == "trial_saved_validation_input"
    assert out["eval_frame_subset"]["by"] == "row_positions"
    assert out["n_rows"] == len(pos) and out["filled_constant_columns"] == 0
    assert out["families_unverifiable"] == []
    capped = run_group_importance(tdir, base, yv[pos], "rmse", n_repeat=1,
                                  row_positions=pos, n_full_rows=len(full),
                                  max_rows=250)
    assert capped["n_rows"] == 250
    assert capped["eval_frame_subset"]["by"] == "row_positions+generic_cap"
    # 配置键生效(无显式 max_rows 时读 run 配置)
    (tdir.parent.parent / "modelevo-config.yaml").write_text(
        "modelevo:\n  evaluator:\n    group_importance_sample_rows: 300\n",
        encoding="utf-8")
    cfg_cap = run_group_importance(tdir, base, yv[pos], "rmse", n_repeat=1,
                                   row_positions=pos, n_full_rows=len(full))
    assert cfg_cap["n_rows"] == 300


# ---------------------------------------------------------------------------
# #22 acknowledge-harness-change --verify-against
# ---------------------------------------------------------------------------
def _scored_trial(run: Path, spec, tid: str, score: np.ndarray, *,
                  parents=(), evaluate_with=None) -> Path:
    from runtime.integrity import bind_prediction_artifact, freeze_trial_source

    tdir = _mk_trial(run, tid, _trial_code(), parents=parents)
    man = freeze_trial_source(tdir)
    v1 = view_splits_dir(run)
    lab = pd.read_parquet(v1 / "labels" / "validation.parquet")
    pd.DataFrame({"score": score}).to_parquet(
        tdir / "validation_score.parquet", index=False)
    bind_prediction_artifact(tdir / "validation_score.parquet", lab,
                             source_manifest=man,
                             split_manifest_path=v1 / "manifest.json",
                             data_view="v1")
    if evaluate_with is not None:
        dump_json(tdir / "evaluation.json", evaluate_with(lab, score))
    return tdir


def _verify_fixture(tmp_path):
    from runtime.config import Config
    from runtime.evaluator import Evaluator
    from runtime.harness import protocol_stamp

    run, spec = _make_run(tmp_path)
    cfg = Config.load(run_config=run / "modelevo-config.yaml")
    lab = pd.read_parquet(view_splits_dir(run) / "labels"
                          / "validation.parquet")
    y = lab["y"].to_numpy()
    rng = np.random.default_rng(3)
    s_base = y + rng.normal(scale=30, size=len(y))
    s_child = y + rng.normal(scale=25, size=len(y))
    _scored_trial(run, spec, "trial-001-g00-explore-xgboost-1", s_base)

    def _ev(lab_, s):
        return Evaluator(spec, cfg).evaluate(
            lab_["y"].to_numpy(), s, parent_score=s_base,
            baseline_score=s_base, is_root_trial=False)

    _scored_trial(run, spec, "trial-002-g01-explore-xgboost-2", s_child,
                  parents=("trial-001-g00-explore-xgboost-1",),
                  evaluate_with=_ev)
    stamp = protocol_stamp(SKILL / "runtime")
    files = dict(stamp["harness_files"])
    files["evaluator.py"] = "0" * 16            # 模拟 measurement 模块变更
    dump_json(run / "run-state.json", {
        "baseline_id": "trial-001-g00-explore-xgboost-1",
        "harness_hash": "0123456789abcdef", "harness_files": files,
        "harness_epoch": 0})
    return run


def test_verify_against_consistent_allows_no_score_impact(tmp_path, capsys):
    run = _verify_fixture(tmp_path)
    rc = _cli(["acknowledge-harness-change", "--run", str(run), "--reason",
               "plumbing refactor", "--measurement-no-score-impact",
               "--verify-against", "trial-002-g01-explore-xgboost-2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "一致" in out and "特征块缓存预判" in out
    st = load_json(run / "run-state.json")
    entry = st["harness_change_log"][-1]
    assert entry["measurement_impact"] == "no-score-impact"
    assert st["harness_epoch"] == 0
    assert entry["verified_against"][0]["consistent"] is True
    assert entry["verified_against"][0]["n_fields"] >= 8
    assert "feature_cache" in entry


def test_verify_against_mismatch_refuses(tmp_path, capsys):
    run = _verify_fixture(tmp_path)
    ev_p = run / "trials" / "trial-002-g01-explore-xgboost-2" / \
        "evaluation.json"
    ev = load_json(ev_p)
    ev["delta"]["vs_parent"] += 1.0                # 评估口径"变了"
    ev["delta"]["vs_parent_paired_bootstrap"]["se"] *= 1.5
    dump_json(ev_p, ev)
    rc = _cli(["acknowledge-harness-change", "--run", str(run), "--reason",
               "x", "--measurement-no-score-impact",
               "--verify-against", "trial-002-g01-explore-xgboost-2"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "不一致" in out and "delta.vs_parent" in out
    assert "vs_parent_paired_bootstrap.se" in out
    st = load_json(run / "run-state.json")
    assert not st.get("harness_change_log")        # 未登记
    # 不带 --verify-against:行为不变(按旧口径降级登记)
    assert _cli(["acknowledge-harness-change", "--run", str(run),
                 "--reason", "x", "--measurement-no-score-impact"]) == 0


# ---------------------------------------------------------------------------
# #16 prescreen
# ---------------------------------------------------------------------------
_PS_TRIAL = '''
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

RUN_DIR = pathlib.Path(__file__).resolve().parents[3]
SPLITS = RUN_DIR / "sample-profile" / "splits"
TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent
SKILL_ROOT = '/nonexistent/old-skill'  # 渲染期注入(prescreen 改写为当前 skill)
if SKILL_ROOT and SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)
SEED = 20260814
ES_FRACTION = 0.1   # 早停尾部比例
FEATS = __FEATS__
USE_BLOCK = __BLOCK__


def _read(p):
    with open(TRIAL_DIR / "access.log", "a") as f:
        f.write(str(pathlib.Path(p).resolve()) + "\\n")
    return pd.read_parquet(p)


def load_labels(split):
    return _read(SPLITS / "labels" / f"{split}.parquet")


def _amt_kernel(b):
    return np.asarray(b["x_f"], dtype="float64").reshape(-1, 1) * 2.0


def _frame(split, df):
    X = df[FEATS].fillna(0).astype("float64")
    if USE_BLOCK:
        from runtime.feature_cache import feature_block

        blk = feature_block("ps_amt", _amt_kernel, split=split,
                            columns=["amt_x2"], sources=["x_f"],
                            plane="features", batch_format="numpy",
                            splits_dir=SPLITS, n_jobs=1, log=None)
        X = pd.concat([X.reset_index(drop=True), blk], axis=1)
    return X.to_numpy()


def main():
    from sklearn.linear_model import Ridge

    tr = _read(SPLITS / "train.parquet")
    va = _read(SPLITS / "validation.parquet")
    ytr = load_labels("train")["y"].to_numpy()
    load_labels("validation")
    es_src = os.environ.get("MODELEVO_ES_SOURCE", "validation")
    cut = int(len(tr) * (1 - ES_FRACTION)) if es_src == "train_tail" \\
        else len(tr)
    X = _frame("train", tr)
    rng = np.random.default_rng(SEED)
    X = X + rng.normal(scale=0.05, size=X.shape)
    model = Ridge(alpha=1.0).fit(X[:cut], ytr[:cut])
    score = model.predict(_frame("validation", va))
    pd.DataFrame({"score": score}).to_parquet(
        TRIAL_DIR / "validation_score.parquet", index=False)
    if (SPLITS / "shadow_test.parquet").exists():
        _read(SPLITS / "shadow_test.parquet")
    with open(TRIAL_DIR / "metrics.json", "w") as f:
        json.dump({"es_source": es_src, "es_fraction": ES_FRACTION,
                   "seed": SEED, "cut": cut, "n_train": len(tr),
                   "n_valid": len(va)}, f)


if __name__ == "__main__":
    main()
'''


def _ps_code(feats, block=False):
    return (_PS_TRIAL.replace("__FEATS__", repr(list(feats)))
            .replace("__BLOCK__", repr(bool(block))))


def test_prescreen_train_only_delta_se_and_no_holdout_labels(tmp_path,
                                                            capsys):
    run, _ = _make_run(tmp_path)
    ref = _mk_trial(run, "trial-020-g06-crossover-xgboost-1",
                    _ps_code(["x_i"]))
    cand = _mk_trial(run, "trial-021-g07-explore-xgboost-2",
                     _ps_code(["x_i", "x_f"], block=True),
                     status="CODE_WRITTEN",
                     parents=("trial-020-g06-crossover-xgboost-1",))
    before = {p: p.stat().st_mtime_ns for p in run.rglob("*") if p.is_file()}
    v1 = view_splits_dir(run)
    forbidden = [v1 / "labels" / "validation.parquet",
                 v1 / "labels" / "shadow_test.parquet",
                 v1 / "validation.parquet", v1 / "shadow_test.parquet",
                 v1 / "final_test.parquet"]
    for p in forbidden:
        p.chmod(0)                    # 任何读取都会 PermissionError
    try:
        rc = _cli(["prescreen", "--run", str(run), "--trial", cand.name,
                   "--seeds", "2", "--holdout-frac", "0.15",
                   "--es-frac", "0.05", "--n-boot", "200", "--threads", "1",
                   "--keep-sandbox"])
    finally:
        for p in forbidden:
            p.chmod(0o644)
    out = capsys.readouterr().out
    assert rc == 0, out
    rep = load_json(run / "prescreen" / cand.name / "report.json")
    assert rep["screening_evidence"] is True
    assert rep["enters_fdr"] is False and rep["enters_leaderboard"] is False
    assert rep["ref"] == ref.name and rep["seeds"] == 2
    assert len(rep["per_seed"]) == 2
    assert rep["per_seed"][0]["seed"] != rep["per_seed"][1]["seed"]
    assert rep["delta_mean"] < 0                     # rmse:候选(含 x_f)更好
    assert rep["se"] > 0 and rep["paired_bootstrap"]["se"] > 0
    assert rep["se_seed"] is not None
    sp = rep["split"]
    assert sp["n_fit"] + sp["n_es"] + sp["n_holdout"] == 1200
    assert sp["holdout_time_range"][0] > sp["fit_time_range"][1]
    # 访问日志:harness 侧只读 train 四产物
    assert rep["data_access"] and all(
        Path(p).stem == "train" for p in rep["data_access"])
    # trial 代码在沙箱里读的全是沙箱文件;早停面在 train 尾部;seed 注入生效
    sandbox = Path(rep["sandbox"]).resolve()
    tdirs = sorted((sandbox / "trials").glob("*__ps*"))
    assert len(tdirs) == 4
    for td in tdirs:
        paths = (td / "access.log").read_text().split()
        assert paths and all(str(sandbox) in p for p in paths), paths
        m = json.loads((td / "metrics.json").read_text())
        assert m["es_source"] == "train_tail"
        assert abs(m["es_fraction"] - sp["n_es"] / (sp["n_fit"] + sp["n_es"])
                   ) < 1e-6
        assert m["n_valid"] == sp["n_holdout"]
    # 特征块缓存在沙箱内按新行集计算(键含数据内容指纹)
    from runtime import feature_cache as FC

    inv = FC.cache_inventory(sandbox)
    assert inv and all(str(sandbox) in m["source"]["path"] for m in inv)
    assert not (run / FC.CACHE_DIRNAME).exists()
    # 真实 trial 目录 / 状态一字未改
    for p, mt in before.items():
        if p.exists() and "prescreen" not in str(p):
            assert p.stat().st_mtime_ns == mt, p
    assert not (cand / "validation_score.parquet").exists()


def test_prescreen_requires_code_written(tmp_path):
    run, _ = _make_run(tmp_path)
    _mk_trial(run, "trial-001", _ps_code(["x_i"]))
    _mk_trial(run, "trial-002", _ps_code(["x_f"]), status="PROPOSED",
              parents=("trial-001",))
    assert _cli(["prescreen", "--run", str(run), "--trial", "trial-002",
                 "--seeds", "1"]) == 1
    _mk_trial(run, "trial-003", _ps_code(["x_f"]), status="CODE_WRITTEN")
    assert _cli(["prescreen", "--run", str(run), "--trial", "trial-003",
                 "--seeds", "1"]) == 2               # 无父代且未给 --ref


def test_prescreen_access_guard_and_env_remap(tmp_path, monkeypatch):
    from runtime.data import (PrescreenAccessLog, build_prescreen_splits,
                              load_labels)

    run, spec = _make_run(tmp_path)
    log = PrescreenAccessLog()
    with pytest.raises(PermissionError):
        log.read(view_splits_dir(run) / "labels" / "validation.parquet")
    sb = tmp_path / "sb"
    info = build_prescreen_splits(run, spec, sb, access_log=log)
    assert info["n_holdout"] > 0
    # 沙箱环境变量下,指向真实 run 的访问被重映射进沙箱(validation=留出段)
    monkeypatch.setenv("MODELEVO_PRESCREEN_ROOT", str(sb))
    lab = load_labels(run)
    assert len(lab["validation"]) == info["n_holdout"]
    assert "final_test" not in lab


# ---------------------------------------------------------------------------
# #19 抽样面噪声预检
# ---------------------------------------------------------------------------
def _sample_fixture(tmp_path, *, runtime_sec=300.0, mde=0.2,
                    with_baseline=True):
    run, spec = _make_run(tmp_path)
    raw = run / "raw.parquet"
    _raw_frame().to_parquet(raw, index=False)
    (run / "data-loader.py").write_text(
        "import pandas as pd\n\ndef load_data():\n"
        f"    return pd.read_parquet({str(raw)!r})\n", encoding="utf-8")
    (run / "modelevo-config.yaml").write_text(
        "modelevo:\n  evaluator:\n"
        f"    minimum_meaningful_improvement: {mde}\n", encoding="utf-8")
    if with_baseline:
        lab = pd.read_parquet(view_splits_dir(run) / "labels"
                              / "validation.parquet")
        rng = np.random.default_rng(5)
        s = lab["y"].to_numpy() + rng.normal(scale=20, size=len(lab))
        tdir = _scored_trial(run, spec, "trial-001-g00-explore-xgboost-1", s)
        import yaml

        y_rec = yaml.safe_load((tdir / "trial.yaml").read_text())
        y_rec["result"] = {"status": "succeeded", "runtime_sec": runtime_sec}
        dump_yaml(tdir / "trial.yaml", y_rec)
        dump_json(run / "run-state.json",
                  {"baseline_id": "trial-001-g00-explore-xgboost-1"})
    return run


def test_noise_precheck_recommends_v1(tmp_path, capsys):
    run = _sample_fixture(tmp_path)
    assert _cli(["write-splits-sample", "--run", str(run),
                 "--max-rows", "1000"]) == 0
    out = capsys.readouterr().out
    man = load_json(view_splits_dir(run, "sample") / "manifest.json")
    npc = man["noise_precheck"]
    for k in ("recommend_v1", "se_ratio", "se_sample", "se_full", "mde",
              "reason"):
        assert k in npc
    assert npc["recommend_v1"] is True
    assert npc["se_sample"] > npc["se_full"] > 0
    assert npc["se_ratio"] > 1.2
    assert npc["mde"] == 0.2 and npc["baseline_runtime_min"] == 5.0
    assert npc["n_sample"] < npc["n_full"] == 400
    assert "噪声预检: recommend_v1=true" in out


def test_noise_precheck_no_recommend_when_slow_or_no_baseline(tmp_path,
                                                              capsys):
    run = _sample_fixture(tmp_path / "a", runtime_sec=3600.0)
    assert _cli(["write-splits-sample", "--run", str(run),
                 "--max-rows", "1000"]) == 0
    npc = load_json(view_splits_dir(run, "sample") / "manifest.json")[
        "noise_precheck"]
    assert npc["recommend_v1"] is False and npc["baseline_runtime_min"] == 60
    run2 = _sample_fixture(tmp_path / "b", with_baseline=False)
    assert _cli(["write-splits-sample", "--run", str(run2),
                 "--max-rows", "1000"]) == 0
    npc2 = load_json(view_splits_dir(run2, "sample") / "manifest.json")[
        "noise_precheck"]
    assert npc2["skipped"] is True and npc2["recommend_v1"] is False
    assert "baseline" in npc2["reason"]
    assert "噪声预检: 跳过" in capsys.readouterr().out
