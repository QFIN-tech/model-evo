"""v2.17.3:req-t7 run4 性能批次(6 项里的 1/3/4/5;2、6 实测已存在)。

覆盖:
  1. 特征块缓存可跨 run 共享(MODELEVO_FEATURE_CACHE_DIR),键仍是内容寻址;
  2. 搜索代跳过 shadow 打分(MODELEVO_SHADOW_SCORE),confirm 仍产出;
  3. 早停集降采样(MODELEVO_ES_EVAL_ROWS)——只影响"何时停",不改变评分口径;
  4. group_importance_repeat 默认 5 → 2。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------- 1) cache_root
def test_cache_root_default_is_run_local(tmp_path, monkeypatch):
    from runtime import feature_cache as FC

    monkeypatch.delenv("MODELEVO_FEATURE_CACHE_DIR", raising=False)
    assert FC.cache_root(tmp_path) == tmp_path / FC.CACHE_DIRNAME


def test_cache_root_shared_dir_absolute_and_relative(tmp_path, monkeypatch):
    from runtime import feature_cache as FC

    shared = tmp_path / "shared"
    monkeypatch.setenv("MODELEVO_FEATURE_CACHE_DIR", str(shared))
    assert FC.cache_root(tmp_path / "run") == shared
    monkeypatch.setenv("MODELEVO_FEATURE_CACHE_DIR", "../shared")
    assert FC.cache_root(tmp_path / "run").resolve() == shared.resolve()
    # 显式关闭回退默认(不与共享目录混用)
    for off in ("off", "0", "none", ""):
        monkeypatch.setenv("MODELEVO_FEATURE_CACHE_DIR", off)
        assert FC.cache_root(tmp_path / "run") == \
            tmp_path / "run" / FC.CACHE_DIRNAME


def test_shared_cache_is_content_addressed_and_hits(tmp_path, monkeypatch):
    """同一份数据 + 同一内核 → 换一个 run 目录也命中;数据变了 → miss。"""
    from runtime import feature_cache as FC

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    shared = tmp_path / "shared"
    for r in (run_a, run_b):
        (r / "sample-profile" / "splits" / "reparse").mkdir(parents=True)
    src = pd.DataFrame({"seq": ["a@DB_0,b@DB_1", "c@DB_0"]})
    for r in (run_a, run_b):
        src.to_parquet(r / "sample-profile" / "splits" / "reparse"
                       / "train.parquet", index=False)
    monkeypatch.setenv("MODELEVO_FEATURE_CACHE_DIR", str(shared))

    def kernel(batch):
        # batch 的每个键是一列(batch_format="pylist" → 原始字符串列表)
        return np.asarray([[float(len(str(x)))] for x in batch["seq"]])

    kw = dict(name="v2173_test", fn=kernel, split="train",
              columns=["n"], sources=["seq"], plane="reparse",
              log=None)
    a = FC.feature_block(splits_dir=str(run_a / "sample-profile"
                                       / "splits"), **kw)
    _want = [float(len(x)) for x in src["seq"]]
    assert a["n"].tolist() == _want
    assert list(shared.glob("v2173_test-*")), "共享目录里应有缓存条目"

    # 另一个 run 目录、同一数据与同一内核 → 命中共享条目(不重算、不新写目录)
    _before = sorted(p.name for p in shared.glob("v2173_test-*"))
    b = FC.feature_block(splits_dir=str(run_b / "sample-profile"
                                       / "splits"), **kw)
    pd.testing.assert_frame_equal(a, b)
    assert sorted(p.name for p in shared.glob("v2173_test-*")) == _before
    assert not (run_b / FC.CACHE_DIRNAME).exists(), \
        "共享模式下不得再往 run 目录写缓存"

    # 数据变了 → 内容寻址失配 → 新条目(键不同,不会串用旧块)
    src2 = pd.DataFrame({"seq": ["a@DB_0", "c@DB_0"]})
    for r in (run_a, run_b):
        src2.to_parquet(r / "sample-profile" / "splits" / "reparse"
                        / "train.parquet", index=False)
    c = FC.feature_block(splits_dir=str(run_b / "sample-profile"
                                       / "splits"), **kw)
    assert c["n"].tolist() == [float(len(x)) for x in src2["seq"]]
    assert len(list(shared.glob("v2173_test-*"))) == len(_before) + 1


# ------------------------------------------------------- 2) shadow 开关(模板层)
def _body_ns() -> dict:
    """TRIAL_BODY 是模板字符串:取其中函数到独立命名空间再测。"""
    from _mvo import templates as T

    import numpy as _np
    import os as _os

    ns: dict = {"__name__": "trial_body_ns", "np": _np, "os": _os}
    exec(compile(T.TRIAL_BODY, "<trial-body>", "exec"), ns)
    return ns


def test_trial_body_shadow_respects_skip_env(monkeypatch):
    """TRIAL_BODY 的 shadow 段在 MODELEVO_SHADOW_SCORE=skip 时不产出工件。"""
    from _mvo import templates as T

    body = T.TRIAL_BODY
    i = body.find('_sh_path = SPLITS / "shadow_test.parquet"')
    assert i > 0
    seg = body[i:i + 900]          # 注释块 + 守卫行
    assert 'MODELEVO_SHADOW_SCORE' in seg
    assert '!= "skip"' in seg
    assert "_last_model is not None" in seg


def test_baseline_template_shadow_respects_skip_env():
    from runtime import baseline as B

    tpl = B.BASELINE_TEMPLATE
    assert "MODELEVO_SHADOW_SCORE" in tpl
    i = tpl.find('_sh_path = SPLITS / "shadow_test.parquet"')
    seg = tpl[i:i + 500]
    assert 'os.environ.get("MODELEVO_SHADOW_SCORE", "1") != "skip"' in seg


def test_upgrade_shadow_skip_patches_old_copies_idempotently():
    """--from-parent / warm-start 复用的旧副本被就地接入开关,且幂等。"""
    from _mvo.render import _upgrade_shadow_skip_code

    old_trial = (
        "try:\n"
        '    _sh_path = SPLITS / "shadow_test.parquet"\n'
        "    if _sh_path.exists() and _last_model is not None:\n"
        "        pass\n")
    out, notes = _upgrade_shadow_skip_code(old_trial)
    assert notes and "MODELEVO_SHADOW_SCORE" in out
    assert "_last_model is not None" in out
    again, again_notes = _upgrade_shadow_skip_code(out)
    assert again == out and again_notes == []            # 幂等

    old_base = (
        "try:\n"
        '    _sh_path = SPLITS / "shadow_test.parquet"\n'
        "    if _sh_path.exists():\n"
        "        pass\n")
    out2, notes2 = _upgrade_shadow_skip_code(old_base)
    assert notes2 and "MODELEVO_SHADOW_SCORE" in out2
    assert "_last_model" not in out2

    # 无关代码不动
    same, same_notes = _upgrade_shadow_skip_code("x = 1\n")
    assert same == "x = 1\n" and same_notes == []


def test_runner_shadow_switch_is_opt_in(tmp_path):
    """runner 只在配置显式列出动作时下发 skip(默认全部产出)。

    fresh confirmation 硬依赖父代 shadow 工件,所以不能默认跳过;
    confirm 即使被写进 skip 列表也会被强制剔除。
    """
    from runtime import runner as R

    src = Path(R.FILE if hasattr(R, "FILE") else R.__file__).read_text(
        encoding="utf-8")
    i = src.find("MODELEVO_SHADOW_SCORE")
    assert i > 0
    assert "shadow_skip_search_actions" in src[i:i + 2400]
    assert "MODELEVO_SHADOW_SCORE\"] = (" in src    # 唯一赋值点
    j = src.find('if "confirm" in _skip_actions:')
    assert j > 0 and "_skip_actions.discard" in src[j:j + 500]


def test_es_eval_cap_disabled_by_default(monkeypatch):
    ns = _body_ns()
    _es_eval_cap = ns["_es_eval_cap"]

    monkeypatch.delenv("MODELEVO_ES_EVAL_ROWS", raising=False)
    X = pd.DataFrame({"a": np.arange(5000.0)})
    y = pd.Series([0, 1] * 2500)
    Xs, ys = _es_eval_cap(X, y)
    assert len(Xs) == 5000 and len(ys) == 5000


def test_es_eval_cap_stratifies_and_caps(monkeypatch):
    ns = _body_ns()
    _es_eval_cap = ns["_es_eval_cap"]

    rng = np.random.default_rng(0)
    n = 40000
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = pd.Series(rng.integers(0, 2, n))
    monkeypatch.setenv("MODELEVO_ES_EVAL_ROWS", "10000")
    Xs, ys = _es_eval_cap(X, y)
    assert len(Xs) == len(ys) <= 10001
    # 分层:两类的抽样比例与原分布一致(±2 个百分点)
    got = float(np.mean(np.asarray(ys)))
    want = float(np.mean(np.asarray(y)))
    assert abs(got - want) < 0.02
    # v2.17.3 验证回写:改为"类内确定性随机抽"(不再是时间前缀),
    # 因此索引不要求连续,但必须严格递增且覆盖原有时间跨度
    assert list(Xs.index) == sorted(Xs.index)
    assert Xs.index.min() > 0 and Xs.index.max() < n


def test_es_eval_cap_noop_when_small(monkeypatch):
    ns = _body_ns()
    _es_eval_cap = ns["_es_eval_cap"]

    monkeypatch.setenv("MODELEVO_ES_EVAL_ROWS", "100000")
    X = pd.DataFrame({"a": np.arange(500.0)})
    y = pd.Series([0, 1] * 250)
    Xs, ys = _es_eval_cap(X, y)
    assert len(Xs) == 500


# ------------------------------------------------------- 4) 置换 repeat 默认
def test_group_importance_repeat_default_is_two():
    from runtime.config import DEFAULTS

    assert DEFAULTS["evaluator"]["group_importance_repeat"] == 2


def test_es_eval_rows_config_default():
    from runtime.config import DEFAULTS

    assert DEFAULTS["training"]["early_stop_eval_rows"] == 150000


# ------------------------------------------------------- 6) GPU 准入门槛
def test_gpu_admission_defaults_are_permissive():
    """默认不拒绝共享卡(旧行为);门槛键存在且可显式开启。"""
    from runtime.config import DEFAULTS

    ex = DEFAULTS["execution"]
    assert ex["require_clean_gpu"] is False
    assert ex["min_free_gpu_gb"] == 20.0


def test_runner_require_clean_gpu_degrades_to_cpu():
    """require_clean_gpu=true 且无干净卡时返回 (None, None) 走 CPU。"""
    from runtime import runner as R

    src = Path(R.FILE if hasattr(R, "FILE") else R.__file__).read_text(
        encoding="utf-8")
    i = src.find('self.config.get(\n            "execution.require_clean_gpu"')
    assert i > 0, "准入门槛读取点缺失"
    seg = src[i:i + 2000]
    assert "require_clean_gpu=true" in seg
    assert "return None, None" in seg
    assert "min_free_gpu_gb" in src


def test_shadow_skip_actions_default_empty():
    """默认空 = 全部产出 shadow(confirm 依赖父代 shadow 工件)。"""
    from runtime.config import DEFAULTS

    assert DEFAULTS["evaluator"]["shadow_skip_search_actions"] == []


def test_upgrade_es_eval_cap_patches_old_copy_idempotently(tmp_path):
    """--from-parent 副本的 _es_stage1 被接入降采样,且补 helper、幂等。"""
    from _mvo.render import _apply_es_source_upgrade

    old = (
        "import os\n\n"
        "def _es_stage1(model, fam, X_head, y_head, X_es, y_es, rounds):\n"
        '    """doc"""\n'
        "    return model.fit(X_head, y_head)\n\n"
        'if __name__ == "__main__":\n    pass\n')
    tp = tmp_path / "train.py"
    tp.write_text(old, encoding="utf-8")
    notes = _apply_es_source_upgrade(tp)
    txt = tp.read_text(encoding="utf-8")
    assert any("降采样" in n for n in notes), notes
    assert "def _es_eval_cap(" in txt
    assert '_cap_fn = globals().get("_es_eval_cap")' in txt
    compile(txt, str(tp), "exec")
    assert _apply_es_source_upgrade(tp) == []      # 幂等


def test_es_eval_cap_keeps_all_time_periods(monkeypatch):
    """按时间排序的帧上,分层随机抽必须覆盖所有日期(不再是时间前缀)。"""
    ns = _body_ns()
    _es_eval_cap = ns["_es_eval_cap"]

    n = 30000
    y = pd.Series([0, 1] * (n // 2))
    period = np.repeat([0, 1, 2], n // 3)          # 三个"日期"块
    X = pd.DataFrame({"a": np.arange(n), "period": period})
    monkeypatch.setenv("MODELEVO_ES_EVAL_ROWS", "6000")
    Xs, ys = _es_eval_cap(X, y)
    got = set(Xs["period"].unique().tolist())
    assert got == {0, 1, 2}, got                # 时间前缀实现只会给出 {0}
    # 每个日期按原比例保留
    share = Xs["period"].value_counts(normalize=True).sort_index()
    assert (share - pd.Series({0: 1 / 3, 1: 1 / 3, 2: 1 / 3})).abs().max() < 0.03


def test_min_free_gpu_gb_only_hard_when_require_clean():
    """默认 false 时显存门槛不生效(旧行为:取空闲最大的卡)。"""
    from runtime import runner as R

    src = Path(R.FILE if hasattr(R, "FILE") else R.__file__).read_text(
        encoding="utf-8")
    i = src.find("_require_free_gb = (float(self.config.get(")
    assert i > 0
    seg = src[i:i + 320]
    assert "if _require_clean else 0.0" in seg
