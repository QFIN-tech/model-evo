"""回归测试:v2.13.2 跨视图 baseline/parent 裁决事故。

事故来源:`example-seq-v3-v2.12.2.4`(2026-09-18)gen-01。

    该代共 15 个 trial,其中 10 个声明 ``data_view: sample`` 的 trial
    **全部**以 verdict=``invalid`` 报废,reason 为
    "预测身份/源码完整性检查失败，禁止进入选择与交付";Agent 只能在
    v1 面重跑 5 个(其中 3 个 accepted)。落盘证据:

        trials/trial-004-g01-lightgbm-1152/evaluation.json
          integrity.errors == [
            "baseline_prediction_incomparable:paired comparison:"
            "新旧样本集合不同(missing=0,new=30110,old=43563)"]
        trials/trial-004-g01-lightgbm-1152/validation_score.parquet.meta.json
          rows=30110, data_view="sample"
        trials/trial-001-g00-lightgbm-1053/validation_score.parquet.meta.json
          rows=43563, data_view="v1"

根因(两层,缺一不可):

1. 调用方把 baseline 的**视图**丢掉了 ——
   ``gen_executor.py`` / ``cmd_trial.py`` / ``cmd_startup.py`` 都写
   ``baseline_score = bl.get("validation")``,而 ``load_parent_scores``
   返回的 dict 里本来就有 ``data_view``。
2. ``runtime/runner.py`` 的 baseline 分支无条件调
   ``align_score_by_keys``,键集合 30110 vs 43563 必然不等 → 抛错 →
   runner 吞掉(注释写"evaluator 会裁为 invalid")→
   ``evaluator.py:244-247`` 置 ``integrity.valid=False`` →
   ``evaluator.py:437-439`` 判 invalid。

**关键区分**(本文件的核心断言):两条路径语义不同,不能合并 ——

    * 视图不同   = 结构性不可比(预期内)→ baseline_comparable=False,
                   走 evaluator 既有的 baseline_incomparable 分支,正常裁决;
    * 对齐失败   = 身份/完整性事故      → 保持 fail-closed(invalid)。

本文件同时锁定"修复没有把 fail-closed 削弱"。
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from runtime.runner import (SAMPLE_KEY_COLUMN,  # noqa: E402
                            baseline_view_compatible)

_RUNNER = SKILL_ROOT / "runtime" / "runner.py"


# ---------------------------------------------------------------------------
# 1. 纯函数真值表
# ---------------------------------------------------------------------------

def test_baseline_view_compatible_truth_table():
    assert baseline_view_compatible("sample", "v1") is False
    assert baseline_view_compatible("v1", "sample") is False
    assert baseline_view_compatible("seqc", "v1") is False
    assert baseline_view_compatible("v1", "v1") is True
    assert baseline_view_compatible("sample", "sample") is True
    # 未知视图按 v1 处理(旧 trial 无 data_view 字段,既有口径是 or "v1")
    assert baseline_view_compatible(None, "v1") is True
    assert baseline_view_compatible("", "v1") is True
    assert baseline_view_compatible("v1", None) is True
    assert baseline_view_compatible(None, None) is True
    assert baseline_view_compatible("sample", None) is False


# ---------------------------------------------------------------------------
# 2. 事故复现:同视图 + 键集合不等 → 必须仍然 invalid(fail-closed)
# ---------------------------------------------------------------------------

def _make_run(tmp_path: Path, *, n_base: int, n_cur: int):
    """造一个最小 run:同时材料化 v1 面与 sample 面(跨视图测试的必需前置)。

    sample 面必须真实存在 —— 事故形态就是"trial 声明 data_view=sample,
    而 baseline 分数来自 v1",只造 v1 面测不出这条路径。
    """
    from runtime.config import Config
    from runtime.data import write_sample_splits, write_splits
    from runtime.spec import TaskSpec

    spec = TaskSpec(data={
        "task_id": "v213xv", "task_name": "v213xv",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
    })
    n = max(n_base, n_cur)
    df = pd.DataFrame({
        "dt": ["2026-02-01"] * n,
        "uid": list(range(n)),
        "x": np.arange(n, dtype=float),
        "label": [0, 1] * (n // 2),
    })
    run = tmp_path / "run"
    write_splits(df, spec, run / "sample-profile" / "splits")
    write_sample_splits(df, spec, run, max_rows=max(1000, n))
    return run, spec, Config.load(skill_config=False)


def _score_df_from_labels(labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    """按标签面的样本键造一份合法预测(键集合与长度都与该视图一致)。

    键必须取自真实标签面 —— 事故的判据正是"键集合不等",用伪造键
    测不出真实形态。
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "score": rng.random(len(labels)),
        SAMPLE_KEY_COLUMN: labels[SAMPLE_KEY_COLUMN].astype(str).values,
    })


def _labels(run_dir: Path, view: str) -> pd.DataFrame:
    from runtime.data import view_splits_dir
    return pd.read_parquet(
        view_splits_dir(Path(run_dir), view) / "labels" / "validation.parquet")


def test_same_view_key_mismatch_still_invalid(tmp_path: Path):
    """**fail-closed 不得被削弱**:同视图但键集合不等,仍是 invalid。

    这是修复的边界。跨视图走"不可比"是预期内的口径缺失;
    同视图键集合不等是身份/完整性事故,必须继续裁 invalid
    (evaluator.py:437-439),否则伪造/错配的预测可以静默进入选择。

    构造:同声明 v1 视图,但喂一份键被整体替换的 baseline 预测
    (行数相同、键集合不同)—— 这正是 align 必须拒绝的形态。
    """
    run, spec, config = _make_run(tmp_path, n_base=100, n_cur=100)
    runner = _TrialRunner(run, spec, config)
    lab = _labels(run, "v1")
    trial = _fake_trial(run, data_view="v1", n_rows=len(lab))

    forged = _score_df_from_labels(lab, 1)
    forged[SAMPLE_KEY_COLUMN] = [f"forged-{i}" for i in range(len(lab))]

    ev = runner._evaluate_trial(
        trial,
        baseline_score=forged,   # 同视图声明,但键集合不等
        baseline_view="v1",
        include_shadow=False,
    )
    assert ev["integrity"]["valid"] is False
    assert any("baseline_prediction_incomparable" in e
               for e in ev["integrity"]["errors"]), ev["integrity"]["errors"]
    assert ev["verdict"] == "invalid"


def test_cross_view_baseline_not_invalid(tmp_path: Path):
    """**修复锁定**:跨视图 baseline → 不再 invalid,记 baseline_incomparable。

    这正是 example-seq-v3-v2.12.2.4 里 10 个 sample 面 trial 的形态:
    trial 在 sample 面(baseline_view="v1"),baseline 分数是 v1 面的。
    修复前:align 抛"新旧样本集合不同" → invalid。
    修复后:视图不同即不喂给 align,走 baseline_incomparable 分支。
    """
    run, spec, config = _make_run(tmp_path, n_base=100, n_cur=100)
    runner = _TrialRunner(run, spec, config)
    v1_lab = _labels(run, "v1")
    sample_lab = _labels(run, "sample")
    trial = _fake_trial(run, data_view="sample", n_rows=len(sample_lab))

    ev = runner._evaluate_trial(
        trial,
        baseline_score=_score_df_from_labels(v1_lab, 1),  # v1 面
        baseline_view="v1",
        include_shadow=False,
    )
    assert ev["integrity"]["valid"] is True, ev["integrity"]["errors"]
    assert ev["delta"].get("vs_baseline") is None
    assert ev["delta"].get("baseline_incomparable") is True
    assert ev["verdict"] != "invalid"


def test_cross_view_parent_skips_vs_parent_without_invalid(tmp_path: Path):
    """跨视图父代:跳过 vs_parent,但不裁 invalid(混合血缘已实测出现)。

    `example-seq-v3-v2.12.2.4` 的血缘是
    v1 baseline → sample trial-003 → sample trial-004;
    trial-003 那一跳就是跨视图父代。旧行为下这一跳被判 invalid。

    另注意 trial 必须**有父代声明**(parent_trial_ids 非空 → is_root=False):
    有父代声明却拿不到可比父代分数,正确语义是"无提升主张"
    (evaluator 的 `delta is None` → rejected),而不是降级成 root trial
    后按"建立起跑线"判 accepted —— 后者会让跨视图 trial 静默变成
    冠军候选,是比 invalid 更危险的失真。
    """
    run, spec, config = _make_run(tmp_path, n_base=100, n_cur=100)
    runner = _TrialRunner(run, spec, config)
    v1_lab = _labels(run, "v1")
    sample_lab = _labels(run, "sample")
    trial = _fake_trial(run, data_view="sample", n_rows=len(sample_lab),
                        parents=["trial-001-g00-lightgbm-0000"])

    ev = runner._evaluate_trial(
        trial,
        parent_scores={"validation": _score_df_from_labels(v1_lab, 2),
                       "data_view": "v1"},   # v1 父代 vs sample 子代
        baseline_score=_score_df_from_labels(sample_lab, 3),
        baseline_view="sample",              # baseline 同视图
        include_shadow=False,
    )
    assert ev["integrity"]["valid"] is True, ev["integrity"]["errors"]
    # 无 vs_parent 即无提升主张 → rejected,但**不是** invalid
    assert "vs_parent" not in ev["delta"]
    assert ev["verdict"] == "rejected"
    # 同视图的 baseline 仍然正常计算
    assert ev["delta"].get("vs_baseline") is not None


def test_same_view_parent_still_paired(tmp_path: Path):
    """同视图父代行为不变(修复不得误伤正常路径)。"""
    run, spec, config = _make_run(tmp_path, n_base=100, n_cur=100)
    runner = _TrialRunner(run, spec, config)
    lab = _labels(run, "v1")
    trial = _fake_trial(run, data_view="v1", n_rows=len(lab))

    ev = runner._evaluate_trial(
        trial,
        parent_scores={"validation": _score_df_from_labels(lab, 2),
                       "data_view": "v1"},
        include_shadow=False,
    )
    assert ev["integrity"]["valid"] is True
    assert "vs_parent" in ev["delta"]


# ---------------------------------------------------------------------------
# 3. AST 守卫:视图门必须真正位于 align 之前,且调用方必须传视图
# ---------------------------------------------------------------------------

def _runner_fn(name: str) -> ast.FunctionDef:
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"runner.py 里找不到 {name}")


def test_runner_gates_view_before_align():
    """`_evaluate_trial` 里 baseline 与 parent 两侧都必须在 align 之前判视图。

    静态守卫的理由:这条门失效时**没有运行时错误** —— 只会静默回到
    "跨视图 trial 全变 invalid",而后者在测试里只表现为一个 verdict
    字符串,容易被后续重构悄悄改回去。
    """
    fn = _runner_fn("_evaluate_trial")
    src = ast.get_source_segment(_RUNNER.read_text(encoding="utf-8"), fn)
    assert src is not None
    assert "baseline_view_compatible(" in src, (
        "_evaluate_trial 未做视图门 —— 跨视图事故会复发")
    # 两处调用:baseline 一次、parent 一次
    assert src.count("baseline_view_compatible(") >= 2, (
        "视图门必须覆盖 baseline 与 parent 两条分支(只修一侧会把事故"
        "从一侧挪到另一侧)")
    # 门的两个分支语义必须可辨
    assert "baseline_comparable = False" in src
    assert "parent_alignment_ok = False" in src


def test_pairing_keys_gates_parent_keys():
    """`_pairing_keys` 必须按 parent_comparable 门控父代键。

    否则跨视图父代下旧键仍会传给 evaluator,evaluator 二次 align 抛错
    又裁 invalid —— 修复等于没做(这是实现期实测到的第三个缺陷)。
    """
    fn = _runner_fn("_pairing_keys")
    src = ast.get_source_segment(_RUNNER.read_text(encoding="utf-8"), fn)
    assert "parent_comparable" in src
    assert "parent_score is not None and parent_comparable" in src, (
        "_pairing_keys 的父代键未加门:跨视图父代会在 evaluator 侧二次抛错")


@pytest.mark.parametrize("rel", [
    "runtime/gen_executor.py",
    "scripts/_mvo/cmd_trial.py",
    "scripts/_mvo/cmd_startup.py",
])
def test_callers_pass_baseline_view(rel: str):
    """三个 runner.run 调用方都必须传 baseline_view。

    漏传不会报错 —— runner 的默认 None 视为 v1,于是 sample 面 trial
    又会被喂 v1 baseline 分数并裁 invalid。这正是本次事故的成因
    (视图在调用方被丢弃),必须逐处静态锁定。
    """
    src = (SKILL_ROOT / rel).read_text(encoding="utf-8")
    assert "baseline_view" in src, (
        f"{rel} 未传 baseline_view —— 跨视图事故会复发")
    # v2.15.7 的 cmd_startup 写成 `(_bl or {}).get("data_view")`,同义
    assert re.search(r'\(?_?bl(?: or \{\})?\)?\.get\("data_view"\)', src), (
        f"{rel} 未从 load_parent_scores 的返回值取 data_view")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _TrialRunner(run_dir, spec, config):
    from runtime.runner import TrialRunner
    return TrialRunner(run_dir, spec, config)


def _fake_trial(run_dir: Path, *, data_view: str, n_rows: int,
                parents: list[str] | None = None):
    """造一个已落盘、源码已冻结的最小 trial。

    `_evaluate_trial` 会走 `verify_trial_source`(要求 code/ 与 config/
    有内容且已冻结)与 `bind_prediction_artifact`(要求预测行数 ==
    标签面行数),因此这里必须落真实的最小事前产物 —— 手写目录而不走
    TrialFactory,是为了不把 factory 的预算护栏/占位检查拖进本测试的
    关注面(本测试只关心跨视图那条判据)。
    """
    from runtime.integrity import freeze_trial_source
    from runtime.trial import TrialRecord

    tid = f"trial-999-g01-lightgbm-{data_view}"
    tdir = Path(run_dir) / "trials" / tid
    (tdir / "code").mkdir(parents=True, exist_ok=True)
    (tdir / "config").mkdir(parents=True, exist_ok=True)
    (tdir / "code" / "train.py").write_text("print('fake')\n",
                                            encoding="utf-8")
    (tdir / "config" / "trial.yaml").write_text(
        "seed: 1\n", encoding="utf-8")
    rng = np.random.default_rng(7)
    pd.DataFrame({"score": rng.random(n_rows)}).to_parquet(
        tdir / "validation_score.parquet", index=False)
    (tdir / "metrics.json").write_text("{}", encoding="utf-8")
    freeze_trial_source(tdir)
    return TrialRecord(data={
        "trial_id": tid,
        "algorithm": "lightgbm",
        "generation": 1,
        "search_action": "explore",
        "changed_modules": ["feature"],
        "data_view": data_view,
        "parent_trial_ids": parents or [],
        "status": "SUCCEEDED",
    }, trial_dir=tdir)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
