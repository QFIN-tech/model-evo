"""v2.15.9 追加修复:经验导出简写适配、全局榜单/经验表方向、分段单调方向。

事故来源:req-t7 run4 regression(2026-10-21,目标 rmse)收尾:

1. export-experience:收尾 subagent 按 references 模板写的四字段简写
   {claim,context,evidence,boundary}(evidence 为字符串、无 id)全部被
   validate 拒收,导入 0 条;
2. top-trial 全局榜单按 Δ_baseline 降序(AUC 方向)→ rmse 任务把基线排
   rank1、冠军挤出 top10;experiences.yaml 有效方向 0 条、current_best
   取到最差值;
3. segment_evaluation.passed 把"段均值递增"与指标方向绑定 → 回归任务
   monotonicity=1.0 仍判 False。
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

from runtime.metrics import segment_passed  # noqa: E402

ATID = "reg-y-200w_1000w-abcd1234"


# ---------------------------------------------------------------------------
# 1. 四字段简写入库
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2. 全局榜单 / experiences.yaml 方向
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 3. 分段单调方向
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("means,mono,expect", [
    ([0.05, 0.2, 0.5, 1.1, 4.7], 1.0, True),      # 回归:递增即通过
    ([0.05, 0.2, 0.5, 1.1, 4.7], 0.5, False),     # 单调性不足
    ([4.7, 1.1, 0.5, 0.2, 0.05], 1.0, False),     # 反向排序
])
def test_segment_passed_direction_free(means, mono, expect):
    assert segment_passed({"bin_means": means, "monotonicity": mono},
                          0.8) is expect


def test_segment_passed_missing():
    assert segment_passed({"bin_means": None, "monotonicity": None}) is None


def test_evaluators_use_segment_passed():
    ev = (SKILL / "runtime" / "evaluator.py").read_text(encoding="utf-8")
    dl = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    assert "segment_passed(" in ev and "segment_passed(" in dl
    assert "not increasing and not self.direction" not in ev
    assert "not metric_direction(spec.primary_metric))))" not in dl
