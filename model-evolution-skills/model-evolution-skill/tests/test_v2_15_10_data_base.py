"""v2.15.10:数据底座指纹去派生列 + 同数据底座检索层。

事故来源(req-t7 run4 regression,2026-10-21):loader 派生标签列
loan_amt_t7_log1p,数据指纹从原生 1513 列的 bc5329dc 变成 e58b4b22;再加上
检索按 model_type 硬过滤、atid 分段从任务类型开始比,同一份数据上二分类
run 的 24 张特征工程卡对回归 run 完全不可见(planner-context "无命中经验")。
"""
from __future__ import annotations

import json
import pathlib
import sys
import types

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401

from runtime.config import Config  # noqa: E402
from runtime.profile import raw_columns_fingerprint  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402

RAW = [f"c{i}" for i in range(20)] + ["req_cnt_t7", "loan_amt_t7", "pday"]
BIN = "binary-req_cnt_t7-200w_1000w-{h}"
REG = "reg-loan_amt_t7_log1p-200w_1000w-{h}"


# ---------------------------------------------------------------------------
# 1. 数据底座指纹
# ---------------------------------------------------------------------------
def _spec(derived=None):
    data = {"task_id": "t", "model_type": "regression",
            "target": {"name": "loan_amt_t7_log1p"},
            "primary_metric": "rmse",
            "data": {"data_loader": "data-loader.py"}}
    if derived is not None:
        data["data"]["derived_columns"] = derived
    return TaskSpec(data=data)


def test_fingerprint_excludes_declared_derived_columns():
    raw = raw_columns_fingerprint(RAW, _spec())
    with_target = raw_columns_fingerprint(RAW + ["loan_amt_t7_log1p"], _spec())
    declared = raw_columns_fingerprint(RAW + ["loan_amt_t7_log1p"],
                                       _spec(["loan_amt_t7_log1p"]))
    assert with_target["raw_columns_hash"] != raw["raw_columns_hash"], \
        "未声明时与旧口径一致(派生列计入)"
    assert declared["raw_columns_hash"] == raw["raw_columns_hash"]
    assert declared["raw_columns_count"] == len(RAW)
    assert declared["derived_columns_excluded"] == ["loan_amt_t7_log1p"]
    assert "derived_columns_excluded" not in raw


def test_fingerprint_backward_compatible_with_legacy_formula():
    import hashlib
    legacy = hashlib.sha256("|".join(sorted(RAW)).encode()).hexdigest()[:8]
    assert raw_columns_fingerprint(RAW, _spec())["raw_columns_hash"] == legacy


def test_spec_validates_derived_columns_type():
    bad = _spec("loan_amt_t7_log1p")
    assert any("derived_columns" in e for e in bad.validate())
    ok = _spec(["loan_amt_t7_log1p"])
    assert not any("derived_columns" in e for e in ok.validate())


# 第 2 节(同数据底座检索层)随旧 memory/ 全局库删除;v2.18.0 的底座相似度与跨底座
# 打分见 tests/test_v2_18_retrieval.py。
