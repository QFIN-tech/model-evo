"""回归测试:v2.12.2.9 泄漏确认门的「确认保留」通道。

事故来源(2026-09-18,req-cnt-t7 run):
    profile 的 phase-1 确认门由
    ``needs_user_confirmation = any(ambiguous) or (not banned and not reparse)``
    判定:只要存在 ambiguous 提案就恒为 true,**与用户是否已裁决无关**。
    而 spec 里原本只有 ``features.banned`` / ``features.reparse`` 两个
    能写回裁决的地方 —— "复核后确认这列合法、保留为普通特征"这个**真实
    且常见**的裁决**无处安放**。

    后果(实测):
      - 把这 10 个 ambiguous 列加入 banned(错误裁决)→ 门翻 false;
      - 保持现状(正确裁决"保留")→ 门恒 true;
      - 操作者只能反复重跑 profile(本 run 白跑一次完整 profile,~5min),
        或委屈地把合法特征误标为泄漏列。

    本文件锁定修复:新增 ``features.leak_confirmation.kept_ambiguous``
    作为第三条裁决通道,它**只关闭确认门,不改变任何数据面**。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from runtime.spec import TaskSpec  # noqa: E402

#: 两个形态上"像泄漏"、语义上属历史回看的列(取自 req-cnt-t7 真实样本)
_AMBIG = [
    "offline_inner_maidian_multidim_loan_loan_enter_days_2m",
    "offline_inner_jtbeh_draw_beforedraw12mapplytimes",
]


def _base_spec(**features) -> dict:
    d = {
        "task_id": "leak-confirm-test",
        "task_name": "泄漏确认门回归",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "y==1"},
        "primary_metric": "auc",
        "split_policy": {"type": "random"},
        "time": {"column": "pday_dt"},
        "entity": {"id_column": "user_no"},
    }
    if features:
        d["features"] = features
    return d


# --------------------------------------------------------------------------
# 1. spec 层:kept_ambiguous 的解析、校验与指纹
# --------------------------------------------------------------------------

def test_kept_ambiguous_is_parsed():
    spec = TaskSpec(data=_base_spec(
        leak_confirmation={"kept_ambiguous": list(_AMBIG), "note": "历史回看"}))
    assert spec.kept_ambiguous_columns == _AMBIG
    assert spec.leak_confirmation_note == "历史回看"


def test_kept_ambiguous_defaults_empty():
    spec = TaskSpec(data=_base_spec())
    assert spec.kept_ambiguous_columns == []
    assert spec.leak_confirmation_note == ""


def test_kept_ambiguous_changes_leak_manifest_hash():
    """裁决面必须进协议指纹:改它 = 改"哪些列被视为已裁决"。"""
    a = TaskSpec(data=_base_spec())
    b = TaskSpec(data=_base_spec(
        leak_confirmation={"kept_ambiguous": list(_AMBIG)}))
    assert a.leak_manifest_hash != b.leak_manifest_hash


def test_kept_ambiguous_cannot_overlap_banned_or_reparse():
    """保留 vs 排除是互斥语义 —— 同一列不能两种裁决都写。"""
    for bad, key in ((list(_AMBIG), "banned"), (list(_AMBIG), "reparse")):
        feats = {
            "banned": [{"column": c, "reason": "future_window", "evidence": "x"}
                       for c in bad] if key == "banned" else [],
            "reparse": [{"column": c, "reason": "form_unusable", "plan": "p"}
                        for c in bad] if key == "reparse" else [],
            "leak_confirmation": {"kept_ambiguous": list(bad)},
        }
        errors = TaskSpec(data=_base_spec(**feats)).validate()
        assert any("不得交叠" in e for e in errors), (key, errors)


def test_kept_ambiguous_rejects_reserved_columns():
    """目标/时间/ID/treatment 不属于泄漏分类表(与 banned 同口径)。"""
    errors = TaskSpec(data=_base_spec(
        leak_confirmation={"kept_ambiguous": ["label"]})).validate()
    assert any("目标/时间/ID/treatment" in e for e in errors), errors


def test_leak_confirmation_must_be_mapping():
    errors = TaskSpec(data=_base_spec(
        leak_confirmation=["not", "a", "dict"])).validate()
    assert any("必须是映射" in e for e in errors), errors


def test_kept_ambiguous_rejects_duplicates():
    errors = TaskSpec(data=_base_spec(
        leak_confirmation={"kept_ambiguous": _AMBIG + _AMBIG[:1]})
    ).validate()
    assert any("重复列" in e for e in errors), errors


# --------------------------------------------------------------------------
# 2. profile 层:门判定 —— 这是本次修复的核心行为
# --------------------------------------------------------------------------

def _leak_classify_with(stub, spec_features: dict) -> dict:
    """用一个最小 stub 跑 SampleProfile._leak_classify 的门判定分支。

    只构造判定所需的最小状态(audit_cols / supervised_df / spec),
    不碰 profile 的其余昂贵阶段。
    """
    import numpy as np
    import pandas as pd

    from runtime.profile import SampleProfile

    rng = np.random.RandomState(0)
    n = 4000
    df = pd.DataFrame({
        # 与 label 中等相关 → 命中 ambiguous 规则 5(AUC ∈ (0.70, 0.85))
        _AMBIG[0]: rng.rand(n).astype("float32"),
        # 无关列 → 不进提案
        "noise_col": rng.rand(n).astype("float32"),
    })
    # noise=1.5 时实测 AUC≈0.798:落在 ambiguous 带内,且远离
    # banned-certain 的 AUC>0.85 哨兵(否则会被规则 2 抢先判成泄漏)
    df["label"] = ((df[_AMBIG[0]] + 1.5 * rng.rand(n)) > 0.78).astype(int)

    stub.df = df
    stub.supervised_df = df
    stub.audit_cols = ["noise_col", _AMBIG[0]]
    stub.spec = TaskSpec(data=_base_spec(**spec_features))
    stub.out_dir = Path(stub._tmpdir)
    return SampleProfile._leak_classify(stub)


@pytest.fixture()
def gate_probe(tmp_path):
    def _run(spec_features: dict) -> dict:
        obj = object.__new__(__import__(
            "runtime.profile", fromlist=["SampleProfile"]).SampleProfile)
        obj._tmpdir = str(tmp_path)
        return _leak_classify_with(obj, spec_features)
    return _run


def test_gate_open_when_ambiguous_pending(gate_probe):
    """未裁决的 ambiguous → 门开(这是我们希望用户看到的信号)。"""
    out = gate_probe({})
    assert out["counts"]["ambiguous"] >= 1, out["counts"]
    assert out["needs_user_confirmation"] is True


def test_gate_closed_by_kept_ambiguous(gate_probe):
    """**核心回归**:写 kept_ambiguous 后门关闭,且该列不进提案表。"""
    out = gate_probe({"leak_confirmation": {"kept_ambiguous": [_AMBIG[0]]}})
    assert out["needs_user_confirmation"] is False, out["counts"]
    assert out["counts"]["ambiguous"] == 0, out["counts"]
    assert not any(p["column"] == _AMBIG[0] for p in out["proposals"])
    assert out["confirmed"]["kept_ambiguous"] == [_AMBIG[0]]


def test_gate_closed_by_banning(gate_probe):
    """另一条合法通道(判泄漏)同样必须能关闭门 —— 防止修复回归。"""
    out = gate_probe({"banned": [
        {"column": _AMBIG[0], "reason": "future_window", "evidence": "x"}]})
    assert out["needs_user_confirmation"] is False, out["counts"]


def test_gate_report_renders_kept_section(gate_probe, tmp_path):
    """leak-classification.md 必须留痕"已确认保留",否则门关了却查不到原因。"""
    out = gate_probe({"leak_confirmation": {
        "kept_ambiguous": [_AMBIG[0]], "note": "历史回看口径,保留"}})
    md = (tmp_path / "leak-classification.md").read_text(encoding="utf-8")
    assert "已确认保留" in md
    assert _AMBIG[0] in md
    assert "历史回看口径,保留" in md
    assert "needs_user_confirmation = False" in md
