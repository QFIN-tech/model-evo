"""回归测试:v22.1 泄漏分类规则 3 的「实体标识误判」与撤销通道。

事故来源(2026-09-22,req-t7-propensity run):
    ``_leak_classify`` 规则 3 原实现是

        唯一率 > 0.95 且非数值  →  entity_identifier  →  banned-certain

    但 **逐人唯一的复合文本字段**唯一率同样≈1:本 run 的 ``seq_d365``
    (逗号分隔的行为事件序列,中位 342 个 token)与 ``seq_gap_d365``
    唯一率均为 0.9976,被机械判为实体标识并封禁。

    后果(实测):
      - 直接丢弃该数据面**最强**信息通道 —— 全局经验库高置信度卡
        exp-202609-8a4fe068 明确记录「seq reparse 加宽在 30 万行信贷
        行为时序表上带来 ~+0.0035 AUC(v1 全量面,独立确认 +0.0032),
        是该数据面最强信息通道」;
      - 同数据源、同 skill 版本的三个历史 run 均正确判为 reparse,
        只有本 run 误判 → 说明原判据不可靠,不是个案;
      - banned 一旦落盘就是物理隔离,**spec 里没有任何地方**能撤销它
        (与 v2.12.2.9 修复前的 kept_ambiguous 缺口同构)。

本文件锁定三层修复:
  1. 规则 3 增加**实体数对齐**否决 + **文本形态证据**判据(识别复合字段);
  2. ``features.leak_confirmation.reinstated_banned`` 撤销通道;
  3. 机械提案与全局经验库对账,冲突时降档为 ambiguous。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

from runtime.profile import (  # noqa: E402
    SampleProfile, _experience_supports_column)
from runtime.spec import TaskSpec  # noqa: E402


def _base_spec(**features) -> dict:
    d = {
        "task_id": "leak-entity-test",
        "task_name": "实体标识误判回归",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "y==1"},
        "primary_metric": "auc",
        "split_policy": {"type": "random"},
        "time": {"column": "pday"},
        "entity": {"id_column": "user_no"},
    }
    if features:
        d["features"] = features
    return d


#: 真实样本里的复合文本列(事故本体)
_SEQ_COL = "seq_d365"


def _composite_frame(n: int = 3000, *, seed: int = 0) -> pd.DataFrame:
    """构造「真 ID 列 + 复合文本列」的最小对照帧。

    ``deviceid_hash``  = 定长、无分隔、无词表复用 → 真实体标识形态
    ``seq_d365``       = 变长、逗号密集、token 高复用 → 复合文本字段形态
    """
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "user_no": [f"U{i:08d}" for i in range(n)],
        "label": rng.randint(0, 2, n),
        "pday": pd.date_range("2026-01-01", periods=n,
                              freq="min").astype(str),
        "deviceid_hash": [f"{rng.randint(0, 10**12):012d}" for _ in range(n)],
        _SEQ_COL: [
            ",".join(f"EV{rng.randint(0, 50)}@HR_noon"
                     for _ in range(rng.randint(5, 80)))
            for _ in range(n)
        ],
    })


def _classify(df: pd.DataFrame, spec_features: dict, tmp_path) -> dict:
    """用最小 stub 跑 ``_leak_classify``(不碰 profile 其余昂贵阶段)。"""
    obj = object.__new__(SampleProfile)
    obj._tmpdir = str(tmp_path)
    obj.out_dir = Path(tmp_path)
    obj.df = df
    obj.supervised_df = df
    obj.audit_cols = [c for c in df.columns
                      if c not in {"user_no", "label", "pday"}]
    obj.spec = TaskSpec(data=_base_spec(**spec_features))
    return SampleProfile._leak_classify(obj)


@pytest.fixture()
def classify(tmp_path):
    def _run(spec_features: dict | None = None, **kw) -> dict:
        return _classify(_composite_frame(**kw), spec_features or {}, tmp_path)
    return _run


def _find(out: dict, col: str) -> dict | None:
    for p in out["proposals"]:
        if p["column"] == col:
            return p
    return None


# --------------------------------------------------------------------------
# 1. 核心回归:复合文本字段不得被判 entity_identifier
# --------------------------------------------------------------------------

def test_composite_text_column_is_not_entity_identifier(classify):
    """**核心回归**:事故本体 —— 唯一率≈1 的复合文本列必须是 reparse。

    修复前:verdict=banned-certain / reason=entity_identifier(物理封禁)。
    """
    out = classify()
    p = _find(out, _SEQ_COL)
    assert p is not None, out["counts"]
    assert p["verdict"] != "banned-certain", p
    assert p["reason"] != "entity_identifier", p
    assert p["verdict"] == "reparse-candidate", p
    assert p["reason"] == "form_unusable", p


def test_composite_text_evidence_is_recorded(classify):
    """形态证据必须落表 —— 复核靠它,不靠"唯一率 0.997"这种不可复核的数字。"""
    p = _find(classify(), _SEQ_COL)
    assert p["len_cv"] is not None and p["len_cv"] > 0.25, p
    assert p["delim_density"] is not None and p["delim_density"] > 0.5, p
    assert p["token_reuse"] is not None and p["token_reuse"] > 50, p
    assert p["token_vocab"] and p["token_vocab"] < p["n_unique"], p


def test_true_identifier_still_banned(classify):
    """反向守卫:真正的实体标识(定长/无分隔/无词表)仍必须封禁。"""
    p = _find(classify(), "deviceid_hash")
    assert p is not None, "真 ID 列必须进提案"
    assert p["verdict"] == "banned-certain", p
    assert p["reason"] == "entity_identifier", p


def test_entity_alignment_is_recorded(classify):
    """实体数对齐证据必须可见(真 ID 的必要条件)。"""
    out = classify()
    assert out["confirmed"]["n_entities"] is not None
    for col in (_SEQ_COL, "deviceid_hash"):
        p = _find(out, col)
        assert p["n_unique"] is not None and p["n_entities"] is not None, p


def test_unique_over_entity_count_forces_reparse(tmp_path):
    """唯一值数 > 实体数 ⇒ 必然是复合字段(真 ID 不可能超过实体数)。

    构造:唯一率恰好压在阈值附近、形态三项都不显著,但唯一值数超过实体数
    (每个实体出现多次、字段逐行不同)——此时只能靠对齐判据兜住。
    """
    n = 3000
    rng = np.random.RandomState(1)
    # 每个 user_no 出现 3 次,但字段值逐行唯一且定长无分隔(形态三项全静默)
    df = pd.DataFrame({
        "user_no": [f"U{i // 3:06d}" for i in range(n)],
        "label": rng.randint(0, 2, n),
        "pday": pd.date_range("2026-01-01", periods=n,
                              freq="min").astype(str),
        "opaque_code": [f"X{i:07d}" for i in range(n)],
    })
    out = _classify(df, {}, tmp_path)
    p = _find(out, "opaque_code")
    assert p is not None, out["counts"]
    # 唯一值 3000 > 实体数 1000 → 形态三项虽静默,仍必须判为复合字段
    assert p["n_unique"] > p["n_entities"], p
    assert p["verdict"] == "reparse-candidate", p


# --------------------------------------------------------------------------
# 2. 撤销通道:reinstated_banned
# --------------------------------------------------------------------------

def test_reinstated_removes_from_proposal_face(classify):
    """撤销后该列回到"待裁决",重新出现在复核清单里。"""
    out = classify({"banned": [{"column": _SEQ_COL,
                                "reason": "entity_identifier",
                                "evidence": "机械误判"}],
                    "leak_confirmation": {"reinstated_banned": [_SEQ_COL]}})
    p = _find(out, _SEQ_COL)
    assert p is not None, "撤销后必须重新进入提案面待裁决"
    assert out["confirmed"]["reinstated_banned"] == [_SEQ_COL]


def test_reinstated_is_traced_in_markdown(classify, tmp_path):
    out = classify({"banned": [{"column": _SEQ_COL,
                                "reason": "entity_identifier",
                                "evidence": "机械误判"}],
                    "leak_confirmation": {"reinstated_banned": [_SEQ_COL]}})
    md = (tmp_path / "leak-classification.md").read_text(encoding="utf-8")
    assert "已撤销机械封禁" in md
    assert _SEQ_COL in md
    assert out["confirmed"]["reinstated_banned"] == [_SEQ_COL]


def test_reinstated_requires_column_already_banned():
    """撤销的是"这个封禁判错了" ⇒ 该列必须仍在 banned 里。"""
    errors = TaskSpec(data=_base_spec(
        leak_confirmation={"reinstated_banned": ["not_banned_col"]})
    ).validate()
    assert any("只能撤销" in e for e in errors), errors


def test_reinstated_conflicts_with_reparse():
    """撤销 ≠ 改判:同时声明两处会让数据面语义打架。"""
    errors = TaskSpec(data=_base_spec(
        banned=[{"column": "c1", "reason": "entity_identifier", "evidence": "x"}],
        reparse=[{"column": "c1", "reason": "form_unusable", "plan": "p"}],
        leak_confirmation={"reinstated_banned": ["c1"]},
    )).validate()
    assert any("不得交叠" in e or "重新裁决" in e for e in errors), errors


def test_reinstated_rejects_reserved_columns():
    errors = TaskSpec(data=_base_spec(
        banned=[{"column": "label", "reason": "entity_identifier",
                 "evidence": "x"}],
        leak_confirmation={"reinstated_banned": ["label"]},
    )).validate()
    assert any("目标/时间/ID/treatment" in e for e in errors), errors


def test_reinstated_rejects_duplicates():
    errors = TaskSpec(data=_base_spec(
        banned=[{"column": "c1", "reason": "entity_identifier",
                 "evidence": "x"}],
        leak_confirmation={"reinstated_banned": ["c1", "c1"]},
    )).validate()
    assert any("重复列" in e for e in errors), errors


def test_reinstated_changes_leak_manifest_hash():
    """撤销封禁 = 分类面变更,必须进协议指纹。"""
    banned = [{"column": "c1", "reason": "entity_identifier", "evidence": "x"}]
    a = TaskSpec(data=_base_spec(banned=banned))
    b = TaskSpec(data=_base_spec(
        banned=banned,
        leak_confirmation={"reinstated_banned": ["c1"]}))
    assert a.leak_manifest_hash != b.leak_manifest_hash


def test_reinstated_keeps_column_banned_until_readjudicated():
    """撤销**不改变数据面**:物理隔离要等用户写下真正的裁决才解除。

    这条是安全底线 —— "撤销"绝不能让一个未重新裁决的列悄悄进入特征面。
    """
    spec = TaskSpec(data=_base_spec(
        banned=[{"column": "c1", "reason": "entity_identifier", "evidence": "x"}],
        leak_confirmation={"reinstated_banned": ["c1"]}))
    assert "c1" in spec.banned_columns
    assert spec.reinstated_banned_columns == ["c1"]


# --------------------------------------------------------------------------
# 3. 机械提案复核清单(不再"可直接采纳")
# --------------------------------------------------------------------------

def test_markdown_lists_mechanical_bans_for_review(classify, tmp_path):
    classify()
    md = (tmp_path / "leak-classification.md").read_text(encoding="utf-8")
    assert "机械提案复核清单" in md
    assert "非终审" in md
    assert "reinstated_banned" in md
    # 旧的"banned-certain 可直接采纳"措辞必须消失
    assert "banned-certain 可直接采纳" not in md


# --------------------------------------------------------------------------
# 4. 经验库对账
# --------------------------------------------------------------------------

def test_experience_supports_column_matches_stem():
    """词根命中:seq_d365 的经验卡应被 seq_ 开头的列认领。"""
    card = {"statement": "seq reparse 加宽带来 ~+0.0035 AUC",
            "context": {}}
    assert _experience_supports_column(card, "seq_d365")
    assert _experience_supports_column(card, "seq_gap_d365")


def test_experience_supports_column_rejects_unrelated():
    """不相关列不得误命中 —— 否则任意卡片都能拦下任意封禁。"""
    card = {"statement": "seq reparse 加宽带来 ~+0.0035 AUC",
            "context": {}}
    assert not _experience_supports_column(card, "user_no")
    assert not _experience_supports_column(card, "loan_amt_t30")


def test_experience_cross_check_downgrades_ban(classify):
    """经验库命中 ⇒ 机械封禁降档为 ambiguous(强制人工裁决)。

    构造一个形态上确实是 ID(定长/无分隔/无词表)、但列名与既有经验卡
    同族的列 —— 此时形态判据无话可说,只有经验库能拦住它。
    """
    n = 3000
    rng = np.random.RandomState(2)
    df = pd.DataFrame({
        "user_no": [f"U{i:08d}" for i in range(n)],
        "label": rng.randint(0, 2, n),
        "pday": pd.date_range("2026-01-01", periods=n,
                              freq="min").astype(str),
        # 定长 12 位、无分隔、token 不复用 → 形态三项全静默 → 会被判 banned
        "seq_token_id": [f"{rng.randint(0, 10**12):012d}" for _ in range(n)],
    })
    out = _classify(df, {}, Path("/tmp"))
    p = _find(out, "seq_token_id")
    assert p is not None, out["counts"]
    # 若全局库不可用(无卡/后端异常),降级为原始封禁也可接受 —— 不误报失败
    if out["experience_cross_check"]:
        assert p["verdict"] == "ambiguous", p
        assert p["reason"] == "experience_conflict", p
        assert p["experience_id"], p
