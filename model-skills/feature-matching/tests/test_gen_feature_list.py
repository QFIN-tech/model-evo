# -*- coding: utf-8 -*-
"""gen_feature_list 加载与落盘单测 + feature_knowledge 索引解析单测。

测试中不硬编码公司特征表/业务域/清单文件名, 均通过 load_index() 动态发现
model-knowledge/assets/feature-knowledge/feature-knowledge.md 索引中已登记的
第一条条目, 以该条目作 fixture 验证解析机制。索引内容由 model-knowledge 维护。
"""
import csv
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from gen_feature_list import (
    load_feature_list,
    write_feature_list_csv,
)
from feature_knowledge import load_index, resolve_feature_list_csv


def _first_index_entry():
    """取 feature-knowledge.md 索引中第一条可用条目作 fixture, 避免硬编码公司信息。"""
    entries = load_index()
    assert entries, "feature-knowledge.md 索引为空, 无法做解析测试"
    return entries[0]


def test_load_default_resolves_from_feature_knowledge():
    """source=None 时按 feature_table 从 feature-knowledge.md 索引识别清单, 且非空。"""
    entry = _first_index_entry()
    features = load_feature_list(None, feature_table=entry["feature_table"])
    assert isinstance(features, list)
    assert len(features) > 0  # 索引登记的清单非空即可, 不绑定具体行数


def test_load_default_unresolved_raises():
    """source=None 且索引未命中时应抛 ValueError, 不静默全量。"""
    with pytest.raises(ValueError, match="feature-knowledge"):
        load_feature_list(None, feature_table="db.not_registered_table")


def test_resolve_by_feature_table():
    """feature_table 精确匹配「特征表」列(忽略大小写), 解析到索引中登记的清单。"""
    entry = _first_index_entry()
    # 大小写不敏感: 用全大写输入验证
    p = resolve_feature_list_csv(feature_table=entry["feature_table"].upper())
    assert p is not None
    assert p == entry["feature_list_csv"]
    assert p.exists()


def test_resolve_by_business_domain_fallback():
    """feature_table 未命中时按 business_domain 匹配「分场景」列, 解析到索引中登记的清单。"""
    entry = _first_index_entry()
    p = resolve_feature_list_csv(
        feature_table="db.unknown", business_domain=entry["sub_domain"]
    )
    assert p is not None
    assert p == entry["feature_list_csv"]


def test_resolve_miss_returns_none():
    """两个匹配键都未命中时返回 None。"""
    assert resolve_feature_list_csv(feature_table="db.unknown", business_domain="不存在的域") is None


def test_load_index_skips_missing_csv():
    """索引里 csv 不存在的行(如「待补充」)被跳过, 不进 entries; 所有 entry 的 csv 均存在。"""
    entries = load_index()
    assert entries, "索引应至少含一条 csv 存在的条目"
    assert all(e["feature_list_csv"].exists() for e in entries)


def test_load_txt_file(tmp_path):
    """普通 .txt 按行读, 跳过空行与 # 注释, 保序去重。"""
    src = tmp_path / "feas.txt"
    src.write_text(
        "# comment\n"
        "fea_a\n"
        "\n"
        "fea_b\n"
        "fea_a\n"   # 重复
        "fea_c\n",
        encoding="utf-8",
    )
    features = load_feature_list(str(src))
    assert features == ["fea_a", "fea_b", "fea_c"]


def test_load_csv_file(tmp_path):
    """.csv 走 csv 解析,取 feature_name 列,忽略其他列。"""
    src = tmp_path / "feas.csv"
    src.write_text(
        "feature_name,note\nfea_a,ok\nfea_b,\nfea_a,dup\nfea_c,ok\n",
        encoding="utf-8",
    )
    features = load_feature_list(str(src))
    assert features == ["fea_a", "fea_b", "fea_c"]


def test_load_csv_missing_column_raises(tmp_path):
    """csv 缺 feature_name 列应抛 ValueError。"""
    src = tmp_path / "bad.csv"
    src.write_text("name,note\nfea_a,ok\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature_name"):
        load_feature_list(str(src))


def test_load_missing_file_raises(tmp_path):
    """来源文件不存在应抛 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_feature_list(str(tmp_path / "does_not_exist.txt"))


def test_load_empty_file_raises(tmp_path):
    """空清单应抛 ValueError(避免下游 SQL 拿到空 features)。"""
    src = tmp_path / "empty.txt"
    src.write_text("# only comment\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        load_feature_list(str(src))


def test_write_feature_list_csv(tmp_path):
    """落盘 csv: 单列 feature_name + 表头,顺序保留。"""
    out = tmp_path / "out" / "feature-list.csv"
    write_feature_list_csv(["x", "y", "z"], str(out))
    assert out.exists()
    with out.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["feature_name"]
    assert [r[0] for r in rows[1:]] == ["x", "y", "z"]
