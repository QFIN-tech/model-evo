# -*- coding: utf-8 -*-
"""特征知识库索引解析: 从 feature-knowledge.md 自动识别特征清单 csv。

索引文件: model-skills/model-knowledge/assets/feature-knowledge/feature-knowledge.md
其中「常用特征列表」markdown 表格每行登记一个业务域:
  | 分场景(sub domain) | 触发方式(trigger) | 特征表(feature table) | 可用特征清单(feature list) |

匹配优先级(resolve_feature_list_csv):
  1. feature_table 精确匹配「特征表」列(忽略大小写/首尾空格)
  2. business_domain 匹配「分场景」列
  3. 都未命中返回 None, 由调用方决定 warn 退全量还是报错

feature list 列填相对路径时按 feature-knowledge.md 所在目录解析;
指向的 csv 不存在的行(如「待补充」)打 warn 跳过。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

# repo 根 = feature-matching/scripts 的父父目录(model-skills)
_REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_KNOWLEDGE_MD = (
    _REPO_ROOT / "model-knowledge" / "assets"
    / "feature-knowledge" / "feature-knowledge.md"
)


def load_index(md_path: Optional[Path] = None) -> List[dict]:
    """解析 feature-knowledge.md 的索引表格。

    Args:
        md_path: 索引 md 路径, 默认 repo 内 feature-knowledge.md。

    Returns:
        [{sub_domain, trigger, feature_table, feature_list_csv(绝对 Path)}] 列表;
        md 不存在返回空列表(打 warn); csv 不存在的行跳过(打 warn)。
    """
    p = md_path or FEATURE_KNOWLEDGE_MD
    if not p.exists():
        print("[feature_knowledge] [WARN] 特征知识库索引不存在: %s" % p)
        return []

    entries: List[dict] = []
    base_dir = p.parent
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        # 跳过表头与分隔行
        if set(cells[0]) <= {"-", ":", " "} or "sub domain" in cells[0].lower() or "分场景" in cells[0]:
            continue
        sub_domain, trigger, feature_table, feature_list = cells[0], cells[1], cells[2], cells[3]
        if not feature_table or not feature_list:
            continue
        # 清单列可能带「(待补充)」等备注, 只取 .csv 路径部分
        m = re.search(r"[\w\-./]+\.csv", feature_list)
        if not m:
            print("[feature_knowledge] [WARN] 行「%s」的特征清单列未解析到 csv 路径, 跳过: %s" % (sub_domain, feature_list))
            continue
        csv_path = Path(m.group(0))
        if not csv_path.is_absolute():
            csv_path = (base_dir / csv_path).resolve()
        if not csv_path.exists():
            print("[feature_knowledge] [WARN] 行「%s」的特征清单 csv 不存在, 跳过: %s" % (sub_domain, csv_path))
            continue
        entries.append({
            "sub_domain": sub_domain,
            "trigger": trigger,
            "feature_table": feature_table,
            "feature_list_csv": csv_path,
        })
    return entries


def resolve_feature_list_csv(
    feature_table: Optional[str] = None,
    business_domain: Optional[str] = None,
    md_path: Optional[Path] = None,
) -> Optional[Path]:
    """按 feature_table 优先、business_domain 兜底, 从索引解析特征清单 csv 路径。

    Args:
        feature_table: 取数配置里的特征表(库.表), 与索引「特征表」列精确匹配(忽略大小写/首尾空格)。
        business_domain: 业务域(如「用户运营」), 与索引「分场景」列匹配。
        md_path: 索引 md 路径, 默认 repo 内 feature-knowledge.md。

    Returns:
        命中的 csv 绝对路径; 未命中返回 None。
    """
    entries = load_index(md_path)
    if not entries:
        return None

    if feature_table:
        ft = feature_table.strip().lower()
        for e in entries:
            if e["feature_table"].strip().lower() == ft:
                print("[feature_knowledge] feature_table=%s 命中「%s」, 特征清单: %s"
                      % (feature_table, e["sub_domain"], e["feature_list_csv"]))
                return e["feature_list_csv"]

    if business_domain:
        bd = business_domain.strip()
        for e in entries:
            if e["sub_domain"].strip() == bd:
                print("[feature_knowledge] business_domain=%s 命中「%s」, 特征清单: %s"
                      % (business_domain, e["sub_domain"], e["feature_list_csv"]))
                return e["feature_list_csv"]

    return None
