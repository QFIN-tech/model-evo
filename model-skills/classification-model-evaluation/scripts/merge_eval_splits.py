#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 train/test/oot 三档 eval JSON 为一份 all（全量）eval JSON。

用法:
  python merge_eval_splits.py \
    --jsons model_train_eval.json model_test_eval.json model_oot_eval.json \
    -o model_all_eval.json

合并规则:
  - metric_by_segment: count 求和, label_rate/auc/ks/accuracy/precision/recall/f1 按 count 加权平均
  - score_buckets: 相同 decile 合并, 重新计算 label_rate/lift/recall/cum_recall
  - biz_avg: 按 count 加权平均
  - data_splits: 汇总各 split 信息

注意: AUC/KS 的加权平均是近似值；精确值需合并原始 CSV 后重新评估。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# Helpers
# ============================================================
def _safe_get(d: dict, key: str, default: Any = None) -> Any:
    v = d.get(key, default)
    return v if v is not None else default


def weighted_avg(values: List[Optional[float]], weights: List[int]) -> Optional[float]:
    """加权平均，忽略 None。全部为 None 则返回 None。"""
    total_w = 0
    total_v = 0.0
    for v, w in zip(values, weights):
        if v is not None and w > 0:
            total_v += v * w
            total_w += w
    return round(total_v / total_w, 6) if total_w > 0 else None


def merge_biz_avg(biz_dicts: List[Dict[str, float]], counts: List[int]) -> Dict[str, float]:
    """合并多个分片的 biz_avg，按 count 加权平均。"""
    keys = set()
    for bd in biz_dicts:
        if bd:
            keys.update(bd.keys())
    result = {}
    for k in keys:
        vals = [bd.get(k) for bd in biz_dicts]
        result[k] = weighted_avg(vals, counts)
    return result


# ============================================================
# Core merge
# ============================================================
def merge_eval_jsons(json_paths: List[Path]) -> Dict[str, Any]:
    """合并多个 split 的 eval JSON 为一份全量 JSON。"""
    if len(json_paths) < 1:
        raise ValueError("至少需要 1 个 eval JSON")

    models = []
    for jp in json_paths:
        with open(jp, 'r', encoding='utf-8') as f:
            models.append(json.load(f))

    # 以第一个 JSON 为模板
    base = copy.deepcopy(models[0])
    base_model_meta = base.get("model_meta", {})
    base_name = base_model_meta.get("name", "模型")
    base_version = "all"

    # ---- data_profile ----
    # data_splits 汇总各 split 信息
    # 保留 observation_window / feature_count 等字段（取首个非空值）
    combined_observation = ""
    combined_feature_count = None
    total_count = 0
    total_label_rate = 0.0
    split_entries = []
    for m in models:
        dp = m.get("data_profile", {})
        obs = dp.get("observation_window", "")
        if obs and not combined_observation:
            combined_observation = obs
        fc = dp.get("feature_count")
        if fc is not None and combined_feature_count is None:
            combined_feature_count = fc
        for s in dp.get("data_splits", []):
            split_entries.append(s)
            total_count += s.get("sample_count", 0)
            total_label_rate += s.get("label_rate", 0) * s.get("sample_count", 0)
    if total_count > 0:
        total_label_rate = round(total_label_rate / total_count, 6)

    merged_data_profile = {
        "observation_window": combined_observation,
        "feature_count": combined_feature_count,
        "data_splits": split_entries,
    }

    # ---- metric_by_segment ----
    # 收集所有出现过 segment 名
    all_segments: List[str] = []
    seen = set()
    for m in models:
        for seg in m.get("metric_by_segment", {}):
            if seg not in seen:
                seen.add(seg)
                all_segments.append(seg)
    # 把 '全量' 放最后
    if '全量' in all_segments:
        all_segments.remove('全量')
        all_segments = sorted(all_segments) + ['全量']
    else:
        all_segments = sorted(all_segments)

    merged_metrics: Dict[str, Dict] = {}
    for seg in all_segments:
        seg_counts = [_safe_get(m.get("metric_by_segment", {}).get(seg, {}), "count", 0) for m in models]
        seg_combined = sum(seg_counts)
        seg_metrics_list = [m.get("metric_by_segment", {}).get(seg, {}) for m in models]

        merged_metrics[seg] = {
            "count": seg_combined,
            "label_rate": weighted_avg([sm.get("label_rate") for sm in seg_metrics_list], seg_counts),
            "auc": weighted_avg([sm.get("auc") for sm in seg_metrics_list], seg_counts),
            "ks": weighted_avg([sm.get("ks") for sm in seg_metrics_list], seg_counts),
            "accuracy": weighted_avg([sm.get("accuracy") for sm in seg_metrics_list], seg_counts),
            "precision": weighted_avg([sm.get("precision") for sm in seg_metrics_list], seg_counts),
            "recall": weighted_avg([sm.get("recall") for sm in seg_metrics_list], seg_counts),
            "f1": weighted_avg([sm.get("f1") for sm in seg_metrics_list], seg_counts),
            "biz_avg": merge_biz_avg(
                [sm.get("biz_avg", {}) for sm in seg_metrics_list],
                seg_counts,
            ),
        }

    # ---- performance.score_buckets ----
    merged_buckets: Dict[str, List[Dict]] = {}
    for seg in all_segments:
        all_buckets_for_seg = [
            m.get("performance", {}).get("score_buckets", {}).get(seg, [])
            for m in models
        ]
        # 如果某 split 没有这个 segment 的 buckets，用空列表
        merged_buckets[seg] = _merge_buckets_segment(all_buckets_for_seg, seg, merged_metrics)

    merged_performance = {
        "score_buckets": merged_buckets,
        "feature_importance": base.get("performance", {}).get("feature_importance", []),
    }

    # ---- hyperparameters / lineage ----
    # 取自第一个模型
    hyperparams = base.get("hyperparameters", {})
    lineage = {
        "replaces": base_model_meta.get("replaces"),
        "replaced_by": base_model_meta.get("replaced_by"),
        "status_note": base_model_meta.get("status_note", ""),
    }

    # ---- 组装 ----
    # model_meta 更新为 all
    merged_meta = copy.deepcopy(base_model_meta)
    merged_meta["version"] = "all"
    merged_meta["eval_date"] = datetime.now().strftime("%Y-%m-%d")

    # tags: 用第一个模型的 tags，追加 "total"
    base_tags = list(base.get("tags", []))
    if "all" not in base_tags:
        base_tags.append("all")

    return {
        "kb_version": base.get("kb_version", "2.0.0"),
        "source_file": ", ".join(str(p.name) for p in json_paths),
        "generated_at": datetime.now().isoformat(),
        "model_meta": merged_meta,
        "tags": base_tags,
        "data_profile": merged_data_profile,
        "metric_by_segment": merged_metrics,
        "performance": merged_performance,
        "hyperparameters": hyperparams,
        "lineage": lineage,
        "_note": "AUC/KS 为按样本量加权近似值；精确值需合并原始 CSV 后重新评估。",
    }


def _merge_buckets_segment(
    buckets_lists: List[List[Dict]],
    seg: str,
    merged_metrics: Dict[str, Dict],
) -> List[Dict]:
    """合并同一 segment 的多个分桶列表。"""
    # 收集所有 decile 的数据
    decile_data: Dict[int, Dict[str, float]] = {}
    biz_keys: set = set()

    for buckets in buckets_lists:
        for b in buckets:
            d = b.get("decile")
            if d is None:
                continue
            if d not in decile_data:
                decile_data[d] = {
                    "count": 0,
                    "pos_count": 0.0,
                    "score_min_sum": 0.0,
                    "score_max_sum": 0.0,
                }
            cnt = b.get("count", 0)
            lr = b.get("label_rate", 0) or 0
            decile_data[d]["count"] += cnt
            decile_data[d]["pos_count"] += cnt * lr
            decile_data[d]["score_min_sum"] += b.get("score_min", 0) * cnt
            decile_data[d]["score_max_sum"] += b.get("score_max", 0) * cnt
            # 收集 biz keys
            for k in b:
                if k not in ("decile", "count", "score_min", "score_max",
                              "label_rate", "lift", "recall", "cum_recall"):
                    biz_keys.add(k)

    total_count = sum(dd["count"] for dd in decile_data.values())
    total_pos = sum(dd["pos_count"] for dd in decile_data.values())
    overall_lr = merged_metrics.get(seg, {}).get("label_rate")
    if overall_lr is None or overall_lr == 0:
        overall_lr = total_pos / total_count if total_count > 0 else 0

    result = []
    cum_pos = 0.0
    for d in range(10, 0, -1):
        dd = decile_data.get(d)
        if dd is None or dd["count"] == 0:
            # 保持 10 行结构，写空
            result.append({
                "decile": d,
                "count": 0,
                "score_min": None,
                "score_max": None,
                "label_rate": None,
                "lift": None,
                "recall": None,
                "cum_recall": None,
            })
            continue

        cnt = dd["count"]
        bucket_lr = dd["pos_count"] / cnt if cnt > 0 else 0
        cum_pos += dd["pos_count"]

        row: Dict[str, Any] = {
            "decile": d,
            "count": cnt,
            "score_min": round(dd["score_min_sum"] / cnt, 4) if cnt > 0 else None,
            "score_max": round(dd["score_max_sum"] / cnt, 4) if cnt > 0 else None,
            "label_rate": round(bucket_lr, 6),
            "lift": round(bucket_lr / overall_lr, 4) if overall_lr > 0 else None,
            "recall": round(dd["pos_count"] / total_pos, 4) if total_pos > 0 else None,
            "cum_recall": round(cum_pos / total_pos, 4) if total_pos > 0 else None,
        }

        # 业务指标：从各 split 加权合并
        biz_vals: Dict[str, List[float]] = {k: [] for k in biz_keys}
        biz_weights: Dict[str, List[int]] = {k: [] for k in biz_keys}
        for buckets in buckets_lists:
            for b in buckets:
                if b.get("decile") == d:
                    for bk in biz_keys:
                        v = b.get(bk)
                        if v is not None:
                            biz_vals[bk].append(v)
                            biz_weights[bk].append(b.get("count", 0))
        for bk in biz_keys:
            if biz_vals[bk]:
                row[bk] = weighted_avg(biz_vals[bk], biz_weights[bk])
            else:
                row[bk] = None

        result.append(row)

    return result


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="合并 train/test/oot 评估 JSON 为全量 JSON+MD+XLSX")
    parser.add_argument(
        "--jsons", nargs="+", required=True,
        help="train/test/oot eval JSON 路径（顺序无关，多个）",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出 JSON 路径（同时产出同名的 .md 和 .xlsx）",
    )
    args = parser.parse_args()

    json_paths = [Path(p) for p in args.jsons]
    for p in json_paths:
        if not p.exists():
            print(f"ERROR: 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)

    merged = merge_eval_jsons(json_paths)

    out_json = Path(args.output)
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] 合并 {len(json_paths)} 份 eval → {out_json}")
    print(f"  全量样本量: {merged['metric_by_segment'].get('全量', {}).get('count', 0):,}")

    # 同时产出 MD 和 XLSX
    model_name = merged["model_meta"]["name"]
    version = merged["model_meta"]["version"]
    metrics = merged["metric_by_segment"]
    segments = [k for k in metrics.keys() if k != "全量"]
    score_buckets = merged.get("performance", {}).get("score_buckets", {})
    # 展平 buckets（build_md/build_xlsx 期望 metrics[seg]["buckets"]）
    metrics_with_buckets = {}
    for seg, m in metrics.items():
        m_with_b = dict(m)
        m_with_b["buckets"] = score_buckets.get(seg, [])
        metrics_with_buckets[seg] = m_with_b
    biz_cols = list(metrics.get("全量", {}).get("biz_avg", {}).keys())

    # 导入 eval_single 的导出函数
    import importlib.util
    _eval_single_path = Path(__file__).resolve().parent / "eval_single.py"
    spec = importlib.util.spec_from_file_location("_eval_single", str(_eval_single_path))
    _eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_eval)

    md_path = out_dir / f"{out_json.stem}.md"
    md_path.write_text(
        _eval.build_md(model_name, version, metrics_with_buckets, segments),
        encoding="utf-8",
    )
    print(f"[OK] {md_path}")

    xlsx_path = out_dir / f"{out_json.stem}.xlsx"
    _eval.build_xlsx(model_name, version, metrics_with_buckets, segments, biz_cols, str(xlsx_path))
    print(f"[OK] {xlsx_path}")


if __name__ == "__main__":
    main()
