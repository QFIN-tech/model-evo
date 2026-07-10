# -*- coding: utf-8 -*-
"""report.md 自动回填器。

按 classification-model-development/SKILL.md §6 契约,从 session_dir 下各 sub-skill
产出的 manifest/JSON/CSV 中提取信息,幂等更新 report.md 的四~七段。

段落归属:
- 四(特征宽表) — 由 `classification-model-orchestration` Step 4B 触发回填
- 五(特征分析) — 由 `classification-model-development` Stage 0 触发回填
- 六(模型迭代) / 七(横向对比) — 由 `classification-model-development` 触发回填

幂等: 同一段落多次调用结果一致 — 用 `## 四、` 等 H2 锚点切分,替换段落内容,
保留首部 (一~三段) 与其他段落不变。

用法:
    python fill_report.py --session-dir <path> --section {IV|V|VI|VII|all}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SECTION_ANCHORS = {
    "IV": ("## 四、特征宽表", "## IV. 特征宽表"),
    "V": ("## 五、特征分析", "## V. 特征分析"),
    "VI": ("## 六、模型迭代", "## VI. 模型迭代"),
    "VII": ("## 七、横向对比", "## VII. 横向对比"),
}
# 元组 (中文锚点, 英文锚点): 中文为写出形式, 英文为读取形式(识别已有 report.md)。
# report.md 章节编号(汉字 一~七)与本脚本 section key 对应:
#   - 一/二/三 (需求 / 样本 / 历史模型推荐) 由 orchestration 自填
#   - 四~七 (特征宽表 / 特征分析 / 模型迭代 / 横向对比) 由本脚本接管
#   - 附录「待处理项与下一步建议」由 orchestration 自填(不带编号)
# 如改 report.md 章节顺序, 务必同步本表锚点。


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[fill_report] WARN: {path} 解析失败: {e}", file=sys.stderr)
        return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(x)


def _fmt_num(x: Any) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_auc(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "—"


def _build_split_manifest_from_parquets(splits_dir: Path) -> Optional[dict]:
    """从 sample-features/splits/{train,test,oot}.parquet 重建 split manifest 供 section IV 回填。

    读三档 parquet 的行数 + 正样本数 + dt 范围, 拼成与 task-spec
    同构的 dict (splits/ranges/actual_ratios/dropped_rows/strategy)。
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    split_names = ("train", "test", "oot")
    splits: Dict[str, dict] = {}
    total = 0
    for name in split_names:
        p = splits_dir / f"{name}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        # 标签列: 优先常见名, 兜底取唯一的 0/1 整数列
        label_col = None
        for c in ("TARGET", "label", "y"):
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            num_cols = [c for c in df.columns if df[c].dtype.kind in "iu"]
            for c in num_cols:
                if set(df[c].dropna().unique()).issubset({0, 1}):
                    label_col = c
                    break
        pos = int(df[label_col].sum()) if label_col else 0
        rows_n = int(len(df))
        # 时间列: 优先 dt / pday
        time_col = None
        for c in ("dt", "pday"):
            if c in df.columns:
                time_col = c
                break
        prange = None
        if time_col:
            prange = [str(int(df[time_col].min())), str(int(df[time_col].max()))]
        splits[name] = {
            "rows": rows_n,
            "pos": pos,
            "pos_rate": (pos / rows_n) if rows_n else 0.0,
            "pday_range": prange,
        }
        total += rows_n

    ranges = {
        name: {"start": splits[name]["pday_range"][0] if splits[name]["pday_range"] else None,
               "end": splits[name]["pday_range"][1] if splits[name]["pday_range"] else None}
        for name in split_names
    }
    actual_ratios = {
        name: (splits[name]["rows"] / total) if total else 0.0
        for name in split_names
    }
    return {
        "strategy": "time_explicit (local_file, rebuilt from splits parquet)",
        "splits": splits,
        "ranges": ranges,
        "actual_ratios": actual_ratios,
        "dropped_rows": 0,
    }


# ===== Section IV: 特征宽表 =====

def build_section_iv(session_dir: Path) -> str:
    """特征宽表段 — 来源: feature-list.csv + 三档切分(切分由 feature-analysis Stage 0 / task-spec 完成)。

    manifest 来源:
      ① sample-features/splits/ (feature-analysis Stage 0 产, 含三档 parquet) — 主
      ② data-profile/_split_manifest.json (task-spec run_sample_analysis 产) — fallback
    任一可用即回填。
    """
    fm_dir = session_dir / "sample-features" / "feature-matching"
    manifest = None
    source_tag = "—"

    splits_dir = session_dir / "sample-features" / "splits"
    if splits_dir.exists() and (splits_dir / "train.parquet").exists():
        manifest = _build_split_manifest_from_parquets(splits_dir)
        source_tag = "feature-analysis(splits)"
    else:
        dp_manifest = _read_json(
            session_dir / "data-profile" / "_split_manifest.json"
        )
        if dp_manifest:
            manifest = dp_manifest
            source_tag = "data-profile"

    if not manifest:
        return (
            f"{SECTION_ANCHORS['IV'][0]}\n\n"
            "（特征宽表/切分尚未执行: sample-features/splits/ 与 data-profile/_split_manifest.json 均缺失）\n"
        )

    splits = manifest.get("splits", {})
    ranges = manifest.get("ranges", {})
    time_col = manifest.get("time_col", "pday")
    label_col = manifest.get("label_col", "label")

    feature_list_csv = fm_dir / "feature-list.csv"
    n_features_listed = 0
    if feature_list_csv.exists():
        with feature_list_csv.open(encoding="utf-8") as f:
            n_features_listed = sum(1 for _ in f) - 1

    rows = [
        f"{SECTION_ANCHORS['IV'][0]}\n",
        f"- 特征清单文件: `feature-list.csv` (共 {n_features_listed} 列)",
        f"- 时间列: `{time_col}` / 标签列: `{label_col}`",
        f"- 切分策略: `{manifest.get('strategy', '—')}`  (数据来源: `{source_tag}`)",
        "",
        "| 集合 | 样本量 | 正样本 | 正样本率 | {col} 范围 |".format(col=time_col),
        "|------|--------|--------|----------|-----------|",
    ]
    for split_name in ("train", "test", "oot"):
        s = splits.get(split_name, {})
        r = ranges.get(split_name, {})
        # 兼容两种区间字段名: feature-matching/data-profile 用 'start'/'end',
        # data-profile split 内也可能用 'pday_range' (列表)
        start = r.get("start") if isinstance(r, dict) else None
        end = r.get("end") if isinstance(r, dict) else None
        if not start or not end:
            pr = s.get("pday_range") if isinstance(s, dict) else None
            if isinstance(pr, list) and len(pr) == 2:
                start, end = pr[0], pr[1]
        rows.append(
            f"| {split_name} | {_fmt_num(s.get('rows'))} | "
            f"{_fmt_num(s.get('pos', s.get('positive')))} | "
            f"{_fmt_pct(s.get('pos_rate', s.get('positive_rate')))} | "
            f"{start or '—'} ~ {end or '—'} |"
        )

    actual_ratios = manifest.get("actual_ratios", {})
    if actual_ratios:
        ratio_str = " / ".join(
            f"{k}={_fmt_pct(v)}" for k, v in actual_ratios.items()
        )
        rows.append("")
        rows.append(f"> 实际切分比例: {ratio_str}")
        rows.append(f"> dropped_rows: {manifest.get('dropped_rows', 0)}")

    rows.append("")
    return "\n".join(rows)


# ===== Section V: 特征分析 =====

def build_section_v(session_dir: Path) -> str:
    """特征分析段 — 来源: feature-analysis/analysis/ 下 report.md + csv。"""
    fa_dir = session_dir / "sample-features" / "feature-analysis" / "analysis"
    manifest = _read_json(fa_dir / "_manifest.json")

    if not manifest:
        return (
            f"{SECTION_ANCHORS['V'][0]}\n\n"
            "（feature-analysis 尚未执行, analysis/_manifest.json 缺失）\n"
        )

    overview = manifest.get("overview", {})
    iv_rows = _read_csv(fa_dir / "iv_table.csv")
    psi_rows = _read_csv(fa_dir / "psi_table.csv")
    stats_rows = _read_csv(fa_dir / "stats.csv")

    iv_top10 = sorted(
        [r for r in iv_rows if r.get("iv") and r["iv"] != "nan"],
        key=lambda r: float(r["iv"]),
        reverse=True,
    )[:10]

    psi_warn = [r for r in psi_rows if r.get("warn", "").lower() == "true"]

    try:
        high_missing = [
            r for r in stats_rows
            if r.get("missing_rate") and float(r["missing_rate"]) > 0.95
        ]
    except (TypeError, ValueError):
        high_missing = []

    lines = [
        f"{SECTION_ANCHORS['V'][0]}\n",
        f"- 总样本: {_fmt_num(overview.get('n_total'))} "
        f"(train={_fmt_num(overview.get('n_train'))} / "
        f"oot={_fmt_num(overview.get('n_oot'))})",
        f"- 分析特征数: {overview.get('n_features', '—')}",
        f"- PSI 预警数 (> {overview.get('psi_warn_threshold', 0.1)}): "
        f"{overview.get('n_psi_warn', '—')}",
        "",
        "### IV Top 10",
        "",
        "| # | feature | IV | 单变量 AUC | 有效分箱 |",
        "|---|---------|----|-----------|---------|",
    ]
    for i, r in enumerate(iv_top10, 1):
        lines.append(
            f"| {i} | {r.get('feature', '—')} | "
            f"{float(r.get('iv', 0)):.4f} | "
            f"{_fmt_auc(r.get('auc'))} | "
            f"{r.get('n_bins_effective', '—')} |"
        )

    lines.extend(["", "### PSI 预警 (train → oot 不稳定)", ""])
    if psi_warn:
        lines.extend([
            "| feature | PSI |",
            "|---------|-----|",
        ])
        for r in sorted(psi_warn, key=lambda x: float(x.get("psi", 0)), reverse=True):
            lines.append(f"| {r.get('feature', '—')} | {float(r.get('psi', 0)):.4f} |")
    else:
        lines.append("（无 PSI 预警特征）")

    lines.extend(["", "### 高缺失率 (> 0.95)", ""])
    if high_missing:
        lines.extend(["| feature | 缺失率 | dtype |", "|---------|--------|-------|"])
        for r in high_missing:
            lines.append(
                f"| {r.get('feature', '—')} | "
                f"{_fmt_pct(r.get('missing_rate'))} | "
                f"{r.get('dtype', '—')} |"
            )
    else:
        lines.append("（无缺失率 > 0.95 的特征）")

    lines.append("")
    lines.append(f"> 详细报告见 `sample-features/feature-analysis/analysis/report.md`")
    lines.append("")
    return "\n".join(lines)


# ===== Section VI: 模型迭代 =====

def _classify_run(config: dict) -> str:
    """从 config.json 判断 run 类型,返回一句话描述。"""
    produced_by = config.get("produced_by", "")
    runtime = config.get("runtime", {})
    algo = config.get("algo", "?")

    if "model-tuning" in produced_by:
        if "selection" in runtime:
            sel = runtime.get("selection", {})
            kept = len(sel.get("kept_features", []))
            dropped = len(sel.get("dropped", []))
            return f"feat 筛选 (kept={kept}, dropped={dropped})"
        if "diagnosis" in runtime:
            diag = runtime.get("diagnosis", {})
            method = runtime.get("method", "rule")
            return f"tuned ({diag.get('status', '?')}, method={method})"
        return "tuning"

    return "baseline"


def build_section_vi(session_dir: Path) -> str:
    """模型迭代段 — 来源: new-models/*/config.json。"""
    new_models_dir = session_dir / "new-models"
    if not new_models_dir.exists():
        return (
            f"{SECTION_ANCHORS['VI'][0]}\n\n"
            "（尚未训练任何模型, new-models/ 不存在）\n"
        )

    run_dirs = sorted(
        [d for d in new_models_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not run_dirs:
        return (
            f"{SECTION_ANCHORS['VI'][0]}\n\n"
            "（new-models/ 为空, Stage 2 baseline 待跑）\n"
        )

    lines = [
        f"{SECTION_ANCHORS['VI'][0]}\n",
        f"共 {len(run_dirs)} 个 run (按目录名排序):",
        "",
        "| # | run_name | algo | n_feat | train AUC | test AUC | oot AUC | 关键变更 |",
        "|---|----------|------|--------|-----------|----------|---------|----------|",
    ]

    for i, run_dir in enumerate(run_dirs, 1):
        cfg = _read_json(run_dir / "config.json")
        if not cfg:
            lines.append(f"| {i} | {run_dir.name} | ? | ? | ? | ? | ? | config.json 缺失 |")
            continue

        runtime = cfg.get("runtime", {})
        metrics = runtime.get("metrics", {})
        train_auc = _fmt_auc(metrics.get("train", {}).get("auc"))
        val_auc = _fmt_auc(metrics.get("val", {}).get("auc"))
        oot_auc = _fmt_auc(metrics.get("oot", {}).get("auc"))
        n_feat = runtime.get("n_features", "—")
        algo = cfg.get("algo", "?")
        change = _classify_run(cfg)

        lines.append(
            f"| {i} | {run_dir.name} | {algo} | {n_feat} | "
            f"{train_auc} | {val_auc} | {oot_auc} | {change} |"
        )

    lines.append("")
    return "\n".join(lines)


# ===== Section VII: 横向对比 =====

def build_section_vii(session_dir: Path) -> str:
    """横向对比段 — 来源: model-comparison/model-comparison_*.json。"""
    mc_dir = session_dir / "model-comparison"
    oot_json = _read_json(mc_dir / "model-comparison_oot.json")

    if not oot_json:
        return (
            f"{SECTION_ANCHORS['VII'][0]}\n\n"
            "（session-level 横向对比尚未生成, model-comparison_oot.json 缺失）\n"
        )

    auc_cmp = oot_json.get("auc_comparison", {}).get("全量", {})
    ks_cmp = oot_json.get("ks_comparison", {}).get("全量", {})

    entries: List[Tuple[str, float, float]] = []
    for model_key, auc_val in auc_cmp.items():
        try:
            auc_f = float(auc_val)
        except (TypeError, ValueError):
            continue
        ks_f = float(ks_cmp.get(model_key, 0)) if ks_cmp.get(model_key) else 0.0
        entries.append((model_key, auc_f, ks_f))

    entries.sort(key=lambda x: x[1], reverse=True)

    lines = [
        f"{SECTION_ANCHORS['VII'][0]}\n",
        f"对比 {len(entries)} 个模型 (按 oot AUC 降序):",
        "",
        "| 排名 | model | oot AUC | oot KS |",
        "|------|-------|---------|--------|",
    ]
    for rank, (model_key, auc, ks) in enumerate(entries, 1):
        clean_name = model_key.replace(" oot", "").strip()
        lines.append(f"| {rank} | {clean_name} | {auc:.4f} | {ks:.4f} |")

    if entries:
        top1 = entries[0]
        lines.append("")
        lines.append(
            f"> Top1: `{top1[0].replace(' oot', '').strip()}` "
            f"AUC={top1[1]:.4f} KS={top1[2]:.4f}"
        )

    lines.extend([
        "",
        f"> 详细对比见 `model-comparison/model-comparison_oot.md` "
        f"(另有 train/test 两档)",
        "",
    ])
    return "\n".join(lines)


# ===== 报告替换逻辑 =====

SECTION_BUILDERS = {
    "IV": build_section_iv,
    "V": build_section_v,
    "VI": build_section_vi,
    "VII": build_section_vii,
}


def _split_report(content: str) -> List[Tuple[str, str]]:
    """把 report.md 切成 [(anchor_or_header, body), ...] 列表。

    每个顶层 `## ` H2 标题起一段。保留 H1 与首部 metadata。
    """
    parts: List[Tuple[str, str]] = []
    current_anchor = ""
    current_lines: List[str] = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_anchor or current_lines:
                parts.append((current_anchor, "\n".join(current_lines)))
            current_anchor = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_anchor or current_lines:
        parts.append((current_anchor, "\n".join(current_lines)))

    return parts


def _is_anchor_match(anchor: str, targets: Tuple[str, str]) -> bool:
    """检查 anchor 是否匹配 targets 中的任意一个。

    targets = ("## 四、特征宽表", "## IV. 特征宽表") 等。
    匹配规则: 比较"序号 token"(四、 / IV.)与"标题关键词"(特征宽表),都一致才算匹配。
    """
    if not anchor:
        return False

    def _parse(line: str) -> Tuple[str, str]:
        """返回 (序号token, 标题剩余)。例: '## 四、特征宽表' → ('四、', '特征宽表')。"""
        if not line.startswith("## "):
            return ("", line)
        rest = line[3:].strip()
        # 序号 token = 第一个空白前或符号后
        parts = rest.split(maxsplit=1)
        if not parts:
            return ("", "")
        num = parts[0]
        title = parts[1] if len(parts) > 1 else ""
        return (num, title)

    a_num, a_title = _parse(anchor)
    for tgt in targets:
        t_num, t_title = _parse(tgt)
        if a_num == t_num and a_title == t_title:
            return True
    return False


def _replace_section(content: str, section_key: str, new_body: str) -> str:
    """替换 report.md 中指定 H2 段落的内容,保留其他段落。

    段落由 H2 锚点开头,到下一个 `## ` 或文件末尾结束。
    锚点为 `## 四、...` 形式(中文序号)。若不存在则追加到末尾。
    """
    primary, fallback = SECTION_ANCHORS[section_key]
    parts = _split_report(content)

    target_anchors = (primary, fallback)

    replaced_idx = None
    for i, (anchor, _body) in enumerate(parts):
        if _is_anchor_match(anchor, target_anchors):
            replaced_idx = i
            break

    body_only = new_body
    for tgt in target_anchors:
        if body_only.startswith(tgt):
            body_only = body_only[len(tgt):].lstrip("\n")
            break

    if replaced_idx is not None:
        parts[replaced_idx] = (primary, body_only)
    else:
        parts.append((primary, body_only))

    out_lines: List[str] = []
    for i, (anchor, body) in enumerate(parts):
        chunk_lines: List[str] = []
        if anchor:
            chunk_lines.append(anchor)
        body_str = body.strip()
        if body_str:
            chunk_lines.append(body_str)
        if not chunk_lines:
            continue
        if out_lines:
            out_lines.append("")
        out_lines.extend(chunk_lines)

    return "\n".join(out_lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="report.md 自动回填器")
    parser.add_argument("--session-dir", required=True, type=Path,
                        dest="session_dir",
                        help="session 目录 (runs/<timestamp>-<model_name>)")
    parser.add_argument(
        "--section",
        required=True,
        choices=list(SECTION_ANCHORS.keys()) + ["all"],
    )
    args = parser.parse_args()

    session_dir = args.session_dir.resolve()
    if not session_dir.is_dir():
        sys.exit(f"[fill_report] session_dir 不存在: {session_dir}")

    report_path = session_dir / "report.md"
    if not report_path.exists():
        sys.exit(f"[fill_report] report.md 不存在: {report_path}")

    content = report_path.read_text(encoding="utf-8")

    sections = (
        list(SECTION_ANCHORS.keys()) if args.section == "all" else [args.section]
    )

    builders = {
        "IV": lambda: build_section_iv(session_dir),
        "V": lambda: build_section_v(session_dir),
        "VI": lambda: build_section_vi(session_dir),
        "VII": lambda: build_section_vii(session_dir),
    }

    for key in sections:
        new_body = builders[key]()
        content = _replace_section(content, key, new_body)
        print(f"[fill_report] 已回填段落 {key}")

    report_path.write_text(content, encoding="utf-8")
    print(f"[fill_report] report.md 已更新: {report_path}")


if __name__ == "__main__":
    main()
