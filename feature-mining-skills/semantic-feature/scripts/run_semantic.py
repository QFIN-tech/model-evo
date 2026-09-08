# -*- coding: utf-8 -*-
"""run_semantic.py: semantic-feature 主入口 Text -> Embedding -> 监督 Head -> 注册。

用法:
    python run_semantic.py --session-dir <session_dir> [--config <semantic_config.yaml>]

流程:
  1. 读 session(须已完成 prepare_session): 三档切分 + 样本契约(text_cols)
  2. 对每个文本源(单列或多列拼接, mode=full/list/chunk):
     三档文本 -> 唯一文本去重编码(磁盘缓存, 换后端自动失效) -> 行向量
  3. 每个 head 变体(lr_score/gbdt_head 走 OOF; pca/quantile/cluster 无监督):
     只用 train 段拟合, 三档推理; 空文本行监督分输出 NaN + present 指示列
  4. 每输出列算方向修正 AUC + PR-AUC(三档); 配了 base_model_score_col 时附条件增益诊断
  5. registry.json + features.parquet(id_cols 去重唯一) 落盘,
     回写 state.semantic_columns -- 语义列自此成为 feature-evolution 候选生成的合法原料

防穿越: 监督 head 只见 train 段 label, train 段分数为 OOF; test/oot 仅推理。

退出码: 0 成功 / 2 配置或契约失败 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

import encoders
import heads
import registry as sem_registry
from evo_core import metrics, paths, state_io

PRODUCED_BY = "feature-mining-skills/semantic-feature"
DEFAULT_HEADS = ["lr_score", "pca_lowdim"]


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def load_semantic_config(session_dir, config_path: str | None) -> dict:
    """语义配置优先级: --config 文件 > session_config 的 semantic 段 > 缺省。"""
    if config_path:
        from config_io import load_config

        cfg = load_config(config_path) or {}
        return cfg.get("semantic") or cfg
    state = state_io.load_state(session_dir)
    return (state.get("config_snapshot") or {}).get("semantic") or {}


def resolve_sources(sem_cfg: dict, state: dict) -> list:
    """文本源列表: 配置显式给出, 或按样本契约 text_cols 每列一源。

    source 字段: name/text_cols/max_chars/mode(full|list|chunk)/agg(list 聚合)/combine。
    combine=separate 且多列时, 每列各自展开为独立源。
    """
    sources = sem_cfg.get("sources")
    out = []
    if sources:
        for s in sources:
            if not s.get("text_cols"):
                raise ValueError("semantic.sources 每项必须给 text_cols")
            cols = [str(c) for c in s["text_cols"]]
            combine = str(s.get("combine") or "concat")
            if combine == "separate" and len(cols) > 1:
                for c in cols:
                    out.append({
                        "name": sem_registry.sanitize_name(s.get("name") or c),
                        "text_cols": [c],
                        "max_chars": int(s.get("max_chars") or 2000),
                        "mode": str(s.get("mode") or "full"),
                        "agg": str(s.get("agg") or "tfidf"),
                    })
                continue
            out.append({
                "name": sem_registry.sanitize_name(s.get("name") or "_".join(cols)),
                "text_cols": cols,
                "max_chars": int(s.get("max_chars") or 2000),
                "mode": str(s.get("mode") or "full"),
                "agg": str(s.get("agg") or "tfidf"),
            })
        return out
    text_cols = ((state.get("config_snapshot") or {}).get("sample") or {}).get("text_cols") or []
    if not text_cols:
        raise ValueError("无文本源: session 未配置 sample.text_cols, 也未配置 semantic.sources")
    return [
        {"name": sem_registry.sanitize_name(c), "text_cols": [str(c)],
         "max_chars": 2000, "mode": "full", "agg": "tfidf"}
        for c in text_cols
    ]


def source_texts(df: pd.DataFrame, text_cols: list, max_chars: int) -> list:
    """多列拼接为单文档(空值补空串), full 模式超长头部截断(list/chunk 模式不截断)。"""
    if len(text_cols) == 1:
        s = df[text_cols[0]].fillna("").astype(str).str.strip()
    else:
        s = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    return [t[:max_chars] for t in s.tolist()]


def encode_source(encode, texts: list, source: dict, cache_path, signature: str) -> np.ndarray:
    """按 source.mode 编码: full(去重+截断) / list(条目级+聚合) / chunk(分窗池化)。

    list 模式缓存键含 mode+agg(签名由 run_semantic 拼), 编码量 = 唯一条目数。
    """
    mode = source.get("mode", "full")
    if mode == "list":
        item_lists = [encoders.split_items(t) for t in texts]
        uniq_items = list(dict.fromkeys(it for lst in item_lists for it in lst))
        idx = {it: i for i, it in enumerate(uniq_items)}
        if not uniq_items:
            probe = encode(["_probe_"])
            return np.zeros((len(texts), probe.shape[1]), dtype=float)
        item_vecs = encoders.encode_cached(
            encode, uniq_items, str(cache_path), signature)
        return encoders.list_aggregate(item_vecs, [[idx[i] for i in lst] for lst in item_lists],
                                       agg=source.get("agg", "tfidf"))
    if mode == "chunk":
        return encoders.chunk_pool(encode, texts, int(source.get("max_chars") or 2000))
    # full: 去重编码
    return encoders.encode_cached(
        lambda u: encoders.encode_unique(encode, u, progress=_progress_cb), texts, cache_path, signature)


def _progress_cb(done: int, total: int) -> None:
    print("  [encode] %d/%d unique texts (%.0f%%)" % (done, total, 100.0 * done / max(total, 1)), flush=True)


def source_signature(encoder_cfg: dict, source: dict) -> str:
    """缓存签名: encoder 配置 + source 编码相关字段(换 mode/agg/max_chars 即失效)。"""
    import json

    key = {k: source.get(k) for k in ("mode", "agg", "max_chars")}
    return json.dumps({"encoder": encoder_cfg, "source": key}, sort_keys=True, ensure_ascii=False)


def run_source_heads(session_dir, source: dict, embeddings: dict, y_train: pd.Series,
                     sem_cfg: dict, dedup_config: dict) -> tuple:
    """对单文本源跑全部 head 变体。

    Returns:
        (items, columns_by_split): items = registry 条目列表;
        columns_by_split = {split: {col: ndarray}} 各档新增语义列值
    """
    head_types = sem_cfg.get("heads") or DEFAULT_HEADS
    pca_dim = int(sem_cfg.get("pca_dim") or heads.DEFAULT_PCA_DIM)
    lr_params = sem_cfg.get("lr_params") or None
    oof_folds = int(sem_cfg.get("oof_folds") or heads.DEFAULT_OOF_FOLDS)
    src = source["name"]

    items, columns_by_split = [], {s: {} for s in paths.SPLITS}
    lr_artifact_cache = None  # quantile_bins 复用
    for head_type in head_types:
        if head_type == "lr_score":
            artifact = heads.fit_lr_head(embeddings["train"], y_train, params=lr_params, oof_folds=oof_folds)
            cols = ["sem_%s_score" % src]
            # train 段用 OOF 分数; 无文本行(present=0)置 NaN
            outputs = {}
            for s in paths.SPLITS:
                arr = heads.apply_lr_head(embeddings[s], artifact)
                outputs[s] = arr
            # OOF 覆盖 train 段
            if artifact.get("oof_score") is not None:
                outputs["train"] = np.asarray(artifact["oof_score"], dtype=float)
            output_type = "score"
        elif head_type == "pca_lowdim":
            artifact = heads.fit_pca_head(embeddings["train"], pca_dim)
            k = artifact["k"]
            cols = ["sem_%s_p%d" % (src, i + 1) for i in range(k)]
            outputs = {s: heads.apply_pca_head(embeddings[s], artifact) for s in paths.SPLITS}
            output_type = "lowdim"
        elif head_type == "gbdt_head":
            artifact = heads.fit_gbdt_head(embeddings["train"], y_train, params=sem_cfg.get("gbdt_params"),
                                           oof_folds=oof_folds)
            cols = ["sem_%s_gscore" % src]
            outputs = {s: heads.apply_gbdt_head(embeddings[s], artifact) for s in paths.SPLITS}
            if artifact.get("oof_score") is not None:
                outputs["train"] = np.asarray(artifact["oof_score"], dtype=float)
            output_type = "score"
        elif head_type == "quantile_bins":
            base = lr_artifact_cache or heads.fit_lr_head(embeddings["train"], y_train, params=lr_params,
                                                          oof_folds=0)
            lr_artifact_cache = base
            artifact = heads.fit_quantile_head(embeddings["train"], y_train, base_artifact=base,
                                               n_bins=int(sem_cfg.get("quantile_bins") or heads.DEFAULT_QUANTILE_BINS))
            k = artifact["n_bins"]
            cols = ["sem_%s_q%d" % (src, i + 1) for i in range(k)]
            outputs = {s: heads.apply_quantile_head(embeddings[s], artifact) for s in paths.SPLITS}
            output_type = "bins"
        elif head_type == "cluster_centroid":
            artifact = heads.fit_cluster_head(embeddings["train"],
                                              n_clusters=int(sem_cfg.get("n_clusters") or heads.DEFAULT_N_CLUSTERS))
            cols = ["sem_%s_cluster" % src]
            outputs = {s: heads.apply_cluster_head(embeddings[s], artifact) for s in paths.SPLITS}
            output_type = "cluster"
        else:
            raise ValueError("未知 head 类型: %s(可选 %s)" % (head_type, heads.HEAD_TYPES))

        item_name = "sem_%s_%s" % (src, head_type)
        heads.save_artifact(paths.semantic_dir(session_dir), item_name, artifact)

        for split in paths.SPLITS:
            arr = np.asarray(outputs[split], dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            for j, col in enumerate(cols):
                columns_by_split[split][col] = arr[:, j]

        items.append({
            "name": item_name,
            "source_text_cols": source["text_cols"],
            "head": head_type,
            "mode": source.get("mode", "full"),
            "output": {"type": output_type, "columns": cols},
            "artifact": "%s/head.json" % item_name,
        })
    return items, columns_by_split


def attach_present(all_columns: dict, sources_present: dict) -> None:
    """每源一列覆盖指示 sem_{src}_present(0/1)。有文本行=1。"""
    for split, per_src in sources_present.items():
        for src, mask in per_src.items():
            all_columns[split]["sem_%s_present" % src] = np.asarray(mask, dtype=float)


def apply_nan_mask(all_columns: dict, sources_present: dict, splits: dict, head_types: list) -> None:
    """无文本行的监督分(pca 无监督仍给几何值)置 NaN, 交给下游树模型原生处理。

    只处理 score 型 head(lr_score/gbdt_head), 其余类型(pca/quantile/cluster)保留几何值。
    """
    supervised = [h for h in head_types if h in ("lr_score", "gbdt_head")]
    if not supervised:
        return
    for split in all_columns:
        for src, mask in sources_present[split].items():
            mask = np.asarray(mask, dtype=bool)
            for h in supervised:
                col = "sem_%s_%s" % (src, "score" if h == "lr_score" else "gscore")
                if col in all_columns[split]:
                    vals = all_columns[split][col].copy()
                    vals[~mask] = np.nan
                    all_columns[split][col] = vals


def compute_item_metrics(items, all_columns, splits, label_col) -> None:
    """每列三档方向修正 AUC + PR-AUC(NaN 行剔除)。"""
    for item in items:
        metrics_by_col = {}
        for col in item["output"]["columns"]:
            metrics_by_col[col] = {}
            for split in paths.SPLITS:
                s = pd.Series(all_columns[split][col])
                y = splits[split][label_col]
                ok = s.notna()
                if ok.sum() == 0 or y[ok].nunique() < 2:
                    metrics_by_col[col]["%s_auc" % split] = None
                    metrics_by_col[col]["%s_pr_auc" % split] = None
                    continue
                auc = metrics.compute_auc_direction_fixed(y[ok], s[ok])
                pr = metrics.compute_pr_auc(y[ok], s[ok])
                metrics_by_col[col]["%s_auc" % split] = None if auc is None else round(float(auc), 6)
                metrics_by_col[col]["%s_pr_auc" % split] = None if pr is None else round(float(pr), 6)
        item["metrics"] = metrics_by_col


def compute_diagnostics(items, all_columns, splits, base_score_col) -> dict:
    """item 级条件增益诊断(配了 base_model_score_col 才算)。

    对每个 score 型输出列:
      conditional_auc_by_baseline_decile: 语义分在 baseline 分数十等分内的条件 AUC
        (判断特征是"补召回"还是"精排"—— top 分位 ≈0.5 说明对 top 人群无二次分层能力)
      spearman_vs_base_score: 与 baseline 分的 spearman 相关(互补性)
    """
    diag_by_item = {}
    if not base_score_col:
        return diag_by_item
    for split in ("oot",):
        df = splits.get(split)
        if df is None or base_score_col not in df.columns:
            return diag_by_item
    for item in items:
        if item["output"]["type"] != "score":
            continue
        diag = {}
        for col in item["output"]["columns"]:
            base = df[base_score_col]
            dec = pd.qcut(base.rank(method="first"), 10, labels=False, duplicates="drop") + 1
            s = pd.Series(all_columns["oot"][col])
            per_dec = []
            for d in sorted(dec.unique()):
                m = (dec == d) & s.notna()
                if m.sum() == 0 or df.loc[m, "y" if "y" in df.columns else df.columns[-1]].nunique() < 2:
                    continue
                label_col = splits["oot"].attrs.get("label_col", "y")
                a = metrics.compute_auc_direction_fixed(df.loc[m, label_col], s[m])
                if a is not None:
                    per_dec.append({"decile": int(d), "auc": round(float(a), 4)})
            diag["conditional_auc_by_baseline_decile"] = per_dec
            ok = s.notna() & base.notna()
            if ok.sum() > 1:
                diag["spearman_vs_base_score"] = round(float(s[ok].corr(base[ok], method="spearman")), 4)
        if diag:
            diag_by_item[item["name"]] = diag
    return diag_by_item


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="semantic-feature: 文本->Embedding->监督Head->注册语义特征")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--config", default=None, help="semantic_config.yaml 路径(缺省用 session_config 的 semantic 段)")
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        state = state_io.load_state(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    try:
        sem_cfg = load_semantic_config(session_dir, args.config)
        sources = resolve_sources(sem_cfg, state)
        encoder_cfg = sem_cfg.get("encoder") or {}
        encode = encoders.build_encoder(encoder_cfg)
    except (ValueError, encoders.EncoderError) as e:
        return _exit(2, "语义配置失败: %s" % e)

    try:
        sample_cfg = (state.get("config_snapshot") or {}).get("sample") or {}
        label_col = str(sample_cfg.get("label_col") or "label")
        id_cols = [str(c) for c in (sample_cfg.get("id_cols") or [])]
        base_score_col = str(sample_cfg.get("base_model_score_col") or "")
        if not id_cols:
            return _exit(2, "样本契约缺 id_cols, 语义特征无法对齐回 join")
        splits = {s: pd.read_parquet(paths.split_parquet(session_dir, s)) for s in paths.SPLITS}
        for s in paths.SPLITS:
            splits[s].attrs["label_col"] = label_col

        # ---- id_cols 唯一性校验(防下游 merge 崩) ----
        dedup_mode = str(sem_cfg.get("dedup") or "error")
        for s in paths.SPLITS:
            keys = splits[s][id_cols]
            dup_mask = keys.duplicated(keep=False)
            if dup_mask.any() and dedup_mode != "first":
                sample_dup = keys[dup_mask].head(3).to_dict("records")
                return _exit(2, "split=%s 的 id_cols %s 有 %d 行重复(样例 %s)。"
                                "请在样本侧保证 join 键唯一, 或在 semantic 配置 dedup: first 自动去重"
                                % (s, id_cols, int(dup_mask.sum()), sample_dup))
            if dup_mask.any():
                splits[s] = splits[s][~dup_mask].reset_index(drop=True)

        signature_base = encoders.encoder_signature(encoder_cfg)
        all_items, all_columns = [], {s: {} for s in paths.SPLITS}
        sources_present = {s: {} for s in paths.SPLITS}
        head_types = sem_cfg.get("heads") or DEFAULT_HEADS
        t_all = time.time()
        cost_by_item = {}
        for source in sources:
            missing = [c for c in source["text_cols"] if c not in splits["train"].columns]
            if missing:
                return _exit(2, "文本列 %s 不在样本中(可用列: %s)" % (missing, list(splits["train"].columns)))
            print("[encode] %s <- %s (mode=%s) ..." % (source["name"], source["text_cols"],
                                                        source.get("mode", "full")), flush=True)
            embeddings = {}
            t_src = time.time()
            for split, df in splits.items():
                texts = source_texts(df, source["text_cols"], source["max_chars"])
                sources_present[split][source["name"]] = [bool(t) for t in texts]
                cache = paths.semantic_dir(session_dir) / "cache" / ("%s_%s.npz" % (source["name"], split))
                sig = source_signature(encoder_cfg, source)
                embeddings[split] = encode_source(encode, texts, source, cache, sig)
            encode_seconds = round(time.time() - t_src, 1)
            print("[encode] %s done in %.1fs" % (source["name"], encode_seconds), flush=True)

            items, columns_by_split = run_source_heads(
                session_dir, source, embeddings, splits["train"][label_col], sem_cfg, sem_cfg
            )
            # 无文本行覆盖/NaN 处理(present 列在全部 item 汇总后统一挂)
            for it in items:
                cost_by_item[it["name"]] = {
                    "encode_seconds": encode_seconds,
                    "coverage": round(float(np.mean(sources_present["train"][source["name"]])), 4),
                }
            all_items.extend(items)
            for split in paths.SPLITS:
                all_columns[split].update(columns_by_split[split])

        # present 指示列 + 无文本行监督分 NaN
        attach_present(all_columns, sources_present)
        apply_nan_mask(all_columns, sources_present, splits, head_types)
        for src in sources_present["train"]:
            all_items.append({
                "name": "sem_%s_present" % src,
                "source_text_cols": [source["text_cols"][0]] if len(source["text_cols"]) == 1 else source["text_cols"],
                "head": "present",
                "mode": "full",
                "output": {"type": "present", "columns": ["sem_%s_present" % src]},
                "metrics": {},
                "artifact": None,
            })

        # 指标 + 诊断 + 成本
        compute_item_metrics(all_items, all_columns, splits, label_col)
        diagnostics = compute_diagnostics(all_items, all_columns, splits, base_score_col)
        for it in all_items:
            it["cost"] = cost_by_item.get(it["name"], {})

        flat_cols = []
        for item in all_items:
            flat_cols.extend(item["output"]["columns"])

        # features.parquet: 三档 id_cols + 语义列(按 id_cols join 回评估帧)
        parts = []
        for split in paths.SPLITS:
            part = splits[split][id_cols].copy()
            for col in flat_cols:
                part[col] = all_columns[split][col]
            parts.append(part)
        sem_dir = paths.semantic_dir(session_dir)
        sem_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_parquet(sem_dir / "features.parquet", index=False)

        # 诊断挂到 item
        for it in all_items:
            if it["name"] in diagnostics:
                it["diagnostics"] = diagnostics[it["name"]]

        sem_registry.write_registry(session_dir, {
            "id_cols": id_cols,
            "encoder": encoder_cfg,
            "items": all_items,
        })
        sem_registry.update_state_columns(session_dir, flat_cols)
        state_io.write_manifest(
            sem_dir, PRODUCED_BY,
            ["registry.json", "features.parquet"] + ["%s/head.json" % it["name"] for it in all_items if it["artifact"]],
            overview={"n_items": len(all_items), "n_columns": len(flat_cols)},
        )
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "语义特征管线失败: %s" % e)

    print("semantic 完成: %d 个变体 / %d 列 -> %s (%.1fs)"
          % (len(all_items), len(flat_cols), paths.semantic_registry_json(session_dir), time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
