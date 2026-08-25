# -*- coding: utf-8 -*-
"""run_semantic.py: semantic-feature 主入口 Text -> Embedding -> 监督 Head -> 注册。

用法:
    python run_semantic.py --session-dir <session_dir> [--config <semantic_config.yaml>]

流程:
  1. 读 session(须已完成 prepare_session): 三档切分 + 样本契约(text_cols)
  2. 对每个文本源(单列或多列拼接): 三档文本 -> encoder 嵌入(磁盘缓存, 换后端自动失效)
  3. 每个 head 变体(缺省 lr_score + pca_lowdim): 只用 train 段拟合, 三档推理
  4. 每个输出列算方向修正 AUC(train/test/oot), registry.json + features.parquet 落盘,
     回写 state.semantic_columns -- 语义列自此成为 feature-evolution 候选生成的合法原料

防穿越: head 只见 train 段的 label; test/oot 仅推理。

退出码: 0 成功 / 2 配置或契约失败 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import sys
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
    """文本源列表: 配置显式给出, 或按样本契约 text_cols 每列一源。"""
    sources = sem_cfg.get("sources")
    if sources:
        out = []
        for s in sources:
            if not s.get("text_cols"):
                raise ValueError("semantic.sources 每项必须给 text_cols")
            out.append({
                "name": sem_registry.sanitize_name(s.get("name") or "_".join(s["text_cols"])),
                "text_cols": [str(c) for c in s["text_cols"]],
                "max_chars": int(s.get("max_chars") or 2000),
            })
        return out
    text_cols = ((state.get("config_snapshot") or {}).get("sample") or {}).get("text_cols") or []
    if not text_cols:
        raise ValueError("无文本源: session 未配置 sample.text_cols, 也未配置 semantic.sources")
    return [
        {"name": sem_registry.sanitize_name(c), "text_cols": [str(c)], "max_chars": 2000}
        for c in text_cols
    ]


def source_texts(df: pd.DataFrame, text_cols: list, max_chars: int) -> list:
    """多列拼接为单文档(空值补空串), 超长截断。"""
    if len(text_cols) == 1:
        s = df[text_cols[0]].fillna("").astype(str).str.strip()
    else:
        s = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
    return [t[:max_chars] for t in s.tolist()]


def run_source_heads(session_dir, source: dict, embeddings: dict, y_train: pd.Series, sem_cfg: dict) -> tuple:
    """对单文本源跑全部 head 变体。

    Returns:
        (items, columns_by_split): items = registry 条目列表;
        columns_by_split = {split: {col: ndarray}} 各档新增语义列值
    """
    head_types = sem_cfg.get("heads") or DEFAULT_HEADS
    pca_dim = int(sem_cfg.get("pca_dim") or heads.DEFAULT_PCA_DIM)
    lr_params = sem_cfg.get("lr_params") or None
    src = source["name"]

    items, columns_by_split = [], {s: {} for s in paths.SPLITS}
    for head_type in head_types:
        if head_type == "lr_score":
            artifact = heads.fit_lr_head(embeddings["train"], y_train, params=lr_params)
            cols = ["sem_%s_score" % src]
            outputs = {s: heads.apply_lr_head(embeddings[s], artifact) for s in paths.SPLITS}
            output_type = "score"
        elif head_type == "pca_lowdim":
            artifact = heads.fit_pca_head(embeddings["train"], pca_dim)
            k = artifact["k"]
            cols = ["sem_%s_p%d" % (src, i + 1) for i in range(k)]
            outputs = {s: heads.apply_pca_head(embeddings[s], artifact) for s in paths.SPLITS}
            output_type = "lowdim"
        else:
            raise ValueError("未知 head 类型: %s(可选 %s)" % (head_type, DEFAULT_HEADS))

        # 工件落盘(item 目录内); 指标在 main 统一计算(需要各档 y)
        item_name = "sem_%s_%s" % (src, head_type)
        heads.save_artifact(paths.semantic_item_dir(session_dir, item_name), item_name, artifact)

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
            "output": {"type": output_type, "columns": cols},
            "artifact": "%s/head.json" % item_name,
        })
    return items, columns_by_split


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
        label_col = str(((state.get("config_snapshot") or {}).get("sample") or {}).get("label_col") or "label")
        id_cols = [str(c) for c in ((state.get("config_snapshot") or {}).get("sample") or {}).get("id_cols") or []]
        if not id_cols:
            return _exit(2, "样本契约缺 id_cols, 语义特征无法对齐回 join")
        splits = {s: pd.read_parquet(paths.split_parquet(session_dir, s)) for s in paths.SPLITS}

        signature = encoders.encoder_signature(encoder_cfg)
        all_items, all_columns = [], {s: {} for s in paths.SPLITS}
        for source in sources:
            missing = [c for c in source["text_cols"] if c not in splits["train"].columns]
            if missing:
                return _exit(2, "文本列 %s 不在样本中(可用列: %s)" % (missing, list(splits["train"].columns)))
            print("[encode] %s <- %s ..." % (source["name"], source["text_cols"]))
            embeddings = {}
            for split, df in splits.items():
                texts = source_texts(df, source["text_cols"], source["max_chars"])
                cache = paths.semantic_dir(session_dir) / "cache" / ("%s_%s.npz" % (source["name"], split))
                embeddings[split] = encoders.encode_cached(encode, texts, cache, signature)

            items, columns_by_split = run_source_heads(
                session_dir, source, embeddings, splits["train"][label_col], sem_cfg
            )
            all_items.extend(items)
            for split in paths.SPLITS:
                all_columns[split].update(columns_by_split[split])

        # 指标(每列方向修正 AUC) + registry
        for item in all_items:
            metrics_by_col = {}
            for col in item["output"]["columns"]:
                metrics_by_col[col] = {}
                for split in paths.SPLITS:
                    auc = metrics.compute_auc_direction_fixed(splits[split][label_col], all_columns[split][col])
                    metrics_by_col[col]["%s_auc" % split] = None if auc is None else round(float(auc), 6)
            item["metrics"] = metrics_by_col

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

        sem_registry.write_registry(session_dir, {
            "id_cols": id_cols,
            "encoder": encoder_cfg,
            "items": all_items,
        })
        sem_registry.update_state_columns(session_dir, flat_cols)
        state_io.write_manifest(
            sem_dir, PRODUCED_BY,
            ["registry.json", "features.parquet"] + ["%s/head.json" % it["name"] for it in all_items],
            overview={"n_items": len(all_items), "n_columns": len(flat_cols)},
        )
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "语义特征管线失败: %s" % e)

    print("semantic 完成: %d 个变体 / %d 列 -> %s" % (len(all_items), len(flat_cols), paths.semantic_registry_json(session_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
