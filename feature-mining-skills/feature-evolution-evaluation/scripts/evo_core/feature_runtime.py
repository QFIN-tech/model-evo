# -*- coding: utf-8 -*-
"""评估帧构造: baseline 特征 + 已接受特征(fid) + 语义特征(semantic/features.parquet)。

所有关卡评估共用同一套帧构造, 保证「比的是特征增益而不是帧差异」:
- base 列: baseline model_meta.json 里的 feature_cols(与 baseline 完全同源, 轮间可比)
- accepted 列: 按 state.champion_fids 顺序逐个执行 accepted/{fid}.py 的 transform,
  后面的特征能看到前面的特征(exploit 候选可在 champion 之上继续加工)
- semantic 列: semantic-feature 注册的语义特征, 按 id_cols 从 semantic/features.parquet 左连接
- transform 的输入帧永远不含 label/id/dt/既有模型分(防泄漏)

大数据量优化:
- champion 物化缓存(data/champion-cache/): 已接受特征值算好落盘, 键含 fid 代码指纹
  与切分/语义文件印记, 命中则直接读列, 不再每轮重执行全部接受代码
- build_frames(splits=None) 返回的帧归调用链所有, 可原地加列(G4/G5/融合帧都依赖此约定)
"""
from __future__ import annotations

import hashlib

import pandas as pd

from . import candidate_exec, paths, state_io

# 候选执行超时(秒); G1 关卡可覆盖
DEFAULT_TIMEOUT_S = 60.0


def load_splits(session_dir) -> dict:
    """读三档切分 parquet。"""
    return {s: pd.read_parquet(paths.split_parquet(session_dir, s)) for s in paths.SPLITS}


def load_baseline_feature_cols(session_dir) -> list:
    """baseline 模型的特征列清单(champion 帧的 base 列唯一来源)。"""
    meta = state_io.read_json(paths.baseline_model_dir(session_dir) / "model_meta.json")
    return list(meta["feature_cols"])


def semantic_columns(session_dir) -> list:
    """注册进来的语义特征列(无注册时为空列表)。"""
    reg_path = paths.semantic_registry_json(session_dir)
    if not reg_path.exists():
        return []
    reg = state_io.read_json(reg_path)
    cols = []
    for item in reg.get("items", []):
        cols.extend(item.get("output", {}).get("columns", []))
    return cols


def _attach_semantic(session_dir, frames: dict) -> dict:
    """把语义特征按 id_cols 左连接到三档帧(缺注册则原样返回)。"""
    reg_path = paths.semantic_registry_json(session_dir)
    if not reg_path.exists():
        return frames
    reg = state_io.read_json(reg_path)
    id_cols = reg.get("id_cols") or []
    if not id_cols:
        return frames
    sem = pd.read_parquet(paths.semantic_dir(session_dir) / "features.parquet")
    value_cols = [c for c in sem.columns if c not in id_cols]
    if not value_cols:
        return frames
    sem = sem[id_cols + value_cols]
    out = {}
    for split, df in frames.items():
        merged = df.merge(sem, on=id_cols, how="left", validate="many_to_one")
        out[split] = merged
    return out


def _apply_accepted(session_dir, frames: dict, fids: list, timeout_s: float, own: bool = True) -> dict:
    """按顺序执行已接受特征代码, 把 fid 列加进三档帧。

    own=True: frames 归调用链所有(内部新构造), 原地加列不做整帧拷贝;
    own=False: 调用方传入的帧, 先整体拷贝一次避免污染(测试路径)。
    每档只构造一次 transform 输入帧(去 label/id/dt/既有分), fid 结果同时回写
    输入帧(后续特征可见)与评估帧(G4/G5 用)。

    Raises:
        candidate_exec.CandidateCodeError 之外的执行失败直接抛出(已接受特征损坏属严重态,
        不静默跳过)。
    """
    if not fids:
        return frames
    if not own:
        frames = {s: df.copy() for s, df in frames.items()}
    input_frames = {s: _input_frame(df) for s, df in frames.items()}
    for fid in fids:
        code = paths.accepted_py(session_dir, fid).read_text(encoding="utf-8")
        for split, df in frames.items():
            run = candidate_exec.run_transform(code, input_frames[split], timeout_s=timeout_s)
            if not run["ok"]:
                raise RuntimeError("已接受特征 %s 在 %s 档执行失败: %s" % (fid, split, run["error"]))
            vals = run["series"].to_numpy()
            df[fid] = vals
            input_frames[split][fid] = vals
    return frames


def _input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """transform 的合法输入列: 去掉 label/id/dt/既有模型分(列名约定见 sample_contract)。

    label/id/dt 防泄漏; 既有模型分防"抄近道"(候选直接复制打分即可过单变量门禁)。
    无可删列时原样返回(避免整帧拷贝)。
    """
    drop = [c for c in df.attrs.get("_drop_cols", []) if c in df.columns]
    return df.drop(columns=drop) if drop else df


# ---- champion 物化缓存 ----
def _file_stamp(p) -> dict | None:
    """文件印记(大小 + mtime_ns), 缓存失效判定用(本仓库文件只由脚本写一次)。"""
    try:
        st = p.stat()
        return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        return None


def _cache_key(session_dir, fids: list) -> dict | None:
    """缓存键: fid 清单 + 各代码指纹 + 切分/语义文件印记。任一文件缺失返回 None。"""
    code_sha = {}
    for fid in fids:
        p = paths.accepted_py(session_dir, fid)
        try:
            code_sha[fid] = hashlib.sha1(p.read_bytes()).hexdigest()
        except OSError:
            return None
    return {
        "schema_version": 1,
        "fids": list(fids),
        "code_sha": code_sha,
        "splits": {s: _file_stamp(paths.split_parquet(session_dir, s)) for s in paths.SPLITS},
        "semantic_features": _file_stamp(paths.semantic_dir(session_dir) / "features.parquet"),
        "semantic_registry": _file_stamp(paths.semantic_registry_json(session_dir)),
    }


def _load_cache(session_dir, splits: dict, key: dict) -> dict | None:
    """命中则返回 {split: DataFrame(仅 fid 列)}, 行序与切分 parquet 一致(按位对齐)。"""
    if key is None:
        return None
    mpath = paths.champion_cache_manifest(session_dir)
    if not mpath.exists():
        return None
    try:
        if state_io.read_json(mpath) != key:
            return None
        out = {}
        for s in paths.SPLITS:
            p = paths.champion_cache_parquet(session_dir, s)
            if not p.exists():
                return None
            cached = pd.read_parquet(p)
            if len(cached) != len(splits[s]):
                return None
            out[s] = cached
        return out
    except Exception:
        return None  # 缓存任何异常都退回重算, 不影响正确性


def _write_cache(session_dir, frames: dict, fids: list, key: dict) -> None:
    """写物化缓存; 失败静默(缓存只是加速, 不承载正确性)。"""
    if key is None:
        return
    try:
        cdir = paths.champion_cache_dir(session_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        for s, df in frames.items():
            df[fids].to_parquet(paths.champion_cache_parquet(session_dir, s), index=False)
        state_io.write_json(paths.champion_cache_manifest(session_dir), key)
    except Exception:
        pass


def build_frames(
    session_dir,
    splits: dict | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    with_accepted: bool = True,
) -> tuple:
    """构造三档评估帧。

    Args:
        session_dir: session 根目录
        splits: 缺省时从 data/splits/ 读
        timeout_s: 已接受特征执行超时
        with_accepted: False 时只给 base+semantic(如 G6 回滚后重建)

    Returns:
        (frames, contract_meta): frames = {split: df}, 每帧 attrs["_drop_cols"]
        记录 label/id/dt 列名; contract_meta = {label_col, id_cols, base_cols, semantic_cols, accepted_fids}
    """
    state = state_io.load_state(session_dir)
    cfg = state.get("config_snapshot") or {}
    sample_cfg = cfg.get("sample") or {}
    label_col = str(sample_cfg.get("label_col") or "label")
    id_cols = [str(c) for c in (sample_cfg.get("id_cols") or [])]
    dt_col = str(sample_cfg.get("dt_col") or "dt")

    caller_splits = splits is not None  # 调用方自带切分(测试)时不走缓存
    splits = splits if splits is not None else load_splits(session_dir)
    base_cols = load_baseline_feature_cols(session_dir)
    frames = _attach_semantic(session_dir, splits)
    accepted_fids = list(state.get("champion_fids") or []) if with_accepted else []
    # 既有模型打分列与 label/id/dt 同等对待: 只进 champion 帧(base_cols), 不进候选输入帧
    score_col = str(sample_cfg.get("base_model_score_col") or "")
    drop_cols = [label_col, dt_col] + id_cols + ([score_col] if score_col else [])
    drop_cols = [c for c in drop_cols if c]
    for df in frames.values():
        df.attrs["_drop_cols"] = drop_cols
    if accepted_fids:
        # champion 物化缓存: 仅当切分来自磁盘(键依赖切分文件印记)时启用;
        # 调用方自带 splits(测试)或文件缺失时键为 None, 直接重算。
        key = _cache_key(session_dir, accepted_fids) if not caller_splits else None
        cached = _load_cache(session_dir, frames, key)
        if cached is not None:
            # 按位对齐(行序与切分 parquet 一致), 直接挂列
            for s, df in frames.items():
                for fid in accepted_fids:
                    df[fid] = cached[s][fid].to_numpy()
        else:
            # 切分来自磁盘 -> 帧归本调用链, 原地加列; 调用方自带切分 -> 拷贝一次保语义
            frames = _apply_accepted(session_dir, frames, accepted_fids, timeout_s, own=not caller_splits)
            _write_cache(session_dir, frames, accepted_fids, key)

    contract_meta = {
        "label_col": label_col,
        "id_cols": id_cols,
        "dt_col": dt_col,
        "base_cols": base_cols,
        "semantic_cols": semantic_columns(session_dir),
        "accepted_fids": accepted_fids,
    }
    return frames, contract_meta


def champion_feature_cols(contract_meta: dict) -> list:
    """champion 模型的特征列 = base + accepted fid(语义列不直接入 champion 模型,
    它们作为候选特征的原料进入 fid)。"""
    return list(contract_meta["base_cols"]) + list(contract_meta["accepted_fids"])


def candidate_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    """候选 transform 的输入帧(去掉 label/id/dt)。"""
    return _input_frame(df)
