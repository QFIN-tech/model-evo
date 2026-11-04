"""v2.18.1:指纹别名建议。

col_hash8 是全量列名的哈希:同一份数据加一列就是新底座,同任务经验随之不可见,
此前只能人工 ``add-alias``。这里按任务画像找候选:同 model_type、同目标(去变换
后缀)、列 MinHash Jaccard ≥ 阈值、尚未互为别名。**只建议,不自动写入** ——
列集大变时同任务经验未必成立,由人确认后用 ``library alias`` 落库。
"""
from __future__ import annotations

from .snapshot import parse_atid
from .task_profile import minhash_jaccard, profile_from_atid, target_base


def suggest_aliases(store, *, at: int | None = None, min_columns: float = 0.9) -> list[dict]:
    at = store._at(at)
    profiles = store.fingerprint_profiles(at=at)
    with store._read() as conn:
        amap = store._alias_map(conn, at)
    fps = store.fingerprints(at=at)
    first_seq = {f["atid"]: f["commit_seq"] for f in fps}
    every = {f["atid"] for f in fps} | set(profiles)
    # 只在规范指纹之间比较:已是别名的指纹由其规范指纹代表(画像缺失时借用别名的)
    atids = sorted({store._canonical(a, amap) for a in every})
    for a in every:
        c = store._canonical(a, amap)
        if c != a and c not in profiles and a in profiles:
            profiles[c] = profiles[a]
    out = []
    for i, a in enumerate(atids):
        pa, ma = profiles.get(a) or {}, (parse_atid(a) or {})
        for b in atids[i + 1:]:
            pb, mb = profiles.get(b) or {}, (parse_atid(b) or {})
            if ma.get("model_type") != mb.get("model_type"):
                continue
            ta = pa.get("target_name") or profile_from_atid(a).get("target_name")
            tb = pb.get("target_name") or profile_from_atid(b).get("target_name")
            # 同一目标且同一变换才可能是"同一任务";log1p 与 raw 是不同口径
            if not ta or ta != tb or target_base(ta) != target_base(tb):
                continue
            jac = minhash_jaccard(pa.get("col_minhash"), pb.get("col_minhash"))
            if jac is None or jac < min_columns:
                continue
            # 列更多(更新)的一侧作别名;列数相同时先登记的一侧作规范 —— 只是默认建议
            na, nb = pa.get("n_columns") or 0, pb.get("n_columns") or 0
            alias, canonical = (a, b) if na > nb else (b, a) if nb > na else \
                tuple(sorted((a, b), key=lambda x: -first_seq.get(x, 0)))
            out.append({"alias_atid": alias, "canonical_atid": canonical,
                        "columns_jaccard": round(jac, 3),
                        "n_columns": {a: na, b: nb},
                        "reason": f"同目标 {ta},列 Jaccard≈{jac:.2f}"})
    out.sort(key=lambda r: -r["columns_jaccard"])
    return out
