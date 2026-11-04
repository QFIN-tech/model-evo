"""v2.18.0 经验库:整库导出 / 导入(Q36)。

用于后端迁移(如本机 SQLite → 团队 PostgreSQL + 对象存储):按表原样导出全部行
(保留 seq 与 id,即保留全部水位与合并结果),并复制产物对象。**不重新合并,
不做双向同步**;导入目标必须是空库。
"""
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import func, insert, select, text

from . import schema as S
from .artifacts import ArtifactStore
from .store import LibraryStore, clean_json

DUMP_SCHEMA = "modelevo.library-dump/1"
#: 迁移搬运的表:由 metadata 依赖顺序派生,新增表自动纳入(漏表会让迁库静默丢数据)。
#: 只有 ``leases`` 例外 —— 它是最长 30 分钟的运维锁,不应跟着库搬走。
SKIP_TABLES = frozenset({"leases"})
TABLES = tuple(t for t in S.metadata.sorted_tables if t.name not in SKIP_TABLES)


def dump(store: LibraryStore, artifacts: ArtifactStore, out_dir) -> dict:
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    counts = {}
    with store.engine.connect() as conn:
        for t in TABLES:
            order = [c for c in t.primary_key.columns]
            n = 0
            with open(out / "tables" / f"{t.name}.jsonl", "w", encoding="utf-8") as fh:
                for row in conn.execute(select(t).order_by(*order)):
                    fh.write(json.dumps(clean_json(dict(row._mapping)),
                                        ensure_ascii=False) + "\n")
                    n += 1
            counts[t.name] = n
    target = ArtifactStore(f"file://{(out / 'cas').resolve()}")
    n_obj = 0
    for h in artifacts.iter_hashes():
        target.put_bytes(artifacts.get_bytes(h), expected=h)
        n_obj += 1
    manifest = {"schema": DUMP_SCHEMA, "watermark": store.watermark(),
                "schema_revision": store.schema_revision(), "tables": counts,
                "artifacts": n_obj}
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load(store: LibraryStore, artifacts: ArtifactStore, in_dir) -> dict:
    src = Path(in_dir)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != DUMP_SCHEMA:
        raise ValueError(f"不是 {DUMP_SCHEMA} 导出目录: {src}")
    if store.watermark() != 0:
        raise ValueError("导入目标必须是空库(水位 0)")
    if manifest.get("schema_revision") != store.schema_revision():
        raise ValueError(
            f"schema 版本不一致: 导出 {manifest.get('schema_revision')} / "
            f"目标 {store.schema_revision()};先把目标升级到同一版本")
    counts = {}
    with store.engine.begin() as conn:
        for t in TABLES:
            path = src / "tables" / f"{t.name}.jsonl"
            rows = [json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for i in range(0, len(rows), 500):
                conn.execute(insert(t), rows[i:i + 500])
            counts[t.name] = len(rows)
        if store.dialect == "postgresql":
            # 显式写入了自增列,序列要追到当前最大值
            for t in TABLES:
                pk = [c for c in t.primary_key.columns if c.autoincrement is True
                      or c.name in ("id", "seq")]
                for c in pk:
                    mx = conn.execute(select(func.max(c))).scalar()
                    if mx:
                        conn.execute(text(
                            f"SELECT setval(pg_get_serial_sequence('{t.name}', "
                            f"'{c.name}'), :v)"), {"v": int(mx)})
    # 导出清单里的行数与实际导入比对:任何一张表少行都说明搬运不完整(旧版曾静默丢
    # topics / dossiers / changelog_entries / evidence_links)
    want = manifest.get("tables") or {}
    short = {k: (want[k], counts.get(k, 0)) for k in want
             if counts.get(k, 0) != want[k]}
    if short:
        raise ValueError(f"导入行数与导出清单不一致(表: 期望/实际): {short}")
    source = ArtifactStore(f"file://{(src / 'cas').resolve()}")
    n_obj = 0
    for h in source.iter_hashes():
        artifacts.put_bytes(source.get_bytes(h), expected=h)
        n_obj += 1
    return {"tables": counts, "artifacts": n_obj, "watermark": store.watermark()}
