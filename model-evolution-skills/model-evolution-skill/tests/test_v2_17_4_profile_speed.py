"""v2.17.4:profile 加速批次(分片并行读取 + split 级并发落盘)。

覆盖四件事:

1. ``read_parquet_shards`` 与"逐片读 + to_pandas + pd.concat"的**逐值等价**
   (含 decimal → float64、空分片跳过、行序=分片名排序);
2. ``shrink_object_columns`` 保留历史 loader 的 0.5 覆盖率口径;
3. ``write_splits`` 并发路径与串行路径的**产物逐字节等价**(parquet 文件
   内容 + manifest 的 splits 键序与取值)——这是本批次最关键的一条:
   并发写不能改变任何一个落盘字节,否则历史 run 的 sha256 对不上;
4. ``_HashingWriter`` 的 sha256 与"写完再 read_bytes"完全一致,且写入
   中途失败不留半成品。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.data import (  # noqa: E402
    _HashingWriter, _split_write_workers, _write_parquet_hashed, write_splits,
)
from runtime.loaders import read_parquet_shards, shrink_object_columns  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402


# --------------------------------------------------------------- 夹具
def _spec(reparse: tuple[str, ...] = ()) -> TaskSpec:
    data = {
        "task_id": "v2174", "task_name": "v2174",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
    }
    if reparse:
        data["features"] = {
            "reparse": [{"column": c, "reason": "form_unusable",
                         "plan": "test"} for c in reparse]}
    return TaskSpec(data=data)


def _frame(n_per_month: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = ("2026-01", "2026-02", "2026-03", "2026-04")
    n = n_per_month * len(months)
    return pd.DataFrame({
        "dt": [m for m in months for _ in range(n_per_month)],
        "uid": [f"u{i}" for i in range(n)],
        "seq": [f"e{i % 3}@DB_{i % 5}" for i in range(n)],
        "x1": rng.normal(size=n).astype("float32"),
        "x2": rng.integers(0, 5, size=n).astype("int32"),
        "label": rng.integers(0, 2, size=n),
    })


# ------------------------------------------- 1) read_parquet_shards 等价性
def _write_shards(tmp_path: Path, n_shards: int = 4) -> list[str]:
    import decimal

    import pyarrow as pa
    import pyarrow.parquet as pq

    paths: list[str] = []
    for i in range(n_shards):
        n = 50
        rng = np.random.default_rng(100 + i)
        table = pa.table({
            "uid": [f"s{i}-{j}" for j in range(n)],
            "x": pa.array(rng.normal(size=n), type=pa.float64()),
            "amt": pa.array(
                [None if j % 7 == 0 else decimal.Decimal(str(j)) +
                 decimal.Decimal("1.25") for j in range(n)],
                type=pa.decimal128(29, 2)),
            "name": pa.array([f"n{j % 3}" for j in range(n)],
                             type=pa.string()),
        })
        p = tmp_path / f"part-{i:04d}.parquet"
        pq.write_table(table, p)
        paths.append(str(p))
    # 中间放一个 0 行占位分片(常见于上游切分)
    empty = tmp_path / "part-0009.parquet"
    pq.write_table(pa.table({"uid": pa.array([], pa.string()),
                             "x": pa.array([], pa.float64()),
                             "amt": pa.array([], pa.decimal128(29, 2)),
                             "name": pa.array([], pa.string())}), empty)
    paths.append(str(empty))
    return paths


def _sequential_reference(paths: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    frames = []
    for p in sorted(paths):
        pf = pq.ParquetFile(p)
        if pf.metadata.num_rows == 0:
            continue
        t = pf.read(use_threads=True)
        frames.append(t.to_pandas())
    return pd.concat(frames, ignore_index=True)


def test_read_parquet_shards_matches_sequential(tmp_path):
    paths = _write_shards(tmp_path)
    got = read_parquet_shards(paths, threads=4)
    ref = _sequential_reference(paths)

    assert list(got.columns) == list(ref.columns)
    assert len(got) == len(ref) == 200
    # 行序 = 分片名排序序
    assert got["uid"].tolist() == ref["uid"].tolist()
    # 空分片被跳过,没有多出一行全空
    assert got["uid"].notna().all()
    # decimal 列在 Arrow 层转 float64(旧路径是 object)。逐值比较时先把
    # object 侧转 float 再比 —— 这里验证的是"值不变",不是 dtype 不变。
    assert str(got["amt"].dtype) == "float64"
    ref_amt = pd.to_numeric(ref["amt"], errors="coerce").to_numpy(dtype="float64")
    assert np.array_equal(got["amt"].to_numpy(), ref_amt, equal_nan=True)
    assert np.array_equal(got["x"].to_numpy(), ref["x"].to_numpy())
    assert got["name"].tolist() == ref["name"].tolist()


def test_read_parquet_shards_serial_and_parallel_agree(tmp_path):
    paths = _write_shards(tmp_path, n_shards=6)
    a = read_parquet_shards(paths, threads=1)
    b = read_parquet_shards(paths, threads=6)
    assert a.equals(b)


def test_read_parquet_shards_single_file_and_empty(tmp_path):
    paths = _write_shards(tmp_path, n_shards=1)
    got = read_parquet_shards(paths)
    assert len(got) == 50

    import pyarrow as pa
    import pyarrow.parquet as pq

    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"a": pa.array([], pa.float64())}), empty)
    with pytest.raises(ValueError):
        read_parquet_shards([str(empty)])


def test_read_parquet_shards_post_hook_is_row_preserving(tmp_path):
    paths = _write_shards(tmp_path)
    got = read_parquet_shards(
        paths, post=lambda f: shrink_object_columns(f, keep_as_is=("name",)))
    assert len(got) == 200
    assert got["name"].dtype == object  # 白名单列不动


def test_shrink_object_columns_keeps_history_threshold():
    """0.5 覆盖率口径:稀疏但可解析的列不降 dtype,密集的才降。"""
    dense = pd.Series([str(v) for v in range(100)], dtype=object)
    sparse = pd.Series([str(v) if i < 10 else None for i, v in
                        enumerate(range(100))], dtype=object)
    keep = pd.Series([f"id{i}" for i in range(100)], dtype=object)
    frame = pd.DataFrame({"dense": dense, "sparse": sparse, "keepme": keep})
    shrink_object_columns(frame, keep_as_is=("keepme",))
    assert frame["dense"].dtype == "float32"
    assert frame["sparse"].dtype == object
    assert frame["keepme"].dtype == object


# ------------------------------------------- 2) write_splits 逐字节等价
def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.parquet"))
    }


def test_write_splits_parallel_matches_serial_bytes(tmp_path):
    """并发落盘必须与串行路径产出**逐字节相同**的 parquet 与 manifest。"""
    df = _frame(n_per_month=60)
    spec = _spec(reparse=("seq",))

    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    m_serial = write_splits(df, spec, serial_dir, workers=1)
    m_parallel = write_splits(df, spec, parallel_dir, workers=4)

    assert _snapshot(serial_dir) == _snapshot(parallel_dir)
    # manifest 除 created_at 外逐值相同,且 splits 键序保持 split_index 顺序
    assert list(m_serial["splits"]) == list(m_parallel["splits"])
    for name in m_serial["splits"]:
        a = dict(m_serial["splits"][name])
        b = dict(m_parallel["splits"][name])
        assert a == b, f"{name} manifest 条目不一致: {a} != {b}"


def test_write_splits_manifest_sha256_matches_file(tmp_path):
    df = _frame()
    out = tmp_path / "splits"
    manifest = write_splits(df, _spec(reparse=("seq",)), out, workers=2)
    for name, entry in manifest["splits"].items():
        p = out / entry["file"]
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        assert entry["sha256"] == digest, f"{name} sha256 与文件不符"


def test_write_splits_env_override_and_auto_guard(monkeypatch, tmp_path):
    df = _frame()
    items = [(n, None) for n in ("train", "validation", "shadow_test",
                                 "final_test")]
    monkeypatch.setenv("MODELEVO_SPLIT_WRITE_WORKERS", "2")
    assert _split_write_workers(items, df, None) == 2
    monkeypatch.setenv("MODELEVO_SPLIT_WRITE_WORKERS", "abc")
    w = _split_write_workers(items, df, None)
    assert w in (1, 2, 3, 4)  # 非法值回退自动路径,不抛错
    # 显式入参优先于环境变量,并夹到 split 数
    assert _split_write_workers(items, df, 99) == 4
    assert _split_write_workers(items, df, 1) == 1
    assert _split_write_workers(items[:1], df, 4) == 1


def test_write_splits_memory_guard_falls_back_to_serial(monkeypatch):
    """可用内存不足预算时降为串行(不赌 OOM)。"""
    monkeypatch.delenv("MODELEVO_SPLIT_WRITE_WORKERS", raising=False)
    df = pd.DataFrame(np.zeros((10_000, 100), dtype="float32"))
    items = [("train", None), ("validation", None)]

    import psutil

    class _VM:
        available = 1024 ** 2  # 1 MB:必然低于预算

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    assert _split_write_workers(items, df, None) == 1


# ------------------------------------------------- 3) 写盘时同步算 hash
def test_hashing_writer_matches_readback(tmp_path):
    df = pd.DataFrame({"a": np.arange(1000, dtype="float32")})
    p = tmp_path / "x.parquet"
    digest = _write_parquet_hashed(df, p)
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    # 产物可正常回读
    assert pd.read_parquet(p).equals(df)


def test_write_parquet_hashed_removes_partial_file(tmp_path):
    class _Boom(pd.DataFrame):
        def to_parquet(self, *a, **k):  # noqa: D401
            raise RuntimeError("boom")

    p = tmp_path / "bad.parquet"
    with pytest.raises(RuntimeError):
        _write_parquet_hashed(_Boom({"a": [1, 2]}), p)
    assert not p.exists(), "写入失败必须删除半成品"


def test_hashing_writer_closes_and_is_idempotent(tmp_path):
    p = tmp_path / "y.bin"
    w = _HashingWriter(p)
    w.write(b"hello")
    assert w.tell() == 5
    w.close()
    w.close()  # 二次 close 不应抛错
    assert p.read_bytes() == b"hello"


# ------------------------------------------- 4) loader 能 import runtime
def test_load_dataframe_exposes_skill_root_on_sys_path(tmp_path, monkeypatch):
    """loader 必须能 ``from runtime.loaders import ...``(模板已这样写)。"""
    import sys as _sys

    loader = tmp_path / "data-loader.py"
    loader.write_text(
        "import pandas as pd\n"
        "def load_data():\n"
        "    from runtime.loaders import read_parquet_shards  # noqa: F401\n"
        "    return pd.DataFrame({'dt': ['2026-01'], 'uid': ['u'],\n"
        "                         'label': [1]})\n", encoding="utf-8")
    spec = TaskSpec(data={**_spec().data,
                          "data": {"data_loader": str(loader)}})
    for p in [x for x in _sys.path if Path(x).resolve() == SKILL]:
        _sys.path.remove(p)  # 模拟"进程里还没有 skill 根"的现场
    from runtime.data import load_dataframe

    out = load_dataframe(spec)
    assert len(out) == 1
    assert str(SKILL) in _sys.path
