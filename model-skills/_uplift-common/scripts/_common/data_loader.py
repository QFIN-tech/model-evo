from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class DataSourceError(Exception):
    pass


class UnsupportedDataSourceError(DataSourceError):
    def __init__(self, message: str, *, kind: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind


class DataLoaderImportError(DataSourceError):
    def __init__(self, package: str, message: str) -> None:
        super().__init__(message)
        self.package = package


def normalize_data_source(source: dict[str, Any] | str, *, base_dir: Path | str | None = None) -> dict[str, Any]:
    if isinstance(source, str):
        return _local_csv_source(_resolve_path(source, base_dir))
    if not isinstance(source, dict):
        raise DataSourceError("data source must be a path string or an object.")
    kind = str(source.get("kind") or "local_csv").strip()
    if kind not in {"local_csv", "local_parquet"}:
        raise UnsupportedDataSourceError(f"Unsupported data source kind: {kind}", kind=kind)
    path_value = source.get("path")
    if not path_value:
        raise DataSourceError("data source path is required.")
    normalized = dict(source)
    normalized["kind"] = kind
    normalized["path"] = str(_resolve_path(str(path_value), base_dir))
    return normalized


def inspect_data_source(source: dict[str, Any] | str, *, base_dir: Path | str | None = None) -> dict[str, Any]:
    normalized = normalize_data_source(source, base_dir=base_dir)
    kind = str(normalized["kind"])
    if kind == "local_parquet":
        return {
            "kind": "local_parquet",
            "path": normalized["path"],
            "supported": False,
            "message": "local_parquet is reserved but not supported in this gate.",
        }
    if kind != "local_csv":
        raise UnsupportedDataSourceError(f"Unsupported data source kind: {kind}", kind=kind)
    path = Path(str(normalized["path"]))
    if not path.exists():
        raise DataSourceError(f"CSV file does not exist: {path}")
    stat = path.stat()
    return {
        **normalized,
        "supported": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "sha256": _sha256(path),
    }


def load_table(
    source: dict[str, Any] | str,
    *,
    base_dir: Path | str | None = None,
    columns: list[str] | None = None,
    nrows: int | None = None,
    dtype: Any = None,
) -> Any:
    normalized = normalize_data_source(source, base_dir=base_dir)
    kind = str(normalized["kind"])
    if kind == "local_parquet":
        raise UnsupportedDataSourceError(
            "local_parquet is reserved but not supported in this gate.",
            kind=kind,
        )
    if kind != "local_csv":
        raise UnsupportedDataSourceError(f"Unsupported data source kind: {kind}", kind=kind)
    path = Path(str(normalized["path"]))
    if not path.exists():
        raise DataSourceError(f"CSV file does not exist: {path}")
    pd = _import_pandas()
    return pd.read_csv(path, usecols=columns, nrows=nrows, dtype=dtype)


def _import_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise DataLoaderImportError("pandas", f"Missing package: pandas") from exc
    return pd


def _resolve_path(value: str, base_dir: Path | str | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir).expanduser().resolve() / path
    return path.resolve()


def _local_csv_source(path: Path) -> dict[str, Any]:
    return {"kind": "local_csv", "path": str(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
