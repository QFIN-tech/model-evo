from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path


def create_flow_folder(output_dir: Path | str, skill_name: str, flow_name: str) -> Path:
    root = Path(output_dir).expanduser().resolve() / skill_name
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_flow_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in flow_name)
    base = f"{timestamp}_{safe_flow_name}"
    candidate = root / base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root / f"{base}_{suffix:02d}"
    candidate.mkdir(parents=False)
    for child in ("inputs", "results", "artifacts"):
        (candidate / child).mkdir()
    return candidate


def ensure_flow_subdirs(flow_dir: Path | str) -> Path:
    root = Path(flow_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for child in ("inputs", "results", "artifacts"):
        (root / child).mkdir(exist_ok=True)
    return root


def infer_flow_folder_from_artifact(path: Path | str) -> Path:
    artifact = Path(path).expanduser().resolve()
    if artifact.parent.name == "artifacts":
        return ensure_flow_subdirs(artifact.parent.parent)
    return ensure_flow_subdirs(artifact.parent)
