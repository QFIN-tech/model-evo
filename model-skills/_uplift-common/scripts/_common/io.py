from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def read_cli_json_input(input_arg: str) -> dict[str, Any]:
    if input_arg == "-":
        if hasattr(sys.stdin, "buffer"):
            raw = sys.stdin.buffer.read().decode("utf-8-sig")
        else:
            raw = sys.stdin.read()
    else:
        raw = Path(input_arg).expanduser().resolve().read_text(encoding="utf-8-sig")
    if not raw.strip():
        raise ValueError("CLI input JSON is empty.")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("CLI input JSON must be an object.")
    return payload


def read_json(path: Path | str) -> dict[str, Any]:
    payload = Path(path).read_text(encoding="utf-8-sig")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding=encoding)
