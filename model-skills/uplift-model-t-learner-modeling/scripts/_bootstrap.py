from __future__ import annotations

import sys
from pathlib import Path

_UPLIFT_COMMON_SCRIPTS = Path(__file__).resolve().parents[2] / "_uplift-common" / "scripts"
if not _UPLIFT_COMMON_SCRIPTS.is_dir():
    raise ImportError(f"Missing uplift common scripts directory: {_UPLIFT_COMMON_SCRIPTS}")

_path = str(_UPLIFT_COMMON_SCRIPTS)
if _path not in sys.path:
    sys.path.insert(0, _path)
