# -*- coding: utf-8 -*-
"""测试公共路径注入: 与 scripts/_bootstrap.py 同逻辑(测试不经 CLI 入口, 需自行注入)。"""
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _TESTS_DIR.parent / "scripts"
_FM_ROOT = _TESTS_DIR.parents[1]

for _base in (_FM_ROOT, _FM_ROOT.parent):
    _shared = _base / "_modelevo-shared" / "scripts"
    if _shared.is_dir():
        if str(_shared) not in sys.path:
            sys.path.insert(0, str(_shared))
        break

_EVAL = _FM_ROOT / "feature-evolution-evaluation" / "scripts"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
