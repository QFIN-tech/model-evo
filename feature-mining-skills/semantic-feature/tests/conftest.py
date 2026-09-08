# -*- coding: utf-8 -*-
"""测试公共路径注入: 与 scripts/_bootstrap.py 同逻辑(测试不经 CLI 入口, 需自行注入)。

单目录拷贝部署时(本 skill 不在 skills 根下), 用 MODELEVO_SKILLS_ROOT 指向 skills 根。
"""
import os
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _TESTS_DIR.parent / "scripts"
_FM_ROOT = _TESTS_DIR.parents[1]

_CANDIDATES = []
_env = os.environ.get("MODELEVO_SKILLS_ROOT")
if _env:
    _CANDIDATES.append(Path(_env))
_CANDIDATES.extend([_FM_ROOT, _FM_ROOT.parent])

for _base in _CANDIDATES:
    _shared = _base / "_modelevo-shared" / "scripts"
    if _shared.is_dir():
        if str(_shared) not in sys.path:
            sys.path.insert(0, str(_shared))
        break

for _base in _CANDIDATES:
    _eval = _base / "feature-evolution-evaluation" / "scripts"
    if _eval.is_dir():
        if str(_eval) not in sys.path:
            sys.path.insert(0, str(_eval))
        break

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
