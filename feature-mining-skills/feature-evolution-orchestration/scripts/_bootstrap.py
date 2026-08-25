# -*- coding: utf-8 -*-
"""orchestration _bootstrap: 注入 sys.path, 供 import evo_prep / evo_core / config_io.

部署形态: 本 skill(feature-evolution-orchestration/) 与 feature-evolution-evaluation/
平级摆放, 二者可同在一个 skill 集合目录下(如 model-evo 的 feature-mining-skills/).
本脚本在 orchestration/scripts/, 注入:
  1. _modelevo-shared/scripts -> config_io
  2. feature-evolution-evaluation/scripts -> evo_core (确定性核心)
  3. 本 scripts/ -> evo_prep
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPTS_DIR.parent               # feature-evolution-orchestration/
_SKILLS_ROOT = _SKILL_DIR.parent               # skill 集合目录(含各 skill + 可能的 _modelevo-shared)

# 向上逐级找 _modelevo-shared/scripts(兼容 skill 集合目录嵌套在仓库子目录下的形态)
_SHARED_SCRIPTS = None
_base = _SKILLS_ROOT
for _ in range(3):
    _candidate = _base / "_modelevo-shared" / "scripts"
    if _candidate.is_dir():
        _SHARED_SCRIPTS = _candidate
        break
    _base = _base.parent
if _SHARED_SCRIPTS is None:
    raise ImportError("缺少公共代码目录 _modelevo-shared/scripts.")
if str(_SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SHARED_SCRIPTS))

# 确定性核心在 feature-evolution-evaluation skill 内
_EVAL_SCRIPTS = _SKILLS_ROOT / "feature-evolution-evaluation" / "scripts"
if not _EVAL_SCRIPTS.is_dir():
    raise ImportError("缺少 feature-evolution-evaluation/scripts (确定性核心).")
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
