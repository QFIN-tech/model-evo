# -*- coding: utf-8 -*-
"""_bootstrap: 注入 sys.path, 让本 skill 脚本可 import config_io / evo_core / evo_prep.

部署形态: 本 skill(feature-evolution-evaluation/) 与 feature-evolution-orchestration/
平级摆放, 二者可同在一个 skill 集合目录下(如 model-evo 的 feature-mining-skills/).
本脚本在 evaluation/scripts/ 下, 注入:
  - 本 scripts/ (供 import evo_core)
  - feature-evolution-orchestration/scripts/ (供 import evo_prep; probe_roi/champion_residual 需要)
  - _modelevo-shared/scripts (公共 config_io / check_sensitive)
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent  # evaluation/scripts/
_SKILL_DIR = _SCRIPTS_DIR.parent               # feature-evolution-evaluation/
_SKILLS_ROOT = _SKILL_DIR.parent               # skill 集合目录(含各 skill + 可能的 _modelevo-shared)

# 本 skill evaluation scripts/ (供 import evo_core)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# orchestration scripts/ (供 import evo_prep; probe_roi/champion_residual 需要 sample_contract 等)
_ORCH_SCRIPTS = _SKILLS_ROOT / "feature-evolution-orchestration" / "scripts"
if _ORCH_SCRIPTS.is_dir() and str(_ORCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_ORCH_SCRIPTS))

# 向上逐级找 _modelevo-shared/scripts(兼容 skill 集合目录嵌套在仓库子目录下的形态)
_SHARED = None
_base = _SKILLS_ROOT
for _ in range(3):
    _cand = _base / "_modelevo-shared" / "scripts"
    if _cand.is_dir():
        _SHARED = _cand
        break
    _base = _base.parent
if _SHARED is None:
    raise ImportError(
        "缺少公共代码目录 _modelevo-shared/scripts."
    )
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))
