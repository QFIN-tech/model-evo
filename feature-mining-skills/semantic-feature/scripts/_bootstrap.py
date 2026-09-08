# -*- coding: utf-8 -*-
"""把公共代码目录注入 sys.path, 让本 skill 脚本可直接 import config_io / evo_core。

部署形态(按优先级):
  ① 环境变量 MODELEVO_SKILLS_ROOT 指向 skills 根(含 _modelevo-shared/ 与各 skill)
  ② 仓库源:   <repo>/feature-mining-skills/<skill>/scripts/_bootstrap.py
  ③ 安装后:   SKILL_ROOT/<skill>/scripts/_bootstrap.py(skills 根与各 skill 平级)
"""
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPTS_DIR.parent
_FM_ROOT = _SKILL_DIR.parent  # feature-mining-skills/ 或 SKILL_ROOT

_CANDIDATE_ROOTS = []
_env_root = os.environ.get("MODELEVO_SKILLS_ROOT")
if _env_root:
    _CANDIDATE_ROOTS.append(Path(_env_root))
_CANDIDATE_ROOTS.extend([_FM_ROOT, _FM_ROOT.parent])

_SHARED_SCRIPTS = None
for _base in _CANDIDATE_ROOTS:
    _candidate = _base / "_modelevo-shared" / "scripts"
    if _candidate.is_dir():
        _SHARED_SCRIPTS = _candidate
        break
if _SHARED_SCRIPTS is None:
    raise ImportError(
        "缺少公共代码目录 _modelevo-shared/scripts。仓库源形态下它应在仓库根; "
        "安装形态下应与各 skill 平级; 单目录拷贝部署请设 MODELEVO_SKILLS_ROOT 指向 skills 根。"
    )
if str(_SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SHARED_SCRIPTS))

# 确定性核心(evo_core: metrics/paths/state_io)在 evaluation skill 内, 整体安装时才可用
_EVAL_SCRIPTS = None
for _base in _CANDIDATE_ROOTS:
    _candidate = _base / "feature-evolution-evaluation" / "scripts"
    if _candidate.is_dir():
        _EVAL_SCRIPTS = _candidate
        break
if _EVAL_SCRIPTS is None:
    raise ImportError(
        "缺少 feature-evolution-evaluation skill。"
        "feature-mining-skills 下的 skill 需整体安装; 单目录拷贝部署请设 MODELEVO_SKILLS_ROOT。"
    )
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
