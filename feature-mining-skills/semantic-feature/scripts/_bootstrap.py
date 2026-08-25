# -*- coding: utf-8 -*-
"""把公共代码目录注入 sys.path, 让本 skill 脚本可直接 import config_io / evo_core。

兼容两种部署形态(与 feature-evolution-evaluation 的 _bootstrap 同款):
  ① 仓库源:   model-evo/feature-mining-skills/<skill>/scripts/_bootstrap.py
  ② 安装后:   SKILL_ROOT/<skill>/scripts/_bootstrap.py
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SKILL_DIR = _SCRIPTS_DIR.parent
_FM_ROOT = _SKILL_DIR.parent  # feature-mining-skills/ 或 SKILL_ROOT

_SHARED_SCRIPTS = None
for _base in (_FM_ROOT, _FM_ROOT.parent):
    _candidate = _base / "_modelevo-shared" / "scripts"
    if _candidate.is_dir():
        _SHARED_SCRIPTS = _candidate
        break
if _SHARED_SCRIPTS is None:
    raise ImportError(
        "缺少公共代码目录 _modelevo-shared/scripts。仓库源形态下它应在仓库根; "
        "安装形态下应与各 skill 平级(参考主 README 的安装说明)。"
    )
if str(_SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SHARED_SCRIPTS))

# 确定性核心(evo_core: metrics/paths/state_io)在 evaluation skill 内, 整体安装时才可用
_EVAL_SCRIPTS = _FM_ROOT / "feature-evolution-evaluation" / "scripts"
if not _EVAL_SCRIPTS.is_dir():
    raise ImportError(
        "缺少 feature-evolution-evaluation skill(%s)。"
        "feature-mining-skills 下的 skill 需整体安装, 不要只拷贝单个目录。" % _EVAL_SCRIPTS
    )
if str(_EVAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_EVAL_SCRIPTS))

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
