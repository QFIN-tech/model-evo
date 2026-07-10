# -*- coding: utf-8 -*-
"""把 model-skills/_modelevo-shared/scripts 与 model-training/scripts 注入 sys.path,
让本 skill 脚本可直接 import config_io / run_layout / write_*_stage 等公共模块。

_modelevo-shared/ 由 install.sh 从仓库根复制到 model-skills/ 下。
"""
import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).resolve().parents[1]                        # classification-model-tuning
_SKILLS_ROOT = _SKILL_DIR.parent                                        # model-skills/
_SHARED_SCRIPTS = _SKILLS_ROOT / "_modelevo-shared" / "scripts"
_MT_SCRIPTS = _SKILLS_ROOT / "classification-model-training" / "scripts"

if not _SHARED_SCRIPTS.is_dir():
    raise ImportError(
        "缺少公共代码目录: %s。请在仓库根目录运行 `bash install.sh` 把 _modelevo-shared/ "
        "复制到 model-skills/ 下。" % _SHARED_SCRIPTS
    )

for p in (_SHARED_SCRIPTS, _MT_SCRIPTS):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
