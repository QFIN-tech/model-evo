# -*- coding: utf-8 -*-
"""把 model-skills/_modelevo-shared/scripts 注入 sys.path,让本 skill 脚本可直接 import config_io。

_modelevo-shared/ 由 install.sh 从仓库根复制到 model-skills/ 下。
"""
import sys
from pathlib import Path

_SKILLS_ROOT = Path(__file__).resolve().parents[2]
_SHARED_SCRIPTS = _SKILLS_ROOT / "_modelevo-shared" / "scripts"
if not _SHARED_SCRIPTS.is_dir():
    raise ImportError(
        "缺少公共代码目录: %s。请在仓库根目录运行 `bash install.sh` 把 _modelevo-shared/ "
        "复制到 model-skills/ 下。" % _SHARED_SCRIPTS
    )
if str(_SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SHARED_SCRIPTS))
