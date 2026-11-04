"""modelevo CLI 的实现包(v2.11.9 从单文件 scripts/modelevo.py 拆出)。

**入口仍是 `scripts/modelevo.py`**(`python <skill-root>/scripts/modelevo.py
<command>`),本包只是它的实现分片。拆分的动机:

- 原单文件 6096 行,其中 811 行是嵌在 `cmd_new_trial`(1325 行)函数体
  内的 trial 代码模板、64 行是 automl 模板——文件无法按"命令实现"浏览;
- 嫁接函数此前**读自己的源码**做字面量考古才拿到模板值,模板一改就崩;
- 违反铁律 13 的 agent 被 guard 拦在 `scripts/modelevo.py` 之外,单文件
  使"哪里实现什么"不可导航。

**包名 `_mvo` 带下划线前缀**:它不是公开 API,不参与 argparse,不会被
`python -m` 直接调用;`modelevo.py` 才是。

`sys.path` 引导在本文件完成(skill 根入 path,`runtime.*` 才能 import),
所以子模块不必各自处理——但它们仍需在文件头写 `from .context import ...`
而不是依赖隐式顺序。
"""
from __future__ import annotations

import sys
from pathlib import Path

#: skill 根(本文件在 <skill>/scripts/_mvo/__init__.py)
SKILL_ROOT: Path = Path(__file__).resolve().parents[2]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

__all__ = ["SKILL_ROOT"]
