"""测试公共引导(v2.11.9)——把 tests/ 从「依赖本机绝对路径」改成自包含。

**背景**:修复前 6 个测试文件把 `/data/oceanus_ctr/j-wangyaoxuan-jk/modelevo/
model-evo-V2.8/modelevo` 写死在源码里(argparse 默认值、REPO 常量、真实
run 目录),换机器/换目录名即失效;另有测试用固定 `/tmp/_widen_cache_test_run`
共享目录,并发跑两份互踩(已实测复现:54 passed/1 failed)。

**用法**(每个测试文件头三行):

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from _common import SKILL, tmp_run_dir   # noqa: E402

之后 `sys.path` 已就绪,`from runtime.X import ...` 与
`from scripts.modelevo import ...` 都能直接工作,与 skill 目录叫什么
名字、放在哪里无关。

**注意**:不要用 `from modelevo.runtime.X import ...` —— `modelevo` 这个名字
在本 skill 里已被 `scripts/modelevo.py`(CLI 模块)占用,同一 pytest 进程内
两种含义会互相遮蔽('modelevo' is not a package)。运行时包一律写
`runtime.X`。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import types

import os

# v2.18.0:测试默认与真实经验库隔离 —— 服务地址指向不可达端口(视图退化为空、
# 写入跳过),本机缓存写临时目录。需要库的用例自行起嵌入式库 / TestClient。
os.environ.setdefault("MODELEVO_LIBRARY_URI", "http://127.0.0.1:9")
os.environ.setdefault("MODELEVO_LIBRARY_CACHE", tempfile.mkdtemp(prefix="mevo-lib-cache-"))

#: skill 根(本文件在 <skill>/tests/_common.py)
SKILL: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]
if not (SKILL / "runtime").is_dir():
    raise RuntimeError(
        f"tests/_common.py 的上一级不是 skill 根(缺 runtime/):{SKILL}")

#: skill 的父目录;旧写法 `sys.path.insert(0, REPO.parent)` 等价物
SKILL_PARENT: pathlib.Path = SKILL.parent

#: CLI 实现目录(scripts/);`import modelevo` / `import _mvo.*` 靠它
SCRIPTS: pathlib.Path = SKILL / "scripts"

#: 工作区工具目录(assets/);`import lint_trial` 靠它
ASSETS: pathlib.Path = SKILL / "assets"

for _p in (str(SKILL), str(SKILL_PARENT), str(SCRIPTS), str(ASSETS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def tmp_run_dir(prefix: str = "modelevo-test-") -> pathlib.Path:
    """每进程独立的临时工作目录(**不要**再用固定 /tmp 路径)。

    调用方负责清理;直接 `shutil.rmtree(path, ignore_errors=True)` 即可。
    """
    return pathlib.Path(tempfile.mkdtemp(prefix=prefix))


def cli_source() -> str:
    """CLI 全部实现源码(`scripts/modelevo.py` + `scripts/_mvo/*.py`)拼接。

    v2.11.9 起 CLI 拆成 `scripts/modelevo.py`(入口/argparse)+
    `scripts/_mvo/`(实现分片),单个文件不再是全集。**凡"源码里含某
    符号"式断言都必须用本函数**,不要读 `scripts/modelevo.py` 单文件
    ——那会让断言在拆分后静默失效(实测:拆模板后 12 条断言从"通过"
    变"找不到",而代码行为完全正确)。
    """
    parts = [(SCRIPTS / "modelevo.py").read_text(encoding="utf-8")]
    for f in sorted((SCRIPTS / "_mvo").glob("*.py")):
        parts.append(f"\n# ==== {f.name} ====\n")
        parts.append(f.read_text(encoding="utf-8"))
    return "".join(parts)


def trial_body() -> str:
    """`_mvo/templates.TRIAL_BODY` —— trial `code/train.py` 的 body 模板。

    旧测试从 `scripts/modelevo.py` 源码里按 `"\\n    body = " + "'''"`
    锚点做字面量切片;v2.11.9 模板已成模块常量,取用即值。
    """
    from _mvo.templates import TRIAL_BODY
    return TRIAL_BODY


def automl_body() -> str:
    """`_mvo/templates.AUTOML_BODY` —— `--action automl` 的渲染模板。"""
    from _mvo.templates import AUTOML_BODY
    return AUTOML_BODY


__all__ = ["SKILL", "SKILL_PARENT", "SCRIPTS", "ASSETS", "tmp_run_dir",
           "cli_source", "trial_body", "automl_body"]
