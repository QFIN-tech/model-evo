"""特征块缓存 + 并行解析(v2.13.5)。

背景(req-t7-rand400w 实测,23.6h wall 拆解):
- seq_d365 的逐行解析是 agent 手写在 train.py 里的纯 Python 循环,单进程
  单核(trial 分到的 17~24 线程只用 1 个),seqd 块 20 min、gapx 30 min、
  stru 47 min;
- **同一个 seqd 块被逐字相同地解析了 9 次**(每个派生 trial 一次),finalize
  又重放 3 次(train+val 24 min、final_test 11 min、shadow 7.5 min);
- 模板里的 `widen_cached` 键含 `TRIAL_DIR.name`,按设计不跨 trial 命中;
  baseline 血缘(`--from-parent` 手写 trial)代码里根本没有它。
  合计 7.2 trial-h(34%),GPU 训练本身只占 7%。

本模块给出两件事,任何血缘的 trial 代码都能 `from runtime.feature_cache
import feature_block` 直接用(harness 模块,不进 trial 源码冻结面):

1. **内容寻址缓存**:键 = 块名 + 内核代码指纹(函数字节码 + 常量 + 递归
   引用的同模块函数/全局常量)+ 解析源文件指纹(大小/mtime/行数/列 schema)
   + split + 输出列 + version。与 trial 身份无关 → 同一内核在不同 trial、
   finalize、shadow 重放之间直接命中;内核任何一处改动 → 指纹变化 → 失配
   重算(宁可重算,不可错用)。落盘 `RUN_DIR/.feature-cache/`。
2. **并行解析**:主进程按 batch 流式读 parquet,fork 出的进程池并行跑内核,
   按 batch 序号回填,行序与 split 严格一致;进程数取 trial 的线程配额。

内核契约(`fn`):`fn(batch: dict[str, list]) -> 2D array-like (n, len(columns))`,
batch 的每个键是一列(默认 python list,见 batch_format)。内核必须是
batch 内容与冻结常量的纯函数(无训练期拟合状态、不读外部文件)——这是
缓存正确性的前提;依赖外部状态的变换用 `version=` 显式换键,或别用本缓存。

v2.17.0(#13 缓存键稳定化,run4 regression.2 实测回写):gen-07 改了
`runtime/__init__.py` 与本文件(数值逐位不变),gen-08 起全部块的
code_fingerprint 漂移(seqd_k0r 14c7… → e3fc…),feature 段重算 1276 s、
交付重训多 10~20 min。键现在**只**由四部分组成:
  1. 内核字节码 + 常量 + 递归引用的(非 harness)函数/全局常量;
  2. 调用参数(块名/split/plane/sources/columns/version/batch_format);
  3. 输入数据**内容**指纹(split manifest 的 sha256,或文件内容 sha256;
     不再用 mtime —— 重新落盘但内容相同的 split 仍命中;行集不同必失配);
  4. 显式 `CACHE_SCHEMA_VERSION`。
harness 自身(`runtime.*`)的函数/类/模块只按身份(名字)记账,不展开代码
也不记 runtime.__version__ —— 其它 runtime 模块改动不再让键失效。指纹算法
(本文件)本身的改动若改变摘要,必须递增 CACHE_SCHEMA_VERSION(由
`fingerprint_probe_digest()` 的金标准测试强制);旧格式缓存目录读取时按
miss 处理,不报错。`predict_invalidation()` 给出"本次 harness 变更是否
会使缓存失效"的预判(acknowledge-harness-change 输出)。
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import os
import re
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

#: v2.17.0:缓存键 schema 版本(显式、唯一的全局失效开关)。落盘布局、键
#: 构成或指纹算法(本文件 _fp_*)任何会改变摘要的修改都必须递增它;其它
#: runtime 模块的改动**不**影响缓存键。v1 = v2.13.5~v2.16.x(键含 mtime 与
#: harness 代码,旧目录读取时自然 miss)。
CACHE_SCHEMA_VERSION = 2
#: 兼容旧名(cmd/测试可能引用)
CACHE_FORMAT = CACHE_SCHEMA_VERSION
CACHE_DIRNAME = ".feature-cache"


def cache_root(run_dir: "Path | str") -> Path:
    """特征块缓存的根目录(v2.17.3):默认仍是 ``<run>/.feature-cache``。

    跨 run 复用的通道 = ``MODELEVO_FEATURE_CACHE_DIR`` 指向一个共享目录
    (例如同一任务多次 run 共用的 ``<share>/feature-cache/<task_id>``)。
    键是**内容寻址**的(内核指纹 + 解析源内容 sha + 参数 + schema),
    与 run 身份无关,共享目录不会串用不同数据的块;写盘用
    ``tmp+os.replace`` 原子替换、并发用 flock,多 run 并发安全。
    相对路径按 run_dir 解析;'off'/空 = 回退默认(不共享)。
    """
    import os as _os

    env = str(_os.environ.get("MODELEVO_FEATURE_CACHE_DIR") or "").strip()
    base = Path(run_dir)
    if not env or env.lower() in ("off", "0", "false", "none"):
        return base / CACHE_DIRNAME
    p = Path(env).expanduser()
    return p if p.is_absolute() else (base / p)
#: 内容哈希的磁盘 memo(按 path+size+mtime 记,避免每进程重读大文件)
_CONTENT_MEMO_FILE = "_content-hashes.json"
_SHA_MEMO: dict[tuple, str] = {}

#: 进程内 memo(同进程 smoke → 全量 → shadow 重放零 IO)
_MEMO: dict[str, pd.DataFrame] = {}
#: fork 子进程继承的内核注册表(避免 pickle 内核函数本身)
_REGISTRY: dict[str, Callable] = {}

_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")


# ---------------------------------------------------------------------------
# 代码指纹
# ---------------------------------------------------------------------------
# 脚本模块(试算代码)的全局常量按顶层赋值源码记账(见 _fp_script_const)。
# 用 thread-local 传根模块:特征块首算按线程配额多进程/多线程并行,模块级
# 变量会互相污染。
_fp_state = threading.local()
#: 脚本源码顶层绑定表缓存:(path, mtime_ns, size) -> {名字: [顶层语句]}
_BINDINGS: dict[tuple, dict[str, list]] = {}


def _is_script_module(mod_name: Any) -> bool:
    return mod_name == "__main__" or (
        mod_name is not None
        and mod_name == getattr(_fp_state, "script_module", None))


def _is_harness_module(mod_name: Any) -> bool:
    """v2.17.0:harness 包(runtime.*)只按身份记账。

    trial 内核引用 harness 的工具函数(safe_to_numeric 等)时,旧实现递归
    展开其字节码、并把 runtime.__version__ 记进摘要 —— 任何 runtime 模块
    改动(哪怕只改注释外的无关行)都让全部键失配。harness 语义变更若确实
    影响内核输出,递增 CACHE_SCHEMA_VERSION(显式、可审计)。
    """
    if not isinstance(mod_name, str):
        return False
    if mod_name == getattr(_fp_state, "script_module", None):
        return False  # 根内核所在模块始终展开(含指纹探针)
    return mod_name == "runtime" or mod_name.startswith("runtime.")


def _toplevel_bindings(path: str) -> dict[str, list] | None:
    """模块顶层(含 if/try/with/for 块,不进 def/class)给每个名字赋值或原地
    修改(X.append(...))的语句,按源码顺序;源码读不到返回 None。"""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        if key in _BINDINGS:
            return _BINDINGS[key]
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None
    out: dict[str, list] = {}

    def visit(stmts) -> None:
        for s in stmts:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
                continue
            if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = s.targets if isinstance(s, ast.Assign) \
                    else [s.target]
                for t in targets:
                    for n in ast.walk(t):   # X = / X[k] = / X.a = / a, X =
                        if isinstance(n, ast.Name):
                            out.setdefault(n.id, []).append(s)
                continue
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) \
                    and isinstance(s.value.func, ast.Attribute) \
                    and isinstance(s.value.func.value, ast.Name):
                out.setdefault(s.value.func.value.id, []).append(s)
                continue
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.stmt):
                    visit([child])
                elif isinstance(child, ast.excepthandler):
                    visit(child.body)

    visit(tree.body)
    _BINDINGS[key] = out
    return out


def _fp_value(h, v, seen: set, depth: int) -> None:
    """常量/全局值的确定性摘要(repr 中的内存地址抹掉,防每次进程都失配)。"""
    if depth > 40:
        h.update(b"<deep>")
        return
    if v is None or isinstance(v, (bool, int, float, complex, str, bytes)):
        h.update(repr(v).encode("utf-8", "surrogatepass"))
        return
    if isinstance(v, types.ModuleType):
        # 模块只记名 + 版本。展开模块 __dict__ 会把整个命名空间(含动态生成的
        # 缓存属性/地址型对象)拉进摘要:同一进程内两次算得相同,跨进程却漂移
        # ——实测内核引用 numpy 时 code_fingerprint 每个进程都不同,特征块缓存
        # 永不命中(DL 的 OOF 状态每 trial 重训 ~65min、父子间列漂移)。
        # v2.17.0:harness 模块(runtime.*)不记版本(升级 harness 不失效)。
        _ver = "" if _is_harness_module(v.__name__) \
            else getattr(v, "__version__", "")
        h.update(f"module:{v.__name__}@{_ver}".encode())
        return
    if isinstance(v, (tuple, list)):
        h.update(b"(" if isinstance(v, tuple) else b"[")
        for x in v:
            _fp_value(h, x, seen, depth + 1)
            h.update(b",")
        h.update(b")")
        return
    if isinstance(v, (set, frozenset)):
        h.update(b"{")
        for r in sorted(repr(x) for x in v):
            h.update(_ADDR_RE.sub("0x?", r).encode("utf-8", "surrogatepass"))
            h.update(b",")
        h.update(b"}")
        return
    if isinstance(v, dict):
        h.update(b"{")
        for k in sorted(v, key=lambda k: repr(k)):
            h.update(repr(k).encode("utf-8", "surrogatepass"))
            h.update(b":")
            _fp_value(h, v[k], seen, depth + 1)
            h.update(b",")
        h.update(b"}")
        return
    if isinstance(v, np.ndarray):
        h.update(f"nd{v.shape}{v.dtype}".encode())
        if v.dtype.hasobject:
            _fp_value(h, v.tolist(), seen, depth + 1)
        else:
            h.update(np.ascontiguousarray(v).tobytes())
        return
    if isinstance(v, (pd.Series, pd.Index)):
        h.update(f"pd{type(v).__name__}{len(v)}".encode())
        _fp_value(h, v.tolist(), seen, depth + 1)
        return
    if isinstance(v, pd.DataFrame):
        h.update(f"pdDF{v.shape}".encode())
        _fp_value(h, [str(c) for c in v.columns], seen, depth + 1)
        try:
            h.update(pd.util.hash_pandas_object(v, index=True).to_numpy()
                     .tobytes())
        except Exception:  # noqa: BLE001  不可哈希单元格 → 逐值
            _fp_value(h, v.to_numpy().tolist(), seen, depth + 1)
        return
    if isinstance(v, types.CodeType):
        _fp_code(h, v, {}, seen, depth + 1)
        return
    if callable(v):
        _fp_callable(h, v, seen, depth + 1)
        return
    # 通用对象(配置对象、查表类实例……):类定义 + 实例状态逐值入摘要。
    # 只记类型名 + repr 会漏掉实例属性(默认 repr 只有地址)→ 两个 trial
    # 的内核引用同类不同参的实例时键相同 = 错用缓存。
    _state = getattr(v, "__dict__", None)
    _slots = getattr(type(v), "__slots__", None)
    if isinstance(_state, dict) or _slots:
        key = ("obj", id(v))
        h.update(f"obj:{type(v).__module__}.{type(v).__qualname__}".encode())
        if key in seen:
            return
        seen.add(key)
        if not _is_harness_module(type(v).__module__):
            _fp_callable(h, type(v), seen, depth + 1)
        if isinstance(_state, dict):
            _fp_value(h, _state, seen, depth + 1)
        if _slots:
            for _sn in ([_slots] if isinstance(_slots, str) else _slots):
                h.update(f".{_sn}=".encode())
                _fp_value(h, getattr(v, _sn, None), seen, depth + 1)
        return
    h.update(type(v).__qualname__.encode())
    try:
        h.update(_ADDR_RE.sub("0x?", repr(v))[:4096].encode(
            "utf-8", "surrogatepass"))
    except Exception:  # noqa: BLE001  repr 失败的对象只记类型
        pass


def _fp_code(h, code: types.CodeType, globs: dict, seen: set,
             depth: int) -> None:
    h.update(code.co_code)
    h.update(repr(code.co_names).encode())
    h.update(repr(code.co_varnames).encode())
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            _fp_code(h, c, globs, seen, depth + 1)
        else:
            _fp_value(h, c, seen, depth + 1)
    names = list(code.co_names)
    # 嵌套代码对象(推导式/lambda/内部函数)引用的全局名也要收
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            names.extend(c.co_names)
    for name in names:
        if name in globs:
            _fp_global(h, name, globs, seen, depth)


def _fp_global(h, name: str, globs: dict, seen: set, depth: int) -> None:
    mod_name = globs.get("__name__")
    v = globs[name]
    tag = f"|g:{name}=".encode()
    if isinstance(v, types.ModuleType):
        _ver = "" if _is_harness_module(v.__name__) \
            else str(getattr(v, "__version__", ""))
        h.update(tag + (v.__name__ + "@" + _ver).encode())
    elif inspect.isfunction(v) or inspect.isclass(v):
        if getattr(v, "__module__", None) == mod_name:
            h.update(tag)
            _fp_callable(h, v, seen, depth + 1)
        else:
            # 第三方/标准库函数:按身份记(版本变化由模块 __version__ 兜)
            h.update(tag + f"{getattr(v, '__module__', '')}."
                     f"{getattr(v, '__qualname__', name)}".encode())
    elif callable(v) and not isinstance(v, (type,)):
        h.update(tag)
        _fp_callable(h, v, seen, depth + 1)
    else:
        h.update(tag)
        if not (_is_script_module(mod_name)
                and _fp_script_const(h, name, globs, seen, depth)):
            _fp_value(h, v, seen, depth + 1)


def _fp_script_const(h, name: str, globs: dict, seen: set,
                     depth: int) -> bool:
    """脚本模块的全局常量按其顶层赋值语句的 AST + 右侧引用的全局名记账,
    不取运行时值:import 期运行态值(_T0 = time.time())、运行中被填充的
    memo dict 跨进程不稳定,赋值源码稳定且改任何字面量都会变。找不到顶层
    赋值(源码不可读、from-import、exec 注入)返回 False → 调用方按值记。"""
    path = globs.get("__file__")
    stmts = ((_toplevel_bindings(path) if path else None) or {}).get(name)
    if not stmts:
        return False
    key = ("g", globs.get("__name__"), name)
    if key in seen:
        return True
    seen.add(key)
    refs: dict[str, None] = {}
    for s in stmts:
        h.update(ast.dump(s).encode("utf-8", "surrogatepass"))
        for n in ast.walk(s):
            if isinstance(n, ast.Name) and n.id != name:
                refs[n.id] = None
    for r in refs:
        if r in globs:
            _fp_global(h, r, globs, seen, depth + 1)
    return True


def _fp_callable(h, fn: Any, seen: set, depth: int) -> None:
    if depth > 40:
        h.update(b"<deep>")
        return
    if isinstance(fn, functools.partial):
        h.update(b"partial:")
        _fp_callable(h, fn.func, seen, depth + 1)
        _fp_value(h, fn.args, seen, depth + 1)
        _fp_value(h, fn.keywords or {}, seen, depth + 1)
        return
    if inspect.ismethod(fn):
        h.update(f"method:{type(fn.__self__).__qualname__}".encode())
        _fp_callable(h, fn.__func__, seen, depth + 1)
        return
    if (inspect.isclass(fn) or inspect.isfunction(fn)) and \
            _is_harness_module(getattr(fn, "__module__", None)):
        # v2.17.0:harness 函数/类按身份记账(见 _is_harness_module)
        h.update(f"harness:{fn.__module__}."
                 f"{getattr(fn, '__qualname__', '')}".encode())
        return
    if inspect.isclass(fn):
        key = ("cls", id(fn))
        h.update(f"class:{fn.__qualname__}".encode())
        if key in seen:
            return
        seen.add(key)
        for attr in sorted(vars(fn)):
            if attr.startswith("__") and attr not in ("__init__", "__call__"):
                continue
            h.update(f".{attr}".encode())
            _fp_value(h, vars(fn)[attr], seen, depth + 1)
        return
    if inspect.isfunction(fn):
        key = ("fn", id(fn))
        h.update(f"fn:{fn.__qualname__}".encode())
        if key in seen:
            return  # 递归/互递归:名字已入摘要,不再展开
        seen.add(key)
        _fp_code(h, fn.__code__, fn.__globals__, seen, depth + 1)
        _fp_value(h, fn.__defaults__, seen, depth + 1)
        _fp_value(h, fn.__kwdefaults__, seen, depth + 1)
        for cell in (fn.__closure__ or ()):
            try:
                _fp_value(h, cell.cell_contents, seen, depth + 1)
            except ValueError:  # 空 cell
                h.update(b"<empty-cell>")
        return
    if callable(fn) and hasattr(fn, "__call__") and \
            inspect.isfunction(getattr(type(fn), "__call__", None)):
        # 可调用实例:类定义 + 实例状态
        _fp_callable(h, type(fn), seen, depth + 1)
        _fp_value(h, getattr(fn, "__dict__", {}), seen, depth + 1)
        return
    # 内建函数等:按身份
    h.update(f"builtin:{getattr(fn, '__module__', '')}."
             f"{getattr(fn, '__qualname__', repr(type(fn)))}".encode())


def code_fingerprint(fn: Callable) -> str:
    """内核的代码指纹:字节码 + 常量 + 默认参数 + 闭包 + 递归引用的
    同模块函数/类与全局常量值。同一源码在不同 trial 文件里 → 同指纹;
    改任何一行逻辑或常量 → 指纹变。Python 版本进入指纹(字节码随版本变)。

    v2.16.0 修复(跨进程确定性):两处曾让同一代码每进程得到不同指纹——
    模块按 __dict__ 展开(见 _fp_value)、试算脚本全局常量按运行时值记账
    (import 期运行态值如 _T0 = time.time())。前者改为按身份,后者改为按
    顶层赋值源码(见 _fp_script_const);辅助函数及其引用仍递归展开。
    """
    h = hashlib.sha256()
    h.update(f"py{sys.version_info[0]}.{sys.version_info[1]}|".encode())
    _fp_state.script_module = getattr(fn, "__module__", None) \
        if inspect.isfunction(fn) else None
    _fp_callable(h, fn, set(), 0)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# v2.17.0:指纹算法探针(金标准)
# ---------------------------------------------------------------------------
#: 探针内核源码:覆盖字节码/常量/同模块辅助函数/全局常量/推导式/默认参数。
#: 在独立的伪模块里编译(不属 runtime.*,不读源文件),摘要只随**指纹算法**
#: 与 Python 版本变化。改动本文件 _fp_* 后若探针摘要变了 → 必须递增
#: CACHE_SCHEMA_VERSION 并更新 _PROBE_GOLDEN(测试强制)。
_PROBE_SRC = '''
_SCALE = 3.0
_TOKENS = ("a", "b")

def _helper(v, k=2):
    return [x * _SCALE + k for x in v]

def kernel(batch):
    xs = _helper(batch["x"])
    return [[x, (1.0 if t in _TOKENS else 0.0)]
            for x, t in zip(xs, batch["t"])]
'''
#: 每个 Python 小版本在当前 CACHE_SCHEMA_VERSION 下的探针摘要
_PROBE_GOLDEN: dict[str, str] = {
    "3.9": "c9dba98c24463e69",
}
_PROBE_FN: Callable | None = None


def _probe_kernel() -> Callable:
    global _PROBE_FN
    if _PROBE_FN is None:
        mod = types.ModuleType("modelevo_fingerprint_probe")
        exec(compile(_PROBE_SRC, "<modelevo-fingerprint-probe>", "exec"),
             mod.__dict__)
        _PROBE_FN = mod.kernel
    return _PROBE_FN


def fingerprint_probe_digest() -> str:
    """当前指纹算法对固定探针内核的摘要(写入缓存 meta,供失效预判比对)。"""
    return code_fingerprint(_probe_kernel())


# ---------------------------------------------------------------------------
# 源文件指纹 / 路径解析
# ---------------------------------------------------------------------------
def _manifest_sha(path: Path, st) -> str | None:
    """split manifest 记录的文件 sha256(前 16 位,与 write_splits 同口径)。

    只在 manifest 晚于(或同时于)文件写入时采信 —— 文件被事后覆写则
    manifest 过期,回退内容哈希。labels/reparse/identity 子目录的文件
    manifest 不记 sha,直接返回 None。
    """
    mpath = path.parent / "manifest.json"
    if not mpath.exists():
        return None
    try:
        if mpath.stat().st_mtime_ns < st.st_mtime_ns:
            return None
        man = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    for entry in (man.get("splits") or {}).values():
        if isinstance(entry, dict) and entry.get("file") == path.name \
                and entry.get("sha256"):
            return str(entry["sha256"])[:16]
    return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(8 << 20), b""):
            h.update(buf)
    return h.hexdigest()[:16]


def content_sha(path: Path, cache_root: Path | None = None) -> tuple[str, str]:
    """文件内容指纹 (sha256 前 16 位, 来源)。来源:memo / manifest / hashed。

    与 write_splits 的 manifest sha256 同口径,所以 manifest 路径与直接哈希
    得到同一个值。按 (路径, size, mtime) 在进程内与 cache_root 下的
    _content-hashes.json 记忆,大文件每个版本只读一次。
    """
    path = Path(path)
    st = path.stat()
    rkey = (str(path.resolve()), int(st.st_size), int(st.st_mtime_ns))
    if rkey in _SHA_MEMO:
        return _SHA_MEMO[rkey], "memo"
    sha = _manifest_sha(path, st)
    if sha:
        _SHA_MEMO[rkey] = sha
        return sha, "manifest"
    memo_path = Path(cache_root) / _CONTENT_MEMO_FILE if cache_root else None
    skey = "|".join(str(x) for x in rkey)
    disk: dict = {}
    if memo_path is not None and memo_path.exists():
        try:
            disk = json.loads(memo_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001  memo 损坏 → 重算
            disk = {}
        if isinstance(disk.get(skey), str):
            _SHA_MEMO[rkey] = disk[skey]
            return disk[skey], "memo"
    sha = _file_sha256(path)
    _SHA_MEMO[rkey] = sha
    if memo_path is not None:
        try:
            memo_path.parent.mkdir(parents=True, exist_ok=True)
            disk[skey] = sha
            tmp = memo_path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(disk, indent=0), encoding="utf-8")
            os.replace(tmp, memo_path)
        except Exception:  # noqa: BLE001  memo 写失败只影响性能
            pass
    return sha, "hashed"


def source_fingerprint(path: Path, columns: Iterable[str],
                       cache_root: Path | None = None) -> dict:
    """解析源文件指纹:内容 sha256 + 行数 + 所用列的 schema(+ 诊断字段)。

    v2.17.0:键只消费 content_sha256 / num_rows / columns;size/mtime_ns
    仅作诊断留痕(旧实现按 mtime 入键,重新落盘内容相同的 split 也全部失配)。
    """
    import pyarrow.parquet as pq

    path = Path(path)
    st = path.stat()
    md = pq.ParquetFile(path).metadata
    schema = md.schema.to_arrow_schema()
    cols = {c: str(schema.field(c).type) for c in columns
            if c in schema.names}
    sha, how = content_sha(path, cache_root)
    return {"path": str(path.resolve()), "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns), "num_rows": int(md.num_rows),
            "columns": cols, "content_sha256": sha, "content_from": how}


def _infer_run_dir(splits_dir: Path) -> Path:
    """<run>/sample-profile/<splits-view> → <run>。"""
    splits_dir = Path(splits_dir).resolve()
    if splits_dir.parent.name == "sample-profile":
        return splits_dir.parent.parent
    return splits_dir.parent


def resolve_source_path(splits_dir: Path, split: str, plane: str,
                        run_dir: Path | None = None) -> Path:
    """split 的解析源路径;视图目录缺该 split(抽样视图不含 final_test)
    时回退基础 v1 splits(与 finalize 的打分回退同口径)。"""
    splits_dir = Path(splits_dir)
    rel = Path("reparse") / f"{split}.parquet" if plane == "reparse" \
        else Path(f"{split}.parquet")
    p = splits_dir / rel
    if p.exists():
        return p
    base = Path(run_dir or _infer_run_dir(splits_dir)) / \
        "sample-profile" / "splits" / rel
    if base.exists():
        return base
    raise FileNotFoundError(f"特征块解析源缺失: {p}(基础面回退 {base} 亦无)")


def block_key(name: str, fn: Callable, *, split: str, columns: list[str],
              sources: list[str], source_fp: dict, plane: str = "reparse",
              version: str = "", batch_format: str = "pylist") -> str:
    """v2.17.0 键:schema 版本 + 调用参数 + 内核指纹 + 输入**内容**指纹。

    source_fp 缺 content_sha256(调用方传旧式指纹)时回退 size/mtime ——
    只为兼容外部调用,feature_block 自身总会给出内容指纹。
    """
    if source_fp.get("content_sha256"):
        data = {k: source_fp[k] for k in
                ("content_sha256", "num_rows", "columns")}
    else:
        data = {k: source_fp.get(k) for k in
                ("size", "mtime_ns", "num_rows", "columns")}
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "name": str(name),
        "split": str(split),
        "plane": plane,
        "sources": list(sources),
        "columns": list(columns),
        "code": code_fingerprint(fn),
        "data": data,
        "version": str(version),
        "batch_format": batch_format,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode())\
        .hexdigest()[:20]


def _prescreen_remap(splits_dir: Path) -> Path:
    """prescreen 沙箱(MODELEVO_PRESCREEN_ROOT)内,把指向沙箱外 run 的
    splits_dir 映射到沙箱内同相对路径(手写 trial 用绝对路径取数时兜底;
    模板 trial 的 SPLITS 本就相对 __file__ 解析到沙箱)。"""
    root = os.environ.get("MODELEVO_PRESCREEN_ROOT")
    if not root:
        return splits_dir
    root_p = Path(root).resolve()
    sd = Path(splits_dir).resolve()
    if sd == root_p or root_p in sd.parents:
        return splits_dir
    parts = sd.parts
    if "sample-profile" in parts:
        i = len(parts) - 1 - parts[::-1].index("sample-profile")
        return root_p.joinpath(*parts[i:])
    raise PermissionError(
        f"prescreen 沙箱内拒绝读取沙箱外的特征块解析源: {splits_dir}")


# ---------------------------------------------------------------------------
# 并行解析
# ---------------------------------------------------------------------------
def _batch_payload(batch, sources: list[str], batch_format: str):
    if batch_format == "pandas":
        return batch.to_pandas()
    if batch_format == "numpy":
        return {c: batch.column(c).to_numpy(zero_copy_only=False)
                for c in sources}
    return {c: batch.column(c).to_pylist() for c in sources}


def _as_block(out: Any, n: int, k: int) -> np.ndarray:
    arr = out.to_numpy(dtype=np.float32) if isinstance(out, pd.DataFrame) \
        else np.asarray(out, dtype=np.float32)
    if arr.ndim == 1 and k == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape != (n, k):
        raise ValueError(
            f"特征块内核返回形状 {arr.shape} != ({n}, {k})"
            "(行数须等于 batch 行数,列数须等于 columns 长度)")
    return arr


def _worker_run(token: str, payload, n: int, k: int) -> np.ndarray:
    """子进程只跑内核。batch 解码(to_pylist)留在主进程:实测把 Arrow
    batch 直接发给子进程再解码反而更慢(44.7 万行 seq 串 27 s vs 21 s),
    且子进程不必碰 pyarrow(fork 安全面更小)。"""
    fn = _REGISTRY[token]
    return _as_block(fn(payload), n, k)


def resolve_jobs(n_jobs: int | None = None) -> int:
    """进程数:MODELEVO_FEATURE_JOBS > 调用方 > OMP_NUM_THREADS(trial 线程配额)。"""
    env = os.environ.get("MODELEVO_FEATURE_JOBS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if n_jobs:
        return max(1, int(n_jobs))
    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS") or 0)
                   or (os.cpu_count() or 2) // 2)
    except ValueError:
        return 1


# ---------------------------------------------------------------------------
# v2.17.2(B1):进程池鲁棒性(run4 opus5.5 trial-006/007:块计算期 cpu 0%
# 空转 20 min 直至被外部 SIGTERM,rc=-15 被判 code_error 并消耗一次 fix_code)
#   - drain 有超时(宽松默认,MODELEVO_FEATURE_DRAIN_TIMEOUT_SEC;0=不限);
#   - worker 异常退出(BrokenProcessPool,多为 OOM-kill)→ 抛带标记的
#     FeatureBlockWorkerLost,scheduler.classify_failure 据标记判 oom
#     (infra 类,不消耗 fix_code 次数);drain 超时 → FeatureBlockDrainTimeout
#     (判 infra_error);两者都先杀掉池内 worker 再抛,不在 shutdown 上挂死;
#   - 给出单 worker 内存估计(worker_mem_gb= 或 MODELEVO_FEATURE_WORKER_GB)
#     时,池并发受有效可用内存约束;没有估计时行为不变。
# ---------------------------------------------------------------------------
#: drain 单个 batch 结果的默认超时(秒)。单 batch(默认 1 万行)正常秒级~
#: 分钟级;取 1 小时只为兜住"worker 卡死/丢失而主进程永远等"的形态
DRAIN_TIMEOUT_SEC_DEFAULT = 3600.0
#: 文件锁(同块并发首算互斥)等待上限(秒);最慢的块实测 ~47 min
LOCK_TIMEOUT_SEC_DEFAULT = 7200.0
LOCK_POLL_SEC = 1.0
#: classify_failure 识别的标记(小写后匹配,见 scheduler.FEATURE_POOL_*)
WORKER_LOST_MARK = "FEATURE_BLOCK_WORKER_LOST"
DRAIN_TIMEOUT_MARK = "FEATURE_BLOCK_DRAIN_TIMEOUT"


class FeatureBlockPoolError(RuntimeError):
    """特征块进程池异常(typed 基类)。"""


class FeatureBlockWorkerLost(FeatureBlockPoolError):
    """池内 worker 异常退出(BrokenProcessPool)——多为内存不足被 OOM-kill。"""


class FeatureBlockDrainTimeout(FeatureBlockPoolError):
    """等待 batch 结果超时(worker 卡死/丢失)。"""


def _env_seconds(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    return None if v <= 0 else v


def resolve_drain_timeout(drain_timeout_sec: float | None = None
                          ) -> float | None:
    """drain 超时:调用方 > MODELEVO_FEATURE_DRAIN_TIMEOUT_SEC > 默认;≤0 = 不限。"""
    if drain_timeout_sec is not None:
        return None if float(drain_timeout_sec) <= 0 \
            else float(drain_timeout_sec)
    return _env_seconds("MODELEVO_FEATURE_DRAIN_TIMEOUT_SEC",
                        DRAIN_TIMEOUT_SEC_DEFAULT)


def cap_jobs_by_memory(jobs: int, worker_mem_gb: float | None = None,
                       *, available_gb: float | None = None
                       ) -> tuple[int, dict | None]:
    """有单 worker 内存估计时,按有效可用内存收紧池并发。

    估计来源:调用方 worker_mem_gb > MODELEVO_FEATURE_WORKER_GB;都没有 →
    原样返回(行为不变)。可用内存取 scheduler.system_memory 的有效可用
    (MemAvailable 与 cgroup 余量取紧者)。返回 (jobs, 留痕 dict|None)。
    """
    if worker_mem_gb is None:
        try:
            worker_mem_gb = float(
                os.environ.get("MODELEVO_FEATURE_WORKER_GB") or 0) or None
        except ValueError:
            worker_mem_gb = None
    if not worker_mem_gb or worker_mem_gb <= 0 or jobs <= 1:
        return jobs, None
    if available_gb is None:
        try:
            from .scheduler import system_memory

            available_gb = float(system_memory().get("available_gb") or 0)
        except Exception:  # noqa: BLE001  探测失败 → 行为不变
            return jobs, None
    cap = max(1, int(float(available_gb) // float(worker_mem_gb)))
    info = {"requested_jobs": int(jobs), "worker_mem_gb": float(worker_mem_gb),
            "available_gb": round(float(available_gb), 1), "cap": cap}
    if cap < jobs:
        info["capped"] = True
        return cap, info
    return jobs, None


def _abort_pool(ex) -> None:
    """杀掉池内 worker 并非阻塞关闭(异常路径;shutdown(wait=True) 会等
    卡死的 worker 一起挂死)。依赖 ProcessPoolExecutor 的私有 _processes,
    取不到时只做非阻塞 shutdown。"""
    try:
        procs = list((getattr(ex, "_processes", None) or {}).values())
    except Exception:  # noqa: BLE001
        procs = []
    for p in procs:
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:  # py<3.9 无 cancel_futures
        ex.shutdown(wait=False)
    except Exception:  # noqa: BLE001
        pass
    for p in procs:
        try:
            p.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass


def parallel_map_batches(fn: Callable, path: Path, sources: list[str],
                         n_out: int, *, n_jobs: int | None = None,
                         batch_rows: int = 10000,
                         batch_format: str = "pylist",
                         log: Callable[[str], None] | None = None,
                         drain_timeout_sec: float | None = None,
                         worker_mem_gb: float | None = None,
                         ) -> np.ndarray:
    """按 batch 流式读 `path` 的 `sources` 列,并行跑 fn,回填 (n, n_out)。

    行序 = 文件行序(按 batch 序号回填,与完成顺序无关)。在途 batch 数
    封顶 n_jobs+1(峰值内存只与 batch_rows 和进程数相关)。n_jobs<=1 或
    平台不支持 fork 时串行。

    v2.17.2:drain 超时(drain_timeout_sec,见 resolve_drain_timeout)、
    worker 丢失 → FeatureBlockWorkerLost、超时 → FeatureBlockDrainTimeout;
    worker_mem_gb 给出时并发受可用内存约束(cap_jobs_by_memory)。
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_total = pf.metadata.num_rows
    out = np.empty((n_total, n_out), dtype=np.float32)
    jobs = resolve_jobs(n_jobs)
    jobs, _mem_cap = cap_jobs_by_memory(jobs, worker_mem_gb)
    if _mem_cap and log:
        log(f"feature block jobs capped by memory: {_mem_cap}")
    drain_timeout = resolve_drain_timeout(drain_timeout_sec)
    batches = pf.iter_batches(batch_size=int(batch_rows), columns=sources)
    t0 = time.time()
    pos = 0
    use_pool = jobs > 1
    ctx = None
    if use_pool:
        import multiprocessing as mp

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            use_pool = False
    if not use_pool:
        for b in batches:
            n = b.num_rows
            out[pos:pos + n] = _as_block(
                fn(_batch_payload(b, sources, batch_format)), n, n_out)
            pos += n
    else:
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures import TimeoutError as _FutTimeout
        from concurrent.futures.process import BrokenProcessPool

        token = f"{os.getpid()}-{id(fn)}-{time.time_ns()}"
        _REGISTRY[token] = fn
        pending: deque = deque()
        ex = ProcessPoolExecutor(max_workers=jobs, mp_context=ctx)
        _ok = False

        def _lost(exc) -> FeatureBlockWorkerLost:
            return FeatureBlockWorkerLost(
                f"[{WORKER_LOST_MARK}] 特征块进程池 worker 异常退出"
                f"(BrokenProcessPool,疑似内存不足被 OOM-kill):{path.name} "
                f"jobs={jobs} batch_rows={batch_rows} 已回填 {pos}/{n_total} "
                "行。降低 MODELEVO_FEATURE_JOBS / batch_rows,或给 "
                f"worker_mem_gb 让并发受内存约束。原始异常: {exc!r}")

        def _drain_one():
            p0, nb, fut = pending.popleft()
            try:
                out[p0:p0 + nb] = fut.result(timeout=drain_timeout)
            except _FutTimeout as exc:
                raise FeatureBlockDrainTimeout(
                    f"[{DRAIN_TIMEOUT_MARK}] 特征块 batch 结果等待超时 "
                    f"{drain_timeout:.0f}s(worker 卡死/丢失):{path.name} "
                    f"rows[{p0}:{p0 + nb}] jobs={jobs}。可用 "
                    "MODELEVO_FEATURE_DRAIN_TIMEOUT_SEC 调整(0=不限)"
                ) from exc

        try:
            for b in batches:
                n = b.num_rows
                payload = _batch_payload(b, sources, batch_format)
                pending.append((pos, n, ex.submit(
                    _worker_run, token, payload, n, n_out)))
                pos += n
                del payload, b
                while len(pending) > jobs:
                    _drain_one()
            while pending:
                _drain_one()
            _ok = True
        except BrokenProcessPool as exc:
            raise _lost(exc) from exc
        finally:
            _REGISTRY.pop(token, None)
            if _ok:
                ex.shutdown(wait=True)
            else:
                _abort_pool(ex)
    if pos != n_total:
        raise RuntimeError(f"特征块解析行数不一致({pos} vs {n_total})")
    if log:
        log(f"feature block parsed: rows={n_total} cols={n_out} "
            f"jobs={jobs if use_pool else 1} {time.time() - t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _log_default(msg: str) -> None:
    print(f"[feature-cache] {msg}", flush=True)


class _FileLock:
    """同一块同时被多个 trial 请求时,后来者等先到者算完直接命中缓存。
    flock 不可用(非 POSIX/网络盘不支持)时退化为无锁(各自计算,原子写
    保证不互相写坏)。

    v2.17.2(B1):有界等待——非阻塞 flock 轮询(LOCK_POLL_SEC),等满
    timeout_sec(缺省 MODELEVO_FEATURE_LOCK_TIMEOUT_SEC → 7200s;≤0 = 不限)
    仍拿不到则告警并退化为无锁计算(与 flock 不可用同语义),不再无限阻塞。
    持锁者把 pid 写进锁文件,等待/超时日志带持有者 pid 便于定位。
    """

    def __init__(self, path: Path, *, timeout_sec: float | None = None,
                 poll_sec: float | None = None,
                 log: Callable[[str], None] | None = None):
        self.path = path
        self.fh = None
        self.timeout_sec = timeout_sec if timeout_sec is not None else \
            _env_seconds("MODELEVO_FEATURE_LOCK_TIMEOUT_SEC",
                         LOCK_TIMEOUT_SEC_DEFAULT)
        if self.timeout_sec is not None and self.timeout_sec <= 0:
            self.timeout_sec = None
        self.poll_sec = float(poll_sec or LOCK_POLL_SEC)
        self.log = log
        self.timed_out = False
        self.waited_sec = 0.0

    def holder_pid(self) -> int | None:
        try:
            raw = Path(self.path).read_text(encoding="utf-8").strip()
            return int(raw.splitlines()[-1]) if raw else None
        except (OSError, ValueError, IndexError):
            return None

    def __enter__(self):
        try:
            import errno
            import fcntl

            self.fh = open(self.path, "a+")
        except Exception:  # noqa: BLE001
            if self.fh is not None:
                self.fh.close()
            self.fh = None
            return self
        t0 = time.time()
        next_note = t0 + 60.0
        while True:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES,
                                     errno.EWOULDBLOCK):
                    # flock 不受支持:旧语义,退化为无锁
                    self.fh.close()
                    self.fh = None
                    return self
            now = time.time()
            self.waited_sec = now - t0
            if self.timeout_sec is not None and \
                    self.waited_sec >= self.timeout_sec:
                if self.log:
                    self.log(f"特征块锁 {self.path.name} 等待 "
                             f"{self.waited_sec:.0f}s 超时(持有者 pid="
                             f"{self.holder_pid()}),退化为无锁计算"
                             "(原子写保证不互相写坏)")
                self.fh.close()
                self.fh = None
                self.timed_out = True
                return self
            if self.log and now >= next_note:
                self.log(f"等待特征块锁 {self.path.name}(持有者 pid="
                         f"{self.holder_pid()},已等 {self.waited_sec:.0f}s)")
                next_note = now + 60.0
            time.sleep(self.poll_sec)
        try:  # 持锁留痕:pid(a+ 模式 truncate 后追加即从头写)
            self.fh.truncate(0)
            self.fh.write(f"{os.getpid()}\n")
            self.fh.flush()
        except OSError:
            pass
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            try:
                import fcntl

                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.fh.close()
        return False


def feature_block(name: str, fn: Callable, *, split: str,
                  columns: list[str], sources: list[str],
                  splits_dir: str | Path, run_dir: str | Path | None = None,
                  plane: str = "reparse", n_jobs: int | None = None,
                  batch_rows: int = 10000, batch_format: str = "pylist",
                  version: str = "", cache: bool = True,
                  log: Callable[[str], None] | None = _log_default,
                  drain_timeout_sec: float | None = None,
                  worker_mem_gb: float | None = None,
                  ) -> pd.DataFrame:
    """计算(或命中缓存)一个 split 的特征块,返回 float32 DataFrame
    (列 = columns,RangeIndex,行序 = split 行序)。

    参数:
      name        块名(人读;同名不同内核会得到不同键,不会串)
      fn          内核,见模块 docstring 的契约
      split       train / validation / shadow_test / final_test
      columns     输出列名(应带解析前缀,如 seqd_*)
      sources     解析源列(plane=reparse 时须是 reparse 面里的列)
      splits_dir  trial 的 SPLITS 常量
      plane       "reparse"(默认,解析源面)| "features"(特征面)
      n_jobs      进程数(缺省取 OMP_NUM_THREADS = trial 线程配额)
      version     内核依赖指纹抓不到的外部状态时手动换键
      cache       False = 只算不存(诊断);环境变量 MODELEVO_FEATURE_CACHE=0 同效
      drain_timeout_sec  v2.17.2:进程池单 batch 结果等待上限(缺省见
                  resolve_drain_timeout;≤0 = 不限)
      worker_mem_gb      v2.17.2:单 worker 内存估计(GB);给出时池并发受
                  有效可用内存约束(缺省读 MODELEVO_FEATURE_WORKER_GB;都无则不约束)
    """
    columns = list(columns)
    sources = list(sources)
    if not columns:
        raise ValueError("feature_block: columns 为空")
    if len(set(columns)) != len(columns):
        raise ValueError("feature_block: columns 有重名")
    splits_dir = _prescreen_remap(Path(splits_dir))
    run_dir = Path(run_dir) if run_dir else _infer_run_dir(splits_dir)
    if os.environ.get("MODELEVO_PRESCREEN_ROOT"):
        run_dir = _infer_run_dir(splits_dir)
    path = resolve_source_path(splits_dir, split, plane, run_dir)
    src_fp = source_fingerprint(path, sources,
                                cache_root=cache_root(run_dir))
    missing = [c for c in sources if c not in src_fp["columns"]]
    if missing:
        raise KeyError(f"feature_block: 解析源 {path} 缺列 {missing}")
    use_cache = cache and os.environ.get("MODELEVO_FEATURE_CACHE", "1") != "0"
    key = block_key(name, fn, split=split, columns=columns, sources=sources,
                    source_fp=src_fp, plane=plane, version=version,
                    batch_format=batch_format)
    cdir = cache_root(run_dir) / f"{_safe(name)}-{key}"
    fdata = cdir / f"{split}.parquet"
    fmeta = cdir / f"{split}.meta.json"
    _hook_info = {"name": name, "fn": fn, "split": split, "key": key,
                  "columns": columns, "sources": sources, "plane": plane,
                  "batch_format": batch_format, "source_path": str(path),
                  "cache_file": str(fdata), "version": version}
    if use_cache and key in _MEMO:
        if log:
            log(f"hit(memo) {name}/{split} key={key}")
        _fire_hook(_hook_info, "memo")
        return _MEMO[key].copy(deep=False)

    def _read_hit() -> pd.DataFrame | None:
        if not (fdata.exists() and fmeta.exists()):
            return None
        try:
            meta = json.loads(fmeta.read_text(encoding="utf-8"))
            # v2.17.0:旧格式(无 cache_schema_version)/版本不符 → miss,不报错
            if meta.get("key") != key or \
                    meta.get("cache_schema_version") != \
                    CACHE_SCHEMA_VERSION or \
                    meta.get("n_rows") != src_fp["num_rows"]:
                return None
            df = pd.read_parquet(fdata)
            if list(df.columns) != columns or len(df) != src_fp["num_rows"]:
                return None
            return df
        except Exception as exc:  # noqa: BLE001  缓存损坏 → 重算
            if log:
                log(f"缓存读取失败,重算: {exc}")
            return None

    if use_cache:
        cdir.mkdir(parents=True, exist_ok=True)
        with _FileLock(cdir / f"{split}.lock", log=log):
            df = _read_hit()
            if df is not None:
                if log:
                    log(f"hit(disk) {name}/{split} key={key} "
                        f"rows={len(df)} cols={df.shape[1]}")
                _MEMO[key] = df
                _fire_hook(_hook_info, "disk")
                return df.copy(deep=False)
            df, elapsed, jobs = _compute(fn, path, sources, columns, n_jobs,
                                         batch_rows, batch_format, log,
                                         drain_timeout_sec=drain_timeout_sec,
                                         worker_mem_gb=worker_mem_gb)
            try:
                tmp = fdata.with_suffix(f".tmp{os.getpid()}.parquet")
                df.to_parquet(tmp, index=False, compression="zstd")
                os.replace(tmp, fdata)
                meta = {
                    "key": key, "name": name, "split": split,
                    "plane": plane, "sources": sources,
                    "columns": columns, "n_rows": int(len(df)),
                    "code_fingerprint": code_fingerprint(fn),
                    "fn": f"{getattr(fn, '__module__', '')}."
                          f"{getattr(fn, '__qualname__', repr(fn))}",
                    "source": src_fp, "version": version,
                    "compute_sec": round(elapsed, 2), "n_jobs": jobs,
                    "writer": _writer_tag(),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "format": CACHE_FORMAT,
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "probe_digest": fingerprint_probe_digest(),
                    "batch_format": batch_format,
                }
                tmpm = fmeta.with_suffix(f".tmp{os.getpid()}")
                tmpm.write_text(json.dumps(meta, ensure_ascii=False,
                                           indent=2), encoding="utf-8")
                os.replace(tmpm, fmeta)
                if log:
                    log(f"written {name}/{split} key={key} -> {fdata}")
            except Exception as exc:  # noqa: BLE001  写盘失败不阻塞 trial
                if log:
                    log(f"缓存写盘失败(不影响本轮): {exc}")
            _MEMO[key] = df
            _fire_hook(_hook_info, "computed")
            return df.copy(deep=False)
    df, _, _ = _compute(fn, path, sources, columns, n_jobs, batch_rows,
                        batch_format, log,
                        drain_timeout_sec=drain_timeout_sec,
                        worker_mem_gb=worker_mem_gb)
    _hook_info["cache_file"] = None
    _fire_hook(_hook_info, "computed_nocache", frame=df)
    return df


#: v2.17.0:feature_block 调用观察钩子(write-splits-view --from-feature-block
#: 用它拿到块的内核/缓存文件,不改 trial 代码)。None = 关闭。
_BLOCK_HOOK: Callable[[dict], None] | None = None


def _fire_hook(info: dict, how: str, frame: pd.DataFrame | None = None) -> None:
    hook = _BLOCK_HOOK
    if hook is None:
        return
    try:
        hook(dict(info, how=how, frame=frame))
    except Exception:  # noqa: BLE001  观察钩子绝不影响主流程
        pass


def _compute(fn, path, sources, columns, n_jobs, batch_rows, batch_format,
             log, *, drain_timeout_sec=None, worker_mem_gb=None):
    t0 = time.time()
    jobs = resolve_jobs(n_jobs)
    arr = parallel_map_batches(fn, path, sources, len(columns),
                               n_jobs=jobs, batch_rows=batch_rows,
                               batch_format=batch_format, log=log,
                               drain_timeout_sec=drain_timeout_sec,
                               worker_mem_gb=worker_mem_gb)
    return pd.DataFrame(arr, columns=columns), time.time() - t0, jobs


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))[:40] or "block"


def _writer_tag() -> str:
    """缓存写入者留痕:trial 进程是 code/train.py 的路径,finalize 是 CLI。"""
    if os.environ.get("MODELEVO_TRIAL_ID"):
        return os.environ["MODELEVO_TRIAL_ID"]
    if sys.argv and sys.argv[0]:
        return str(Path(sys.argv[0]).resolve())
    return ""


def align_rows(block: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """把整 split 的特征块按基础帧的行位置对齐(子样本/quick 轮)。

    行数相等:直接换成 X 的 index;否则要求 X.index 是合法行位置
    (0..len(block)-1)——不合法宁可报错,绝不静默错位拼接(那会把
    A 用户的事件接到 B 用户身上)。
    """
    n = len(X)
    if n == len(block):
        out = block.copy(deep=False)
        out.index = X.index
        return out
    pos = np.asarray(X.index)
    if (pos.dtype.kind not in "iu" or pos.size == 0
            or pos.min() < 0 or pos.max() >= len(block)):
        raise ValueError(
            f"特征块与基础帧行数不一致({len(block)} vs {n})且 X.index "
            "不是合法行位置,无法安全对齐")
    out = block.iloc[pos].copy(deep=False)
    out.index = X.index
    return out


def cache_inventory(run_dir: str | Path) -> list[dict]:
    """列出 run 的特征块缓存(CLI `feature-cache` 与报告消费)。"""
    root = cache_root(run_dir)
    out = []
    if not root.exists():
        return out
    for meta_p in sorted(root.glob("*/*.meta.json")):
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        data_p = meta_p.with_name(meta_p.name.replace(".meta.json",
                                                      ".parquet"))
        meta["bytes"] = data_p.stat().st_size if data_p.exists() else 0
        meta["dir"] = meta_p.parent.name
        out.append(meta)
    return out


def predict_invalidation(run_dir: str | Path,
                         changed_files: Iterable[str] | None = None) -> dict:
    """v2.17.0:预判"当前 harness 下,已有特征块缓存是否仍会命中"。

    判据(不 import trial 代码、不读数据,毫秒级):
      - 条目的 cache_schema_version != 当前 → 失效(旧格式 v1 条目全部失效);
      - 条目记录的 probe_digest != 当前指纹算法的探针摘要 → 失效
        (指纹算法被改却没递增 CACHE_SCHEMA_VERSION —— 应修 harness);
      - 条目的解析源文件内容指纹(manifest/内存/磁盘 memo 可得时)已变 →
        该条目失效(数据重写;与 harness 无关,单列 data_changed)。
    `changed_files` 只用于说明(如 ["feature_cache.py"]):除 schema 版本与
    指纹算法外,任何 runtime 模块改动都不改变缓存键。
    """
    entries = cache_inventory(run_dir)
    cur_probe = fingerprint_probe_digest()
    changed = sorted(set(changed_files or []))
    by_reason: dict[str, int] = {}
    stale_dirs: list[str] = []
    valid = 0
    for m in entries:
        reason = None
        ver = m.get("cache_schema_version")
        if ver is None:
            reason = "legacy_format_v1"
        elif ver != CACHE_SCHEMA_VERSION:
            reason = "schema_version_changed"
        elif m.get("probe_digest") and m["probe_digest"] != cur_probe:
            reason = "fingerprint_algorithm_changed"
        else:
            src = m.get("source") or {}
            sp = Path(src.get("path") or "")
            want = src.get("content_sha256")
            if want and sp.exists():
                st = sp.stat()
                rkey = (str(sp.resolve()), int(st.st_size),
                        int(st.st_mtime_ns))
                got = _SHA_MEMO.get(rkey) or _manifest_sha(sp, st)
                if got is None:
                    memo = cache_root(run_dir) / _CONTENT_MEMO_FILE
                    try:
                        got = (json.loads(memo.read_text(encoding="utf-8"))
                               or {}).get("|".join(str(x) for x in rkey))
                    except Exception:  # noqa: BLE001
                        got = None
                if got is not None and got != want:
                    reason = "data_changed"
            elif want and not sp.exists():
                reason = "source_missing"
        if reason:
            m["_stale_reason"] = reason
        else:
            valid += 1
    # 旧格式条目:同 (块名, split) 已有当前 schema 的有效条目 = 已被取代
    # (不再计入失效);否则升级后首次使用必然重算(计入 harness 失效)
    live = {(m.get("name"), m.get("split")) for m in entries
            if "_stale_reason" not in m}
    for m in entries:
        reason = m.pop("_stale_reason", None)
        if reason is None:
            continue
        if reason == "legacy_format_v1":
            reason = ("legacy_superseded"
                      if (m.get("name"), m.get("split")) in live
                      else "legacy_not_recomputed")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if reason != "legacy_superseded":
            stale_dirs.append(f"{m.get('dir')}/{m.get('split')}")
    harness_stale = sum(v for k, v in by_reason.items()
                        if k in ("schema_version_changed",
                                 "fingerprint_algorithm_changed",
                                 "legacy_not_recomputed"))
    if not entries:
        summary = "无特征块缓存,无失效影响"
    elif harness_stale:
        summary = (f"当前 harness 下 {harness_stale}/{len(entries)} 个缓存条目"
                   "不会命中(缓存 schema 版本 / 指纹算法变化,或 v2.16 及更早的"
                   "旧格式条目尚未按新键重算),对应特征块首次使用时重算一次")
    else:
        summary = (f"本次 harness 变更不影响缓存键:{valid}/{len(entries)} 个"
                   "条目仍会命中")
        if by_reason:
            summary += ";另有 " + ", ".join(
                f"{k}={v}" for k, v in sorted(by_reason.items())) + \
                "(与本次 harness 变更无关)"
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "probe_digest": cur_probe,
        "n_entries": len(entries),
        "n_valid": valid,
        "n_stale": len(entries) - valid,
        "stale_by_reason": by_reason,
        "will_invalidate": bool(harness_stale),
        "feature_cache_py_changed": "feature_cache.py" in changed,
        "changed_files": changed,
        "stale_examples": stale_dirs[:10],
        "summary": summary,
    }
