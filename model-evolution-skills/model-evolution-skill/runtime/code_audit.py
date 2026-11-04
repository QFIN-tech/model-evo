"""生成代码自洽性检查(v2.12.2.10):渲染出来的 trial/baseline 代码
里"调用了一个模块级不存在、也没 import 的名字"这类缺陷的静态拦截。

## 为什么需要它

Harness 的 trial / baseline 代码不是手写的,而是 `str.format` 渲染出来的。
模板里少一个 import 或一个 `def`,**模板文件本身照样能 import、能通过单元
测试**(被调用名只出现在字符串里),缺陷只在生成的代码真正跑起来时才炸。

实测事故(example-seq-v3-v2.12.2.9 / run-jt-draw-t7,2026-09-18):
`runtime/baseline.py` 的 `BASELINE_TEMPLATE` 里 dtype 治理段改用了
`safe_to_numeric`(v2.12.2.8 的修复),但那次改动只改了**调用点**,没有把
函数定义随模板一起发出去。于是:

- `preflight` 通过(它只探测运行时依赖,不渲染模板);
- `baseline` 起两个 gen-00 trial,各跑 3 秒后在 `dtype_governance` 段
  `NameError: name 'safe_to_numeric' is not defined`;
- 两个 trial 变成 `invalid/code_error`。按协议 `delete-trial` 只接受
  PROPOSED,失败 trial **不可删除**,只能永久留在 leaderboard 上;
- 用户只能自己写补丁把函数定义塞回模板字符串。

同一批次里 `roc_auc_score` 是同类风险的第二个实例(它只靠 `leak_scan`
函数体内的局部 import 侥幸成立)。

## 检查口径

只查"模块级绑定 + 函数内调用"这一层,不做跨函数的数据流分析:

1. 模块级按序绑定(import / def / class / 赋值)——与 lint 规则 13 同口径;
2. 逐个函数体收集它**读到的所有 Name**(不只调用名——漏掉 `_NUM_STR_RE`
   这种"以参数形式被引用"的常量会留下同类缺口),若该名字既不在模块级
   绑定、也不在该函数自己的局部绑定、也不在 builtins 与豁免表,即判为
   未定义;
3. 语法错误直接报错。

误报控制:只用"函数体局部绑定 + 模块级绑定 + builtins + 小豁免表"四档
豁免,不做更宽的猜测。历史事故(含本文件头部那两例)都能被这四档之外的
名字抓到。
"""

from __future__ import annotations

import ast
import builtins

__all__ = ["check_generated_code", "undefined_module_level_names"]

#: 运行期由解释器/cwd/环境提供、ast.parse 看不见的常见注入名。
_INJECTED_NAMES = frozenset({
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__",
    # trial 模板 marker 段运行期注入的常量(见 assets/lint_trial.py 规则 13)
    "TARGET", "MODEL_TYPE", "TREATMENT", "SEED", "BANNED",
    "REPARSE_SOURCES", "SPLITS", "LABELS_DIR", "REPARSE_DIR",
    "RUN_DIR", "TRIAL_DIR", "SKILL_ROOT", "METRIC_NAME",
})


def _bindings_from_body(body: list[ast.stmt]) -> set[str]:
    """收集一段语句块在**模块级**绑定的名字。"""
    names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                names |= _target_names(tgt)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names |= _target_names(node.target)
        elif isinstance(node, (ast.If, ast.Try, ast.For, ast.While,
                               ast.With, ast.AsyncWith, ast.AsyncFor)):
            # 条件/循环/异常块里的绑定同样落到模块级命名空间
            inner: list[ast.stmt] = []
            inner.extend(getattr(node, "body", []) or [])
            inner.extend(getattr(node, "orelse", []) or [])
            inner.extend(getattr(node, "finalbody", []) or [])
            for handler in getattr(node, "handlers", []) or []:
                inner.extend(handler.body)
                if handler.name:
                    names.add(handler.name)
            names |= _bindings_from_body(inner)
    return names


def _target_names(tgt: ast.AST) -> set[str]:
    if isinstance(tgt, ast.Name):
        return {tgt.id}
    if isinstance(tgt, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in tgt.elts:
            out |= _target_names(elt)
        return out
    if isinstance(tgt, ast.Starred):
        return _target_names(tgt.value)
    return set()


def _local_bindings(fn: ast.AST) -> set[str]:
    """函数体内的局部绑定(参数 / 局部 import / 赋值 / 嵌套 def)。"""
    local: set[str] = set()
    for arg in ast.walk(fn):
        if isinstance(arg, ast.arg):
            local.add(arg.arg)
    for node in ast.walk(fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node is not fn:
            local.add(node.name)
        elif isinstance(node, ast.Lambda):
            for a in node.args.args:
                local.add(a.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                local |= _target_names(tgt)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            local |= _target_names(node.target)
        elif isinstance(node, ast.comprehension):
            local |= _target_names(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            local |= _target_names(node.target)
        elif isinstance(node, ast.withitem):
            if node.optional_vars is not None:
                local |= _target_names(node.optional_vars)
        elif isinstance(node, ast.NamedExpr):
            local |= _target_names(node.target)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            local.add(node.name)
        elif isinstance(node, ast.Global):
            local.update(node.names)
    return local


def undefined_module_level_names(code: str) -> list[str]:
    """扫描生成代码,返回"函数体内引用但无法解析到定义"的名字列表。

    返回空列表表示通过;语法错误由 `ast.parse` 抛出 SyntaxError,
    由 `check_generated_code` 统一转成问题项。
    """
    tree = ast.parse(code)
    module_bindings = (_bindings_from_body(tree.body)
                       | set(dir(builtins)) | set(_INJECTED_NAMES))
    problems: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = _local_bindings(node)
        # 收集所有被**读**到的名字。Store 上下文(赋值的左值)不算依赖,
        # 这里统一走 ast.Name 的 Load 语义 + 嵌套函数的参数名豁免。
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)}
        missing = sorted(u for u in used
                         if u not in module_bindings and u not in local)
        if missing:
            problems.append(f"{node.name}(): 未定义 {missing}")
    return problems


def check_generated_code(code: str, *, warn_only_prefixes: tuple[str, ...] = ()
                         ) -> list[str]:
    """生成代码自洽性检查;返回问题列表(空 = 通过)。

    语法错误也作为问题返回(渲染失败的代码不该落盘)。

    `warn_only_prefixes` 命中的符号由调用方自行决定如何呈现:骨架路径
    (marker 段)里模板提供 `widen_cached` 包装、真实 `_widen` 由 Trial
    主体定义(如 `_widen`/`_widen_feats`),渲染时还看不见——这类名字
    不该硬拦,但保留在返回值里供调用方做提示。
    """
    try:
        problems = undefined_module_level_names(code)
    except SyntaxError as exc:
        return [f"生成代码语法错误:{exc}"]
    if not warn_only_prefixes:
        return problems
    hard: list[str] = []
    for p in problems:
        _names = p.split("未定义", 1)[-1]
        if any(pre in _names for pre in warn_only_prefixes):
            hard.append("[warn] " + p)
        else:
            hard.append(p)
    return hard
