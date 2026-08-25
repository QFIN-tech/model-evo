# -*- coding: utf-8 -*-
"""候选特征代码的安全执行: G1 关卡的确定性部分。

候选契约: 一个 .py 文件, 模块级定义 `def transform(df: pd.DataFrame) -> pd.Series`,
输入帧含 原始特征列 + 语义特征列 + 已接受特征列(以 fid 命名), **不含 label/id/dt**
(从源头防标签泄漏)。

安全模型(子进程隔离执行, 加静态白名单前置):
- 静态: AST 白名单校验(import 仅限 math/numpy/pandas/statistics/re; 禁双下划线名;
  禁 open/eval/exec 等调用; 必须有 transform)
- 动态: 子进程隔离 + 超时; transform 双跑比对(确定性); 输出可数值化/非常数/非全空
"""
from __future__ import annotations

import ast
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# import 白名单(根模块); 候选特征只做数值/向量化计算, 不需要其它任何东西
ALLOWED_IMPORT_ROOTS = {"math", "numpy", "pandas", "statistics", "re"}
# 禁止调用的内建名(网络/文件/动态执行/自省逃逸)
FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "input", "vars", "globals", "locals",
    "breakpoint", "exit", "quit", "help", "__import__",
}


class CandidateCodeError(ValueError):
    """候选代码不满足 G1(静态违规 / 执行失败 / 输出非法)。"""


def check_code_safety(code: str) -> list:
    """AST 静态校验, 返回违规描述列表(空列表 = 通过)。

    检查项: 语法可解析 / import 白名单(禁相对导入) / 双下划线名 / 危险调用 /
    存在模块级 transform 函数。
    """
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ["语法错误: %s (line %s)" % (e.msg, e.lineno)]

    has_transform = False
    for node in tree.body:  # 只认模块级定义
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            has_transform = True
            break
    if not has_transform:
        violations.append("缺少模块级 transform(df) 函数定义")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    violations.append("禁止 import %s(白名单: %s)" % (alias.name, sorted(ALLOWED_IMPORT_ROOTS)))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                violations.append("禁止相对导入(from . import ...)")
            elif node.module and node.module.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
                violations.append("禁止 from %s import(白名单: %s)" % (node.module, sorted(ALLOWED_IMPORT_ROOTS)))
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                violations.append("禁止双下划线名 %s(如 __import__/__builtins__)" % node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                violations.append("禁止双下划线属性 .%s" % node.attr)
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in FORBIDDEN_CALLS:
                violations.append("禁止调用 %s()" % name)
    return violations


# 子进程 runner: 读 df pickle -> exec 候选代码 -> transform 双跑比对 -> 落结果
# 批量模式(可选第 4 参 params_path): 对一组参数各跑一次 transform(常数精调用, 不做双跑)
_RUNNER_SOURCE = '''
# -*- coding: utf-8 -*-
"""candidate runner(由 candidate_exec 在子进程中拉起, 勿手动执行)。"""
import inspect
import json
import pickle
import sys

import numpy as np
import pandas as pd


def _to_numeric_series(out, n):
    """把 transform 输出规整为等长数值 Series; 失败抛 ValueError。"""
    if isinstance(out, pd.DataFrame):
        if out.shape[1] != 1:
            raise ValueError("transform 返回 DataFrame 须恰有 1 列, 实际 %d 列" % out.shape[1])
        out = out.iloc[:, 0]
    if not hasattr(out, "__len__"):
        raise ValueError("transform 须返回 Series/数组, 实际 %r" % type(out))
    s = pd.Series(pd.to_numeric(pd.Series(out), errors="coerce"), dtype=float)
    if len(s) != n:
        raise ValueError("输出长度 %d != 输入行数 %d" % (len(s), n))
    return s


def _call_transform(transform, df, params, accepts_params):
    if params is None or not accepts_params:
        return transform(df)
    return transform(df, params=params)


def _param_spec(ns):
    """提取候选的常数精调约定(PARAM_BOUNDS/DEFAULT_PARAMS), 无效/缺省返回 None。"""
    bounds, defaults = ns.get("PARAM_BOUNDS"), ns.get("DEFAULT_PARAMS")
    if not isinstance(bounds, dict) or not isinstance(defaults, dict):
        return None
    try:
        return {
            "bounds": {str(k): [float(v[0]), float(v[1])] for k, v in bounds.items()},
            "defaults": {str(k): float(v) for k, v in defaults.items()},
        }
    except (TypeError, ValueError, IndexError):
        return None


def main(code_path, df_path, out_path, params_path=None):
    with open(df_path, "rb") as f:
        df = pickle.load(f)
    with open(code_path, "r", encoding="utf-8") as f:
        code = f.read()

    ns = {"__builtins__": __builtins__}
    exec(compile(code, code_path, "exec"), ns)  # noqa: S102 - 父进程已做 AST 白名单
    transform = ns.get("transform")
    if not callable(transform):
        raise ValueError("缺少模块级 transform(df) 函数")

    with open(out_path, "wb") as f:
        if params_path is None:
            s1 = _to_numeric_series(transform(df), len(df))
            s2 = _to_numeric_series(transform(df), len(df))
            deterministic = bool(s1.equals(s2))
            pickle.dump({"series": s1, "deterministic": deterministic}, f)
        else:
            with open(params_path, "r", encoding="utf-8") as pf:
                params_list = json.load(pf)
            accepts_params = "params" in inspect.signature(transform).parameters
            series_list = [
                _to_numeric_series(_call_transform(transform, df, p, accepts_params), len(df))
                for p in params_list
            ]
            pickle.dump(
                {"series_list": series_list, "param_spec": _param_spec(ns),
                 "accepts_params": accepts_params},
                f,
            )


if __name__ == "__main__":
    code_path, df_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    params_path = sys.argv[4] if len(sys.argv) > 4 else None
    try:
        main(code_path, df_path, out_path, params_path)
        print(json.dumps({"ok": True}))
    except Exception as e:  # runner 内任何异常都转为结构化失败
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        sys.exit(1)
'''


def run_transform(code: str, df: pd.DataFrame, timeout_s: float = 60.0) -> dict:
    """子进程执行候选 transform, 返回 {ok, series?, deterministic?, error?}。

    Args:
        code: 候选代码全文
        df: 输入帧(特征列, 不含 label/id/dt)
        timeout_s: 超时秒数, 超时杀进程
    """
    with tempfile.TemporaryDirectory(prefix="evo_cand_") as td:
        td = Path(td)
        code_path = td / "candidate.py"
        df_path = td / "input.pkl"
        out_path = td / "output.pkl"
        runner_path = td / "_runner.py"
        code_path.write_text(code, encoding="utf-8")
        runner_path.write_text(_RUNNER_SOURCE, encoding="utf-8")
        with open(df_path, "wb") as f:
            pickle.dump(df, f)

        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(code_path), str(df_path), str(out_path)],
                capture_output=True,
                timeout=timeout_s,
                cwd=str(td),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "执行超时(>%ss), 疑似死循环或过重计算" % timeout_s}

        try:
            payload = json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "runner 输出不可解析: %s" % proc.stderr.decode("utf-8", "replace")[:500]}
        if not payload.get("ok"):
            return {"ok": False, "error": str(payload.get("error"))}
        with open(out_path, "rb") as f:
            result = pickle.load(f)
        return {
            "ok": True,
            "series": result["series"],
            "deterministic": bool(result["deterministic"]),
        }


def run_transform_batch(code: str, df: pd.DataFrame, params_list: list, timeout_s: float = 60.0) -> dict:
    """子进程批量执行: 对一组参数各跑一次 transform(常数精调用, 不做双跑比对)。

    Args:
        params_list: 每项为一个参数 dict; 项为 None 时按默认(不传 params)调用
    Returns:
        {ok, series_list?, param_spec?, accepts_params?, error?}
        param_spec = {bounds: {name: [lo, hi]}, defaults: {...}}(候选未声明时为 None)
    """
    with tempfile.TemporaryDirectory(prefix="evo_cand_") as td:
        td = Path(td)
        code_path = td / "candidate.py"
        df_path = td / "input.pkl"
        out_path = td / "output.pkl"
        params_path = td / "params.json"
        runner_path = td / "_runner.py"
        code_path.write_text(code, encoding="utf-8")
        runner_path.write_text(_RUNNER_SOURCE, encoding="utf-8")
        with open(df_path, "wb") as f:
            pickle.dump(df, f)
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(list(params_list), f)

        try:
            proc = subprocess.run(
                [sys.executable, str(runner_path), str(code_path), str(df_path), str(out_path), str(params_path)],
                capture_output=True,
                timeout=timeout_s,
                cwd=str(td),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "批量执行超时(>%ss)" % timeout_s}

        try:
            payload = json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "runner 输出不可解析: %s" % proc.stderr.decode("utf-8", "replace")[:500]}
        if not payload.get("ok"):
            return {"ok": False, "error": str(payload.get("error"))}
        with open(out_path, "rb") as f:
            result = pickle.load(f)
        return {
            "ok": True,
            "series_list": result["series_list"],
            "param_spec": result.get("param_spec"),
            "accepts_params": bool(result.get("accepts_params")),
        }


def validate_output(series: pd.Series) -> dict:
    """输出质量检查(G1 后半): 非全 NaN / 非常数 / 数值化后的覆盖率。

    Returns:
        dict: {valid, reason?, coverage} coverage = 非 NaN 比例
    """
    s = pd.Series(series, dtype=float)
    coverage = float(s.notna().mean()) if len(s) else 0.0
    if len(s) == 0:
        return {"valid": False, "reason": "输出为空", "coverage": 0.0}
    if coverage == 0.0:
        return {"valid": False, "reason": "输出全为 NaN(无法数值化)", "coverage": 0.0}
    if s.dropna().nunique() <= 1:
        return {"valid": False, "reason": "输出为常数(无区分度)", "coverage": coverage}
    if not np.isfinite(s.dropna().to_numpy()).all():
        return {"valid": False, "reason": "输出含 inf(须先做截断/平滑)", "coverage": coverage}
    return {"valid": True, "coverage": round(coverage, 6)}


def execute_candidate(code: str, df: pd.DataFrame, timeout_s: float = 60.0) -> dict:
    """G1 完整流程: 静态校验 -> 子进程执行 -> 输出校验。

    Returns:
        dict: {passed, reason?, series?, details}
        details 恒存在(静态违规数/确定性/覆盖率), 供 result.json 留痕
    """
    violations = check_code_safety(code)
    if violations:
        return {"passed": False, "reason": "静态校验违规: %s" % "; ".join(violations[:3]),
                "details": {"static_violations": violations}}

    run = run_transform(code, df, timeout_s=timeout_s)
    if not run["ok"]:
        return {"passed": False, "reason": "执行失败: %s" % run["error"],
                "details": {"static_violations": []}}
    if not run["deterministic"]:
        return {"passed": False, "reason": "transform 双跑结果不一致(须确定性: 禁随机数/当前时间)",
                "details": {"static_violations": [], "deterministic": False}}

    check = validate_output(run["series"])
    details = {"static_violations": [], "deterministic": True, "coverage": check.get("coverage")}
    if not check["valid"]:
        return {"passed": False, "reason": check["reason"], "details": details}
    return {"passed": True, "series": run["series"], "details": details}
