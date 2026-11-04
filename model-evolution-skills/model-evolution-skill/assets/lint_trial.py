"""Trial 代码静态校验脚本。

用法:
    python tools/lint_trial.py <run_dir> <trial_dir_or_id>

退出码:
    0 — 通过
    1 — 违规(打印原因)
    2 — 用法错误
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# 规则 1: algorithm 字段一致性
# ---------------------------------------------------------------------------

ALGO_REQUIRED_IMPORTS = {
    "lightgbm": ["lightgbm"],
    "xgboost": ["xgboost"],
    "catboost": ["catboost"],
    "mlp": ["torch"],
    "nn": ["torch"],
    "torch": ["torch"],
    "logistic": ["sklearn.linear_model"],
    "lr": ["sklearn.linear_model"],
    "rf": ["sklearn.ensemble"],
    "randomforest": ["sklearn.ensemble"],
}

GBM_LIBS = {"lightgbm", "xgboost", "catboost"}


def extract_algorithm_from_trial_id(trial_id: str) -> str | None:
    """trial-{NNN}-g{gen:02d}-{algorithm}-{hhmm} -> algorithm"""
    m = re.match(r"trial-\d+-g\d+-([a-zA-Z_]+)-\d+$", trial_id)
    return m.group(1).lower() if m else None


def scan_imports(code: str) -> set[str]:
    """提取 code 中所有 import 的顶级包名。"""
    pkgs = set()
    for m in re.finditer(r"^\s*import\s+([a-zA-Z_][\w.]*)", code, re.M):
        pkgs.add(m.group(1).split(".")[0])
    for m in re.finditer(r"^\s*from\s+([a-zA-Z_][\w.]*)\s+import", code, re.M):
        pkgs.add(m.group(1).split(".")[0])
    return pkgs


def check_algorithm_consistency(trial_id: str, code: str) -> list[str]:
    """规则 1: trial_id 的 algorithm 字段必须与 code 实际 import 一致。"""
    errors = []
    declared = extract_algorithm_from_trial_id(trial_id)
    if not declared:
        return []  # trial_id 格式异常,跳过
    if declared == "ensemble":
        return []  # ensemble 是特殊值,允许多模型库
    pkgs = scan_imports(code)
    required = ALGO_REQUIRED_IMPORTS.get(declared)
    if required is None:
        # 未识别的 algorithm 字段(可能是新类型)
        return []
    for req in required:
        if req not in pkgs:
            errors.append(
                f"规则 1 违反: trial_id 声明 algorithm={declared},"
                f"但 code/train.py 未 import {req}"
            )
    return errors


# ---------------------------------------------------------------------------
# 规则 2: 未经人工批准的融合拓扑时禁止单 trial ensemble
# (v2.16.0:放行条件由 runtime/topology.py 的拓扑门给出,spec.ensemble_enabled
#  不再生效;级联与融合的主检测也在拓扑门,本规则保留为正则补充)
# ---------------------------------------------------------------------------

def _fit_count_by_lib(code: str) -> dict[str, int]:
    """统计每个 GBM 库被 fit 的次数(粗略)。"""
    counts = {}
    for lib in GBM_LIBS:
        # 匹配 LGBMClassifier/Regressor、XGBClassifier/Regressor 等
        # 实例化次数(Regressor 与 Classifier 同口径,回归任务同样适用)
        patterns = {
            "lightgbm": r"LGBM(?:Classifier|Regressor|Model)\s*\(",
            "xgboost": r"XGB(?:Classifier|Regressor|Model)\s*\(",
            "catboost": r"CatBoost(?:Classifier|Regressor|Model)\s*\(",
        }
        if lib in patterns:
            counts[lib] = len(re.findall(patterns[lib], code))
    return counts


def check_no_ensemble(code: str) -> list[str]:
    """规则 2: 检测单 trial 内的 ensemble 行为。"""
    errors = []

    # 1. 多 GBM 库并存(v19.4 修正:探测型 import 不算使用——
    #    模板 _lgbm_cuda_ok/_xgb_cuda_built 以 64x4 小样本探测 CUDA
    #    编译是否可用,别名带 _probe 后缀且只出现在 try 块内,
    #    不构成"同 trial 双库训练"。此前 baseline 渲染的 xgboost
    #    trial 因自带 lgbm 探测被误拒(baseline 路径不经 lint 硬闸
    #    所以当时未暴露;--from-parent 复制该血缘后才炸)。
    imported_gbm = GBM_LIBS & scan_imports(code)
    if len(imported_gbm) >= 2:
        import ast as _ast

        try:
            _tree = _ast.parse(code)
        except SyntaxError:
            _tree = None
        _probe_only = set()
        if _tree is not None:
            for _node in _ast.walk(_tree):
                if isinstance(_node, _ast.Try):
                    for _sub in _ast.walk(_node):
                        if isinstance(_sub, _ast.Import):
                            for _al in _sub.names:
                                _root = _al.name.split(".")[0]
                                if _root in GBM_LIBS and (
                                        (_al.asname and
                                         "_probe" in _al.asname)
                                        or _al.asname is None):
                                    _probe_only.add(_root)
                        elif isinstance(_sub, _ast.ImportFrom):
                            _root = (_sub.module or "").split(".")[0]
                            if _root in GBM_LIBS and (
                                    _sub.names and _sub.names[0].asname
                                    and "_probe" in _sub.names[0].asname):
                                _probe_only.add(_root)
        _real = imported_gbm - _probe_only
        if len(_real) >= 2:
            errors.append(
                f"规则 2 违反: 同 trial 使用多个 GBM 库 ({sorted(_real)}),"
                f"疑似多模型融合"
            )
        elif len(imported_gbm) >= 2 and not _probe_only:
            # 非 try 块/无 _probe 别名的第二库 import:保守仍报
            errors.append(
                f"规则 2 违反: 同 trial 多个 GBM 库 import "
                f"({sorted(imported_gbm)});若属 CUDA 编译探测,请把探测"
                f" import 放入 try 块并使用带 _probe 后缀的别名"
                f"(模板 _lgbm_cuda_ok 同款)"
            )

    # 2. rank 平均 / 概率平均
    if re.search(r"rankdata\s*\(", code):
        errors.append(
            "规则 2 违反: 使用 scipy.stats.rankdata 做 rank-ensemble"
        )
    if re.search(r"np\.mean\s*\(\s*scores\b", code):
        errors.append(
            "规则 2 违反: 对 scores 列表取 np.mean,疑似多模型/多 seed 平均"
        )

    # 3. 多 seed 循环
    if re.search(r"for\s+(?:seed|s)\s+in\s+(?:SEEDS|\[\s*SEED)", code):
        errors.append(
            "规则 2 违反: 存在多 seed 训练循环 (for seed/s in SEEDS/[SEED,...])"
        )

    # 4. sklearn 高层 ensemble
    if re.search(r"\b(?:StackingClassifier|VotingClassifier|BaggingClassifier)\b", code):
        errors.append(
            "规则 2 违反: 使用 sklearn 高层 ensemble "
            "(StackingClassifier/VotingClassifier/BaggingClassifier)"
        )

    # 5. Stacking: LogisticRegression 输入是其他模型的 predict_proba
    if re.search(r"LogisticRegression", code) and re.search(r"predict_proba", code):
        # 粗糙判断:LR 与 predict_proba 同时出现大概率是 stacking
        if re.search(r"(?:meta|stack|meta_learner|oof)", code, re.I):
            errors.append(
                "规则 2 违反: 检测到 stacking 模式"
                "(LogisticRegression + predict_proba + meta/stack/oof)"
            )

    return errors


# ---------------------------------------------------------------------------
# 规则 4: regression 任务 score 列必须是预测值(不是概率)
# ---------------------------------------------------------------------------

#: v2.17.0 规则 4:概率化输出的来源(调用名)
_R4_PROBA_ATTRS = frozenset({"predict_proba"})
_R4_PROBA_FUNCS = frozenset({"sigmoid", "expit", "_sigmoid", "softmax"})
#: 不把实参污点传给返回值的调用(模型推断/训练/标量化/指标)
_R4_NO_PASSTHRU = frozenset({
    "predict", "fit", "fit_predict", "transform", "fit_transform",
    "decision_function", "score", "inplace_predict", "len", "int", "float",
    "str", "bool", "print", "repr", "isinstance", "type", "round", "mean",
    "sum", "std", "var", "median", "nanmean", "nansum", "count_nonzero",
    "roc_auc_score", "log_loss", "mean_squared_error", "mean_absolute_error",
    "r2_score", "compute_metric", "step", "dump_json", "dump_model",
    "dump_eval_frame", "dump_feature_importance", "leak_scan"})
_R4_SCORE_DUMP_RE = re.compile(r"^(?:dump_\w*score|_?write_\w*score\w*)$")


def _r4_static_truth(test, proba_absent: bool):
    """规则 4 的静态可达性:按 MODEL_TYPE=regression 折叠(与 runtime.templates.
    static_model_type_truth 同口径;lint 须可独立运行,不 import runtime)。"""
    import ast as _ast

    def lit(node):
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, (_ast.Tuple, _ast.List, _ast.Set)) and all(
                isinstance(e, _ast.Constant) and isinstance(e.value, str)
                for e in node.elts):
            return {e.value for e in node.elts}
        return None

    def ev(node):
        if isinstance(node, _ast.Compare) and len(node.ops) == 1:
            a, b, op = node.left, node.comparators[0], node.ops[0]
            if isinstance(b, _ast.Name) and b.id == "MODEL_TYPE" and \
                    isinstance(op, (_ast.Eq, _ast.NotEq)):
                a, b = b, a
            if not (isinstance(a, _ast.Name) and a.id == "MODEL_TYPE"):
                return None
            vals = lit(b)
            if vals is None:
                return None
            one = len(vals) == 1 and "regression" in vals
            if isinstance(op, _ast.Eq):
                return one
            if isinstance(op, _ast.NotEq):
                return not one
            if isinstance(op, _ast.In):
                return "regression" in vals
            if isinstance(op, _ast.NotIn):
                return "regression" not in vals
            return None
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) \
                and node.func.id == "hasattr" and len(node.args) == 2 and \
                isinstance(node.args[1], _ast.Constant) and \
                node.args[1].value == "predict_proba":
            return False if proba_absent else None
        if isinstance(node, _ast.BoolOp):
            vals = [ev(v) for v in node.values]
            if isinstance(node.op, _ast.And):
                if any(v is False for v in vals):
                    return False
                return True if all(v is True for v in vals) else None
            if any(v is True for v in vals):
                return True
            return False if all(v is False for v in vals) else None
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.Not):
            v = ev(node.operand)
            return None if v is None else (not v)
        return None

    return ev(test)


class _R4Analyzer:
    """规则 4 的污点分析:predict_proba / sigmoid 的结果是否流入 score 写出路径。

    - 源:`.predict_proba(...)`(及其切片/方法链)、sigmoid/expit/softmax 调用;
    - 传播:赋值(流敏感,分支并集)、算术、容器、实参透传(模型推断/训练/
      标量化调用除外)、跨函数返回值摘要(模块函数按名、类方法按方法名);
    - 消解:概率 × 非常量的另一模型输出(hurdle p·μ)不再是"概率口径";
    - 汇:`DataFrame({"score": v})`、`x["score"] = v`、`dump_*score(v)`、
      `fit_and_score` 的返回值(首元素),以及形参流入这些汇的函数的调用点;
    - 可达性:MODEL_TYPE 分支按 regression 折叠、无自定义 predict_proba 类时
      `hasattr(x, "predict_proba")` 判 False、return/raise 之后不可达。
    """

    def __init__(self, tree):
        import ast as _ast

        self.ast = _ast
        self.tree = tree
        self.proba_absent = not any(
            isinstance(n, _ast.ClassDef) and any(
                isinstance(s, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                and s.name == "predict_proba" for s in n.body)
            for n in _ast.walk(tree))
        self.funcs = {n.name: n for n in tree.body
                      if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        self.methods: dict[str, list] = {}
        for n in tree.body:
            if isinstance(n, _ast.ClassDef):
                for s in n.body:
                    if isinstance(s, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        self.methods.setdefault(s.name, []).append(s)
        self.ret_taint: dict[str, bool] = {}
        self.meth_taint: dict[str, bool] = {}
        self.param_sink: dict[str, bool] = {}
        self.mod_env: set = set()
        self.findings: list[dict] = []
        self._record = False
        self._local_funcs: list[dict] = [{}]
        self._cur_fn = "<module>"
        self._ret_hit = False

    # ---------------------------------------------------------------- 表达式
    def _const(self, e) -> bool:
        _a = self.ast
        if isinstance(e, _a.Constant):
            return True
        if isinstance(e, _a.UnaryOp):
            return self._const(e.operand)
        if isinstance(e, _a.BinOp):
            return self._const(e.left) and self._const(e.right)
        return False

    def _call_name(self, f) -> str:
        _a = self.ast
        if isinstance(f, _a.Name):
            return f.id
        if isinstance(f, _a.Attribute):
            return f.attr
        return ""

    def taint(self, e, env) -> bool:
        _a = self.ast
        if e is None:
            return False
        if isinstance(e, _a.Name):
            return e.id in env or e.id in self.mod_env
        if isinstance(e, _a.Call):
            f = e.func
            name = self._call_name(f)
            if isinstance(f, _a.Attribute):
                if f.attr in _R4_PROBA_ATTRS or f.attr in _R4_PROBA_FUNCS:
                    return True
                if self.meth_taint.get(f.attr):
                    return True
                if f.attr not in _R4_NO_PASSTHRU and self.taint(f.value, env):
                    return True
            elif isinstance(f, _a.Name):
                if f.id in _R4_PROBA_FUNCS:
                    return True
                for lf in reversed(self._local_funcs):
                    if f.id in lf:
                        if lf[f.id]:
                            return True
                        break
                else:
                    if self.ret_taint.get(f.id):
                        return True
            if name in _R4_NO_PASSTHRU:
                return False
            return any(self.taint(a, env) for a in e.args) or any(
                self.taint(k.value, env) for k in e.keywords)
        if isinstance(e, (_a.Subscript, _a.Attribute, _a.Starred,
                          _a.NamedExpr)):
            return self.taint(e.value, env)
        if isinstance(e, _a.BinOp):
            lt, rt = self.taint(e.left, env), self.taint(e.right, env)
            if lt and rt:
                return True
            if lt or rt:
                return self._const(e.right if lt else e.left)
            return False
        if isinstance(e, _a.UnaryOp):
            return self.taint(e.operand, env)
        if isinstance(e, _a.IfExp):
            return self.taint(e.body, env) or self.taint(e.orelse, env)
        if isinstance(e, (_a.Tuple, _a.List, _a.Set)):
            return any(self.taint(x, env) for x in e.elts)
        if isinstance(e, _a.BoolOp):
            return any(self.taint(x, env) for x in e.values)
        if isinstance(e, (_a.ListComp, _a.GeneratorExp, _a.SetComp)):
            return self.taint(e.elt, env) or any(
                self.taint(g.iter, env) for g in e.generators)
        return False

    # ---------------------------------------------------------------- 汇
    def _finding(self, node, why: str) -> None:
        if not self._record:
            return
        try:
            snip = self.ast.unparse(node)[:80]
        except Exception:  # noqa: BLE001
            snip = type(node).__name__
        self.findings.append({"line": getattr(node, "lineno", None),
                              "why": why, "code": snip,
                              "function": self._cur_fn})

    def _scan(self, e, env) -> bool:
        """表达式内的写分汇;返回是否命中(param_sink 摘要用)。"""
        _a = self.ast
        hit = False
        if e is None:
            return False
        for c in _a.walk(e):
            if not isinstance(c, _a.Call):
                continue
            name = self._call_name(c.func)
            if name == "DataFrame":
                data = c.args[0] if c.args else next(
                    (k.value for k in c.keywords if k.arg == "data"), None)
                if isinstance(data, _a.Dict):
                    for k, v in zip(data.keys, data.values):
                        if isinstance(k, _a.Constant) and k.value == "score" \
                                and self.taint(v, env):
                            self._finding(c, "DataFrame({'score': 概率})")
                            hit = True
            elif _R4_SCORE_DUMP_RE.match(name or "") and any(
                    self.taint(a, env) for a in c.args):
                self._finding(c, f"{name}(概率)")
                hit = True
            elif isinstance(c.func, _a.Name) and self.param_sink.get(c.func.id) \
                    and (any(self.taint(a, env) for a in c.args) or any(
                        self.taint(k.value, env) for k in c.keywords)):
                self._finding(c, f"{c.func.id}(...) 的形参流入写分路径")
                hit = True
        return hit

    # ---------------------------------------------------------------- 语句
    def _assign(self, tgt, t: bool, env: set, value=None) -> bool:
        _a = self.ast
        hit = False
        if isinstance(tgt, _a.Name):
            (env.add if t else env.discard)(tgt.id)
        elif isinstance(tgt, (_a.Tuple, _a.List)):
            for x in tgt.elts:
                hit |= self._assign(x, t, env)
        elif isinstance(tgt, _a.Starred):
            hit |= self._assign(tgt.value, t, env)
        elif isinstance(tgt, _a.Subscript):
            sl = tgt.slice
            if isinstance(sl, getattr(_a, "Index", ())):  # py<3.9
                sl = sl.value
            if t and isinstance(sl, _a.Constant) and sl.value == "score":
                self._finding(tgt, "x['score'] = 概率")
                hit = True
            if t and isinstance(tgt.value, _a.Name):
                env.add(tgt.value.id)
        return hit

    def block(self, stmts, env: set):
        """返回 (出口 env, 是否必然终止, 是否命中汇)。"""
        hit = False
        for s in stmts:
            env, term, h = self.stmt(s, env)
            hit |= h
            if term:
                return env, True, hit
        return env, False, hit

    def stmt(self, s, env: set):
        _a = self.ast
        env = set(env)
        if isinstance(s, (_a.FunctionDef, _a.AsyncFunctionDef)):
            saved = self._cur_fn
            self._local_funcs.append({})
            rets = self._fn_body(s, set(env))
            self._local_funcs.pop()
            self._cur_fn = saved
            self._local_funcs[-1][s.name] = rets
            return env, False, False
        if isinstance(s, _a.ClassDef):
            return env, False, False
        if isinstance(s, _a.Return):
            h = self._scan(s.value, env)
            t = self.taint(s.value, env)
            if t:
                self._ret_hit = True
            if self._cur_fn == "fit_and_score" and s.value is not None:
                v = s.value.elts[0] if isinstance(s.value, _a.Tuple) and \
                    s.value.elts else s.value
                if self.taint(v, env):
                    self._finding(s, "fit_and_score 返回概率作 score")
                    h = True
            return env, True, h
        if isinstance(s, _a.Raise):
            return env, True, False
        if isinstance(s, _a.If):
            v = _r4_static_truth(s.test, self.proba_absent)
            if v is True:
                return self.block(s.body, env)
            if v is False:
                return self.block(s.orelse, env)
            h = self._scan(s.test, env)
            e1, t1, h1 = self.block(s.body, set(env))
            e2, t2, h2 = self.block(s.orelse, set(env))
            h = h or h1 or h2
            if t1 and t2:
                return e1 | e2, True, h
            if t1:
                return e2, False, h
            if t2:
                return e1, False, h
            return e1 | e2, False, h
        if isinstance(s, (_a.Assign, _a.AnnAssign, _a.AugAssign)):
            val = s.value
            h = self._scan(val, env)
            if isinstance(s, _a.AugAssign):
                t = self.taint(_a.BinOp(left=s.target, op=s.op, right=val), env)
                targets = [s.target]
            else:
                t = self.taint(val, env) if val is not None else False
                targets = s.targets if isinstance(s, _a.Assign) else [s.target]
            for tg in targets:
                h |= self._assign(tg, t, env)
            return env, False, h
        if isinstance(s, _a.Expr):
            return env, False, self._scan(s.value, env)
        if isinstance(s, (_a.For, _a.AsyncFor, _a.While)):
            h = self._scan(getattr(s, "iter", None) or getattr(s, "test", None),
                           env)
            if isinstance(s, (_a.For, _a.AsyncFor)):
                self._assign(s.target, self.taint(s.iter, env), env)
            e1, _, h1 = self.block(s.body, set(env))
            e2, _, h2 = self.block(s.body, env | e1)
            e3, _, h3 = self.block(s.orelse, env | e1 | e2)
            return env | e1 | e2 | e3, False, h or h1 or h2 or h3
        if isinstance(s, (_a.With, _a.AsyncWith)):
            h = False
            for it in s.items:
                h |= self._scan(it.context_expr, env)
                if it.optional_vars is not None:
                    self._assign(it.optional_vars,
                                 self.taint(it.context_expr, env), env)
            e1, t1, h1 = self.block(s.body, env)
            return e1, t1, h or h1
        if isinstance(s, _a.Try):
            eb, tb, hb = self.block(s.body, set(env))
            merged, hh, all_term = env | eb, hb, tb
            for hd in s.handlers:
                eh, th, h_ = self.block(hd.body, env | eb)
                merged |= eh
                hh |= h_
                all_term = all_term and th
            eo, to, ho = self.block(s.orelse, set(eb))
            merged |= eo
            ef, tf, hf = self.block(s.finalbody, set(merged))
            return merged | ef, tf, hh or ho or hf
        # 其它语句:只查汇
        h = False
        for child in self.ast.iter_child_nodes(s):
            if isinstance(child, self.ast.expr):
                h |= self._scan(child, env)
        return env, False, h

    def _fn_body(self, fn, env: set) -> bool:
        """分析函数体,返回"返回值是否携带概率污点"。"""
        saved_fn, saved_ret = getattr(self, "_cur_fn", None), \
            getattr(self, "_ret_hit", False)
        self._cur_fn, self._ret_hit = fn.name, False
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs
                  + fn.args.posonlyargs]
        env = set(env) - set(params)
        self.block(fn.body, env)
        out = self._ret_hit
        self._cur_fn, self._ret_hit = saved_fn, saved_ret
        return out

    def _param_sink_of(self, fn) -> bool:
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs
                  + fn.args.posonlyargs if a.arg not in ("self", "cls")]
        if not params:
            return False
        saved = self._record
        self._record = False
        self._cur_fn = fn.name
        _, _, hit = self.block(fn.body, set(params))
        self._record = saved
        return hit

    def run(self) -> list[dict]:
        _a = self.ast
        module_body = [n for n in self.tree.body if not isinstance(
            n, (_a.FunctionDef, _a.AsyncFunctionDef, _a.ClassDef))]
        for _ in range(6):
            before = (dict(self.ret_taint), dict(self.meth_taint),
                      dict(self.param_sink), set(self.mod_env))
            for name, fn in self.funcs.items():
                self.ret_taint[name] = self._fn_body(fn, set())
                self.param_sink[name] = self._param_sink_of(fn)
            for name, fns in self.methods.items():
                self.meth_taint[name] = any(self._fn_body(f, set())
                                            for f in fns)
            self._cur_fn = "<module>"
            self.mod_env, _, _ = self.block(module_body, set())
            if before == (self.ret_taint, self.meth_taint, self.param_sink,
                          self.mod_env):
                break
        self._record = True
        self.findings = []
        for name, fn in self.funcs.items():
            self._fn_body(fn, set())
        for fns in self.methods.values():
            for f in fns:
                self._fn_body(f, set())
        self._cur_fn = "<module>"
        self.block(module_body, set())
        seen, out = set(), []
        for f in self.findings:
            k = (f["line"], f["why"])
            if k not in seen:
                seen.add(k)
                out.append(f)
        return sorted(out, key=lambda f: f["line"] or 0)


def regression_score_findings(code: str) -> list[dict] | None:
    """v2.17.0 规则 4 的发现列表;代码无法解析返回 None(调用方回退正则口径)。"""
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return None
    return _R4Analyzer(tree).run()


def check_regression_score_semantics(run_dir: Path, code: str) -> list[str]:
    """规则 4(仅 regression):validation_score 的 score 不得是概率口径。

    回归任务里 evaluator 把 score 列直接当预测值计算 rmse/mae/r2;
    若 trial 写出 predict_proba / sigmoid 输出,指标会整体失真
    (概率 ∈[0,1],而目标可能量纲任意)。

    v2.17.0(run4 regression.2 #7):旧版是全文正则,baseline 渲染代码与 HPO
    嫁接段里**不可达**的分类/uplift 通用分派(`predict_proba(...)[:, 1]`)每代
    都误报,只能 adhoc 收窄。现改为静态污点分析(见 `_R4Analyzer`):只有
    概率化输出**实际流入** score 写出路径(validation/shadow 落盘、
    fit_and_score 返回)才拒绝;MODEL_TYPE 折叠后不可达的分支不报;
    hurdle 的 p·max(μ,0) 不是概率口径。代码无法解析时回退旧正则。
    """
    spec_path = run_dir / "task-spec.yaml"
    if not spec_path.exists():
        return []
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if str(spec.get("model_type")) != "regression":
        return []
    findings = regression_score_findings(code)
    if findings is None:
        for pat, why in (
            (r"predict_proba[^\n]*\[[^\]]*[:,]\s*1[^\]]*\]", "score=正类概率"),
            (r"torch\.sigmoid\([^\n]*\)[^\n]*(score|to_parquet)",
             "score=sigmoid 输出"),
        ):
            if re.search(pat, code):
                return [
                    "规则 4 违反: regression 任务的 score 列是预测值,"
                    f"检测到概率化写分({why});改用 model.predict(X) 或回归头输出"
                ]
        return []
    if not findings:
        return []
    where = "; ".join(f"第 {f['line']} 行 {f['why']}(`{f['code']}`)"
                      for f in findings[:4])
    return [
        "规则 4 违反: regression 任务的 score 列是预测值,检测到概率化输出"
        f"(predict_proba/sigmoid)流入 score 写出路径:{where};改用 "
        "model.predict(X) 或回归头输出(hurdle 用 p·max(μ,0),见 "
        "runtime.hurdle.HurdleRegressor)"
    ]


# ---------------------------------------------------------------------------
# 规则 3: 禁止无基数防护的 one-hot
# ---------------------------------------------------------------------------

def check_unsafe_onehot(run_dir: Path, code: str) -> list[str]:
    """规则 3: get_dummies 前必须有基数/可数值化防护。

    历史教训(example-full-3 trial-001/002):parquet decimal128 列读入
    pandas 为 object(Decimal),is_numeric_dtype=False,直接 get_dummies
    按唯一值膨胀出 ~120 万哑变量列,单 trial 峰值内存 932GB 被 cgroup OOM
    kill。凡出现 get_dummies,必须伴随 nunique/基数上限判断或
    to_numeric 数值化路径(如模板 prepare_features());否则拒绝。
    """
    if not re.search(r"get_dummies\s*\(", code):
        return []
    guarded = (re.search(r"nunique\s*\(", code) is not None
               or re.search(r"to_numeric\s*\(", code) is not None)
    if not guarded:
        return [
            "规则 3 违反: get_dummies 前无基数/数值化防护"
            "(decimal 等 object 列会按唯一值膨胀成百万哑变量列导致 OOM;"
            "请使用模板 prepare_features() 或先做 nunique<=阈值检查)"
        ]
    return []


# ---------------------------------------------------------------------------
# 规则 5: uplift 任务必须使用 treatment 列(S-learner/T-learner 口径)
# ---------------------------------------------------------------------------

def check_uplift_treatment(run_dir: Path, code: str) -> list[str]:
    """规则 5(仅 uplift):train.py 必须引用 treatment 列并产出 CATE 口径 score。

    uplift 的 score 是 p1-p0(处置减对照)或双模型差分;若代码从头到
    尾没有引用 treatment 列(常量或按 TaskSpec 声明),模型根本无法
    区分处置/对照,输出的只是转化率概率——auuc/qini 会完全失真。
    检测:代码中出现 TREATMENT 常量、treatment 字样、或 TaskSpec 声明的
    treatment 列名,三者其一即视为合规。
    """
    spec_path = run_dir / "task-spec.yaml"
    if not spec_path.exists():
        return []
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    if str(spec.get("model_type")) != "uplift":
        return []
    declared_col = str((spec.get("treatment") or {}).get("column") or "")
    has_reference = (
        "TREATMENT" in code
        or re.search(r"\btreatment\b", code, re.I) is not None
        or (declared_col and declared_col in code)
    )
    if not has_reference:
        return [
            "规则 5 违反: uplift 任务的 train.py 未引用 treatment 列"
            f"(声明列: {declared_col or '未声明'});"
            "S-learner 需把 treatment 入特征并按 p1-p0 打分,"
            "否则 score 只是转化率概率,auuc/qini 全部失真"
        ]
    return []


# ---------------------------------------------------------------------------
# 规则 6: 模型工件持久化(trial 必须落盘 model/)
# ---------------------------------------------------------------------------

def check_model_persistence(code: str) -> list[str]:
    """规则 6: code/train.py 必须包含模型持久化调用。

    dump_model(model, ...)(模板内置)或直接写 model/model.joblib 等
    工件路径均视为合规;完全缺失则拒绝——模型对象只在内存里训练完
    就丢,事后无法复用/上线(withdraw_t7 实测教训:交付时 trial 与
    final-model 均无模型文件,被迫重训补存)。纯只读/分析型 trial
    可在行内注释加 lint:skip-model-persist 豁免。

    v15.2 追加(规则 6b):特征加宽 trial(transform 新增列,典型形态
    X.assign(new_col=…)/insert/concat 轴向加列)必须调用 dump_eval_frame
    ——分组置换在 harness 侧重放不了 transform,缺失时加宽族全部
    unverifiable(实测 gen-02:8 个加宽族被 0 填充伪判 dead)。
    基础特征面 trial 不要求(基础帧即模型输入)。
    """
    if "lint:skip-model-persist" in code:
        return []
    # 去掉 dump_model 的整个 def 块(防止"只定义不调用"误报):从 def 行
    # 起删除到下一个顶层 def/class 或文件尾
    body = re.sub(
        r"^def dump_model\s*\([^)]*\).*?(?=^def |^class |\Z)",
        "", code, flags=re.M | re.S)
    # 去掉注释与 docstring 中的字样,只看真实可执行调用
    body_nc = re.sub(r"#[^\n]*", "", body)
    body_nc = re.sub(r'""".*?"""', "", body_nc, flags=re.S)
    errors = []
    if not (re.search(r"\bdump_model\s*\(", body_nc)
            or re.search(r"\bsave_model\s*\(", body_nc)
            or re.search(r"\bjoblib\.dump\s*\(", body_nc)
            or re.search(r"model[/\\]model\.(joblib|txt|json|bin|pt)",
                         body_nc)):
        errors.append(
            "规则 6 违反: 未检测到模型持久化(dump_model/save_model/"
            "model/model.* 落盘);在 fit_and_score 训练完成后调用"
            "模板内置 dump_model(model, X_tr.columns),"
            "或注释 lint:skip-model-persist 豁免"
        )
    # ---- 规则 6b(v15.2):加宽 trial 必须落盘评估帧 ----
    # 去掉 dump_eval_frame 自身的 def 块(同 dump_model 的防误报口径)
    body_6b = re.sub(
        r"^def dump_eval_frame\s*\([^)]*\).*?(?=^def |^class |\Z)",
        "", body, flags=re.M | re.S)
    body_6b = re.sub(r"#[^\n]*", "", body_6b)
    body_6b = re.sub(r'""".*?"""', "", body_6b, flags=re.S)
    has_eval_frame_call = re.search(r"\bdump_eval_frame\s*\(", body_6b)
    has_eval_frame_path = re.search(
        r"model[/\\]validation_input\.parquet", body_6b)
    widened = _detect_feature_widening(body_6b)
    if widened and not (has_eval_frame_call or has_eval_frame_path):
        errors.append(
            "规则 6b 违反: 检测到特征加宽(transform 新增列,如 "
            f"{widened[:3]}),但未调用 dump_eval_frame(X_va)——分组置换"
            "重放不了 trial 的 transform,缺失时新增列族全部 "
            "unverifiable(方向生死无证据,v15.2 修复 0 填充伪判 dead "
            "的实测事故);在训练完成后调用模板内置 "
            "dump_eval_frame(X_va) 落盘 model/validation_input.parquet"
        )
    return errors


def _detect_feature_widening(code: str) -> list[str]:
    """启发式检测特征加宽(transform 在特征面上新增列)的典型形态。

    只认明确的加列调用(assign/insert/concat 轴向=1/方括号新列赋值),
    误报方向保守:漏检的代价只是"少一次提醒"(该 trial 评估时
    eval_frame_source 如实标注),误报的代价是拦住合法 trial。
    排除两类非加宽形态(模板 marker 段实测误报):
    - `sys.path.insert(...)` / `os.environ[...] =`(环境装配,非加列);
    - RHS 不引用帧对象自身的方括号赋值(leak_scan 的
      `out["error"] = str(_e)`)——transform 新列的 RHS 必然引用帧
      (`df["r"] = df["a"] / df["b"]`),据此过滤。
    """
    hits: set[str] = set()
    # X.assign(new_col=...) / train_w.assign(...)
    for m in re.finditer(r"(\w[\w.]*)\s*\.\s*assign\s*\(", code):
        hits.add(f"{m.group(1)}.assign()")
    # frame.insert(loc, "new_col", ...);排除 sys.path.insert
    for m in re.finditer(r"(\w[\w.]*)\s*\.\s*insert\s*\(", code):
        if m.group(1) == "sys.path":
            continue
        hits.add(f"{m.group(1)}.insert()")
    # pd.concat([...], axis=1)
    for m in re.finditer(r"pd\s*\.\s*concat\s*\(", code):
        hits.add("pd.concat(axis=1)")
    # df["new_col"] = <RHS 引用 df 自身>(见 docstring 的过滤依据)
    for m in re.finditer(
            r"(\w[\w.]*)\s*\[\s*['\"]([A-Za-z_][\w]*)['\"]\s*\]\s*=[^=]"
            r"([^\n]*)", code):
        obj, col, rhs = m.group(1), m.group(2), m.group(3)
        base_obj = obj.split(".")[0]
        if base_obj in ("sys", "os") or obj == "sys.path":
            continue
        if base_obj not in rhs:
            continue
        hits.add(f'{obj}["{col}"] = …')
    return sorted(hits)


# ---------------------------------------------------------------------------
# 规则 7: hpo trial 禁止在 outer validation 上做选择
# ---------------------------------------------------------------------------

def check_hpo_selection_leak(run_dir: Path, trial_dir: Path,
                             code: str) -> list[str]:
    """规则 7(v11):hpo trial 的参数选择只能用 inner-validation。

    hpo 语义 = train 内部切 inner(时间序尾部)搜索,outer validation
    只评一次。若搜索循环/采样目标引用了外层 validation 的 y 或分数
    (典型泄漏:用 validation AUC 挑 best_params),noise gate 的输入
    不再是干净比较,validation 分数虚高、shadow gap 爆表。检测:
    search/optuna/best_params 选择路径中出现 validation 标签或
    validation_score 的读取。
    """
    cfg_path = trial_dir / "config" / "trial.yaml"
    if not cfg_path.exists():
        return []
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(cfg.get("hpo"), dict):
        return []  # 非 hpo trial 不检查
    errors = []
    # v14:config 有 hpo 节但代码无 hpo 分支 = 运行期 NameError。
    # 实测事故 trial-020/021:参数快通道复制的父代代码来自 baseline
    # 模板,没有 fit_and_score/_hpo_fit_inner/run_hpo_search,三个符号
    # 陆续 NameError,手工修了三轮。规则 7 原本只查泄漏,查不出
    # "config 与代码不匹配"这一整类事故。
    if "run_hpo_search" not in code:
        errors.append(
            "规则 7 违反: config 有 hpo 节但代码无 run_hpo_search 调用"
            "(参数快通道复制的父代代码不含 hpo 分支;运行期会 NameError)"
        )
    elif not _main_calls(code, "fit_and_score"):
        # v2.13.5(req-t7-rand400w gen-04 实测回写):--action hpo
        # --from-parent 对 baseline 血缘父代码嫁接了模板 hpo 段,但父
        # main() 是内联 build_estimator+fit,从不调用 fit_and_score ——
        # config.hpo 永不被消费,trial 静默退化为默认超参重训;两个 HPO
        # trial 各跑 53~65 min 才在评估期暴露(缺 validation_score)。
        errors.append(
            "规则 7 违反: config 有 hpo 节但 main() 不调用 fit_and_score"
            "(hpo 分支在 fit_and_score 内;父代码 main 内联训练时嫁接段"
            "是死代码,HPO 静默退化为普通重训)。修复:把 main() 中"
            "'build_estimator → fit → 打分 → 写 validation_score' 段换成 "
            "`score, _m = fit_and_score(X_tr, y_tr, X_va, y_va, seed, cfg)`"
            " 并保留 validation_score.parquet 落盘(new-trial --action hpo "
            "--from-parent 在 v2.13.5 起会自动改线,失败时报此错)"
        )
    # 选参路径触及 outer validation 的标志
    if re.search(r"optuna.*valid|valid.*optuna", code, re.I):
        errors.append(
            "规则 7 违反: hpo 搜索路径疑似引用 outer validation"
            "(选择必须在 train 内切的 inner-validation 上)"
        )
    # 在含 best_params 的选择语境里读 validation 标签
    if re.search(r"best_params[^\n]{0,200}y_va", code) or re.search(
            r"y_va[^\n]{0,200}best_params", code):
        errors.append(
            "规则 7 违反: best_params 选择与 outer validation 标签"
            "(y_va)同现;重训后的单次打分除外,选择必须只用 inner"
        )
    return errors


def _main_calls(code: str, func_name: str) -> bool:
    """main() 函数体内是否(直接)调用 func_name。语法错/无 main 时保守
    返回 True(交给规则 13/19 报语法与未定义名)。"""
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return True
    mains = [n for n in tree.body
             if isinstance(n, _ast.FunctionDef) and n.name == "main"]
    if not mains:
        return True
    for node in _ast.walk(mains[-1]):
        if isinstance(node, _ast.Call):
            f = node.func
            if (isinstance(f, _ast.Name) and f.id == func_name) or (
                    isinstance(f, _ast.Attribute) and f.attr == func_name):
                return True
    return False


# ---------------------------------------------------------------------------
# 规则 8: 早停/调参 eval_set 禁止使用 outer validation(裁决面解耦)
# ---------------------------------------------------------------------------

def check_eval_set_leak(code: str) -> list[str]:
    """规则 8(v12,P0-4):eval_set / early stopping 不得引用 outer validation。

    validation 同时是主指标/delta/noise gate/榜单排名/最终选型的裁决面;
    每个 trial 都在它上面做早停,等于让"训练自由度"直接优化裁决面,
    产生与搜索方向相关的系统性乐观偏差(实测全 run 最大两笔增益恰是
    早停自由度最大的配置)。早停集必须从 train 内部切(es_valid /
    inner-validation)。hpo 路径的 best_params 与 y_va 同现已由
    规则 7 覆盖。

    检测注意:模板脚手架的 fit_and_score **签名**含 y_va 入参(签名
    兼容,规则不针对签名);只拦 eval_set=[(X_va/y_va…)] 的实际使用。
    """
    errors = []
    # eval_set=[(X_va, y_va)] / eval_set=[(X_valid, y_valid)] 形态
    if re.search(r"eval_set\s*=\s*\[\s*\(\s*X_v(?:a|alid)\w*\s*,", code):
        errors.append(_RULE8_MSG)
    # 变体:eval_set 含 valid 关键字的裸变量(区分 es_valid/inner_valid)
    m = re.search(r"eval_set\s*=\s*\[\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\]", code)
    if m:
        x_name, y_name = m.group(1), m.group(2)
        for name in (x_name, y_name):
            if re.fullmatch(r"(X_va|X_valid|X_validation|X_val)\b", name) \
                    or re.fullmatch(r"(y_va|y_valid|y_validation|y_val)\b", name):
                if _RULE8_MSG not in errors:
                    errors.append(_RULE8_MSG)
    return errors


_RULE8_MSG = (
    "规则 8 违反: eval_set 使用 outer validation"
    "——早停/调参验证集必须从 train 内部切(尾部时间序 es_valid"
    "或 inner-validation);validation 是裁决面,只评分不参与"
    "任何训练决策(v12/P0-4)"
)


def resolve_es_source(run_dir: Path) -> str:
    """v2.16.0:早停面口径(决定规则 8 是否生效)。

    MODELEVO_ES_SOURCE > run 配置 modelevo-config.yaml 的
    training.early_stop_source > 默认 validation。validation 模式下
    outer validation 本就是早停面,规则 8 不再适用;train_tail(v2.15 两段式
    回退)仍执行规则 8。规则 7(超参选择不得用 outer validation)始终生效。
    """
    import os

    v = os.environ.get("MODELEVO_ES_SOURCE")
    if not v:
        try:
            cfg = yaml.safe_load((Path(run_dir) / "modelevo-config.yaml")
                                 .read_text(encoding="utf-8")) or {}
            node = cfg.get("modelevo", cfg) if isinstance(cfg, dict) else {}
            v = ((node or {}).get("training") or {}).get("early_stop_source")
        except Exception:  # noqa: BLE001  读不到配置按默认
            v = None
    return "train_tail" if str(v or "").strip().lower() == "train_tail" \
        else "validation"


# ---------------------------------------------------------------------------
# 规则 9(v13):泄漏防御——banned 列消费拦截 / leak_scan 存活 / 禁绕数据面
# ---------------------------------------------------------------------------

def _load_banned_columns(run_dir: Path) -> set[str]:
    """从 task-spec.yaml 读 banned 列集合(旧 flat exclude 全部视为 banned)。"""
    spec_path = run_dir / "task-spec.yaml"
    if not spec_path.exists():
        return set()
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return set()
    feats = spec.get("features") or {}
    banned: set[str] = set()
    for item in feats.get("banned") or []:
        banned.add(item["column"] if isinstance(item, dict) else str(item))
    if feats.get("banned") is None:
        banned.update(str(c) for c in (feats.get("exclude") or []))
    return banned


def _strip_scaffold_markers(code: str) -> str:
    """剥掉脚手架 marker 段(BANNED/exclude_columns 定义本身),只留
    Agent 可编辑的 transform 区——否则规则 9a 会把 harness 注入的
    列名集合本身当违规。"""
    body = code
    # 去注释与 docstring(引用字面量多在实现里)
    body = re.sub(r'"""[\s\S]*?"""', "", body)
    body = re.sub(r"#[^\n]*", "", body)
    return body


def check_banned_consumption(run_dir: Path, code: str) -> list[str]:
    """规则 9a:trial code 的 transform 区不得出现 banned 列名字面量。

    req-t7-drawdown 实测事故:flat exclude 只挡列名,前缀聚合
    (agg_loan_nonzero_cnt 吞 loan_cnt_t7/t15/t30/t0)把信息偷渡回
    特征面,以 5.85% gain 成为头号特征。v13 数据面已物理隔离(banned
    列不在任何 parquet 里),本规则把"运行期 KeyError"提前为
    "写码期明确报错"——名字防御,不是信息流防御(后者靠物理隔离
    + leak_scan 哨兵)。reparse 列不检查(它们经 reparse_frame()
    访问器显式消费是合法的)。
    """
    banned = _load_banned_columns(run_dir)
    if not banned:
        return []
    body = _strip_scaffold_markers(code)
    # 排除 exclude_columns/load_labels 等脚手架函数体的干扰:只扫
    # BANNED 集合定义与函数定义区之外的真实引用太脆,改为全文扫描
    # 但跳过注入行本身(harness 生成的 frozenset([...]) 行)
    injected = re.search(
        r"BANNED:\s*frozenset\[str\]\s*=\s*frozenset\((\[[^\]]*\])\)", code)
    injected_names: set[str] = set()
    if injected:
        injected_names = set(re.findall(r"'([^']+)'", injected.group(1)))
    errors = []
    for col in sorted(banned):
        # 字符串字面量形式引用('col' / "col")
        pat = r"['\"]" + re.escape(col) + r"['\"]"
        hits = re.findall(pat, code)
        # 排除注入行自身的出现
        if hits and col not in injected_names:
            errors.append(
                f"规则 9a 违反: transform 代码引用 banned 列 {col!r}"
                "(信息有毒,任何形态不得进特征;数据面已物理隔离,"
                "此处引用说明代码试图重建它)")
    return errors


def check_leak_scan_alive(run_dir: Path, code: str) -> list[str]:
    """规则 9b:leak_scan( 调用必须存活(同规则 6 的 dump_model 模式)。

    训练后哨兵是深纵深的最后一段;Agent 删掉它等于拆掉了
    "从合法列重建极端未来信息"的唯一动态防线。豁免:纯只读/
    分析型 trial(与规则 6 同口径,lint:skip-model-persist 同时豁免)。
    """
    if "lint:skip-model-persist" in code:
        return []
    body = re.sub(r"^def leak_scan\s*\([\s\S]*?(?=^def |^class |\Z)",
                  "", code, flags=re.M)
    body_nc = re.sub(r"#[^\n]*", "", body)
    body_nc = re.sub(r'"""[\s\S]*?"""', "", body_nc)
    if not re.search(r"\bleak_scan\s*\(", body_nc):
        # metrics.json 写入处直接内联 suspects 也算(灵活口径)
        if not re.search(r"leak_scan", body_nc):
            return [
                "规则 9b 违反: 未检测到 leak_scan( 调用(模板内置的"
                "训练后泄漏哨兵);在 fit_and_score 训练完成后调用"
                "leak_scan(model, X_tr, y_tr) 并把结果写入 metrics.json,"
                "或注释 lint:skip-model-persist 豁免"
            ]
    return []


def check_no_direct_load_data(code: str) -> list[str]:
    """规则 9c:trial code 禁止直接调用 load_data()(绕过数据面)。

    load_data() 返回含 banned 列的原始全表;从它重建特征等于
    绕开四平面物理隔离。数据视图构建脚本(views/*.py)不受本规则
    约束(它本来就要消费原始数据),但视图产物落盘时有
    assert_face_clean 后置断言拦截。
    """
    body = _strip_scaffold_markers(code)
    # 排除 data-loader 自身/视图脚本的判定交给调用方;trial 的
    # train.py 里出现 load_data( 调用即拒绝(脚手架模板不含它)
    if re.search(r"\bload_data\s*\(", body):
        return [
            "规则 9c 违反: train.py 直接调用 load_data()——绕过四平面"
            "数据隔离(特征/标签/解析源/身份),banned 列会随原始表回流;"
            "特征从 splits/*.parquet、标签从 labels/、解析源从"
            "reparse/ 取(模板访问器)"
        ]
    return []


def check_scaffold_fit_and_score(code: str) -> list[str]:
    """规则 10:未改造的模板脚手架 fit_and_score 不得 run(不分 model_type)。

    脚手架 fit_and_score 的末尾 else 分支是显式 raise(带
    `# lint:scaffold-fit-and-score` 标记):v12 前是 DummyClassifier,
    Agent 未改造的 trial 会静默成功并产出 AUC≈0.5 进 leaderboard。
    **标记仍在 = 脚手架未改造,run 前静态拒绝**,把"训练尾才爆"提前为
    "写码期报错"。v2.17.0(D6)据实更正:本规则对回归/uplift 脚手架同样
    生效 —— 回归分支只是 DummyRegressor(中位数常数预测)占位,uplift
    分支是默认 T-learner,二者虽可执行,但都不是按 hypothesis 改造过的
    模型路径(旧说明"回归/uplift 分支可执行、不拦"与实际行为相反)。
    放行条件:Agent 实现真实模型路径并删除标记;或模块级已注入 `_HPO_ALGO`
    且算法化分支(`elif globals().get("_HPO_ALGO"):`)在 else 之前接管
    —— hpo 模板渲染、confirm/--set 参数子代,以及 v2.17.0 起
    `new-trial --algorithm xgboost|lightgbm|catboost` 渲染的回归脚手架
    (harness 创建期注入算法常量并中和标记)。
    """
    if re.search(r"lint[:\s-]*scaffold-fit-and-score", code):
        # v2.15.0:模块级已注入 _HPO_ALGO 且算法化分支
        # (`elif globals().get("_HPO_ALGO"):`)在 else 之前接管时,脚手架
        # raise 不可达(harness 创建期也会中和标记;此处兜底存量代码)
        if re.search(r"(?m)^_HPO_ALGO\s*=\s*['\"]\w+['\"]", code) and \
                'elif globals().get("_HPO_ALGO"):' in code:
            return []
        return [
            "规则 10 违反: fit_and_score 仍是脚手架默认实现(脚手架标记 "
            "lint:scaffold-fit-and-score 仍在;分类分支是 raise,回归分支只是 "
            "DummyRegressor 占位,uplift 是默认 T-learner)——按 hypothesis "
            "实现真实模型路径并删除标记后重跑(参考 baseline 模板);回归任务"
            "可改用 new-trial --algorithm xgboost|lightgbm 重新渲染(harness "
            "注入算法常量,走算法化分支构造真实回归器,本规则自动放行)",
        ]
    return []


def check_hyperparameters_consumption(run_dir: Path, trial_dir: Path,
                                      code: str) -> list[str]:
    """规则 11(v16/P0-1):config 超参必须被代码真实消费。

    快通道 `new-trial --from-parent --set k=v` 只把覆写合并进
    config/trial.yaml 的 hyperparameters 节,父代代码若不读它,
    覆写静默失效——trial 在重跑父代参数,hypothesis 从未被注入
    (实测 400w run trial-006:--set 假设"从未被测试")。本规则在
    run 前拦截:config 含非空 hyperparameters(dict,至少一键)且
    code 里找不到任何消费点(cfg["hyperparameters"] /
    cfg.get("hyperparameters") / get("hyperparameters"))→ 拒绝。

    hpo trial 的 best_params 路径天然消费 config(hpo 节);本规则
    只查非空 hyperparameters 节。豁免:代码注释含
    `lint:skip-hyperparameters`(如纯特征面 trial 确认无需超参)。
    """
    if re.search(r"lint[:\s-]*skip-hyperparameters", code):
        return []
    cfg_path = trial_dir / "config" / "trial.yaml"
    if not cfg_path.exists():
        return []
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return []
    hp = cfg.get("hyperparameters")
    if not isinstance(hp, dict) or not hp:
        return []
    # baseline 的 hyperparameters 是说明字符串,不触发
    if all(isinstance(v, str) for v in hp.values()) and any(
            "baseline" in str(v) or "默认" in str(v) for v in hp.values()):
        return []
    consumed = ('"hyperparameters"' in code
                or "'hyperparameters'" in code)
    if not consumed:
        return [
            f"规则 11 违反: config hyperparameters 非空({len(hp)} 键)"
            "但 code/train.py 无消费点(缺 cfg[\"hyperparameters\"] / "
            "cfg.get(\"hyperparameters\") 读取)——快通道 --set 覆写会"
            "静默失效,trial 将重跑父代参数。请在 _build_model/"
            "fit_and_score 注入 config 超参(参考 v16 模板的 _cfg_hp "
            "口径),或确认无需超参后加注释 lint:skip-hyperparameters",
        ]
    return []


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def check_undefined_names(code: str) -> list[str]:
    """规则 13(v18/P0-2):AST 层未定义名 / 先用后导检查。

    lint 此前是纯 regex,catch 不到运行期 NameError——实测 400w run
    trial-006(hpo 模板渲染):Agent 在 `import time as _time` 之前
    引用 `_time`,两次 code_error 各烧一轮全量重训(每次 ~2h),lint
    两次放行。本规则用 ast 静态 walk:
    1. 模块级语句按序扫描,记录已绑定名(import/def/class/赋值);
       模块级表达式引用未绑定名 → 报错(运行期必炸 NameError);
    2. 模块内函数体不深查(动态绑定太常见,误报不可控),只查
       模块级——历史事故全部是模块级先用后导;
    3. builtins 与常见环境注入名(MODELEVO_*  env 读取不在此列)豁免;
    4. `if TYPE_CHECKING`/字符串内引用天然不触发。

    误报兜底:注释 `lint:skip-undefined-names` 全规则豁免。
    """
    if re.search(r"lint[:\s-]*skip-undefined-names", code):
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"规则 13 违反: code/train.py 语法错误(运行必炸):{exc}"]

    import builtins as _b

    builtin_names = set(dir(_b))
    # 模板/常见注入名(trial 模板 marker 段之外 Agent 可能假设存在的)
    injected = {
        "TARGET", "MODEL_TYPE", "TREATMENT", "SEED", "BANNED",
        "REPARSE_SOURCES", "SPLITS", "LABELS_DIR", "REPARSE_DIR",
        "RUN_DIR", "TRIAL_DIR", "SKILL_ROOT", "SMOKE_MAX_ROWS",
        "SMOKE_MAX_VALID", "_GPU_MIN_ROWS", "METRIC_NAME",
        "_INNER_METRIC_UP", "__file__", "__name__",
    }
    errors: list[str] = []
    bound: set[str] = set(builtin_names) | injected

    def _bind_target(t) -> None:
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                _bind_target(e)
        elif isinstance(t, ast.Starred):
            _bind_target(t.value)

    def _scan_expr(node) -> None:
        """模块级表达式里引用未绑定名 → 报错(只查 Name load 上下文)。

        手动栈 walk(不可用 ast.walk):comprehension 的绑定变量
        (c/i 等)在其自身作用域内,普通 walk 会继续深入其子树把
        target 名当模块级引用误报(实测 trial-007 `_WKMO_IDX = {c: i
        for i, c in enumerate(...)}` 的 c)。命中 comprehension 时
        单独处理并**剪枝**其子树。
        """
        stack = [node]
        while stack:
            sub = stack.pop()
            if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp,
                                ast.GeneratorExp)):
                # comprehension 自成作用域:target 自绑(c/i/x 等),
                # iter/ifs/elt 引用 target 名合法;只查真正未绑定的
                # 外层名。多 generator 时后者可见前者的 target(从左
                # 到右累积绑定)。
                comp_bound = set(bound)
                for comp in sub.generators:
                    for t in ([comp.target] if not isinstance(
                            comp.target, (ast.Tuple, ast.List))
                            else list(comp.target.elts)):
                        if isinstance(t, ast.Name):
                            comp_bound.add(t.id)
                    for expr in [comp.iter, *comp.ifs]:
                        for n2 in ast.walk(expr):
                            if isinstance(n2, ast.Name) and isinstance(
                                    n2.ctx, ast.Load) and \
                                    n2.id not in comp_bound:
                                errors.append(_err(n2))
                inners = []
                if isinstance(sub, ast.DictComp):
                    inners = [sub.key, sub.value]
                elif hasattr(sub, "elt"):
                    inners = [sub.elt]
                for inner in inners:
                    if inner is not None:
                        _scan_comp_inner(inner, set(comp_bound))
                continue
            if isinstance(sub, ast.Name) and isinstance(
                    sub.ctx, ast.Load) and sub.id not in bound:
                errors.append(_err(sub))
                continue
            for child in ast.iter_child_nodes(sub):
                stack.append(child)

    def _scan_comp_inner(node, scope: set) -> None:
        """comprehension elt/key/value 的引用检查(作用域 = 外层
        bound ∪ 全部 target;嵌套 comprehension 同规)。"""
        stack = [node]
        while stack:
            sub = stack.pop()
            if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp,
                                ast.GeneratorExp)):
                inner_scope = set(scope)
                for comp in sub.generators:
                    for t in ([comp.target] if not isinstance(
                            comp.target, (ast.Tuple, ast.List))
                            else list(comp.target.elts)):
                        if isinstance(t, ast.Name):
                            inner_scope.add(t.id)
                    for expr in [comp.iter, *comp.ifs]:
                        for n2 in ast.walk(expr):
                            if isinstance(n2, ast.Name) and isinstance(
                                    n2.ctx, ast.Load) and \
                                    n2.id not in inner_scope:
                                errors.append(_err(n2))
                inners = []
                if isinstance(sub, ast.DictComp):
                    inners = [sub.key, sub.value]
                elif hasattr(sub, "elt"):
                    inners = [sub.elt]
                for inner in inners:
                    if inner is not None:
                        _scan_comp_inner(inner, inner_scope)
                continue
            if isinstance(sub, ast.Name) and isinstance(
                    sub.ctx, ast.Load) and sub.id not in scope:
                errors.append(_err(sub))
                continue
            for child in ast.iter_child_nodes(sub):
                stack.append(child)

    def _err(sub) -> str:
        return (
            f"规则 13 违反: 模块级第 {sub.lineno} 行引用了未定义名 "
            f"`{sub.id}`(先于其 import/赋值)——运行期必炸 "
            f"NameError(实测 400w trial-006:两次 code_error "
            f"各烧一轮全量重训)。把 import 移到首个使用点之前。")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            # def/class:先登记名字(装饰器体内引用已 bound 的才合法),
            # 体不深查
            for dec in getattr(node, "decorator_list", []):
                _scan_expr(dec)
            bound.add(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
            continue
        if isinstance(node, ast.Assign):
            _scan_expr(node.value)
            for t in node.targets:
                _bind_target(t)
            continue
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                _scan_expr(node.value)
            if node.target:
                _bind_target(node.target)
            continue
        if isinstance(node, ast.AugAssign):
            _scan_expr(node.value)
            continue
        if isinstance(node, ast.If):
            # 模块级 if:两条分支都扫(main 守卫的 `if __name__` 体内
            # 引用 main——main 已在 bound)
            _scan_expr(node.test)
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    bound.add(sub.name)
            for stmt in node.body + node.orelse:
                if isinstance(stmt, ast.Assign):
                    _scan_expr(stmt.value)
                    for t in stmt.targets:
                        _bind_target(t)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    for a in stmt.names:
                        bound.add((a.asname or a.name).split(".")[0])
                elif isinstance(stmt, ast.Expr):
                    _scan_expr(stmt)
                elif isinstance(stmt, ast.Try):
                    for s2 in (stmt.body + stmt.orelse
                               + stmt.finalbody + stmt.handlers):
                        if isinstance(s2, ast.Assign):
                            _scan_expr(s2.value)
                            for t in s2.targets:
                                _bind_target(t)
                        elif isinstance(s2, (ast.Import, ast.ImportFrom)):
                            for a in s2.names:
                                bound.add(
                                    (a.asname or a.name).split(".")[0])
            continue
        if isinstance(node, ast.Try):
            for stmt in (node.body + node.orelse + node.finalbody
                         + [h for h in node.handlers]):
                if isinstance(stmt, ast.Assign):
                    _scan_expr(stmt.value)
                    for t in stmt.targets:
                        _bind_target(t)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    for a in stmt.names:
                        bound.add((a.asname or a.name).split(".")[0])
                elif isinstance(stmt, ast.Expr):
                    _scan_expr(stmt)
                elif isinstance(stmt, ast.ExceptHandler):
                    for s2 in stmt.body:
                        if isinstance(s2, ast.Assign):
                            _scan_expr(s2.value)
                            for t in s2.targets:
                                _bind_target(t)
                        elif isinstance(s2, (ast.Import, ast.ImportFrom)):
                            for a in s2.names:
                                bound.add(
                                    (a.asname or a.name).split(".")[0])
            continue
        if isinstance(node, ast.Expr):
            _scan_expr(node)
            continue
        if isinstance(node, ast.For):
            _scan_expr(node.iter)
            _bind_target(node.target)
            for stmt in node.body + node.orelse:
                if isinstance(stmt, ast.Assign):
                    _scan_expr(stmt.value)
                    for t in stmt.targets:
                        _bind_target(t)
                elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    for a in stmt.names:
                        bound.add((a.asname or a.name).split(".")[0])
            continue
        if isinstance(node, ast.With):
            for item in node.items:
                _scan_expr(item.context_expr)
                if item.optional_vars is not None:
                    _bind_target(item.optional_vars)
            continue
    # 同名只报一次(同一名字可能引用多次)
    seen = set()
    deduped = []
    for e in errors:
        key = e.split("`")[1] if "`" in e else e
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


def check_function_scope_shadowing(code: str) -> list[str]:
    """规则 14(v19.3):函数内"先用后导"检测(UnboundLocalError 形态)。

    背景(harness 自身实测,2026-09-11):`cmd_write_splits_view` 在
    函数**前段**引用 `SPLIT_NAMES`,而 `from ... import SPLIT_NAMES`
    写在函数**后段** —— Python 因该名在函数内被绑定而判其为局部变量,
    前段引用直接抛 `UnboundLocalError`;该异常又被同一段的
    `except Exception` 吞掉,表现为"注入静默失效"(视图骨架的泄漏
    防线 `_PROTECTED_COLS` 恒为空 → 标签列被加权成 AUC≈1 伪特征)。

    规则 13 只扫**模块级**语句(函数体不深查,防动态绑定误报),
    这一整类"函数内先用后导"从它底下穿过。本规则补上:对每个函数,
    若某名字在**该函数自身作用域**内被绑定,且存在行号更早的 Load
    引用 → 报错(必然 UnboundLocalError)。

    误报控制(实测回写:本规则首版在 14 个真跑通的 trial 上误报 40 次,
    全部来自 comprehension 作用域——`[c for c in cols]` 的 `c` 是
    comprehension 自绑名,却被当成函数级绑定):
    - **comprehension 目标名整体豁免**:凡在该函数内作为
      ListComp/SetComp/DictComp/GeneratorExp 目标出现的名字,一律不报
      (其作用域自包含,与函数级同名绑定无因果关系);
    - 形参 / `global` / `nonlocal` 声明的名字跳过;
    - 嵌套 def/lambda 是独立作用域:其体内的绑定与引用都不计入外层,
      嵌套函数自身另行单独检查;
    - 只在"绑定行号**严格晚于**引用行号"时报(同行不报);
    - 名字在该函数内从未被绑定(是模块级/内建)→ 不报;
    - 注释 `lint:skip-undefined-names` 全规则豁免(与规则 13 同标记)。
    """
    if re.search(r"lint[:\s-]*skip-undefined-names", code):
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # 语法错由规则 13 报,避免重复
    import builtins as _b

    builtin_names = set(dir(_b))
    errors: list[str] = []

    def _bind_target(target, out: dict, lineno: int) -> None:
        if isinstance(target, ast.Name):
            out.setdefault(target.id, lineno)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                _bind_target(e, out, lineno)
        elif isinstance(target, ast.Starred):
            _bind_target(target.value, out, lineno)

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in (fn.args.args + fn.args.posonlyargs
                                  + fn.args.kwonlyargs)}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)
        declared: set[str] = set()
        first_bind: dict[str, int] = {}
        loads: dict[str, int] = {}
        # comprehension 自绑名:这些名字在 comprehension 内自成作用域,
        # 与函数级绑定无因果关系(实测误报 40/40 都来自这里)。
        comp_names: set[str] = set()

        def _collect_comp_names(node) -> None:
            """登记 comprehension 的自绑目标名(整个函数范围内豁免)。

            首版靠 `_bind_target(comp.target, {}, 0)` 的副作用收集,是
            写法冗余;这里显式遍历目标树取名字。
            """
            def _names_of(t, out: set) -> None:
                stack_t = [t]
                while stack_t:
                    cur = stack_t.pop()
                    if isinstance(cur, ast.Name):
                        out.add(cur.id)
                    elif isinstance(cur, ast.Starred):
                        stack_t.append(cur.value)
                    elif isinstance(cur, (ast.Tuple, ast.List)):
                        stack_t.extend(cur.elts)

            for sub in ast.walk(node):
                if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp,
                                    ast.GeneratorExp)):
                    for comp in sub.generators:
                        _names_of(comp.target, comp_names)

        def _visit(node, *, in_nested: bool = False) -> None:
            """遍历函数自身作用域;嵌套 def/lambda 只登记名字不深入。

            行号取自"最近的带 lineno 的祖先节点":withitem/ExceptHandler
            等辅助节点自身不一定带 lineno。
            """
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.Lambda)):
                if isinstance(node, ast.Lambda):
                    return  # lambda 无名字,且其体是独立作用域
                first_bind.setdefault(node.name, node.lineno)
                return
            if isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
                declared.update(node.names)
                return
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                 ast.GeneratorExp)):
                _collect_comp_names(node)
            _ln = getattr(node, "lineno", None)
            if isinstance(node, ast.Import):
                for a in node.names:
                    first_bind.setdefault(
                        (a.asname or a.name).split(".")[0], _ln)
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    first_bind.setdefault(
                        (a.asname or a.name).split(".")[0], _ln)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    _bind_target(t, first_bind, _ln)
            elif isinstance(node, ast.AnnAssign):
                if node.target is not None:
                    _bind_target(node.target, first_bind, _ln)
            elif isinstance(node, ast.AugAssign):
                _bind_target(node.target, first_bind, _ln)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                _bind_target(node.target, first_bind, _ln)
            elif isinstance(node, ast.withitem):
                if node.optional_vars is not None:
                    _bind_target(node.optional_vars, first_bind, _ln)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                first_bind.setdefault(node.name, _ln)
            elif isinstance(node, ast.Name) and isinstance(node.ctx,
                                                           ast.Load):
                loads.setdefault(node.id, _ln)
            for child in ast.iter_child_nodes(node):
                _visit(child, in_nested=in_nested)

        for stmt in fn.body:
            _visit(stmt)

        for name, use_line in loads.items():
            if name in params or name in declared or name in builtin_names:
                continue
            if name in comp_names:
                continue  # comprehension 自绑名,与函数级绑定无关
            bind_line = first_bind.get(name)
            if bind_line is None:
                continue  # 该函数内从未绑定 → 是模块级/闭包名,不报
            # 判据:引用行 **早于** 首个绑定行 → Python 视该名为局部
            # 变量,引用时尚未赋值 → UnboundLocalError。
            if use_line >= bind_line:
                continue
            errors.append(
                f"规则 14 违反: 函数 `{fn.name}` 内第 {use_line} 行引用了 "
                f"`{name}`,而该名在第 {bind_line} 行才被绑定"
                "(import/赋值)——Python 会把它判为局部变量,前段引用直接"
                "抛 UnboundLocalError(实测:harness 自身 "
                "cmd_write_splits_view 的 SPLIT_NAMES;外层若还套着 "
                "`except Exception` 就是**静默失效**)。把绑定移到首个"
                "使用点之前。")
    return errors



# 规则 14(v19.4):hpo 嫁接产物的 sys.path bootstrap 检查
def check_runtime_import_bootstrap(code: str) -> list[str]:
    """规则 12: `from runtime.X import` / `import runtime.X` 出现,
    但文件内无 SKILL_ROOT/sys.path bootstrap → 拒 run。

    编号说明(v2.11.9):本规则此前在 docstring 里错标为"规则 14",与
    `check_function_scope_shadowing` 撞号(SKILL.md、MAINTENANCE.md 与
    测试均按 14 = function_scope_shadowing 引用)。12 是编号表里的空位,
    改标 12 不动其余任何编号——历史 CHANGELOG 里的旧引用保持原义。

    背景(gen-03 trial-008 实测):hpo 嫁接段引用 runtime.hpo,而
    trial 以 cwd=code/ 起进程、不带 PYTHONPATH——缺 bootstrap 首次
    run 必抛 ModuleNotFoundError,被失败分类误判 infra_error 触发
    无效重试,Agent 只能读 harness 源码手工补(v19.3 已修注入,本
    规则是兜底:任何来源的手写/嫁接代码都过同一道机检)。
    豁免:文件内显式 `# lint:skip-runtime-bootstrap` 标记
    (诊断场景逃生口)。
    """
    if "# lint:skip-runtime-bootstrap" in code:
        return []
    import re as _re

    has_runtime_import = bool(_re.search(
        r"^\s*(from runtime[\w.]* import |import runtime[\w.]*)",
        code, _re.M))
    if not has_runtime_import:
        return []
    has_bootstrap = (
        "SKILL_ROOT" in code
        and "sys.path.insert" in code
    ) or "sys.path.append" in code or (
        "# v19.3" in code and "sys.path.insert" in code)
    if not has_bootstrap:
        return [
            "规则 12 违反: 代码 import runtime.* 但缺 SKILL_ROOT + "
            "sys.path bootstrap —— trial 以 cwd=code/ 起进程,无 "
            "bootstrap 首次 run 必抛 ModuleNotFoundError(infra_error "
            "误分类)。修复:文件头加模板同款 SKILL_ROOT 注入块;或对"
            " hpo 嫁接场景重建 trial(v19.3+ 嫁接自动补 bootstrap)。"
            "诊断豁免:代码内加 `# lint:skip-runtime-bootstrap`",
        ]
    return []


# ---------------------------------------------------------------------------
# 规则 15(v2.11):特征加宽必须经 widen_cached(重复 widen 是
# v2.10.1 run 实测的第一大计算浪费:同一张加宽面 smoke/全量/
# multi_seed/shadow 重放 13~15 次,单次 830~1000s,占 compute 55%+;
# widen_cached 提供进程内 memo + 落盘缓存,把同进程多轮与跨进程
# finalize-retrain 全部归一)
# ---------------------------------------------------------------------------
def check_widen_cache(code: str) -> list[str]:
    """规则 15: 检测到特征加宽(_widen 定义/调用)但未走 widen_cached。

    加宽 trial 的 _widen 是确定性行内变换——同一口径 repeated call
    纯浪费。检测口径(保守,防误报):
    - 触发条件 = 定义了 `def _widen(` 或直接调用 `_widen(`;
    - 合规 = 出现 `widen_cached(` 调用;
    - 豁免 = `# lint:skip-widen-cache`(诊断/一次性口径)。
    """
    if "# lint:skip-widen-cache" in code:
        return []
    body_nc = re.sub(r"#[^\n]*", "", code)
    body_nc = re.sub(r'""".*?"""', "", body_nc, flags=re.S)
    widens = bool(re.search(r"\bdef _widen\s*\(", body_nc)) or \
        bool(re.search(r"(?<![\w.])_widen\s*\(", body_nc))
    if not widens:
        return []
    if re.search(r"\bwiden_cached\s*\(", body_nc):
        return []
    return [
        "规则 15 违反: 检测到 _widen(特征加宽)但未走 widen_cached"
        "——同一张加宽面在 smoke/全量/multi_seed/shadow 重放会被重复"
        "重算(v2.10.1 实测一个 run 13~15 次全量 widen ≈ 3.2h,占 "
        "compute 55%+)。修复:把 `_widen(a, b, c, d)` 的调用点换成 "
        "`widen_cached('tag', a, b, c, d)`(同口径同 tag;口径变更换 "
        "tag);一次性口径可加 `# lint:skip-widen-cache` 豁免",
    ]


def check_widen_cache_provenance(code: str) -> list[str]:
    """规则 16(v2.12.2.5):加宽缓存键必须含 RUN_DIR(跨 run 隔离)。

    背景(drawdown-t7-sv24 实测事故):缓存目录是 run 级的
    `RUN_DIR/.widen-cache`,但键只由 `TRIAL_DIR.name | tag | 两侧帧指纹`
    组成。同一数据面复制出的多个 run 目录里,**同名 trial 名 + 同 tag +
    同帧指纹**会算出同一个 key —— 实测 example-seq-v3-v2.12.2.4 从 .3
    硬链接继承了前序 run(.2)的 14 组缓存(8.2GB,stats 里带
    `plane`/`desc`/`plane_want` 等本 run 物化器从不写的字段),两组实现的
    J 面分别是 1503 / 1412 新列。一旦新 run 复用同名 trial 名并撞 tag,
    `widen_cached` 会**静默**返回别的 run 实现的列,下游 AUC 与归因全部
    失准且无任何告警。

    检测口径(保守,防误报):仅当代码里出现 `def _widen_face_key(`
    且该函数体内找不到 `RUN_DIR` 时报错。豁免
    `# lint:skip-widen-cache-provenance`(只用于历史代码冻结复现)。
    """
    if "# lint:skip-widen-cache-provenance" in code:
        return []
    m = re.search(r"def _widen_face_key\s*\(", code)
    if not m:
        return []
    seg = code[m.start():]
    nxt = re.search(r"\ndef ", seg[1:])
    seg = seg[:nxt.start() + 1] if nxt else seg
    if "RUN_DIR" in seg:
        return []
    return [
        "规则 16 违反: `_widen_face_key` 未把 RUN_DIR 纳入缓存键 —— "
        "不同 run 目录里同名 trial + 同 tag + 同帧指纹会算出同一 key,"
        "`widen_cached` 会静默命中**别的 run 实现的列**(实测两个 run 的 "
        "同一平面分别是 1503 / 1412 新列,AUC 与归因失准且无告警)。"
        "修复:键串前缀加 RUN_DIR(如 "
        "f\"{RUN_DIR.resolve()}|{TRIAL_DIR.name}|{tag}|…\");"
        "历史代码冻结复现可加 `# lint:skip-widen-cache-provenance` 豁免",
    ]


def check_generated_code_consistency(code: str) -> list[str]:
    """规则 19(v2.12.2.10):生成代码自洽性(未定义名 / 语法错误)。

    与规则 13 的分工:规则 13 只查**模块级**的先用后导(历史事故形态);
    本规则下探到**每个函数体**,查"函数里引用了模块级不存在、也没在
    函数内 import/定义的名字"。

    依据实测事故(2026-09-18,example-seq-v3-v2.12.2.9 / run-jt-draw-t7):
    `runtime/baseline.py` 的 BASELINE_TEMPLATE 里 dtype 治理段调用
    `safe_to_numeric`,而模板体既未定义也未 import 它 —— 模板文件自身
    import 正常、既有测试全绿(名字只出现在字符串里),两个 gen-00
    trial 各跑 3 秒后 NameError;失败 trial 按协议不可删除,永久留在
    leaderboard 上。同类风险的第二个实例是 `roc_auc_score`(只靠函数体
    内的局部 import 侥幸成立)。

    实现复用 `runtime/code_audit.check_generated_code`(preflight 走同一
    实现,口径不会漂移)。lint 在工作区里可能以单文件形式存在、拿不到
    runtime 包,此时静默跳过(该场景下 preflight 已覆盖)。
    """
    if re.search(r"lint[:\s-]*skip-generated-code-consistency", code):
        return []
    try:
        import sys as _sys
        from pathlib import Path as _P

        _skill = _P(__file__).resolve().parent.parent
        for _cand in (_skill, _skill.parent):
            if (_cand / "runtime" / "code_audit.py").exists():
                if str(_cand) not in _sys.path:
                    _sys.path.insert(0, str(_cand))
                break
        from runtime.code_audit import check_generated_code  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — 单文件工作区场景,preflight 已覆盖
        return []
    return [f"规则 19 违反: {p}(生成代码引用了不存在的名字,运行必炸 "
            f"NameError)" for p in check_generated_code(code)]


# ---------------------------------------------------------------------------
# 规则 21(v2.13.5):大数据上的 reparse 解析必须走 feature_block / widen_cached
# ---------------------------------------------------------------------------
#: reparse train 行数超过该值才触发(小数据手写循环无所谓)
_FEATURE_CACHE_MIN_ROWS = 200_000


def _reparse_consumers(code: str) -> list[str]:
    """reparse 解析源的消费点(函数名);`reparse_frame` 自身的定义不算。"""
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return []
    hits: list[str] = []

    def _scan(node, owner: str) -> None:
        for sub in _ast.walk(node):
            if isinstance(sub, _ast.Call) and isinstance(sub.func, _ast.Name) \
                    and sub.func.id == "reparse_frame":
                hits.append(owner)
                return
            if isinstance(sub, _ast.Name) and sub.id == "REPARSE_DIR":
                hits.append(owner)
                return
            if isinstance(sub, _ast.Constant) and sub.value == "reparse":
                hits.append(owner)
                return

    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if node.name == "reparse_frame":
                continue
            _scan(node, node.name)
        elif isinstance(node, _ast.ClassDef):
            _scan(node, node.name)
        elif not isinstance(node, (_ast.Import, _ast.ImportFrom)):
            # 模块级赋值 REPARSE_DIR = SPLITS / "reparse" 是模板常量,不算
            if isinstance(node, _ast.Assign) and any(
                    isinstance(t, _ast.Name) and t.id == "REPARSE_DIR"
                    for t in node.targets):
                continue
            _scan(node, "<module>")
    return sorted(set(hits))


def check_reparse_feature_cache(run_dir: Path, code: str) -> list[str]:
    """规则 21: 大数据上消费 reparse 解析源却不走 feature_block/widen_cached。

    req-t7-rand400w 实测:agent 手写的 seq 解析是单核纯 Python 循环,
    每个派生 trial 逐字重算同一块(seqd 9 次 × 21 min),finalize 再重放
    3 次;解析占全部 trial 时间 34%,GPU 训练只占 7%。
    `runtime.feature_cache.feature_block` 给出跨 trial 内容寻址缓存 +
    按线程配额并行;有训练期拟合状态的变换用 widen_cached。
    触发条件:代码消费 reparse(reparse_frame 调用 / REPARSE_DIR /
    "reparse" 路径段)且 run 的 reparse train 行数 ≥ 20 万,而全文无
    feature_block( / widen_cached( 调用。豁免 `# lint:skip-feature-cache`。
    """
    if "# lint:skip-feature-cache" in code:
        return []
    body_nc = re.sub(r"#[^\n]*", "", code)
    if re.search(r"\bfeature_block\s*\(|\bwiden_cached\s*\(", body_nc):
        return []
    owners = _reparse_consumers(code)
    if not owners:
        return []
    rows = 0
    try:
        import pyarrow.parquet as _pq

        _p = Path(run_dir) / "sample-profile" / "splits" / "reparse" / \
            "train.parquet"
        if _p.exists():
            rows = int(_pq.ParquetFile(_p).metadata.num_rows)
    except Exception:  # noqa: BLE001  读不到行数不拦
        return []
    if rows < _FEATURE_CACHE_MIN_ROWS:
        return []
    return [
        f"规则 21 违反: {owners[:3]} 消费 reparse 解析源({rows} 行)但未走 "
        "feature_block / widen_cached —— 手写逐行解析单核运行且每个派生 "
        "trial/交付重放都重算同一块(req-t7-rand400w:seqd 块 9×21 min)。"
        "修复:把内核包成 `feature_block(name, kernel, split=..., "
        "columns=..., sources=..., splits_dir=SPLITS)` 并定义 "
        "`trial_feature_blocks(split)`(契约见 references/template-api.md;"
        "跨 trial 缓存 + 按线程配额并行);有训练期拟合状态的变换用 "
        "widen_cached。一次性口径可加 `# lint:skip-feature-cache` 豁免",
    ]


def check_validation_score_output(code: str) -> list[str]:
    """规则 22(v2.13.5):代码必须落盘 validation_score.parquet(评估入口)。

    req-t7-rand400w gen-04 trial-019:修复脚本第一版把 main 的训练段换成
    fit_and_score 调用,漏了分数落盘 —— 训练 53 min 后评估期才报
    "Trial 未产出 validation_score.parquet"。输出契约是纯静态可查的,
    不该靠一次全量训练来发现。豁免 `# lint:skip-output-contract`。
    """
    if "# lint:skip-output-contract" in code:
        return []
    body_nc = re.sub(r"#[^\n]*", "", code)
    if "validation_score.parquet" in body_nc:
        return []
    return [
        "规则 22 违反: 代码里没有 validation_score.parquet 落盘(评估入口;"
        "fit_and_score 只返回分数不落盘,main 必须 "
        "`pd.DataFrame({\"score\": score}).to_parquet(TRIAL_DIR / "
        "\"validation_score.parquet\", index=False)`)",
    ]


# ---------------------------------------------------------------------------
# 规则 23(v2.16.2):生效的 fit_and_score 不得甩掉父代 structural 训练入口
# ---------------------------------------------------------------------------

#: structural 训练入口的命名口径(hurdle p·μ / T-learner μ1−μ0 / 两部模型)
_STRUCT_ENTRY_RE = re.compile(r"(?i)hurdle|t_?learner|two_?part|zero_?inflat")


def structural_override_findings(code: str) -> list[dict]:
    """生效(最后绑定)的 fit_and_score 是否覆盖了经 structural 入口训练的旧版。

    run4 regression D-008:`--action hpo --from-parent <hurdle 冠军>` 的嫁接段在
    文件尾重定义模板版 fit_and_score,父代 hurdle 版被"后定义覆盖",inner 搜索与
    重训都退化成单个 XGBRegressor,创建期只有一行 WARNING。判据按 Python 的名字
    解析语义做静态可达性:函数体里的全局名在调用时取模块**最终绑定**;模块级
    `alias = fit_and_score` 捕获的是赋值当时的定义(包装 / 委托链据此可达)。
    被覆盖的某个 fit_and_score 可达 structural 入口(名字匹配 hurdle/T-learner
    等的顶层函数/类),而生效版一个都到不了 → 记一条发现。
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return []
    _defs = (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)
    current: dict = {}
    fs_defs: list = []
    for stmt in tree.body:
        if isinstance(stmt, _defs):
            current[stmt.name] = stmt
            if stmt.name == "fit_and_score":
                fs_defs.append(stmt)
        elif isinstance(stmt, _ast.Assign) and isinstance(
                stmt.value, _ast.Name) and stmt.value.id in current:
            for t in stmt.targets:
                if isinstance(t, _ast.Name):
                    current[t.id] = current[stmt.value.id]
    if len(fs_defs) < 2 or "fit_and_score" not in current:
        return []
    effective = current["fit_and_score"]
    refs: dict = {}

    def _refs(node) -> set:
        key = id(node)
        if key not in refs:
            refs[key] = {n.id for n in _ast.walk(node)
                         if isinstance(n, _ast.Name)
                         and isinstance(n.ctx, _ast.Load)}
        return refs[key]

    def _entries_from(node) -> set:
        seen, stack, hits = set(), [node], set()
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if cur is not node and _STRUCT_ENTRY_RE.search(cur.name or ""):
                hits.add(cur.name)
            for name in _refs(cur):
                tgt = current.get(name)
                if tgt is not None and id(tgt) not in seen:
                    stack.append(tgt)
        return hits

    eff_hits = _entries_from(effective)
    out = []
    for d in fs_defs:
        if d is effective:
            continue
        d_hits = _entries_from(d)
        if d_hits and not (d_hits & eff_hits):
            out.append({"effective_line": effective.lineno,
                        "overridden_line": d.lineno,
                        "entries": sorted(d_hits)})
    return out


def check_structural_override(code: str) -> list[str]:
    """规则 23:生效的 fit_and_score 不再到达父代 structural 训练入口即拒绝。

    豁免:注释 `lint:allow-structural-override`(确属有意把结构化拓扑换成
    单模型时,Agent 必须在 hypothesis 里声明拓扑变化,Δ_parent 才可解释)。
    """
    if re.search(r"lint[:\s-]*allow-structural-override", code):
        return []
    errs = []
    for f in structural_override_findings(code):
        errs.append(
            f"规则 23 违反: 生效的 fit_and_score(第 {f['effective_line']} 行)"
            f"不再到达父代 structural 训练入口 {f['entries'][:6]}(被覆盖的第 "
            f"{f['overridden_line']} 行 fit_and_score 经它们训练 hurdle/"
            "T-learner 等结构化拓扑)——HPO 嫁接/模板段整段覆盖会把结构化拓扑"
            "静默退化为单模型,Δ_parent 混入拓扑变化。hurdle 父代的同算法 HPO 用 "
            "`new-trial --action hpo --from-parent <id>`(v2.16.2 起自动注入 "
            "hurdle 超参段,包住而非覆盖父代入口);确属有意替换时注释 "
            "lint:allow-structural-override 并在 hypothesis 声明拓扑变化")
    return errs


def lint_trial(run_dir: Path, trial_dir: Path, ensemble_enabled: bool) -> list[str]:
    code_path = trial_dir / "code" / "train.py"
    if not code_path.exists():
        return [f"缺少 code/train.py: {code_path}"]
    code = code_path.read_text(encoding="utf-8")
    trial_id = trial_dir.name

    errors = []
    errors.extend(check_algorithm_consistency(trial_id, code))
    if not ensemble_enabled:
        errors.extend(check_no_ensemble(code))
    errors.extend(check_unsafe_onehot(run_dir, code))
    errors.extend(check_regression_score_semantics(run_dir, code))
    errors.extend(check_uplift_treatment(run_dir, code))
    errors.extend(check_model_persistence(code))
    errors.extend(check_hpo_selection_leak(run_dir, trial_dir, code))
    # v2.16.0:validation 早停模式下 outer validation 即早停面,规则 8 只在
    # train_tail 回退模式下执行(规则 7 不变)
    if resolve_es_source(run_dir) == "train_tail":
        errors.extend(check_eval_set_leak(code))
    errors.extend(check_banned_consumption(run_dir, code))
    errors.extend(check_leak_scan_alive(run_dir, code))
    errors.extend(check_no_direct_load_data(code))
    errors.extend(check_scaffold_fit_and_score(code))
    errors.extend(check_hyperparameters_consumption(run_dir, trial_dir, code))
    errors.extend(check_undefined_names(code))
    errors.extend(check_generated_code_consistency(code))
    errors.extend(check_function_scope_shadowing(code))
    errors.extend(check_runtime_import_bootstrap(code))
    errors.extend(check_widen_cache(code))
    errors.extend(check_widen_cache_provenance(code))
    errors.extend(check_reparse_feature_cache(run_dir, code))
    errors.extend(check_validation_score_output(code))
    errors.extend(check_structural_override(code))
    return errors


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python lint_trial.py <run_dir> <trial_dir_or_id>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1])
    trial_arg = sys.argv[2]
    trial_dir = Path(trial_arg)
    if not trial_dir.is_absolute():
        if not trial_dir.exists():
            trial_dir = run_dir / "trials" / trial_arg
    if not trial_dir.exists():
        print(f"trial_dir 不存在: {trial_dir}", file=sys.stderr)
        return 2

    # v2.16.0:spec.ensemble_enabled 不再授权融合(它由 Agent 起草)。
    # 融合/级联只能经用户 approve-topology 批准,由 run 路径的拓扑门把
    # 批准结果传给 lint_trial();独立自检一律按未批准处理。
    ensemble_enabled = False

    errors = lint_trial(run_dir, trial_dir, ensemble_enabled)
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return 1
    print(f"[PASS] {trial_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
