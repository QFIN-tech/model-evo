"""模型拓扑管控(v2.16.0):trial 默认单模型;级联与融合须人工批准。

用户裁决(2026-10-22):每个 trial 默认只有单一模型——不支持"训练一个或
多个子分作为特征"的级联模型,也不支持模型融合;要突破必须人介入确认。
hurdle(回归)与 T-learner(uplift)归为 structural,按任务类型白名单放行。

事故背景:全局库 rank 配方曾在 `_widen` 里用 5 折 OOF 训练 DL 模型,把
`seqtab_logit` 当特征喂给 XGBoost(级联)。lint 规则 2 是正则,只认"同
trial 两个 GBM 库 / rankdata / sklearn Voting…",这个形态一条都不命中;
放行开关 `ensemble_enabled` 又写在 Agent 自己起草的 spec 里。

拓扑四档:

- single:一个可训练模型(HPO inner、早停、confirm 的 seed 复现不算多模型);
- structural:多个子模型各自只吃原始特征,固定公式组合、无可学习组合器
  (hurdle = p·μ;T-learner = μ1−μ0);白名单见 STRUCTURAL_WHITELIST;
- cascade:任一模型的输出(predict/logit/OOF)成为另一个模型的输入或目标,
  或在特征管线(`_widen` / feature_block 内核 / trial_feature_blocks)里训练
  了模型;
- ensemble:多个模型的输出被平均/加权/rank-mean/投票,或用 sklearn 融合
  容器/AutoML。

本模块三部分:

1. ``analyze_code``:AST 静态检测(预测值污点 + 组合/喂给训练的汇点 +
   特征管线内训练)。按模块常量(MODEL_TYPE / _HPO_ALGO)折叠死分支——
   模板里 uplift 的 T-learner 分支对二分类 trial 是死代码,不能误判。
2. 批准账本 ``approvals/model-topology.jsonl``:哈希链,只由
   ``modelevo.py approve-topology``(交互式终端 + 随机确认码)写入;
   guard hook 拒绝 Agent 调用该命令或写账本。链断裂 ⇒ 全部批准作废。
3. ``evaluate_gate``:runner 执行 trial 前的唯一裁决点(immutable harness,
   不经工作区 lint 副本,不受 MODELEVO_SKIP_LINT 影响)。

静态检测的边界(诚实标注):它识别的是常见写法,不是证明;运行期拟合
审计(fit-ledger)与产物核对是后续层,不在本版。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from .logging_utils import get_logger, now_iso

logger = get_logger("modelevo.topology")

KINDS = ("single", "structural", "cascade", "ensemble")

#: 任务类型 → 免批准的结构型组合(用户裁决 2026-10-22)。写死在 Harness:
#: 白名单是放行面,不允许 run 配置扩大。
STRUCTURAL_WHITELIST: dict[str, frozenset] = {
    "regression": frozenset({"hurdle"}),
    "uplift": frozenset({"t_learner"}),
}

LEDGER_REL = Path("approvals") / "model-topology.jsonl"
REQUESTS_REL = Path("approvals") / "requests"
APPROVAL_SCOPES = ("trial", "recipe", "run")
#: run / wait 在拓扑门拦截时使用的退出码(与 wait 的 3/4 区分)
EXIT_APPROVAL_REQUIRED = 5


# ===========================================================================
# 1. 静态检测
# ===========================================================================

_PRED_ATTRS = frozenset({"predict", "predict_proba", "predict_log_proba",
                         "decision_function", "inplace_predict"})
_FIT_ATTRS = frozenset({"fit", "partial_fit"})
_AGG_NAMES = frozenset({"mean", "average", "median", "nanmean", "nanmedian",
                        "sum", "nansum", "fmean"})
_STACK_NAMES = frozenset({"vstack", "stack", "column_stack", "hstack",
                          "array", "asarray", "concatenate", "dstack"})
_GBM_MODULES = {"lightgbm": "lightgbm", "xgboost": "xgboost",
                "catboost": "catboost"}
_LEARNER_CLASS_RE = re.compile(
    r"^(?:LGBM\w*|XGB\w*|CatBoost\w*|"
    r"LogisticRegression\w*|LinearRegression|Ridge\w*|Lasso\w*|"
    r"ElasticNet\w*|SGD(?:Classifier|Regressor)|Perceptron|"
    r"(?:RandomForest|ExtraTrees|GradientBoosting|HistGradientBoosting|"
    r"DecisionTree|AdaBoost|KNeighbors|MLP)(?:Classifier|Regressor)|"
    r"SVC|SVR|NuSVC|NuSVR|LinearSVC|LinearSVR|GaussianNB|BernoulliNB|"
    r"MultinomialNB|IsotonicRegression|CalibratedClassifierCV|TabNet\w*)$")
#: 本身就是"多模型组合器"的容器/框架:出现即 ensemble
_ENSEMBLE_CLASS_RE = re.compile(
    r"^(?:(?:Voting|Stacking|Bagging)(?:Classifier|Regressor)|SeedBag|"
    r"TabularPredictor)$")
#: 返回标量指标的函数:预测值经过它们不再是"预测"
_METRIC_FN_RE = re.compile(
    r"(?:^|_)(?:auc|roc_auc_score|log_loss|logloss|metric|loss|gini|ks|"
    r"rmse|mse|mae|mape|r2|r2_score|corr|corrcoef|spearmanr|pearsonr|qini|"
    r"auuc|average_precision_score|accuracy_score|f1_score)(?:$|_)",
    re.I)
_SCALAR_BUILTINS = frozenset({"len", "print", "int", "float", "bool", "str",
                              "repr", "isinstance", "type", "hash", "id",
                              "round", "min", "max", "any", "all"})
#: CUDA 编译探测类函数(64x4 小样本 fit,不构成训练)
_PROBE_FN_RE = re.compile(r"cuda_ok|cuda_built|_probe|gpu_ok|gpu_capab", re.I)
#: 特征管线入口:runtime 按名字调用(widen_cached 动态取 `_widen`)
_FEATURE_ENTRY_NAMES = frozenset({"trial_feature_blocks", "_widen",
                                  "_bundle_widen"})
_FEATURE_WRAPPERS = frozenset({"feature_block", "widen_cached"})
#: v2.17.0:官方结构型容器类名 → structural 形态(构造即识别)
_OFFICIAL_STRUCTURAL = {"HurdleRegressor": "hurdle"}


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001
        return type(node).__name__


class _Origin(str):
    """预测值来源:'<接收者>#<版本>',末尾带 '*' 表示可能代表多个模型
    (循环内创建/循环变量/推导式变量)。"""

    @property
    def multi(self) -> bool:
        return self.endswith("*")


class _ModuleFacts:
    """模块级事实:常量、GBM 别名、函数表、调用图、训练点。"""

    def __init__(self, tree: ast.Module):
        self.tree = tree
        self.consts: dict[str, Any] = {}
        self.module_names: set[str] = set()
        self.gbm_alias: dict[str, str] = {}
        self.probe_alias: set[str] = set()
        self.funcs: dict[str, ast.AST] = {}
        self.func_class: dict[str, str | None] = {}
        for node in tree.body:
            for tgt in self._assign_targets(node):
                self.module_names.add(tgt)
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.Constant):
                self.consts[node.targets[0].id] = node.value.value
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[node.name] = node
                self.func_class[node.name] = None
            elif isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef,
                                        ast.AsyncFunctionDef)):
                        key = f"{node.name}.{sub.name}"
                        self.funcs[key] = sub
                        self.func_class[key] = node.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    root = al.name.split(".")[0]
                    name = al.asname or root
                    if root in _GBM_MODULES:
                        self.gbm_alias[name] = _GBM_MODULES[root]
                    if "_probe" in name:
                        self.probe_alias.add(name)
            elif isinstance(node, ast.ImportFrom):
                for al in node.names:
                    if "_probe" in (al.asname or ""):
                        self.probe_alias.add(al.asname)

    @staticmethod
    def _assign_targets(node) -> list[str]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            return [node.name]
        if isinstance(node, ast.Assign):
            return [n.id for t in node.targets for n in ast.walk(t)
                    if isinstance(n, ast.Name)]
        if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and \
                isinstance(node.target, ast.Name):
            return [node.target.id]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return [a.asname or a.name.split(".")[0] for a in node.names]
        return []

    # ---- 静态求值(死分支折叠)----
    def static_value(self, node: ast.AST):
        """返回 (known, value)。只认模块常量与 globals().get(...)。"""
        if isinstance(node, ast.Constant):
            return True, node.value
        if isinstance(node, ast.Name) and node.id in self.consts:
            return True, self.consts[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func,
                                                     ast.Attribute) \
                and node.func.attr == "get" \
                and isinstance(node.func.value, ast.Call) \
                and _call_name(node.func.value.func) == "globals" \
                and node.args and isinstance(node.args[0], ast.Constant):
            name = node.args[0].value
            if name in self.consts:
                return True, self.consts[name]
            if name not in self.module_names:
                return True, (node.args[1].value if len(node.args) > 1
                              and isinstance(node.args[1], ast.Constant)
                              else None)
        return False, None

    def static_truth(self, test: ast.AST) -> bool | None:
        if isinstance(test, ast.BoolOp):
            vals = [self.static_truth(v) for v in test.values]
            if isinstance(test.op, ast.And):
                if any(v is False for v in vals):
                    return False
                return True if all(v is True for v in vals) else None
            if any(v is True for v in vals):
                return True
            return False if all(v is False for v in vals) else None
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            v = self.static_truth(test.operand)
            return None if v is None else (not v)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op = test.ops[0]
            right = test.comparators[0]
            # "_X" in globals()
            if isinstance(op, (ast.In, ast.NotIn)) \
                    and isinstance(test.left, ast.Constant) \
                    and isinstance(right, ast.Call) \
                    and _call_name(right.func) == "globals":
                present = test.left.value in self.module_names
                return present if isinstance(op, ast.In) else not present
            kl, lv = self.static_value(test.left)
            if not kl:
                return None
            if isinstance(op, (ast.In, ast.NotIn)) and isinstance(
                    right, (ast.Tuple, ast.List, ast.Set)):
                items = []
                for e in right.elts:
                    ke, ve = self.static_value(e)
                    if not ke:
                        return None
                    items.append(ve)
                res = lv in items
                return res if isinstance(op, ast.In) else not res
            kr, rv = self.static_value(right)
            if not kr:
                return None
            if isinstance(op, ast.Eq):
                return lv == rv
            if isinstance(op, ast.NotEq):
                return lv != rv
            return None
        known, val = self.static_value(test)
        return bool(val) if known else None


class _FuncAnalyzer:
    """单函数的预测值污点分析(语句序,循环体走两遍)。"""

    def __init__(self, mod: _ModuleFacts, key: str, node: ast.AST,
                 summaries: dict[str, dict], record: bool):
        self.mod = mod
        self.key = key
        self.node = node
        self.summaries = summaries
        self.record = record
        self.env: dict[str, frozenset] = {}
        self.pred_lists: set[str] = set()
        self.version: dict[str, int] = {}
        self.loop_assigned: set[str] = set()
        self.iter_names: set[str] = set()
        self.loop_depth = 0
        self.returns_pred = False
        self.trains: list[tuple[str, int]] = []
        self.calls: set[str] = set()
        self.preds_made = False
        self.signals: list[dict] = []
        self.combos: list[dict] = []
        self.param_names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            self.param_names = [p.arg for p in (a.posonlyargs + a.args
                                                + a.kwonlyargs)]
            if self.param_names and self.param_names[0] in ("self", "cls"):
                self.param_names = self.param_names[1:]
        self.param_sink: set[int] = set()
        self.param_origin = {p: _Origin(f"param:{p}")
                             for p in self.param_names}

    # ---- 工具 ----
    def _sig(self, kind: str, rule: str, node: ast.AST, detail: str):
        if not self.record:
            return
        self.signals.append({"kind": kind, "rule": rule,
                             "line": getattr(node, "lineno", None),
                             "function": self.key, "detail": detail[:160]})

    def _origin_for_receiver(self, recv: ast.AST) -> _Origin:
        root = _root_name(recv)
        text = _unparse(recv)
        if root is not None and root in self.iter_names:
            return _Origin(f"{text}*")
        if isinstance(recv, ast.Name):
            v = self.version.get(recv.id, 0)
            multi = recv.id in self.loop_assigned
            return _Origin(f"{recv.id}#{v}{'*' if multi else ''}")
        return _Origin(text)

    def _bind(self, name: str, taint: frozenset):
        self.version[name] = self.version.get(name, 0) + 1
        if self.loop_depth:
            self.loop_assigned.add(name)
        self.env[name] = taint

    @staticmethod
    def _distinct(t: Iterable[str]) -> set[str]:
        return {o for o in t if not str(o).startswith("param:")}

    def _combo(self, op: str, node: ast.AST, origins: set[str]):
        if not self.record:
            return
        self.combos.append({"op": op, "line": getattr(node, "lineno", None),
                            "function": self.key,
                            "origins": sorted(origins)[:6]})

    def _learner_ctor(self, call: ast.Call) -> str | None:
        """learner 构造/原生训练 → 族名;CUDA 探测别名不算。"""
        func = call.func
        name = _call_name(func)
        root = _root_name(func)
        if root in self.mod.probe_alias:
            return None
        if isinstance(func, ast.Attribute) and func.attr == "train" \
                and isinstance(func.value, ast.Name) \
                and func.value.id in self.mod.gbm_alias:
            return self.mod.gbm_alias[func.value.id]
        if _ENSEMBLE_CLASS_RE.match(name):
            return "ensemble_container"
        if _LEARNER_CLASS_RE.match(name):
            if name.startswith("LGBM"):
                return "lightgbm"
            if name.startswith("XGB"):
                return "xgboost"
            if name.startswith("CatBoost"):
                return "catboost"
            if name.startswith("TabNet"):
                return "torch"
            return "sklearn"
        if name == "build_gbm":
            return "gbm"
        return None

    # ---- 表达式污点 ----
    def taint(self, node: ast.AST | None) -> frozenset:
        if node is None:
            return frozenset()
        if isinstance(node, ast.Name):
            out = self.env.get(node.id, frozenset())
            if node.id in self.param_origin and node.id not in self.version:
                out = out | {self.param_origin[node.id]}
            return out
        if isinstance(node, ast.Attribute):
            key = _unparse(node)
            if key in self.env:
                return self.env[key]
            return self.taint(node.value) if not isinstance(
                node.value, ast.Name) or node.value.id not in ("self",) \
                else frozenset()
        if isinstance(node, ast.Subscript):
            return self.taint(node.value)
        if isinstance(node, ast.Starred):
            return self.taint(node.value)
        if isinstance(node, ast.Await):
            return self.taint(node.value)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            out = frozenset()
            for e in node.elts:
                out |= self.taint(e)
            return out
        if isinstance(node, ast.Dict):
            out = frozenset()
            for v in node.values:
                out |= self.taint(v)
            return out
        if isinstance(node, ast.IfExp):
            return self.taint(node.body) | self.taint(node.orelse)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
            return self._comp_taint(node.generators, node.elt)
        if isinstance(node, ast.DictComp):
            return self._comp_taint(node.generators, node.value)
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.UnaryOp):
            return self.taint(node.operand)
        if isinstance(node, ast.Call):
            return self._call(node)
        return frozenset()

    def _comp_taint(self, generators, elt) -> frozenset:
        added = []
        for g in generators:
            for n in ast.walk(g.target):
                if isinstance(n, ast.Name) and n.id not in self.iter_names:
                    self.iter_names.add(n.id)
                    added.append(n.id)
        try:
            out = self.taint(elt)
            for g in generators:
                out |= frozenset(o for o in self.taint(g.iter)
                                 if _Origin(o).multi)
        finally:
            for n in added:
                self.iter_names.discard(n)
        return out

    def _binop(self, node: ast.BinOp) -> frozenset:
        lt, rt = self.taint(node.left), self.taint(node.right)
        out = lt | rt
        if lt and rt:
            ld, rd = self._distinct(lt), self._distinct(rt)
            if ld and rd and (ld | rd) != ld & rd and len(ld | rd) >= 2:
                op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mult",
                      ast.Div: "div"}.get(type(node.op))
                if op:
                    self._combo(op, node, ld | rd)
        if isinstance(node.op, ast.Add) and self.loop_depth \
                and any(_Origin(o).multi for o in self._distinct(out)):
            self._combo("accumulate", node, self._distinct(out))
        return out

    def _is_pred_collection(self, node: ast.AST) -> bool:
        """多个预测的集合:预测列表变量 / 含 ≥2 来源或多模型来源的字面量
        或推导式 / 它们的 stack。"""
        if isinstance(node, ast.Name):
            return node.id in self.pred_lists
        if isinstance(node, (ast.List, ast.Tuple, ast.ListComp,
                             ast.GeneratorExp)):
            d = self._distinct(self.taint(node))
            return len(d) >= 2 or any(_Origin(o).multi for o in d)
        if isinstance(node, ast.Call) and _call_name(node.func) \
                in _STACK_NAMES and node.args:
            return self._is_pred_collection(node.args[0])
        return False

    def _call(self, node: ast.Call) -> frozenset:
        func = node.func
        name = _call_name(func)
        # 预测源
        if isinstance(func, ast.Attribute) and func.attr in _PRED_ATTRS:
            self.preds_made = True
            for a in node.args:
                self.taint(a)
            return frozenset({self._origin_for_receiver(func.value)})
        # 训练点
        fam = self._learner_ctor(node)
        if fam is not None:
            self.trains.append((fam, node.lineno))
            if fam == "ensemble_container":
                self._sig("ensemble", "ensemble_container", node,
                          f"{name}(...)")
        is_fit = isinstance(func, ast.Attribute) and func.attr in _FIT_ATTRS
        is_native_train = isinstance(func, ast.Attribute) \
            and func.attr == "train" and isinstance(func.value, ast.Name) \
            and func.value.id in self.mod.gbm_alias
        if isinstance(func, ast.Attribute) and func.attr == "backward":
            self.trains.append(("torch", node.lineno))
        arg_taints = [self.taint(a) for a in node.args]
        kw_taints = {k.arg: self.taint(k.value) for k in node.keywords}
        # 汇点 1:预测值喂给训练(特征或目标)
        callee_key = None
        if isinstance(func, ast.Name) and func.id in self.mod.funcs:
            callee_key = func.id
        elif isinstance(func, ast.Attribute) and isinstance(
                func.value, ast.Name) and func.value.id == "self":
            cls = self.mod.func_class.get(self.key)
            if cls and f"{cls}.{func.attr}" in self.mod.funcs:
                callee_key = f"{cls}.{func.attr}"
        if callee_key:
            self.calls.add(callee_key)
        if isinstance(func, ast.Call):
            self.taint(func)   # `f(x)(y)`:内层调用也要走一遍
        if is_fit or is_native_train:
            for i, t in enumerate(arg_taints[:2]):
                self._fit_sink(t, node, "X" if i == 0 else "y",
                               _unparse(func))
            for k in ("X", "y", "dtrain", "train_set"):
                if kw_taints.get(k):
                    self._fit_sink(kw_taints[k], node, k, _unparse(func))
        elif callee_key and callee_key in self.summaries:
            sinks = self.summaries[callee_key].get("param_sink", set())
            params = self.summaries[callee_key].get("params", [])
            for i, t in enumerate(arg_taints):
                if i in sinks:
                    self._fit_sink(t, node, params[i] if i < len(params)
                                   else str(i), callee_key)
            for k, t in kw_taints.items():
                if k in params and params.index(k) in sinks:
                    self._fit_sink(t, node, k, callee_key)
        # 聚合:多个模型的预测求均值/加和
        if name in _AGG_NAMES and node.args and \
                self._is_pred_collection(node.args[0]):
            self._combo("aggregate", node,
                        self._distinct(self.taint(node.args[0])))
        if isinstance(func, ast.Attribute) and name in _AGG_NAMES \
                and self._is_pred_collection(func.value):
            self._combo("aggregate", node,
                        self._distinct(self.taint(func.value)))
        # 列表累积:循环里 .append(预测) ⇒ 预测集合
        if isinstance(func, ast.Attribute) and func.attr in (
                "append", "extend") and isinstance(func.value, ast.Name) \
                and node.args:
            t = arg_taints[0]
            if t:
                base = func.value.id
                real = self._distinct(t)
                extra = frozenset({_Origin(f"{base}@loop*")}) \
                    if (self.loop_depth and real) else frozenset()
                self.env[base] = self.env.get(base, frozenset()) | t | extra
                if real:
                    self.pred_lists.add(base)
            return frozenset()
        # 净化:指标/标量函数
        if name in _SCALAR_BUILTINS or _METRIC_FN_RE.search(name or ""):
            return frozenset()
        # 跨函数:返回预测的模块函数
        if callee_key and self.summaries.get(callee_key, {}).get(
                "returns_pred"):
            out = frozenset({_Origin(f"call:{callee_key}@{node.lineno}"
                                     f"{'*' if self.loop_depth else ''}")})
            for t in arg_taints:
                out |= frozenset(o for o in t if str(o).startswith("param:"))
            return out
        out = frozenset()
        if isinstance(func, ast.Attribute):
            out |= self.taint(func.value)
        for t in arg_taints:
            out |= t
        for t in kw_taints.values():
            out |= t
        return out

    def _fit_sink(self, t: frozenset, node: ast.AST, which: str,
                  callee: str):
        if not t:
            return
        for o in t:
            if str(o).startswith("param:"):
                p = str(o)[len("param:"):]
                if p in self.param_names:
                    self.param_sink.add(self.param_names.index(p))
        if self._distinct(t):
            rule = "prediction_as_target" if which in ("y",) \
                else "prediction_as_feature"
            self._sig("cascade", rule, node,
                      f"预测值作为 {callee} 的 {which}")

    # ---- 语句 ----
    def run(self):
        body = self.node.body if hasattr(self.node, "body") else []
        # 调用图静态收集(含 `f(x)(y)`、回调传参等不经污点路径的形态)
        cls = self.mod.func_class.get(self.key)
        for stmt in body:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and n.id in self.mod.funcs:
                    self.calls.add(n.id)
                elif cls and isinstance(n, ast.Attribute) and isinstance(
                        n.value, ast.Name) and n.value.id == "self" and \
                        f"{cls}.{n.attr}" in self.mod.funcs:
                    self.calls.add(f"{cls}.{n.attr}")
        self._block(body)
        return self

    def _block(self, stmts):
        for s in stmts:
            self._stmt(s)

    def _stmt(self, s: ast.stmt):
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef,
                          ast.ClassDef)):
            return  # 嵌套定义单独分析(模块级函数/方法)或忽略
        if isinstance(s, ast.If):
            truth = self.mod.static_truth(s.test)
            self.taint(s.test)
            if truth is True:
                self._block(s.body)
            elif truth is False:
                self._block(s.orelse)
            else:
                self._block(s.body)
                self._block(s.orelse)
            return
        if isinstance(s, (ast.For, ast.AsyncFor, ast.While)):
            if isinstance(s, (ast.For, ast.AsyncFor)):
                self.taint(s.iter)
                tnames = [n.id for n in ast.walk(s.target)
                          if isinstance(n, ast.Name)]
            else:
                self.taint(s.test)
                tnames = []
            self.loop_depth += 1
            added = [n for n in tnames if n not in self.iter_names]
            self.iter_names.update(added)
            saved_record = self.record
            try:
                self.record = False
                self._block(s.body)       # 第一遍:收敛循环携带的污点
                self.record = saved_record
                self._block(s.body)
            finally:
                self.record = saved_record
                self.loop_depth -= 1
                for n in added:
                    self.iter_names.discard(n)
            self._block(s.orelse)
            return
        if isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                self.taint(it.context_expr)
            self._block(s.body)
            return
        if isinstance(s, ast.Try):
            self._block(s.body)
            for h in s.handlers:
                self._block(h.body)
            self._block(s.orelse)
            self._block(s.finalbody)
            return
        if isinstance(s, ast.Return):
            if s.value is not None and self._distinct(self.taint(s.value)):
                self.returns_pred = True
            return
        if isinstance(s, ast.Assign):
            t = self.taint(s.value)
            is_coll = self._is_pred_collection(s.value)
            for tgt in s.targets:
                self._assign(tgt, t, is_coll)
            return
        if isinstance(s, ast.AnnAssign):
            if s.value is not None:
                self._assign(s.target, self.taint(s.value),
                             self._is_pred_collection(s.value))
            return
        if isinstance(s, ast.AugAssign):
            t = self.taint(s.value)
            cur = self.taint(s.target)
            if isinstance(s.op, (ast.Add, ast.Sub)) and t:
                d = self._distinct(t)
                if self.loop_depth and d:
                    self._combo("accumulate", s, d | self._distinct(cur))
                elif cur and d and (self._distinct(cur) | d) != d:
                    self._combo("add" if isinstance(s.op, ast.Add)
                                else "sub", s, self._distinct(cur) | d)
            if isinstance(s.target, ast.Name):
                self.env[s.target.id] = cur | t
            elif isinstance(s.target, ast.Subscript):
                base = _root_name(s.target)
                if base:
                    self.env[base] = self.env.get(base, frozenset()) | t
            return
        if isinstance(s, ast.Expr):
            self.taint(s.value)
            return
        for child in ast.iter_child_nodes(s):
            if isinstance(child, ast.expr):
                self.taint(child)

    def _assign(self, tgt: ast.AST, t: frozenset, is_coll: bool):
        if isinstance(tgt, ast.Name):
            self._bind(tgt.id, t)
            if is_coll:
                self.pred_lists.add(tgt.id)
            else:
                self.pred_lists.discard(tgt.id)
        elif isinstance(tgt, (ast.Tuple, ast.List)):
            for e in tgt.elts:
                self._assign(e, t, False)
        elif isinstance(tgt, ast.Starred):
            self._assign(tgt.value, t, False)
        elif isinstance(tgt, ast.Subscript):
            base = _root_name(tgt)
            if base:
                self.env[base] = self.env.get(base, frozenset()) | t
        elif isinstance(tgt, ast.Attribute):
            self.env[_unparse(tgt)] = t


def _module_body_as_func(tree: ast.Module) -> ast.AST:
    body = [n for n in tree.body if not isinstance(
        n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return ast.Module(body=body, type_ignores=[])


def analyze_code(code: str) -> dict:
    """静态检测 trial 代码的模型拓扑。返回::

        {"kind", "kinds", "structural_forms", "signals", "notes",
         "combos", "families", "signature", "parse_error"}

    signals 每条 {kind, rule, line, function, detail};kind ∈ cascade/
    ensemble/structural。signature 是去掉行号的规则集合(批准按它比对)。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"kind": "single", "kinds": ["single"],
                "structural_forms": [], "signals": [], "notes": [],
                "combos": [], "families": [], "signature": [],
                "parse_error": f"SyntaxError: {exc}"}
    mod = _ModuleFacts(tree)
    keys = [k for k in mod.funcs
            if not _PROBE_FN_RE.search(k.split(".")[-1])]
    # 函数摘要不动点(returns_pred / param_sink 跨函数传播)
    summaries: dict[str, dict] = {}
    for _ in range(4):
        changed = False
        for k in keys:
            fa = _FuncAnalyzer(mod, k, mod.funcs[k], summaries,
                               record=False).run()
            new = {"returns_pred": fa.returns_pred,
                   "param_sink": set(fa.param_sink),
                   "params": fa.param_names}
            if summaries.get(k) != new:
                summaries[k] = new
                changed = True
        if not changed:
            break
    analyzers: dict[str, _FuncAnalyzer] = {}
    for k in keys:
        analyzers[k] = _FuncAnalyzer(mod, k, mod.funcs[k], summaries,
                                     record=True).run()
    analyzers["<module>"] = _FuncAnalyzer(
        mod, "<module>", _module_body_as_func(tree), summaries,
        record=True).run()

    signals: list[dict] = []
    combos: list[dict] = []
    for fa in analyzers.values():
        signals.extend(fa.signals)
        combos.extend(fa.combos)

    # 特征管线内训练:入口按名字 + feature_block/widen_cached 的回调参数
    entries = {k for k in keys if k in _FEATURE_ENTRY_NAMES}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and \
                _call_name(node.func) in _FEATURE_WRAPPERS:
            for a in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(a, ast.Name) and a.id in analyzers:
                    entries.add(a.id)
    reach, stack = set(), list(entries)
    while stack:
        k = stack.pop()
        if k in reach or k not in analyzers:
            continue
        reach.add(k)
        stack.extend(analyzers[k].calls)
    notes: list[dict] = []
    for k in sorted(reach):
        fa = analyzers[k]
        fams = [f for f, _ in fa.trains if f != "ensemble_container"]
        if not fams:
            continue
        line = fa.trains[0][1]
        if "torch" in fams or fa.preds_made or fa.returns_pred:
            signals.append({
                "kind": "cascade", "rule": "model_trained_in_feature_pipeline",
                "line": line, "function": k,
                "detail": f"特征管线函数 {k} 内训练模型({sorted(set(fams))})"})
        else:
            notes.append({"rule": "model_fit_in_feature_pipeline",
                          "line": line, "function": k,
                          "detail": f"特征管线函数 {k} 构造了 learner 但未产出"
                                    "预测(疑似特征选择),未作级联判定"})

    structural_forms: set[str] = set()
    # v2.17.0:官方结构型容器(runtime.hurdle.HurdleRegressor)—— p·max(μ,0)
    # 在 harness 模块内组合,trial 代码里看不到"两个预测相乘",按构造点识别为
    # structural:hurdle(回归白名单,免批准;两个子模型原型作为构造参数传入
    # 不构成级联/融合)。
    _in_funcs = {id(n) for f in mod.funcs.values() for n in ast.walk(f)}
    for fname, fnode in [("<module>", tree)] + sorted(mod.funcs.items()):
        for node in ast.walk(fnode):
            if not (isinstance(node, ast.Call) and _call_name(
                    node.func) in _OFFICIAL_STRUCTURAL):
                continue
            if fname == "<module>" and id(node) in _in_funcs:
                continue
            form = _OFFICIAL_STRUCTURAL[_call_name(node.func)]
            structural_forms.add(form)
            signals.append({
                "kind": "structural", "rule": form,
                "line": getattr(node, "lineno", None), "function": fname,
                "detail": f"官方结构型容器 {_call_name(node.func)}"
                          "(runtime.hurdle;score = p·max(μ,0))"})
    for c in combos:
        if c["op"] in ("add", "aggregate", "accumulate", "div"):
            signals.append({"kind": "ensemble", "rule": f"combine_{c['op']}",
                            "line": c["line"], "function": c["function"],
                            "detail": "多个模型的预测被组合: "
                                      + ", ".join(c["origins"])})
        elif c["op"] == "mult":
            structural_forms.add("hurdle")
            signals.append({"kind": "structural", "rule": "hurdle",
                            "line": c["line"], "function": c["function"],
                            "detail": "两个子模型预测相乘(hurdle p·μ): "
                                      + ", ".join(c["origins"])})
        elif c["op"] == "sub":
            structural_forms.add("t_learner")
            signals.append({"kind": "structural", "rule": "t_learner",
                            "line": c["line"], "function": c["function"],
                            "detail": "两个子模型预测相减(T-learner μ1−μ0): "
                                      + ", ".join(c["origins"])})

    families = sorted({f for fa in analyzers.values() for f, _ in fa.trains
                       if f not in ("ensemble_container", "gbm")})
    if len([f for f in families if f != "sklearn"]) >= 2:
        notes.append({"rule": "multi_family", "line": None,
                      "function": None,
                      "detail": f"代码中出现多个模型族 {families}(静态可达,"
                                "不等于实际训练;仅作提示)"})

    # 去重(同函数同规则同行)
    seen, uniq = set(), []
    for s in signals:
        k = (s["kind"], s["rule"], s.get("function"), s.get("line"))
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    signals = sorted(uniq, key=lambda s: (s.get("line") or 0, s["rule"]))
    kinds = sorted({s["kind"] for s in signals}) or ["single"]
    if "cascade" in kinds or "ensemble" in kinds:
        kind = "+".join(k for k in ("cascade", "ensemble") if k in kinds)
    elif "structural" in kinds:
        kind = "structural"
    else:
        kind = "single"
    return {"kind": kind, "kinds": kinds,
            "structural_forms": sorted(structural_forms),
            "signals": signals, "notes": notes, "combos": combos,
            "families": families,
            "signature": signature_of(signals), "parse_error": None}


def signature_of(signals: Iterable[dict]) -> list[str]:
    return sorted({f"{s['kind']}:{s['rule']}" for s in signals})


# ===========================================================================
# 2. 批准账本(哈希链)
# ===========================================================================

def ledger_path(run_dir: Path) -> Path:
    return Path(run_dir) / LEDGER_REL


def _record_sha(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "sha"}
    return hashlib.sha256(json.dumps(body, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")
                          ).hexdigest()


def load_ledger(run_dir: Path) -> dict:
    """读取并校验账本。链断裂 ⇒ chain_valid=False(调用方按无批准处理)。"""
    path = ledger_path(run_dir)
    out = {"records": [], "chain_valid": True, "error": None,
           "path": str(path)}
    if not path.exists():
        return out
    prev = "genesis"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        out.update(chain_valid=False, error=f"账本不可读: {exc}")
        return out
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            out.update(chain_valid=False, error=f"第 {i} 行不是合法 JSON")
            return out
        if rec.get("prev_sha") != prev or rec.get("sha") != _record_sha(rec):
            out.update(chain_valid=False,
                       error=f"第 {i} 行哈希链断裂(记录被改写或手工追加)")
            return out
        prev = rec["sha"]
        out["records"].append(rec)
    return out


def active_approvals(run_dir: Path) -> tuple[list[dict], dict]:
    led = load_ledger(run_dir)
    if not led["chain_valid"]:
        return [], led
    revoked = {r.get("revokes") for r in led["records"]
               if r.get("action") == "revoke"}
    return ([r for r in led["records"] if r.get("action") == "approve"
             and r.get("approval_id") not in revoked], led)


def append_ledger(run_dir: Path, record: dict) -> dict:
    """追加一条记录(批准/撤销)。仅供 approve-topology 命令调用。"""
    led = load_ledger(run_dir)
    if not led["chain_valid"]:
        raise RuntimeError(f"批准账本哈希链已断裂,拒绝追加: {led['error']}"
                           "(先人工核查 approvals/model-topology.jsonl)")
    recs = led["records"]
    rec = dict(record)
    rec["seq"] = len(recs) + 1
    if rec.get("action") == "approve":
        rec["approval_id"] = f"apr-{rec['seq']:03d}"
    rec.setdefault("at", now_iso())
    rec["prev_sha"] = recs[-1]["sha"] if recs else "genesis"
    rec["sha"] = _record_sha(rec)
    path = ledger_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


# ===========================================================================
# 3. 门
# ===========================================================================

def code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _trial_code(trial) -> str | None:
    p = Path(trial.trial_dir) / "code" / str(
        (trial.data.get("execution") or {}).get("code_entry") or "train.py")
    if not p.is_file():
        p = Path(trial.trial_dir) / "code" / "train.py"
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


def _ancestors(run_dir: Path, trial) -> set[str]:
    from .trial import TrialRecord

    seen: set[str] = set()
    stack = list(trial.parent_ids or [])
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pdir = Path(run_dir) / "trials" / pid
        if (pdir / "trial.yaml").exists():
            try:
                stack.extend(TrialRecord.load(pdir).parent_ids or [])
            except Exception:  # noqa: BLE001
                pass
    return seen


def _needs(analysis: dict, model_type: str | None,
           declared: str = "single") -> list[str]:
    """需要人工批准的项:检测到的 ∪ Agent 声明的(new-trial --topology)。

    声明只能**增加**需批准项——检测是底线,声明 single 不能抵消检测。
    """
    allow = STRUCTURAL_WHITELIST.get(str(model_type or ""), frozenset())
    needs = []
    for k in ("cascade", "ensemble"):
        if k in analysis["kinds"] or declared == k:
            needs.append(k)
    forms = list(analysis["structural_forms"])
    if declared.startswith("structural:"):
        forms.append(declared.split(":", 1)[1])
    for form in dict.fromkeys(forms):
        if form not in allow:
            needs.append(f"structural:{form}")
    return needs


DECLARABLE = ("single", "structural:hurdle", "structural:t_learner",
              "cascade", "ensemble")


def _approval_covers(appr: dict, trial, needs: list[str], signature: list,
                     sha: str, ancestors: set[str]) -> bool:
    """单条批准是否覆盖本 trial 的全部需批准项。

    run:按种类;trial:同 trial + 同代码 sha + 签名不超出;recipe:本 trial
    或其祖先的批准 + 签名不超出(confirm/hpo 子代沿用,新增组合方式须重批)。
    """
    kinds_ok = {n.split(":")[0] for n in needs} <= set(appr.get("kinds") or [])
    if not kinds_ok:
        return False
    scope = appr.get("scope")
    if scope == "run":
        return True
    sig_ok = set(signature) <= set(appr.get("signature") or [])
    if scope == "trial":
        return appr.get("trial_id") == trial.trial_id \
            and appr.get("code_sha256") == sha and sig_ok
    if scope == "recipe":
        return (appr.get("trial_id") == trial.trial_id
                or appr.get("trial_id") in ancestors) and sig_ok
    return False


def approve_command(run_dir: Path, trial_id: str | None = None) -> str:
    cli = Path(__file__).resolve().parents[1] / "scripts" / "modelevo.py"
    tgt = f"--trial {trial_id}" if trial_id else "--scope run --kind <kind>"
    return (f"python {cli} approve-topology --run {Path(run_dir).resolve()} "
            f"{tgt} --reason \"<批准理由>\"")


def evaluate_gate(run_dir: Path, trial, model_type: str | None, *,
                  code: str | None = None) -> dict:
    """trial 执行前的拓扑裁决。返回门结果(写入 trial.yaml.model_topology)。

    decision:single / whitelisted / approved / exempt_baseline / blocked。
    allowed=False 时调用方必须拒跑(不冻结、不训练)。
    """
    code = _trial_code(trial) if code is None else code
    if code is None:
        return {"decision": "no_code", "allowed": True, "kind": "single",
                "kinds": ["single"], "signals": [], "needs": []}
    an = analyze_code(code)
    sha = code_sha256(code)
    declared = str(trial.data.get("declared_topology") or "single")
    needs = _needs(an, model_type, declared)
    gate = {
        "kind": an["kind"], "kinds": an["kinds"], "declared": declared,
        "structural_forms": an["structural_forms"],
        "signals": an["signals"], "notes": an["notes"],
        "families": an["families"], "signature": an["signature"],
        "code_sha256": sha, "needs": needs, "approvals": [],
        "ledger_valid": True, "checked_at": now_iso(),
        "parse_error": an["parse_error"],
    }
    if not needs:
        gate["decision"] = "whitelisted" if an["kind"] == "structural" \
            else "single"
        gate["allowed"] = True
        return gate
    approvals, led = active_approvals(run_dir)
    gate["ledger_valid"] = led["chain_valid"]
    if not led["chain_valid"]:
        gate["ledger_error"] = led["error"]
    anc = _ancestors(run_dir, trial)
    covering = [a for a in approvals
                if _approval_covers(a, trial, needs, an["signature"], sha,
                                    anc)]
    if covering:
        gate["decision"] = "approved"
        gate["allowed"] = True
        gate["approvals"] = [a["approval_id"] for a in covering]
        return gate
    if (trial.data.get("execution") or {}).get("baseline_recipe"):
        # baseline 由 Runtime 渲染/固化,不是 Agent 写码;只留痕不拦
        gate["decision"] = "exempt_baseline"
        gate["allowed"] = True
        return gate
    gate["decision"] = "blocked"
    gate["allowed"] = False
    gate["approve_command"] = approve_command(run_dir, trial.trial_id)
    return gate


def gate_message(trial_id: str, gate: dict) -> str:
    lines = [f"{trial_id}:模型拓扑 {gate.get('kind')} 未获人工批准"
             f"(需要: {', '.join(gate.get('needs') or [])})——不训练、"
             "不冻结源码,trial 保持 CODE_WRITTEN。"]
    for s in (gate.get("signals") or [])[:6]:
        lines.append(f"  - [{s['kind']}/{s['rule']}] L{s.get('line')} "
                     f"{s.get('function')}: {s.get('detail')}")
    if gate.get("ledger_valid") is False:
        lines.append(f"  ! 批准账本无效: {gate.get('ledger_error')}")
    lines.append("  出路:(a) 改写为单模型后重跑(源码尚未冻结);(b) 由**用户"
                 "在自己的终端**批准:" + str(gate.get("approve_command")))
    lines.append("  Agent 不得自行执行批准命令;按「运行中的裁决不阻塞算力」"
                 "继续其它方向,并已写入 logs/user-decisions.md。")
    return "\n".join(lines)


def record_request(run_dir: Path, trial, gate: dict) -> None:
    """拦截留痕:approvals/requests/<trial>.json + logs/user-decisions.md
    (PENDING,同 trial 同签名幂等)。"""
    run_dir = Path(run_dir)
    req_dir = run_dir / REQUESTS_REL
    req_dir.mkdir(parents=True, exist_ok=True)
    hyp = (trial.data.get("hypothesis") or {}).get("statement", "")
    payload = {"trial_id": trial.trial_id, "requested_at": now_iso(),
               "hypothesis": hyp, "gate": gate}
    (req_dir / f"{trial.trial_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sig_hash = hashlib.sha256(json.dumps(gate.get("signature") or [])
                              .encode()).hexdigest()[:10]
    marker = f"<!-- topology-request:{trial.trial_id}:{sig_hash} -->"
    ud = run_dir / "logs" / "user-decisions.md"
    ud.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = ud.read_text(encoding="utf-8") if ud.exists() else ""
    except OSError:
        old = ""
    if marker in old:
        return
    sig_lines = "\n".join(
        f"  - [{s['kind']}/{s['rule']}] L{s.get('line')} {s.get('function')}"
        f": {s.get('detail')}" for s in (gate.get("signals") or [])[:8])
    entry = (f"\n## [PENDING] 模型拓扑批准:{trial.trial_id} {marker}\n"
             f"- 时间:{now_iso()}\n"
             f"- 假设:{hyp}\n"
             f"- 检测拓扑:{gate.get('kind')}(需要批准:"
             f"{', '.join(gate.get('needs') or [])})\n{sig_lines}\n"
             "- 默认分支:不执行该 trial,演进按单模型方向继续\n"
             f"- 批准(用户在自己的终端执行):`{gate.get('approve_command')}`\n")
    with open(ud, "a", encoding="utf-8") as f:
        f.write(entry)


def composite_approved(gate: dict | None) -> bool:
    """lint 规则 2(工作区 ensemble 正则)是否放行:仅当拓扑门判 approved
    且批准覆盖 cascade/ensemble。spec.ensemble_enabled 不再授权。"""
    if not gate:
        return False
    return gate.get("decision") == "approved" and any(
        n in ("cascade", "ensemble") for n in gate.get("needs") or [])


def policy_summary(model_type: str | None) -> dict:
    return {"default": "single",
            "structural_whitelist": sorted(
                STRUCTURAL_WHITELIST.get(str(model_type or ""), ())),
            "requires_human_approval": ["cascade", "ensemble",
                                        "structural(白名单外)"],
            "approve_via": "approve-topology(仅用户在交互式终端执行)"}


def describe_for_human(gate: dict) -> str:
    """approve-topology 展示给人看的检测事实(不是 Agent 的描述)。"""
    lines = [f"检测拓扑:{gate.get('kind')}  需要批准:"
             f"{', '.join(gate.get('needs') or []) or '无'}",
             f"代码 sha256:{gate.get('code_sha256')}"]
    for s in gate.get("signals") or []:
        lines.append(f"  - [{s['kind']}/{s['rule']}] L{s.get('line')} "
                     f"{s.get('function')}: {s.get('detail')}")
    for n in gate.get("notes") or []:
        lines.append(f"  · 提示 {n['rule']}: {n['detail']}")
    return "\n".join(lines)


def env_identity() -> dict:
    import getpass
    import socket

    try:
        tty = os.ttyname(0)
    except OSError:
        tty = None
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = None
    return {"approver": user, "host": socket.gethostname(), "tty": tty}
