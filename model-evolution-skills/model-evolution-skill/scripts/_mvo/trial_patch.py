"""v2.19.0(#15)补丁式写码:`new-trial --from-parent <p> --patch <patch.py>`
与只读大纲 `trial-skeleton --trial <p>`。

背景:trial 的 train.py 2850~3700 行(最长 5536 行 / 160 KB),相对父代只改
3~5%,LLM 每代读写整份文件(代间空窗 30~108 分钟)。补丁式写码让 Agent 只写
**新增或替换的顶层定义**,harness 以父代 train.py 为底按 AST 合并:

- 同名顶层函数/类/单名常量 → 原位替换父代**最后一个**同名绑定(运行期生效的那个);
- 新名字 → 追加进 `# ==== v2.19.0 patch 段 BEGIN … END ====`(main 守卫之前);
- `PATCH_REGISTER_BLOCKS = ["_xxx_block", ...]` 声明的块包装函数自动注册进
  `trial_feature_blocks`(段尾包一层,原有块 + 新块按列右接,列重名报错;
  与 `--add-blocks` 同一注册表前缀 `_ADDED_FEATURE_BLOCKS_`,互相可见);
- 可选 `PATCH_REPLACES = [...]`:声明"本 patch 应替换的父代名字",与合并实际
  结果不一致(拼错名字变成新增 / 意外覆盖)即拒绝;
- harness 保护的符号(数据加载 / 评估 / 早停 / HPO 入口 / 设备 / 产物 / 特征块
  基础设施)不可替换也不可新增同名,见 PROTECTED_NAMES / PROTECTED_PREFIXES。

合并结果就是落进 trial code/ 的完整 train.py,之后照常走 Lint、拓扑门、源码
冻结与完整性 —— 被冻结/审计的永远是合并后的完整源码,patch 只是写法。
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .context import logger

PATCH_BEGIN = "# ==== v2.19.0 patch 段 BEGIN"
PATCH_END = "# ==== v2.19.0 patch 段 END ===="
#: patch 文件里允许的指令常量(其余 PATCH_* 名字一律拒绝,防拼写错误静默失效)
PATCH_DIRECTIVES = ("PATCH_REGISTER_BLOCKS", "PATCH_REPLACES")

# ---------------------------------------------------------------------------
# 受保护符号(黑名单):来源 = 模板(scripts/_mvo/templates.py)/ baseline 血缘
# (runtime/baseline.py)的 harness 段 + lint(assets/lint_trial.py)规则依赖的
# 符号 + render 里的 HPO/早停硬化符号(_HPO_REQUIRED_SYMBOLS / _HARD_FUNC_SOURCES)。
# 例外:`fit_and_score` 是模板明确留给 Agent 改造的训练入口(TODO(Agent)),
# 不在黑名单;`trial_feature_blocks` 走 PATCH_REGISTER_BLOCKS 增量注册。
# ---------------------------------------------------------------------------
_PROTECTED_GROUPS = {
    "数据加载/特征面/泄漏防线": (
        "RUN_DIR", "TRIAL_DIR", "SKILL_ROOT", "SPLITS", "LABELS_DIR",
        "REPARSE_DIR", "TARGET", "MODEL_TYPE", "TREATMENT", "SEED", "BANNED",
        "REPARSE_SOURCES", "_SPEC_NONFEATURE", "_ALGORITHM", "_NUM_STR_RE",
        "load_config", "load_labels", "load_data", "reparse_frame",
        "exclude_columns", "assert_banned_clean", "leak_scan",
        "safe_to_numeric"),
    "评估/产物/打分口径": (
        "main", "run_smoke", "SMOKE_MAX_ROWS", "SMOKE_MAX_VALID", "step", "_T0",
        "dump_json", "dump_model", "dump_eval_frame", "dump_shadow_score",
        "dump_feature_importance", "dump_learning_curve",
        "_load_treatment_for_metrics", "prepare_shadow_features",
        "_spec_primary_metric", "METRIC_NAME", "_model_score", "_es_info_of",
        "best_iteration_of", "_stratified_head_labels", "_last_model"),
    "早停/轮数上限": (
        "_es_source", "ES_SOURCE", "ES_FRACTION", "EARLY_STOP_ROUNDS",
        "N_ESTIMATORS_CAP", "_MAX_TREES", "_clamp_trees",
        "_fit_es_on_validation", "_fit_with_es", "_fit_with_eval",
        "_es_eval_cap", "_es_stage1", "_es_rounds_for", "_model_cap",
        "_set_cap", "_es_curve_of", "_es_best_trees", "_LAST_EVAL_XY",
        "_PRIMARY_FIT_PARAMS", "_PRIMARY_FIT_XY", "_fit_seed_variant",
        "_fit_hurdle_refit_full"),
    "HPO 入口": (
        "_hpo_fit_inner", "_inner_metric", "_INNER_METRIC_UP", "_build_model",
        "_HPO_ALGO"),
    "设备/线程/seed 并行": (
        "n_jobs", "pick_gpu", "_gpu_allocated", "_GPU_MIN_ROWS",
        "_xgb_cuda_built", "_lgbm_cuda_ok", "_LGBM_CUDA", "_lgbm_fit_safe",
        "align_predict_device", "_SEED_WORKER_ENV", "_confirm_parallel_setting",
        "_seed_parallel_plan", "_seed_parallel_release", "_seed_worker_run",
        "_run_seed_variants_parallel"),
    "特征块/加宽缓存基础设施": (
        "trial_feature_blocks", "feature_block", "_append_feature_blocks",
        "widen_cached", "_widen_face_key", "_widen_impl_sig", "_WIDEN_MEMO"),
}
PROTECTED_NAMES = {n: g for g, ns in _PROTECTED_GROUPS.items() for n in ns}
#: 前缀黑名单:harness 注册表/注入段与 patch 指令保留字
PROTECTED_PREFIXES = (
    ("_tfb_pre_", "特征块注册表(harness 生成)"),
    ("_ADDED_FEATURE_BLOCKS_", "特征块注册表(harness 生成)"),
    ("_hhp_", "hurdle 超参段(harness 注入)"),
    ("_hurdle_hp_", "hurdle 超参段(harness 注入)"),
    ("PATCH_", "patch 指令保留字"),
)
#: 受保护名字的规范导入:patch 以这一来源 import 不算改写(块内核需要它)
_CANONICAL_IMPORTS = {"feature_block": "runtime.feature_cache:feature_block"}
#: 白名单(典型可改,仅供 skeleton 标注与文档;判定以黑名单为准)
EDITABLE_NAMES = ("fit_and_score", "build_estimator", "prepare_features",
                  "_widen", "_bundle_widen")
EDITABLE_SUFFIXES = ("_kernel", "_block", "_row", "_cols", "_COLS")


def protected_reason(name: str) -> str | None:
    """受保护符号 → 原因;可替换 → None。"""
    if name in PROTECTED_NAMES:
        return PROTECTED_NAMES[name]
    for pre, why in PROTECTED_PREFIXES:
        if name.startswith(pre) and name not in PATCH_DIRECTIVES:
            return why
    return None


def _is_editable_hint(name: str) -> bool:
    return name in EDITABLE_NAMES or name.endswith(EDITABLE_SUFFIXES) or \
        name.startswith("HURDLE_")


# ---------------------------------------------------------------------------
# AST 工具(复用 add-blocks 的绑定/引用口径,保证两条路径判定一致)
# ---------------------------------------------------------------------------
def _render():
    from . import render as R

    return R


def _is_main_guard(st) -> bool:
    return isinstance(st, ast.If) and isinstance(st.test, ast.Compare) and \
        isinstance(st.test.left, ast.Name) and st.test.left.id == "__name__"


def _kind(st) -> str:
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "function"
    if isinstance(st, ast.ClassDef):
        return "class"
    if isinstance(st, ast.Assign) and len(st.targets) == 1 and \
            isinstance(st.targets[0], ast.Name):
        return "const"
    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and \
            st.value is not None:
        return "const"
    if isinstance(st, (ast.Import, ast.ImportFrom)):
        return "import"
    return "stmt"


def _import_targets(st) -> dict:
    """{绑定名: 导入目标串};`from a import b as c` → {"c": "a:b"}。"""
    out = {}
    for a in st.names:
        if isinstance(st, ast.ImportFrom):
            out[a.asname or a.name] = f"{'.' * st.level}{st.module or ''}:{a.name}"
        else:
            out[(a.asname or a.name).split(".")[0]] = \
                a.name if a.asname else a.name.split(".")[0]
    return out


def _stmt_span(st) -> tuple[int, int]:
    """语句行区间(0 基左闭右开;含装饰器,不含上方注释)。"""
    lo = min([st.lineno] + [d.lineno for d in getattr(st, "decorator_list", [])])
    return lo - 1, st.end_lineno


def _def_time_refs(st) -> set:
    """定义期(而非调用期)就会求值的名字:常量右值、装饰器、默认参数、基类。"""
    R = _render()
    if _kind(st) == "const":
        return R._referenced_names(st)
    nodes = list(getattr(st, "decorator_list", []))
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nodes += list(st.args.defaults) + [d for d in st.args.kw_defaults if d]
    if isinstance(st, ast.ClassDef):
        nodes += list(st.bases) + [k.value for k in st.keywords]
    out = set()
    for n in nodes:
        out |= {x.id for x in ast.walk(n) if isinstance(x, ast.Name)
                and isinstance(x.ctx, ast.Load)}
    return out


def _insert_before_main_guard(code: str, block: str) -> str:
    """把 block 插到顶层 main 守卫(及其紧邻上方注释)之前;无守卫则追加文末。"""
    lines = code.splitlines(keepends=True)
    guards = [st for st in ast.parse(code).body if _is_main_guard(st)]
    if not guards:
        return code.rstrip("\n") + "\n\n\n" + block
    lo = guards[-1].lineno - 1
    while lo > 0 and lines[lo - 1].lstrip().startswith("#"):
        lo -= 1
    head = "".join(lines[:lo]).rstrip("\n")
    return head + "\n\n\n" + block + "\n\n" + "".join(lines[lo:])


# ---------------------------------------------------------------------------
# patch 解析与合并(纯函数,创建前 dry-run 与创建后实写同一实现)
# ---------------------------------------------------------------------------
def parse_patch(text: str) -> dict:
    """解析 patch 文本 → {tree, lines, stmts, register, replaces}。格式错误抛 SystemExit。"""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise SystemExit(f"--patch:语法错误 第 {exc.lineno} 行:{exc.msg}")
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and \
            isinstance(body[0].value, ast.Constant) and \
            isinstance(body[0].value.value, str):
        body = body[1:]                                # 模块 docstring 不合并
    directives: dict = {}
    stmts = []
    for st in body:
        if _is_main_guard(st):
            raise SystemExit(f"--patch 第 {st.lineno} 行:patch 不得包含 "
                             "`if __name__ == \"__main__\"` 守卫(main 属 harness)")
        if isinstance(st, ast.ImportFrom) and st.module == "__future__":
            continue                                   # 父代已有;不可插到文件中部
        if _kind(st) == "const":
            nm = (st.targets[0] if isinstance(st, ast.Assign) else st.target).id
            if nm.startswith("PATCH_"):
                if nm not in PATCH_DIRECTIVES:
                    raise SystemExit(
                        f"--patch 第 {st.lineno} 行:未知指令 {nm}(允许:"
                        f"{', '.join(PATCH_DIRECTIVES)})")
                try:
                    val = ast.literal_eval(st.value)
                except ValueError:
                    val = None
                if not isinstance(val, (list, tuple)) or \
                        not all(isinstance(x, str) and x for x in val):
                    raise SystemExit(f"--patch:{nm} 必须是字符串字面量列表")
                if nm in directives:
                    raise SystemExit(f"--patch:指令 {nm} 重复声明")
                directives[nm] = list(dict.fromkeys(val))
                continue
        stmts.append(st)
    return {"tree": tree, "lines": text.splitlines(keepends=True),
            "stmts": stmts,
            "register": directives.get("PATCH_REGISTER_BLOCKS", []),
            "replaces": directives.get("PATCH_REPLACES")}


def merge_patch(parent_code: str, patch_text: str, *, label: str = "") -> tuple:
    """父代代码 + patch → (合并后代码, 报告)。违反约定抛 SystemExit(带全部原因)。

    报告:{replaced, added, imports_added, skipped_identical, registered_blocks,
    segment}。
    """
    R = _render()
    P = parse_patch(patch_text)
    ptree = ast.parse(parent_code)
    pbody = ptree.body
    last_binder: dict[str, int] = {}
    import_map: dict[str, str] = {}
    for i, st in enumerate(pbody):
        if _is_main_guard(st):
            continue
        for nm in R._ab_bound_names(st):
            last_binder[nm] = i
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            import_map.update(_import_targets(st))
    errors: list[str] = []
    replacements: dict[int, tuple] = {}
    added_stmts: list = []
    rep = {"replaced": [], "added": [], "imports_added": [],
           "skipped_identical": [], "registered_blocks": [], "segment": False}
    seen: dict[str, int] = {}
    new_names: set = set()
    for st in P["stmts"]:
        kind = _kind(st)
        names = R._ab_bound_names(st)
        where = f"第 {st.lineno} 行"
        _imp = _import_targets(st) if kind == "import" else {}
        for nm in sorted(names):
            why = protected_reason(nm)
            if why and nm in _imp and _imp[nm] == _CANONICAL_IMPORTS.get(nm):
                why = None                             # 规范导入(如 feature_block)放行
            if why:
                errors.append(
                    f"{where}:{nm} 是受保护符号({why}),patch 不得替换或新增"
                    + (";新特征块用 PATCH_REGISTER_BLOCKS 注册"
                       if nm == "trial_feature_blocks" else ""))
            if nm in seen and kind != "stmt":
                errors.append(f"{where}:{nm} 在 patch 内重复定义(首次第 "
                              f"{seen[nm]} 行)")
            seen.setdefault(nm, st.lineno)
        if kind == "import":
            fresh = []
            for nm, tgt in _import_targets(st).items():
                if nm in import_map:
                    if import_map[nm] != tgt:
                        errors.append(f"{where}:import 绑定 {nm}→{tgt} 与父代 "
                                      f"{nm}→{import_map[nm]} 不同(会遮蔽父代)")
                elif nm in last_binder:
                    errors.append(f"{where}:import 绑定的 {nm} 会遮蔽父代同名定义")
                else:
                    fresh.append(nm)
            if fresh:
                added_stmts.append(st)
                rep["imports_added"].extend(fresh)
                new_names.update(fresh)
            continue
        if kind in ("function", "class", "const"):
            (nm,) = tuple(names)
            if nm in last_binder:
                pi = last_binder[nm]
                pst = pbody[pi]
                pk = _kind(pst)
                if pk != kind or R._ab_bound_names(pst) != {nm}:
                    errors.append(
                        f"{where}:{nm} 在父代第 {pst.lineno} 行是"
                        f"{'多名绑定语句' if pk == kind else pk}(patch 是 {kind}),"
                        "只能同类整体替换")
                    continue
                if pi in replacements:
                    continue                           # 重复定义已记错
                if ast.dump(pst) == ast.dump(st):
                    rep["skipped_identical"].append(nm)
                    continue
                replacements[pi] = (nm, st)
                rep["replaced"].append(nm)
            else:
                added_stmts.append(st)
                rep["added"].append(nm)
                new_names.add(nm)
            continue
        # 其它顶层语句(for/assert/表达式/增量赋值/多名赋值):只能作用于本 patch 新名字
        hit = sorted(n for n in names if n in last_binder)
        if hit:
            errors.append(
                f"{where}:{type(st).__name__} 语句改写了父代已有名字 {hit};"
                "改父代常量/函数请写同名整体定义(原位替换)")
            continue
        added_stmts.append(st)
        for nm in sorted(names - new_names):
            rep["added"].append(nm)
        new_names |= names
    # 替换语句在父代原位置执行:定义期不得依赖本 patch 新增的名字(新增段在文末)
    for pi, (nm, st) in replacements.items():
        dep = sorted(_def_time_refs(st) & new_names)
        if dep:
            errors.append(
                f"{nm}(替换父代第 {pbody[pi].lineno} 行)定义期引用了本 patch 新增的 "
                f"{dep};新增段位于文末,原位替换处尚未定义 —— 把 {dep} 的定义内联"
                "或改为调用期引用")
    want = P["replaces"]
    if want is not None:
        got = set(rep["replaced"]) | set(rep["skipped_identical"])
        miss, extra = sorted(set(want) - got), sorted(got - set(want))
        if miss or extra:
            errors.append(
                f"PATCH_REPLACES 与合并结果不符:声明但未替换 {miss}(父代无此名 → "
                f"被当作新增)、替换但未声明 {extra}")
    if errors:
        raise SystemExit("--patch 合并被拒绝:\n  - " + "\n  - ".join(errors))

    # ---- 落代码:先自下而上原位替换,再插新增段
    lines = parent_code.splitlines(keepends=True)
    tag = f"patch {label}" if label else "patch"
    for pi in sorted(replacements, reverse=True):
        nm, st = replacements[pi]
        lo, hi = _stmt_span(pbody[pi])
        src = R._stmt_source(P["lines"], st).rstrip("\n") + "\n"
        lines[lo:hi] = [f"# [v2.19.0 {tag}] 替换父代 {nm}\n" + src]
    out = "".join(lines)
    reg = P["register"]
    if reg:
        wr = R.block_wrappers(out + "\n" + "".join(
            R._stmt_source(P["lines"], s) + "\n" for s in added_stmts))
        already = R._registered_wrappers(parent_code)
        bad = []
        for w in reg:
            if w not in wr:
                bad.append(f"{w} 不是调用 feature_block 的顶层包装函数")
            elif w in already:
                bad.append(f"{w} 已在父代 trial_feature_blocks 中注册(重复注册会"
                           "产生重名列)")
        if bad:
            raise SystemExit("--patch PATCH_REGISTER_BLOCKS 被拒绝:\n  - "
                             + "\n  - ".join(bad))
        rep["registered_blocks"] = [{"wrapper": w, "feature_block_name": wr[w]}
                                    for w in reg]
    if added_stmts or reg:
        k = parent_code.count(PATCH_BEGIN) + 1
        chunks = [R._stmt_source(P["lines"], s).rstrip("\n") for s in added_stmts]
        seg = (f"{PATCH_BEGIN}(harness 按 AST 合并 {tag};新增定义为 patch 原文)"
               "====\n" + ("\n\n\n".join(chunks) + "\n" if chunks else ""))
        if reg:
            pre = "trial_feature_blocks" if "trial_feature_blocks" in last_binder \
                else "None"
            regs = "\n".join(f"    ({('patch:' + w)!r}, {w})," for w in reg)
            seg += (
                ("\n\n" if chunks else "")
                + f"_tfb_pre_patch_{k} = {pre}\n"
                f"_ADDED_FEATURE_BLOCKS_P{k} = (\n{regs}\n)\n\n\n"
                "def trial_feature_blocks(split: str) -> \"pd.DataFrame\":\n"
                f"    \"\"\"v2.19.0 patch:原有块(若有)+ 注册块({', '.join(reg)}),"
                "按列右接。\"\"\"\n"
                "    import pandas as _pd_pt\n"
                "    parts = []\n"
                f"    if _tfb_pre_patch_{k} is not None:\n"
                f"        parts.append(_tfb_pre_patch_{k}(split).reset_index(drop=True))\n"
                f"    for _tag, _fn in _ADDED_FEATURE_BLOCKS_P{k}:\n"
                "        parts.append(_fn(split).reset_index(drop=True))\n"
                "    W = _pd_pt.concat(parts, axis=1)\n"
                "    _dup = W.columns[W.columns.duplicated()].tolist()\n"
                "    if _dup:\n"
                "        raise ValueError(f\"patch 特征块列重名: {_dup[:5]}\")\n"
                "    return W.astype(\"float32\")\n")
        seg += f"{PATCH_END}\n"
        out = _insert_before_main_guard(out, seg)
        rep["segment"] = True
    try:
        ast.parse(out)
    except SyntaxError as exc:                       # 理论不可达(兜底)
        raise SystemExit(f"--patch 合并结果语法错误 第 {exc.lineno} 行:{exc.msg}")
    if not (rep["replaced"] or rep["added"] or rep["registered_blocks"]):
        raise SystemExit("--patch 合并后与父代无差异(全部定义与父代 AST 等价);"
                         "零差异子代请改用 --set / --features")
    return out, rep


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dry_run_patch(run_dir: Path, parent_id: str, patch_path: str) -> dict:
    """创建 trial 之前对父代冻结代码做一次合并预演(失败抛 SystemExit,不落盘)。"""
    pp = Path(patch_path)
    if not pp.is_file():
        raise SystemExit(f"--patch 文件不存在:{patch_path}")
    ptp = Path(run_dir) / "trials" / parent_id / "code" / "train.py"
    if not ptp.exists():
        raise SystemExit(f"--patch:父代 {parent_id} 无 code/train.py")
    _, rep = merge_patch(ptp.read_text(encoding="utf-8"),
                         pp.read_text(encoding="utf-8"))
    return rep


def apply_patch(trial, patch_path: str) -> tuple:
    """把 patch 合并进新 trial 的 code/train.py(此时已是父代副本 + 创建期升级)。

    返回 (报告, 合并前代码);失败抛 SystemExit,不改文件。
    """
    pp = Path(patch_path).resolve()
    sha = _sha256_file(pp)
    tp = Path(trial.trial_dir) / "code" / "train.py"
    before = tp.read_text(encoding="utf-8")
    merged, rep = merge_patch(before, pp.read_text(encoding="utf-8"),
                              label=sha[:12])
    tp.write_text(merged, encoding="utf-8")
    rep.update({"source_path": str(pp), "source_sha": sha,
                "merged_sha": hashlib.sha256(merged.encode("utf-8")).hexdigest()})
    return rep, before


def _norm_lint(errs, trial_id: str) -> set:
    return {str(e).replace(trial_id, "<trial>") for e in errs or []}


def verify_patched_trial(run_dir: Path, trial, parent_id: str, rep: dict) -> dict:
    """合并后(含 add-blocks / 单分支渲染之后的最终代码)的创建期校验:

    1. 导入探针(NameError/ImportError 拒绝);
    2. 注册块探针:包装函数确实调用 feature_block,记块名/列数/内核指纹;
    3. Lint:合并结果相对父代**新增**的 lint 错误 → 拒绝(父代原有错误只告警,
       run 前 lint 照常执行)。
    失败抛 SystemExit。
    """
    R = _render()
    tdir = Path(trial.trial_dir)
    msg = R._import_probe(tdir)
    if msg:
        raise SystemExit(f"--patch 合并结果导入失败:{msg}")
    reg = [b["wrapper"] for b in rep.get("registered_blocks") or []]
    if reg:
        fps = R.probe_block_fingerprints(tdir / "code", reg)
        bad = {w: (fps.get(w) or {}).get("error") for w in reg
               if not fps.get(w) or fps[w].get("error")}
        if bad:
            raise SystemExit(f"--patch 注册块探针失败:{bad}")
        rep["registered_blocks"] = [
            {"wrapper": w, "feature_block_name": fps[w].get("name"),
             "code_fingerprint": fps[w].get("code_fingerprint"),
             "n_columns": fps[w].get("n_columns"),
             "train_block_key": fps[w].get("train_block_key"),
             "cache_present": fps[w].get("train_cache_present")} for w in reg]
    L = R._lint_module()
    child = L.lint_trial(Path(run_dir), tdir, False)
    pdir = Path(run_dir) / "trials" / parent_id
    try:
        parent = L.lint_trial(Path(run_dir), pdir, False)
    except Exception as exc:  # noqa: BLE001  父代 lint 失败不影响判定口径
        logger.warning("父代 lint 失败(按无错误处理):%s", exc)
        parent = []
    base = _norm_lint(parent, parent_id)
    new = [e for e in child if str(e).replace(trial.trial_id, "<trial>") not in base]
    old = [e for e in child if e not in new]
    if new:
        raise SystemExit("--patch 合并结果新增 lint 错误(父代没有):\n  - "
                         + "\n  - ".join(str(e) for e in new))
    if old:
        logger.warning("合并结果沿用父代已有的 lint 错误 %d 条(run 前仍会拦截):%s",
                       len(old), old[:3])
    rep["lint"] = {"new_errors": 0, "inherited_errors": len(old)}
    return rep


def trial_patch_record(rep: dict) -> dict:
    """trial.yaml 的 patch 留痕字段。"""
    keys = ("source_path", "source_sha", "merged_sha", "replaced", "added",
            "imports_added", "skipped_identical", "registered_blocks", "lint")
    return {k: rep.get(k) for k in keys if rep.get(k) is not None}


# ---------------------------------------------------------------------------
# trial-skeleton:父代大纲(只读)
# ---------------------------------------------------------------------------
_SKEL_PROBE_SRC = r'''
import importlib.util, json, pathlib, sys
names = json.loads(sys.argv[1])
spec = importlib.util.spec_from_file_location("_mvo_skel_probe", pathlib.Path("train.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["_mvo_skel_probe"] = mod
spec.loader.exec_module(mod)
import runtime.feature_cache as fc
calls = []
class _Stop(Exception):
    pass
def _rec(name, fn, **kw):
    calls.append((name, list(kw.get("columns") or []), list(kw.get("sources") or [])))
    raise _Stop()
mod.feature_block = _rec
fc.feature_block = _rec
out = {}
for w in names:
    calls.clear()
    try:
        getattr(mod, w)("train")
    except _Stop:
        pass
    except Exception as e:
        out[w] = {"error": "%s: %s" % (type(e).__name__, e)}
        continue
    if calls:
        out[w] = {"name": calls[0][0], "columns": calls[0][1], "sources": calls[0][2]}
print("SKEL_PROBE_JSON " + json.dumps(out, ensure_ascii=False))
'''


def _col_prefixes(cols: list) -> dict:
    out: dict = {}
    for c in cols:
        s = str(c)
        p = s[: s.index("_") + 1] if "_" in s else s
        out[p] = out.get(p, 0) + 1
    return out


def _probe_columns(code_dir: Path, wrappers: list) -> dict:
    try:
        r = subprocess.run([sys.executable, "-c", _SKEL_PROBE_SRC,
                            json.dumps(wrappers)], cwd=str(code_dir),
                           capture_output=True, text=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}
    line = next((ln for ln in r.stdout.splitlines()
                 if ln.startswith("SKEL_PROBE_JSON ")), None)
    if line is None:
        tail = (r.stderr or r.stdout).strip().splitlines()[-1:]
        return {"_error": " | ".join(tail) or f"rc={r.returncode}"}
    return json.loads(line[len("SKEL_PROBE_JSON "):])


def _kw_src(call, key: str):
    for k in call.keywords:
        if k.arg == key:
            return ast.unparse(k.value) if hasattr(ast, "unparse") else "?"
    return None


def build_skeleton(code: str, code_dir: Path | None = None) -> dict:
    """父代 train.py 的大纲:顶层定义、特征块、段标记、受保护标记。"""
    R = _render()
    tree = ast.parse(code)
    lines = code.splitlines()
    defs = []
    imports = []
    last: dict = {}
    for i, st in enumerate(tree.body):
        if not _is_main_guard(st):
            for n in R._ab_bound_names(st):
                last[n] = i
    for i, st in enumerate(tree.body):
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and \
                isinstance(st.value.value, str):
            continue
        kind = _kind(st)
        if kind == "import":
            imports.extend(_import_targets(st))
            continue
        names = sorted(R._ab_bound_names(st))
        if _is_main_guard(st):
            kind = "main_guard"
        lo, hi = _stmt_span(st)
        item = {"lines": [lo + 1, hi], "kind": kind, "names": names}
        if kind == "function" and hasattr(ast, "unparse"):
            item["signature"] = f"{st.name}({ast.unparse(st.args)})"
        prot = sorted({protected_reason(n) for n in names} - {None})
        if prot:
            item["protected"] = prot
        elif any(_is_editable_hint(n) for n in names):
            item["editable"] = True
        if kind in ("function", "class", "const") and \
                any(last.get(n) != i for n in names):
            item["shadowed"] = True                # 被后面的同名定义覆盖
        defs.append(item)
    # 段标记(# ==== … / # ---- …)
    sections = [{"line": i + 1, "text": ln.strip()[:120]}
                for i, ln in enumerate(lines)
                if ln.startswith("# ====") or ln.startswith("# ----")]
    # 特征块
    wr = R.block_wrappers(code)
    registered = R._registered_wrappers(code)
    blocks = []
    funcs = {st.name: st for st in tree.body
             if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for w, fb in wr.items():
        call = R._fb_calls(funcs[w])[0]
        kern = call.args[1] if len(call.args) > 1 else None
        try:
            closure = R.block_dependency_closure(code, w)
            n_lines = sum(_stmt_span(s)[1] - _stmt_span(s)[0] for s in closure)
        except SystemExit:
            closure, n_lines = [], None
        blocks.append({
            "wrapper": w, "feature_block_name": fb,
            "kernel": ast.unparse(kern) if kern is not None and
            hasattr(ast, "unparse") else None,
            "columns_expr": _kw_src(call, "columns"),
            "sources_expr": _kw_src(call, "sources"),
            "version": _kw_src(call, "version"),
            "registered": w in registered,
            "line": funcs[w].lineno, "closure_lines": n_lines,
            "closure_names": sorted({n for s in closure
                                     for n in R._ab_bound_names(s)})[:40]})
    if code_dir is not None and blocks:
        got = _probe_columns(code_dir, [b["wrapper"] for b in blocks])
        for b in blocks:
            g = got.get(b["wrapper"]) or {}
            if g.get("columns") is not None:
                b["n_columns"] = len(g["columns"])
                b["column_prefixes"] = _col_prefixes(g["columns"])
                b["sources"] = g.get("sources")
            else:
                b["probe_error"] = g.get("error") or got.get("_error")
    return {"n_lines": len(lines), "n_bytes": len(code.encode("utf-8")),
            "sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "imports": sorted(set(imports)), "defs": defs,
            "sections": sections, "feature_blocks": blocks,
            "patch_segments": code.count(PATCH_BEGIN),
            "protected_names": sorted(PROTECTED_NAMES),
            "protected_prefixes": [p for p, _ in PROTECTED_PREFIXES]}


def _short_list(xs, k: int = 4):
    if not isinstance(xs, list):
        return xs
    return xs if len(xs) <= k else xs[:k] + [f"…共 {len(xs)} 列"]


def format_skeleton(sk: dict, trial_id: str = "") -> str:
    out = [f"# trial-skeleton {trial_id}:{sk['n_lines']} 行 / {sk['n_bytes']} B "
           f"sha256={sk['sha256'][:12]}  patch 段 {sk['patch_segments']} 个",
           "# 标记:[P:原因] 受保护(patch 不可替换/新增) [E] 典型可改 "
           "[B] 特征块包装函数",
           f"imports: {', '.join(sk['imports'])}", "", "## 顶层定义(最后一个同名绑定 = "
           "patch 替换目标)"]
    blk = {b["wrapper"]: b for b in sk["feature_blocks"]}
    sec = list(sk["sections"])
    for d in sk["defs"]:
        while sec and sec[0]["line"] <= d["lines"][0]:
            out.append(f"   {sec.pop(0)['text']}")
        mark = ""
        if d.get("protected"):
            mark = f" [P:{'/'.join(d['protected'])}]"
        elif d.get("editable"):
            mark = " [E]"
        if d.get("shadowed"):
            mark += " (被后定义覆盖)"
        nm = d.get("signature") or ",".join(d["names"]) or "-"
        if d["names"] and d["names"][0] in blk:
            mark += f" [B {blk[d['names'][0]]['feature_block_name']}]"
        n = d["lines"][1] - d["lines"][0] + 1
        out.append(f"L{d['lines'][0]}-{d['lines'][1]} ({n}) {d['kind']} {nm}{mark}")
    for s in sec:
        out.append(f"   {s['text']}")
    out += ["", "## 特征块"]
    if not sk["feature_blocks"]:
        out.append("(无 feature_block 包装函数)")
    for b in sk["feature_blocks"]:
        cols = (f"{b['n_columns']} 列 前缀 {b['column_prefixes']}"
                if "n_columns" in b else f"columns={b['columns_expr']}"
                + (f"(探针失败:{b['probe_error']})" if b.get("probe_error") else ""))
        out.append(
            f"- {b['wrapper']} → 块 {b['feature_block_name']} 内核 {b['kernel']} "
            f"{'已注册' if b['registered'] else '未注册'};{cols};sources="
            f"{_short_list(b.get('sources')) or b['sources_expr']};"
            f"依赖闭包 {b['closure_lines']} 行"
            + (f";version={b['version']}" if b.get("version") else ""))
    out += ["", "## 受保护符号(黑名单)", ", ".join(sk["protected_names"]),
            "前缀:" + ", ".join(sk["protected_prefixes"])]
    return "\n".join(out)


def cmd_trial_skeleton(args) -> int:
    """`trial-skeleton --trial <p> [--json] [--no-probe]`:只读输出父代大纲。"""
    from .context import _run_dir_from

    run_dir = _run_dir_from(args)
    tdir = Path(run_dir) / "trials" / str(args.trial)
    tp = tdir / "code" / "train.py"
    if not tp.exists():
        logger.error("trial %s 无 code/train.py(%s)", args.trial, tp)
        return 1
    sk = build_skeleton(tp.read_text(encoding="utf-8"),
                        None if getattr(args, "no_probe", False) else tp.parent)
    if getattr(args, "json", False):
        print(json.dumps(sk, ensure_ascii=False, indent=1))
    else:
        print(format_skeleton(sk, str(args.trial)))
    return 0
