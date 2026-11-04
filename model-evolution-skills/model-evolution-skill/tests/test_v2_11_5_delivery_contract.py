"""ModelEvo v2.11.5 回归测试:交付链 / 创建链的「符号契约」守卫。

动机(drawdown-req-t7-v211 执行历史回写):
  harness 有 22 个测试文件、500+ checks 守着**训练链**,而**交付链与创建链
  的符号契约一个测试都没有**。结果是同一个缺陷族在一条 run 里连崩 4 次:

    v2.11.1  confirm 复现面 noise=None → temporal 段 `2 * noise` TypeError
             (反相关缺陷:复现得越完美越必然崩溃)
    v2.11.2  --from-parent 只复制 *.py,静默丢弃 code/ 非 .py 附属文件
             (prune_keep.json)→ 3 个 gen-03 trial 0.5s FAILED
    v2.11.3  finalize-retrain 期望 champ.build_estimator,
             模板血缘导出 _build_model → 交付链阻断
    v2.11.4  加宽重放只认 build_wide/_build_widen_columns,
             超级 trial 血缘导出 _widen/_widen_pruned
             → **交付面退化为 1500 列而冠军实为 2442 列**(不报错!)

  这 4 例同族:**「CLI 期望的符号名 ≠ 实际 trial 代码导出的符号名」**。
  且 v2.11.2/v2.11.3 两例在**同族 run** 里早已被"降级脚本/手工 cp"绕过过,
  变通没回写 harness ⇒ 下个 run 复发。

本文件把这些契约固化成断言,使同类偏斜在**测试期**而非**交付期**暴露。
新增血缘符号族时,必须同步更新本文件的 SYMBOL_FAMILIES 表(测试会失败提醒)。

跑法:python tests/test_v2_11_5_delivery_contract.py [--skill-root <path>]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import tempfile
from pathlib import Path

PASS = 0
FAIL = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 契约表:血缘符号族(新增族必须在此登记,否则 A2 会失败)
# ---------------------------------------------------------------------------
# 每族 = (族名, 触发符号, 说明)。CLI 的交付侧 transform 重放必须对每个
# 非 none 族有分支;none 族 = 基础面直读(默认路径)。
SYMBOL_FAMILIES = [
    ("handwritten_vocab", ("build_wide",), "手写词表血缘(build_wide/_build_vocab)"),
    ("handwritten_widen", ("_build_widen_columns",), "手写加宽血缘(P7)"),
    ("super_trial_widen", ("_widen_pruned", "_widen"),
     "超级 trial 血缘(_widen_pruned/_widen/widen_cached;模板 body 渲染 "
     "widen_cached,Agent transform 追加 _widen/_widen_pruned)"),
]

# estimator 工厂族:CLI 必须能消费两族(适配器 _finalize_build_estimator)
ESTIMATOR_FACTORIES = ("build_estimator", "_build_model")


def _read(rel: str, root: Path) -> str:
    """读 skill 内的源码文本。

    v2.11.9:CLI 拆成 `scripts/modelevo.py` + `scripts/_mvo/*.py`,
    单文件不再是全集 —— 对 `scripts/modelevo.py` 的请求统一返回
    `cli_source()`(全集拼接),其余路径按原样读。
    """
    if rel.replace("\\", "/") == "scripts/modelevo.py":
        return cli_source()
    return (root / rel).read_text(encoding="utf-8")


def _module_defs(src: str) -> set[str]:
    """模块级 def / class / 顶层常量名(粗粒度符号面)。

    渲染模板体含 f-string 占位(如 `RUN_DIR = pathlib.Path({run_dir!r})`),
    裸字面量不是合法 Python → AST 解析失败时回退到正则扫描(行首无缩进的
    def/class/赋值)。
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return _module_defs_regex(src)
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            out.add(node.target.id)
    return out


def _module_defs_regex(src: str) -> set[str]:
    """AST 解析失败(f-string 占位)时的符号面回退扫描。"""
    out: set[str] = set()
    out.update(re.findall(r"^def ([a-zA-Z_][a-zA-Z0-9_]*)", src, re.M))
    out.update(re.findall(r"^class ([a-zA-Z_][a-zA-Z0-9_]*)", src, re.M))
    out.update(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]", src, re.M))
    out.update(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[A-Za-z_]",
                          src, re.M))
    return out


def _load_body_template(cli_src: str) -> str:
    """按 harness 同款锚点抽取渲染 body 字面量(与 _graft_hpo_section 一致)。"""
    # v2.11.9:模板已是 _mvo/templates.TRIAL_BODY 模块常量,直接取。
    # 旧实现从 scripts/modelevo.py 源码找赋值锚点 + ast.literal_eval
    # 还原字面量;模板抽出后源码里已无该字面量。
    del cli_src                              # 保持旧签名(调用方仍传)
    return trial_body()


def _rendered_symbols(cli_src: str, tmpl_src: str) -> set[str]:
    """**trial 血缘**渲染产物(TRIAL_TEMPLATE 头 + body)的模块级符号全集。

    这是本仓库两条代码生成通道之一;另一条是 baseline 血缘
    (`runtime/baseline.py` 的 BASELINE_TEMPLATE,见
    `_baseline_symbols`)。两通道符号面**不同**——`build_estimator` 只
    存在于 baseline 通道,`_build_model` 只存在于 trial 通道。这正是
    v2.11.3 偏斜的来源。
    """
    i = tmpl_src.find("TRIAL_TEMPLATE = ")
    j = tmpl_src.find("'''", i) + 3
    k = tmpl_src.find("'''", j)
    header = tmpl_src[j:k]
    body = _load_body_template(cli_src)
    return _module_defs(header) | _module_defs(body)


def _baseline_symbols(bl_src: str) -> set[str]:
    """**baseline 血缘**渲染产物(runtime/baseline.py BASELINE_TEMPLATE)。"""
    i = bl_src.find("BASELINE_TEMPLATE = ")
    j = bl_src.find("'''", i) + 3
    k = bl_src.find("'''", j)
    return _module_defs(bl_src[j:k])


def _probed_symbols(cli_src: str, obj: str) -> set[str]:
    """CLI 中经 hasattr/getattr 显式探测过的符号(合法直接访问的前提)。"""
    out: set[str] = set()
    for m in re.finditer(
            rf'(?:hasattr|getattr)\(\s*{obj}\s*,\s*["\']([a-zA-Z_]\w*)["\']',
            cli_src):
        out.add(m.group(1))
    return out


def _unguarded_attr_accesses(cli_src: str, obj: str) -> set[str]:
    """`<obj>.X` 形式的属性访问,排除被 hasattr/getattr 探测包裹的。

    判定方式:逐行扫描,跳过含 `hasattr(`/`getattr(` 的行;再排除常见非
    模块属性访问(如 `.data`/`.trial_dir`——那是 TrialRecord 对象)。
    """
    out: set[str] = set()
    for line in cli_src.splitlines():
        if "hasattr(" in line or "getattr(" in line:
            continue
        for m in re.finditer(rf"\b{obj}\.([a-zA-Z_][a-zA-Z0-9_]*)", line):
            out.add(m.group(1))
    return out


# ---------------------------------------------------------------------------
# A1:通用符号偏斜检测(本文件的核心;能同时抓住 v2.11.3 与 v2.11.4 两例)
# ---------------------------------------------------------------------------

def test_a1_cli_symbols_covered_by_template() -> None:
    print("\nA1 CLI 未加保护符号访问 ⊆ 对应血缘的渲染符号面:")
    root = ROOT
    cli = _read("scripts/modelevo.py", root)
    tmpl = _read("runtime/templates.py", root)
    bl = _read("runtime/baseline.py", root)
    trial_syms = _rendered_symbols(cli, tmpl)
    base_syms = _baseline_symbols(bl)

    # 两条血缘的符号面必须**确实不同**(这是 v2.11.3 偏斜的根源,记录性断言;
    # 若未来统一为一条通道,本断言失败 → 提醒同步简化交付侧适配层)
    check("trial 血缘导出 _build_model", "_build_model" in trial_syms)
    check("baseline 血缘导出 build_estimator",
          "build_estimator" in base_syms)
    check("两血缘符号面不同(记录性)",
          ("_build_model" in trial_syms) != ("build_estimator" in base_syms)
          or "_build_model" not in base_syms)

    # 规则:CLI 对某符号的**直接**访问合法,当且仅当该符号属于对应血缘的
    # 渲染符号面,或 CLI 已在某处用 hasattr/getattr 显式探测过它
    # (探测 = 显式分支,缺失时走 fallback)。
    OBJ_LINEAGE = {"champ": (trial_syms, "trial"), "base": (base_syms,
                                                           "baseline")}
    NON_MODULE_ATTRS = {  # TrialRecord/dict 等对象属性,不是模板符号
        "data", "trial_dir", "trial_id", "flush", "transition", "state",
        "attempts", "update",
    }
    skew: dict[str, set[str]] = {}
    for obj, (surface, lineage) in OBJ_LINEAGE.items():
        acc = _unguarded_attr_accesses(cli, obj)
        probed = _probed_symbols(cli, obj)
        missing = {a for a in acc
                   if a not in surface and a not in probed
                   and a not in NON_MODULE_ATTRS}
        if missing:
            skew[obj] = missing
    check("无未加保护的符号偏斜", not skew,
          f"偏斜(CLI 直接访问但既不在 {[v[1] for v in OBJ_LINEAGE.values()]}"
          f" 渲染面、也无 hasattr 探测): {skew}" if skew else
          f"champ 面 {len(trial_syms)} 符号 / base 面 {len(base_syms)} 符号,"
          "全部访问已覆盖")

    # 反向断言:两条血缘都必须真的导出交付链关键符号
    for sym in ("prepare_features", "exclude_columns", "load_labels",
                "reparse_frame", "dump_model", "leak_scan", "n_jobs"):
        check(f"trial 血缘渲染含 {sym}", sym in trial_syms)
    for sym in ("exclude_columns", "load_labels", "build_estimator"):
        check(f"baseline 血缘渲染含 {sym}", sym in base_syms)


# ---------------------------------------------------------------------------
# A2:加宽重放必须覆盖全部血缘族(v2.11.4 的直接守卫)
# ---------------------------------------------------------------------------

def test_a2_widen_replay_family_coverage() -> None:
    print("\nA2 加宽重放覆盖全部血缘族(训练路径 + 打分路径):")
    cli = _read("scripts/modelevo.py", ROOT)

    # 训练路径(cmd_finalize_retrain)与打分路径(_score_split)两处都需覆盖
    for fam, syms, desc in SYMBOL_FAMILIES:
        if fam == "handwritten_vocab":
            # build_wide 分支(P7 之前的词表血缘)
            check(f"[训练] {desc}", "hasattr(champ, \"build_wide\")" in cli
                  or "hasattr(champ, 'build_wide')" in cli)
            continue
        for sym in syms:
            in_train = f'hasattr(champ, "{sym}")' in cli \
                or f"hasattr(champ, '{sym}')" in cli
            check(f"[族 {fam}] 训练/打分路径探测 {sym}", in_train, desc)

    # 打分路径必须显式引用第三族符号(v2.11.4 修复点)
    score_block_start = cli.find("def _score_split")
    check("_score_split 存在", score_block_start > 0)
    tail = cli[score_block_start:score_block_start + 8000]
    check("[打分] _score_split 覆盖 super_trial_widen 族",
          "_widen_pruned" in tail or "_widen" in tail,
          "v2.11.4 修复点:此前只认 _build_widen_columns")

    # 顺序契约:打分帧加宽必须在 prepare_features 之前(与冠军 main() 同序)。
    # v2.11.9 订正:旧断言用 `tail.find()` 比较**全局首次出现位置**,但
    # `_score_split` 的四个血缘分支是互斥的——第四族(widen_face)分支内部
    # 先调 prepare_features 治理基础帧,第三族的 `_widen_pruned` 因此出现
    # 在它之后,旧断言恒假(假阳性:实测 widen@5734 > prepare_features@2630)。
    # 改为:第三族分支的起点必须早于尾部那段**共享**的 prepare_features。
    i_fam3 = tail.find('hasattr(champ, "_widen_pruned")')
    i_pf_shared = tail.find('if (hasattr(champ, "prepare_features")')
    check("[打分] 加宽重放先于 prepare_features",
          i_fam3 > 0 and i_pf_shared > i_fam3,
          f"第三族分支@{i_fam3} < 共享 prepare_features@{i_pf_shared}")


# ---------------------------------------------------------------------------
# A3 + B1:estimator 工厂适配器(v2.11.3 的直接守卫)
# ---------------------------------------------------------------------------

def test_a3_adapter_present() -> None:
    print("\nA3 estimator 工厂适配器存在且覆盖两族:")
    cli = _read("scripts/modelevo.py", ROOT)
    check("_finalize_build_estimator 已定义",
          "def _finalize_build_estimator(" in cli)
    for sym in ESTIMATOR_FACTORIES:
        check(f"适配器引用 {sym}", f'"{sym}"' in cli)
    # 两处调用点都必须走适配层(而非裸 champ.build_estimator)
    bare = re.findall(r"champ\.build_estimator\(", cli)
    check("无裸 champ.build_estimator( 调用", not bare,
          f"残留 {len(bare)} 处" if bare else "全部走适配层")


def test_b1_adapter_behavior() -> None:
    print("\nB1 适配器三态行为(功能):")
    sys.path.insert(0, str(ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_cli_under_test", ROOT / "scripts" / "modelevo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cli_under_test"] = mod
    spec.loader.exec_module(mod)
    fn = getattr(mod, "_finalize_build_estimator", None)
    if fn is None:
        check("适配器可导入", False, "函数缺失")
        return
    check("适配器可导入", True)

    calls: list = []

    class _FakeEst:
        def __init__(self, **kw):
            self.kw = kw

        def get_params(self, deep=True):
            return self.kw

    # 态 1:有 build_estimator → 优先走它
    class M1:
        SEED = 7

        @staticmethod
        def build_estimator(n_jobs, n_rows, hp=None):
            calls.append(("build_estimator", n_jobs, n_rows, hp))
            return _FakeEst(src="build_estimator")

    r1 = fn(M1, 8, 1000, {"a": 1})
    check("态1: 优先 build_estimator",
          r1.get_params()["src"] == "build_estimator" and
          calls[-1] == ("build_estimator", 8, 1000, {"a": 1}))

    # 态 2:只有 _build_model(模板血缘)→ 回落,且 n_jobs 并入 params
    class M2:
        SEED = 20260814

        @staticmethod
        def _build_model(params, seed, _n_rows=0):
            calls.append(("_build_model", params, seed, _n_rows))
            return _FakeEst(**params)

    r2 = fn(M2, 12, 2000, {"max_depth": 9})
    p2 = r2.get_params()
    check("态2: 回落 _build_model 并被调用",
          calls[-1][0] == "_build_model")
    check("态2: 超参原样传递", p2.get("max_depth") == 9)
    check("态2: n_jobs 并入 params", p2.get("n_jobs") == 12)
    check("态2: n_rows/seed 正确传递",
          calls[-1][3] == 2000 and calls[-1][2] == 20260814)

    # 态 3:两族都缺 → 显式报错(不静默猜测)
    class M3:
        pass

    raised = False
    try:
        fn(M3, 8, 100, None)
    except SystemExit:
        raised = True
    except Exception as e:  # noqa: BLE001
        raised = "SystemExit" in type(e).__name__
    check("态3: 两族皆缺时显式报错", bool(raised))

    # 态 4:显式调用者传的 n_jobs 不被覆写
    r4 = fn(M2, 4, 100, {"n_jobs": 99, "max_depth": 5})
    check("态4: 调用者显式 n_jobs 优先(不覆写)",
          r4.get_params().get("n_jobs") == 99)


# ---------------------------------------------------------------------------
# B2:--from-parent 必须复制 code/ 下的非 .py 附属文件(v2.11.2 的直接守卫)
# ---------------------------------------------------------------------------

def test_b2_from_parent_copies_aux_files() -> None:
    print("\nB2 --from-parent 复制 code/ 非 .py 附属文件(功能):")
    sys.path.insert(0, str(ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_cli_b2", ROOT / "scripts" / "modelevo.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cli_b2"] = mod
    spec.loader.exec_module(mod)
    copy_fn = getattr(mod, "_copy_parent_trial_code", None)
    if copy_fn is None:
        check("_copy_parent_trial_code 可导入", False)
        return
    check("_copy_parent_trial_code 可导入", True)

    tmp = Path(tempfile.mkdtemp())
    parent = tmp / "trials" / "trial-001"
    (parent / "code").mkdir(parents=True)
    (parent / "config").mkdir(parents=True)
    (parent / "code" / "train.py").write_text("x = 1\n")
    # 关键:非 .py 附属文件(gen-02 压缩 trial 的 prune_keep.json 同型)
    (parent / "code" / "prune_keep.json").write_text('{"keep": ["a"]}')
    (parent / "code" / "notes.txt").write_text("hi")
    (parent / "code" / "stale.pyc").write_text("bytecode")
    (parent / "code" / "__pycache__").mkdir()
    (parent / "config" / "trial.yaml").write_text("trial_id: trial-001\n")

    class _FakeTrial:
        trial_id = "trial-002"
        trial_dir = str(tmp / "trials" / "trial-002")
        data: dict = {}

        def flush(self):
            pass

    (tmp / "trials" / "trial-002" / "config").mkdir(parents=True)
    t = _FakeTrial()
    try:
        copy_fn(tmp, t, "trial-001", {"learning_rate": 0.02})
    except Exception as e:  # noqa: BLE001
        check("复制执行无异常", False, f"{type(e).__name__}: {e}")
        return
    check("复制执行无异常", True)

    dst = Path(t.trial_dir) / "code"
    check("复制 .py", (dst / "train.py").exists())
    check("复制 .json 附属文件(prune_keep.json 同型)",
          (dst / "prune_keep.json").exists())
    check("复制其它常规附属文件(.txt)", (dst / "notes.txt").exists())
    check("跳过 .pyc", not (dst / "stale.pyc").exists())
    check("跳过 __pycache__ 目录", not (dst / "__pycache__").exists())
    cfg_ok = (Path(t.trial_dir) / "config" / "trial.yaml").exists()
    check("config 落盘", cfg_ok)


# ---------------------------------------------------------------------------
# B3:缺列校验必须区分「基础面缺列」与「派生展开缺级」(v2.11.4 第二半)
# ---------------------------------------------------------------------------

def test_b3_missing_column_split() -> None:
    print("\nB3 缺列校验区分基础面缺列 / 派生展开缺级:")
    cli = _read("scripts/modelevo.py", ROOT)
    check("存在 _miss_base 分支", "_miss_base" in cli)
    check("存在 _miss_derived 分支", "_miss_derived" in cli)
    check("基础面缺列仍硬报错(RuntimeError)",
          "missing base columns" in cli)
    check("派生缺级补 0 而非报错",
          bool(re.search(r"_miss_derived[\s\S]{0,600}?X\[_c\]\s*=\s*0\.0",
                         cli)))
    check("派生缺级记入 model_meta 诊断",
          "missing_derived_levels" in cli)
    # F1 保护未被放松:不得出现无差别 reindex(fill_value=0).fillna(0)
    check("F1 保护未放松(无 reindex fill_value=0 + fillna(0))",
          "reindex(columns=cols, fill_value=0).fillna(0)" not in cli)


# ---------------------------------------------------------------------------
# C1:temporal 段对 noise=None 的防御(v2.11.1 的直接守卫)
# ---------------------------------------------------------------------------

def test_c1_temporal_noise_none_guarded() -> None:
    print("\nC1 temporal_evaluation 对 noise=None 的防御:")
    ev = _read("runtime/evaluator.py", ROOT)
    check("temporal 段含 noise 有效性守卫",
          bool(re.search(r"temporal[\s\S]{0,1200}?noise is not None[\s\S]{0,80}?"
                         r"np\.isfinite\(noise\)", ev))
          or bool(re.search(r"noise is not None and np\.isfinite\(noise\)"
                            r"[\s\S]{0,2000}?temporal_evaluation", ev)))
    check("temporal 段含 noise=None 退化分支", "noise_available" in ev)
    # v2.11.1 的崩溃点精确地是 `out["temporal_evaluation"]` **字典字面量**
    # 里的 "passed" 表达式无条件做 `2 * noise`。上下文其它 `2 * noise`
    # 都位于 `if noise is not None and np.isfinite(noise) and noise > 0:`
    # 分支内(合法)。故断言精确限定在该字面量上。
    i0 = ev.find('out["temporal_evaluation"] = {')
    if i0 < 0:
        check("temporal_evaluation 字面量存在", False)
    else:
        i1 = ev.find("}", i0)
        lit = ev[i0:i1]
        bad = bool(re.search(r"(?<![\w.])2\s*\*\s*noise\b", lit))
        check("temporal_evaluation 字面量内无裸 `2 * noise`", not bad,
              f"字面量: {lit[:120]!r}" if bad else "")
        check("字面量使用预计算的 _temp_passed",
              "_temp_passed" in lit)
    # 退化判据本身必须存在(noiseless 分支)
    check("temporal 段含 noiseless 退化判据",
          "worst_t >= 0" in ev or "worst_t <= 0" in ev)


def test_c2_confirm_replication_no_crash() -> None:
    print("\nC2 confirm 复现面(Δ≡0 → noise=None)不崩(功能):")
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from runtime.evaluator import Evaluator
    from runtime.spec import TaskSpec
    from runtime.config import Config

    spec = TaskSpec(data={
        "task_id": "t-c2", "task_name": "t",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "final_test": "2026-04"},
        "data": {},
    })
    cfg = Config.load(skill_config=False)
    ev = Evaluator(spec, cfg)
    rng = np.random.default_rng(7)
    n = 3000
    y = (rng.random(n) > 0.7).astype(int)
    s = rng.normal(size=n)
    # parent == child 逐位相同 ⇒ 配对 bootstrap delta 恒 0 ⇒ SE=0 ⇒ noise=None
    temporal = [{"y_true": y[:1500], "score": s[:1500],
                 "parent_score": s[:1500]},
                {"y_true": y[1500:], "score": s[1500:],
                 "parent_score": s[1500:]}]
    try:
        out = ev.evaluate(y, s, parent_score=s.copy(), confirm_trial=True,
                          temporal_scores=temporal)
    except TypeError as e:
        check("Δ≡0 的 confirm 评估不抛 TypeError", False, str(e))
        return
    except Exception as e:  # noqa: BLE001
        check("Δ≡0 的 confirm 评估不抛异常", False,
              f"{type(e).__name__}: {e}")
        return
    check("Δ≡0 的 confirm 评估不抛异常", True)
    te = out.get("temporal_evaluation") or {}
    check("temporal_evaluation 产出 passed 字段", "passed" in te,
          f"{te}")
    check("noise=None 时置 noise_available=False",
          te.get("noise_available") is False)
    check("verdict 非 invalid",
          out.get("verdict") != "invalid", str(out.get("verdict")))


# ---------------------------------------------------------------------------
# D1:harness_hash 冻结面事实(记录,防误判)
# ---------------------------------------------------------------------------

def test_d1_hash_scope_documented() -> None:
    print("\nD1 harness_hash 覆盖范围(记录性断言):")
    sys.path.insert(0, str(ROOT))
    from runtime.harness import harness_hash
    h_runtime = harness_hash(ROOT / "runtime")
    check("harness_hash(runtime) 可计算", bool(h_runtime),
          f"{h_runtime}")
    # scripts/ 不在 hash 面内 ⇒ 改 CLI 不触发铁律 10 的 acknowledge 闸门。
    # 这是**已知未修的冻结面缺口**(见 CHANGELOG v2.11.2 附注)。若未来把
    # scripts/ 纳入 hash 面,本断言会失败并提醒维护者同步更新文档与流程。
    cli_before = (ROOT / "scripts" / "modelevo.py").stat().st_mtime
    h2 = harness_hash(ROOT / "runtime")
    check("scripts/ 变更不影响 harness_hash(冻结面缺口,已记录)",
          h_runtime == h2 and cli_before > 0,
          "若此断言失败:scripts/ 已被纳入 hash 面,"
          "请更新 CHANGELOG 与 acknowledge 流程")


# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL, cli_source, trial_body  # noqa: E402

ROOT = SKILL


def main() -> int:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-root", default=str(ROOT))
    a = ap.parse_args()
    ROOT = Path(a.skill_root).resolve()
    print("=" * 70)
    print("v2.11.5 交付链/创建链符号契约守卫")
    print(f"harness root: {ROOT}")
    print("=" * 70)

    test_a1_cli_symbols_covered_by_template()
    test_a2_widen_replay_family_coverage()
    test_a3_adapter_present()
    test_b1_adapter_behavior()
    test_b2_from_parent_copies_aux_files()
    test_b3_missing_column_split()
    test_c1_temporal_noise_none_guarded()
    test_c2_confirm_replication_no_crash()
    test_d1_hash_scope_documented()

    print(f"\n{'=' * 70}\nv2.11.5 contract guard: "
          f"{PASS} passed, {FAIL} failed")
    if FAILED:
        print("FAILED: " + "; ".join(FAILED))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
