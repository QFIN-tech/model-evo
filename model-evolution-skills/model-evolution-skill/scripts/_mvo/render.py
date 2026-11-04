"""trial 代码生成:特征引用展开 + hpo 嫁接/注入

v2.11.9 从单文件 `scripts/modelevo.py` 拆出(拆分动机见
`_mvo/__init__.py`)。**本文件是机械搬运,未改语义** —— 便于对照原
实现审阅。新增命令请按职责放进对应模块,不要回填到 `modelevo.py`
(那里只留 argparse 与分发)。
"""
from __future__ import annotations

from pathlib import Path
from _mvo.templates import AUTOML_BODY, TRIAL_BODY  # noqa: E402
import json
from runtime.logging_utils import (dump_json, dump_yaml, get_logger,  # noqa: E402
                                   load_json, load_yaml, locked_state_write,
                                   now_iso)
import sys
from .context import _SKILL_ROOT, _load_config, logger


def _parse_overrides(pairs: list[str] | None) -> dict:
    """--set key=value 解析:值按 YAML 字面量解释(0.03→float,6→int,
    "auto"→str),解析失败按原始字符串。供参数 trial 快通道使用。"""
    out: dict = {}
    for kv in pairs or []:
        if "=" not in kv:
            raise SystemExit(f"--set 需要 key=value 形式,得到: {kv}")
        k, v = kv.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"--set key 不能为空: {kv}")
        try:
            import yaml as _yaml

            out[k] = _yaml.safe_load(v)
        except Exception:  # noqa: BLE001
            out[k] = v
    return out


def _ref_domain(trial_dir: Path, derived: list[str]) -> list[str]:
    """v2.10.1/F9:引用展开(裁决产物 → 列)的列域。

    优先父 trial 的**权威特征面** `model/feature_columns.json`(交付时
    落盘,含 transform 派生列);缺失时退回基础帧派生面 `derived`。
    旧实现一律用 derived,于是加宽血缘的 trial 在
    `@id:feature_importance` / `@id:group_importance` 展开时,
    派生列被静默筛掉(实测 gen-02/gen-03)。
    """
    fc = Path(trial_dir) / "model" / "feature_columns.json"
    if fc.exists():
        try:
            cols = json.loads(fc.read_text(encoding="utf-8"))
            if isinstance(cols, list) and cols:
                return [c for c in cols if isinstance(c, str)]
        except (OSError, ValueError):
            pass
    return list(derived)


def _explicit_face(cols, drop, derived: list[str],
                   label: str) -> list[str] | None:
    """v2.10.1/F9:显式列名清单(文件引用 / 父代 config 引用)。

    显式清单本身就是权威 —— **不与基础帧派生面求交**。派生列(transform
    新增列)按定义不在基础帧里,求交会把它们静默丢光,而调用方(Agent)
    正是用 `--features @file.yaml` 来表达"就要这几列"。
    不在基础面里的列打出告警(提示按 transform 派生列处理),不报错。
    """
    drop = set(drop or ())
    out = [c for c in (cols or [])
           if isinstance(c, str) and c not in drop]
    unknown = [c for c in out if c not in set(derived)]
    if unknown:
        logger.warning(
            "引用白名单(%s)含 %d 个不在基础特征面的列,按 transform "
            "派生列放行(前 10): %s", label, len(unknown), unknown[:10])
    return out or None


def _load_features_ref(ref: str, run_dir: Path,
                       derived: list[str]) -> list[str] | None:
    """--features @引用模式解析:特征列表不进命令行/上下文。

    支持的引用(v2.8):
      @path/to/face.yaml            YAML 文件,key 为 features(dict:
                                    exclude/columns)或直接列名 list
      @<trial_id>                   继承该 trial config/trial.yaml 的
                                    features(整体:exclude+columns)
      @<trial_id>:feature_importance  按 trial 落盘的
                                    feature_importance.csv/gain 裁决,
                                    取覆盖累计 share 阈值的头部列
      @<trial_id>:group_importance  按 feature/group_importance.json 的
                                    families_alive ∪ families_unverifiable
                                    (族白名单;unverifiable=无证据非死,
                                    v15.2)展开为列

    返回 None 表示引用不成立(调用方报错);引用文件缺失/格式错时
    SystemExit 带明确原因,不静默回退。
    """
    ref = ref.strip()
    if not ref.startswith("@"):
        return None
    src = ref[1:]

    def _families_expand(trial_dir: Path, alive_key: str) -> list[str]:
        gi_path = trial_dir / "feature" / "group_importance.json"
        if not gi_path.exists():
            raise SystemExit(
                f"引用失败: {gi_path} 不存在(该 trial 未产出分组置换裁决)")
        gi = load_json(gi_path) or {}
        families = set(gi.get(alive_key) or [])
        # v15.2:unverifiable = 无证据(评估帧缺列,0 填充常数列置换
        # delta≡0),不是死——展开白名单时并入 alive,防止"帧缺失"
        # 被下游当成"方向已证死"把列族筛掉(实测 gen-02:8 个加宽族
        # 被伪判 dead,引用展开后新列族全被丢弃)
        if alive_key == "families_alive":
            families |= set(gi.get("families_unverifiable") or [])
            if not families:
                raise SystemExit(
                    f"引用失败: group_importance.json 的 {alive_key} 与 "
                    "families_unverifiable 均为空")
        elif not families:
            raise SystemExit(
                f"引用失败: group_importance.json 的 {alive_key} 为空")
        # v2.10.1/F9:同 _fi_head,展开域用父代权威特征面
        return [c for c in _ref_domain(trial_dir, derived)
                if (c.split("_", 1)[0] if "_" in c else "base") in families]

    def _fi_head(trial_dir: Path) -> list[str]:
        fi_path = trial_dir / "feature" / "feature_importance.csv"
        if not fi_path.exists():
            raise SystemExit(
                f"引用失败: {fi_path} 不存在(该 trial 未产出 gain)")
        import pandas as _pd

        fi = _pd.read_csv(fi_path).sort_values(
            "importance", ascending=False)
        total = float(fi["importance"].sum()) or 1.0
        share = fi["importance"].cumsum() / total
        keep = fi.loc[share < 0.95, "feature"].tolist() \
            or fi["feature"].head(1).tolist()
        # v2.10.1/F9:展开域用父代权威特征面(含派生列)
        return [c for c in _ref_domain(trial_dir, derived)
                if c in set(keep)]

    # trial 引用:@id / @id:feature_importance / @id:group_importance
    # (Windows 盘符如 C:/ 不含 trial.yaml,自然落到文件分支,不受影响)
    if ":" in src and src[1:2] != ":":
        tid, what = src.split(":", 1)
    else:
        tid, what = src, ""
    trial_dir = run_dir / "trials" / tid
    if trial_dir.exists() and ((trial_dir / "trial.yaml").exists()
                               or (trial_dir / "config" / "trial.yaml").exists()):
        if what == "":
            pcfg = load_yaml(trial_dir / "config" / "trial.yaml") or {}
            feats = pcfg.get("features")
            if isinstance(feats, dict):
                # v2.10.1/F9:父代 config 的 columns 是显式清单,不求交
                return _explicit_face(feats.get("columns"),
                                      feats.get("exclude"), derived, tid)
            if isinstance(feats, list):
                return _explicit_face(feats, None, derived, tid)
            raise SystemExit(
                f"引用失败: {tid} 的 config features 缺失或格式不支持")
        if what == "feature_importance":
            return _fi_head(trial_dir)
        if what == "group_importance":
            return _families_expand(trial_dir, "families_alive")
        raise SystemExit(f"引用失败: 不支持的引用形式 {ref}"
                         "(可用 :feature_importance / :group_importance)")

    # 文件引用:YAML(features dict 或列名 list)。相对路径优先按
    # run 目录解析(工作区根,Agent 的通常 cwd),再按进程 cwd 兜底
    for cand in (run_dir / src, Path(src)):
        if cand.is_file():
            data = load_yaml(cand)
            if isinstance(data, dict) and "features" in data:
                data = data["features"]
            # v2.10.1/F9:显式清单(文件里的列名)是权威,不求交基础面
            if isinstance(data, dict):
                return _explicit_face(data.get("columns"),
                                      data.get("exclude"), derived,
                                      str(cand))
            if isinstance(data, list):
                return _explicit_face(data, None, derived, str(cand))
            raise SystemExit(
                f"引用失败: {cand} 既不是 features 映射也不是列表")
    raise SystemExit(f"引用失败: {ref} 既不是已落盘 trial 也不是存在的文件")


_HPO_REQUIRED_SYMBOLS = ("fit_and_score", "_hpo_fit_inner",
                         "_INNER_METRIC_UP", "dump_json")

#: [v2.13.4] 模板 HPO 段引用的**符号化依赖**:父代码若定义了这些名字,
#: 它就不再是"缺四件套的朴素父代",而是"自带(旧)HPO 实现的父代" ——
#: 此时整段嫁接会用模板版覆盖父版,而父版可能是为**自己的算法**定制
#: 的(见 `_parent_hpo_overridden_symbols`)。仅用于检测,不参与校验。
_HPO_SECTION_SYMBOLS = ("_hpo_fit_inner", "_fit_with_es", "_fit_with_eval",
                        "_inner_metric", "_build_model")


def _parent_hpo_overridden_symbols(code: str, trial_algo: str) -> list[str]:
    """[v2.13.4] 整段嫁接会**覆盖掉**的、父代码里已有定义的符号清单。

    背景(req-t7-drawdown-v21331 实测事故):`--action hpo --from-parent`
    对**已自带四件套**的父代码(如确认冠军)走整段嫁接,嫁接段追加在
    文件尾 —— 模板版 `_hpo_fit_inner` / `_fit_with_es` 后定义覆盖父版。
    而父版是 Agent 为**父代自己的算法**定制的:

      - 冠军 trial-008(train.py 内 `_hpo_fit_inner` 调 `_lgbm_fit_safe`,
        无 `_fit_with_es`),`import lightgbm` 已在上一次嫁接中被
        importlib 化(lint 规则 2 的机械裁剪),而 `_lgbm_fit_safe`
        对 xgboost 只做裸 `model.fit(X, y)`;
      - v2.13 起 `build_gbm` 对 xgboost **构造期**注入
        `early_stopping_rounds`,裸 fit 无 eval_set →
        `ValueError: Must have at least 1 validation dataset for early
        stopping`(trial-019-g05 实测 25s 即 FAILED);
      - 模板版 `_hpo_fit_inner` 走 `_fit_with_es` 能正确处理 —— 但只有
        **旧版嫁接段**(从 28d2306 前的 templates.py 抽取)才带该函数;
        现行模板 body 抽不出 `_fit_with_es`(它已不与 `dump_json`/
        `step` 同段,且定义在 `fit_and_score` 抽取边界之内),
        于是嫁接产物里 `_fit_with_es` 从未被定义。

    两种情况都让"在冠军加宽面上做 HPO"这条路走不通,而失败发生在
    25 秒的训练期、以 xgboost 的深层 ValueError 呈现,归因成本极高。
    本函数把"将要覆盖父版"这件事显式化,由调用方决定拒绝还是留痕。
    """
    out = []
    for sym in _HPO_SECTION_SYMBOLS:
        if f"\ndef {sym}(" in code:
            out.append(sym)
    return out


def _hpo_parent_face_risk(trial, from_parent: str) -> str | None:
    """[v2.13.4] `--hpo-parent` 继承面 ≠ 父代真实特征面时的风险描述。

    背景(同一次事故):planner 的 `_materialize_command` 对 hpo 动作发
    `--hpo-parent <id>`,而该路径(模板渲染)按 `cmd_trial` 只继承父代
    **config.features 声明**(exclude/banned/reparse/columns)。父代的
    加宽列是**代码 transform 的运行时产物**(如冠军 trial-008 的
    `widen_cached(...)` 生成 387 个 sq_ 列),声明面里根本没有它们 ——
    于是"在冠军特征面上做 HPO"实际跑在**基础面**上,静默降级。

    实测对照(同一数据面两个 run):

      | trial                  | 创建路径       | n_features |
      |------------------------|----------------|------------|
      | v2.13.3.1 trial-018    | --hpo-parent   | 1500(基础) |
      | v2.13.0   trial-014    | --from-parent  | 1887(加宽) |

    同一份 planner 命令,先行 run 之所以拿到加宽面,是因为 Agent 手工
    换成了 --from-parent —— 正确性依赖"操作者偏离官方命令",这才是缺
    陷本身。判据用父代 metrics.json 的 `widen_stats.n_new_cols`
    (transform 加宽的确定性标记),取不到时保守返回 None(不阻断)。
    """
    import json as _json

    try:
        run_dir = Path(trial.trial_dir).parent.parent
        meta_path = run_dir / "trials" / str(from_parent) / "metrics.json"
        if not meta_path.exists():
            return None
        meta = _json.loads(meta_path.read_text(encoding="utf-8")) or {}
        stats = meta.get("widen_stats") or {}
        n_new = int(stats.get("n_new_cols") or 0)
    except Exception:  # noqa: BLE001 读不到就不阻断(与旧行为一致)
        return None
    if n_new <= 0:
        return None
    return (
        f"父代 {from_parent} 的特征面含 {n_new} 个 **transform 加宽列**"
        f"(widen_stats.n_new_cols={n_new});`--hpo-parent` 只继承其 "
        "config.features 声明(列名级),不含代码 transform 的运行时产物 "
        "→ HPO 会静默跑在基础面上,与冠军特征面不一致。"
        "在加宽特征面上做 HPO 必须用 `--from-parent <id>`"
        "(逐字复制父代码,保留 transform;缺四件套时 harness 自动嫁接)。"
    )


_GRAFT_BEGIN = "# ==== v19 hpo 嫁接段 BEGIN"
_GRAFT_END = "# ==== v19 hpo 嫁接段 END ===="
_GRAFT_AUDIT = "# v19.3 嫁接后自动裁剪:"


def _toplevel_names(src: str) -> set | None:
    """片段的顶层定义名(def/class/赋值/import);解析失败返回 None。"""
    import ast as _ast

    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return None
    out: set = set()
    for node in tree.body:
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                             _ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name):
                    out.add(t.id)
        elif isinstance(node, _ast.AnnAssign) and isinstance(
                node.target, _ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (_ast.Import, _ast.ImportFrom)):
            for a in node.names:
                out.add((a.asname or a.name).split(".")[0])
    return out


def _strip_previous_graft_blocks(code: str, new_section: str
                                 ) -> tuple[str, int]:
    """[v2.15.0] 移除代码里已有的嫁接段(BEGIN…END,含其上方 audit 行)。

    返回 (新代码, 移除段数)。安全条件:被移除段的每个顶层名字,要么新
    嫁接段也定义,要么在剩余代码里另有定义;任一段解析失败或不满足
    条件即整体放弃移除(返回原代码, 0),调用方退回追加语义。
    """
    if _GRAFT_BEGIN not in code:
        return code, 0
    new_names = _toplevel_names(new_section)
    if new_names is None:
        return code, 0
    lines = code.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(_GRAFT_BEGIN):
            start = i
            while start > 0 and lines[start - 1].startswith(_GRAFT_AUDIT):
                start -= 1
            j = i
            while j < len(lines) and not lines[j].startswith(_GRAFT_END):
                j += 1
            if j >= len(lines):
                return code, 0  # 有头无尾:结构异常,不动
            spans.append((start, j))
            i = j + 1
        else:
            i += 1
    if not spans:
        return code, 0
    keep = [ln for k, ln in enumerate(lines)
            if not any(a <= k <= b for a, b in spans)]
    rest_names = _toplevel_names("".join(keep)) or set()
    for a, b in spans:
        body = "".join(ln for ln in lines[a:b + 1]
                       if not ln.startswith(_GRAFT_AUDIT))
        old_names = _toplevel_names(body)
        if old_names is None:
            return code, 0
        orphan = old_names - new_names - rest_names
        if orphan:
            logger.warning(
                "hpo 嫁接:旧嫁接段独有符号 %s,保留旧段(退回追加语义)",
                sorted(orphan)[:8])
            return code, 0
    return "".join(keep), len(spans)


def _graft_hpo_section(trial) -> None:
    """v19:向缺四件套的父代码副本嫁接模板 hpo 段(gen-03 实测回写)。

    --action hpo --from-parent 的父代码(如 baseline 血缘 confirm trial
    或手写 widen trial)常缺 hpo 四件套;旧实现直接 SystemExit,迫使
    Agent 二选一:模板渲染路径(丢父代码的加宽 transform)或手改代码
    (违反"零写码快通道"并易引入偏差)。嫁接 = 从渲染 body 源抽取
    "hpo 段"(import time 后的 _T0 到 prepare_features 之前的四件套
    与其依赖),追加注入父代码;父代码已有同名符号时不覆盖(缺什么
    补什么)。嫁接后 _assert_hpo_capable 重验,仍缺则 SystemExit。

    抽取边界约定(与 `_mvo/templates.TRIAL_BODY` 的段落顺序耦合,
    模板结构变更时本函数的三个锚点字符串同步更新):起锚 =
    "\ndef dump_json("(四件套之一),终锚 = "\ndef step(" 前的
    prepare_features 定义起点。fit_and_score 特殊:hpo 分支在
    模板 body 的 fit_and_score 里,若父代码的 fit_and_score 缺
    hpo 分支,嫁接段会在其后追加"hpo 分支注入版"并重定义
    fit_and_score(父版在前、模板版在后 = 模板版生效——与
    Python 后定义覆盖语义一致),同时打告警提示人工确认。
    """
    code_path = Path(trial.trial_dir) / "code" / "train.py"
    code = code_path.read_text(encoding="utf-8")
    # v2.11.9:模板 body 直接来自 _mvo/templates.py。
    # 旧实现读**自己的源文件**、拼 `"\n    body = " + "\'\'\'"` 找锚点、
    # 再 ast.literal_eval 切片还原字符串——模板结构一改就 SystemExit,
    # 还在本文件里留下一段"注释里不能出现该赋值行原文,否则会破坏
    # 测试的朴素抽取"的自我限制(代码被测试的解析方式反向塑形)。
    # 现在模板是模块常量,取用即值,无考古、无转义陷阱。
    body_src = TRIAL_BODY
    # 从渲染 body(实际字符串值)抽取 hpo 段:
    # 起 = "def dump_json(" 行(dump_json 是四件套之一,baseline
    # 血缘父代码缺失,重定义无害);终 = "def step(" 行之前
    # (prepare_features 及之后的内容不在嫁接范围)。
    _anchor_start = "\ndef dump_json("
    _anchor_end = "\ndef step("
    _i0 = body_src.find(_anchor_start)
    _i1 = body_src.find(_anchor_end, _i0)
    if _i0 < 0 or _i1 < 0:
        raise SystemExit(
            "hpo 嫁接失败:模板 body 锚点未找到(模板结构变更,"
            "请同步 _graft_hpo_section 的锚点)")
    hpo_section = body_src[_i0:_i1]
    # hpo 分支在模板 fit_and_score 内(1531-1650 区段)——一并抽取:
    # 模板的完整 fit_and_score 以"后定义覆盖"生效,同时保留父版
    # (注释引用);父 fit_and_score 自定义逻辑(transform 之后的
    # 训练调用形态)若与模板契约不同,hpo 分支行为以模板为准。
    # 抽取段终点 = 模板版 fit_and_score 的函数体结束(_last_model
    # 赋值块后),不含其后的 dump_eval_frame 等 helper(父代码
    # baseline 血缘已带;非 baseline 血缘缺它们时 lint/运行期另有
    # 明确报错,不属 hpo 嫁接职责)。
    _fs_start = body_src.find("\ndef fit_and_score(", _i1)
    _fs_end = body_src.find("\n\n\ndef dump_eval_frame(", _fs_start)
    if _fs_start < 0 or _fs_end < 0:
        raise SystemExit(
            "hpo 嫁接失败:模板 fit_and_score 段锚点未找到")
    fs_section = body_src[_fs_start:_fs_end]
    # ---- v19.3:嫁接段三项裁剪/注入(fix #4) ----------------------------
    # 背景(gen-03 trial-008 实测三处卡点,operator 全部手工就地修补):
    #  (a) 嫁接段不含 `_HPO_ALGO`,模板 _build_model 回退到 lightgbm
    #      ——trial_id 声明 xgboost 却在训 lightgbm,算法口径错;
    #  (b) 嫁接段内 `import lightgbm` 与 LGBM 回退分支对 xgboost trial
    #      触发 lint 规则 2(同 trial 两个 GBM 库 = 疑似融合,直接拒 run);
    #  (c) 嫁接段引用的 _lgbm_fit_safe/_gpu_allocated/n_jobs 定义在
    #      body 的**前置段**(_i0 之前),抽取边界不含它们 —— 父代码
    #      是手写加宽 trial 时这三个名字在模块级不存在,运行期 NameError
    #      (lint 规则 13 的模块级扫描也报不出函数体内的引用)。
    # 抽依赖范围先于 path 注入确定(用 code 原文判定)。
    from runtime.hpo import SUPPORTED_HPO_ALGOS as _SUP_ALGOS

    _trial_algo = str(
        (trial.data or {}).get("algorithm") or "").strip().lower()
    _trial_declared = (Path(trial.trial_dir).name.split("-")[3]
                       if len(Path(trial.trial_dir).name.split("-")) > 3
                       else "")
    if _trial_algo not in _SUP_ALGOS:
        _trial_algo = (_trial_declared if _trial_declared in _SUP_ALGOS
                       else "lightgbm")
    # 依赖闭包:嫁接段引用、但父代码里没有定义的 helper。
    # 三件套 + 其模块级常量(_GPU_MIN_ROWS)一并搬运 —— 只搬函数体
    # 会在运行期炸 NameError,而 lint 规则 13 只扫模块级引用,查不出
    # 函数体内的名字。
    #
    # [FIX-2026-09-17] 依赖原先只从 body **前置段**(_i0 之前)查找,
    # 但 fs_section 引用 helper 的位置并不都在前置段:
    # `_stratified_head_labels` 定义在 body 偏移 ~23k(body 的
    # run_smoke 段,远在 _i1/fs_end 之后)。于是 `--hpo-sample-rows`
    # 在嫁接路径上必然 NameError,又被 fit_and_score 的 `except
    # Exception` 吞成一句 [warn] 后**静默退回全量 inner 搜索** ——
    # 用户以为限制了采样,实际烧全量(实测 drawdown/t7 gen-03
    # trial-016:1433s inner,`hpo_sample_rows` 从未写入 metrics)。
    # 改为在整个 body 里查找(不限前置段),并把找不到的依赖**按
    # 名字**记为致命错误——旧实现只对三件套名单内的名字报
    # _missing_deps,名单外的引用缺定义时无任何提示。
    # 前置段优先:同名符号若在 body 里重复定义,取靠前的那份,保持
    # 与旧行为一致。
    _deps = ""
    _missing_deps = []
    if _i0 > 0:
        import re as _re

        _prefix = body_src[:_i0]
        _called = set(_re.findall(r"\b(_?[A-Za-z]\w*)\s*\(", hpo_section)) \
            | set(_re.findall(r"\b(_?[A-Za-z]\w*)\s*\(", fs_section))
        # v2.15.8:_clamp_trees 入名单。v2.15.0 的 lr-shrink 重训块与
        # _fit_with_es 的 cap 外扩都调用它,但它定义在 body 前置段(_i0
        # 之前),旧名单不含它 → 嫁接路径运行期 NameError(refactor8
        # trial-011-g03-hpo:train.py:2400 `_refit_cap = _clamp_trees(...)`)。
        _dep_names = ["_lgbm_fit_safe", "_gpu_allocated", "n_jobs",
                      "_stratified_head_labels", "_clamp_trees"]
        for _dep in _dep_names:
            # v2.13.5:判"父代码已定义"用 `def <名>(` 而非子串 —— baseline
            # 血缘 main() 里有局部变量 `n_jobs = ...`,子串命中使模块级
            # n_jobs() 从不搬运,嫁接段 uplift/HPO 路径运行期 NameError
            # (req-t7-rand400w gen-04:operator 手补模块级 n_jobs)
            if _dep not in _called or _re.search(
                    rf"(?m)^def {_dep}\s*\(", code):
                continue
            # 先在前置段找(旧行为),找不到再退回整个 body
            # [FIX-2026-09-17] 见上方注释。
            _d = _prefix.find(f"\ndef {_dep}(")
            _src = _prefix
            if _d < 0:
                _d = body_src.find(f"\ndef {_dep}(")
                _src = body_src
            if _d < 0:
                _missing_deps.append(_dep)
                continue
            _dend = _src.find("\ndef ", _d + 1)
            _dend = len(_src) if _dend < 0 else _dend
            _deps += _src[_d:_dend]
        # 被搬运 helper 的模块级依赖常量(_gpu_allocated 用 _GPU_MIN_ROWS)
        if "_gpu_allocated" in _deps:
            _rows = _prefix.find("\n_GPU_MIN_ROWS = ")
            if _rows >= 0 and "_GPU_MIN_ROWS = " not in code:
                _rend = _prefix.find("\n\n", _rows)
                _deps += "\n" + _prefix[_rows:_rend if _rend > 0
                                        else _rows + 80].strip() + "\n"
        # v2.15.8:_clamp_trees 读模块级 _MAX_TREES(树数铁律上限),同法搬运
        if "def _clamp_trees(" in _deps and not _re.search(
                r"(?m)^_MAX_TREES\s*=", code):
            _mt = _prefix.find("\n_MAX_TREES = ")
            if _mt >= 0:
                _mend = _prefix.find("\n\n", _mt)
                _deps = ("\n" + _prefix[_mt:_mend if _mend > 0
                                        else _mt + 40].strip() + "\n"
                         + _deps)
            else:
                _missing_deps.append("_MAX_TREES")
    # [FIX-2026-09-18] 嫁接段引用的**模块别名**必须由父代码提供。
    # 模板 body 假设文件头有 `import pandas as pd` 等(模板渲染路径由
    # TRIAL_TEMPLATE 头部提供),但嫁接路径的父代码可能是任意形态 ——
    # 实测 tests/test_v29_fixes.py 的夹具就是 `import pandas`(无 as pd),
    # 而搬运的 `_stratified_head_labels(y: pd.Series, ...)` 在**签名注解**
    # 阶段就求值 `pd.Series` → NameError,被可导入性探针拦在创建期:
    # "PROBE_FAIL NameError: name 'pd' is not defined"。
    #
    # 修法与 _path_block 的 SKILL_ROOT 注入同构:探到哪个别名缺就补哪个
    # import,幂等(父代码已有 `import pandas as pd` 时 `pd` 已在 code 里,
    # 不重复注入)。
    # 只在**确实搬运了引用该别名的段**时才补 —— 避免给无关 trial 塞 import。
    _ALIAS_IMPORTS = {
        "pd": "import pandas as pd",
        "np": "import numpy as np",
        "yaml": "import yaml",
        "json": "import json",
        "os": "import os",
    }
    _alias_block = ""
    _need_alias = set()
    if _deps:
        for _al in _ALIAS_IMPORTS:
            if _re.search(rf"\b{_al}\.", _deps) and \
                    _re.search(rf"(?m)^\s*(import\s+{_al}\s*$|import\s+\w+\s+as\s+{_al}\s*$)",
                               code):
                continue
            if _re.search(rf"\b{_al}\.", _deps):
                _need_alias.add(_al)
    for _al in sorted(_need_alias):
        _alias_block += f"{_ALIAS_IMPORTS[_al]}\n"
    if _alias_block:
        _alias_block = (
            "\n# v2.12.2.9:搬运段引用的模块别名(父代码可能只写了\n"
            "# `import pandas` 而没有 `as pd`,而搬运 helper 的签名注解\n"
            "# 在模块级求值 pd.Series → NameError 被探针拦在创建期)\n"
            + _alias_block)
    if _missing_deps:
        raise SystemExit(
            f"hpo 嫁接失败:嫁接段依赖 {_missing_deps} 未能在模板 body "
            "定位(不是仅前置段——[FIX-2026-09-17] 已放宽到全 body;"
            "模板结构变更时请同步 _graft_hpo_section 的依赖名单)")
    # v19.3 修复(嫁接段缺 skill 根入 path):嫁接段取自模板 **body**,
    # 而 `SKILL_ROOT = {...}` + sys.path.insert 写在 TRIAL_TEMPLATE 的
    # 头部(marker 段,不在 body 内)—— 于是嫁接出来的代码里
    # `from runtime.hpo import ...` 找不到包。实测 gen-03 trial-008:
    # 首次 run 直接 ModuleNotFoundError(父代码手写加宽 trial 本就不
    # import runtime,operator 只能手补 sys.path 才跑通;该失败还被
    # classify_failure 归成 infra_error 触发无意义重试)。
    # 这里按模板头部同口径注入;名为幂等:父代码若已有 SKILL_ROOT
    # 定义则不再插(手写过 sys.path 的父代码也走这条)。
    _skill_root_tok = repr(str(_SKILL_ROOT))
    _path_block = ""
    if "SKILL_ROOT" not in code:
        # v19.4:自带 import sys —— 嫁接段插在文件尾,引用头部
        # import 过的 sys;但手写父代码可能没 import sys(实测
        # test_v29 夹具 "import pandas" 即触发 NameError),
        # 探针会把它拦在创建期。幂等 import 无副作用。
        _path_block = (
            f"\n# v19.3:skill 根入 sys.path(嫁接段 `from runtime.hpo "
            f"import ...` 的依赖;\n# 模板渲染路径由 TRIAL_TEMPLATE 头部"
            f"注入,嫁接路径此前缺这一步 → ModuleNotFoundError)\n"
            f"import sys as _sys_graft\n"
            f"SKILL_ROOT = {_skill_root_tok}\n"
            f"if SKILL_ROOT not in _sys_graft.path:\n"
            f"    _sys_graft.path.insert(0, SKILL_ROOT)\n")
    else:
        logger.info("父代码已含 SKILL_ROOT 定义,跳过 path 注入")
    _grafted = (f"""{_path_block}{_alias_block}{_deps}
# ==== v19 hpo 嫁接段 BEGIN(harness 自动注入;源 trial 缺 hpo 四件套,
# 从渲染模板抽取,父代码已有符号在下方重定义处被模板版覆盖;
# v19.3 起附带 path/算法/依赖三件修复,见段首 audit 行)====
{hpo_section}
{fs_section}
# ==== v19 hpo 嫁接段 END ====
""")
    # [v2.15.0] 嫁接幂等:父代码若已含**旧嫁接段**(父代本身是嫁接过的
    # HPO trial),先整段移除再追加新段,而不是叠加。旧实现逐代追加 +
    # "后定义覆盖":run3 trial-021 的 train.py 叠了 3 份嫁接段;更糟的是
    # 父代里**已手工修好**的 `_build_model`(补了 cap 透传)被新追加的
    # 模板版覆盖,同一缺陷每建一代复发一次(run2 gen-04、run3 trial-016/
    # 018/021 共 4 次手补)。只在"旧段定义的顶层名字新段都有、或旧段外
    # 另有定义"时移除 —— 否则移除会让旧段独有的符号失效,退回追加语义。
    code, _replaced = _strip_previous_graft_blocks(
        code, hpo_section + "\n" + fs_section)
    if _replaced:
        logger.info("hpo 嫁接:移除父代码中 %d 段旧嫁接段(幂等替换)",
                    _replaced)
        trial.data.setdefault("execution", {})[
            "hpo_graft_replaced_blocks"] = int(_replaced)
        try:
            trial.flush()
        except Exception:  # noqa: BLE001 留痕失败不阻断创建
            pass
    # 嫁接段追加在父代码末尾(后定义覆盖);父代码若在文件尾有
    # main() 调用,嫁接段插在其前(定义先于调用)
    _tail = code.rstrip("\n")
    _main_idx = _tail.rfind('\nif __name__ == "__main__"')
    if _main_idx >= 0:
        new_code = (_tail[:_main_idx] + _grafted
                    + _tail[_main_idx:] + "\n")
    else:
        new_code = _tail + "\n" + _grafted
    code_path.write_text(new_code, encoding="utf-8")

    # ---- v19.3 事后自检 + 机械裁剪 ------------------------------------
    # 上述 (a)(b) 依赖对模板文本做正则,不如"写完再审"稳:直接对产物
    # 做 AST/文本自检,发现口径错就**自动裁掉**并留痕,而不是把问题
    # 留给操作者手改(gen-03 的三处修订全是手工做的,每次嫁接都要重做)。
    _audit_notes: list[str] = []
    _new_code = code_path.read_text(encoding="utf-8")
    # (a) 算法常量:_HPO_ALGO 缺失 → 模板 _build_model 回退 lightgbm
    #     (判定走 AST 的模块级绑定,不用子串包含——见 _has_module_level_algo)
    if "def _build_model" in _new_code \
            and not _has_module_level_algo(_new_code):
        # 锚点优先 SEED;父代码无 SEED 常量(单测/极简父代码)时回退
        # 嫁接段内已存在的 `import time as _time` 行 —— 嫁接段自带该行,
        # 锚点必定存在(否则模板结构已变,上面的抽取早已 SystemExit)。
        _anchor = ("SEED = 20260814" if "SEED = 20260814" in _new_code
                   else "import time as _time")
        if _anchor not in _new_code:
            raise SystemExit(
                "hpo 嫁接后自检失败:需要注入 _HPO_ALGO 但锚点(SEED / "
                "`import time as _time`)均未命中(模板结构变更,请人工"
                "检查 code/train.py)")
        _new_code = _new_code.replace(
            _anchor, f"{_anchor}\n_HPO_ALGO = {_trial_algo!r}", 1)
        _audit_notes.append(f"注入 _HPO_ALGO={_trial_algo!r}")
    # (b) 规则 2:同 trial 两个 GBM 库。嫁接段保留了模板默认的 lightgbm
    #     构造/回退分支;非 lightgbm 算法 trial 下必须裁掉,否则 lint
    #     规则 2 直接拒 run(gen-03 的 trial-008 即因此被拒后再手改)。
    if _trial_algo != "lightgbm" and _has_module_level_algo(_new_code):
        # 裁剪必须**全量**:模板 hpo 段 + fit_and_score 段有多处
        # `import lightgbm`(_build_model 回退分支 / _lgbm_cuda_ok 探测 /
        # _fit_with_eval 早停回调 / uplift 分支 T-learner)。只裁一处
        # 仍会被 lint 规则 2 判"双 GBM 库"直接拒 run(gen-03 trial-008
        # 实测:operator 手改 7 处才过)。
        # 手法统一为 importlib 动态加载(与模板 _xgb_cuda_built 同源):
        # 运行期语义完全不变,静态 import 扫描不再命中。
        import re as _re2

        _pat = _re2.compile(
            r"^(?P<ind>[ \t]*)import lightgbm as (?P<alias>\w+)[ \t]*$",
            _re2.M)
        _n_imp = len(_pat.findall(_new_code))
        _new_code = _pat.sub(
            lambda m: (f"{m.group('ind')}import importlib as "
                       f"_il_{m.group('alias')}\n"
                       f"{m.group('ind')}{m.group('alias')} = "
                       f"_il_{m.group('alias')}.import_module(\"lightgbm\")"),
            _new_code)
        if _n_imp:
            _audit_notes.append(
                f"{_n_imp} 处 lightgbm import 改 importlib(规则 2)")
        # _build_model 的 lgbm 回退分支:即使 import 已动态化,仍把
        # "非 lightgbm 算法走到 lgbm 回退"这一路径显式化(口径清晰)，
        # 便于日后审计。仅当 _HPO_ALGO 已注入(不会回退)时替换。
        _bm_i = _new_code.find("def _build_model(")
        _bm_end = _new_code.find("\ndef ", _bm_i + 1)
        if _bm_i >= 0 and _bm_end > _bm_i:
            _bm = _new_code[_bm_i:_bm_end]
            _imp_i = _bm.find("import importlib as _il__lgb")
            if _imp_i >= 0:
                _bm_new = _bm[:_imp_i] + (
                    f"    # v19.3: 非 lightgbm 算法 trial,lgbm 回退分支已由\n"
                    f"    # harness 显式化(规则 2:同 trial 双 GBM 库 = 疑似融合)\n"
                    f"    raise RuntimeError(\n"
                    f"        \"_HPO_ALGO={_trial_algo} 但 _build_model 走到了\"\n"
                    f"        \" lightgbm 回退分支\")\n")
                _new_code = _new_code[:_bm_i] + _bm_new + _new_code[_bm_end:]
                _audit_notes.append("_build_model 的 LGBM 回退分支显式化")
    if _audit_notes:
        _new_code = _new_code.replace(
            "# ==== v19 hpo 嫁接段 BEGIN",
            "# v19.3 嫁接后自动裁剪: " + "; ".join(_audit_notes)
            + "\n# ==== v19 hpo 嫁接段 BEGIN", 1)
        code_path.write_text(_new_code, encoding="utf-8")
        logger.info("hpo 嫁接后自动裁剪: %s", "; ".join(_audit_notes))
    logger.warning(
        "trial %s: 父代码缺 hpo 四件套,已自动嫁接模板 hpo 段"
        "(父 transform 保留;fit_and_score 以模板 hpo 分支版生效,"
        "请人工确认其训练调用形态与父代码契约一致)",
        trial.trial_id)
    # ---- v2.13.5:死嫁接改线 ------------------------------------------
    # 嫁接段的 fit_and_score 只有被 main() 调用才生效。baseline 血缘父代码
    # 的 main() 内联 build_estimator → fit → 打分,从不调用它 ⇒ config.hpo
    # 永不被消费,trial 静默退化为默认超参重训(req-t7-rand400w gen-04:
    # 两个 HPO trial 各跑 53~65 min 才在评估期暴露,修复 + 重跑又 98 min)。
    # 这里把 main 的内联训练段机械改线为 fit_and_score 调用;改不了就在
    # 创建期失败(回滚 scaffold),而不是等全量训练烧完。
    _code_now = code_path.read_text(encoding="utf-8")
    _rewired, _rw_notes = _rewire_main_to_fit_and_score(_code_now)
    if _rewired is not None:
        code_path.write_text(_rewired, encoding="utf-8")
        logger.warning("trial %s: hpo 嫁接改线 main() → fit_and_score:%s",
                       trial.trial_id, "; ".join(_rw_notes))
    # v2.17.0(#7):嫁接段来自通用模板 body(含分类/uplift 打分分派与
    # hasattr(predict_proba) 分派);回归任务折叠为单分支,lint 规则 4 不再误报
    _nr_n = _narrow_dispatch_file(code_path)
    if _nr_n:
        logger.info("trial %s: 嫁接段回归单分支渲染 %d 处", trial.trial_id, _nr_n)
    # 重验
    still_missing = _assert_hpo_capable(trial, skip=False)
    if still_missing:
        raise SystemExit(
            f"hpo 嫁接后仍缺符号 {still_missing};父代码结构异常,"
            "请人工检查 code/train.py")
    # ---- v19.4:可导入性探针(创建期,不做任何训练) ------------------
    # 符号自检只证"字符串在文件里",不证"能 import"。trial 以
    # cwd=code/ 起进程,`from runtime.hpo import ...` 依赖 SKILL_ROOT
    # bootstrap——历史上这一步只能靠 Agent 首次 run 报
    # ModuleNotFoundError 后读 harness 源码手工补(实测 gen-03 烧掉
    # 20+ 次源码刑侦调用)。探针 = runner 同款环境(cwd=code/,不带
    # PYTHONPATH)做一次模块级编译+导入检查:
    #   1) ast.parse 全文件(语法错,如嫁接锚点错位产出的悬空缩进);
    #   2) importlib 加载(模拟 `python train.py` 的模块级执行路径;
    #      缺 sys.path 时 `from runtime.hpo import` 即抛 ImportError,
    #      嫁接段引用未搬运符号时抛 NameError)。
    # 二分类脚手架的模块级 raise 是渲染路径的合法初始态,探针只拦
    # ImportError/SyntaxError/NameError 三类"结构性不可导入"。
    # 失败即抛 SystemExit —— 调用方(v2.9 事务化创建)回滚 scaffold,
    # Agent 拿到的是创建期报错而非运行期 infra_error 重试。
    import subprocess as _sp

    _probe_src = "import ast,importlib.util,pathlib,sys\np=pathlib.Path('train.py')\nsrc=p.read_text(encoding='utf-8')\nast.parse(src)\nspec=importlib.util.spec_from_file_location('_graft_probe', p)\nmod=importlib.util.module_from_spec(spec)\ntry:\n    spec.loader.exec_module(mod)\nexcept (ImportError, NameError) as e:\n    print('PROBE_FAIL %s: %s' % (type(e).__name__, e))\n    sys.exit(4)\nexcept Exception:\n    pass\nprint('PROBE_OK')\n"
    _probe = _sp.run(
        [sys.executable, "-c", _probe_src],
        cwd=str(Path(trial.trial_dir) / "code"),
        capture_output=True, text=True, timeout=60)
    if _probe.returncode == 4:
        _msg = (_probe.stdout.strip().splitlines()
                or ["PROBE_FAIL(无输出)"])[-1]
        raise SystemExit(
            "hpo 嫁接后可导入性探针失败:" + _msg +
            " —— 嫁接段无法在 cwd=code/ 环境下 import(通常是缺 "
            "SKILL_ROOT/sys.path 或嫁接段引用了未搬运符号)。trial "
            "scaffold 已在创建期失败回滚,请勿手工修改 code/train.py;"
            "确属父代码结构异常时,换模板渲染路径"
            "(--action hpo --algorithm ...)重建")


def _main_calls_name(tree, func_name: str) -> bool:
    import ast as _ast

    mains = [n for n in tree.body
             if isinstance(n, _ast.FunctionDef) and n.name == "main"]
    if not mains:
        return False
    for node in _ast.walk(mains[-1]):
        if isinstance(node, _ast.Call):
            f = node.func
            if (isinstance(f, _ast.Name) and f.id == func_name) or (
                    isinstance(f, _ast.Attribute) and f.attr == func_name):
                return True
    return False


def _rewire_main_to_fit_and_score(code: str) -> tuple[str | None, list[str]]:
    """[v2.13.5] 把 main() 的内联训练段改线为嫁接段 fit_and_score。

    返回 (新代码, 留痕) —— main 已调用 fit_and_score 时返回 (None, [])。
    识别口径(baseline 模板血缘,`--from-parent` 派生的手写 trial 同形):
      起点 = main 顶层第一条含 `build_estimator(` 的语句;
      终点 = 其后第一条含 `validation_score.parquet` 的语句(落盘分数)。
    整段替换为 fit_and_score 调用 + 分数落盘;段内产出、段后仍被引用的
    名字按已知语义补回(model → _last_model、curve_info → 学习曲线、
    score → fit_and_score 返回值),出现未知名字即拒绝(SystemExit)——
    宁可创建期失败,不可生成语义不明的代码。metrics.json 落盘前并入
    fit_and_score 的 hpo 摘要(best_params / hpo_n_configs …,只补不覆盖)。
    同时去掉嫁接 fit_and_score 分类分支的脚手架 raise 标记(hpo trial 走
    hpo 分支,该分支不可达;标记会让 lint 规则 10 直接拒 run)。
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError as exc:
        raise SystemExit(f"hpo 嫁接改线失败:嫁接后代码语法错误 {exc}")
    if _main_calls_name(tree, "fit_and_score"):
        return None, []
    mains = [n for n in tree.body
             if isinstance(n, _ast.FunctionDef) and n.name == "main"]
    if not mains:
        # 无 main() 的代码不是可运行 trial(片段/单测夹具);不改线,由 lint
        # 规则 7/22 在 run 前拦截(无入口 = 无 validation_score 落盘)
        logger.warning("hpo 嫁接:父代码无 main(),跳过改线(run 前 lint "
                       "会拦截无入口/无分数落盘的代码)")
        return None, []
    main = mains[-1]
    lines = code.splitlines(keepends=True)

    def _seg(node) -> str:
        return "".join(lines[node.lineno - 1:node.end_lineno])

    body = main.body
    i0 = next((i for i, st in enumerate(body)
               if "build_estimator(" in _seg(st)), None)
    i1 = None
    if i0 is not None:
        i1 = next((i for i in range(i0, len(body))
                   if "validation_score.parquet" in _seg(body[i])), None)
    if i0 is None or i1 is None:
        raise SystemExit(
            "hpo 嫁接改线失败:父代码 main() 不调用 fit_and_score,且识别不出"
            "内联训练段(需含 `build_estimator(` … `validation_score.parquet`)。"
            "嫁接段是死代码 —— HPO 会静默退化为默认超参重训。请在父代码"
            "main() 里把训练+打分段改为 `score, _m = fit_and_score(X_tr, y_tr,"
            " X_va, y_va, seed, cfg)` 后再派生 hpo trial")

    def _stored(nodes) -> set:
        out = set()
        for st in nodes:
            for n in _ast.walk(st):
                if isinstance(n, _ast.Name) and isinstance(
                        n.ctx, (_ast.Store, _ast.Del)):
                    out.add(n.id)
        return out

    def _loaded(nodes) -> set:
        out = set()
        for st in nodes:
            for n in _ast.walk(st):
                if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Load):
                    out.add(n.id)
        return out

    before = _stored(body[:i0]) | {a.arg for a in main.args.args}
    need = {"X_tr", "y_tr", "X_va", "y_va", "cfg"}
    missing_in = sorted(need - before)
    if missing_in:
        raise SystemExit(
            f"hpo 嫁接改线失败:main() 训练段之前缺 {missing_in}(fit_and_score "
            "入参);请人工改线后再派生 hpo trial")
    produced = _stored(body[i0:i1 + 1])
    used_after = _loaded(body[i1 + 1:])
    carried = produced & used_after
    known = {"model", "score", "curve_info", "eval_set", "n_jobs"}
    unknown = sorted(carried - known)
    if unknown:
        raise SystemExit(
            f"hpo 嫁接改线失败:训练段产出的 {unknown} 在其后仍被引用,"
            "无法机械补回;请人工改线(main 调用 fit_and_score)后再派生")
    ind = " " * (body[i0].col_offset)
    has_seed = "SEED" in {n.id for n in _ast.walk(tree)
                          if isinstance(n, _ast.Name)}
    seed_expr = ('int(cfg.get("seed", SEED))' if has_seed
                 else 'int(cfg.get("seed", 0))')
    repl = [
        ind + "# ==== v2.13.5 harness 自动改线(hpo 嫁接)BEGIN ====",
        ind + "# 原段 build_estimator → fit → 打分 → validation_score 落盘不消费",
        ind + "# config.hpo;改由嫁接段 fit_and_score 在 train 内切的 inner 上",
        ind + "# 搜索、best_params 全量重训,outer validation 只评一次。",
        ind + "score, _fs_metrics = fit_and_score(",
        ind + "    X_tr, y_tr, X_va, y_va, " + seed_expr + ", cfg)",
        ind + "model = _last_model",
    ]
    if "curve_info" in carried:
        repl += [
            ind + "try:",
            ind + "    curve_info = dump_learning_curve(model)",
            ind + "except Exception:  # noqa: BLE001  hpo 重训模型无 eval 曲线时",
            ind + "    curve_info = {}",
        ]
    if "eval_set" in carried:
        # [v2.15.0] 训练段之前已切出 es 切片时把它交给后续段(multi-seed
        # 变体的 xgboost 构造期早停需要 eval_set;旧实现恒置 None →
        # `Must have at least 1 validation dataset for early stopping`)
        if {"X_es", "y_es"} <= before:
            repl.append(ind + "eval_set = [(X_es, y_es)]")
        else:
            repl.append(ind + "eval_set = None")
    if "n_jobs" in carried:
        repl.append(ind + 'n_jobs = max(1, int(os.environ.get('
                    '"OMP_NUM_THREADS", "4") or 4))')
    if "def dump_feature_importance(" in code:
        repl += [
            ind + "try:",
            ind + "    dump_feature_importance(model, X_tr.columns)",
            ind + "except Exception:  # noqa: BLE001",
            ind + "    pass",
        ]
    repl += [
        ind + 'pd.DataFrame({"score": score}).to_parquet(',
        ind + '    TRIAL_DIR / "validation_score.parquet", index=False)',
        ind + "# ==== v2.13.5 harness 自动改线 END ====",
    ]
    repl = [ln + "\n" for ln in repl]
    notes = [f"main 第 {body[i0].lineno}-{body[i1].end_lineno} 行内联训练段 → "
             "fit_and_score"]
    # metrics.json 落盘前并入 hpo 摘要(只补不覆盖)
    merge_at = None
    merge_var = None
    for st in body[i1 + 1:]:
        if isinstance(st, _ast.With) and "metrics.json" in _seg(st):
            for n in _ast.walk(st):
                if isinstance(n, _ast.Call) and isinstance(
                        n.func, _ast.Attribute) and n.func.attr == "dump" \
                        and n.args and isinstance(n.args[0], _ast.Name):
                    merge_var = n.args[0].id
            merge_at = st
            break
    new_lines = list(lines)
    if merge_at is not None and merge_var:
        mind = " " * merge_at.col_offset
        new_lines[merge_at.lineno - 1:merge_at.lineno - 1] = [
            mind + "# v2.13.5 改线:并入 fit_and_score 的 hpo 摘要(只补不覆盖)\n",
            mind + "for _k, _v in (_fs_metrics or {}).items():\n",
            mind + "    " + merge_var + ".setdefault(_k, _v)\n",
            # v2.16.0:main 的 baseline 血缘段会先按 best_iteration/cap 比值
            # 写一份启发式 es_info(v2.15.8),setdefault 会把 fit_and_score
            # 的真实早停遥测(stage1_rounds/cap 外扩/es_source)挡掉
            mind + "if (_fs_metrics or {}).get(\"es_info\"):\n",
            mind + "    " + merge_var + "[\"es_info\"] = _fs_metrics[\"es_info\"]\n",
        ]
        notes.append(f"metrics.json 并入 hpo 摘要({merge_var})")
    else:
        notes.append("未找到 metrics.json 落盘点:hpo 摘要仅在 hpo-study.json")
    # 先替换靠后的 merge 插入(上面已做)不影响前面的行号;训练段在其之前
    new_lines[body[i0].lineno - 1:body[i1].end_lineno] = repl
    out = "".join(new_lines)
    # 嫁接 fit_and_score 分类分支的脚手架 raise 标记(hpo trial 不可达)
    _marker = "  # lint:scaffold-fit-and-score"
    if _marker in out:
        out = out.replace(_marker, "  # v2.13.5:hpo 改线后本分支不可达")
        notes.append("去掉嫁接段脚手架 raise 标记(规则 10)")
    if "\n_last_model = None" not in out:
        _anchor = "SEED = 20260814"
        if _anchor in out:
            out = out.replace(
                _anchor, _anchor + "\n_last_model = None  # fit_and_score 写入"
                "(v2.13.5 改线:main 引用)", 1)
    # v2.17.2(A5):setdefault 并入挡不住 main 自己写的 n_features(基础列数),
    # 改线同时让拟合期模型列数胜出(基础列数另存 n_features_base)
    out, _nf_notes = _upgrade_n_features_code(out)
    notes.extend(_nf_notes)
    try:
        _ast.parse(out)
    except SyntaxError as exc:
        raise SystemExit(
            f"hpo 嫁接改线后语法错误(harness 缺陷,请上报):{exc}")
    return out, notes


_HPO_RENDER_ALGOS = ("lightgbm", "xgboost", "catboost")


def _has_module_level_algo(code: str) -> bool:
    """模块级是否已绑定 `_HPO_ALGO`(AST 判定,不被注释/字符串误命中)。"""
    import ast as _ast

    try:
        _tree = _ast.parse(code)
    except SyntaxError:
        # 语法有问题的代码不该静默放行注入;交给调用方的语法检查暴露。
        return False
    for _node in _tree.body:
        if isinstance(_node, _ast.Assign):
            for _t in _node.targets:
                if isinstance(_t, _ast.Name) and _t.id == "_HPO_ALGO":
                    return True
        elif isinstance(_node, _ast.AnnAssign) and \
                isinstance(_node.target, _ast.Name) and \
                _node.target.id == "_HPO_ALGO":
            return True
    return False


def _inject_hpo_algo_at(code: str, algo: str) -> str | None:
    """在 SEED 锚点后注入 `_HPO_ALGO = algo`(幂等;已存在返回 None)。

    v19.3:hpo 三条创建路径(模板渲染 / --from-parent 快通道 / 嫁接)
    共用同一注入实现。缺该常量时模板 _build_model 会**回退 lightgbm**
    —— 搜索空间的键按父代算法命名,trial_id 也按算法命名,不注入等于
    "声明 xgboost 实训 lightgbm",整轮搜索口径作废。

    v19.3a(实测回写):幂等判定必须是"**模块级赋值**存在",不能是
    子串包含 —— 嫁接段正文里有 `globals().get("_HPO_ALGO")` 与注释
    提及该名(共 6 处子串、0 处赋值),子串判定会让注入静默跳过,
    正是本函数要修的那个坑在自身复现。

    锚点未命中抛 SystemExit(拒绝静默生成口径不符的 hpo trial)。
    """
    if _has_module_level_algo(code):
        return None
    anchor = "SEED = 20260814"
    if anchor not in code:
        raise SystemExit(
            "SEED 锚点未命中(模板/父代码结构变更,拒绝静默生成"
            "无算法常量的 hpo trial)")
    out = code.replace(anchor, f"{anchor}\n_HPO_ALGO = {algo!r}", 1)
    return _neutralize_scaffold_marker(out)


#: [v2.15.0] 模板 fit_and_score 的算法化非 hpo 分支(confirm/参数子代)起止锚
_GENERIC_BRANCH_BEGIN = '    elif globals().get("_HPO_ALGO"):\n'
_GENERIC_BRANCH_END = '    elif MODEL_TYPE == "regression":\n'
_SCAFFOLD_MARKER = "  # lint:scaffold-fit-and-score"


def _neutralize_scaffold_marker(code: str) -> str:
    """[v2.15.0] `_HPO_ALGO` 已注入且算法化分支在位时,脚手架 raise 不可达。

    模板 fit_and_score 的二分类 else 分支带 lint 规则 10 的标记。旧版
    只有"嫁接改线"会去掉它,模板渲染的 hpo trial 与重复嫁接(父代已
    改线、本次不再改线)都保留标记 → lint 拒跑,run3 每轮 HPO 都靠
    Agent 手删(gen-04 反思留痕"去 lint 标记")。算法化分支
    (`elif globals().get("_HPO_ALGO")`)在 else 之前接管全部二分类/回归
    路径,标记即可安全替换为说明注释。
    """
    if _SCAFFOLD_MARKER not in code or _GENERIC_BRANCH_BEGIN not in code \
            or not _has_module_level_algo(code):
        return code
    return code.replace(
        _SCAFFOLD_MARKER,
        "  # v2.15.0:_HPO_ALGO 已注入,算法化分支接管,本分支不可达")


def _upgrade_hpo_lineage_code(code: str) -> tuple[str, list[str]]:
    """[v2.15.0] 给 HPO 血缘的旧代码补算法化非 hpo 分支与 seed 变体实现。

    `--action confirm`(或 --set 参数子代)`--from-parent <HPO trial>`
    剥离 hpo 节后,旧模板的 fit_and_score 没有可执行的二分类路径 → 落进
    脚手架 raise(run3 trial-017 code_error;run2/run3 共 5 次)。旧父代
    代码已冻结,这里在**子代副本**上补两段(均取自现行模板,单源):
      1. 生效的(最后一个)fit_and_score 里,在 regression 分支之前插入
         算法化分支;
      2. 文件尾补 `_fit_seed_variant`(与主轮同配方/同数据/同早停机制,
         修 eval_set=None 与"变体用默认超参"两处缺陷)。
    只对含模块级 `_HPO_ALGO` 的代码生效;已含算法化分支时不动。
    """
    notes: list[str] = []
    if not _has_module_level_algo(code):
        return code, notes
    out = code
    if _GENERIC_BRANCH_BEGIN not in out:
        i = TRIAL_BODY.find(_GENERIC_BRANCH_BEGIN)
        j = TRIAL_BODY.find(_GENERIC_BRANCH_END, i)
        k = out.rfind("\ndef fit_and_score(")
        if i >= 0 and j > i and k >= 0:
            branch = TRIAL_BODY[i:j]
            # 旧代码没有 _es_info_of:分支内改用就地读取(同语义)
            branch = branch.replace(
                "_es = _es_info_of(model)",
                "_es = {_k: _v for _k, _v in (getattr(model, "
                "\"mvo_es_info_\", None) or {}).items() "
                "if _k != \"stage1_curve\"}")
            m = out.find(_GENERIC_BRANCH_END, k)
            if m > k:
                out = out[:m] + branch + out[m:]
                notes.append("fit_and_score 补算法化非 hpo 分支(confirm 可执行)")
    if "\n_PRIMARY_FIT_XY = None" not in out:
        i = TRIAL_BODY.find("\n#: [v2.15.0] 主轮 fit_and_score 的重训配方")
        j = TRIAL_BODY.find("\ndef step(", i)
        if i >= 0 and j > i:
            seg = TRIAL_BODY[i:j].rstrip("\n") + "\n"
            tail = out.rstrip("\n")
            mi = tail.rfind('\nif __name__ == "__main__"')
            out = (tail[:mi] + "\n" + seg + tail[mi:] + "\n") if mi >= 0 \
                else (tail + "\n" + seg)
            notes.append("补 _fit_seed_variant(multi-seed 与主轮同配方)")
    out = _neutralize_scaffold_marker(out)
    return out, notes


#: [v2.16.0] validation 早停升级段标记(幂等判据)
_ES_UPGRADE_BEGIN = "# ==== v2.16 validation-ES 段 BEGIN"
_ES_UPGRADE_END = "# ==== v2.16 validation-ES 段 END ===="
#: train 尾部切分的锚点行(baseline 血缘 main / 手写 fit_and_score)与守卫标记
_ES_MAIN_ANCHOR = "y_fit, y_es = y_tr.iloc[:_es_cut], y_tr.iloc[_es_cut:]"
_ES_MAIN_GUARD = "# [v2.16.0 es_source]"
#: 新段依赖的模板 helper(旧代码缺哪个补哪个;已有的保留,后定义覆盖)
_ES_DEP_FUNCS = ("_clamp_trees", "_es_eval_cap", "_es_stage1",
                 "_es_rounds_for",
                 "_model_cap", "_set_cap", "_es_curve_of", "_es_best_trees",
                 "_fit_with_es")
#: hurdle 血缘(回归 run 的 `_fit_hurdle` / `_fit_hurdle_refit_full` 两段式)
_ES_HURDLE_BEGIN = "# ==== v2.16 hurdle validation-ES 段 BEGIN"
_ES_HURDLE_END = "# ==== v2.16 hurdle validation-ES 段 END ===="
_ES_HURDLE_BLOCK = '''_LAST_EVAL_XY = globals().get("_LAST_EVAL_XY")
_fit_hurdle_refit_full_v215 = _fit_hurdle_refit_full
_fit_and_score_v215 = fit_and_score


def fit_and_score(X_tr, y_tr, X_va, y_va=None, *args, **kwargs):
    """v2.16.0:记下 outer validation(hurdle 单段早停面),其余原样委托。"""
    global _LAST_EVAL_XY
    if X_va is not None and y_va is not None:
        _LAST_EVAL_XY = (X_va, y_va)
    return _fit_and_score_v215(X_tr, y_tr, X_va, y_va, *args, **kwargs)


def _fit_hurdle_refit_full(n_jobs, X_tr, y_tr):
    """v2.16.0:validation 口径下 hurdle 两个子模型在 100% train 上只训一次、
    在 outer validation 上早停(_fit_hurdle 原样),不再"train 尾部早停 →
    全量定轮重训"两段式;train_tail 或拿不到 validation 时回退原实现。
    stage1_ 遥测键保持不变。"""
    _ev = globals().get("_LAST_EVAL_XY")
    if _es_source() != "validation" or _ev is None or _ev[1] is None:
        return _fit_hurdle_refit_full_v215(n_jobs, X_tr, y_tr)
    X_ev, y_ev = _ev
    m = _fit_hurdle(n_jobs, X_tr, y_tr, X_ev, y_ev)
    _cb = int(N_ESTIMATORS_CAP)
    _sr = globals().get("_stage1_rounds")
    _rc = _sr(m.clf_) if _sr else None
    _rr = _sr(m.reg_) if _sr else None
    _npos = int(np.sum(np.asarray(y_tr) > 0))
    st = {"clf_best_iteration": int(m.clf_.best_iteration) + 1,
          "reg_best_iteration": int(m.reg_.best_iteration) + 1,
          "clf_rounds": _rc, "reg_rounds": _rr,
          "n_fit_rows": int(len(X_tr)), "n_es_rows": int(len(X_ev)),
          "n_reg_fit_rows": _npos, "refit_reg_rows": _npos,
          "clf_cap_bound": bool(_rc is not None and _rc >= _cb),
          "reg_cap_bound": bool(_rr is not None and _rr >= _cb),
          "es_source": "validation", "fit_mode": "single"}
    try:
        m.stage1_ = st
    except Exception:  # noqa: BLE001
        pass
    return m'''


def _template_func_sources() -> dict[str, str]:
    """TRIAL_BODY 顶层函数名 → 源码(单源:升级段一律取现行模板原文)。"""
    import ast as _ast

    tree = _ast.parse(TRIAL_BODY)
    return {n.name: _ast.get_source_segment(TRIAL_BODY, n)
            for n in tree.body if isinstance(n, _ast.FunctionDef)}


def _insert_before_main(out: str, block: str) -> str:
    tail = out.rstrip("\n")
    mi = tail.rfind('\nif __name__ == "__main__"')
    return (tail[:mi] + "\n\n\n" + block + tail[mi:] + "\n") if mi >= 0 \
        else (tail + "\n\n\n" + block)


def _guard_es_anchors(out: str) -> tuple[str, int]:
    """每处 train 尾部切分锚点行之后注入 validation 早停守卫(已有则跳过)。"""
    n = 0
    pos = 0
    while True:
        at = out.find(_ES_MAIN_ANCHOR, pos)
        if at < 0:
            return out, n
        le = out.find("\n", at)
        le = len(out) if le < 0 else le
        nxt = out[le + 1:out.find("\n", le + 1) if out.find("\n", le + 1) >= 0
                  else len(out)]
        if _ES_MAIN_GUARD in nxt:
            pos = le + 1
            continue
        ls = out.rfind("\n", 0, at) + 1
        indent = out[ls:at]
        guard = (
            f"\n{indent}if (_es_source() == \"validation\" and MODEL_TYPE != "
            f"\"uplift\" and locals().get(\"y_va\") is not None):  "
            f"{_ES_MAIN_GUARD}\n"
            f"{indent}    X_fit, y_fit, X_es, y_es = X_tr, y_tr, X_va, y_va\n"
            f"{indent}    step(\"v2.16.0: early stopping on outer validation"
            f"(单段,100% train)\")")
        out = out[:le] + guard + out[le:]
        pos = le + len(guard) + 1
        n += 1


def _upgrade_es_source_code(code: str) -> tuple[str, list[str]]:
    """[v2.16.0] 把 --from-parent / warm-start / 全局库复制来的旧代码升级到
    "outer validation 早停、100% train 只训一次"。

    旧父代代码已冻结,这里只改**副本**,三处(模板段取自现行模板,单源):
      1. 含 `_fit_with_eval` 但没有 `_fit_es_on_validation` 的代码:在
         `if __name__ == "__main__"` 之前插入带标记的段,重定义
         `_es_source / _fit_es_on_validation / _fit_with_eval /
         dump_learning_curve`(有 `_build_model` 时连同 `_fit_seed_variant`),
         缺失的依赖 helper 一并补齐(后定义覆盖);
      2. 每处 train 尾部切分(锚点行,baseline 血缘 main / 手写
         fit_and_score)之后注入守卫:validation 模式且拿得到 y_va 时
         X_fit/X_es 改为全量 train / outer validation(uplift 维持 train 尾部);
      3. hurdle 血缘(`_fit_hurdle` + `_fit_hurdle_refit_full` 两段式):
         包一层 fit_and_score 记下 outer validation,`_fit_hurdle_refit_full`
         在 validation 模式下改为 `_fit_hurdle` 全量单段早停。
    运行期仍按 MODELEVO_ES_SOURCE 分派:train_tail 时行为与 v2.15 相同。
    幂等:各处已升级时不动。
    """
    notes: list[str] = []
    out = code
    needs_fit = ("\ndef _fit_with_eval(" in out
                 and "\ndef _fit_es_on_validation(" not in out
                 and _ES_UPGRADE_BEGIN not in out)
    needs_hurdle = ("\ndef _fit_hurdle_refit_full(" in out
                    and "\ndef _fit_hurdle(" in out
                    and "\ndef fit_and_score(" in out
                    and _ES_HURDLE_BEGIN not in out)
    out, n_guard = _guard_es_anchors(out)
    if n_guard:
        notes.append(f"train 尾部切分注入 validation 早停守卫 ×{n_guard}")
    if not (needs_fit or needs_hurdle or n_guard):
        return code, notes
    if needs_fit or "\ndef _es_source(" not in out:
        src = _template_func_sources()
        parts: list[str] = []
        if needs_fit:
            if "\n_MAX_TREES = " not in out:
                parts.append("_MAX_TREES = 8000")
            for name in _ES_DEP_FUNCS:
                if f"\ndef {name}(" not in out and name in src:
                    parts.append(src[name])
        if "\ndef _es_source(" not in out:
            parts.append(src["_es_source"])
        if needs_fit:
            parts.append("_LAST_EVAL_XY = None")
            parts.extend(src[n] for n in ("_fit_es_on_validation",
                                          "_fit_with_eval",
                                          "dump_learning_curve"))
            if "\ndef _build_model(" in out:
                parts.append(src["_fit_seed_variant"])
        block = (f"{_ES_UPGRADE_BEGIN}(harness 自动注入;源 trial 早于 "
                 f"v2.16.0,早停改为 outer validation 单段拟合)====\n"
                 + "import os\n\n\n"
                 + "\n\n\n".join(parts) + f"\n{_ES_UPGRADE_END}\n")
        out = _insert_before_main(out, block)
        notes.append("插入 v2.16 validation-ES 段"
                     + ("(_fit_with_eval 改为单段)" if needs_fit else ""))
    if needs_hurdle:
        block = (f"{_ES_HURDLE_BEGIN}(harness 自动注入;hurdle 两段式改为 "
                 f"outer validation 单段早停)====\n"
                 + _ES_HURDLE_BLOCK + f"\n{_ES_HURDLE_END}\n")
        out = _insert_before_main(out, block)
        notes.append("hurdle 两段式改为 validation 单段早停")
    return out, notes


# ==================== 全局库配方单段硬化 ====================
# `_upgrade_es_source_code` 只做运行期分派:validation 走单段,两段式
# (`_fit_with_es` 第二段全量重训 / `_fit_hurdle_refit_full`)作为 train_tail
# 与异常回退仍留在代码里。全局库配方是跨 run 复用的起点,这里把它们改成
# 任何口径都只训一次:
#   - validation(默认):100% train 拟合、outer validation 早停;
#   - train_tail(交付在 train+validation 上复跑复合容器配方时据此避开样本内
#     早停;run 配置 train_tail 同样落到这里)/ 拿不到 y_va:train 头部拟合、
#     尾部 ES_FRACTION 早停,不再全量重训。
# 单段拟合异常直接抛出,不再静默回退两段式。

#: 硬化版函数的标记(docstring / 注释里出现,便于审计)
_ES_HARD_MARK = "[全局库单段]"

_HARD_ES_SOURCE = '''def _es_source() -> str:
    """[全局库单段] outer 拟合的早停面;两种口径都只训一次,没有全量重训第二段。

    - validation(默认):100% train 拟合、outer validation 早停;
    - train_tail:train 头部拟合、尾部 ES_FRACTION 早停。交付在
      train+validation 上复跑复合容器配方时据此避开样本内早停;run 配置
      training.early_stop_source: train_tail 同样落到这里。
    MODELEVO_ES_SOURCE 优先,其次模块常量 ES_SOURCE(baseline 模板)。
    """
    _v = str(os.environ.get("MODELEVO_ES_SOURCE")
             or globals().get("ES_SOURCE") or "validation")
    return "train_tail" if _v.strip().lower() == "train_tail" else "validation"'''

_HARD_FIT_ES_ON_VALIDATION = '''def _fit_es_on_validation(model, X_tr, y_tr, X_va, y_va, rounds=None,
                          es_source="validation", extend_cap=True):
    """[全局库单段] 单段拟合:X_tr 只训一次,早停在 (X_va, y_va) 上。

    - 早停面默认是 outer validation;_fit_with_es 以 train 尾部调用时
      es_source="train_tail";
    - 复用 _es_stage1(各库原生早停),早停耐心沿用 _es_rounds_for;
    - extend_cap=True 时 cap 一次性放到 cap × MODELEVO_CAP_EXTEND_FACTOR ^
      MODELEVO_CAP_EXTEND_TIMES(钳到 _MAX_TREES):早停会先叫停,大 cap 无
      额外成本,不撞 cap 重跑;
    - 遥测挂 mvo_es_info_(stage1_* / cap / cap_bound / stopped_by /
      es_source / fit_mode=single),无 refit_trees;v2.17.2:外扩只在剩余
      余量内(cand > 当前 cap 才外扩),撞 cap 且 cap 已达 _MAX_TREES 时
      cap_bound_unresolved=True;
    - 预测走最优轮数:xgboost/lightgbm 的 sklearn API 早停后自动用 best,
      catboost 带 eval_set 时默认 use_best_model;
    - 非 GBM 族(sklearn/torch)直接 fit 一次;拟合异常直接抛出,不回退两段式。
    """
    import math as _math_v

    if rounds is None:
        try:
            rounds = int(os.environ.get("MODELEVO_EARLY_STOP_ROUNDS")
                         or globals().get("EARLY_STOP_ROUNDS") or 50)
        except (TypeError, ValueError):
            rounds = 50
    _fam = str(type(model).__module__ or "").split(".")[0]
    if _fam not in ("lightgbm", "xgboost", "catboost"):
        return _lgbm_fit_safe(model, X_tr, y_tr) or model
    rounds = _es_rounds_for(model, rounds)
    _cap0 = _model_cap(model, _fam)
    _times = (int(os.environ.get("MODELEVO_CAP_EXTEND_TIMES", "1") or 0)
              if extend_cap else 0)
    _factor = float(os.environ.get("MODELEVO_CAP_EXTEND_FACTOR", "2") or 2)
    _cap_ext = None
    if _cap0 and _times > 0 and _factor > 1:
        _cand = _clamp_trees(int(_math_v.ceil(
            int(_cap0) * (_factor ** _times))))
        if _cand > int(_cap0):
            model = _set_cap(model, _fam, _cand)
            _cap_ext = _cand
    info = {"es_rounds": int(rounds), "cap_initial": _cap0,
            "cap_extended_to": _cap_ext, "es_source": str(es_source),
            "fit_mode": "single", "n_es_rows": int(len(X_va)),
            # v2.17.3:早停集可能被降采样,实际参与早停的行数单列
            # (由 MODELEVO_ES_EVAL_ROWS 决定,与 _es_stage1 同口径)
            "n_es_eval_rows": min(
                int(len(X_va)),
                int(os.environ.get("MODELEVO_ES_EVAL_ROWS") or 0)
                or int(len(X_va)))}
    model = _es_stage1(model, _fam, X_tr, y_tr, X_va, y_va, rounds)
    _curve = _es_curve_of(model, _fam)
    _best = _es_best_trees(model, _fam)
    _cap = _model_cap(model, _fam)
    _bound = bool(_cap and len(_curve) >= int(_cap))
    # v2.17.2(A1):撞 cap 且 cap 已达 _MAX_TREES(无外扩余量)= 截断未解除
    info["cap_bound_unresolved"] = bool(_bound and int(_cap or 0) >= int(
        globals().get("_MAX_TREES") or 8000))
    info.update({"stage1_best_iteration": _best,
                 "stage1_rounds": len(_curve) or None,
                 "cap": _cap, "cap_bound": _bound,
                 "stopped_by": "cap" if _bound else "early_stop",
                 "stage1_curve": [round(float(v), 6) for v in _curve]})
    try:
        model.mvo_es_info_ = info
    except Exception:  # noqa: BLE001
        pass
    return model'''

_HARD_FIT_WITH_ES = '''def _fit_with_es(model, X, y, es_fraction=0.1, rounds=None, refit=False,
                 auto_extend=False):
    """[全局库单段] train 尾部早停:X 头部 1-es_fraction 拟合、尾部早停,只训一次。

    全局库配方没有"定轮 → 全量重训"第二段:refit 只为调用兼容保留、不生效,
    返回的就是早停模型;auto_extend=True 时 cap 一次性外扩(同
    _fit_es_on_validation),不撞 cap 重跑。调用方:HPO inner(inner 训练集)
    与 _fit_with_eval 的 train 尾部口径。只切**传入的 X/y**,不引用 outer
    validation。非 GBM 族或切不出尾部(数据过小)时全量 fit 一次;拟合异常
    直接抛出,不回退。
    """
    _fam = str(type(model).__module__ or "").split(".")[0]
    _n = len(X)
    _cut = int(_n * (1.0 - float(es_fraction)))
    if _fam not in ("lightgbm", "xgboost", "catboost") \\
            or _cut <= 0 or _cut >= _n:
        return _lgbm_fit_safe(model, X, y) or model
    return _fit_es_on_validation(model, X.iloc[:_cut], y.iloc[:_cut],
                                 X.iloc[_cut:], y.iloc[_cut:], rounds=rounds,
                                 es_source="train_tail",
                                 extend_cap=bool(auto_extend))'''

_HARD_FIT_WITH_EVAL = '''def _fit_with_eval(model, X_tr, y_tr, X_va, y_va):
    # [全局库单段] outer 拟合只训一次:默认 100% train 拟合、outer validation
    # 早停(_fit_es_on_validation);MODELEVO_ES_SOURCE=train_tail 或拿不到
    # y_va 时 train 尾部早停(_fit_with_es,同样单段,不全量重训)。
    # 返回拟合后的模型——catboost 重建路径会换对象,调用方必须
    # 用返回值重绑定(model = _fit_with_eval(...))。
    global _LAST_EVAL_XY
    if X_va is not None and y_va is not None:
        _LAST_EVAL_XY = (X_va, y_va)
        if _es_source() == "validation":
            return _fit_es_on_validation(model, X_tr, y_tr, X_va, y_va)
    return _fit_with_es(model, X_tr, y_tr, auto_extend=True)'''

#: hurdle 血缘的硬化段(替换 `_ES_HURDLE_BLOCK`:去掉 `_v215` 两段式别名与回退)
_ES_HURDLE_HARD_BLOCK = '''_LAST_EVAL_XY = globals().get("_LAST_EVAL_XY")
_fit_and_score_v215 = fit_and_score


def fit_and_score(X_tr, y_tr, X_va, y_va=None, *args, **kwargs):
    """v2.16.0:记下 outer validation(hurdle 单段早停面),其余原样委托。"""
    global _LAST_EVAL_XY
    if X_va is not None and y_va is not None:
        _LAST_EVAL_XY = (X_va, y_va)
    return _fit_and_score_v215(X_tr, y_tr, X_va, y_va, *args, **kwargs)


def _fit_hurdle_refit_full(n_jobs, X_tr, y_tr):
    """[全局库单段] hurdle 两个子模型只训一次(_fit_hurdle 原样),不再
    "train 尾部早停定轮 → 100% train 定轮重训"两段式。

    - 默认:100% train 拟合、outer validation 早停;
    - MODELEVO_ES_SOURCE=train_tail(交付在 train+validation 上复跑)或拿不到
      validation:train 头部拟合、尾部 ES_FRACTION 早停。
    函数名沿用旧名(fit_and_score / _fit_seed_variant 调用点不变);遥测挂
    stage1_,键名沿用,另记 es_source / fit_mode=single。
    """
    _ev = globals().get("_LAST_EVAL_XY")
    if _es_source() == "validation" and _ev is not None \\
            and _ev[1] is not None:
        X_fit, y_fit = X_tr, y_tr
        X_es, y_es = _ev
        _src = "validation"
    else:
        _cut = int(len(X_tr) * (1.0 - float(
            globals().get("ES_FRACTION") or 0.1)))
        X_fit, X_es = X_tr.iloc[:_cut], X_tr.iloc[_cut:]
        y_fit, y_es = y_tr.iloc[:_cut], y_tr.iloc[_cut:]
        _src = "train_tail"
    m = _fit_hurdle(n_jobs, X_fit, y_fit, X_es, y_es)
    _cb = int(N_ESTIMATORS_CAP)
    _sr = globals().get("_stage1_rounds")
    _rc = _sr(m.clf_) if _sr else None
    _rr = _sr(m.reg_) if _sr else None
    _npos = int(np.sum(np.asarray(y_fit) > 0))
    st = {"clf_best_iteration": int(m.clf_.best_iteration) + 1,
          "reg_best_iteration": int(m.reg_.best_iteration) + 1,
          "clf_rounds": _rc, "reg_rounds": _rr,
          "n_fit_rows": int(len(X_fit)), "n_es_rows": int(len(X_es)),
          "n_reg_fit_rows": _npos,
          "clf_cap_bound": bool(_rc is not None and _rc >= _cb),
          "reg_cap_bound": bool(_rr is not None and _rr >= _cb),
          "es_source": _src, "fit_mode": "single"}
    try:
        m.stage1_ = st
    except Exception:  # noqa: BLE001
        pass
    return m'''

#: 硬化时整段替换的函数(仅文件里已有定义时替换;保留最后一处定义的位置)
_HARD_FUNC_SOURCES = {
    "_es_source": _HARD_ES_SOURCE,
    "_fit_es_on_validation": _HARD_FIT_ES_ON_VALIDATION,
    "_fit_with_es": _HARD_FIT_WITH_ES,
    "_fit_with_eval": _HARD_FIT_WITH_EVAL,
}
#: 被后定义覆盖的同名旧定义一并删除(死代码里是两段式实现与旧口径注释)
_HARD_DEDUP_FUNCS = ("_es_source", "_fit_es_on_validation", "_fit_with_es",
                     "_fit_with_eval", "_fit_seed_variant",
                     "dump_learning_curve", "_fit_hurdle_refit_full")
#: 日志 / metrics / 注释里的旧口径字样(字面替换,不匹配则跳过)
_HARD_TEXT_SUBS = (
    ('f"(early stopping 专用,validation 不参与训练)")',
     'f"(早停面={\'outer validation\' if X_es is X_va else \'train 尾部\'},'
     '单段拟合)")'),
    ("step(f\"final refit done: trees={_es.get('refit_trees') or "
     "_es.get('stage1_best_iteration')} \"",
     "step(f\"final fit done: trees={_es.get('stage1_best_iteration')} \""),
    ("step(f\"final refit done: trees={_es.get('refit_trees')} \"",
     "step(f\"final fit done: trees={_es.get('stage1_best_iteration')} \""),
    ('"two_stage": _st,', '"fit_stats": _st,'),
    ('"refit_full_train": bool(REFIT_FULL_TRAIN),',
     '"full_train_fit": bool(REFIT_FULL_TRAIN),'),
    ('int(_st.get("refit_reg_rows") or 0)', 'int(_st.get("n_reg_fit_rows") or 0)'),
    ("    # P0-4:早停专用 es_valid 从 train 尾部按时间切 ES_FRACTION。\n"
     "    # validation 只评分,不参与任何训练决策——切断\"早停面=裁决面\"的\n"
     "    # 系统性乐观偏差。注意切分在 dtype 治理之后(X_es 与 X_tr 同特征面)。\n"
     "    # uplift 同口径(S-learner 特征面含 __treatment__)。\n",
     "    # [全局库单段] 早停面:默认 outer validation(100% train 只训一次,见\n"
     "    # 下方 [v2.16.0 es_source] 守卫);MODELEVO_ES_SOURCE=train_tail、uplift\n"
     "    # 或拿不到 y_va 时为 train 尾部 ES_FRACTION(头部拟合、尾部早停,同样\n"
     "    # 只训一次)。切分在 dtype 治理之后(X_es 与 X_tr 同特征面)。\n"),
    ("    # early stopping 的 eval_set:P0-4 起固定用 train 尾部切出的 es_valid,\n"
     "    # 不再触碰 validation(裁决面只评分)。uplift 用同特征面的处置侧副本。\n",
     "    # early stopping 的 eval_set = 上方早停面([全局库单段]:默认 outer\n"
     "    # validation,只训一次)。uplift 用同特征面的处置侧副本。\n"),
    ("        # metrics.best_params 继承重训配方),机制与 hpo 重训同(_fit_with_es\n"
     "        # 两段式 + cap 自动外扩)。\n",
     "        # metrics.best_params 继承重训配方),机制与 hpo 重训同\n"
     "        # (_fit_with_eval:[全局库单段],只训一次)。\n"),
    ("    # 内部按同一 ES_FRACTION 从 train 尾部切早停面,validation 只评分)。\n",
     "    # 早停面同上方 [v2.16.0 es_source] 守卫,单段拟合)。\n"),
    ("    早停 es_valid 从 train 尾部 ES_FRACTION 切(与父代同口径);y_va 只用于\n"
     "    外层评分,不进入任何训练决策。",
     "    [全局库单段] 早停面默认 outer validation(100% train 只训一次);\n"
     "    train_tail 或拿不到 y_va 时为 train 尾部 ES_FRACTION。"),
    ("# 早停仍从 train 尾部 ES_FRACTION 切出(outer validation 只评分)。\n",
     "# 早停面见 fit_and_score([全局库单段]:默认 outer validation,只训一次)。\n"),
    ("- 早停专用 es_valid 从 train 尾部按时间切 10%(v12/P0-4):validation\n"
     "  不再承担早停职责——它同时是主指标/delta/noise gate/榜单排名/最终\n"
     "  选型的裁决面,早停在上面会引入与搜索方向相关的系统性乐观偏差\n"
     "  (实测全 run 最大两笔增益恰是早停自由度最大的配置);\n",
     "- [全局库单段] 早停默认在 outer validation 上、100% train 只训一次\n"
     "  (v2.16.0;validation 分数因此略偏乐观,样本外以 shadow/final 为准);\n"
     "  train_tail 口径为 train 尾部 10% 早停,同样只训一次;\n"),
    # hurdle 全量重训臂(gen-08 血缘):常量注释、容器遥测注释、seed 变体
    ("REFIT_FULL_TRAIN = True\nREFIT_ROUND_SCALE = 1.0\n",
     "REFIT_FULL_TRAIN = True\nREFIT_ROUND_SCALE = 1.0\n"
     "# [全局库单段] 入库后第二段已删除:_fit_hurdle_refit_full 只训一次(默认\n"
     "# 100% train + outer validation 早停),REFIT_ROUND_SCALE 不再生效;\n"
     "# REFIT_FULL_TRAIN 只决定 multi-seed 变体轮复用完整 train。\n"),
    ("        # gen-08:第一段(头 90% 拟合 / 尾 10% 早停)遥测;None = 单段拟合\n",
     "        # 拟合遥测(_fit_hurdle_refit_full 写入;None = 直接 _fit_hurdle)\n"),
    ("        # gen-08:两段式下取**第一段** held-out 最优(0-based,与 v2.15\n"
     "        # best_iteration 取第一段的口径一致;第二段无早停、无 best_iteration)。\n",
     "        # [全局库单段] stage1_ 是唯一一段拟合的遥测,取其最优轮(0-based)。\n"),
    ("    gen-08(trial-031):主轮是「头 90% 早停 → 100% train 定轮重训」,变体轮\n"
     "    必须同配方 ⇒ 复用 fit_and_score 登记的全量 train(_FULL_TRAIN,同一对象,\n"
     "    不复制),走同一 _fit_hurdle_refit_full;行数与 X_fit+es 不一致时拒绝。\n",
     "    gen-08(trial-031):变体轮复用 fit_and_score 登记的全量 train(_FULL_TRAIN,\n"
     "    同一对象,不复制),走同一 _fit_hurdle_refit_full([全局库单段]只训一次);\n"
     "    行数与 X_fit(validation 早停)、X_fit+es(train 尾部早停)都对不上时拒绝。\n"),
    # v2.16 守卫下 main 的 X_fit 已是全量 train、X_es 是 validation:旧判据
    # len(_FULL_TRAIN) == len(X_fit)+len(X_es) 恒假,变体轮必抛 → multi-seed 失效
    ("        if _FULL_TRAIN is None or len(_FULL_TRAIN[0]) != len(X_fit) + len(X_es):\n"
     "            raise RuntimeError(\"全量重训变体轮缺少主轮登记的完整 train\")\n",
     "        if _FULL_TRAIN is None or len(_FULL_TRAIN[0]) not in (\n"
     "                len(X_fit), len(X_fit) + len(X_es)):\n"
     "            raise RuntimeError(\"变体轮缺少主轮登记的完整 train\")\n"),
)


def _top_level_defs(code: str) -> dict:
    """顶层函数名 → [(起始行 0-based 含装饰器, 结束行 exclusive), ...](源码序)。"""
    import ast as _ast

    out: dict = {}
    for n in _ast.parse(code).body:
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            start = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
            out.setdefault(n.name, []).append((start, n.end_lineno))
    return out


def _captured_between(code: str, name: str, lo: int, hi: int) -> bool:
    """行区间 [lo, hi) 内的模块级语句是否在导入期引用了 name(别名捕获等)。"""
    import ast as _ast

    for n in _ast.parse(code).body:
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                          _ast.ClassDef)):
            continue
        if not (lo <= n.lineno - 1 < hi):
            continue
        for sub in _ast.walk(n):
            if isinstance(sub, _ast.Name) and sub.id == name \
                    and isinstance(sub.ctx, _ast.Load):
                return True
    return False


def _residual_two_stage(code: str) -> list[str]:
    """硬化后仍带"定轮 → 关早停重训"特征的顶层函数(只报告,不改写)。"""
    import ast as _ast

    hits = []
    for n in _ast.parse(code).body:
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            seg = _ast.get_source_segment(code, n) or ""
            if "early_stopping_rounds=None" in seg \
                    or '["refit_trees"] =' in seg:
                hits.append(n.name)
    return hits


def _residual_kfold_oof(code: str) -> list[str]:
    """硬化后仍有 K 折拟合(GroupKFold / KFold / cross_val_predict)的顶层函数
    (OOF 特征 / 交叉拟合;只报告,不改写)。"""
    import ast as _ast
    import re as _re

    pat = _re.compile(r"\b(?:Group|Stratified|StratifiedGroup)?KFold\s*\(|"
                      r"\bcross_val_predict\s*\(")
    return [n.name for n in _ast.parse(code).body
            if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
            and pat.search(_ast.get_source_segment(code, n) or "")]


# ==================== 全局库配方:DL 5 折 OOF 特征块剥离 ====================
# 用户裁决(2026-10-22):全局库配方去掉 DL 特征的 5 折 OOF。深度 logit 列
# (seqtab_logit / hier_logit)要给 train 行样本外值,只能 K 折训练;只训一次
# 而用样本内 logit 是 stacking 泄漏(样本内 AUC 远低于 leak_scan 的 0.9,
# 拦不住),因此整列删除,配方回到纯 GBM 特征面。
# 适用形态:gen-01 DL 段(段首标记)到 gen-02 槽位配置(段尾标记)整段是 DL
# 实现,段外只有 trial_feature_blocks 的 fns 注册表引用段内函数;块键 ∈
# _DL_OOF_BLOCK_KEYS,每块 1 列。不满足即不改(残留由 _residual_kfold_oof 报告)。
_DL_SEG_BEGIN = "# ================== gen-01 DL 段"
_DL_SEG_STOP = "# ------------------------------------------------------- gen-02 槽位配置"
_DL_OOF_BLOCK_KEYS = ("seqtab", "dl", "p1")


def _bound_names(node) -> list[str]:
    """模块级语句绑定的名字(def / class / 赋值目标 / import 别名)。"""
    import ast as _ast

    if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                         _ast.ClassDef)):
        return [node.name]
    if isinstance(node, (_ast.Assign, _ast.AnnAssign, _ast.AugAssign)):
        tgts = node.targets if isinstance(node, _ast.Assign) else [node.target]
        return [s.id for t in tgts for s in _ast.walk(t)
                if isinstance(s, _ast.Name)]
    if isinstance(node, (_ast.Import, _ast.ImportFrom)):
        return [a.asname or a.name.split(".")[0] for a in node.names]
    return []


def _strip_dl_oof_blocks(code: str) -> tuple[str, list[str]]:
    """删除 DL 5 折 OOF 特征块(整段实现 + 块注册 + 块列数)。返回 (新代码, 说明)。"""
    import ast as _ast
    import re as _re

    lines = code.splitlines(keepends=True)
    b = next((i for i, ln in enumerate(lines)
              if ln.startswith(_DL_SEG_BEGIN)), None)
    e = next((i for i, ln in enumerate(lines)
              if b is not None and i > b and ln.startswith(_DL_SEG_STOP)),
             None)
    if b is None or e is None:
        return code, []
    tree = _ast.parse(code)
    removed: set = set()
    for n in tree.body:
        if b <= n.lineno - 1 < e:
            removed.update(_bound_names(n))
    tfb = None
    for n in tree.body:
        if b <= n.lineno - 1 < e:
            continue
        if not any(isinstance(s, _ast.Name) and s.id in removed
                   for s in _ast.walk(n)):
            continue
        if isinstance(n, _ast.FunctionDef) and n.name == "trial_feature_blocks":
            tfb = n
            continue
        return code, []                  # 段外还有别处引用 DL 实现:不改
    if tfb is None:
        return code, []
    keys: set = set()
    for s in _ast.walk(tfb):
        if isinstance(s, _ast.Dict):
            keys.update(k.value for k, v in zip(s.keys, s.values)
                        if isinstance(k, _ast.Constant)
                        and isinstance(v, _ast.Name) and v.id in removed)
        elif isinstance(s, _ast.Assign) and isinstance(s.value, _ast.Name) \
                and s.value.id in removed:
            keys.update(t.slice.value for t in s.targets
                        if isinstance(t, _ast.Subscript)
                        and isinstance(t.slice, _ast.Constant))
    if not keys or not keys <= set(_DL_OOF_BLOCK_KEYS):
        return code, []
    if "p1" in keys:                     # P1 段一臂一列;多臂则列数推不准
        parts = [n for n in tree.body if "_P1_PARTS" in _bound_names(n)]
        if len(parts) != 1 or not isinstance(parts[0].value, _ast.Tuple) \
                or len(parts[0].value.elts) != 1:
            return code, []
    # 1. trial_feature_blocks:删 fns 注册项
    s0, s1 = tfb.lineno - 1, tfb.end_lineno
    seg = "".join(lines[s0:s1])
    _alt = "|".join(sorted(keys))
    seg = _re.sub(r',\s*"(?:%s)":\s*\w+' % _alt, "", seg)
    seg = _re.sub(r'\{\s*"(?:%s)":\s*\w+,\s*' % _alt, "{", seg)
    seg = _re.sub(r'^[ \t]*fns\["(?:%s)"\]\s*=\s*\w+[ \t]*\n' % _alt, "",
                  seg, flags=_re.M)
    # 2. 删段(段尾标记保留;接缝处空行收成两行)
    if s0 < e:
        return code, []
    head = lines[:b]
    while head and not head[-1].strip():
        head.pop()
    out = "".join(head + ["\n", "\n"] + lines[e:s0] + [seg] + lines[s1:])
    # 3. 槽位常量:_G2_BLOCKS 去键、_G2_N_BLOCK_COLS 扣列、_G2_SLOT 去标签
    tree2 = _ast.parse(out)
    lines2 = out.splitlines(keepends=True)
    edits = []
    for n in tree2.body:
        nm = _bound_names(n)
        if nm == ["_G2_BLOCKS"] and isinstance(n.value, _ast.Tuple):
            kept = [x.value for x in n.value.elts
                    if isinstance(x, _ast.Constant) and x.value not in keys]
            head = "_G2_BLOCKS = ("
            row, rows = head, []
            for k in kept:
                item = f'"{k}", '
                if len(row) + len(item) > 78:
                    rows.append(row.rstrip() + "\n")
                    row = " " * len(head)
                row += item
            rows.append(row.rstrip(", ") + ("," if len(kept) == 1 else "")
                        + ")\n")
            edits.append((n.lineno - 1, n.end_lineno, "".join(rows)))
        elif nm == ["_G2_N_BLOCK_COLS"] and isinstance(n.value, _ast.Constant):
            edits.append((n.lineno - 1, n.end_lineno,
                          f"_G2_N_BLOCK_COLS = {int(n.value.value) - len(keys)}"
                          f"  # {_ES_HARD_MARK} 已扣除 DL 块 {sorted(keys)} 各 1 列\n"))
        elif nm == ["_G2_SLOT"] and isinstance(n.value, _ast.Constant):
            slot = _re.sub(r"\+(?:seqtab|hier|daymat|dl)\b", "",
                           str(n.value.value))
            edits.append((n.lineno - 1, n.end_lineno,
                          f"# {_ES_HARD_MARK} DL 5 折 OOF 特征块(seqtab_logit 等深度 "
                          f"logit 列)已删除:只训一次给不出 train 行的样本外值\n"
                          f"_G2_SLOT = \"{slot}\"\n"))
    for s, t, txt in sorted(edits, reverse=True):
        lines2[s:t] = [txt]
    out = "".join(lines2)
    # 4. 自检:段内名字不得再被引用
    left = {s.id for s in _ast.walk(_ast.parse(out))
            if isinstance(s, _ast.Name) and s.id in removed}
    if left:
        return code, []
    return out, [f"删除 DL 5 折 OOF 特征块 {sorted(keys)}(段内 {len(removed)} 个"
                 f"顶层名字)"]


def harden_library_es_code(code: str) -> tuple[str, list[str]]:
    """全局库配方单段硬化:任何口径都只训一次,删除两段式实现与回退。

    先跑 `_upgrade_es_source_code`(v2.16 前的旧代码补齐单段段 / 守卫 /
    hurdle 段),再:
      1. hurdle 段换成硬化版(去掉 `_fit_hurdle_refit_full_v215` 别名与
         两段式回退;train_tail / 无 validation 时 train 尾部单段早停);
      2. `_es_source / _fit_es_on_validation / _fit_with_es / _fit_with_eval`
         的最后一处定义换成硬化版(`_fit_with_es` 只剩 train 尾部单段,
         `_fit_es_on_validation` 异常不再回退两段式);
      3. 这些函数及 `_fit_seed_variant / dump_learning_curve /
         _fit_hurdle_refit_full` 被后定义覆盖的旧定义删除(导入期被别名
         捕获的跳过);
      4. 日志 / metrics / 注释里的旧口径字样按 `_HARD_TEXT_SUBS` 字面替换;
      5. DL 5 折 OOF 特征块整列删除(`_strip_dl_oof_blocks`;形态不符不改,
         残留 K 折拟合由 `_residual_kfold_oof` 报告)。
    幂等;产物必须可编译。返回 (新代码, 说明)。
    """
    import re as _re

    out, notes = _upgrade_es_source_code(code)
    # 1. hurdle 段
    _hb = _re.compile(_re.escape(_ES_HURDLE_BEGIN) + r"[^\n]*\n.*?"
                      + _re.escape(_ES_HURDLE_END) + r"\n?", _re.S)
    _hard_hb = (f"{_ES_HURDLE_BEGIN}(harness 自动注入;{_ES_HARD_MARK} hurdle "
                f"只训一次,无全量重训)====\n" + _ES_HURDLE_HARD_BLOCK
                + f"\n{_ES_HURDLE_END}\n")
    if _hb.search(out) and _hb.search(out).group(0) != _hard_hb:
        out = _hb.sub(lambda _m: _hard_hb, out, count=1)
        notes.append("hurdle 段改为硬化版(删除两段式别名与回退)")
    # 2/3. 函数替换与旧定义删除:按行区间自下而上改写
    defs = _top_level_defs(out)
    lines = out.splitlines(keepends=True)
    edits: list[tuple[int, int, str]] = []
    n_rep = n_dropped = 0
    for name in _HARD_DEDUP_FUNCS:
        spans = defs.get(name) or []
        if not spans:
            continue
        s, e = spans[-1]
        new_src = _HARD_FUNC_SOURCES.get(name)
        if new_src is not None and "".join(lines[s:e]) != new_src + "\n":
            edits.append((s, e, new_src + "\n"))
            n_rep += 1
        for i, (s, e) in enumerate(spans[:-1]):
            if _captured_between(out, name, e, spans[i + 1][0]):
                continue
            while e < len(lines) and not lines[e].strip():
                e += 1          # 连同其后空行,避免留下大段空白
            edits.append((s, e, ""))
            n_dropped += 1
    for s, e, text in sorted(edits, key=lambda t: t[0], reverse=True):
        lines[s:e] = [text] if text else []
    out = "".join(lines)
    if n_rep:
        notes.append(f"单段硬化替换函数 ×{n_rep}")
    if n_dropped:
        notes.append(f"删除被覆盖的旧定义 ×{n_dropped}")
    # 4. 旧口径字样(new 以 old 为前缀的追加式替换按 new 是否已在判幂等)
    n_sub = 0
    for old, new in _HARD_TEXT_SUBS:
        if old in out and not (old in new and new in out):
            n_sub += out.count(old)
            out = out.replace(old, new)
    if n_sub:
        notes.append(f"日志/metrics/注释旧口径字样替换 ×{n_sub}")
    # 5. DL 5 折 OOF 特征块
    out, dl_notes = _strip_dl_oof_blocks(out)
    notes.extend(dl_notes)
    if notes:
        compile(out, "<harden_library_es_code>", "exec")
    return out, notes


def _upgrade_leak_scan_code(code: str) -> tuple[str, list[str]]:
    """[v2.16.2] 子代副本的 leak_scan 升级为按任务类型分派的单源实现(D-004)。

    旧模板 leak_scan 对回归目标算单变量 AUC,必抛 "multi_class must be in
    ('ovo','ovr')"(run4 regression baseline 起全部 trial `n_scanned=0`),
    泄漏哨兵从未运行。父代冻结代码不动,只改 --from-parent / warm-start /
    固化 baseline 复用的**副本**:模块级(最后一个)`def leak_scan` 仍是旧版
    (含 roc_auc_score、无回归分支)时整函数替换为
    `runtime.templates.LEAK_SCAN_SOURCE`。幂等;分类口径逐字不变。
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, []
    defs = [n for n in tree.body
            if isinstance(n, _ast.FunctionDef) and n.name == "leak_scan"]
    if not defs:
        return code, []
    node = defs[-1]
    lines = code.splitlines(keepends=True)
    start = node.lineno - 1 - len(node.decorator_list)
    end = node.end_lineno
    seg = "".join(lines[start:end])
    if "abs_spearman" in seg or "roc_auc_score" not in seg:
        return code, []
    from runtime.templates import LEAK_SCAN_SOURCE

    out = "".join(lines[:start]) + LEAK_SCAN_SOURCE + "\n" + "".join(lines[end:])
    return out, ["leak_scan 升级为按任务类型分派(回归 |Spearman|>0.95)"]


#: v2.16.3(D-011):旧 baseline 模板的撞 cap 判据行(字面量,逐字匹配)
_OLD_CAP_BOUND_LINE = \
    "            _bound = int(_bi) >= 0.95 * int(N_ESTIMATORS_CAP)\n"
_NEW_CAP_BOUND_BLOCK = (
    "            # v2.16.3 撞 cap 单一语义(runtime.hpo.es_cap_telemetry:实训轮数\n"
    "            # ≥ cap;无轮数时 best + 早停耐心 ≥ cap;复合容器任一段撞即 True)\n"
    "            try:\n"
    "                from runtime.hpo import es_cap_telemetry as _ect_v2163\n"
    "                _bound = bool(_ect_v2163(\n"
    "                    model,\n"
    "                    cap=int(N_ESTIMATORS_CAP),\n"
    "                    es_rounds=int(globals().get('EARLY_STOP_ROUNDS') or 0),\n"
    "                    best_iteration=_bi,\n"
    "                    rounds=metrics.get('n_estimators_used'))"
    ".get('cap_bound'))\n"
    "            except Exception:  # noqa: BLE001\n"
    "                _r_v2163 = metrics.get('n_estimators_used')\n"
    "                _bound = (int(_r_v2163) >= int(N_ESTIMATORS_CAP)) "
    "if _r_v2163 else (\n"
    "                    int(_bi) + int(globals().get('EARLY_STOP_ROUNDS') or 0)\n"
    "                    >= int(N_ESTIMATORS_CAP))\n")


def _upgrade_cap_bound_code(code: str) -> tuple[str, list[str]]:
    """[v2.16.3] 子代副本里旧 baseline 模板的撞 cap 判据升级为单一语义(D-011)。

    旧行 ``_bound = int(_bi) >= 0.95 * int(N_ESTIMATORS_CAP)`` 与 hurdle /
    模板单段遥测(实训轮数 ≥ cap)对同一 trial 判定相反(run4 regression
    trial-022:best 7923 ≥ 0.95×8000 判撞 cap,而 clf 实训 7973 < 8000、早停
    真实触发)。只做逐字替换(父代冻结代码不动,只改 --from-parent /
    warm-start / 固化 baseline 复用的副本),``source`` 字面量同步改名。幂等。
    """
    if _OLD_CAP_BOUND_LINE not in code:
        return code, []
    out = code.replace(_OLD_CAP_BOUND_LINE, _NEW_CAP_BOUND_BLOCK)
    out = out.replace('"source": "baseline_best_iteration_ratio"',
                      '"source": "unified_cap_bound"')
    return out, ["es_info.cap_bound 升级为单一语义(实训轮数 ≥ cap)"]


#: v2.17.2(A1):旧版 _fit_es_on_validation 的撞 cap 判定行(缩进不限)
_CAP_BOUND_ASSIGN_RE = (r"^(?P<ind>[ \t]+)_bound = bool\(_cap and "
                        r"len\(_curve\) >= int\(_cap\)\)[ \t]*$")
#: v2.17.0 官方 hurdle 段 es_info 字面量(旧版无 cap_bound_unresolved)
_OLD_HURDLE_ES_TAIL = '                    "source": "hurdle_segments"},\n'
_NEW_HURDLE_ES_TAIL = (
    '                    "source": "hurdle_segments",\n'
    '                    # v2.17.2(A1):撞 cap 且 cap 已达树数上限 = 截断未解除\n'
    '                    "cap_bound_unresolved": bool(_bound and _cap >= int(\n'
    '                        globals().get("_MAX_TREES") or 8000))},\n')


def _upgrade_cap_unresolved_code(code: str) -> tuple[str, list[str]]:
    """[v2.17.2] 子代副本补 ``es_info.cap_bound_unresolved``(A1,评估侧契约)。

    run4 trial-017:lr-shrink 重训停在 7999/8000 且 cap 已达树数上限,无从
    外扩,es_info 只有 cap_bound=true,评估侧把 −0.00095 当方向证伪。现行
    模板 / 全局库硬化版 ``_fit_es_on_validation`` 在撞 cap 且 cap ≥ _MAX_TREES
    时写 ``cap_bound_unresolved=True``;--from-parent / warm-start / 固化
    baseline 复用的**副本**里旧版 helper 没有该键,这里就地补齐(父代冻结
    代码不动):
      1. 每个模块级 ``_fit_es_on_validation``(含 v2.16 升级段与嫁接段两份)
         在 ``_bound = bool(_cap and len(_curve) >= int(_cap))`` 之后插入同口径
         赋值(要求其前已有 ``info = {``;不符即不动);
      2. v2.17.0 官方 hurdle 段的 es_info 字面量追加同名键。
    幂等:已含该键的函数 / 字面量不动。
    """
    import ast as _ast
    import re as _re

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, []
    lines = code.splitlines(keepends=True)
    inserts: list[tuple[int, str]] = []
    for n in tree.body:
        if not (isinstance(n, _ast.FunctionDef)
                and n.name == "_fit_es_on_validation"):
            continue
        if _re.search(r"info\[\"cap_bound_unresolved\"\]|"
                      r"\"cap_bound_unresolved\"\s*:",
                      "".join(lines[n.lineno - 1:n.end_lineno])):
            continue
        for i in range(n.lineno - 1, n.end_lineno):
            m = _re.match(_CAP_BOUND_ASSIGN_RE, lines[i].rstrip("\r\n"))
            if not m:
                continue
            if "info = {" not in "".join(lines[n.lineno - 1:i]):
                break
            ind = m.group("ind")
            inserts.append((i + 1, (
                f"{ind}# v2.17.2(A1):撞 cap 且 cap 已达 _MAX_TREES(无外扩余量)"
                f"= 截断未解除\n"
                f"{ind}info[\"cap_bound_unresolved\"] = bool(_bound and "
                f"int(_cap or 0) >= int(\n"
                f"{ind}    globals().get(\"_MAX_TREES\") or 8000))\n")))
            break
    for idx, text in sorted(inserts, reverse=True):
        lines[idx:idx] = [text]
    out = "".join(lines)
    notes: list[str] = []
    if inserts:
        notes.append(f"_fit_es_on_validation 补 es_info.cap_bound_unresolved "
                     f"×{len(inserts)}")
    if _OLD_HURDLE_ES_TAIL in out:
        out = out.replace(_OLD_HURDLE_ES_TAIL, _NEW_HURDLE_ES_TAIL)
        notes.append("官方 hurdle 段 es_info 补 cap_bound_unresolved")
    return out, notes


#: v2.17.2(A5):n_features 口径升级段标记(幂等判据)
_NF_UPGRADE_MARK = "# v2.17.2(A5):n_features 以拟合期模型列数为准"


#: v2.17.3:搜索代跳过 shadow 打分的开关标记(幂等判据)
_SHADOW_SKIP_MARK = "MODELEVO_SHADOW_SCORE"
#: 旧副本的 shadow 段入口(两种血缘:baseline 手写 main / 模板 body)
#: 旧副本 shadow 段入口:缩进无关(4~8 空格都接受),两种血缘各一条
import re as _re_shadow
_SHADOW_OLD_RE = _re_shadow.compile(
    r'^(?P<ind>[ \t]*)try:\n'
    r'(?P<ind2>[ \t]*)_sh_path = SPLITS / "shadow_test\.parquet"\n'
    r'(?P<ind3>[ \t]*)if _sh_path\.exists\((?P<cond>[^\n]*):\n',
    _re_shadow.MULTILINE)


#: v2.17.3:早停降采样接入标记(幂等判据)
_ES_EVAL_CAP_MARK = "_es_eval_cap"


def _upgrade_es_eval_cap_code(code: str) -> tuple[str, list[str]]:
    """[v2.17.3] 子代副本的 ``_es_stage1`` 接入早停集降采样。

    为什么:每轮评估都在整个早停集(本任务 44.7 万行 × 2000+ 列)上打分,
    是单 trial wall 的可观部分;降采样只改变"在哪一行停",落盘分数仍全量
    口径(2.55M/450k 实测 best_iter 804 vs 818、全量 AUC 一致到 6 位)。

    已带 ``_es_eval_cap`` 的现行代码不动;旧副本(``--from-parent`` /
    warm-start / 固化 baseline 复制来的)就地补:
      1. 缺失时在 ``if __name__`` 之前插入模板原文的 ``_es_eval_cap``;
      2. 每个模块级 ``_es_stage1`` 的函数体开头插入两行调用(缺失才插)。
    幂等:函数体已引用 ``_es_eval_cap`` 的不再插。
    """
    import ast as _ast
    import re as _re

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, []
    names = {n.name for n in tree.body if isinstance(n, _ast.FunctionDef)}
    notes: list[str] = []
    out = code
    if "_es_stage1" not in names:
        return code, []
    lines = out.splitlines(keepends=True)
    inserts: list[tuple[int, str]] = []
    for n in tree.body:
        if not (isinstance(n, _ast.FunctionDef) and n.name == "_es_stage1"):
            continue
        body = "".join(lines[n.lineno - 1:n.end_lineno])
        if _ES_EVAL_CAP_MARK in body:
            continue
        _doc_end = n.lineno
        if (n.body and isinstance(n.body[0], _ast.Expr)
                and isinstance(getattr(n.body[0], "value", None), _ast.Constant)
                and isinstance(n.body[0].value.value, str)):
            _doc_end = n.body[0].end_lineno
        ind = "    "
        inserts.append((_doc_end, (
            f"{ind}# v2.17.3:早停集降采样(只影响\"何时停\",评分仍全量)\n"
            f"{ind}_cap_fn = globals().get(\"_es_eval_cap\")\n"
            f"{ind}if _cap_fn is not None:\n"
            f"{ind}    X_es, y_es = _cap_fn(X_es, y_es)\n")))
    if not inserts:
        return code, []
    for idx, text in sorted(inserts, reverse=True):
        lines[idx:idx] = [text]
    out = "".join(lines)
    notes.append(f"_es_stage1 接入早停集降采样 ×{len(inserts)}")

    if "_es_eval_cap" not in names:
        src = _template_func_sources().get("_es_eval_cap")
        if src:
            out = _insert_before_main(
                out, "# ==== v2.17.3 早停集降采样 helper ====\n" + src
                + "\n# ==== v2.17.3 早停集降采样 helper END ====\n")
            notes.append("补 _es_eval_cap helper")
    return out, notes


def _upgrade_shadow_skip_code(code: str) -> tuple[str, list[str]]:
    """[v2.17.3] 子代副本的 shadow 段接入 MODELEVO_SHADOW_SCORE 开关。

    为什么:shadow 分数只在 fresh confirmation 里被消费(runner 的
    shadow_policy=confirmation_only),普通搜索 trial 的 shadow 工件从不
    进反馈环,却要在 44.7 万行上重放 transform + 特征块 + 全量打分 ——
    req-t7 run4 实测冷缓存世代 557~686 s/trial(热缓存 ~55 s),占单 trial
    wall 的 2%~12%。runner 按 search_action 下发 skip,confirm 仍产出。

    只改**副本**(--from-parent / warm-start / 固化 baseline);父代冻结代码
    不动。缩进无关(生成的 main 体是 4 空格、模板 body 是 4/8 空格两种)。
    幂等:已含 MODELEVO_SHADOW_SCORE 的代码直接返回。
    """
    if _SHADOW_SKIP_MARK in code:
        return code, []

    def _sub(m) -> str:
        ind, ind2, ind3 = m.group("ind"), m.group("ind2"), m.group("ind3")
        # cond 是被截断的尾部(如 " and _last_model is not None"):去掉
        # 首尾的 `and`/`)`,左括号尾统一补一个 " and "
        cond = (m.group("cond") or "").strip()
        while cond.startswith(")"):
            cond = cond[1:].strip()
        for _pre in ("and ", "and", "&"):
            if cond.startswith(_pre):
                cond = cond[len(_pre):].strip()
                break
        cond = cond.rstrip(")").strip()
        extra = (" and " + cond) if cond else ""
        return (f"{ind}try:\n"
                f"{ind2}_sh_path = SPLITS / \"shadow_test.parquet\"\n"
                f"{ind3}if (_sh_path.exists(){extra}\n"
                f"{ind3}        and os.environ.get("
                f"\"MODELEVO_SHADOW_SCORE\", \"1\") != \"skip\"):\n")

    out, n = _SHADOW_OLD_RE.subn(_sub, code, count=1)
    if not n:
        return code, []
    return out, ["shadow 段接入 MODELEVO_SHADOW_SCORE 开关"]


def _upgrade_n_features_code(code: str) -> tuple[str, list[str]]:
    """[v2.17.2] 旧 main 的 ``metrics.n_features`` 改以拟合期模型列数为准(A5)。

    新模板 fit_and_score 写 ``n_features = X_tr.shape[1]``(模型真实列数),但
    baseline 血缘 / v2.13.5 改线的旧 main 先写 ``metrics = {"n_features":
    len(feats), …}``(基础面列数,run4 trial-017 记 700,模型实为 1640 列),
    改线段只 ``setdefault`` 并入 fit_and_score 摘要 → 基础列数胜出,planner
    据此误判"同面"。这里在 main 的 metrics.json 落盘语句之前插入一段:
    n_features = fit_and_score 摘要的 n_features(无则 main 里的
    ``X_tr.shape[1]``),原值另存 ``n_features_base``。
    只处理"main 自己写 n_features 且落盘变量是字面量 dict"的形态;幂等。
    """
    import ast as _ast

    if _NF_UPGRADE_MARK in code:
        return code, []
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, []
    mains = [n for n in tree.body
             if isinstance(n, _ast.FunctionDef) and n.name == "main"]
    if not mains:
        return code, []
    main = mains[-1]
    best = None
    for st in _ast.walk(main):
        if not isinstance(st, (_ast.With, _ast.Expr)):
            continue
        seg = _ast.get_source_segment(code, st) or ""
        if "metrics.json" not in seg:
            continue
        var = None
        for c in _ast.walk(st):
            if not isinstance(c, _ast.Call):
                continue
            f = c.func
            fname = f.attr if isinstance(f, _ast.Attribute) else (
                f.id if isinstance(f, _ast.Name) else "")
            if fname == "dump" and c.args \
                    and isinstance(c.args[0], _ast.Name):
                var = c.args[0].id
            elif fname == "dump_json" and len(c.args) >= 2 \
                    and isinstance(c.args[1], _ast.Name):
                var = c.args[1].id
        if var and (best is None or st.lineno < best[0].lineno):
            best = (st, var)
    if best is None:
        return code, []
    st, var = best
    owns_nf = any(
        isinstance(a, _ast.Assign) and any(
            isinstance(t, _ast.Name) and t.id == var for t in a.targets)
        and isinstance(a.value, _ast.Dict) and any(
            isinstance(k, _ast.Constant) and k.value == "n_features"
            for k in a.value.keys)
        for a in _ast.walk(main))
    if not owns_nf:
        return code, []
    ind = " " * st.col_offset
    block = [
        f"{ind}{_NF_UPGRADE_MARK}(fit_and_score 的\n",
        f"{ind}# X_tr.shape[1],缺失时取本函数的 X_tr);main 记的基础列数另存 "
        "n_features_base\n",
        f"{ind}_nf_fs_v2172 = locals().get(\"_fs_metrics\")\n",
        f"{ind}_nf_fit_v2172 = (_nf_fs_v2172.get(\"n_features\")\n",
        f"{ind}                 if isinstance(_nf_fs_v2172, dict) else None)\n",
        f"{ind}if _nf_fit_v2172 is None and hasattr(locals().get(\"X_tr\"), "
        "\"shape\"):\n",
        f"{ind}    _nf_fit_v2172 = locals()[\"X_tr\"].shape[1]\n",
        f"{ind}if _nf_fit_v2172 is not None:\n",
        f"{ind}    {var}.setdefault(\"n_features_base\", "
        f"{var}.get(\"n_features\"))\n",
        f"{ind}    {var}[\"n_features\"] = int(_nf_fit_v2172)\n",
    ]
    lines = code.splitlines(keepends=True)
    lines[st.lineno - 1:st.lineno - 1] = block
    out = "".join(lines)
    try:
        _ast.parse(out)
    except SyntaxError:
        return code, []
    return out, [f"main 的 {var}.n_features 改为拟合期模型列数"
                 "(基础列数另存 n_features_base)"]


def _apply_es_source_upgrade(train_py: Path, harden: bool = False) -> list[str]:
    """对子代副本 train.py 就地应用 `_upgrade_es_source_code`,返回说明。

    harden=True(副本取自全局库:warm-start / 固化 baseline 复用)时改用
    `harden_library_es_code`,与库内配方同为单段口径。
    v2.16.2:同一入口顺带把旧版 leak_scan 升级为按任务类型分派
    (`_upgrade_leak_scan_code`,D-004)。
    """
    try:
        text = train_py.read_text(encoding="utf-8")
    except OSError:
        return []
    new_text, notes = (harden_library_es_code(text) if harden
                       else _upgrade_es_source_code(text))
    new_text, _ls_notes = _upgrade_leak_scan_code(new_text)
    # v2.17.0(#7):回归任务的副本(--from-parent / warm-start / 固化 baseline)
    # 折叠分类/uplift 打分分派为单分支(MODEL_TYPE 取自代码常量;分类不变)
    new_text, _nr = _narrow_dispatch_text(new_text)
    _nr_notes = [f"v2.17.0 回归单分支渲染 {len(_nr)} 处"] if _nr else []
    # v2.16.3 批次(以 2.17.1 发布):旧 baseline 撞 cap 判据 → 单一语义(D-011)
    new_text, _cb_notes = _upgrade_cap_bound_code(new_text)
    # v2.17.2:es_info.cap_bound_unresolved(A1)与 n_features 模型列口径(A5)
    new_text, _cu_notes = _upgrade_cap_unresolved_code(new_text)
    new_text, _nf_notes = _upgrade_n_features_code(new_text)
    # v2.17.3:搜索代跳过 shadow 打分(shadow 只服务 fresh confirm)
    new_text, _sk_notes = _upgrade_shadow_skip_code(new_text)
    # v2.17.3:早停集降采样(只影响"何时停")
    new_text, _ec_notes = _upgrade_es_eval_cap_code(new_text)
    _v2172_notes = _cu_notes + _nf_notes
    if (notes or _ls_notes or _nr_notes or _cb_notes
            or _v2172_notes or _sk_notes or _ec_notes):
        compile(new_text, str(train_py), "exec")  # 升级产物必须可编译
        train_py.write_text(new_text, encoding="utf-8")
        if notes:
            print(f"[info] v2.16.0 早停口径升级: {notes}")
        if _ls_notes:
            print(f"[info] v2.16.2 泄漏哨兵升级: {_ls_notes}")
        if _nr_notes:
            print(f"[info] {_nr_notes[0]}(分类/uplift 打分分派对回归不可达,"
                  "lint 规则 4 无需 adhoc 收窄)")
        if _cb_notes:
            print(f"[info] v2.16.3 撞 cap 判据升级: {_cb_notes}")
        if _v2172_notes:
            print(f"[info] v2.17.2 遥测口径升级: {_v2172_notes}")
        if _sk_notes:
            print(f"[info] v2.17.3 搜索代跳过 shadow: {_sk_notes}")
        if _ec_notes:
            print(f"[info] v2.17.3 早停集降采样: {_ec_notes}")
    return (notes + _ls_notes + _nr_notes + _cb_notes + _v2172_notes
            + _sk_notes + _ec_notes)


# ==================== v2.16.2:hurdle 父代的超参段(D-008) ====================
# run4 regression(loan_amt_t7)gen-04:`new-trial --action hpo --from-parent
# <hurdle 冠军 trial-008>` 的嫁接段在文件尾重定义模板版 fit_and_score,覆盖父代
# hurdle 版;inner 搜索与 outer 重训都退化成单个 XGBRegressor(HURDLE_STRUCT /
# HURDLE_LR 不被消费),创建期只打一行 WARNING,lint 只报规则 4。修复:hurdle
# 父代不走模板嫁接,改注入 `templates.HURDLE_HP_SEGMENT`(包住父代入口而不是
# 覆盖它;inner/重训都构造父代 hurdle)。非 hurdle 父代嫁接后再过一道
# "生效 fit_and_score 仍到达父代 structural 入口"的检查(lint 规则 23 同源)。

#: hurdle 超参段依赖的父代训练入口(创建期校验;缺任一项即拒绝并给替代路径)
_HURDLE_ENTRY_FUNCS = ("_fit_hurdle", "_hurdle_estimators", "fit_and_score")


def _lint_module():
    """加载 skill 的 assets/lint_trial.py(规则函数单源,创建期与 run 前同口径)。"""
    import importlib.util as _ilu

    p = _SKILL_ROOT / "assets" / "lint_trial.py"
    spec = _ilu.spec_from_file_location("_mvo_lint_trial", p)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hurdle_lineage_status(code: str) -> dict:
    """父代码是否 hurdle 拓扑、hurdle 超参段的契约入口是否齐全。

    判据:runtime.topology 的 structural:hurdle 信号(两个子模型预测相乘),或
    模块级定义了 `_fit_hurdle` / `Hurdle*` 类(静态分析失败时的兜底)。
    """
    import ast as _ast
    import re as _re

    st = {"hurdle": False, "entries_ok": False, "missing": [],
          "main_calls_fit_and_score": False, "source": None}
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return st
    names = {n.name for n in tree.body
             if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                               _ast.ClassDef))}
    try:
        from runtime.topology import analyze_code

        if "hurdle" in (analyze_code(code).get("structural_forms") or []):
            st.update(hurdle=True, source="topology:structural:hurdle")
    except Exception:  # noqa: BLE001 静态分析失败走命名兜底
        pass
    if not st["hurdle"] and ("_fit_hurdle" in names or any(
            _re.match(r"Hurdle\w*$", n) for n in names)):
        st.update(hurdle=True, source="names")
    st["missing"] = [n for n in _HURDLE_ENTRY_FUNCS if n not in names]
    st["main_calls_fit_and_score"] = _main_calls_name(tree, "fit_and_score")
    st["entries_ok"] = bool(st["hurdle"] and not st["missing"]
                            and st["main_calls_fit_and_score"])
    return st


def _hurdle_reject_message(trial_id: str, st: dict, what: str) -> str:
    gaps = list(st.get("missing") or [])
    if not st.get("main_calls_fit_and_score"):
        gaps.append("main() 调用 fit_and_score")
    return (
        f"{what}:父代 {trial_id} 是 hurdle 拓扑(structural:hurdle),但缺 hurdle "
        f"超参段的契约入口 {gaps}(需要 `_hurdle_estimators(n_jobs, n_rows_clf, "
        "n_rows_reg) -> (clf, reg)`、`_fit_hurdle(n_jobs, X_fit, y_fit, X_es, "
        "y_es)` 与 main 调用的 `fit_and_score`)。不嫁接模板 hpo 段 —— 那会把 "
        "hurdle 静默换成单个回归器,Δ_parent 混入拓扑变化(D-008)。替代路径:"
        "(a) `--allow-code-edit` 在子代副本补齐上述入口后重建;(b) 结构参数子代 "
        "`--from-parent <id> --set clf__max_depth=8` 等(同样需要上述入口,"
        "harness 自动注入消费点);(c) 确要换成单模型时用模板渲染路径 "
        "`--action hpo --algorithm xgboost --hpo-parent <id>` 并在 hypothesis "
        "声明拓扑变化")


def _inject_hurdle_hp_segment(train_py: Path) -> list[str]:
    """在子代副本文件尾(main 守卫前)注入 hurdle 超参段;已有旧段先整段替换(幂等)。

    父代冻结代码不动。注入后编译自检,失败抛 SystemExit(调用方回滚)。
    """
    from .templates import HURDLE_HP_BEGIN, HURDLE_HP_END, HURDLE_HP_SEGMENT

    code = Path(train_py).read_text(encoding="utf-8")
    notes: list[str] = []
    if HURDLE_HP_BEGIN in code:
        i = code.find(HURDLE_HP_BEGIN)
        i = code.rfind("\n", 0, i) + 1
        j = code.find(HURDLE_HP_END, i)
        if j < 0:
            raise SystemExit("hurdle 超参段有头无尾(代码被手改?),请人工检查")
        j = code.find("\n", j)
        j = len(code) if j < 0 else j + 1
        code = code[:i] + code[j:]
        notes.append("替换旧 hurdle 超参段")
    block = (f"{HURDLE_HP_BEGIN}(harness 自动注入;包住父代 hurdle 训练入口:"
             "HPO / config 超参覆写 / 原样委托)====\n"
             + HURDLE_HP_SEGMENT + f"\n{HURDLE_HP_END}\n")
    out = _insert_before_main(code, block)
    try:
        compile(out, str(train_py), "exec")
    except SyntaxError as exc:
        raise SystemExit(f"hurdle 超参段注入后语法错误(harness 缺陷,请上报):{exc}")
    Path(train_py).write_text(out, encoding="utf-8")
    notes.append("注入 v2.16.2 hurdle 超参段(HPO/超参覆写构造父代 hurdle)")
    return notes


def _import_probe(trial_dir: Path) -> str | None:
    """cwd=code/ 下做一次模块级编译 + 导入(不训练);ImportError/NameError/
    SyntaxError 返回失败行,其余异常不拦(与嫁接探针同口径)。"""
    import subprocess as _sp

    _probe_src = "import ast,importlib.util,pathlib,sys\np=pathlib.Path('train.py')\nsrc=p.read_text(encoding='utf-8')\nast.parse(src)\nspec=importlib.util.spec_from_file_location('_graft_probe', p)\nmod=importlib.util.module_from_spec(spec)\ntry:\n    spec.loader.exec_module(mod)\nexcept (ImportError, NameError) as e:\n    print('PROBE_FAIL %s: %s' % (type(e).__name__, e))\n    sys.exit(4)\nexcept Exception:\n    pass\nprint('PROBE_OK')\n"
    _probe = _sp.run([sys.executable, "-c", _probe_src],
                     cwd=str(Path(trial_dir) / "code"),
                     capture_output=True, text=True, timeout=120)
    if _probe.returncode in (0,) and "PROBE_OK" in _probe.stdout:
        return None
    if _probe.returncode == 4:
        return (_probe.stdout.strip().splitlines()
                or ["PROBE_FAIL(无输出)"])[-1]
    return ((_probe.stderr or _probe.stdout).strip().splitlines()
            or [f"PROBE_FAIL rc={_probe.returncode}"])[-1]


def _hurdle_parent_hpo_best(run_dir: Path, parent_id: str | None) -> dict | None:
    """父代本身是 hurdle HPO trial 时,其重训配方(hurdle 键口径)作额外基准点。"""
    if not parent_id:
        return None
    pdir = Path(run_dir) / "trials" / str(parent_id)
    st = load_json(pdir / "hpo-study.json") if (pdir / "hpo-study.json"
                                                ).exists() else {}
    if str((st or {}).get("topology") or "") != "structural:hurdle":
        return None
    bp = (st or {}).get("best_params")
    if isinstance(bp, dict) and bp:
        return {k: v for k, v in bp.items()
                if isinstance(v, (int, float, str, bool))}
    return None


def _hurdle_lr_cap_advice(run_dir: Path, trial, code: str,
                          hpo_cfg: dict) -> str | None:
    """lr 收缩重训的 cap 是否会被树数上限钳住(重训撞 cap 风险)的创建期提示。"""
    import math as _m
    import re as _re

    lr_mode = str(hpo_cfg.get("lr_mode") or "search").lower()
    space = hpo_cfg.get("search_space") or {}
    # v2.16.3:显式 shrink 不再因"空间无 lr 维"回落 search(plan_lr_mode),
    # 重训 lr 恒收缩 → 无论空间有无 lr 维都要检查;auto 仍需 lr 维(含分段前缀)
    from runtime.hpo import param_base_name as _pbn
    _has_lr = any(_pbn(k) in ("learning_rate", "eta", "hurdle_lr")
                  for k in space)
    if lr_mode == "auto" and not _has_lr or lr_mode not in ("shrink", "auto"):
        return None
    m = _re.search(r"(?m)^HURDLE_LR\s*=\s*([0-9.eE+-]+)", code)
    if not m:
        return None
    parent_lr = float(m.group(1))
    search_lr = float(hpo_cfg.get("search_lr") or 0.05)
    factor = min(max(float(hpo_cfg.get("lr_shrink_factor") or 0.25), 0.05), 1.0)
    refit_lr = search_lr * factor
    pids = list((trial.data or {}).get("parent_trial_ids") or [])
    pm = load_json(Path(run_dir) / "trials" / pids[0] / "metrics.json") \
        if pids and (Path(run_dir) / "trials" / pids[0] / "metrics.json"
                     ).exists() else {}
    h = (pm or {}).get("hurdle") or {}
    best = max(int(h.get("clf_best_iteration") or 0),
               int(h.get("reg_best_iteration") or 0))
    if not best or refit_lr <= 0:
        return None
    need = int(_m.ceil(best * parent_lr / refit_lr))
    if need <= 8000:
        return None
    ok_factor = round(parent_lr / search_lr, 4)
    return (f"lr 收缩重训 lr={refit_lr:g}(search_lr {search_lr:g} × factor "
            f"{factor:g})低于父代 HURDLE_LR={parent_lr:g};按父代最优轮数 {best} "
            f"外推需要约 {need} 棵树,超过树数上限 8000 → 重训大概率撞 cap。"
            f"建议 --hpo-lr-shrink-factor {ok_factor:g}(重训 lr 回到父代 "
            f"{parent_lr:g}),或 --hpo-lr-mode search 并收窄 learning_rate 维")


def _baseline_hp_algos() -> tuple:
    from .templates import BASELINE_HP_ALGOS

    return tuple(BASELINE_HP_ALGOS)


def _baseline_lineage_status(code: str) -> dict:
    """[v2.17.0 D2] 父代码是否 baseline 模板血缘、能否注入 --set 消费段。

    判据(静态):模块级 ``def build_estimator(n_jobs, n_rows...)`` + 模块级常量
    ``_ALGORITHM = '<algo>'``(runtime/baseline.py BASELINE_TEMPLATE 的结构)。
    可注入 = 算法在 BASELINE_HP_ALGOS(估计器支持 set_params)。
    """
    import ast as _ast

    st = {"baseline": False, "injectable": False, "algorithm": None}
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return st
    has_build = False
    for n in tree.body:
        if isinstance(n, _ast.FunctionDef) and n.name == "build_estimator":
            args = [a.arg for a in n.args.args]
            has_build = bool(args) and args[0] == "n_jobs"
        elif isinstance(n, _ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], _ast.Name) \
                and n.targets[0].id == "_ALGORITHM" \
                and isinstance(n.value, _ast.Constant) \
                and isinstance(n.value.value, str):
            st["algorithm"] = n.value.value
    st["baseline"] = bool(has_build and st["algorithm"])
    st["injectable"] = bool(st["baseline"]
                            and st["algorithm"] in _baseline_hp_algos())
    return st


def _inject_baseline_hp_segment(train_py: Path) -> list[str]:
    """在子代副本 main 守卫前注入 baseline 超参消费段(已有旧段先整段替换,幂等)。"""
    from .templates import (BASELINE_HP_BEGIN, BASELINE_HP_END,
                            BASELINE_HP_SEGMENT)

    code = Path(train_py).read_text(encoding="utf-8")
    notes: list[str] = []
    if BASELINE_HP_BEGIN in code:
        i = code.find(BASELINE_HP_BEGIN)
        i = code.rfind("\n", 0, i) + 1
        j = code.find(BASELINE_HP_END, i)
        if j < 0:
            raise SystemExit("baseline 超参消费段有头无尾(代码被手改?),请人工检查")
        j = code.find("\n", j)
        j = len(code) if j < 0 else j + 1
        code = code[:i] + code[j:]
        notes.append("替换旧 baseline 超参消费段")
    block = (f"{BASELINE_HP_BEGIN}(harness 自动注入;包住父代 build_estimator,"
             "以 config.hyperparameters 经 set_params 覆写)====\n"
             + BASELINE_HP_SEGMENT + f"{BASELINE_HP_END}\n")
    out = _insert_before_main(code, block)
    try:
        compile(out, str(train_py), "exec")
    except SyntaxError as exc:
        raise SystemExit(
            f"baseline 超参消费段注入后语法错误(harness 缺陷,请上报):{exc}")
    Path(train_py).write_text(out, encoding="utf-8")
    notes.append("注入 v2.17.0 baseline 超参消费段(build_estimator 外层 "
                 "set_params 覆写 config.hyperparameters)")
    return notes


def _ensure_hyperparameters_consumed(run_dir: Path, trial, parent_id: str,
                                     action: str, *,
                                     allow_code_edit: bool = False) -> list[str]:
    """[v2.16.2] `--from-parent --set` 的创建期消费检查(lint 规则 11 同口径)。

    config.hyperparameters 非空而父代副本不读它 → 覆写静默失效(run4 regression
    gen-04:hurdle 冠军不读 hyperparameters,三个结构参数子代靠 adhoc 脚本手补
    消费点)。hurdle 父代(契约入口齐全)自动注入 hurdle 超参段作为消费点;
    v2.17.0(D2):baseline 血缘父代(xgboost/lightgbm)自动注入 baseline 超参
    消费段(包住 build_estimator,set_params 覆写);其它父代**创建期拒绝**
    (SystemExit → 调用方回滚,exit 1,不留 trial),除非 --allow-code-edit
    (此时告警并留痕 execution.hyperparameters_unconsumed,run 前 lint 规则 11
    仍会拒跑,届时必须已补消费点)。hpo 动作由嫁接/超参段负责,不在此检查。
    返回留痕说明。
    """
    from runtime.logging_utils import load_yaml as _ly

    if action == "hpo":
        return []
    tdir = Path(trial.trial_dir)
    cfg = _ly(tdir / "config" / "trial.yaml") or {}
    hp = cfg.get("hyperparameters")
    if not isinstance(hp, dict) or not hp:
        return []
    tp = tdir / "code" / "train.py"
    code = tp.read_text(encoding="utf-8")
    if '"hyperparameters"' in code or "'hyperparameters'" in code:
        return []
    st = _hurdle_lineage_status(code)
    if st["hurdle"] and st["entries_ok"]:
        notes = _inject_hurdle_hp_segment(tp)
        bad = _import_probe(tdir)
        if bad:
            raise SystemExit(f"hurdle 超参段注入后可导入性探针失败:{bad}")
        logger.info("父代 %s 是 hurdle 且不读 config.hyperparameters:已注入 "
                    "hurdle 超参段作为 --set 消费点(无前缀=两段共用,clf__/reg__="
                    "单段;hurdle_lr/hurdle_cap 为 lr/cap 别名)", parent_id)
        return notes
    # v2.17.0(D2):baseline 血缘(模块级 build_estimator + _ALGORITHM 常量)
    # → 在子代副本包住 build_estimator,以 set_params 消费 config 超参
    bst = _baseline_lineage_status(code)
    if not st["hurdle"] and bst["baseline"] and bst["injectable"]:
        _prefixed = sorted(k for k in hp if "__" in str(k))
        if _prefixed:
            raise SystemExit(
                f"--set 键 {_prefixed} 带子模型前缀(clf__/reg__ 是 hurdle 父代的"
                f"口径),但父代 {parent_id} 是单估计器 baseline({bst['algorithm']})"
                f":去掉前缀后重建,例如 --set "
                + ",".join(f"{str(k).split('__', 1)[1]}={hp[k]}"
                           for k in _prefixed))
        notes = _inject_baseline_hp_segment(tp)
        bad = _import_probe(tdir)
        if bad:
            raise SystemExit(f"baseline 超参消费段注入后可导入性探针失败:{bad}")
        logger.info("父代 %s 是 baseline 血缘(%s)且不读 config.hyperparameters:"
                    "已在子代副本注入超参消费段(build_estimator 外层 set_params"
                    " 覆写 %s)", parent_id, bst["algorithm"], sorted(hp))
        return notes
    _why = (_hurdle_reject_message(parent_id, st, "hurdle 消费点无法注入") + ";"
            if st["hurdle"] else
            (f"父代是 baseline 血缘但算法 {bst['algorithm']} 的估计器不支持 "
             "set_params 覆写(可注入算法:"
             + "/".join(_baseline_hp_algos()) + ");")
            if bst["baseline"] else
            "父代既非 hurdle 也非 baseline 血缘,harness 无法自动注入消费点;")
    msg = (f"父代 {parent_id} 的 code/train.py 不读 config.hyperparameters,"
           f"--set {sorted(hp)} 不会生效(子代会原样重跑父代配方,run 前 lint "
           "规则 11 也会拒跑)。" + _why
           + "选项:--allow-code-edit 后在 fit_and_score / 估计器构造处读取 "
           "cfg[\"hyperparameters\"](参考模板 _cfg_hp 口径);或改用已有消费点的父代")
    if not allow_code_edit:
        # v2.17.0(D2):不可行时创建期直接拒绝(调用方回滚,不留 trial)。
        # 旧实现只告警并留下一个必被 lint 规则 11 拒跑的 trial,diagnose 也不报。
        raise SystemExit(
            "创建期拒绝(--set 无消费点):" + msg
            + f"。替代命令:同一 new-trial 命令加 --allow-code-edit "
            f"--changed-modules model(创建后在 trials/<id>/code/train.py 补消费点"
            f"再 run);或 --from-parent 换成已读 config.hyperparameters 的父代")
    logger.warning("创建期提示:" + msg
                   + "(--allow-code-edit:请在 run 前补齐消费点)")
    try:
        trial.data.setdefault("execution", {})[
            "hyperparameters_unconsumed"] = sorted(hp)
        trial.flush()
    except Exception:  # noqa: BLE001 留痕失败不阻断
        pass
    return []


def _validate_grafted_hpo_deps(trial) -> list[str]:
    """[v2.13.4] 嫁接产物自检:`_hpo_fit_inner` 引用的符号必须都存在。

    事故(req-t7-drawdown-v21331 trial-019,实测 25s FAILED):

      1. 冠军代码自带 `_hpo_fit_inner`(旧机制,调 `_lgbm_fit_safe`),
         `_assert_hpo_capable` 因四件套齐全而**不触发嫁接**;
      2. harness 对**任何** hpo 路径都调 `_inject_hpo_algo_at` 注入
         `_HPO_ALGO` → 模板版 `_build_model` 走 `runtime.hpo.build_gbm`;
      3. `build_gbm` 对 xgboost **构造期**注入 `early_stopping_rounds`,
         而冠军版 `_hpo_fit_inner` 做裸 `model.fit(X, y)`(无 eval_set)
         → `ValueError: Must have at least 1 validation dataset for
         early stopping`。

    正确的修复是让 `_hpo_fit_inner` 走 `_fit_with_es`(cap + 早停,与
    交付同机制)。两条来源:
      - 模板渲染路径:模板自带(且已修好);父代码**定义过**这些符号
        (说明是"自带(旧)HPO 实现的父代")时,必须整段嫁接覆盖父版;
      - 已缺符号的父代码:本来就整段嫁接,天然带全。

    本函数只做**自检**,不产生修复动作 —— 避免自动搬半套依赖造成
    "看起来能跑"的假象;缺什么就明确说缺什么,由创建期报错拦下。
    """
    import re as _re

    code_path = Path(trial.trial_dir) / "code" / "train.py"
    code = code_path.read_text(encoding="utf-8")
    if "def _hpo_fit_inner(" not in code:
        return []
    _body = _re.search(r"def _hpo_fit_inner\(.*?\n(?=\ndef |\Z)",
                       code, _re.S)
    if _body is None:
        return []
    src = _body.group(0)
    # `_build_model` 经 `_HPO_ALGO` 分派到 runtime.hpo.build_gbm:该函数
    # 对 **xgboost/catboost** 在**构造期**注入早停参数
    # (`early_stopping_rounds` / `od_type+od_wait+use_best_model`),
    # 其 fit 必须带 eval_set,否则第一棵树就抛
    # "Must have at least 1 validation dataset for early stopping"。
    # lightgbm 无此约束(早停靠 fit 期 callbacks),裸 fit 合法。
    # 这是**符号齐全但机制错**的形态 —— 冠军版 `_hpo_fit_inner` 恰恰
    # 如此,`_assert_hpo_capable` 查不出(它只查符号在不在)。
    algo = str((trial.data or {}).get("algorithm") or "").lower()
    if algo in ("xgboost", "catboost") and "_fit_with_es(" not in src:
        return ["_fit_with_es(未使用:inner fit 仍是裸 fit,而 "
                f"{algo} 的早停参数在构造期注入(build_gbm),"
                "fit 缺 eval_set 会在第一棵树抛 "
                "`Must have at least 1 validation dataset for early "
                "stopping`)"]
    required = []
    if "_fit_with_es(" in src:
        required.append("_fit_with_es")
    if "build_gbm(" in src:
        required.append("build_gbm")
    missing = [name for name in required
               if f"def {name}(" not in code
               and f"from runtime.hpo import {name}" not in code
               and f"import {name}" not in code]
    return missing


def _assert_hpo_capable(trial, *, skip: bool = False) -> None:
    """hpo 快通道前置校验:复制来的父代码必须真能消费 cfg["hpo"]。

    _copy_parent_trial_code 逐字复制父 .py,_inject_hpo_config 只写
    config —— 两边没有任何一处校验"这份代码里有没有 hpo 分支"。父代
    血缘一旦回溯到 baseline 模板(runtime/baseline.py),就没有
    fit_and_score/_hpo_fit_inner/pick_gpu/dump_json,运行期才炸
    NameError,一个符号一轮(实测 trial-020/021 手工补了三轮)。
    lint 规则 7 只查 outer-validation 泄漏,查不到"根本没有 hpo 分支"。

    skip=True:模板渲染路径调用——代码由 harness 刚按算法渲染(自带
    四件套经 runtime.hpo.build_gbm 分发),跳过父代码符号检查;
    [FIX-2026-09-03] 旧实现固定校验,渲染路径在 code 写盘**之前**
    注入 config 时 FileNotFoundError。

    v19(req-t7-drawdown gen-03 实测回写):返回值改为缺失符号清单
    (空 = capable);SystemExit 行为上移到调用方——gen-03 实测
    "--hpo-parent 只继承 config features,冠军代码内加宽列拿不到;
    --from-parent 又因 baseline 血缘缺四件套被拒",两条路都到不了
    "冠军加宽面 + HPO"。修复 = 调用方对缺四件套的父代码**自动嫁接**
    (渲染模板的 hpo 段注入父代码副本,父 transform 原样保留),
    而不是拒绝。
    """
    code_path = Path(trial.trial_dir) / "code" / "train.py"
    code = code_path.read_text(encoding="utf-8")
    missing = [s for s in _HPO_REQUIRED_SYMBOLS if s not in code]
    if "runtime.hpo" not in code:
        missing.append("from runtime.hpo import ...")
    if skip:
        return []
    return missing


def _hpo_lr_mode_preview(run_dir: Path, parents: list, hpo_cfg: dict,
                         run_cfg) -> dict:
    """[v2.16.3] 创建期预判 HPO 的生效 lr 模式(与运行期 plan_lr_mode 同源)。

    inner 行数取父代 metrics 的 ``n_train × (1 − inner_fraction)``,再按
    ``sample_rows`` 截断(与运行期同口径);取不到行数时 auto
    无法预判,生效值记 None、说明"运行期裁定"。hurdle 的 ``hurdle_lr`` 别名按
    lr 维处理。返回 ``{"fields": {...写进 cfg.hpo}, "message", "warn"}``。
    """
    from runtime.hpo import param_base_name, plan_lr_mode

    space = {}
    for k, v in dict(hpo_cfg.get("search_space") or {}).items():
        base = param_base_name(k)
        if base == "hurdle_lr":
            k = str(k)[:len(str(k)) - len(base)] + "learning_rate"
        space[k] = v
    requested = str(hpo_cfg.get("lr_mode") or "search").strip().lower()
    frac = float(hpo_cfg.get("inner_fraction")
                 or run_cfg.get("evolution.hpo.inner_fraction", 0.2) or 0.2)
    n_inner = 0
    for pid in parents or []:
        mf = Path(run_dir) / "trials" / str(pid) / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = load_json(mf) or {}
        except Exception:  # noqa: BLE001
            continue
        n_tr = int(m.get("n_train") or 0)
        n_inner = int(n_tr * (1.0 - frac)) if n_tr else 0
        if n_inner:
            break
    sample = hpo_cfg.get("sample_rows")
    if sample and n_inner:
        n_inner = min(int(sample), n_inner)
    fields = {"lr_mode_requested": requested}
    if requested == "auto" and not n_inner:
        fields["lr_mode_effective"] = None
        fields["lr_mode_note"] = ("auto:创建期无父代 inner 行数,生效模式由运行期"
                                  "按 inner 行数裁定(见 metrics.hpo_lr_mode_effective)")
        return {"fields": fields, "message": fields["lr_mode_note"],
                "warn": False}
    plan = plan_lr_mode(space, hpo_cfg, n_inner)
    fields["lr_mode_effective"] = plan["mode"]
    if plan.get("note"):
        fields["lr_mode_note"] = plan["note"]
    msg = None
    if plan.get("note"):
        msg = (f"请求 {requested} → 生效 {plan['mode']}"
               + (f"(预估 inner {n_inner} 行)" if n_inner else "")
               + f":{plan['note']}")
    return {"fields": fields, "message": msg, "warn": bool(plan.get("warn"))}


def _inject_hpo_config(run_dir: Path, trial, search_space_path: str,
                       budget: int, *, skip: bool = False,
                       sample_rows: int | None = None,
                       hpo_opts: dict | None = None) -> None:
    """hpo 快通道(v11):把搜索空间与预算写进 config/trial.yaml。

    空间 YAML 格式(每键一个分布声明):
        learning_rate: {type: float, low: 0.01, high: 0.2, log: true}
        num_leaves:   {type: int, low: 16, high: 128}
        feature_fraction: {type: float, low: 0.6, high: 1.0}
    trial 代码侧(模板 hpo 入口)读取后在 inner-validation 上搜索;
    lint 规则 7 禁止 hpo trial 在 outer validation 上做选择。

    v18/P1-4:sample_rows 非 None 时写入 hpo.sample_rows——inner 搜索
    在分层抽样的子样本上做(标签分层,防前缀偏差),best_params 仍
    全量重训。400w run 实测:inner 构建 6217s + 单 config ~900s,
    360min heavy 只装 1 个 config;抽样面搜索 + 全量 confirm 是
    大数据 HPO 唯一可规划路径。成本闸外推按抽样后行数等比缩放。
    """
    from runtime.logging_utils import dump_yaml, load_yaml

    # v14:先验证复制来的父代码真能跑 hpo(缺符号在创建期报,
    # 不是运行期 NameError 一轮一轮补)
    # v19:缺四件套时自动嫁接(不再是 SystemExit)——gen-03 实测
    # "冠军加宽列在父 code transform 里,--hpo-parent 拿不到;
    # --from-parent 又被四件套卡死"两条路都到不了加宽面 HPO。
    # 嫁接 = 从渲染模板 body 抽取 hpo 段(四件套 + _lgbm_fit_safe/
    # step/n_jobs 等 hpo 依赖)注入父代码副本;父 transform 逻辑
    # (加宽/特征工程)原样保留。父代码自定义的 fit_and_score 若与
    # 模板签名冲突,以父代码为准并告警(嫁接只补缺失符号)。
    _missing = _assert_hpo_capable(trial, skip=skip)
    # [v2.16.2] hurdle 父代(structural:hurdle):不嫁接模板 hpo 段(会覆盖父代
    # hurdle 版 fit_and_score,inner/重训都退化成单个回归器,D-008),改注入
    # hurdle 超参段:inner 搜索与 outer 重训都构造父代 hurdle,搜索键作用于
    # 两个子模型(共用 / clf__ / reg__)。契约入口不齐时拒绝并给替代路径。
    _hurdle_st = None
    _cp_h = Path(trial.trial_dir) / "code" / "train.py"
    # v2.17.0:模板渲染路径(skip=True)同样识别 hurdle —— `--hurdle-template
    # --action hpo` 的 hurdle 不得注入 _HPO_ALGO / 走单模型口径
    if _cp_h.exists():
        _hurdle_st = _hurdle_lineage_status(_cp_h.read_text(encoding="utf-8"))
    _is_hurdle = bool(_hurdle_st and _hurdle_st["hurdle"])
    if _is_hurdle:
        _pid_h = ((trial.data or {}).get("parent_trial_ids") or ["?"])[0]
        if not _hurdle_st["entries_ok"]:
            raise SystemExit(_hurdle_reject_message(
                _pid_h, _hurdle_st, "hurdle HPO 拒绝创建"))
        _h_notes = _inject_hurdle_hp_segment(_cp_h)
        _bad = _import_probe(Path(trial.trial_dir))
        if _bad:
            raise SystemExit(
                f"hurdle 超参段注入后可导入性探针失败:{_bad}(trial 已回滚;"
                "请勿手改 code/train.py,检查父代 hurdle 入口)")
        trial.data.setdefault("execution", {})[
            "hpo_topology"] = "structural:hurdle"
        trial.data["execution"]["hpo_hurdle_segment"] = _h_notes
        try:
            trial.flush()
        except Exception:  # noqa: BLE001 留痕失败不阻断创建
            pass
        logger.info("hpo:父代 %s 为 hurdle(%s),已注入 hurdle 超参段(inner 与"
                    "重训均构造父代 hurdle;不嫁接模板 hpo 段、不注入 _HPO_ALGO)",
                    _pid_h, _hurdle_st.get("source"))
        _missing = []
    # [v2.13.4] "自带旧 HPO 实现"的父代码(四件套齐全 → _missing 为空
    # → 旧实现**不嫁接**)是另一个坑:harness 仍会注入 `_HPO_ALGO`、
    # 把 `_build_model` 切到 `build_gbm`(构造期早停),而父版
    # `_hpo_fit_inner` 的裸 `model.fit(X, y)` 不满足该契约 → 25 秒
    # FAILED(见 `_validate_grafted_hpo_deps` docstring)。
    # 判据:父代码**定义过**模板 hpo 段的符号,说明它是"自带 HPO 实现
    # 的父代",其实现可能与 harness 的 HPO 契约不兼容 → 走整段嫁接,
    # 用模板版覆盖(模板版是唯一与 build_gbm 契约对齐的实现)。
    # 这不会波及普通加宽父代:它们不定义这些符号,_missing 非空,
    # 本来就走嫁接。
    if not _missing and not skip and not _is_hurdle:
        _cp_chk = Path(trial.trial_dir) / "code" / "train.py"
        if _cp_chk.exists():
            _owns = _parent_hpo_overridden_symbols(
                _cp_chk.read_text(encoding="utf-8"),
                str((trial.data or {}).get("algorithm") or ""))
            if _owns:
                logger.warning(
                    "hpo 快通道:父代码自带 %s(自带 HPO 实现的父代),"
                    "其 inner fit 机制可能与 runtime.hpo 的构造期早停契约"
                    "不兼容;改用模板 hpo 段整段嫁接覆盖,保持与交付同机制。",
                    ", ".join(_owns))
                trial.data.setdefault("execution", {})[
                    "hpo_graft_overrides_parent_symbols"] = _owns
                try:
                    trial.flush()
                except Exception:  # noqa: BLE001 留痕失败不阻断创建
                    pass
                _missing = list(_HPO_REQUIRED_SYMBOLS)
    if _missing:
        # [v2.13.4] 嫁接前先看父代码**已有**哪些 hpo 符号:整段嫁接会
        # 用模板版覆盖它们(见 _parent_hpo_overridden_symbols)。
        # 这是"父代自带旧 HPO 实现"的信号 —— 覆盖后行为可能反而不如
        # 父版(尤其父版是为自己的算法定制的),必须显式留痕。
        _cp_pre = Path(trial.trial_dir) / "code" / "train.py"
        if _cp_pre.exists():
            _overridden = _parent_hpo_overridden_symbols(
                _cp_pre.read_text(encoding="utf-8"),
                str((trial.data or {}).get("algorithm") or ""))
            if _overridden:
                logger.warning(
                    "hpo 嫁接:父代码已定义 %s,模板版将在文件尾覆盖它们"
                    "(整段嫁接语义)。若嫁接产物在 HPO inner 搜索期报 "
                    "early stopping 相关错误,优先检查这里 —— 模板 body "
                    "抽不出 _fit_with_es 时,被覆盖的父版可能是唯一能"
                    "正确消费构造期早停参数的实现。",
                    ", ".join(_overridden))
                trial.data.setdefault("execution", {})[
                    "hpo_graft_overrides_parent_symbols"] = _overridden
                try:
                    trial.flush()
                except Exception:  # noqa: BLE001 留痕失败不阻断创建
                    pass
        _graft_hpo_section(trial)
        # [v2.13.4] 嫁接后自检:inner fit 必须与 build_gbm 的构造期早停
        # 契约对齐(否则 xgboost/catboost 在 inner 搜索第一棵树就炸)。
        # 缺符号直接拒绝创建 —— 让失败发生在创建期(秒级、信息完整),
        # 而不是训练期(25s 后抛 xgboost 深层 ValueError)。
        _dep_gap = _validate_grafted_hpo_deps(trial)
        if _dep_gap:
            raise SystemExit(
                "hpo 嫁接产物自检失败:inner fit 机制与 runtime.hpo 的"
                f"构造期早停契约不兼容,缺 {_dep_gap}。"
                "这是「嫁接段从模板 body 抽不全依赖」的形态(模板结构"
                "变更时需同步 _graft_hpo_section 的抽取边界)。"
                "处置:不要手工补代码,请改用 `--action hpo --algorithm "
                "<algo>`(模板渲染路径,四件套由 harness 按算法生成)。")
        # [v2.16.2] 嫁接后结构检查(lint 规则 23 同源):生效的 fit_and_score
        # 不再到达父代 structural 训练入口(hurdle / T-learner …)= 拓扑被模板
        # 单模型静默替换。旧实现只打一行 WARNING(D-008),现拒绝创建。
        try:
            _fx = _lint_module().structural_override_findings(
                (Path(trial.trial_dir) / "code" / "train.py").read_text(
                    encoding="utf-8"))
        except Exception as _exc:  # noqa: BLE001 检查器异常不掩盖创建
            logger.warning("嫁接后结构检查失败(不阻断):%s", _exc)
            _fx = []
        if _fx:
            raise SystemExit(
                "hpo 嫁接拒绝:嫁接段的 fit_and_score 覆盖了父代 structural 训练"
                f"入口 {_fx[0]['entries'][:6]}(生效第 {_fx[0]['effective_line']} 行,"
                f"被覆盖第 {_fx[0]['overridden_line']} 行)——inner 搜索与重训会"
                "退化为单模型,Δ_parent 混入拓扑变化。hurdle 父代应自动走 hurdle "
                "超参段(若到这里说明契约入口识别失败,请上报);确要换单模型请用"
                "模板渲染路径 `--action hpo --algorithm <algo> --hpo-parent <id>`"
                "并在 hypothesis 声明拓扑变化")
    # v19.3 修复(算法口径错):嫁接/快通道路径同样必须落 _HPO_ALGO ——
    # 模板 _build_model 在缺该常量时**回退 lightgbm**,而搜索空间的键
    # 按父代算法命名,trial_id 也按算法命名:不注入 = trial 声明 xgboost
    # 却训 lightgbm,整轮搜索口径作废(gen-03 trial-008 实测,手工补才
    # 纠正)。模板渲染路径在 cmd_new_trial 里已注入;此处覆盖
    # --from-parent 快通道(父代码自带 _HPO_ALGO 时幂等短路)。
    _algo_eff = str(
        (trial.data or {}).get("algorithm")
        or Path(trial.trial_dir).name.split("-")[3]
        or "lightgbm").strip().lower()
    if _algo_eff in _HPO_RENDER_ALGOS and not _is_hurdle:
        _cp = Path(trial.trial_dir) / "code" / "train.py"
        if _cp.exists():
            try:
                _code_new = _inject_hpo_algo_at(_cp.read_text("utf-8"),
                                                _algo_eff)
            except SystemExit as _exc:
                raise SystemExit(
                    f"hpo 快通道算法常量注入失败({_exc});"
                    "父代码结构异常,请人工检查 code/train.py") from _exc
            if _code_new is not None:
                _cp.write_text(_code_new, encoding="utf-8")
                logger.info("hpo 快通道:已注入 _HPO_ALGO=%r(父代码缺该"
                            "常量,缺省会回退 lightgbm 造成算法口径错)",
                            _algo_eff)
            # [v2.15.0] 重复嫁接(父代已改线)时改线不再执行,脚手架 raise
            # 的 lint 标记会残留 → 规则 10 拒跑(run3 每轮靠 Agent 手删)
            _c_now = _cp.read_text(encoding="utf-8")
            _c_neu = _neutralize_scaffold_marker(_c_now)
            if _c_neu != _c_now:
                _cp.write_text(_c_neu, encoding="utf-8")
                logger.info("hpo:算法化分支在位,脚手架 raise 标记已中和")
    sp = Path(search_space_path)
    space = load_yaml(sp) if sp.exists() else None
    if not isinstance(space, dict) or not space:
        raise SystemExit(f"搜索空间 YAML 无效或为空: {sp}")
    cfg_path = Path(trial.trial_dir) / "config" / "trial.yaml"
    cfg = load_yaml(cfg_path) or {}
    cfg["hpo"] = {
        "search_space": space,
        "budget_configs": int(budget),
        "inner_split": "time_tail",   # train 时间序后 20% 为 inner-valid
        "sampler": "tpe_or_random",   # optuna 在→TPE,缺失→随机
        "space_source": str(sp.resolve()),
    }
    if _is_hurdle:
        cfg["hpo"]["topology"] = "structural:hurdle"
    if sample_rows is not None and int(sample_rows) > 0:
        cfg["hpo"]["sample_rows"] = int(sample_rows)
        cfg["notes"] = (f"hpo trial:inner 搜索 {budget} configs"
                        f"(分层抽样 {int(sample_rows)} 行),"
                        f"空间来自 {sp.name};best_params 全量重训,"
                        "outer validation 只评一次")
    else:
        cfg["notes"] = (f"hpo trial:inner 搜索 {budget} configs,"
                        f"空间来自 {sp.name};outer validation 只评一次")
    # ---- v2.15.0:HPO 引擎参数(CLI 覆写 > run 配置 evolution.hpo.*) ----
    _rd = Path(trial.trial_dir).parent.parent
    _rc = _load_config(_rd)
    _opts = dict(hpo_opts or {})
    for _key in ("lr_mode", "parallel_configs", "auto_expand", "lr_floor",
                 "fidelity", "search_lr", "lr_shrink_factor",
                 "lr_shrink_min_rows"):
        _v = _opts.get(_key)
        if _v is None:
            _v = _rc.get(f"evolution.hpo.{_key}")
        if _v is not None:
            cfg["hpo"][_key] = _v
    # v2.17.2(A4):抽样面保真复评(opt-in;0 = 关,不写入 config)
    _rk = _opts.get("fidelity_recheck_top_k")
    if _rk is None:
        _rk = _rc.get("evolution.hpo.fidelity_recheck_top_k")
    try:
        _rk = int(_rk or 0)
    except (TypeError, ValueError):
        _rk = 0
    if _rk > 0:
        cfg["hpo"]["fidelity_recheck_top_k"] = _rk
    # 基准点:父代配方(best_params ← hyperparameters ← 算法默认)enqueue
    # 为第一个评估点 —— 搜索永远知道"父代配方在 inner 上得几分"。
    _algo_inc = str((trial.data or {}).get("algorithm") or "").lower()
    _parents = list((trial.data or {}).get("parent_trial_ids") or [])
    try:
        from runtime.hpo import collect_warm_start, incumbent_params

        _pbest, _php = None, None
        if _parents:
            _pdir = _rd / "trials" / _parents[0]
            _pm = load_json(_pdir / "metrics.json") \
                if (_pdir / "metrics.json").exists() else {}
            _pbest = (_pm or {}).get("best_params")
            _pc = load_yaml(_pdir / "config" / "trial.yaml") \
                if (_pdir / "config" / "trial.yaml").exists() else {}
            _php = (_pc or {}).get("hyperparameters")
        if _is_hurdle:
            # hurdle:父代配方基准点在运行期从父代 _hurdle_estimators 读回
            # (算法默认值不是父代配方);父代本身是 hurdle HPO 时追加其重训配方
            _hb = _hurdle_parent_hpo_best(_rd, _parents[0] if _parents
                                          else None)
            cfg["hpo"]["enqueue"] = [_hb] if _hb else []
            _inc = None
        else:
            _inc = incumbent_params(_algo_inc, _pbest if isinstance(
                _pbest, dict) else None,
                _php if isinstance(_php, dict) else None)
        if _inc:
            cfg["hpo"]["enqueue"] = [_inc]
        # 热启动:run 内签名一致的历史观测(运行期按 inner 签名精确比对)
        _ws_on = _opts.get("warm_start")
        if _ws_on is None:
            _ws_on = _rc.get("evolution.hpo.warm_start", True)
        _ws_path = Path(trial.trial_dir) / "config" / "hpo-warm-start.json"
        if _ws_on:
            _srcs = collect_warm_start(_rd, algorithm=_algo_inc or None,
                                       exclude=(trial.trial_id,))
            if _srcs:
                dump_json(_ws_path, {"created_at": now_iso(),
                                     "sources": _srcs})
                logger.info("hpo 热启动候选:%d 个历史 study(%s),运行期按"
                            "inner 签名精确匹配", len(_srcs),
                            ", ".join(s["source_trial"] for s in _srcs))
        cfg["hpo"]["warm_start"] = bool(_ws_on)
    except Exception as _exc:  # noqa: BLE001 基准点/热启动是优化项,失败不阻断
        logger.warning("hpo 基准点/热启动准备失败(不阻断创建): %s", _exc)
    # ---- v2.16.3:创建期就裁定 lr 模式并显式留痕 ----
    # 运行期 plan_lr_mode 才真正裁定(auto 依赖 inner 行数),旧实现创建期完全
    # 不看它:gen-05 trial-022 显式 `--hpo-lr-mode shrink`,空间无 lr 维,运行期
    # 静默变成 search(inner 用父代 lr、重训不收缩),trial.yaml 仍写 shrink。
    # 这里用创建期可知的 inner 行数预判一次,把裁定写进 trial.yaml 的
    # hpo.lr_mode_effective / lr_mode_requested / lr_mode_note,请求 ≠ 生效或
    # 显式 shrink 移除 lr 维时 WARNING;运行期 record_lr_mode 以实测覆写。
    try:
        _lr_plan = _hpo_lr_mode_preview(_rd, _parents, cfg["hpo"], _rc)
        cfg["hpo"].update(_lr_plan["fields"])
        if _lr_plan.get("message"):
            (logger.warning if _lr_plan.get("warn") else logger.info)(
                "hpo lr_mode:%s", _lr_plan["message"])
            trial.data.setdefault("execution", {})[
                "hpo_lr_mode_note"] = _lr_plan["message"]
    except Exception as _exc:  # noqa: BLE001 预判失败不阻断创建
        logger.warning("hpo lr 模式预判失败(不阻断创建): %s", _exc)
    if _is_hurdle:
        _adv = _hurdle_lr_cap_advice(_rd, trial, _cp_h.read_text(
            encoding="utf-8"), cfg["hpo"])
        if _adv:
            logger.warning("hurdle HPO:%s", _adv)
            trial.data.setdefault("execution", {})["hpo_lr_cap_advice"] = _adv
    dump_yaml(cfg_path, cfg)
    # 搜索空间拷贝进 trial 目录(自包含,交付可复现)
    import shutil as _sh

    _sh.copy2(sp, Path(trial.trial_dir) / "config" / "search-space.yaml")
    trial.data["execution"]["action_description"] = [
        f"hpo: inner 搜索 {budget} configs(space={sp.name})"]
    # v14:hpo = 重活。不打 heavy 标记就落回 execution.trial_timeout_minutes
    # (默认 90 分钟),40 config 的搜索必被 SIGTERM(实测 trial-020/021
    # 各 4 次 rc=-15 白跑数小时)。此处与 --heavy 同机制放大超时。
    _run_dir = Path(trial.trial_dir).parent.parent
    er = dict(trial.data.get("expected_resources") or {})
    er.setdefault("heavy", True)
    er.setdefault("timeout_multiplier", float(
        _load_config(_run_dir).get(
            "evolution.hpo.timeout_multiplier", 4)))
    er["hpo_budget_configs"] = int(budget)
    trial.data["expected_resources"] = er
    trial.flush()

    # v16/P0-2:HPO 创建期成本闸(实测 400w run trial-005:inner split
    # 构建 6217s + 单 config ~900s,heavy 360min 只装下 1 个 config,
    # "40 configs TPE"实为单点评估,预算全耗在构建上)。按父代实测
    # runtime 外推搜索成本,装不下时在创建期拒绝并给降级选项——
    # 比 deadline 中途截断(信息量≈单点)早一代止损。
    _hpo_cost_gate(trial, _run_dir, budget)


def _hurdle_segment_seconds(parent_dir: Path) -> dict | None:
    """父代主轮 hurdle 两段的实测拟合秒数(stdout.log 的 step 行)。

    `_fit_hurdle` 写 "[step] hurdle clf fit start/done … (Ts)" 与 reg 同款;
    取第一对(主轮;multi-seed 变体轮在其后)。缺日志/格式不符返回 None。
    """
    import re as _re

    p = Path(parent_dir) / "stdout.log"
    if not p.exists():
        return None
    pat = _re.compile(r"^\[step\] hurdle (clf|reg) fit (start|done)\b.*"
                      r"\((\d+(?:\.\d+)?)s\)\s*$")
    marks: dict = {}
    try:
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = pat.match(line.strip())
                if m:
                    marks.setdefault((m.group(1), m.group(2)),
                                     float(m.group(3)))
                if len(marks) >= 4:
                    break
    except OSError:
        return None
    out = {}
    for seg in ("clf", "reg"):
        a, b = marks.get((seg, "start")), marks.get((seg, "done"))
        if a is None or b is None or b < a:
            return None
        out[seg] = round(b - a, 1)
    return out


def _hpo_cost_estimate(trial, run_dir: Path, budget: int) -> dict | None:
    """hpo 成本外推(v16 口径,v2.16.2 订正;None = 无父代实测,不外推)。

    v2.16.2 订正(D-008 实测 trial-008 被估 796 min 拒绝):
    - 父代 runtime_sec 含 harness 评估段(group_importance 等,trial-008 2244s)
      与 confirm 的 multi-seed 变体轮(3 次全量拟合)——两者都不是搜索成本:
      改用"训练进程耗时 / 拟合次数"作单次全量拟合成本;
    - `--hpo-sample-rows` 从未生效:旧实现读 trial.data["config"](trial.yaml
      没有该键),现读 config/trial.yaml 的 hpo.sample_rows;
    - 并行 config(hpo.parallel_configs;auto = GPU 上的 xgboost 取 2)按
      并行度折算搜索段墙钟;
    - hurdle(hpo.topology=structural:hurdle):每个 config 训练两段(分类段全部
      inner 行 + 条件回归段其中 y>0 的行),按父代主轮两段实测秒数(stdout.log)
      × inner 拟合行数比外推;取不到分段秒数时用单次拟合成本 × 行数比。
    """
    er = dict(trial.data.get("expected_resources") or {})
    cfg = _load_config(run_dir)
    base_timeout = float(cfg.get("execution.trial_timeout_minutes", 90))
    mult = float(er.get("timeout_multiplier", 4))
    budget_sec = base_timeout * 60 * mult
    parent_ids = trial.data.get("parent_trial_ids") or []
    parent_runtime = None
    pt_best = None
    for pid in parent_ids:
        try:
            from runtime.trial import TrialRecord as _TR

            pt = _TR.load(Path(run_dir) / "trials" / pid)
            rt = (pt.data.get("result") or {}).get("runtime_sec")
            if isinstance(rt, (int, float)) and float(rt) > (
                    parent_runtime or 0):
                parent_runtime, pt_best = float(rt), pt
        except Exception:  # noqa: BLE001
            continue
    if parent_runtime is None or parent_runtime <= 0:
        return None  # 无父代实测数据,不做外推(首代 hpo 不拦)
    _res = pt_best.data.get("result") or {}
    eval_sec = _res.get("evaluate_sec")
    train_sec = parent_runtime
    if isinstance(eval_sec, (int, float)) and 0 < float(eval_sec) < parent_runtime:
        train_sec = parent_runtime - float(eval_sec)
    _pdir = Path(pt_best.trial_dir)
    pm = load_json(_pdir / "metrics.json") if (_pdir / "metrics.json"
                                                ).exists() else {}
    pm = pm or {}
    try:
        n_fits = max(1, int(pm.get("multi_seed_n") or 1))
    except (TypeError, ValueError):
        n_fits = 1
    single = train_sec / n_fits
    tcfg = load_yaml(Path(trial.trial_dir) / "config" / "trial.yaml") \
        if (Path(trial.trial_dir) / "config" / "trial.yaml").exists() else {}
    hcfg = dict((tcfg or {}).get("hpo") or {})
    frac = float(hcfg.get("inner_fraction")
                 or cfg.get("evolution.hpo.inner_fraction", 0.2))
    sample_rows = hcfg.get("sample_rows")
    hurdle = str(hcfg.get("topology") or "") == "structural:hurdle"
    try:
        n_train = int(pm.get("n_train") or 0)
    except (TypeError, ValueError):
        n_train = 0
    if not n_train and (sample_rows or hurdle):
        try:
            from runtime.data import load_split_frames

            _tr = load_split_frames(Path(run_dir)).get("train")
            n_train = int(len(_tr)) if _tr is not None else 0
        except Exception:  # noqa: BLE001
            n_train = 0
    est: dict = {"parent_runtime_sec": round(parent_runtime, 1),
                 "parent_train_sec": round(train_sec, 1),
                 "parent_fits": n_fits, "single_fit_sec": round(single, 1),
                 "budget_configs": int(budget),
                 "sample_rows": int(sample_rows) if sample_rows else None,
                 "topology": "structural:hurdle" if hurdle else "single"}
    if hurdle:
        inner_rows = n_train * (1.0 - frac) if n_train else None
        if inner_rows and sample_rows:
            inner_rows = min(inner_rows, float(int(sample_rows)))
        ratio = (inner_rows / float(n_train)) if inner_rows and n_train \
            else (1.0 - frac)
        seg = _hurdle_segment_seconds(_pdir)
        if seg:
            per_config = (seg["clf"] + seg["reg"]) * ratio
            est["segments_sec_parent"] = seg
        else:
            per_config = single * ratio
        est["inner_rows_ratio"] = round(ratio, 4)
        retrain = single
    else:
        # 单模型:沿用 v16 外推口径(单 config ≈ 单次拟合 × inner_fraction,
        # sample_rows 按行数等比缩放),只把父代 runtime 换成单次拟合成本
        if sample_rows and n_train and n_train > int(sample_rows):
            frac = frac * float(int(sample_rows)) / float(n_train)
        per_config = single * frac
        retrain = single
    try:
        from runtime.hpo import hpo_parallelism

        _gpu = (pt_best.data.get("environment") or {}).get(
            "gpu_index") is not None
        par = int(hpo_parallelism(hcfg, str(trial.data.get("algorithm") or ""),
                                  bool(_gpu)))
    except Exception:  # noqa: BLE001
        par = 1
    search = per_config * int(budget) / max(1, par)
    total = search + retrain
    est.update({"per_config_sec": round(per_config, 1), "parallel": par,
                "search_sec": round(search, 1), "retrain_sec": round(retrain, 1),
                "total_min": round(total / 60.0, 1),
                "budget_min": round(budget_sec / 60.0, 1),
                "limit_min": round(budget_sec * 0.8 / 60.0, 1),
                "fits": bool(total <= budget_sec * 0.8)})
    return est


def _hpo_cost_gate(trial, run_dir: Path, budget: int) -> None:
    """hpo 成本预估闸:按父代实测保守外推,超预算即拒绝创建。

    外推口径见 `_hpo_cost_estimate`(v2.16.2:单次拟合成本 = 父代训练进程
    耗时 / 拟合次数;hurdle 计两段;sample_rows 与并行 config 生效)。
    预估总成本 > heavy 超时上限的 80%(留评估余量)→ 拒绝。估算明细写入
    expected_resources.hpo_cost_estimate(豁免时同样留痕)。
    逃逸口:--force-hpo-budget 显式声明(留痕 expected_resources)。
    """
    er = dict(trial.data.get("expected_resources") or {})
    try:
        est = _hpo_cost_estimate(trial, run_dir, budget)
    except Exception as exc:  # noqa: BLE001 外推失败不阻断(与无父代同口径)
        logger.warning("hpo 成本外推失败(不阻断):%s", exc)
        est = None
    if est is not None:
        er["hpo_cost_estimate"] = est
        trial.data["expected_resources"] = er
        try:
            trial.flush()
        except Exception:  # noqa: BLE001
            pass
    if er.get("hpo_force_budget"):
        return  # 用户显式豁免
    if est is None or est["fits"]:
        return
    per_config = max(est["per_config_sec"], 1e-6)
    par = max(1, int(est.get("parallel") or 1))
    _room = est["limit_min"] * 60.0 - est["retrain_sec"]
    _fit_n = max(1, int(_room * par / per_config)) if _room > 0 else 1
    raise SystemExit(
        f"hpo 成本闸:预估搜索成本 {est['total_min']:.0f} 分钟"
        f"(单 config ≈ {per_config / 60:.1f} 分钟 × {budget} configs"
        f" / 并行 {par} + 重训 {est['retrain_sec'] / 60:.0f} 分钟;"
        f"父代单次拟合 {est['single_fit_sec'] / 60:.0f} 分钟,"
        f"拓扑 {est['topology']})> heavy 预算 "
        f"{est['budget_min']:.0f} 分钟的 80%。"
        "降级选项:(a) --budget-configs 缩到 "
        f"{_fit_n} 以内;"
        "(b) --hpo-sample-rows N(inner 拟合池分层抽样,best_params 仍全量重训);"
        "(c) --hpo-parallel 2(GPU 上 xgboost 线程并发 config);"
        "(d) 降低 evolution.hpo.inner_fraction;"
        "(e) 确认要跑则加 --force-hpo-budget(留痕 expected_resources)")


def _copy_parent_trial_code(run_dir: Path, trial, parent_id: str,
                            overrides: dict,
                            features_override: list[str] | None = None,
                            action: str = "explore") -> None:
    """参数 trial 快通道:复制父 trial 的 code/(全部文件)+ config/,
    把 overrides 合并进 config/trial.yaml 的 hyperparameters 节。

    零 LLM 写码:代码路径与父代完全一致,差异只走配置注入
    (模板约定:超参优先从 config 读,代码内只放默认值)。
    features_override 显式传入时覆写 config 的 features.columns
    (特征面单源存 config,代码运行期 exclude_columns() 读取,子代
    特征面与父代的差异同样走配置注入,不碰代码)。
    无法安全注入时显式报错,不做静默猜测。
    """
    import shutil as _shutil

    src = Path(run_dir) / "trials" / parent_id
    src_code = src / "code"
    if not (src_code / "train.py").exists():
        raise SystemExit(f"父 trial 无 code/train.py: {src}")
    dst_code = Path(trial.trial_dir) / "code"
    dst_code.mkdir(parents=True, exist_ok=True)
    # v2.11.2:原来只复制 .py,与 docstring"复制 code/(全部文件)"相悖。
    # code/ 下的非 .py 附属物(gen-02 压缩面 trial 的 prune_keep.json
    # ——模块级 _load_prune_keep() 读取、缺失即 raise)曾被静默丢弃,
    # 子 trial 创建成功/lint 通过,直到 run 才 0.5s 炸(infra_error)。
    # gen-02 反思首记、gen-03 复发致 3 个 trial 白跑,故收口到"全部
    # 常规文件":__pycache__ 是目录,is_file() 天然跳过;仅排除 .pyc。
    for f in sorted(src_code.iterdir()):
        if not f.is_file() or f.suffix == ".pyc":
            continue
        _shutil.copy2(f, dst_code / f.name)
    # config:父 trial.yaml + hyperparameters 覆写
    from runtime.logging_utils import load_yaml

    cfg = load_yaml(src / "config" / "trial.yaml") or {}
    # [v2.16.0] 早停口径升级(对所有 run 立即生效):旧父代代码里的两段式
    # _fit_with_eval / baseline 血缘的 train 尾部切分,在子代副本上改为
    # outer validation 单段早停(父代冻结代码不动;运行期按
    # MODELEVO_ES_SOURCE 分派,train_tail 时行为不变)。
    _es_up_notes = _apply_es_source_upgrade(dst_code / "train.py")
    if _es_up_notes:
        cfg["notes"] = (str(cfg.get("notes") or "") + ";"
                        + ";".join(_es_up_notes)).lstrip(";")
    # [FIX-2026-09-01] 快通道剥离 hpo 节:父代若是 hpo trial,其
    # config 含 hpo 搜索配置;子代(非 hpo 动作)继承后会在 fit_and_score
    # 里误入 hpo 搜索分支,把"参数型 exploit"变成 40-config 内搜白跑
    # (req-t7-drawdown 实测:t031/t032/t037 各误跑 ~60min,搜索后 outer
    # 分数仍只评一次,白白浪费预算)。非 hpo 动作的 from-parent 一律
    # 移除 hpo 节;真 hpo 动作走 _inject_hpo_config 单独写回。
    # [FIX-2026-09-01b] 原 hpo 剥离块引用了本函数作用域不存在的
    # args(NameError):凡 --from-parent 且父代 config 带 hpo 节,
    # 快通道创建必炸(实测 withdraw-t7:父=trial-006(hpo),全部
    # 参数型子代创建失败,被迫回退手写码)。action 改为显式入参。
    if cfg.get("hpo") and action != "hpo":
        # [FIX-2026-09-17] 父代是 HPO trial 时,真正的冠军参数不在
        # config.hyperparameters(它是 {} —— HPO 是运行期搜出来的),
        # 而在 trials/<parent>/metrics.json 的 best_params。
        # 旧实现只 pop 掉 hpo 节就完事,子代于是跑在**默认超参**上:
        # `--action confirm --from-parent <hpo_trial>` 会静默确认一个
        # 与父代配方不同的模型("审计上确认了冠军、实际确认的是另一个
        # 模型"),`--set` 参数型子代也会丢掉父代冠军参数。实测
        # drawdown/t7 gen-03:confirm 只能靠手写 child trial 绕过。
        # 这里在剥离前把 best_params 捞进 hyperparameters。
        _best = None
        try:
            _mj = src / "metrics.json"
            if _mj.exists():
                _bp = (json.loads(_mj.read_text(encoding="utf-8"))
                       or {}).get("best_params")
                if isinstance(_bp, dict) and _bp:
                    _best = _bp
        except Exception as _exc:  # noqa: BLE001
            print(f"[warn] 读取父代 {parent_id} 的 best_params 失败"
                  f"({type(_exc).__name__}: {_exc}),子代将不继承 HPO "
                  "冠军参数", flush=True)
        cfg.pop("hpo", None)
        _note = ";快通道剥离父代 hpo 节(非 hpo 动作)"
        if _best:
            _bp2 = cfg.get("hyperparameters")
            _bp2 = dict(_bp2) if isinstance(_bp2, dict) else {}
            # overrides 优先级最高:显式 --set 覆盖继承来的冠军参数
            _merged = dict(_best)
            _merged.update(_bp2)
            cfg["hyperparameters"] = _merged
            _note += (f";已从父代 metrics.json.best_params 继承 HPO "
                      f"冠军参数 {sorted(_best)}")
            print(f"[info] 父代 {parent_id} 是 HPO trial,已继承其 "
                  f"best_params {sorted(_best)} 到 config."
                  f"hyperparameters(旧实现会静默退回默认超参)")
        elif cfg.get("hyperparameters") in (None, {}, ""):
            print(f"[warn] 父代 {parent_id} 是 HPO trial 但 "
                  "metrics.json.best_params 缺失/为空:子代将跑在默认"
                  "超参上,与父代冠军配方不等价——若本 trial 用于 "
                  "confirm,该确认无效", flush=True)
            _note += ";父代 best_params 缺失(子代=默认超参)"
        # [FIX-2026-10-11 / 已迁移] 曾经的 hpo 置空修复写在这里 —— 但这
        # 整块位于 `if cfg.get("hpo") and action != "hpo":` 之内,对 hpo
        # 动作**恒不可达**(gen-05 复核发现:新建 hpo trial 的 config.
        # hyperparameters 仍是父代 7 参,掩码未去)。正确位置是函数末尾
        # `hp = dict(hp_raw); hp.update(overrides)` 之后、`cfg.update(
        # {"hyperparameters": hp})` 之前的**平级**位置(见下方同名注释)。
        # 留在原地的旧语句已删除,避免"注释在、代码没跑"的假象。
        # 作用域仅限 hpo:confirm / exploit 等非 hpo 动作必须保留上面的
        # 继承(confirm 正是靠 config.hyperparameters 才能复现冠军配方)。
        # [v2.15.0] HPO 血缘的旧代码剥离 hpo 节后,二分类没有可执行路径
        # (脚手架 raise)、multi-seed 变体用默认超参且 eval_set=None ——
        # 在子代副本上补齐(见 _upgrade_hpo_lineage_code;父代不动)。
        _tp = dst_code / "train.py"
        try:
            _src_up, _up_notes = _upgrade_hpo_lineage_code(
                _tp.read_text(encoding="utf-8"))
            if _up_notes:
                _tp.write_text(_src_up, encoding="utf-8")
                _note += ";" + ";".join(_up_notes)
                print(f"[info] HPO 血缘子代代码已补齐: {_up_notes}")
        except OSError:
            pass
        cfg["notes"] = str(cfg.get("notes") or "") + _note
    hp_raw = cfg.get("hyperparameters")
    # baseline 的 hyperparameters 是说明字符串("baseline 固定默认参数"),
    # 快通道覆写时重置为空 dict 再注入
    hp = dict(hp_raw) if isinstance(hp_raw, dict) else {}
    hp.update(overrides)
    # [FIX-2026-10-11] **hpo 动作必须清空继承来的 hyperparameters**
    # (req-t7-rand400w run5 gen-03/gen-05 两次实测事故回写)。
    #
    # 父代若是 HPO trial,上面的 `action != "hpo"` 分支会把父代
    # metrics.best_params 视为"冠军重训配方"继承进 config.hyperparameters
    # —— 这对 confirm / exploit 是**正确且必需**的(confirm 正是靠它才能
    # 复现冠军配方)。但当 action == "hpo" 时,该分支被整段跳过,别名块
    # 由父 config 原样拷贝而来;运行期模板 hpo 分支
    # `_final_params = dict(_search_best); _final_params.update(_cfg_hp)`
    # 于是用父代配方**覆盖掉全部搜索维度** ⇒ 搜索赢家从未被训练,
    # metrics.best_params == hyperparameters_injected == 父代配方,而
    # hpo_search_best_params 是另一个点。同一 harness 下,父代 config 为
    # 空 dict 的 run 拿到真实 Δ=+0.00129 —— 差别只在父代 config 是否带
    # 别名块,故这是**静默**口径失效。
    #
    # v2.15.0 已为"父代是 HPO trial"的路径收口:`_inject_hpo_config`
    # 经 runtime.hpo.incumbent_params(best_params ← hyperparameters)把
    # 父代配方 **enqueue 为搜索的第一个评估点**,故清空 config.
    # hyperparameters 不会丢基准点 —— "父代配方在 inner 上得几分"仍可观测;
    # 且回归构造器取的是**算法默认**而非父代别名块(后者会让所有未在空间
    # 中出现的父代超参悄悄生效)。
    #
    # 位置要求:必须在上面 `hp = ...; hp.update(overrides)` **之后**
    # (否则被覆盖),且在下面 `cfg.update({"hyperparameters": hp})` 之前。
    # 第一版修复曾把本判断写进 `action != "hpo"` 分支内,对 hpo 恒不可达
    # (gen-05 复核发现并回写;真正生效的是 `--set` 快通道参数子代 ——
    # 那类子代**必须**保留覆写,故不能在更上层无差别清空)。
    if action == "hpo":
        _hp_parent = dict(hp)
        hp = {}
        if _hp_parent:
            # 显式 --set 覆写不静默丢弃:它是调用方的有意输入,保留下来
            # (会被 enqueue 基准点消费),而让**搜索空间**负责探索其余维度。
            _ov = {k: v for k, v in _hp_parent.items() if k in overrides}
            hp = _ov
            print(f"[info] hpo 动作:父代 {parent_id} 的别名块 "
                  f"{sorted(_hp_parent)} 已从 config.hyperparameters 清除,"
                  "避免 refit 段 _cfg_hp 覆盖搜索结果(基准点仍由 "
                  "hpo.enqueue 保留)", flush=True)
            if _ov:
                print(f"[warn] hpo 动作:显式 --set {sorted(_ov)} 已保留"
                      "(不再并入配置别名块,改由 enqueue 基准点消费)",
                      flush=True)
    # v19(req-t7-drawdown 实测事故回写):harness 级键从超参覆写中
    # 分流到 config 顶层——main() 读顶层 cfg.get("multi_seed"),而
    # --set 把覆写并进 hyperparameters 节,键位错位导致 multi_seed=3
    # 静默失效(t011/t012 实测单 seed;键被吞时无任何报错)。已知
    # harness 级键显式搬运 + 告警;未知键留在 hyperparameters(lint
    # 规则 11 的消费门兜底)。
    _harness_keys = ("multi_seed",)
    _moved = {}
    for _hk in _harness_keys:
        if _hk in hp:
            _moved[_hk] = hp.pop(_hk)
    if _moved:
        cfg.update(_moved)
        print(f"[warn] 快通道覆写中的 harness 级键 {_moved} 已写入 "
              "config 顶层(而非 hyperparameters;键位错位会使 "
              "main() 静默忽略——v19 修复)")
    # [FIX-2026-09-17] notes 必须保留上面写入的继承/告警留痕:
    # 旧实现下面这行 cfg.update 会整体覆写 notes,把 hpo 节剥离与
    # best_params 继承的说明冲掉(审计上看不出子代到底是"继承了冠军
    # 参数"还是"跑在默认超参上")。
    _prev_notes = str(cfg.get("notes") or "")
    # v2.12.2.7:真实算法 = trial.data.algorithm(trial.yaml 与 trial_id
    # 同源),缺失时退回命名里的算法段;绝不沿用父代 config 的值。
    _true_algo = str(trial.data.get("algorithm") or "").strip().lower()
    if not _true_algo:
        _tid_parts = str(trial.trial_id).split("-")
        _true_algo = _tid_parts[3] if len(_tid_parts) > 3 else ""
    cfg.update({
        "trial_id": trial.trial_id,
        "hyperparameters": hp,
        # v18:search_action 进 config(confirm 子代默认 multi_seed=3)
        "search_action": action,
        # v2.12.2.7:algorithm 必须写**本 trial 的真实算法**,不能沿用
        # 父代 config 里的值。旧实现整份 cfg 由父代拷贝而来,`--algorithm`
        # 只写进 trial.yaml/命名,config 里保留父代算法 ⇒ `trial_id`/
        # `trial.yaml` 声明 xgboost 而 config 写 lightgbm(实测
        # drawdown-t7-sv24 的 trial-030~034 五条 HPO 子代,父代是
        # lightgbm)。该字段无运行期消费点(实际算法由 trial.yaml 与
        # code 决定),但会误导按 config 复现的人,且随血缘一路传播。
        "algorithm": _true_algo,
        "notes": (f"参数 trial 快通道:代码复制自 {parent_id},"
                  f"仅超参覆写 {overrides}") + _prev_notes,
    })
    # 特征面白名单覆写(单源 config):父代 features(含 exclude)整体
    # 继承,columns 显式覆写;与超参同机制,代码零改动
    if features_override:
        feats_cfg = cfg.get("features")
        feats_cfg = dict(feats_cfg) if isinstance(feats_cfg, dict) else {}
        feats_cfg["columns"] = list(features_override)
        cfg["features"] = feats_cfg
        cfg["notes"] = (str(cfg.get("notes") or "")
                        + f";特征面白名单覆写 {len(features_override)} 列")
    from runtime.logging_utils import dump_yaml

    dump_yaml(Path(trial.trial_dir) / "config" / "trial.yaml", cfg)
    trial.data["execution"] = {
        "action_description": [f"参数覆写: {overrides}(代码=父代 {parent_id})"],
        "code_entry": "train.py",
        "command": "python train.py --config ../config/trial.yaml",
        "changed_modules": [],
    }
    trial.flush()


def _rollback_trial(trial) -> None:
    """[FIX-2026-09-03] new-trial 失败事务化:删除本次创建的孤儿目录。

    旧实现 scaffold 落盘(factory.create)先于父代码校验/快通道注入,
    _assert_hpo_capable 等中途 SystemExit 后目录残留——孤儿 trial
    伪装成"半成品"(root 标 sample / config 抄父代 v1 / 无 hpo 节),
    后续盘点误判为设计缺陷并触发契约外 rm -rf(drawdown-t7 gen-03
    实测:trial-005/006 两个孤儿 + 编号空洞)。

    安全边界:仅删 status==PROPOSED 的目录(创建失败路径上不可能
    越过 PROPOSED);删除失败只告警不掩盖原异常。
    """
    import shutil as _shutil_r

    try:
        from runtime.trial import TrialRecord as _TR_r

        rec = _TR_r.load(Path(trial.trial_dir))
        if rec.status != "PROPOSED":
            logger.warning(
                "回滚跳过:%s 状态 %s ≠ PROPOSED(可能有并发写入,人工核查)",
                trial.trial_id, rec.status)
            return
        _shutil_r.rmtree(Path(trial.trial_dir), ignore_errors=False)
        logger.info("已回滚本次创建的 trial 目录(失败事务清理): %s",
                    trial.trial_dir)
    except Exception as exc:  # noqa: BLE001  回滚失败不掩盖原异常
        logger.warning("孤儿目录回滚失败(需人工删除 %s): %s",
                       trial.trial_dir, exc)


# ===========================================================================
# v2.17.0(run4 regression.2 #7):回归单分支渲染(子代副本 / 嫁接段)
# ===========================================================================
def _narrow_dispatch_text(code: str, model_type: str | None = None
                          ) -> tuple[str, list[str]]:
    """runtime.templates.narrow_model_type_dispatch 的薄包装(失败不阻断)。"""
    try:
        from runtime.templates import narrow_model_type_dispatch

        return narrow_model_type_dispatch(code, model_type)
    except Exception as exc:  # noqa: BLE001  折叠是优化项,失败保持原文
        logger.warning("回归单分支渲染跳过(%s: %s)", type(exc).__name__, exc)
        return code, []


def _narrow_dispatch_file(train_py: Path, model_type: str | None = None) -> int:
    """就地折叠 train.py 的打分分派;返回改写处数(0 = 未改)。"""
    try:
        text = Path(train_py).read_text(encoding="utf-8")
    except OSError:
        return 0
    new, notes = _narrow_dispatch_text(text, model_type)
    if notes and new != text:
        compile(new, str(train_py), "exec")
        Path(train_py).write_text(new, encoding="utf-8")
    return len(notes)


# ===========================================================================
# v2.17.0(#16):new-trial --add-blocks <trial>:<block>[,...]
# ===========================================================================
# run4 regression.2 每代用 adhoc/build_gen0x_trials.py 按行区间截取源 trial 的
# feature_block 内核段、锚点校验后拼进新 trial,再用 cachecheck_*.py 人工确认
# 缓存命中。这里收编为确定性实现:
#   1. 解析源 trial 冻结的 code/train.py,定位块包装函数(块名 = feature_block
#      的第一个参数、包装函数名或 `_<块>_block`);
#   2. 依赖闭包:包装函数 → 引用的模块级名字(函数/类/常量/import,含对常量
#      的顶层原地修改)递归求闭包,按源文件顺序取语句;
#   3. 新 trial 已有 AST 等价的同名定义 → 跳过;已有不同定义 → 保留新 trial
#      的、记冲突;其余**字节级**复制(含紧邻的上方注释),放进 BEGIN/END 段;
#   4. 段尾包一层 trial_feature_blocks(原有块 + 追加块,按列右接,重名报错);
#   5. 创建期指纹校验:源/新 trial 各起一个探针进程(cwd=code/),拦截
#      feature_block 调用,用 runtime.feature_cache.code_fingerprint 算内核指纹,
#      块名/列/源列/平面/version/batch_format 一并比对;不一致拒绝并给差异。
ADD_BLOCKS_BEGIN = "# ==== v2.17.0 add-blocks 段 BEGIN"
ADD_BLOCKS_END = "# ==== v2.17.0 add-blocks 段 END ===="


def parse_add_blocks(spec: str | None) -> list[tuple[str, str]]:
    """`trial-013:seqd_k0r,trial-018:amt_v1` → [(trial, block), ...](去重保序)。"""
    out: list[tuple[str, str]] = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"--add-blocks 项 {part!r} 格式应为 <trial_id>:<block>")
        tid, blk = (s.strip() for s in part.split(":", 1))
        if not tid or not blk:
            raise SystemExit(f"--add-blocks 项 {part!r} 格式应为 <trial_id>:<block>")
        if (tid, blk) not in out:
            out.append((tid, blk))
    return out


def _fb_calls(fn_node) -> list:
    import ast as _ast

    return [n for n in _ast.walk(fn_node) if isinstance(n, _ast.Call) and (
        (isinstance(n.func, _ast.Name) and n.func.id == "feature_block") or
        (isinstance(n.func, _ast.Attribute) and n.func.attr == "feature_block"))]


def block_wrappers(code: str) -> dict:
    """{包装函数名: feature_block 块名或 None}:顶层函数里调用 feature_block 的。"""
    import ast as _ast

    out = {}
    for n in _ast.parse(code).body:
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and \
                n.name != "trial_feature_blocks":
            calls = _fb_calls(n)
            if calls:
                a0 = calls[0].args[0] if calls[0].args else None
                out[n.name] = a0.value if isinstance(a0, _ast.Constant) else None
    return out


def resolve_block_wrapper(code: str, block: str) -> str:
    """块标识 → 包装函数名(块名 / 函数名 / `_<块>_block` / 块名前缀,唯一匹配)。"""
    wr = block_wrappers(code)
    if block in wr:
        return block
    hits = [w for w, nm in wr.items() if nm == block]
    if not hits:
        hits = [w for w in wr if w in (f"_{block}_block", f"{block}_block")]
    if not hits:
        hits = [w for w, nm in wr.items()
                if nm and str(nm).split("_v")[0].split("_k")[0] == block]
    if len(hits) != 1:
        avail = ", ".join(f"{w}({nm})" for w, nm in sorted(wr.items())) or "无"
        raise SystemExit(
            f"--add-blocks:块 {block!r} 在源代码里{'不唯一' if hits else '找不到'}"
            f"(可用:包装函数(块名) = {avail})")
    return hits[0]


def _ab_bound_names(stmt) -> set:
    import ast as _ast

    out: set = set()
    if isinstance(stmt, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, (_ast.Import, _ast.ImportFrom)):
        for a in stmt.names:
            out.add((a.asname or a.name).split(".")[0])
        return out
    if isinstance(stmt, _ast.Expr) and isinstance(stmt.value, _ast.Call) and \
            isinstance(stmt.value.func, _ast.Attribute) and \
            isinstance(stmt.value.func.value, _ast.Name):
        return {stmt.value.func.value.id}            # X.append(...) 原地修改
    # 只收模块作用域的绑定:推导式/lambda/嵌套 def 体内的名字不算(推导式
    # 变量 i/k/w 会把无关常量误判成同名冲突)
    stack = [stmt]
    while stack:
        n = stack.pop()
        if isinstance(n, _ast.Name) and isinstance(n.ctx, (_ast.Store, _ast.Del)):
            out.add(n.id)
            continue
        if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                          _ast.ClassDef)):
            out.add(n.name)
            continue
        if isinstance(n, (_ast.Import, _ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
            continue
        if isinstance(n, (_ast.Lambda, _ast.ListComp, _ast.SetComp,
                          _ast.DictComp, _ast.GeneratorExp)):
            continue
        stack.extend(_ast.iter_child_nodes(n))
    return out


def _referenced_names(stmt) -> set:
    """语句引用的(非局部)名字;函数体内的参数/局部赋值名不算。"""
    import ast as _ast

    loads = {n.id for n in _ast.walk(stmt)
             if isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Load)}
    if isinstance(stmt, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        local = set()
        glob = set()
        for n in _ast.walk(stmt):
            if isinstance(n, _ast.arg):
                local.add(n.arg)
            elif isinstance(n, _ast.Name) and isinstance(n.ctx, _ast.Store):
                local.add(n.id)
            elif isinstance(n, (_ast.Global, _ast.Nonlocal)):
                glob.update(n.names)
        loads -= (local - glob)
    return loads


def block_dependency_closure(code: str, wrapper: str) -> list:
    """包装函数及其依赖的顶层语句(源文件顺序)。"""
    import ast as _ast

    tree = _ast.parse(code)
    body = tree.body
    binds: dict[str, list[int]] = {}
    for i, st in enumerate(body):
        if isinstance(st, _ast.If) and isinstance(st.test, _ast.Compare) and \
                isinstance(st.test.left, _ast.Name) and \
                st.test.left.id == "__name__":
            continue                                  # main 守卫
        for nm in _ab_bound_names(st):
            binds.setdefault(nm, []).append(i)
    if wrapper not in binds:
        raise SystemExit(f"--add-blocks:源代码无顶层函数 {wrapper}")
    picked: set[int] = set()
    stack = list(binds[wrapper])
    while stack:
        i = stack.pop()
        if i in picked:
            continue
        picked.add(i)
        for nm in _referenced_names(body[i]):
            for j in binds.get(nm, ()):
                if j not in picked:
                    stack.append(j)
    return [body[i] for i in sorted(picked)]


def _stmt_source(lines: list[str], stmt) -> str:
    """语句原文(含装饰器与紧邻上方的注释行),字节级。"""
    lo = min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list",
                                                          [])]) - 1
    while lo > 0 and lines[lo - 1].lstrip().startswith("#") and \
            not lines[lo - 1].startswith("# ===="):
        lo -= 1
    return "".join(lines[lo:stmt.end_lineno])


def _toplevel_defs(code: str) -> dict:
    """{名字: [语句 ast.dump]}(新 trial 已有定义的等价性判定)。"""
    import ast as _ast

    out: dict[str, list[str]] = {}
    for st in _ast.parse(code).body:
        for nm in _ab_bound_names(st):
            out.setdefault(nm, []).append(_ast.dump(st))
    return out


def _registered_wrappers(code: str) -> set:
    """新 trial 的 trial_feature_blocks 已注册的块包装函数(直接调用或经
    add-blocks 注册表 `_ADDED_FEATURE_BLOCKS_*`)。"""
    import ast as _ast

    tree = _ast.parse(code)
    out: set = set()
    for f in tree.body:
        if isinstance(f, _ast.FunctionDef) and f.name == "trial_feature_blocks":
            out |= {n.id for n in _ast.walk(f) if isinstance(n, _ast.Name)
                    and isinstance(n.ctx, _ast.Load)}
        elif isinstance(f, _ast.Assign) and any(
                isinstance(t, _ast.Name) and t.id.startswith(
                    "_ADDED_FEATURE_BLOCKS_") for t in f.targets):
            out |= {x.id for x in _ast.walk(f.value)
                    if isinstance(x, _ast.Name)}
    return out


def assemble_add_blocks(dst_code: str, sources: list[dict]) -> tuple[str, dict]:
    """把各源 trial 的块段拼进 dst_code。

    sources: [{"trial": id, "block": 标识, "code": 源 train.py 文本}]。
    返回 (新代码, 报告 {blocks:[...], copied:[名字], skipped_identical:[...],
    conflicts:[{name, source_trial}]})。
    """
    import ast as _ast

    report = {"blocks": [], "copied": [], "skipped_identical": [],
              "conflicts": []}
    have = _toplevel_defs(dst_code)
    already = _registered_wrappers(dst_code)
    chunks: list[str] = []
    copied_dump: set[str] = set()
    registered: list[tuple[str, str]] = []
    for src in sources:
        code = src["code"]
        lines = code.splitlines(keepends=True)
        wrapper = resolve_block_wrapper(code, src["block"])
        if wrapper in already and wrapper in have:
            raise SystemExit(
                f"--add-blocks:块 {src['trial']}:{src['block']}(包装函数 "
                f"{wrapper})已在新 trial 的 trial_feature_blocks 中注册,"
                "重复追加会产生重名列")
        fb_name = block_wrappers(code).get(wrapper)
        stmts = block_dependency_closure(code, wrapper)
        n_copy = 0
        for st in stmts:
            names = _ab_bound_names(st)
            dump = _ast.dump(st)
            if dump in copied_dump:
                continue
            existing = [d for nm in names for d in have.get(nm, [])]
            if existing and dump in existing:
                report["skipped_identical"].append(sorted(names)[0])
                continue
            if existing:
                report["conflicts"].append({
                    "names": sorted(names), "source_trial": src["trial"],
                    "source_text": _stmt_source(lines, st)[:600]})
                continue
            chunks.append(_stmt_source(lines, st).rstrip("\n"))
            copied_dump.add(dump)
            report["copied"].extend(sorted(names))
            for nm in names:
                have.setdefault(nm, []).append(dump)
            n_copy += 1
        tag = f"{src['trial']}:{src['block']}"
        if wrapper not in [w for _, w in registered]:
            registered.append((tag, wrapper))
        report["blocks"].append({"source_trial": src["trial"],
                                 "block": src["block"], "wrapper": wrapper,
                                 "feature_block_name": fb_name,
                                 "n_statements_copied": n_copy})
    k = dst_code.count(ADD_BLOCKS_BEGIN) + 1
    pre = "trial_feature_blocks" if "trial_feature_blocks" in have else "None"
    tags = ", ".join(t for t, _ in registered)
    reg = "\n".join(f"    ({t!r}, {w})," for t, w in registered)
    seg = (
        f"{ADD_BLOCKS_BEGIN}(harness 字节级复制 {tags};勿手改 —— 改动会使"
        "特征块缓存键失配)====\n"
        + ("\n\n\n".join(chunks) + "\n\n\n" if chunks else "")
        + f"_tfb_pre_add_{k} = {pre}\n"
        f"_ADDED_FEATURE_BLOCKS_{k} = (\n{reg}\n)\n\n\n"
        "def trial_feature_blocks(split: str) -> \"pd.DataFrame\":\n"
        f"    \"\"\"v2.17.0 add-blocks:原有块(若有)+ 追加块({tags}),按列右接。\"\"\"\n"
        "    import pandas as _pd_ab\n"
        "    parts = []\n"
        f"    if _tfb_pre_add_{k} is not None:\n"
        f"        parts.append(_tfb_pre_add_{k}(split).reset_index(drop=True))\n"
        f"    for _tag, _fn in _ADDED_FEATURE_BLOCKS_{k}:\n"
        "        parts.append(_fn(split).reset_index(drop=True))\n"
        "    W = _pd_ab.concat(parts, axis=1)\n"
        "    _dup = W.columns[W.columns.duplicated()].tolist()\n"
        "    if _dup:\n"
        "        raise ValueError(f\"add-blocks 特征块列重名: {_dup[:5]}\")\n"
        "    return W.astype(\"float32\")\n"
        f"{ADD_BLOCKS_END}\n")
    out = _insert_before_main(dst_code, seg)
    _ast.parse(out)
    return out, report


_BLOCK_PROBE_SRC = r'''
import importlib.util, json, pathlib, sys
names = json.loads(sys.argv[1])
p = pathlib.Path("train.py")
spec = importlib.util.spec_from_file_location("_mvo_block_probe", p)
mod = importlib.util.module_from_spec(spec)
sys.modules["_mvo_block_probe"] = mod
spec.loader.exec_module(mod)
import runtime.feature_cache as fc
calls = []
class _Stop(Exception):
    pass
def _rec(name, fn, **kw):
    calls.append((name, fn, kw))
    raise _Stop()
mod.feature_block = _rec
_orig = fc.feature_block
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
    if not calls:
        out[w] = {"error": "包装函数未调用 feature_block"}
        continue
    name, fn, kw = calls[0]
    rec = {"name": name, "code_fingerprint": fc.code_fingerprint(fn),
           "fn": "%s" % getattr(fn, "__qualname__", repr(fn)),
           "columns_sha": __import__("hashlib").sha256(
               json.dumps(list(kw.get("columns") or [])).encode()).hexdigest()[:16],
           "n_columns": len(list(kw.get("columns") or [])),
           "sources": list(kw.get("sources") or []),
           "plane": kw.get("plane", "reparse"),
           "version": kw.get("version", ""),
           "batch_format": kw.get("batch_format", "pylist")}
    try:
        sd = pathlib.Path(kw.get("splits_dir"))
        src = fc.resolve_source_path(sd, "train", rec["plane"])
        sfp = fc.source_fingerprint(src, rec["sources"])
        key = fc.block_key(name, fn, split="train",
                           columns=list(kw.get("columns") or []),
                           sources=rec["sources"], source_fp=sfp,
                           plane=rec["plane"], version=rec["version"],
                           batch_format=rec["batch_format"])
        rec["train_block_key"] = key
        rd = fc._infer_run_dir(sd)
        rec["train_cache_present"] = (rd / fc.CACHE_DIRNAME /
                                      ("%s-%s" % (fc._safe(name), key)) /
                                      "train.parquet").exists()
    except Exception as e:
        rec["train_block_key"] = None
        rec["key_note"] = "%s: %s" % (type(e).__name__, e)
    out[w] = rec
print("BLOCK_PROBE_JSON " + json.dumps(out, ensure_ascii=False))
'''

#: 指纹比对的字段(缓存键里与代码相关的全部组成)
_BLOCK_FP_FIELDS = ("name", "code_fingerprint", "columns_sha", "sources",
                    "plane", "version", "batch_format")


def probe_block_fingerprints(code_dir: Path, wrappers: list[str]) -> dict:
    """cwd=code/ 起探针进程,返回 {包装函数: 指纹记录}(失败抛 SystemExit)。"""
    import subprocess as _sp

    r = _sp.run([sys.executable, "-c", _BLOCK_PROBE_SRC, json.dumps(wrappers)],
                cwd=str(code_dir), capture_output=True, text=True, timeout=300)
    line = next((ln for ln in r.stdout.splitlines()
                 if ln.startswith("BLOCK_PROBE_JSON ")), None)
    if line is None:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        raise SystemExit(f"--add-blocks 指纹探针失败({code_dir}):"
                         + " | ".join(tail))
    return json.loads(line[len("BLOCK_PROBE_JSON "):])


def apply_add_blocks(run_dir: Path, trial, spec: str) -> dict:
    """new-trial --add-blocks 主流程;失败抛 SystemExit(调用方回滚)。"""
    import difflib

    items = parse_add_blocks(spec)
    if not items:
        return {}
    tp = Path(trial.trial_dir) / "code" / "train.py"
    dst = tp.read_text(encoding="utf-8")
    sources, by_trial = [], {}
    for tid, blk in items:
        sdir = Path(run_dir) / "trials" / tid
        sp = sdir / "code" / "train.py"
        if not sp.exists():
            raise SystemExit(f"--add-blocks:源 trial {tid} 无 code/train.py")
        if (sdir / "source-manifest.json").exists():
            try:
                from runtime.integrity import verify_trial_source

                verify_trial_source(sdir)
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(f"--add-blocks:源 trial {tid} 的 code/ 与冻结"
                                 f"清单不一致({exc}),拒绝复制被改动的源码")
        code = by_trial.setdefault(tid, sp.read_text(encoding="utf-8"))
        sources.append({"trial": tid, "block": blk, "code": code})
    new_code, rep = assemble_add_blocks(dst, sources)
    tp.write_text(new_code, encoding="utf-8")
    try:
        return _verify_add_blocks(run_dir, tp, new_code, by_trial, rep)
    except BaseException:
        tp.write_text(dst, encoding="utf-8")      # 失败恢复原代码(调用方另行回滚)
        raise


def _verify_add_blocks(run_dir: Path, tp: Path, new_code: str, by_trial: dict,
                       rep: dict) -> dict:
    import difflib

    # 指纹校验:源 trial(冻结代码)vs 新 trial
    checks = []
    for tid in by_trial:
        wrs = [b["wrapper"] for b in rep["blocks"] if b["source_trial"] == tid]
        src_fp = probe_block_fingerprints(Path(run_dir) / "trials" / tid / "code",
                                          wrs)
        dst_fp = probe_block_fingerprints(tp.parent, wrs)
        for w in wrs:
            a, b = src_fp.get(w) or {}, dst_fp.get(w) or {}
            diff = {f: {"source": a.get(f), "new": b.get(f)}
                    for f in _BLOCK_FP_FIELDS if a.get(f) != b.get(f)}
            if a.get("error") or b.get("error"):
                diff["error"] = {"source": a.get("error"), "new": b.get("error")}
            checks.append({"source_trial": tid, "wrapper": w,
                           "feature_block_name": a.get("name"),
                           "code_fingerprint": a.get("code_fingerprint"),
                           "n_columns": a.get("n_columns"),
                           "train_block_key": b.get("train_block_key"),
                           "cache_present": b.get("train_cache_present"),
                           "match": not diff, "diff": diff})
    bad = [c for c in checks if not c["match"]]
    rep["fingerprints"] = checks
    if bad:
        detail = []
        for c in bad:
            detail.append(f"{c['source_trial']}:{c['wrapper']} 差异 "
                          f"{json.dumps(c['diff'], ensure_ascii=False)[:400]}")
        for cf in rep["conflicts"][:4]:
            have = _toplevel_src_of(new_code, cf["names"])
            ud = "".join(difflib.unified_diff(
                cf["source_text"].splitlines(True), have.splitlines(True),
                fromfile=f"{cf['source_trial']}(源)", tofile="新 trial",
                n=1))[:800]
            detail.append(f"同名定义冲突 {cf['names']}:\n{ud}")
        raise SystemExit(
            "--add-blocks 指纹校验失败(新 trial 的块内核指纹 ≠ 源 trial,"
            "特征块缓存不会命中,且块语义可能已变):\n  - "
            + "\n  - ".join(detail)
            + "\n处置:新 trial 里与块依赖同名的定义和源不一致时,换一个不含"
              "冲突定义的父代 / 先对齐同名定义,或给块改名(version=)后按新块重算")
    return rep


def _toplevel_src_of(code: str, names) -> str:
    import ast as _ast

    lines = code.splitlines(keepends=True)
    out = []
    for st in _ast.parse(code).body:
        if _ab_bound_names(st) & set(names):
            out.append(_stmt_source(lines, st))
    return "".join(out)


# ===========================================================================
# v2.17.0(#5 第 4 点):structural 拓扑 trial 的交付契约自检(只告警不拦截)
# ===========================================================================
_DELIVERABILITY_PROBE_SRC = r'''
import importlib.util, json, pathlib, sys
mt = sys.argv[1]
p = pathlib.Path("train.py")
spec = importlib.util.spec_from_file_location("_mvo_deliv_probe", p)
mod = importlib.util.module_from_spec(spec)
sys.modules["_mvo_deliv_probe"] = mod
out = {"import_ok": True, "issues": []}
try:
    spec.loader.exec_module(mod)
except Exception as e:
    out["import_ok"] = False
    out["issues"].append("模块导入失败 %s: %s" % (type(e).__name__, e))
    print("DELIV_JSON " + json.dumps(out, ensure_ascii=False)); sys.exit(0)
from sklearn.base import clone
out["fit_and_score"] = callable(getattr(mod, "fit_and_score", None))
out["has_last_model"] = hasattr(mod, "_last_model")
container = None
clone_ok, clone_detail = None, None
he = getattr(mod, "_hurdle_estimators", None)
H = getattr(mod, "HurdleRegressor", None)
if callable(he):
    try:
        segs = he(1, 0, 0)
        for s in segs:
            clone(s)
        if H is not None:
            container = H(clf=segs[0], reg=segs[1])
            clone(container)
            clone_ok, clone_detail = True, "runtime.hurdle.HurdleRegressor 可 clone"
    except Exception as e:
        clone_ok, clone_detail = False, "%s: %s" % (type(e).__name__, e)
classes = []
import inspect
for nm, obj in vars(mod).items():
    if inspect.isclass(obj) and getattr(obj, "__module__", "") == mod.__name__ \
            and callable(getattr(obj, "predict", None)):
        classes.append(obj)
for c in classes:
    info = {"name": c.__name__, "sklearn": hasattr(c, "get_params"),
            "predict_proba": hasattr(c, "predict_proba")}
    out.setdefault("containers", []).append(info)
    if clone_ok is None:
        clone_ok = bool(info["sklearn"])
        clone_detail = ("自定义容器 %s %s" % (c.__name__, "继承 sklearn 协议" if
                        info["sklearn"] else "不是 sklearn 估计器(无 get_params),"
                        "交付 frozen_rounds 的 clone 会失败"))
    if mt == "regression" and info["predict_proba"]:
        out["issues"].append("回归容器 %s 暴露 predict_proba(交付/打分可能误走概率口径)"
                             % c.__name__)
if H is not None and mt == "regression" and hasattr(H, "predict_proba"):
    out["issues"].append("HurdleRegressor 暴露 predict_proba")
out["clone_ok"] = clone_ok
out["clone_detail"] = clone_detail
print("DELIV_JSON " + json.dumps(out, ensure_ascii=False))
'''


def deliverability_probe(trial_dir: Path, model_type: str | None) -> dict:
    """structural trial 的交付契约自检(导入探针 + 静态检查;不训练)。

    返回 trial.yaml.deliverability:{checked_at, clone_ok, clone_detail,
    rerun_recipe_ok, deliverable, preferred_refit, issues, containers}。
    clone_ok:模型容器可 sklearn.clone(frozen_rounds 可用);rerun_recipe_ok:
    fit_and_score 在位且被 main 调用、fit_and_score 写 _last_model、容器可
    predict、回归容器不暴露 predict_proba。
    """
    import ast as _ast
    import subprocess as _sp

    code_dir = Path(trial_dir) / "code"
    res: dict = {"checked_at": now_iso(), "issues": []}
    try:
        code = (code_dir / "train.py").read_text(encoding="utf-8")
        tree = _ast.parse(code)
    except (OSError, SyntaxError) as exc:
        res.update(deliverable=False, clone_ok=None, rerun_recipe_ok=False,
                   issues=[f"代码不可读/语法错误:{exc}"])
        return res
    try:
        r = _sp.run([sys.executable, "-c", _DELIVERABILITY_PROBE_SRC,
                     str(model_type or "")], cwd=str(code_dir),
                    capture_output=True, text=True, timeout=180)
        line = next((ln for ln in r.stdout.splitlines()
                     if ln.startswith("DELIV_JSON ")), None)
        dyn = json.loads(line[len("DELIV_JSON "):]) if line else {
            "import_ok": False, "issues": [
                "探针无输出:" + " | ".join((r.stderr or "").strip()
                                          .splitlines()[-2:])]}
    except Exception as exc:  # noqa: BLE001
        dyn = {"import_ok": False, "issues": [f"探针异常:{exc}"]}
    res["issues"].extend(dyn.get("issues") or [])
    fs = [n for n in tree.body if isinstance(n, _ast.FunctionDef)
          and n.name == "fit_and_score"]
    sets_last = any(isinstance(g, _ast.Global) and "_last_model" in g.names
                    for f in fs for g in _ast.walk(f))
    main_calls = _main_calls_name(tree, "fit_and_score")
    if not fs:
        res["issues"].append("缺 fit_and_score(rerun_recipe 入口)")
    if fs and not main_calls:
        res["issues"].append("main() 不调用 fit_and_score(交付复跑与搜索路径不同源)")
    if fs and not sets_last:
        res["issues"].append("fit_and_score 未写 _last_model(交付取不到模型对象)")
    proba_issue = any("predict_proba" in i for i in res["issues"])
    rr_ok = bool(dyn.get("import_ok") and fs and main_calls and sets_last
                 and not proba_issue)
    clone_ok = dyn.get("clone_ok")
    res.update({
        "import_ok": bool(dyn.get("import_ok")),
        "clone_ok": clone_ok, "clone_detail": dyn.get("clone_detail"),
        "containers": dyn.get("containers") or [],
        "rerun_recipe_ok": rr_ok,
        "deliverable": bool(rr_ok or (clone_ok and not proba_issue)),
        "preferred_refit": ("frozen_rounds" if clone_ok and not proba_issue
                            else ("rerun_recipe" if rr_ok else None)),
    })
    return res
