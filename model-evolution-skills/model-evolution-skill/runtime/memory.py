"""Experience & Memory:run 内两层经验(§14)。

Level 1 Evidence       :原始 Trial/Generation 事实(Source of Truth);
Level 2 Task Experience:当前任务压缩总结,Planner 高频读取;
另含收尾导出文件 ``experience/exported-global-experiences.yaml`` 的逐条校验。

跨任务经验(原 Level 3,skill 内置 memory/ 全局库)自 v2.18.0 起由中心化经验库
承担:``runtime/library/``(存储与服务)与 ``runtime/library_view.py``(run 侧视图)。
经验必须 Evidence-based:Claim + Context + Evidence + Boundary(§2.3)。
"""

from __future__ import annotations

from pathlib import Path

from .logging_utils import (append_jsonl, dump_yaml, load_yaml, now_iso,
                            read_jsonl, get_logger)
from .spec import TaskSpec
from .trial import TrialRecord

logger = get_logger(__name__)

#: 原则 category 标准枚举(v2.15.10 显式化;references/experience-and-memory.md)
STANDARD_CATEGORIES = frozenset({
    "feature_engineering", "architecture", "hpo", "evaluation",
    "top_trial_baseline"})


# ---------------------------------------------------------------------------
# v2.17.0:经验口径(target_transform / primary_metric)
# ---------------------------------------------------------------------------
# 事故(run4 regression.2 #18):同一数据底座、同一业务目标先后有 log1p 口径
# (target=loan_amt_t7_log1p,rmse≈2.7)与原值口径(target=loan_amt_t7,
# rmse≈12932)两个回归 run。本 run 的 seqg_ 经验(原值口径无独立增益)与库内
# exp-202610-ef8acd28(log1p 口径有增益)被判"冲突";planner-context 7c 又把
# log1p run 的 rmse 2.72 当"同任务历史 top trial"展示。两者口径不同,结论
# 不可直接比较。导出时把口径写进 condition,冲突检测/检索/展示据此分流。

#: 简写经验字段的枚举(export-experience 逐条校验)
EXPERIENCE_OUTCOMES = ("positive", "negative", "neutral")
EXPERIENCE_ACTION_TYPES = ("add", "avoid")
EXPERIENCE_CONFIDENCES = ("high", "medium", "low")

#: 目标列名后缀 → 变换口径(按后缀长度优先匹配)
_TRANSFORM_SUFFIXES = (("_log1p", "log1p"), ("_log", "log"),
                       ("_sqrt", "sqrt"), ("_boxcox", "boxcox"),
                       ("_rank", "rank"))


def _spec_dict(spec_like) -> dict:
    if isinstance(spec_like, TaskSpec):
        return spec_like.data or {}
    if isinstance(spec_like, dict):
        return spec_like
    return getattr(spec_like, "data", None) or {}


def infer_target_transform(spec_like) -> str:
    """由 spec 推断目标口径(v2.17.0):``raw`` / ``log1p`` / ``log`` …

    优先级:spec 显式声明(``target.transform`` / ``target.target_transform``
    / 顶层 ``target_transform``)> 目标列名后缀(``*_log1p`` → log1p)>
    ``data.derived_columns`` 声明了目标列(loader 派生的标签:按列名推断,
    推断不出记 ``derived``)> ``raw``。
    """
    d = _spec_dict(spec_like)
    tgt = d.get("target") if isinstance(d.get("target"), dict) else {}
    for v in (tgt.get("transform"), tgt.get("target_transform"),
              d.get("target_transform")):
        if v:
            return str(v).strip().lower()
    name = str(tgt.get("name") or "").strip().lower()
    for suf, tt in _TRANSFORM_SUFFIXES:
        if name.endswith(suf):
            return tt
    derived = (d.get("data") or {}).get("derived_columns") \
        if isinstance(d.get("data"), dict) else None
    if isinstance(derived, list) and name and \
            name in {str(c).strip().lower() for c in derived}:
        for suf, tt in _TRANSFORM_SUFFIXES:
            if suf.strip("_") in name:
                return tt
        return "derived"
    return "raw"


def _yaml_item_lines(text: str) -> list[int | None]:
    """导出文件里每个条目的起始行号(1-based;解析不出返回空表)。"""
    try:
        import yaml as _yaml

        node = _yaml.compose(text)
    except Exception:  # noqa: BLE001
        return []
    if node is None:
        return []
    seq = None
    if isinstance(node, _yaml.SequenceNode):
        seq = node
    elif isinstance(node, _yaml.MappingNode):
        for k, v in node.value:
            if getattr(k, "value", None) == "experiences" and \
                    isinstance(v, _yaml.SequenceNode):
                seq = v
                break
    if seq is None:
        return []
    return [n.start_mark.line + 1 for n in seq.value]


#: v2.18.4:YAML 解析失败时最容易踩的形态 —— 文本里带裸冒号
#: (`summary: 这是 extra: 未加引号`)。PyYAML 把第二个冒号当映射分隔符。
_YAML_COLON_HINT = (
    "若报 mapping values are not allowed here:该行文本里含未加引号的冒号,"
    "把整段用引号包起来或改写成 `|`/`>` 块标量")

EXPORT_FIX_HINT = (
    "修复 experience/exported-global-experiences.yaml 后重跑 "
    "`python <skill>/scripts/modelevo.py export-experience --run <run_dir>`:"
    "顶层写成列表(`- claim: ...`)或 `experiences:` 包装的列表;每条必须是"
    "mapping 且含非空 claim(或 statement);outcome ∈ positive/negative/"
    "neutral;action_type ∈ add/avoid;confidence ∈ high/medium/low;"
    "context/evidence 可为字符串(自动规范化);condition 若写必须是 mapping。")


def validate_experience_item(item: object, index: int,
                             line: int | None = None) -> tuple[list, list]:
    """单条导出经验的 schema 校验 → (errors, warnings)。

    每条错误含 index(1-based 条目序号)/ line / field / error / fix。
    """
    errors: list[dict] = []
    warnings: list[dict] = []

    def _e(field, msg, fix):
        errors.append({"index": index, "line": line, "field": field,
                       "error": msg, "fix": fix})

    def _w(field, msg):
        warnings.append({"index": index, "line": line, "field": field,
                         "warning": msg})

    if not isinstance(item, dict):
        _e(None, f"条目必须是 mapping(当前 {type(item).__name__}:"
                 f"{str(item)[:60]!r})",
           "每个条目写成 `- claim: ...` 形式的字典;若顶层是 mapping,"
           "经验列表应放在 `experiences:` 键下")
        return errors, warnings
    claim = item.get("claim", item.get("statement"))
    if not isinstance(claim, str) or not claim.strip():
        _e("claim", "缺少非空字符串 claim(或旧字段 statement)",
           "补一句可证伪的结论,如 `claim: X 方向在原值 rmse 口径下无独立增益`")
    for field, allowed in (("outcome", EXPERIENCE_OUTCOMES),
                           ("action_type", EXPERIENCE_ACTION_TYPES),
                           ("confidence", EXPERIENCE_CONFIDENCES)):
        if field in item and item[field] is not None and \
                str(item[field]) not in allowed:
            _e(field, f"取值 {item[field]!r} 不在枚举 {list(allowed)}",
               f"把 {field} 改成 {' / '.join(allowed)} 之一")
    if "do_not_retry" in item and \
            not isinstance(item["do_not_retry"], bool):
        _e("do_not_retry", f"必须是布尔值(当前 {item['do_not_retry']!r})",
           "写 true 或 false")
    if "condition" in item and item["condition"] is not None and \
            not isinstance(item["condition"], dict):
        _e("condition", f"必须是 mapping(当前 "
                        f"{type(item['condition']).__name__})",
           "写成 `condition: {model_type: regression, ...}`;自由文本请放 "
           "context")
    if "category" in item and not isinstance(item["category"], str):
        _e("category", f"必须是字符串(当前 {type(item['category']).__name__})",
           f"取 {sorted(STANDARD_CATEGORIES)} 之一")
    for field in ("context", "evidence"):
        v = item.get(field)
        if v is not None and not isinstance(v, (str, dict)):
            _w(field, f"{field} 为 {type(v).__name__},已按文本规范化")
    if item.get("evidence") in (None, "", {}):
        _w("evidence", "缺少 evidence(建议写 trial id、Δ 与 CI)")
    return errors, warnings


def load_experience_export(src: Path) -> dict:
    """读并逐条校验 exported-global-experiences.yaml(v2.17.0,#9)。

    接受三种顶层写法:列表;``{experiences: [...]}`` 包装;单条 mapping
    (含 claim/statement)。旧实现把 ``experiences:`` 包装当列表迭代,拿到
    的是键名字符串,validate 里 ``'str'.get`` 抛 AttributeError,导出整体
    崩溃且 diagnose 无 finding(实测 run4 regression.2)。

    返回 ``{items: [(index, line, item)], errors, warnings, structure,
    fix_hint}``——items 只含通过校验的条目(已做 context/evidence 文本规范化
    之前的原样 dict),错误条目逐条带序号与字段名。
    """
    out: dict = {"items": [], "errors": [], "warnings": [],
                 "structure": None, "fix_hint": EXPORT_FIX_HINT,
                 "n_raw": 0}
    try:
        text = Path(src).read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        out["structure"] = "unreadable"
        out["errors"].append({"index": None, "line": None, "field": None,
                              "error": f"读取失败: {exc}",
                              "fix": "确认文件存在且为 UTF-8 文本"})
        return out
    import yaml as _yaml

    try:
        raw = _yaml.safe_load(text)
    except _yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        out["structure"] = "yaml_error"
        out["errors"].append({
            "index": None, "line": (mark.line + 1) if mark else None,
            "field": None, "error": f"YAML 解析失败: {exc}",
            "fix": "检查该行缩进/引号;" + _YAML_COLON_HINT})
        return out
    items: list = []
    if raw is None:
        out["structure"] = "empty"
    elif isinstance(raw, list):
        out["structure"] = "list"
        items = raw
    elif isinstance(raw, dict):
        if "experiences" in raw:
            exps = raw.get("experiences")
            if isinstance(exps, list):
                out["structure"] = "wrapped:experiences"
                items = exps
            elif exps is None:
                out["structure"] = "wrapped:experiences"
            else:
                out["structure"] = "invalid"
                out["errors"].append({
                    "index": None, "line": None, "field": "experiences",
                    "error": f"experiences 必须是列表(当前 "
                             f"{type(exps).__name__})",
                    "fix": "写成 `experiences:` 下逐条 `- claim: ...`"})
        elif "claim" in raw or "statement" in raw:
            out["structure"] = "single_mapping"
            items = [raw]
        else:
            out["structure"] = "invalid"
            out["errors"].append({
                "index": None, "line": 1, "field": None,
                "error": "顶层是 mapping 但没有 experiences 键(键: "
                         f"{sorted(map(str, raw))[:8]})",
                "fix": "顶层改成列表,或把经验列表放到 `experiences:` 键下"})
    else:
        out["structure"] = "invalid"
        out["errors"].append({
            "index": None, "line": 1, "field": None,
            "error": f"顶层必须是列表或 mapping(当前 {type(raw).__name__})",
            "fix": "顶层改成列表(`- claim: ...`)"})
    lines = _yaml_item_lines(text) if items else []
    out["n_raw"] = len(items)
    for i, item in enumerate(items, start=1):
        line = lines[i - 1] if i - 1 < len(lines) else None
        errs, warns = validate_experience_item(item, i, line)
        out["warnings"].extend(warns)
        if errs:
            out["errors"].extend(errs)
            continue
        out["items"].append((i, line, item))
    return out


# ---------------------------------------------------------------------------
# Level 1: Evidence(facts)
# ---------------------------------------------------------------------------

class EvidenceStore:
    """原始事实层:trials.jsonl / generations.jsonl,只追加。"""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "memory" / "facts"
        self.dir.mkdir(parents=True, exist_ok=True)

    def record_trial(self, trial: TrialRecord) -> None:
        append_jsonl(self.dir / "trials.jsonl", {
            "recorded_at": now_iso(),
            "trial_id": trial.trial_id,
            "status": trial.status,
            "verdict": trial.verdict,
            "failure_reason": trial.data.get("failure_reason"),
            "algorithm": trial.data.get("algorithm"),
            "generation": trial.generation,
            "search_action": trial.data.get("search_action"),
            "hypothesis": trial.data.get("hypothesis"),
            "primary_metric": ((trial.data.get("evaluation") or {})
                               .get("primary_metric")),
            "delta": ((trial.data.get("evaluation") or {}).get("delta")),
            "trial_dir": str(trial.trial_dir),
        })

    def record_generation(self, payload: dict) -> None:
        append_jsonl(self.dir / "generations.jsonl", {
            "recorded_at": now_iso(), **payload,
        })

    def all_trial_facts(self) -> list[dict]:
        return read_jsonl(self.dir / "trials.jsonl")


# ---------------------------------------------------------------------------
# Level 2: Task Experience
# ---------------------------------------------------------------------------

class TaskExperience:
    """当前任务压缩总结(§14.1 Level 2)。"""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "experience" / "task-experience.yaml"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        if self.path.exists():
            return load_yaml(self.path) or {}
        return {}

    def save(self, data: dict) -> None:
        data.setdefault("updated_at", now_iso())
        dump_yaml(self.path, data)

    def record_final_recipe(self, recipe: dict) -> dict:
        """写入冠军交付配方摘要(v2.16.3,D-014 表内第 5 条)。

        背景:``auto_summarize`` 恒把 ``final_recipe`` 写 None,而
        ``cmd_trial.finalize-generation`` 的合并只**保留**已有值 ——
        全链路没有任何地方写过它,所以 task-experience.yaml 的
        ``final_recipe`` 从 v2.14 起恒为 null,报告与跨 run 复用都拿不到
        "本任务最终交付的是什么配方"。

        本方法走既有 TaskExperience 通道(load → merge → save,不手写
        YAML),交付阶段(finalize-retrain 成功落盘 / evaluate-final)调用:

        - 幂等:同 ``model_sha256`` + 同 ``refit_mode`` 已记录时不动;
        - 只补 ``final_recipe``,不改其它字段;文件不存在时只写这一项
          (其余字段由 finalize-generation 的 auto_summarize 合并补齐,
          该合并对 ``final_recipe`` 是"已有即保留")。

        返回写入后的 `final_recipe`。
        """
        data = self.load() or {}
        prev = data.get("final_recipe")
        if isinstance(prev, dict) and prev.get("model_sha256") == \
                recipe.get("model_sha256") and \
                prev.get("refit_mode") == recipe.get("refit_mode"):
            return prev
        data["final_recipe"] = dict(recipe, recorded_at=now_iso())
        self.save(data)
        return data["final_recipe"]

    def auto_summarize(self, trials: list[TrialRecord], spec: TaskSpec,
                       generations: list[dict] | None = None) -> dict:
        """从 Trial 事实自动压缩(确定性部分);LLM 反思可补充语义字段。"""
        accepted, failed, diminishing = [], [], []
        for t in trials:
            ev = t.data.get("evaluation") or {}
            verdict = ev.get("verdict")
            hyp = (t.data.get("hypothesis") or {}).get("statement", "")
            if verdict == "accepted":
                delta = (ev.get("delta") or {}).get("vs_parent")
                accepted.append({
                    "trial_id": t.trial_id, "hypothesis": hyp,
                    "delta_vs_parent": delta,
                    "algorithm": t.data.get("algorithm"),
                })
            elif verdict in ("rejected", "invalid") or t.status == "FAILED":
                failed.append({
                    "trial_id": t.trial_id, "hypothesis": hyp,
                    "failure_reason": t.data.get("failure_reason"),
                    # v2.18.4:`failure_reason` 只承载运行时失败,方向性拒绝
                    # (切片门 / 无父代可比 / 相对父代无提升)的原因在
                    # evaluation.reason。旧口径落盘的一律 null,报告 §9 因此
                    # 只有 trial 名 + 假设,读者看不出"为什么失败"。
                    "reason": ev.get("reason"),
                    "verdict": verdict,
                })
        for g in generations or []:
            for d in (g.get("reflection") or {}).get(
                    "diminishing_return_directions", []):
                diminishing.append(d)
        return {
            "task_id": spec.task_id,
            "task_summary": spec.data.get("objective_description", ""),
            "key_findings": [],
            "successful_directions": accepted,
            "failed_directions": failed,
            "diminishing_return_directions": diminishing,
            "final_recipe": None,
            "reusable_experiences": [],
            "task_specific_experiences": [],
            "n_trials": len(trials),
            "updated_at": now_iso(),
        }
