"""warm-start 配方源码改写(v2.15.2)。

从 `runtime/memory.py` 拆出的**纯 plumbing** 模块:只做源码文本改写与静态
校验,不参与任何评分/落盘数字,放在 scripts 侧可避免触碰 harness hash。

背景(v2.15.0 实测事故,req-t7-rand400w run4-memory-refactor4):
gen-00 会复制 `memory/top-trials/<task_id>/rank<N>/train.py` 的配方作为
warm trial(v2.18.0 起配方来自经验库,渲染后仍经本模块校验)。配方行是**相对式**

    RUN_DIR = pathlib.Path(__file__).resolve().parents[3]

旧实现用一条正则改写:

    re.subn(r'(?m)^RUN_DIR\\s*=\\s*pathlib\\.Path\\([^\\)]*\\)',
            f"RUN_DIR = pathlib.Path({run_dir!r})", text)

`[^\\)]*` 在 `pathlib.Path(__file__` 处遇**第一个右括号即停**,匹配区间不含
其后的 `.resolve().parents[3]`;替换串又只插入新路径、不删除匹配区间内的原文
⇒ 结果行是

    RUN_DIR = pathlib.Path('/…/run-x').resolve().parents[3]

新 run 路径**又被上跳 3 层** → `SPLITS` 指向 `<share>/sample-profile/splits/`
→ 两个 warm trial 同因 `FileNotFoundError` 失败(runtime 0.48s,被归类
infra_error);且只要 subn 命中就打 "RUN_DIR 已改写" info 日志,错误完全不可见。

本模块的修复:
1. **整行替换**(`^RUN_DIR = pathlib.Path(.*$`,无条件吃到行尾),不保留残留;
2. 写盘前做**解析级硬校验**(`_run_dir_resolves_in_source`):改写后的赋值
   必须能 `Path(...).resolve() == run_dir.resolve()`,且不含 `__file__` /
   `.parents[` 残留;
3. 校验不过返回 `not_found`,由调用方**放弃该 warm trial**,绝不带错路径进训练。
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["rewrite_recipe_run_dir", "run_dir_resolves_in_source",
           "warm_recipe_fingerprint", "config_semantic_hash"]

#: 改写后的目标行模板(绝对路径,残留无处藏身)
_ROW_TMPL = "RUN_DIR = pathlib.Path({path!r})"


def rewrite_recipe_run_dir(code_text: str, run_dir: Path) -> tuple[str, str]:
    """把 warm-start 配方源码里的 `RUN_DIR = pathlib.Path(...)` 改写为本 run 路径。

    返回 `(new_text, status)`,status ∈ {"verified", "not_found"}:
    - "verified" : 已整行改写且通过解析级校验,`new_text` 可安全写盘;
    - "not_found": 找不到可改写的行(或改写后校验不过),`new_text` 是
      **原文本**(调用方据此放弃该 warm trial,而不是用可能错的路径去训练)。

    - 只改**顶层**赋值(行首即 `RUN_DIR`);
    - 改写成目标 run_dir 的绝对路径;
    - 等号两侧空白可变(`RUN_DIR=pathlib.Path(...)` / `RUN_DIR  =  ...`)。
    """
    import re as _re

    target = _ROW_TMPL.format(path=str(Path(run_dir).resolve()))
    # 整行匹配:`RUN_DIR = pathlib.Path(` 起头,**无条件吃到行尾**。
    # 注意不能写成 `...Path\(.*\)$` —— 配方行的结尾是 `parents[3]` 的 `]`,
    # `pathlib.Path(__file__)` 的右括号在行中,要求"行尾是右括号"会漏匹配
    # (v2.15.2 首版实现即栽在这里)。
    pat = _re.compile(r"^RUN_DIR\s*=\s*pathlib\.Path\(.*$")
    out_lines: list[str] = []
    hit = False
    for line in code_text.splitlines():
        if not hit and pat.match(line):
            out_lines.append(target)
            hit = True
            continue
        out_lines.append(line)
    if not hit:
        return code_text, "not_found"
    new_text = "\n".join(out_lines)
    if code_text.endswith("\n"):
        new_text += "\n"

    if run_dir_resolves_in_source(new_text, Path(run_dir)):
        return new_text, "verified"
    return code_text, "not_found"


def run_dir_resolves_in_source(code_text: str, run_dir: Path) -> bool:
    """静态校验:源码里 RUN_DIR 是否**确实**指向本 run。

    判据(全部满足才算通过):
    1. 存在 `RUN_DIR = pathlib.Path(<字面量>)` 赋值行,且该路径经
       `resolve()` 后 == run_dir.resolve();
    2. 该赋值行**不含** `__file__` / `.parents[`(残留形态会静默错位);
    3. 用 ast 解析确认右侧就是 `pathlib.Path(...)` 调用。

    这是防"正则命中但改错"的第二道闸——v2.15.0 的 bug 正是因为只判
    `subn` 命中数、不判结果正确性。
    """
    import ast
    import re as _re

    line_pat = _re.compile(r"^RUN_DIR\s*=\s*pathlib\.Path\((.*)\)")
    wanted = Path(run_dir).resolve()
    for line in code_text.splitlines():
        stripped = line.strip()
        m = line_pat.match(stripped)
        if not m or not stripped.endswith(")"):
            continue
        if "__file__" in stripped or ".parents[" in stripped:
            return False
        try:
            node = ast.parse(stripped).body[0].value
        except SyntaxError:
            return False
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "Path"):
            return False
        try:
            candidate = Path(ast.literal_eval(m.group(1).strip()))
        except (ValueError, SyntaxError):
            return False
        try:
            if candidate.resolve() == wanted:
                return True
        except OSError:
            return False
    return False


# ---------------------------------------------------------------------------
# v2.19.0:warm-start 去重指纹 = 代码语义指纹 + config 语义哈希 + 附带模块
#
# v2.18.4 的兜底去重只看 train.py 的 AST 规范形(runtime.recipe_fp)。它抓住了
# ds-flash 的 trial-002/003(代码语义相同、config 只差 notes/trial_id),但有两
# 个漏洞:① 两份配方代码相同、trial.yaml 的超参/特征排除不同 → 训练行为不同,
# 却会被误判重复而少跑一个;② 配方 bundle 的 code/ 下除 train.py 外的模块
# (如 deep_seq.py)不同 → 同样误判。这里把三者合成一个指纹:只有代码、
# 训练相关 config、附带模块全部相同才判重复。
# ---------------------------------------------------------------------------

#: config 中不影响训练行为的键(标签 / 溯源 / 状态);其余键全部进哈希
CONFIG_VOLATILE_KEYS = frozenset({
    "trial_id", "notes", "note", "hypothesis", "rationale", "created_at",
    "run_date", "parent_trial_ids", "generation", "status", "status_history",
    "experience_bindings", "execution", "source", "source_trial_id",
    "recipe_id", "run_id", "task_id", "g2_slot", "seed"})


def config_semantic_hash(cfg) -> str:
    """trial.yaml(训练 config)的语义哈希:剔除 CONFIG_VOLATILE_KEYS 后
    按键排序 JSON 序列化再 sha256。"""
    import hashlib
    import json

    if not isinstance(cfg, dict):
        cfg = {}
    norm = {k: v for k, v in cfg.items() if str(k) not in CONFIG_VOLATILE_KEYS}
    blob = json.dumps(norm, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def warm_recipe_fingerprint(recipe_dir: Path) -> str:
    """warm 配方目录(train.py + trial.yaml + code/ 附带模块)的去重指纹。

    train.py 不可解析时返回 ``"unparsable"``(调用方不去重,偏保守)。
    """
    import hashlib

    import yaml

    from runtime.recipe_fp import recipe_semantic_fingerprint

    rd = Path(recipe_dir)
    code_fp = recipe_semantic_fingerprint(
        (rd / "train.py").read_text(encoding="utf-8"))
    if code_fp == "unparsable":
        return code_fp
    try:
        cfg = yaml.safe_load((rd / "trial.yaml").read_text(
            encoding="utf-8")) if (rd / "trial.yaml").exists() else {}
    except Exception:  # noqa: BLE001 — 解析失败按"不同"处理(不去重)
        return "unparsable"
    h = hashlib.sha256()
    h.update(code_fp.encode())
    h.update(b"\0")
    h.update(config_semantic_hash(cfg or {}).encode())
    code_dir = rd / "code"
    if code_dir.is_dir():
        for f in sorted(code_dir.rglob("*")):
            if f.is_file() and f.name != "train.py" \
                    and "__pycache__" not in f.parts:
                h.update(b"\0")
                h.update(str(f.relative_to(code_dir)).encode())
                h.update(b"\0")
                h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()
