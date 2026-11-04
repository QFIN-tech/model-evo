"""v2.18.0 经验库:配方规范化与渲染(Q21)。

库内配方(train.py / trial.yaml / task-spec.yaml …)此前写死源 run 目录与 skill
根目录:41 份 train.py 的 ``SKILL_ROOT`` 指向主工作树,热启动只改写 ``RUN_DIR``
(scripts/_mvo/warmstart.py)。换机器 / 换 worktree 时要么 import 失败,要么静默
加载另一份 runtime。

入库前把这些路径换成占位符再算哈希(同一配方在任何机器上哈希相同);取用时按
本地环境填回。
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

import yaml

#: 配方与 runtime 的契约版本(RUN_DIR 约定、早停口径、loader 接口任一变化时提升)。
#: v2.16 起库内配方统一为单段早停,至今未变。
RUNTIME_API_VERSION = "2.16"

#: P1-B: 向前兼容声明表。当 runtime 升级时在此追加记录,旧版配方不会被错误排除。
#: 格式: {存入库时的版本: [可运行该配方的 runtime 版本集]}
#: 仅需声明存在版本跳跃的历史区间;表中未出现的版本默认与自身兼容。
RUNTIME_COMPAT_MAP: dict[str, list[str]] = {
    "2.16": ["2.16", "2.17", "2.18"],
    "2.17": ["2.17", "2.18"],
    "2.18": ["2.18"],
}


def runtime_api_compatible(recipe_version: str | None,
                           current_version: str) -> bool:
    """判断入库时版本为 ``recipe_version`` 的配方是否能在 ``current_version`` 运行。

    规则(P1-B):
    1. recipe_version 为 None(旧配方未记录):允许运行,视为兼容。
    2. recipe_version 在 RUNTIME_COMPAT_MAP 中,且 current_version 也在 COMPAT_MAP
       的已知版本集里:精确检查 allowed 列表(防止低版本配方在高版本运行时产生静默错误)。
    3. recipe_version 在 RUNTIME_COMPAT_MAP 中,但 current_version 是未来未知版本
       (不在任何 allowed 列表里):回退到数值比较 recipe <= current,保持向前兼容。
    4. recipe_version 不在 COMPAT_MAP:回退到数值比较 recipe <= current。
    """
    if not recipe_version:
        return True

    def _t(v: str) -> tuple:
        return tuple(int("".join(c for c in p if c.isdigit()) or "0")
                     for p in str(v).split("."))

    allowed = RUNTIME_COMPAT_MAP.get(recipe_version)
    if allowed is not None:
        # 收集 COMPAT_MAP 里所有已知的 current 版本
        all_known_currents: set[str] = set()
        for v_list in RUNTIME_COMPAT_MAP.values():
            all_known_currents.update(v_list)
        if current_version in all_known_currents:
            # 精确检查:已知版本做显式校验
            return current_version in allowed
        # 未知的 future 版本:数值 >= recipe_version 则兼容
        return _t(recipe_version) <= _t(current_version)
    # fallback: 旧版 <= 当前版本才兼容
    return _t(recipe_version) <= _t(current_version)

RUN_DIR_TOKEN = "{{MODELEVO_RUN_DIR}}"
SKILL_ROOT_TOKEN = "{{MODELEVO_SKILL_ROOT}}"
TOKENS = (RUN_DIR_TOKEN, SKILL_ROOT_TOKEN)

_RUN_DIR_RE = re.compile(
    r"^RUN_DIR\s*=\s*pathlib\.Path\(\s*(['\"])(?P<p>[^'\"]+)\1\s*\)", re.M)
_SKILL_ROOT_RE = re.compile(r"^SKILL_ROOT\s*=\s*(['\"])(?P<p>[^'\"]*)\1", re.M)
#: 规范化后仍残留的本机绝对路径(只报告,不阻断)
_ABS_PATH_RE = re.compile(r"(?<![\w{}.])/(?:data|home|root|mnt|tmp|opt)/[^\s'\"),;]+")
#: 解释器等环境路径,不算配方对本机的依赖
_BENIGN_PREFIXES = ("/root/miniconda3/", "/opt/conda/", "/usr/")


def _decode(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def detect_paths(files: dict[str, bytes]) -> dict[str, str]:
    """从 .py 文件的顶层赋值行找出源 RUN_DIR / SKILL_ROOT。"""
    found: dict[str, str] = {}
    for name in sorted(files):
        if not name.endswith(".py"):
            continue
        text = _decode(files[name]) or ""
        m = _RUN_DIR_RE.search(text)
        if m and "run_dir" not in found and m.group("p") not in TOKENS:
            found["run_dir"] = m.group("p")
        m = _SKILL_ROOT_RE.search(text)
        if m and m.group("p") and "skill_root" not in found and \
                m.group("p") not in TOKENS:
            found["skill_root"] = m.group("p")
    return found


def _path_pattern(path: str) -> re.Pattern:
    # 路径后不能紧跟路径字符:run4-v2.16.0-2 不能吃掉 run4-v2.16.0-2.opus5.5
    return re.compile(re.escape(path.rstrip("/")) + r"(?![\w.\-])")


def canonicalize(files: dict[str, bytes], *, run_dir: str | None = None,
                 skill_root: str | None = None) -> tuple[dict[str, bytes], dict]:
    """返回 (规范化后的文件, 信息)。信息含识别到的路径与残留绝对路径。

    显式给出的 run_dir / skill_root 与代码顶层赋值行里写的不一致时(例如配方被
    后续 run 热启动、RUN_DIR 已改写成那个 run 的目录后又导出),两者都换成占位符。
    """
    detected = detect_paths(files)
    run_dir = run_dir or detected.get("run_dir")
    skill_root = skill_root or detected.get("skill_root")
    repl = [(p, tok) for p, tok in ((run_dir, RUN_DIR_TOKEN),
                                    (skill_root, SKILL_ROOT_TOKEN),
                                    (detected.get("run_dir"), RUN_DIR_TOKEN),
                                    (detected.get("skill_root"), SKILL_ROOT_TOKEN)) if p]
    repl = list(dict.fromkeys(repl))
    repl.sort(key=lambda x: len(x[0]), reverse=True)
    pats = [(_path_pattern(p), tok) for p, tok in repl]

    out: dict[str, bytes] = {}
    residual: list[str] = []
    for name, data in files.items():
        text = _decode(data)
        if text is None:
            out[name] = data
            continue
        for pat, tok in pats:
            text = pat.sub(tok, text)
        for m in _ABS_PATH_RE.findall(text):
            if not m.startswith(_BENIGN_PREFIXES) and m not in residual:
                residual.append(m)
        out[name] = text.encode("utf-8")
    info = {"source_run_dir": run_dir, "source_skill_root": skill_root,
            "residual_abs_paths": residual[:20]}
    return out, info


def render(files: dict[str, bytes], *, run_dir: str,
           skill_root: str) -> dict[str, bytes]:
    """把占位符填回本地路径。"""
    out: dict[str, bytes] = {}
    for name, data in files.items():
        text = _decode(data)
        if text is None:
            out[name] = data
            continue
        text = text.replace(RUN_DIR_TOKEN, str(run_dir).rstrip("/"))
        text = text.replace(SKILL_ROOT_TOKEN, str(skill_root).rstrip("/"))
        out[name] = text.encode("utf-8")
    return out


def check_python(files: dict[str, bytes]) -> list[str]:
    """渲染后的 .py 文件逐个 ast.parse;返回错误列表(空 = 通过)。"""
    errors = []
    for name, data in sorted(files.items()):
        if not name.endswith(".py"):
            continue
        text = _decode(data)
        if text is None:
            errors.append(f"{name}: 非 UTF-8")
            continue
        if any(tok in text for tok in TOKENS):
            errors.append(f"{name}: 仍含未渲染的占位符")
            continue
        try:
            ast.parse(text, filename=name)
        except SyntaxError as exc:
            errors.append(f"{name}:{exc.lineno}: {exc.msg}")
    return errors


#: 定义"模型是什么"的配置键;seed / trial_id / notes / confirmation 等不算
RECIPE_IDENTITY_KEYS = ("algorithm", "model_type", "target", "features", "hyperparameters",
                        "data_view")
#: 只表示"配方在哪儿 / 用哪个 seed"的赋值行,不改变配方本身
_LOCATION_LINE_RES = ((re.compile(r"^SEED\s*=.*$", re.M), "SEED = <seed>"),
                      (re.compile(r"^RUN_DIR\s*=.*$", re.M), "RUN_DIR = <run_dir>"),
                      (re.compile(r"^SKILL_ROOT\s*=.*$", re.M), "SKILL_ROOT = <skill_root>"))


def recipe_family(files: dict[str, bytes]) -> str:
    """配方身份:同一配方换 seed 复跑(confirm 子代)、或同一 trial 的多份副本,得到同一值。

    = sha256(代码文件的语义规范形 + trial.yaml 中 RECIPE_IDENTITY_KEYS)。只比较 .py
    代码与 trial.yaml;hypothesis / evaluation / environment 不参与。

    v2.18.4:.py 按 :func:`runtime.recipe_fp.semantic_normal_form` 规范化(顶层语句
    AST 排序;去掉 SEED / RUN_DIR / SKILL_ROOT 与标签常量)。旧口径只按行去掉
    SEED / RUN_DIR / SKILL_ROOT,语句顺序与注释排版不同的同一配方被当成两份 ——
    实测 req-t7 榜单 rank-1/rank-2 逐字节差 435 行、模型工件 sha256 全同,两份都
    被热启动,各烧约 69 min × 2 卡。无法解析的 .py 退回旧的按行口径。
    """
    from ..recipe_fp import semantic_normal_form  # noqa: PLC0415

    parts = []
    for name in sorted(files):
        if name.endswith(".py") and not name.startswith("data_view/"):
            text = _decode(files[name]) or ""
            norm = semantic_normal_form(text)
            if norm is None:
                for pat, repl in _LOCATION_LINE_RES:
                    text = pat.sub(repl, text)
                norm = text
            parts.append((name, norm))
    try:
        cfg = yaml.safe_load((_decode(files.get("trial.yaml", b"")) or "")) or {}
    except yaml.YAMLError:
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}
    parts.append(("trial.yaml", json.dumps({k: cfg.get(k) for k in RECIPE_IDENTITY_KEYS},
                                           sort_keys=True, ensure_ascii=False, default=str)))
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode("utf-8")).hexdigest()
