#!/usr/bin/env python3
"""会话工具边界 hook(v17.2):把铁律 13 从软规则变成硬边界。

PreToolUse hook:

1. **主会话**(非 subagent):
   a. Read/Grep/Glob 指向 harness runtime 源码(scripts/modelevo.py、
      runtime/*.py、assets/lint_trial.py)→ 拒绝。主会话遇到异常状态
      的唯一合法动作是 `modelevo.py diagnose`(确定性 triage,输出含
      remedy),不是回读 harness 源码做刑侦——实测一次 triage 主会话
      grep/sed runtime 30+ 次,~140k tokens 未缓存输入全部烧在
      "harness 内部如何工作"上,而这些信息 diagnose 一条命令就能给出;
   b. Bash 执行演进循环保留命令(new-trial/run/run-generation/
      finalize-generation/evaluate-final/report/export-experience/baseline
      等；profile 是允许主会话直跑的例外)→ 拒绝并提示改派 subagent。
2. **subagent(v17.2 新增覆盖;v19.4 补 Bash 直读拦截)**:
   a. Read/Grep/Glob 指向 harness runtime 源码 → 同主会话拒绝;
   b. Bash 的"文本读取命令(sed/grep/cat/head/tail/awk/wc/...)
      直接作用于 harness 源码路径"同样拒绝(v19.4——实测单 run
      26 次严格口径 Bash 源码刑侦,工具级拦截挡不住 Bash 绕行;
      只拦子句首的读取形态,python 执行 CLI 不受影响);
   c. 其余 Bash 放行:subagent 是被派来干活的(写 trial code、
      跑 run、做 adhoc 探查都是合法 Bash)。

豁免通道:
- CLAUDE_PROJECT_DIR 环境变量缺省(非 Claude Code 运行)→ 放行;
- MODELEVO_ALLOW_MAIN_EXEC=1(用户显式豁免,等价于"不用子 Agent
  直接跑",需在 run-state 记录授权);
- MODELEVO_ALLOW_SOURCE_READ=1(诊断 harness 自身的开发场景豁免
  源码拦截,主会话与 subagent 都生效);
- 只读命令(status/leaderboard/snapshot/planner-context/diagnose/
  wait/validate/preflight/init/check-products/capacity-check/
  acknowledge-harness-change/config-template/experience)放行。

3. **模型拓扑批准(v2.16.0,无豁免)**:主会话与 subagent 一律拒绝
   `approve-topology`(`--list` 只读除外)以及对 approvals/ 账本的写入
   (Write/Edit,或带重定向/写命令的 Bash)。上面两个豁免变量都不放行——
   级联/融合的批准只能由用户本人在自己的终端完成。
4. **holdout 标签面(v2.17.2,主会话与 subagent 同规)**:拒绝在官方 CLI
   之外读取 `labels/(validation|shadow_test|final_test).parquet`——
   Read/Grep 的路径、Bash 命令文本(含 `python -c` / heredoc)、Bash 执行的
   `adhoc/`/`feature-lab/` 脚本内容、以及 Write/Edit 写入这两个目录的脚本
   内容。官方 CLI(`python …/scripts/modelevo.py …`)与 skill 根内 harness
   脚本豁免;train 标签(`labels/train.parquet`)与 prescreen 沙箱
   (`prescreen/_sandbox/…`,由 train 切出)不拦。MODELEVO_ALLOW_MAIN_EXEC /
   MODELEVO_ALLOW_SOURCE_READ 不放行(它们豁免的是编排与源码阅读,不是
   留出面口径);唯一豁免是用户在会话环境显式设
   MODELEVO_ALLOW_HOLDOUT_READ=1(调试 harness 评估器自身)。

sidechain(subagent)判定:hook 输入的 transcript_path 检测——最近
sidechain 标记存在则按 subagent 规则(仅 2a 生效)。无法判定时按
主会话规则处理(fail-open 仅指 hook 异常,不做身份降级放行;
hook 异常不阻塞执行是有意设计)。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# harness 源码路径特征(相对 skill root;匹配命令/文件路径中出现即拦)
HARNESS_PATTERNS = (
    re.compile(r"skills[/\\]modelevo[/\\]scripts[/\\]modelevo\.py"),
    re.compile(r"skills[/\\]modelevo[/\\]runtime[/\\][\w/\\]*\.py"),
    re.compile(r"skills[/\\]modelevo[/\\]assets[/\\]lint_trial\.py"),
    # 共享安装形态:/…/model-evo-V2.x/modelevo/runtime/…
    re.compile(r"model-evo-V[\d.]+[/\\]modelevo[/\\]"
               r"(runtime[/\\][\w/\\]*\.py|scripts[/\\]modelevo\.py)"),
    # v2.11.9:CLI 实现拆到 scripts/_mvo/;不覆盖这里 = 铁律 13 在拆分
    # 面板上出现缺口(主会话可以合法读 cmd_trial.py 的 1160 行实现)。
    # 注:_BASH_HARNESS_TARGET 靠 [\w./\-] 已天然覆盖本目录,无需另加分支。
    re.compile(r"skills[/\\]modelevo[/\\]scripts[/\\]_mvo[/\\][\w/\\]*\.py"),
    re.compile(r"model-evo-V[\d.]+[/\\]modelevo[/\\]"
               r"scripts[/\\]_mvo[/\\][\w/\\]*\.py"),
)

#: v19.4:Bash 源码读取形态检测。子句首是文本读取命令且该子句引用
#: harness 源码路径(与 HARNESS_PATTERNS 同一目标集)。只匹配子句首:
#: `python …/modelevo.py status | tail -40` 的 tail 在管道右端、
#: `cat tools/lint_trial.py` 是工作区副本,均不命中。
_BASH_READER_HEAD = re.compile(
    r"(?:^|;|&&|\|\||\||\n)\s*"
    r"(?:sudo\s+)?(?:sed|grep|egrep|awk|cat|head|tail|wc|nl|less|"
    r"strings|diff)\b[^;\n]*")
_BASH_HARNESS_TARGET = re.compile(
    r"(?:skills[/\\]modelevo|modelevo)[/\\]"
    r"(?:runtime|scripts|assets)[/\\][\w./\\-]+\.py"
    r"|model-evo-V[\d.]+[/\\]modelevo[/\\]"
    # v2.11.9:CLI 实现拆到 scripts/_mvo/;

    # 这里靠 [\w./\-] 已天然覆盖(_mvo 是该字符集的子串),

    # 保留本注释说明依赖关系 —— 若将来放宽该字符集,需显式补 _mvo 分支。
    r"(?:runtime|scripts|assets)[/\\][\w./\\-]+\.py")


def _bash_reads_harness_source(command: str) -> bool:
    """True = 命令中存在'文本读取命令直接作用于 harness 源码'的子句。"""
    for m in _BASH_READER_HEAD.finditer(command):
        if _BASH_HARNESS_TARGET.search(m.group(0)):
            return True
    return False


# 主会话允许直接执行的 modelevo 子命令(只读/确定性/固定输出，外加机械
# 材料化步骤 profile)。
# v17.2:增 wait(纯只读等待——只读 trial.yaml/mtime,不改任何状态)
# v2.13.5:增 timing-report / feature-cache(只读汇总;feature-cache --clear
# 只删可重算的缓存,不碰任何裁决产物)
ALLOWED_SUBCOMMANDS = (
    "status", "leaderboard", "snapshot", "planner-context", "diagnose",
    "validate", "preflight", "init", "check-products", "capacity-check",
    "acknowledge-harness-change", "config-template",
    "write-splits-view", "write-splits-sample", "wait", "profile",
    "timing-report", "feature-cache",
    # v2.16.0:只读 --list;批准形态已被 _topology_guard 先行拒绝
    "approve-topology",
    # v2.18.0:experience search/get/evidence/topics 只读本 run 固定的经验库视图。
    # library 不放行:它含合并写入(merge-apply/revert),由收尾 subagent 执行
    "experience",
    # v2.19.0:trial-skeleton 只读父代大纲(预规划下一代 patch);ceiling 与
    # profile 同属机械测量步骤(只写 <run>/ceiling/),benchmark 只读汇总
    "trial-skeleton", "ceiling", "benchmark",
)


def _deny(code: int, message: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": message,
        }
    }, ensure_ascii=False))
    sys.exit(code)


# ---- v2.16.0 模型拓扑批准:只能由用户本人做 ----
#: 批准命令(--list 只读除外)、账本写入函数与批准账本/请求目录。命令名
#: 出现在任何位置都拦(含 heredoc/python -c/变量拼接前的字面量),不要求
#: 前面紧跟 modelevo.py
_APPROVE_CMD = re.compile(r"approve[-_]topology")
_LEDGER_API = re.compile(r"append_ledger|cmd_approve_topology")
_APPROVALS_PATH = re.compile(r"approvals[/\\](?:model-topology\.jsonl|"
                             r"requests\b)")
#: Bash 中可能改写文件的形态(重定向/写命令/解释器);只读查看放行
_BASH_WRITE_HINT = re.compile(
    r">|\btee\b|\bcp\b|\bmv\b|\brm\b|\bln\b|\btruncate\b|\bdd\b|"
    r"\bsed\s+-i|\bperl\b|\bpython\d*\b|\bnode\b|\bchmod\b|\btouch\b")
_TOPOLOGY_DENY_MSG = (
    "模型拓扑批准只能由用户本人在自己的交互式终端执行 approve-topology,"
    "Agent(主会话与 subagent)不得调用该命令,也不得写 approvals/ 账本。"
    "请把 logs/user-decisions.md 里的 PENDING 请求转告用户,并按默认分支"
    "(单模型方向)继续。")


#: v2.17.0:Bash 命令分词(引号感知);只用于识别"modelevo 子命令的长选项
#: 引号参数值"这一种惰性文本,其余形态一律按原文检查
_PYTHON_WORD = re.compile(r"(?:^|/)python[\d.]*$")


def _shell_words(command: str) -> list[tuple] | None:
    """[(kind, start, end, text, quoted_whole, has_subst)];kind ∈ word/op。

    quoted_whole = 整个词是单个引号串(单引号或双引号);has_subst = 双引号
    串内含 $( 或反引号(命令替换会执行,不算惰性文本)。遇到 heredoc(<<)、
    未闭合引号、反斜杠续行之外的异常形态返回 None(调用方不做任何豁免)。
    """
    out: list[tuple] = []
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if c in " \t":
            i += 1
            continue
        if command.startswith("<<", i):
            return None
        op = next((o for o in ("&&", "||") if command.startswith(o, i)), None)
        if op is None and c in ";|&\n()":
            op = c
        if op is not None:
            out.append(("op", i, i + len(op), op, False, False))
            i += len(op)
            continue
        start, parts, quoted_parts, subst = i, [], [], False
        while i < n and command[i] not in " \t;|&\n()":
            ch = command[i]
            if ch in ("'", '"'):
                j = command.find(ch, i + 1)
                while ch == '"' and j > 0 and command[j - 1] == "\\":
                    j = command.find(ch, j + 1)
                if j < 0:
                    return None
                seg = command[i + 1:j]
                if ch == '"' and ("$(" in seg or "`" in seg):
                    subst = True
                parts.append(seg)
                quoted_parts.append(True)
                i = j + 1
                continue
            if ch == "\\" and i + 1 < n:
                parts.append(command[i + 1])
                quoted_parts.append(False)
                i += 2
                continue
            if ch == "`" or command.startswith("$(", i):
                subst = True
            parts.append(ch)
            quoted_parts.append(False)
            i += 1
        whole = len(quoted_parts) == 1 and quoted_parts[0]
        out.append(("word", start, i, "".join(parts), whole, subst))
    return out


def _strip_inert_args(command: str) -> str:
    """把 `python …modelevo.py <子命令≠approve-topology> --opt "<文本>"` 里的
    引号参数值替换为空串(v2.17.0,#22 guard 误伤)。

    run4 regression.2 实测:new-trial 的 --rationale/--hypothesis 里引用了
    批准命令原文(向用户转述待批请求),整条命令被拓扑守卫拒绝。引号内的
    参数值是传给 argparse 的数据,不会被 shell 执行。豁免只覆盖这一种形态:
    - 必须是 modelevo.py 的非批准子命令(批准子命令本身不豁免任何文本);
    - 必须是长选项(--xxx)的值,且值整体是单个引号串;
    - 双引号串内含 $( 或反引号(命令替换)不豁免;
    - 含 heredoc / 未闭合引号的命令不做任何豁免。
    其余一律按原文检查(无豁免语义不变)。
    """
    words = _shell_words(command)
    if not words:
        return command
    spans: list[tuple[int, int]] = []
    seg: list[tuple] = []
    for w in words + [("op", len(command), len(command), "", False, False)]:
        if w[0] == "word":
            seg.append(w)
            continue
        # 一个简单命令结束:找 modelevo.py 与其子命令
        k = next((x for x, t in enumerate(seg)
                  if t[3].replace("\\", "/").endswith("modelevo.py")), None)
        if k is not None and k + 1 < len(seg) and any(
                _PYTHON_WORD.search(t[3]) for t in seg[:k]) and \
                not _APPROVE_CMD.search(seg[k + 1][3]):
            rest = seg[k + 2:]
            for x, t in enumerate(rest):
                txt = t[3]
                if not txt.startswith("--") or t[4]:
                    continue
                if "=" in txt:
                    # --opt="文本":值部分是引号串
                    raw = command[t[1]:t[2]]
                    eq = raw.find("=")
                    val = raw[eq + 1:]
                    if len(val) >= 2 and val[0] in "'\"" and val[-1] == val[0] \
                            and not t[5]:
                        spans.append((t[1] + eq + 1, t[2]))
                    continue
                if x + 1 < len(rest):
                    nxt = rest[x + 1]
                    if nxt[4] and not nxt[5]:
                        spans.append((nxt[1], nxt[2]))
        seg = []
    if not spans:
        return command
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(command[last:a])
        out.append('""')
        last = b
    out.append(command[last:])
    return "".join(out)


def _topology_guard(tool: str, payload: dict) -> None:
    """拓扑批准边界:对主会话与 subagent 同规生效,任何豁免变量都不放行。

    v2.17.0:Bash 检查前先剔除惰性文本(modelevo 非批准子命令长选项的引号
    参数值,见 `_strip_inert_args`);批准命令作为实际子命令调用、出现在
    可执行位置(含 heredoc / python -c / bash -c / 命令替换)或对 approvals/
    的写入仍一律拒绝。
    """
    if tool == "Bash":
        command = _strip_inert_args(str(payload.get("command", "")))
        if _LEDGER_API.search(command):
            _deny(2, _TOPOLOGY_DENY_MSG)
        if _APPROVE_CMD.search(command) and not re.search(
                r"approve-topology\s[^;\n|&]*--list\b", command):
            _deny(2, _TOPOLOGY_DENY_MSG)
        if _APPROVALS_PATH.search(command) and _BASH_WRITE_HINT.search(
                command):
            _deny(2, _TOPOLOGY_DENY_MSG)
    elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        target = str(payload.get("file_path") or payload.get(
            "notebook_path") or "")
        if _APPROVALS_PATH.search(target) or target.replace(
                "\\", "/").rstrip("/").endswith("/approvals"):
            _deny(2, _TOPOLOGY_DENY_MSG)


# ---- v2.17.2(E1)holdout 标签面边界:主会话与 subagent 同规 ----
# 事故(run4 req-t7-400w gen-06):adhoc/gen06_cb_*probe.py 共 5 个探针脚本
# 直接 pd.read_parquet(".../sample-profile/splits/labels/validation.parquet")
# 在 validation 标签上调 CatBoost 参数——hook 只拦 harness 源码,留出面标签
# 畅通无阻,"validation/shadow 标签只允许固定 evaluator 消费"全靠自觉。
_HOLDOUT_SPLITS = r"(?:validation|shadow_test|final_test)"
#: 字面路径:`labels` 是完整路径分量(覆盖 splits*/ 与 sample-profile/…)
_HOLDOUT_LABEL_PATH = re.compile(
    r"(?:^|(?<=[/\\\s'\"`=(,{}]))labels[/\\]" + _HOLDOUT_SPLITS
    + r"\.parquet")
#: 分量拼接:SPL / "labels" / "validation.parquet"、os.path.join(…)、"labels/"+…
_HOLDOUT_LABEL_JOIN = re.compile(
    r"[\"']labels[/\\]?[\"']\s*[,/+]\s*[rRfF]{0,2}[\"'][/\\]?"
    + _HOLDOUT_SPLITS + r"\.parquet[\"']")
#: 动态标签目录(split 名来自变量/glob);配合 holdout split 字面量才判命中。
#: "labels" 须处在路径拼接位置(/ 或 join 的逗号两侧),df["labels"] 不算
_LABEL_DIR_DYNAMIC = re.compile(
    r"/\s*[\"']labels[\"']|[\"']labels[\"']\s*/|,\s*[\"']labels[\"']\s*,"
    r"|labels[/\\](?:\{|%|\*|\$)|labels[/\\][\"']\s*\+|\bLABELS_SUBDIR\b")
_HOLDOUT_LITERAL = re.compile(r"[\"']" + _HOLDOUT_SPLITS + r"[\"']")
_LOAD_LABELS_CALL = re.compile(r"\bload_labels\s*\(")
#: runtime.data.load_labels(run_dir) 不传 splits = 读全部 split(含留出面)
_RUNTIME_LOAD_LABELS = re.compile(r"\bruntime\b")
_LAB_DIR = re.compile(r"(?:^|[/\\])(?:adhoc|feature-lab)[/\\]")
_LAB_SCRIPT_SUFFIX = (".py", ".sh", ".bash", ".zsh", ".ipynb", ".sql", ".r")
_INTERP_WORD = re.compile(r"(?:^|/)(?:python[\d.]*|bash|sh|zsh)$")
#: 只看元数据/文本检索的子句(ls/stat/grep 不会解析 parquet 内容)
_META_HEADS = ("ls", "stat", "du", "grep", "egrep", "rg", "file")
_SKILL_ROOT = Path(__file__).resolve().parents[1]
_HOLDOUT_DENY_MSG = (
    "holdout 标签面边界(v2.17.2,%s):不得在官方 CLI 之外读取 "
    "validation/shadow_test/final_test 标签(命中:%s)。这些标签只允许固定 "
    "evaluator 经 `modelevo.py` 消费;adhoc/feature-lab 脚本读它们等于用留出面"
    "做选择,确认面与盲测面的口径随之失效。合法替代:① trial 代码用模板访问器"
    "(load_labels/split 访问函数)取数,由 `modelevo.py run` 执行与评估;② trial "
    "间对照用 `modelevo.py contrast --trials A,B`(harness 配对 bootstrap,"
    "只读 validation 打分);③ 训练前方向预筛用 `modelevo.py prescreen "
    "--trial <候选>`(train 内时间切片,只读 train 标签);④ 只用 train 标签的"
    "分析可直接读 labels/train.parquet。确需豁免(调试 harness 评估器自身)时"
    "由用户在会话环境设 MODELEVO_ALLOW_HOLDOUT_READ=1。")


def _call_args(text: str, start: int, cap: int = 400) -> str:
    """从 `(` 之后取到配对 `)` 的参数文本(最多 cap 字符,容忍一层嵌套)。"""
    depth, out = 1, []
    for ch in text[start:start + cap]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    return "".join(out)


def _holdout_in_text(text: str, *, code: bool) -> str | None:
    """文本里读取 holdout 标签面的证据(无则 None)。

    - 字面路径 `…/labels/<holdout>.parquet`(prescreen 沙箱路径除外);
    - 分量拼接 `"labels" / "validation.parquet"`;
    - code=True(脚本内容 / python -c / heredoc)时再看 `load_labels(` 调用
      与动态标签目录:参数里有 holdout 字面量即命中;split 由变量给出时,
      全文出现 holdout split 字面量才命中;runtime.data.load_labels 不传
      splits(缺省读全部 split)同样命中。只取 train 的调用放行。
    """
    for m in _HOLDOUT_LABEL_PATH.finditer(text):
        a = m.start()
        while a > 0 and text[a - 1] not in " \t\n'\"`=(,;|&<>":
            a -= 1
        token = text[a:m.end()]
        if re.search(r"(?:^|[/\\])prescreen[/\\]", token) and not re.search(
                r"(?:^|[/\\])\.\.[/\\]", token):
            continue  # prescreen 沙箱的 validation = train 留出段
        return token[-120:]
    m = _HOLDOUT_LABEL_JOIN.search(text)
    if m:
        return m.group(0)
    if not code:
        return None
    has_literal = _HOLDOUT_LITERAL.search(text) is not None
    runtime_api = _RUNTIME_LOAD_LABELS.search(text) is not None
    for m in _LOAD_LABELS_CALL.finditer(text):
        args = _call_args(text, m.end())
        if _HOLDOUT_LITERAL.search(args):
            return "load_labels(%s)" % args.strip()[:80]
        if re.fullmatch(r"\s*[\"']train[\"']\s*", args) or re.search(
                r"[\[(]\s*[\"']train[\"']\s*,?\s*[\])]", args):
            continue  # 只取 train
        if has_literal:
            return "load_labels(%s) + holdout split 字面量" % args.strip()[:60]
        if runtime_api and not re.search(r"\bsplits\s*=", args) \
                and args.count(",") < 2:
            return "load_labels(%s) 缺省读全部 split" % args.strip()[:60]
    if has_literal:
        m = _LABEL_DIR_DYNAMIC.search(text)
        if m:
            return "动态标签路径 %s + holdout split 字面量" % m.group(0)
    return None


def _shell_segments(command: str) -> list[list[tuple]] | None:
    """按 shell 操作符切成简单命令(词列表);heredoc/未闭合引号返回 None。"""
    words = _shell_words(command)
    if words is None:
        return None
    segs, cur = [], []
    for w in words:
        if w[0] == "op":
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(w)
    if cur:
        segs.append(cur)
    return segs


def _segment_program(seg: list[tuple]) -> tuple[str, str | None, int]:
    """(首个程序词, 解释器执行的脚本路径或 None, 程序词下标)。

    跳过前导环境变量赋值与 nohup/time/timeout N/env/nice 包装;
    解释器(python/bash/sh)后跳过 `-u` 一类开关取第一个位置参数。
    """
    i, n = 0, len(seg)
    while i < n:
        t = seg[i][3]
        if re.fullmatch(r"[A-Za-z_]\w*=.*", t) or t in (
                "nohup", "time", "env", "nice", "exec", "command"):
            i += 1
            continue
        if t == "timeout" and i + 1 < n:
            i += 2
            continue
        break
    if i >= n:
        return "", None, i
    prog = seg[i][3]
    if _INTERP_WORD.search(prog):
        j = i + 1
        while j < n and seg[j][3].startswith("-"):
            if seg[j][3] in ("-c", "-m"):
                return prog, None, i
            j += 1
        return prog, (seg[j][3] if j < n else None), i
    if prog.endswith(_LAB_SCRIPT_SUFFIX):
        return prog, prog, i
    return prog, None, i


def _is_harness_script(path: str, resolved: Path | None) -> bool:
    """官方 CLI 与 skill 根内的 harness 脚本(内部进程,标签访问受 harness 约束)。

    相对路径按 payload cwd / 命令内 cd 解析后的结果判断(不用 hook 进程自己
    的 cwd,否则在 skill 根下运行的 hook 会把 run 的 adhoc/x.py 误认成 harness)。
    """
    p = path.replace("\\", "/")
    if p.endswith("scripts/modelevo.py") or "skills/modelevo/" in p:
        return True
    if resolved is None:
        return False
    try:
        rp = resolved.resolve()
        return _SKILL_ROOT in rp.parents
    except Exception:  # noqa: BLE001
        return False


def _resolve_script(path: str, command: str, cwd: str | None) -> Path | None:
    """把命令里的脚本路径解析成存在的文件(cwd / 命令内 cd 目录 / 项目根)。"""
    raw = os.path.expanduser(os.path.expandvars(path))
    if "$" in raw:
        return None
    p = Path(raw)
    bases: list[Path] = []
    base0 = Path(cwd) if cwd else Path.cwd()
    bases.append(base0)
    for m in re.finditer(r"(?:^|[;&|\n(])\s*cd\s+([^\s;&|)]+)", command):
        d = Path(os.path.expanduser(m.group(1).strip("'\"")))
        bases.insert(0, d if d.is_absolute() else base0 / d)
    if os.environ.get("CLAUDE_PROJECT_DIR"):
        bases.append(Path(os.environ["CLAUDE_PROJECT_DIR"]))
    for cand in ([p] if p.is_absolute() else [b / p for b in bases]):
        try:
            if cand.is_file():
                return cand
        except OSError:
            continue
    return None


def _read_script(p: Path, cap: int = 2_000_000) -> str:
    try:
        with open(p, "rb") as f:
            return f.read(cap).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _bash_holdout_evidence(command: str, cwd: str | None) -> str | None:
    """Bash 命令读取 holdout 标签面的证据。

    1. 剔除官方 CLI / harness 脚本与只看元数据的子句(ls/stat/grep)后,
       检查命令文本(含内联代码);
    2. 命令执行 adhoc/ 或 feature-lab/ 下的脚本时读取脚本内容同样检查。
    含 heredoc/命令替换的命令不做任何子句剔除(按原文检查)。
    """
    segs = _shell_segments(command)
    scripts: list[str] = []
    if segs is None or "$(" in command or "`" in command:
        text = command
        for m in re.finditer(
                r"(?:python[\d.]*|bash|sh)(?:\s+-\w+)*\s+([^\s;|&'\"<>()]*"
                r"(?:adhoc|feature-lab)[/\\][^\s;|&'\"<>()]+)", command):
            scripts.append(m.group(1))
    else:
        keep: list[str] = []
        for seg in segs:
            prog, script, _ = _segment_program(seg)
            if script and _is_harness_script(
                    script, _resolve_script(script, command, cwd)):
                continue
            if prog in _META_HEADS:
                continue
            if script:
                scripts.append(script)
            keep.append(command[seg[0][1]:seg[-1][2]])
        text = "\n".join(keep)
    code = bool(re.search(r"python[\d.]*\b[^\n;|&]*\s-c\b", text)) or \
        "<<" in text
    ev = _holdout_in_text(text, code=code)
    if ev:
        return ev
    for s in scripts:
        if not _LAB_DIR.search(s.replace("\\", "/")) or \
                not s.endswith(_LAB_SCRIPT_SUFFIX):
            continue
        p = _resolve_script(s, command, cwd)
        if p is None:
            continue
        ev = _holdout_in_text(_read_script(p), code=True)
        if ev:
            return "%s → %s" % (s, ev)
    return None


def _edited_content(tool: str, payload: dict, target: str) -> str:
    """Write/Edit/MultiEdit/NotebookEdit 落盘后的脚本内容(尽力还原)。"""
    if tool == "Write":
        return str(payload.get("content") or "")
    if tool == "NotebookEdit":
        return str(payload.get("new_source") or "")
    edits = payload.get("edits") if tool == "MultiEdit" else [payload]
    edits = [e for e in (edits or []) if isinstance(e, dict)]
    p = Path(target)
    base = _read_script(p) if p.is_file() else ""
    if not base:
        return "\n".join(str(e.get("new_string") or "") for e in edits)
    for e in edits:
        old, new = str(e.get("old_string") or ""), str(
            e.get("new_string") or "")
        if old and old in base:
            base = base.replace(old, new) if e.get("replace_all") else \
                base.replace(old, new, 1)
        else:
            base += "\n" + new
    return base


def _holdout_label_guard(tool: str, payload: dict, hook_input: dict) -> None:
    """holdout 标签面边界(v2.17.2):主会话与 subagent 同规生效。

    MODELEVO_ALLOW_MAIN_EXEC / MODELEVO_ALLOW_SOURCE_READ 不放行;唯一豁免
    MODELEVO_ALLOW_HOLDOUT_READ=1(用户在会话环境显式设置)。判定异常时
    放行(与 hook 整体 fail-open 设计一致)。
    """
    if os.environ.get("MODELEVO_ALLOW_HOLDOUT_READ") == "1":
        return
    ev = None
    try:
        if tool in ("Read", "Grep", "Glob"):
            if tool != "Glob":  # Glob 只列文件名,不读内容
                for k in ("file_path", "path", "notebook_path"):
                    v = payload.get(k)
                    if v and not ev:
                        ev = _holdout_in_text(str(v), code=False)
        elif tool == "Bash":
            ev = _bash_holdout_evidence(str(payload.get("command", "")),
                                        hook_input.get("cwd"))
        elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            target = str(payload.get("file_path") or payload.get(
                "notebook_path") or "")
            if _LAB_DIR.search(target.replace("\\", "/")) and \
                    target.lower().endswith(_LAB_SCRIPT_SUFFIX):
                ev = _holdout_in_text(
                    _edited_content(tool, payload, target), code=True)
                if ev:
                    ev = "%s → %s" % (target, ev)
    except Exception:  # noqa: BLE001
        return
    if ev:
        who = "subagent" if _in_subagent(hook_input) else "主会话"
        _deny(2, _HOLDOUT_DENY_MSG % (who, ev))


def _allow() -> None:
    sys.exit(0)


def _in_subagent(hook_input: dict) -> bool:
    """身份判定:优先 payload 字段,transcript tail-scan 作回退。

    v17.3 修复(subagent 识别失效事故,2026-09 实测):
    Claude Code 的 PreToolUse payload 中,主会话与 subagent 的
    `transcript_path` **可能指向同一个主会话转录**(实测:subagent
    payload 的 transcript_path = 主会话 jsonl,该文件内
    `isSidechain` 恒为 false)——此时 tail-scan 恒判 False,subagent
    被按主会话口径拦截,演进循环整体无法启动(实测某 run 的
    subagent 被拒 27 次,phase 卡在 INITED)。

    可靠信号是 payload 的 `agent_id` / `agent_type`:实测**仅**
    subagent 的 payload 携带这两个键,主会话 payload 完全没有。
    transcript tail-scan 保留为回退路径(在 transcript_path 确实
    指向 subagent 转录的形态下仍然有效,经对照实验验证)。
    """
    # 主路径:payload 自带的 subagent 标识(实测最可靠)
    if hook_input.get("agent_id") or hook_input.get("agent_type"):
        return True
    # 回退:transcript 侧判定(transcript_path 指向 subagent 转录时有效)
    tp = hook_input.get("transcript_path")
    if not tp or not Path(tp).exists():
        return False  # 无法判定 → 按主会话处理(保守拦截面)
    try:
        # 只扫尾部 200KB:判定只需要"最近的说话者"
        with open(tp, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            tail = f.read().decode("utf-8", errors="replace")
        last_main = -1
        last_side = -1
        for m in re.finditer(
                r'"isSidechain":\s*(true|false)', tail):
            idx = m.start()
            val = m.group(1) == "true"
            if val:
                last_side = idx
            else:
                last_main = idx
        return last_side > last_main
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    if os.environ.get("CLAUDE_PROJECT_DIR") is None:
        _allow()  # 非 Claude Code 上下文(测试/手工调用)

    try:
        hook_input = json.loads(sys.stdin.read() or "{}")
    except Exception:  # noqa: BLE001
        _allow()  # 输入不可解析 → fail-open

    tool = hook_input.get("tool_name", "")
    payload = hook_input.get("tool_input") or {}
    # v2.16.0:拓扑批准边界先于一切豁免(MODELEVO_ALLOW_MAIN_EXEC 只豁免
    # "主会话不直接跑演进命令",不等于授权 Agent 代替用户批准)
    _topology_guard(tool, payload)
    # v2.17.2:holdout 标签面同样先于 MAIN_EXEC/SOURCE_READ 豁免(那两个变量
    # 豁免的是编排与源码阅读,不是留出面口径)
    _holdout_label_guard(tool, payload, hook_input)
    if os.environ.get("MODELEVO_ALLOW_MAIN_EXEC") == "1":
        _allow()  # 用户显式豁免(需在 run-state 记录授权)

    is_sub = _in_subagent(hook_input)

    if tool in ("Read", "Grep", "Glob"):
        # v17.2:源码拦截对主会话与 subagent 同规则生效(subagent 的
        # 契约出口是 references/ 文档与 diagnose,不是 grep runtime/)。
        if os.environ.get("MODELEVO_ALLOW_SOURCE_READ") == "1":
            _allow()  # 开发 harness 自身的诊断场景
        target = json.dumps(payload, ensure_ascii=False)
        if any(p.search(target) for p in HARNESS_PATTERNS):
            who = "subagent" if is_sub else "主会话"
            _deny(2, (
                "铁律 13(%s 边界):不得回读 harness 源码。异常状态用"
                " `python <skill-root>/scripts/modelevo.py diagnose "
                "--run <run_dir> --spec <spec> [--trial <id>]` 一键归因"
                "(输出含 remedy);字段级契约读 references/"
                "protocol-digest.md 与 template-api.md,不要 grep "
                "runtime/。确需豁免(开发 harness 自身)时设 "
                "MODELEVO_ALLOW_SOURCE_READ=1。" % who
            ))
        _allow()

    if tool == "Bash":
        command = str(payload.get("command", ""))
        # ---- v19.4:源码直读拦截(主会话与 subagent 同规) ----
        # 实测一次 run 里 26 次(严格口径)Bash sed/grep/cat 直读
        # runtime/|scripts/|assets/,全部是契约明令禁止的"源码刑侦";
        # hook 此前只拦 Read/Grep/Glob 工具,Bash 绕行畅通。
        # 逃生口与工具级拦截一致:MODELEVO_ALLOW_SOURCE_READ=1。
        if os.environ.get("MODELEVO_ALLOW_SOURCE_READ") != "1":
            if _bash_reads_harness_source(command):
                who = "subagent" if is_sub else "主会话"
                _deny(2, (
                    "铁律 13(%s 边界):Bash 直读 harness 源码与 Read/"
                    "Grep/Glob 同罪,已拦截。异常状态用 `python "
                    "<skill-root>/scripts/modelevo.py diagnose --run "
                    "<run_dir> --spec <spec> [--trial <id>]` 一键归因"
                    "(输出含 remedy);字段级契约读 references/"
                    "protocol-digest.md 与 template-api.md。确需豁免"
                    "(开发 harness 自身)时设 MODELEVO_ALLOW_SOURCE_READ=1。"
                    % who
                ))
        if not is_sub:
            # 只拦主会话的 modelevo CLI 演进循环命令;subagent 的
            # Bash 不拦(它是被派来执行的一线角色)。v19.4 正则收紧:
            # 旧 `modelevo\.py\s+(\w+)` 会把 `cat log | grep
            # 'modelevo.py run'`(在日志里 grep 字符串)误判为执行;
            # 现要求 python 解释器前缀的执行形态。
            m = re.search(
                r"(?:python3?|/usr/bin/env\s+python3?)\s+[^;\n]*?"
                r"modelevo\.py\s+(\w[\w-]*)", command)
            if m and m.group(1) not in ALLOWED_SUBCOMMANDS:
                _deny(2, (
                    "铁律 13(主会话边界):`%s` 属演进循环执行命令,"
                    "主会话只做调度,应派 generation subagent 执行。"
                    "主会话允许的是:%s。用户显式豁免时设 "
                    "MODELEVO_ALLOW_MAIN_EXEC=1 并记录 run-state。"
                    % (m.group(1), ", ".join(ALLOWED_SUBCOMMANDS))
                ))
        _allow()

    _allow()


if __name__ == "__main__":
    main()
