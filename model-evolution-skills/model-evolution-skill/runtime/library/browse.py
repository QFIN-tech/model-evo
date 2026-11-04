"""v2.18.0 经验库明文镜像:把库的最新状态渲染成一棵给人看的目录树。

库的权威数据在数据库 + 内容寻址存储里(按哈希命名,适合机器);本模块只读,
生成 Markdown 与明文配方目录,供 ls / grep / 编辑器浏览:

    <dest>/
      README.md                       水位、生成时间、任务列表、原则统计、最近更新
      CHANGELOG.md                    更新日志
      topics.md                       主题词表
      principles/INDEX.md             全部原则一览
      principles/<主题前两段>.md       按主题分组的原则(方向、口径、置信度、证据)
      tasks/<atid>/dossier.md         任务卷宗 + 数据底座卷宗
      tasks/<atid>/leaderboard.md     去重后的榜单与固化 baseline
      tasks/<atid>/trials.md          方向表 + 全部历史 trial
      tasks/<atid>/recipes/<NN>-<trial>-<id8>/   明文配方(占位符保留原样)

只依赖仓库读接口,LocalRepository(服务端自动生成)与 HttpRepository
(``library browse --out``)通用。服务端发布用 :func:`publish`:生成到新目录后
原子切换软链接,读者总看到完整的一版。镜像只读、可随时重建;改经验走合并流程。
"""
from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import retrieval
from .curation import render_changelog

MARKER = ".modelevo-browse"
#: 跨命名空间的全局提交(指纹、指纹别名)使用的命名空间
GLOBAL_NAMESPACE = "*"
RECIPE_MARKER = ".recipe_id"
_DIRECTION = {1: "建议做", 0: "无增益 / 边际递减", -1: "避免"}
_SECTIONS = (("ceiling", "天花板与冠军配方"), ("effective", "已证实有效"),
             ("falsified", "已证伪 / 无增益"), ("open_questions", "未决问题"),
             ("data_notes", "数据注意事项"))
_CONF_ORDER = {"high": 0, "medium": 1, "mixed": 2, "contradicted": 3, "low": 4}


def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s)).strip("._") or "_"


def _fmt(v, nd: int = 5) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _cell(s, n: int = 120) -> str:
    s = " ".join(str(s or "").split())
    s = s if len(s) <= n else s[: n - 1] + "…"
    return s.replace("|", "\\|")


def _ts(v) -> str:
    """ISO 时间截到分钟(``2026-10-30T12:34``→``2026-10-30 12:34``);缺失(旧服务端)显示 —。"""
    return str(v)[:16].replace("T", " ") if v else "—"


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str, sort_keys=True)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _group_of(p: dict) -> str:
    t = (p.get("topics") or ["_未分类"])[0]
    return ".".join(str(t).split(".")[:2])


def _confidence(p: dict, stats: dict) -> tuple[str, str]:
    """(置信度, 来源):有证据时用证据计算值,否则是撰写者自报。"""
    st = stats.get(p["principle_id"])
    if st:
        return str(st.get("confidence") or "low"), "证据"
    return str(p.get("confidence") or "low"), "自报"


# --------------------------------------------------------------------------- 原则

def _principle_md(p: dict, stats: dict) -> list[str]:
    conf, src = _confidence(p, stats)
    st = stats.get(p["principle_id"]) or {}
    head = (f"### `{p['principle_id']}` v{p.get('version')} · "
            f"{_DIRECTION.get(p.get('direction'), p.get('direction'))} · 置信度 {conf}({src})")
    lines = [head, "", f"**结论**:{p.get('claim')}", ""]
    sc = p.get("scope") or {}
    lines.append(f"- 主题:{', '.join(f'`{t}`' for t in p.get('topics') or []) or '—'}"
                 f";对象:{', '.join(f'`{o}`' for o in p.get('objects') or []) or '—'}")
    lines.append(f"- 口径:{sc.get('model_type') or '—'} / {sc.get('primary_metric') or '—'}"
                 f" / {sc.get('target_transform') or '—'}")
    lines.append(f"- 首次提交:{_ts(p.get('first_committed_at'))};"
                 f"最近更新:{_ts(p.get('updated_at'))}")
    if p.get("condition"):
        lines.append(f"- 适用条件:`{_j(p['condition'])}`")
    if p.get("effect"):
        lines.append(f"- 效应:`{_j(p['effect'])}`")
    if st:
        lines.append(f"- 证据:{st.get('n_links', 0)} 条关联(显式 {st.get('n_explicit', 0)}),"
                     f"{st.get('n_runs', 0)} 个 run;加权 支持 {_fmt(st.get('w_support'), 3)} / "
                     f"反驳 {_fmt(st.get('w_refute'), 3)};后验 {_fmt(st.get('posterior'), 3)}")
    ev = p.get("evidence") or {}
    if ev.get("summary"):
        lines.append(f"- 原始证据:{_cell(ev['summary'], 400)}")
    if ev.get("trial_ids"):
        lines.append(f"- 来源 trial:{', '.join(ev['trial_ids'][:10])}"
                     + (" …" if len(ev["trial_ids"]) > 10 else ""))
    for lim in p.get("limitations") or []:
        lines.append(f"- 局限:{lim}")
    if p.get("contested_with"):
        lines.append(f"- **争议**:与 {', '.join(f'`{x}`' for x in p['contested_with'])} 结论相反,并存")
    lines.append("")
    return lines


def _principle_files(principles: list[dict], stats: dict) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in principles:
        groups[_group_of(p)].append(p)

    def _key(p):
        return (_CONF_ORDER.get(_confidence(p, stats)[0], 9), -(p.get("direction") or 0),
                p["principle_id"])

    out = {}
    idx = ["# 原则一览", "", f"共 {len(principles)} 条有效原则。点主题进入详情。", "",
           "| 原则 | 方向 | 置信度 | 主题 | 首次提交 | 最近更新 | 结论 |",
           "|---|---|---|---|---|---|---|"]
    for g in sorted(groups):
        ps = sorted(groups[g], key=_key)
        body = [f"# 主题 `{g}`", "", f"{len(ps)} 条原则,按置信度、方向排序。", ""]
        for p in ps:
            body += _principle_md(p, stats)
            conf, src = _confidence(p, stats)
            idx.append(f"| [`{p['principle_id']}`]({safe_name(g)}.md) | "
                       f"{_DIRECTION.get(p.get('direction'), p.get('direction'))} | {conf}({src}) | "
                       f"{_cell(', '.join(p.get('topics') or []), 60)} | "
                       f"{_ts(p.get('first_committed_at'))} | {_ts(p.get('updated_at'))} | "
                       f"{_cell(p.get('claim'), 100)} |")
        out[f"principles/{safe_name(g)}.md"] = "\n".join(body)
    out["principles/INDEX.md"] = "\n".join(idx)
    return out


# --------------------------------------------------------------------------- 任务

def _dossier_md(atid: str, view: dict) -> str:
    lines = [f"# 卷宗 · `{atid}`", ""]
    for key, title in (("task", "任务卷宗"), ("data_base", f"数据底座卷宗 `{view.get('data_base')}`")):
        d = (view.get("dossiers") or {}).get(key)
        lines += [f"## {title}", ""]
        if not d:
            lines += ["(暂无)", ""]
            continue
        lines += [f"版本 v{d.get('version')} · 提交 {d.get('commit_seq')}", ""]
        sec = d.get("sections") or {}
        for k, t in _SECTIONS:
            if sec.get(k):
                lines += [f"### {t}", "", str(sec[k]), ""]
    return "\n".join(lines)


def _recipe_dirname(rank, row: dict) -> str:
    return safe_name(f"{rank}-{row.get('source_trial_id') or 'recipe'}-{row['recipe_id'][:8]}")


def _leaderboard_md(atid: str, view: dict, names: dict[str, str]) -> str:
    lb = view.get("leaderboard") or []
    lines = [f"# 榜单 · `{atid}`", "",
             "同一配方(代码去掉 SEED / RUN_DIR 行 + 定义模型的配置相同)只占一个名次;"
             "「同组」是被并入的副本与 confirm 子代数。增益按指标方向归一(越大越好)。", "",
             "| 名次 | 配方 | 来源 run / trial | 算法 | 测量 | 值 | 增益 | 同组 | 树数 | 契约 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in lb:
        m = r.get("measurement") or {}
        d = names.get(r["recipe_id"])
        link = f"[`{r['recipe_id'][:12]}`](recipes/{d}/)" if d else f"`{r['recipe_id'][:12]}`"
        lines.append(f"| {r.get('rank')} | {link} | {_cell(r.get('source_run_id'), 50)} / "
                     f"{r.get('source_trial_id') or '—'} | {r.get('algorithm') or '—'} | "
                     f"{m.get('kind') or '—'} | {_fmt(m.get('value'))} | {_fmt(m.get('gain'))} | "
                     f"{r.get('family_size') or 1} | {_fmt(r.get('n_trees'))} | "
                     f"{r.get('runtime_api_version') or '—'} |")
    if not lb:
        lines.append("| — | (暂无) | | | | | | | | |")
    bl = view.get("baselines") or {}
    lines += ["", "## 固化 baseline", "", "| 算法 | 配方 | 来源 run / trial |", "|---|---|---|"]
    for algo, b in sorted(bl.items()):
        d = names.get(b["recipe_id"])
        link = f"[`{b['recipe_id'][:12]}`](recipes/{d}/)" if d else f"`{b['recipe_id'][:12]}`"
        lines.append(f"| {algo} | {link} | {_cell(b.get('source_run_id'), 50)} / "
                     f"{b.get('source_trial_id') or '—'} |")
    if not bl:
        lines.append("| — | (暂无) | |")
    return "\n".join(lines)


def _trials_md(atid: str, view: dict) -> str:
    trials = view.get("trial_outcomes") or []
    dt = retrieval.direction_table(view, top=10)
    lines = [f"# 历史 trial · `{atid}`", "",
             f"{len(trials)} 个 trial,来自 {len({t['run_id'] for t in trials})} 个 run。", ""]
    for key, title in (("effective", "有效方向(Top 10)"), ("failed", "失败 / 无增益方向(Top 10)")):
        rows = dt.get(key) or []
        lines += [f"## {title}", "",
                  "| 方向(主题 / 对象) | 代表 trial | 裁决 | 增益 [CI] | 同方向 trial | 假设 / 失败原因 |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            ci = r.get("ci") or [None, None]
            lines.append(
                f"| {_cell(', '.join(r.get('topics') or []), 50)} / {r.get('object') or '—'} | "
                f"{_cell(r.get('run_id'), 40)} / {r.get('trial_id')} | {r.get('verdict') or '—'} | "
                f"{_fmt(r.get('gain'))} [{_fmt(ci[0])}, {_fmt(ci[1])}] | {r.get('n_trials')} | "
                f"{_cell(r.get('failure_reason') or r.get('hypothesis'), 80)} |")
        if not rows:
            lines.append("| (暂无) | | | | | |")
        lines.append("")
    lines += ["## 全部 trial(按 run、代次)", ""]
    by_run: dict[str, list] = defaultdict(list)
    for t in trials:
        by_run[t["run_id"]].append(t)
    for run in sorted(by_run):
        lines += [f"### {run}", "",
                  "| 代 | trial | 动作 | 算法 | 裁决 | 主指标 | Δ父代 [CI] | Δbaseline | 依据 | 假设 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for t in sorted(by_run[run], key=lambda x: (x.get("generation") or 0, x["trial_id"])):
            basis = ((t.get("extra") or {}).get("basis") or {})
            refs = basis.get("refs") or [{"id": e} for e in (t.get("experience_ids") or [])]
            btxt = "novel" if basis.get("novel") else ", ".join(
                f"{r['id']}:{r.get('relation', 'follows')}" for r in refs) or "—"
            if basis.get("source") == "prefilled":
                btxt += "(沿用预填)"
            lines.append(
                f"| {t.get('generation') if t.get('generation') is not None else '—'} | "
                f"{t['trial_id']} | {t.get('action') or '—'} | {t.get('algorithm') or '—'} | "
                f"{t.get('verdict') or t.get('status') or '—'} | {_fmt(t.get('metric_value'))} | "
                f"{_fmt(t.get('delta_parent'))} [{_fmt(t.get('delta_parent_ci_low'))}, "
                f"{_fmt(t.get('delta_parent_ci_high'))}] | {_fmt(t.get('delta_baseline'))} | "
                f"{_cell(btxt, 40)} | {_cell(t.get('hypothesis'), 80)} |")
        lines.append("")
    return "\n".join(lines)


def _write_recipe(repo, rid: str, dest: Path, prev: dict[str, Path]) -> bool:
    """配方按内容寻址不可变:上一版镜像里已有就硬链接 / 复制,否则从库取。返回是否复用。"""
    old = prev.get(rid)
    if old is not None and old.is_dir():
        shutil.copytree(old, dest, copy_function=_link_or_copy)
        return True
    files = repo.get_bundle(rid)
    for name, data in files.items():
        out = dest / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    (dest / RECIPE_MARKER).write_text(rid + "\n", encoding="utf-8")
    return False


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prev_recipes(prev_dir: Path | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if prev_dir is None or not prev_dir.is_dir():
        return out
    for m in prev_dir.glob(f"tasks/*/recipes/*/{RECIPE_MARKER}"):
        try:
            out[m.read_text(encoding="utf-8").strip()] = m.parent
        except OSError:
            continue
    return out


# --------------------------------------------------------------------------- 总装

def task_atids(view: dict) -> list[str]:
    aliases = view.get("aliases") or {}
    return sorted({f["atid"] for f in view.get("fingerprints") or []
                   if f.get("atid") and f["atid"] not in aliases})


def render(repo, namespace: str, dest, *, prev_dir=None, at: int | None = None) -> dict:
    """把命名空间 ``namespace`` 在水位 ``at``(缺省最新)的状态渲染到空目录 ``dest``。"""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    at = at if at is not None else repo.watermark()
    base = repo.run_view(namespace, None, at=at)
    principles, stats = base.get("principles") or [], base.get("stats") or {}
    prev = _prev_recipes(Path(prev_dir) if prev_dir else None)
    files = _principle_files(principles, stats)

    tasks = []
    n_recipes = n_reused = 0
    for atid in task_atids(base):
        view = repo.run_view(namespace, atid, at=at)
        trials, lb = view.get("trial_outcomes") or [], view.get("leaderboard") or []
        if not (trials or lb or view.get("baselines") or (view.get("dossiers") or {}).get("task")):
            continue
        tdir = f"tasks/{safe_name(atid)}"
        names: dict[str, str] = {}
        for r in lb:
            names[r["recipe_id"]] = _recipe_dirname(f"{r['rank']:02d}", r)
        for algo, b in (view.get("baselines") or {}).items():
            names.setdefault(b["recipe_id"], _recipe_dirname(f"baseline-{algo}", b))
        for rid, d in names.items():
            n_reused += _write_recipe(repo, rid, dest / tdir / "recipes" / d, prev)
            n_recipes += 1
        files[f"{tdir}/dossier.md"] = _dossier_md(atid, view)
        files[f"{tdir}/leaderboard.md"] = _leaderboard_md(atid, view, names)
        files[f"{tdir}/trials.md"] = _trials_md(atid, view)
        best = (lb[0].get("measurement") or {}).get("gain") if lb else None
        tasks.append({"atid": atid, "dir": tdir, "data_base": view.get("data_base"),
                      "runs": len({t["run_id"] for t in trials}), "trials": len(trials),
                      "recipes": len(lb), "best_gain": best,
                      "dossier": bool((view.get("dossiers") or {}).get("task"))})

    changelog = _all_changelog(repo, namespace)
    files["CHANGELOG.md"] = render_changelog(changelog, namespace)
    topics = sorted(base.get("topics") or [], key=lambda t: t["code"])
    files["topics.md"] = "\n".join(
        ["# 主题词表", "", "| 主题 | 定义 | 别名 |", "|---|---|---|"]
        + [f"| `{t['code']}` | {_cell(t.get('definition'), 80)} | "
           f"{_cell(', '.join(t.get('aliases') or []), 60)} |" for t in topics])
    files["README.md"] = _readme(namespace, at, principles, stats, tasks, changelog, len(topics))
    for rel, text in files.items():
        _write(dest / rel, text)
    (dest / MARKER).write_text(_j({"namespace": namespace, "watermark": at,
                                   "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n",
                               encoding="utf-8")
    return {"namespace": namespace, "watermark": at, "dest": str(dest), "tasks": len(tasks),
            "principles": len(principles), "recipes": n_recipes,
            "recipes_reused": n_reused}


#: HTTP 接口单次最多返回的更新日志条数(服务端上限 5000)
_CHANGELOG_PAGE = 5000


def _all_changelog(repo, namespace: str) -> list[dict]:
    """按提交序号分页取全部更新日志;页尾若截在一个提交中间,丢掉该提交的残片下页重取。"""
    rows: list[dict] = []
    since = 0
    while True:
        page = repo.changelog(namespace, since=since, limit=_CHANGELOG_PAGE)
        if len(page) < _CHANGELOG_PAGE:
            return rows + page
        last = page[-1]["commit_seq"]
        complete = [r for r in page if r["commit_seq"] < last]
        if not complete:                  # 单个提交超过一页:整页收下,不再细分
            rows += page
            since = last
            continue
        rows += complete
        since = complete[-1]["commit_seq"]


def _readme(ns, at, principles, stats, tasks, changelog, n_topics) -> str:
    by_dir = Counter(_DIRECTION.get(p.get("direction"), p.get("direction")) for p in principles)
    by_conf = Counter(_confidence(p, stats)[0] for p in principles)
    lines = [f"# 经验库明文镜像 · {ns}", "",
             f"- **水位** {at} · 生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}",
             "- 本目录由经验库服务自动生成,**只读**:修改会在下次生成时被覆盖。改经验走 run 收尾的"
             "快照 + 合并流程(references/experience-library-merge.md)。",
             "- 展示的是库的**最新状态**;某个 run 实际使用的经验以该 run 的 "
             "`experience/library-view.json`(固定水位)为准。",
             "- 配方里的路径保留占位符 `{{MODELEVO_RUN_DIR}}` / `{{MODELEVO_SKILL_ROOT}}`,"
             "不能直接运行;要可运行副本用 `modelevo.py library recipe export <配方id> --run <run_dir>`。",
             "", "## 任务", "",
             "| 任务 | 数据底座 | run | trial | 榜单配方 | 最佳增益 | 卷宗 |",
             "|---|---|---|---|---|---|---|"]
    for t in tasks:
        lines.append(f"| [`{t['atid']}`]({t['dir']}/) | {t['data_base'] or '—'} | {t['runs']} | "
                     f"{t['trials']} | [{t['recipes']}]({t['dir']}/leaderboard.md) | "
                     f"{_fmt(t['best_gain'])} | {'[有](' + t['dir'] + '/dossier.md)' if t['dossier'] else '—'} |")
    lines += ["", "## 原则", "",
              f"共 {len(principles)} 条([一览](principles/INDEX.md)),主题词 {n_topics} 个"
              "([词表](topics.md))。", "",
              "- 按方向:" + ";".join(f"{k} {v}" for k, v in by_dir.most_common()),
              "- 按置信度:" + ";".join(f"{k} {v}" for k, v in sorted(
                  by_conf.items(), key=lambda kv: _CONF_ORDER.get(kv[0], 9))),
              "", "## 最近更新", ""]
    recent: dict[int, list] = defaultdict(list)
    for r in changelog:
        recent[r["commit_seq"]].append(r)
    for seq in sorted(recent, reverse=True)[:10]:
        items = recent[seq]
        acts = Counter(i["action"] for i in items)
        lines.append(f"- 提交 {seq} · {str(items[0].get('created_at'))[:16]} · "
                     + ", ".join(f"{k}×{v}" for k, v in acts.most_common())
                     + (" (已撤销)" if any(i.get("reverted") for i in items) else ""))
    lines += ["", "完整记录见 [CHANGELOG.md](CHANGELOG.md)。"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- 发布

def _make_readonly(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                p.chmod(0o444)
            except OSError:
                pass


def publish(repo, namespace: str, browse_root, *, keep: int = 2) -> dict:
    """服务端发布:渲染到 ``<browse_root>/.gen/<ns>-<水位>-<时间>``,再把
    ``<browse_root>/<ns>`` 软链接原子切换过去;保留最近 ``keep`` 版(正在读旧版的人不受影响)。"""
    browse_root = Path(browse_root)
    gen_root = browse_root / ".gen"
    gen_root.mkdir(parents=True, exist_ok=True)
    ns = safe_name(namespace)
    link = browse_root / ns
    prev = link.resolve() if link.is_symlink() else None
    at = repo.watermark()
    # 同一秒内可能连续发布(水位不变时也会重建):纳秒时间戳保证目录名唯一
    target = gen_root / f"{ns}-{at}-{time.strftime('%Y%m%d%H%M%S')}-{time.time_ns() % 10**9:09d}"
    try:
        info = render(repo, namespace, target, prev_dir=prev, at=at)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    _make_readonly(target)
    if link.exists() and not link.is_symlink():
        if (link / MARKER).exists():
            shutil.rmtree(link)           # 旧版本留下的普通目录(自己生成的)
        else:
            raise RuntimeError(f"{link} 不是镜像目录,拒绝覆盖")
    tmp = browse_root / f".{ns}.link.{os.getpid()}"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(os.path.relpath(target, browse_root), tmp)
    os.replace(tmp, link)
    gens = sorted(gen_root.glob(f"{ns}-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in gens[keep:]:
        if old.resolve() != target.resolve():
            shutil.rmtree(old, ignore_errors=True)
    info["link"] = str(link)
    return info


def export_dir(repo, namespace: str, out) -> dict:
    """客户端导出(``library browse --out``):先渲染到临时目录,成功后替换 ``out``。"""
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not (out / MARKER).exists():
        raise RuntimeError(f"{out} 非空且不是镜像目录,拒绝覆盖")
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        info = render(repo, namespace, tmp, prev_dir=out if out.exists() else None)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if out.exists():
        old = out.with_name(f".{out.name}.old.{os.getpid()}")
        os.replace(out, old)
        os.replace(tmp, out)
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(tmp, out)
    info["dest"] = str(out)
    return info


# --------------------------------------------------------------------------- 服务端触发

class BrowsePublisher:
    """写入后防抖发布镜像:``notify()`` 由服务在每次成功写入后调用;静默 ``quiet``
    秒(最长攒 ``max_delay`` 秒)后,对自上次发布以来有新提交的命名空间重新生成。
    失败只记日志,不影响写入。上次发布的水位记在 ``<browse_root>/.state.json``。"""

    def __init__(self, repo, browse_root, *, quiet: float = 5.0, max_delay: float = 60.0,
                 logger=None):
        import logging
        import threading

        self.repo = repo
        self.root = Path(browse_root)
        self.quiet, self.max_delay = quiet, max_delay
        self.log = logger or logging.getLogger("modelevo.library.browse")
        self._event = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._first = self._last = 0.0
        self._thread = None

    # ---- 状态
    def _state_path(self) -> Path:
        return self.root / ".state.json"

    def _load_state(self) -> dict:
        try:
            return json.loads(self._state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self, st: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path().with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(_j(st), encoding="utf-8")
        os.replace(tmp, self._state_path())

    def changed_namespaces(self) -> dict[str, int]:
        """{命名空间: 最新提交序号},只含自上次发布以来有新提交、或镜像缺失的。"""
        st = self._load_state()
        since = min(st.values()) if st else 0
        latest: dict[str, int] = {}
        while True:
            page = self.repo.commits(since=since, limit=1000)
            for c in page:
                latest[c["namespace"]] = max(latest.get(c["namespace"], 0), c["seq"])
            if len(page) < 1000:
                break
            since = page[-1]["seq"]
        # 全局提交(命名空间 "*",如指纹别名)不单独成镜像:它变了,所有命名空间都要重建
        glob_seq = latest.pop(GLOBAL_NAMESPACE, 0)
        out = {}
        for ns, seq in latest.items():
            if seq > st.get(ns, 0) or glob_seq > st.get(ns, 0) or \
                    not (self.root / safe_name(ns)).exists():
                out[ns] = max(seq, glob_seq)
        if glob_seq:
            for ns, seq in st.items():
                if ns != GLOBAL_NAMESPACE and ns not in out and glob_seq > seq:
                    out[ns] = glob_seq
        for ns, seq in st.items():          # 镜像被删了也要补
            if ns != GLOBAL_NAMESPACE and ns not in out and not (self.root / safe_name(ns)).exists():
                out[ns] = seq
        return out

    def run_once(self) -> list[dict]:
        done = []
        with self._lock:
            st = self._load_state()
            for ns, seq in sorted(self.changed_namespaces().items()):
                try:
                    info = publish(self.repo, ns, self.root)
                    st[ns] = max(seq, info["watermark"])
                    self._save_state(st)
                    done.append(info)
                    self.log.info("明文镜像已更新: %s 水位 %s", ns, info["watermark"])
                except Exception:  # noqa: BLE001  镜像失败不影响服务
                    self.log.exception("明文镜像生成失败: %s", ns)
        return done

    # ---- 线程
    def notify(self) -> None:
        now = time.monotonic()
        if not self._event.is_set():
            self._first = now
        self._last = now
        self._event.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._event.wait(timeout=1.0):
                continue
            while not self._stop.is_set():
                now = time.monotonic()
                if now - self._last >= self.quiet or now - self._first >= self.max_delay:
                    break
                self._stop.wait(0.5)
            self._event.clear()
            self.run_once()

    def start(self) -> "BrowsePublisher":
        import threading

        self._thread = threading.Thread(target=self._loop, name="library-browse", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
