"""按 run 目录重新推出旧 run 的任务指纹,修正 atid 为空 / 笼统的 trial 行与指纹别名。

判定规则(每个 atid 一个动作,全部需要 ``reason`` 写清依据):

- ``alias``     :fingerprints 行与目标列集相同(别名行不删除,只登记别名);
- ``retag``     :trial 行的 atid 写错 / 为空,原地修正(唯一键不允许追加第二行;
                  旧值写进 ``extra.atid_fix`` 与变更日志,可逐条还原);随后 ``relink``
                  按新指纹重算证据关联;
- ``drop_alias``:已经指向正确规范的别名,标记 ``canonical == alias`` 撤销;
- ``note``      :无法自动判定的(其他用户 run 目录不可读等),只写日志。

阈值:**只有列集完全相同才自动 retag / alias**(不靠相似度猜);其它情形一律只报告。
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import yaml

from .. import curation
from .. import schema as S
from ..snapshot import parse_atid, run_id_for
from ..store import LibraryValidationError, _latest, _rows, clean_json
from sqlalchemy import and_, select

_SCALES = (("lt10w", 100_000), ("10w_50w", 500_000), ("50w_200w", 2_000_000),
           ("200w_1000w", 10_000_000))


def _read_json(p: Path):
    import json

    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_yaml(p: Path):
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _parquet_columns(path: Path) -> list[str] | None:
    try:
        import pyarrow.parquet as pq

        return sorted(str(c) for c in pq.read_schema(str(path)).names)
    except Exception:  # noqa: BLE001
        return None


def col_hash(cols: list[str] | None) -> str | None:
    if not cols:
        return None
    return hashlib.sha256("|".join(sorted(cols)).encode("utf-8")).hexdigest()[:8]


def scale_bucket(rows: int | None) -> str | None:
    if not rows:
        return None
    for name, bound in _SCALES:
        if rows < bound:
            return name
    return "ge1000w"


def _native_columns(run_dir: Path) -> tuple[list[str] | None, str | None, str]:
    """原生全量面的列集:run 自己的 data/ 分片 → splits 兜底。返回 (列, 来源, 类别),
    类别是 ``native`` / ``splits`` / ``none`` —— 不能靠路径里含 "splits" 判断(临时目录名
    里就可能有)。

    profile 里存的是**原生面**的哈希与列数;`data/` 目录不在(别的机器采样)时拿不到
    原生列集,splits 的列集比原生面少(去掉泄漏 / 不可用列),哈希通常不同 —— 此时只用
    它核对,不能拿它当指纹。
    """
    cands: list[Path] = []
    data = run_dir / "data"
    if data.is_dir():
        cands += sorted(data.glob("*.parquet"))[:1]
        cands += sorted(data.rglob("train_sample_*.parquet"))[:1]
    # 与 spec.resolve_abstract_task_id 同口径:原始分片可能在上一 / 两级目录
    for anc in (run_dir, run_dir.parent, run_dir.parent.parent):
        if (anc / "data").is_dir():
            cands += sorted((anc / "data").glob("train_400w-part-*.parquet"))[:1]
    for c in cands:
        cols = _parquet_columns(c)
        if cols:
            return cols, str(c), "native"
    sp = run_dir / "sample-profile" / "splits" / "train.parquet"
    if sp.is_file():
        return _parquet_columns(sp), str(sp) + " (splits)", "splits"
    return None, None, "none"


def _parquet_rows(paths) -> int:
    n = 0
    for p in paths:
        try:
            import pyarrow.parquet as pq

            n += pq.read_metadata(str(p)).num_rows
        except Exception:  # noqa: BLE001
            continue
    return n


def _rows_estimate(run_dir: Path) -> tuple[int | None, str | None, bool]:
    """(行数, 来源, 是否只是下界)。profile 没记 n_rows 时:run 的 data/ 物化样本行数;
    再不行用 splits 各切分行数之和 —— 那是下界(切分前会过滤),只在落入最高档时可用。"""
    data = run_dir / "data"
    if data.is_dir():
        files = sorted(data.rglob("*.parquet"))
        if files:
            n = _parquet_rows(files)
            if n:
                return n, f"data/ 下 {len(files)} 个 parquet", False
    sp = run_dir / "sample-profile" / "splits"
    if sp.is_dir():
        n = _parquet_rows(sorted(sp.glob("*.parquet")))
        if n:
            return n, "splits 行数之和(下界)", True
    return None, None, False


def derive_atid(run_dir) -> dict:
    """由 run 目录推出可信 atid 与依据。不写任何东西。"""
    r = Path(run_dir)
    out = {"run_dir": str(r), "atid": None, "evidence": {}, "problems": []}
    rs = _read_json(r / "run-state.json") or {}
    spec = _read_yaml(r / "task-spec.yaml")
    prof = _read_json(r / "sample-profile" / "sample-profile.json") or \
        _read_json(r / "sample-profile" / "profile.json") or {}
    mt = {"binary_classification": "binary", "regression": "reg",
          "uplift": "uplift"}.get(str(spec.get("model_type") or ""), "cls")
    target = str((spec.get("target") or {}).get("name") or "unknown")
    target = "".join(c if (c.isalnum() or c in "-_") else "-" for c in target)
    rows = prof.get("n_rows") or prof.get("total_rows")
    rows_src = "sample-profile.n_rows" if rows else None
    scale = scale_bucket(int(rows)) if rows else None
    if not rows:
        rows, rows_src, lower = _rows_estimate(r)
        scale = scale_bucket(rows) if rows else None
        if lower and scale != "ge1000w":
            scale = None          # 下界只能确定最高档
    cols, col_src, src_kind = _native_columns(r)
    ch = col_hash(cols)
    stored = prof.get("raw_columns_hash")
    if ch and src_kind == "splits":
        # splits 不是原生面(切分前会去掉泄漏 / 不可用列):只记下来供 plan 做
        # "splits 列集 → 原生面"的对照推断,本身不能当指纹
        out["evidence"]["splits_col_hash8"] = ch
        cols, ch, col_src = None, None, None
    if not ch and stored:
        ch, col_src = stored, "profile.raw_columns_hash"
        cols = None
    out["evidence"] = {"run_state_atid": rs.get("abstract_task_id"),
                       "target": target, "n_rows": rows, "rows_source": rows_src,
                       "scale": scale,
                       "col_hash8": ch, "columns_source": col_src,
                       "n_columns": len(cols) if cols else prof.get("raw_columns_count"),
                       "raw_columns_hash": stored,
                       "skill_version": rs.get("skill_version")}
    sp = r / "sample-profile" / "splits" / "train.parquet"
    if sp.is_file() and "splits_col_hash8" not in out["evidence"]:
        out["evidence"]["splits_col_hash8"] = col_hash(_parquet_columns(sp))
    if not scale:
        out["problems"].append("量级不可得(n_rows 缺失)")
    if not ch:
        out["problems"].append("列集不可得")
    out["prefix"] = f"{mt}-{target}-{scale}" if scale else None
    if rs.get("abstract_task_id"):
        out["atid"] = rs["abstract_task_id"]
    elif not out["problems"]:
        out["atid"] = f"{mt}-{target}-{scale}-{ch}"
    return out


def _infer_from_splits(derived: dict) -> None:
    """只有 splits 列集的 run:若库里**所有**同时知道原生面与 splits 的 run,其 splits
    列集 X 都对应同一个原生面 Y,则 X → Y(同一份数据的同一建模面)。不唯一就不推。"""
    pairs: dict[str, set] = defaultdict(set)
    for d in derived.values():
        ev = d["evidence"]
        if ev.get("col_hash8") and ev.get("splits_col_hash8"):
            pairs[ev["splits_col_hash8"]].add(ev["col_hash8"])
    for d in derived.values():
        ev = d["evidence"]
        if d["atid"] or ev.get("col_hash8") or not d.get("prefix"):
            continue
        native = pairs.get(ev.get("splits_col_hash8") or "")
        if native and len(native) == 1:
            n = next(iter(native))
            k = sum(1 for x in derived.values()
                    if x["evidence"].get("splits_col_hash8") == ev["splits_col_hash8"]
                    and x["evidence"].get("col_hash8") == n)
            ev.update({"col_hash8": n, "columns_source":
                       f"splits 列集 {ev['splits_col_hash8']} 与 {k} 个已知原生面为 {n} 的 run "
                       f"的 splits 列集相同"})
            d["problems"] = [p for p in d["problems"] if p != "列集不可得"]
            if not d["problems"]:
                d["atid"] = f"{d['prefix']}-{n}"


def plan(store, namespace: str, *, run_dirs: dict[str, str] | None = None,
         dry_run: bool = True) -> dict:
    """比对库内状态,给出修复计划。``run_dirs``:{run_id: run_dir} 覆盖快照里缺失的目录。"""
    at = store.watermark()
    with store._read() as conn:
        snaps = _rows(conn.execute(select(S.snapshots).where(
            S.snapshots.c.namespace == namespace)))
        trials = _rows(conn.execute(select(S.trial_outcomes).where(
            S.trial_outcomes.c.namespace == namespace)))
    dirs: dict[str, str] = dict(run_dirs or {})
    for s in snaps:
        if s["run_id"] and s.get("run_dir"):
            dirs.setdefault(s["run_id"], s["run_dir"])
    derived = {}
    for rid, rd in dirs.items():
        if rd and Path(rd).is_dir():
            derived[rid] = derive_atid(rd)
    _infer_from_splits(derived)
    latest = _latest(trials, lambda r: (r["run_id"], r["trial_id"]))
    with store._read() as conn:
        amap0 = store._alias_map(conn, at)

    def _canon(a):
        return store._canonical(a, amap0) if a else a

    actions, notes = [], []
    for rid, d in sorted(derived.items()):
        if not d["atid"]:
            notes.append({"run_id": rid, "reason": "无法推出指纹", **d["evidence"],
                          "problems": d["problems"]})
            continue
        d["atid"] = _canon(d["atid"])
        # 在规范指纹空间比较:已登记别名的(7580b665 → bc5329dc)不算错
        wrong = sorted({t["trial_id"] for t in latest.values()
                        if t["run_id"] == rid and _canon(t["atid"]) != d["atid"]})
        if wrong:
            froms = sorted({str(t["atid"]) for t in latest.values()
                            if t["run_id"] == rid and t["trial_id"] in wrong})
            actions.append({"action": "retag", "run_id": rid, "to_atid": d["atid"],
                            "from_atids": froms, "trial_ids": wrong, "reason": _reason(d),
                            "evidence": d["evidence"]})
    actions = _prefer_alias(actions, latest, derived, _canon)
    # 指纹:空 atid 的快照对应指纹、或 trial retag 后留下的笼统指纹
    with store._read() as conn:
        fps = store.fingerprints(at=at)
        amap = store._alias_map(conn, at)
    canon = {f["atid"]: store._canonical(f["atid"], amap) for f in fps}
    by_cols: dict[str, set] = defaultdict(set)
    for rid, d in derived.items():
        if d["atid"] and d["evidence"].get("col_hash8"):
            by_cols[d["evidence"]["col_hash8"]].add(d["atid"].rsplit("-", 1)[0])
    for atid in sorted({f["atid"] for f in fps}):
        suffix = atid.rsplit("-", 1)[-1]
        target = (parse_atid(atid) or {}).get("data_base")
        if target is None and suffix in by_cols:
            # 笼统指纹(unknown-unverified 等):指向同目标、同列集的规范指纹
            cands = sorted(by_cols[suffix] | {atid})
            good = [c for c in cands if parse_atid(c) and parse_atid(c).get("data_base")]
            if len(good) == 1:
                actions.append({"action": "alias", "alias_atid": atid,
                                "canonical_atid": canon.get(good[0], good[0]),
                                "reason": f"指纹 {atid} 无可信底座({suffix});同列集 {suffix} "
                                          f"的唯一可信指纹为 {good[0]},trial 已按 run 目录重打",
                                "evidence": {"col_hash8": suffix}})
            else:
                notes.append({"atid": atid, "reason": "笼统指纹,但同列集可信指纹不唯一",
                              "candidates": good})
    return {"watermark": at, "run_dirs": len(dirs), "derived_runs": len(derived),
            "actions": actions, "notes": notes}


def _prefer_alias(actions: list[dict], latest: dict, derived: dict, canon) -> list[dict]:
    """错的是**指纹本身**(同一数据算出的漂移哈希)而不是 trial 归属时,登记别名而不是
    重打 trial:别名同时覆盖引用该指纹的配方、原则与卷宗,与 7580b665 的处理同口径。

    条件:错指纹与正确指纹只差列哈希(同任务类型、同目标、同量级),且库里**所有**带这个
    错指纹的 run 都推出同一个正确指纹。"""
    out, by_from = [], defaultdict(list)
    for a in actions:
        if a["action"] == "retag" and len(a["from_atids"]) == 1 and a["from_atids"][0] != "None":
            by_from[a["from_atids"][0]].append(a)
        else:
            out.append(a)
    for frm, acts in by_from.items():
        to = {a["to_atid"] for a in acts}
        pf, pt = parse_atid(frm) or {}, parse_atid(next(iter(to))) or {}
        runs_with = {t["run_id"] for t in latest.values() if t["atid"] == frm}
        same_task = len(to) == 1 and all(pf.get(k) == pt.get(k) for k in
                                         ("model_type", "target_name", "sample_scale"))
        all_agree = runs_with <= {a["run_id"] for a in acts}
        if same_task and all_agree:
            ev = acts[0]["evidence"]
            out.append({"action": "alias", "alias_atid": frm, "canonical_atid": canon(next(iter(to))),
                        "reason": (f"{frm} 是同一数据的漂移指纹:带它的 run "
                                   f"{sorted(runs_with)} 由原生全量面重新推出 {next(iter(to))}"
                                   f"({ev.get('n_columns')} 列,来源 {ev.get('columns_source')})"),
                        "evidence": {"runs": sorted(runs_with), **ev}})
        else:
            out += acts
    return out


def _reason(d: dict) -> str:
    ev = d["evidence"]
    return (f"run 目录重新推出指纹:target={ev['target']} rows={ev['n_rows']} "
            f"scale={ev['scale']} col_hash8={ev['col_hash8']}"
            f"({ev['n_columns']} 列,来源 {ev['columns_source']});"
            f"run-state 记录的 atid={ev['run_state_atid']}")


def apply(store, namespace: str, actions: list[dict], *, actor: dict,
          reason: str = "") -> dict:
    """执行修复。alias 追加别名行;retag **原地修正** trial 行的 atid。

    retag 是追加写不变式的唯一例外:``trial_outcomes`` 唯一键是
    ``(namespace, snapshot_id, trial_id)``,同一 trial 无法追加第二行;而错的是导入时
    写下的那一行事实。修正把 ``commit_seq`` 推到新提交(水位语义不变),并把旧值
    (整个旧行的 atid 与原因)写进 ``extra.atid_fix`` 与变更日志的 evidence —— 据此可
    逐条还原。每次修复是一条独立提交,带 actor 与理由。
    """
    from sqlalchemy import update

    from .. import evidence

    retagged = aliased = 0
    with store._write() as conn:
        alias_acts = [a for a in actions if a["action"] == "alias"]
        for a in alias_acts:
            seq = store._new_commit(conn, namespace="*", kind="alias", actor=actor,
                                    message=a["reason"], ref=a["alias_atid"])
            store._insert(conn, S.fingerprint_aliases, [{
                "alias_atid": a["alias_atid"], "canonical_atid": a["canonical_atid"],
                "reason": a["reason"]}], namespace=None, seq=seq)
            store._ensure_fingerprints(conn, [a["alias_atid"], a["canonical_atid"]], seq)
            aliased += 1
        retag = [a for a in actions if a["action"] == "retag"]
        fixed_rows = []
        if retag:
            seq = store._new_commit(conn, namespace=namespace, kind="fix",
                                    actor=actor, message=reason or _retag_message(retag))
            for a in retag:
                rows = _rows(conn.execute(select(S.trial_outcomes).where(and_(
                    S.trial_outcomes.c.namespace == namespace,
                    S.trial_outcomes.c.run_id == a["run_id"],
                    S.trial_outcomes.c.trial_id.in_(a["trial_ids"])))))
                has = {(r["snapshot_id"], r["trial_id"]) for r in rows
                       if r["atid"] == a["to_atid"]}
                latest = _latest(rows, lambda r: (r["run_id"], r["trial_id"]))
                for t in latest.values():
                    if t["atid"] == a["to_atid"] or (t["snapshot_id"], t["trial_id"]) in has:
                        continue
                    conn.execute(update(S.trial_outcomes).where(
                        S.trial_outcomes.c.id == t["id"]).values(
                        atid=a["to_atid"], commit_seq=seq,
                        extra=clean_json({**(t.get("extra") or {}),
                                          "atid_fix": {"from": t["atid"],
                                                        "reason": a["reason"]}})))
                    fixed_rows.append({"run_id": t["run_id"], "trial_id": t["trial_id"],
                                       "snapshot_id": t["snapshot_id"],
                                       "from": t["atid"], "to": a["to_atid"]})
                store._ensure_fingerprints(conn, [a["to_atid"]], seq)
            retagged = len(fixed_rows)
            store._insert(conn, S.changelog_entries, [{
                "snapshot_id": None, "entry_type": "fix", "action": "retag_atid",
                "subject_type": "run", "subject_id": a["run_id"],
                "summary": f"按 run 目录重打 {len(a['trial_ids'])} 个 trial 的任务指纹 → "
                           f"{a['to_atid']}", "reason": a["reason"], "evidence": a["evidence"]}
                for a in retag], namespace=namespace, seq=seq)
            store._insert(conn, S.changelog_entries, [{
                "snapshot_id": None, "entry_type": "fix", "action": "retag_atid_row",
                "subject_type": "trial", "subject_id": f"{r['run_id']}/{r['trial_id']}",
                "summary": f"{r['from']} → {r['to']}", "reason": reason or "fix-fingerprints",
                "evidence": r} for r in fixed_rows], namespace=namespace, seq=seq)
    out = {"retagged_trials": retagged, "aliased": aliased}
    if retagged:
        # 证据关联按新指纹重算(unknown → same_task/cross_base),只追加尚不存在的判定
        out["evidence"] = evidence.relink(store, namespace, actor=actor)
    return out


def _retag_message(actions: list[dict]) -> str:
    return "fix: 按 run 目录重打 trial 指纹 — " + "; ".join(
        f"{a['run_id']}→{a['to_atid']}({len(a['trial_ids'])})" for a in actions)
