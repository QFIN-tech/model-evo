"""v2.18.0 经验库:LibraryStore —— 提交、水位、写入与只读视图。

- 写:每次写操作一个数据库事务,先插 ``commits`` 拿到 seq,再插事实行。
  SQLite 用 ``BEGIN IMMEDIATE`` 串行化写者;PostgreSQL 在事务开头取同一个
  advisory xact lock,保证 seq 分配顺序 = 提交顺序(否则读者可能先看到 seq=n、
  后看到 seq=n-1,水位读不可复现)。
- 读:所有查询接受 ``at``(水位),只看 ``commit_seq <= at`` 的行;缺省取当前水位。
- 视图(榜单、固化 baseline、trial 最新行)在 Python 侧计算:库规模在千行量级,
  方言无关比 SQL 窗口函数更重要。
"""
from __future__ import annotations

import datetime as _dt
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

from sqlalchemy import (and_, create_engine, event, func, insert, or_, select,
                        text)
from sqlalchemy.pool import StaticPool

from . import schema as S
from .snapshot import (GOVERNANCE_COLUMNS, MEASUREMENT_COLUMNS, RECIPE_COLUMNS,
                       TRIAL_COLUMNS, parse_atid, snapshot_id,
                       validate_snapshot)

#: PostgreSQL 写串行化用的 advisory lock 键(任意常量)
_PG_WRITE_LOCK = 0x6D65766F  # "mevo"
#: 榜单默认剔除树数过大的配方(与旧 MAX_TREES 一致)
MAX_TREES = 8000
SUBJECT_TYPES = ("recipe", "principle", "snapshot", "topic", "atid")
SNAPSHOT_STATES = ("submitted", "merged", "merge_failed", "reverted")


class LibraryValidationError(ValueError):
    """写入内容不合契约。``errors`` 为逐条原因。"""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:5]) + (" …" if len(errors) > 5 else ""))
        self.errors = errors


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def clean_json(obj):
    """NaN / inf → None(PostgreSQL JSONB 不接受),元组 → 列表。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    return obj


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int(v):
    f = _num(v)
    return int(f) if f is not None else None


def _bool(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return bool(v)


def make_engine(url: str):
    if url.startswith("sqlite"):
        from sqlalchemy.engine import make_url

        db = make_url(url).database
        if db and db != ":memory:":
            Path(db).parent.mkdir(parents=True, exist_ok=True)
        kw: dict = {"connect_args": {"timeout": 60, "check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kw["poolclass"] = StaticPool
        engine = create_engine(url, **kw)

        @event.listens_for(engine, "connect")
        def _on_connect(dbapi_conn, _rec):  # noqa: ANN001
            # 由 SQLAlchemy 的 begin 事件发 BEGIN(pysqlite 自带的隐式事务
            # 不会对 SELECT 开事务,也无法发 BEGIN IMMEDIATE)
            dbapi_conn.isolation_level = None
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            # 回滚日志模式:WAL 依赖共享内存,不适合网络文件系统(CephFS)
            cur.execute("PRAGMA journal_mode=DELETE")
            cur.execute("PRAGMA synchronous=FULL")
            cur.execute("PRAGMA busy_timeout=60000")
            cur.close()

        @event.listens_for(engine, "begin")
        def _on_begin(conn):  # noqa: ANN001
            conn.exec_driver_sql(
                conn.get_execution_options().get("sqlite_begin", "BEGIN"))
        return engine
    return create_engine(url, pool_pre_ping=True)


def _rows(result) -> list[dict]:
    return [dict(r._mapping) for r in result]


def _api_tuple(v) -> tuple:
    """契约版本 "2.16" → (2, 16);非数字段按 0。"""
    return tuple(int("".join(ch for ch in p if ch.isdigit()) or 0) for p in str(v).split("."))


def _latest(rows: Iterable[dict], key: Callable[[dict], object]) -> dict:
    """按 key 分组,每组取 (commit_seq, id) 最大的一行。"""
    out: dict = {}
    for r in rows:
        k = key(r)
        cur = out.get(k)
        if cur is None or (r["commit_seq"], r.get("id", 0)) > \
                (cur["commit_seq"], cur.get("id", 0)):
            out[k] = r
    return out


class LibraryStore:
    def __init__(self, url: str, *, migrate: bool = True):
        self.url = url
        self.engine = make_engine(url)
        self.dialect = self.engine.dialect.name
        if migrate:
            from .migrate import upgrade

            upgrade(self.engine)

    # ------------------------------------------------------------------ 事务
    @contextmanager
    def _write(self):
        with self.engine.connect() as conn:
            if self.dialect == "sqlite":
                conn = conn.execution_options(sqlite_begin="BEGIN IMMEDIATE")
            with conn.begin():
                if self.dialect == "postgresql":
                    conn.execute(text("SELECT pg_advisory_xact_lock(:k)"),
                                 {"k": _PG_WRITE_LOCK})
                yield conn

    @contextmanager
    def _read(self):
        with self.engine.connect() as conn:
            yield conn

    @staticmethod
    def _new_commit(conn, *, namespace: str, kind: str, actor: dict,
                    message: str = "", ref: str | None = None) -> int:
        res = conn.execute(insert(S.commits).values(
            namespace=namespace, kind=kind, actor=clean_json(actor or {}),
            message=message or "", ref=ref, created_at=now_iso()))
        return int(res.inserted_primary_key[0])

    @staticmethod
    def _insert(conn, table, rows: list[dict], *, namespace: str | None,
                seq: int) -> None:
        if not rows:
            return
        cols = [c.name for c in table.columns if c.name != "id"]
        full = []
        for r in rows:
            d = {c: r.get(c) for c in cols}
            if "namespace" in cols:
                d["namespace"] = namespace
            d["commit_seq"] = seq
            full.append(d)
        conn.execute(insert(table), full)

    # ------------------------------------------------------------------ 水位
    def watermark(self) -> int:
        with self._read() as conn:
            v = conn.execute(select(func.max(S.commits.c.seq))).scalar()
        return int(v or 0)

    def _at(self, at: int | None) -> int:
        return self.watermark() if at is None else int(at)

    def schema_revision(self) -> str | None:
        try:
            with self._read() as conn:
                return conn.execute(text(
                    "SELECT version_num FROM alembic_version")).scalar()
        except Exception:  # noqa: BLE001  未经 Alembic 建表
            return None

    def health(self) -> dict:
        return {"backend": self.dialect, "watermark": self.watermark(),
                "schema_revision": self.schema_revision()}

    def commits(self, since: int = 0, limit: int = 100) -> list[dict]:
        with self._read() as conn:
            return _rows(conn.execute(
                select(S.commits).where(S.commits.c.seq > since)
                .order_by(S.commits.c.seq).limit(limit)))

    # ------------------------------------------------------------------ 指纹
    def _alias_map(self, conn, at: int) -> dict[str, str]:
        rows = _rows(conn.execute(select(S.fingerprint_aliases).where(
            S.fingerprint_aliases.c.commit_seq <= at)))
        return {k: r["canonical_atid"] for k, r in
                _latest(rows, lambda r: r["alias_atid"]).items()}

    @staticmethod
    def _canonical(atid: str, amap: dict[str, str]) -> str:
        seen = set()
        while atid in amap and amap[atid] != atid and atid not in seen:
            seen.add(atid)
            atid = amap[atid]
        return atid

    def _atid_family(self, conn, atid: str, at: int) -> set[str]:
        """atid 归一后的规范指纹,及所有指向它的别名。"""
        amap = self._alias_map(conn, at)
        canon = self._canonical(atid, amap)
        fam = {canon, atid}
        fam.update(a for a in amap if self._canonical(a, amap) == canon)
        return fam

    def resolve_atid(self, atid: str, at: int | None = None) -> str:
        with self._read() as conn:
            return self._canonical(atid, self._alias_map(conn, self._at(at)))

    def add_fingerprint_alias(self, alias_atid: str, canonical_atid: str, *,
                              reason: str, actor: dict) -> int:
        """登记指纹别名;``canonical == alias`` 表示撤销该别名。"""
        if not alias_atid or not canonical_atid:
            raise LibraryValidationError(["alias_atid / canonical_atid 不能为空"])
        with self._write() as conn:
            amap = self._alias_map(conn, 1 << 62)
            amap[alias_atid] = canonical_atid
            node, seen = alias_atid, set()
            while node in amap and amap[node] != node:
                if node in seen:
                    raise LibraryValidationError(
                        [f"别名成环: {alias_atid} → {canonical_atid}"])
                seen.add(node)
                node = amap[node]
            seq = self._new_commit(conn, namespace="*", kind="alias", actor=actor,
                                   message=f"{alias_atid} → {canonical_atid}",
                                   ref=alias_atid)
            self._insert(conn, S.fingerprint_aliases, [{
                "alias_atid": alias_atid, "canonical_atid": canonical_atid,
                "reason": reason or ""}], namespace=None, seq=seq)
            self._ensure_fingerprints(conn, [alias_atid, canonical_atid], seq)
        return seq

    def _ensure_fingerprints(self, conn, atids: Iterable[str], seq: int) -> None:
        want = {a for a in atids if a}
        if not want:
            return
        have = set(conn.execute(select(S.fingerprints.c.atid).where(
            S.fingerprints.c.atid.in_(sorted(want)))).scalars())
        rows = []
        for a in sorted(want - have):
            p = parse_atid(a) or {"atid": a}
            rows.append({k: p.get(k) for k in
                         ("atid", "model_type", "target_name", "sample_scale",
                          "col_hash8", "data_base")})
        self._insert(conn, S.fingerprints, rows, namespace=None, seq=seq)

    def fingerprints(self, at: int | None = None) -> list[dict]:
        at = self._at(at)
        with self._read() as conn:
            return _rows(conn.execute(select(S.fingerprints).where(
                S.fingerprints.c.commit_seq <= at).order_by(S.fingerprints.c.atid)))

    # ------------------------------------------------------------------ 快照
    def submit_snapshot(self, payload: dict, *, namespace: str, actor: dict,
                        artifact_exists: Callable[[str], bool] | None = None
                        ) -> dict:
        """提交快照。同一内容(snapshot_id)重复提交幂等,返回原回执。"""
        payload = clean_json(payload)
        errors = validate_snapshot(payload)
        if not namespace:
            errors.append("namespace 不能为空")
        if errors:
            raise LibraryValidationError(errors)
        sid = snapshot_id(payload)
        recipes = payload.get("recipes") or []
        if artifact_exists is not None:
            missing = [r["recipe_id"] for r in recipes
                       if not artifact_exists(r["recipe_id"])]
            if missing:
                raise LibraryValidationError(
                    [f"配方 bundle 未上传: {m}" for m in missing])
        run = payload.get("run") or {}
        with self._write() as conn:
            prev = conn.execute(select(S.snapshots.c.commit_seq).where(and_(
                S.snapshots.c.namespace == namespace,
                S.snapshots.c.snapshot_id == sid))).scalar()
            if prev is not None:
                return {"snapshot_id": sid, "commit_seq": int(prev),
                        "duplicate": True}
            seq = self._new_commit(
                conn, namespace=namespace, kind="snapshot", actor=actor, ref=sid,
                message=f"{payload['kind']} {run.get('run_id') or ''}".strip())
            counts = self._write_snapshot_rows(conn, payload, sid, namespace, seq)
        return {"snapshot_id": sid, "commit_seq": seq, "duplicate": False,
                "counts": counts}

    def _write_snapshot_rows(self, conn, payload: dict, sid: str,
                             ns: str, seq: int) -> dict:
        run = payload.get("run") or {}
        big = ("discoveries", "trial_outcomes", "recipes", "measurements",
               "governance")
        meta = {k: v for k, v in payload.items() if k not in big}
        self._insert(conn, S.snapshots, [{
            "snapshot_id": sid, "kind": payload["kind"],
            "run_id": run.get("run_id"), "run_dir": run.get("run_dir"),
            "atid": run.get("atid"), "meta": meta}], namespace=ns, seq=seq)
        self._insert(conn, S.snapshot_states, [{
            "snapshot_id": sid, "state": "submitted", "detail": {}}],
            namespace=ns, seq=seq)
        self._insert(conn, S.discoveries, [{
            "snapshot_id": sid, "local_id": str(d["local_id"]),
            "payload": d["payload"]} for d in payload.get("discoveries") or []],
            namespace=ns, seq=seq)

        trials = []
        for t in payload.get("trial_outcomes") or []:
            row = {c: t.get(c) for c in TRIAL_COLUMNS}
            row["atid"] = row["atid"] or run.get("atid")
            for c in ("metric_value", "delta_parent", "delta_parent_ci_low",
                      "delta_parent_ci_high", "p_positive", "delta_baseline"):
                row[c] = _num(row[c])
            row["generation"] = _int(row["generation"])
            row["higher_is_better"] = _bool(row["higher_is_better"])
            row["snapshot_id"] = sid
            row["extra"] = {k: v for k, v in t.items() if k not in TRIAL_COLUMNS} or None
            trials.append(row)
        self._insert(conn, S.trial_outcomes, trials, namespace=ns, seq=seq)

        have = set(conn.execute(select(S.recipes.c.recipe_id).where(
            S.recipes.c.namespace == ns)).scalars())
        new_recipes, recipe_links, skipped = [], [], []
        for r in payload.get("recipes") or []:
            row = {c: r.get(c) for c in RECIPE_COLUMNS}
            row["atid"] = row["atid"] or run.get("atid")
            # 归属单独记:同一 recipe_id 在第二个任务里出现时,配方行不重插,但本任务
            # 必须拿到自己的归属(否则该任务固化不了 baseline、榜单缺行)
            if row["atid"]:
                recipe_links.append({"recipe_id": r["recipe_id"], "atid": row["atid"],
                                     "kind": row["kind"]})
            if r["recipe_id"] in have:
                skipped.append(r["recipe_id"])
                continue
            have.add(r["recipe_id"])
            row["is_hpo"] = bool(_bool(row["is_hpo"]))
            row["n_trees"] = _int(row["n_trees"])
            row["meta"] = {k: v for k, v in r.items() if k not in RECIPE_COLUMNS}
            new_recipes.append(row)
        self._insert(conn, S.recipes, new_recipes, namespace=ns, seq=seq)
        self._write_recipe_atids(conn, recipe_links, namespace=ns, seq=seq)

        meas = []
        for m in payload.get("measurements") or []:
            if m["recipe_id"] not in have:
                raise LibraryValidationError(
                    [f"测量引用了未登记的配方: {m['recipe_id']}"])
            row = {c: m.get(c) for c in MEASUREMENT_COLUMNS}
            for c in ("value", "baseline_value", "delta_baseline", "gain",
                      "ci_low", "ci_high", "p_positive"):
                row[c] = _num(row[c])
            for c in ("higher_is_better", "confirmed"):
                row[c] = _bool(row[c])
            if row["gain"] is None and row["delta_baseline"] is not None and \
                    row["higher_is_better"] is not None:
                row["gain"] = row["delta_baseline"] if row["higher_is_better"] \
                    else -row["delta_baseline"]
            row["extra"] = {k: v for k, v in m.items()
                            if k not in MEASUREMENT_COLUMNS} or None
            meas.append(row)
        self._insert(conn, S.measurements, meas, namespace=ns, seq=seq)

        gov = []
        for g in payload.get("governance") or []:
            self._check_governance(g)
            gov.append({c: g.get(c) for c in GOVERNANCE_COLUMNS})
            gov[-1]["payload"] = g.get("payload") or {}
            gov[-1]["reason"] = g.get("reason") or ""
        self._insert(conn, S.governance_events, gov, namespace=ns, seq=seq)

        atids = [run.get("atid")] + [t.get("atid") for t in trials] + \
            [r.get("atid") for r in new_recipes]
        self._ensure_fingerprints(conn, atids, seq)
        n_profile = self._write_profile(conn, run.get("atid"), run.get("task_profile"), seq)

        # v2.18.1:检索反馈落表(此前只存在快照 meta 里,两张表恒为空)。
        # 遥测文件逐 run 追加:重新导出快照会带上旧条目,按内容哈希跳过已入库的行;
        # 依据用 payload 里的原始 trial 行(extra 里也有 basis),按 (原则, run, trial) 去重。
        from .feedback import application_rows, new_log_rows, retrieval_rows

        logs = new_log_rows(conn, S, ns, retrieval_rows(payload))
        apps = application_rows(trials)
        if apps:
            have_apps = {tuple(r) for r in conn.execute(select(
                S.applications.c.experience_id, S.applications.c.run_id,
                S.applications.c.trial_id).where(and_(
                    S.applications.c.namespace == ns,
                    S.applications.c.run_id.in_(sorted({a["run_id"] for a in apps})))))}
            apps = [a for a in apps
                    if (a["experience_id"], a["run_id"], a["trial_id"]) not in have_apps]
        self._insert(conn, S.retrieval_logs, clean_json(logs), namespace=ns, seq=seq)
        self._insert(conn, S.applications, clean_json(apps), namespace=ns, seq=seq)
        return {"discoveries": len(payload.get("discoveries") or []),
                "trial_outcomes": len(trials), "recipes": len(new_recipes),
                "recipes_already_present": len(skipped),
                "measurements": len(meas), "governance": len(gov),
                "retrieval_logs": len(logs), "applications": len(apps),
                "task_profile": n_profile}

    # ------------------------------------------------------------------ 任务画像
    def _write_profile(self, conn, atid: str | None, profile, seq: int) -> int:
        """画像按 atid 追加;与最新一行内容相同时不写。"""
        if not atid or not isinstance(profile, dict) or not profile:
            return 0
        from .artifacts import canonical_json, sha256_hex

        h = sha256_hex(canonical_json(clean_json(profile)))
        last = conn.execute(select(S.fingerprint_profiles.c.profile_hash).where(
            S.fingerprint_profiles.c.atid == atid).order_by(
            S.fingerprint_profiles.c.id.desc()).limit(1)).scalar()
        if last == h:
            return 0
        self._insert(conn, S.fingerprint_profiles, [{
            "atid": atid, "profile": clean_json(profile), "profile_hash": h}],
            namespace=None, seq=seq)
        return 1

    def put_task_profile(self, atid: str, profile: dict, *, actor: dict) -> int | None:
        """单独登记 / 更新一个任务画像(存量任务回填用);内容未变时不提交,返回 None。"""
        if not atid or not isinstance(profile, dict) or not profile:
            raise LibraryValidationError(["atid 与 profile 必填"])
        from .artifacts import canonical_json, sha256_hex

        h = sha256_hex(canonical_json(clean_json(profile)))
        with self._write() as conn:
            last = conn.execute(select(S.fingerprint_profiles.c.profile_hash).where(
                S.fingerprint_profiles.c.atid == atid).order_by(
                S.fingerprint_profiles.c.id.desc()).limit(1)).scalar()
            if last == h:
                return None
            seq = self._new_commit(conn, namespace="*", kind="profile", actor=actor,
                                   message=f"task profile {atid}", ref=atid)
            self._write_profile(conn, atid, profile, seq)
            self._ensure_fingerprints(conn, [atid], seq)
        return seq

    def fingerprint_profiles(self, at: int | None = None) -> dict[str, dict]:
        """{atid: 最新画像}(水位内)。"""
        at = self._at(at)
        with self._read() as conn:
            rows = _rows(conn.execute(select(S.fingerprint_profiles).where(
                S.fingerprint_profiles.c.commit_seq <= at).order_by(
                S.fingerprint_profiles.c.id)))
        return {r["atid"]: r["profile"] for r in rows}

    def set_snapshot_state(self, namespace: str, sid: str, state: str, *,
                           actor: dict, detail: dict | None = None) -> int:
        if state not in SNAPSHOT_STATES:
            raise LibraryValidationError([f"未知快照状态: {state}"])
        with self._write() as conn:
            exists = conn.execute(select(S.snapshots.c.id).where(and_(
                S.snapshots.c.namespace == namespace,
                S.snapshots.c.snapshot_id == sid))).scalar()
            if exists is None:
                raise LibraryValidationError([f"快照不存在: {sid}"])
            seq = self._new_commit(conn, namespace=namespace, kind="snapshot_state",
                                   actor=actor, ref=sid, message=state)
            self._insert(conn, S.snapshot_states, [{
                "snapshot_id": sid, "state": state,
                "detail": clean_json(detail or {})}], namespace=namespace, seq=seq)
        return seq

    def list_snapshots(self, namespace: str, *, kind: str | None = None,
                       state: str | None = None, run_id: str | None = None,
                       at: int | None = None) -> list[dict]:
        at = self._at(at)
        with self._read() as conn:
            q = select(S.snapshots).where(and_(
                S.snapshots.c.namespace == namespace,
                S.snapshots.c.commit_seq <= at))
            if kind:
                q = q.where(S.snapshots.c.kind == kind)
            if run_id:
                q = q.where(S.snapshots.c.run_id == run_id)
            snaps = _rows(conn.execute(q.order_by(S.snapshots.c.commit_seq)))
            states = _latest(_rows(conn.execute(select(S.snapshot_states).where(and_(
                S.snapshot_states.c.namespace == namespace,
                S.snapshot_states.c.commit_seq <= at)))),
                lambda r: r["snapshot_id"])
        for s in snaps:
            st = states.get(s["snapshot_id"])
            s["state"] = st["state"] if st else None
            s["state_detail"] = st["detail"] if st else None
        if state:
            snaps = [s for s in snaps if s["state"] == state]
        return snaps

    def get_snapshot(self, namespace: str, sid: str, *,
                     at: int | None = None) -> dict | None:
        at = self._at(at)
        snaps = [s for s in self.list_snapshots(namespace, at=at)
                 if s["snapshot_id"] == sid]
        if not snaps:
            return None
        snap = snaps[0]
        with self._read() as conn:
            snap["discoveries"] = _rows(conn.execute(select(S.discoveries).where(and_(
                S.discoveries.c.namespace == namespace,
                S.discoveries.c.snapshot_id == sid)).order_by(S.discoveries.c.id)))
            snap["n_trial_outcomes"] = conn.execute(select(func.count()).where(and_(
                S.trial_outcomes.c.namespace == namespace,
                S.trial_outcomes.c.snapshot_id == sid))).scalar()
        return snap

    # ------------------------------------------------------------------ 证据
    def trial_outcomes(self, namespace: str, *, atid: str | None = None,
                       run_id: str | None = None, at: int | None = None
                       ) -> list[dict]:
        """每个 (run_id, trial_id) 的最新一行(历史 run 可重复增量导入)。"""
        at = self._at(at)
        with self._read() as conn:
            q = select(S.trial_outcomes).where(and_(
                S.trial_outcomes.c.namespace == namespace,
                S.trial_outcomes.c.commit_seq <= at))
            if atid:
                q = q.where(S.trial_outcomes.c.atid.in_(
                    sorted(self._atid_family(conn, atid, at))))
            if run_id:
                q = q.where(S.trial_outcomes.c.run_id == run_id)
            rows = _rows(conn.execute(q))
        latest = _latest(rows, lambda r: (r["run_id"], r["trial_id"]))
        return sorted(latest.values(), key=lambda r: (r["run_id"], r["trial_id"]))

    # ------------------------------------------------------------------ 配方
    @staticmethod
    def _write_recipe_atids(conn, links: list[dict], *, namespace: str, seq: int) -> None:
        """登记配方归属;已存在的 (recipe, atid, kind) 不重复写。"""
        if not links:
            return
        have = {(r["recipe_id"], r["atid"], r["kind"]) for r in _rows(conn.execute(
            select(S.recipe_atids.c.recipe_id, S.recipe_atids.c.atid, S.recipe_atids.c.kind)
            .where(S.recipe_atids.c.namespace == namespace)))}
        new = []
        for lk in links:
            key = (lk["recipe_id"], lk["atid"], lk["kind"])
            if key not in have:
                have.add(key)
                new.append(lk)
        LibraryStore._insert(conn, S.recipe_atids, new, namespace=namespace, seq=seq)

    def recipe_atids(self, namespace: str, *, recipe_id: str | None = None,
                     at: int | None = None) -> list[dict]:
        at = self._at(at)
        with self._read() as conn:
            q = select(S.recipe_atids).where(and_(
                S.recipe_atids.c.namespace == namespace,
                S.recipe_atids.c.commit_seq <= at))
            if recipe_id:
                q = q.where(S.recipe_atids.c.recipe_id == recipe_id)
            return _rows(conn.execute(q.order_by(S.recipe_atids.c.commit_seq,
                                                 S.recipe_atids.c.id)))

    def recipes(self, namespace: str, *, atid: str | None = None,
                kind: str | None = None, at: int | None = None) -> list[dict]:
        """配方(可按任务过滤)。归属走 ``recipe_atids``:同一配方在多个任务里各有一行,
        因此同一 ``recipe_id`` 可能返回多次(每次带该任务的 atid),调用方按 recipe_id
        去重即可。"""
        at = self._at(at)
        with self._read() as conn:
            q = select(S.recipes).where(and_(
                S.recipes.c.namespace == namespace,
                S.recipes.c.commit_seq <= at))
            if atid:
                fam = sorted(self._atid_family(conn, atid, at))
                linked = select(S.recipe_atids.c.recipe_id).where(and_(
                    S.recipe_atids.c.namespace == namespace,
                    S.recipe_atids.c.atid.in_(fam),
                    S.recipe_atids.c.commit_seq <= at))
                # atid 为空的旧配方仍按 recipes.atid 兜底(0004 之前的行已回填,这里
                # 是防御:迁移未跑或 meta 缺 atid 时不至于整条配方消失)
                q = q.where(or_(S.recipes.c.recipe_id.in_(linked),
                                S.recipes.c.atid.in_(fam)))
            if kind:
                q = q.where(S.recipes.c.kind == kind)
            rows = _rows(conn.execute(q.order_by(S.recipes.c.commit_seq,
                                                 S.recipes.c.id)))
        if not atid:
            return rows
        # 同一配方在该任务下只保留一行(按 (recipe_id, kind) 去重)
        seen, out = set(), []
        for r in rows:
            key = (r["recipe_id"], r["kind"])
            if key in seen:
                continue
            seen.add(key)
            out.append({**r, "atid": atid})
        return out

    def measurements(self, namespace: str, *, recipe_ids: Iterable[str] | None = None,
                     at: int | None = None) -> list[dict]:
        at = self._at(at)
        with self._read() as conn:
            q = select(S.measurements).where(and_(
                S.measurements.c.namespace == namespace,
                S.measurements.c.commit_seq <= at))
            if recipe_ids is not None:
                q = q.where(S.measurements.c.recipe_id.in_(list(recipe_ids)))
            return _rows(conn.execute(q.order_by(S.measurements.c.commit_seq,
                                                 S.measurements.c.id)))

    def leaderboard(self, namespace: str, atid: str, *, at: int | None = None,
                    top_n: int = 10, max_trees: int = MAX_TREES,
                    runtime_api_version: str | None = None,
                    family_of: Callable[[dict], str | None] | None = None) -> list[dict]:
        """同任务榜单:非 HPO、树数不超限、未被 exclude 的 top_trial 配方,
        每个配方取最新一次测量,按归一化增益 gain 降序。

        **同一配方只占一个名次**。``family_of(recipe) -> key`` 给出配方身份(仓库按
        bundle 内容计算:代码去掉 SEED 行 + 定义模型的配置),相同即同一配方 ——
        覆盖两种重复:旧 top-trials 里同一 trial 的多份副本(被后续 run 热启动后
        RUN_DIR 已改写、内容哈希不同),以及 confirm 子代(只换 seed 复跑父代配方)。
        未给 / 取不到时按来源 trial(source_run_id + source_trial_id)去重。组内保留
        规范化无残留绝对路径的一份,其次增益高者,再次最新提交;``family_size`` 记组内
        配方数。

        给出 ``runtime_api_version`` 时去掉契约版本比它新的配方(当前 runtime 跑不了);
        更旧的配方在渲染副本上硬化,照常保留。"""
        at = self._at(at)
        recs = [r for r in self.recipes(namespace, atid=atid, kind="top_trial", at=at)
                if not r["is_hpo"] and (r["n_trees"] is None or r["n_trees"] <= max_trees)]
        excluded = self._excluded(namespace, "recipe", at)
        recs = [r for r in recs if r["recipe_id"] not in excluded]
        if runtime_api_version:
            from .recipes import runtime_api_compatible  # noqa: PLC0415 (avoid circular at module level)
            recs = [r for r in recs if runtime_api_compatible(
                r.get("runtime_api_version"), runtime_api_version)]
        meas = _latest(self.measurements(
            namespace, recipe_ids=[r["recipe_id"] for r in recs], at=at),
            lambda m: m["recipe_id"])
        groups: dict = {}
        for r in recs:
            key = family_of(r) if family_of else None
            if key is None:
                key = (r.get("source_run_id"), r.get("source_trial_id")) \
                    if r.get("source_trial_id") else ("recipe", r["recipe_id"])
            residual = bool((((r.get("meta") or {}).get("canonicalization") or {})
                             .get("residual_abs_paths")))
            g = (meas.get(r["recipe_id"]) or {}).get("gain")
            order = (not residual, g is not None, g or 0.0, r["commit_seq"], r["id"])
            cur = groups.get(key)
            if cur is None or order > cur[0]:
                groups[key] = (order, r, (cur[2] if cur else 0) + 1)
            else:
                groups[key] = (cur[0], cur[1], cur[2] + 1)
        rows = []
        for _order, r, n in groups.values():
            rows.append({**{k: r[k] for k in (
                "recipe_id", "atid", "algorithm", "source_run_id", "source_trial_id",
                "runtime_api_version", "n_trees", "commit_seq")},
                "measurement": meas.get(r["recipe_id"]), "family_size": n})

        def _gain_key(row):
            g = (row["measurement"] or {}).get("gain")
            return (g is None, -(g or 0.0), row["commit_seq"])

        rows.sort(key=_gain_key)
        for i, row in enumerate(rows[:top_n], 1):
            row["rank"] = i
        return rows[:top_n]

    def frozen_baseline(self, namespace: str, atid: str, algorithm: str, *,
                        at: int | None = None) -> dict | None:
        """固化 baseline:有 pin_baseline 治理事件的最新配方;否则最新 baseline 配方。"""
        at = self._at(at)
        cands = [r for r in self.recipes(namespace, atid=atid, kind="baseline", at=at)
                 if r["algorithm"] == algorithm]
        if not cands:
            return None
        by_id = {r["recipe_id"]: r for r in cands}
        pins = [g for g in self.governance(namespace, subject_type="recipe", at=at)
                if g["action"] == "pin_baseline" and g["subject_id"] in by_id]
        if pins:
            return by_id[pins[-1]["subject_id"]]
        return cands[-1]

    # ------------------------------------------------------------------ 治理
    @staticmethod
    def _check_governance(g: dict) -> None:
        if g.get("subject_type") not in SUBJECT_TYPES:
            raise LibraryValidationError(
                [f"subject_type 必须是 {SUBJECT_TYPES} 之一: {g.get('subject_type')}"])
        if not g.get("subject_id") or not g.get("action"):
            raise LibraryValidationError(["治理事件缺 subject_id / action"])

    def govern(self, namespace: str, events: list[dict], *, actor: dict,
               message: str = "") -> int:
        if not events:
            raise LibraryValidationError(["治理事件为空"])
        for g in events:
            self._check_governance(g)
        with self._write() as conn:
            seq = self._new_commit(conn, namespace=namespace, kind="governance",
                                   actor=actor, message=message)
            self._insert(conn, S.governance_events, [{
                **{c: g.get(c) for c in GOVERNANCE_COLUMNS},
                "payload": clean_json(g.get("payload") or {}),
                "reason": g.get("reason") or ""} for g in events],
                namespace=namespace, seq=seq)
        return seq

    def governance(self, namespace: str, *, subject_type: str | None = None,
                   subject_id: str | None = None, at: int | None = None) -> list[dict]:
        at = self._at(at)
        with self._read() as conn:
            q = select(S.governance_events).where(and_(
                S.governance_events.c.namespace == namespace,
                S.governance_events.c.commit_seq <= at))
            if subject_type:
                q = q.where(S.governance_events.c.subject_type == subject_type)
            if subject_id:
                q = q.where(S.governance_events.c.subject_id == subject_id)
            return _rows(conn.execute(q.order_by(S.governance_events.c.commit_seq,
                                                 S.governance_events.c.id)))

    def _excluded(self, namespace: str, subject_type: str, at: int) -> set[str]:
        evs = [g for g in self.governance(namespace, subject_type=subject_type, at=at)
               if g["action"] in ("exclude", "restore")]
        last = _latest(evs, lambda g: g["subject_id"])
        return {k for k, g in last.items() if g["action"] == "exclude"}
