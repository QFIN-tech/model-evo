"""v2.18.0 经验库:Repository —— 调用方唯一入口(dev 文档 §2)。

- :class:`LocalRepository`:进程内直接读写 LibraryStore + ArtifactStore。服务端的
  每个接口都调用它的同名方法;嵌入式模式(pytest / 开发 / 离线应急)也用它。
- :class:`HttpRepository`:正式 run 使用,经 HTTP ``/v1`` 访问服务。只用标准库
  (urllib),不给 runtime 增加依赖。

两者方法名、参数、返回结构一致。写方法都接受 ``actor``(缺省取本机自报身份)。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Callable

from . import curation, evidence
from .artifacts import (ArtifactError, ArtifactStore, build_bundle_manifest,
                        parse_bundle_manifest, sha256_hex)
from .identity import actor_headers, current_actor
from .recipes import check_python, recipe_family, render
from .store import LibraryStore, LibraryValidationError

API_VERSION = "1"


class LibraryError(Exception):
    """经验库返回错误。``status`` 为 HTTP 状态码(本地调用时为 None)。"""

    def __init__(self, message: str, *, status: int | None = None,
                 errors: list[str] | None = None):
        super().__init__(message)
        self.status = status
        self.errors = errors or []


class LibraryUnavailable(LibraryError):
    """服务不可达(连接失败 / 超时)。调用方按 Q26 降级,不阻塞 run。"""


def _safe_name(name: str) -> str:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ArtifactError(f"bundle 内文件名非法: {name!r}")
    return str(p)


class _BundleMixin:
    """bundle(一组文件)在单对象接口之上的组合操作,本地 / HTTP 共用。"""

    def put_bundle(self, files: dict[str, bytes]) -> str:
        hashes = {}
        for name, data in files.items():
            h = sha256_hex(data)
            if not self.has_artifact(h):
                self.put_artifact(data, expected=h)
            hashes[_safe_name(name)] = h
        manifest = build_bundle_manifest(hashes)
        bid = sha256_hex(manifest)
        if not self.has_artifact(bid):
            self.put_artifact(manifest, expected=bid)
        return bid

    def get_bundle(self, bundle_id: str) -> dict[str, bytes]:
        files = parse_bundle_manifest(self.get_artifact(bundle_id))
        return {_safe_name(n): self.get_artifact(h) for n, h in files.items()}

    def materialize(self, recipe_id: str, dest, *, run_dir: str,
                    skill_root: str) -> list[str]:
        """把配方渲染到 ``dest``。.py 校验不过时返回错误、**不写盘**。"""
        files = render(self.get_bundle(recipe_id), run_dir=run_dir,
                       skill_root=skill_root)
        errors = check_python(files)
        if errors:
            return errors
        dest = Path(dest)
        for name, data in files.items():
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        return []


class LocalRepository(_BundleMixin):
    def __init__(self, store: LibraryStore, artifacts: ArtifactStore):
        self.store = store
        self.artifacts = artifacts
        #: recipe_id → 配方身份;bundle 按内容寻址不可变,缓存永久有效
        self._family_cache: dict[str, str | None] = {}

    def _recipe_family(self, recipe: dict) -> str | None:
        """榜单去重用的配方身份(见 recipes.recipe_family);bundle 缺失时 None。"""
        rid = recipe["recipe_id"]
        if rid not in self._family_cache:
            try:
                self._family_cache[rid] = recipe_family(self.get_bundle(rid))
            except (ArtifactError, LibraryError):
                self._family_cache[rid] = None
        return self._family_cache[rid]

    @staticmethod
    def _actor(actor: dict | None) -> dict:
        return actor if actor else current_actor()

    # ---- 状态
    def health(self) -> dict:
        return {"status": "ok", "api_version": API_VERSION,
                "artifacts": self.artifacts.url, **self.store.health()}

    def watermark(self) -> int:
        return self.store.watermark()

    def commits(self, since: int = 0, limit: int = 100) -> list[dict]:
        return self.store.commits(since, limit)

    # ---- 产物
    def has_artifact(self, h: str) -> bool:
        return self.artifacts.exists(h)

    def put_artifact(self, data: bytes, expected: str | None = None) -> str:
        return self.artifacts.put_bytes(data, expected=expected)

    def get_artifact(self, h: str) -> bytes:
        return self.artifacts.get_bytes(h)

    # ---- 快照
    def submit_snapshot(self, payload: dict, *, namespace: str,
                        actor: dict | None = None) -> dict:
        actor = self._actor(actor)
        rec = self.store.submit_snapshot(payload, namespace=namespace, actor=actor,
                                         artifact_exists=self.artifacts.exists)
        if not rec["duplicate"] and (rec.get("counts") or {}).get("trial_outcomes"):
            # M3:为本快照的 trial 结果建立证据关联;失败不影响提交本身
            try:
                rec["evidence"] = evidence.link_snapshot(self.store, namespace,
                                                         rec["snapshot_id"], actor=actor)
            except Exception as exc:  # noqa: BLE001
                rec["evidence_error"] = f"{type(exc).__name__}: {exc}"
        return rec

    def set_snapshot_state(self, namespace: str, snapshot_id: str, state: str, *,
                           detail: dict | None = None,
                           actor: dict | None = None) -> int:
        return self.store.set_snapshot_state(namespace, snapshot_id, state,
                                             actor=self._actor(actor), detail=detail)

    def list_snapshots(self, namespace: str, *, kind: str | None = None,
                       state: str | None = None, run_id: str | None = None,
                       at: int | None = None) -> list[dict]:
        return self.store.list_snapshots(namespace, kind=kind, state=state,
                                         run_id=run_id, at=at)

    def get_snapshot(self, namespace: str, snapshot_id: str, *,
                     at: int | None = None) -> dict | None:
        return self.store.get_snapshot(namespace, snapshot_id, at=at)

    # ---- 证据与配方
    def trial_outcomes(self, namespace: str, *, atid: str | None = None,
                       run_id: str | None = None, at: int | None = None) -> list[dict]:
        return self.store.trial_outcomes(namespace, atid=atid, run_id=run_id, at=at)

    def recipes(self, namespace: str, *, atid: str | None = None,
                kind: str | None = None, at: int | None = None) -> list[dict]:
        return self.store.recipes(namespace, atid=atid, kind=kind, at=at)

    def recipe_atids(self, namespace: str, *, recipe_id: str | None = None,
                     at: int | None = None) -> list[dict]:
        return self.store.recipe_atids(namespace, recipe_id=recipe_id, at=at)

    def measurements(self, namespace: str, *, recipe_ids=None,
                     at: int | None = None) -> list[dict]:
        return self.store.measurements(namespace, recipe_ids=recipe_ids, at=at)

    def leaderboard(self, namespace: str, atid: str, *, at: int | None = None,
                    top_n: int = 10) -> list[dict]:
        return self.store.leaderboard(namespace, atid, at=at, top_n=top_n,
                                      family_of=self._recipe_family)

    def frozen_baseline(self, namespace: str, atid: str, algorithm: str, *,
                        at: int | None = None) -> dict | None:
        return self.store.frozen_baseline(namespace, atid, algorithm, at=at)

    # ---- 治理与指纹
    def govern(self, namespace: str, events: list[dict], *, message: str = "",
               actor: dict | None = None) -> int:
        return self.store.govern(namespace, events, actor=self._actor(actor),
                                 message=message)

    def governance(self, namespace: str, *, subject_type: str | None = None,
                   subject_id: str | None = None, at: int | None = None) -> list[dict]:
        return self.store.governance(namespace, subject_type=subject_type,
                                     subject_id=subject_id, at=at)

    def add_fingerprint_alias(self, alias_atid: str, canonical_atid: str, *,
                              reason: str, actor: dict | None = None) -> int:
        return self.store.add_fingerprint_alias(alias_atid, canonical_atid,
                                                reason=reason, actor=self._actor(actor))

    def resolve_atid(self, atid: str, *, at: int | None = None) -> str:
        return self.store.resolve_atid(atid, at=at)

    def fingerprints(self, *, at: int | None = None) -> list[dict]:
        return self.store.fingerprints(at=at)

    def fingerprint_profiles(self, *, at: int | None = None) -> dict[str, dict]:
        return self.store.fingerprint_profiles(at=at)

    def put_task_profile(self, atid: str, profile: dict, *,
                         actor: dict | None = None) -> int | None:
        return self.store.put_task_profile(atid, profile, actor=self._actor(actor))

    def suggest_aliases(self, *, at: int | None = None, min_columns: float = 0.9) -> list[dict]:
        from .aliases import suggest_aliases

        return suggest_aliases(self.store, at=at, min_columns=min_columns)

    # ---- 存量修复(v2.18.1)
    def fix_fingerprints(self, namespace: str, *, apply: bool = False,
                         run_dirs: dict | None = None, actor: dict | None = None) -> dict:
        from .hygiene import legacy_fingerprint as LF

        p = LF.plan(self.store, namespace, run_dirs=run_dirs)
        if not apply:
            return {**p, "applied": False}
        res = LF.apply(self.store, namespace, p["actions"], actor=self._actor(actor),
                       reason="library fix-fingerprints")
        return {**p, "applied": True, **res}

    # ---- 检索反馈(v2.18.1)
    def feedback_backfill(self, namespace: str, *, actor: dict | None = None) -> dict:
        from . import feedback

        return feedback.backfill(self.store, namespace, actor=self._actor(actor))

    def feedback_stats(self, namespace: str, *, at: int | None = None,
                       top: int = 20) -> dict:
        from . import feedback

        return feedback.feedback_stats(self.store, namespace, at=at, top=top)

    # ---- 合并与整理(M2,curation.py)
    def merge_pending(self, namespace: str) -> list[dict]:
        return curation.merge_pending(self.store, namespace)

    def merge_context(self, namespace: str, snapshot_id: str, *,
                      actor: dict | None = None) -> dict:
        return curation.merge_context(self.store, namespace, snapshot_id,
                                      holder=self._actor(actor))

    def merge_apply(self, namespace: str, decision: dict, *,
                    actor: dict | None = None) -> dict:
        actor = self._actor(actor)
        rec = curation.merge_apply(self.store, namespace, decision, actor=actor)
        if not rec.get("dry_run") and rec["counts"]["principle_versions"]:
            # 新原则 / 新版本出现后回补历史 trial 的证据(只追加尚不存在的关联)
            try:
                rec["evidence"] = evidence.relink(self.store, namespace, actor=actor)
            except Exception as exc:  # noqa: BLE001
                rec["evidence_error"] = f"{type(exc).__name__}: {exc}"
        return rec

    # ---- 异步整理(v2.18.2,consolidation.py)
    def curation_plan(self, namespace: str, *, include_unchanged: bool = False) -> dict:
        from . import consolidation as CO

        return CO.plan(self.store, namespace, include_unchanged=include_unchanged)

    def curation_open(self, namespace: str, group: str, *,
                      actor: dict | None = None) -> dict:
        from . import consolidation as CO

        return CO.open_group(self.store, namespace, group, actor=self._actor(actor))

    def curation_context(self, namespace: str, snapshot_id: str, *,
                         actor: dict | None = None) -> dict:
        from . import consolidation as CO

        return CO.curation_context(self.store, namespace, snapshot_id,
                                   holder=self._actor(actor))

    def curation_apply(self, namespace: str, decision: dict, *,
                       actor: dict | None = None) -> dict:
        from . import consolidation as CO

        actor = self._actor(actor)
        rec = CO.curation_apply(self.store, namespace, decision, actor=actor)
        if not rec.get("dry_run") and rec["counts"]["principle_versions"]:
            # 修订 / 上位原则出现后回补历史 trial 的证据(与 merge_apply 相同)
            try:
                rec["evidence"] = evidence.relink(self.store, namespace, actor=actor)
            except Exception as exc:  # noqa: BLE001
                rec["evidence_error"] = f"{type(exc).__name__}: {exc}"
        return rec

    def merge_abort(self, namespace: str, snapshot_id: str, *, lease_token: str,
                    reason: str, actor: dict | None = None) -> int:
        return curation.merge_abort(self.store, namespace, snapshot_id, token=lease_token,
                                    reason=reason, actor=self._actor(actor))

    def renew_lease(self, namespace: str, lease_token: str, *,
                    extra_ttl: float | None = None) -> dict:
        kw = {"extra_ttl": float(extra_ttl)} if extra_ttl else {}
        return curation.renew_lease(self.store, curation._lease_name(namespace),
                                    lease_token, **kw)

    def revert_merge(self, namespace: str, snapshot_id: str, *, reason: str,
                     actor: dict | None = None) -> int:
        return curation.revert_merge(self.store, namespace, snapshot_id, reason=reason,
                                     actor=self._actor(actor))

    def principles(self, namespace: str, *, at: int | None = None,
                   include_inactive: bool = False) -> list[dict]:
        return curation.principles_current(self.store, namespace, at=at,
                                           include_inactive=include_inactive)

    def topics(self, namespace: str, *, at: int | None = None) -> list[dict]:
        cur = curation.topics_current(self.store, namespace, at=at)
        return [cur[k] for k in sorted(cur)]

    def dossier(self, namespace: str, scope_type: str, scope_id: str, *,
                at: int | None = None) -> dict | None:
        return curation.dossier_current(self.store, namespace, scope_type, scope_id, at=at)

    def changelog(self, namespace: str, *, since: int = 0, limit: int = 1000) -> list[dict]:
        return curation.changelog(self.store, namespace, since=since, limit=limit)

    def init_topics(self, namespace: str, topics: list[dict], *, reason: str,
                    actor: dict | None = None) -> int:
        return curation.init_topics(self.store, namespace, topics, reason=reason,
                                    actor=self._actor(actor))

    # ---- 证据与检索视图(M3)
    def relink(self, namespace: str, *, actor: dict | None = None) -> dict:
        return evidence.relink(self.store, namespace, actor=self._actor(actor))

    def replay(self, namespace: str, *, k: int = 8, mode: str = "same_task",
               run_dedup: bool = False, multi_factor: float | None = None,
               at: int | None = None, max_queries: int | None = None) -> dict:
        """检索回放评测(v2.18.2,replay.py)。``run_dedup`` / ``multi_factor`` 指定
        证据汇总口径(缺省 = 现行口径),用于比较候选口径。"""
        from . import replay as RP

        at = self.store._at(at)
        view = self.run_view(namespace, None, at=at)
        run_sources: dict[str, set] = {}
        for s in self.store.list_snapshots(namespace, at=at):
            if s.get("run_id"):
                run_sources.setdefault(s["run_id"], set()).add(s["snapshot_id"])
        kw = {"run_dedup": bool(run_dedup)}
        if multi_factor is not None:
            kw["multi_factor"] = float(multi_factor)
        out = RP.replay(view, self.store.trial_outcomes(namespace, at=at),
                        evidence.evidence_links(self.store, namespace, at=at),
                        run_sources=run_sources, k=k, mode=mode, aggregate_kw=kw,
                        max_queries=max_queries)
        out.update({"watermark": at, "evidence": {
            "run_dedup": kw["run_dedup"],
            "multi_factor": kw.get("multi_factor", evidence.MULTI_FACTOR_POWER)}})
        return out

    def evidence(self, namespace: str, principle_id: str, *,
                 at: int | None = None) -> dict:
        links = evidence.evidence_links(self.store, namespace, at=at,
                                        principle_id=principle_id)
        # 用原则当前方向重算 relation,与 principle_stats 同一口径(direction=0 的中性带)
        directions = evidence.principle_directions(self.store, namespace, at=at)
        return {"principle_id": principle_id, "links": links,
                "stats": evidence.aggregate(links, directions=directions)
                .get(principle_id) or {}}

    def run_view(self, namespace: str, atid: str | None, *, at: int | None = None,
                 runtime_api_version: str | None = None) -> dict:
        """一个 run 需要的全部读数据(固定水位,客户端缓存后离线检索)。

        ``runtime_api_version`` 是**客户端** runtime 的配方契约版本:榜单去掉比它新的配方。"""
        from .snapshot import parse_atid

        st = self.store
        at = st._at(at)
        with st._read() as conn:
            aliases = st._alias_map(conn, at)
        canon = st._canonical(atid, aliases) if atid else None
        base = (parse_atid(canon) or {}).get("data_base") if canon else None
        baselines = {}
        if canon:
            for r in st.recipes(namespace, atid=canon, kind="baseline", at=at):
                if r["algorithm"] not in baselines:
                    baselines[r["algorithm"]] = st.frozen_baseline(
                        namespace, canon, r["algorithm"], at=at)
        return {
            "schema": "modelevo.run-view/1", "namespace": namespace, "watermark": at,
            "atid": canon, "data_base": base,
            "principles": curation.principles_current(st, namespace, at=at),
            "stats": evidence.principle_stats(st, namespace, at=at),
            "topics": list(curation.topics_current(st, namespace, at=at).values()),
            "aliases": aliases, "fingerprints": st.fingerprints(at=at),
            # v2.18.1:各任务的多维画像(分级检索的任务相似度)
            "fingerprint_profiles": st.fingerprint_profiles(at=at),
            "dossiers": {
                "task": curation.dossier_current(st, namespace, "task", canon, at=at)
                if canon else None,
                "data_base": curation.dossier_current(st, namespace, "data_base", base, at=at)
                if base else None},
            "leaderboard": st.leaderboard(namespace, canon, at=at,
                                          runtime_api_version=runtime_api_version,
                                          family_of=self._recipe_family)
            if canon else [],
            "baselines": baselines,
            "trial_outcomes": st.trial_outcomes(namespace, atid=canon, at=at) if canon else [],
            # v2.19.1:同数据底座(sample_scale + 列集哈希)的 σ_seed/SE 实测比值,
            # 供新 run 的配对检验先验使用(σ_seed = ratio×SE)。不新增表:
            # 比值随 trial_outcomes.extra.sigma_seed_ratio 落库,这里按底座聚合。
            "seed_variance_ratio": seed_variance_ratio_of(st, namespace, base, at=at),
        }


def seed_variance_ratio_of(store, namespace: str, data_base: str | None, *,
                           at: int | None = None, min_n: int = 3) -> dict | None:
    """同数据底座的历史 σ_seed/SE 比值中位数(v2.19.1)。

    底座 = atid 的 ``{sample_scale}-{col_hash8}``(快照 schema.parse_atid)。
    取该底座全部 trial_outcomes 里 ``extra.sigma_seed_ratio`` 的中位数;
    少于 ``min_n`` 个样本时返回 None(让调用方用公式先验,避免单点噪声)。
    返回值:``{"ratio", "n", "source": "library"}``。
    """
    import statistics

    if not data_base:
        return None
    fp = store.fingerprints(at=at) if hasattr(store, "fingerprints") else []
    family = sorted({f["atid"] for f in fp if f.get("data_base") == data_base})
    if not family:
        return None
    vals: list[float] = []
    for a in family:
        for row in store.trial_outcomes(namespace, atid=a, at=at):
            r = (row.get("extra") or {}).get("sigma_seed_ratio")
            try:
                r = float(r)
            except (TypeError, ValueError):
                continue
            if r > 0:
                vals.append(r)
    if len(vals) < int(min_n):
        return None
    return {"ratio": round(float(statistics.median(vals)), 6),
            "n": len(vals), "source": "library"}


# --------------------------------------------------------------------------
# HTTP 客户端
# --------------------------------------------------------------------------

Transport = Callable[[str, str, dict, "bytes | None", float], "tuple[int, dict, bytes]"]


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None,
                     timeout: float) -> tuple[int, dict, bytes]:
    # 不走 http_proxy:库服务在内网直连(本机 127.0.0.1 经代理反而不通)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read()
    except (urllib.error.URLError, OSError) as exc:
        raise LibraryUnavailable(f"经验库服务不可达: {url}: {exc}") from exc


class HttpRepository(_BundleMixin):
    def __init__(self, base_url: str, *, timeout: float = 15.0,
                 actor: dict | None = None, transport: Transport | None = None,
                 token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.actor = actor
        self.transport = transport or urllib_transport
        #: 服务启用 token 认证时(``serve --tokens``)每个请求带 Bearer token;
        #: 缺省取环境变量 MODELEVO_LIBRARY_TOKEN
        self.token = token if token is not None else os.environ.get("MODELEVO_LIBRARY_TOKEN")

    # ---- 传输
    def _call(self, method: str, path: str, *, params: dict | None = None,
              body=None, raw: bytes | None = None, actor: dict | None = None,
              want_bytes: bool = False, ok_404: bool = False):
        url = f"{self.base_url}/v{API_VERSION}{path}"
        if params:
            q = []
            for k, v in params.items():
                if v is None:
                    continue
                for item in (v if isinstance(v, (list, tuple)) else [v]):
                    q.append((k, str(item)))
            if q:
                url += "?" + urllib.parse.urlencode(q)
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if method in ("POST", "PUT"):
            headers.update(actor_headers(actor or self.actor or current_actor()))
        data = None
        if raw is not None:
            data = raw
            headers["Content-Type"] = "application/octet-stream"
        elif body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        status, _hdrs, content = self.transport(method, url, headers, data,
                                                self.timeout)
        if status == 404 and ok_404:
            return None
        if status >= 400:
            detail, errors = content[:500].decode("utf-8", "replace"), []
            try:
                obj = json.loads(content.decode("utf-8"))
                d = obj.get("detail", obj)
                if isinstance(d, dict):
                    errors = d.get("errors") or []
                    detail = d.get("error") or json.dumps(d, ensure_ascii=False)
                else:
                    detail = str(d)
            except (ValueError, AttributeError):
                pass
            raise LibraryError(f"{method} {path} → {status}: {detail}",
                               status=status, errors=errors)
        if want_bytes:
            return content
        return json.loads(content.decode("utf-8")) if content else None

    # ---- 状态
    def health(self) -> dict:
        return self._call("GET", "/health")

    def watermark(self) -> int:
        return int(self._call("GET", "/watermark")["watermark"])

    def commits(self, since: int = 0, limit: int = 100) -> list[dict]:
        return self._call("GET", "/commits", params={"since": since, "limit": limit})

    # ---- 产物
    def has_artifact(self, h: str) -> bool:
        return self._call("HEAD", f"/artifacts/{h}", ok_404=True,
                          want_bytes=True) is not None

    def put_artifact(self, data: bytes, expected: str | None = None) -> str:
        h = sha256_hex(data)
        if expected is not None and expected != h:
            raise ArtifactError(f"内容哈希 {h} 与声明的 {expected} 不符")
        self._call("PUT", f"/artifacts/{h}", raw=data)
        return h

    def get_artifact(self, h: str) -> bytes:
        data = self._call("GET", f"/artifacts/{h}", want_bytes=True, ok_404=True)
        if data is None:
            raise ArtifactError(f"对象不存在: {h}")
        if sha256_hex(data) != h:
            raise ArtifactError(f"下载的对象内容与哈希不符: {h}")
        return data

    # ---- 快照
    def submit_snapshot(self, payload: dict, *, namespace: str,
                        actor: dict | None = None) -> dict:
        return self._call("POST", "/snapshots", actor=actor,
                          body={"namespace": namespace, "payload": payload})

    def set_snapshot_state(self, namespace: str, snapshot_id: str, state: str, *,
                           detail: dict | None = None,
                           actor: dict | None = None) -> int:
        return int(self._call("POST", f"/snapshots/{snapshot_id}/state", actor=actor,
                              body={"namespace": namespace, "state": state,
                                    "detail": detail or {}})["commit_seq"])

    def list_snapshots(self, namespace: str, *, kind: str | None = None,
                       state: str | None = None, run_id: str | None = None,
                       at: int | None = None) -> list[dict]:
        return self._call("GET", "/snapshots", params={
            "namespace": namespace, "kind": kind, "state": state,
            "run_id": run_id, "at": at})

    def get_snapshot(self, namespace: str, snapshot_id: str, *,
                     at: int | None = None) -> dict | None:
        return self._call("GET", f"/snapshots/{snapshot_id}", ok_404=True,
                          params={"namespace": namespace, "at": at})

    # ---- 证据与配方
    def trial_outcomes(self, namespace: str, *, atid: str | None = None,
                       run_id: str | None = None, at: int | None = None) -> list[dict]:
        return self._call("GET", "/trial-outcomes", params={
            "namespace": namespace, "atid": atid, "run_id": run_id, "at": at})

    def recipes(self, namespace: str, *, atid: str | None = None,
                kind: str | None = None, at: int | None = None) -> list[dict]:
        return self._call("GET", "/recipes", params={
            "namespace": namespace, "atid": atid, "kind": kind, "at": at})

    def measurements(self, namespace: str, *, recipe_ids=None,
                     at: int | None = None) -> list[dict]:
        return self._call("GET", "/measurements", params={
            "namespace": namespace, "recipe_id": list(recipe_ids or []) or None,
            "at": at})

    def leaderboard(self, namespace: str, atid: str, *, at: int | None = None,
                    top_n: int = 10) -> list[dict]:
        return self._call("GET", f"/leaderboard/{urllib.parse.quote(atid, safe='')}",
                          params={"namespace": namespace, "at": at, "top_n": top_n})

    def frozen_baseline(self, namespace: str, atid: str, algorithm: str, *,
                        at: int | None = None) -> dict | None:
        return self._call(
            "GET", f"/baselines/{urllib.parse.quote(atid, safe='')}/"
                   f"{urllib.parse.quote(algorithm, safe='')}",
            ok_404=True, params={"namespace": namespace, "at": at})

    # ---- 治理与指纹
    def govern(self, namespace: str, events: list[dict], *, message: str = "",
               actor: dict | None = None) -> int:
        return int(self._call("POST", "/governance", actor=actor, body={
            "namespace": namespace, "events": events,
            "message": message})["commit_seq"])

    def governance(self, namespace: str, *, subject_type: str | None = None,
                   subject_id: str | None = None, at: int | None = None) -> list[dict]:
        return self._call("GET", "/governance", params={
            "namespace": namespace, "subject_type": subject_type,
            "subject_id": subject_id, "at": at})

    def add_fingerprint_alias(self, alias_atid: str, canonical_atid: str, *,
                              reason: str, actor: dict | None = None) -> int:
        return int(self._call("POST", "/fingerprint-aliases", actor=actor, body={
            "alias_atid": alias_atid, "canonical_atid": canonical_atid,
            "reason": reason})["commit_seq"])

    def resolve_atid(self, atid: str, *, at: int | None = None) -> str:
        return self._call("GET", "/fingerprints/resolve",
                          params={"atid": atid, "at": at})["atid"]

    def fingerprints(self, *, at: int | None = None) -> list[dict]:
        return self._call("GET", "/fingerprints", params={"at": at})

    def fingerprint_profiles(self, *, at: int | None = None) -> dict[str, dict]:
        return self._call("GET", "/fingerprints/profiles", params={"at": at})

    def put_task_profile(self, atid: str, profile: dict, *,
                         actor: dict | None = None) -> int | None:
        return self._call("POST", "/fingerprints/profiles", actor=actor,
                          body={"atid": atid, "profile": profile})["commit_seq"]

    def suggest_aliases(self, *, at: int | None = None, min_columns: float = 0.9) -> list[dict]:
        return self._call("GET", "/fingerprints/alias-suggestions",
                          params={"at": at, "min_columns": min_columns})

    # ---- 存量修复(v2.18.1)
    def fix_fingerprints(self, namespace: str, *, apply: bool = False,
                         run_dirs: dict | None = None, actor: dict | None = None) -> dict:
        return self._call("POST", "/fix/fingerprints", actor=actor, body={
            "namespace": namespace, "apply": bool(apply), "run_dirs": run_dirs or {}})

    # ---- 检索反馈(v2.18.1)
    def feedback_backfill(self, namespace: str, *, actor: dict | None = None) -> dict:
        return self._call("POST", "/feedback/backfill", actor=actor,
                          body={"namespace": namespace})

    def feedback_stats(self, namespace: str, *, at: int | None = None,
                       top: int = 20) -> dict:
        return self._call("GET", "/feedback", params={"namespace": namespace, "at": at,
                                                      "top": top})

    # ---- 合并与整理(M2)
    def merge_pending(self, namespace: str) -> list[dict]:
        return self._call("GET", "/merges/pending", params={"namespace": namespace})

    def merge_context(self, namespace: str, snapshot_id: str, *,
                      actor: dict | None = None) -> dict:
        return self._call("POST", "/merges/context", actor=actor, body={
            "namespace": namespace, "snapshot_id": snapshot_id})

    def merge_apply(self, namespace: str, decision: dict, *,
                    actor: dict | None = None) -> dict:
        return self._call("POST", "/merges/apply", actor=actor, body={
            "namespace": namespace, "decision": decision})

    # ---- 异步整理(v2.18.2)
    def curation_plan(self, namespace: str, *, include_unchanged: bool = False) -> dict:
        return self._call("GET", "/curation/plan", params={
            "namespace": namespace,
            "include_unchanged": "true" if include_unchanged else "false"})

    def curation_open(self, namespace: str, group: str, *,
                      actor: dict | None = None) -> dict:
        return self._call("POST", "/curation/open", actor=actor, body={
            "namespace": namespace, "group": group})

    def curation_context(self, namespace: str, snapshot_id: str, *,
                         actor: dict | None = None) -> dict:
        return self._call("POST", "/curation/context", actor=actor, body={
            "namespace": namespace, "snapshot_id": snapshot_id})

    def curation_apply(self, namespace: str, decision: dict, *,
                       actor: dict | None = None) -> dict:
        return self._call("POST", "/curation/apply", actor=actor, body={
            "namespace": namespace, "decision": decision})

    def merge_abort(self, namespace: str, snapshot_id: str, *, lease_token: str,
                    reason: str, actor: dict | None = None) -> int:
        return int(self._call("POST", "/merges/abort", actor=actor, body={
            "namespace": namespace, "snapshot_id": snapshot_id,
            "lease_token": lease_token, "reason": reason})["commit_seq"])

    def renew_lease(self, namespace: str, lease_token: str, *,
                    extra_ttl: float | None = None) -> dict:
        return self._call("POST", "/merges/renew", body={
            "namespace": namespace, "lease_token": lease_token, "extra_ttl": extra_ttl})

    def revert_merge(self, namespace: str, snapshot_id: str, *, reason: str,
                     actor: dict | None = None) -> int:
        return int(self._call("POST", "/merges/revert", actor=actor, body={
            "namespace": namespace, "snapshot_id": snapshot_id,
            "reason": reason})["commit_seq"])

    def principles(self, namespace: str, *, at: int | None = None,
                   include_inactive: bool = False) -> list[dict]:
        return self._call("GET", "/principles", params={
            "namespace": namespace, "at": at,
            "include_inactive": "true" if include_inactive else None})

    def topics(self, namespace: str, *, at: int | None = None) -> list[dict]:
        return self._call("GET", "/topics", params={"namespace": namespace, "at": at})

    def dossier(self, namespace: str, scope_type: str, scope_id: str, *,
                at: int | None = None) -> dict | None:
        return self._call("GET", f"/dossiers/{urllib.parse.quote(scope_type, safe='')}/"
                                 f"{urllib.parse.quote(scope_id, safe='')}",
                          ok_404=True, params={"namespace": namespace, "at": at})

    def changelog(self, namespace: str, *, since: int = 0, limit: int = 1000) -> list[dict]:
        return self._call("GET", "/changelog", params={
            "namespace": namespace, "since": since, "limit": limit})

    def init_topics(self, namespace: str, topics: list[dict], *, reason: str,
                    actor: dict | None = None) -> int:
        return int(self._call("POST", "/topics/init", actor=actor, body={
            "namespace": namespace, "topics": topics, "reason": reason})["commit_seq"])

    # ---- 证据与检索视图(M3)
    def relink(self, namespace: str, *, actor: dict | None = None) -> dict:
        return self._call("POST", "/evidence/relink", actor=actor, body={"namespace": namespace})

    def replay(self, namespace: str, *, k: int = 8, mode: str = "same_task",
               run_dedup: bool = False, multi_factor: float | None = None,
               at: int | None = None, max_queries: int | None = None) -> dict:
        return self._call("GET", "/replay", params={
            "namespace": namespace, "k": k, "mode": mode, "at": at,
            "run_dedup": "true" if run_dedup else "false", "multi_factor": multi_factor,
            "max_queries": max_queries})

    def evidence(self, namespace: str, principle_id: str, *,
                 at: int | None = None) -> dict:
        return self._call("GET", f"/principles/{urllib.parse.quote(principle_id, safe='')}"
                                 "/evidence", params={"namespace": namespace, "at": at})

    def run_view(self, namespace: str, atid: str | None, *, at: int | None = None,
                 runtime_api_version: str | None = None) -> dict:
        return self._call("GET", "/view", params={"namespace": namespace, "atid": atid,
                                                  "at": at,
                                                  "runtime_api_version": runtime_api_version})


def open_repository(uri: str, *, artifacts_url: str | None = None,
                    timeout: float = 15.0, actor: dict | None = None,
                    token: str | None = None):
    """``http(s)://…`` → HttpRepository;数据库 URL → 嵌入式 LocalRepository
    (须同时给 ``artifacts_url``)。"""
    if uri.startswith(("http://", "https://")):
        return HttpRepository(uri, timeout=timeout, actor=actor, token=token)
    if not artifacts_url:
        raise ValueError("嵌入式经验库需要 artifacts_url")
    return LocalRepository(LibraryStore(uri), ArtifactStore(artifacts_url))


__all__ = ["API_VERSION", "HttpRepository", "LibraryError", "LibraryUnavailable",
           "LibraryValidationError", "LocalRepository", "open_repository",
           "urllib_transport"]
