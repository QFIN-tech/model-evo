"""v2.18.0 经验库服务:FastAPI 应用(dev 文档 §8.1)。

每个接口直接调用 :class:`runtime.library.LocalRepository` 的同名方法,嵌入式与
服务式行为一致。写请求的身份取自 ``X-Modelevo-*`` 请求头(Q25)。

v2.18.1:给 ``tokens``({sha256: {user, team, write}})时启用认证 —— 除 /health 外
所有请求须带 ``Authorization: Bearer <token>``;写请求须 write 权限;actor 的
user / team 以 token 登记为准。

注:路由签名用 typing.Optional / List(Python 3.9 运行时不支持 ``X | None``)。
"""
from typing import List, Optional

from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from runtime.library import API_VERSION, LocalRepository
from runtime.library.artifacts import ArtifactError
from runtime.library.curation import LeaseBusy
from runtime.library.identity import actor_from_headers, bearer, token_hash
from runtime.library.store import LibraryValidationError


def _actor(request: Request) -> dict:
    actor = actor_from_headers(request.headers)
    actor["remote"] = request.client.host if request.client else ""
    auth = getattr(request.state, "auth", None)
    if auth:
        actor.update({"user": auth["user"], "team": auth["team"], "auth": "token"})
    return actor


def create_app(repo: LocalRepository, *, on_write=None, tokens: Optional[dict] = None) -> FastAPI:
    """``on_write()``:每次写请求(POST / PUT)成功后回调,用于触发明文镜像重新生成;
    回调异常不影响响应。``tokens``:见模块说明;None = 不认证(只应在本机监听时)。"""
    app = FastAPI(title="ModelEvo Experience Library", version=API_VERSION)
    v1 = APIRouter(prefix=f"/v{API_VERSION}")

    if tokens:
        @app.middleware("http")
        async def _auth(request: Request, call_next):
            if request.url.path == f"/v{API_VERSION}/health":
                return await call_next(request)
            tok = bearer(request.headers)
            entry = tokens.get(token_hash(tok)) if tok else None
            if entry is None:
                return JSONResponse(status_code=401, content={
                    "detail": {"error": "需要有效的 Bearer token", "errors": ["unauthorized"]}})
            if request.method in ("POST", "PUT", "DELETE") and not entry["write"]:
                return JSONResponse(status_code=403, content={
                    "detail": {"error": "该 token 只读", "errors": ["forbidden"]}})
            request.state.auth = entry
            return await call_next(request)

    if on_write is not None:
        @app.middleware("http")
        async def _after_write(request: Request, call_next):
            response = await call_next(request)
            if request.method in ("POST", "PUT") and response.status_code < 400:
                try:
                    on_write()
                except Exception:  # noqa: BLE001
                    pass
            return response

    @app.exception_handler(LibraryValidationError)
    async def _validation(_req: Request, exc: LibraryValidationError):
        return JSONResponse(status_code=422, content={
            "detail": {"error": str(exc), "errors": exc.errors}})

    @app.exception_handler(LeaseBusy)
    async def _lease_busy(_req: Request, exc: LeaseBusy):
        return JSONResponse(status_code=409, content={
            "detail": {"error": str(exc), "errors": exc.errors}})

    @app.exception_handler(ArtifactError)
    async def _artifact(_req: Request, exc: ArtifactError):
        status = 404 if "不存在" in str(exc) else 400
        return JSONResponse(status_code=status, content={
            "detail": {"error": str(exc), "errors": [str(exc)]}})

    # ---- 状态
    @v1.get("/health")
    def health():
        return repo.health()

    @v1.get("/watermark")
    def watermark():
        return {"watermark": repo.watermark()}

    @v1.get("/commits")
    def commits(since: int = 0, limit: int = Query(100, le=1000)):
        return repo.commits(since, limit)

    # ---- 产物
    @v1.api_route("/artifacts/{sha}", methods=["HEAD"])
    def head_artifact(sha: str):
        return Response(status_code=200 if repo.has_artifact(sha) else 404)

    @v1.get("/artifacts/{sha}")
    def get_artifact(sha: str):
        return Response(content=repo.get_artifact(sha),
                        media_type="application/octet-stream")

    @v1.put("/artifacts/{sha}")
    async def put_artifact(sha: str, request: Request):
        data = await request.body()
        h = await run_in_threadpool(repo.put_artifact, data, sha)
        return {"sha256": h, "size": len(data)}

    # ---- 快照
    @v1.post("/snapshots")
    def submit_snapshot(request: Request, body: dict = Body(...)):
        payload, namespace = body.get("payload"), body.get("namespace")
        if not isinstance(payload, dict) or not namespace:
            raise LibraryValidationError(["请求体须为 {namespace, payload}"])
        return repo.submit_snapshot(payload, namespace=namespace,
                                    actor=_actor(request))

    @v1.post("/snapshots/{sid}/state")
    def set_snapshot_state(sid: str, request: Request, body: dict = Body(...)):
        seq = repo.set_snapshot_state(body.get("namespace") or "", sid,
                                      body.get("state") or "",
                                      detail=body.get("detail") or {},
                                      actor=_actor(request))
        return {"commit_seq": seq}

    @v1.get("/snapshots")
    def list_snapshots(namespace: str, kind: Optional[str] = None,
                       state: Optional[str] = None, run_id: Optional[str] = None,
                       at: Optional[int] = None):
        return repo.list_snapshots(namespace, kind=kind, state=state,
                                   run_id=run_id, at=at)

    @v1.get("/snapshots/{sid}")
    def get_snapshot(sid: str, namespace: str, at: Optional[int] = None):
        snap = repo.get_snapshot(namespace, sid, at=at)
        if snap is None:
            raise HTTPException(404, detail={"error": f"快照不存在: {sid}"})
        return snap

    # ---- 证据与配方
    @v1.get("/trial-outcomes")
    def trial_outcomes(namespace: str, atid: Optional[str] = None,
                       run_id: Optional[str] = None, at: Optional[int] = None):
        return repo.trial_outcomes(namespace, atid=atid, run_id=run_id, at=at)

    @v1.get("/recipes")
    def recipes(namespace: str, atid: Optional[str] = None,
                kind: Optional[str] = None, at: Optional[int] = None):
        return repo.recipes(namespace, atid=atid, kind=kind, at=at)

    @v1.get("/measurements")
    def measurements(namespace: str, recipe_id: Optional[List[str]] = Query(None),
                     at: Optional[int] = None):
        return repo.measurements(namespace, recipe_ids=recipe_id, at=at)

    @v1.get("/leaderboard/{atid}")
    def leaderboard(atid: str, namespace: str, at: Optional[int] = None,
                    top_n: int = Query(10, ge=1, le=100)):
        return repo.leaderboard(namespace, atid, at=at, top_n=top_n)

    @v1.get("/baselines/{atid}/{algorithm}")
    def frozen_baseline(atid: str, algorithm: str, namespace: str,
                        at: Optional[int] = None):
        rec = repo.frozen_baseline(namespace, atid, algorithm, at=at)
        if rec is None:
            raise HTTPException(404, detail={"error": "无固化 baseline"})
        return rec

    # ---- 治理与指纹
    @v1.post("/governance")
    def govern(request: Request, body: dict = Body(...)):
        seq = repo.govern(body.get("namespace") or "", body.get("events") or [],
                          message=body.get("message") or "", actor=_actor(request))
        return {"commit_seq": seq}

    @v1.get("/governance")
    def governance(namespace: str, subject_type: Optional[str] = None,
                   subject_id: Optional[str] = None, at: Optional[int] = None):
        return repo.governance(namespace, subject_type=subject_type,
                               subject_id=subject_id, at=at)

    @v1.post("/fingerprint-aliases")
    def add_alias(request: Request, body: dict = Body(...)):
        seq = repo.add_fingerprint_alias(
            body.get("alias_atid") or "", body.get("canonical_atid") or "",
            reason=body.get("reason") or "", actor=_actor(request))
        return {"commit_seq": seq}

    @v1.get("/fingerprints/resolve")
    def resolve_atid(atid: str, at: Optional[int] = None):
        return {"atid": repo.resolve_atid(atid, at=at)}

    @v1.get("/fingerprints")
    def fingerprints(at: Optional[int] = None):
        return repo.fingerprints(at=at)

    @v1.get("/fingerprints/profiles")
    def fingerprint_profiles(at: Optional[int] = None):
        return repo.fingerprint_profiles(at=at)

    @v1.post("/fingerprints/profiles")
    def put_task_profile(request: Request, body: dict = Body(...)):
        return {"commit_seq": repo.put_task_profile(body.get("atid") or "",
                                                    body.get("profile") or {},
                                                    actor=_actor(request))}

    @v1.get("/fingerprints/alias-suggestions")
    def alias_suggestions(at: Optional[int] = None, min_columns: float = 0.9):
        return repo.suggest_aliases(at=at, min_columns=min_columns)

    # ---- 存量修复(v2.18.1;apply 需写权限,默认只出计划)
    @v1.post("/fix/fingerprints")
    def fix_fingerprints(request: Request, body: dict = Body(...)):
        return repo.fix_fingerprints(body.get("namespace") or "",
                                     apply=bool(body.get("apply")),
                                     run_dirs=body.get("run_dirs") or {},
                                     actor=_actor(request))

    # ---- 检索反馈(v2.18.1)
    @v1.post("/feedback/backfill")
    def feedback_backfill(request: Request, body: dict = Body(...)):
        return repo.feedback_backfill(body.get("namespace") or "", actor=_actor(request))

    @v1.get("/feedback")
    def feedback_stats(namespace: str, at: Optional[int] = None, top: int = 20):
        return repo.feedback_stats(namespace, at=at, top=top)

    # ---- 检索回放评测(v2.18.2)
    @v1.get("/replay")
    def replay(namespace: str, k: int = 8, mode: str = "same_task",
               run_dedup: bool = False, multi_factor: Optional[float] = None,
               at: Optional[int] = None, max_queries: Optional[int] = None):
        return repo.replay(namespace, k=k, mode=mode, run_dedup=run_dedup,
                           multi_factor=multi_factor, at=at, max_queries=max_queries)

    # ---- 合并与整理(M2)
    @v1.get("/merges/pending")
    def merge_pending(namespace: str):
        return repo.merge_pending(namespace)

    @v1.post("/merges/context")
    def merge_context(request: Request, body: dict = Body(...)):
        return repo.merge_context(body.get("namespace") or "",
                                  body.get("snapshot_id") or "", actor=_actor(request))

    @v1.post("/merges/apply")
    def merge_apply(request: Request, body: dict = Body(...)):
        return repo.merge_apply(body.get("namespace") or "", body.get("decision") or {},
                                actor=_actor(request))

    # ---- 异步整理(v2.18.2;放弃用 /merges/abort,撤销用 /merges/revert)
    @v1.get("/curation/plan")
    def curation_plan(namespace: str, include_unchanged: bool = False):
        return repo.curation_plan(namespace, include_unchanged=include_unchanged)

    @v1.post("/curation/open")
    def curation_open(request: Request, body: dict = Body(...)):
        return repo.curation_open(body.get("namespace") or "", body.get("group") or "",
                                  actor=_actor(request))

    @v1.post("/curation/context")
    def curation_context(request: Request, body: dict = Body(...)):
        return repo.curation_context(body.get("namespace") or "",
                                     body.get("snapshot_id") or "", actor=_actor(request))

    @v1.post("/curation/apply")
    def curation_apply(request: Request, body: dict = Body(...)):
        return repo.curation_apply(body.get("namespace") or "", body.get("decision") or {},
                                   actor=_actor(request))

    @v1.post("/merges/abort")
    def merge_abort(request: Request, body: dict = Body(...)):
        seq = repo.merge_abort(body.get("namespace") or "", body.get("snapshot_id") or "",
                               lease_token=body.get("lease_token") or "",
                               reason=body.get("reason") or "", actor=_actor(request))
        return {"commit_seq": seq}

    @v1.post("/merges/renew")
    def merge_renew(body: dict = Body(...)):
        return repo.renew_lease(body.get("namespace") or "", body.get("lease_token") or "",
                                extra_ttl=body.get("extra_ttl"))

    @v1.post("/merges/revert")
    def merge_revert(request: Request, body: dict = Body(...)):
        seq = repo.revert_merge(body.get("namespace") or "", body.get("snapshot_id") or "",
                                reason=body.get("reason") or "", actor=_actor(request))
        return {"commit_seq": seq}

    @v1.get("/principles")
    def principles(namespace: str, at: Optional[int] = None,
                   include_inactive: bool = False):
        return repo.principles(namespace, at=at, include_inactive=include_inactive)

    @v1.post("/topics/init")
    def init_topics(request: Request, body: dict = Body(...)):
        seq = repo.init_topics(body.get("namespace") or "", body.get("topics") or [],
                               reason=body.get("reason") or "", actor=_actor(request))
        return {"commit_seq": seq}

    @v1.get("/topics")
    def topics(namespace: str, at: Optional[int] = None):
        return repo.topics(namespace, at=at)

    @v1.get("/dossiers/{scope_type}/{scope_id}")
    def dossier(scope_type: str, scope_id: str, namespace: str, at: Optional[int] = None):
        d = repo.dossier(namespace, scope_type, scope_id, at=at)
        if d is None:
            raise HTTPException(404, detail={"error": "无卷宗"})
        return d

    @v1.get("/changelog")
    def changelog(namespace: str, since: int = 0, limit: int = Query(1000, le=5000)):
        return repo.changelog(namespace, since=since, limit=limit)

    # ---- 证据与检索视图(M3)
    @v1.post("/evidence/relink")
    def relink(request: Request, body: dict = Body(...)):
        return repo.relink(body.get("namespace") or "", actor=_actor(request))

    @v1.get("/principles/{principle_id}/evidence")
    def principle_evidence(principle_id: str, namespace: str, at: Optional[int] = None):
        return repo.evidence(namespace, principle_id, at=at)

    @v1.get("/view")
    def run_view(namespace: str, atid: Optional[str] = None, at: Optional[int] = None,
                 runtime_api_version: Optional[str] = None):
        return repo.run_view(namespace, atid, at=at, runtime_api_version=runtime_api_version)

    app.include_router(v1)
    return app
