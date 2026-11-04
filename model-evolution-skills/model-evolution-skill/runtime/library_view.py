"""v2.18.0 M3:run 侧经验库视图 —— 固定水位、本机缓存、离线可读(Q4 Q17 Q26)。

一个 run 第一次需要经验库时(profile 的泄漏对账)调用 :func:`ensure` 固定库水位;
之后拿到任务指纹(profile 末尾 / baseline)再按**同一水位**补取任务相关数据。视图
写到 ``<run>/experience/library-view.json``,并备份到本机缓存目录;热启动与固化
baseline 需要的配方 bundle 一并预取到缓存。之后 planner-context、new-trial、
planner、experience search 等只读这份视图,不再访问服务:

- 并行 run 互不可见(Q4):各自固定启动时的水位;
- 服务不可达(Q26):用本机缓存里最近的视图并标 ``stale``;没有缓存则空视图,
  本 run 不用经验,diagnose 报 warning;
- 写入(baseline 固化、收尾快照)不受水位限制,直接走 HttpRepository。

依赖方向:runtime 其它模块 → 本模块 → runtime.library(纯库代码)。
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

from .library import retrieval
from .library.recipes import RUNTIME_API_VERSION, check_python, render
from .logging_utils import get_logger, load_json, locked_state_write

logger = get_logger("modelevo.library_view")

VIEW_REL = Path("experience") / "library-view.json"
VIEW_SCHEMA = "modelevo.run-view/1"
_MODEL_TYPES = {"binary_classification": "binary_classification", "binary": "binary_classification",
                "classification": "binary_classification", "regression": "regression",
                "uplift": "uplift"}
_TRANSFORMS = (("_log1p", "log1p"), ("_log", "log"), ("_sqrt", "sqrt"), ("_boxcox", "boxcox"))


def _truthy(value, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _config_for(run_dir):
    """按 run 目录解析配置:run 级 modelevo-config.yaml 覆盖 skill 级。

    [FIX-2026-10-29] 旧版 ``client_settings(None)`` 直接用 ``Config.load()``,
    只读 skill 级 ``assets/global-config.yaml``,把 run 级配置里**同样由 init
    写出**的 ``library`` 段静默忽略。实测(run4 req-t7 run1):run config 的
    ``library.uri`` 改成不可达地址后,profile 仍然连上真实服务并固化了真实
    库视图(watermark 243),baseline 据此自动创建 2 个 warm-start trial ——
    配置写了却不生效,且没有任何提示。
    """
    from .config import Config

    if run_dir is None:
        return Config.load()
    return Config.load(run_config=Path(run_dir) / "modelevo-config.yaml")


def config_for_run(run_dir=None):
    """对外暴露的配置解析:有 run 目录就用 run 级配置(带 skill 级基线)。"""
    return _config_for(run_dir)


def client_settings(config=None, *, run_dir=None) -> dict:
    """客户端配置:Config 的 library.* ,环境变量覆盖。

    优先级(高 → 低):``MODELEVO_LIBRARY_URI/_NS/_TEAM/_CACHE/_ENABLED`` >
    run 级 ``modelevo-config.yaml`` 的 ``library.*`` > skill 级
    ``assets/global-config.yaml``。

    ``config`` 显式给定时以它为准(调用方已解析好);只给 ``run_dir`` 时按
    run 级配置解析;都不给才退回 skill 级(CLI 无 run 上下文的场景)。
    ``library.enabled: false``(或 ``MODELEVO_LIBRARY_ENABLED=0``)是本 run
    主动关闭经验库的唯一正式开关:不连服务、不读本机缓存、不固化视图,
    baseline 也不会创建 warm-start trial。
    """
    if config is None:
        config = _config_for(run_dir)
    cache = os.environ.get("MODELEVO_LIBRARY_CACHE") or config.get("library.cache_dir") or \
        str(Path.home() / ".cache" / "modelevo-library")
    enabled_env = os.environ.get("MODELEVO_LIBRARY_ENABLED")
    enabled = _truthy(enabled_env if enabled_env is not None
                      else config.get("library.enabled"), True)
    return {
        "enabled": enabled,
        "uri": os.environ.get("MODELEVO_LIBRARY_URI") or config.get("library.uri"),
        "namespace": os.environ.get("MODELEVO_LIBRARY_NS") or config.get("library.namespace")
        or "prod",
        "team": os.environ.get("MODELEVO_LIBRARY_TEAM", config.get("library.team") or ""),
        "timeout": float(config.get("library.timeout_sec", 15) or 15),
        "token": os.environ.get("MODELEVO_LIBRARY_TOKEN") or config.get("library.token") or None,
        "cache_dir": Path(cache),
        "warm_rank_limit": int(config.get("baseline.warm_rank_limit", 2) or 2),
    }


def describe_resolution(settings: dict, run_dir=None) -> str:
    """一行可审计的解析结果:pinned_at 与 run-state 都记它。"""
    if not settings.get("enabled", True):
        return (f"LIBRARY_RESOLVED enabled=false uri={settings['uri']} "
                f"run={run_dir} (本 run 主动禁用经验库)")
    return (f"LIBRARY_RESOLVED enabled=true uri={settings['uri']} "
            f"namespace={settings['namespace']} cache={settings['cache_dir']} run={run_dir}")


def client(settings: dict, *, model: str | None = None, run: str | None = None):
    from .library import HttpRepository
    from .library.identity import current_actor

    return HttpRepository(settings["uri"], timeout=settings["timeout"],
                          token=settings.get("token"),
                          actor=current_actor(team=settings["team"], model=model, run=run))


def spec_scope(spec) -> dict:
    """任务口径:model_type / primary_metric / target_transform(与原则 scope 同口径)。"""
    data = getattr(spec, "data", spec) or {}
    target = data.get("target") if isinstance(data.get("target"), dict) else {}
    name = str(target.get("name") or "")
    tt = target.get("transform") or target.get("target_transform") or \
        data.get("target_transform")
    if not tt:
        tt = next((v for suf, v in _TRANSFORMS if name.endswith(suf)), "raw")
    return {"model_type": _MODEL_TYPES.get(str(data.get("model_type") or "").lower(),
                                           data.get("model_type")),
            "primary_metric": str(data.get("primary_metric") or "").lower(),
            "target_transform": str(tt).lower()}


def raw_columns(run_dir) -> list[str]:
    """本 run 的原生列名(任务画像的列 MinHash 用):原始分片 schema(与 atid 同口径)
    → splits/train.parquet → feature-quality.csv。拿不到时返回空列表。"""
    run = Path(run_dir).resolve()
    cands = []
    for anc in (run, run.parent, run.parent.parent):
        shards = sorted((anc / "data").glob("train_400w-part-*.parquet")) \
            if (anc / "data").is_dir() else []
        if shards:
            cands.append(shards[0])
            break
    cands.append(run / "sample-profile" / "splits" / "train.parquet")
    for pq in cands:
        if pq.is_file():
            try:
                import pyarrow.parquet as _pq

                return [str(c) for c in _pq.read_schema(str(pq)).names]
            except Exception:  # noqa: BLE001
                continue
    fq = run / "sample-profile" / "feature-quality.csv"
    if fq.is_file():
        try:
            import csv

            with open(fq, encoding="utf-8") as fh:
                rows = csv.DictReader(fh)
                key = next((k for k in (rows.fieldnames or [])
                            if k in ("feature", "column", "name")), None)
                if key:
                    return [r[key] for r in rows if r.get(key)]
        except (OSError, csv.Error):
            pass
    return []


def task_profile_for(run_dir, spec, atid: str | None = None) -> dict:
    """本 run 的多维任务画像(v2.18.1,分级检索与快照提交用)。失败时返回空 dict。"""
    from .library.task_profile import build_profile

    try:
        data = getattr(spec, "data", spec) or {}
        return build_profile(spec_data=data, columns=raw_columns(run_dir) if run_dir else [],
                             atid=atid)
    except Exception as exc:  # noqa: BLE001  画像只用于相似度,失败不影响 run
        logger.warning("任务画像构造失败(检索退回 atid 画像): %s", exc)
        return {}


# --------------------------------------------------------------------------
# 视图
# --------------------------------------------------------------------------

class LibraryView:
    def __init__(self, data: dict | None, *, run_dir=None):
        self.data = data or {}
        self.run_dir = Path(run_dir) if run_dir else None

    # ---- 元数据
    @property
    def pinned(self) -> bool:
        return self.data.get("watermark") is not None

    @property
    def watermark(self):
        return self.data.get("watermark")

    @property
    def stale(self) -> bool:
        return bool(self.data.get("stale"))

    @property
    def atid(self):
        return self.data.get("atid")

    @property
    def namespace(self):
        return self.data.get("namespace")

    # ---- 数据
    @property
    def principles(self) -> list[dict]:
        return self.data.get("principles") or []

    @property
    def stats(self) -> dict:
        return self.data.get("stats") or {}

    @property
    def dossiers(self) -> dict:
        return self.data.get("dossiers") or {}

    @property
    def leaderboard(self) -> list[dict]:
        return self.data.get("leaderboard") or []

    @property
    def baselines(self) -> dict:
        return self.data.get("baselines") or {}

    def principle(self, pid: str) -> dict | None:
        return next((p for p in self.principles if p["principle_id"] == pid), None)

    def exists(self, pid: str) -> bool:
        return self.principle(pid) is not None

    def search(self, *, topics=(), hypothesis=None, k: int = 8, scope=None,
               directions=(-1, 0, 1)) -> list[dict]:
        return retrieval.search(self.data, topics=topics, scope=scope or self.data.get("scope"),
                                atid=self.atid, hypothesis=hypothesis, k=k,
                                directions=directions)

    def direction_table(self, top: int = 5) -> dict:
        return retrieval.direction_table(self.data, top=top)

    def failed_for(self, topics) -> list[dict]:
        return retrieval.failed_for_topics(self.data, topics)

    def resolver(self) -> dict[str, str]:
        return retrieval.resolver_of(self.data)

    # ---- 配方
    def bundle_dir(self, recipe_id: str) -> Path | None:
        root = self.data.get("bundles_dir")
        d = Path(root) / recipe_id if root else None
        return d if d is not None and (d / ".complete").exists() else None

    def materialize(self, recipe_id: str, dest, *, run_dir, skill_root) -> list[str]:
        """把缓存里的配方渲染到 ``dest``(填回本 run 的 RUN_DIR / SKILL_ROOT);
        .py 校验不过时返回错误、不写盘。"""
        src = self.bundle_dir(recipe_id)
        if src is None:
            return [f"配方 {recipe_id[:12]} 不在本机缓存(视图固定时服务不可达?)"]
        files = {str(p.relative_to(src)): p.read_bytes() for p in src.rglob("*")
                 if p.is_file() and p.name != ".complete"}
        files = render(files, run_dir=str(run_dir), skill_root=str(skill_root))
        errors = check_python(files)
        if errors:
            return errors
        dest = Path(dest)
        for name, data in files.items():
            out = dest / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        return []


def empty_view(reason: str, **meta) -> dict:
    return {"schema": VIEW_SCHEMA, "watermark": None, "stale": True, "reason": reason,
            "principles": [], "stats": {}, "topics": [], "aliases": {}, "fingerprints": [],
            "dossiers": {}, "leaderboard": [], "baselines": {}, "trial_outcomes": [], **meta}


def load(run_dir) -> LibraryView:
    """只读本 run 已固定的视图;未固定时返回空视图(不访问服务)。"""
    p = Path(run_dir) / VIEW_REL
    if p.exists():
        try:
            return LibraryView(json.loads(p.read_text(encoding="utf-8")), run_dir=run_dir)
        except (OSError, ValueError) as exc:
            logger.warning("经验库视图损坏,按空视图处理: %s", exc)
    return LibraryView(empty_view("not_pinned"), run_dir=run_dir)


def seed_variance_ratio(run_dir) -> dict | None:
    """本 run 数据底座的历史 σ_seed/SE 比值(v2.19.1)。

    读取固定视图里的 ``seed_variance_ratio``(服务端按 data_base 聚合)。
    返回 ``{"ratio", "n", "source"}`` 或 None(库未启用 / 无历史 / 样本不足)。
    调用方(runner._seed_sd_context)优先用它,拿不到时退回按 n_val 的公式先验。
    """
    try:
        data = load(run_dir).data or {}
    except Exception:  # noqa: BLE001  视图不可读不影响裁决
        return None
    r = data.get("seed_variance_ratio")
    if not isinstance(r, dict):
        return None
    try:
        ratio = float(r.get("ratio"))
    except (TypeError, ValueError):
        return None
    if ratio <= 0:
        return None
    return {"ratio": ratio, "n": r.get("n"),
            "source": r.get("source") or "library"}


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _prefetch_bundles(repo, data: dict, bundles_root: Path, top: int) -> list[str]:
    want = [r["recipe_id"] for r in (data.get("baselines") or {}).values() if r]
    want += [r["recipe_id"] for r in (data.get("leaderboard") or [])[:top]]
    fetched = []
    for rid in dict.fromkeys(want):
        d = bundles_root / rid
        if (d / ".complete").exists():
            continue
        tmp = bundles_root / f".{rid}.{uuid.uuid4().hex}.tmp"
        try:
            for name, blob in repo.get_bundle(rid).items():
                out = tmp / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(blob)
            (tmp / ".complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            os.replace(tmp, d)
            fetched.append(rid)
        except Exception as exc:  # noqa: BLE001  单个 bundle 失败不影响视图
            shutil.rmtree(tmp, ignore_errors=True)
            logger.warning("配方 %s 预取失败: %s", rid[:12], exc)
    return fetched


def _latest_cache(cache_ns: Path, atid: str | None) -> dict | None:
    cands = []
    for sub in ([atid] if atid else []) + ["_none"]:
        d = cache_ns / "views" / str(sub).replace("/", "_")
        if d.is_dir():
            cands += [(int(p.stem), p) for p in d.glob("*.json") if p.stem.isdigit()]
        if cands:
            break
    if not cands:
        return None
    try:
        return json.loads(max(cands)[1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _record_run_state(run_dir: Path, data: dict, settings: dict, source: str) -> None:
    st = run_dir / "run-state.json"
    if not st.exists():
        return
    with locked_state_write(st):
        state = load_json(st)
        state["library"] = {"uri": settings["uri"], "namespace": settings["namespace"],
                            "watermark": data.get("watermark"), "stale": bool(data.get("stale")),
                            "atid": data.get("requested_atid"), "source": source,
                            "pinned_at": data.get("pinned_at")}
        tmp = st.with_name(f".{st.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        os.replace(tmp, st)


def pin(run_dir, *, atid: str | None = None, spec=None, config=None) -> LibraryView:
    """固定(或在同一水位补取)本 run 的经验库视图。"""
    from .library import LibraryError

    run_dir = Path(run_dir)
    settings = client_settings(config, run_dir=run_dir)
    logger.info(describe_resolution(settings, run_dir))
    prev = load(run_dir)
    if not settings.get("enabled", True):
        # [FIX-2026-10-29] 主动禁用:连本机缓存都不读。旧版只有"服务不可达 →
        # 退化到本机缓存(stale)"这一条被动路径,于是把 uri 指到不可达地址后
        # 仍然拿到了缓存里的真实视图(watermark 243、157 条原则),baseline 照旧
        # 创建 warm-start trial。禁用必须是显式的、可审计的。
        data = empty_view("library_disabled")
        data.update({"requested_atid": atid,
                     "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "uri": settings["uri"], "namespace": settings["namespace"]})
        if spec is not None:
            data["scope"] = spec_scope(spec)
            try:
                data["task_profile"] = task_profile_for(run_dir, spec, atid)
            except Exception:  # noqa: BLE001  禁用时画像失败无影响
                pass
        _write_json_atomic(run_dir / VIEW_REL, data)
        _record_run_state(run_dir, data, settings, "disabled")
        logger.warning("本 run 已主动禁用经验库(library.enabled=false),"
                       "不连接服务、不读本机缓存、不创建 warm-start trial")
        return LibraryView(data, run_dir=run_dir)
    at = prev.watermark if prev.pinned and not prev.stale else None
    ns = settings["namespace"]
    cache_ns = settings["cache_dir"] / ns.replace("/", "__")
    source = "server"
    try:
        repo = client(settings)
        data = repo.run_view(ns, atid, at=at, runtime_api_version=RUNTIME_API_VERSION)
        data["bundles_dir"] = str(cache_ns / "bundles")
        _prefetch_bundles(repo, data, cache_ns / "bundles", max(settings["warm_rank_limit"], 3))
        data["stale"] = False
        _write_json_atomic(cache_ns / "views" / str(atid or "_none").replace("/", "_")
                           / f"{data['watermark']}.json", data)
    except LibraryError as exc:
        cached = _latest_cache(cache_ns, atid)
        if cached is not None:
            data, source = cached, "cache"
            data["stale"] = True
            logger.warning("经验库服务不可达,使用本机缓存视图(水位 %s,stale):%s",
                           data.get("watermark"), exc)
        else:
            data, source = empty_view("library_unreachable"), "none"
            logger.warning("经验库服务不可达且无本机缓存,本 run 不使用经验:%s", exc)
    data.update({"requested_atid": atid, "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "uri": settings["uri"], "namespace": ns})
    if spec is not None:
        data["scope"] = spec_scope(spec)
        data["task_profile"] = task_profile_for(run_dir, spec, atid)
    elif prev.data.get("scope"):
        data["scope"] = prev.data["scope"]
        if prev.data.get("task_profile"):
            data["task_profile"] = prev.data["task_profile"]
    _write_json_atomic(run_dir / VIEW_REL, data)
    _record_run_state(run_dir, data, settings, source)
    return LibraryView(data, run_dir=run_dir)


def ensure(run_dir, *, atid: str | None = None, spec=None, config=None) -> LibraryView:
    """已固定且覆盖所需任务指纹时直接返回;否则固定 / 补取。"""
    view = load(run_dir)
    if view.pinned and (atid is None or view.data.get("requested_atid") == atid):
        if spec is not None and not view.data.get("scope"):
            view.data["scope"] = spec_scope(spec)
        if spec is not None and not view.data.get("task_profile"):
            view.data["task_profile"] = task_profile_for(run_dir, spec, view.atid)
        return view
    if view.data.get("reason") in ("library_unreachable", "library_disabled") \
            and atid is None:
        return view
    return pin(run_dir, atid=atid, spec=spec, config=config)
