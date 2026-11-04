"""v2.18.0 经验库服务进程入口(dev 文档 §8.2)。

    python service/server.py --root /data/oceanus_share/modelevo-library \\
        --host 127.0.0.1 --port 18765

启动时:Alembic 升级到最新 schema → 维护线程(SQLite 每日在线备份、每小时重写
CHANGELOG/<namespace>.md)→ 明文镜像发布器(``{root}/browse/<namespace>``:写入后防抖
重新生成,启动时与每小时兜底各一次;``--no-browse`` 关闭)→ 写 pidfile → uvicorn。
通常由 ``modelevo library serve [--daemon]`` 调用。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[1]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from runtime.library import ArtifactStore, LibraryStore, LocalRepository  # noqa: E402

log = logging.getLogger("modelevo.library.server")
_BACKUP_PREFIX = "library-"


def default_urls(root: str) -> tuple[str, str]:
    root = str(Path(root).resolve())
    return f"sqlite:///{root}/library.db", f"file://{root}/cas"


def sqlite_path(db_url: str) -> str | None:
    from sqlalchemy.engine import make_url

    url = make_url(db_url)
    if url.get_backend_name() != "sqlite" or not url.database or \
            url.database == ":memory:":
        return None
    return url.database


def backup_sqlite(db_path: str, backup_dir: Path, keep_days: int,
                  today: _dt.date | None = None) -> Path | None:
    """当天没有备份时做一次在线备份(sqlite3 backup API),并清理过期备份。"""
    today = today or _dt.date.today()
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{_BACKUP_PREFIX}{today:%Y%m%d}.db"
    made = None
    if not target.exists():
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        src = sqlite3.connect(db_path, timeout=60)
        dst = sqlite3.connect(str(tmp))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        os.replace(tmp, target)
        made = target
    cutoff = today - _dt.timedelta(days=keep_days)
    for p in backup_dir.glob(f"{_BACKUP_PREFIX}*.db"):
        try:
            d = _dt.datetime.strptime(p.stem[len(_BACKUP_PREFIX):], "%Y%m%d").date()
        except ValueError:
            continue
        if d < cutoff:
            p.unlink()
    return made


def write_changelogs(repo, out_dir: Path) -> list[Path]:
    """每个命名空间一份 Markdown 更新日志(按日期分节),整份重写。"""
    from sqlalchemy import select

    from runtime.library import schema as S
    from runtime.library.curation import changelog, render_changelog

    with repo.store.engine.connect() as conn:
        spaces = sorted({r[0] for r in conn.execute(
            select(S.changelog_entries.c.namespace).distinct())})
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ns in spaces:
        path = out_dir / f"{ns.replace('/', '__')}.md"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(render_changelog(changelog(repo.store, ns, limit=100000), ns),
                       encoding="utf-8")
        os.replace(tmp, path)
        written.append(path)
    return written


def _maintenance_loop(repo, db_path: str | None, root: Path, keep_days: int,
                      stop: threading.Event, publisher=None) -> None:
    """每小时:SQLite 在线备份(当天没有时)+ 重写 CHANGELOG/<namespace>.md + 镜像兜底。"""
    while not stop.is_set():
        if db_path:
            try:
                made = backup_sqlite(db_path, root / "backups", keep_days)
                if made:
                    log.info("备份完成: %s", made)
            except Exception:  # noqa: BLE001  备份失败不影响服务
                log.exception("备份失败")
        try:
            write_changelogs(repo, root / "CHANGELOG")
        except Exception:  # noqa: BLE001
            log.exception("更新日志生成失败")
        if publisher is not None:
            publisher.notify()
        stop.wait(3600)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ModelEvo 经验库服务")
    ap.add_argument("--root", required=True,
                    help="库根目录:library.db / cas / backups / run 的缺省位置")
    ap.add_argument("--db-url", default="", help="空 = sqlite:///{root}/library.db")
    ap.add_argument("--artifacts-url", default="", help="空 = file://{root}/cas")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--backup-keep-days", type=int, default=30)
    ap.add_argument("--pidfile", default="", help="空 = {root}/run/library.pid")
    ap.add_argument("--no-browse", action="store_true",
                    help="不生成明文镜像({root}/browse/<namespace>)")
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--tokens", default="",
                    help="token 文件(JSON/YAML);启用认证。监听非本机地址时必需")
    ap.add_argument("--insecure-no-auth", action="store_true",
                    help="允许在非本机地址上无认证监听(仅限隔离网络调试)")
    return ap


_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tokens = None
    if args.tokens:
        from runtime.library.identity import load_tokens

        tokens = load_tokens(args.tokens)
        log.info("token 认证已启用:%d 个 token", len(tokens))
    elif args.host not in _LOOPBACK and not args.insecure_no_auth:
        log.error("监听 %s(非本机)必须给 --tokens;隔离网络调试可加 --insecure-no-auth",
                  args.host)
        return 2
    root = Path(args.root)
    (root / "run").mkdir(parents=True, exist_ok=True)
    d_db, d_art = default_urls(str(root))
    db_url, art_url = args.db_url or d_db, args.artifacts_url or d_art

    store = LibraryStore(db_url)          # 含 Alembic upgrade head
    repo = LocalRepository(store, ArtifactStore(art_url))
    log.info("经验库就绪: %s", json.dumps(repo.health(), ensure_ascii=False))

    stop = threading.Event()
    publisher = None
    if not args.no_browse:
        from runtime.library.browse import BrowsePublisher
        publisher = BrowsePublisher(repo, root / "browse", logger=log).start()
    threading.Thread(target=_maintenance_loop, name="library-maintenance", daemon=True,
                     args=(repo, sqlite_path(db_url), root, args.backup_keep_days,
                           stop, publisher)).start()

    pidfile = Path(args.pidfile or root / "run" / "library.pid")
    pidfile.write_text(json.dumps({
        "pid": os.getpid(), "host": socket.gethostname(), "bind": args.host,
        "auth": "token" if tokens else "none",
        "port": args.port, "db_url": db_url, "artifacts_url": art_url,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}), encoding="utf-8")

    import uvicorn

    from service.app import create_app

    try:
        uvicorn.run(create_app(repo, on_write=publisher.notify if publisher else None,
                               tokens=tokens),
                    host=args.host, port=args.port,
                    log_level=args.log_level, access_log=False)
    finally:
        stop.set()
        if publisher is not None:
            publisher.stop()
        try:
            if json.loads(pidfile.read_text(encoding="utf-8")).get("pid") == os.getpid():
                pidfile.unlink()
        except (OSError, ValueError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
