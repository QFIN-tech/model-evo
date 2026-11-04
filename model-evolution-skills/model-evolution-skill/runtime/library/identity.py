"""v2.18.0 经验库:写入方身份(Q25)。

客户端每次写请求自报 用户 / 主机 / 团队 / run / 模型,服务写进 ``commits.actor`` 与
更新日志。HTTP 传输用 ``X-Modelevo-*`` 请求头。

v2.18.1:服务可选 token 认证(``serve --tokens <file>``,跨机器开放监听时必需)。启用后
user / team 以 token 登记的为准,不再信任自报;host / run / model 仍自报。
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket

ACTOR_FIELDS = ("user", "host", "team", "run", "model")
_HEADER_PREFIX = "X-Modelevo-"
#: 模型名:字母开头,允许 . _ : - 与 ``[1m]`` 一类后缀,≤64 字符
_MODEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:\[\]-]{0,63}$")
#: 明显不是模型名的写法(v2.18.6:正式库 5 个 run 快照里 2 个把 trial id 当 --model 传了)
_NOT_MODEL_RE = re.compile(r"^(trial|prop|run\d*|gen|g\d+)[-_]", re.IGNORECASE)


def valid_model(value) -> bool:
    v = str(value or "").strip()
    return bool(_MODEL_RE.match(v)) and not _NOT_MODEL_RE.match(v)


def resolve_model(model: str | None = None) -> tuple[str, str | None]:
    """→ (模型名, 被拒的显式取值)。显式 ``model`` 不像模型名(trial / run id、路径等)
    时丢弃并回落到环境变量 ``MODELEVO_MODEL`` / ``ANTHROPIC_MODEL``;都没有则为空。
    不抛错:收尾导出不能因为来源字段写错而失败。"""
    explicit = str(model or "").strip()
    rejected = explicit if explicit and not valid_model(explicit) else None
    for v in (explicit, os.environ.get("MODELEVO_MODEL"), os.environ.get("ANTHROPIC_MODEL")):
        if v and valid_model(v):
            return str(v).strip(), rejected
    return "", rejected


def current_actor(*, run: str | None = None, model: str | None = None,
                  team: str | None = None) -> dict:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001  容器里可能没有 passwd 条目
        user = os.environ.get("USER") or "unknown"
    return {
        "user": user,
        "host": socket.gethostname(),
        "team": team if team is not None else os.environ.get("MODELEVO_LIBRARY_TEAM", ""),
        "run": run or "",
        "model": resolve_model(model)[0],
    }


def actor_headers(actor: dict) -> dict[str, str]:
    return {f"{_HEADER_PREFIX}{k.capitalize()}": str(actor.get(k) or "")
            for k in ACTOR_FIELDS}


def actor_from_headers(headers) -> dict:
    """``headers``:任意大小写不敏感的映射(Starlette Headers / dict)。"""
    lower = {str(k).lower(): v for k, v in dict(headers).items()}
    return {k: lower.get(f"{_HEADER_PREFIX}{k}".lower(), "") for k in ACTOR_FIELDS}


# --------------------------------------------------------------------------
# token 认证(v2.18.1)
# --------------------------------------------------------------------------

def token_hash(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def load_tokens(path) -> dict[str, dict]:
    """token 文件(JSON 或 YAML):``{"tokens": [{"sha256": <hex> | "token": <明文>,
    "user": .., "team": .., "write": true}]}`` → {sha256: 条目}。建议只存 sha256
    (``library token-hash`` 生成)。``write`` 缺省 true;false 的 token 只能读。"""
    text = open(path, encoding="utf-8").read()
    try:
        data = json.loads(text)
    except ValueError:
        import yaml

        data = yaml.safe_load(text)
    out: dict[str, dict] = {}
    for e in (data or {}).get("tokens") or []:
        if not isinstance(e, dict):
            continue
        h = str(e.get("sha256") or (token_hash(e["token"]) if e.get("token") else "")).lower()
        if not h or not e.get("user"):
            raise ValueError(f"token 条目须有 sha256/token 与 user: {e!r}")
        out[h] = {"user": str(e["user"]), "team": str(e.get("team") or ""),
                  "write": bool(e.get("write", True))}
    if not out:
        raise ValueError(f"{path} 里没有 token")
    return out


def bearer(headers) -> str | None:
    lower = {str(k).lower(): v for k, v in dict(headers).items()}
    v = str(lower.get("authorization") or "")
    return v[7:].strip() or None if v.lower().startswith("bearer ") else None
