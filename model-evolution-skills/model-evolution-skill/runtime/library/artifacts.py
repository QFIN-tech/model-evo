"""v2.18.0 经验库:内容寻址产物存储(fsspec:file / s3 / hdfs)。

对象路径 ``<root>/sha256/<h[:2]>/<h>``,只写一次、不修改。
bundle = 一组文件的清单 ``{"schema": "modelevo.bundle/1", "files": {name: sha256}}``,
清单按规范 JSON 存为对象,其哈希即 bundle id(配方的 recipe_id)。
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid

BUNDLE_SCHEMA = "modelevo.bundle/1"


class ArtifactError(Exception):
    """产物缺失或内容与哈希不符。"""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    """规范 JSON:排序键、UTF-8、无多余空白。哈希与 bundle 清单都用它。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def is_sha256(h: str) -> bool:
    return isinstance(h, str) and len(h) == 64 and \
        all(c in "0123456789abcdef" for c in h)


class ArtifactStore:
    """按 sha256 存取对象。``url`` 例:``file:///data/.../cas``、``s3://b/p``。"""

    def __init__(self, url: str):
        import fsspec  # 延迟导入:只读元数据的调用方不需要 fsspec

        self.url = url
        self.fs, root = fsspec.core.url_to_fs(url)
        self.root = root.rstrip("/")
        protocol = self.fs.protocol
        protocols = protocol if isinstance(protocol, (tuple, list)) else (protocol,)
        self._local = any(p in ("file", "local") for p in protocols)

    def _path(self, h: str) -> str:
        return f"{self.root}/sha256/{h[:2]}/{h}"

    def exists(self, h: str) -> bool:
        return is_sha256(h) and self.fs.exists(self._path(h))

    def put_bytes(self, data: bytes, expected: str | None = None) -> str:
        """写入对象,返回 sha256。已存在则不重写;``expected`` 不符时报错。"""
        h = sha256_hex(data)
        if expected is not None and expected != h:
            raise ArtifactError(f"内容哈希 {h} 与声明的 {expected} 不符")
        path = self._path(h)
        if self.fs.exists(path):
            return h
        if self._local:
            # 同目录临时文件 + os.replace:并发写同一对象时结果一致且无半截文件
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                with open(tmp, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        else:
            # 对象存储的单次 put 本身是原子的
            self.fs.pipe_file(path, data)
        return h

    def get_bytes(self, h: str) -> bytes:
        if not is_sha256(h):
            raise ArtifactError(f"非法对象哈希: {h!r}")
        path = self._path(h)
        if not self.fs.exists(path):
            raise ArtifactError(f"对象不存在: {h}")
        data = self.fs.cat_file(path)
        if sha256_hex(data) != h:
            raise ArtifactError(f"对象内容损坏: {h}")
        return data

    def iter_hashes(self):
        """遍历全部对象哈希(dump / 迁移用)。"""
        base = f"{self.root}/sha256"
        if not self.fs.exists(base):
            return
        for path in sorted(self.fs.find(base)):
            name = path.rstrip("/").rsplit("/", 1)[-1]
            if is_sha256(name):
                yield name


def build_bundle_manifest(file_hashes: dict[str, str]) -> bytes:
    return canonical_json({"schema": BUNDLE_SCHEMA,
                           "files": dict(sorted(file_hashes.items()))})


def parse_bundle_manifest(data: bytes) -> dict[str, str]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"bundle 清单不是合法 JSON: {exc}") from exc
    if not isinstance(obj, dict) or obj.get("schema") != BUNDLE_SCHEMA or \
            not isinstance(obj.get("files"), dict):
        raise ArtifactError("不是 modelevo.bundle/1 清单")
    files = obj["files"]
    bad = [n for n, h in files.items() if not is_sha256(h)]
    if bad:
        raise ArtifactError(f"bundle 清单含非法哈希: {bad[:3]}")
    return files
