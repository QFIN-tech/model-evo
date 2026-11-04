"""pytest 全局夹具:v2.18.0 起所有测试与真实经验库隔离。

经验库客户端默认连 ``library.uri``(本机服务)。测试进程一律指向不可达端口、
本机缓存写临时目录 —— 放在 conftest(而非各文件 import 的 _common.py)里,
保证单独运行任一测试文件时也不会读写正式库。需要库的用例自行起嵌入式库。
"""
import os
import tempfile

os.environ.setdefault("MODELEVO_LIBRARY_URI", "http://127.0.0.1:9")
os.environ.setdefault("MODELEVO_LIBRARY_CACHE", tempfile.mkdtemp(prefix="mevo-lib-cache-"))
