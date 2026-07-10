# -*- coding: utf-8 -*-
"""model-evo 公共 Spark 提交默认值(yaml 加载器)。

所有取数类 skill(feature-matching / classification-model-task-spec /
classification-model-recommend)的 fetch_*.py 通过 _bootstrap 注入 _modelevo-shared/scripts
后, 从这里 import DEFAULT_SPARK_BIN / DEFAULT_SPARK_OPTIONS / default_hdfs_base,
避免三处副本漂移。

加载顺序:
  1. 同目录 spark_defaults.yaml (真实配置; 不入库, 由 install.sh 从 template 拷贝生成)
  2. 同目录 spark_defaults.template.yaml (占位模板; 走这条路径会 RuntimeWarning)

default_hdfs_base 因含 getpass.getuser() 动态值, 仍以 Python 函数形式提供。
"""
from __future__ import annotations

import getpass
import os
import warnings

import yaml


_HERE = os.path.dirname(os.path.abspath(__file__))
_YAML_PATH = os.path.join(_HERE, "spark_defaults.yaml")
_TEMPLATE_YAML_PATH = os.path.join(_HERE, "spark_defaults.template.yaml")


def _load_yaml():
    """优先读 spark_defaults.yaml, 缺失时回退到 spark_defaults.template.yaml。"""
    path = _YAML_PATH if os.path.exists(_YAML_PATH) else _TEMPLATE_YAML_PATH
    if path == _TEMPLATE_YAML_PATH:
        warnings.warn(
            "spark_defaults.yaml 未找到, 已回退到 spark_defaults.template.yaml "
            "中的占位符。请复制该模板为 spark_defaults.yaml 并填入你集群的真实值, "
            "否则 spark-submit 会因占位符而失败。",
            RuntimeWarning,
            stacklevel=2,
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["bin"], list(data.get("options") or []), data.get("hdfs_user")


DEFAULT_SPARK_BIN, DEFAULT_SPARK_OPTIONS, DEFAULT_HDFS_USER = _load_yaml()


def default_hdfs_base(skill_name: str) -> str:
    """返回 <skill_name> 默认 HDFS 中间目录: /user/<hdfs_user>/<skill_name>。

    hdfs_user 优先取 spark_defaults.yaml 的 hdfs_user 字段(钉死 spark-submit
    实际认证用户, 与 OS 登录用户解耦); yaml 缺失时回退到 getpass.getuser()。

    Args:
        skill_name: skill 短名(如 "feature-matching" / "model-recommend")

    Returns:
        HDFS 路径字符串
    """
    user = DEFAULT_HDFS_USER or getpass.getuser()
    return "/user/%s/%s" % (user, skill_name)
