#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""skills 公共取数提交脚本生成器。

读取 yaml 配置,生成一个可直接提交的 spark-submit wrapper 脚本
(generated/fetch_{name}_{version}.sh),供用户手动提交到集群运行 fetch_spark.py。
被 feature-analysis 与 model-training 共用。

用法:
    python gen_fetch_command.py --config <config.yaml> \
        [--out runs/sample.parquet]
"""

import argparse
import os
import stat
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_spark_defaults():
    """从同目录 spark_defaults.py 取默认值。

    不做 ImportError 兜底: 本模块与 spark_defaults.py 同处 _modelevo-shared/scripts/,
    若不可 import 说明部署形态异常, 应直接抛错而非用过期副本静默运行。
    """
    from spark_defaults import DEFAULT_SPARK_BIN, DEFAULT_SPARK_OPTIONS
    return DEFAULT_SPARK_BIN, DEFAULT_SPARK_OPTIONS


_DEFAULT_SPARK_BIN, _DEFAULT_SPARK_OPTIONS = _load_spark_defaults()


def _shquote(value):
    """对含空格/单引号的参数值做最简单的 shell 引用。"""
    if value is None:
        return value
    s = str(value)
    if s == "" or any(ch in s for ch in " '\"()=<>"):
        return "'%s'" % s.replace("'", "'\\''")
    return s


def _resolve_spark_cfg(spark_cfg: dict) -> tuple:
    """解析 spark_submit 段, 缺字段时从 spark_defaults 兜底。

    Args:
        spark_cfg: cfg["spark_submit"] (可能为 {} 或缺 bin/options)

    Returns:
        (submit_bin, options_list, hdfs_base) 三元组
    """
    spark_cfg = spark_cfg or {}
    submit_bin = spark_cfg.get("bin") or _DEFAULT_SPARK_BIN
    options = spark_cfg.get("options")
    if not options:
        options = list(_DEFAULT_SPARK_OPTIONS)
    hdfs_base = (spark_cfg.get("hdfs_base") or "").rstrip("/")
    return submit_bin, options, hdfs_base


def build_command(cfg, out_path, hdfs_out_path=None):
    """根据配置拼出完整 spark-submit 命令。

    参数:
        cfg: build_config 字典(含 spark_submit/model 段)
        out_path: 最终本地输出 parquet 路径
        hdfs_out_path: HDFS 中间输出路径(None 则直写本地)
    返回:
        (命令字符串, 输出路径, hdfs_out_path)
    """
    model = cfg["model"]
    spark_cfg = cfg.get("spark_submit", {})
    submit_bin, options, _ = _resolve_spark_cfg(spark_cfg)

    features = model.get("features", [])
    id_cols = model.get("id_cols", [])
    fetch_dt = model["fetch_dt"]

    # spark-submit 写 HDFS(executor 有权限), 再 hdfs dfs -get 拉到本地
    spark_out = hdfs_out_path if hdfs_out_path else out_path

    # 样本表⋈特征表模式: 配置含 feature_table 时启用, 样本表用 sample_table
    feature_table = model.get("feature_table")
    if feature_table:
        eval_parts = [
            os.path.join(HERE, "fetch_spark.py"),
            "--sample-table", model["sample_table"],
            "--feature-table", feature_table,
            "--dt-col", model.get("dt_col", "pday"),
            "--fetch-start", str(fetch_dt[0]),
            "--fetch-end", str(fetch_dt[1]),
            "--out", spark_out,
        ]
        # features 留空 = 取特征表全部列, 不传 --features
        if features:
            eval_parts += ["--features", ",".join(features)]
        join_keys = model.get("join_keys") or ["user_no", model.get("dt_col", "pday")]
        join_keys = ",".join(join_keys) if isinstance(join_keys, (list, tuple)) else str(join_keys)
        eval_parts += ["--join-keys", join_keys]
        if id_cols:
            eval_parts += ["--id-cols", ",".join(id_cols)]
        if model.get("label_expr"):
            eval_parts += ["--label-expr", model["label_expr"]]
        elif model.get("label_col"):
            eval_parts += ["--label-col", model["label_col"]]
        if model.get("where"):
            eval_parts += ["--where", model["where"]]
        if model.get("feature_lag_day", 0) != 0:
            eval_parts += ["--feature-lag-day", str(model["feature_lag_day"])]
        eval_str = " ".join(_shquote(p) if i >= 1 else p for i, p in enumerate(eval_parts))
        prefix_lines = [submit_bin] + [str(o) for o in options]
        cmd = " \\\n  ".join(prefix_lines) + " \\\n  " + eval_str
        return cmd, out_path, hdfs_out_path

    # fetch_spark.py 及其参数(单表模式)
    eval_parts = [
        os.path.join(HERE, "fetch_spark.py"),
        "--table", model["sample_table"],
        "--dt-col", model.get("dt_col", "pday"),
        "--fetch-start", str(fetch_dt[0]),
        "--fetch-end", str(fetch_dt[1]),
        "--out", spark_out,
    ]
    # features 留空 = 仅拉样本三列(task-spec 阶段), 不传 --features
    if features:
        eval_parts += ["--features", ",".join(features)]
    if id_cols:
        eval_parts += ["--id-cols", ",".join(id_cols)]
    if model.get("label_expr"):
        eval_parts += ["--label-expr", model["label_expr"]]
    elif model.get("label_col"):
        eval_parts += ["--label-col", model["label_col"]]
    if model.get("where"):
        eval_parts += ["--where", model["where"]]

    eval_str = " ".join(_shquote(p) if i >= 1 else p for i, p in enumerate(eval_parts))

    # 集群固化参数前缀
    prefix_lines = [submit_bin] + [str(o) for o in options]
    cmd = " \\\n  ".join(prefix_lines) + " \\\n  " + eval_str
    return cmd, out_path, hdfs_out_path


def main():
    """入口:读配置、拼命令、写出可执行 wrapper 脚本。"""
    parser = argparse.ArgumentParser(description="生成 skills 公共取数提交脚本")
    parser.add_argument("--config", required=True, help="配置 yaml 路径")
    parser.add_argument("--out", default=None, help="输出 parquet 路径,默认 runs/sample.parquet")
    args = parser.parse_args()

    sys.path.insert(0, HERE)
    from config_io import load_config, validate_common
    cfg = load_config(args.config)
    validate_common(cfg)

    model = cfg["model"]
    out_path = args.out or os.path.join(
        os.path.dirname(args.config), "..", "runs", "sample.parquet"
    )
    out_path = os.path.abspath(out_path)

    # HDFS 中间路径: {hdfs_base}/{name}_{version}/sample.parquet
    spark_cfg = cfg.get("spark_submit", {})
    _, _, hdfs_base = _resolve_spark_cfg(spark_cfg)
    if hdfs_base:
        version_tag = model.get("version", "vX")
        model_dir = "%s_%s" % (model["name"], version_tag)
        hdfs_out_path = "%s/%s/sample.parquet" % (hdfs_base, model_dir)
    else:
        hdfs_out_path = None

    cmd, out_path, hdfs_out_path = build_command(cfg, out_path, hdfs_out_path)

    gen_dir = os.path.join(HERE, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    tag = "%s_%s" % (model["name"], model.get("version", "vX"))
    script_path = os.path.join(gen_dir, "fetch_%s.sh" % tag)

    # 拼 wrapper 脚本: spark-submit → hdfs dfs -get
    script_lines = ["#!/bin/bash", "set -e", "# 自动生成的 skills 公共取数提交脚本,请检查后提交", cmd]
    if hdfs_out_path:
        local_dir = os.path.dirname(out_path)
        script_lines += [
            "",
            "# 从 HDFS 拉取到本地(先清本地目标, 防 -f 只覆盖同名文件残留旧 part)",
            "mkdir -p %s" % _shquote(local_dir),
            "rm -rf %s" % _shquote(out_path),
            "hdfs dfs -get -f %s %s" % (_shquote(hdfs_out_path), _shquote(out_path)),
            'echo "已拉取到本地: %s"' % out_path,
        ]
    script_content = "\n".join(script_lines) + "\n"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR)

    print("已生成取数提交脚本: %s" % script_path)
    if hdfs_out_path:
        print("HDFS 中间输出:    %s" % hdfs_out_path)
    print("样本将输出到:     %s" % out_path)
    print("\n提交命令:\n  bash %s\n" % script_path)
    print("脚本内容:\n%s" % script_content)


if __name__ == "__main__":
    main()
