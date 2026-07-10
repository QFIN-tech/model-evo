# -*- coding: utf-8 -*-
"""classification-model-task-spec 样本拉取脚本。

职责: 需求确认后, 从样本表拉取 user_no/label/pday + 补充字段(不拉特征列),
落成 sample.parquet 供 run_sample_analysis_task_spec.py 分析。

与 feature-matching/scripts/fetch_sample.py 的区别:
  - 不做特征清单加载 / feature-list.csv 派生(task-spec 阶段不需要特征)
  - 不支持 feature_list_source / 全列模式
  - features 字段在 task-spec 语义里 = "补充字段清单"(供后续报告补充分析)
  - 复用同一套公共取数代码: config_io / gen_fetch_command / fetch_spark

两种模式:
  - --mode spark (默认): 走 spark-submit 拉 your_db 表, 落 yaml + 生成 fetch 脚本 + (可选) --submit
  - --mode local_file: 跳过 spark, shutil.copyfile 本地 parquet 到 data-profile/,
    落 yaml (model.mode=local_file), 不生成 spark-submit 脚本, --submit 无效

CLI 模式: --session-dir + 关键参数直传, 脚本内部自动生成 yaml 落到
<session_dir>/task-spec/sample_config.<model_name>.yaml, 不再接收 --config,
机制上强制配置文件落 session 目录, 避免误落 skill config/ 目录。

公共模块: model-evo/_modelevo-shared/scripts, 通过 _bootstrap.py 注入。
安全: 仅取所需列, 不输出用户级明细到日志。
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

import yaml


def _parse_range(text: str) -> List[str]:
    """解析 '起,止' 为 [起, 止] 列表 (YYYYMMDD 8 位)。"""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("区间须为两元素 起,止, 如 20260312,20260430, 当前 %r" % text)
    for d in parts:
        if not (d.isdigit() and len(d) == 8):
            raise ValueError("区间日期须为 8 位 YYYYMMDD, 当前 %r" % d)
    if parts[0] > parts[1]:
        raise ValueError("区间起始 %s 不应大于结束 %s" % (parts[0], parts[1]))
    return parts


def _build_cfg(args: argparse.Namespace) -> dict:
    """从 CLI 参数构造配置 dict (含 spark_submit + model 段)。

    local_file 模式下 sample_table/fetch_dt 仅作占位/记录, spark_submit.hdfs_base 留空。
    """
    features: List[str] = []
    if args.features:
        features = [c.strip() for c in args.features.split(",") if c.strip()]

    id_cols: List[str] = [c.strip() for c in (args.id_cols or "user_no").split(",") if c.strip()]

    if args.mode == "local_file":
        # local_file 模式: 不走 spark, hdfs_base 留空, sample_table 占位
        cfg = {
            "spark_submit": {"hdfs_base": ""},
            "model": {
                "name": args.model_name,
                "version": args.version,
                "mode": "local_file",
                "sample_table": args.sample_table or "local_file",
                "local_parquet_path": args.local_parquet_path,
                "dt_col": args.dt_col,
                "id_cols": id_cols,
                "fetch_dt": [args.fetch_start or "", args.fetch_end or ""],
                "where": args.where,
                "split": {
                    "train_range": _parse_range(args.train_range),
                    "test_range": _parse_range(args.test_range),
                    "oot_range": _parse_range(args.oot_range),
                },
                "features": features,
                "feature_list_source": None,
                "label_col": args.label_col,
                "label_expr": args.label_expr,
                "driving_mode": args.driving_mode or "manual",
            },
        }
        return cfg

    # spark 模式 (默认)
    from spark_defaults import default_hdfs_base

    hdfs_base = args.hdfs_base or default_hdfs_base("feature-matching")

    cfg = {
        "spark_submit": {
            # bin/options 留空, 走 gen_fetch_command._resolve_spark_cfg 兜底 _modelevo-shared/spark_defaults
            "hdfs_base": hdfs_base,
        },
        "model": {
            "name": args.model_name,
            "version": args.version,
            "sample_table": args.sample_table,
            "dt_col": args.dt_col,
            "id_cols": id_cols,
            "fetch_dt": [args.fetch_start, args.fetch_end],
            "where": args.where,
            "split": {
                "train_range": _parse_range(args.train_range),
                "test_range": _parse_range(args.test_range),
                "oot_range": _parse_range(args.oot_range),
            },
            "features": features,
            "feature_list_source": None,
            "label_col": args.label_col,
            "label_expr": args.label_expr,
            "driving_mode": args.driving_mode or "manual",
        },
    }
    return cfg


def _dump_yaml(cfg: dict, session_dir: str, model_name: str) -> str:
    """把 cfg 落成 yaml 到 <session_dir>/task-spec/sample_config.<model_name>.yaml。"""
    out_dir = os.path.join(session_dir, "task-spec")
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "sample_config.%s.yaml" % model_name)
    # 去掉 _config_dir 等运行时字段
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# task-spec 样本拉取配置 (由 fetch_sample_task_spec.py 自动落盘)\n")
        f.write("# model: %s\n\n" % model_name)
        yaml.safe_dump(clean, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return yaml_path


def _link_or_copy_local(src: str, dst: str) -> None:
    """把本地样本文件落到 dst: 优先硬链接(同盘零拷贝), 失败则软链接, 再失败才 copyfile。

    local_file 模式下 sample.parquet 由后续脚本只读消费(run_sample_analysis_task_spec /
    feature-analysis / model-training 都不改源文件), 无需整文件复制。
    .csv 输入仍走 pandas 读后写 parquet(格式转换必须复制)。

    Args:
        src: 源文件路径(.parquet 或 .csv)
        dst: 目标 parquet 路径
    """
    src_lower = src.lower()
    if src_lower.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(src)
        df.to_parquet(dst, index=False)
        return
    # .parquet: 优先硬链接(同盘 inode 共享, 零拷贝), 跨盘则软链接, 都不行才 copyfile
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(os.path.abspath(src), dst)
        return
    except OSError:
        pass
    shutil.copyfile(src, dst)


def main() -> None:
    """命令行入口: 读参数 → 构造 cfg → 校验 → 落 yaml → (spark 模式) 生成 spark-submit 脚本 → (可选) 提交。"""
    import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts
    from config_io import validate_split_ranges, check_sensitive

    parser = argparse.ArgumentParser(description="task-spec 样本拉取(--session-dir 自动落盘模式)")
    parser.add_argument("--session-dir", required=True, help="session 目录 (runs/<timestamp>-<model_name>)")
    parser.add_argument("--model-name", required=True, help="模型简称, 如 draw_willingness-t7")
    parser.add_argument("--mode", choices=["spark", "local_file"], default="spark",
                        help="取数模式: spark=走 spark-submit 拉 your_db 表; local_file=shutil.copyfile 本地 parquet")
    parser.add_argument("--local-parquet-path", default=None,
                        help="[mode=local_file 必填] 本地 parquet 路径, 含 id_cols+label_col+dt_col+(可选)features")
    parser.add_argument("--sample-table", default=None,
                        help="样本表 库.表, 如 your_db.xxx (spark 模式必填, local_file 模式仅记录用)")
    parser.add_argument("--fetch-start", default=None, help="取数起始日期 YYYYMMDD (spark 模式必填)")
    parser.add_argument("--fetch-end", default=None, help="取数结束日期 YYYYMMDD (spark 模式必填)")
    parser.add_argument("--train-range", required=True, help="Train pday 闭区间 起,止 (YYYYMMDD)")
    parser.add_argument("--test-range", required=True, help="Test pday 闭区间 起,止 (YYYYMMDD)")
    parser.add_argument("--oot-range", required=True, help="OOT pday 闭区间 起,止 (YYYYMMDD)")
    parser.add_argument("--label-col", default="label", help="标签列名 (默认 label)")
    parser.add_argument("--label-expr", default=None, help="SQL 标签表达式, 非空时替代 --label-col")
    parser.add_argument("--features", default=None, help="补充字段清单(逗号分隔), 不直接入模; 留空=仅样本三列")
    parser.add_argument("--id-cols", default="user_no", help="ID 列(逗号分隔), 默认 user_no")
    parser.add_argument("--dt-col", default="pday", help="日期分区字段, 默认 pday")
    parser.add_argument("--where", default=None, help="可选客群筛选条件")
    parser.add_argument("--version", default="v1", help="模型版本, 默认 v1")
    parser.add_argument("--hdfs-base", default=None, help="HDFS 中间目录, 默认 /user/<whoami>/feature-matching")
    parser.add_argument("--spark-bin", default=None, help="spark-submit 路径, 默认集群 3.3.2")
    parser.add_argument("--out", default=None, help="输出 parquet 路径, 默认 <session_dir>/data-profile/<model_name>_sample_<YYYYMMDD>.parquet")
    parser.add_argument("--submit", action="store_true", help="生成脚本后同步执行 bash <script> 提交集群 (spark 模式有效, local_file 模式无效)")
    parser.add_argument("--driving-mode", choices=["auto", "manual"], default=None,
                        help="驾驶模式: auto=全自动驾驶(写入 manifest 后下游 feature-matching/development 跳过交互, 用默认值); manual=辅助驾驶(默认, 走交互约定); 优先级: CLI > 默认 manual; task-spec 自身不读 manifest, 由 LLM 在 §3.1.5 检测关键词后通过 --driving-mode 透传")
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        raise SystemExit("session 目录不存在: %s" % session_dir)

    # local_file 模式必填校验
    if args.mode == "local_file":
        if not args.local_parquet_path:
            raise SystemExit("--mode local_file 时必须传 --local-parquet-path")
        if not os.path.isfile(args.local_parquet_path):
            raise SystemExit("本地 parquet 不存在: %s" % args.local_parquet_path)
    else:  # spark 模式必填校验
        if not args.sample_table:
            raise SystemExit("--mode spark 时必须传 --sample-table")
        if not args.fetch_start or not args.fetch_end:
            raise SystemExit("--mode spark 时必须传 --fetch-start / --fetch-end")

    cfg = _build_cfg(args)
    model = cfg["model"]

    # 校验: 安全线 + split 区间合法性 (local_file 模式跳过 spark 必填校验)
    check_sensitive(model.get("where") or "")
    check_sensitive(model.get("sample_table") or "")
    validate_split_ranges(model)

    # 落 yaml 到 session 目录 (机制上强制, 不可能落 skill config/)
    yaml_path = _dump_yaml(cfg, session_dir, model["name"])
    print("[fetch_sample_task_spec] 配置已落盘: %s" % yaml_path)


    # 输出 parquet 路径: 默认 <session_dir>/data-profile/<model_name>_sample_<YYYYMMDD>.parquet
    today = datetime.now().strftime("%Y%m%d")
    out_path = args.out or os.path.join(
        session_dir, "data-profile", "%s_sample_%s.parquet" % (model["name"], today)
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # local_file 模式: 用硬链接(同盘)或软链接(跨盘)替代 shutil.copyfile, 避免
    # 把同一份本地 parquet 整文件复制一份到 data-profile/(后续 run_sample_analysis_task_spec.py
    # 只读不改, 无需副本)。源是 .csv 时仍 copyfile 转 parquet。
    if args.mode == "local_file":
        _link_or_copy_local(args.local_parquet_path, out_path)
        print("[fetch_sample_task_spec] mode=local_file, 已链接本地 parquet:")
        print("  源: %s" % args.local_parquet_path)
        print("  目: %s" % out_path)
        extra_cols = model.get("features") or []
        if extra_cols:
            print("补充字段(不直接入模): %s" % ", ".join(extra_cols))
        else:
            print("仅样本三列(%s/%s/%s), 无补充字段" % (
                "/".join(model.get("id_cols") or []),
                model.get("label_col"),
                model.get("dt_col"),
            ))
        return

    # === spark 模式: 生成 spark-submit 包装脚本 ===
    from gen_fetch_command import build_command

    # HDFS 中间路径
    spark_cfg = cfg.get("spark_submit", {})
    hdfs_base = (spark_cfg.get("hdfs_base") or "").rstrip("/")
    if hdfs_base:
        version_tag = model.get("version", "vX")
        model_dir = "%s_%s" % (model["name"], version_tag)
        hdfs_out_path = "%s/%s/sample.parquet" % (hdfs_base, model_dir)
    else:
        hdfs_out_path = None

    # 生成 spark-submit 包装脚本
    cmd, _, _ = build_command(cfg, out_path, hdfs_out_path)

    gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated")
    os.makedirs(gen_dir, exist_ok=True)

    tag = "%s_%s" % (model["name"], model.get("version", "vX"))
    script_path = os.path.join(gen_dir, "fetch_%s.sh" % tag)

    # 拼 wrapper 脚本: spark-submit → hdfs dfs -get
    script_lines = [
        "#!/bin/bash", "set -e",
        "# 自动生成的 task-spec 样本拉取脚本, 请检查后提交",
        cmd,
    ]
    if hdfs_out_path:
        local_dir = os.path.dirname(out_path)
        script_lines += [
            "",
            "# 从 HDFS 拉取到本地",
            "mkdir -p %s" % shlex.quote(local_dir),
            "hdfs dfs -get -f %s %s" % (shlex.quote(hdfs_out_path), shlex.quote(out_path)),
            "echo %s" % shlex.quote("已拉取到本地: %s" % out_path),
        ]

    script_content = "\n".join(script_lines) + "\n"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR)

    print("已生成样本拉取脚本: %s" % script_path)
    if hdfs_out_path:
        print("HDFS 中间输出:    %s" % hdfs_out_path)
    print("样本将输出到:     %s" % out_path)
    extra_cols = model.get("features") or []
    if extra_cols:
        print("补充字段(不直接入模): %s" % ", ".join(extra_cols))
    else:
        print("仅拉样本三列(user_no/label/pday), 无补充字段")
    print("\n提交命令:\n  bash %s\n" % script_path)
    print("脚本内容:\n%s" % script_content)

    if args.submit:
        import subprocess
        print("[fetch_sample_task_spec] 提交到集群: bash %s" % script_path)
        result = subprocess.run(["bash", script_path])
        if result.returncode != 0:
            print("[fetch_sample_task_spec] 提交失败, returncode=%d" % result.returncode)
            sys.exit(result.returncode)
        print("[fetch_sample_task_spec] 完成, sample.parquet 已落到 %s" % out_path)


if __name__ == "__main__":
    main()
