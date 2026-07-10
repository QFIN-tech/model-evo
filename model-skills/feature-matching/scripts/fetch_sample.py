# -*- coding: utf-8 -*-
"""feature-matching 样本+特征拉取脚本。

职责: 从样本表(⋈特征表)拉取 user_no/label/pday + 特征列,
落成 sample.parquet 供下游 feature-analysis / classification-model-training 复用。

与 classification-model-task-spec/scripts/fetch_sample_task_spec.py 的区别:
  - 支持样本表⋈特征表 LEFT JOIN 模式(--feature-table)
  - 支持 features 留空 = 取特征表全部列(由 fetch_spark 运行时展开)
  - 支持 --feature-list-source / --no-filter-feas 等特征清单控制
  - 生成 feature-list.csv 供下游引用
  - 复用同一套公共取数代码: config_io / gen_fetch_command / fetch_spark

两种模式:
  - --mode spark (默认): 走 spark-submit 拉样本表/特征宽表, 落 yaml + 生成 fetch 脚本 + (可选) --submit
  - --mode local_file: 跳过 spark, 把本地 parquet/csv 转写为 sample.parquet 落到
    <session_dir>/sample-features/feature-matching/, 落 yaml (model.mode=local_file),
    调 derive_feature_list.py 派生 feature-list.csv。
    输入 .parquet 走 shutil.copyfile 直接复制, .csv 走 pandas 读后写 parquet,
    输出统一为 sample.parquet。
    不生成 spark-submit 脚本, --submit 无效(转写+派生已完成)

具体公司样本表/特征表/业务域取值由 model-knowledge/assets/ 下知识库登记,
本脚本不内置任何公司表名/业务域名/模型简称, CLI 参数 help 中均用占位符。

CLI 模式: --session-dir + 关键参数直传, 脚本内部自动生成 yaml 落到
<session_dir>/sample-features/feature-matching/sample_config.<model_name>.yaml,
不接收 --config, 机制上强制配置文件落 session 目录, 避免误落 skill config/ 目录。

公共模块: model-evo/_modelevo-shared/scripts, 通过 _bootstrap.py 注入。
安全: 仅取所需列, 不输出用户级明细到日志。
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import stat
import sys
from datetime import datetime
from typing import List, Optional

import yaml


def _build_cfg(args: argparse.Namespace) -> dict:
    """从 CLI 参数构造配置 dict (含 spark_submit + model 段)。

    local_file 模式下 sample_table/feature_table/join_keys/fetch_dt 仅作占位/记录,
    spark_submit.hdfs_base 留空。
    """
    features: List[str] = []
    if args.features:
        features = [c.strip() for c in args.features.split(",") if c.strip()]

    id_cols: List[str] = [c.strip() for c in (args.id_cols or "user_no").split(",") if c.strip()]

    join_keys: Optional[List[str]] = None
    if args.join_keys:
        join_keys = [c.strip() for c in args.join_keys.split(",") if c.strip()]
    elif args.feature_table:
        join_keys = ["user_no", args.dt_col or "pday"]

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
                "feature_table": args.feature_table,
                "join_keys": join_keys,
                "dt_col": args.dt_col,
                "id_cols": id_cols,
                "fetch_dt": [args.fetch_start or "", args.fetch_end or ""],
                "where": args.where,
                "features": features,
                "feature_list_source": args.feature_list_source,
                "business_domain": args.business_domain,
                "label_col": args.label_col,
                "label_expr": args.label_expr,
                "feature_lag_day": args.feature_lag_day,
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
            "feature_table": args.feature_table,
            "join_keys": join_keys,
            "dt_col": args.dt_col,
            "id_cols": id_cols,
            "fetch_dt": [args.fetch_start, args.fetch_end],
            "where": args.where,
            "features": features,
            "feature_list_source": args.feature_list_source,
            "business_domain": args.business_domain,
            "label_col": args.label_col,
            "label_expr": args.label_expr,
            "feature_lag_day": args.feature_lag_day,
        },
    }
    return cfg


def _dump_yaml(cfg: dict, session_dir: str, model_name: str) -> str:
    """把 cfg 落成 yaml 到 <session_dir>/sample-features/feature-matching/sample_config.<model_name>.yaml。"""
    out_dir = os.path.join(session_dir, "sample-features", "feature-matching")
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "sample_config.%s.yaml" % model_name)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# feature-matching 样本+特征拉取配置 (由 fetch_sample.py 自动落盘)\n")
        f.write("# model: %s\n\n" % model_name)
        yaml.safe_dump(clean, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return yaml_path


def main() -> None:
    """命令行入口: 读参数 → 构造 cfg → 校验 → 落 yaml → (spark 模式) 生成 spark-submit 脚本 → (可选) 提交 / (local_file 模式) 复制+派生。"""
    import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts
    from config_io import validate_common, check_sensitive
    from feature_knowledge import resolve_feature_list_csv
    from gen_feature_list import load_feature_list, write_feature_list_csv

    parser = argparse.ArgumentParser(description="feature-matching 取数(--session-dir 自动落盘模式)")
    parser.add_argument("--session-dir", required=True, help="session 目录 (runs/<timestamp>-<model_name>)")
    parser.add_argument("--model-name", required=True, help="模型简称, 由 task-spec 阶段确认")
    parser.add_argument("--mode", choices=["spark", "local_file"], default="spark",
                        help="取数模式: spark=走 spark-submit 拉样本表/特征宽表; local_file=本地 parquet/csv 转写为 sample.parquet + 派生特征清单")
    parser.add_argument("--local-parquet-path", default=None,
                        help="[mode=local_file 必填] 本地 parquet/csv 路径, 含 id_cols+label_col+dt_col+features 全量列; .csv 输入会被读后转写为 sample.parquet")
    parser.add_argument("--sample-table", default=None,
                        help="样本表 库.表 (spark 模式必填, local_file 模式仅记录用); 公司实际表名由 model-knowledge 知识库登记")
    parser.add_argument("--feature-table", default=None, help="特征表 库.表(拼接模式); 留空=单表模式 (local_file 模式仅记录用); 公司实际表名由 model-knowledge 知识库登记")
    parser.add_argument("--join-keys", default=None, help="拼接键(逗号分隔), 默认 user_no,pday; 仅拼接模式生效 (local_file 模式仅记录用)")
    parser.add_argument("--fetch-start", default=None, help="取数起始日期 YYYYMMDD (spark 模式必填)")
    parser.add_argument("--fetch-end", default=None, help="取数结束日期 YYYYMMDD (spark 模式必填)")
    parser.add_argument("--label-col", default="label", help="标签列名 (默认 label)")
    parser.add_argument("--label-expr", default=None, help="SQL 标签表达式, 非空时替代 --label-col")
    parser.add_argument("--features", default=None, help="特征列(逗号分隔); 留空=取特征表全部列(拼接模式) 或 仅样本三列(单表模式) / local_file 模式下留空=派生全量列")
    parser.add_argument("--feature-list-source", default=None, help="特征清单文件(.txt/.csv); 留空按 feature-knowledge.md 索引自动识别")
    parser.add_argument("--business-domain", default=None, help="业务域, 自动识别特征清单时的兜底匹配键; feature_table 未命中索引时用它匹配「分场景」列; 具体取值由 model-knowledge 知识库登记")
    parser.add_argument("--id-cols", default="user_no", help="ID 列(逗号分隔), 默认 user_no")
    parser.add_argument("--dt-col", default="pday", help="日期分区字段, 默认 pday")
    parser.add_argument("--where", default=None, help="可选客群筛选条件; 严禁硬编码用户ID/手机号/身份证号")
    parser.add_argument("--version", default="v1", help="模型版本, 默认 v1")
    parser.add_argument("--hdfs-base", default=None, help="HDFS 中间目录, 默认 /user/<whoami>/feature-matching")
    parser.add_argument("--spark-bin", default=None, help="spark-submit 路径, 默认集群 3.3.2")
    parser.add_argument("--out", default=None, help="输出 parquet 路径, 默认 <session_dir>/sample-features/feature-matching/sample.parquet")
    parser.add_argument("--submit", action="store_true", help="生成脚本后同步执行 bash <script> 提交集群 (spark 模式有效, local_file 模式无效)")
    parser.add_argument("--no-filter-feas", action="store_true", help="跳过特征清单过滤, 派生全量 feature-list.csv")
    parser.add_argument("--feature-lag-day", type=int, default=0,
                        help="特征表滞后天数: 0=同日JOIN(默认), 1=特征表 t-1 vs 样本表 t (样本表 day t LEFT JOIN 特征表 day t-1)")
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        raise SystemExit("session 目录不存在: %s" % session_dir)

    # 驾驶模式感知: 读 task-spec/_manifest.json 的 driving_mode
    # task-spec 阶段由 LLM 在 §3.1.5 检测关键词后通过 --driving-mode 透传写入 manifest
    # 本 skill 作为下游消费者, 读 manifest 决定是否跳过交互约定
    driving_mode = "manual"
    manifest_path = os.path.join(session_dir, "task-spec", "_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            driving_mode = manifest.get("driving_mode", "manual")
        except Exception as e:
            print("[fetch_sample] [WARN] 读 manifest 失败, 视为 manual: %s" % e)
    else:
        print("[fetch_sample] [WARN] task-spec/_manifest.json 不存在: %s, 视为 manual" % manifest_path)

    if driving_mode == "auto":
        # autopilot: 覆盖 args 中未显式传入的交互项为默认值, 跳过 4 个 ⚠️ 交互约定
        # 1. 拼接键: 用默认 user_no,pday (args.join_keys=None 时 _build_cfg 已有兜底, 此处显式打日志)
        if args.join_keys is None and args.feature_table:
            args.join_keys = "user_no,%s" % (args.dt_col or "pday")
            print("[autopilot] driving_mode=auto, 拼接键用默认: %s" % args.join_keys)
        # 2. 特征清单: 走 feature-knowledge 索引自动识别 (识别不到时全量派生)
        if not args.features and not args.feature_list_source:
            print("[autopilot] driving_mode=auto, 特征清单走 feature-knowledge 索引自动识别 (识别不到时全量派生)")
        # 3. HDFS 中间路径: 用 default_hdfs_base (spark 模式)
        if args.hdfs_base is None and args.mode == "spark":
            from spark_defaults import default_hdfs_base
            args.hdfs_base = default_hdfs_base("feature-matching")
            print("[autopilot] driving_mode=auto, HDFS 路径用默认: %s" % args.hdfs_base)
        # 4. t-1 滞后 JOIN: 默认 0, 用户显式传非 0 时尊重并打 warn
        if args.feature_lag_day != 0:
            print("[autopilot] [WARN] driving_mode=auto 但 --feature-lag-day=%d, 已尊重用户显式传入" % args.feature_lag_day)
        print("[autopilot] driving_mode=auto, 跳过 4 个交互约定, 用默认值推进")

    # local_file 模式必填校验
    if args.mode == "local_file":
        if not args.local_parquet_path:
            raise SystemExit("--mode local_file 时必须传 --local-parquet-path")
        if not os.path.isfile(args.local_parquet_path):
            raise SystemExit("本地样本文件不存在: %s" % args.local_parquet_path)
        if not args.local_parquet_path.lower().endswith((".parquet", ".csv")):
            raise SystemExit(
                "--local-parquet-path 仅支持 .parquet / .csv, 实际: %s" % args.local_parquet_path
            )
    else:  # spark 模式必填校验
        if not args.sample_table:
            raise SystemExit("--mode spark 时必须传 --sample-table")
        if not args.fetch_start or not args.fetch_end:
            raise SystemExit("--mode spark 时必须传 --fetch-start / --fetch-end")

    cfg = _build_cfg(args)
    model = cfg["model"]

    # 拼接模式 + features 留空 + 无 feature_list_source = 取特征表全部列, 跳过特征清单加载/校验
    # local_file 模式: features 留空也走"派生全量列"逻辑 (等同 join_all_columns)
    join_all_columns = (args.mode == "local_file" or bool(model.get("feature_table"))) \
        and not model.get("features") and not model.get("feature_list_source")

    # features 三档来源: CLI features > CLI feature_list_source > 按 feature-knowledge.md 索引自动识别
    # local_file 模式不在此加载, 留给 derive_feature_list.py 派生
    if not join_all_columns and not model.get("features"):
        model["features"] = load_feature_list(
            model.get("feature_list_source"),
            feature_table=model.get("feature_table"),
            business_domain=args.business_domain,
        )

    # 全列模式的过滤清单: 按 feature_table 优先 / business_domain 兜底, 从 feature-knowledge.md 自动识别
    filter_csv = None
    if join_all_columns and not args.no_filter_feas:
        filter_csv = resolve_feature_list_csv(model.get("feature_table"), args.business_domain)
        if filter_csv is not None:
            print(
                "[fetch_sample] [特征清单约定] %s: sample.parquet 将含全量列,"
                " feature-list.csv 按自动识别的清单过滤: %s\n"
                "  - 如需指定其他清单: Ctrl-C 后用 --feature-list-source <path> 或 --features <list>;\n"
                "  - 如需派生全量列(不过滤): 加 --no-filter-feas;\n"
                "  - 继续即视为接受该清单过滤。"
                % ("local_file 模式" if args.mode == "local_file" else "拼接全列模式", filter_csv)
            )
        else:
            print(
                "[fetch_sample] [WARN] 按 feature_table=%s / business_domain=%s 未在"
                " feature-knowledge.md 索引中识别到特征清单, feature-list.csv 将派生全量列"
                % (model.get("feature_table"), args.business_domain)
            )

    # 校验
    if join_all_columns:
        # local_file 模式 sample_table 占位为 "local_file", 不强制非空
        for key in ("name", "dt_col", "fetch_dt"):
            if not model.get(key):
                raise ValueError("配置 model.%s 缺失或为空" % key)
        if not (model.get("label_col") or model.get("label_expr")):
            raise ValueError("配置 model.label_col 与 model.label_expr 必须至少填一个")
        check_sensitive(model.get("where") or "")
        check_sensitive(model.get("sample_table") or "")
    else:
        validate_common(cfg)

    # 落 yaml 到 session 目录 (机制上强制, 不可能落 skill config/)
    yaml_path = _dump_yaml(cfg, session_dir, model["name"])
    print("[fetch_sample] 配置已落盘: %s" % yaml_path)

    # 输出 parquet 路径: 默认 <session_dir>/sample-features/feature-matching/sample.parquet
    out_path = args.out or os.path.join(
        session_dir, "sample-features", "feature-matching", "sample.parquet"
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # local_file 模式: 输入 parquet 直接复制 / 输入 csv 读后写 parquet → 派生 feature-list.csv → return
    if args.mode == "local_file":
        _local_sample_to_parquet(args.local_parquet_path, out_path)
        print("[fetch_sample] mode=local_file, 已转写为 sample.parquet:")
        print("  源: %s" % args.local_parquet_path)
        print("  目: %s" % out_path)

        # 派生 feature-list.csv (与 spark 全列模式逻辑一致)
        feature_list_csv = os.path.join(os.path.dirname(out_path), "feature-list.csv")
        _derive_feature_list_for_local(model, out_path, feature_list_csv, filter_csv, args.no_filter_feas)
        return

    # === spark 模式: 生成 spark-submit 包装脚本 ===
    from gen_fetch_command import build_command as build_fetch_cmd

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
    cmd, _, _ = build_fetch_cmd(cfg, out_path, hdfs_out_path)

    # wrapper 落 session 自包含目录(与 sample.parquet 同目录), 不再写 skill 安装目录
    gen_dir = os.path.dirname(out_path)
    os.makedirs(gen_dir, exist_ok=True)

    tag = "%s_%s" % (model["name"], model.get("version", "vX"))
    script_path = os.path.join(gen_dir, "fetch_%s.sh" % tag)

    # 拼 wrapper 脚本: spark-submit → hdfs dfs -get
    script_lines = [
        "#!/bin/bash", "set -e",
        "# 自动生成的 feature-matching 取数提交脚本, 请检查后提交",
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

    feature_list_csv = os.path.join(os.path.dirname(out_path), "feature-list.csv")

    # 全列模式: 列在集群运行时才确定, 取数回本地后从 sample.parquet schema 派生 feature-list.csv,
    # 过滤清单已在前面按 feature-knowledge.md 索引解析(filter_csv)。
    if join_all_columns:
        derive_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derive_feature_list.py")
        exclude_cols: List[str] = []
        for c in list(model.get("id_cols") or []) + [model.get("dt_col")]:
            if c and c not in exclude_cols:
                exclude_cols.append(c)
        label_name = model.get("label_col") or ("label" if model.get("label_expr") else None)
        if label_name and label_name not in exclude_cols:
            exclude_cols.append(label_name)
        for jk in (model.get("join_keys") or []):
            if jk and jk not in exclude_cols:
                exclude_cols.append(jk)

        use_filter = filter_csv is not None
        if args.no_filter_feas:
            print("[fetch_sample] --no-filter-feas: feature-list.csv 将派生全量列(不过滤)")

        derive_cmd = [
            "python %s" % shlex.quote(derive_py),
            "--input %s" % shlex.quote(out_path),
            "--output %s" % shlex.quote(feature_list_csv),
            "--exclude %s" % shlex.quote(",".join(exclude_cols)),
        ]
        if use_filter:
            derive_cmd.append("--filter %s" % shlex.quote(str(filter_csv)))
        script_lines += [
            "",
            "# 全列模式: 从落地的 sample.parquet 派生特征清单 feature-list.csv"
            + (" (按 feature-knowledge 清单过滤)" if use_filter else ""),
            " ".join(derive_cmd),
            "echo %s" % shlex.quote("已派生特征清单: %s" % feature_list_csv),
        ]

    script_content = "\n".join(script_lines) + "\n"

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR)

    # 指定 features 模式: 清单在生成阶段已知, 直接落 feature-list.csv;
    # 全列模式: 清单由 wrapper 脚本在取数回本地后派生(此处不写)。
    if not join_all_columns:
        write_feature_list_csv(model["features"], feature_list_csv)

    print("已生成取数提交脚本: %s" % script_path)
    if hdfs_out_path:
        print("HDFS 中间输出:    %s" % hdfs_out_path)
    print("样本将输出到:     %s" % out_path)
    if join_all_columns:
        print("特征列:           取特征表全部列; 取数回本地后自动派生 feature-list.csv")
        if use_filter:
            print("feature-list.csv: 按 %s 过滤(取数全量, 下游用 feature-knowledge 清单)" % filter_csv)
        else:
            print("feature-list.csv: 派生全量列(不过滤)")
        print("特征清单 csv:     %s (取数完成后生成)" % feature_list_csv)
    else:
        print("特征清单 csv:     %s (%d 个特征)" % (feature_list_csv, len(model["features"])))
    print("\n提交命令:\n  bash %s\n" % script_path)
    print("脚本内容:\n%s" % script_content)

    if args.submit:
        import subprocess
        print("[fetch_sample] 提交到集群: bash %s" % script_path)
        result = subprocess.run(["bash", script_path])
        if result.returncode != 0:
            print("[fetch_sample] 提交失败, returncode=%d" % result.returncode)
            sys.exit(result.returncode)
        print("[fetch_sample] 完成, sample.parquet 已落到 %s" % out_path)


def _local_sample_to_parquet(src: str, dst: str) -> None:
    """把本地样本文件转写为 dst.parquet。

    local_file 模式下 sample.parquet 由下游(feature-analysis / model-training)
    只读消费, 无需整文件复制。.parquet 优先硬链接(同盘零拷贝), 失败则软链接,
    再失败才 copyfile; .csv 仍走 pandas 读后写 parquet(格式转换必须复制)。

    Args:
        src: 源文件路径(.parquet 或 .csv)
        dst: 目标 parquet 路径

    Raises:
        ValueError: src 扩展名既非 .parquet 也非 .csv
    """
    src_lower = src.lower()
    if src_lower.endswith(".parquet"):
        # 优先硬链接(同盘 inode 共享, 零拷贝), 跨盘则软链接, 都不行才 copyfile
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
        return
    if src_lower.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(src)
        df.to_parquet(dst, index=False)
        return
    raise ValueError(
        "不支持的本地样本格式: %s (仅支持 .parquet / .csv)" % src
    )


def _derive_feature_list_for_local(
    model: dict, sample_path: str, feature_list_csv: str, filter_csv, no_filter_feas: bool
) -> None:
    """local_file 模式: 从 sample.parquet schema 派生 feature-list.csv。

    filter_csv 为 main 中按 feature-knowledge.md 索引解析出的过滤清单(可为 None);
    no_filter_feas 时派生全量列。
    """
    import subprocess

    derive_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "derive_feature_list.py")

    exclude_cols: List[str] = []
    for c in list(model.get("id_cols") or []) + [model.get("dt_col")]:
        if c and c not in exclude_cols:
            exclude_cols.append(c)
    label_name = model.get("label_col") or ("label" if model.get("label_expr") else None)
    if label_name and label_name not in exclude_cols:
        exclude_cols.append(label_name)
    for jk in (model.get("join_keys") or []):
        if jk and jk not in exclude_cols:
            exclude_cols.append(jk)

    use_filter = filter_csv is not None
    if no_filter_feas:
        print("[fetch_sample] --no-filter-feas: feature-list.csv 将派生全量列(不过滤)")

    cmd = [
        "python", derive_py,
        "--input", sample_path,
        "--output", feature_list_csv,
        "--exclude", ",".join(exclude_cols),
    ]
    if use_filter:
        cmd += ["--filter", str(filter_csv)]
    print("[fetch_sample] 派生 feature-list.csv: %s" % " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[fetch_sample] 派生 feature-list.csv 失败, returncode=%d" % result.returncode)
        sys.exit(result.returncode)
    print("[fetch_sample] 已派生特征清单: %s" % feature_list_csv)


if __name__ == "__main__":
    main()
