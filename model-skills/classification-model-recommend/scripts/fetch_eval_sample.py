# -*- coding: utf-8 -*-
"""classification-model-recommend 评估取数+切分+委托评估 entry。

复用 model-evo/_modelevo-shared/scripts/ 公共代码(config_io / gen_fetch_command / fetch_spark),
采用 --session-dir 模式。评估逻辑委托 classification-model-evaluation/scripts/eval_single.py。

机制: 把 score_table 当 feature_table 喂给 _modelevo-shared/fetch_spark.py, 取 (keys + label + score)
一份 parquet, 再用 split_sample.py 按 pday 区间切三档, 最后 invoke_evaluation.py 产标准化
三件套 (JSON+MD+XLSX)。

CLI 模式: --session-dir + 关键参数直传, 脚本内部自动生成 yaml 落到
<session_dir>/model-recommend/{model_id}/eval_config.yaml, 并生成可直接 bash 提交的
4 步 wrapper (spark-submit 取数 → hdfs dfs -get → split_sample → invoke_evaluation)。

公共模块: model-evo/_modelevo-shared/scripts, 通过 _bootstrap.py 注入。
安全: 仅取所需列, 不输出用户级明细到日志。
"""

from __future__ import annotations

import argparse
import os
import shlex
import stat
import sys
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

    recommend 语义: feature_table = 模型表(score_table), features = [score_col]。
    """
    from spark_defaults import default_hdfs_base

    join_keys: Optional[List[str]] = None
    if args.join_keys:
        join_keys = [c.strip() for c in args.join_keys.split(",") if c.strip()]
    else:
        join_keys = ["user_no", args.dt_col or "pday"]

    hdfs_base = args.hdfs_base or default_hdfs_base("model-recommend")

    cfg = {
        "spark_submit": {
            # bin/options 留空, 走 gen_fetch_command._resolve_spark_cfg 兜底 _modelevo-shared/spark_defaults
            "hdfs_base": hdfs_base,
        },
        "model": {
            "name": args.model_id,
            "version": args.version,
            "sample_table": args.sample_table,
            "feature_table": args.score_table,  # recommend: feature_table = 模型表
            "join_keys": join_keys,
            "dt_col": args.dt_col,
            "id_cols": [c.strip() for c in (args.id_cols or "user_no").split(",") if c.strip()],
            "fetch_dt": [args.fetch_start, args.fetch_end],
            "where": args.where,
            "split": {
                "train_range": _parse_range(args.train_range),
                "test_range": _parse_range(args.test_range),
                "oot_range": _parse_range(args.oot_range),
            },
            "features": [args.score_col],  # recommend: features 就是这一个 score 列
            "label_col": args.label_col,
            "feature_lag_day": args.score_lag_day,  # CLI --score-lag-day 映射到 yaml feature_lag_day (_modelevo-shared 统一字段名)
        },
    }
    return cfg


def _dump_yaml(cfg: dict, session_dir: str, model_id: str) -> str:
    """把 cfg 落成 yaml 到 <session_dir>/model-recommend/{model_id}/eval_config.yaml。"""
    out_dir = os.path.join(session_dir, "model-recommend", model_id)
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "eval_config.yaml")
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# classification-model-recommend 评估配置 (由 fetch_eval_sample.py 自动落盘)\n")
        f.write("# model_id: %s\n" % model_id)
        f.write("# recommend 语境: feature_table = 模型表(score_table), features = [score_col]\n\n")
        yaml.safe_dump(clean, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return yaml_path


def _shquote(value):
    """POSIX-safe shell 引用; None 透传方便调用侧兜底。

    历史版本用白名单只覆盖 空格/'/\"/()=<> 等字符, 漏掉 ; & | $ ` \\n 等,
    含 ${IFS} 的 payload 可绕过。改用 shlex.quote 一次性覆盖全部 POSIX 元字符。
    """
    if value is None:
        return value
    return shlex.quote(str(value))


def main() -> None:
    """命令行入口: 读参数 → 构造 cfg → 校验 → 落 yaml → 生成 wrapper → (可选) 提交。"""
    import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts
    from config_io import validate_common, validate_split_ranges, check_sensitive
    from gen_fetch_command import build_command as build_fetch_cmd

    parser = argparse.ArgumentParser(description="recommend 评估取数+切分+委托评估 (--session-dir 模式)")
    parser.add_argument("--session-dir", required=True, help="session 目录 (runs/<timestamp>-<model_name>)")
    parser.add_argument("--model-id", required=True, help="模型ID, 用作文件名前缀")
    parser.add_argument("--sample-table", required=True, help="样本表 库.表, 提供 label")
    parser.add_argument("--score-table", required=True,
                        help="模型表 库.表, 提供 score; 内部映射到 feature_table")
    parser.add_argument("--join-keys", default=None,
                        help="拼接键(逗号分隔), 默认 user_no,pday")
    parser.add_argument("--fetch-start", required=True, help="取数起始日期 YYYYMMDD (须覆盖 train+test+oot 并集)")
    parser.add_argument("--fetch-end", required=True, help="取数结束日期 YYYYMMDD (须覆盖 train+test+oot 并集)")
    parser.add_argument("--train-range", required=True, help="Train pday 闭区间 起,止 (YYYYMMDD)")
    parser.add_argument("--test-range", required=True, help="Test  pday 闭区间 起,止 (YYYYMMDD)")
    parser.add_argument("--oot-range", required=True, help="OOT   pday 闭区间 起,止 (YYYYMMDD)")
    parser.add_argument("--score-col", default="score", help="模型分列名 (默认 score)")
    parser.add_argument("--label-col", default="label", help="标签列名 (默认 label)")
    parser.add_argument("--id-cols", default="user_no", help="ID 列(逗号分隔), 默认 user_no")
    parser.add_argument("--dt-col", default="pday", help="日期分区字段(两表须同名), 默认 pday")
    parser.add_argument("--where", default=None, help="可选客群筛选条件")
    parser.add_argument("--version", default="v1", help="模型版本, 默认 v1")
    parser.add_argument("--hdfs-base", default=None, help="HDFS 中间目录, 默认 /user/<whoami>/model-recommend")
    parser.add_argument("--spark-bin", default=None, help="spark-submit 路径, 默认集群 3.3.2")
    parser.add_argument("--out", default=None,
                        help="sample.parquet 输出路径, 默认 <session_dir>/model-recommend/<model_id>/predictions/sample.parquet")
    parser.add_argument("--submit", action="store_true", help="生成脚本后同步执行 bash <script> 提交集群")
    parser.add_argument("--no-eval", action="store_true",
                        help="跳过 wrapper 末尾的 invoke_evaluation, 仅产 predictions 三档 parquet")
    parser.add_argument("--score-lag-day", type=int, default=0,
                        help="模型分表(score_table)滞后天数: 0=同日JOIN(默认), 1=模型分表 t-1 vs 样本表 t (recommend 内部映射到 feature_lag_day)")
    args = parser.parse_args()

    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        raise SystemExit("session 目录不存在: %s" % session_dir)

    cfg = _build_cfg(args)
    model = cfg["model"]

    # 校验: 走 _modelevo-shared config_io
    validate_common(cfg)
    validate_split_ranges(model)
    check_sensitive(model.get("where") or "")
    check_sensitive(model.get("sample_table") or "")

    # 落 yaml 到 session 目录
    yaml_path = _dump_yaml(cfg, session_dir, args.model_id)
    print("[fetch_eval_sample] 配置已落盘: %s" % yaml_path)

    # 输出路径
    rec_dir = os.path.join(session_dir, "model-recommend", args.model_id)
    predictions_dir = os.path.join(rec_dir, "predictions")
    out_path = args.out or os.path.join(predictions_dir, "sample.parquet")
    out_path = os.path.abspath(out_path)

    # HDFS 中间路径(gen_fetch_command._resolve_spark_cfg 会从 spark_submit.hdfs_base 解析;
    # 上文 _build_cfg 已把 hdfs_base 填了 default_hdfs_base("model-recommend"))
    spark_cfg = cfg.get("spark_submit", {})
    hdfs_base = (spark_cfg.get("hdfs_base") or "").rstrip("/")
    if hdfs_base:
        version_tag = model.get("version", "vX")
        model_dir = "%s_%s" % (model["name"], version_tag)
        hdfs_out_path = "%s/%s/sample.parquet" % (hdfs_base, model_dir)
    else:
        hdfs_out_path = None

    # 生成 spark-submit 取数段(写 HDFS)
    fetch_cmd, _, _ = build_fetch_cmd(cfg, out_path, hdfs_out_path)

    # 拼 wrapper 脚本: STEP1 取数 → STEP2 hdfs get → STEP3 split → STEP4 eval
    here = os.path.dirname(os.path.abspath(__file__))
    split_script = os.path.join(here, "split_sample.py")
    invoke_script = os.path.join(here, "invoke_evaluation.py")

    def _fmt_range(r):
        return "%s,%s" % (r[0], r[1])

    split_cmd = " ".join([
        "python", _shquote(split_script),
        "--input", _shquote(out_path),
        "--train-range", _fmt_range(model["split"]["train_range"]),
        "--test-range", _fmt_range(model["split"]["test_range"]),
        "--oot-range", _fmt_range(model["split"]["oot_range"]),
        "--time-col", str(model.get("dt_col", "pday")),
        "--label-col", str(model.get("label_col", "label")),
        "--output_dir", _shquote(predictions_dir),
    ])

    eval_cmd = " ".join([
        "python", _shquote(invoke_script),
        "--train-parquet", _shquote(os.path.join(predictions_dir, "train.parquet")),
        "--test-parquet",  _shquote(os.path.join(predictions_dir, "test.parquet")),
        "--oot-parquet",   _shquote(os.path.join(predictions_dir, "oot.parquet")),
        "--score-col", str(args.score_col),
        "--label-col", str(args.label_col),
        "--out-dir", _shquote(os.path.join(rec_dir, "evaluation")),
        "--model-id", args.model_id,
    ])

    script_lines = [
        "#!/bin/bash", "set -e",
        "# 自动生成的 recommend 评估提交脚本, 请检查后提交",
        "# recommend 语境: feature_table = 模型表(score_table), features = [score_col]",
        "",
        'echo "[STEP1] spark-submit 取数(样本表⋈模型表 JOIN)写 HDFS: %s"' % (hdfs_out_path or out_path),
        fetch_cmd,
        "",
        'echo "[STEP2] hdfs dfs -get 拉本地: %s"' % out_path,
        "mkdir -p %s" % _shquote(predictions_dir),
    ]
    if hdfs_out_path:
        script_lines += [
            "hdfs dfs -get -f %s %s" % (_shquote(hdfs_out_path), _shquote(out_path)),
        ]
    script_lines += [
        'echo "[STEP3] 本地 split_sample.py 切三档: %s"' % predictions_dir,
        split_cmd,
    ]
    if not args.no_eval:
        script_lines += [
            'echo "[STEP4] invoke_evaluation.py 委托评估: %s"' % os.path.join(rec_dir, "evaluation"),
            eval_cmd,
            'echo "[DONE] 报告已生成于: %s"' % os.path.join(rec_dir, "evaluation"),
        ]
    else:
        script_lines += [
            'echo "[DONE] --no-eval: 仅产 predictions, 跳过评估"',
        ]

    script_content = "\n".join(script_lines) + "\n"

    script_path = os.path.join(rec_dir, "fetch_eval_%s.sh" % args.model_id)
    os.makedirs(rec_dir, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR)

    print("已生成评估提交脚本: %s" % script_path)
    if hdfs_out_path:
        print("HDFS 中间输出:    %s" % hdfs_out_path)
    print("样本将输出到:     %s" % out_path)
    print("三档切分输出到:   %s/{train,test,oot}.parquet" % predictions_dir)
    if not args.no_eval:
        print("评估报告输出到:   %s/evaluation/" % rec_dir)
    print("\n提交命令:\n  bash %s\n" % script_path)
    print("脚本内容:\n%s" % script_content)

    if args.submit:
        import subprocess
        print("[fetch_eval_sample] 提交到集群: bash %s" % script_path)
        result = subprocess.run(["bash", script_path])
        if result.returncode != 0:
            print("[fetch_eval_sample] 提交失败, returncode=%d" % result.returncode)
            sys.exit(result.returncode)
        print("[fetch_eval_sample] 完成, 报告见: %s/evaluation/" % rec_dir)


if __name__ == "__main__":
    main()
