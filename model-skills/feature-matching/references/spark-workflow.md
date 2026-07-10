# Spark 提交模式工作流

> 本文件从 `feature-matching/SKILL.md` 2.1 节抽出,包含 spark 模式两步法(生成 wrapper → 确认后 `--submit` 提交)、提交方式行为说明、超时提醒。SKILL.md 中保留 4 行摘要 + 指向本文件的指针;`2.1 节` 仍作为 stub heading 存在,内部 `2.1 节` 锚点继续指向该 stub。

配置 yaml **自动落到** `<session_dir>/sample-features/feature-matching/sample_config.<model_name>.yaml`(脚本内部自动构造 cfg 并落盘,不走 `--config`),机制上不可能误落 skill 自身 `config/` 目录。

**第一步:生成提交脚本(不提交)**

```bash
python <skill_dir>/scripts/fetch_sample.py \
    --session-dir <session_dir> \
    --model-name <model_name> \
    --sample-table <db>.<sample_table> \
    [--feature-table <db>.<feature_table>] \
    [--join-keys user_no,pday] \
    --fetch-start <YYYYMMDD> --fetch-end <YYYYMMDD> \
    --label-col label \
    [--features f0,f1,f2] \
    [--feature-list-source path/to/feature-list.csv] \
    [--business-domain <business_domain>] \
    [--id-cols user_no] [--dt-col pday] \
    [--where "seg='<your_segment>'"] [--label-expr "(CASE WHEN ... THEN 1 ELSE 0 END)"] \
    [--hdfs-base /user/<whoami>/feature-matching] \
    [--out <session_dir>/sample-features/feature-matching/sample.parquet]
```

特征清单:`--features` 留空 + 给 `--feature-table` = 取特征表全部列(运行时展开),`feature-list.csv` 按 `feature-knowledge.md` 索引自动识别的清单过滤(feature_table 优先命中「特征表」列,`--business-domain` 兜底命中「分场景」列);或用 `--feature-list-source <path>` 指定清单(`.txt` 按行 / `.csv` 取 `feature_name` 列);或直接 `--features f0,f1,...` 指定。

产出:
- `<session_dir>/sample-features/feature-matching/fetch_<name>_<version>.sh`(spark-submit wrapper)
- `<session_dir>/sample-features/feature-matching/feature-list.csv`(单列 `feature_name`,供下游引用)

**第二步:确认后自动提交(`--submit`)**

生成 wrapper 后,让用户确认是否提交到集群;用户确认后加 `--submit`:

```bash
python <skill_dir>/scripts/fetch_sample.py \
    --session-dir <session_dir> \
    --model-name <model_name> \
    --sample-table <db>.<sample_table> \
    [--feature-table <db>.<feature_table>] \
    --fetch-start <YYYYMMDD> --fetch-end <YYYYMMDD> \
    --label-col label \
    [--features ... | --feature-list-source ... | --no-filter-feas] \
    --submit
```

spark-submit 跑完后 wrapper 自动 `hdfs dfs -get` 回本地,**`sample.parquet` 落地即完成**。

**提交方式说明**(`--submit` 行为):
- **同步阻塞**: `subprocess.run(["bash", script_path])` 等到 spark job 完成才返回
- **流式回显**: 不开 `capture_output`,spark-submit 日志直接打到当前 stdout/stderr(长 job 更直观)
- **失败传播**: wrapper 退出码非 0 时,`fetch_sample.py` 以同样退出码 `sys.exit` 退出
- **不切分**: `--submit` 只产 `sample.parquet`,不产 `train/test/oot.parquet`;切分由下游 skill 内部完成
- **不传 `--submit`**: 只生成脚本不提交,由用户手动 `bash <script>`
- 确认点放在 SKILL.md 流程层(Claude 在调 `--submit` 前向用户确认),不在脚本里弹交互 — 保持脚本本身非交互、CI 友好

⚠️ **超时提醒**: spark job 通常 5–30 分钟,Claude Bash 工具默认 timeout 2 分钟,调用 `--submit` 时需 `run_in_background=True` 或加大 timeout,否则会中断。

---

> 关联:SKILL.md 2.1 节;local_file 模式见 SKILL.md 2.2 节;参数表见 `references/fetch-parameters.md`。
