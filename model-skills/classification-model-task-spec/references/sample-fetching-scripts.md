# 样本拉取与分析脚本

> 本文件从 `classification-model-task-spec/SKILL.md` 3.6.1 + 3.6.2 + 3.6.3 节抽出,包含 spark 模式 / local_file 模式样本拉取、样本分析脚本的 bash 调用、脚本行为差异、产出文件清单。SKILL.md 中保留 4 行摘要 + 指向本文件的指针。

## 3.6.1 拉取样本数据（spark 模式）

按需求文档中的样本表名和 pday 范围拉取全量样本（只拉样本列 `user_no`/`label`/`pday`，不拉特征列），保存至 `data-profile/{model_name}_sample_{YYYYMMDD}.parquet`。

```bash
python scripts/fetch_sample_task_spec.py \
    --session-dir <session_dir> \
    --model-name {model_name} \
    --sample-table {table_name} \
    --fetch-start {fetch_start} --fetch-end {fetch_end} \
    --train-range {train_start},{train_end} \
    --test-range  {eval_start},{eval_end} \
    --oot-range   {oot_start},{oot_end} \
    --label-col label \
    [--id-cols user_no] [--dt-col pday] \
    [--where "seg='活跃户'"] [--label-expr "(CASE WHEN ... THEN 1 ELSE 0 END)"] \
    [--hdfs-base /user/{user}/feature-matching] \
    [--out .../data-profile/{model_name}_sample_{YYYYMMDD}.parquet] \
    [--submit]
```

脚本复用 `model-evo/_modelevo-shared/scripts/` 下的公共取数代码（`config_io` / `gen_fetch_command` / `fetch_spark`），仅落 `sample.parquet` + 自动落 yaml 到 `<session_dir>/task-spec/sample_config.<model_name>.yaml`。可选 `--submit` 同步执行 spark-submit + hdfs dfs -get，不传则只生成脚本由用户手动跑。

## 3.6.2 mode=local_file 分支（本地 parquet/csv）

当上游选择"本地 parquet"模式透传 `mode=local_file` 标记时，或用户首轮即表达"已有本地 parquet"/"离线实验"时：

**首轮提问调整**：
- SAMPLE 维度问题为 **"本地 parquet/csv 路径是什么？"**
- 同时确认 `--label-col`/`--dt-col`/`--id-cols`（本地文件列名可能不是 `user_no`/`label`/`pday`，必须显式问清楚）
- 其他维度按标准流程询问；用户不提供时 WHO/WHAT/HOW GOOD/CONSTRAINTS 按默认值填写，**Train-Test-OOT 切分仍强制由用户提供**

**调用方式**：

```bash
python scripts/fetch_sample_task_spec.py \
    --session-dir <session_dir> \
    --model-name {model_name} \
    --mode local_file \
    --local-parquet-path /path/to/local_sample.parquet \
    --train-range {train_start},{train_end} \
    --test-range  {eval_start},{eval_end} \
    --oot-range   {oot_start},{oot_end} \
    --label-col {label_col} \
    --dt-col {dt_col} \
    --id-cols {id_cols} \
    [--where ""]
```

> `--local-parquet-path` 接受 `.parquet`（直接复制）或 `.csv`（读后写 parquet）。

**脚本行为差异**：
- 跳过 `gen_fetch_command.build_command`，不生成 spark-submit 包装脚本
- `_local_sample_to_parquet` 转写：`.parquet` 直接复制，`.csv` 读后写 parquet
- 落 yaml 含 `model.mode="local_file"`、`model.local_parquet_path=<path>`
- `--sample-table / --fetch-start / --fetch-end` 仅作记录用；`--submit` 在 local_file 模式下无效
- 校验仅做 `check_sensitive` + `validate_split_ranges`，跳过 spark 必填校验

**`_manifest.json` 额外字段**：`mode: "local_file"` / `local_parquet_path: <path>` / `id_cols`/`label_col`/`dt_col` 沿用用户指定

> local_file 模式下 task-spec 完成后，编排器**跳过 recommend**，直接进入 feature-matching（同样以 `--mode local_file` 调 `feature-matching/scripts/fetch_sample.py`）。

## 3.6.3 运行样本分析脚本

调起 `scripts/run_sample_analysis_task_spec.py`，由脚本统一完成：分时间段标签分布计算、稳定性判定、样本充足度判定、Train/Test/OOT 切分（**仅切样本列，不含特征**）、报告/清单/切分文件产出。

```bash
python scripts/run_sample_analysis_task_spec.py \
    --sample .../data-profile/{model_name}_sample_{YYYYMMDD}.parquet \
    --train-range {train_start},{train_end} \
    --test-range  {eval_start},{eval_end} \
    --oot-range   {oot_start},{oot_end} \
    --model-name {model_name} \
    --timestamp {timestamp} \
    --output-dir .../data-profile/ \
    --dt-col {dt_col} \
    --label-col {label_col} \
    --id-cols {id_cols} \
    [--sample-table {source_table}] \
    [--mode {spark|local_file}] [--local-parquet-path {path}]
```

> **列名约定**：`--dt-col` / `--label-col` / `--id-cols` 与 `fetch_sample_task_spec.py` 对齐。
> - spark 模式拉取的样本列名固定 `pday`/`label`/`user_no`，三参数可省（走默认值）。
> - **local_file 模式下必传**：本地 parquet 列名可能不是 `pday`/`label`/`user_no`（如 Home Credit 的 `dt`/`TARGET`/`SK_ID_CURR`），必须显式传入，否则校验失败。
> - `--id-cols` 支持逗号分隔多列，分析侧取首列作主 ID；老参数 `--time-col` / `--id-col` 仍可传入（已弃用 alias）。

> **模式与报告头**：`--mode spark`（缺省）时报告头展示 `--sample-table` 的表名；`--mode local_file` 时展示 `--local-parquet-path`（未传则回退到 `--sample`）的数据位置。

**产出文件**：

| 产出 | 路径 |
|------|------|
| 样本分析报告 (md/xlsx) | `data-profile/report.md` / `report.xlsx` |
| 关键信息清单 | `data-profile/_manifest.json` |
| 切分清单 | `data-profile/_split_manifest.json` |
| Train/Test/OOT 集 | `data-profile/train.parquet` / `test.parquet` / `oot.parquet` |

> 关联: `classification-model-task-spec/SKILL.md` 3.6 节需求确认后的样本分析
