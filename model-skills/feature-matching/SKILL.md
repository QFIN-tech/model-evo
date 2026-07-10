---
name: feature-matching
description: 从 Spark 现成宽表拉取样本+特征+标签,生成 spark-submit 提交脚本,用户确认后由本 skill 自动提交集群,落本地 sample.parquet;或在 local_file 模式下从本地 parquet/csv 直接转写+派生 feature-list.csv。当用户要"取数""拉样本""拉宽表""准备建模样本"时使用。
---

# feature-matching

> 取数 skill:配置一份样本+特征清单 → 生成 spark-submit 提交脚本 → 用户确认后由本 skill 自动提交集群 → 落本地 `sample.parquet`;或在 local_file 模式下从本地 parquet/csv 直接转写为 `sample.parquet` + 派生 `feature-list.csv`。所有产物落到 `<session_dir>/sample-features/feature-matching/`。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| `session_dir` | ✅ | CLAUDE.md 会话约定(`runs/<timestamp>-<model_name>`) | 只在 CLI 上拼接,不写进任何 yaml/csv,保持配置可移植 |
| `model_name` | ✅ | 用户输入 / task-spec 文档 | 模型简称,由 task-spec 阶段确认 |
| 样本表 `sample_table` | spark 模式 ✅ | task-spec 文档 > 其他上游 md > 交互询问 | 库.表;拼接模式下为提供 label + 主键的主表 |
| 特征表 `feature_table` | 否 | 同上 | 传时启用样本表⋈特征表拼接模式 |
| 取数窗口 `fetch_dt` | spark 模式 ✅ | 同上 | `[起, 止]` 两元素列表,8 位 `YYYYMMDD` |
| 标签 `label_col` / `label_expr` | ✅(至少一个) | 同上 | `label_expr` 非空时替代 `label_col`(`config_io.validate_common` 校验) |
| HDFS 中间路径 `hdfs_base` | spark 模式 ✅ | task-spec 文档 > 交互询问,**必须回显确认**(见 `references/interaction-conventions.md`) | 留空会把本地路径误当 HDFS 写 → 权限错误 |
| 特征清单 | 否 | 三档来源优先级:yaml `features` > `feature_list_source` 文件 > 按 `model-knowledge/assets/feature-knowledge/feature-knowledge.md` 索引自动识别(feature_table 优先/business_domain 兜底) | 单表模式必须指定 `features`;拼接模式可留空=取特征表全部列 |
| 本地 parquet / csv | local_file 模式 ✅ | 用户提供 / 上游 orchestration 透传 | 预组装好的本地 parquet 或 csv(含 `id_cols + label_col + dt_col + features`);csv 输入会被读后转写为 sample.parquet,输出统一 parquet |

**入参来源优先级**:`classification-model-task-spec` 规格文档 > 其他上游 markdown(`classification-model-recommend` 的 `reports/{model_id}_*.md`、`classification-model-training` 的 run `report.md` 等)> 交互询问用户。task-spec 规格文档字段契约:

| 文档字段 | 映射到 | 示例 |
|---|---|---|
| 样本表 | `model.sample_table` | `<db>.<sample_table>` |
| 特征表 | `model.feature_table` | `<db>.<feature_table>` |
| join-key | `model.join_keys` | `user_no, pday` |
| 特征列 | `model.features` / 留空取全列 | 清单 或 "全部" |
| label 列 | `model.label_col` | `label` |
| 取数窗口 | `model.fetch_dt` | `20260312 ~ 20260524` |
| HDFS 中间路径 | `spark_submit.hdfs_base` | `/user/{用户}/feature-matching` |

本 skill 支持两种取数模式,由 `--feature-table` 参数决定:

1. **单表模式**(默认): 仅传 `--sample-table` 一张宽表,直接取 `features + label`。
2. **样本表⋈特征表模式**: 额外传 `--feature-table` 时启用。样本表(`--sample-table`,提供 label + 主键)为主,LEFT JOIN 特征表(`--feature-table`,提供特征)。
   - `--features` 留空 + 无 `--feature-list-source` → 取**特征表全部列**(`b.* EXCEPT (join_keys)`)
   - 指定 `--features f0,f1,...` → 只取特征表中这些列
   - 拼接键 `--join-keys`(默认 `user_no,pday`)

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录(即本文件所在目录),执行时替换为实际绝对路径,不要依赖当前工作目录。

### 2.0 驾驶模式感知(进入 skill 首要动作)

进入本 skill 后,**首要动作是读** `<session_dir>/task-spec/_manifest.json` 的 `driving_mode` 字段,按其取值调整交互约定与默认值。字段定义、关键词检测、`auto`/`manual` 各阶段行为、本 skill 的具体跳过项与默认值(jion-key / 特征清单 / HDFS 路径 / feature_lag_day)详见 [../classification-model-orchestration/references/driving-mode.md](../classification-model-orchestration/references/driving-mode.md) 第 6.3 节。本 skill 是 `driving_mode` 的**消费者**,不改字段。

本 skill 有两条互斥的执行分支:**2.1 spark 提交模式**(从 Spark 宽表取数)与 **2.2 mode=local_file 模式**(本地 parquet/csv 直接转写+派生,跳过 spark-submit)。走哪条分支由 `--mode`(默认 `spark`)决定。

### 2.1 spark 提交模式

两步法:① `fetch_sample.py` 不带 `--submit` 生成 spark-submit wrapper;② 用户确认后加 `--submit` 提交集群。完整两步命令、`--submit` 行为、特征清单三档来源、⚠️ Bash 工具超时提醒详见 `references/spark-workflow.md`。

### 2.2 mode=local_file 模式(本地 parquet/csv,跳过 spark-submit)

当用户已有预组装好的本地 parquet 或 csv(含 `id_cols + label_col + dt_col + features`),或上游 `classification-model-orchestration` 在第零步 B 选择"本地样本"透传 `mode=local_file` 标记时,走此分支,**只做"转写为 sample.parquet + 派生 feature-list.csv"两件事,不走 spark-submit**:

```bash
python <skill_dir>/scripts/fetch_sample.py \
    --session-dir <session_dir> \
    --model-name <model_name> \
    --mode local_file \
    --local-parquet-path <local_sample.parquet | local_sample.csv> \
    [--features f0,f1,f2 | --no-filter-feas]
```

**脚本行为**:
- 跳过 `gen_fetch_command.build_command`,不生成 spark-submit wrapper 脚本
- 输入 `.parquet` → `shutil.copyfile` 直接复制为 `sample.parquet`;输入 `.csv` → `pandas.read_csv` 读后 `to_parquet` 转写为 `sample.parquet`
- 调 `derive_feature_list.py` 派生 `feature-list.csv`:`exclude_cols = id_cols + dt_col + label_col + join_keys`;默认按 `feature-knowledge.md` 索引自动识别的清单过滤(`--filter <识别到的csv>`);`--no-filter-feas` 或未识别到时派生全量列
- 落 yaml 到 `<session_dir>/sample-features/feature-matching/sample_config.<model_name>.yaml`,含 `model.mode="local_file"`、`model.sample_table="local_file"`(占位)、`model.local_parquet_path=<path>`、`spark_submit.hdfs_base=""`

**local_file 模式下参数语义**(与 spark 模式的差异):
- `--sample-table / --fetch-start / --fetch-end / --feature-table / --join-keys` 均可选(仅记录用,不强制)
- `--submit` 在 local_file 模式下无效(转写+派生已在脚本内同步完成,无需提交集群)
- `--features` 留空时,`feature-list.csv` 派生用 sample 全列减去 exclude_cols
- `--local-parquet-path` 接受 `.parquet` / `.csv`,其他扩展名报错终止

**与 spark 模式的产物布局一致**: local_file 模式产出的 `sample.parquet` + `feature-list.csv` 落在 `<session_dir>/sample-features/feature-matching/`,下游 `feature-analysis` / `classification-model-training` 按 `sample-features/feature-matching/` 路径消费即可。

## 3. 参数说明

`fetch_sample.py`(取数 entry,21 个参数,含 `--mode` 切 spark/local_file、`--feature-lag-day` t-1 滞后 JOIN)与 `derive_feature_list.py`(派生 `feature-list.csv`,4 个参数)的完整 CLI 表详见 `references/fetch-parameters.md`。

## 4. 输出产物

产物统一落 `<session_dir>/sample-features/feature-matching/`(session 自包含,多 session 不互相覆盖):

```text
<session_dir>/sample-features/feature-matching/
├── sample_config.<model_name>.yaml   # 自动落盘的配置
├── sample.parquet                    # 主样本
├── feature-list.csv                  # 特征清单(1 列 feature_name)
└── fetch_<name>_<version>.sh         # spark-submit wrapper(仅 spark 模式)
```

### 4.1 产物内容

| 产物 | 必选 | 说明 |
|---|:---:|---|
| `sample_config.<model_name>.yaml` | ✅ | 由脚本自动落盘,含 mode / sample_table / feature_table / join_keys / fetch_dt / hdfs_base 等 |
| `sample.parquet` | ✅ | 主样本,列 = `id_cols + features + dt_col + label`(或 `label_col`) |
| `feature-list.csv` | ✅ | 1 列 `feature_name`,**始终产出**:指定模式在生成阶段写;全量模式取数回本地后派生,默认按 `feature-knowledge.md` 索引自动识别的清单过滤,`--no-filter-feas` 或未识别到时退回全量派生 |
| `fetch_<name>_<version>.sh` | 条件生成 | spark-submit wrapper,仅 spark 模式生成;`--submit` 模式下由 `fetch_sample.py` 自动执行 |
| HDFS 中间产物 | spark 模式 ✅ | `{hdfs_base}/{name}_{version}/sample.parquet`,spark 先写此处再拉回本地(临时) |

**明确不做**:
- 不写任何 `selected_features.txt`
- 不产 `train/test/oot.parquet` / 切分清单(切分由下游 skill 内部完成)
- 不改 catalog/reports(数据资产,不属于 session)
- 同一 session 重复跑用时间戳 / `run_<HHMM>/` 后缀,不互相覆盖

### 4.2 工具一览

| 工具 | 作用 |
|------|------|
| `scripts/fetch_sample.py` | 取数 entry: `--session-dir` 模式,生成 spark-submit wrapper + 可选 `--submit` 提交集群;local_file 模式下转写(parquet 复制 / csv 读后写)+派生 |
| `scripts/derive_feature_list.py` | 从 sample.parquet / sample.csv 派生 feature-list.csv,可选按清单文件(csv/txt)过滤 |
| `scripts/gen_feature_list.py` | 特征清单加载/落盘工具(内部依赖,由 `fetch_sample.py` 调用):三档来源优先级同 1. 节 |
| `scripts/_bootstrap.py` | 注入 `model-skills/_modelevo-shared/scripts` 到 sys.path |

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `classification-model-task-spec` | 规格文档提供样本表/特征表/join-key/取数窗口/HDFS 路径等入参(优先级最高) |
| 上游 | `classification-model-orchestration` | 编排调用本 skill(Step 4B);本地 parquet 场景透传 `mode=local_file` |
| 下游 | `feature-analysis` | 读 `sample.parquet` / `feature-list.csv`,内部切分做 IV/PSI/相关性分析 |
| 下游 | `classification-model-training` | 读 `sample.parquet`,内部切分,在配置指定入模 `features` 训新模型 |
| 依赖 | `model-evo/shared`(父目录) | 复用 `config_io`(配置读写+安全校验)、`fetch_spark`(PySpark 集群取数)、`gen_fetch_command`(wrapper 生成);公共 Spark 基础设施,各 skill 通过 `_bootstrap.py` 注入 sys.path 共用 |

## 6. 执行约束

数据安全红线、取数必填校验、覆盖范围、架构约束详见 `references/constraints-and-exceptions.md`;⚠️ 交互约定(4 条)详见 `references/interaction-conventions.md`。

## 7. 异常处理

异常分类与处理方式详见 `references/constraints-and-exceptions.md` 第 7 节。

## 8. 测试

```bash
python -m pytest feature-matching/tests/ -q
```

---

数据来源:Spark 宽表由 `fetch_spark.py` 集群取数;本地 parquet 由用户提供或上游 orchestration 透传;session 产物落 `${session_dir}/sample-features/feature-matching/`。
最后更新:2026-07-07
