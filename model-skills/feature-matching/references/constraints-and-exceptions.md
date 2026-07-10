# feature-matching 执行约束与异常处理

> 本文件从 `feature-matching/SKILL.md` 第 6/7 节抽出,包含覆盖范围、架构约束、取数必填校验、数据安全红线、变更前置流程、异常处理全表。SKILL.md 第 6 节只保留红线级约束(数据安全 / 必填校验 / 变更前置),第 7 节保留单行指针指向本文件。⚠️ 交互约定(4 条)不在本文件,详见 `interaction-conventions.md`。

## 1. 覆盖范围

本 skill **只做**:

- ① 从 Spark 宽表拉样本+特征+标签生成 spark-submit wrapper
- ② 用户确认后自动 `--submit` 提交集群并 `hdfs dfs -get` 回本地
- ③ 本地 parquet/csv 模式(`--mode local_file`)转写为 sample.parquet + 派生 `feature-list.csv`(csv 输入读后写 parquet,输出统一 parquet)
- ④ 派生 `feature-list.csv`(默认按 feature-knowledge 索引识别的清单过滤,`--no-filter-feas` 退回全量)

**不做**:特征质量分析(IV/PSI/相关性,归 `feature-analysis`)、训练新模型(归 `classification-model-training`)、特征筛选/调参(归 `classification-model-tuning`)、历史模型推荐(归 `classification-model-recommend`)、train/test/oot 数据切分(下游内部切分,本 skill 只产 `sample.parquet`)、会话编排/任务路由(归 `model-task-routing` / `classification-model-orchestration`)。

## 2. 何时用

- 用户要从宽表拉一份样本到本地(后续做特征分析/建模)
- 下游 `feature-analysis` / `classification-model-training` 需要 `sample.parquet` 作为输入

## 3. 架构约束

不在 driver 进程内直连 Spark:本 skill 生成 spark-submit wrapper 脚本,用户确认后由 `fetch_sample.py` 通过 subprocess 同步调用 `bash <script>` 提交集群,输出流式回显。

## 4. 取数必填校验(`config_io.validate_common` / `fetch_sample`)

- `model.name` / `sample_table` / `dt_col` / `fetch_dt` 非空
- `label_col` 与 `label_expr` 至少一个
- `fetch_dt` 为 `[起,止]` 两元素、8 位 `YYYYMMDD`
- `hdfs_base` 必填(spark 模式)

## 5. 数据安全红线(全模式强制,`check_sensitive`)

- yaml / where / label_expr 严禁硬编码用户 ID/手机号/身份证号
- 自动扫描 `where` 与 `sample_table`,命中 18 位身份证或 11 位手机号正则即抛错
- 仅取所需列,不输出用户级明细到日志
- 脚本仅做分组聚合统计,**不输出任何用户级明细**

## 6. 变更前置流程(强制遵循 CLAUDE.md)

修改取数代码前,先输出「变更计划」(①修改内容 ②预期影响 ③回滚方案)并等确认。

## 7. 异常处理

| 异常 | 处理方式 |
|---|---|
| `model.name`/`sample_table`/`dt_col`/`fetch_dt` 为空 | `validate_common` 报错终止 |
| `label_col` 与 `label_expr` 都为空 | `validate_common` 报错终止 |
| `hdfs_base` 留空(spark 模式) | 报错终止,不允许 spark 直接写本地(会触发权限错误) |
| `--local-parquet-path` 既非 `.parquet` 也非 `.csv`(local_file 模式) | 报错终止 |
| `where`/`sample_table` 命中身份证/手机号正则 | `check_sensitive` 抛错终止 |
| spark wrapper 退出码非 0 | `fetch_sample.py` 以同样退出码退出,失败传播 |
| 识别到的清单里的特征不在 sample 中 | 打 warn 并丢弃 |
| feature-knowledge 索引未命中(feature_table/business_domain 都匹配不到) | 打 warn 自动退回全量派生;显式来源(features/feature_list_source)都为空且非全列模式时报错终止 |
| spark job 超过 Bash 工具 2 分钟默认 timeout | 调用时用 `run_in_background=True` 或加大 timeout |

---

> 关联:SKILL.md 第 6/7 节;⚠️ 交互约定(4 条)详见 `references/interaction-conventions.md`。
