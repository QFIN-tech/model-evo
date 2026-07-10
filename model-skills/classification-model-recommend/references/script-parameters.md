# 脚本参数说明

> 本文件从 `classification-model-recommend/SKILL.md` 3.2–3.4 节抽出,包含 3 个脚本的完整 CLI 参数表:fetch_eval_sample.py(评估 entry)、split_sample.py(本地三档切分)、invoke_evaluation.py(评估委托)。模型推荐阶段(Claude 直接读台账语义筛选)不涉及脚本。SKILL.md 中保留摘要 + 指向本文件的指针。

## 1. fetch_eval_sample.py(评估 entry:取数+切分+委托评估一条龙)

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--session-dir` | ✅ | - | session 目录(`runs/<timestamp>-<model_name>`) |
| `--model-id` | ✅ | - | 模型 ID,用作文件名前缀 |
| `--sample-table` | ✅ | - | 样本表 库.表,提供 label |
| `--score-table` | ✅ | - | 模型表 库.表,提供 score;内部映射到 feature_table |
| `--join-keys` | 否 | `user_no,pday` | 拼接键(逗号分隔) |
| `--fetch-start` / `--fetch-end` | ✅ | - | 取数起止日期 YYYYMMDD(须覆盖 train+test+oot 并集) |
| `--train-range` / `--test-range` / `--oot-range` | ✅ | - | 三档 pday 闭区间 `起,止`(YYYYMMDD) |
| `--score-col` | 否 | `score` | 模型分列名 |
| `--label-col` | 否 | `label` | 标签列名 |
| `--id-cols` | 否 | `user_no` | ID 列(逗号分隔) |
| `--dt-col` | 否 | `pday` | 日期分区字段(两表须同名) |
| `--where` | 否 | - | 可选客群筛选条件 |
| `--version` | 否 | `v1` | 模型版本 |
| `--hdfs-base` | 否 | `/user/<whoami>/model-recommend` | HDFS 中间目录 |
| `--spark-bin` | 否 | 集群 3.3.2 | spark-submit 路径 |
| `--out` | 否 | `<session_dir>/model-recommend/<model_id>/predictions/sample.parquet` | sample.parquet 输出路径 |
| `--submit` | 否 | `false` | 生成脚本后同步执行 bash 提交集群 |
| `--no-eval` | 否 | `false` | 跳过 wrapper 末尾的 invoke_evaluation,仅产 predictions 三档 parquet |
| `--score-lag-day` | 否 | `0` | 模型分表(score_table)滞后天数: `0`=同日JOIN(默认), `1`=模型分表 t-1 vs 样本表 t。语义同 feature-matching 的 `--feature-lag-day`(recommend 语境下 score_table 即 feature_table), 内部映射到 yaml `model.feature_lag_day` |

## 2. split_sample.py(本地三档切分)

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--input` | ✅ | - | sample.parquet 路径 |
| `--ratios` | 否 | - | 比例模式:`train,test,oot` 如 `0.6,0.2,0.2`(与 `*-range` 互斥) |
| `--train-range` / `--test-range` / `--oot-range` | 否 | - | 显式模式:各档 pday 闭区间,如 `20260312,20260430` |
| `--time-col` | 否 | `pday` | 时间切分列 |
| `--label-col` | 否 | `label` | 标签列(仅用于统计正样本率) |
| `--output_dir` | 否 | 与 input 同目录 | 输出目录 |

## 3. invoke_evaluation.py(评估委托)

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--train-parquet` / `--test-parquet` / `--oot-parquet` | ✅ | - | 三档 parquet 路径 |
| `--score-col` | 否 | `score` | 模型分列名 |
| `--label-col` | 否 | `label` | 标签字段名 |
| `--out-dir` | ✅ | - | 报告输出目录 |
| `--model-id` | 否 | `model` | 模型 ID,用于输出文件名前缀 |
| `--model-type` | 否 | `xgboost` | 模型类型(透传 eval_single.py) |

调用 `eval_single.py` 目录模式:三档 parquet 转 CSV 落临时目录,一次性传入产 4 份三件套(train / test / oot / all 合并)。

---

> 关联:SKILL.md 3.2 / 3.3 / 3.4 节;脚本实现见 `scripts/`。
