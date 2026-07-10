# 取数脚本参数表

> 本文件从 `feature-matching/SKILL.md` 第 3 节抽出,包含 `fetch_sample.py` 与 `derive_feature_list.py` 的完整 CLI 参数表。SKILL.md 中保留摘要 + 指向本文件的指针;`第 3 节` 仍作为 stub heading 存在。

## 1. fetch_sample.py(取数 entry)

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--session-dir` | ✅ | - | session 目录(`runs/<timestamp>-<model_name>`) |
| `--model-name` | ✅ | - | 模型简称 |
| `--mode` | 否 | `spark` | `spark` / `local_file`,决定走 2.1 还是 2.2 节分支 |
| `--local-parquet-path` | local_file 必填 | - | 本地样本路径(local_file 模式),接受 `.parquet` / `.csv`;csv 输入会被读后转写为 sample.parquet |
| `--sample-table` | spark 必填 | - | 样本表 库.表(local_file 模式仅记录用) |
| `--feature-table` | 否 | - | 特征表 库.表;传时启用拼接模式,留空=单表模式 |
| `--join-keys` | 否 | `user_no,pday` | 拼接键(逗号分隔),仅拼接模式生效;⚠️ **需交互确认,别闷头用默认**(见第 6 节) |
| `--fetch-start` / `--fetch-end` | spark 必填 | - | 取数起止日期 `YYYYMMDD` |
| `--label-col` | 否 | `label` | 标签列名 |
| `--label-expr` | 否 | - | SQL 标签表达式,非空时替代 `--label-col` |
| `--features` | 否 | - | 特征列(逗号分隔);留空=取特征表全部列(拼接模式)或仅样本三列(单表模式)/ local_file 模式派生全量列 |
| `--feature-list-source` | 否 | - | 特征清单文件(`.txt` 按行 / `.csv` 取 `feature_name` 列);留空按 `feature-knowledge.md` 索引自动识别 |
| `--business-domain` | 否 | - | 业务域,自动识别清单时的兜底匹配键;由上层根据建模描述与业务知识库推断后传入(具体取值由 model-knowledge 知识库登记) |
| `--id-cols` | 否 | `user_no` | ID 列(逗号分隔) |
| `--dt-col` | 否 | `pday` | 日期分区字段 |
| `--where` | 否 | - | 可选客群筛选条件 |
| `--version` | 否 | `v1` | 模型版本 |
| `--hdfs-base` | 否 | `/user/<whoami>/feature-matching` | HDFS 中间目录(代码兜底默认);⚠️ **交互层仍须回显确认**(见第 6 节),不能直接吃默认值 |
| `--spark-bin` | 否 | 集群 3.3.2 | spark-submit 路径 |
| `--out` | 否 | `<session_dir>/sample-features/feature-matching/sample.parquet` | 输出 parquet 路径 |
| `--submit` | 否 | `false` | 生成脚本后同步执行 `bash <script>` 提交集群(spark 模式有效,local_file 模式无效) |
| `--no-filter-feas` | 否 | `false` | 跳过特征清单过滤,派生全量 `feature-list.csv` |
| `--feature-lag-day` | 否 | `0` | 特征表滞后天数: `0`=同日JOIN(默认), `1`=特征表 t-1 vs 样本表 t。lag=1 时 ON 子句去掉 dt_col 等值比较改用日期算术对齐(`a.pday = date_format(date_add(to_date(b.pday,'yyyyMMdd'),1),'yyyyMMdd')`),特征表时间窗自动平移为 `[fetch_start-1, fetch_end-1]` |

## 2. derive_feature_list.py(派生 feature-list.csv)

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--input` | ✅ | - | 本地 `sample.parquet` / `sample.csv` 路径(按扩展名判断,parquet 走 pyarrow schema,csv 走 pandas 读首行) |
| `--output` | ✅ | - | 输出 `feature-list.csv` 路径 |
| `--exclude` | 否 | 空 | 逗号分隔的非特征列(id/dt/label/base) |
| `--filter` | 否 | - | 过滤清单路径(`.csv` 取 `feature_name` 列 / `.txt` 按行);传时只输出该清单与 sample schema 的交集(按清单顺序),不传则全量派生;相对路径按 repo 根解析 |

---

> 关联:SKILL.md 第 3 节;spark 模式工作流见 `references/spark-workflow.md`;脚本实现见 `scripts/`。
