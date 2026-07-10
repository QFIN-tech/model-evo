# feature-matching 交互约定

> 本文件从 `feature-matching/SKILL.md` 第 6 节抽出,包含 4 个 ⚠️ 强制交互约定 + 取数定位说明 + t-1 滞后 JOIN + 全自动驾驶模式跳过规则。SKILL.md 第 6 节保留单行引用指向本文件;2.0 节驾驶模式感知作为进入 skill 的首要动作仍 inline 保留,内部引用「四个 ⚠️ 交互约定」时指向本文件。

## 0. 定位:取数只负责"拉全量",特征筛选从下游开始

**feature-matching 是「取数」步骤,不是「特征筛选」步骤**。默认把特征表的**全部列**拉到本地(拼接模式 `features` 留空即取全部列),让下游基于完整数据再做特征选择,避免上游过早裁列丢信息。

`feature-list.csv` 默认按 `feature-knowledge.md` 索引自动识别的清单过滤:
- feature_table 优先命中「特征表」列,business_domain 兜底命中「分场景」列
- 只输出「清单指定 且 在 sample.parquet schema 中存在」的交集特征(按清单顺序)
- 清单里不在 sample 中的特征打 warn 并丢弃
- `--no-filter-feas` 可退回全量派生;索引未命中时打 warn 自动退回全量派生

特征指定/筛选应从下游开始:
- `feature-analysis` 对 `feature-list.csv` 做 IV/PSI/相关性
- `classification-model-training` 在配置里指定 `model.features` 入模清单
- 下游可直接从 `feature-list.csv` 继续查找/筛选,不必自己枚举特征表列名

若确需在取数阶段就限定特征(表特别宽、只关心子集),才在本 skill 显式填 `model.features` 或 `feature_list_source`。一句话:**取数拉全量 → 按 feature-knowledge 识别的清单过滤落 feature-list.csv → 下游从它定特征**,不要指望 feature-matching 帮你选特征。

## 1. ⚠️ 特征清单交互约定(强制,生成脚本前必须先问用户)

**生成 spark-submit 脚本前,必须先问用户是否指定 feature-list,不能闷头走默认**。

三档分支:

- **A. 按 `feature-knowledge.md` 索引自动识别(推荐默认)**
  - 先按建模描述与业务知识库推断业务域,用 feature_table 优先/business_domain 兜底匹配索引
  - 取数拉全量,`feature-list.csv` 按识别到的清单过滤
  - yaml `features` 与 `feature_list_source` 都留空
  - 询问话术:「按 feature-knowledge 索引识别到清单 `<csv 路径>`(取数拉全量, 下游按该清单过滤),用它吗?」
  - **须回显识别到的路径**;识别不到时告知将全量派生

- **B. 指定其他特征清单文件(.txt/.csv)**
  - 让用户填路径,写入 `model.feature_list_source`
  - 取数阶段即限定特征列,`feature-list.csv` 生成阶段即写好

- **C. 手动圈定特征列表**
  - 让用户列特征名,写入 `model.features`,行为同 B

**强制点**:
- 不问就生成脚本 = 违反约定
- 即使用户上次选过,新 session 也要重新确认
- task-spec 文档已显式指定特征清单时,跳过询问,直接用文档值(优先级:task-spec > 交互询问)

## 2. ⚠️ HDFS 中间路径交互约定(强制,生成脚本前必须先确认)

**生成 spark-submit 脚本前,必须先向用户确认 `spark_submit.hdfs_base`,不能闷头用 yaml 默认值或 task-spec 文档值**;即便文档里已带路径,也要回显让用户确认。

询问话术:

- **A(推荐默认)「用当前用户的 HDFS 家目录 `/user/{当前用户}/feature-matching`」** → 自动拼当前 `whoami`
- **B. 指定其他 HDFS 目录** → 让用户填绝对路径,写入 `spark_submit.hdfs_base`
- **C. task-spec 文档已指定,直接用** → 仅当 task-spec 文档存在该字段时可选,回显值让用户确认

来源优先级:task-spec 文档 > 交互询问(**不取 yaml 默认值**,yaml 留空待填)。

**强制点**:
- 不问就生成脚本 = 违反约定
- 即使用户上次填过,新 session 也要重新确认
- yaml `hdfs_base` 留空 → 报错终止,不允许 spark 直接写本地(会触发权限错误)
- 确认后写入 `spark_submit.hdfs_base`,最终 HDFS 中间路径 = `{hdfs_base}/{model.name}_{model.version}/sample.parquet`

## 3. ⚠️ 拼接模式交互约定(强制,启用样本表⋈特征表模式时必须先确认五要素)

**启用样本表⋈特征表模式时,生成脚本前必须先确认以下五要素,确认齐全前不要生成脚本**:

1. 样本表 → `model.sample_table`(提供 label + 主键的主表)
2. 特征表 → `model.feature_table`(提供特征的副表)
3. join-key → `model.join_keys`,**默认不要闷头用 `user_no+pday`**
4. 特征列 → `model.features` / `feature_list_source`,留空=取特征表全部列或指定清单
5. HDFS 中间路径 → `spark_submit.hdfs_base`,**必填**,留空会把本地路径误当 HDFS 写触发权限错误

获取来源优先级:

1. 优先从 `classification-model-task-spec` 的规格文档读取(解析到的值回显用户确认后写入配置)
2. 无 task-spec 时退而从其他上游 markdown(`classification-model-recommend` 的 `reports/{model_id}_*.md`、`classification-model-training` 的 run `report.md` 或用户提供的任意 markdown)解析
3. 文档缺失或解析不全时逐项交互询问:
   - 样本表/特征表(库.表)
   - join-key(`user_no + pday` 时序样本 / 仅 `user_no` 无日期维度 / 其他列名)
   - 特征列(全部列/指定清单)
   - HDFS 中间路径(默认家目录或用户指定)
   - 另需确认 label 列名与取数窗口

确认齐全后写入 `model.{sample_table, feature_table, join_keys, features}` 与 `spark_submit.hdfs_base` 再生成脚本。

## 4. ⚠️ t-1 滞后 JOIN 模式(`--feature-lag-day 1`)

启用时: 样本表 day t LEFT JOIN 特征表 day t-1, 适用于"当天样本用昨日特征"场景(避免特征泄漏)。

要求:
- `dt_col` 必须在 `join_keys` 中(默认即如此)
- 仅 spark 模式 + 拼接模式(`--feature-table` 非空)生效
- 单表模式 + lag=1 报错终止

lag=1 时特征表窗口自动平移为 `[fetch_start-1, fetch_end-1]`, 无需手动改 `--fetch-start/--fetch-end`。

## 5. ⚠️ 全自动驾驶模式(driving_mode=auto)跳过规则

进入 skill 先读 `<session_dir>/task-spec/_manifest.json` 的 `driving_mode`:

- `driving_mode == "auto"`: 上述第 1~3 节三个交互约定**全部跳过**(第 4 节 t-1 lag 仍尊重用户显式传参),用默认值推进:
  - 拼接键 join-key → 用默认 `user_no,pday`
  - 特征清单 → 走 `feature-knowledge.md` 索引自动识别(`feature_table` 优先 / `business_domain` 兜底),识别不到时全量派生
  - HDFS 中间路径 → 用 `default_hdfs_base("feature-matching")`(当前用户 HDFS 家目录)
  - t-1 滞后 JOIN → 默认 `feature_lag_day=0`(同日 JOIN)
- 用户显式传入 `--feature-lag-day N`(N≠0)时尊重并打 warn(用户可能有意为之)
- 日志打印 `[autopilot] driving_mode=auto, 跳过 3 个交互约定, 用默认值推进`
- **仍执行**: 取数本身(spark-submit / local_file 转写)、feature-list.csv 派生、yaml 落盘
- `driving_mode == "manual"`(默认): 走第 1~3 节交互约定,3 个 ⚠️ 点逐个问用户
- manifest 缺失或字段缺失: 视为 `manual`,打 warn 继续

> task-spec 自身是 manifest 的**生产者**,在 3.1.5 节检测到「全自动驾驶」/「自动驾驶」关键词后,通过 `fetch_sample.py --driving-mode auto` 写入 `sample_config.<model_name>.yaml` 的 `model.driving_mode` 字段;后续编排器把该值落到 `task-spec/_manifest.json`。本 skill 作为**消费者**读 manifest,不改 driving_mode 字段。

---

> 关联:SKILL.md 第 6 节执行约束;spark 模式工作流见 `references/spark-workflow.md`;参数表见 `references/fetch-parameters.md`。
