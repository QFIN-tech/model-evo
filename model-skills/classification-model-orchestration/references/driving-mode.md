# 驾驶模式（driving_mode）

> 本文件是驾驶模式的单一事实源。原散落在 4 处的内容（orchestration 3.2/3.8 节、task-spec 3.1.5 节/第 6 节约束 14、feature-matching 2.0 节、development 3.0 节）已合并到此。各 skill 的 SKILL.md 仅保留 1-2 行摘要 + 指向本文件的指针。

## 1. 字段定义与生产消费链

| 项 | 内容 |
|---|---|
| 字段名 | `driving_mode` |
| 取值 | `"auto"`（全自动驾驶）/ `"manual"`（辅助驾驶，默认） |
| 缺失语义 | 视为 `"manual"`，打 warn 继续 |
| 生产者 | `classification-model-task-spec`（3.1.5 节关键词检测后写入） |
| 消费者 | `feature-matching` / `classification-model-development`（读 manifest 调整行为） |
| 落盘位置 1 | `<session_dir>/task-spec/_manifest.json` 顶层 `"driving_mode": "auto" \| "manual"` |
| 落盘位置 2 | `task-spec.md` 顶部标注 `> 驾驶模式: 全自动驾驶` 或 `> 驾驶模式: 辅助驾驶` |
| 落盘位置 3 | `sample_config.<model_name>.yaml` 的 `model.driving_mode` 字段，由 `fetch_sample_task_spec.py --driving-mode` 透传 |

**消费者不改字段**：feature-matching / development 只读 `driving_mode`，不修改。task-spec 是唯一生产者。

## 2. 关键词检测（task-spec 3.1.5 节）

第零步问题类型判定通过后，扫描用户原始诉求（`routing_input.user_raw_request` 或用户首轮回复）是否包含关键词：

- 「全自动驾驶」/「自动驾驶」→ `driving_mode = "auto"`
- 其他 → `driving_mode = "manual"`（默认）

## 3. 辅助驾驶 vs 全自动驾驶对比表

| # | 阶段 | 辅助驾驶 | 全自动驾驶 |
|---|------|---------|-----------|
| 1 | 数据源 | 用户选 spark / local_file | **仅 local_file**（不提供 spark 选项） |
| 2 | 需求澄清 | 5 项维度 + 样本 + 切分全问 | **只采集 local_file 信息**：本地 parquet/csv 路径 + 列名 + 切分 |
| 3 | WHO/WHAT/HOW GOOD/CONSTRAINTS | 逐项确认 | **全部填默认值** |
| 4 | recommend | 见 orchestration 3.5 节（spark 执行 / local_file 跳过） | 跳过（local_file 模式） |
| 5 | feature-matching | 走对应模式 | 走 local_file 模式（`--mode local_file`）；读 manifest `driving_mode=auto` 后跳过 4 个交互约定 |
| 6 | 建模决策 | 询问用户 | **跳过询问直接推进到 development** |

## 4. 全自动驾驶流程要点

1. 调起 task-spec 透传 `driving_mode=auto` 标记 → task-spec **只采集 local_file 信息**：本地 parquet/csv 路径 + 列名（`--label-col`/`--dt-col`/`--id-cols`）+ 切分，其余维度填默认值，**不发起需求确认轮次**（自动落盘）。**不询问样本表名，不提供 spark 选项**。
2. task-spec.md 顶部标注 `> 驾驶模式: 全自动驾驶`；`_manifest.json` 写入 `driving_mode: "auto"`
3. recommend：跳过（local_file 模式）
4. 调起 feature-matching，调 `feature-matching/scripts/fetch_sample.py --mode local_file`；读 manifest 跳过 4 个交互约定
5. **跳过建模决策询问**，直接调用 `classification-model-development`
6. report.md 第「一、需求」节标注 `> 驾驶模式: 全自动驾驶（默认值填充，未逐项确认）`

## 5. 默认值表（task-spec 落盘时填写）

| 维度 | 默认值 |
|------|--------|
| 业务场景 | 待业务方确认 |
| 标签人群 | 全量样本 |
| 预测目标 | {label_col} 列对应的二分类目标 |
| 表现窗口 | 待业务方确认 |
| 预估正样本率 | 待数据探查 |
| 效果目标 | 当前无目标（模型建立后确定基线） |
| 触达方式 | 离线T+1跑批打分 |
| 约束条件 | 无 |

## 6. 各 skill 消费行为

### 6.1 task-spec（生产者）

**全自动驾驶模式下的行为调整**：

- **数据模式**：**仅 local_file**。全自动驾驶下不询问、不提供 spark 选项；用户表达 spark 诉求时拒绝（见 orchestration 3.2 节）
- **仍强制询问**（无默认值）：本地 parquet/csv 路径 + 列名（`--label-col`/`--dt-col`/`--id-cols`）+ Train/Test/OOT 切分
- **跳过询问，按默认值填充**：WHO 人群 → 全量样本；WHAT → `{label_col}`；HOW GOOD → 当前无目标；CONSTRAINTS → 无；触达方式 → 离线T+1跑批打分；Train/Test/OOT 切分 → **按 `dt_col` 升序后 6:2:2 时间顺序切分，同一天的数据放入同一 set**（用户显式给时间区间或比例时尊重）
- **跳过 3.5 节需求确认复述**：直接落 task-spec.md，标注「需求成熟度 A（全自动驾驶，默认值填充）」
- **跳过补充字段询问**

### 6.2 classification-model-orchestration（编排者）

- 3.2 节新建 session 时首先询问驾驶模式；关键词触发时自动选全自动驾驶
- 3.8 节全自动驾驶下：跳过建模决策询问，直接调用 development
- report.md 第「一」节标注驾驶模式

### 6.3 feature-matching（消费者）

进入本 skill 后，**首要动作是读** `<session_dir>/task-spec/_manifest.json` 的 `driving_mode` 字段：

- `driving_mode == "auto"`：
  - `references/interaction-conventions.md` 第 1~3 节三个 ⚠️ 强制交互约定**全部跳过**，直接用默认值推进：
    - 拼接键 join-key → 用默认 `user_no,pday`
    - 特征清单 → 走 `feature-knowledge.md` 索引自动识别（`feature_table` 优先 / `business_domain` 兜底），识别不到时全量派生
    - HDFS 中间路径 → 用 `default_hdfs_base("feature-matching")`（当前用户 HDFS 家目录）
    - t-1 滞后 JOIN → 默认 `feature_lag_day=0`（同日 JOIN）
  - 用户显式传入 `--feature-lag-day N`（N≠0）时尊重并打 warn
  - 日志打印 `[autopilot] driving_mode=auto, 跳过 3 个交互约定, 用默认值推进`
  - **仍执行**：取数本身（local_file 转写）、feature-list.csv 派生、yaml 落盘
- `driving_mode == "manual"`（默认）：走 `references/interaction-conventions.md` 第 1~3 节交互约定，3 个 ⚠️ 点逐个问用户
- manifest 缺失或字段缺失：视为 `manual`，打 warn 继续

### 6.4 classification-model-development（消费者）

进入本 skill 后，**首要动作是读** `<session_dir>/task-spec/_manifest.json` 的 `driving_mode` 字段：

- `driving_mode == "auto"`（全自动驾驶）：
  - 第 5 节各 Stage 决策点话术**全部跳过**，按默认路径推进：
    - Stage 0 特征分析完成后 → 直接 continue 进 Stage 1（不问 A/B/C）
    - Stage 1 baseline 训练完成后 → 直接进 Stage 3（不问 A-E，跳过 Stage 2 迭代）
    - Stage 2 迭代 → 跳过（autopilot 默认不迭代，最简路径：xgb baseline → Stage 3 对比 → Stage 4 自动选最优）
    - Stage 3 横向对比 → 自动执行（本就是 `run_build.py` 末尾自动触发）
    - Stage 4 收口 → 自动选 OOT AUC top1 落盘为上线候选，不再问用户
  - 默认 baseline 算法：**xgb**（最简路径，单算法不切换）
  - 日志打印 `[autopilot] driving_mode=auto, Stage X 自动推进`
- `driving_mode == "manual"`（辅助驾驶，默认）：走第 5 节话术，各决策点必问
- manifest 缺失或字段缺失：视为 `manual`，打 warn 继续

## 7. 切分硬规则（两种模式通用）

Train/Test/OOT 必须按 `dt_col` 升序后切分，**禁止随机切分**。用户输入二选一：

- (a) 显式时间区间
- (b) 比例（如 7:2:1），由脚本按时间顺序切到对应比例

不得接受随机 seed 切分或 sklearn shuffle 切分。

**全自动驾驶下的切分**：默认按 `dt_col` 升序后 6:2:2 时间顺序切分（同一天数据放同一 set），无需用户确认；用户显式给时间区间或比例时尊重。

## 8. 异常处理

| 场景 | 处理 |
|---|---|
| manifest 缺失 `driving_mode` 字段 | 视为 `"manual"`，打 warn 继续 |
| 全自动驾驶下需求成熟度 C 级 | 不出现（默认值填充保证成熟度至少 B 级） |
| 全自动驾驶下建模决策 | 跳过询问（用户选全自动即默认同意建模） |
| 全自动驾驶下用户表达 spark 诉求 | 拒绝并提示"仅支持 local_file"，建议改用本地 parquet 或切回辅助驾驶（见 orchestration 3.2 节） |

> 关联: `classification-model-orchestration/SKILL.md` 3.2/3.8 节、`classification-model-task-spec/SKILL.md` 3.1.5 节/第 6 节约束 14、`feature-matching/SKILL.md` 2.0 节、`classification-model-development/SKILL.md` 3.0 节
