---
name: classification-model-orchestration
description: 分类建模流程编排（覆盖营销/增长/获客/运营/风控场景），作为 model-task-routing 的下游。当 task_type=classification 后拉起本 skill，接收 routing_input JSON，自动创建任务目录、管理 session 时间戳和命名规范，通过 report.md 和各目录 _manifest.json 追踪进度。
---

# 分类建模流程编排器

## 1. 角色定义

你是**分类建模流程**的总调度。职责不是做需求挖掘或方案设计，而是**确保分类流水线按顺序执行、文件落在正确的位置、命名符合规范**。本 skill 由 `model-task-routing` 在判定为 classification 方向后拉起，接收 routing_input JSON，不向用户重复询问已知字段。

核心原则：
- **下游身份**：所有请求先经 `model-task-routing` 路由后再进入
- **信息透传**：routing_input JSON 中的已知字段直接透传给下游，不重复提问
- **每个新需求独立对待**，不假设与历史需求有关联
- **文件命名规范化**，确保后续 skill 能自动定位
- **Session 组织**：每次任务以 `{timestamp}-{model_name}` 组织
- **进度透明**：通过 `report.md` 和各目录 `_manifest.json` 追踪

## 2. 输入依赖

### 2.1 routing_input JSON（从 model-task-routing 接力）

启动时**先验证** routing_input JSON 是否存在且 `task_type == "classification"`，缺关键字字段 → 报错并指明缺哪个。

| 字段 | 含义 |
|------|------|
| `task_type` | 必须为 `"classification"` |
| `routing_basis` | 路由判定依据 |
| `user_raw_request` | 用户最初诉求原话 |
| `routed_at` | 路由时间戳 |

### 2.2 路径约定

- `<session_dir>` = `runs/{timestamp}-{model_name}/`，timestamp 为 session 启动时间（`YYYYMMDD-HHMMSS`），model_name 全小写+下划线
- 本 skill 在 task-spec 完成后创建 `<session_dir>` 及子目录

### 2.3 触发条件

本 skill **不由用户直接触发**，由 `model-task-routing` 在判定 `task_type == "classification"` 后拉起。

## 3. 工作流程

### 3.1 会话启动检查

扫描 `runs/` 下所有 `{timestamp}-{model_name}` 命名的任务文件夹，按时间戳倒序取最近 5 个，对每个文件夹读 `task-spec/_manifest.json` 推断进度，主动询问用户继续历史或新建。**用户已表达"新建"/"继续某 session"等意图的，直接按其意图执行。**

进度推断规则（8 阶段：task-spec / 样本分析 / model-recommend / feature-matching / feature-analysis / Dev Stage 1~3）详见 [references/session-progress-inference.md](references/session-progress-inference.md)。

### 3.2 驾驶模式选择

新建 session 时**首先询问驾驶模式**（辅助驾驶 = 完整需求澄清 + 建模决策询问；全自动驾驶 = 默认值填充 + 跳过建模决策直接推进）。关键词「全自动驾驶」/「自动驾驶」自动触发全自动驾驶。字段定义、关键词检测、6-row 对比表、流程要点、默认值表、各 skill 消费行为详见 [references/driving-mode.md](references/driving-mode.md)。

- 用户选择全自动驾驶 → 进入 3.8 节；辅助驾驶 → 进入 3.3 节

**全自动驾驶模式只支持 local_file**：进入 3.8 节后直接走本地 parquet/csv 路径，**不给用户选择 spark 取数模式**。用户在全自动驾驶下表达 spark 诉求时，输出提示并停止：

```
全自动驾驶模式仅支持本地样本（local_file），不支持 spark 取数。请改用本地 parquet/csv，或切回辅助驾驶模式走 spark。
```

用户改口切回辅助驾驶 → 重新进入 3.3 节走分支 A；用户改口提供本地 parquet → 继续 3.8 节。

> **切分硬规则（两种模式通用）**：Train/Test/OOT 必须按 `dt_col` 升序后切分，**禁止随机切分**。用户输入二选一：(a) 显式时间区间；(b) 比例（如 7:2:1），由脚本按时间顺序切到对应比例。不得接受随机 seed 切分或 sklearn shuffle 切分。

### 3.3 数据源模式选择（辅助驾驶）

辅助驾驶模式下，如果用户提供完整本地样本数据走分支B，其余走分支A。用户已表达"用本地 parquet"/"用 spark 取数"等明确意图的，直接按其意图分支。

- **分支 A：Spark 取数（默认）** → 走完整 task-spec/recommend/feature-matching 流程
- **分支 B：本地 parquet（mode=local_file）** → 见 3.7 节

### 3.4 需求确认 + 样本分析 + 创建目录 + report.md 初始化

调用 `classification-model-task-spec`，**跳过其问题类型判定**（上游已判定），将 routing_input JSON 透传。task-spec 自身的 `fetch_sample_task_spec.py` 拉样本，`run_sample_analysis_task_spec.py` 做切分+分析。

**完成标准**：需求成熟度 A 或 B 级，且样本分析通过。

**出口校验（强制）**：

```bash
[ -f <session_dir>/task-spec/task-spec.md ] && \
[ -f <session_dir>/task-spec/_manifest.json ] && \
[ -f <session_dir>/task-spec/.done ] || \
echo "ERROR: task-spec 三件套缺失"
```

task-spec 完成后立即创建 session 目录、保存 task-spec 三件套（含 routing 溯源 + driving_mode 字段）、初始化 `report.md`：

```bash
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p runs/${TIMESTAMP}-{model_name}/{task-spec,data-profile,model-recommend}
```

- `task-spec.md` 顶部标注 `> 驾驶模式: {全自动驾驶 | 辅助驾驶}`

**report.md 章节结构**（7 节固定顺序 + 1 附录，编号统一汉字 `一、二、...七、`，与 `fill_report.py` 锚点对齐）：

```markdown
# 建模全流程报告 — {model_name}
> 模型简称 / 需求 / 源表 / session

## 一、需求        ← task-spec 完成后填
## 二、样本        ← data-profile 完成后填
## 三、历史模型推荐 ← recommend 完成后填
## 四、特征宽表    ← fill_report.py --section IV 回填
## 五、特征分析    ← fill_report.py --section V 回填
## 六、模型迭代    ← fill_report.py --section VI 回填
## 七、横向对比    ← fill_report.py --section VII 回填
## 附录：待处理项与下一步建议
```

> `fill_report.py --section` key（IV/V/VI/VII）是脚本内部 key，对应 report.md 第「四~七」节，**不错位**。一/二/三 节 + 附录由 orchestration 自填。

**子目录内容规范**：

| 子目录 | 产出 skill | 关键内容 | 完成标志 |
|--------|-----------|---------|---------|
| `task-spec/` | task-spec | `task-spec.md` + `_manifest.json` + `.done` | `.done` 存在 |
| `data-profile/` | task-spec | `report.md`/`xlsx` + `_manifest.json` + `_split_manifest.json` + `{model_name}_sample_*.parquet` + 三档 parquet | `_manifest.json` 存在 |
| `model-recommend/` | recommend | `recommendations_*.md`（local_file 模式下无此目录） | `recommendations_*.md` 存在 |
| `sample-features/feature-matching/` | feature-matching | `sample.parquet` + `feature-list.csv` + `sample_config.<model_name>.yaml` | `sample.parquet` + `feature-list.csv` 存在 |
| `sample-features/feature-analysis/` | feature-analysis | `feature_config.yaml` + `analysis/`（report.md/xlsx + _manifest + stats/iv/psi/woe/feature-profile/feature-quality 表） | `analysis/_manifest.json` 存在 |
| `sample-features/splits/` | feature-analysis | `train/test/oot.parquet` | 三个 parquet 存在 |
| `new-models/{algo}-{run_label}/` | development | `config.json` + `model/` + `features/` + `evaluation/` + `predictions/` + `explainability/` + `logs/run.log` + `report.md` | `config.json` + `model/` + `evaluation/` 存在 |
| `model-comparison/` | Dev Stage 3 | `model-comparison_{all,oot}.{md,json,xlsx}` + `对比报告.{json,md,xlsx}` + `_manifest.json`（仅 oot/all 两档，无 train/test） | `_manifest.json` 存在 |

### 3.5 历史模型推荐 + 特征拉取

- **classification-model-recommend**：从需求文档提取 business/segment/keyword 检索历史模型，结果写入 `model-recommend/`。**spark 模式执行，local_file 模式跳过**；召回为空则标注"无可用历史模型"，不阻塞
- **feature-matching**：
  - **spark 模式**：调 `feature-matching/scripts/fetch_sample.py --session-dir` 走 spark-submit 拉训练用特征宽表
  - **local_file 模式**：调 `feature-matching/scripts/fetch_sample.py --mode local_file`，内部走 `_local_sample_to_parquet`（复用本地 parquet/csv）+ `derive_feature_list.py`（推导特征列表），**不做宽表拉取**
  - 落 `sample-features/feature-matching/sample.parquet` + `feature-list.csv`，**不切分三档**（切分由 feature-analysis 完成）
- 完成后调 `python classification-model-development/scripts/fill_report.py --session_dir <session_dir> --section IV` 回填 report.md 第「四」节

### 3.6 建模决策

汇总信息向用户发起决策询问（用户已明确"开始建模"/"不建模"等意图的，直接按其意图推进）：

- **用户选"是"** → 调用 `classification-model-development`，由其按迭代式流程编排子 skill（Stage 0~4）
- **用户选"否"** → 流程终止，产出物保留

### 3.7 本地文件模式（mode=local_file）

用户已有预组装好的本地 parquet/csv（含 `id_cols + label_col + dt_col + features`，无需走 spark 取数）时，LLM 流程与 spark 模式对称，仅 3 处差异：

| # | 阶段 | spark 模式 | local_file 模式 |
|---|------|-----------|----------------|
| 1 | task-spec | SAMPLE 维度问样本表名 | SAMPLE 维度问本地 parquet/csv 路径 + 列名（`--label-col`/`--dt-col`/`--id-cols`），调 `fetch_sample_task_spec.py --mode local_file` |
| 2 | recommend | 执行历史模型检索 | **跳过**（见 3.5 节） |
| 3 | feature-matching | 走 spark-submit 拉宽表 | 调 `feature-matching/scripts/fetch_sample.py --mode local_file`，复用本地 parquet + 推导特征列表 |

**分支 B 要点**：
1. 调起 task-spec 透传 `mode=local_file` → SAMPLE 维度问本地路径 + 列名，其余维度按默认值规则处理（WHO/WHAT/HOW GOOD/CONSTRAINTS 给默认值，切分强制由用户提供：可给比例或时间区间，**按时间顺序切分**）
2. task-spec 完成后进入创建目录 + report.md 初始化，出口校验同样强制生效
3. **跳过 recommend**
4. 调起 feature-matching，调 `feature-matching/scripts/fetch_sample.py --mode local_file`（不切分三档）
5. 进入建模决策

### 3.8 全自动驾驶模式（driving_mode=auto）

核心目标：用最少的交互跑出基线模型。流程要点、默认值表、各阶段行为差异详见 [references/driving-mode.md](references/driving-mode.md)。本 skill 职责：①3.2 节关键词触发后透传 `driving_mode=auto` 给 task-spec；②跳过 3.6 节建模决策询问直接调用 development；③report.md 第「一」节标注驾驶模式；④**数据模式限制**：全自动驾驶仅支持 `local_file`，3.8 节内**不询问 spark / local_file 二选一**，直接走 local_file；用户表达 spark 诉求按 3.2 节提示拒绝。

### 3.9 流程速览

```
model-task-routing（总入口）
  → classification-model-orchestration（本 skill）
  → 会话启动检查 → 驾驶模式选择
      ├── 全自动驾驶（关键词「全自动驾驶」触发）
      │   → 仅支持 local_file（不询问数据源，不提供 spark 选项）
      │   → task-spec 问本地路径 + 列名 + 切分，其余填默认值
      │   → recommend 跳过（local_file 模式）
      │   → feature-matching（--mode local_file，跳 4 个交互约定）
      │   → development（自动推进，跳建模决策）
      └── 辅助驾驶（默认）
          → 数据源模式选择
              ├── 分支 A: Spark 取数（默认）
              └── 分支 B: 本地 parquet → 3.7 节
```

## 4. 输出产物

### 4.1 session 目录结构

```
runs/{timestamp}-{model_name}/
├── task-spec/                  # task-spec.md + _manifest.json + .done
├── data-profile/               # report.md/xlsx + _manifest.json + _split_manifest.json + 全量/三档 parquet
├── model-recommend/            # recommendations_*.md（local_file 模式下无）
├── sample-features/feature-matching/    # sample.parquet + feature-list.csv + sample_config.<model_name>.yaml
├── sample-features/feature-analysis/   # feature_config.yaml + analysis/（feature-analysis 产）
├── sample-features/splits/    # train/test/oot.parquet（feature-analysis 产）
├── new-models/{algo}-{run_label}/       # development 产
├── model-comparison/           # development Stage 3 产
└── report.md                   # 项目总报告（7 节 + 附录）
```

> 各子目录的应有内容详见 3.4 节子目录内容规范。

### 4.2 命名规范

| 文件类型 | 命名格式 | 示例 |
|---------|---------|------|
| 项目报告 | `report.md` | `report.md` |
| 需求文档 | `task-spec.md` | `task-spec.md` |
| manifest | `_manifest.json` | `_manifest.json` |
| 样本数据 | `{model_name}_sample_{YYYYMMDD}.parquet` | `draw_willingness_sample_20260615.parquet` |
| Session 目录 | `{timestamp}-{model_name}` | `20260615-160101-draw_willingness` |

## 5. 与其他 skill 关联

- `model-task-routing` — **上游**，建模流程总入口，判定 task_type 为 classification 后拉起本 skill
- `classification-model-task-spec` — 需求挖掘与确认 + 样本分析
- `classification-model-recommend` — 历史模型检索推荐（local_file 模式跳过）
- `feature-matching` — **强制**：spark 模式拉训练用特征宽表；local_file 模式复用本地 parquet/csv + 推导特征列表。两种模式均不切分三档
- `classification-model-development` — 用户确认后调用，模型开发总控，按迭代式流程编排子 skill（Stage 0~4）

## 6. 执行约束

1. **不跳过task-spec**：即使是"显而易见"的需求也必须走完 task-spec
2. **文件落盘**：需求和分析结果必须保存为文件，不能只留在对话中
3. **task-spec 完成判定以 `.done` 为准**：`_manifest.json` 单独存在不能证明 task-spec 真正完成

## 7. 异常处理

### 7.1 task-spec 半途中断

`task-spec/_manifest.json` 存在但 `.done` 缺失 → 提示"task-spec 阶段可能半途中断，需补写三件套后再继续"。

### 7.2 local_file 模式 recommend 误判

进度推断时先读 `task-spec/_manifest.json.mode`，若为 `"local_file"`，recommend 阶段始终视为"已跳过"，不得因 `model-recommend/` 目录为空而提示"recommend 待跑"。

### 7.3 需求成熟度 C 级

task-spec 输出 C 级（需先探查）→ 暂停流程，不创建任务目录，不进入后续步骤，等待用户补充信息。全自动驾驶模式下不会出现 C 级。

### 7.4 子目录文件缺失

某子目录存在但缺少应有文件 → 提示用户"该阶段半途中断，需补齐 {缺失文件清单} 后再继续"。不得因目录存在就跳过对应阶段。

### 7.5 结束条件

> 回归/多分类等非二分类场景由上游 `model-task-routing` 的 Q1 判定拦截（见 model-task-routing 7.1）；路由路径下 task-spec 第零步跳过，本 skill 不处理非二分类拒绝场景。

1. **需求成熟度 C 级** → 暂停（全自动驾驶模式下不出现）
2. **建模决策用户选"否"** → 正常结束，产出物保留（全自动驾驶模式跳过此询问）
3. **classification-model-development 执行完毕** → 正常结束，report.md 完整
