# Model Skills

业务建模全流程 Skill 集合，覆盖从需求采集、样本准备、特征工程、模型开发、评估对比到归档沉淀的完整链路。所有 Skill 遵循统一的命名约定与产物规范，由 `model-task-routing` 作为总入口统一分流到 **classification（分类）** 与 **uplift（增益/因果）** 两种建模类型。

> 每个 Skill 位于独立目录下的 `SKILL.md`，包含 frontmatter（`name` + `description`）与正文（输入依赖 / 执行命令 / 参数说明 / 输出产物 / 关联 skill / 执行约束 / 异常处理）。

## 整体能力

- **任务路由**：一次性提问判定 classification / uplift / 拒绝，按建模类型分流
- **分类建模**：需求规格化 → 历史模型推荐 → 特征准备 → baseline 模型开发（特征分析 / 训练 / 评估）→ LOOP 迭代式开发（确定优化方案 → 模型开发 → 横向对比，多轮循环）→ 收口归档
  - 当前已支持的优化方向：超参调优（Optuna）、特征筛选（PSI/IV/缺失率）、换算法（xgb / dnn / lr）；特征衍生、样本调整、loss 优化、网络结构优化等为规划中的演化方向
- **Uplift 建模**：任务规格化 → 样本准备 → 同质性检查 → 特征质量分析 → S/T-Learner 建模 → 调参 → 多模型对比 → 最终报告
- **共享能力**：特征匹配、特征分析、模型知识库（业务领域知识 / 特征资产 / 历史模型档案 / 建模经验）
- **会话连续性**：基于 `runs/` 下的 `_manifest.json` 自动推断进度，支持断点续跑


## 目录结构

```
model-skills/
├── model-task-routing/                  # 总入口：判定 classification / uplift / 拒绝
├── model-knowledge/                     # 模型知识库（业务领域 / 特征 / 历史模型 / 建模经验）
│
├── classification-model-orchestration/  # 分类流程编排器（接收 routing_input）
├── classification-model-task-spec/      # 分类任务规格化 + 样本分析
├── classification-model-recommend/      # 历史模型检索推荐
├── classification-model-development/    # 模型开发总控（迭代式编排）
├── classification-model-training/       # 模型训练（xgb / dnn / lr）
├── classification-model-tuning/         # 模型调参 / 特征筛选
├── classification-model-evaluation/     # 单模型标准化评估
├── classification-model-comparison/     # 多模型 N-way 对比
│
├── uplift-model-task-spec/              # uplift 任务规格化（流程入口）
├── uplift-model-sample-preparation/     # 样本切分方案 + 建模样本产物
├── uplift-model-sample-homogeneity-check/ # treatment/control 协变量均衡性诊断
├── uplift-model-feature-quality-analysis/ # 特征质量诊断 + 筛选建议 + feature plan
├── uplift-model-s-learner-modeling/     # S-Learner baseline 训练
├── uplift-model-t-learner-modeling/     # T-Learner baseline 训练
├── uplift-model-tuning/                 # LightGBM learner 有边界调参
├── uplift-model-result-comparison/      # 多模型/版本确定性比较
├── uplift-model-reporting/              # 最终建模报告（MD + HTML + facts）
│
├── feature-matching/                    # 特征匹配（拉样本+特征，跨流程共用）
├── feature-analysis/                    # 特征分析（IV / PSI / 基础统计，跨流程共用）
│
└── _uplift-common/                      # uplift 公共代码（非 skill，project_layout / data_loader / io 等）
    ├── scripts/_common/
    └── tests/

```

> **公共代码说明**：[`model-evo/_modelevo-shared/`](../_modelevo-shared/)也会自动安装到SKILL_ROOT目录下供各skill公共使用。

## Skill 清单

### 总入口 / 共享

| Skill | 说明 | 触发词示例 |
|---|---|---|
| `model-task-routing` | 建模流程总入口，一次性提问判定 classification / uplift / 拒绝，构造 routing_input JSON 透传下游 | 建模、新模型、模型需求、帮我建模、模型立项 |
| `feature-matching` | 从 Spark 宽表拉取样本+特征+标签，生成 spark-submit 提交脚本，确认后自动提交集群落 `sample.parquet`；或 `local_file` 模式下从本地 parquet/csv 直接转写+派生 `feature-list.csv` | 取数、拉样本、拉宽表、准备建模样本 |
| `feature-analysis` | 对候选特征做基础统计 / 单变量预测力（IV+AUC）/ 训练-OOT 稳定性（PSI）报告，仅产报告不自动剔特征，并按配置切分 Train/Test/OOT | 特征分析、特征 IV、特征 PSI |
| `model-knowledge` | 沉淀建模方法论、业务领域知识、特征资产、历史模型档案与建模经验教训，供检索复用 | 查历史建模经验、归档建模知识、查业务字段定义 |

### Classification 建模

| Skill | 说明 |
|---|---|
| `classification-model-orchestration` | 分类流程总调度，承接 routing_input，串联 task-spec → recommend → feature-matching → development，管理 session 目录与断点续跑 |
| `classification-model-task-spec` | 需求挖掘 + 样本分析，输出 4 段式 task-spec.md 与 `_manifest.json`，拉取样本（仅 user_no/label/pday + 补充字段）并按时间顺序切分 Train/Test/OOT |
| `classification-model-recommend` | 从历史模型台账检索可复用模型，语义筛选排序 + 适配度评估，可选委托 evaluation 产三档评估 |
| `classification-model-development` | 开发总控，按 Stage 0~4 迭代式编排 feature-analysis / training / tuning / comparison，管理路径接力、决策点询问、report.md 回填 |
| `classification-model-training` | 训练 xgb / dnn / lr 模型，读上游 feature-analysis 切分数据，产八阶段产物，并与历史 baseline 做 AUC/KS/分档多维对比 |
| `classification-model-tuning` | 基于 baseline run 做超参调优（Optuna）或特征筛选（PSI/IV/缺失率），产 `-tuned` / `-feat` 新 run |
| `classification-model-evaluation` | 单模型标准化评估：AUC/KS/准确率/F1/十分桶排序性/业务指标 + 客群拆分，输出 JSON+MD+XLSX 三件套 |
| `classification-model-comparison` | 多模型 N-way 横向对比，消费 evaluation 的 JSON 做 delta 分析与缺口清单，输出含条件格式的 Excel |

> 模型上线（`model-publication`）、指标匹配（`metric-matching`）、演化方案（`classification-model-evolution-plan`）、分群建模（`classification-segment-model`）等为规划中的能力，**当前尚未实现**，不包含在本次交付内。

### Uplift 建模

| Skill | 说明 |
|---|---|
| `uplift-model-task-spec` | uplift 任务规格化（流程入口）：收集并确认结构化 `task_config`（干预定义、目标增量、实验/对照组、样本主体、时间窗口、字段映射） |
| `uplift-model-sample-preparation` | 检查样本语义、起草并确认切分方案、生成建模样本产物，产出 confirmed `modeling_sample_spec_path` |
| `uplift-model-sample-homogeneity-check` | treatment/control 协变量均衡性诊断，判断两组是否同质/可比 |
| `uplift-model-feature-quality-analysis` | 特征质量诊断（缺失率/唯一值/Split PSI/Monthly PSI/Uplift Bivar）+ 筛选建议，沉淀 confirmed `feature_plan_path` |
| `uplift-model-s-learner-modeling` | 训练 baseline S-Learner 模型，输出模型元数据、评估指标、打分表、uplift 分箱、特征重要性、报告 |
| `uplift-model-t-learner-modeling` | 训练 baseline T-Learner 模型，输出同上（特征重要性按 component 分段） |
| `uplift-model-tuning` | 基于 S/T-Learner baseline 做有边界的 LightGBM learner 调参 study，复用 trial_000 作对照，输出 study winner |
| `uplift-model-result-comparison` | 在同一任务和评估 split 下做多模型/版本确定性比较，输出比较表、摘要报告和推荐模型 path |
| `uplift-model-reporting` | 从上游结构化结果生成最终 Uplift 建模报告（Markdown + HTML + facts + manifest） |

## 完整流程

```
用户建模诉求
   │
   ▼
model-task-routing（一次性提问 Q1/Q2/Q3 → 判定 task_type）
   │
   ├── classification ──► classification-model-orchestration
   │                          │
   │                          ├─ 会话启动检查（扫描 runs/ → _manifest.json 推断进度）
   │                          ├─ classification-model-task-spec（需求确认 + 样本分析）
   │                          ├─ 创建任务目录 + 初始化 report.md
   │                          ├─ classification-model-recommend（历史模型推荐）
   │                          ├─ feature-matching（拉特征宽表 + 派生 feature-list.csv）
   │                          └─ 建模决策（询问用户）
   │                                  │
   │                                  ├── 是 ──► classification-model-development
   │                                  │            ├─ Stage 0: feature-analysis（一次性必跑）
   │                                  │            ├─ Stage 1: classification-model-training（baseline）
   │                                  │            ├─ Stage 2: 迭代（tuning 调参 / 特征筛选 / 换算法，loop）
   │                                  │            ├─ Stage 3: classification-model-comparison（session 级）
   │                                  │            └─ Stage 4: 收口 → report.md
   │                                  └── 否 ──► 流程结束
   │
   ├── uplift ──► uplift-model-task-spec
   │                  │
   │                  ▼
   │            uplift-model-sample-preparation
   │                  │
   │                  ▼
   │            uplift-model-sample-homogeneity-check
   │                  │
   │                  ▼
   │            uplift-model-feature-quality-analysis
   │                  │
   │                  ▼
   │            uplift-model-s-learner-modeling  /  uplift-model-t-learner-modeling
   │                  │
   │                  ▼
   │            uplift-model-tuning（可选，基于 baseline 调参）
   │                  │
   │                  ▼
   │            uplift-model-result-comparison（多模型/版本比较 → 推荐模型）
   │                  │
   │                  ▼
   │            uplift-model-reporting（最终报告）
   │
   └── 拒绝（回归 / 多分类 / 聚类 / 时序 / NLP/CV）──► 给出转化建议
```

## 前置依赖

### 基础环境
参考仓库主[`model-evo/README.md`](../README.md)的前置依赖部分。


### 大数据取数（可选）

`feature-matching` / `classification-model-task-spec` / `classification-model-recommend` 的 **spark 模式**依赖 Spark 3.x + YARN + HDFS 集群、PySpark 与 Kerberos 认证，资源默认值见 [`_modelevo-shared/scripts/spark_defaults.template.yaml`](../_modelevo-shared/scripts/spark_defaults.template.yaml)（复制为 `spark_defaults.yaml` 并填本集群值，不入库）。**无集群时**用 `--mode local_file` 直接读本地 parquet/csv，可跑通除「Spark 取数」外的全部流程。

### 上下游数据前置

| 场景 | 前置数据 |
|---|---|
| 分类建模（local_file） | 一份含 `id + 特征列 + label`（可含日期列）的 parquet/csv |
| 分类建模（spark） | 数仓样本表（含 `user_no/label/pday`）+ 特征宽表 |
| Uplift 建模 | 含 treatment 标记（实验/对照组）与 outcome 的样本文件 |
| 历史模型推荐 | `model-knowledge` 台账 `model_catalog.csv` 中有可检索的历史模型条目 |

## 使用说明

参考仓库[`model-evo/README.md`](../README.md)的**使用说明**部分。

## Session 产物结构

每个建模任务以 `runs/{YYYYMMDD-HHMMSS}-{model_name}/` 组织：

```
runs/20260624-114630-draw_willingness/
├── report.md                              # 项目总报告，各阶段逐步回填
├── task-spec/
│   ├── task-spec.md                       # 4 段式需求规格
│   └── _manifest.json                     # 结构化核心信息（含 routing 溯源字段）
├── data-profile/
│   ├── report.md / report.xlsx            # 样本分析报告
│   ├── _manifest.json
│   ├── _split_manifest.json               # 切分清单
│   └── {model_name}_sample_{YYYYMMDD}.parquet
├── model-recommend/                       # 历史模型推荐结果（local_file 模式下跳过）
├── sample-features/
│   ├── feature-matching/
│   │   ├── sample.parquet                 # 全量样本（id + features + label）
│   │   └── feature-list.csv
│   ├── feature-analysis/
│   │   └── analysis/{stats,iv_table,psi_table}.csv
│   └── splits/{train,test,oot}.parquet    # 三档切分（feature-analysis 产）
├── new-models/                            # 各次 run 的训练产物
│   └── {algo}-v{N}/                       # xgb-v1 / xgb-v1-feat / xgb-v1-tuned ...
│       ├── config/train_config.yaml
│       ├── features/ · model/ · evaluation/
│       ├── predictions/ · explainability/
│       ├── comparison/ · logs/
│       └── _manifest.json
└── model-comparison/                      # N-way 横向对比产物
```

uplift 建模产物按 experiment 组织在 `new-models/{experiment_id}/` 下，由 `_uplift-common` 的 `project_layout` 统一管理 `_flow_manifest.json` / `_experiment_manifest.json` / `_flow_log.jsonl` 等。

## 命名约定

| 类型 | 格式 | 示例 |
|---|---|---|
| Session 目录 | `YYYYMMDD-HHMMSS-{model_name}` | `20260624-114630-draw_willingness` |
| 模型简称 | 全小写英文 + 下划线，`{业务动作}_{预测目标}` | `draw_willingness`、`coupon_response` |
| 需求文档 | `task-spec.md` | 固定名称 |
| 元信息 | `_manifest.json` | 固定名称 |
| 项目报告 | `report.md` | 固定名称 |
| Uplift experiment_id | `exp-{3位数字}-{标识}` | `exp-001-draw-uplift` |

命名前缀规则：仅 classification 专属 skill 加 `classification-` 前缀，uplift 专属加 `uplift-` 前缀，跨流程共享 skill（`model-task-routing`、`feature-matching`、`feature-analysis`、`model-knowledge`）不加前缀。每个 `SKILL.md` 的 `name` 字段必须等于其所在目录名。

## 公共代码

| 位置 | 作用 |
|---|---|
| `model-evo/_modelevo-shared/scripts/config_io.py` | yaml 配置读写 + 必填校验 + 数据安全红线（`load_config` / `validate_common` / `check_sensitive`，命中身份证/手机号即抛错） |
| `model-evo/_modelevo-shared/scripts/fetch_spark.py` | PySpark 集群取数 |
| `model-evo/_modelevo-shared/scripts/gen_fetch_command.py` | spark-submit wrapper 脚本生成 |
| `model-evo/_modelevo-shared/scripts/spark_defaults.template.yaml` | Spark 提交默认资源档模板（复制为 `spark_defaults.yaml` 后填本集群值，不入库） |
| `model-evo/_modelevo-shared/tests/` | 公共代码单元测试 |
| `model-skills/_uplift-common/scripts/_common/` | uplift 管线公共布局与 IO（`project_layout` / `data_loader` / `io` / `run_folder` / `report_language`） |


## 关键约束

- **路由在最前**：任何建模诉求先经 `model-task-routing`，下游不得直接承接未经路由的请求
- **信息透传不丢失**：routing_input JSON 中非 null 字段直接透传，下游不重复提问
- **文件落盘**：需求和分析结果必须保存为文件，不能只留在对话中
- **演化闭环**：evaluation/comparison → 回流 development/training 形成多轮迭代；完成后归档至 `model-knowledge`
- **数据安全红线**：`config_io.check_sensitive` 拦截配置中硬编码的身份证号 / 手机号；`where` / `sample_table` 等字段严禁写入明细个人数据
