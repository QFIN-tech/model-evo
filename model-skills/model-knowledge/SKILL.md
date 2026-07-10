---
name: model-knowledge
description: 模型知识库 skill。沉淀建模过程中的业务领域知识、特征资产、历史模型档案与建模经验教训，供后续建模检索复用。当需要查询业务字段定义、查询历史模型/建模经验、复用既有特征或方案、或在建模完成后归档沉淀知识时使用。
---

# 模型知识库（model-knowledge）

本 skill 是**数据资产库**，不产生 session 产物（内容不落 `runs/`）。所有知识按四个知识域组织在 `assets/` 下，供上下游 skill 按需加载。

## 知识域总览

| 知识域 | 入口文件 | 说明 | 加载时机 |
|---|---|---|---|
| 业务领域知识 | `assets/business-domain-knowledge/business-domain-knowledge.md` | 业务字段语义、客群标签、用户状态、常用分析指标 | 建模任务启动时默认加载 |
| 特征知识 | `assets/feature-knowledge/feature-knowledge.md` | 各业务域特征宽表与特征清单（`feature-list/*.csv`） | `feature-matching` / `feature-analysis` 选特征时 |
| 历史模型知识 | `assets/historical-model-knowledge/historical-model-knowledge.md` | 模型台账 `model_catalog.csv` + 模型报告 `reports/` | `classification-model-recommend` 检索复用时 |
| 建模经验知识 | `assets/modeling-experience-knowledge/modeling-experience-knowledge.md` | 方法论、调参经验、踩坑记录 | training / tuning 阶段参考；建模完成后归档 |

## 目录结构

```text
model-knowledge/
├── SKILL.md
└── assets/
    ├── business-domain-knowledge/
    │   └── business-domain-knowledge.md      # 字段对应关系、客群/状态标签解释
    ├── feature-knowledge/
    │   ├── feature-knowledge.md              # 特征表索引（业务域 → 特征表 → 清单）
    │   └── feature-list/                     # 各特征表的特征清单 csv
    ├── historical-model-knowledge/
    │   ├── historical-model-knowledge.md     # 历史模型检索入口与归档规范
    │   ├── model_catalog.csv                 # 模型台账
    │   └── reports/                          # {model_id}_{模型简称}.md(+.json) 模型报告
    └── modeling-experience-knowledge/
        ├── modeling-experience-knowledge.md  # 入口：分类组织 + 条目模板 + 通用经验（EXP-G）
        ├── classification-experience.md      # 分类模型专属经验（EXP-C）
        └── uplift-experience.md              # uplift 模型专属经验（EXP-U）
```

## 何时使用

- **查询**：需要理解业务字段/客群定义；检索历史模型、既有特征表、调参方案与踩坑记录。
- **复用**：新建模任务匹配相似历史案例，给出可借鉴的特征、算法与参数方案。
- **归档**：当前建模完成（模型发布/对比结论产出）后，沉淀模型档案与经验教训。

## 工作流程

### 1. 检索（新任务启动 / 建模过程中）
1. 默认读 `business-domain-knowledge.md` 理解业务字段。
2. 按任务的业务域在 `feature-knowledge.md` 中定位特征表与特征清单。
3. 按预测目标/客群在 `model_catalog.csv` 中匹配历史模型，`模型报告路径` 非空的进一步读 `reports/` 下报告提取 KS/AUC/PSI 与超参数。
4. 在 `modeling-experience-knowledge.md` 中匹配经验条目：通用经验（EXP-G）+ 按任务类型读 `classification-experience.md`（EXP-C）或 `uplift-experience.md`（EXP-U）。
5. 输出：可复用的特征/模型/调参建议 + 注意事项。

### 2. 归档（建模完成后）
当前归档动作要求手动指定操作或者手动触发。
1. **模型档案**：复制 `reports/_template_model_report.md` 填写，按 `reports/{model_id}_{模型简称}.md` 命名落盘；在 `model_catalog.csv` 追加/更新一行并登记 `模型报告路径`（规范见 `reports/README.md`）。
2. **经验条目**：按任务类型追加到对应文件（通用→`modeling-experience-knowledge.md`、分类→`classification-experience.md`、uplift→`uplift-experience.md`），并在该文件索引表登记（任务背景 → 做法 → 结论 → 教训）。
3. **特征资产**：如新挖掘了可复用特征表，在 `feature-knowledge.md` 登记，并将特征清单 csv 落 `feature-list/`。

## 输入
- 检索意图（业务域、预测目标、客群），或待归档的建模产物与结论（session 的 `report.md`、评估指标、`config.json`）。

## 输出
- 知识条目（业务定义、特征清单、模型档案、经验案例）。
- 针对新任务的复用建议与风险提示。

## 约束
- 本目录为数据资产，**不属于 session**，不要往 `runs/` 搬，也不要把 session 临时产物直接落入本目录。
- 台账 `model_catalog.csv` 为 CSV，字段含逗号时用引号包裹。
- 报告落盘前脱敏：不含用户 ID、手机号、身份证号等明细数据，仅保留聚合统计与指标。

## 关联 skill
- 上游：`classification-model-development` / `classification-model-comparison`（归档来源）
- 下游：`model-task-routing`、`classification-model-task-spec`、`classification-model-recommend`、`feature-matching`、`feature-analysis`（检索复用）
