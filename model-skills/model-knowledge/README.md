# model-knowledge（模型知识库）

本 skill 是 model-skills 全流程的**数据资产库**：沉淀建模过程中的业务领域知识、特征资产、历史模型档案与建模经验教训，供上下游 skill 按需检索复用。它不产生 session 产物（内容不落 `runs/`），所有知识按四个知识域组织在 `assets/` 下。

## 知识域总览

| 知识域 | 入口文件 | 说明 | 加载时机 |
|---|---|---|---|
| 业务领域知识 | `assets/business-domain-knowledge/business-domain-knowledge.md` | 业务字段语义、客群标签、用户状态、常用分析指标 | 建模任务启动时默认加载 |
| 特征知识 | `assets/feature-knowledge/feature-knowledge.md` | 各业务域特征宽表与特征清单（`feature-list/*.csv`） | `feature-matching` / `feature-analysis` 选特征时 |
| 历史模型知识 | `assets/historical-model-knowledge/historical-model-knowledge.md` | 模型台账 `model_catalog.csv` + 模型报告 `reports/` | `classification-model-recommend` 检索复用时 |
| 建模经验知识 | `assets/modeling-experience-knowledge/modeling-experience-knowledge.md` | 方法论、调参经验、踩坑记录（EXP-G / EXP-C / EXP-U） | training / tuning 阶段参考；建模完成后归档 |

## 目录结构

```text
model-knowledge/
├── SKILL.md
├── README.md
└── assets/
    ├── business-domain-knowledge/
    │   ├── business-domain-knowledge.md      # 业务知识路由入口（知识分类 → 触发方式 → 文档）
    │   ├── common-knowledge.md               # 公司公共业务知识（业务概况、核心链路、通用字段）
    │   └── user-operation-knowledge.md       # 业务域专属知识示例（用户运营）
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

## 工作流程

### 检索（新任务启动 / 建模过程中）

1. 默认读 `business-domain-knowledge.md` 理解业务字段，按路由表加载业务域专属知识。
2. 按任务的业务域在 `feature-knowledge.md` 中定位特征表与特征清单。
3. 按预测目标/客群在 `model_catalog.csv` 中匹配历史模型，`模型报告路径` 非空的进一步读 `reports/` 下报告提取 KS/AUC/PSI 与超参数。
4. 在建模经验知识库中匹配经验条目（通用 EXP-G + 按任务类型读 EXP-C / EXP-U）。
5. 输出：可复用的特征/模型/调参建议 + 注意事项。

### 归档（建模完成后，手动触发）

1. **模型档案**：复制 `reports/_template_model_report.md` 填写，按 `reports/{model_id}_{模型简称}.md` 命名落盘；在 `model_catalog.csv` 追加/更新一行并登记 `模型报告路径`（规范见 `reports/README.md`）。
2. **经验条目**：按任务类型追加到对应经验文件，并在该文件索引表登记（任务背景 → 做法 → 结论 → 教训）。
3. **特征资产**：新挖掘的可复用特征表在 `feature-knowledge.md` 登记，特征清单 csv 落 `feature-list/`。

## 约束

- 本目录为数据资产，**不属于 session**：不要往 `runs/` 搬，也不要把 session 临时产物直接落入本目录。
- 报告与知识文件落盘前脱敏：不含用户 ID、手机号、身份证号等明细数据，仅保留聚合统计与指标。

## 关联 skill

- 上游（归档来源）：`classification-model-development` / `classification-model-comparison`
- 下游（检索复用）：`model-task-routing`、`classification-model-task-spec`、`classification-model-recommend`、`feature-matching`、`feature-analysis`

---

## 公司初次接入流程

仓库自带的知识内容均为**示例/占位**（示例业务线、`yx_001` 示例模型、示例特征表等）。一家公司初次接入时，需要把四个知识域的示例内容替换为自己的真实资产，建模流程才能给出有意义的检索与推荐结果。建议按以下顺序进行：

### 第 1 步：填写公司公共业务知识

编辑 `assets/business-domain-knowledge/common-knowledge.md`：

- 公司业务概况：一句话说明公司主营业务与用户对象。
- 核心业务链路：参照示例（流量获取 → 注册 → … → 复借/流失）改写为自己公司的业务阶段，并标注各阶段常见建模目标。
- 常用字段含义：登记全公司通用的字段（字段名、中文含义、类型、备注）。
- 补全文档信息表（维护人、创建时间、版本号）。

### 第 2 步：梳理业务域并配置知识路由

编辑 `assets/business-domain-knowledge/business-domain-knowledge.md` 的路由表：

- 每个业务域一行：知识分类、触发方式（哪些建模任务/模型类型时加载）、文档位置。
- 为每个业务域新建 `{业务域}-knowledge.md`（可参照 `user-operation-knowledge.md` 的结构：业务概况 → 核心链路 → 常用字段）。
- 初次接入可以只建 1 个最先要建模的业务域，其余按「新业务接入流程」后续补充。

### 第 3 步：登记特征资产

编辑 `assets/feature-knowledge/feature-knowledge.md`：

- 每张可复用特征宽表一行：业务域、触发方式、特征表名（库.表）、特征清单路径。
- 为每张特征表导出特征清单 csv（首列 `feature_name`），落 `feature-list/` 目录。

### 第 4 步：初始化模型台账与报告

- 删除/替换 `model_catalog.csv` 中的 `yx_001` 示例行，按台账字段规范导入存量模型（`model_id` 规则：`{业务线缩写}_{三位序号}`）。
- 有完整评估报告的存量模型，按 `reports/README.md` 规范补充报告文件并在台账登记 `模型报告路径`（报告须脱敏）。
- 没有存量模型也可以留空台账（仅保留表头），首次建模归档后自然生长。

### 第 5 步（可选）：沉淀存量建模经验

把团队已有的方法论与踩坑记录，按 `modeling-experience-knowledge.md` 中的条目模板整理为 EXP-G / EXP-C / EXP-U 条目，并在对应索引表登记。

### 第 6 步：验证

发起一次真实建模任务（走 `model-task-routing` 入口），确认：业务字段能被正确解读、特征表能被 `feature-matching` 定位、`classification-model-recommend` 能从台账筛出候选模型。

### 公司初次接入 Checklist

| # | 项目 | 文件 | 完成 |
|---|---|---|---|
| 1 | 公司业务概况、核心业务链路已填写 | `common-knowledge.md` | ☐ |
| 2 | 公司通用字段含义已登记 | `common-knowledge.md` | ☐ |
| 3 | 业务域路由表已配置，至少 1 个业务域知识文档已建 | `business-domain-knowledge.md` + `{业务域}-knowledge.md` | ☐ |
| 4 | 至少 1 张特征宽表已登记，特征清单 csv 已落盘 | `feature-knowledge.md` + `feature-list/*.csv` | ☐ |
| 5 | 台账示例行已清除，存量模型已导入（或确认留空） | `model_catalog.csv` | ☐ |
| 6 | 存量模型报告已按规范落盘并在台账登记路径 | `reports/` | ☐ |
| 7 | 所有落盘内容已脱敏（无用户 ID / 手机号 / 身份证号等明细） | 全部 | ☐ |
| 8 | 已跑通一次检索验证（字段解读 / 特征定位 / 模型筛选） | — | ☐ |

---

## 新业务接入流程

公司已完成初次接入后，新增一个业务域（如从"用户运营"扩展到"广告投放"）时，只需增量补充以下内容：

### 第 1 步：新建业务域知识文档

- 在 `assets/business-domain-knowledge/` 下新建 `{业务域}-knowledge.md`，参照 `user-operation-knowledge.md` 的结构填写：文档信息 → 业务概况 → 核心业务链路 → 常用字段含义。
- 在 `business-domain-knowledge.md` 路由表追加一行，写明触发方式（该业务域涉及哪些建模任务/模型类型）。

### 第 2 步：登记新业务特征资产

- 在 `feature-knowledge.md` 追加新业务域的特征宽表行。
- 导出对应特征清单 csv 落 `feature-list/`。

### 第 3 步：导入新业务存量模型（如有）

- 确定新业务线的 `model_id` 缩写前缀（如 广告投放 → `gg`），在 `model_catalog.csv` 导入该业务线存量模型。
- 有报告的按 `reports/README.md` 规范落盘并登记路径。

### 第 4 步：验证

用新业务域的一个建模需求跑一次检索，确认路由表能触发新知识文档、特征表可被定位、台账按新业务线筛选正常。

### 新业务接入 Checklist

| # | 项目 | 文件 | 完成 |
|---|---|---|---|
| 1 | 新业务域知识文档已建（概况 / 链路 / 字段） | `{业务域}-knowledge.md` | ☐ |
| 2 | 路由表已追加该业务域及触发方式 | `business-domain-knowledge.md` | ☐ |
| 3 | 新业务特征宽表与特征清单已登记 | `feature-knowledge.md` + `feature-list/*.csv` | ☐ |
| 4 | 新业务线 model_id 前缀已确定，存量模型已导入（或确认无） | `model_catalog.csv` | ☐ |
| 5 | 新增内容已脱敏 | 全部 | ☐ |
| 6 | 已用新业务需求跑通一次检索验证 | — | ☐ |
