# 历史模型知识库

沉淀已上线/已归档模型的档案，供 `classification-model-recommend` 等下游 skill 检索复用。

## 组成

| 资产 | 路径 | 说明 |
|---|---|---|
| 模型台账 | `model_catalog.csv` | 一行一个模型 |
| 模型报告 | `reports/{model_id}_{模型简称}.md`（可附同名 `.json`） | 完整评估报告：KS/AUC/PSI、分档分布、实验设置 |
| 报告规范 | `reports/README.md` | 命名规范与上传步骤 |
| 报告模板 | `reports/_template_model_report.md` | 归档时复制填写 |

## 台账字段

| 列 | 说明 |
|---|---|
| model_id | 唯一 ID，`{业务线缩写}_{三位序号}`，如 `yx_001` |
| 业务线 | 如 营销线 |
| 模型中文名 | |
| 预测目标 / 正样本定义 | 检索匹配的主要依据 |
| 模型表 | 分数落库 Hive 表 |
| 算法类型 | 如 xgboost |
| 训练时间窗 | `train：yyyymmdd~yyyymmdd｜oot：yyyymmdd~yyyymmdd` |
| 训练客群 | 自由文本，如 `活跃户` |
| 状态 | 可用 / 不可用 |
| 原始文档路径 / 模型报告路径 | 报告路径相对本目录，如 `reports/yx_001_模型简称.md` |
| 负责人 | - |

## 检索流程

1. 按新任务的**预测目标 / 正样本定义 / 训练客群**在台账中匹配候选（只看状态=可用）。
2. `模型报告路径` 非空的，读报告提取历史 KS/AUC/PSI、样本时间窗与核心超参数。
3. 输出候选模型 + 复用建议（直接复用打分表 / 复用特征与参数重训 / 仅作 baseline 参考）。

## 归档流程（新模型入库）

1. 复制 `reports/_template_model_report.md` 填写，按 `reports/{model_id}_{模型简称}.md` 命名保存。
2. 在 `model_catalog.csv` 追加一行（model_id 顺延），登记 `模型报告路径`。
3. 报告须包含：KS/AUC/PSI、分档分布（默认 10 档）、训练时间窗、正负样本比、核心超参数；PSI > 0.1 的特征标注 `[PSI_WARN]`。

## 当前已归档模型索引

| model_id | 模型 | 预测目标 | 报告 |
|---|---|---|---|
| yx_001 | 示例模型 | 示例预测目标 | `reports/yx_001_example_model.md` |

> 本索引为快捷视图，**以 `model_catalog.csv` 为准**；台账更新后请同步维护本表。上方为占位示例，实际归档时替换为真实模型信息。
