# 模型台账与工具说明

> 本文件从 `classification-model-recommend/SKILL.md` 第 8 节抽出,包含模型台账 catalog 字段定义、模型报告规范、工具清单、安全约束。SKILL.md 中保留摘要 + 指向本文件的指针。

## 1. 模型列表(catalog)

模型列表文件:`model-knowledge/assets/historical-model-knowledge/model_catalog.csv`(CSV)

| 字段 | 含义 |
|------|------|
| model_id | 模型唯一标识 |
| 业务线 | 所属业务线 |
| 模型中文名 | 模型名称 |
| 预测目标 | 模型预测的目标(如 T+7 是否发生某行为) |
| 正样本定义 | 正样本口径 |
| 模型表 | 模型分数落库的 Hive 表 |
| 算法类型 | 如 xgboost |
| 训练时间窗 | train / oot 区间(可能为空) |
| 训练客群 | 训练样本客群(自由文本;含"全量/全客群/全部"等视为不限客群) |
| 状态 | 可用 / 不可用 |
| 原始文档路径 | 模型原始设计文档链接(可能为空) |
| 模型报告路径 | 完整模型评估报告路径(KS/AUC/PSI/分档等,相对 `model-knowledge/assets/historical-model-knowledge/` 如 `reports/<model_id>_<简称>.md`,可能为空) |
| 负责人 | 模型负责人 |

> 字段为空表示未登记,匹配时该维度按"不限"处理,并在推荐时提示信息缺失。

## 2. 模型报告

每个模型的完整评估报告存放在 `model-knowledge/assets/historical-model-knowledge/reports/`:
- 上传规范见 `model-knowledge/assets/historical-model-knowledge/reports/README.md`,统一模板见 `model-knowledge/assets/historical-model-knowledge/reports/_template_model_report.md`
- 报告命名 `reports/{model_id}_{模型简称}.md`,并在模型列表 `模型报告路径` 列登记
- 推荐输出时若该列非空,引导业务人员查看报告;为空则提示报告待补充

## 3. 工具说明

| 工具 | 作用 |
|------|------|
| `scripts/recall.py` | 规则召回, 从 `model_catalog.csv` 找候选模型 |
| `scripts/_bootstrap.py` | 注入 `model-skills/_modelevo-shared/scripts` 到 sys.path |
| `scripts/fetch_eval_sample.py` | 评估 entry: `--session-dir` 模式, 取数+切分+委托评估一条龙 |
| `scripts/split_sample.py` | 本地 pandas 按 pday 区间切 train/test/oot 三档 |
| `scripts/invoke_evaluation.py` | 评估委托: 三档 parquet → 临时目录 → `eval_single.py` 目录模式一次产 4 份三件套 + `_manifest.json` |
| `scripts/eval_config.example.yaml` | 评估配置模板 |

> 模型推荐阶段(解析需求 → 读台账 → 语义筛选排序)由 Claude 直接完成,不涉及独立脚本。

## 4. 安全

- 脚本仅做分组聚合统计,**不输出任何用户级明细**。
- 严禁在 `eval_config.yaml` 或 `--where` 中硬编码用户ID/手机号/身份证号。

---

> 关联:SKILL.md 第 8 节;数据来源:模型台账 `model-knowledge/assets/historical-model-knowledge/model_catalog.csv`,历史报告 `model-knowledge/assets/historical-model-knowledge/reports/`。
