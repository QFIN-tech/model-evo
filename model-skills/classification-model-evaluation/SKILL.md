---
name: classification-model-evaluation
description: 单一分类模型标准化评估——输入含打分列和标签列的CSV文件，自动计算AUC/KS/准确率/十分桶排序性/业务指标均值，输出JSON+MD+XLSX。支持单档评估和合并评估两种模式。当用户说"模型评估""跑AUC""跑KS""分桶排序性""模型报告""打分评估""单模型评估""帮我评估这个模型"时使用。
---

# 单一分类模型标准化评估

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| 模型打分 CSV | ✅ | 上游训练/取数 | 上游已切分好的 train / val / oot CSV 文件，含打分列 + 二分类标签列 |

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录（即本文件所在目录），执行时替换为实际绝对路径，不要依赖当前工作目录。

**目录模式**（eval_single.py `--input-dir`），一次调用自动完成所有文件评估 + all 合并：

```bash
python <skill_dir>/scripts/eval_single.py \
  --input-dir <data_dir> \
  --score-col score \
  --name "模型名" \
  -o <output_dir>/
```

目录下每个 CSV/parquet 各评估一份（`version = 文件名 stem`），再纵向拼接所有文件行重算指标得出 all 合并一份（`version = all`）。合并是真正拼接原始数据重算，非 JSON 加权近似。

## 3. 参数说明

### eval_single.py

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--input-dir` | ✅ | - | 输入目录（目录下所有 CSV / parquet 各评估一份 + all 合并一份） |
| `--score-col` | ✅ | - | 打分列名 |
| `--name` | 否 | `模型名` | 模型名称，用于输出文件名与报告标题 |
| `--label-col` | 否 | `label` | 标签列名 |
| `--model-type` | 否 | `xgboost` | 模型类型 |
| `--segment-cols` | 否 | `None` | 分群列名，逗号分隔；`None` 表示不分群 |
| `--metric-cols` | 否 | `None` | 业务指标列名，逗号分隔；`None` 表示不汇总 |
| `--hyperparams` | 否 | `None` | 超参 JSON 字符串 |
| `--n-bins` | 否 | `10` | 分桶数 |
| `-o` / `--output-dir` | 否 | 输入同目录 | 输出目录 |

## 4. 输出产物

对目录下每个 CSV/parquet 各产 1 份三件套（`version = 文件名 stem`），再产 1 份 all 合并三件套（`version = all`，纵向拼接所有文件行重算指标）。共 `(N+1) × 3` 个文件，N 为目录下表文件数。

```text
<output_dir>/
├── {name}_train_eval.{json,md,xlsx}
├── {name}_test_eval.{json,md,xlsx}
├── {name}_oot_eval.{json,md,xlsx}
└── {name}_all_eval.{json,md,xlsx}    # 全量样本整体一份（纵向拼接所有文件行重算）
```

### 4.1 产物内容

| 产物 | 说明 |
|---|---|
| `*_eval.json` | 标准化评估结果，`classification-model-comparison` 的唯一合法输入。关键字段：`model_meta`（身份信息）、`metric_by_segment`（全量 + 各客群的 AUC/KS/accuracy/precision/recall/biz_avg）、`performance.score_buckets`（全量 + 各客群十分桶明细） |
| `*_eval.md` | 人类可读报告：标题摘要行 + 按客群指标表 + 全量分桶表（含 label率/lift/召回率/累计召回/业务指标） + 模型信息 |
| `*_eval.xlsx` | 三 Sheet：`1-AUC_KS_by客群`、`2-分桶排序性by客群`、`3-全量分桶`（明细） |

### 4.2 评估口径

AUC 用 `sklearn.roc_auc_score`；KS = 按打分降序累计正负样本占比曲线最大差值；准确率/精确率/召回率 以 0.5 为阈值；分桶为 10 档等频降序（decile 10 = 最高分组）。`all` 版本 = train+test+oot 合并样本整体一份评估。

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `classification-model-training` | 训练产出打分数据，各 run 的三档评估由其管线调本脚本落 `new-models/{run}/evaluation/` |
| 上游 | `classification-model-development` | 编排调用 |
| 下游 | `classification-model-comparison` | 消费 `*_oot_eval.json` 和 `*_all_eval.json` 做 N-way 对比（本 skill 为其强制前置） |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| 职责边界 | 只做单一模型评估，不计算 PSI、不做多模型对比（稳定性与横向对比由 `classification-model-comparison` 负责） |
| 不调用下游 | 本 skill 不被 evaluation 直接调用 comparison，由 `classification-model-development` 统一编排 |
| 评估后必检 | 评估完成后复核指标合理性（AUC/KS 区间、分桶单调性、各客群趋势一致），命中异常须明确标注 |

## 7. 异常处理

| 条件 | 处理方式 |
|---|---|
| 找不到标签列 | 脚本报错退出并列出可用列，检查 `--label-col` |
| AUC < 0.5 | 标注打分方向可能反了，取反后重跑 |
| AUC > 0.95 | 标注疑似标签泄露，必须解释后才可采信 |
| 客群 N ≤ 50 | KS 标空并警告 |
| 总样本 < 200 | 严重警告"评估结果仅供参考" |
| 目录下无 CSV/parquet 文件 | 报错退出 |
