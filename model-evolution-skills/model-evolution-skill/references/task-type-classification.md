# 分类任务细则(binary_classification;渐进披露)

`model_type: binary_classification`(默认路径)时必读。铁律、主循环、
编排纪律见 [SKILL.md](../SKILL.md),本文档只讲分类特有的口径。

## score 语义

- `validation_score.parquet` 的 score 列 = **正样本概率**
  (`predict_proba(...)[:, 1]` / sigmoid 输出);
- Trial 产物、evaluator、leaderboard、stop 判定全链路按此口径。

## 指标白名单

- `primary_metric`/`secondary_metrics` 只能 **auc / ks / logloss**
  (validate 拦截分类×回归/uplift 指标错配);
- 方向自动识别:auc/ks 越大越好,logloss 越小越好。

## TaskSpec 要点

- binary 必须有 `target.positive_definition`(正样本的业务定义);
- 字段语义见 [task-and-data.md](task-and-data.md)；完整可执行模板以
  `modelevo.py init` / `config-template` 的输出为准。

## profile 特有诊断

- `feature-quality.csv` 的 IV 等级刻度:useless/weak/medium/strong/
  suspicious;suspicious(IV≥0.5)进反泄漏复核清单;
- 单变量 AUC>0.85 → leakage warning;
- `feature-woe.csv` 有 WOE 分箱明细(仅 binary/uplift 产出)。

## baseline 与 Trial 模板

- baseline(gen-00)默认 LightGBM(+XGBoost)分类器,不做 HPO;
- 模板输出 predict_proba 口径的 score;
- Trial Lint 无分类特有规则(规则 4/5 仅适用于 regression/uplift)。

## evaluator 与分段排序性口径

- noise-aware improvement gate、切片、时序回归按默认口径
  (见 [evolution-and-evaluation.md](evolution-and-evaluation.md));
- 分段排序性(segment_evaluation):段内**正样本率**;spearman 适用;
- 重采样按 validation 行。

## 分桶报表(segment-report)列集

按 score 分位分桶、降序(桶 1 = 最高分段),列集:

分桶人数、样本占比、正样本数、正样本占比、lift、累计样本数、
累计正样本数、累计准确率、累计 lift、累计召回率、KS
(逐桶累计正样本率 − 累计负样本率)。

## 升级到人

无分类特有项;通用场景(泄漏 critical、预算不可达等)见 SKILL.md
「升级到用户」。
