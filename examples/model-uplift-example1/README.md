# Criteo Uplift Demo 使用说明

本示例基于 **Criteo Uplift Prediction Dataset v2.1** 广告实验数据。数据记录了广告处理组、对照组及用户访问和转化结果，可用于评估广告干预带来的增量效果。本仓库内置 20k 采样，适合快速跑通流程；如需获得相对稳定的效果验证结果，可按下文说明下载官方完整数据集。本示例仅支持 CSV 数据。

## 提示词示例

> **使用提示：** 请根据数据文件的实际存放位置和文件名，调整提示词中的路径。

| 编号  | 提示词                                                                                                                                                                                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 基于本地路径下 `data/criteo-uplift-v2.1.sample_20k.csv` 进行 uplift 建模                                                                                                                         |
| 2   | 使用本目录下的 `data/criteo-uplift-v2.1.sample_20k.csv` 建模，结果列为 `visit`，实验分组列为 `treatment`                                                                                                   |
| 3   | 基于 `/home/data/criteo-uplift-v2.1.sample_20k.csv` 建一个广告干预对用户访问增量效果的 uplift 模型                                                                                                         |
| 4   | 基于 `data/criteo-uplift-v2.1.sample_20k.csv`，评估广告处理对用户访问的增量效果；`visit` 为二分类结果，`treatment=1` 为处理组、`treatment=0` 为对照组，排除 `exposure` 字段                                                    |
| 5   | 使用 `data/criteo-uplift-v2.1.sample_20k.csv` 完成 uplift 建模全流程，包括样本准备、同质性检查、特征质量分析、S-Learner 和 T-Learner 训练、模型对比及最终报告；结果列为 `visit`，处理列为 `treatment`，处理组值为 `1`，对照组值为 `0`，排除 `exposure` 字段 |
| 6   | 使用 `data/criteo-research-uplift-v2.1.csv` 完成 uplift 建模全流程，结果列为 `visit`，处理列为 `treatment`，处理组值为 `1`，对照组值为 `0`，排除 `exposure` 字段                                                          |

## 数据使用方式

本示例提供两种使用方式：

| 使用方式 | 数据 | 适用目的 |
|---|---|---|
| 快速体验 | 仓库内置的 20k 极小采样 | 验证流程、交互和产物能否完整跑通 |
| 效果验证 | Criteo 官方 v2.1 完整数据集 | 观察相对稳定的训练、评估和模型比较结果 |

## 快速体验：使用仓库内置样本

仓库内置样本：

`data/criteo-uplift-v2.1.sample_20k.csv`

该文件采样自 Criteo Uplift Prediction Dataset v2.1，共 20,000 条记录，其中处理组 16,968 条、对照组 3,032 条，`visit=1` 共有 901 条，`conversion=1` 仅有 59 条。

> **重要提示**
>
> 该文件是用于快速体验流程的极小规模采样。经过 Train/Eval/OOT 切分后，各数据分区中的对照组和正样本会进一步减少，因此 AUUC、Qini、uplift 分箱及模型排序可能出现较大波动。
>
> 使用该样本成功跑通流程，不代表模型效果具有业务参考价值。请勿使用该样本评判算法优劣、比较模型效果或复现论文指标。

快速体验模式的成功标准是完整生成样本准备、同质性检查、特征质量分析、模型训练、模型比较和最终报告等流程产物，而不是达到特定的 AUUC 或 Qini 指标。

## 效果验证：使用官方完整数据集

如需获得相对稳定的训练和评估结果，请使用 Criteo 官方发布的无偏 v2.1 完整数据集：

- [Criteo 官方数据说明及许可](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)
- [Criteo Uplift v2.1 官方下载](https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz)

官方 v2.1 数据约包含 13,979,592 条记录，压缩文件约 297 MB。考虑到完整数据体积较大，本仓库仅提供用于流程体验的 20k 采样；完整数据请从 Criteo 官方渠道下载，并按照 CC BY-NC-SA 4.0 许可使用。

Linux/macOS 下载并解压：

```bash
cd examples/model-uplift-example1
mkdir -p data

curl -L --fail \
  https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz \
  -o data/criteo-research-uplift-v2.1.csv.gz

gzip -dk data/criteo-research-uplift-v2.1.csv.gz
```

下载完成后，请确认以下文件存在：

`examples/model-uplift-example1/data/criteo-research-uplift-v2.1.csv`


## 数据来源与授权

本示例数据来源于 Criteo AI Lab 发布的 **Criteo Uplift Prediction Dataset**。Criteo 官方页面说明该数据采用 CC BY-NC-SA 4.0 许可，包含署名、非商业用途和相同方式共享等要求。

使用仓库内置采样或下载完整数据前，请阅读并遵守 [Criteo 官方数据说明及许可](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)。本项目仅提供建模流程示例，不对该数据集提供任何担保。

## 样本字段说明

| 字段 | 含义 |
|---|---|
| `f0`-`f11` | 可用于建模的匿名特征字段 |
| `treatment` | 处理组字段，`1` 表示广告处理组，`0` 表示对照组 |
| `visit` | 二分类访问目标字段，推荐作为 `outcome_column` |
| `conversion` | 二分类转化目标字段，正样本率较低，可按任务需要指定 |
| `exposure` | 是否实际曝光的字段，通常不作为协变量进入模型 |

## 推荐建模参数

| 参数                 | 推荐值         |
| ------------------ | ----------- |
| `outcome_column`   | `visit`     |
| `outcome_type`     | `binary`    |
| `treatment_column` | `treatment` |
| `treatment_value`  | `1`         |
| `control_value`    | `0`         |
| `excluded_columns` | `exposure`  |
