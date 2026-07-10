# 该 Demo 任务使用说明

本示例用于二分类模型建模，仅支持 CSV 数据。

示例数据：`data/sample.csv`

## 提示词示例

| 编号 | 提示词 |
|---|-----|
| 1 | 基于本地路径下 data/sample.csv 进行二分类建模 |
| 2 | 使用本目录下的 data/sample.csv 建模，目标列为 default payment next month |
| 3 | 基于 /home/data/sample.csv 建一个信用卡违约风险二分类模型 |
| 4 | 基于 data/sample.csv 这份样本，按 dt 划分训练与 OOT，针对客户违约建模 |
| 5 | 使用data/sample.csv样本进行建模，全自动驾驶模式 |

## 推荐建模参数

| 参数 | 推荐值 |
|---|---|
| label_column | `default payment next month` |
| label_positive_value | `1` |
| id_column | `ID` |
| oot_column | `dt` |
| excluded_columns | `ID` |


## 数据来源与授权

本示例数据来自 UCI Machine Learning Repository 公开数据集 **Default of Credit Card Clients**。

- 数据集主页：<https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- 原始数据发布方：I-Cheng Yeh
- DOI：<https://doi.org/10.24432/C55S3H>
- 数据集创建年份：2009；捐赠年份：2016

### License

本数据集采用 **Creative Commons Attribution 4.0 International (CC BY 4.0)** 协议授权，允许任何目的（含商业用途）的共享与改编，前提是给出适当署名。完整协议文本见 <https://creativecommons.org/licenses/by/4.0/legalcode>。

### 修改声明

相对原始数据，本仓库做了一处修改：

- 追加 `dt` 列：取值范围 `20260601` ~ `20260630`，由固定随机种子（seed=42）生成，用于演示 OOT 切分；其余字段保持原始数据集口径。

除上述修改外，本仓库未对原始数据做任何其他变更。

### 学术引用

如在本示例基础上产出研究成果，建议同时引用原始数据集与论文：

- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.
- Yeh, I. C. (2016). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C55S3H.

### 免责声明

本仓库以"现状"提供数据，不附带任何明示或暗示的担保。如需完整或权威版本，请从 UCI 官方页面下载。

## 样本字段说明

| 字段 | 含义 |
|---|---|
| dt | 样本日期字段（`yyyymmdd`），本示例追加，可作为 OOT 切分依据 |
| ID | 客户唯一标识，不作为特征进入模型 |
| LIMIT_BAL | 信用额度（NT 美元） |
| SEX | 性别（1=男，2=女） |
| EDUCATION | 教育程度（1=研究生，2=本科，3=高中，4=其他） |
| MARRIAGE | 婚姻状况（1=已婚，2=未婚，3=其他） |
| AGE | 年龄 |
| PAY_0 | 9 月还款状态（-1=按时还，1=逾期 1 个月，…，9=逾期 9 个月+） |
| PAY_2 | 8 月还款状态 |
| PAY_3 | 7 月还款状态 |
| PAY_4 | 6 月还款状态 |
| PAY_5 | 5 月还款状态 |
| PAY_6 | 4 月还款状态 |
| BILL_AMT1 | 9 月账单金额（NT 美元） |
| BILL_AMT2 | 8 月账单金额 |
| BILL_AMT3 | 7 月账单金额 |
| BILL_AMT4 | 6 月账单金额 |
| BILL_AMT5 | 5 月账单金额 |
| BILL_AMT6 | 4 月账单金额 |
| PAY_AMT1 | 9 月还款金额（NT 美元） |
| PAY_AMT2 | 8 月还款金额 |
| PAY_AMT3 | 7 月还款金额 |
| PAY_AMT4 | 6 月还款金额 |
| PAY_AMT5 | 5 月还款金额 |
| PAY_AMT6 | 4 月还款金额 |
| default payment next month | 二分类目标字段（`0`/`1`），作为 label_column |

