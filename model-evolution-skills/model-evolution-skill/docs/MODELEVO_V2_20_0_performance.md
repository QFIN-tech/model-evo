# v2.20.0 开源数据集效果对比

> 一句话: 在本次五个公开数据集、固定切分的评测中，ModelEvo 获得**单模型组最佳平均名次，并在包含集成的比较中位居第二**；其在多表历史任务上的表现尤其突出。相对于“让 LLM 直接写代码”，结构化的建模与搜索流程值得继续投入，对模型效果提升方面具有不错的潜力。
---

## 榜单说明

### 数据集说明

5 个数据集覆盖不同的"形状"(类别型 / 高基数 / 极度不平衡 / 时间切分回归 / 多表历史),
切分口径固定为 train / validation / shadow_test，最终对比都在shadow_test上进行。(时间切分的数据集按时间窗口,不是随机)。

| 数据集 | 来源 | 原始行数 | 特征数 | 目标 / 指标 | 切分 | shadow_test 行数 | 其中正例 | train / validation 行数 | 档位 |
|---|---|---|---|---|---|---|---|---|---|
| adult | OpenML 1590 | 48,842 | 15 | `class` / auc | 分层随机 | 4,885 | 1,169(23.9%) | 29,305 / 7,326 | full |
| amazon_employee_access | OpenML 4135 | 32,769 | 10 | `target` / auc | 分层随机 | 3,276 | 189(5.8%) | 19,661 / 4,916 | full |
| creditcard | OpenML 1597 | 284,807 | 31 | `Class` / auc | 分层随机 | 28,481 | **49(0.17%)** | 170,884 / 42,721 | full |
| house_sales | OpenML 42092 | 21,613 | 21 | `price`(log1p)/ rmse | **时间切分** | 1,875 | — | 14,633 / 2,228 | full |
| home_credit | Kaggle HDFS 副本 | 307,511 | 主表 + 5 列历史序列 | `TARGET` / auc | 分层随机 | 30,751 | 2,482(8.1%) | 184,507 / 46,126 | large |

**任务性质**

- **adult**:人口普查收入二分类,含类别型与缺失值;基准较饱和(多个系统挤在 0.92 附近)。
- **amazon_employee_access**:员工访问审批,类别高基数 + 5.8% 正例;样本量小,单次 AUC 抖动大。
- **creditcard**:信用卡欺诈,正例率 0.17%——**shadow_test 只有 49 个正例**。这一集的 AUC
  置信区间天然很宽,±0.01 以内的差异基本不可分辨。
- **house_sales**:房价回归,目标做 `log1p`,所以 rmse 0.17 是**对数尺度**上的值;
  按时间切分(训练 2014-05~12、验证 2015-01~02、shadow 2015-03)。
- **home_credit**:违约预测,含申请人历史明细序列;是唯一 `large` 档(更大预算)。

### 榜单候选说明(3 个 LLM 系统 + 4 个 AutoML 参照,共 9 个条目)

* 针对同一标准测试集数据面评分,所有系统取各自**在该数据集上的最佳可用结果**。
* "机制"列说明该系统属于哪一类:LLM 驱动的框架 / 端到端代码搜索 / 自动模型搜索。
* 区分单模型/集成。raw-agent 与 AIDE 各有自由(多模型)与单模型两批,分别条目化。
* 在本次对比中ModelEvo关闭了经验系统，仅做冷启动建模能力对比。

| 系统 | 机制 | 单模型/集成 |
|---|---|---|
| ModelEvo | LLM 编排 + 确定性 runtimes + 裁决门 | 单模型 |
| raw-agent 集成 | LLM 直接写代码,单链迭代 | 多模型(自由) |
| raw-agent 单模型 | LLM 直接写代码,单链迭代 | 单模型 |
| AIDE 集成 | LLM 树搜索 + 执行反馈 | 多模型(自由) |
| AIDE 单模型 | LLM 树搜索 + 执行反馈 | 单模型 |
| AutoGluon 集成 | 自动模型搜索 + 多层堆叠 | 多模型(自由) |
| AutoGluon 单模型 | 同上,取单模型 | 单模型 |
| FLAML | 自动超参/算法搜索 | 单模型 |
| TabPFN v2 | 预训练小样本模型 | 单模型 |

### 口径说明(读表前必看)

1. **配对口径**: **正 = 该参与者更好**。
2. **胜负判定**:CI 全正 = better,全负 = worse,跨 0 = tie。名次与显著性是两个口径:
   名次只是点值排序,可能有 4 个系统"打平"却排出 1/2/3/4。
3. **单模型约束**:单一估计器实例、单一算法/单一超参。禁止多模型加权/平均/投票/堆叠、
   同族多 seed/多配置/多折平均、训练多个候选再挑一个;早停允许。静态扫描
   (`strategy_audit`)对 10 份单模型方案(2 对照 × 5 集)零命中,全部合规。
4. **补位标注**:`[fill:base]` 表示该参照是事后补齐的(creditcard 的 strong LGBM/混合、
   creditcard 与 home_credit 的 TabPFN),口径与原参照不完全一致,见附录 C。

---

## 榜单

### 单模型单独榜单

只放"只训练一个估计器"的条目。

| 数据集(指标, n) | ModelEvo | raw-agent 单模型 | AIDE 单模型 | AutoGluon 单模型 | FLAML | TabPFN v2 |
|---|---|---|---|---|---|---|
| home_credit(auc, 30751) | **0.79315** | 0.76773 | 0.76813 | 0.77318 | 0.78634 | 0.73524 |
| creditcard(auc, 28481) | **0.99601** | 0.99423 | 0.98654 | 0.92278 | 0.96664 | 0.98399 |
| amazon(auc, 3276) | **0.87980** | 0.86044 | 0.86979 | 0.82792 | 0.81435 | 0.76180 |
| house_sales(rmse, 1875) | 0.17804 | 0.18264 | 0.17610 | 0.17836 | 0.17844 | **0.17026** |
| adult(auc, 4885) | 0.92423 | **0.92610** | 0.92305 | 0.90409 | 0.91991 | 0.90881 |

**名次(6 个单模型条目之间比较,N=6)**

| 系统 | 平均名次 | 归一化 | 各集名次(按上表数据集顺序) |
|---|---|---|---|
| **ModelEvo** | **1.60** | **0.120** | 1, 1, 1, 3, 2 |
| AIDE 单模型 | 2.80 | 0.360 | 4, 3, 2, 2, 3 |
| raw-agent 单模型 | 3.40 | 0.480 | 5, 2, 3, 6, 1 |
| FLAML | 4.20 | 0.640 | 2, 5, 5, 5, 4 |
| TabPFN v2 | 4.40 | 0.680 | 6, 4, 6, 1, 5 |
| AutoGluon 单模型 | 4.60 | 0.720 | 3, 6, 4, 4, 6 |

### 单模型结论

* **纯单模型对纯单模型，ModelEvo框架第一：** 整体表现最佳，零线覆盖多个数据集而且优势较为稳定。
  * ModelEvo 在 6 条目单模型池里平均名次 1.60,
   拿下 home_credit / creditcard / amazon 三集冠军,adult 第二、house_sales 第三;
   两个 LLM 对照的单模型版本都排在它后面(AIDE 单模型 2.80、raw-agent 单模型 3.40)。
  * 对 AIDE 单模型：4/5 个任务点值更优。对 raw-agent 单模型：4/5 个任务点值更优。对 FLAML、AutoGluon 单模型：均为 5/5。
* **AIDE 是最接近的竞争者：** 平均名次第二，五个数据集均进入前四
* **LLM系统整体优于传统AutoML系统：** ModelEvo/AIDE/raw-agent > FLAML/AutoGluon ≈ TabPFN V2。其中，TabPFN v2 虽然整体排名靠后，但在 house_sales 上取得第一，显示出特定任务上的竞争力。


### 单模型与集成统一榜单

同一批样本上,把 raw-agent / AIDE / AutoGluon 的自由(多模型)与单模型两种策略拆成独立条目,
与 ModelEvo、AutoML 参照一起排名(9 条目)。

| 数据集(指标, n) | ModelEvo | raw-agent 集成 | raw-agent 单模型 | AIDE 集成 | AIDE 单模型 | AutoGluon 集成 | AutoGluon 单模型 | FLAML | TabPFN v2 |
|---|---|---|---|---|---|---|---|---|---|
| home_credit(auc, 30751) | **0.79315** | 0.76917 | 0.76773 | 0.77112 | 0.76813 | 0.77674 | 0.77318 | 0.78634 | 0.73524 |
| creditcard(auc, 28481) | 0.99601 | 0.99470 | 0.99423 | **0.99606** | 0.98654 | 0.99018 | 0.92278 | 0.96664 | 0.98399 |
| amazon(auc, 3276) | 0.87980 | 0.86571 | 0.86044 | **0.88164** | 0.86979 | 0.83938 | 0.82792 | 0.81435 | 0.76180 |
| house_sales(rmse, 1875) | 0.17804 | **0.16804** | 0.18264 | 0.17111 | 0.17610 | 0.17580 | 0.17836 | 0.17844 | 0.17026 |
| adult(auc, 4885) | 0.92423 | 0.92448 | **0.92610** | 0.92494 | 0.92305 | 0.91190 | 0.90409 | 0.91991 | 0.90881 |

| 系统 | 多模型/单模型 | 平均名次 | 归一化 | 各集名次 |
|---|---|---|---|---|
| AIDE 集成 | **多模型** | **2.40** | **0.175** | 5, 1, 1, 3, 2 |
| ModelEvo | **单模型** | **3.00** | **0.250** | 1, 2, 2, 6, 4 |
| raw-agent 集成 | 多模型 | 3.40 | 0.300 | 6, 3, 4, 1, 3 |
| AutoGluon 集成 | 多模型 | 5.00 | 0.500 | 3, 5, 6, 4, 7 |
| AIDE 单模型 | 单模型 | 5.20 | 0.525 | 7, 6, 3, 5, 5 |
| raw-agent 单模型 | 单模型 | 5.40 | 0.550 | 8, 4, 5, 9, 1 |
| FLAML | 单模型 | 6.40 | 0.675 | 2, 8, 8, 8, 6 |
| TabPFN v2 | 单模型 | 7.00 | 0.750 | 9, 7, 9, 2, 8 |
| AutoGluon 单模型 | 单模型 | 7.20 | 0.775 | 4, 9, 7, 7, 9 |

### 统一榜单结论

* **统一榜单中，AIDE 集成第一，ModelEvo 以单模型位居第二:** AIDE 集成平均名次为 2.40，在 creditcard、amazon 上取得第一；ModelEvo 平均名次为 3.00，领先 raw-agent 集成（3.40）和 AutoGluon 集成（5.00），说明其单模型在本次评测中具备与集成方案竞争的能力。
（说明：ModelEvo考虑到真实业务场景，默认不采纳模型集成方案。）
* **ModelEvo 在 home_credit明显领先其他 LLM 条目:** 因为home_credit 包含多表历史序列，而ModelEvo的优势可能更多来自把复杂原始信息转成有效特征，而不仅是调参。
* **LLM 系统包揽总榜前三，其他方法仍有局部优势:** FLAML 在 home_credit 上第二，TabPFN v2 在 house_sales 上第二，体现出特定任务上的竞争力。


---

## 附录 A:对照基线的完整输入

两批(自由 / 单模型)只差"单模型约束"段落。

### A.1 raw-agent System prompt

~~~text
你是一位资深机器学习工程师。你要为一个表格任务写出**一个可执行的
predict.py**。

约束:
- 只能读工作目录里的 train.parquet / train_labels.parquet / validation.parquet /
  validation_labels.parquet,以及你自己生成的文件。数据路径在 data_dir.txt 里。
- 目标列名固定为 `target`;行序必须与输入 parquet 一致。
- predict.py 必须定义:
    def fit(train: pd.DataFrame, y: np.ndarray) -> object   # 返回模型对象
    def predict(model, frame: pd.DataFrame) -> np.ndarray   # 返回与 frame 行数一致的预测
- 允许 import 本环境已装的库(numpy/pandas/scikit-learn/xgboost/lightgbm/catboost/torch)。
- 不要访问网络,不要尝试读取除上述之外的数据文件。
- {env}
{rules}
- 每次回答输出**完整的 predict.py 全文**,放在一个 ```python 代码块里,不要输出别的内容。
~~~

`{env}`(有空闲 GPU 时):

```text
本机有 GPU 可用(CUDA_VISIBLE_DEVICES 已设置好,直接用 `device='cuda'` / xgboost 的
`device='cuda'` / lightgbm 的 `device_type='gpu'` 即可,不要自己指定物理卡号)。
训练线程数上限 24,不要把 n_jobs / nthread 设为 -1。
```

`{env}`(无空闲 GPU 时,本轮新增):

```text
本机当前没有空闲的 GPU,全部用 CPU(不要写 task_type='GPU' / device='cuda',
也不要尝试回退到 GPU)。训练线程数上限 24,不要把 n_jobs / nthread 设为 -1。
```

`{rules}`(单模型批次):

```text
- **本次是单模型对照**:predict.py 的 fit() 里只能训练**一个**估计器实例
  (一种算法、一套超参),predict() 直接用它输出预测。禁止:
    * 多个模型的加权/平均/投票/堆叠(包括同族多 seed、多配置、多折平均);
    * Cross-Validation 平均、OOF 融合、bagging 包装器;
    * 先训练多个候选再挑一个(挑选本身也是多次拟合)。
  早停是允许的:可以在 train 内部留一个尾部验证集做早停,但最终交付仍是一个模型。
  评估你的方案时,用**单次留出**(例如从 train 里留最后 20% 或随机 20%)算一次分数,
  不要做 k 折。
```

User prompt(每轮):

```text
任务:{desc}

数据:{shape}
列:{cols}

已发现的数据字典(若有):
{dict}

本轮是第 {rnd}/{total} 轮。{prev}
请输出完整的 predict.py。
```

### A.2 AIDE 任务说明

```text
# 任务

{task_name}:预测 `target` 列。

- 数据:`train.csv`(含 `target` 列)。`test.csv` 只有特征、没有标签;
  {eval}。提交 `submission.csv`,列 `target`(预测值,行序与 `test.csv` 一致)。
- 指标:`{metric}`(越大越好 / 越小越好)。
- 约束:只能用本工作目录里的数据,不得访问网络;最终方案要写进
  `runfile.py`,并在 `./working/submission.csv` 输出 `target` 列。
{single_rules}
- 执行环境:{env}
```

- 自由批次:`{eval}` = "用 train.csv 上的交叉验证评估方案",`{single_rules}` 为空。
- 单模型批次:`{eval}` = "用 train.csv 上的**单次留出**(如最后 20%)评估方案,不要做交叉验证",
  `{single_rules}`:

```text
- **本次是单模型对照**:最终方案只能训练**一个**估计器实例(一种算法、
  一套超参),预测时直接用它输出。禁止多模型加权/平均/投票/堆叠,禁止
  同族多 seed、多配置、多折平均,也禁止"训练多个候选再挑一个"。
  早停允许(训练集内部留尾部验证集),但交付的仍是一个模型。
```

AIDE 框架内置指引(不可配置,节选):

```text
- Your response should only contain a single code block.
- All the provided input data is stored in "./input" directory.
- **If there is test data provided for this task, please save the test predictions in a
  `submission.csv` file in the "./working" directory ...**
- The implementation should include a predict() function ...
- The evaluation should be based on {k}-fold cross-validation but only if that's an
  appropriate evaluation for the task at hand.        # k_fold_validation>1 时才加
```

---
