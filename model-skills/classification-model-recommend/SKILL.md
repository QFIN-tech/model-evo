---
name: classification-model-recommend
description: 给业务人员推荐历史营销模型——解析自然语言需求(客群+用途)→读模型台账语义筛选排序→展示详情→询问是否评估→(可选)委托 classification-model-evaluation 产三档评估。当用户说"推荐历史模型""找可复用模型""有没有现成模型""历史模型""模型推荐"时使用。
---

# 历史模型推荐

> 给业务人员从历史营销模型列表中找到可复用模型,支持可选的样本上评估(委托 classification-model-evaluation 产三档标准化三件套)。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| 需求自由描述 | ✅ | 用户输入 | 一段话,从中抽取 `business`(业务线)/ `segment`(训练客群)/ `keyword`(用途关键词,可多个);条件缺失不臆造,留空=该维度不限 |
| 模型台账 | ✅ | `model-knowledge/assets/historical-model-knowledge/model_catalog.csv` | 14 列 CSV;数据资产,不属于 session |
| 模型报告 | 否 | `model-knowledge/assets/historical-model-knowledge/reports/` | `{model_id}_{模型简称}.md` 及同名 `.json`;`模型报告路径` 列非空时提取历史指标展示 |
| `session_dir` | 评估时必选 | 会话启动确认(见根目录 CLAUDE.md) | 推荐落盘与评估产物目录 `${session_dir}/model-recommend/{model_id}/` |
| `split_ranges` | 评估时必选 | `task-spec/_manifest.json` 或 `data-profile/_manifest.json` | train_range / test_range / oot_range 三档切分区间 |
| 样本表 / 模型表 | 评估时必选 | task-spec manifest / 台账 `模型表` 字段 | 样本表提供 label,模型表提供 score,默认按 `[user_no, pday]` JOIN |

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录(即本文件所在目录),执行时替换为实际绝对路径,不要依赖当前工作目录。

**模型推荐**(工作流程 Step 2-3):无独立召回脚本。Claude 直接读 `model-knowledge/assets/historical-model-knowledge/model_catalog.csv`(14 列 CSV)作为候选池,按解析出的 `business`/`segment`/`keyword` 语义筛选 + 排序,默认 Top3。台账规模建议 ≤ 数百行;超出时由调用方先按业务线等维度自行切片。

**评估委托**(工作流程 Step 6,用户选择评估时;单条 entry,wrapper 内部 4 步: fetch → get → split → eval):

```bash
python <skill_dir>/scripts/fetch_eval_sample.py \
  --session-dir <session_dir> \
  --model-id <model_id> \
  --sample-table <db>.<sample_table> \
  --score-table <db>.<score_table> \
  --join-keys user_no,pday \
  --fetch-start <YYYYMMDD> --fetch-end <YYYYMMDD> \
  --train-range <YYYYMMDD>,<YYYYMMDD> \
  --test-range  <YYYYMMDD>,<YYYYMMDD> \
  --oot-range   <YYYYMMDD>,<YYYYMMDD> \
  --score-col score --label-col label
# 加 --submit 同步执行 wrapper; 加 --no-eval 仅产 predictions 跳过评估
```

`fetch_eval_sample.py` 内部生成的 `fetch_eval_{model_id}.sh` 4 步:
1. **STEP1** spark-submit 调 shared `fetch_spark.py`,样本表⋈模型表 LEFT JOIN,写 HDFS `sample.parquet`
2. **STEP2** `hdfs dfs -get` 拉本地 `predictions/sample.parquet`
3. **STEP3** 本地 `split_sample.py` 按 pday 区间切 train/test/oot 三档
4. **STEP4** `invoke_evaluation.py` 委托 `classification-model-evaluation/scripts/eval_single.py` 产三件套

## 3. 参数说明

### 3.1 完整工作流程

#### 3.1.1 解析自由描述(Step 1)

从业务人员的一段话里抽取:
- `business`(业务线,与台账 `业务线` 列取值对齐;未提到则不限)
- `segment`(训练客群,与台账 `训练客群` 列对齐;未提到则不限)
- `keyword`(用途/场景关键词,可多个)

抽取后**回显解析出的条件**给用户确认。条件缺失不要臆造,留空即可(留空=该维度不限)。

**客群口径**:模型列表 `训练客群` 字段是自由文本,Claude 在语义匹配时综合处理:状态口径(细分状态)、全量口径(含"全量/全客群/全部"等视为不限客群)、自由文本(语义近似匹配)。无需额外规则配置。

#### 3.1.2 语义筛选与排序(Step 2-3)

直接读 `model-knowledge/assets/historical-model-knowledge/model_catalog.csv` 作为候选池。先按 `状态=可用` 过滤(用户要求含不可用模型时放宽),再结合以下因素给出最终排序(模型列表无 KS/AUC/PSI 指标列,故主要靠语义与时效):
- **语义相关性**:候选的 `模型中文名`、`预测目标`、`正样本定义`、`训练客群` 与用户真实意图的贴合度
- **时效**:`训练时间窗` 越近越好(注意区分 train 与 oot 区间)
- **可用性**:`状态`=可用 优先;非可用需明确告知

#### 3.1.3 展示推荐详情(Step 4,强制)

**必须**向用户展示推荐结果,默认 Top3(不足则按实际)。每个模型完整展示 14 个字段(排名+适配度 / model_id / 模型中文名 / 预测目标 / 正样本定义 / 算法类型 / 训练客群 / 训练时间窗 / 模型表 / 状态 / 负责人 / 推荐理由 / 历史指标 / 可复用点),展示原则、落盘规则、降级处理、报告指标提取(含 `data_profile.data_splits[]` 性能表与 `performance.score_buckets[]` 分档表)详见 `references/recommendation-details.md`。**不允许只输出 model_id 列表**。

#### 3.1.4 询问是否在样本上评估(Step 5,强制)

**推荐展示完成后,必须主动询问用户是否在当前样本上评估推荐模型的表现**,不能跳过、不能默认不评估。

询问话术(示例):
```
> 推荐已完成,Top1 是 `<model_id>`(高适配度)。
>
> 是否需要基于当前任务样本(50,000 条,5 个 pday)在样本上评估 `<model_id>` 的实际表现?
> - 评估 → 按 Train/Test/OOT 三档切分,分别计算 KS/AUC/分档分布,产出三份报告
> - 跳过 → 直接进入下一步(feature-matching 拉训练特征宽表)
```

**用户选择"评估"**:走「Step 6 生成评估脚本并提交」;评估脚本调用 `fetch_eval_sample.py`(取数+切分+委托评估一条龙),一次作业产三档标准化三件套。

**用户选择"跳过"**:不生成评估脚本,直接进入下游 feature-matching;在 `_manifest.json` 中标注 `eval_performed: false`。

#### 3.1.5 生成评估脚本并提交(Step 6,当用户选择评估时)

**输入信息传递**:
- 样本表: `task-spec/_manifest.json` 中的 `source_table`(含 label + 关联键)
- 模型表: 推荐结果 Top1 的 `model_table`(提供 score)
- 关联键: 默认 `[user_no, pday]`,两表同名(否则需建视图别名)
- 切分区间: 复用 `data-profile/_manifest.json` 中的 `split_ranges`
- 分数字段: 默认 `score`,可在 CLI `--score-col` 覆盖
- 标签字段: 默认 `label`,可在 CLI `--label-col` 覆盖

**配置文件**:`<session_dir>/model-recommend/{model_id}/eval_config.yaml`(由 `fetch_eval_sample.py` 自动落盘)。完整模板见 `references/recommendation-details.md` 第 2 节。

**评估完成后**:
- 读取三份 JSON,向用户展示 Train/Test/OOT 的 AUC/KS/标签率汇总表 + 分档分布摘要
- 将评估结果摘要追加到 `<session_dir>/report.md` 的"三、历史模型推荐"段
- 在 `<session_dir>/model-recommend/{model_id}/_manifest.json` 中记录 `eval_performed: true` + 三档指标
- 若发现模型在 OOT 上明显衰减(AUC 跌幅 > 0.03),提示用户注意

### 3.2 fetch_eval_sample.py(评估 entry:取数+切分+委托评估一条龙)

详见 `references/script-parameters.md` 第 2 节。关键参数:`--session-dir` / `--model-id` / `--sample-table` / `--score-table` / 三档 `*-range` / `--score-lag-day`(t-1 滞后 JOIN)。

### 3.3 split_sample.py(本地三档切分)

详见 `references/script-parameters.md` 第 2 节。支持比例模式(`--ratios`)与显式区间模式(`--train-range/--test-range/--oot-range`)。

### 3.4 invoke_evaluation.py(评估委托)

详见 `references/script-parameters.md` 第 3 节。三档 parquet 转 CSV 落临时目录,调 `eval_single.py` 目录模式一次产 4 份三件套(train/test/oot/all)。

## 4. 输出产物

标准输出目录(遵循 CLAUDE.md session 约定,按推荐模型 ID 分子目录):

```text
<session_dir>/model-recommend/{model_id}/
├── recommendations_<HHMM>.md          # 推荐摘要(markdown, 人工复盘)
├── _manifest.json                     # 结构化推荐结果 + eval_performed 标记 + 三档指标
├── eval_config.yaml                   # 评估配置(自动落盘, 仅评估时)
├── fetch_eval_{model_id}.sh           # 生成的 4 步 wrapper(仅评估时)
├── predictions/
│   ├── sample.parquet                 # 样本表⋈模型表 JOIN 结果
│   ├── train.parquet / test.parquet / oot.parquet
│   └── _split_manifest.json
└── evaluation/
    ├── {model_id}_{train,test,oot,all}_eval.{json,md,xlsx}   # 三档 + all 合并, 共 4 份标准化三件套
    └── _manifest.json                 # 评估元信息
```

### 4.1 产物内容

| 产物 | 必选 | 说明 |
|---|:---:|---|
| `recommendations_<HHMM>.md` | ✅(session 模式) | 推荐摘要,人工阅读 |
| `_manifest.json` | ✅(session 模式) | 结构化推荐结果;含 `eval_performed` 标记 |
| `eval_config.yaml` / `fetch_eval_{model_id}.sh` | 条件生成 | 仅用户选择评估时生成 |
| `predictions/*.parquet` + `_split_manifest.json` | 条件生成 | 取数与三档切分产物 |
| `evaluation/{model_id}_{split}_eval.{json,md,xlsx}` | 条件生成 | 四档评估三件套(train/test/oot/all,KS/AUC/准确率/精确率/召回率/F1/10 档分布/评估质量评分);`all` 档为三档样本纵向拼接后的整体评估;未提供切分区间时降级为单份 `{model_id}_eval.*` |
| `<session_dir>/report.md` 第三段 | 条件生成 | 评估摘要回填"三、历史模型推荐"段 |

### 4.2 评估口径

- **评估逻辑委托 `classification-model-evaluation` skill**: 评估阶段调 `classification-model-evaluation/scripts/eval_single.py` 产出标准化三件套(JSON+MD+XLSX)
- **效果指标**: KS / AUC / 准确率 / 精确率 / 召回率 / F1(由 `eval_single.py` 内置)
- **分档**: 默认 10 档等频,含占比 / 正样本率 / lift(由 `eval_single.py` 内置)
- **三档切分**: 复用 task-spec 切分区间, Train / Test / OOT 分别评估, 便于观察 OOT 衰减
- **能力边界**: split 间 PSI(如 test vs train 分数分布 PSI)不在本 skill 产出范围内;如需 split 间 PSI 或多版本对比, 调 `classification-model-comparison` skill
- 若用户未提供切分区间,降级为整体评估(单份 `{model_id}_eval.{json,md,xlsx}`)

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `model-task-routing` / `classification-model-orchestration` | 编排流程中"历史模型推荐"阶段拉起本 skill |
| 上游 | `classification-model-task-spec` | 提供 `source_table` 与 `split_ranges`(评估输入) |
| 上游 | `model-knowledge` | 提供模型台账 catalog 与历史报告 reports(数据资产) |
| 下游 | `classification-model-evaluation` | 评估统一委托其 `eval_single.py` 产标准化三件套 |
| 下游 | `classification-model-comparison` | 三档评估 JSON 参与会话级 N-way 对比(推荐基线 vs 新模型) |
| 下游 | `feature-matching` | 用户跳过或完成评估后,进入拉训练特征宽表环节 |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ Step 4 强制(展示详情) | 不允许只输出 model_id 列表,推荐理由必填 |
| ⚠️ Step 5 强制(询问评估) | 不能跳过、不能默认不评估 |
| ⚠️ 需求解析原则 | 条件缺失不臆造,留空即不限;解析结果必须回显给用户确认 |
| ⚠️ t-1 滞后 JOIN 模式(`--score-lag-day 1`) | 样本表 day t LEFT JOIN 模型分表 day t-1,适用于"当天样本用昨日模型分"场景;`dt_col` 必须在 `join_keys` 中;lag=1 时模型分表窗口自动平移为 `[fetch_start-1, fetch_end-1]`,split 区间仍按样本表 pday 不变 |

> 覆盖范围、职责边界、评估委托边界、PSI 能力边界、三档切分与取数、数据资产边界、模型报告规范、异常处理全表详见 `references/constraints-and-exceptions.md`。

## 7. 异常处理

异常分类与处理方式详见 `references/constraints-and-exceptions.md` 第 13 节。

## 8. 模型列表与工具说明

模型台账字段、模型报告规范、工具清单、安全约束详见 `references/model-catalog-spec.md`。台账文件:`model-knowledge/assets/historical-model-knowledge/model_catalog.csv`;报告目录:`model-knowledge/assets/historical-model-knowledge/reports/`。

---

数据来源:模型台账 `model-knowledge/assets/historical-model-knowledge/model_catalog.csv`;历史报告 `model-knowledge/assets/historical-model-knowledge/reports/`;session 产物落 `${session_dir}/model-recommend/{model_id}/`。
最后更新:2026-07-07
