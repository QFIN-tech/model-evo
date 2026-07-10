# 推荐详情展示与评估配置

> 本文件从 `classification-model-recommend/SKILL.md` 3.1.3 + 3.1.5 节抽出,包含展示推荐详情的字段表、展示原则、落盘规则、降级处理、报告指标提取,以及评估委托的 `eval_config.yaml` 模板。SKILL.md 中保留 4 行摘要 + 指向本文件的指针。

## 1. 展示推荐详情(Step 4,强制)

**必须**向用户展示推荐结果,默认 Top3(不足则按实际)。每个模型**必须**完整展示以下信息:

| 字段 | 说明 |
|------|------|
| 排名 + 适配度 | #1/#2/#3 + 高/中/低 |
| model_id | 模型唯一标识 |
| 模型中文名 | 业务可读的名称 |
| 预测目标 | 模型预测什么 |
| 正样本定义 | 正样本口径(让业务确认是否一致) |
| 算法类型 | 如 xgboost |
| 训练客群 | 训练样本客群 |
| 训练时间窗 | train/oot 区间(可能为空) |
| 模型表 | 模型分数落库的 Hive 表(便于取数) |
| 状态 | 可用 / 不可用 |
| 负责人 | 模型负责人 |
| **推荐理由** | 为什么匹配该需求(必填,不能省略) |
| 历史指标 | 若 `模型报告路径` 非空,从 JSON 报告提取 KS/AUC 等指标展示;为空则提示"报告待补充" |
| 可复用点 | 可作 baseline / 可作 base_score_table / 超参作初始点 等 |

**展示原则**:
- 用表格 + 文字结合,让业务人员一眼看清"为什么推荐这个模型"
- 高适配度模型要重点突出,中/低适配度要说明差距
- **不允许只输出 model_id 列表**,必须带详情

**推荐结果落盘(session 模式)**:推荐对话给出后,把 markdown 形式的推荐摘要写到
`<session_dir>/model-recommend/{model_id}/recommendations_<HHMM>.md`(`<session_dir>` 由会话启动时确认,见根目录 CLAUDE.md;`{model_id}` 为推荐模型 ID),
便于业务复盘。同时写 `<session_dir>/model-recommend/{model_id}/_manifest.json` 结构化版本。

**降级处理**:若召回为空或无高适配,说明原因(业务线/客群/关键词哪个维度无匹配),给出最接近的次优模型或建议(放宽客群、调整关键词、或提示需新建模型)。

**报告指标提取**:当模型列表 `模型报告路径` 非空时,将路径中的 `.md` 替换为 `.json` 即得对应的 JSON 报告文件。读取该 JSON 后提取以下内容并展示。

性能指标(`data_profile.data_splits[]`):

| 数据集 | 时间范围 | 样本量 | 标签率 | AUC | KS |
|--------|----------|--------|--------|-----|-----|

- `segment`: 数据集类型(如 train/val/test/oot)
- `split_name`: 客群名
- `time_range`: 时间范围
- `sample_count`: 样本量
- `label_rate`: 标签率
- `auc`: AUC
- `ks`: KS
- 空值填 `-`

分档分布(`performance.score_buckets[]`,如有):

| 分箱 | 分数区间 | 人数 | 标签率 | Lift |
|------|----------|------|--------|------|

- `bucket_id` → 分箱编号,`score_range` → [min, max],`user_count` → 人数,`label_rate` → 标签率,`lift` → Lift
- buckets 为空则跳过此表

展示原则:
- 优先展示与用户需求客群匹配的数据集
- 指标表后附报告路径,引导查看完整报告(含特征重要性、超参数等)
- JSON 文件不存在时提示"报告 JSON 缺失,可联系负责人"

## 2. 评估配置模板(Step 6,用户选择评估时)

**配置文件**:`<session_dir>/model-recommend/{model_id}/eval_config.yaml`(由 `fetch_eval_sample.py` 自动落盘,模板见 `scripts/eval_config.example.yaml`)

```yaml
# recommend 语境: feature_table = 模型表(score_table), features = [score_col]
# dt_col 须样本表与模型表同名(否则需建视图别名); shared fetch_spark.py 用单一 dt_col 同时过滤两表
spark_submit:
  hdfs_base: null                   # 留空走代码默认 /user/<whoami>/model-recommend
  # bin/options 留空 = 走 _modelevo-shared/scripts/spark_defaults.yaml 默认档

model:
  name: <model_id>                      # = model_id, 用作文件名前缀
  version: v1
  sample_table: <db>.<sample_table>     # 样本表(提供 label)
  feature_table: <db>.<score_table>     # 模型表(提供 score)
  join_keys: [user_no, pday]
  dt_col: pday                          # 须样本表与模型表同名
  id_cols: [user_no]
  fetch_dt: [<YYYYMMDD>, <YYYYMMDD>]    # 须覆盖 train+test+oot 并集
  where: null                           # 可选客群筛选; 严禁硬编码用户ID/手机号/身份证号
  label_col: label
  features: [score]                     # 模型分列, recommend 语境下 features 就这一个
  split:
    train_range: [<YYYYMMDD>, <YYYYMMDD>]
    test_range:  [<YYYYMMDD>, <YYYYMMDD>]
    oot_range:   [<YYYYMMDD>, <YYYYMMDD>]
```

---

> 关联:SKILL.md 3.1.3 / 3.1.5 节;数据来源:模型台账 `model-knowledge/assets/historical-model-knowledge/model_catalog.csv`、报告 `model-knowledge/assets/historical-model-knowledge/reports/`。
