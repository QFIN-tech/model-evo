# task-spec.md 4 段式需求规格文档模板

> 本文件从 `classification-model-task-spec/SKILL.md` 4.2 节抽出,包含 task-spec.md 的完整 4 段式模板（建模目标 / 核心参数 / 样本数据 / 待处理项）。SKILL.md 中保留 3 行摘要 + 指向本文件的指针。

```
# 建模需求规格

> 沟通时间: {日期}
> 需求成熟度: A-可直接设计 / B-部分待确认 / C-需先探查
> 模型简称: {model_name}

## 一、建模目标

{一句话说清楚：在 XX 人群上预测 XX 行为，用于 XX 触达，达到 XX 效果}

## 二、核心参数

| 维度 | 内容 | 状态 |
|------|------|:---:|
| 业务场景 | {增长/促活/获客/推荐} | |
| 标签人群 | {圈选条件} | |
| 预测目标 | {未来N天是否发生X行为} | |
| 目标变量 | `label`（样本表固定列名） | |
| 表现窗口 | {N天} | |
| 预估正样本率 | ~{X}%（如有） | |
| 效果目标 | {P0指标 + 基线（如有）} | |
| 触达方式 | 离线T+1跑批打分（默认） | |
| 约束条件 | {有/无，说明} | |

> 假设与默认值：{如用户分层标签="是"}（待确认）、{如沉默阈值=7天}（待数据探查）。

## 三、样本数据

**源表**: `{table_name}`

| 列名 | 含义 | 类型 |
|------|------|------|
| user_no | 用户唯一标识 | string |
| label | 正负样本标记（0/1） | int |
| pday | 样本观察日期（yyyyMMdd） | string |

**数据文件**: `runs/{timestamp}-{model_name}/data-profile/{model_name}_sample_{YYYYMMDD}.parquet`

### 样本分析结果

| 指标 | 值 |
|------|-----|
| 总样本量 | |
| 正样本率 | |
| pday 范围 | |
| 标签稳定性 | 正样本率波动幅度 = Xpp ({稳定/轻微波动/显著波动}) |
| 样本充足度 | {充足/基本可用/不足} |

### Train/Test/OOT 切分

| 集合 | 样本量 | 正样本率 | pday 范围 |
|------|--------|----------|-----------|
| Train | | | |
| Test | | | |
| OOT | | | |

> 切分方式：按 pday 时间顺序。
> 详细报告见 `data-profile/report.md` 和 `data-profile/report.xlsx`

## 四、待处理项

| 优先级 | 事项 | 类型 | 建议 |
|:---:|------|:---:|------|
| P0 | {阻塞建模的} | 待确认/待探查 | |
| P1 | {不阻塞的} | 待确认/待探查 | |

## 下一步

- [ ] 用户确认 Train/Test/OOT 切分结果
- [ ] 编排器调用 classification-model-recommend 检索历史模型（**spark 模式**；local_file 模式跳过此步）
- [ ] 用户确认是否建模 → classification-model-development
```

> 关联: `classification-model-task-spec/SKILL.md` 4.2 节 task-spec.md 模板
