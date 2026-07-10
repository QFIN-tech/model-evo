---
name: model-task-routing
description: 建模任务总入口（覆盖营销/增长/获客/运营/风控场景）。当用户提出建模需求时首先触发，按固定格式一次性列出全部待确认问题，收集回答后判定属于 classification（分类预测）还是 uplift（增益/因果干预），并据此路由到对应管线：classification → classification-model-orchestration；uplift → uplift-model-task-spec。判定时将已收集的需求基础信息按固定 JSON 格式传递给下游。触发词：建模、建模型、新模型、模型需求、帮我建模、设计模型、模型立项、模型规划。
---

# 模型任务路由

## 1. 角色定义

你是建模流程的**总入口**。所有建模诉求都从本 skill 进入。职责是：

1. **一次性**向用户列出全部待确认问题（固定格式，不逐条挤牙膏）
2. 收集回答后，判定任务类型（classification / uplift / 拒绝）
3. 把已收集的需求基础信息按**固定 JSON 格式**传递给下游 skill
4. 不做需求挖掘细化、不做样本分析、不创建 session 目录 —— 这些由下游负责

核心原则：
- **总入口唯一性**：用户提出建模诉求先走本 skill，由本 skill 分流，下游不得直接承接未经路由的请求
- **一次性提问**：所有问题在一个回合内全部抛出，用户可一次性回复，避免多轮往返
- **判据前置**：classification vs uplift 的判定维度必须显式问清楚，不靠猜
- **信息透传**：判定完成后，已收集到的所有信息以 JSON 形式传递给下游，不要求用户重复回答

触发词：建模、建模型、新模型、模型需求、帮我建模、设计模型、模型立项、模型规划。

## 2. 输入依赖

### 2.1 用户输入

- 用户的建模诉求（自然语言描述）
- 用户对 3 个固定问题（Q1/Q2/Q3）的回复

### 2.2 业务知识前置

Skill 调起后，首先读取 `model-knowledge/assets/business-domain-knowledge/business-domain-knowledge.md` 了解业务定义，再进入提问环节。

### 2.3 classification vs uplift 判定要点

| 维度       | classification（分类）               | uplift（增益/因果）                  |
|------------|---------------------------------------|---------------------------------------|
| 业务问题   | "谁会发生某结果"（流失/转化/逾期…）   | "对谁干预能带来增量"（发券/触达是否值得） |
| 预测目标   | 结果发生的概率                        | 干预带来的处理效应（ITE/CATE）        |
| 是否有干预 | 无干预，纯预测                        | 有明确干预动作（treatment）           |
| 数据形态   | 样本+标签                             | 实验/对照组（treatment vs control）   |
| 决策用途   | 排序/筛选高风险或高价值人群           | 圈定"干预敏感"人群、优化资源投放      |

**关键判据**：

1. 是否存在一个"干预动作"，且关心的是该动作带来的**增量**而非结果本身？
   - 是 → `uplift`
   - 否 → `classification`
   - 模糊 → 在提问环节让用户澄清（见 3.1 节 Q1）
2. 是否存在充足的随机试验数据？
   - 是 → `uplift`
   - 否 → `classification`

## 3. 工作流程

### 3.1 第一步：扫描诉求 → 补齐缺项 → 进入确认

收到建模诉求后，先扫描用户原始表达，对 Q1/Q2/Q3 中已隐含的回答直接提取，不重复提问；仅对未覆盖的问题按以下固定格式一次性抛出，让用户在一轮内补齐。全部问题有答案后直接进入 3.2 节复述确认环节，不再次逐条追问。

```
为了把建模需求路由到正确的管线，请一次性回答以下问题（不确定的填"待探查"，不要留空）：

Q1. 预测目标是什么？（如：未来7天是否动支 / 是否响应外呼 / 发券后是否核销 / 流失概率…）
    - 若是"是否XX"二分类 → 倾向 classification
    - 若是"干预后相比对照多多少"增量 → 倾向 uplift

Q2. 是否存在一个"干预动作"（如发券、外呼、Push、利率优惠等），且关心的是该动作带来的"增量效果"而非结果本身？
    - 是 → uplift 方向
    - 否 → classification 方向
    - 不确定 → 请描述业务场景，由我判断

Q3. 是否有实验/对照组数据（treatment vs control）？
    - 有 → 倾向 uplift
    - 无 → 必走 classification
```

> **提问纪律**：Q1/Q2/Q3 中已在用户原始表达里给出回答的，直接提取，不重复提问；仅就缺项一次性抛出补问。所有问题有答案后直接进入 3.2 节复述确认，不再逐条追问。不替用户填答未给出的项（填"待探查"由用户在确认环节校正）。

### 3.2 第二步：收集回答 + 判定任务类型

收到用户回复后，按以下顺序判定：

1. **路由判定**（基于 Q1/Q2/Q3，按顺序短路）：
   - Q1 描述为回归目标（金额/时长/次数/频次等连续值）→ **拒绝**（见 7.1，建议转化为二分类）
   - Q2 = 是 **且** Q3 = 有 **且** Q1 描述为"增量/处理效应/干预效果" → `task_type = uplift`
   - Q1 描述为"是否XX"二分类 **且**（Q2 = 否 **或** Q3 = 无）→ `task_type = classification`
   - 仍模糊 → 向用户追问一次 Q1，仍不明确则**拒绝**并说明原因

2. **复述确认**：判定完成后，向用户复述判定结论与依据，明确下一步去向：
   ```
   > 任务类型判定：{classification | uplift}
   > 判定依据：{Q1/Q2/Q3 的回答摘要}
   > 下一步：{拉起 classification-model-orchestration / uplift-model-task-spec}，下方 Q1/Q2/Q3 三项答复会以 JSON 透传，下游不会再次询问；但下游仍会就业务窗口、客群划分、评估口径等其他需求细节继续追问。
   ```

### 3.3 第三步：构造需求基础信息 JSON 并传递

判定完成后，把第一步收集到的全部信息按以下**固定 JSON 格式**封装，作为下游 skill 的输入契约：

```json
{
  "task_type": "classification | uplift",
  "routing_basis": {
    "q1_target": "用户原话描述的预测目标",
    "q2_intervention": "yes | no | uncertain",
    "q3_experiment_data": "yes | no"
  },
  "user_raw_request": "用户最初的建模诉求原话",
  "routed_at": "路由时间戳 YYYY-MM-DD HH:MM:SS"
}
```

**传递方式**：
- **classification 方向**：拉起 `classification-model-orchestration`，将上述 JSON 作为 `routing_input` 传入。该 JSON 不落盘，由 orchestration 在第一步消化后写入 `task-spec/_manifest.json` 的 `routing` 字段。
- **uplift 方向**：拉起 `uplift-model-task-spec`，将上述 JSON 作为 `routing_input` 传入。该 JSON 由 uplift-model-task-spec 在规格化时消费。

### 3.4 流程速览

```
用户建模诉求
  → model-task-routing（本 skill，总入口）
  → 一次性抛出 Q1/Q2/Q3
  → 判定 task_type
      ├── classification → classification-model-orchestration（透传 routing_input JSON）
      ├── uplift         → uplift-model-task-spec（透传 routing_input JSON）
      └── 拒绝           → 输出拒绝信息 + 建议，结束
```

## 4. 输出产物

### 4.1 路由决策

- `task_type` ∈ {classification, uplift, 拒绝}
- `routing_basis`：判定依据摘要（Q1/Q2/Q3 回答）

### 4.2 需求基础信息 JSON

按 3.3 节固定格式封装，作为 `routing_input` 透传给下游。**本 skill 不落盘任何文件**，JSON 由下游 skill 消费。

### 4.3 下游拉起动作

调起对应 skill（`classification-model-orchestration` 或 `uplift-model-task-spec`）并传入 `routing_input` JSON。

## 5. 与其他 skill 关联

- **下游（classification）**：`classification-model-orchestration` —— 接收 `routing_input` JSON，接管分类建模全流程（task-spec / recommend / feature-matching / development）
- **下游（uplift）**：`uplift-model-task-spec` —— 接收 `routing_input` JSON，接管 uplift 任务规格化与后续管线
- **无上游**：本 skill 是建模流程的总入口

## 6. 执行约束

1. **不跳过路由**：任何建模诉求都先走本 skill，不直接进入 `classification-model-orchestration` 或 `uplift-model-task-spec`
2. **不替用户判定**：Q1/Q2/Q3 必须让用户明确表态，不靠 LLM 臆测；但用户原始表达已回答的项直接提取，不重复提问
3. **不创建目录**：本 skill 不创建 session 目录、不落盘任何文件，所有产物由下游负责
4. **信息透传不丢失**：JSON 必须包含全部已收集信息，下游不得要求用户重复回答

## 7. 异常处理

### 7.1 既非 classification 也非 uplift

- **场景**：用户诉求属于回归（金额/时长/次数）、多分类、聚类、时序预测、NLP/CV 等
- **处理**：直接拒绝，不路由
- **拒绝模板**：
  ```
  当前需求不属于 classification / uplift 任一管线，超出本框架能力范围，无法路由。
  原因：{具体原因}
  建议：{如"可尝试将金额预测转化为'是否高额动支'的二分类问题" / "可咨询其他团队"}
  ```

### 7.2 结束条件

以下任一条件满足即结束：

1. **路由成功**：判定为 classification 或 uplift，JSON 已透传给下游，本 skill 退出
2. **拒绝**：判定为非 classification/uplift，输出拒绝信息后结束
