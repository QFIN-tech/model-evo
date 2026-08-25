---
name: feature-evolution-orchestration
description: 特征进化编排--围绕当前建模任务持续挖掘有价值的特征。可挖性前置诊断(开跑前判ROI,避免在榨干的池里空转)+ 数据驱动候选生成(原料相关性矩阵黑名单,champion残差反推)+ 多seed降噪Stage A + 严苛全量Stage B。LLM提假设、确定性系统判G1-G6,只认valid/OOT稳定增益。当用户说"特征进化""特征挖掘""帮我找特征""提升模型效果的特征""feature evolution"时使用。
---

# 特征进化编排(Feature Evolution)

## 0. 核心机制

| 机制 | 解决的问题 |
|---|---|
| **Stage 0 可挖性诊断** | 开跑前算池内冗余分布+单变量AUC分布,预判"该跑还是该劝退",避免在榨干的池里空跑 |
| **原料相关性矩阵** | 候选生成前自动算"未用特征 vs champion"的\|Spearman\|,高冗余原料自动标黑,LLM 只在低冗余原料里做交互,杜绝候选栽在冗余上 |
| **多 seed bootstrap** | Stage A 用 3-5 个不同种子子样本+置信区间,救回"子样本略降但全量能涨"的候选,降低子样本单点估计噪声 |
| **分场景默认** | 区分冷启动(cold_start)与饱和模型(saturated)场景:饱和场景更高轮次预算、更严前置诊断、更严 G3 口径 |

总体框架:LLM 提假设+代码,确定性系统判 G1-G6,只认 valid/OOT 稳定增益,只增不改,断点可续。

## 1. 角色定义

你是**特征进化流程**的总调度。职责:驱动「前置诊断 -> 数据驱动生成 -> 多seed快筛 -> 全量终判 -> 反馈 -> 下一轮」闭环。

- **LLM 负责**:提 Feature Hypothesis + Feature Code(候选代码);
- **确定性系统判定**:前置诊断、原料相关性、G1-G6 关卡、champion 更新/回滚(`feature-evolution-evaluation`);
- **你永远不自己下结论**:任何"这个特征有用"的判断必须交给脚本。包括"还有没有信号可挖"--交给 Stage 0 诊断,不许靠 LLM 直觉判断 ROI。

核心原则(AlphaEvolve 闭环):
- **只认 valid/OOT 稳定增益**:train 再好看,valid/OOT 不涨一律拒绝
- **候选多样性**:每轮 K 个,≥1 explore(新原料/新算子)+ ≥1 exploit(成功方向加工)
- **不重复劳动**:假设指纹 + **原料指纹**双黑名单(原料指纹见 §3.2)
- **断点可续**:state.json / ledger.jsonl / .done 全落盘

## 2. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| 本地样本 | ✅ | 用户提供 | parquet/csv,含 id_cols+label_col+dt_col+特征列(可含文本列) |
| 建模任务说明 | ✅ | 用户提供 | 业务目标一句话(正样本定义) |
| 切分方案 | ✅ | 用户提供 | 时间比例或显式区间,禁止随机切分 |
| **场景标识** | ✅ | 用户提供 | `cold_start`(从零训baseline) 或 `saturated`(给已有强模型加特征)。决定轮次预算/诊断严苛度 |
| 已有模型打分列 | 可选 | 用户提供 | `base_model_score_col`:饱和场景下 baseline=该列,融合判定须超既有分单独 |

## 3. 工作流程

### 3.1 会话启动 + 场景确认

```bash
python <orch>/scripts/list_sessions.py --runs-dir runs --last 5
```

有历史 session 询问「续跑/新建」。新建时**先与用户确认场景**:

- **cold_start**(冷启动):baseline 从零训固定超参 XGB,低垂果实多,默认 max_rounds=15。
- **saturated**(饱和):已有训练好的强模型(填 `base_model_score_col`),默认 max_rounds=30-50、前置诊断更严、原料黑名单更激进。

### 3.2 session 初始化 + **Stage 0 可挖性诊断**

复制 `config/session_config.example.yaml` 为 `<session>/session_config.yaml` 填值后:

```bash
python <orch>/scripts/prepare_session.py --config <session>/session_config.yaml --session-dir <session>
```

幂等六步:
1. 契约校验(样本快照)
2. 时序切分
3. 特征画像
4. baseline 训练
5. 台账初始化(champion=baseline)
6. **Stage 0 可挖性诊断**(见下)

**Stage 0 诊断逻辑**(`feature-evolution-evaluation/scripts/probe_roi.py`, prepare_session 第 6 步自动调用):

饱和场景下,开跑前必须回答"池里还有多少信号":

a. **池内冗余扫描**:未用特征(池 - champion 列)逐个算与 champion 全列的 \|Spearman\| max。统计分布:
   - 高冗余(\|corr\|>0.9):被 champion 隐式覆盖,换族不换信息,**作原料黑名单**
   - 低冗余(\|corr\|<0.5):真·新信息维度,作候选原料白名单

b. **单变量质量预筛**:对低冗余子集算方向修正 AUC(对齐 G2 口径),AUC>0.52 的进"有信号原料池"。

c. **ROI 判定**:
   - 低冗余+有信号原料数 ≥ K×3 -> ROI 可观,正常进入进化循环
   - 低冗余+有信号原料数 < K(每轮候选数)-> **ROI 枯竭,停止并向用户报告**:池内已榨干,建议换范式(池外数据/语义特征/调参),不要空跑 30 轮
   - 介于之间 -> 进入循环但提高轮次预算

诊断结果落 `profile/roi_report.md` + `profile/material_whitelist.txt` + `profile/material_blacklist.txt`,候选生成必读。

**champion 残差反推**(可选,饱和场景强烈建议,需 baseline 预测):
```bash
python <eval>/scripts/champion_residual.py --session-dir <session>
```
产出 `profile/champion_residual.md`,候选生成 Step 2 的必读材料之一。

### 3.3 语义特征(可选,池外通道)

样本含文本列时,先跑 semantic-feature skill 注册语义特征,语义列成为候选合法原料(池内榨干时这是唯一活路)。

### 3.4 进化循环(核心)

对 round = 1, 2, 3, ... 依次:

**Step 1 生成轮简报**(确定性):
```bash
python <orch>/scripts/build_round_brief.py --session-dir <session> --round <N>
```
简报含 **`material-whitelist.md`**(低冗余有信号原料)+ `material-blacklist.md`(高冗余原料,勿用)+ champion 残差分析(见下)。

**Step 2 提出候选**(LLM 职责)。必须先读:
- `evolution/rounds/rXXX/round-brief.md`
- **`profile/material-whitelist.md`**(候选原料**只能从这里选**,杜绝赌中高冗余)
- `profile/material-blacklist.md`(近几轮已证实冗余的原料族,整族勿用)
- `evolution/rounds/rXXX/case-batch.json`(均衡采样,看正负差异)
- `references/operator-catalog.md`
- `profile/data-desc.md`
- `evolution/feedback/latest-feedback.md`
- **`profile/champion_residual.md`**(champion 预测错的样本在哪些未用特征上取值异常--这才是"模型缺什么"的数据驱动答案)

写 K 个候选到 `evolution/rounds/rXXX/candidates/`(契约见 `references/candidate-contract.md`):
- `cNNN.py` + `cNNN.meta.json`
- 候选设计指引:
  - 先多字段交互算数值(差分/比率/排名/分箱/语义×数值),再代入非线性复合算子(log1p/sqrt/多项式/tanh/分式/高斯核/erf/分段复合)
  - explore:whitelist 里未用过的原料、新算子族;exploit:已接受特征或上轮成功方向加工
  - 只增不改;假设具体可证伪;NaN 显式 fillna

**Step 3 评估**(确定性):
```bash
python <eval>/scripts/evaluate_round.py --session-dir <session> --round <N>
```

**Step 4 结算**(G6 融合+champion更新/回滚):
```bash
python <eval>/scripts/commit_round.py --session-dir <session> --round <N>
```

**Step 5 反馈**:
```bash
python <eval>/scripts/build_feedback.py --session-dir <session>
```

**Step 6 终止判定**:读 `state.json`,`stop_reason` 非空(轮次上限/连续无接受/ROI枯竭/目标达成/时间预算)或用户喊停 -> §3.5;否则 round+1 回 Step 1。

### 3.5 收口
```bash
python <eval>/scripts/export_features.py --session-dir <session>
python <eval>/scripts/finalize_session.py --session-dir <session>
```

## 4. 输出产物

```
runs/{timestamp}-{name}/
├── session_config.yaml
├── data/                          # 样本快照+三档切分+清单
├── profile/
│   ├── feature-profile.csv        # 特征画像
│   ├── data-desc.md
│   ├── roi_report.md              # 可挖性诊断报告
│   ├── material_whitelist.txt     # 低冗余+有信号原料(候选必读)
│   ├── material_blacklist.txt     # 高冗余原料(勿用)
│   └── champion_residual.md       # champion残差反推"缺什么"
├── baseline/                      # baseline模型+预测+三档评估
├── semantic/                      # (可选)语义特征
├── evolution/
│   ├── state.json                 # champion/轮次/stop_reason/场景
│   ├── ledger.jsonl               # 全历史(append-only)
│   ├── rounds/rXXX/               # 简报/case batch/候选/结果/摘要
│   ├── accepted/                  # 已接受特征
│   ├── feedback/latest-feedback.md
│   └── export/accepted-features/  # 交付包
└── report.md                      # 总报告
```

## 5. 与其他 skill 关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 下游 | `feature-evolution-evaluation` | 确定性执行核心(含 Stage0诊断/原料矩阵/多seed快筛/全量终判) |
| 下游 | `semantic-feature` | 池内榨干时的池外通道 |
| 边界 | `classification-model-tuning` | 特征进化枯竭时自动建议切调参(stop_reason=roi_exhausted 时) |

## 6. 执行约束

1. **不许自评**:任何指标只能来自 evaluation 脚本输出,LLM 不得估算
2. **不许放水**:挂关卡就是挂了,不许改阈值/数据救候选
3. **不许手改台账**:state.json/ledger.jsonl 只由脚本写
4. **候选代码红线**(G1):模块级 transform,import 仅 math/numpy/pandas/statistics/re,确定性,60s内,不含label/id/dt/既有分
5. **断点必续**:重入先 list_sessions 推断进度,从断点轮继续
6. **饱和场景必跑 Stage 0**:不开诊断直接进循环是违规,易在榨干的池里空转

## 7. 异常处理

| 条件 | 处理 |
|---|---|
| Stage 0 判 ROI 枯竭 | 不进循环,直接报告用户"池内已榨干",给换范式建议(池外/语义/调参) |
| prepare_session 退出码 2/3 | 配置/契约问题,修正后重跑(幂等) |
| session broken | 按 list_sessions 提示补齐或 --force 重建 |
| 候选执行超时 | G1 拒,反馈提示降复杂度 |
| commit_round 回滚 | G6 防过拟合正常机制;连续回滚减同方向 exploit 增 explore |
| 连续 N 轮无接受 | stop_reason 命中终止;N 随场景变(saturated: N=5, cold_start: N=3) |
