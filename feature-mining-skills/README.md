# Feature Mining Skills

特征智能挖掘 Skill 集合：围绕当前建模任务持续挖掘有价值的特征。以 **LLM 提假设 + 确定性系统判定** 的闭环（AlphaEvolve 式）驱动「前置诊断 -> 数据驱动生成 -> 多seed快筛 -> 全量终判 -> 反馈 -> 下一轮」的特征进化循环，只认 valid/OOT 稳定增益。

> 每个 Skill 位于独立目录下的 `SKILL.md`，包含 frontmatter（`name` + `description`）与正文（输入依赖 / 执行命令 / 输出产物 / 关联 skill / 执行约束 / 异常处理）。

## 整体能力

- **特征进化编排**：场景化（冷启动 / 饱和模型）驱动特征进化闭环，LLM 提候选、确定性系统判 G1~G6 关卡，champion 只增不改、断点可续
- **可挖性前置诊断（Stage 0）**：开跑前算池内冗余分布 + 单变量 AUC 分布，预判「该跑还是该劝退」，避免在已榨干的池里空跑
- **数据驱动候选生成**：原料相关性矩阵自动产出白名单/黑名单，LLM 只在低冗余原料里做交互；champion 残差反推「模型缺什么」
- **多seed降噪 + 严苛终判**：G4/G5 在 K 个种子子样本上取增益均值 + bootstrap CI，G6 全量帧轮级融合防贪心过拟合
- **语义特征（池外通道）**：把业务文本（对话/工单/营销交互等）经 Embedding -> 监督 Head 转成结构化特征，注册为候选原料；hash / sentence-transformers / api 三种后端可替换

## 目录结构

```
feature-mining-skills/
├── feature-evolution-orchestration/    # 特征进化编排（LLM 调度 + 候选生成，流程总入口）
│   ├── SKILL.md
│   ├── config/session_config.example.yaml
│   ├── references/                     # 算子目录 + 候选代码契约
│   └── scripts/                        # prepare_session / build_round_brief / evo_prep 等
│
├── feature-evolution-evaluation/       # 确定性评估核心（无 LLM，唯一判定权威）
│   ├── SKILL.md
│   ├── config/gate_config.example.yaml
│   ├── scripts/                        # probe_roi / evaluate_round / commit_round / evo_core 等
│   └── tests/                          # （占位，待补）
│
└── semantic-feature/                   # 语义特征挖掘（文本 -> Embedding -> 监督 Head -> 注册）
    ├── SKILL.md
    ├── config/semantic_config.example.yaml
    ├── scripts/                        # run_semantic / encoders / heads / registry
    └── tests/
```

> **公共代码说明**：[`model-evo/_modelevo-shared/`](../_modelevo-shared/) 供各 skill 公共使用（`config_io` 等），三个 skill 的 `_bootstrap.py` 会自动向上定位并注入。

## Skill 清单

| Skill | 说明 | 触发词示例 |
|---|---|---|
| `feature-evolution-orchestration` | 特征进化编排：场景确认 -> session 初始化（含 Stage 0 可挖性诊断）-> 进化循环（轮简报 -> LLM 提候选 -> 评估 -> 结算 -> 反馈）-> 收口导出 | 特征进化、特征挖掘、帮我找特征、提升模型效果的特征 |
| `feature-evolution-evaluation` | 确定性评估核心：G1~G6 关卡（AST 白名单安全执行 / 单变量质量 / 冗余 / 多seed增益 / 轮级融合）、台账、反馈生成、交付包导出，零 LLM | 评估这些候选特征、结算这一轮、导出接受的特征 |
| `semantic-feature` | 语义特征挖掘：文本 -> Embedding（hash / 本地模型 / API 可替换；full / list / chunk 三种文本模式）-> 监督 Head（OOF LR / GBDT / PCA / 分箱 / 聚类）-> 注册为特征进化候选原料 | 文本特征、语义特征、把文本变成特征、embedding 特征 |

## 核心流程

```
样本(含特征列, 可含文本列) + 建模任务说明 + 切分方案
   │
   ▼
feature-evolution-orchestration
   ├─ 场景确认（cold_start 从零训 / saturated 给已有强模型加特征）
   ├─ prepare_session（契约校验 / 时序切分 / 特征画像 / baseline / 台账 / Stage 0 诊断）
   │     └─ Stage 0: 池内冗余扫描 + 单变量预筛 -> ROI 判定（该跑 / 劝退）
   ├─ （可选）semantic-feature 注册语义特征（池外通道）
   │
   ▼ 进化循环（round = 1, 2, 3, ...）
   ├─ build_round_brief（轮简报: 白名单/黑名单/残差分析/case-batch）
   ├─ LLM 提 K 个候选（explore + exploit, 原料只从白名单选）
   ├─ evaluate_round（G1~G5, 多seed 降噪）
   ├─ commit_round（G6 轮级融合, champion 更新/回滚）
   └─ build_feedback -> 下一轮（或 stop_reason 终止）
   │
   ▼
export_features（交付包: 特征代码 + 特征值 + 融合模型）+ finalize_session（总报告）
```

## 前置依赖

### 基础环境
参考仓库主 [`model-evo/README.md`](../README.md) 的前置依赖部分（Python 3.9+，numpy / pandas / scikit-learn / scipy / xgboost）。

### 可选依赖

| 依赖 | 用途 |
|---|---|
| `pip install sentence-transformers` | semantic-feature 的本地 Embedding 模型后端（如 `BAAI/bge-m3`，GPU 上可开 fp16）；不装可用 hash 后端（无外部依赖） |
| OpenAI 兼容 `/embeddings` 端点 | semantic-feature 的 api 后端（key 只从环境变量读，严禁写进配置） |

### 上下游数据前置

| 场景 | 前置数据 |
|---|---|
| 冷启动（cold_start） | 一份含 `id_cols + label_col + dt_col + 特征列` 的 parquet/csv + 时间切分方案 |
| 饱和模型（saturated） | 同上，另需已有强模型的打分列（`base_model_score_col`） |
| 语义特征 | 样本中含文本列（`sample.text_cols`） |

## Session 产物结构

每个特征进化任务以 `runs/{timestamp}-{name}/` 组织：

```
runs/{timestamp}-{name}/
├── session_config.yaml                 # 会话配置
├── data/                               # 样本快照 + 三档切分 + 清单
├── profile/                            # 特征画像 / roi_report / 原料白黑名单 / champion残差
├── baseline/                           # baseline 模型 + 预测 + 三档评估
├── semantic/                           # （可选）语义特征 registry + features.parquet
├── evolution/
│   ├── state.json                      # champion / 轮次 / stop_reason / 场景
│   ├── ledger.jsonl                    # 全历史（append-only）
│   ├── rounds/rXXX/                    # 简报 / case batch / 候选 / 结果 / 摘要
│   ├── accepted/                       # 已接受特征（只增不改）
│   ├── feedback/latest-feedback.md     # 下一轮候选生成必读
│   └── export/accepted-features/       # 交付包
└── report.md                           # 总报告
```

## 关键约束

- **LLM 不自评**：任何「这个特征有用」的判断必须交给评估脚本，LLM 不得估算指标
- **关卡不放水**：挂关卡就是挂了，不许改阈值/数据救候选；阈值覆盖只能来自 session_config 的 gates 段（留痕）
- **只增不改**：已接受特征代码冻结只读，修正通过新增候选叠加实现
- **台账不可手改**：state.json / ledger.jsonl 只由脚本写，ledger 只追加
- **防泄漏**：候选输入帧不含 label/id/dt/既有模型分，从源头防标签泄漏
- **数据安全红线**：严禁硬编码用户ID/手机号/身份证号（`config_io.check_sensitive` 拦截）
