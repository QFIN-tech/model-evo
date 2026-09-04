---
name: semantic-feature
description: 语义特征挖掘(semantic-feature)--把业务文本(对话/客户描述/工单/营销交互/催收记录/App列表/联系人列表等)转化成传统业务模型可直接使用的特征。技术路线 Text -> Embedding Encoder(BGE/Qwen Embedding/哈希/API/条目聚合 可替换) -> 监督 Head(OOF LR/GBDT/PCA/分箱/聚类) -> Semantic Feature，注册为特征进化(feature-evolution)的候选原料。当用户说"文本特征""语义特征""把文本变成特征""embedding 特征"时使用；正常流程由 feature-evolution-orchestration 在检测到文本列时调用。
---

# 语义特征挖掘（semantic-feature · Semantic Feature）

## 1. 定位

传统业务模型主要依赖结构化变量，但大量业务信息存在于非结构化文本中。本 skill 把这些文本变成结构化模型能直接消费的特征，并注册进特征进化流程作为候选原料：

```
Text -> Embedding Encoder -> Supervised Head -> Semantic Feature -> 注册(feature-evolution 候选原料)
```

与 feature-evolution 的分工：**semantic-feature 负责创造新的语义特征能力；feature-evolution 负责在所有特征中持续搜索、组合和演进**。最终统一以主模型稳定 OOT 增益作为评价标准。

## 2. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| 已初始化的 session | ✅ | `feature-evolution-orchestration` 的 prepare_session | 含三档切分与样本契约 |
| 文本列 | ✅ | 样本契约 `sample.text_cols` 或 `semantic.sources` 配置 | 中文/英文均可（hash 后端按字符 n-gram，不依赖分词） |

若原始数据还不是 session 形态（JSONL 分片 / 独立文本补表 / 无 parquet），见第 8 节「冷启动指引」——先拼样本再回来跑本 skill。

## 3. 执行命令

```bash
python <skill_dir>/scripts/run_semantic.py \
  --session-dir <session_dir> [--config <session_dir>/semantic_config.yaml]
```

配置优先级：`--config` 文件（`semantic:` 段）> session_config.yaml 的 `semantic:` 段 > 缺省（hash 后端 + 每个 text_cols 一源 + 双 head）。模板见 `config/semantic_config.example.yaml`。

### 3.1 Encoder 后端（可替换）

| 后端 | 依赖 | 适用 |
|---|---|---|
| `hash` | 无（sklearn 自带） | 冒烟/单测/无 GPU 环境；字符 1~2-gram 哈希，确定性 |
| `sentence_transformers` | `pip install sentence-transformers`（首次运行下载权重） | BGE / Qwen Embedding 等本地模型（`encoder.model` 指定，如 `BAAI/bge-m3`）；支持 `fp16`（GPU 吞吐 +~50%） |
| `api` | 无（标准库 urllib） | OpenAI 兼容 `/embeddings` 端点；key 只从环境变量读（`encoder.api.api_key_env`），**严禁写进配置文件** |

**编码前自动做文本级去重**：对唯一文本集合编码后按行映射回向量（业务文本大量重复/空串时编码量可降 50%+，结果与逐行编码完全一致）。

**文本模式（`source.mode`）**：

| mode | 行为 | 适用 |
|---|---|---|
| `full`（缺省） | 整行文本一次编码（超 `max_chars` 头部截断） | 描述/对话/备注等自然语言 |
| `list` | 条目去重 → 逐条目短编码 → 按 `agg` 聚合成行向量 | `^`/`,`/空格分隔的离散条目列表（App 列表/联系人列表/标签集）。成本从 O(行数×长文本) 降到 O(唯一条目数×短文本)，语义不被无关条目稀释 |
| `chunk` | 按 `max_chars` 分窗编码 → mean 池化 | 长文本且信息分布在全文（日志/长记录） |

`list` 模式的 `agg`：`mean` / `sum` / `tfidf`（条目 TF-IDF 权重加权平均，缺省）。

### 3.2 监督 Head（变体 = 文本源 × head）

| Head | 输出 | 说明 |
|---|---|---|
| `lr_score` | `sem_{源}_score`（1 列） | Embedding → LogisticRegression → 语义打分。**train 段输出 out-of-fold 分数**（K 折交叉拟合，防 in-sample 虚高传导进特征）；test/oot 用全量 head 推理。极不平衡场景建议配 `lr_params.class_weight: balanced` |
| `pca_lowdim` | `sem_{源}_p1..k`（k 列） | Embedding → PCA → 低维语义表征（无监督压缩，不平衡场景无偏） |
| `gbdt_head` | `sem_{源}_gscore`（1 列） | Embedding → 浅层 XGB（非线性，捕获维度交互）；同 OOF 口径 |
| `quantile_bins` | `sem_{源}_q1..k`（k 列） | 监督分的分箱重表达（序数），树模型友好 |
| `cluster_centroid` | `sem_{源}_cluster`（1 列） | KMeans 簇 ID，人群画像式特征 |

多文本源通过 `semantic.sources` 配置；多列组合策略 `source.combine`：`concat`（拼接成一文档，缺省）/ `separate`（各列独立成源注册）。

## 4. 输出产物

```
<session_dir>/semantic/
├── registry.json        # 注册表: 每变体一条(来源/head/输出列/三档 AUC+PR-AUC/条件增益诊断/成本)
├── features.parquet     # id_cols(去重唯一) + present 指示 + 全部语义特征列
├── {变体名}/head.json   # head 工件(系数/投影/簇心, json 落盘不用 pickle)
└── cache/*.npz          # 嵌入缓存(float32 压缩; 同文本+同后端不重算, 换后端自动失效)
```

registry 每列指标：`{split}_auc`（方向修正）+ `{split}_pr_auc`（低正样本率场景主口径，与 gates.g4_g5_metric=pr_auc 对齐）。
item 级诊断（配了 `base_model_score_col` 时自动产出）：语义分在 baseline 十分位内的**条件 AUC**（判断特征是"补召回"还是"精排"用法）、与 baseline 分的 spearman 相关性、覆盖率、编码耗时。

同时回写 `evolution/state.json` 的 `semantic_columns`。此后：
- `feature-evolution-evaluation` 的评估帧自动 join 语义列（与普通列同权过 G1~G6）
- `build_round_brief` 的轮简报会列出语义特征清单（含 OOT 指标），供候选生成引用

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `feature-evolution-orchestration` | 准备 session（含 text_cols 契约）；正常流程由其在进化前调用本 skill |
| 下游 | `feature-evolution-evaluation` | 消费 registry + features.parquet；语义列作为候选原料过关卡（registry 带 PR-AUC，与 g4_g5_metric 口径联动） |
| 下游 | `feature-evolution-orchestration` | 轮简报引用语义特征清单，LLM 在其上做组合/交互候选 |

## 6. 执行约束

1. **防穿越**：监督 head 只见 train 段的 label，且 train 段分数为 OOF；test/oot 仅推理
2. **key 不落盘**：api 后端的 key 只从环境变量读，任何配置文件/日志中不得出现
3. **工件不用 pickle**：head 工件以 json 落盘（系数/投影/簇心），避免反序列化安全面
4. **嵌入必须缓存**：float32 + 压缩；换 encoder 后端/模型/模式自动失效缓存，同配置重跑不重复计算
5. **join 键唯一性**：features.parquet 落盘前按 id_cols 去重校验——重复时显式报错（或按 `dedup: first` 配置去重），杜绝下游 merge 崩溃
6. **空文本不冒充中性语义**：每源自带 `sem_{源}_present`（0/1 覆盖指示）；无文本行监督分输出 NaN 交给树模型原生处理，而不是给一个空串伪概率

## 7. 异常处理

| 条件 | 处理方式 |
|---|---|
| 无文本源（未配 text_cols/sources） | 退出码 2：提示在 session_config 补 `sample.text_cols` 或 `semantic.sources` |
| sentence-transformers 未安装 | 启动即报错并给出安装指引（不静默降级） |
| api 后端缺 base_url/model | 退出码 2，指明缺哪项 |
| 文本列不在样本中 | 退出码 2，列出实际可用列 |
| 样本 join 键（id_cols）重复 | 退出码 2，列出重复样例；配 `dedup: first` 可自动去重 |
| 文本完全无信号 | 正常完成：注册表里 AUC 接近 0.5，特征仍可用（交由 feature-evolution 关卡判定价值） |

## 8. 冷启动指引（原始数据 → 可用 session）

原始数据常见形态与拼装步骤（全部产物写在任务自己的目录，勿污染源数据）：

1. **海量 JSONL 分片 + 独立文本补表**（如主表 400+ 数值列 + 用户文本表按 user_id 一一对应）：
   - 从主表抽 `id + label + dt` 三列（流式逐分片，只取需要的字段）
   - 文本表抽 `id + text_cols`；按 id many-to-one 左连主表
   - 若有已训练好的主模型打分列，作为 `base_model_score_col` 一起进样本——融合判定即"语义特征必须超既有分单独"，这是饱和模型场景下最稳的增益锚点
2. **单 parquet/csv 已含文本列**：直接把路径填进 session_config 的 `sample.path`，text_cols 指到文本列
3. **超大样本（>300 万行）**：先按时序块分层采样（如 1/5）做实验口径，确认增益后再全量
4. 之后按 feature-evolution-orchestration 的 prepare_session 初始化 session（三档时序切分 + baseline + 台账），再回来跑本 skill

**耗时预期**（A100-40G, bge-m3, batch 256 实测量级）：自然语言短文本 ~240 条/s；≥500 字符长文本 ~70 条/s（`max_chars` 每降一档耗时可观下降）；`list` 模式成本≈唯一条目数（万级秒成）。编码阶段逐 split 打进度日志（已完成条数/吞吐/ETA）。

## 9. 工件口径速查

| 想要 | 配置 |
|---|---|
| 最快冒烟 | 缺省（hash 后端 + lr_score + pca_lowdim） |
| 真 embedding | `encoder.backend: sentence_transformers` + `model: <本地路径或模型名>` |
| App/联系人等条目列表 | `source.mode: list`（+ `agg: tfidf`） |
| 长文本不截断丢信息 | `source.mode: chunk` |
| 低正样本率（<1%） | head 配 `lr_params.class_weight: balanced`；看指标以 `pr_auc` 为主 |
| 增判"补召回 vs 精排" | 样本配 `base_model_score_col`，registry 自动附条件 AUC |
| 既有模型很强，怕语义特征白加 | 必配 `base_model_score_col`（融合判定锚点） |
| 多列异质文本 | `source.combine: separate`（各列独立成源） |
