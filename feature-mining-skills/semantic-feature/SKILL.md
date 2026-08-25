---
name: semantic-feature
description: 语义特征挖掘(semantic-feature)--把业务文本(对话/客户描述/工单/营销交互/催收记录等)转化成传统业务模型可直接使用的特征。技术路线 Text -> Embedding Encoder(BGE/Qwen Embedding/哈希/API 可替换) -> 监督 Head -> Semantic Feature，注册为特征进化(feature-evolution)的候选原料。当用户说"文本特征""语义特征""把文本变成特征""embedding 特征"时使用；正常流程由 feature-evolution-orchestration 在检测到文本列时调用。
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
| `sentence_transformers` | `pip install sentence-transformers`（首次运行下载权重） | BGE / Qwen Embedding 等本地模型（`encoder.model` 指定，如 `BAAI/bge-small-zh-v1.5`） |
| `api` | 无（标准库 urllib） | OpenAI 兼容 `/embeddings` 端点；key 只从环境变量读（`encoder.api.api_key_env`），**严禁写进配置文件** |

### 3.2 监督 Head（变体 = 文本源 × head）

| Head | 输出 | 说明 |
|---|---|---|
| `lr_score` | `sem_{源}_score`（1 列） | Embedding -> LogisticRegression -> 语义打分；label 监督，只用 train 段拟合 |
| `pca_lowdim` | `sem_{源}_p1..k`（k 列） | Embedding -> PCA -> 低维语义表征（无监督压缩） |

多文本源（多列拼接/不同截断窗口）通过 `semantic.sources` 配置，每个源独立产出变体。

## 4. 输出产物

```
<session_dir>/semantic/
├── registry.json        # 注册表: 每变体一条(名称/来源文本列/head/输出列/三档方向修正 AUC)
├── features.parquet     # id_cols + 全部语义特征列(三档, 供评估帧 join)
├── {变体名}/head.json   # head 工件(系数/PCA 投影, json 落盘不用 pickle)
└── cache/*.npz          # 嵌入缓存(同文本+同后端不重算, 换后端自动失效)
```

同时回写 `evolution/state.json` 的 `semantic_columns`。此后：
- `feature-evolution-evaluation` 的评估帧自动 join 语义列（与普通列同权过 G1~G6）
- `build_round_brief` 的轮简报会列出语义特征清单（含 OOT AUC），供候选生成引用

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `feature-evolution-orchestration` | 准备 session（含 text_cols 契约）；正常流程由其在进化前调用本 skill |
| 下游 | `feature-evolution-evaluation` | 消费 registry + features.parquet；语义列作为候选原料过关卡 |
| 下游 | `feature-evolution-orchestration` | 轮简报引用语义特征清单，LLM 在其上做组合/交互候选 |

## 6. 执行约束

1. **防穿越**：监督 head 只见 train 段的 label；test/oot 仅推理
2. **key 不落盘**：api 后端的 key 只从环境变量读，任何配置文件/日志中不得出现
3. **工件不用 pickle**：head 工件以 json 落盘（系数/投影矩阵），避免反序列化安全面
4. **嵌入必须缓存**：换 encoder 后端/模型自动失效缓存，同配置重跑不重复计算

## 7. 异常处理

| 条件 | 处理方式 |
|---|---|
| 无文本源（未配 text_cols/sources） | 退出码 2：提示在 session_config 补 `sample.text_cols` 或 `semantic.sources` |
| sentence-transformers 未安装 | 启动即报错并给出安装指引（不静默降级） |
| api 后端缺 base_url/model | 退出码 2，指明缺哪项 |
| 文本列不在样本中 | 退出码 2，列出实际可用列 |
| 文本完全无信号 | 正常完成：注册表里 AUC 接近 0.5，特征仍可用（交由 feature-evolution 关卡判定价值） |
