# 经验与记忆（中心化经验库）

跨任务经验存放在**经验库服务**里，不再存放在 skill 内置的 `memory/` 目录。本页说明一个 run
怎样读经验、怎样写依据，以及收尾时怎样沉淀经验。合并决议的写法见
[experience-library-merge.md](experience-library-merge.md)。

## 存放位置

| 内容 | 位置 | 写入者 |
|---|---|---|
| Trial / Generation 原始事实 | `<run>/memory/facts/*.jsonl` | Harness 自动 |
| 本任务压缩总结、`final_recipe` | `<run>/experience/task-experience.yaml` | Harness 汇总 + 交付命令 |
| 本 run 固定的经验库视图 | `<run>/experience/library-view.json` | profile 自动 |
| 本 run 提炼的候选原则 | `<run>/experience/exported-global-experiences.yaml` | **收尾 subagent 撰写** |
| 未能提交的快照 | `<run>/experience/outbox/` | `export-experience`（服务不可达时） |
| 跨任务原则、证据、卷宗、配方、榜单 | 经验库服务（`assets/global-config.yaml` 的 `library.uri`） | 快照提交 + 合并 |

经验库是追加式账本。每次写入都会产生一个递增序号，称为**水位**。读取时总是指定水位，所以
同时在跑的 run 互相看不到对方的新经验。

## 业务域

task-spec 的 `business_domains` 参与经验库的任务相似度（决定能否拿到"相关任务"档的经验）。
可以写业务域词表（`runtime/library/business_domains.yaml`）里的代码、别名或中文名，同一业务的
不同写法会被归一；词表外的写法原样保留，只与完全相同的写法匹配。`library domains` 可查词表。

## 一个 run 怎样读经验

1. **固定水位**（自动）：`profile` 第一次需要经验（泄漏对账）时取当前水位，拿到可信任务指纹后
   按同一水位补取本任务的数据，写入 `library-view.json`，并在 `run-state.json` 中记录
   `library{uri, namespace, watermark, stale, source}`。整个 run 都只读这个视图。
   - 服务不可达时用本机最近的缓存（`stale=true`）；没有缓存时视图为空，本 run 不使用经验。
   - `diagnose` 会报告 `library_unreachable`、`library_stale` 或 `library_outbox_pending`。
     服务由用户在终端启动：`python <skill-root>/scripts/modelevo.py library serve --daemon`。

   **配置解析与主动禁用**：连接参数优先级为环境变量
   （`MODELEVO_LIBRARY_URI/_NS/_TEAM/_CACHE/_ENABLED`）> 本 run 的 `modelevo-config.yaml`
   的 `library.*` > `assets/global-config.yaml`；改 run config 就能把这个 run 指向别的库。
   `library.enabled: false`（或 `MODELEVO_LIBRARY_ENABLED=0`）是**主动禁用**：不连服务、
   不读本机缓存、`baseline` 不创建 warm-start trial。它与"服务不可达 → 退到缓存"不同，
   后者仍会带出真实经验；需要本 run 完全不用历史经验时必须用它。
2. **planner-context 第 7 节**：
   - 7.0 任务卷宗、7.1 数据底座卷宗：天花板、有效方向、已证伪方向、未决问题、数据注意事项；
   - 7.2 同任务方向表：历史 trial 按方向聚合（有效 / 失败）；
   - 7.3 同任务避免清单：同口径、方向为避免、且证据没有冲突的原则；
   - 7.4 Warm Baselines：同任务榜单。gen-00 已对前 `baseline.warm_rank_limit`（默认 2）名做热启动。
3. **起草假设后主动检索**：

   ```bash
   python <skill-root>/scripts/modelevo.py experience search --run <run> \
       --topics feature.seq.window_count --hypothesis "事件×窗口计数网格能补充近因特征"
   python <skill-root>/scripts/modelevo.py experience get <原则id> --run <run>
   python <skill-root>/scripts/modelevo.py experience evidence <原则id> --run <run>
   python <skill-root>/scripts/modelevo.py experience topics --run <run>      # 主题词表
   ```

   `search` 是**分级检索**：先取精确档，凑不满 `k` 才逐级放宽，并附上同主题下已经
   失败过的方向。每张卡片标注档位（`tier`）和匹配理由（`why`）：

   | 档 | 含义 | 怎么用 |
   |---|---|---|
   | L0 同任务 | 同一任务、口径一致 | 可直接 `follows` / `extends` |
   | L1 近似任务 | 同数据底座跨目标，或同目标、列集小改（列重合 ≥0.7） | 可 `follows`，注意对象列是否还在 |
   | L2 相关任务 | 同业务域 / 列族重合的相关任务，或仅指标同族（如 auc↔ks） | 只作假设来源，用 `tests` |
   | L3 通用·需验证 | 无任务指纹的历史原则、有迁移证据的跨业务原则；冷启动时的兜底 | 用 `tests`，小步验证 |
   | L4 方法与工程 | 与指标 / 任务类型无关的评估与工程原则（OOM、CV 协议…） | 照常遵守 |

   精确档（L0/L1）按分数混排；放宽档按档位依次补位。有精确档时，跨任务卡片最多 2 张。卡片还会
   标注置信度（high / medium / low），以及证据冲突标记（`mixed` / `contradicted`）。任务相似度
   来自多维任务画像（业务域、目标、列 MinHash、量级、切分方式），固定视图时由 task-spec 与原生列
   算出，并随快照入库。
4. **Planner 程序化使用**：`plan` 会拦截命中避免清单的候选，记录在 `experience_avoid_rejected`。
   有对象（列前缀）的原则按对象拦截，没有对象的原则按主主题拦截。物化命令会预填 `--topics`
   和 `--basis`。

## 依据：每个新 trial 必须写明（`--basis`）

```bash
new-trial ... --topics feature.seq.window_count --basis p-202610-ab12cd34:tests
new-trial ... --topics model.deep.architecture --basis novel
```

| 关系 | 含义 |
|---|---|
| `follows` | 按原则建议去做（省略关系时默认） |
| `extends` | 在原则基础上推进一步（更细粒度、更多对象） |
| `tests` | 原则在本任务 / 本口径上是否成立尚不确定，本 trial 用来检验 |
| `contradicts` | 有意与原则反着做（原则说避免、本 trial 去做），需要写明理由 |
| `novel` | 库里没有相关原则 |

- 视图已固定时 `--basis` 必填；原则 id 必须在视图中，关系必须是上表之一。校验失败时
  trial 不会创建。
- `--from-parent` 的子代没有给依据时继承父代依据；confirm 子代总是继承。
- 视图不可用时照常记录依据，并标注 `unverified`。旧参数 `--experience-ids <id>` 等价于
  `--basis <id>:follows`。
- `--topics` 填主题词或别名（会规范化到词表）；词表外的写法会原样保留，供合并时补词。
- `finalize-generation` 在 `generation.yaml` 中汇总本代依据：`cited` / `novel` / `missing`，以及
  `shown_never_cited`（展示过但从未被引用的原则）。

依据会成为**显式证据**：trial 结束后，它对所引用原则是支持、反驳还是中性，由实测 Δ 及其 CI
判定。没有引用、但主题和口径完全一致的 trial，会作为**被动证据**以较低权重计入。原则的置信度
由加权证据计算，不由撰写者自报。

- 判定按 `direction`：`+1` 要求 CI 下界 > 0 才算支持；`-1` 要求 CI 上界 < 0；**`0`（无增益）要求
  CI 上界 < 0 才算支持** —— CI 跨越 0 的 trial 什么也没证明，记 `neutral`，既不加支持也不加反驳。
- 原则的 `evidence.abstract_task_ids` 由**服务端在合并时自动写入**本次快照的任务指纹，不需要在
  合并决议里手填。没有指纹的原则拿不到同任务权重（检索首档、planner 避免清单、方向表、profile
  对账都依赖它）。
- 卡片上的置信度带来源：`confidence_source=evidence` 表示由证据算出；`author` 表示尚无证据关联、
  用的是撰写者自报值（评估 / 工程类原则按设计不建被动证据）。后者不代表结论弱，只代表未被
  trial 读数验证过。

## 热启动与固化 baseline（gen-00）

- `baseline` 先跑 trial-001 锚点。视图里有同任务、同算法的固化 baseline 配方时直接渲染复用；
  没有时现场渲染，训练评估后作为 `kind=baseline` 快照提交（先到者为准）。
- 然后把视图榜单前 `warm_rank_limit` 名配方渲染成 warm trial：配方 bundle 从缓存取出，
  `{{MODELEVO_RUN_DIR}}` / `{{MODELEVO_SKILL_ROOT}}` 填成本 run 与本 skill 的路径，并做 AST 校验。
  `execution.warm_start` 记录 `recipe_id`、水位与参照分数。warm trial 是普通根 trial，不计入
  “本 run 新发现”。收尾快照会把它作为该配方的一次复测提交。

## 任务指纹 abstract_task_id（atid）

atid 是经验体系的主键，唯一入口是 `TaskSpec.resolve_abstract_task_id(run_dir)`，**不要**自己拿
splits 列名计算 hash。loader 派生的列必须写进 spec `data.derived_columns`，否则数据底座指纹会变。

## 收尾：沉淀经验

1. 收尾 subagent 撰写 `experience/exported-global-experiences.yaml`（格式见下）。成功、失败、
   边际递减三类结论都要写。
2. 提交快照：

   ```bash
   python <skill-root>/scripts/modelevo.py export-experience --run <run_dir> --model <模型名>
   ```

   快照包含候选原则、全部 trial 结果（含依据与主题）、入榜配方及其测量、热启动复测和检索日志。
   退出码：`0` 已提交；`1` 导出文件有 schema 错误（合法条目已提交，修好后重跑会提交新快照）；
   `3` 服务不可达，快照已写入 outbox（服务恢复后执行 `library submit-pending --run <run_dir>`）。
3. 按 [experience-library-merge.md](experience-library-merge.md) 合并：`library merge-context` →
   写决议 → `library merge-apply`。合并决定每条候选是新增、并入、按口径拆分、推翻、并存争议还是
   丢弃，并改写任务卷宗与数据底座卷宗。所有变更写入更新日志，可以按快照撤销。

## 导出文件格式（`experience/exported-global-experiences.yaml`）

顶层可以写成**列表**（推荐），也可以写成 `experiences:` 包装的列表，或者单条 mapping。

```yaml
- claim: 零膨胀原值金额回归上,hurdle(P(y>0)×E[y|y>0])相对单模型父代 rmse 显著下降
  category: architecture            # feature_engineering|architecture|hpo|evaluation
  action_type: add                  # add | avoid
  outcome: positive                 # positive | negative | neutral
  confidence: high                  # high | medium | low(参考;最终由证据计算)
  topics: [model.hurdle]            # 推荐:主题词(见 experience topics),合并时据此归类
  objects: []                       # 推荐:具体对象(列前缀 / 信息源),抽象结论留空
  condition:                        # 必须是 mapping
    model_type: regression
    target_transform: raw
    primary_metric: rmse
  context: 动支金额 87% 为 0,原值 rmse≈12932
  evidence:
    trial_ids: [trial-013-g03-explore-xgboost-254, trial-015-g04-confirm-xgboost-353]
    summary: 013 vs 单模型父代 Δ=−134.8;confirm 015 vs baseline Δ=−250.9(−1.94%)
  boundary: 只在原值口径验证;log1p 目标上未测
- claim: 金额二阶派生(amt2_ 族)在原值 rmse 口径下无独立增益,不应再试
  action_type: avoid
  outcome: negative
  topics: [feature.base.derivation]
  objects: [amt2_]
  evidence: trial-021 vs_parent Δ=+10.7(rmse 变差,rejected)
  boundary: 仅本数据底座、原值 rmse 口径
```

逐条 schema 校验（`runtime.memory.load_experience_export`）：

- 每条必须是 mapping，并且有非空 `claim`（或旧字段 `statement`）。
- `outcome` / `action_type` / `confidence` 必须取上述枚举值；`do_not_retry` 必须是布尔值；
  `condition` 如果写了就必须是 mapping；`category` 必须是字符串。
- `context` / `evidence` 可以写成字符串。缺少 evidence 只告警。
- 错误逐条附带**条目序号、行号、字段和修复提示**。`diagnose` 对导出文件做同样的只读校验，
  发现问题时报告 `experience_export_schema_error`。
- 旧字段 `target` 仍然接受；合并模型会把它映射到主题词或对象。**口径写进 `condition`**：
  回归必须写 `target_transform`。不同口径的结论在库里不会互相并入或推翻。
