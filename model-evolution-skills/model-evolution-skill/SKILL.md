---
name: modelevo
description: 使用 ModelEvo V2.20 对结构化/表格监督学习任务进行可审计的端到端自动建模、自主搜索、断点恢复、效果评估和模型交付。适用于二分类、回归与 uplift，以及需要特征探索、跨任务经验、代际 Trial、HPO、确认实验或一次性 final_test 的任务；不用于无需训练模型的一次性数据分析。
---

# ModelEvo V2.20(v2.20.0)

把 bundled Runtime 作为执行与裁决的唯一权威。Agent 负责理解任务、提出
可证伪假设、解释结果和与用户沟通；Runtime 负责数据隔离、状态机、训练、
统计裁决、资源调度、完整性校验与交付。任务计划、假设、反思和报告默认用
中文，除非用户要求其他语言。

把本文件所在目录记为 `<skill-root>`：

```bash
python <skill-root>/scripts/modelevo.py --version
python <skill-root>/scripts/modelevo.py --help
```

不要另写一套训练或评估流程，不要模拟 Trial 指标，不要手改状态产物，也不要
为了跑通而削弱依赖、泄漏、资源、源码或 final_test 检查。

## 渐进加载

只读取当前动作需要的参考文档，不要一次性通读全部 references：

| 当前动作 | 必读 |
|---|---|
| 定义目标、时间窗、loader、split、profile、数据视图 | [task-and-data.md](references/task-and-data.md) |
| 二分类 / 回归 / uplift | 先看 [task-type-conventions.md](references/task-type-conventions.md) 的分发表，再读对应的 [classification](references/task-type-classification.md) / [regression](references/task-type-regression.md) / [uplift](references/task-type-uplift.md) 文档 |
| baseline 后的特征探索 | [feature-exploration.md](references/feature-exploration.md) |
| Planner、评估、晋级、crossover、调度、停止 | [evolution-and-evaluation.md](references/evolution-and-evaluation.md) |
| 编写或修复 Trial 代码 | [template-api.md](references/template-api.md)；Lint 细节见 [stage-playbook.md](references/stage-playbook.md#trial-lint-细则) |
| 阶段命令、状态语义、交付清单 | [stage-playbook.md](references/stage-playbook.md) 的对应小节 |
| subagent 命令签名、等待和回传契约 | [protocol-digest.md](references/protocol-digest.md) 的对应小节 |
| 检索经验、写 trial 依据、收尾沉淀经验 | [experience-and-memory.md](references/experience-and-memory.md) |
| 收尾合并经验快照（写合并决议） | [experience-library-merge.md](references/experience-library-merge.md) |
| 天花板估计（用户要求时） | [benchmark.md](references/benchmark.md) |

需要定位长文档时先用标题或关键词搜索，再读命中的小节。不要通过阅读
`runtime/` 或 CLI 实现来学习操作协议；异常先用 `diagnose`。

## 核心不变量

1. **尺子不可演进。** 同一 protocol 内 target、split、metric、evaluator、
   不确定性口径、约束和 final_test 策略固定；Harness 或泄漏分类改变时走
   `acknowledge-harness-change` 或新 run，不混排旧结果。
2. **四平面隔离。** 特征、标签、reparse、identity 分开；banned 不进入任何
   parquet，identity 只用于框架对齐。Trial 禁止直接调用 `load_data()`。
3. **final_test 只触碰一次。** 搜索阶段不落盘 final 标签；冻结配方后仅
   Finalizer 可经 gate 在内存读取。gate 不退款，失败后必须更换 holdout 或 run。
4. **预测必须可配对。** prediction artifact 携带唯一 sample key 和
   source/split/model/prediction hash；键集合不一致即不可比较。
5. **训练与裁决分离。** 调参（超参选择）只用 train 内部切分；早停默认在
   outer validation 上（只训一次；`training.early_stop_source: train_tail`
   回退为 train 尾部早停），validation 除早停外只用于固定裁决；shadow 只用于
   confirmation；final 结果不回流搜索。
6. **代码裁决晋级。** accepted 必须通过配对统计、同代 BH 和跨代 alpha
   spending；promotion/delivery 还要求完整性、复现和 fresh confirmation。
   晋级对照由 run 配置固定（见「评估口径由 run 配置决定」），同一 run 内不换尺。
7. **源码与状态不可暗改。** Trial code/config 冻结后不得覆盖；修订必须创建
   child Trial。禁止手改 evaluation、leaderboard、run-state 或最终产物。
8. **失败也是证据。** timeout、invalid、inconclusive 和 infra failure 全部留痕；
   基础设施失败不向模型方向记负奖励；同因连续失败触发熔断和 root-cause。
9. **执行边界由工具强制。** Trial 必须通过 `lint_trial.py`、源码完整性和
   产物校验；每个 Trial 默认单模型，级联与融合须用户经 `approve-topology`
   人工批准（Runtime 拓扑门）；不得绕过失败门。
10. **经验必须有本地证据。** 跨任务经验只是 transfer prior；只有经过当前
    任务验证，才能晋级或导出 Claim + Context + Evidence + Boundary。每个 trial
    用 `--basis` 声明依据；原则的置信度由证据计算，不由撰写者自报。

## 全局运行约束

以下约束贯穿整个 run。前三条由 Runtime/guard hook 拦截；后三条**没有工具兜底**，
靠执行纪律。

**留出面标签只经官方 CLI 读取**（guard hook，主会话与 subagent 同规）：任何方式
读取 `labels/(validation|shadow_test|final_test).parquet` 都会被拒，包括 `python -c`、
heredoc 和 `adhoc/`/`feature-lab/` 脚本。只有用户在会话环境显式设
`MODELEVO_ALLOW_HOLDOUT_READ=1` 才豁免。train 标签与 `prescreen` 沙箱照常可用；
配对对照用 `contrast`。

**内存准入由 Runtime 排队，Agent 不手工错峰**。规划并发看
`snapshot --run <run_dir>` 的 `admission.suggested_parallel`；`admission_timeout`
属 infra 失败，不进熔断、不耗预算。已单独 `run --trial <id> --detach` 起跑的
trial，`run-generation` 会记入 `already_running` 并跳过。

**每个 Trial 默认只有一个模型**（拓扑门）。不设计"子模型分数/OOF/logit 当特征再训"
的级联，也不做多模型/多 seed 的平均、加权、stacking 或 AutoML 融合；hurdle（回归）
与 T-learner（uplift）按任务类型白名单放行。未获批准的级联/融合 trial 不训练、
状态保持 CODE_WRITTEN（`run`/`wait` 退出码 5，`TOPOLOGY_BLOCKED`），请求写入
`logs/user-decisions.md`。批准只能由用户在**自己的交互式终端**执行
`approve-topology`；Agent 不得执行该命令、不得写 `approvals/`。按单模型方向继续，
把请求转告用户。细则见 [template-api.md](references/template-api.md#模型拓扑)。

**长时训练必须 `--detach`**：训练挂在 agent 会话的进程树上时，会话边界会连带
杀掉它（实测三个 HPO trial 同时死亡，空烧 7.9 小时）。

**等待用 `wait`，不用 `sleep`**：`wait --generation/--trial/--job <…> --timeout 570`
循环调用（Bash 单次上限 10 min）。退出码 3 表示未完、`wait` 被杀（137/143）时都原样
再调；输出 `STALE_RUNNING` 时重入 `run --trial <id>`。

**经验库**：profile 自动固定本 run 的经验库视图（`experience/library-view.json`），
之后整个 run 只读这个视图。服务由用户在自己的终端启动（`library serve --daemon`），
Agent 不代启；不可达时 `diagnose` 报告 `library_unreachable`。要本 run 完全不用历史
经验，必须设 `library.enabled: false`——只把 `uri` 改成不可达地址**不够**，会回落到
本机缓存。细则见 [experience-and-memory.md](references/experience-and-memory.md)。

## 评估口径由 run 配置决定

`init` 把当时版本的评估口径写进 `<run_dir>/modelevo-config.yaml` 的
`modelevo.evaluator.*`；在途老 run 的配置里没有这些键，保持旧口径。同一 run 内
不换尺，Agent 不改这些键。续跑老 run 时，裁决和停止报告的含义可能不同（例如
`promotion_reference: parent` 口径下小步链不能晋级，也不会出现
`confirm_or_consolidate`），一律以本 run 配置为准，不要按文档里的新 run 口径推断。
例外：warm-start 确认资格修正按 `run-state.json` 的 `skill_version ≥ 2.20.0` 判定，
不由配置键控制（`status` 不输出该字段）。

## 标准工作流

### 1. 初始化或恢复

新任务先只读检查样本 schema、目标、时间列、实体列和规模，然后初始化：

```bash
python <skill-root>/scripts/modelevo.py init --dir <run_dir> --task-id <task_id>
```

补全 `<run_dir>/task-spec.yaml` 与 `data-loader.py` 后再执行：

```bash
python <skill-root>/scripts/modelevo.py validate --run <run_dir>
python <skill-root>/scripts/modelevo.py preflight --run <run_dir>
```

目标含义、观测/标签时点、切分、主指标或泄漏列不能安全推断时才询问用户。
`profile` 首次运行若报告 `needs_user_confirmation=true`，必须停下逐列裁决
ambiguous/reparse 提案，写回 spec 后重跑同一命令；第一次产物只是提案面，
不得供 baseline 使用。用户明确要求直接开跑时记录豁免。

裁决写回 spec 有四条去向：

- 判为泄漏 → `features.banned`（带受控 `reason`）；
- 判为形态不可用 → `features.reparse`（带 `plan`）；
- **复核后判为合法保留 → `features.leak_confirmation.kept_ambiguous`**（只关闭
  确认门，不改数据面）；
- **机械封禁判错了 → `features.leak_confirmation.reinstated_banned`**（撤销该
  封禁，让它回到待裁决）。

后两档不可省：缺了它们，"复核后确认合法"与"机械规则判错了"这两类真实裁决
无处安放，正确做法反而关不上确认门，误封只能靠新建 run 挽回。

**机械提案复核：banned-certain 与 reparse-candidate 是机械提案，不是终审。**
它们由正则/统计规则产生，`leak-classification.md` 会单独给出「机械提案复核清单」
（含 n_unique vs 实体数、长度 CV、分隔符密度、token 复用率）。至少核对一遍：
**真 ID 的唯一值数不可能超过实体数**；唯一率≈1 的变长复合文本字段（事件序列/
JSON/编码串）是 reparse 候选而非实体标识。实测事故：行为序列列 `seq_d365`
（唯一率 0.9976）被误封为 `entity_identifier`，丢掉了该数据面最强的信息通道
（约 +0.0035 AUC），而同数据源的历史 run 都正确判为 reparse。详见
[task-and-data.md](references/task-and-data.md#机械提案复核清单)。

已有任务以磁盘状态为准，不重复已完成且 hash 兼容的工作：

```bash
python <skill-root>/scripts/modelevo.py status --run <run_dir>
python <skill-root>/scripts/modelevo.py leaderboard --run <run_dir>
```

### 2. 建立可信起点

按顺序执行 `profile →（可选 adhoc）→ baseline → write-splits-sample`。
`capacity-check` 是可选诊断，只作参考、不参与放行；`ceiling` 不属于日常流程
（见「按需能力」）。

**profile**：spec、`loader`、`validate` 和 `preflight` 就绪后，主会话直接前台
执行：

```bash
python <skill-root>/scripts/modelevo.py profile --run <run_dir>
```

profile 是机械材料化步骤。执行前不派 subagent、不研读 Runtime、不自行分析样本，
也不先设计探索方案；命令失败才运行 `diagnose`。成功后只检查
`needs_user_confirmation` 和 critical finding：命中确认门则请用户确认并重跑同一
命令，**并复核 `leak-classification.md` 的「机械提案复核清单」**（误封高价值列
等于永久丢掉一条信息通道）；否则立即进入下一阶段。

**adhoc**：没有 CLI 子命令，只在 profile 已完成且确有未覆盖问题时，把一次性
脚本先落盘到 run 的 `adhoc/` 再执行。

**baseline**：建立固定锚点，交给 stage subagent 执行（见「编排与上下文」）。

**write-splits-sample**：baseline 不会自动材料化抽样面，特征探索与
`capacity-check` 都依赖这一步。

**capacity-check**（可选）：`target_reachability` 对照 spec `performance_target`
给出 reachable/unlikely/not_reachable，用于尽早向用户说明目标是否可达。

### 3. 特征优先搜索

baseline 后先按 [feature-exploration.md](references/feature-exploration.md)
探索可归因的 feature bundles，再进入模型与 HPO。使用 ≤20 万抽样面筛选，
晋级方向回到全量面复核并做独立确认。搜索宽度、heavy 配额、消融数量与停止
由配置、Planner、预算和 Runtime 门控制；不要在提示词中发明第二套门槛。

- **先看噪声预检再选探索面**：`splits-sample/manifest.json.noise_precheck`
  （planner-context §6c）给出 `recommend_v1=true`（抽样面噪声大于 2×MDE 且全量
  trial 不贵）时，**直接在 v1 全量面上探索**，并在反思中留痕。
- 复用已验证的特征块用 `new-trial --add-blocks <trial>:<block>[,...]`，不要手拼代码段。
- `prescreen` 只在 train 内产出筛选证据，只用来排除坏区，晋级仍以正式 trial 为准。
- **解析类特征必须走特征块缓存**（lint 规则 21）：reparse 序列/JSON 等**无训练期
  拟合状态**的行内变换，写成 `runtime.feature_cache.feature_block` 包装的内核并定义
  `trial_feature_blocks(split)`（契约见
  [template-api.md](references/template-api.md#特征块-trial_feature_blocks)）；有拟合
  状态的变换仍用 `widen_cached`。
- 条件消融（drop-group retrain）**没有 runtime 实现**，消融 Trial 由 Agent 手建。

Runtime 在这一阶段强制的门，以及对应的放行方式：

- **特征期门**：已完成探索代数 < `feature_exploration.min_generations`（默认 2，
  gen-00 不计）时，model/loss/hpo 方向 Trial 被拒。放行：Planner 释放的名额
  （`--planner-proposal-id`）或 `--override-feature-phase "理由"`。
- 每代特征 Trial 数受 `feature_exploration.bundle_beam_width` 限制（默认 4），超宽需
  `--override-beam "理由"`。
- gen-01 禁止 `--action exploit` / `--action crossover`。
- 跨视图（sample ↔ v1）Trial 记 `baseline_incomparable`，这是预期行为，不是失败；
  相对排序用同视图父代。

细则见 [feature-exploration.md](references/feature-exploration.md#抽样与全量复核)。

### 4. 代际演进

#### 4.1 每代闭环

```bash
python <skill-root>/scripts/modelevo.py snapshot --run <run_dir>          # 含内存准入的建议并发
python <skill-root>/scripts/modelevo.py planner-context --run <run_dir>
python <skill-root>/scripts/modelevo.py plan --run <run_dir> --slots <N>
# 根据 proposal 创建并完成 Trial code/config
python <skill-root>/scripts/modelevo.py run-generation --run <run_dir> --generation <N> --detach
python <skill-root>/scripts/modelevo.py wait --run <run_dir> --generation <N> --timeout 570   # 退出码 3 = 未完，原样再调
python <skill-root>/scripts/modelevo.py finalize-generation --run <run_dir> --generation <N>
python <skill-root>/scripts/modelevo.py timing-report --run <run_dir>     # 本代耗时拆解
```

读取 Runtime 生成的 verdict、leaderboard、reflection、Archive、方向后验和
stop report 决定是否继续；不得由 Agent 自行宣布 accepted 或穷尽。

#### 4.2 编写 Trial

只使用 active-protocol、可排名的父代；普通 Trial 一次只改变一个主要组件，
写明 hypothesis、evidence、falsification、changed_modules 和 family hints。

**批量并行优先**：
整代代码一次写齐并通过 Lint 后再 `run-generation`；结果驱动的调整放到下一代
child Trial，**禁微调循环**。

**每个新 Trial 都要写依据**（`new-trial` 强制）：起草假设后先检索
`experience search --run <run_dir> --topics <主题> --hypothesis "…"`，再用
`--basis <原则id>:<follows|extends|tests|contradicts>` 引用；库里确实没有相关原则时
写 `--basis novel`，并用 `--topics` 标主题。`plan` 的物化命令已预填 `--topics` /
`--basis`（同时带 `--basis-prefill`，勿删）；改了方向就同步改依据。原样沿用的预填
依据只按被动证据计权——真的依据了某条原则，就自己写明关系。细则见
[experience-and-memory.md](references/experience-and-memory.md)。

**写子代：patch / `--set` 优先**：

- 默认用 `new-trial --from-parent <父代>`：纯参数差异用 `--set k=v`；代码差异用
  `--patch next.py`，patch 只写新增或整体替换的顶层定义（新特征块、换内核、改
  fit_and_score/build_estimator），不要读写整份 train.py。
- 写 patch 前先 `modelevo.py trial-skeleton --run <run> --trial <父代>` 看大纲，只在
  需要时按行区间读局部源码。
- 新块用 `PATCH_REGISTER_BLOCKS=["_xxx_block"]` 注册，建议声明 `PATCH_REPLACES=[...]`；
  受保护符号（数据加载/评估/早停/HPO 入口/设备）不可改。
- 复用别的 trial 的块用 `--add-blocks`，可与 `--patch`/`--set` 组合；confirm 只能
  `--from-parent`。
- 长 trial 运行期间可以先起草下一代 patch，等 finalize-generation 确定父代后再
  new-trial；父代变了就按新父代大纲调整。

细则见 [template-api.md](references/template-api.md#补丁式写码)。

**Planner 的提名**来自 planner-context（第 7 节的任务卷宗、同任务方向表、避免清单
和 Warm Baselines；避免清单上的方向会被 `plan` 拦截）以及画像规则与经验库原则。
提名口径见 [evolution-and-evaluation.md](references/evolution-and-evaluation.md#planner-与-archive)。

**兄弟 trial 的配对对照**用 `contrast --trials A,B,... [--ref A]`，**不要手写
bootstrap 脚本**（实测 60 min vs 约 1 min/对）。

**回归零膨胀目标用官方 hurdle**：`new-trial --hurdle-template`，可与 `--action hpo`
组合；不要手写容器（实测手写容器在交付时 clone 失败）。见
[template-api.md](references/template-api.md#官方-hurdle-容器与子模型缓存)。

#### 4.3 HPO 与早停

**同面 HPO 一次做对**：用 `new-trial --action hpo --from-parent <冠军>`，预算给足
（默认 40）。热启动、贴边外扩、lr 收缩重训和撞 cap 外扩都由 Runtime 在同一 trial
内完成，不要多轮手工追边界（实测 3 轮 84 个 config 吃掉 65% 算力只换 +0.0022）。
两道门：同面上一轮 HPO 的 Δ_parent < MDE 时再开一轮需
`--override-hpo-diminishing "理由"`；父代有在途 HPO 子代时 confirm 需
`--allow-confirm-during-hpo "理由"`。

**判断是否撞 cap 只看 `evaluation.complexity.cap_bound`**；`near_cap` 只是提示。
带 `cap_confounded` 的负 Δ 是树数上限造成的，不能当作方向被证伪。

**早停**默认在 outer validation 上（不变量 5），HPO 的超参选择只在 train 内部
（lint 规则 7）。validation 分数因此略偏乐观；样本外结论以 shadow / final_test 为准。

细则见 [template-api.md](references/template-api.md#hpo) 与
[stage-playbook.md](references/stage-playbook.md#同面-hpo-只做一次做完整)。

#### 4.4 停止与收尾判定

**STOPPED 只是建议**：`stop_report.advisory=true` 时继续演进不会被拒。真正的硬拦截
只有 `blocks` 列出的预算护栏：`budget_max_trials` 拦 new-trial，`budget_wall_time`
拦 run-generation。

- 收尾前先看 `undelivered_gain`（同视图最优 trial 相对交付锚点、尚未交付的增益）。
  remaining 中出现 `confirm_or_consolidate:<trial>` 时，先对它 confirm 再收尾——通过
  且优于现任冠军才换冠军，不通过维持原交付。开了收尾整合通道的 run，被同代 BH 拦下
  的最佳候选也会以同样形式出现，处理方式相同。
- `single_data_view_insufficient` 是单口径护栏的状态码，不是统计收敛结论；各停止
  码的含义见 `reason_detail`。

细则见 [evolution-and-evaluation.md](references/evolution-and-evaluation.md#停止语义)。

### 5. 确认与交付

只有通过选择校正、源码完整性和新 seed 独立确认的 fresh-confirmed Trial 可交付，
且它必须**属于搜索冠军的配方**（其它配方的陈旧确认只在落后冠军 ≤ MDE 时可用）。
确认失败记 `unconfirmed`，不是 `rejected`。每配方 confirm 默认 ≤2 次，不要"失败就
再抽"。确认门只看 `fresh_confirmation.passed`，`shadow_test.gap_sanity_passed` 只是
诊断。细则见
[evolution-and-evaluation.md](references/evolution-and-evaluation.md#fresh-confirmation)。

**warm-start 的确认与先验豁免**：

- gen-00 warm-start trial **必须过预注册家族校正才可 confirm**，未过记
  `warm_start_not_confirmable`。仅对 `skill_version ≥ 2.20.0` 的 run 生效；更早的在途
  run 中 warm trial 仍直接获得 confirm 资格。
- 开了 `prior_waiver` 的 run，换 seed 后增益仍稳健的 warm-start 冠军由 Runtime 判定
  免 confirm（`delivery_resolution.status=waived_by_prior`）。报告与回传必须如实写明
  豁免及其判据，不得写成 confirmed。

收尾前先看 `status` 输出的 `delivery_resolution`：`bound`（或 `waived_by_prior`）
才是正常收尾；`diverged`/`unconfirmed` 先向用户呈报（见「升级到用户」）。停止后
使用确定性 CLI，**不写脚本**：

```bash
# finalize-all = finalize-retrain ∥ finalize-baseline-score 并行；只生成预测，不读标签
python <skill-root>/scripts/modelevo.py finalize-all --run <run_dir> --detach
python <skill-root>/scripts/modelevo.py wait --run <run_dir> --job finalize-all --timeout 570
python <skill-root>/scripts/modelevo.py evaluate-final --run <run_dir>   # 唯一一次 gate；log1p 回归目标可加 --segment-inverse-transform expm1
# report 之前先写 report/journey-narrative.yaml（见下文）
python <skill-root>/scripts/modelevo.py report --run <run_dir>              # report.md + 训练历程 report.html
python <skill-root>/scripts/modelevo.py export-experience --run <run_dir> --model <模型名>   # 提交经验快照
# 然后按 references/experience-library-merge.md：library merge-context → 写决议 → library merge-apply
```

**交付的模型必须是 `finalize-retrain` 的重训模型**。trial 目录里的
`model/model.joblib` 只在 train 上训练，是搜索工件，任何情况下都不是交付物：

- 确认门预期会拦时**也必须执行 `finalize-retrain`**：它会对搜索冠军按同一口径重训到
  `final-model/unconfirmed/model/`（标注未确认、不打 final_test、退出码 1）。
- 禁止：把 `trials/*/model/` 拷进交付包；手写重训脚本；加载冻结 `train.py` 调用
  `fit_and_score`（会覆写冻结工件）。实测事故：手工组包加手写重训，交付轮数与官方
  口径不一致（522 vs 383 轮），且两次覆写冠军的冻结工件。
- 重训口径（frozen_rounds / rerun_recipe）由 Runtime 按冠军早停面自动选择，不要
  显式传 `--refit-mode`。口径与留痕字段见
  [stage-playbook.md](references/stage-playbook.md#交付重训口径)。
- 交付物只有单个冠军配方的重训模型，没有跨 trial 融合通道。

`evaluate-final` 前的两个需要用户理由的放行参数：重训模型在 shadow 上显著回退时
需 `--ack-shadow-regression "<理由>"`；交付位落后搜索冠军超过容差时需
`--ack-divergence "<用户裁决理由>"`。回归任务出现 `level_shift_warning` 时，在报告
与回传中写明"排序可用、绝对金额需校准"。完成前逐项执行
[交付校验](references/stage-playbook.md#交付校验)。

**经验沉淀**：收尾 subagent 先撰写 `experience/exported-global-experiences.yaml`
（成功、失败、边际递减三类结论都要写，口径写进 `condition`），再用 `export-experience`
提交本 run 的快照。退出码 3 表示服务不可达、快照已进 outbox，不阻塞交付，服务恢复后
执行 `library submit-pending`。提交后由同一个 subagent 按
[experience-library-merge.md](references/experience-library-merge.md) 合并：决定每条
候选是新增、并入、按口径拆分、推翻、争议并存还是丢弃，并改写任务卷宗。

**训练历程报告**：`report` 默认另生成 `report/report.html`（`--no-html` 关闭）。
**在 `report` 之前**由收尾 subagent 撰写 `report/journey-narrative.yaml`（冠军路径
每步"为什么有效/证据"、证伪原因、经验要点、业务提醒；schema 见
[stage-playbook.md](references/stage-playbook.md#训练历程报告叙述文件)），缺省时只能
回退到 trial 假设与导出经验卡。

## 按需能力（非默认流程）

**天花板估计**：仅在用户明确要求时执行，Agent 不得自行加入日常迭代。baseline 后执行
`modelevo.py ceiling --run <run_dir> [--engines strong,autogluon] [--budget-minutes 45]`
得到等预算强参照；特征加宽后用 `--from-trial <冠军>` 在冠军特征面上重测。参照只是
测量，不是可交付 trial。细则见 [benchmark.md](references/benchmark.md)。

## 编排与上下文

主会话是调度器。初始化、spec/loader 准备和 profile 由主会话直接前台执行，不创建
subagent。baseline 起的执行阶段、每一代以及最终收尾分别使用独立 subagent；主会话
只读取状态摘要并回传 ≤2K tokens。派发时提供 run/spec 绝对路径、代号、
`planner-context`、`snapshot`、剩余预算及当前阶段所需的参考小节，不要把全部文档、
源码或原始日志塞入 prompt。

profile 之外，主会话只运行状态和诊断类命令，例如 `status`/`leaderboard`/
`snapshot`/`planner-context`/`diagnose`（读状态与诊断）；实际允许面以
`assets/guard_main_session.py` 为准。**异常状态(triage)也受本条约束**：
主会话与 subagent 都先运行 `diagnose`，不得通过 Read/Grep/Glob 或 shell
绕行研读 Harness 源码。仅维护 Harness 本身时才使用
`MODELEVO_ALLOW_SOURCE_READ=1`；用户明确授权主会话直接执行时才使用
`MODELEVO_ALLOW_MAIN_EXEC=1`，并在 run-state 留痕。

subagent 必须完成本阶段并核对状态/产物后才能回传；禁止 `nohup ... &`
后提前结束。单 Trial 滚动执行时使用受跟踪后台任务并以前台
`wait --trial <id>` 等待；普通一代使用 `run-generation`。回传只包含
Trial 摘要、stop 判定、最多五条反思、需要用户处理的问题和启动/完成时刻，
不要粘贴原始日志。

## 故障处置

任何 Trial 缺榜、verdict 异常、产物缺失、baseline delta 为空或 Harness
报错，第一步都是：

```bash
python <skill-root>/scripts/modelevo.py diagnose --run <run_dir> [--trial <id>]
```

按 finding 的 remedy 处理；无 finding 或需要业务判断时升级给用户。不要用
临时脚本替代失败的官方命令；旁证只能标注为非官方补充，不能单独支撑交付。
环境变化写入 `env-ledger.jsonl`，不要因为安装可选包作废整个 run。

Harness 在 run 中途升级时（`harness_hash` 变化），用复评证明分数不变后再登记：

```bash
python <skill-root>/scripts/modelevo.py acknowledge-harness-change --run <run_dir> \
  --measurement-no-score-impact --verify-against <最近若干 trial,逗号分隔> --reason "..."
```

`--verify-against` 用当前 evaluator 复评这些 trial 已落盘的预测（不重训），全部一致
才放行；确认影响口径则改用 `--allow-measurement-impact`（epoch +1）。细则见
[stage-playbook.md](references/stage-playbook.md#harness-变更登记)。

## 升级到用户

**运行中的裁决不阻塞算力**（实测：gen-01 后用 AskUserQuestion 等一个业务裁决，
过夜空等 7.3 h，占 wall 31%）。规则：

- 能预见的裁决（spec 里已写"若…则升级"的对照、资源档位、辅助标签等）在
  **启动期一次问完**；
- 运行中新出现的裁决：写入 `logs/user-decisions.md`（状态 `PENDING`、默认
  分支、各分支影响），**按预注册默认分支继续下一代**（必要时两个分支并行
  作为同代对照），同时告知用户；用户回复后再采纳或回退，不重跑已完成代；
- 只有"继续跑会造成不可逆后果"（触碰 final_test、交付、删除产物、协议
  变更）时才阻塞等待。**交付位分叉属于此类**：`status` 或
  `leaderboard.json.delivery_resolution.status` 为 `diverged`/`unconfirmed`
  时，`evaluate-final` 之前必须先向用户呈报搜索冠军、交付位与差距，不走默认
  分支（实测：gate 被花在落后冠军 1.7×MDE 的陈旧确认上，不可恢复）。

只在以下事项无法由磁盘状态或确定性检查解决时暂停：

- 目标含义、观察/标签时点、指标、切分或 ambiguous 泄漏列需要业务判断；
- leakage suspect、数据质量、回归目标量纲或 uplift 非随机分配需要确认；
- 预算不足、容量诊断显示目标不可达，或需要改变协议；
- 需要级联/融合拓扑：按「全局运行约束」的拓扑门处理，写入 user-decisions 并
  转告用户，不阻塞其它方向；
- 同因连续失败、资源严重超预期，或官方诊断无法给出安全 remedy。
