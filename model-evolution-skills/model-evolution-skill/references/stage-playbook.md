# 阶段手册

本页是执行检查表，不重复专题规则。命令签名与等待见
[protocol-digest.md](protocol-digest.md)，数据协议见
[task-and-data.md](task-and-data.md)，搜索与裁决见
[evolution-and-evaluation.md](evolution-and-evaluation.md)。

## 目录

- [阶段检查表](#阶段检查表)
- [Plan 与 Run](#plan-与-run)
- [Finalize 与反思](#finalize-与反思)
- [Trial Lint 细则](#trial-lint-细则)
- [工作区工具与状态](#工作区工具与状态)
- [交付校验](#交付校验)
- [分桶报表](#分桶报表)

## 阶段检查表

### 1. 初始化与恢复

- 新任务：`init → validate → preflight`，补全 spec 与 loader 后再 profile。
- 恢复任务：先 `status`、`leaderboard`；异常跑 `diagnose`，不要重做 hash
  兼容且已完成的阶段。
- TaskSpec、时间窗口、泄漏分类或指标改变属于协议变更，不与旧结果混排。

### 2. Profile

Profile 的 CLI 是可重入的单命令，没有 `--phase` 参数。`validate` 和 `preflight`
通过后，主会话立即直接前台执行，不创建 subagent，也不在执行前做额外样本分析、
源码阅读或工具探查：

```bash
python <skill-root>/scripts/modelevo.py profile --run <run_dir>
```

只有命令失败时才运行 `diagnose`。按以下方式处理确认门：

1. 首次 `profile` 生成 `leak-classification.md`；检查
   `sample-profile.json.leak_classification.needs_user_confirmation`。为 true
   时，当前材料化结果只视为提案面，禁止继续 baseline；由用户逐列裁决
   ambiguous / reparse-candidate，**并把裁决写回 spec**。裁决有四种去向：
   - 判定为泄漏 → 加进 `features.banned`（附 `reason`，见
     [task-and-data.md](task-and-data.md) 的受控枚举）；
   - 判定为形态不可用 → 加进 `features.reparse`（须补 `plan`）；
   - **判定为保留为普通特征 → 加进 `features.leak_confirmation.kept_ambiguous`**；
   - **判定为机械误封 → 加进 `features.leak_confirmation.reinstated_banned`**。
     撤销的是"这个封禁判错了"；该列必须仍在 banned 里，撤销后回到待裁决并重新出现在复核清单中。

   **复核清单是必读项**：即使 `needs_user_confirmation=false`，
   也必须过一遍 `leak-classification.md` 的「机械提案复核清单」。
   banned-certain 是机械判决而非人工裁决，重点核对
   **`n_unique` 是否超过实体数**（真 ID 不可能超过）以及变长复合文本字段
   （事件序列/JSON/编码串）是否被误判为 `entity_identifier`。报告里的
   「经验库对账」小节会自动标出与已证经验冲突的封禁。
2. 重跑同一个 `profile`，用已确认 spec 覆写材料化 splits、labels、reparse、
   identity，再产出质量、漂移、泄漏、压缩与信息源清单。`kept_ambiguous` 里的
   列不再进提案表，确认门随之关闭。数据面未变时重跑会复用已有四平面（日志
   `write_splits 复用已有四平面(<原因>),跳过重写`），不必为省时间跳过重跑。

结束前确认：四平面可对齐、banned 未落盘、`leak_manifest_hash` 已冻结、
critical finding 已处理，并执行：

```bash
python <skill-root>/scripts/modelevo.py check-products \
  --spec <run_dir>/task-spec.yaml --run <run_dir>
```

字段和视图细节见 [task-and-data.md](task-and-data.md)。

### 3. Adhoc、Baseline 与抽样面

- adhoc 是可选的手工分析阶段，没有 CLI 子命令；只补固定 profile 未覆盖的
  问题，脚本先落盘到 `adhoc/` 再执行。
- baseline 为 gen-00，按 `baseline.algorithms` 运行，不做 HPO。默认 `[xgboost]`，
  需要 LightGBM 对照时显式加上；`baseline.skip_algorithms` 显式排除，
  `baseline.anchor_algorithm` 指定锚点（trial-001；缺省时配置含 xgboost 即取
  xgboost）。warm-start 不“替代”已配置算法，未覆盖的按常规 baseline 补跑。
- baseline 后立即落抽样面（`baseline` 不会自动材料化，特征探索与可选的
  `capacity-check` 都依赖它；产物为 `sample-profile/splits-sample/`）：

  ```bash
  python <skill-root>/scripts/modelevo.py write-splits-sample --run <run_dir>
  ```

- **抽样面噪声预检**：落盘时自动执行，结果在
  `splits-sample/manifest.json.noise_precheck`，stdout 打印一行“噪声预检”。
  `recommend_v1=true`（抽样面 SE > 2×MDE 且 baseline 运行时长 <
  `feature_exploration.sample_precheck_max_full_minutes`）→ 特征探索直接在 v1 上进行。
  预检只是建议，不是门。
- baseline_id 缺失或 `delta_vs_baseline` 全空时（多见于历史 run 或被 kill 的
  中断），先 `diagnose`，必要时用 `tools/backfill_baseline.py` 修复锚点。

**`capacity-check` 是可选的参考诊断**：结论进 `planner-context` §10 与报告，
**不参与 model/loss/hpo 放行**。读数注意：样本曲线“仍陡”常常只说明抽样面比
全量小（全量重训会兑现这部分）；局部歧义指数不是误差下界；“尾部列为负 →
裁列”在经验库中有反例（`exp-202610-5657727f`），只作止损信号。`--seeds`（默认 2）
控制每格点重复数；指纹未变时缓存秒回，`--no-cache` 可强制重算。

门槛未通过前只建 feature/data Trial；不得提前把预算转向模型族或 HPO。

### 5. 模型与 HPO 探索

- 特征期门放行后（见下）才投 model/loss/hpo。
- 配比与预算以配置和 Planner 为准；HPO 在单 Trial 内做 inner 搜索，outer
  validation 只评一次，大数据优先抽样搜索后全量重训。
- crossover 只组合兼容且有证据的父代，必须作为新 Trial 完整重训重评。
- 每代保持数据/信息轴探索；heavy 配额与停止条件以 Runtime 输出为准。

**特征期门**：「先特征、后模型」由 Planner 与 `new-trial` 共用同一判定，与
`capacity-report.json` 无关：

1. **模型方向**：`--action hpo`，或 `--changed-modules` 含 model/loss（不带
   `--changed-modules` 的 hpo 同样计入）；confirm 不算。
2. **特征期内**（已评估的探索代数 < `feature_exploration.min_generations`，
   默认 2，gen-00 不计）：模型方向被拒（Planner 错误码
   `feature_phase_model_locked`）。放行通道两条，都留痕
   `trial.yaml.feature_phase_override`：
   - Planner 在**特征侧已收敛**（信息清单 `unconsumed==0`，或连续两代无新
     accepted）时每代确定性释放**一个**名额，记在 `planner/latest.json` 的
     `feature_phase_gate.released_proposal_id`；`new-trial
     --planner-proposal-id <id>` 认账；
   - 人工 `--override-feature-phase "理由"`。
3. **特征期满后**：不拦，model/loss/hpo 候选按 acquisition 正常竞争，也不需要
   任何 override。`min_generations: 0` 关闭本门。
4. 报告中「未执行的高潜力方向」会**显式列出被特征期门拒绝的候选**。被拒 ≠ 已
   探索过。已证伪的 family 会从候选中剔除，记入 `rejected_candidates`
   （`family_falsified`）与 batch 的 `falsified_families`。

#### 加宽面上的 HPO 必须走 `--from-parent`

**创建 HPO Trial 前先确认父代特征面能不能被 `--hpo-parent` 表达。**
`--hpo-parent` 只继承父代 `config.features` 声明；转型加宽列（`widen_cached` /
`_widen` 产出的 `sq_`、`miss_` 等）是代码 transform 的运行时产物，用它会静默跑在
**基础面**上。父代 `widen_stats.n_new_cols > 0` 时 `--hpo-parent` 会被拒绝并打印替代命令。

规则：

- 父代是**基础面**（无 transform 加宽）→ `--hpo-parent` 或
  `--from-parent` 均可；
- 父代**含加宽 transform**（如已确认的冠军）→ **必须**
  `--from-parent <champ>`（逐字复制父代码，保留 transform；缺四件套时
  harness 自动嫁接模板 hpo 段）；
- 两种路径产出的 Trial 只要 `metrics.json` 的 `n_features` 与冠军一致，
  才算"在该特征面上跑过 HPO"。**只看"有没有 hpo 子代"会把跑错面的
  trial 当成已调参**。

#### 同面 HPO 只做一次、做完整

（曾在同一冠军面连开三轮 HPO，第二轮已触发预注册证伪仍续开。）规则:

- 第一轮就给足预算(默认 40),用 `--from-parent <冠军>`;不要为了"避开 lr
  撞下界"人为抬高 lr 下界——大数据默认 lr 收缩模式,搜索期根本不搜 lr;
- 读 `hpo-study.json.diagnostics`:`boundary_hits` 非空或 outer
  `evaluation.complexity.cap_bound=true`(实训轮数 ≥ cap;`near_cap` 只是提示)
  说明搜索不完整,Runtime 已在同 trial 内尽力外扩;
  仍不完整且 Δ ≥ MDE 时可**续一轮**(热启动复用全部观测,planner 会以
  上一轮增益×0.5 提名);
- 同面上一轮 Δ_parent < MDE 时 `new-trial` 直接拒绝,需
  `--override-hpo-diminishing "理由"` 并在反思里写清"为什么预注册判据不适用";
- confirm 等 HPO 出结果后对**届时的**搜索冠军发(父代有在途 HPO 子代时
  `new-trial` 拒绝,需 `--allow-confirm-during-hpo "理由"`)。

#### 模型轴被截断 ≠ 已探索

`plan` 的批次里若**生成了**模型/HPO 候选却一个都没入选，batch 会带
`model_axis_not_selected`（含被截断的 family、分数与截断阈值）。
读批次时以它为准：**不要在批次里看不到模型候选就判定"模型方向无
潜力"**，更不要据此进入收尾（曾因此漏掉冠军面 HPO 就交付）。

## Plan 与 Run

### 每代计划

1. `snapshot` 获取并发槽位；`planner-context` 获取状态、经验和未消费信息源。
2. `plan --slots N` 落盘 ProposalBatch；所有候选写明 hypothesis、evidence、
   falsification、parents、changed_modules 与 generation。
3. Trial 数在预算允许时填满槽位；参数型差异用 `--from-parent --set`。
4. 先创建整代脚手架，再并行完成 code/config；全部通过 Lint 后一次
   `run-generation`。结果依赖型修改放到下一代。

**HPO 父代的两处快通道检查**：

- `--from-parent <hpo_trial>` 会自动继承父代 `metrics.json.best_params` 并在
  notes 留痕；若打印 best_params 缺失告警（该确认无效）——换父代，不要当成
  确认成功。
- `--hpo-sample-rows` 跑完 HPO 后核对 `metrics.hpo_sample_fallback` /
  `hpo_inner_rows`，确认抽样是否生效。

### 执行纪律

- `run-generation` 是同代 ≥2 个就绪 Trial 的默认入口；逐 Trial 只用于真正
  的滚动依赖。
- **内存准入**：runner 自动排队（见
  [evolution-and-evaluation.md](evolution-and-evaluation.md)）。并发计划按
  `snapshot --run <run_dir>` 的 `admission.suggested_parallel` 填。不要手工错峰，
  也不要为防 OOM 自行串行。`wait` 里的 `ADMISSION_QUEUED` 不是卡死；
  `admission_timeout` 属 infra，重入 `run --trial <id>` 即可。
- runner 负责 CPU/GPU 和进程组；Trial 只用 `n_jobs()` 与模板 GPU helper。
- `run`/`wait` 只解析末尾 `SUMMARY`，原始日志不进入常规上下文。
- 禁止 `nohup` 脱管、`sleep; tail` 轮询、shell `&&` 串行整代。等待一律
  `wait --generation <N> | --trial <id> | --job <tag> --timeout 570`(exit 3 = 未完,
  原样再调),细则见 [protocol-digest.md](protocol-digest.md)。
- 每代 `finalize-generation` 后跑一次 `timing-report`,把"评估 / 特征解析 /
  训练 / HPO 搜索 / 代际间隔"占比写进反思的预算项。
- 兄弟臂对照(如"当日口径 vs 保守口径")用 `contrast --trials ... --ref ...`,
  结果落 `logs/contrast-*.json`,反思直接引用;不写 adhoc bootstrap 脚本。
- Trial 输出、临时分析和接力摘要都写 run 目录，不污染 skill 根目录。

## Finalize 与反思

正常流程每代只调用一次 `finalize-generation`；若命令中断，可用同一代号安全
重入。反思写入 `generations/gen-XX/reflection.md`，至少覆盖：

- key findings、successful/failed/diminishing directions；
- timeout/unverified directions 与下一代计划；
- 结论适用的数据视图 `scope_of_conclusions`；
- gain concentration 或 leakage suspect 的信息流归因；
- 命中经验是否改变计划、信息源清单本代消费/证伪项；
- 特征阶段的八维覆盖与 `feature_exploration_complete`；
- 预算、heavy 配额、并行度 warning 与未执行方向。

是否继续只读取 Runtime 的 verdict、Archive、方向后验与 stop report；停止条件见
[evolution-and-evaluation.md](evolution-and-evaluation.md)。

停止后执行：`finalize-all(= finalize-retrain ∥ finalize-baseline-score) →
evaluate-final → (写 report/journey-narrative.yaml) → report → export-experience`。不要手写收尾脚本。`finalize-all
--detach` 后用 `wait --job finalize-all` 等待;两个子命令的日志分别在
`logs/finalize-retrain.log` / `logs/finalize-baseline-score.log`。

前两个 finalize 命令不读 final_test 标签、不 acquire gate；`evaluate-final`
只 acquire 一次，同时计算冠军指标与基线对照。

### 交付重训口径

`--refit-mode` 缺省按冠军早停面自动选择:

| 模式 | 流程 | 适用 |
|---|---|---|
| `rerun_recipe`(train_tail 冠军的默认) | 从冠军 code/config 副本(`final-model/delivery-recipe/`)以 train+validation 调冠军 `fit_and_score`,**最新尾部** `ES_FRACTION` 早停,单段只训一次,`_last_model` 直接交付 | 所有遵守 `fit_and_score` 契约的血缘,含复合容器(hurdle) |
| `frozen_rounds`(validation 早停冠军的唯一口径;train_tail 冠军可选) | train 上复现 → 冻结最优树数 → `sklearn.clone` → train+validation 关闭早停一次重拟合 | 单一 GBM 与 sklearn 协议容器(官方 `HurdleRegressor`)。不可 clone:validation 早停冠军**自动回落** `rerun_recipe`,留痕 `refit_mode_fallback_reason`;train_tail 冠军显式 `frozen_rounds` → 立即报错。validation 早停冠军显式 `--refit-mode rerun_recipe` 会被拒绝 |

HPO 冠军两种口径都先剥 hpo 节、注入 best_params。回落时 stdout 另打一行
`[refit] 交付重训口径回落…`;实际路径记在 `model_meta.json` 的
`refit_mode_requested/effective`、`refit_mode_fallback_reason`、`clone_probe`
(按[交付校验](#交付校验)核对)。

冠军无 `validation_input.parquet` 或加 `--full-reproduction` 时,`rerun_recipe`
会先在 train 上完整复现(约 40 min)再复跑。重训后与冠军 train-only 模型在
shadow_test 上做配对预检(`model_meta.shadow_precheck`);交付模型显著
更差时 `evaluate-final` 在 gate 前拒绝,需 `--ack-shadow-regression "<理由>"`。
回归任务:交付模型水位偏离时 `level_shift_warning=true`(只诊断,不阻塞),
`evaluate-final` 写 `evaluation.regression_diagnostics`,有告警时 stdout 打印
`[回归水位]` 行——报告须写"排序可用、绝对金额需校准"。
log1p 回归目标可加 `evaluate-final --segment-inverse-transform expm1`,在同一次
标签读取里给 final_test/shadow_test 各出一份原值尺度分桶表
(`final-model/segment-raw/<split>/`)——final_test 标签事后无法再读。

**停止后的交付可能被确认门挡住（不是故障）**：交付位要求存在**绑定搜索
冠军配方**的 `fresh_confirmation.passed=true`(陈旧配方的确认只在
落后冠军 ≤ 容差时可用)。确认失败时 `finalize-retrain` **仍会重训**搜索冠军,
产物落 `final-model/unconfirmed/model/`,附 `UNCONFIRMED.md`；**不写**
`final-model/model/`、`recipe.yaml`、`model_meta.json`，**不打**
final_test/shadow，退出码 1。`evaluate-final` 因交付位缺失而不 acquire gate。

**收尾前必看交付位**:`status` 与
`leaderboard.json.delivery_resolution` 给出搜索冠军、交付位与状态。
`status ∈ {diverged, unconfirmed}` 意味着默认收尾**不会**交付冠军配方 ——
这是用户裁决项,**不适用"按预注册默认分支继续"**:`evaluate-final` 消耗
唯一一次盲测,不可逆动作前必须先向用户呈报(选项:对冠军配方补 confirm /
接受未确认交付 / `--ack-divergence` 明知落后仍交付某个已确认 trial)。
(曾在反思已写明分叉后仍启动收尾,把 gate 花在落后冠军的 trial 上。)harness
会拒绝静默分叉并写 `final-model/DELIVERY-DIVERGENCE.md`。

此时的正确做法：

- **必须照常执行 `finalize-retrain`**，不要因为"预期会被拒"而跳过——跳过
  就连未确认重训模型都没有；只要证据不要模型时才加 `--snapshot-only`。
- 不要用 `--champion` 强指、不要手写重训脚本、不要把 `trials/*/model/`
  （train-only 搜索工件）拷进任何交付包、不要伪造 `final-model/model/`。
- 报告写明哪些交付校验项未通过，并说明可用模型在 `final-model/unconfirmed/`、
  属未确认、无样本外估计。

**确认门失败的归因**必须逐项读 `evaluation.json.fresh_confirmation`
(`failure_reasons` 逐条列出),判据见
[evolution-and-evaluation.md](evolution-and-evaluation.md)。
`legacy_two_sided_equivalence` **只作诊断**。`sigma_seed_source=floor_se_ratio`
时 margin 是保守下限 —— 先看 `multi_seed.status`:`missing` 表示 train.py 没有
multi-seed 段,不是模型问题。已知形态：

- **非劣失败**:父代成绩不可复现(winner's curse)。**不要**靠重抽 confirm
  抹掉它(每配方默认 ≤2 次,超额须 `--allow-extra-confirm`)。
- **保真失败**:子代远优于父代 —— confirm 不是同配方复现(`--allow-code-edit`
  改了配方?),改动应作为搜索 Trial 提交并过代级 FDR。
- **参照失败**:父代宣称的增益在独立 shadow 上不可复现。
- **切片/时序**:对 confirm 只是诊断（`slice_temporal_waived`）。
- **shadow 口径问题**:先查 `metrics.json.shadow_scoring_mode` 是否为
  `nan_native`、`artifact_model` 是否为 `primary_seed`；不是则属旧模板工件问题，
  不是模型退化的证据。

validation↔shadow 绝对 gap 检查是**不阻塞 verdict 的诊断**，与配对非劣/保真门
是两把不同的尺子。

**经验回路写入侧**：`export-experience` 的源是
`experience/exported-global-experiences.yaml`（不是 `task-experience.yaml`）。
收尾 subagent 需要**撰写**这个文件（claim/context/evidence/boundary 四字段
简写即可）。若它缺失，`export-experience` 会以 `no_source` 返回 exported=0——
这是"写入侧断路"，不要当成"本任务无可导出经验"。停止前的最后一代 reflection
应确认该文件已落盘。

## Trial Lint 细则

`init` 会准备工作区工具；每次运行前由 Harness 执行静态 Lint。以下规则由
[assets/lint_trial.py](../assets/lint_trial.py) 实现，文档只解释边界：

| 规则 | 拒绝条件 |
|---|---|
| 算法一致 | trial_id 声明与代码 import 的模型族不一致 |
| 禁融合 | 未经用户 `approve-topology` 批准时出现多 GBM 库、stacking、voting 或多 seed 平均(主检测是 runner 的拓扑门,见 [template-api.md](template-api.md);`spec.ensemble_enabled` 不授权) |
| dtype | 无基数/数值化防护的 `get_dummies` |
| 任务语义 | 回归写概率(规则 4:`predict_proba`/sigmoid 的结果**实际流入** score 写出路径才拒绝;hurdle `p·μ` 不算概率口径);uplift 未使用 treatment |
| 模型工件 | 未调用 `dump_model` 或等价持久化 |
| HPO 隔离 | 超参搜索/选择读取 outer validation label(规则 7);train_tail 口径下早停 eval_set 用 outer validation(规则 8) |
| 泄漏三连 | 引用 banned 字面量、删除 `leak_scan`、直接调用 `load_data()` |
| 脚手架 | `fit_and_score` 占位标记仍在(规则 10,不分 model_type;回归 `new-trial --algorithm xgboost\|lightgbm\|catboost` 已渲染真实回归器) |
| 评估帧 | 加宽特征但未调用 `dump_eval_frame` |
| 配置消费 | config 有 hyperparameters 但代码未读取 |
| 名称绑定 | 模块或函数内存在先用后导/未定义名 |
| Runtime bootstrap | HPO 嫁接代码无法导入 Runtime |
| HPO 死嫁接(规则 7) | config 有 hpo 节但 `main()` 不调用 `fit_and_score` |
| 特征缓存(规则 21) | reparse train ≥ 20 万行,代码消费 reparse 却无 `feature_block(`/`widen_cached(` |
| 输出契约(规则 22) | 代码里没有 `validation_score.parquet` 落盘 |

仅只读分析 Trial 可使用文档化的 lint 豁免标记；不能为跑通训练任意关闭
规则。Lint 失败先修 code/config 或新建 child Trial，不手改裁决产物。

## 工作区工具与状态

### 工具

| 工具 | 用途 |
|---|---|
| [lint_trial.py](../assets/lint_trial.py) | Trial 运行前的静态硬门 |
| [backfill_baseline.py](../assets/backfill_baseline.py) | 修复历史 run 缺失的 baseline 锚点 |
| [guard_main_session.py](../assets/guard_main_session.py) | 限制主会话执行面，并阻止主会话/subagent 直接研读 Harness 源码 |

Guard hook 的具体安装随宿主环境变化，以工具文件顶部说明为准；维护 Harness
时才使用 `MODELEVO_ALLOW_SOURCE_READ=1`，用户授权主会话执行时才使用
`MODELEVO_ALLOW_MAIN_EXEC=1`。guard 只对 `python …modelevo.py <非批准子命令> --长选项 "引号文本"` 里的
**引号参数值**按惰性文本放行;真实的批准调用、命令替换(`$(...)`/反引号)、heredoc、
`bash -c`、写 `approvals/` 一律拒绝。

### 缓存(feature-cache)

```bash
python <skill-root>/scripts/modelevo.py feature-cache --run <run_dir>   # 列表 + 失效预判 + 子模型缓存汇总
python <skill-root>/scripts/modelevo.py feature-cache --run <run_dir> --clear            # 清空 .feature-cache
python <skill-root>/scripts/modelevo.py feature-cache --run <run_dir> --clear-submodel   # 清空 .submodel-cache
```

子模型缓存 `RUN_DIR/.submodel-cache/` 没有自动回收,占用大时用 `--clear-submodel`。
两类缓存都可重算,不碰任何裁决产物。

### Harness 变更登记

`harness_hash` 变化后(skill 升级或热修),在途 run 先复评、后登记,不手改
`run-state.json`:

```bash
python <skill-root>/scripts/modelevo.py acknowledge-harness-change --run <run_dir> \
  --measurement-no-score-impact --verify-against <trial,trial,...> --reason "..."
```

1. 命令先打印**特征块缓存预判**(`特征块缓存预判: …`)。
2. `--verify-against`:用当前 evaluator 对所列 trial 已落盘的预测重新评估,不重训。
   建议选最近 2~3 个有 vs_parent 的搜索 trial。
3. 全部一致才放行 `--measurement-no-score-impact`(epoch 不变);任一不一致即拒绝并
   列出差异,确认改变口径时改用 `--allow-measurement-impact`(epoch +1,新旧不混排)。
   `harness_hash` 未变时 `--verify-against` 也照常复评(只是不登记),不一致 exit 1。

### 状态语义

- run phase 单调推进：`INITED → PROFILED → BASELINED → EVOLVING →
  FINAL_EVALUATED/STOPPED`；Trial 失败不回退 run phase。
- `STOPPED` 是建议性状态,硬拦截只看 `stop_report.blocks`,见
  [evolution-and-evaluation.md](evolution-and-evaluation.md)。
- 并行时 `current_trial` 只是最近启动项；真实状态以各自 `trial.yaml` 为准。
- `run-state.json.lock` 是并发锁，正常存在，不要删除。
- Trial 生命周期为 `PROPOSED → CODE_WRITTEN → RUNNING → SUCCEEDED/FAILED →
  EVALUATED → REFLECTED`；同因连续失败触发熔断与 root-cause。

## 交付校验

- [ ] 所有 Trial 可由 code + config 复现；leaderboard 含 failed/invalid 与原因
      （`verdict_reason` 列带 `evaluation.reason`，report §7 有"未通过 trial 与原因"
      清单；只查 `failure_reason` 会漏掉全部方向性拒绝）。
- [ ] accepted Trial 通过 Lint、完整性与泄漏检查；模型工件可加载。
- [ ] baseline 锚点存在，`delta_vs_baseline` 非空（跨视图 Trial 例外：
      应记 `baseline_incomparable`，不得因此判 invalid）；预算双口径和停止原因可查。
- [ ] 特征阶段有 sample manifest、bundle、分组置换、条件消融（Agent 手建）、
      八维覆盖、reparse 清账、全量复核和独立 confirm；`generation.yaml.
      feature_beam.underfilled` 若 >0 有可解释原因（候选供给薄 or 编排漏建）。
- [ ] 所有数据视图脚本及 source hash 留存；banned 未进入四平面，身份面未进模型。
- [ ] 冠军来自可排名候选，`fresh_confirmation.passed=true`，shadow artifact、
      sample key、源码 hash、实际 seed 和 recipe 一致（shadow 分数落盘即带
      `__mvo_sample_key`；缺该列说明工件来自旧模板）。
- [ ] **交付位绑定搜索冠军**:`delivery_resolution.status=bound`
      (或 `stale_within_tolerance` 且差距已写进报告);`diverged/unconfirmed`
      时 evaluate-final 之前已向用户呈报并取得裁决,不走默认分支;
      `model_meta.json.search_champion` 与 leaderboard 的搜索冠军一致。
- [ ] `finalize-retrain` 产出的 champion 与 evaluation/model_meta/recipe 一致；
      基线 final_test 对照也已生成。
- [ ] **交付模型是重训模型**：`model_meta.json` 的 `delivery_status`、
      `n_train_rows + n_validation_rows == n_train_validation_rows`、
      `final_model_sha256`（= 磁盘实际哈希）≠ `trial_model_sha256`；
      `refit_mode=rerun_recipe` 时 `face_check.passed=true`(或
      `reproduction.passed=true`)、`es_holdout` 落在 validation 期、
      `delivery_recipe_sha256` 与冠军源码一致、`shadow_precheck` 未显示显著
      回退(否则有用户 `--ack-shadow-regression` 留痕);`frozen_rounds` 时
      `reproduction.passed=true`（或 HPO 冠军的 `retrain_builder=estimator_adapter`）;
      `estimator_class` 与冠军 trial 模型同一算法族；交付包里
      没有任何来自 `trials/*/model/` 的文件。确认门未过时，可用模型只在
      `final-model/unconfirmed/model/`，报告须标注"未确认、无样本外估计"。
- [ ] **refit 口径留痕一致**:`model_meta.json` 的 `refit_mode_effective`
      == `refit_mode`;`refit_mode_fallback=true` 时 `refit_mode_fallback_reason` 非空,
      且报告 §12/§14(h) 已写明回落;`clone_probe.status` 与实际路径一致(`error` 时
      不应出现 train 上复现耗时);`skill_version` / `generated_by` 为当前版本;
      `FINALIZE_RETRAIN_SUMMARY` 行与 `model_meta` 一致。
- [ ] **回归水位诊断**(仅回归):`final-model/evaluation.json.regression_diagnostics`
      存在(或带 `error` 说明);`level_shift_warning=true` 时报告 §14 已写明"排序可用、
      绝对金额需校准"及偏离期;`shadow_precheck.level_stats` 与之方向一致。
- [ ] `final-test-gate.json` count ≤ 1；final 失败不退款，结果未回流搜索。
- [ ] final-model 中模型、特征列、预测及 hash、segment report、evaluation 和报告齐备。
- [ ] Task Experience 已更新；跨任务经验仅导出有本地证据和边界的条目；
      `experience/exported-global-experiences.yaml` 已撰写（否则导出为 0）。
- [ ] 用户报告说明提升、稳定性、适用范围、未执行方向和剩余不确定性。
- [ ] `report/report.html`(训练历程)已生成,冠军路径每一步都有"为什么有效 + 证据"
      (来自 journey-narrative.yaml,而非仅自动回退);页脚无"分项降级"告警,或已说明原因。

### 训练历程报告叙述文件

`report` 读取可选的 `report/journey-narrative.yaml`,为 HTML 报告提供说明文字;键均可省略,
trial 可写完整 id 或三位短号(如 `"031"`):

```yaml
headline: 一句话结论(页首)
steps:            # 冠军路径上的步骤(非 confirm),按 trial 键
  "031": {title: 第二段在 100% train 定轮重训, why: 为什么有效(1~3 句), evidence: [证据 1, 证据 2]}
falsified:        # 证伪/递减方向
  "024": {title: tweedie loss, why: MAE 变好但 RMSE +0.0058}
labels:           # 图上短标签(进化地图/森林图)
  "019": +seqb2_ 30 天周期相位窗
insights:         # 经验要点卡片
  - {title: 零膨胀改结构不改 loss, text: ...}
notes:            # 业务口径提醒(黄色提示框)
  - 模型输出为 log1p 金额,适合排序,不能直接 expm1 当金额
```

写作要求:每条结论能在 leaderboard / evaluation / reflection 中找到对应数字;不写
未经确认的推测;Δ 的符号与主指标方向一致(页面会另行展示方向归一化增益)。

## 分桶报表

每个 Trial 与最终评测会自动生成 `segment-report.xlsx` 和
`segment-report.md`；桶按 score 降序，桶 1 为最高分段。分类、回归与 uplift
的列定义分别见对应的 task-type 文档。缺少 openpyxl 时降级 CSV；样本不足或
score 退化时不生成报表，但不影响主评估。

**回归 `Bias` 列口径**:桶内 **预测 − 真实** 的均值,正 = 高估、负 =
低估,与 `evaluation.raw_scale.mean_bias`、报告的"均值偏置"同号。早期 run 落盘的
旧报表是 **真实 − 预测**(符号相反,不追溯重算),跨 run 对比 Bias 时先看报表生成版本。
