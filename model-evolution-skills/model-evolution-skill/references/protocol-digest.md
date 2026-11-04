# 命令与编排契约

本页只回答三件事：命令怎么调、长任务怎么等、subagent 怎么交接。数据、
评估、停止与模板细节分别以 [task-and-data.md](task-and-data.md)、
[evolution-and-evaluation.md](evolution-and-evaluation.md) 和
[template-api.md](template-api.md) 为准。

## 目录

- [调用约定](#调用约定)
- [常用子命令签名表](#常用子命令签名表)
- [Trial 创建与产物](#trial-创建与产物)
- [等待与调度](#等待与调度)
- [Subagent 体量护栏](#subagent-体量护栏)
- [派发与异常回传](#派发与异常回传)

## 调用约定

统一入口：

```bash
python <skill-root>/scripts/modelevo.py <command>
```

- 跨目录调用显式传 `--run <run_dir>`；在 run 目录内可省略。运行目录按
  `--run > --spec > MODELEVO_RUN_DIR > cwd` 解析。
- `status`、`leaderboard`、`snapshot`、`planner-context` 是只读命令，均接受
  `--run`/`--spec`。
- 异常第一动作是 `diagnose --run <run_dir> [--trial <id>]`。输出
  `code/severity/detail/remedy`；exit 1 表示存在 critical。**主会话与 subagent 同规**：
  禁止为 triage 直接 Read/Grep/Glob `runtime/` 源码；
  只在维护 Harness 时用 `MODELEVO_ALLOW_SOURCE_READ=1`。
- `preflight` 记录 `env-ledger.jsonl`。安装或出现可选依赖不作废 run；
  Harness、数据协议或泄漏分类变化才需要新协议或新 run。
- `profile` 是主会话可直接执行的确定性命令。`validate`/`preflight` 通过后立即
  前台调用，不派 subagent，不先做额外工具探查；失败才转 `diagnose`。

## 常用子命令签名表

| 命令 | 关键参数 | 契约 |
|---|---|---|
| `init` / `validate` / `preflight` | `--dir` / `--run` | 建目录、校验 spec、检查依赖 |
| `profile` / `baseline` / `capacity-check` | `--run` | 建数据面、基线；天花板诊断可选，仅供参考、不参与放行 |
| `status` / `leaderboard` | `--run` | 读阶段与成绩 |
| `snapshot` | `[--run <run_dir>]`(缺省按 cwd 解析) | 读资源、GPU 占用与并发计划；并发按 `admission.suggested_parallel`(内存准入口径) |
| `planner-context` | `--run [--top-k N]` | 汇总 Planner 所需状态(MDE、晋级门预览、噪声预检、§7 经验库视图)，不靠人工拼 prompt |
| `plan` | `--run --slots N` | 校验并落盘 ProposalBatch；confirm 只提名冠军配方(或容差内候选)；物化命令预填 `--topics`/`--basis` |
| `new-trial` | `--algorithm X` 或 `--from-parent ID` | 创建 trial；显式传 `--generation N`；**`--basis <原则id>:<follows\|extends\|tests\|contradicts>[,…]` 或 `--basis novel` 必填**(子代未给则继承父代)，`--topics` 标主题；`--topology` 声明拓扑；`--changed-modules` 只取 `feature/sample/model/loss/data`，未知值 exit 1 |
| `new-trial --add-blocks` | `<trial>:<block>[,<trial>:<block>...]` | 字节级复制源 trial 的特征块段并校验指纹(保证缓存命中，不一致拒绝)；不可用于 confirm/automl |
| `new-trial --hurdle-template` | `[--action explore\|exploit\|crossover\|hpo]` | 渲染官方 `HurdleRegressor` trial(仅 regression/xgboost，隐含 `--topology structural:hurdle`，不与 `--from-parent` 同用) |
| `new-trial --action hpo` | `--search-space FILE --budget-configs N` | inner 搜索、outer 只评一次；大数据可加 `--hpo-sample-rows`；可选 `--hpo-lr-mode/--hpo-parallel/--hpo-fidelity/--hpo-no-expand/--hpo-no-warm-start/--hpo-search-lr/--hpo-lr-shrink-factor`；同面上一轮 Δ<MDE 时需 `--override-hpo-diminishing "理由"` |
| `run` | `--trial ID [--detach]` | 阻塞到终态并输出一行 `SUMMARY\t...`;拓扑未获批准时输出 `TOPOLOGY_BLOCKED`,exit 5,不训练 |
| `wait` | `--trial ID [--timeout SEC]` | 等当前 attempt 的终态；超时输出 `WAIT_TIMEOUT`，exit 3，不 kill，原样再调；`TOPOLOGY_BLOCKED` 为 exit 5；`ADMISSION_QUEUED` 表示在内存准入排队，不是卡住；自身被杀(137/143)原样重调，训练不受影响 |
| `run-generation` | `--generation N [--wait-code SEC] [--detach]` | 整代 barrier 并行，逐个内存准入后启动；跳过尚未写码的 PROPOSED trial、拓扑未获批准的 trial(摘要 `topology_blocked`)和已在单独运行的 trial(摘要 `already_running`) |
| `approve-topology` | `--trial ID [--scope trial\|recipe\|run --kind K] [--revoke ID] [--list]` | **仅用户在交互式终端执行**;Agent 只可 `--list`(guard hook 拒绝其余形态) |
| `finalize-generation` | `--generation N [--unexplored "..."]` | 每代调用一次，输出 archive、stop 与并行度摘要 |
| `finalize-retrain` | `[--champion ID] [--snapshot-only] [--ack-divergence "..."] [--refit-mode rerun_recipe\|frozen_rounds] [--full-reproduction]` | 交付位在 train+validation 重训(默认 `rerun_recipe`)；确认门未过或交付位分叉时重训到 `final-model/unconfirmed/`，exit 1(分叉另写 `DELIVERY-DIVERGENCE.md`)；stdout 一行 `FINALIZE_RETRAIN_SUMMARY\t{json}`。口径见 [stage-playbook.md](stage-playbook.md#交付重训口径) |
| `finalize-baseline-score` | `[--baseline ID]` | 基线重训并产出 final_test 对照分数 |
| `evaluate-final` | `[--best-trial ID] [--ack-divergence "..."] [--ack-shadow-regression "..."] [--segment-inverse-transform expm1]` | 默认确定性选交付位；显式指定也不能绕过确认、完整性与 gate；交付模型落后搜索冠军超过容差、或 shadow 预检显著回退时在 gate 前拒绝。见 [stage-playbook.md](stage-playbook.md#交付校验) |
| `report` | `--run [--no-html]` | 生成交付报告(report.md + report.html，可读 journey-narrative.yaml) |
| `export-experience` | `--run [--model M]` | 提交本 run 的经验快照；exit 0 已提交、1 导出文件 schema 错误(合法条目已提交，按报错的条目/字段修)、3 服务不可达已进 outbox |
| `experience search` / `get` / `evidence` / `topics` | `--run [--topics T[,…]] [--hypothesis "…"]` / `<原则id>` | 按主题检索原则与同主题已失败方向、看原则全文与证据、看主题词表(evidence 需服务可达) |
| `library merge-pending` / `merge-context` / `merge-apply` / `merge-abort` | 见 [experience-library-merge.md](experience-library-merge.md) | 收尾 subagent 把本 run 快照合并进全局原则(`merge-apply --dry-run` 先校验) |
| `library status` / `submit-pending` | `[--run <run_dir>]` | 经验库服务健康与水位;补交 outbox 快照。`library serve`/`stop` 由用户在终端执行 |
| `library leaderboard` / `recipe export` / `browse` | `--atid A` / `<配方id> --out D [--run R]` / `--out D` | 只读：榜单、导出明文配方(给 `--run` 渲染成可运行副本)、拉明文镜像 |
| `write-splits-view` | `--view-id ID (--template recency\|active\|kwin \| --source F \| --from-feature-block T:B[,...]) [--max-mem-gb G] [--chunk-rows N] [--no-stream] [--note "..."]` | 构建数据视图(默认流式，build_view 须为行内变换)；`--from-feature-block` 把已缓存特征块材料化为视图，与 `--source/--template` 互斥 |
| `write-splits-sample` | `[--max-rows N] [--note "..."]` | 抽样面落盘，自动噪声预检 → `manifest.json.noise_precheck` |
| `prescreen` | `--trial ID [--ref ID] [--seeds K] [--holdout-frac F] [--es-frac F] [--max-rows N] [--gpu I] [--n-boot B] [--threads T] [--timeout-min M] [--keep-sandbox]` | 只用 train 在沙箱配对多 seed 跑候选与对照(缺省第一父代)，输出 `prescreen/<trial>/report.json`；仅筛选证据，不进 FDR/leaderboard/方向台账，不改状态 |
| `feature-cache` | `[--clear] [--clear-submodel]` | 列出特征块缓存、失效预判与子模型缓存；`--clear` / `--clear-submodel` 分别清除 |
| `acknowledge-harness-change` | `--reason "..." [--measurement-no-score-impact \| --allow-measurement-impact] [--verify-against T[,T...]]` | 登记 harness 变更(替代手改 run-state)；`--verify-against` 复评已落盘预测，全部一致才放行 `--measurement-no-score-impact` |
| `delete-trial` | `--trial ID --reason "..."` | 只删除未启动且无依赖的误建 trial；禁止 `rm -rf trials/...` |

`run-generation` 已**反转为默认路径**：同代有两个及以上就绪 Trial 时一次
提交；只有后一个 Trial 必须读取前一个结果才能设计时，才滚动执行。

## Trial 创建与产物

### 创建路径

- 只改配置：`new-trial --from-parent <id> --set key=value`，不写代码。父代
  无超参消费点时创建期拒绝(exit 1、不留 trial)，除非加 `--allow-code-edit`
  后自行补消费点；diagnose 对已建的此类 trial 报 `hyperparameters_unconsumed`。
- 改特征或模型结构：普通 `new-trial`，按 [template-api.md](template-api.md)
  实现并先过 Lint。
- HPO：冠军面(同算法)用 `--action hpo --from-parent <id>`；基础面或异算法
  用模板渲染 `--action hpo --algorithm ... [--hpo-parent <id>]`。`plan` 会在
  `planner/default-search-space-<algo>.yaml` 生成合法默认空间；inner 只能读
  train 内切分。**一个预算充足的 HPO trial 代替多轮手工追边界**；hurdle 父代
  与搜索键写法见 [template-api.md](template-api.md#hpo)。
- 独立确认(confirm 的假设是确定性的,**可省略 `--hypothesis`**；其余动作缺省
  即报错退出)：`new-trial --from-parent <candidate> --action confirm
  --generation <N>`。只能指定一个通过代级选择校正且源码完整的父代；Harness
  自动换 seed 并保留父代代码/transform。每配方默认最多
  完成 2 次 confirm,超额须 `--allow-extra-confirm "<用户裁决理由>"`;父代
  train.py 无 multi-seed 段(`seed_scores`)时会告警。
- 数据视图：先 `write-splits-view`，再以 `--action data_view --data-view`
  创建；特征抽样面用 `write-splits-sample --max-rows 200000`。
- 复用特征块:`new-trial --from-parent <id> --add-blocks <trial>:<block>`
  ——不要手拼代码段或手写缓存核验脚本。
- 零膨胀回归:`new-trial --hurdle-template [--action hpo ...]` 渲染官方
  `HurdleRegressor`;hurdle 父代的子代直接 `--from-parent`。
- `--from-parent` 表示代码/配置载体，`--parents` 只表示血缘，不能互换。
- 依据:每个新 trial 用 `--basis` 写明依据的经验库原则与关系,或 `novel`;
  起草假设后先 `experience search --topics …` 再建 trial。`plan` 物化命令已预填,
  改了方向就改依据。细则见 [experience-and-memory.md](experience-and-memory.md#依据每个新-trial-必须写明--basis)。

### run 前最低产物契约

1. `validation_score.parquet`：含 `score`；分类为正类概率，回归为预测值，
   uplift 为 CATE。
2. `metrics.json`：含 `model_num` 与 `leak_scan`；HPO 另记搜索摘要。
3. `model/model.joblib`、原生模型与 `feature_columns.json`。
4. `feature/feature_importance.csv`；加宽特征 Trial 另调用
   `dump_eval_frame(X_va)` 产出 `model/validation_input.parquet`。
5. confirm Trial 用新 seed 重训并写 `shadow_test_score.parquet`；普通搜索
   Trial 的 shadow 结果不得进入反馈。

模型线程取模板 `n_jobs()`，GPU 取 `MODELEVO_GPU`，禁止硬编码设备或线程。
特征只能消费 splits、labels、reparse 三个授权信息面；identity 只供框架
对齐，Trial 禁止 `load_data()`。完整 API 见 [template-api.md](template-api.md)。

## 等待与调度

- 分钟级确定性命令直接前台跑完。
- 同代多个 Trial：**带 `--detach`** 启动 `run-generation`，再用
  `wait --trial <id>` 或 `status` 追踪；也可前台阻塞到 barrier 结束
  (此时同样建议 `--detach`)。不要用 shell `&&` 串行。
- 滚动单 Trial：`run --trial <id> --detach` 后跟一条 `wait --trial <id>`。
  禁止 `nohup ... &` 后空等通知，也禁止 `sleep; tail` 轮询日志。
- **长时训练一律 `--detach`**：训练不随 agent 会话退出而终止(日志在
  `logs/detached-*.log`)；不带则会话边界会杀死训练。
- `wait` 返回 `STALE_RUNNING` 时重入 `run --trial <id>`；有分数则只补
  评估，无分数则回到可训练态(HPO 只补未评的 config)。`WAIT_TIMEOUT` 不会终止训练。
- `diagnose` 的 `lb_rows_absent` 包含已有分数的 Trial 时，重入
  `run --trial <id>`：EVALUATED/RUNNING Trial 会走 evaluate-only，并由
  命令末尾重建 leaderboard；不得手工补行或再次训练。
- `prefix_rate_rel_dev > 0.2` 时前缀样本存在标签偏差风险，Runtime 会退化
  到**分层**抽样；解读旧 quick/smoke/capacity 产物时核对 manifest。
- **内存准入**:`wait` 打印 `ADMISSION_QUEUED` 表示 trial 在排队;准入超时
  FAILED(`admission_timeout`)时重入 `run --trial <id>`(不耗 retry 预算)。
  并发按 `snapshot` 的 `admission.suggested_parallel` 规划,不手工错峰。
  口径见 [evolution-and-evaluation.md](evolution-and-evaluation.md)。

### 代间流水线

上代 barrier 运行时可以**预起草**本代 reflection 骨架与下一代命令的
**预案分支**，barrier 结束后再填入真实 verdict。confirmation 因依赖最终
冠军和特征面，属于**硬依赖串行**，不得预建或猜测裁决。

## Subagent 体量护栏

- 一个 generation subagent 只负责一代 `plan → build → run → finalize →
  reflection`；跨代时落盘 `logs/<id>-next.md` 后结束，由新 subagent 接力。
- 连续多次同主题探查时改用 `planner-context`、`diagnose` 或专题文档，不逐段
  翻源码。
- 代际交接依赖落盘产物；最终回传不超过 2K tokens，不贴原始日志。
- 长训练仍遵守上面的确定性等待姿势；结束回合不是等待机制。

## 派发与异常回传

Generation subagent 的 prompt 只需包含：skill/run/spec 绝对路径、代号、
`planner-context`(第 7 节即经验库视图,不另行粘贴经验)、`snapshot`、上一代
reflection、剩余预算、已试数据视图，以及当前阶段的参考小节。写码时附 [template-api.md](template-api.md)，不要
内嵌全部 references。

回传固定为：Trial 摘要(含每个 trial 的依据)、stop 判定、最多五条反思、用户待决
事项、启动与完成时间。

收尾 subagent 在 `report` 之后依次执行:撰写 `experience/exported-global-experiences.yaml` →
`export-experience --run <run> --model <模型名>` → 按
[experience-library-merge.md](experience-library-merge.md) 完成合并(exit 3 时回传
"快照在 outbox,待服务恢复后 submit-pending 并合并",不阻塞交付)。泄漏疑点、final_test gate 违规、不可达目标或 `diagnose` 的
critical remedy 超出授权时升级给用户。
