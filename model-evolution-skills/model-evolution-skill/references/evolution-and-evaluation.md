# 演进与评估协议

本页是搜索、晋级裁决和停止语义的单一文档权威。统计裁决全部由 Runtime 计算,
Agent 只读结果字段、不自行计算或宣布 accepted。命令签名见
[protocol-digest.md](protocol-digest.md)，特征阶段细节见
[feature-exploration.md](feature-exploration.md)。

## 目录

- [搜索分期与动作](#搜索分期与动作)
- [Planner 与 Archive](#planner-与-archive)
- [Crossover](#crossover)
- [Evaluator](#evaluator)
- [特征贡献证据](#特征贡献证据)
- [生命周期、失败与资源](#生命周期失败与资源)
- [停止条件](#停止条件)

## 搜索分期与动作

搜索分两期：

1. **特征探索期**：baseline 后开始；模型和超参冻结，100% 预算用于
   feature/data_view。以 bundle、分组置换和条件消融产生可归因证据。
2. **模型/超参探索期**：只有特征充分性四门槛全部通过后才进入；抽样面
   晋级方向先在 v1 全量面复核并完成 fresh confirmation。

特征收敛只是阶段转换信号，不是全局停止信号。

| 动作 | 用途 | 约束 |
|---|---|---|
| `explore` | 新特征族、模型族、损失或任务分解 | 一次只改一个主要组件 |
| `exploit` | 已验证方向的窗口、筛选或正则细化 | 必须有父代证据 |
| `crossover` | 合并互补的 accepted 改动 | 作为新 Trial 重训重评 |
| `data_view` | 重建聚合口径、复活被压缩的信息 | 先材料化并校验视图 |
| `hpo` | 一个 Trial 内做多配置 inner 搜索 | outer validation 只评一次 |
| `confirm` | 新 seed 独立复现候选 | validation 与 shadow 均非劣且保真(margin 含 seed 方差);每配方默认 ≤2 次 |
| `automl` | 外部 AutoML 参照 | reference-only，不参与冠军选择 |

参数差异不应拆成大量手工 Trial：HPO 用 `--action hpo`；纯配置开关用
`--from-parent --set`。特征期内（已完成探索代数 < `min_generations`）
model/loss/hpo 需 Planner 释放名额或 `--override-feature-phase` 理由。HPO 预估成本超过 heavy
预算门时应缩小 configs、抽样搜索或显式记录豁免。

## Planner 与 Archive

每代先运行确定性 Planner：

```bash
python <skill-root>/scripts/modelevo.py plan --run <run_dir> --slots <N>
```

ProposalBatch 必须通过：

- final_test 禁用、父代合法、候选去重、预算/资源可行；
- hypothesis 可证伪，action、changed_modules、数据视图和 generation 完整；
- 特征期禁止 model/loss/hpo（特征期门）；heavy 配额满足。

leaderboard 记录全部结果；Archive 只保留值得继续搜索的候选，并永久保留 global best。

读方向台账与提名时:

- `directions.json` 的 `mean_delta`/`lcb_delta` 是**方向感知增益**(越大越好,
  rmse 等已取 −Δ);原始 Δ 在 `mean_raw_delta`。
- 列集与父代相同的换 seed 复测记 `replicate:seed`,其 Δ 是 seed 噪声,不据此 exploit。
- confirm 提名只给搜索冠军配方上已过校正、未确认的父代(理由见 `rationale` /
  `confirm_nomination`)。
- 所有 MDE 均为量纲感知值,由 Runtime 计算;planner-context 顶部出现
  `scale_warning` 时说明 MDE 与指标量纲不匹配。

## Crossover

Crossover 不允许把父代收益相加当作组合收益。创建前检查：

- changed_modules 冲突、样本定义和特征口径不一致；
- 父代数据视图差异、强交互风险、重复解决同一问题；
- 该组合是否已经实验过。

跨视图组合必须声明基准视图，并在新 Trial 内重新对齐另一父代的工程逻辑。
跨任务经验只能形成假设，不能直接注入最终模型。多父代 crossover 的 vs_parent
口径见[晋级对照与 alpha 排程](#晋级对照与-alpha-排程)。

## Evaluator

### 数据访问层级

`train → validation → shadow_test → final_test`：

- 调参(超参选择)只在 train 内部完成;早停默认在 validation 上;
- validation 是搜索反馈面;
- shadow 只供新 seed confirm，不参与候选生成或排名；
- final_test 标签不落盘，冻结配方后由 FinalTestGate 在内存读取一次；获取后
  即使失败也不退款。

预测 artifact 必须带唯一 sample key 和 source/split/model/prediction hash；
键集合或协议 hash 不一致时禁止配对比较。

### 指标语义

- binary：score 为正类概率；
- regression：score 为预测值，rmse/mae/mape 自动按越小越好处理；
- uplift：score 为 CATE，可正可负，bootstrap 同时携带 treatment。

更细的指标和分桶列见对应 task-type 文档。

### Noise-aware gate 与 verdict

提升量使用按 sample key 严格配对的 bootstrap：

- `delta < 0.5 × noise`：likely noise；
- `0.5 × noise ≤ delta < noise`：promising；
- `delta ≥ noise`：进入候选验证。

单 Trial 通过后仍需 Runtime 的代级多重校正才能 accepted。
搜索 Trial 的 verdict 只有 `accepted / promising / rejected / invalid`，
confirm Trial 只有 `confirmed / unconfirmed / invalid`,只信 `evaluation.json`。
rejected 是有效证据；invalid 表示泄漏、运行异常或评估不完整；unconfirmed
表示"同配方换 seed 的合法模型未通过独立确认",**不是**"模型差",不计方向
失败、不进冠军/Archive 池。无父代的 root Trial 与 baseline 做同尺配对 bootstrap。

稳定性、最差切片、时序回归和复杂度共同参与裁决。`model_num` 从
`metrics.json` 进入复杂度与报告；单模型为 1，多模型按实际交付数量记录。

#### 晋级对照与 alpha 排程

裁决口径以本 run `modelevo-config.yaml` 为准(在途 run 不换尺):

- `evaluator.promotion_reference`:`anchor`(新 run)时代级校正按 vs 交付锚点计账,
  另要求子代不显著差于直接父代、通过质量门;未过则降回 vs_parent 单试验 verdict
  (accepted → promising)。`parent`(老 run)按 vs_parent 计账,小步链可能永远
  只是 promising。
- 结果字段:`delta.vs_anchor`、`delta.anchor_id`;`selection_correction.passed` 与
  `promotion_blocked_by` 说明为何未晋级;`anchor_incomparable=true` 时该 trial 按
  parent 口径。`evaluation.parent_verdict` 是校正前的 vs_parent verdict。
- `evaluator.crossover_parent_test: iut`(新 run)时多父代 crossover 的 vs_parent 取
  最保守的父代(明细 `delta.vs_parents_iut`)。
- planner-context §1b 给出下一代晋级门槛预览(`q_next`、z 门槛、`delta_hint`):
  用它判断一步要改善多少才可能晋级。

### Fresh confirmation

confirm Trial 必须使用新 seed 重训，并同时产出 validation 与此前未进入反馈
的 shadow 分数。判据由 Runtime 计算(结果在 `fresh_confirmation`):

- **非劣**:validation 与 shadow 上均不显著差于父代(margin 含 seed 方差);
- **保真**:子代远优于父代时判不通过——那不是同配方复现,配方改动必须作为
  搜索 Trial 过代级校正;
- **参照对照**:shadow 上不得显著差于父代当初的比较对象(锚点或祖父代);
  `gain_reproduced` 只是诊断;
- multi-seed 稳定性、源码完整性、样本键与 hash 全部通过。

确认门只看 `fresh_confirmation.passed`;`shadow_test.gap_sanity_passed` 是非阻断诊断。

**交付位必须绑定搜索冠军配方**:只有与搜索冠军同配方(沿 confirm 边回溯
的配方根相同)的 `fresh_confirmation.passed=true` confirmed Trial 才能成为
交付冠军;其它配方的陈旧确认仅在落后搜索冠军 ≤ 容差(默认 MDE,且不低于
2×noise)时可用,否则交付位空缺(`delivery_resolution.status=diverged`),
收尾走未确认路径,人工指定须 `--ack-divergence`。缺 shadow 不可豁免。
每配方 confirm 完成次数默认 ≤2(`evolution.max_confirm_attempts_per_recipe`),
超额须 `new-trial --allow-extra-confirm "<理由>"`;confirm 失败不要靠反复换 seed 再抽。

**cap 混杂的负结论**:`cap_confounded=true`(Δparent<0 且最终拟合撞树数上限)时
**verdict 不改**,但方向台账不记负奖励、不算证伪,HPO 边际递减门也不当证据。

**warm-start**:gen-00 warm trial 按 warm 家族单独校正,未通过记
`warm_start_not_confirmable`,不能 confirm。

**先验豁免**:`delivery_resolution.status=waived_by_prior`(交付状态
`confirmation_waived_by_prior`)表示冠军配方未确认、但 warm-start 先验证据足够强,
交付位照常绑定冠军;只缺换 seed 复现证明(审计见 `model_meta.prior_waiver`)。配置缺省关闭。

### Segment 诊断

Evaluator 按 score 分位输出 monotonicity、Spearman、top lift 和 passed。
样本不足或 score 退化时 `passed=None`，不单独阻断 verdict。完整业务分桶表
由 Harness 自动生成，见 [stage-playbook.md](stage-playbook.md)。

## 特征贡献证据

模型内 gain 只能解释，不能单独裁决方向。模型工件齐备时，Harness 按列名前缀
族在 validation 上做分组置换，生成 `feature/group_importance.csv`：

- bundle 初筛参考 permutation delta；
- 晋级 bundle 必须通过 drop-group retrain 条件消融（**由 Agent 手建
  Trial**：`feature_exploration.conditional_ablation` / `ablation_trials`
  在 runtime 无实现，已标 RESERVED，框架不自动派生消融）；
- gain 与消融冲突时，以置换/消融证据为准；
- 加宽 Trial 必须保存真实 `validation_input.parquet`。缺失时新增列族标为
  `families_unverifiable`，不得判 dead。

大 validation 上置换在分层子样本上做,delta≈0 附近的族排序不构成证据。
`group_importance_policy` 缺省 `skip_rejected`:被拒 trial 没有置换结果;需要时设 `all`。

`screening-only` 宽 Trial（`screen_only: true`）的 group_importance 是
**方向先验**，不是某族的证据：它不进 FDR/alpha 台账、不计 accepted、不作
冠军/Archive/父代、不入方向台账。用它选题，用它裁决不行。

反思应同时检查重要性迁移、gain concentration、特征稳定性和高 PSI 风险。

## 生命周期、失败与资源

```text
PROPOSED → CODE_WRITTEN → RUNNING → SUCCEEDED/FAILED
         → EVALUATED → REFLECTED
```

- timeout、oom、infra_error 可在同一 Trial 内按配置重试并降配；
- code_error 只允许有限修复轮，超限后 invalid；evaluator_error 只补评估；
- generation deadline 到期仍运行的 Trial 记 timeout，反思记录未验证方向；
- 同因连续失败或一代失败率过高触发熔断，root-cause 后才能恢复；
- 所有失败都计入预算并进入 leaderboard，但 infra failure 不给模型方向记负奖励。

`snapshot` 决定并发和线程预算；runner 注入 OMP/MKL 与 `MODELEVO_GPU`。Trial
不得硬编码线程或设备。预算按 compute hours 与 wall time 双轨计。

**内存准入与选卡**:Runtime 按可用内存与预估峰值逐个准入,不够就排队,GPU 按
实际占用选卡。

- 定并发前看 `snapshot --run <run_dir>` 的 `admission.suggested_parallel`
  (内存口径可能远低于 CPU 口径)。
- 无父代/同 lineage 峰值可参考时,在 `trial.yaml.expected_resources.peak_memory_gb`
  声明预估峰值。
- 排队超时记 `failure_reason=admission_timeout`:infra 类,不进熔断、不耗预算、
  不入方向台账。
- 逃生口 `MODELEVO_MEMORY_ADMISSION=0`;日志 `logs/admission.jsonl`。
- 资源受限导致串行时 finalize 的低并行告警会注明"资源受限串行",不需在反思中答辩。

Successive halving 默认关闭；启用时 quick 只作同代筛选，晋级者仍全量重跑。
若旧数据的前缀样本标签偏差过大，quick/capacity 结论降级处理。

## 停止条件

Runtime 可以因以下原因停止：

1. **Budget**：max trials/generations/compute hours/wall time；
2. **Performance**：达到 TaskSpec 目标；
3. **Convergence**：连续多代 best delta 小于 noise 且无新 accepted；
4. **Diminishing return**：方向后验上界低、无未探索高潜力方向；
5. **Information channels exhausted**：固定样本源下的信息通道已清账。

只有一个数据视图时,单口径护栏阻断 `convergence_no_accepted`;budget/performance
与 `roi_diminishing_return` 不受此限。信息穷尽是独立出口，至少同时要求：

- 最新一代无新 accepted，best delta 无有意义正增益；
- `information-inventory.yaml` 无 unconsumed 项；
- 最新 Planner 的 `candidate_value_upper_bound` 低于最小有意义提升；
- 存在**绑定搜索冠军配方**的合法 fresh confirmation(`delivery_resolution`
  非 diverged/unconfirmed;陈旧确认不算),且反思已记录结论的数据视图范围。

停止报告必须列出
近代 delta、累计提升、方向递减、未执行高潜力方向及原因、数据视图、预算和
接近上限的置信度；Agent 不能用未知代替“已穷尽”。

### 停止语义

stop_report 关键字段:

| 字段 | 含义 |
|---|---|
| `advisory` | `blocks` 为空时为 true。phase=STOPPED 只是建议:后续命令都不检查 phase,继续演进不会被拒,也可能再次命中停止 |
| `blocks` | 真正硬拦截的只有两种:`budget_max_trials` → new-trial(`BudgetExhausted`);`budget_wall_time` → run-generation 整代拒绝启动 |
| `reason_detail` | 每个停止码的解释,例如 `budget_max_generations` 只是建议 |
| `undelivered_gain` | 同视图最优 trial 相对**交付锚点**的增益,附 z、`meets_mde` 与前 3 名候选。只进报告,不触发停止 |

- anchor 口径下 `undelivered_gain.meets_mde=true` 时,`remaining_high_potential_directions`
  会有 `confirm_or_consolidate:<trial>`,收尾前先对它 confirm。parent 口径只写
  `action_hint`:小步链在该口径下无法晋级,要纳入交付需开 anchor 口径的新 run,
  或由用户裁决手工 confirm。
- **`single_data_view_insufficient` 不是"可以停"的统计结论**:它表示搜索已停滞且
  证据面单一(≥3 代、最近一代无新 accepted、只试过 1 个非 sample 视图),不得据此
  宣布统计收敛。出口有两个:补一个独立数据视图(`write-splits-view` 流式构建或
  `--from-feature-block`),或满足上面的信息穷尽条件。与 `roi_diminishing_return`
  同时出现时,ROI 才是实际的停止依据。
- **代际 delta 口径**(`generation.yaml.generation_delta_protocol`):
  `legacy_best_vs_prev_generation_best`(老 run)下冠军没换时 delta 为 0 或负,ROI
  会当成"无增益";`champion_vs_champion`(新 run)为本代前后冠军之差。要区分
  "真没增益"和"增益没晋级",都看 `undelivered_gain`。
- **空代**(本代没有任何已评估 trial):代际 delta 记 None,不计入收敛/ROI,只保留
  预算类停止;finalize 输出 `empty_generation` 告警。
