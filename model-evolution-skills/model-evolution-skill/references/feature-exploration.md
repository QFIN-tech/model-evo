# 特征优先探索

baseline 后先系统探索信息与特征，再进入模型/HPO。目标不是穷举单列，而是用
少量可归因 bundle 判断哪些信息源和加工机制值得继续。

## 目录

- [数据与信息边界](#数据与信息边界)
- [Bundle beam](#bundle-beam)
- [抽样与全量复核](#抽样与全量复核)
- [八个探查维度](#八个探查维度)
- [充分性门槛](#充分性门槛)
- [交付检查](#交付检查)

## 数据与信息边界

Trial 只能消费三个授权信息面：

- 特征：`splits/{split}.parquet`；
- 标签：`load_labels(split)`；
- 解析源：`reparse_frame(split, columns)`，只允许 spec 白名单。解析内核一律
  经 `feature_block` + `trial_feature_blocks(split)` 接入(跨 trial
  缓存 + 并行;契约见 [template-api.md](template-api.md#特征块-trial_feature_blocks)),
  不要在 main 里手写单核逐行循环。

`splits/identity/` 只供框架对齐。banned 列不在任何数据面，Trial 和 adhoc 都
禁止调用 `load_data()` 绕开隔离。transform 不得引用 banned 字面量，训练后
必须保留 `leak_scan`。API 见 [template-api.md](template-api.md)。

`sample-profile/information-inventory.yaml` 是候选方向清单：按信息源记录窗口、
近因、分桶、聚合、序列结构和组合的消费状态。每代 reflection 回填
implemented 或 explicitly_falsified；未清账的信息源不能被宣称“已榨干”。
`finalize-generation` 按 trial 的 `info:<column>:<dimension>` 形式 family_hints 自动
回填，不覆盖 agent 写入的更高状态；用 `new-trial --family-hints "info:<列>:<维度>"`
声明即可免去每代人工回填。

**只有方向证据才回填证伪**：`rejected` 不必然是"该粒度没有信息"。
跨数据视图的 trial（`data_view≠v1`）没有配对对照、记 `baseline_incomparable`，
其 verdict 只是"不可比"；`reason` 落在切片/时序门或"无父代可比"时同理
——切片门拦的是分布敏感度。这些形态一律不计回填；不再成立的自动回填证据会被
降级（见返回值的 `demotions`）。

## Bundle beam

默认宽度 4，首轮组织 3–4 个 bundle。宽度由
`feature_exploration.bundle_beam_width` 定义，语义是**每代特征 Trial 槽位
总数**（`new-trial` 在建满后拒绝继续创建；`--override-beam "理由"` 可
显式放行并留痕）。开启 `super_trial` 时，其中一个槽位留给筛选用宽 Trial，
剩下 `width-1` 个给 bundle。bundle 按“共同信息源 + 作用机制 +
预期交互”划分：

- **加列不替换**：保留基础特征面，新列以 `seq_`、`ratio_`、`miss_` 等前缀
  标明来源；
- **包内充分、包间可辨**：同机制的多个口径可同包，不同机制不提前混合；
- **模型固定**：使用 baseline 最优算法和超参，不建 model/loss Trial；
- **beam 而非穷举**：首轮只保留风险调整后有价值且互补的方向；
- **super Trial 仅作筛选**：不能单独判方向死亡或信息穷尽；入选族仍拆包验证。

`screening-only` 宽 Trial（`new-trial --screen-only`，
`feature_exploration.super_trial=true` 时由 Planner 提名）是上一条的
机器可执行形态：一次性加宽全部候选族，产出 `group_importance` 的族级
置换 delta 作**方向先验**，但不参与任何裁决——不进代级 FDR/alpha 台账、
不计 accepted、不作冠军/Archive/父代、不入方向台账，仅在 leaderboard 里
带 `screen_only` 列可见。它的分数不可归因到任何单一方向，因此**永远
不能**被当作某族的证据。

每个 bundle 除标准 Trial 产物外还必须提供：

1. `feature/feature_importance.csv`，仅作解释；
2. `dump_eval_frame(X_va)` 生成的 `model/validation_input.parquet`；
3. `feature/group_importance.csv`（Harness 自动计算）；
4. hypothesis、falsification、changed_modules 与 family hints。

加宽后不保存真实 validation 输入时，Harness 无法重放 transform；新增列族只能
标为 `families_unverifiable`，不能判 dead。

**复合容器的分组置换**:hurdle 等复合容器走"按族置换 → 容器 `predict`";
`HurdleRegressor` 另给 `delta_clf_mean` / `delta_reg_mean`,即该族对分类段与条件
回归段的贡献。族生死仍只是参考信号,不进 verdict/leaderboard/alpha。

### 裁决顺序

1. 以分组置换 delta 初筛方向；
2. 合并互补的晋级 bundle；
3. 从合并候选中分别移除晋级方向，至少完成两个 drop-group retrain 条件消融；
4. gain 只辅助解释；与置换/消融冲突时以后两者为准；
5. 最佳组合进入全量复核和新 seed confirm。

> **条件消融由 Agent 手建。** 第 3 步的 drop-group retrain 是协议要求，不是
> runtime 保证：`feature_exploration.conditional_ablation` / `ablation_trials`
> 未实现（RESERVED），框架不会自动派生消融 Trial。做法：对晋级 bundle 用
> `--features @<trial_id>:group_importance` 取族白名单，手工排除目标族后建
> drop-family trial。

## 抽样与全量复核

先建立可复现的 20 万行以内 sample 面：

```bash
python <skill-root>/scripts/modelevo.py write-splits-sample \
  --run <run_dir> --max-rows 200000 --note "特征探索抽样面"
python <skill-root>/scripts/modelevo.py check-products \
  --spec <run_dir>/task-spec.yaml --run <run_dir> \
  --splits-dir <run_dir>/sample-profile/splits-sample
```

**先读噪声预检,再决定探索面**:落盘时自动写
`splits-sample/manifest.json.noise_precheck`(planner-context §6c),关键字段为
`recommend_v1` 与 `reason`。

- `recommend_v1=true`(抽样面 SE > 2×MDE,且 baseline 全量运行 <
  `sample_precheck_max_full_minutes`)→ 本任务的特征探索**直接在 v1 全量面上**进行,
  跳过下面的抽样面锚点。反思里引用该预检作为偏离"先抽样"协议的依据。
- `recommend_v1=false` 且 reason 写"SE > 2×MDE 但全量太贵"→ 仍用抽样面,但阴性结论
  不可信,方向必须回 v1 复核。

**train 内预筛 `prescreen`**:

```bash
python <skill-root>/scripts/modelevo.py prescreen --run <run_dir> --trial <候选> \
  [--ref <对照,缺省第一父代>] --seeds 3 --holdout-frac 0.15 --es-frac 0.05 \
  [--max-rows N] [--gpu <序号>]
```

- 只用 train 面:按时间列排序,前段拟合、中段早停、尾段留出。候选与对照的 trial
  代码在镜像 run 布局的**沙箱副本**里运行 K 个配对 seed,不读 validation/shadow/
  final_test 的标签与特征(访问日志记入报告),也不写 trial 目录、不改状态。
- 输出 `prescreen/<trial>/report.json`:Δ(候选 − 对照,原始方向)、配对 bootstrap SE、
  seed 间 SE、合成 `se`、`z`、`p_better`、逐 seed 结果、`screen_verdict`
  (`likely_better`/`likely_worse`/`inconclusive`)。报告标注
  `screening_evidence:true`,**不进 FDR / leaderboard / 方向台账**,晋级仍须正式
  trial 在 validation 上裁决。
- 边界:
  - 默认 CPU 串行,`--gpu` 指定卡;
  - 沙箱数据面 `prescreen/_sandbox/<view>-<tag>/` 会按相同参数复用,不会自动删除,
    需要手动清理(沙箱 trial 副本缺省会删,`--keep-sandbox` 保留);
  - 首次运行要在沙箱数据面上重算特征块,因为行集不同,缓存键随之不同。

**复用已验证的特征块**:`new-trial --add-blocks <trial>:<block>[,...]`,字节级复制
加指纹校验,保证缓存命中,细则见
[template-api.md](template-api.md#特征块-trial_feature_blocks)。

**先建同视图锚点**:`new-trial --from-parent <baseline_id> --action
data_view --data-view sample` —— baseline 代码原样跑在抽样面(分钟级,
`SPLITS` 自动改写到 `splits-sample`)。之后的 sample 面 Trial 一律
`--from-parent <该锚点> --data-view sample`,vs_parent 才是同视图配对比较。
父代在 v1 面的 sample Trial 既无 vs_parent 也无 vs_baseline,配对检验全缺席;
`--from-parent` 会按子 trial 的 `--data-view` 改写 `SPLITS`。

在 sample 面创建 `--action data_view --data-view sample --heavy` Trial。抽样面
保留 split 归属与基本分布，final_test 不参与。它适合方向筛选和相对排序，
绝对指标不得直接与全量面比较。

**跨视图没有 vs_baseline，这是预期行为**：baseline 锚点在 v1 面，
与 sample 面的行集不同，配对比较结构性不可比。评估器会记
`delta.baseline_incomparable=true` 并把 `vs_baseline` 置空，Trial 按
`vs_parent` 正常裁决——**不会**被判 invalid。相对排序用 sample 面内部的
父代对照（同视图），绝对水平必须回到 v1 面测。

抽样面晋级方向必须组合后在 v1 全量 splits 上重训；再以新 seed 创建 confirm
Trial，要求 validation 与 shadow 均非劣且保真(margin 含 seed 方差,见
evolution-and-evaluation.md「Fresh confirmation」)、参照对照不显著更差、
multi-seed 稳定性通过。

```bash
python <skill-root>/scripts/modelevo.py new-trial --run <run_dir> \
  --from-parent <full_data_candidate> --action confirm --generation <N>
```

父代必须先经 `finalize-generation` 的代级选择校正并通过源码完整性检查。

若 compression audit 或 reparse 指向冻结宽表之外的信息，使用
`write-splits-view` 重建全量视图；构建脚本与 hash 必须留在 run 目录。

## 八个探查维度

每个维度都需要 Trial 证据或明确“不适用”理由：

| 维度 | 最低探查 |
|---|---|
| 单特征深挖 | top 特征的分箱、饱和、断点及非线性变换 |
| 特征族结构 | 来源/前缀聚类、冗余簇与信息缺口 |
| 交互组合 | top 特征对的比值、差、乘积、交叉分箱；目标编码只用 train |
| 时序口径 | 多窗口、环比、斜率、最近事件和生命周期 |
| 聚合层复活 | compression 候选的近 K 期、趋势和事件序列视图 |
| 缺失语义 | missing indicator 与填补策略对照 |
| 分群异质性 | 关键业务分群的效力差异和专属特征 |
| 样本偏移 | 高 PSI 特征的比率化、去趋势或剔除 |

这些分析是产生假设的漏斗，不是“完成清单即通过”的形式检查。

## 充分性门槛

以下条件全部满足，reflection 才能写
`feature_exploration_complete: true`：

1. 达到 `feature_exploration.min_generations`（默认至少两代）；
2. 八维均有 bundle 证据或不适用结论；compression 候选和 reparse backlog
   都已实现或明确证伪；
3. 连续规定代数无新增 accepted 特征方向，且所有晋级 bundle 都有条件消融
   （消融 Trial 由 Agent 手建，见上文「裁决顺序」的说明）；
4. 最佳组合已在 v1 全量面复核，并通过新 seed validation + shadow confirm。

Runtime 只强制第 1 条（特征期门，见
[stage-playbook.md](stage-playbook.md) §5）：未满 `min_generations` 时
model/loss/hpo Trial 被拒。第 2–4 条由反思举证，无 runtime 实现。

未通过时，下一代保持 explore/data_view 100%；exploit 仅做特征筛选和窗口
细化，crossover 仅组合特征。只有用户明确授权并在 run-state 留痕时可跳过。
**gen-01 例外（硬门）**：首代无父代证据可加码/组合，`new-trial` 直接拒绝
`--action exploit` 与 `--action crossover`（方向台账此时尚不存在，
Planner 同代也不会提名这两类候选）。

特征收敛只表示可以切换搜索阶段，不等于全局停止。

**`feature_exploration_complete: true` 之后、进入收尾之前，必须在已确认的
冠军特征面上跑一轮同算法 HPO。** 特征充分性门只说明"特征轴
够了"，不代表模型轴已经榨过 —— 两者是独立的信息通道。

三条硬约束：

1. **必须在冠军的特征面上跑**，不是基础面。含 transform 加宽的冠军
   必须用 `--from-parent <champ>`（`--hpo-parent` 只继承 config 声明面，
   会静默跑在基础面上）。细则见
   [stage-playbook.md](stage-playbook.md#加宽面上的-hpo-必须走---from-parent)。
2. **算法取冠军的算法**。异算法 HPO 是另一个问题（换族是否有增益），
   不能替代"同族调参"。
3. 判据是**实测 `n_features` 与冠军一致**，不是"存在 hpo 子代"。跑错
   面的 HPO trial 不构成该面已调参的证据。

若批次里模型/HPO 候选被 slots 截断（batch 带 `model_axis_not_selected`），
**不得**据此判定模型方向无潜力 —— 截断 ≠ 已探索。

## 交付检查

- [ ] 所有 bundle 新列可由前缀追溯到合法信息源；
- [ ] transform 只消费授权三平面，identity 未进入模型；
- [ ] Lint 的 banned、leak_scan、load_data 和 eval-frame 规则全部通过；
- [ ] 置换、至少两个条件消融、八维覆盖与 reparse 清账有落盘证据
      （消融为 Agent 手建，框架不自动派生）；
- [ ] sample 晋级方向有 v1 全量复核和 fresh confirmation；
- [ ] 视图、adhoc 和 feature-lab 脚本及 hash 均保留；
- [ ] 经验条目标注 sample/full scope，未全量复核的证据不导出为强结论。
