# 回归任务细则(regression;渐进披露)

`model_type: regression` 时必读(在 SKILL.md 之后)。铁律、主循环、
编排纪律见 [SKILL.md](../SKILL.md),本文档只讲回归特有的口径。

## score 语义(最容易错的一条)

- score 列 = **预测值本身**,不是概率——Trial 产物、evaluator、
  leaderboard、stop 判定全链路按此口径;
- **lint 规则 4**:检测到 `predict_proba(...)[:, 1]` / `torch.sigmoid(...)`
  写入 score → 拒绝(概率量纲会让 rmse/mae/r2 全部失真)。只有概率**实际流入**
  score 写出路径才拒绝,hurdle 的 `p·μ` 不算。

## 量纲感知 MDE

`evaluator.minimum_meaningful_improvement` 默认 0.001 是**绝对值**,对原值 rmse
毫无意义(所有 Δ 都"有意义",穷尽/HPO 边际/confirm 稳定性/交付容差门全部失效)。

- `MDE = max(minimum_meaningful_improvement, minimum_meaningful_improvement_relative ×
  |baseline 主指标|)`,所有消费点同口径。
- 新 run(以本 run 配置为准)由 init 写入 relative=0.0008(原值 rmse 12932 时
  MDE≈10.3);配置无该键时仍取绝对值,原值量纲的回归 run 须自行设 relative 或绝对值。
- 未设 relative 且绝对 MDE < 1e-4×|baseline| 时,planner-context 顶部显示
  `MDE 量纲告警`;§1 显示 MDE 数值与来源(absolute/relative)。

## 指标白名单

- `primary_metric` 只能 **rmse / mae / mape / r2**(validate 拦截
  回归+分类指标错配);
- 停止判定与 improvement_vs_baseline 按 rmse 等"越小越好"方向自动修正。

## TaskSpec 差异

```yaml
model_type: regression
target:
  name: amount_30d
  value_definition: 名单下发后30天内累计动支金额(元)   # 预测什么量
  unit: CNY                                            # 单位,报告呈现用
primary_metric: rmse                # rmse | mae | mape | r2
secondary_metrics: [mae, mape]
```

- `target` 建议写 `value_definition`(预测什么量)与 `unit`(单位),
  替代 positive_definition(binary 才必填)。
- **在 loader 里做标签变换**(如 log1p 金额)时:派生列名写进 `target.name`,
  原始列进 `features.banned`(label_equivalent),派生列名写进
  `data.derived_columns`——否则数据底座指纹变化,同数据上
  其它任务(如同源二分类)的特征经验检索不到。见
  [task-and-data.md](task-and-data.md#data-loader)。

## profile 特有诊断

- IV 列 = **|Spearman corr|**(同一等级刻度:suspicious≥0.5 进反泄漏
  复核);单特征泄漏检查 |corr|>0.95 → critical;
- 目标分布摘要(mean/std/分位)+ **重尾诊断**(q99/q50>10 提示
  log1p/分段口径);
- train/validation **目标均值漂移告警**;
- `feature-woe.csv` 不产出(仅 binary/uplift)。

## baseline 与 Trial 模板

- 模板自动切 Regressor + `model.predict(X)` 分支(LGBM/XGB/torch MLP
  均已适配),metrics.json 报 valid_mse;
- 目标重尾时(见 profile 重尾诊断)优先与业务确认换 log1p 目标 +
  rmse 口径,或分位数/两段式建模,而不是直接硬拟合。
- 两段式 hurdle(P(y>0) 分类器 × 条件均值回归器,各自只吃原始特征)在
  拓扑门里判为 `structural:hurdle`,回归任务白名单放行;把分类器
  的概率当特征喂给回归器则是级联,须用户批准(见
  [template-api.md](template-api.md#模型拓扑))。
- hurdle 冠军面的同算法 HPO:`new-trial --action hpo --from-parent
  <hurdle 冠军>`,搜索空间键无前缀 = 两段共用,`clf__max_depth` /
  `reg__reg_lambda` 等 = 只作用于分类段 / 条件回归段(`hurdle_lr` /
  `hurdle_cap` 是 lr / 轮数上限的别名);inner 指标是组合预测 p·max(μ,0)
  上的主指标(rmse)。父代需提供 `_hurdle_estimators` / `_fit_hurdle` /
  `fit_and_score` 契约入口,缺失时创建期拒绝并给替代路径。
- 回归的训练后泄漏哨兵(leak_scan)用单变量 |Spearman ρ| > 0.95(与 profile
  回归泄漏规则同口径)。
- **官方 hurdle 模板**:`new-trial --hurdle-template` 渲染
  `runtime.hurdle.HurdleRegressor`,它可 clone、可 joblib,两段各自早停,
  `predict = p·max(μ,0)`,不暴露 `predict_proba`;支持 `clf__`/`reg__` 分段键与
  `hurdle_lr`/`hurdle_cap` 别名,可与 `--action hpo` 组合。两段拟合走子模型缓存:
  只改一段的子代,另一段在同输入、同参数、同 seed 下命中缓存(`MODELEVO_SUBMODEL_CACHE=0`
  关闭)。API 见 [template-api.md](template-api.md#官方-hurdle-容器与子模型缓存)。
- 手写复合容器在 new-trial 时会写 `trial.yaml.deliverability`(clone_ok /
  rerun_recipe_ok;只告警)。交付时不可 clone 的 validation 早停冠军自动回落 train_tail
  口径复跑配方,并留痕。
- hurdle 的分组置换走"按族置换 → 容器 predict",另给 `delta_clf_mean` /
  `delta_reg_mean`(见 [feature-exploration.md](feature-exploration.md))。

## evaluator 与分段排序性口径

- evaluator 对 rmse/mae/mape/r2 全链路方向感知(越小越好自动翻转
  delta 与 stop 语义);noise-aware gate、切片、时序回归同样适用;
- 重采样同 binary(按 validation 行);
- 分段排序性:段内**目标均值**;spearman 适用。

## 回归诊断(只诊断,不进 verdict)

主指标会掩盖绝对水位漂移与中低额段变化,以下字段供解读与升级判断:

- **搜索期**:每个回归 trial 的 `evaluation.regression_diagnostics` 包含三部分:
  - `level`:`level_ratio`(pred/y),|水位比 − 1| > max(10%, 2·`level_ratio_se`)时
    `level_shift_warning`,超过 10% 但仍在噪声带内的记 `level_within_noise`;
  - `level_by_period`:按 validation 时间列分期的水位;
  - `vs_parent_divergence`:主指标与次指标(spec `secondary_metrics` 中首个回归指标,
    缺省 mae)相对父代方向相反、且次指标 Δ 超过 2·SE 时 `diverged=true`。
- **交付期**:`finalize-retrain` 的 `shadow_precheck.level_stats` 与
  `level_shift_warning`;`evaluate-final` 在 gate 内写 final_test/shadow_test 的
  `evaluation.regression_diagnostics`,交付模型另有 `vs_baseline_divergence`。
  水位告警时 stdout 打印 `[回归水位]` 行,`advice` 给出"排序可用、绝对金额需校准"。
- **报告**:§12 显示回归水位(交付 vs 基线);§14「模型局限」自动汇总水位偏差与
  主次指标背离。

## 分桶报表(segment-report)列集

分桶人数、样本占比、预测平均值、真实均值、Bias、MAE、RMSE、
中位数、最大值、最小值、label 求和、label 占比、累计 label、
累计 label 占比。

`Bias` = 桶内 **预测 − 真实** 的均值(正 = 高估),与原值尺度
`evaluation.raw_scale.mean_bias` 同号(老 run 已落盘的报表可能是 真实 − 预测,
以报表所属 run 版本为准)。MAE/RMSE 按残差绝对值/平方计,不受符号约定影响。

**原值尺度分桶**:目标经 log1p 等变换时,官方分桶在变换尺度上。
业务需要原值(如元)统计时,在 `evaluate-final` 加
`--segment-inverse-transform expm1`:同一次内存标签读取中,对 final_test 与
shadow_test 各出一份同列集的原值分桶表(`final-model/segment-raw/<split>/`),
分桶仍按变换尺度预测排序(expm1 单调,桶划分不变),并在
`evaluation.raw_scale` 记原值 RMSE/MAE/均值偏置/总额比/Spearman。注意 log
尺度的条件均值经 expm1 回算接近几何口径,原值总额比显著 < 1 属口径(Jensen),
不是模型偏置;该报表只供业务解读,不参与裁决。final_test 标签事后无法补读,
需要就在这一次里打开。

## 升级到人

- 目标口径异常(重尾 q99/q50>10、train/validation 目标均值漂移告警)
  需业务方确认预测量纲与口径。
- 交付期 `level_shift_warning=true`时,在报告与回传中写明"排序可用、绝对
  金额需校准"及偏离期,由业务方决定是否做水位校准;不在 harness 外自行缩放交付模型。
