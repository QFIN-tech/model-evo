# Uplift 任务细则(增量响应;渐进披露)

`model_type: uplift` 时必读(在 SKILL.md 之后)。铁律、主循环、
编排纪律见 [SKILL.md](../SKILL.md),本文档只讲 uplift 特有的口径。

## TaskSpec 必填差异

```yaml
model_type: uplift
target:
  name: label_convert
  positive_definition: T0~T+30 内发生转化(响应)
treatment:
  column: is_treated                # 1=处置 0=对照,整数列
  definition: 名单内下发优惠券(1)vs 未下发(0)
primary_metric: qini                # uplift_auc | qini | auuc
secondary_metrics: [auuc]
```

- **treatment.column 必填**(1=处置 0=对照,整数列)+
  `treatment.definition`(处置语义,如"下发优惠券");target 是处置
  意义下的响应(转化)标签;
- `primary_metric`/`secondary_metrics` 只能 **uplift_auc / qini / auuc**
  (validate 拦截 auc 等普通分类指标——那评的是转化率而非增量)。

## score 语义

- score 列 = **CATE 估计(个体处置效应,可正可负)**,不是转化概率:
  S-learner 的 p1−p0 或 T-learner 双模型差分;
- Trial 产物、evaluator、leaderboard、stop 判定全链路按此口径;
- evaluator 自动从 validation split 取 treatment 列传入指标计算
  (bootstrap/配对采样连 treatment 一起重采样)。

## baseline 与 Trial 脚手架

- **baseline 走 S-learner**(单模型,treatment 作为特征进入,打分时
  同一批样本分别置 w=1/w=0,`p1−p0` 即 CATE):单次 fit;mlp baseline 不支持 uplift 双次打分,配置层自动剔除;
- Trial 默认脚手架是 **T-learner**(处置/对照各训一个 LGBM 取
  差分,`model_num=2`,可执行非占位)——双模型差分是 uplift 建模
  语义的一部分,不属于 ensemble 融合,但须在 hypothesis 中声明
  T-learner/S-learner 口径。拓扑门把它判为 `structural:t_learner`,
  uplift 任务白名单放行;在 T-learner 之上再做子分级联或多模型融合仍须
  用户批准(见 [template-api.md](template-api.md#模型拓扑))。

## profile 特有诊断

- treatment 分布(总体/分 split/分月)、treated vs control 响应率与
  观测 ATE(split 间实验条件一致性检查);
- IV 按 treated/control 拆分为 `iv_treated`/`iv_control`,差值
  `iv_diff` 大 = 处置改变了特征与转化的关系 = uplift 调节
  (moderator)候选——Planner 优先围绕 top moderators 造特征×处置
  交互(见 feature-quality.csv);
- `feature-woe.csv` 有产出(仅 binary/uplift)。

## Trial Lint 规则 5

uplift Trial 的 train.py 必须**引用 treatment 列**(TREATMENT 常量/
treatment 字样/TaskSpec 声明的列名任一);未引用即拒绝——模型学不到
处置效应,score 退化为转化率概率,auuc/qini 全部失真。

## 非随机实验数据慎用

观测 ATE 与 uplift 指标都假设处置分配近似随机(或可忽略性);
profile 的 treatment 分布倾斜应在 adhoc 里做处置分配偏倚检查
(处置组/对照组特征 PSI)。

## evaluator 与分段排序性口径

- 指标方向均为越大越好;noise-aware gate、stop 判定与 binary 同一套;
- bootstrap/配对重采样与切片评测均连 treatment 子集一起;
- 分段排序性:段内**处置−对照增量**;spearman 不算(uplift 不算)。

## 分桶报表(segment-report)列集

人群占比、分桶人数、T/C 人数、T/C 转化率、实际 Uplift、预测
Uplift、增量人数(实际 uplift × 桶人数,处置组增量外推到整桶)、
累计 Uplift、累计增量人数、95% CI(桶内 bootstrap;桶样本 <30 为空)。
桶内缺处置/对照组时整表不产出(评估照常)。

## 升级到人

- 处置分配严重非随机(treatment 分布倾斜、处置组/对照组特征 PSI 大)
  时,uplift 指标解释力受限,需业务方确认实验口径。
