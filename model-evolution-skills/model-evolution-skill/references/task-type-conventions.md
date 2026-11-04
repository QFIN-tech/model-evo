# 任务类型细则(入口:按 model_type 渐进加载对应单类型文档)

按 TaskSpec 的 `model_type` **只读对应的一篇**,不整篇通读:

| model_type | 必读文档 | 说明 |
|---|---|---|
| `binary_classification` | [task-type-classification.md](task-type-classification.md) | 默认路径;本文档概要已覆盖大半,写 Trial 前读 |
| `regression` | [task-type-regression.md](task-type-regression.md) | score=预测值、指标白名单、重尾/漂移诊断、lint 规则 4 |
| `uplift` | [task-type-uplift.md](task-type-uplift.md) | treatment 必填、score=CATE、S/T-learner、lint 规则 5 |

multiclass/ordinal 属实验性支持(指标方向已注册,profile 的 IV/
单变量 AUC 对其不生效),无单独细则文档,按最接近的类型口径执行
并在 hypothesis 中声明实验性质。

## evaluator 侧差异速查(三类型一览)

| 维度 | binary(默认) | regression | uplift |
|---|---|---|---|
| score 语义 | 正样本概率 | 预测值本身 | CATE 估计(可正可负) |
| 指标 | auc/ks/logloss | rmse/mae/mape/r2(越小越好,方向自动翻转) | uplift_auc/qini/auuc(越大越好) |
| 重采样 | 按 validation 行 | 同 binary | bootstrap/配对采样连 treatment 一起 |
| 分段排序性 | 段内正样本率 | 段内目标均值 | 段内处置−对照增量 |
| 相关 lint | — | 规则 4(禁概率入 score) | 规则 5(必须引用 treatment) |
