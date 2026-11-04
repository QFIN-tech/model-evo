# 天花板估计

回答**这份数据大致能做到多好**。benchmark 套件(框架在一批数据上离强参照多远)由维护者
运行,见 `docs/reference-archive/benchmark.md`。

## 1. ceiling:等预算强参照

按需能力:不属于日常迭代流程,只在用户明确要求时执行,Agent 不得自行加入。

```bash
python <skill-root>/scripts/modelevo.py ceiling --run <run_dir> \
    [--engines strong,autogluon] [--budget-minutes 45] [--n-seeds 5] \
    [--from-trial <trial_id>] [--max-rows N] [--out-dir <dir>]
```

- 前置:profile 与 baseline 已完成(参照读 profile 材料化的数据面)。
- **只在 validation 上评估**,与搜索 trial 同一裁决面;不读 shadow / final_test。
- engine:`strong`(强参数 GBDT + K-seed bagging,必跑)与可选 `autogluon`(装不上时记
  `unavailable`,不阻塞)。
- 预算按 engine 计:每个 engine 各得 `--budget-minutes`。
- 特征面:
  - 缺省 `base`:profile 材料化的特征面。冠军加宽后超过它是常态,**不代表到顶**;
    只用于判断模型侧(HPO / 模型族)还有没有空间。
  - `--from-trial <id>`:该 trial 的特征面(基础列 + `trial_feature_blocks`,经
    特征块缓存复建)。只支持第五族特征块契约;旧加宽族明确报错。
- 结果写 `<run>/ceiling/ceiling-estimate.json`。

参照里允许 bagging / AutoML 集成——它只是测量,不是可交付 trial,不受单模型
拓扑门约束;参照 bagging 值与单 seed 值之差就是"单模型约束的代价"的估计。

### 谁消费

| 消费方 | 用法 |
|---|---|
| planner-context §14 | 显示天花板、`gap_to_ref`、`feature_surface` |
| stop report `ceiling` | 同上;trial 面且 `gap_to_ref ≤ MDE` 时追加停止码 `ceiling_gap_within_mde`(建议性) |
