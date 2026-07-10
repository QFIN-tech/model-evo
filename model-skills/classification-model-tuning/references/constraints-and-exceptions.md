# classification-model-tuning 执行约束与异常处理

> 本文件从 `classification-model-tuning/SKILL.md` 第 6/7 节抽出,包含覆盖范围、不覆盖清单、何时用、session 约定、数据安全红线(项目级通用约定复述)、变更前置流程、Optuna 搜索空间详情、异常处理全表。SKILL.md 第 6 节只保留红线级约束(交互确认流程 / Optuna 依赖 / select_features 不调参 / 变更前置),第 7 节保留单行指针指向本文件。

## 1. 覆盖范围

### 1.1 做什么

- 对 baseline run 做规则诊断(欠/过拟合/不稳定/未收敛)→ 推荐超参 / Optuna 搜索 → 重训产 `-tuned` 新 run
- 基于 feature-analysis csv 按高 PSI / 低 IV / 高缺失率剔除特征 → 用 baseline 超参重训产 `-feat` 新 run

### 1.2 不做什么

- 不训练 baseline(`classification-model-training` 职责)
- 不做特征质量分析 IV/PSI/相关性(`feature-analysis` 职责)
- 不做历史模型推荐(`classification-model-recommend` 职责)
- 不做会话级横向对比聚合(`classification-model-comparison` 职责)

## 2. 何时用

用户已有一个 `classification-model-training` 产出的 baseline run,想做超参调优或特征筛选;支持算法 xgb(默认)/ dnn / lr,按 baseline.algo 自动分流,无需手动指定。

## 3. 不覆盖 baseline

两个入口都产生新的 run 目录,不会覆盖 baseline。

## 4. select_features 不调参

仅缩小特征集,用 baseline 的 `used_params` 直接重训;如需调参走流程 A 或串联 `-feat → -tuned`。

## 5. session 约定

`--output_dir` 传 `session_dir` 本身(或默认从 baseline 推断),新 run 与 baseline 同级落 `new-models/`;输入是已有 baseline run 目录,不需额外取数;`session_dir` 不写进任何 yaml/csv 配置。

## 6. 数据安全红线(全模式强制)

baseline 的 data_path 必须可访问;严禁在配置中硬编码用户 ID。

## 7. 变更前置流程(强制遵循 CLAUDE.md)

修改训练/调优代码前,先输出「变更计划」(一、修改内容 二、预期影响 三、回滚方案)并等确认。

## 8. Optuna 搜索空间详情

搜索空间按 algo 分流,均在 baseline 周围 ±30%(int 参数离散、float 连续 uniform),评价指标 val_auc:

- **xgb**:`max_depth / min_child_weight / learning_rate / subsample / colsample_bytree / reg_lambda`
- **dnn**:`dropout / learning_rate / weight_decay / batch_size / epochs / patience`
- **lr**:`C / max_n_bins / min_bin_size / max_iter`

dnn / lr 路径下 `train_fn` 用 lambda 包装丢弃 `info` 元素,统一为 `(predictor, metrics)` 协议。

## 9. 异常处理

| 条件 | 处理方式 |
|---|---|
| `baseline_run` 目录或其 `config.json` 不存在 | 停止执行,提示先跑 `classification-model-training` 产出 baseline |
| baseline 的 data_path(train/test/oot parquet)不可访问 | 停止执行,提示恢复数据或重跑上游 |
| baseline 无 `used_params` | 防御性兜底:退到该 algo 的默认参数(正常路径不会触发) |
| `--method optuna` 但未安装 optuna | 清晰报错,提示 `pip install --user "optuna<4"` |
| 诊断为 well_fit 且未 `--auto-apply` | 默认询问后退出(`status=skipped_well_fit`),不强行重训 |
| 用户在 `[Y/n]` 确认时拒绝 | 退出,不落新 run |
| `analysis_dir` 下某 csv 缺失 | 对应规则静默跳过,其余规则正常执行 |
| 筛选后所有特征被剔除 | 停止执行,提示放宽阈值后重试 |
| baseline 无 `config/train_config.yaml` | 打 warning 跳过 yaml 复制,其他产物正常 |
| 某 split 缺基线 eval JSON | comparison 对应档跳过,不影响主流程 |

---

> 关联:SKILL.md 第 6/7 节。
