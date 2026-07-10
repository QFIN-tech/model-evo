---
name: classification-model-tuning
description: 对 classification-model-training 的 baseline run 做超参调优（run_tuning.py：规则诊断 / 可选 Optuna 搜索）或特征筛选（select_features.py：按 PSI/IV/缺失率剔除特征），产 `-tuned` / `-feat` 新 run，按 baseline.algo 直通 xgb/dnn/lr。当用户说"调优""调参""优化模型""超参搜索""欠拟合""过拟合""不收敛""特征筛选""剔除高 PSI 特征""剔除低 IV 特征""剔除高缺失特征""缩小特征集"时使用。
---

# classification-model-tuning

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| `baseline_run` | ✅ | 上游 `classification-model-training` 产物 | baseline run 目录（如 `<session_dir>/new-models/xgb-v1`）；从其 `config.json` 恢复 cfg / data_path / algo / used_params / metrics / best_iteration |
| `analysis_dir` | 流程 B 必选 | 上游 `feature-analysis` 产物 | 含 `stats.csv` / `iv_table.csv` / `psi_table.csv` 的目录，仅 `select_features.py` 用 |
| baseline 的 data_path | ✅ | baseline `config.json` 记录 | train/test/oot 三档 parquet 必须可访问 |
| `session_dir` | 否 | 从 `baseline_run` 推断（parent.parent） | 输出根目录；可用 `--output_dir` 显式覆盖 |
| Optuna 包 | 否 | `pip install --user "optuna<4"` | 仅 `--method optuna` 需要 |

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录（即本文件所在目录），执行时替换为实际绝对路径，不要依赖当前工作目录。

### 2.1 流程 A：超参调优（run_tuning.py）

```bash
# 假设上游 classification-model-training 已产出 baseline:
#   <session_dir>/new-models/xgb-v1/

python <skill_dir>/scripts/run_tuning.py \
    --baseline_run <session_dir>/new-models/xgb-v1 \
    [--method rule|optuna] \
    [--n_trials 30] \
    [--version v1] \
    [--output_dir <session_dir>] \
    [--auto-apply]
```

执行步骤：
1. **读 baseline**：从 `config.json` 恢复 cfg / data_path / algo / used_params / metrics / best_iteration
2. **算法直通**：按 `baseline.algo` 走 xgb / dnn / lr 路径
3. **规则诊断**（algo-aware）：根据 train/val/oot AUC gap、PSI、训练动力学信号推断状态
   - `overfit` / `underfit` / `unstable_psi` 三条规则 algo-agnostic，只看 AUC gap 与 PSI
   - `underconverged` 按 algo 分流：xgb 看 `best_iteration / n_estimators >= 0.95`；dnn 看 `early_stopped=False` 或 `best_epoch / epochs >= 0.95`；lr 看 `converged=False`（凸优化罕见，兜底）

   | 状态 | 触发条件（algo-agnostic 部分） |
   |---|---|
   | `underfit` | `train_auc < 0.70` 或 `train-oot gap < 0.005` |
   | `overfit` | `train-oot gap > 0.05` |
   | `unstable_psi` | `train→oot psi > 0.10` |

4. **推荐超参**（algo-aware）：

   xgb 策略表：

   | 状态 | 动作 |
   |---|---|
   | underfit | depth+1, mcw/2, lr/2, n_est×1.5（各受 bounds 约束） |
   | overfit | depth-1, reg_lambda×2, mcw×2 |
   | underconverged | n_estimators ×2（受上限 2000 约束） |
   | unstable_psi | subsample -0.1, colsample_bytree -0.1（受下限 0.5 约束） |
   | well_fit | 不动 |

   dnn 策略表：

   | 状态 | 动作 |
   |---|---|
   | underfit | lr×2, dropout×0.7, epochs×1.5 |
   | overfit | dropout+0.1, weight_decay×2 |
   | underconverged | epochs×2, patience+5 |
   | unstable_psi | dropout+0.1, batch_size×2（减梯度噪声） |
   | well_fit | 不动 |

   lr 策略表：

   | 状态 | 动作 |
   |---|---|
   | underfit | C×2（减弱正则）, max_n_bins+2（更细 WoE 分箱） |
   | overfit | C/2（加强正则）, max_n_bins-2 |
   | underconverged | max_iter×2 |
   | unstable_psi | max_n_bins-2, min_bin_size×2（粗分箱更稳） |
   | well_fit | 不动 |

5. **问用户**：打印诊断 + 推荐参数，要 `[Y/n]` 确认（`--auto-apply` 跳过）
6. **训练**：`--method rule` 直接用推荐参数训练；`--method optuna` 在 baseline 周围 ±30% 搜索 `n_trials` 次，取 val_auc 最优 params 重训
7. **复用 classification-model-training 产物管线**：落八阶段产物到新 run_dir，`config.json.runtime` 含 `baseline_run / diagnosis / method / recommended_params / final_params / trials_log / baseline_metrics`

### 2.2 流程 B：特征筛选（select_features.py）

```bash
# 上游需要先跑 feature-analysis，产出含 stats/iv_table/psi_table.csv 的目录

python <skill_dir>/scripts/select_features.py \
    --baseline_run <session_dir>/new-models/xgb-v1 \
    --analysis_dir <session_dir>/sample-features/feature-analysis/analysis \
    [--version v1] \
    [--output_dir <session_dir>] \
    [--no-psi] [--no-iv] [--no-missing] \
    [--psi_threshold 0.10] \
    [--iv_threshold 0.02] \
    [--missing_threshold 0.95] \
    [--auto-apply]
```

执行步骤：
1. **读 baseline**：同流程 A
2. **算法直通**：同流程 A（按 baseline.algo 走 xgb / dnn / lr）
3. **应用规则**（三条独立，可分别启停）：

   | 规则 | 数据源 | 阈值 | 默认 |
   |---|---|---|---|
   | `high_psi` | `psi_table.csv.psi` | `> psi_threshold` | 0.10 |
   | `low_iv` | `iv_table.csv.iv` | `< iv_threshold` 或 IV=NaN | 0.02 |
   | `high_missing` | `stats.csv.missing_rate` | `> missing_threshold` | 0.95 |

   csv 不存在时对应规则**静默跳过**（不抛错，见 `selection_rules._read_csv`）
4. **打印剔除明细**（分规则列出 + 总数），要 `[Y/n]` 确认（`--auto-apply` 跳过）
5. **重训**：用 baseline 的 `used_params`（不调参） + 筛选后的特征集训练，落八阶段产物
6. `config.json.runtime` 含 `baseline_run / selection {kept/dropped/dropped_by_rule/thresholds/rules_enabled} / analysis_dir / final_params / baseline_metrics`

## 3. 参数说明

### 3.1 run_tuning.py

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--baseline_run` | ✅ | - | `classification-model-training` 产出的 baseline run 目录 |
| `--method` | 否 | `rule` | 调优方法：`rule`（规则推荐）\| `optuna`（贝叶斯搜索） |
| `--n_trials` | 否 | `30` | Optuna 搜索次数（仅 `--method optuna`） |
| `--version` | 否 | `None` | 新 run 显式版本号（仅纯版本号：`v1` / `v2` / `custom-tag`；**不要带 algo/suffix 前缀**如 `xgb-v1` / `tuned-v1` / `feat`，会被拦截报错，避免产出 `xgb-tuned-tuned-v1` 重复前缀目录）；留空自动自增（xgb-tuned-v1, v2, ...）；与 `--label` 都传时 `--version` 优先 |
| `--label` | 否 | `None` | `--version` 的别名 |
| `--output_dir` | 否 | 从 baseline 推断 | 输出根目录；默认 `baseline_run` 的 parent.parent（即 session_dir） |
| `--auto-apply` | 否 | `False` | 跳过交互式确认，自动应用推荐参数 |

### 3.2 select_features.py

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--baseline_run` | ✅ | - | `classification-model-training` 产出的 baseline run 目录 |
| `--analysis_dir` | ✅ | - | `feature-analysis` 输出目录（含 stats.csv / iv_table.csv / psi_table.csv） |
| `--version` | 否 | `None` | 新 run 显式版本号（仅纯版本号：`v1` / `v2` / `custom-tag`；**不要带 algo/suffix 前缀**如 `xgb-v1` / `feat-v1` / `tuned`，会被拦截报错，避免产出 `xgb-feat-feat-v1` 重复前缀目录）；留空自动自增（xgb-feat-v1, v2, ...）；与 `--label` 都传时 `--version` 优先 |
| `--label` | 否 | `None` | `--version` 的别名 |
| `--output_dir` | 否 | 从 baseline 推断 | 输出根目录；默认 `baseline_run` 的 parent.parent（即 session_dir） |
| `--auto-apply` | 否 | `False` | 跳过交互式确认，自动应用筛选结果 |
| `--no-psi` | 否 | `False` | 关闭高 PSI 剔除规则（默认开） |
| `--no-iv` | 否 | `False` | 关闭低 IV 剔除规则（默认开） |
| `--no-missing` | 否 | `False` | 关闭高缺失率剔除规则（默认开） |
| `--psi_threshold` | 否 | `0.10` | PSI 剔除阈值，与 CLAUDE.md PSI 红线一致 |
| `--iv_threshold` | 否 | `0.02` | IV 最低阈值 |
| `--missing_threshold` | 否 | `0.95` | missing_rate 上限 |

## 4. 输出产物

- 调优 run：`<session_dir>/new-models/{algo}-tuned-v{N}/`（例：xgb-tuned-v1）
- 筛选 run：`<session_dir>/new-models/{algo}-feat-v{N}/`（例：xgb-feat-v1）
- baseline run：`<session_dir>/new-models/{algo}-v{N}/`（例：xgb-v1）

`{N}` 为按 algo+suffix 维度自动自增的版本号（扫 `new-models/` 下已有目录取 max+1）。结构均与 `classification-model-training` 完全一致（八阶段：config / features / model / evaluation / predictions / explainability / comparison / logs）。`config.json` 顶部多了 `produced_by: "skills/model-tuning"`，所有 stage 的 `_manifest.json.produced_by` 也标 `skills/model-tuning` 便于事后追溯。

`logs/` 子目录除 `run.log`（tee 训练核心阶段）外，还多一个进程级日志：`run_tuning` 入口落 `logs/run_tuning.log`，`select_features` 入口落 `logs/select_features.log`（均为 process_tee 捕获 run_dir 创建→完成回执全过程，写入 `logs/_manifest.json` 的 `files` 列表）。

```text
<session_dir>/new-models/{algo}-tuned-v{N}/    (或 {algo}-feat-v{N}/)
├── config.json                       # 含 runtime 快照
├── config/train_config.yaml          # 条件生成，见 4.2
├── features/used-feature-list.csv    # feat run 含 dropped 明细
├── model/model.json|.pkl             # 扩展名按 algo
├── model/model_meta.json             # dnn / lr 必有；xgb 由引擎自身落盘
├── model/scorecard.csv               # 仅 lr，algo=lr 且 predictor 持有 scorecard_df 时
├── evaluation/{run_name}_{split}_eval.{json,md,xlsx}   # 三档标准化三件套
├── predictions/*.parquet + report.md
├── explainability/feature-importance.csv (+ shap-summary.csv，仅 xgb)
├── comparison/comparison_{split}.{json,md,xlsx}        # 某 split 缺 JSON 时跳过
├── logs/run.log                      # tee 训练核心阶段
├── logs/run_tuning.log 或 logs/select_features.log    # 进程级日志
└── report.md                         # 单 run 顶层整合报告
<session_dir>/model-comparison/        # 会话级横向对比聚合，每次 run 完成后自动刷新
```

### 4.1 产物内容

| 产物 | 必选 | 说明 |
|---|:---:|---|
| `config.json`（含 runtime） | ✅ | run_tuning：`baseline_run / diagnosis / method / recommended_params / final_params / trials_log / baseline_metrics`；select_features：`baseline_run / selection / analysis_dir / final_params / baseline_metrics`；通用字段：`version / suffix / n_features / metrics / split_mode / split_report` 及 algo 专属 `train_info`（xgb: `best_iteration`；dnn: `best_epoch / total_epochs / early_stopped / best_val_auc`；lr: `n_iter / converged`） |
| `config/train_config.yaml` | 条件生成 | run_tuning：复制 baseline yaml 并用 `final_params` 覆写 `model.params`；select_features：复制并用筛选后特征覆写 `model.features`；baseline 无 yaml 时打 warning 跳过，其他产物正常。PyYAML 未安装时直接复制 baseline yaml（不覆写）。`config/_manifest.json` 含 `source_yaml` 指向 baseline yaml 绝对路径，便于追溯 |
| `features/used-feature-list.csv` | ✅ | 实际使用的特征清单；feat run 三列 `feature_name / status / dropped_by_rule`，被规则剔除的特征单列 `status=dropped_<rule>` 行；tuned run 仅 kept 行 |
| `model/model.json\|.pkl` | ✅ | 模型文件，扩展名按 algo（xgb→json，dnn/lr→pkl） |
| `model/model_meta.json` | ✅ | dnn / lr 必有（algo=xgb 时由 xgb 引擎自身落盘）。结构 `{algo, feature_names, feature_importance, train_info, params, created_at}` |
| `model/scorecard.csv` | 条件生成 | 仅 `algo=lr` 且 predictor 持有 `scorecard_df` 时落盘；列 `[feature, bin, woe, coef, score]`，`model/_manifest.json` 标 `has_scorecard=True` |
| `evaluation/{run_name}_{split}_eval.{json,md,xlsx}` | ✅ | 三档标准化三件套，委托 `classification-model-evaluation` 产出（本 skill 不自带评估报告逻辑）；predictions 阶段写完三档 parquet 后由 `invoke_evaluation.py` 对每档分别调 `classification-model-evaluation/scripts/eval_single.py` |
| `predictions/*.parquet` + `report.md` | ✅ | 三档分档预测明细 |
| `explainability/feature-importance.csv` | ✅ | 特征重要性；`shap-summary.csv` 仅 xgb |
| `comparison/comparison_{split}.{json,md,xlsx}` | ✅ | 评估完成后自动链式调用 `classification-model-comparison`，以调优/筛选所基于的 baseline run 的 `evaluation/` 目录（`snap.run_dir / "evaluation"`）为对比基准，对 train/test/oot 三档做 N-way 对比；未配置 baseline_eval_dir 或某 split 缺 JSON 时跳过，不影响主流程 |
| `logs/run.log` | ✅ | `tee_stdout` 捕获训练核心阶段（数据清洗 / dispatch_train / 评估 / 落盘）的 stdout/stderr |
| `logs/run_tuning.log` 或 `logs/select_features.log` | ✅ | 按入口二选一，`process_tee` 捕获 run_dir 创建→完成回执全过程；两份日志均写入 `logs/_manifest.json` 的 files 列表 |
| `report.md` | ✅ | 单 run 顶层整合报告：模型概览 / 三档指标 / 入模特征 Top20 / 特征重要性 Top20 / vs baseline 对比 / 产物索引；纯读已有产物，不引入新计算 |
| 所有 stage 的 `_manifest.json` | ✅ | `produced_by` 标 `skills/model-tuning`，便于事后追溯 |
| `<session_dir>/model-comparison/` | ✅ | 会话级横向对比聚合，每次 run 完成后自动刷新；聚合脚本缺失或失败时仍 mkdir + 落 fallback `_manifest.json`（标 `status=skipped/failed`）保证目录始终存在 |

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `classification-model-training` | 必须先有其产出的 baseline run 才能调优/筛选；读其 `config.json` 恢复完整 cfg。与 training 的区别：training 做边界安全过滤(剔除危险特征)产 baseline(suffix 为空),本 skill 在其基础上做优化筛选/调参,产 `-tuned` / `-feat` 新 run |
| 上游 | `feature-analysis`（仅 select_features） | 读其 `stats.csv / iv_table.csv / psi_table.csv` |
| 下游 | `classification-model-evaluation` | 评估依赖（强制）：本 skill 不自带评估报告逻辑，predictions 阶段写完三档 parquet 后调其 `eval_single.py` 产标准化三件套到 `evaluation/` |
| 下游 | `classification-model-comparison` | 对比依赖（强制链式）：评估完成后自动以 baseline run 的 `evaluation/` 目录为基准，调其 `compare_models.py` 与新 run 做 N-way 对比，产 `comparison/` 子目录 |
| 下游 | `classification-model-recommend` | tuned / feat run 同样可人工登记到台账 |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ 交互确认流程 | 默认打印诊断/剔除明细后 `[Y/n]` 确认再重训;`--auto-apply` 跳过;非 TTY 环境按默认值处理;**well_fit 状态下未 `--auto-apply` 时默认不重训** |
| ⚠️ Optuna 依赖 | `--method rule`(默认)无新依赖;`--method optuna` 需 `pip install --user "optuna<4"`,首次缺包清晰报错 |
| ⚠️ select_features 不调参 | 仅缩小特征集,用 baseline 的 `used_params` 直接重训;如需调参走流程 A 或串联 `-feat → -tuned` |

> 覆盖范围、不覆盖清单、何时用、session 约定、Optuna 搜索空间详情(各 algo 调哪些超参)、异常处理全表详见 `references/constraints-and-exceptions.md`。

## 7. 异常处理

异常分类与处理方式详见 `references/constraints-and-exceptions.md` 第 9 节。

## 8. 测试

```bash
python -m pytest <skill_dir>/tests/ -q                 # 全量
python -m pytest <skill_dir>/tests/ -q -m "not slow"    # 仅快测
```

---

数据来源：baseline run 来自 `classification-model-training` 产出的 `<session_dir>/new-models/{algo}-v{N}/`；特征分析 csv 来自 `feature-analysis` 产出的 `<session_dir>/sample-features/feature-analysis/analysis/`。
最后更新：2026-07-05
