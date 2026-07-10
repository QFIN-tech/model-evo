---
name: uplift-model-t-learner-modeling
description: "已有 confirmed task_config_path、modeling_sample_spec_path 和 feature_plan_path 后，用于训练 baseline T-Learner 模型并输出模型元数据、评估指标、打分表、uplift 分箱、分 component 特征重要性和报告。不要用于特征诊断、调参、模型比较或最终报告生成。"
---

# T-Learner 基线建模

## 1. 输入依赖

本 skill 只负责 baseline T-Learner 训练和评估：

- 读取已确认的 `task_config_path`、`modeling_sample_spec_path` 和 `feature_plan_path`。
- 只使用 `feature_plan_path` 中已确认的业务特征训练模型。
- treatment column 只用于拆分 treatment/control arm，不进入模型特征。
- 在完整 train split 的 `selected_features` 上 fit 一套共享 encoder，两个子模型共用同一 encoded feature space。
- binary outcome 使用两个 `LGBMClassifier`，continuous outcome 使用两个 `LGBMRegressor`。
- 输出模型候选、模型元数据、评估指标、打分表、uplift 分箱、分 component 特征重要性和报告。

已有 confirmed `task_config_path`、`modeling_sample_spec_path` 和 `feature_plan_path` 后使用。

## 2. 执行入口

统一 CLI：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

`--output-dir` 必须是已存在的 `run_dir`；runner 不创建 run_dir，不扫描 `latest/current`。`train` 必须显式提供 `experiment_id`，格式为 `exp-001-name` 这类 `^exp-[0-9]{3}-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$`。目标 experiment 已存在时返回 `needs_input / EXPERIMENT_ALREADY_EXISTS`，不覆盖、不恢复。

请求 JSON 必须通过 stdin 传入：

```json
{"action": "train", "payload": {"experiment_id": "exp-001-t_learner", "task_config_path": "...", "modeling_sample_spec_path": "...", "feature_plan_path": "..."}}
```

runner stdout 是 JSON object。调用方只读取 stdout 或 result JSON 中的 `outputs.*` 字段，例如 `outputs.modeling_result_path`、`outputs.model_candidate_path`、`outputs.report_path`。

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

## 3. Actions 与参数说明

`train`

- 输入：`experiment_id`、`task_config_path`、`modeling_sample_spec_path`、`feature_plan_path`。
- 可选输入：`parameter_mode`、`parameter_overrides`。
- 输出：
  - `experiment_id`
  - `experiment_dir`
  - `modeling_result_path`
  - `model_candidate_path`
  - `model_metadata_path`
  - `evaluation_metrics_path`
  - `score_frame_path`
  - `uplift_bins_path`
  - `feature_importance_path`
  - `model_artifact_path`
  - `report_path`
  - `learner`
  - `modeling_metrics`
  - `split_evaluations`

`parameter_mode` 支持：

- `recommended_defaults`
- `user_overrides`

`parameter_overrides` 只允许 LightGBM 训练参数，例如 `n_estimators`、`learning_rate`、`num_leaves`、`max_depth`、`min_child_samples`、`random_state`、`n_jobs`。

## 4. 输出产物

典型输出目录：

```text
{run_dir}/new-models/{experiment_id}/
  inputs/0001_train.request.json
  results/0001_train.result.json
  artifacts/
    t_learner_model.v1.joblib
    t_learner_treatment_model.v1.txt
    t_learner_control_model.v1.txt
    model_candidate.v1.json
    model_metadata.v1.json
    evaluation_metrics.v1.json
    score_frame.v1.csv
    uplift_bins.v1.csv
    feature_importance.v1.csv
  _experiment_manifest.json
  _experiment_log.jsonl
  report.md
```

所有 stdout、result、manifest 和 log 中的持久化 path 都是 `run_dir` 相对路径。`candidate_id` 使用 `experiment_id`。下游比较优先通过 `experiment_ids` 进入 `uplift-model-result-comparison` 的 `draft_comparison_plan`。

`score_frame.v1.csv` 使用 `treatment_prediction`、`control_prediction` 和 `uplift_score` 表达两个潜在 outcome 预测差值。

`feature_importance.v1.csv` 是单文件，包含 `component` 字段；`component` 取值为 `treatment_outcome` 或 `control_outcome`。

`outputs.split_evaluations` 使用和 `uplift-model-tuning` winner 一致的结构，按 split 暴露 `population_fingerprint`、`metric_name`、`metric_value`、`metric_method`、`metric_version`、`auuc_version` 和 `support`。`auuc_version` 固定标记 `{"package": "scikit-uplift", "version": "0.5.1"}`。

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-feature-quality-analysis` | 提供 confirmed `feature_plan_path` |
| 上游 | `uplift-model-sample-preparation` | 提供 confirmed `modeling_sample_spec_path` |
| 下游 | `uplift-model-tuning` | 消费 successful T-Learner `modeling_result_path` 生成调参 study |
| 下游 | `uplift-model-result-comparison` | 消费 `experiment_ids` 或 `modeling_result_path` 参与模型比较 |

## 6. 执行约束

本 skill 不选择特征，不调参，不比较多个模型，不生成最终业务报告。

### 样本支持与风险

- train split 缺 treatment 或 control arm 时返回 `needs_input`，`issues[].code = "INSUFFICIENT_TRAIN_ARM_SUPPORT"`。
- binary outcome 下，train split 任一 arm 只有单一 outcome class 时返回 `needs_input`，`issues[].code = "SINGLE_CLASS_TRAIN_ARM"`。
- evaluation split 缺 treatment 或 control arm 时训练仍可成功，但该 split 的 `metric_value` 为 `null`，并记录 `issues[].code = "INSUFFICIENT_EVALUATION_ARM_SUPPORT"`。
- T-Learner 依赖 treatment/control 两个 arm 的可比性；如果上游样本选择或 treatment assignment 存在偏差，uplift 排序应保守解读。

### 模型文件安全

`t_learner_model.v1.joblib` 只能从可信来源加载。不要加载来源不明或被外部修改过的 joblib 模型文件。

## 7. 异常处理

### 7.1 缺依赖处理

本 skill 需要 `pandas`、`numpy`、`joblib`、`lightgbm` 和 `scikit-uplift==0.5.1`。runner 不安装依赖。缺包时返回结构化失败，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

AUUC 由 `sklift.metrics.uplift_auc_score` 和 `sklift.metrics.uplift_curve` 计算。

### 7.2 字段兼容与 experiment 异常

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

目标 experiment 已存在时返回 `needs_input / EXPERIMENT_ALREADY_EXISTS`，不覆盖、不恢复。

## 8. 面向用户回复

普通用户回复默认使用中文。先给模型训练是否完成、测试集评估质量、主要风险和下一步。不要把 raw JSON、内部目录结构、长路径清单或实现术语作为主回复内容。

baseline 建模成功后，默认提示“下一步可进行模型调参”。如果当前 run_dir 下已有多个 baseline 模型，则提示“下一步可对当前模型调参，或先进行多模型比较以选择 baseline winner；若选择跳过调参，可直接进入比较”。不要仅因为已有多个 baseline 候选就只推荐多模型比较。

除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径；`modeling_result_path`、`model_candidate_path` 等 JSON 路径只作为内部/下游输入传递。人类可读报告可以按需给出。

确需精确交接时，可在“技术详情”里保留 `modeling_result_path`、`model_candidate_path`、`report_path` 等字段名，并说明用途。
