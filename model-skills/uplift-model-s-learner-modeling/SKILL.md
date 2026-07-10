---
name: uplift-model-s-learner-modeling
description: "已有 confirmed task_config_path、modeling_sample_spec_path 和 feature_plan_path 后，用于训练 baseline S-Learner 模型并输出模型元数据、评估指标、打分表、uplift 分箱、特征重要性和报告。不要用于特征诊断、调参、模型比较或最终报告生成。"
---

# S-Learner 基线建模

## 1. 输入依赖

本 skill 只负责 baseline S-Learner 训练和评估：

- 读取已确认的 `task_config_path`、`modeling_sample_spec_path` 和 `feature_plan_path`。
- 使用 `feature_plan_path` 中的已确认特征列表训练 LightGBM S-Learner。
- 输出模型候选、模型元数据、评估指标、打分表、uplift 分箱、特征重要性和报告。

已有 confirmed `task_config_path`、`modeling_sample_spec_path` 和 `feature_plan_path` 后使用。

## 2. 执行入口

统一 CLI：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

请求 JSON 必须通过 stdin 传入：

```json
{"action": "train", "payload": {"experiment_id": "exp-001-s_learner_baseline", "task_config_path": "...", "modeling_sample_spec_path": "...", "feature_plan_path": "..."}}
```

runner stdout 是 JSON object。调用方只读取 stdout 或 result JSON 中的 `outputs.*` 字段，例如 `outputs.modeling_result_path`、`outputs.model_candidate_path`、`outputs.report_path`。

`--output-dir` 必须是已存在的项目运行目录 `run_dir`。`train` 不再写入普通 skill 调用目录，而是写入 `{run_dir}/new-models/{experiment_id}/`。`experiment_id` 由调用方显式提供，runner 只校验安全字符并创建对应 experiment 目录；如果目录已存在，返回 `needs_input` / `EXPERIMENT_ALREADY_EXISTS`，不覆盖、不自动恢复。`candidate_id` 使用 `experiment_id`。

`outputs.split_evaluations` 使用和 `uplift-model-tuning` winner 一致的结构，按 split 暴露 `population_fingerprint`、`metric_name`、`metric_value`、`metric_method`、`metric_version`、`auuc_version` 和 `support`。`auuc_version` 固定标记 `{"package": "scikit-uplift", "version": "0.5.1"}`。旧的 `outputs.modeling_metrics` 仍保留，供历史链路兼容。

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
    s_learner_model.v1.joblib
    s_learner_model.v1.txt
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

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-feature-quality-analysis` | 提供 confirmed `feature_plan_path` |
| 上游 | `uplift-model-sample-preparation` | 提供 confirmed `modeling_sample_spec_path` |
| 下游 | `uplift-model-tuning` | 消费 successful S-Learner `modeling_result_path` 生成调参 study |
| 下游 | `uplift-model-result-comparison` | 消费 `experiment_ids` 或 `modeling_result_path` 参与模型比较 |

## 6. 执行约束

本 skill 不选择特征，不调参，不比较多个模型，不生成最终业务报告。

`parameter_overrides` 只允许 LightGBM 训练参数，例如 `n_estimators`、`learning_rate`、`num_leaves`、`max_depth`、`min_child_samples`、`random_state`、`n_jobs`。

## 7. 异常处理

### 7.1 缺依赖处理

本 skill 需要 `pandas`、`numpy`、`joblib`、`lightgbm` 和 `scikit-uplift==0.5.1`。runner 不安装依赖。缺包时返回结构化失败，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

AUUC 由 `sklift.metrics.uplift_auc_score` 和 `sklift.metrics.uplift_curve` 计算.

### 7.2 字段兼容与 experiment 异常

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

如果目标 experiment 目录已存在，返回 `needs_input` / `EXPERIMENT_ALREADY_EXISTS`，不覆盖、不自动恢复。

## 8. 面向用户回复

普通用户回复默认使用中文。先给模型训练是否完成、测试集评估质量、主要风险和下一步。不要把 raw JSON、内部目录结构、长路径清单或实现术语作为主回复内容。

baseline 建模成功后，默认提示“下一步可进行模型调参”。如果当前 run_dir 下已有多个 baseline 模型，则提示“下一步可对当前模型调参，或先进行多模型比较以选择 baseline winner；若选择跳过调参，可直接进入比较”。不要仅因为已有多个 baseline 候选就只推荐多模型比较。

除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径；`modeling_result_path`、`model_candidate_path` 等 JSON 路径只作为内部/下游输入传递。人类可读报告可以按需给出。

确需精确交接时，可在“技术详情”里保留 `modeling_result_path`、`model_candidate_path`、`report_path` 等字段名，并说明用途。
