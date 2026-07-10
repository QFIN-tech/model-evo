---
name: uplift-model-tuning
description: "已有 successful S-Learner 或 T-Learner modeling_result_path 后，用于生成、确认并执行有边界的 LightGBM learner 调参 study。不要用于首次训练、多模型比较、最终报告或模型采用决策。"
---

# 模型调参

## 1. 输入依赖

本 skill 只负责基于已完成的 baseline S-Learner 或 T-Learner 做有边界调参：

- `draft_plan` 生成调参计划草稿，不训练模型。
- `confirm_plan` 记录用户确认和必要风险确认。
- `execute` 基于 confirmed plan 真实训练候选 trial，复用 `trial_000` baseline 作为对照，并输出 study winner。

本 skill 不比较多个候选集合，不生成最终建模报告，不声明 primary/final/deployment model。

本 skill 是当前 bundle 中唯一调参入口；不要新增或调用 `uplift-t-learner-tuning`。

已有 successful S-Learner 或 T-Learner `modeling_result_path` 后使用。

## 2. 执行入口

统一 CLI：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

`--output-dir` 必须是已存在的 `run_dir`；runner 不创建 run_dir，不扫描 `latest/current`。`draft_plan` / `confirm_plan` 输出到 tuning flow；`confirm_plan` 必须显式传入 `outputs.flow_dir` 复用该 flow。`draft_plan` 校验但不创建 `experiment_id`，`execute` 只能使用 confirmed tuning plan 内的 `experiment_id` 创建 winner experiment，目标 experiment 已存在时返回 `needs_input / EXPERIMENT_ALREADY_EXISTS`。

请求 JSON 必须通过 stdin 传入：

```json
{"action": "draft_plan", "payload": {"experiment_id": "exp-003-s_learner_tuned", "task_config_path": "...", "modeling_sample_spec_path": "...", "feature_plan_path": "...", "modeling_result_path": "..."}}
```

runner stdout 是 JSON object。调用方只读取 stdout 或 result JSON 中的 `outputs.*` 字段，例如 `outputs.tuning_plan_path`、`outputs.winner_model_result_path`、`outputs.model_candidate_path`、`outputs.report_path`。

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

## 3. Actions 与参数说明

`draft_plan`

- 输入：`experiment_id`、`task_config_path`、`modeling_sample_spec_path`、`feature_plan_path`、`modeling_result_path`。
- 可选输入：`preferences.profile`、`preferences.selection_dataset_role`、`preferences.budget`、`preferences.fixed_parameters`、`preferences.search_space_overrides`、`preferences.candidate_grid`。
- 输出：`tuning_plan_path`。

`tuning_plan` 必须显式记录：

- `model_spec.model_type`
- `model_spec.base_estimators`
- `parameter_strategy`

v1 只支持 `parameter_strategy = "shared"`。收到 `parameter_strategy = "per_arm"`、`per_arm_parameters`、`component_parameters`、`treatment_parameters` 或 `control_parameters` 时，runner 返回 `needs_input`，`issues[].code = "UNSUPPORTED_PARAMETER_STRATEGY"`。

如果 execute 发现 confirmed plan 中的 `model_spec` 和 baseline result 暴露的 `model_spec` 不一致，runner 返回 `needs_input`，`issues[].code = "MODEL_SPEC_MISMATCH"`。

`confirm_plan`

- 输入：`source_plan_path` 或 `tuning_plan_path`、`warning_acknowledgements`、`confirmed_by`。
- 输出：confirmed `tuning_plan_path`。

`execute`

- 输入：confirmed `tuning_plan_path`。
- 输出：
  - `experiment_id`
  - `experiment_dir`
  - `tuning_plan_path`
  - `tuning_result_path`
  - `trial_metrics_path`
  - `leaderboard_path`
  - `winner_result_path`
  - `winner_model_result_path`
  - `winner_metrics`
  - `winner_metrics_path`
  - `model_candidate_path`
  - `evaluation_metrics_path`
  - `model_artifact_path`
  - `report_path`
  - `tuning_report_path`

读取 baseline 时，selection split 必须存在 `outputs.split_evaluations.<split>.auuc_version = {"package": "scikit-uplift", "version": "0.5.1"}`；缺失或版本不一致时 runner 返回明确错误并提示重跑 baseline。

快速验收只能通过显式小 `max_trials`、小 `candidate_grid` 或窄搜索空间控制耗时；这仍是真实调参执行，不是 mock。

## 4. 输出产物

典型输出目录：

```text
{run_dir}/uplift-model-tuning/{timestamp}_model_tuning/
  inputs/0001_draft_plan.request.json
  results/0001_draft_plan.result.json
  artifacts/
    tuning_plan.draft.v1.json
    tuning_plan.confirmed.v1.json
  _flow_manifest.json
  _flow_log.jsonl

{run_dir}/new-models/{experiment_id}/
  inputs/0001_execute.request.json
  results/0001_execute.result.json
  artifacts/
    trial_metrics.v1.json
    leaderboard.v1.csv
    winner_metrics.v1.json
    winner_model_result.v1.json
    model_candidate.v1.json
    tuning_report.md
    s_learner_tuned_model.v1.joblib
    t_learner_tuned_model.v1.joblib
    t_learner_tuned_treatment_model.v1.txt
    t_learner_tuned_control_model.v1.txt
  _experiment_manifest.json
  _experiment_log.jsonl
  report.md
```

所有 stdout、result、manifest 和 log 中的持久化 path 都是 `run_dir` 相对路径。下游比较优先通过 `experiment_ids` 进入 `uplift-model-result-comparison` 的 `draft_comparison_plan`。

如果 winner 是 `trial_000`，`model_artifact_path` 可指向 baseline 模型文件；`winner_model_result_path` 仍由本次 study 生成，便于下游比较。

T-Learner 的非 baseline trial 使用同一组 shared LightGBM 参数训练 treatment/control 两个子模型。`trial_000` 复用 baseline T-Learner model 和 baseline evidence，不重新训练。

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-s-learner-modeling` / `uplift-model-t-learner-modeling` | 提供 successful baseline `modeling_result_path` |
| 下游 | `uplift-model-result-comparison` | 消费 winner `experiment_id`、`winner_model_result_path` 或 `model_candidate_path` 参与模型比较 |
| 下游 | `uplift-model-reporting` | 可消费 `winner_model_result_path`、`model_candidate_path`、`tuning_result_path` 生成最终报告 |

## 6. 执行约束

本 skill 不比较多个候选集合，不生成最终建模报告，不声明 primary/final/deployment model。

本 skill 是当前 bundle 中唯一调参入口；不要新增或调用 `uplift-t-learner-tuning`。

快速验收只能通过显式小 `max_trials`、小 `candidate_grid` 或窄搜索空间控制耗时；这仍是真实调参执行，不是 mock。

## 7. 异常处理

### 7.1 缺依赖处理

本 skill 需要 `pandas`、`numpy`、`joblib`、`lightgbm` 和 `scikit-uplift==0.5.1`。runner 不安装依赖。缺包或 LightGBM 无法加载时返回结构化失败，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

### 7.2 计划、版本与 experiment 异常

收到 `parameter_strategy = "per_arm"`、`per_arm_parameters`、`component_parameters`、`treatment_parameters` 或 `control_parameters` 时，runner 返回 `needs_input`，`issues[].code = "UNSUPPORTED_PARAMETER_STRATEGY"`。

如果 execute 发现 confirmed plan 中的 `model_spec` 和 baseline result 暴露的 `model_spec` 不一致，runner 返回 `needs_input`，`issues[].code = "MODEL_SPEC_MISMATCH"`。

读取 baseline 时，selection split 必须存在 `outputs.split_evaluations.<split>.auuc_version = {"package": "scikit-uplift", "version": "0.5.1"}`；缺失或版本不一致时 runner 返回明确错误并提示重跑 baseline。

目标 experiment 已存在时返回 `needs_input / EXPERIMENT_ALREADY_EXISTS`。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

## 8. 面向用户回复

普通用户回复默认使用中文。先给调参是否完成、winner 是否改善、使用的 selection split、主要风险和下一步。不要把 raw JSON、内部目录结构、长路径清单或实现术语作为主回复内容。

除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径；`tuning_plan_path`、`winner_model_result_path`、`model_candidate_path` 等 JSON 路径只作为内部/下游输入传递。人类可读报告可以按需给出。

确需精确交接时，可在“技术详情”里保留 `tuning_plan_path`、`winner_model_result_path`、`model_candidate_path`、`report_path` 等字段名，并说明用途。
