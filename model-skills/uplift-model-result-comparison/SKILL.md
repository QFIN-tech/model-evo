---
name: uplift-model-result-comparison
description: "已有两个或更多 experiment_ids 或底层 candidate_result_paths 时，用于在同一任务和评估 split 下先生成比较方案，用户确认后做确定性模型比较，输出比较表、摘要报告和推荐模型 path。不要用于训练、调参、最终报告或模型采用决策。"
---

# 多模型结果比较

## 1. 输入依赖

本 skill 读取显式候选 experiment 或底层 result path，规范化 baseline modeling 与 tuning winner 的指标，校验可比性，并在可公平比较且用户确认后给出推荐模型 path。

已有两个或更多 `experiment_ids` 或底层 `candidate_result_paths` 后使用。

## 2. 执行入口

统一 CLI：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

`--output-dir` 必须是已存在的 `run_dir`；runner 不创建 run_dir，不扫描 `latest/current`。`draft_comparison_plan` 创建 comparison flow；正式 `compare` 应显式传入上一步 `outputs.flow_dir` 和 `outputs.comparison_plan_path` 复用该 flow。所有 stdout、result、manifest 和 log 中的持久化 path 都是 `run_dir` 相对路径。

请求 JSON 必须通过 stdin 传入：

```json
{"action": "draft_comparison_plan", "payload": {"experiment_ids": ["exp-001-s_learner_baseline", "exp-002-t_learner_baseline"], "comparison_dataset_role": "test"}}
```

用户确认 comparison plan 后，再执行正式比较：

```json
{"action": "compare", "payload": {"flow_dir": "...", "comparison_plan_path": "..."}}
```

runner stdout 是 JSON object。调用方只读取 stdout 或 result JSON 中的 `outputs.*` 字段，例如 `outputs.comparison_plan_path`、`outputs.recommended_comparison_mode`、`outputs.comparison_table_path`、`outputs.model_comparison_summary_path`、`outputs.recommended_candidate_result_path`、`outputs.report_path`。

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

## 3. Actions 与参数说明

`draft_comparison_plan`

- 输入：`experiment_ids`；底层调试或临时手动接力时可改用 `candidate_result_paths`。
- 可选输入：`comparison_dataset_role`，支持 `test`、`oot`、`train`、`valid` / `validation`。
- 输出：
  - `comparison_plan_path`
  - `recommended_comparison_mode`
  - `requires_user_confirmation=true`

`draft_comparison_plan` 只生成 comparison plan，不执行正式比较；返回 `status=needs_confirmation`。推荐使用 `experiment_ids`，runner 从 `{run_dir}/new-models/{experiment_id}/_experiment_manifest.json` 读取候选信息。`experiment_ids` 和 `candidate_result_paths` 二选一，不能混用。

`compare`

- 推荐输入：`comparison_plan_path`，通常来自 `draft_comparison_plan` 的 `outputs.comparison_plan_path`。
- 底层调试输入：`task_config_path`、`candidate_result_paths`。
- 可选输入：`comparison_dataset_role`，支持 `test`、`oot`、`train`、`valid` / `validation`。
- 输出：
  - `comparison_table_path`
  - `model_comparison_summary_path`
  - `report_path`
  - `recommended_primary_model_path`
  - `recommended_candidate_result_path`

`compare` 的主链路入口是已确认的 `comparison_plan_path`。`candidate_result_paths` 仅作为底层显式输入保留，用于调试、未注册候选或临时手动接力；不作为新体系推荐入口。后续应通过外部候选注册机制进一步弱化或剔除面向用户的裸 `candidate_result_paths` 用法。

`candidate_result_paths` 可以包含 `uplift-model-s-learner-modeling` 或 `uplift-model-t-learner-modeling` 的 `modeling_result_path`，也可以包含 `uplift-model-tuning` 的 S-Learner/T-Learner execute result path；必须至少两个。

comparison 从 `model_candidate.model_spec.model_type` 或 `outputs.learner` 识别 learner，不固定为 `s_learner`。

以下候选会进入 comparison table，但不可推荐：

- 指定 split 缺少 `split_evaluations`：`MISSING_SPLIT_EVALUATION`
- `metric_value = null` 或不可转成 finite number：`NON_RECOMMENDABLE_CANDIDATE`
- population fingerprint 或 dataset role 不一致：`EVALUATION_POPULATION_MISMATCH`
- `metric_name`、`metric_direction`、`auuc_version.package` 或 `auuc_version.version` 不一致：`EVALUATION_METHOD_MISMATCH`

候选缺少 `auuc_version` 或不是 `{"package": "scikit-uplift", "version": "0.5.1"}` 时不可比较。comparison 以 `outputs.split_evaluations.<split>` 作为 AUUC 结果主 contract，不再用旧 `outputs.modeling_metrics.splits` 绕过 split evaluation contract。

## 4. 输出产物

典型输出目录：

```text
{run_dir}/uplift-model-result-comparison/{timestamp}_model_comparison/
  inputs/0001_draft_comparison_plan.request.json
  results/0001_draft_comparison_plan.result.json
  artifacts/
    comparison_plan.v1.json
    model_comparison.v1.csv
    model_comparison_summary.v1.json
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

报告保持轻量业务摘要形态，包含：最终推荐模型、模型产物索引、AUUC 对比、分箱性能对比、可比性与限制。

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-s-learner-modeling` / `uplift-model-t-learner-modeling` | 提供 baseline `experiment_ids` 或 `modeling_result_path` |
| 上游 | `uplift-model-tuning` | 提供 tuned winner `experiment_id` 或 `winner_model_result_path` |
| 下游 | `uplift-model-reporting` | 消费 `model_comparison_summary_path`、`recommended_candidate_result_path` 和报告证据 |

## 6. 执行约束

本 skill 不训练模型，不调参，不重算上游指标，不生成最终建模报告，也不代表模型采用或部署。

`draft_comparison_plan` 只生成 comparison plan，不执行正式比较；返回 `status=needs_confirmation`。推荐使用 `experiment_ids`，runner 从 `{run_dir}/new-models/{experiment_id}/_experiment_manifest.json` 读取候选信息。`experiment_ids` 和 `candidate_result_paths` 二选一，不能混用。

`compare` 的主链路入口是已确认的 `comparison_plan_path`。`candidate_result_paths` 仅作为底层显式输入保留，用于调试、未注册候选或临时手动接力；不作为新体系推荐入口。

## 7. 异常处理

### 7.1 字段兼容与比较计划异常

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

`experiment_ids` 和 `candidate_result_paths` 二选一，不能混用；`candidate_result_paths` 必须至少两个。

### 7.2 不可推荐候选

以下候选会进入 comparison table，但不可推荐：

- 指定 split 缺少 `split_evaluations`：`MISSING_SPLIT_EVALUATION`
- `metric_value = null` 或不可转成 finite number：`NON_RECOMMENDABLE_CANDIDATE`
- population fingerprint 或 dataset role 不一致：`EVALUATION_POPULATION_MISMATCH`
- `metric_name`、`metric_direction`、`auuc_version.package` 或 `auuc_version.version` 不一致：`EVALUATION_METHOD_MISMATCH`

候选缺少 `auuc_version` 或不是 `{"package": "scikit-uplift", "version": "0.5.1"}` 时不可比较。

## 8. 面向用户回复

普通用户回复默认使用中文。先给推荐模型是否产生、候选数量、可比性风险和下一步。不要把 raw JSON、内部目录结构、长路径清单或实现术语作为主回复内容。

除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径；`model_comparison_summary_path`、`recommended_candidate_result_path` 等 JSON 路径只作为内部/下游输入传递。人类可读报告可以按需给出。

确需精确交接时，可在“技术详情”里保留 `comparison_table_path`、`model_comparison_summary_path`、`recommended_candidate_result_path`、`report_path` 等字段名，并说明用途。
