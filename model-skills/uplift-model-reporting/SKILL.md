---
name: uplift-model-reporting
description: "已有 task_config_path、subject_result_paths 和可选 supporting_paths 后，用于从上游结构化结果生成最终 Uplift 建模报告、HTML、facts 和 manifest。不要用于重算指标、训练、调参、比较排序或模型采用执行。"
---

# Uplift 最终报告

## 1. 输入依赖

本 skill 只基于显式 path 聚合上游事实，生成面向业务用户的最终建模报告：

- `report_facts.v1.json`
- `report_manifest.v1.json`
- Markdown 报告
- HTML 报告
- flow root `report.md`

已有 `task_config_path`、`subject_result_paths` 和可选 `supporting_paths` 后使用。

候选模型来源支持：

- `uplift-model-s-learner-modeling`
- `uplift-model-t-learner-modeling`
- `uplift-model-tuning` 产出的 S-Learner tuning result
- `uplift-model-tuning` 产出的 T-Learner tuning result

展示名至少包括 `S-Learner Baseline`、`S-Learner Tuned`、`T-Learner Baseline`、`T-Learner Tuned`。T-Learner 的 `feature_importance.v1.csv` 如果包含 `component` 字段，报告按 component 分段展示 top features。

## 2. 执行入口

统一 CLI：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

`--output-dir` 必须是已存在的 `run_dir`；runner 不创建 run_dir，不扫描 `latest/current`。本 skill 输出到：

```text
{run_dir}/uplift-model-reporting/{timestamp}_uplift_reporting/
  inputs/0001_generate_report.request.json
  results/0001_generate_report.result.json
  artifacts/
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

所有 stdout、result、manifest 和 log 中的持久化 path 都是 `run_dir` 相对路径。输入只通过显式 `task_config_path`、`subject_result_paths`、`supporting_paths` 接力，不读取 latest/current，也不回填 `{run_dir}/report.md`。

请求 JSON 必须通过 stdin 传入：

```json
{"action": "generate_report", "payload": {"task_config_path": "...", "subject_result_paths": ["..."], "supporting_paths": {"comparison_result_path": "..."}}}
```

runner stdout 是 JSON object。调用方只读取 stdout 或 result JSON 中的 `outputs.*` 字段，例如 `outputs.final_report_path`、`outputs.markdown_report_path`、`outputs.html_report_path`、`outputs.report_path`。

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

## 3. Actions 与参数说明

`generate_report`

- 输入：`task_config_path`、`subject_result_paths`。
- 可选输入：`supporting_paths`、`report_decision`、`recommended_candidate_result_path`。
- 输出：
  - `final_report_path`
  - `report_facts_path`
  - `report_manifest_path`
  - `markdown_report_path`
  - `html_report_path`
  - `report_path`

`supporting_paths` 可包含：

- `sample_preparation_result_path`
- `sample_homogeneity_result_path`
- `feature_quality_result_path`
- `comparison_result_path`
- `candidate_result_paths`
- `modeling_sample_spec_path`
- `feature_plan_path`

生成完整最终报告时，应显式传入 `sample_preparation_result_path`、`sample_homogeneity_result_path`、`feature_quality_result_path` 和 `comparison_result_path`；缺少任一项时，报告仍可生成，但会返回 `partial_success` 并在缺失证据中标记对应章节。

如果 comparison 产生推荐候选，调用方应显式传入 `report_decision`，例如：

```json
{"adoption": "adopt_recommended", "decision_maker": "user"}
```

## 4. 输出产物

典型输出目录：

```text
{run_dir}/uplift-model-reporting/{timestamp}_uplift_reporting/
  inputs/
  results/
  artifacts/
    report_facts.v1.json
    report_manifest.v1.json
    uplift-report.v1.md
    uplift-report.v1.html
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

报告保持面向业务用户的最终建模报告形态，包含：执行摘要、任务定义、样本质量、同质性、特征质量、候选模型、模型比较、采用决策、风险与限制、建议下一步、追踪摘要。

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-result-comparison` | 可提供 `model_comparison_summary_path`、`recommended_candidate_result_path` 和采用决策证据 |
| 上游 | `uplift-model-s-learner-modeling` / `uplift-model-t-learner-modeling` / `uplift-model-tuning` | 提供 `subject_result_paths` 中的候选模型结果 |
| 下游 | 无 | 作为最终报告产出，不触发后续 skill |

## 6. 执行约束

本 skill 不重算样本、同质性、特征、模型、调参或比较指标，不训练模型，不创建模型采用状态。

输入只通过显式 `task_config_path`、`subject_result_paths`、`supporting_paths` 接力，不读取 latest/current，也不回填 `{run_dir}/report.md`。

如果 comparison 产生推荐候选，调用方应显式传入 `report_decision`，例如 `{"adoption": "adopt_recommended", "decision_maker": "user"}`。

## 7. 异常处理

### 7.1 字段兼容异常

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

### 7.2 证据缺失处理

报告应基于显式 path 聚合上游事实；缺少可选证据时在报告中表达缺失证据和主要风险，不重算或补造上游指标。

## 8. 面向用户回复

普通用户回复默认使用中文。先给报告是否生成、采用表达、缺失证据和主要风险，再提供 Markdown 报告路径。不要把 raw JSON、内部目录结构、长路径清单或实现术语作为主回复内容。

除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径；`report_facts`、`report_manifest` 等 JSON 路径只作为内部/审计输入传递。Markdown/HTML 报告是面向人阅读的产物，可以按需给出。

确需精确交接时，可在“技术详情”里保留 `final_report_path`、`markdown_report_path`、`html_report_path`、`report_path` 等字段名，并说明用途。
