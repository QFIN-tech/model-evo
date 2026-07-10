---
name: uplift-model-sample-homogeneity-check
description: 当用户想对两组数据做同质性检验时使用，例如“我想判断这两组数据是否同质/可比”；也可以在样本准备产出 confirmed modeling_sample_spec_path 后使用：起草、确认并运行 treatment/control 协变量均衡性诊断。不要用于样本切分、特征选择、模型训练、调参、模型比较或最终报告。
---

# 样本同质性检查

## 1. 输入依赖

本 skill 在 `uplift-model-sample-preparation` 已产出 `modeling_sample_spec_path` 后使用，用来检查 treatment/control 两组在已确认干预前协变量上的可比性。

流程：

```text
plan_covariates
-> confirm homogeneity_covariate_plan
-> run_diagnostics
-> homogeneity report/result artifacts
```

## 2. 执行入口

使用统一 runner：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

`--output-dir` 必须是已存在的 `run_dir`；runner 不创建 run_dir，不扫描 `latest/current`。本 skill 输出到：

```text
{run_dir}/uplift-model-sample-homogeneity-check/{timestamp}_homogeneity_check/
  inputs/0001_<action>.request.json
  results/0001_<action>.result.json
  artifacts/
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

所有 stdout、result、manifest 和 log 中的持久化 path 都是 `run_dir` 相对路径。`plan_covariates` 创建 flow；后续 action 必须显式传入上一步 `outputs.flow_dir`，不得从 artifact path 反推 flow。

request JSON 必须通过 UTF-8 stdin 传入：

```json
{"action": "plan_covariates", "payload": {"modeling_sample_spec_path": "..."}}
```

stdout 和落盘 result JSON 都使用共同返回 shape：

```text
status, summary, outputs, issues, next_steps
```

只从 `outputs.result_path` 读取当前 action 的 result JSON。存在报告时读取 `outputs.report_path`。复用同一个 flow 时读取 `outputs.flow_dir` 后作为下一次请求的 `flow_dir` 输入。不要读取或传播顶层 `flow_dir`、`result_path`、`warnings`。

旧输入字段如果以 `_ref` 或 `_refs` 结尾，runner 会返回 `needs_input`，并在 `issues` / `next_steps` 中给出对应 `_path` 或 `_paths` 替代字段。

如果缺少 Python 包，runner 返回 `status=failed`、`code=PYTHON_IMPORT_ERROR` issue，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

## 3. Actions 与参数说明

### plan_covariates

必填 payload：

```json
{
  "modeling_sample_spec_path": "<confirmed modeling_sample_spec artifact>"
}
```

可选 payload：

```json
{
  "user_covariates": ["age", "tenure", "region"],
  "max_categorical_levels": 10,
  "prior_plan_path": null
}
```

如果省略 `user_covariates`，runner 会从启用的 split 数据集中推荐干预前候选协变量。

输出：

```text
outputs.homogeneity_covariate_plan_path
```

该 action 写入 `artifacts/homogeneity_covariate_plan.draft.v1.json`，并返回 `needs_confirmation`。

### confirm_artifact

必填 payload：

```json
{
  "source_draft_path": "<draft homogeneity_covariate_plan path>",
  "artifact_kind": "homogeneity_covariate_plan",
  "confirmed_by": "user"
}
```

该 action 会基于启用的 split 数据集校验被选协变量，然后写入：

```text
artifacts/homogeneity_covariate_plan.confirmed.v1.json
```

输出：

```text
outputs.homogeneity_covariate_plan_path
```

### run_diagnostics

必填 payload：

```json
{
  "modeling_sample_spec_path": "<confirmed modeling_sample_spec artifact>",
  "homogeneity_covariate_plan_path": "<confirmed covariate plan artifact>"
}
```

可选 payload：

```json
{
  "diagnostic_config_overrides": {}
}
```

该 action 计算 split 级 treatment predictability AUC 和 SMD 诊断。严重或关键均衡性风险不代表 runner 失败，而是表示继续下游前需要用户明确接受风险。

输出：

```text
outputs.sample_homogeneity_result_path
outputs.report_path
outputs.covariates_path
outputs.smd_detail_path
outputs.auc_feature_importance_path
outputs.diagnostic_config_path
```

产物：

```text
artifacts/covariates.v1.csv
artifacts/smd_detail.v1.csv
artifacts/auc_feature_importance.v1.csv
  artifacts/diagnostic_config.json
  report.md
```

## 4. 输出产物

典型输出目录：

```text
{run_dir}/uplift-model-sample-homogeneity-check/{timestamp}_homogeneity_check/
  inputs/0001_<action>.request.json
  results/0001_<action>.result.json
  artifacts/
    homogeneity_covariate_plan.draft.v1.json
    homogeneity_covariate_plan.confirmed.v1.json
    covariates.v1.csv
    smd_detail.v1.csv
    auc_feature_importance.v1.csv
    diagnostic_config.json
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

关键下游路径：

- `outputs.homogeneity_covariate_plan_path`
- `outputs.sample_homogeneity_result_path`
- `outputs.report_path`
- `outputs.covariates_path`
- `outputs.smd_detail_path`
- `outputs.auc_feature_importance_path`
- `outputs.diagnostic_config_path`

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-sample-preparation` | 提供 confirmed `modeling_sample_spec_path` |
| 下游 | `uplift-model-feature-quality-analysis` | 消费 `sample_homogeneity_result_path`；如跳过同质性检查，必须由用户明确接受风险 |

## 6. 执行约束

本 skill 只负责起草、确认并运行 treatment/control 协变量均衡性诊断，不做样本切分、特征选择、模型训练、调参、模型比较或最终报告。

所有 split 数据集读取必须经过 `_common.data_loader`，其实现位于 `model-skills/_uplift-common/scripts/_common/data_loader.py`，并与 `uplift-model-sample-preparation` 使用同一公共实现。Gate 1 支持 `local_csv`；`local_parquet` 仅预留 kind，并返回明确 unsupported 结果，不做 fallback。

- `plan_covariates` 返回 `needs_confirmation` 时，必须请用户确认或修改协变量计划，不能直接运行诊断。
- `run_diagnostics` 如果发现 severe 或 critical 级别均衡性风险，必须明确询问用户是否接受该风险后再进入特征或建模链路。
- 只有拿到 `outputs.sample_homogeneity_result_path` 且用户已处理必要风险确认后，才建议继续后续 skill。

## 7. 异常处理

### 7.1 缺依赖处理

如果缺少 Python 包，runner 返回 `status=failed`、`code=PYTHON_IMPORT_ERROR` issue，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

### 7.2 字段兼容与诊断异常

旧输入字段如果以 `_ref` 或 `_refs` 结尾，runner 会返回 `needs_input`，并在 `issues` / `next_steps` 中给出对应 `_path` 或 `_paths` 替代字段。

如果存在跳过变量、缺失协变量或诊断失败，必须说明影响范围，并请用户补充字段、修订协变量计划或返回样本准备步骤。

## 8. 面向用户回复

- 默认使用中文回复，除非用户明确要求英文。
- 先给业务结论：均衡性风险等级、主要不均衡驱动变量、是否需要用户接受风险后才能继续。
- 有报告时可以链接 `outputs.report_path`；需要交接时内部使用 `outputs.sample_homogeneity_result_path`，默认不要把该 JSON 路径展示给用户。
- 不展示 raw stdout、raw result JSON 或内部长路径清单，除非用户明确要求调试。不要把路径放进独立代码块作为主要回复内容。
- 除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径。JSON 产物和 `outputs.*_path` 仍然是机器接口与下游 skill 的契约，可以在内部读取和传递。
