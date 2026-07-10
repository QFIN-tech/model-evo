---
name: uplift-model-sample-preparation
description: "在 uplift-model-task-spec 产出 confirmed task_config_path 后使用：检查样本语义、起草并确认切分方案、生成建模样本产物。不要用于任务定义、特征选择、建模、调参、模型比较或最终报告。"
---

# 样本准备

## 1. 输入依赖

本 skill 在 `uplift-model-task-spec` 已产出 confirmed `task_config_path` 后使用，用来准备 uplift 建模样本。

流程：

```text
raw_sample_check
-> confirm data_semantics_plan
-> split_planning
-> confirm split_plan
-> post_split_validation
-> modeling_sample_spec.confirmed
-> uplift-model-sample-homogeneity-check
```

## 2. 执行入口

使用统一 runner：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

`--output-dir` 必须是已存在的 `run_dir`；runner 不创建 run_dir，不扫描 `latest/current`。本 skill 输出到：

```text
{run_dir}/uplift-model-sample-preparation/{timestamp}_sample_preparation/
  inputs/0001_<action>.request.json
  results/0001_<action>.result.json
  artifacts/
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

所有 stdout、result、manifest 和 log 中的持久化 path 都是 `run_dir` 相对路径。`raw_sample_check` 创建 flow；后续 action 必须显式传入上一步 `outputs.flow_dir`，不得从 artifact path 反推 flow。

request JSON 必须通过 UTF-8 stdin 传入：

```json
{"action": "raw_sample_check", "payload": {"task_config_path": "..."}}
```

stdout 和落盘 result JSON 都使用共同返回 shape：

```text
status, summary, outputs, issues, next_steps
```

只从 `outputs.result_path` 读取当前 action 的 result JSON。存在报告时读取 `outputs.report_path`。复用同一个 flow 时读取 `outputs.flow_dir` 后作为下一次请求的 `flow_dir` 输入。不要读取或传播顶层 `flow_dir`、`result_path`、`warnings`。

旧输入字段如果以 `_ref` 或 `_refs` 结尾，runner 会返回 `needs_input`，并在 `issues` / `next_steps` 中给出对应 `_path` 或 `_paths` 替代字段。业务数据字段 `data_ref` 仍可出现在 task config 内。

如果缺少 Python 包，runner 返回 `status=failed`、`code=PYTHON_IMPORT_ERROR` issue，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

## 3. Actions 与参数说明

### raw_sample_check

必填 payload：

```json
{
  "task_config_path": "<confirmed task_config artifact>"
}
```

可选 payload：

```json
{
  "data_source": {"kind": "local_csv", "path": "..."}
}
```

如果省略 `data_source`，runner 使用 `task_config.external_inputs.data_source` 或 task config 中的 `data_ref`。

输出：

```text
outputs.data_semantics_plan_path
```

该 action 写入 `artifacts/data_semantics_plan.draft.v1.json`，并返回 `needs_confirmation`。

### confirm_artifact

必填 payload：

```json
{
  "source_draft_path": "<draft artifact path>",
  "artifact_kind": "data_semantics_plan",
  "confirmed_by": "user"
}
```

`artifact_kind` 可为 `data_semantics_plan` 或 `split_plan`。确认后写入：

```text
artifacts/data_semantics_plan.confirmed.v1.json
artifacts/split_plan.confirmed.v1.json
```

输出会暴露 `outputs.data_semantics_plan_path` 或 `outputs.split_plan_path`。

### split_planning

必填 payload：

```json
{
  "task_config_path": "<confirmed task_config artifact>",
  "data_semantics_plan_path": "<confirmed data_semantics_plan artifact>"
}
```

可选 payload：

```json
{
  "split_method": "random",
  "ratios": {"train": 0.8, "test": 0.2, "valid": 0.0},
  "random_seed": 42
}
```

时间切分需提供 `split_method=time`、`time_column`、`oot_window.start`、可选 `oot_window.end_exclusive` 和可选 `non_oot_random_ratios`。

输出：

```text
outputs.split_plan_path
```

该 action 写入 `artifacts/split_plan.draft.v1.json`，并返回 `needs_confirmation`。

### post_split_validation

必填 payload：

```json
{
  "task_config_path": "<confirmed task_config artifact>",
  "data_semantics_plan_path": "<confirmed data_semantics_plan artifact>",
  "split_plan_path": "<confirmed split_plan artifact>"
}
```

该 action 会 dry-run 切分，校验启用切分中的 treatment/control 和 outcome 支撑；只有验证通过才写入产物。

输出：

```text
outputs.modeling_sample_spec_path
outputs.modeling_sample_path
outputs.modeling_sample_paths
outputs.report_path
```

产物：

```text
artifacts/modeling_sample_spec.confirmed.v1.json
artifacts/modeling_sample.v1.csv
artifacts/modeling_sample.train.v1.csv
artifacts/modeling_sample.test.v1.csv
artifacts/modeling_sample.valid.v1.csv
artifacts/modeling_sample.oot.v1.csv
report.md
```

只写入启用的 split 数据集。

## 4. 输出产物

典型输出目录：

```text
{run_dir}/uplift-model-sample-preparation/{timestamp}_sample_preparation/
  inputs/0001_<action>.request.json
  results/0001_<action>.result.json
  artifacts/
    data_semantics_plan.draft.v1.json
    data_semantics_plan.confirmed.v1.json
    split_plan.draft.v1.json
    split_plan.confirmed.v1.json
    modeling_sample_spec.confirmed.v1.json
    modeling_sample.v1.csv
    modeling_sample.train.v1.csv
    modeling_sample.test.v1.csv
    modeling_sample.valid.v1.csv
    modeling_sample.oot.v1.csv
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

关键下游路径：

- `outputs.data_semantics_plan_path`
- `outputs.split_plan_path`
- `outputs.modeling_sample_spec_path`
- `outputs.modeling_sample_path`
- `outputs.modeling_sample_paths`
- `outputs.report_path`

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-task-spec` | 提供 confirmed `task_config_path` |
| 下游 | `uplift-model-sample-homogeneity-check` | 消费 confirmed `modeling_sample_spec_path` 做 treatment/control 同质性检查 |

## 6. 执行约束

本 skill 只负责样本语义检查、切分方案确认和建模样本产物生成，不做任务定义、特征选择、建模、调参、模型比较或最终报告。

所有表格读取必须经过 `_common.data_loader`，其实现位于 `model-skills/_uplift-common/scripts/_common/data_loader.py`。Gate 1 支持 `local_csv`；`local_parquet` 仅预留 kind，并返回明确 unsupported 结果，不做 fallback。

- `raw_sample_check` 返回 `needs_confirmation` 时，必须请用户确认或修改样本语义计划，不能直接进入切分。
- `split_planning` 返回 `needs_confirmation` 时，必须请用户确认或修改切分方案，不能直接生成建模样本。
- 只有拿到 confirmed `modeling_sample_spec_path` 后，才建议进入 `uplift-model-sample-homogeneity-check`。

## 7. 异常处理

### 7.1 缺依赖处理

如果缺少 Python 包，runner 返回 `status=failed`、`code=PYTHON_IMPORT_ERROR` issue，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

### 7.2 字段兼容与流程异常

旧输入字段如果以 `_ref` 或 `_refs` 结尾，runner 会返回 `needs_input`，并在 `issues` / `next_steps` 中给出对应 `_path` 或 `_paths` 替代字段。业务数据字段 `data_ref` 仍可出现在 task config 内。

`post_split_validation` 如果返回失败或需要输入，必须说明阻塞原因，并请用户补充字段、修订语义计划或修订切分方案。

## 8. 面向用户回复

- 默认使用中文回复，除非用户明确要求英文。
- 先给业务结论：样本语义是否可确认、切分方案是否需要确认、建模样本 spec 是否已生成。
- 有报告时可以链接 `outputs.report_path`；需要交接到下一步时内部使用 `outputs.modeling_sample_spec_path`，默认不要把该 JSON 路径展示给用户。
- 不展示 raw stdout、raw result JSON 或内部长路径清单，除非用户明确要求调试。不要把路径放进独立代码块作为主要回复内容。
- 除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径。JSON 产物和 `outputs.*_path` 仍然是机器接口与下游 skill 的契约，可以在内部读取和传递。
- 当 action 产出 draft artifact 且需要用户确认时，必须读取 draft artifact，用业务字段和值解释“要确认什么”，不要只展示 `outputs.*_path`。
- 技术路径最多作为末尾“技术详情”一行展示；用户没有要求调试时，可以省略草稿路径。

### `raw_sample_check` 需要确认时的回复结构

`raw_sample_check` 返回 `needs_confirmation` 后，回复必须面向业务用户，使用类似结构：

```text
样本语义检查已通过，下一步需要你确认这套字段解释是否符合业务口径。

请确认以下 4 点：
- 人群分组：字段 `<treatment_column>` 中，`<treatment_value>` 表示处理组，`<control_value>` 表示对照组。
- 目标结果：字段 `<outcome_column>` 是 `<outcome_type>` 类型；如果是二分类，`<positive_value>` 表示正例，`<negative_value>` 表示负例。
- 样本支撑：共 `<row_count>` 行，处理组 `<treated_rows>` 行，对照组 `<control_rows>` 行；如果是二分类，同时给出处理组/对照组的正负例数量。
- 缺失和异常：关键字段缺失数、警告和假设。没有问题时写“未发现关键字段缺失或阻塞风险”。

如果以上口径正确，请回复：确认样本语义。
如果不正确，请直接指出要改的字段或取值。
```

不要使用下面这种只展示路径的回复：

```text
数据源：<long path>
语义计划草稿：<long path>
如果确认无误，请回复：确认样本语义
```

### `split_planning` 需要确认时的回复结构

`split_planning` 返回 `needs_confirmation` 后，必须解释切分方式，而不是只展示 `split_plan_path`：

```text
切分方案已生成，下一步需要你确认是否按这个方案产出建模样本。

请确认以下 3 点：
- 切分方式：随机切分或时间切分；如果是时间切分，说明时间字段和 OOT 窗口。
- 样本比例/行数：train/test/valid/oot 的比例和预计行数。
- 风险提示：未启用分层、空切分、时间字段解析等警告；没有问题时写“未发现阻塞风险”。

如果以上切分方案正确，请回复：确认切分方案。
如果不正确，请说明希望调整的切分方式、比例或 OOT 时间窗口。
```

### `post_split_validation` 成功时的回复结构

`post_split_validation` 成功后，默认只展示业务摘要，不展示 `modeling_sample_spec_path`、`modeling_sample_path` 或 `modeling_sample_paths`：

```text
样本准备已完成，切分后验证通过。

样本切分结果：
- 总样本：<row_count>
- 训练集：<train_rows>
- 测试集：<test_rows>
- 验证集：<valid_rows 或 未启用>
- OOT：<oot_rows 或 未启用>

未发现阻塞风险。下一步可以进入样本同质性检查。
```
