---
name: uplift-model-task-spec
description: 由 model-task-routing 判定 task_type=uplift 后，或用户已明确指定这是 uplift/增益建模任务且需要定义任务配置时使用；负责收集并确认结构化 task_config。写入 confirmed artifact 前，必须由用户明确提供字段名和值。不要用于通用建模路由、字段发现、样本校验、特征分析、模型训练、评估、调参、比较或报告生成。
---

# 任务定义

使用本 skill 为一次增益（uplift）建模运行定义并确认 `task_config`。本 self-contained 版本只负责工作流入口：判断任务是否适用、收集字段映射、调用确定性运行器生成草稿，并在用户明确确认后确认 artifact。

## 1. 输入依赖

适用：已由 `model-task-routing` 判定为 uplift，或用户明确表达“uplift / 增益 / 干预相对不干预的增量效果 / 有 treatment-control 实验数据”等 uplift 语义，并需要生成或澄清 `task_config`。

不支持：用户明确要求非 uplift 的分类、聚类、报告生成或归因分析；用户要做字段发现、样本校验、特征分析、训练、评估、调参或模型比较。

### 从 model-task-routing 接力

当本 skill 由 `model-task-routing` 拉起时，调用方可以把 routing JSON 作为 `routing_input` 传入：

```json
{
  "task_type": "uplift",
  "routing_basis": {
    "q1_target": "用户原话描述的预测目标",
    "q2_intervention": "yes | no | uncertain",
    "q3_experiment_data": "yes | no"
  },
  "user_raw_request": "用户最初的建模诉求原话",
  "routed_at": "路由时间戳 YYYY-MM-DD HH:MM:SS"
}
```

承接规则：

- 启动时如果提供 `routing_input`，必须校验 `routing_input.task_type == "uplift"`；否则不要生成 uplift `task_config`。
- `routing_basis` 中非 null 信息视为已知条件，不重复询问 routing 阶段的 Q1/Q2/Q3。
- `user_raw_request` 和 `routing_basis` 可用于补充 `business_context`，并必须在 task_config artifact 的 `routing` 字段中保留溯源。
- routing 判到 uplift 只表示进入本 skill 做入口复核；不代表样本已满足 uplift 建模条件。

本 skill 只做轻量入口复核：

```text
1. 业务目标是否确认为“干预相对不干预的增量效果”，而不是“预测被干预人群响应概率”？
2. treatment/control 的字段和值是否能由用户明确给出？
```

如果第 1 点不成立，不写 confirmed uplift `task_config`，建议转 classification。
如果第 2 点不成立，暂停在 `task-spec`，要求用户补充字段和值。
样本量是否充足、treatment/control 是否支撑切分、两组是否同质、特征质量是否可用，交给后续 `uplift-model-sample-preparation`、`uplift-model-sample-homogeneity-check` 和 `uplift-model-feature-quality-analysis` 判断。

### 对话规则

对于不熟悉 uplift 的用户，先简要解释：

- 干预是业务动作、触达或暴露。
- 对照是可比较的未干预状态。
- 结果变量是要衡量的业务结果。
- 增益是干预相对对照对结果变量产生的增量效果。

随后要求用户提供实际 CSV 路径、列名和值，以及本次建模运行目录。不要从长字段列表中推断列，不要为了字段发现而扫描或展示完整表头。

### 必需输入

只有在以下值都已获得后，才调用运行器：

```text
data_ref
output_dir
outcome_column
outcome_type: binary | continuous
treatment_column
treatment_value
control_value
```

始终询问以下可选上下文字段，但允许为 `null`：

```text
unit_id_column
time_column
business_context
```

## 2. 执行入口

使用确定性运行器：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

`--output-dir` 必须是已存在的项目运行目录 `run_dir`，例如 `runs/{YYYYMMDD-HHMMSS}-{project_slug}`。runner 不负责选择工作区、不创建 `runs_root`，也不自动创建或恢复历史 `run_dir`。

请求 JSON 必须通过 stdin 传入。中文 `payload` 走 stdin 时，调用方必须明确使用 UTF-8；不要依赖 Windows、PowerShell 或 Python 默认文本编码。

stdout 和落盘 result JSON 都必须包含 `status`、`summary`、`outputs`、`issues`、`next_steps`。当前 action 的 result 文件路径必须读取 `outputs.result_path`；flow folder 必须读取 `outputs.flow_dir`。不要读取或传播顶层 `flow_dir`、`result_path`、`warnings`。outputs 中的产物路径作为下游接力 contract，按 `run_dir` 相对路径记录和传递。

支持的 actions：

```text
draft_task_config
confirm_artifact
```

一次业务 skill flow 会写入一个 timestamped flow folder。`draft_task_config` 创建该 folder，后续 `confirm_artifact` 复用同一个 folder：

```text
{run_dir}/uplift-model-task-spec/{timestamp}_task_spec/
  inputs/
    0001_draft_task_config.request.json
    0002_confirm_artifact.request.json
  results/
    0001_draft_task_config.result.json
    0002_confirm_artifact.result.json
  artifacts/
    task_config.draft.v1.json
    task_config.confirmed.v1.json
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

## 3. Actions 与参数说明

### draft_task_config

- 输入：必需 `data_ref`、`output_dir`、`outcome_column`、`outcome_type`、`treatment_column`、`treatment_value`、`control_value`。
- 可选输入：`unit_id_column`、`time_column`、`business_context`、`routing_input`。
- 输出：草稿 `task_config_path`、`flow_dir`、`result_path` 和 `confirmation_summary`。

`draft_task_config` 只读取 CSV 表头并检查指定列是否存在，不读取数据值，不计算统计量。

`draft_task_config` 可接收可选 `routing_input`。runner 只校验其 `task_type` 并把原始 routing 信息写入 artifact 顶层 `routing` 字段；不会把 `routing_input` 混入 `payload` 业务字段。

### confirm_artifact

确认调用示例：

```json
{
  "action": "confirm_artifact",
  "flow_dir": "<draft_result.outputs.flow_dir>",
  "source_draft_path": "<draft_result.outputs.task_config_path>",
  "artifact_kind": "task_config",
  "confirmed_by": "user"
}
```

`confirm_artifact` 只确认已有草稿，不接受 `payload` 修改。如果用户要求“确认并修改”，先重新调用 `draft_task_config`，展示新的确认摘要，然后再次请求确认。

### Flow 规则

一次业务上的 task definition flow 从 `draft_task_config` 开始，到 `confirm_artifact` 结束。

- 调用 `draft_task_config` 时，runner 会创建新的 `flow_dir`。
- `draft_task_config` 返回后，必须保留 `result.outputs.flow_dir` 和 `result.outputs.task_config_path`。
- 后续 `confirm_artifact` 必须复用同一个 `flow_dir`，并使用同一个 flow 下的 draft artifact。
- 不要为 `confirm_artifact` 创建新的 flow。
- 不接受 `source_draft_ref` 或其他旧 `_ref` 输入；收到旧字段时，runner 会返回 `needs_input`，并在 `issues` / `next_steps` 中给出对应 `_path` 字段建议。
- 不要使用 `latest`、`current`、`active` 或目录扫描来猜测 flow。
- 如果找不到上一轮返回的 `flow_dir` 或 `task_config_path`，应向用户说明需要重新运行 `draft_task_config`，或请用户提供草稿路径。

## 4. 输出产物

典型输出目录：

```text
{run_dir}/uplift-model-task-spec/{timestamp}_task_spec/
  inputs/
    0001_draft_task_config.request.json
    0002_confirm_artifact.request.json
  results/
    0001_draft_task_config.result.json
    0002_confirm_artifact.result.json
  artifacts/
    task_config.draft.v1.json
    task_config.confirmed.v1.json
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

关键下游路径：

- `outputs.result_path`
- `outputs.flow_dir`
- `outputs.task_config_path`

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `model-task-routing` | 可提供 `routing_input`，触发本 skill 做 uplift 入口复核 |
| 下游 | `uplift-model-sample-preparation` | 消费 confirmed `task_config_path`，继续样本语义检查和切分 |

## 6. 执行约束

本 skill 只做轻量入口复核和 `task_config` 草稿/确认，不做字段发现、样本校验、特征分析、训练、评估、调参或模型比较。

当用户只是提出通用“建模 / 建模型 / 新模型 / 开始建模”诉求，且尚未明确 `task_type=uplift` 时，不要直接使用本 skill；必须先交给 `model-task-routing` 判定 classification/uplift。

禁止事项：

```text
从长表头中猜测列。
读取数据值。
校验干预/对照样本是否充足。
推断二分类结果变量的正负标签。
判断 treatment/control 是否均衡或可比。
训练或评估模型。
在没有用户明确确认的情况下写入 confirmed artifact。
在 `confirm_artifact` 阶段修改 `payload`。
```

只有当用户确认具体草稿路径时，才执行确认：

```json
{
  "action": "confirm_artifact",
  "flow_dir": "uplift-model-task-spec/<timestamp>_task_spec",
  "source_draft_path": "uplift-model-task-spec/<timestamp>_task_spec/artifacts/task_config.draft.v1.json",
  "artifact_kind": "task_config",
  "confirmed_by": "user"
}
```

## 7. 异常处理

### 7.1 routing 输入不匹配

启动时如果提供 `routing_input`，必须校验 `routing_input.task_type == "uplift"`；否则不要生成 uplift `task_config`。

如果业务目标不是“干预相对不干预的增量效果”，不写 confirmed uplift `task_config`，建议转 classification。如果 treatment/control 的字段和值不能由用户明确给出，暂停在 `task-spec`，要求用户补充字段和值。

### 7.2 Flow 或字段兼容异常

不接受 `source_draft_ref` 或其他旧 `_ref` 输入；收到旧字段时，runner 会返回 `needs_input`，并在 `issues` / `next_steps` 中给出对应 `_path` 字段建议。

不要使用 `latest`、`current`、`active` 或目录扫描来猜测 flow。如果找不到上一轮返回的 `flow_dir` 或 `task_config_path`，应向用户说明需要重新运行 `draft_task_config`，或请用户提供草稿路径。

## 8. 面向用户回复

- 默认使用中文回复，除非用户明确要求英文。
- 先给业务摘要、确认事项、风险和下一步；不要把工程实现细节放在主回复里。
- 避免大段英文和不必要工程术语。必须提到字段名、action 名、命令或文件路径时，用反引号保留原文。
- 不向普通用户展示 raw stdout、raw result JSON、内部 flow 目录结构或长路径清单，除非用户明确要求调试或审计。不要把路径放进独立代码块作为主要回复内容。
- 除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径，包括草稿或确认后的 `task_config` 路径。JSON 产物和 `outputs.*_path` 仍然是机器接口与下游 skill 的契约，可以在内部读取和传递。
- 确认成功后，默认给出业务摘要、关键字段口径、风险和下一步；下游样本准备需要的 confirmed `task_config` 路径由 agent 内部传递，不作为普通用户主回复内容。

### 确认

执行 `draft_task_config` 后，向用户展示 `result.outputs.confirmation_summary`。不要自行改写这些事实。

只有当用户确认具体草稿路径时，才执行确认：

```json
{
  "action": "confirm_artifact",
  "flow_dir": "uplift-model-task-spec/<timestamp>_task_spec",
  "source_draft_path": "uplift-model-task-spec/<timestamp>_task_spec/artifacts/task_config.draft.v1.json",
  "artifact_kind": "task_config",
  "confirmed_by": "user"
}
```

确认后，回复中给出业务摘要和下一步建议，默认不要展示 confirmed `task_config` JSON 路径。需要继续 `uplift-model-sample-preparation` 时，内部带上明确的 `task_config_path`；只有用户要求手动接力、查看产物路径、调试或审计时，才在“技术详情”里展示该路径。
