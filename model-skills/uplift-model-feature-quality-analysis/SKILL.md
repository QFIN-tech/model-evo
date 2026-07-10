---
name: uplift-model-feature-quality-analysis
description: "已有 confirmed task_config_path 和 modeling_sample_spec_path 后，用于运行特征质量诊断、生成筛选建议、确认或复用 feature plan。不要用于模型训练、调参、模型比较或最终报告。"
---

# 特征质量分析

## 1. 输入依赖

本 skill 负责特征质量诊断和建模特征方案交接：

- Basic Quality：缺失率、唯一值数量和值集中度。
- Split PSI：训练集和测试、验证、OOT 样本之间的分布稳定性。
- Monthly PSI：按月观察特征分布稳定性，仅作为参考。
- Uplift Bivar：观察特征分组内 treatment/control outcome 差异，仅作为参考。
- Feature plan：把用户确认后的特征列表沉淀为 `feature_plan_path`，供建模 skill 消费。

已有 confirmed `task_config_path` 和 `modeling_sample_spec_path` 后使用。

## 2. 执行入口

统一 CLI：

```bash
python <skill-dir>/scripts/run.py --input - --output-dir <output_dir>
```

Windows 环境可按需将路径分隔符改为反斜杠。

`--output-dir` 必须是已存在的 `run_dir`；runner 不创建 run_dir，不扫描 `latest/current`。本 skill 输出到：

```text
{run_dir}/uplift-model-feature-quality-analysis/{timestamp}_feature_quality/
  inputs/0001_<action>.request.json
  results/0001_<action>.result.json
  artifacts/
  _flow_manifest.json
  _flow_log.jsonl
  report.md
```

所有 stdout、result、manifest 和 log 中的持久化 path 都是 `run_dir` 相对路径。需要复用同一 flow 的 action 必须显式传入 `outputs.flow_dir`，不得从 recommendation、draft 或 feature_plan artifact path 反推 flow。

请求 JSON 必须通过 stdin 传入：

```json
{"action": "diagnostics_only", "payload": {"task_config_path": "...", "modeling_sample_spec_path": "...", "analysis_scope": {"diagnostics": ["basic_quality", "split_psi"], "confirmed": true}}}
```

runner stdout 是 JSON object。调用方只读取 stdout 或 result JSON 中的 `outputs.*` 字段，例如 `outputs.report_path`、`outputs.feature_selection_recommendation_path`、`outputs.feature_plan_path`。

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

## 3. Actions 与参数说明

### 默认交互流程

诊断、筛选推荐、feature plan 确认是三个独立步骤。用户确认前一步不代表确认后一步。

当用户只说“继续”“开始特征质量分析”“先看一下特征”时，调用方必须先向用户展示候选诊断项和推荐诊断计划，不得直接生成筛选推荐或 feature plan。

诊断前面向用户展示的内容只包含候选项、用途、推荐诊断计划和本步产物边界，不写“建议运行”“参与筛选依据”“仅作参考”“默认不纳入”等判断性表述。推荐展示口径：

```text
我将先做特征质量诊断。本步只生成诊断报告，不生成筛选推荐，也不生成 feature plan。

候选诊断项：
1. basic_quality：检查缺失率、唯一值数量和值集中度。
2. split_psi：检查 train/test/valid/OOT 之间的特征分布稳定性。
3. monthly_psi：按月检查特征分布稳定性，需要可用时间字段。
4. uplift_bivar：观察特征分组内 treatment/control outcome 差异。

推荐诊断计划：
- basic_quality
- split_psi
- uplift_bivar

如果按这个计划执行，我会先跑诊断；诊断完成后再展示摘要，并询问是否按默认规则生成筛选推荐。
```

默认诊断计划固定为：

```text
diagnostics:
  - basic_quality
  - split_psi
  - uplift_bivar
```

`monthly_psi` 可由用户加入；加入时必须有明确 `date_column`。如果 `modeling_sample_spec` / `task_config` 中不能明确识别时间字段，则停止并询问用户提供字段名，不自动猜测。

用户确认诊断计划后，默认调用 `diagnostics_only`，并传入 `analysis_scope.confirmed=true`。缺少该确认时 runner 返回 `needs_confirmation`。诊断完成后，先展示诊断摘要，再询问用户是否按默认规则生成筛选推荐，并展示默认规则：

```text
默认规则：
- 缺失率 > 80%：进入剔除候选
- PSI > 0.25：进入剔除候选
- 众数占比 >= 99%：进入剔除候选
- 缺失率 > 50%：标记为预警
- PSI > 0.10：标记为预警
- 众数占比 >= 95%：标记为预警
```

用户确认筛选规则后，调用 `selection_only` 复用本轮诊断结果，并显式传入默认阈值和 `selection_scope.confirmed=true`：

```json
{
  "new_thresholds": {
    "missing_rate_warning": 0.5,
    "missing_rate_exclude": 0.8,
    "psi_warning": 0.1,
    "psi_exclude": 0.25,
    "near_constant_mode_ratio": 0.99,
    "high_concentration_mode_ratio": 0.95
  },
  "selection_scope": {
    "confirmed": true,
    "rule_source": "default"
  }
}
```

如果用户不接受默认规则，则让用户调整阈值；未提到的阈值沿用默认值。

筛选推荐生成后，必须展示推荐方案，并停止等待用户选择接受、修改或调整阈值重跑。展示粒度至少包含保留数量、剔除数量、剔除特征及原因。保留特征数量较少时完整展示；数量较多时可展示预览，但必须说明可以展开完整列表。在用户明确选择接受或修改前，不调用 `accept_recommendation`、`modify_recommendation` 或 `confirm_artifact`。

`diagnostics_only`

- 输入：`task_config_path`、`modeling_sample_spec_path`、`analysis_scope`。
- `analysis_scope.confirmed=true` 必填，表示调用方已经展示候选诊断项和推荐诊断计划并获得用户确认；缺少时返回 `needs_confirmation`。
- `analysis_scope.diagnostics` 支持 `basic_quality`、`split_psi`、`monthly_psi`、`uplift_bivar`。
- 只输出请求的诊断表和报告，不生成特征筛选建议。
- 定位：默认诊断入口。用于执行用户已确认的诊断计划。
- 默认不生成 `feature_selection_recommendation_path`，不生成或修改 feature plan。

`selection_only`

- 输入：`source_result_path`、`new_thresholds`、`selection_scope`。
- `selection_scope.confirmed=true` 必填，表示调用方已经展示诊断摘要和筛选规则并获得用户确认；缺少时返回 `needs_confirmation`。
- `selection_scope.rule_source` 必须是 `default` 或 `custom`。
- 复用已有 Basic Quality / Split PSI 明细，生成新的 `feature_selection_recommendation_path`。
- 定位：诊断完成后的筛选推荐入口。默认用于复用本轮 `diagnostics_only` 结果生成推荐。
- 调用前必须先向用户展示诊断摘要和筛选规则，并获得用户确认。
- 只生成筛选推荐，不生成 confirmed feature plan。

`accept_recommendation`

- 输入：`recommendation_path`，可选 `source_result_path`、`task_config_path`、`modeling_sample_spec_path`。
- 输出：已确认的 `feature_plan_path`。
- 定位：用户看过推荐方案并明确接受后使用。
- 调用前必须展示推荐保留/剔除方案和剔除原因。用户没有明确接受前不得调用。

`modify_recommendation`

- 输入：`recommendation_path`、`include_features`、`exclude_features`。
- 输出：草稿 `feature_plan_path`，需要再调用 `confirm_artifact`。
- 定位：用户看过推荐方案并指定额外保留或剔除特征后使用。
- 输出草稿后必须再次展示草稿方案，并等待用户确认。

`manual_feature_plan`

- 输入：`task_config_path`、`modeling_sample_spec_path`、`feature_list` 或 `feature_list_path`。
- 输出：草稿 `feature_plan_path`，需要再调用 `confirm_artifact`。
- 定位：用户明确提供人工特征列表时使用。
- 输出草稿后必须展示草稿方案，并等待用户确认。

`confirm_artifact`

- 输入：`source_draft_path`、`artifact_kind="feature_plan"`。
- 输出：已确认的 `feature_plan_path`。
- 定位：只用于确认已经展示给用户并由用户明确接受的 feature plan 草稿。
- 不得把用户对诊断计划、筛选规则或推荐生成的确认解释为 feature plan 确认。

`reuse_feature_set`

- 输入：`previous_feature_plan_path`、`task_config_path`、`modeling_sample_spec_path`。
- 输出：已确认的 `feature_plan_path`，并提示未对这组输入重跑诊断。
- 定位：用户明确要求复用既有 confirmed feature plan 时使用。
- 调用前必须提示未对当前输入重跑诊断。

`validate_feature_plan`

- 输入：`feature_plan_path`、`task_config_path`、`modeling_sample_spec_path`。
- 输出：校验通过的 `feature_plan_path`。
- 定位：校验已有 confirmed feature plan 与当前 task/sample 输入是否匹配。

## 4. 输出产物

典型输出目录：

```text
{run_dir}/uplift-model-feature-quality-analysis/{timestamp}_feature_quality/
  inputs/
  results/
  artifacts/
    feature_basic_quality.v1.csv
    feature_psi_split.v1.csv
    feature_selection_recommendation.v1.json
    feature_plan.confirmed.v1.json
  report.md
  _flow_manifest.json
  _flow_log.jsonl
```

`selection_only` 以及创建、修改、确认或复用 feature plan 的 action 会刷新 `report.md`。报告复用已有诊断产物，不重算指标，并显示当前最新的 feature plan 状态与特征列表。

关键下游路径：

- `feature_basic_quality_path`
- `feature_psi_split_path`
- `feature_selection_recommendation_path`
- `feature_plan_path`
- `report_path`

## 5. 与其他 skill 的关联

| 方向 | Skill | 关系 |
|---|---|---|
| 上游 | `uplift-model-sample-preparation` | 提供 confirmed `modeling_sample_spec_path` |
| 下游 | `uplift-model-s-learner-modeling` / `uplift-model-t-learner-modeling` | 消费 confirmed `feature_plan_path` 训练 baseline 模型 |

## 6. 执行约束

本 skill 不训练模型，不生成调参方案，不做模型比较，不生成最终报告，不用大模型计算确定性指标，不手写结果文件。

## 7. 异常处理

### 7.1 缺依赖处理

本 skill 需要 `pandas` 和 `numpy`。runner 不安装依赖。缺包时返回结构化失败，并在 issue/next_steps 中提示阅读本 skill 的缺依赖处理说明，不返回 skill 安装目录文件路径。

### 7.2 字段兼容异常

只接受显式 `*_path` / `*_paths` 字段。收到旧的 `*_ref` / `*_refs` 输入时，runner 返回 `needs_input`，并在 `issues` 与 `next_steps` 中给出对应替代字段。

## 8. 面向用户回复

普通用户回复默认使用中文。先给业务摘要，再给关键风险和下一步决策。不要把 raw JSON、内部目录结构、长路径清单或实现术语作为主回复内容。

除非用户明确要求查看 JSON、调试、获取产物路径或手动接力下游流程，否则不要在用户回复中展示工程中的 `.json` 文件路径；`feature_plan_path`、`recommendation_path` 等 JSON 路径只作为内部/下游输入传递。人类可读报告可以按需给出。

不要让用户看到大量字段名。确需精确交接时，可在“技术详情”里保留 `feature_plan_path`、`recommendation_path`、`report_path` 等字段名，并说明用途。
