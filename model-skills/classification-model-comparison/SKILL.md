---
name: classification-model-comparison
description: 多模型标准化评估横向对比——输入多个模型的标准化评估 JSON，对 oot 和 all 两档分别做 N-way 全维度对比（AUC、KS、分桶标签率/lift/召回率/累计召回），输出含条件格式的 Excel 报告与 delta 分析。由 classification-model-development 统一编排调用。当用户说"模型对比""对比评估""哪个模型更好""新旧对比""多版本对比""横向对比"时使用。
---

# 模型对比

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| oot eval JSON（≥2 份） | ✅ | `classification-model-evaluation` 产出的 `*_oot_eval.json` | 同数据、同客群、同指标口径 |
| all eval JSON（≥2 份） | ✅ | `classification-model-evaluation` 产出的 `*_all_eval.json`（`--input-dir` 模式自动产出）| 同数据、同客群、同指标口径 |

**前置约束**：对比必须在相同数据集、相同客群、相同指标口径上进行；不对比在不同测试集上评估的模型。

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录（即本文件所在目录），执行时替换为实际绝对路径，不要依赖当前工作目录。

**单次 N-way 对比**（手动指定 JSON 列表）：

```bash
# oot 单独对比
python <skill_dir>/scripts/compare_models.py \
  --jsons V1_oot_eval.json V2_oot_eval.json V3_oot_eval.json \
  -o /tmp/oot --fmt all

# all（train+test+oot 合并）对比
python <skill_dir>/scripts/compare_models.py \
  --jsons V1_all_eval.json V2_all_eval.json V3_all_eval.json \
  -o /tmp/all --fmt all
```

**会话级聚合对比**（自动扫描 session 内所有 run，对 oot 和 all 两档各产一份）：

```bash
python <skill_dir>/scripts/aggregate_session_comparison.py \
  --session-dir <session_dir> \
  --produced-by skills/model-comparison
```

脚本扫描 `new-models/*/evaluation/` 和 `model-recommend/*/evaluation/` 下的 eval JSON，对 oot 和 all 两档分别调 `compare_models.py` 对比，最终合成一份 combined 输出（中间产物落临时目录自动清理）。

## 3. 参数说明

### compare_models.py

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--jsons` | ✅ | - | 模型 `*_eval.json` 列表，`nargs="+"`，可传多个（脚本本身不校验数量下限，传 1 个也会跑出退化的对比结果，需上层调用方保证 ≥2 个） |
| `-o` / `--output` | 否 | `对比报告` | 输出文件前缀（含路径），父目录不存在会自动创建 |
| `--fmt` | 否 | `all` | 输出格式，取值 `json` / `md` / `xlsx` / `all` |

### aggregate_session_comparison.py

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--session-dir` | ✅ | - | session 根目录，需含 `new-models/` 和/或 `model-recommend/` 子目录 |
| `--produced-by` | 否 | `skills/model-comparison` | 写入 `_manifest.json` 的来源标识 |

## 4. 输出产物

### compare_models.py

```text
<output_prefix>.json
<output_prefix>.md
<output_prefix>.xlsx
```

### aggregate_session_comparison.py

中间产物流入临时目录，最终产出 **10 个文件**：6 个 per-split 文件 + 3 个 combined 文件 + 1 个 manifest。

```text
<session_dir>/model-comparison/
├── model-comparison_oot.json    ← oot 单 split 对比结果
├── model-comparison_oot.md
├── model-comparison_oot.xlsx
├── model-comparison_all.json    ← all 单 split 对比结果
├── model-comparison_all.md
├── model-comparison_all.xlsx
├── 对比报告.json    ← {"oot": {...comparison...}, "all": {...comparison...}}
├── 对比报告.md      ← oot MD + 空行 + all MD 拼接
├── 对比报告.xlsx    ← 3 Sheet，oot/all 同 Sheet 上下堆叠
└── _manifest.json
```

> `_SPLITS = ("oot", "all")` —— 只产出 oot 和 all 两档，不产出 train/test 档。

### 4.1 产物内容

| 产物 | 说明 |
|---|---|
| `model-comparison_{oot,all}.json` | 单 split 结构化对比结果，`fill_report.py --section VII` 的输入源。内含 `comparison_meta`（对比模型列表/数量）、`auc_comparison` / `ks_comparison`（按客群分组，行=客群，列=模型名）、`buckets_by_model`（各模型全量十分桶明细） |
| `model-comparison_{oot,all}.md` | 单 split 人类可读报告：标题 + 对比模型列表 + AUC 对比表（按客群） + KS 对比表（按客群） |
| `model-comparison_{oot,all}.xlsx` | 单 split 三 Sheet：`1-指标对比`、`2-分桶并排对比`、`3-raw_data`。Sheet 2 含 DataBar 条件格式 |
| `对比报告.json` | 结构化对比结果，顶层为 `{"oot": {...}, "all": {...}}`，每个 split 内容同 `model-comparison_{split}.json` |
| `对比报告.md` | 人类可读报告：oot 和 all 两段拼接，每段含标题 + 对比模型列表 + AUC 对比表（按客群） + KS 对比表（按客群） |
| `对比报告.xlsx` | 三 Sheet：`1-指标对比`、`2-分桶并排对比`、`3-raw_data`。每个 Sheet 内 oot 和 all 上下堆叠，以 split 标签行分隔。**条件格式**：详见 [references/databar-formatting-spec.md](references/databar-formatting-spec.md)。 |
| `_manifest.json` | 记录 `stage`、`produced_by`、`created_at`、`compare_engine`、`session_dir`、`included_runs`（命中的 run 目录名列表）、`splits`（实际产出的档位）、`skipped`（因样本不足跳过的档位及原因） |

## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `classification-model-evaluation` | 提供本 skill 的唯一合法输入 `*_eval.json`（含 `*_oot_eval.json` 和 `*_all_eval.json`），是本 skill 的强制前置 |
| 编排方 | `classification-model-development` | 统一编排调用本 skill，本 skill **不**被 `classification-model-evaluation` 直接调用 |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| 前置强制 | 不跳过 `classification-model-evaluation` 直接对比原始打分文件 |
| 对比口径 | 不对比在不同测试集 / 不同客群 / 不同指标口径上评估的模型 |
| 结论产出 | 不只凭 AUC 宣布胜者——必须结合分桶指标（label率/lift/召回率/累计召回）综合判断，DataBar 辅助观察单调性 |
| 噪声阈值 | delta < 0.005 不认定为有意义差异（N < 5000 时是噪声） |
| 产出覆盖 | 每次运行直接覆盖上次结果，不保留历史版本 |

## 7. 异常处理

| 条件 | 处理方式 |
|---|---|
| 某档 eval JSON 不足 2 个 | `aggregate_session_comparison.py` 跳过该档，不产出对应三件套，原因记入 `_manifest.json` 的 `skipped` 字段 |
| all eval JSON 不存在 | 回退调用 `merge_eval_splits.py` 从 train/test/oot 合并生成；正常流程由 `eval_single.py --input-dir` 直接产出，不经过此路径 |
| `compare_models.py` 找不到 | `aggregate_session_comparison.py` 抛 `FileNotFoundError` 并终止 |
| `session_dir` 不存在 | `aggregate_session_comparison.py` 抛 `FileNotFoundError` |
| `new-models/` 与 `model-recommend/` 均不存在 | 停止执行，提示检查 `--session-dir` |
| `compare_models.py` 子进程非 0 退出 | `aggregate_session_comparison.py` 打印其 stdout/stderr 后抛 `RuntimeError` |
| 预期产物文件未生成 | `aggregate_session_comparison.py` 抛 `FileNotFoundError` 并附 `compare_models` 的 stdout |
