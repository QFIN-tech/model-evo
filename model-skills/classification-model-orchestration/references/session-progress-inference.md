# 会话启动进度推断规则

> 本文件从 `classification-model-orchestration/SKILL.md` 3.1 节抽出,包含 8 阶段进度推断表（完成标志 / 缺失时推断）。SKILL.md 中保留 2 行摘要 + 指向本文件的指针。

扫描 `runs/` 下所有 `{timestamp}-{model_name}` 命名的任务文件夹，按时间戳倒序取最近 5 个，对每个文件夹读 `task-spec/_manifest.json` 推断进度。

**进度推断规则**：

| 阶段 | 完成标志 | 缺失时推断 |
|------|----------|-----------|
| task-spec | `task-spec/.done` 存在 | 待跑或半途中断 |
| 样本分析 | `data-profile/_manifest.json` 存在 | data-profile 待跑 |
| model-recommend | spark 模式：`model-recommend/_manifest.json`；local_file 模式：始终视为"已跳过" | spark 模式下待跑 |
| feature-matching | `sample-features/feature-matching/sample.parquet` + `feature-list.csv` 存在 | 待跑 |
| feature-analysis（Dev Stage 0） | `sample-features/feature-analysis/analysis/_manifest.json` 存在 | Stage 0 待跑 |
| Dev Stage 1 | `new-models/` 非空 | Stage 1 待跑 |
| Dev Stage 2 | `new-models/` 下多 run，但 `model-comparison/_manifest.json` 不存在 | Stage 2 loop 中 |
| Dev Stage 3 | `model-comparison/_manifest.json` 存在 | Stage 3 已完成，待收口 |

> 关联: `classification-model-orchestration/SKILL.md` 3.1 节会话启动检查
