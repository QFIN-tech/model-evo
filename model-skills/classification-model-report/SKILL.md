---
name: classification-model-report
description: 聚合 session 内 task-spec / data-profile / feature-analysis / new-models/*/evaluation 的关键信息,产 6-sheet Excel 报告(模型概览/样本分析/特征质量/三档评估/分桶排序性对比/特征清单)。定位与 classification-model-comparison 互补 — comparison 聚焦 AUC/KS 单指标 + 分桶并排;本 skill 聚合需求规格 + 样本分析 + 特征质量 + 多 run 全量评估 + 分桶并排 + AUC 最高 run 的特征重要性/SHAP。仅用户主动调起,不被 development / orchestration 调度。当用户说"出模型报告""汇总报告""生成报告 excel""model_report"时使用。
---

# model_report

把 session 内多个 run 与上游产物(task-spec / data-profile / feature-analysis / `new-models/*/`)聚合到一个 6-sheet Excel,落到 `<session_dir>/{session_name}_report.xlsx`。定位与 `classification-model-comparison` 互补:comparison 只产 AUC/KS 对比表 + 分桶并排(JSON + md + xlsx);本 skill 补位**全链路汇总**(需求规格 → 样本分析 → 特征质量 → 多 run 三档评估 → 多 run 分桶并排 → AUC 最高 run 特征清单)。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| `<session_dir>` | ✅ | 上游 `classification-model-orchestration` 产 | session 根目录,如 `runs/20260701-110624-draw_willingness/`;必须存在 |
| `task-spec/_manifest.json` | ✅ | `classification-model-task-spec` 产 | 需求规格 + 路由溯源 + 样本概况 + 切分配置;Sheet 1 数据源 |
| `data-profile/_manifest.json` | 否 | `classification-model-task-spec` 产 | 样本分析详情(分时段 / 稳定性 / 充足性 / 三档切分);Sheet 2 数据源 |
| `sample-features/feature-analysis/analysis/{_manifest.json, iv_table.csv, psi_table.csv, stats.csv}` | 否 | `feature-analysis`(orchestration Step 4C)产 | 特征质量(IV / PSI / 缺失率 / 基础统计);Sheet 3 数据源 |
| `new-models/*/config.json` | ✅ | `classification-model-training` / `classification-model-tuning` 产 | 每个 run 的入参 + runtime(metrics / n_features / best_iteration);Sheet 4/6 数据源 |
| `new-models/*/model/_manifest.json` | 否 | `classification-model-training` 产 | `used_params`(标量 dict);Sheet 6 训练参数列 |
| `new-models/*/evaluation/{run_name}_{train,test,oot}_eval.json` | ✅ | `classification-model-evaluation` 产 | 每个 run 三档 eval JSON;Sheet 4 全量指标 + Sheet 5 分桶 |
| `new-models/*/explainability/{feature-importance,shap-summary}.csv` | 否 | `classification-model-training` Stage 5 产 | Sheet 6(AUC 最高 run)的特征重要性 / SHAP Top 30 |
| `new-models/*/features/used-feature-list.csv` | 否 | `classification-model-training` 产 | Sheet 6 入模特征清单统计 |

**扫描范围**: 仅扫 `<session_dir>/new-models/*/`,**不扫 `model-recommend/`**(历史模型通常无 eval JSON / explainability,纳入会全是占位行,无信息量)。

**容错**: 每个输入独立加载,缺失时对应 sheet 写"上游产物缺失"占位,不阻断整体流程。仅 `<session_dir>` 或 `new-models/` 不存在时才立即退出。

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录(即本文件所在目录),执行时替换为实际绝对路径。

```bash
python <skill_dir>/scripts/build_report.py \
    --session-dir <session_dir> \
    [-o <output.xlsx>]
```

**示例**:

```bash
# 默认: 输出落 <session_dir>/{session_name}_report.xlsx
python <skill_dir>/scripts/build_report.py \
    --session-dir /path/to/runs/20260701-110624-draw_willingness

# 显式指定输出路径
python <skill_dir>/scripts/build_report.py \
    --session-dir /path/to/runs/20260701-110624-draw_willingness \
    -o /tmp/session_report.xlsx
```

**Python 环境**: 需要 openpyxl。当前默认 `python` 无 openpyxl,可用 `/data/oceanus_ctr_wkdir/pyenv/anaconda_data_ai/bin/python` 跑(Python 3.9 + openpyxl 3.0.10)。

## 3. 参数说明

### build_report.py

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--session-dir` | ✅ | - | session 根目录,如 `runs/20260701-110624-draw_willingness/`;必须存在 |
| `-o` / `--output` | 否 | `<session_dir>/{session_name}_report.xlsx` | 输出 xlsx 路径;父目录不存在则自动创建 |

**session_name 推断**: 取 `session_dir.name`(即目录名,如 `20260701-110624-draw_willingness`),作为默认输出文件名前缀。

## 4. 输出产物

单个 xlsx 文件,默认落在 `<session_dir>/{session_name}_report.xlsx`:

```text
<session_dir>/
└── {session_name}_report.xlsx     # 6-sheet session 级全链路汇总报告
```

### 4.1 Sheet 列表

| # | Sheet 名 | 内容 | 数据来源 |
|---|---------|------|---------|
| 1 | 1-模型概览 | KV 格式 8 个段: 基础信息 / 路由溯源 / WHO / WHAT / HOW GOOD / CONSTRAINTS / HOW TO USE / 数据源 | `<session_dir>/task-spec/_manifest.json` |
| 2 | 2-样本分析 | KV 格式 4 段(总体样本 / 稳定性 / 样本充足性 / 切分元信息)+ 2 张附表(分时段样本 / Train-Eval-OOT 切分);**不画 DataBar**(纯数值) | `<session_dir>/data-profile/_manifest.json` + `sample-features/feature-matching/_split_manifest.json` |
| 3 | 3-特征质量 | **全量特征单表**(17 列): # / feature / dtype / IV / 单变量 AUC / PSI / PSI 预警 / 有效分箱 / 缺失率 / unique / mean / std / min / q25 / median / q75 / max;合并 iv_table + psi_table + stats 三张 csv,特征 union 后**按 IV 降序排序**(缺失 IV 排末尾);**无 KV 概况段**(样本分析已在 Sheet 2) | `<session_dir>/sample-features/feature-analysis/analysis/{_manifest.json, iv_table.csv, psi_table.csv, stats.csv}` |
| 4 | 4-三档评估 | 按 split 分块(**train / test / oot / all 四档**纵向堆叠),每子表列: run_name / 样本量 / 正样本率 / AUC / KS / 准确率 / 精确率 / 召回率 / F1;多 run 横向对比;**不画 DataBar** | `new-models/*/evaluation/{run_name}_{train,test,oot,all}_eval.json` 的 `metric_by_segment['全量']`(all = train+test+oot 合并评估) |
| 5 | 5-分桶排序性对比 | **严格参考 `classification-model-comparison` 对比报告.xlsx Sheet 2 格式**: 顶部指标计算逻辑 + 基线版本声明; **仅 oot + all 两档**按 split 分块, 每子表两层表头(metric group header + 模型名子表头, 基线加 "(基线)" 后缀), 列按 metric 分组(label率×n / 召回率×n / 累计召回×n), 10 decile 数据行 + 主表 DataBar(label率绿/召回率蓝/累计召回蓝); 主表下方接 Lift 子表(基线列 "-", 其他列比值, >1 绿/<1 红, 不画 DataBar) | `new-models/*/evaluation/{run_name}_{oot,all}_eval.json` 的 `performance.score_buckets['全量']` |
| 6 | 6-特征清单 | KV(基本信息 / 三档 AUC-KS / 训练参数 / 入模特征清单统计)+ **1 张合并全量表**(feature / importance / mean_abs_shap / mean_shap, 按 importance 降序, SHAP 列缺失写 "—");**用 AUC 最高的 run** | `_find_best_run` 选定 run 的 `config.json` + `model/_manifest.json` + `explainability/{feature-importance,shap-summary}.csv` |

### 4.2 样式约定

**通用样式**:
- 标题行(第 1 行): 微软雅黑 14pt bold #1A3060,合并 A1 起的所有列
- source_note 行(第 2 行): 微软雅黑 9pt italic 灰字,合并所有列,内容为**实际文件绝对路径**(非占位符)
- definitions 行(第 3 行起,可选): 微软雅黑 9pt 灰字,合并所有列,说明本 sheet 布局 + 指标定义
- 表头: 微软雅黑 11pt bold 白字,深蓝底色(#1A3060),居中,thin border
- 数据行: 微软雅黑 10pt,thin border,数字右对齐/文本左对齐
- `ws.sheet_view.showGridLines = False`
- 列宽: 按 header 与数据内容长度自适应(min 10 / max 50)
- freeze_panes: 标题行下方(A2 或表头行下方)

**KV 段头样式**(Sheet 1 / 2 / 6 的 KV 部分):
- 段头行: 灰底 #D9D9D9 + 蓝字 bold #1A3060,合并所有列,前缀 `▌`
- 字段列(A 列): 微软雅黑 10pt bold 深蓝字
- 值列(B 列): 微软雅黑 10pt,thin border

**Split 标签行样式**(Sheet 4 / 5):
- 灰底 #D9D9D9 + 蓝字 bold #1A3060,合并所有列,前缀 `▼`,格式如 `▼ train 档 (共 N 个 run)`

**Sheet 1 特定样式**:
- 8 个 KV 段(基础信息 / 路由溯源 / WHO / WHAT / HOW GOOD / CONSTRAINTS / HOW TO USE / 数据源),无 DataBar(纯文本概览);样本概况 + 切分配置已下沉到 Sheet 2

**Sheet 2 特定样式**:
- 4 个 KV 段 + 2 张附表
- 附表"分时段样本": 列 pday / 样本量 / 正样本 / 正样本率 / 正负比
- 附表"Train-Eval-OOT 切分": 列 split / 样本量 / 正样本 / 正样本率 / pday 范围
- KV 与附表之间仅 1 行空行间隔(`_write_kv_sheet` 返回 `next_row`, 调用方 `cur_row = next_row + 1` 定位, 不依赖 `ws.max_row` — 后者在 KV 只填 A:B 两列时会被 openpyxl 虚报)
- **不画 DataBar**(纯数值, 与其他 sheet 区分)

**Sheet 3 特定样式**:
- **全量特征单表**(17 列): `# / feature / dtype / IV / 单变量 AUC / PSI / PSI 预警 / 有效分箱 / 缺失率 / unique / mean / std / min / q25 / median / q75 / max`
- **无 KV 概况段**(样本分析已在 Sheet 2, 避免重复);title + source_note + 表头 + 数据行直连
- 特征 union: stats.csv 为主序(含 dtype/缺失率/quantiles 最完整), iv_table / psi_table 补全 iv/auc/psi/有效分箱, 缺字段写 "—"; union 后**按 IV 降序排序**(缺失 IV 的特征排末尾)
- 标题: "3. 特征质量 (全量特征 — IV / 单变量 AUC / PSI / 缺失率 / 基础统计)"
- DataBar:
  - IV 列(列 4): 蓝色(#1A3060), 全表归一化
  - PSI 列(列 6): 红色(#C00000), 全表归一化
  - 缺失率 列(列 9): 红色(#C00000), 全表归一化
- freeze_panes = "C2"(冻住 # + feature 两列)
- 列宽: A=6 / B=60(feature 名) / C=10 / D-F=12 / G-H=10 / I=12 / J=10 / K-Q=14-16

**Sheet 4 特定样式**:
- **4 个 split 子表纵向堆叠**(train / test / oot / all), 每个子表前插入 split 标签行
- 9 列: run_name / 样本量 / 正样本率 / AUC / KS / 准确率 / 精确率 / 召回率 / F1
- `all` = train+test+oot 合并(全集)评估, 指标口径与单 split 一致
- **不画 DataBar**(用户要求: 仅看指标数值, 不用条形可视化; 与 AUC/KS 单指标对比交给 Sheet 5 + comparison skill)

**Sheet 5 特定样式**(严格参考 `classification-model-comparison` 的对比报告.xlsx Sheet 2 格式):
- 顶部 2 行声明(灰底小字 9pt):
  - R3 指标计算逻辑: `分桶=各模型按自己打分降序十分位(Decile10=最高分) | label率=桶内正样本占比 | 召回率=桶正样本/总正样本 | 累计召回=从高到低累计捕获正样本占比`
  - R4 基线版本声明: `基线版本={runs_with_buckets[0].run_name} | Lift=各版本指标 / 基线同指标 (>1 表示优于基线)`
- **仅 oot + all 两档**, 2 个 split 子表纵向堆叠, 每个子表前插入 `▼ {sp} 档 (共 N 个 run)` 标签行
- **列分组按 metric(非按 run)**: `分桶`(1) + `人数`(1) + `label率 × n_runs` + `召回率 × n_runs` + `累计召回 × n_runs`
- **每个 split 子表两层表头**:
  - 第 1 层 metric group header(灰底 #D9D9D9 + 蓝字 bold #1A3060, 合并 n_runs 列): `label率(正样本率)` / `召回率` / `累计召回`
  - 第 2 层模型名子表头(灰底 + 灰字 bold #333333): 每 group 下重复 n_runs 个 run_name, 基线 run 加 `(基线)` 后缀
- 10 个 decile 数据行(decile 10 → 1, 高分 → 低分), `分桶/人数` 取首个有 buckets 的 run 的值
- **DataBar(主表, 各列独立归一化)**:
  - label率 列: 绿色(#2E7D32)
  - 召回率 列: 蓝色(#1A3060)
  - 累计召回 列: 蓝色(#1A3060)
- **Lift 子表**(仅当 n_runs > 1 且基线 split 有 buckets 时, 在主表下方):
  - 副标题: `各版本 vs {baseline} Lift（label率 / 召回率 / 累计召回 相除）`
  - 同主表两层表头结构, group 改为 `label率 Lift` / `召回率 Lift` / `累计召回 Lift`
  - 基线列 = `-`(灰字), 其他列 = 比值(>1 绿 #006100 / <1 红 #9C0006 / =1 灰 #333333)
  - **不画 DataBar**(Lift 仅看数值)

**Sheet 6 特定样式**:
- 4 个 KV 段(基本信息 / 三档 AUC-KS / 训练参数 / 入模特征清单统计)
- 1 张**合并全量表**(非 2 张独立表): `# / feature / importance / mean_abs_shap / mean_shap`, 按 importance 降序
- 特征 join: fi_sorted 为主序, SHAP 按 feature 名 map 到对应行; SHAP-only(不在 fi 中)的特征追加在末尾
- importance 缺失写 "—", mean_abs_shap / mean_shap 缺失写 "—"
- KV 与全量表之间仅 1 行空行间隔(同 Sheet 2, 用 `_write_kv_sheet(return_next_row=True)` 定位, 不用 `ws.max_row`)
- DataBar: 仅 importance 列(蓝 #1A3060), 全表归一化
- 段头标题: `▌ 特征重要性 + SHAP (全量, 按 importance 降序; SHAP 列缺失写 —)`
- run 选择逻辑: `_find_best_run` 按 oot AUC 降序(oot 缺失则按 train AUC 兜底)选 AUC 最高的 run

### 4.3 容错行为

| 异常 | 处理方式 |
|------|---------|
| `--session-dir` 不存在 | 立即退出,提示路径不存在 |
| `<session_dir>/new-models/` 不存在 | 立即退出,提示无有效 run |
| `new-models/` 下无有效 run(全缺 config 或 eval) | 立即退出,提示无有效 run |
| `task-spec/_manifest.json` 缺失 | Sheet 1 写"task-spec/_manifest.json 缺失"占位,不阻断 |
| `data-profile/_manifest.json` 缺失 | Sheet 2 写"data-profile/_manifest.json 缺失"占位,不阻断 |
| `feature-analysis/analysis/_manifest.json` 缺失 | Sheet 3 写"feature-analysis 尚未执行"占位,不阻断 |
| 某个 run 的 `config.json` 缺失 | 该 run 记入 skipped,不纳入 Sheet 4/5/6 |
| 某个 run 的 train/test/oot/all eval JSON 全缺 | 该 run 记入 skipped,不纳入 Sheet 4/5/6 |
| 某个 run 的某档 eval JSON 缺失 | Sheet 4 对应 split 子表该 run 行写 "—",Sheet 5 该 run 列写 "—",不阻断 |
| 某个 run 的 `explainability/*.csv` 缺失 | Sheet 6 若选中该 run 则 Top 表写"文件缺失"占位,不阻断 |
| 全部 run 失效 | 立即退出,提示无有效 run |
| skipped 清单 | Sheet 4/5/6 末尾追加 warning 行(微软雅黑 9pt 红字) |

## 5. 与其他 skill 的关联

| skill / 模块 | 关系 | 用途 |
|---|---|---|
| `classification-model-task-spec` | 上游 | 产 `task-spec/_manifest.json` + `data-profile/_manifest.json`(Sheet 1/2 数据源) |
| `feature-analysis`(orchestration Step 4C) | 上游 | 产 `sample-features/feature-analysis/analysis/*.csv`(Sheet 3 数据源) |
| `classification-model-training` | 上游 | 产 `new-models/*/config.json` + `model/_manifest.json` + `evaluation/*_{split}_eval.json` + `explainability/*.csv`(Sheet 4/5/6 数据源) |
| `classification-model-evaluation` | 上游 | 产 `new-models/*/evaluation/{run_name}_{split}_eval.json`(Sheet 4/5 数据源) |
| `classification-model-tuning` | 上游 | 产 `-feat` / `-tuned` 新 run,同结构被本 skill 扫描 |
| `classification-model-comparison` | **互补** | comparison 产 JSON/md/xlsx,聚焦 AUC/KS 单指标对比表 + 分桶并排(label率/lift/召回率/累计召回);本 skill 产单一 xlsx,聚合全链路(需求 → 样本 → 特征质量 → 三档评估 → 分桶并排 → 特征清单)。两者产物不重叠,本 skill Sheet 5 的分桶并排与 comparison 的分桶并排格式不同(本 skill 按 split 分块纵向堆叠,comparison 按 split 分文件横向并排) |
| `classification-model-development` | **不调度本 skill** | 本 skill 仅用户主动调起 |
| `classification-model-orchestration` | **不调度本 skill** | 本 skill 仅用户主动调起 |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| 覆盖范围 | 读 session 内 task-spec / data-profile / feature-analysis / new-models 全量前序产物,产 6-sheet xlsx 落到 `<session_dir>/` |
| 不覆盖:调度 | 不被 development / orchestration 调度,仅用户主动调起 |
| 不覆盖:run 内部产物生成 | evaluation / comparison / feature-analysis 等均由各自上游 skill 产,本 skill 只读不写 |
| 不覆盖:AUC/KS 单指标对比表 | 由 `classification-model-comparison` skill 覆盖;本 skill Sheet 4 含 AUC/KS 但定位是"多 run 全量评估指标对比",非纯 AUC/KS 对比表 |
| 数据安全红线 | 只读前序产物,不修改任何上游文件;唯一写操作是 `<session_dir>/{session_name}_report.xlsx` |
| 文件来源真实 | 所有 sheet 的 source_note 必须写实际文件绝对路径(`f"{session_dir}/..."`),不允许写占位符 |
| 扫描范围 | 仅扫 `<session_dir>/new-models/*/`,不扫 `model-recommend/`(历史模型通常无 eval JSON) |
| AUC 最高 run 选择 | Sheet 6 用 `_find_best_run` 动态选 oot AUC 最高的 run(oot 缺失则 train AUC 兜底),不硬编码 |
| 依赖 | openpyxl(Python 3.x);当前默认 `python` 无 openpyxl,可用 `/data/oceanus_ctr_wkdir/pyenv/anaconda_data_ai/bin/python` 跑 |
| 何时用 | 当用户对一个已完成多个 run 的 session 想要一份聚合 Excel 报告(含需求 / 样本 / 特征质量 / 三档评估 / 分桶并排 / 特征清单)时使用 |

## 7. 异常处理

| 异常 | 处理方式 |
|---|---|
| `--session-dir` 不存在 | 立即退出,提示路径不存在 |
| `<session_dir>/new-models/` 不存在 | 立即退出,提示无有效 run |
| `new-models/` 下无有效 run(全缺 config 或 eval) | 立即退出,提示无有效 run |
| `task-spec/_manifest.json` 缺失 | Sheet 1 写占位,不阻断 |
| `data-profile/_manifest.json` 缺失 | Sheet 2 写占位,不阻断 |
| `feature-analysis/analysis/_manifest.json` 缺失 | Sheet 3 写占位,不阻断 |
| 某个 run 的 `config.json` 缺失 | 该 run 记入 skipped,sheet 4/5/6 末尾追加 warning 行 |
| 某个 run 的 train/test/oot/all eval JSON 全缺 | 该 run 记入 skipped,sheet 4/5/6 末尾追加 warning 行 |
| 某个 run 的某档 eval JSON 缺失 | sheet 4 对应 split 子表该 run 行写 "—";sheet 5 该 run 列写 "—";不阻断 |
| 某个 run 的 `explainability/*.csv` 缺失 | Sheet 6 若选中该 run 则 Top 表写"文件缺失"占位,不阻断 |
| openpyxl 未安装 | 立即报错 `ModuleNotFoundError: No module named 'openpyxl'`,提示用 `/data/oceanus_ctr_wkdir/pyenv/anaconda_data_ai/bin/python` |
