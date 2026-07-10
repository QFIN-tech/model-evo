---
name: classification-model-training
description: 端到端训练 xgboost/dnn/lr 二分类模型——读上游 feature-analysis 产出的 splits/{train,test,oot}.parquet，产八阶段产物(features/model/evaluation/predictions/explainability/comparison/logs/config)，并与历史 baseline 做 AUC/KS/分档多维对比。本 skill 不取数也不切分。当用户说"建模""训练新模型""跑模型""对比base""迭代模型"时使用。
---

# classification-model-training

训练为进程内实现(xgboost / dnn / lr),按 `model.algo` 切换;由 `scripts/run_build.py` 编排。
共享配置读写代码位于 `model-skills/_modelevo-shared/scripts/`(config_io),通过 `scripts/_bootstrap.py` 注入;源在仓库根 `_modelevo-shared/scripts/`,由 `install.sh` 复制。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| `splits/{train,test,oot}.parquet` | ✅ | 上游 `feature-analysis` 按 `model.split` 切分产出 | 本 skill 直接消费,不取数、不切分;缺失时回到 `classification-model-development` Stage 0 跑 `feature-analysis` 补齐,本 skill 不做切分兜底 |
| 特征清单(`features` / `feature_list_source`) | ✅ | 用户配置 或 上游 `feature-matching` 派生的 `feature-list.csv` | 候选特征清单强制过**边界安全过滤**(可配 `model.boundary_filter.enable_*=false` 关闭),剔除 4 类会让训练失败或泄漏的特征(常量/泄漏/ID 类/全缺失,规则表见第 3 节);过滤后剩余特征即入模特征集;剔除弱特征(低 IV/高 PSI)的优化筛选不在本 skill,走 `classification-model-tuning` 产 `-feat` 新 run;`features` 留空则走 `feature_list_source`(各业务域清单见 `model-knowledge/assets/feature-knowledge/feature-knowledge.md` 索引,如 `feature-list/feature-list-user-operation-v1.csv`) |
| `train_config.yaml` | ✅ | 复制 `config/train_config.example.yaml` 后填写 | 填模型名/标签列/特征清单;可选 `model.baseline_eval_dir` 配基线评估目录以走 N-way 对比;可选 `model.run_label` 当作本次 run 的版本号(如 v1/v2)。**输入 yaml 必须放 `<session_dir>/new-models/{algo}-v{N}/config/` 下**(即 model 内部 config 目录),**严禁落到 session 根目录或 `<skill_dir>/config/` 下**;`run_build.py` 会把 `--config` 指向的 yaml 视为输入源,`write_train_config_yaml` 在同目录原地写 `_manifest.json`(含 `source_yaml` 指向自身),不做副本拷贝 |
| `session_dir` | ✅ | 上游 `classification-model-development` / `classification-model-orchestration` 传入 | 本 skill 不负责 session 决议;无 session 上下文时请先调 `classification-model-development/scripts/list_sessions.py` 列历史 sessions |

## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录(即本文件所在目录),执行时替换为实际绝对路径,不要依赖当前工作目录。

**Session 决议**(由上游负责):本 skill 假定 `session_dir` 已由上游 `classification-model-development` / `classification-model-orchestration` 确定。若用户直接调本 skill 且无 session 上下文,请先跑:

```bash
python <model-skills>/classification-model-development/scripts/list_sessions.py
```

用 `AskUserQuestion` 询问选历史 session 还是新建,确认 `session_dir` 后再执行训练。

**输入 yaml 落盘(强制)**:输入 yaml 必须落 `<session_dir>/new-models/{algo}-v{N}/config/train_config.yaml`(model 内部 config 目录),不要放 `<skill_dir>/config/` 或 session 根目录。流程:

1. 复制 `<skill_dir>/config/train_config.example.yaml` 到 `<session_dir>/new-models/{algo}-v{N}/config/train_config.yaml`(`{algo}` 取 `xgb|dnn|lr`,`{N}` 由 `next_version` 自增;目录不存在则 `mkdir -p` 创建)
2. 编辑该 yaml 填真实值
3. 调 `run_build.py` 指向该 yaml:

```bash
python <skill_dir>/scripts/run_build.py \
--config <session_dir>/new-models/{algo}-v{N}/config/train_config.yaml \
--output_dir <session_dir> \
--version v1                   # 可选; 否则按 yaml.run_label → 自动自增
```

`--data_dir` 可选,默认从同 session 下 `<session_dir>/sample-features/` 读 `splits/{train,test,oot}.parquet`(由 `feature-analysis` 切分产出);若需用其他数据,显式传 `--data_dir` override(指向含 `splits/` 子目录的目录)。`--output_dir` 直接传 `<session_dir>`,`run_build` 会在其下落 `new-models/{algo}-v{N}/`(无 `classification-model-training/` 中间层)。test.parquet 当 val 段(early stopping);进程内用调优超参训练(`tune_train.TUNED_PARAMS`: depth 6 / lr 0.03 / n 800 + early-stop;比 `engines/_xgb/entry.py` 的强正则默认更高容量,避免欠拟合)。

训练完成后,模型报告路径需人工登记到 `classification-model-recommend` 台账(本 skill 不自动改 csv)。

## 3. 参数说明

### run_build.py

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--config` | ✅ | - | 训练配置 yaml 路径(load_config + validate 后使用,并复制到 run_dir/config/) |
| `--output_dir` | ✅ | - | session_dir 本身(编排器在其下落 `new-models/`) |
| `--data_dir` | 否 | 自动推断 | 含 `splits/{train,test,oot}.parquet` 的目录(通常 `<session_dir>/sample-features/`);留空从 `<output_dir>/sample-features/` 推断,推断失败则报错退出 |
| `--version` | 否 | `None` | 显式版本号(仅纯版本号: `v1` / `v2` / `custom-tag` / `20260710`;**不要带 algo/suffix 前缀**如 `xgb-v1`/`tuned-v1`/`feat`,会被拦截报错);与 `--label` 都传时 `--version` 优先 |
| `--label` | 否 | `None` | `--version` 的别名;留空则按优先级回退 yaml `model.run_label`,再空则自动自增 |

**version 决议优先级**(本次 run 的 version 用作目录后缀,形如 `v1` / `v2` / `custom-tag`):

| 优先级 | 来源 | 备注 |
|:---:|---|---|
| 1 | CLI `--version`(别名 `--label`) | 显式指定,跳过自增 |
| 2 | yaml `model.run_label` | 语义化短名(如 v1 / v2) |
| 3 | 自动自增 | 扫 `new-models/` 下同 algo 已有目录取 max+1,首次为 `v1` |

字符集限制 `[a-zA-Z0-9_.-]`,非白名单字符归一为 `_`;空值走自动自增。

**⚠️ version 保留字拦截**:目录命名规则为 `{algo}{suffix}-{version}`,若 version 含 algo(`xgb`/`dnn`/`lr`/`lgb`)或 suffix(`tuned`/`feat`)保留字 token,会叠加产生重复前缀目录(如 `xgb-xgb-v1` / `xgb-tuned-tuned-v1` / `xgb-feat`)。`validate_config` 与 CLI 入口(`run_build.py` / `run_tuning.py` / `select_features.py`)在 run_dir 创建前统一拦截,报错示例:

```
ValueError: model.run_label 非法: version 标识 'lgb-v1' 含算法/后缀保留字 ['lgb'], ...
```

写 yaml 或跑 CLI 时,`run_label` / `--version` / `--label` 只填纯版本号(如 `v1` / `v2` / `20260710` / `exp01`),不要重复 algo 或 suffix。

### model.boundary_filter(边界特征过滤)

**职责定位**:边界过滤是**安全过滤**(剔除会让训练失败或泄漏的特征),基于 IV/PSI 的优化筛选由 `classification-model-tuning` 负责。本层不参与特征优选,只剔除数据安全/正确性问题特征。

**4 条规则**(任一命中即剔除,并集生效):

| 规则 | 读哪个 csv | 判定条件 | 默认阈值 |
|------|-----------|---------|---------|
| `constant`(常量) | `stats.csv` | `unique <= const_unique_max` **或** `std == 0` | `const_unique_max=1` |
| `leakage`(泄漏/特征穿越) | `feature-quality.csv`(优先)→ `iv_table.csv`(回退) | `iv > iv_max`(IV NaN 不删) | `iv_max=1.0` |
| `id_like`(ID 类) | `stats.csv` + manifest `overview.n_total` | `unique / sample_total > id_like_ratio`(`sample_total <= 0` 时跳过) | `id_like_ratio=0.9` |
| `all_missing`(全缺失) | `stats.csv` | `missing_rate >= missing_max` | `missing_max=1.0` |

**与 tuning 阈值不重叠**:本 skill `missing_max=1.0` 只删真全缺失;tuning `missing_threshold=0.95` 删高缺失。

**配置字段**(`model.boundary_filter` 段):

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enable_constant` / `enable_leakage` / `enable_id_like` / `enable_all_missing` | bool | `true` | 单规则开关;设 `false` 关闭对应规则 |
| `iv_max` | float | `1.0` | IV > 此值视为泄漏;须 > 0 |
| `const_unique_max` | int | `1` | unique <= 此值视为常量;须 >= 0 |
| `id_like_ratio` | float | `0.9` | unique / sample_total > 此值视为 ID 类;须在 (0, 1] |
| `missing_max` | float | `1.0` | missing_rate >= 此值视为全缺失;须在 (0, 1] |

**warn-and-skip 行为**:上游 `feature-analysis` 缺 `stats.csv`/`feature-quality.csv`/`iv_table.csv` 时,对应规则打印 warning 并跳过(返回空剔除集),不阻断训练;`sample_total <= 0` 时 `id_like` 规则跳过。

**产物落盘**:被剔除特征落 `features/used-feature-list.csv` 的 `dropped_<rule>` 行(三列 csv: feature_name / status / dropped_by_rule);`features/report.md` 含"边界特征过滤"段按规则分组展示;`config.json.runtime.boundary_filter` 含 `n_before`/`n_after`/`n_dropped`/`dropped_by_rule`/`thresholds`/`rules_enabled`/`sample_total` 摘要。

## 4. 输出产物

单次 run 在 `<session_dir>/new-models/{algo}-v{N}/` 下落以下子目录(`--output_dir` 传的是 `<session_dir>` 本身,无 `classification-model-training/` 中间层):
- baseline run: `{algo}-v1` / `{algo}-v2` / ...(suffix 为空)
- 特征筛选 run(classification-model-tuning 产): `{algo}-feat-v1` / `{algo}-feat-v2` / ...
- 调参 run(classification-model-tuning 产): `{algo}-tuned-v1` / `{algo}-tuned-v2` / ...
- `{N}` 按 algo+suffix 维度自动自增(扫 `new-models/` 下已有目录取 max+1);`--version` 显式指定时跳过自增。

```text
{run_dir}/
├── config.json                    # 入参+训练快照(metrics/best_iteration/n_features)
├── config/                        # 入参 yaml(原地存,独立复现用)
│   ├── _manifest.json             # 含 source_yaml 指向自身 yaml 绝对路径
│   └── train_config.yaml          # 输入 yaml 原件(--config 指向此文件;未传则整段缺失)
├── features/
│   ├── _manifest.json
│   ├── used-feature-list.csv      # 单列 feature_name(供下游复现/部署)
│   └── report.md
├── model/
│   ├── _manifest.json             # 含 algo / used_params / best_iteration / has_scorecard
│   ├── model_meta.json            # 训练元信息(结构随 algo 不同, 供下游 tuning 复用)
│   ├── model.json  (xgb)          # 算法原生扩展名
│   ├── model.pkl   (dnn|lr)
│   └── scorecard.csv (lr)         # 评分卡: [feature, bin, woe, coef, score]
├── evaluation/
│   ├── _manifest.json
│   ├── {run_name}_train_eval.json     # classification-model-evaluation 产出
│   ├── {run_name}_train_eval.md
│   ├── {run_name}_train_eval.xlsx
│   ├── {run_name}_test_eval.{json,md,xlsx}
│   ├── {run_name}_oot_eval.{json,md,xlsx}
│   └── {run_name}_all_eval.{json,md,xlsx}   # train+test+oot 纵向拼接整体评估
├── predictions/
│   ├── _manifest.json
│   ├── train_predictions.parquet  # train 分档: id_cols + label + score + bucket
│   ├── test_predictions.parquet   # test  分档: 同上
│   ├── oot_predictions.parquet    # oot   分档: 同上
│   └── report.md                  # 三档分布直方图 + 数据安全提示
├── explainability/
│   ├── _manifest.json
│   ├── feature-importance.csv
│   └── shap-summary.csv           # 仅 xgb (TreeExplainer; 其他算法在 manifest 标 skipped)
├── comparison/                     # 仅当 model.baseline_eval_dir 配置时产出(延迟创建, 无产物则目录不存在)
│   ├── _manifest.json             # 含 baseline_eval_dir / splits / skipped
│   ├── comparison_train.{json,md,xlsx}  # 新模型 vs 基线 (train 档)
│   ├── comparison_test.{json,md,xlsx}   # 新模型 vs 基线 (test 档)
│   └── comparison_oot.{json,md,xlsx}    # 新模型 vs 基线 (oot 档)
├── logs/
│   ├── _manifest.json
│   ├── run.log                    # tee 捕获训练核心阶段 stdout/stderr
│   └── run_build.log              # process_tee 捕获 run_dir 创建→完成回执全过程
└── report.md                       # 单 run 顶层整合报告(各阶段摘要, 自动落)
```

### 4.0 session 级横向对比产物

每个 run 跑完自身 `comparison/` 后, `run_build.py` 末尾固定调 `invoke/session_aggregate.py:invoke_session_aggregate(output_dir)`, 在 `<session_dir>/model-comparison/` 下产跨 run 横向 N-way 对比(与单 run 的 `comparison/` 互补, 聚合 session 内所有 run 的 eval JSON):

```text
<session_dir>/model-comparison/
├── _manifest.json                       # 含 included_runs / splits / skipped
├── model-comparison_train.{json,md,xlsx}   # session 内所有 run 的 train 段 eval 横向对比
├── model-comparison_test.{json,md,xlsx}    # 同上, test 段
└── model-comparison_oot.{json,md,xlsx}     # 同上, oot 段
```

聚合失败不影响主流程(单 run `comparison/` 已落盘), 仅打 warning。`included_runs` 字段记录实际纳入对比的 run 列表(扫 `new-models/*/evaluation/` + `model-recommend/*/evaluation/` 命中的 eval JSON)。

### 4.1 产物内容

**评估依赖(强制)**:
- 本 skill 不自带评估报告逻辑,评估阶段统一委托 `classification-model-evaluation` skill。
- predictions 阶段写完三档 parquet 后,由 `scripts/invoke_evaluation.py` 一次性把三档 predictions 转 CSV 喂给 `classification-model-evaluation/scripts/eval_single.py`(目录模式),产出标准化三件套(JSON + MD + XLSX)到 `evaluation/` 子目录。
- 输出文件命名:`{run_name}_{split}_eval.{json,md,xlsx}`,split = train / test / oot / all(共 4 档);`all` 档为三档样本纵向拼接后的整体评估,由 eval_single 自动产。
- 透传字段:`model.algo` → `--model-type`,`model.label_expr` → `--target-def`,`model.owner` → `--owner`,`model.status` → `--status`,训练实际超参 → `--hyperparams`。
- base 对比 / 训练段 vs OOT 段 PSI / 三段汇总表 不在本 skill 产出范围内;如需 base 对比请走 `classification-model-comparison` skill。

**对比阶段(可选,评估完成后链式调用)**:
- `model.baseline_eval_dir` 配置时:评估完成后自动调 `classification-model-comparison` skill,对 train/test/oot 三档分别做 N-way 对比(新模型 eval JSON vs 基线 eval JSON)。
- `model.baseline_eval_dir` 未设置时:默认扫描 `<session_dir>/model-recommend/*/evaluation/`(兼容 `classification-model-recommend` skill 推荐的 yx_001/yx_002 等多 model_id 子目录),命中即对每个 split 做 N-way 对比;无命中则跳过,不影响主流程。
- `model.baseline_eval_dir: null` 显式为 null:关闭默认扫描,不产 `comparison/` 子目录。
- 基线 JSON 定位:在每个 baseline_eval_dir 下 glob `*_{split}_eval.json`(按 model_id 去重,避免同名重复计入);新模型 JSON:`{run_name}_{split}_eval.json`。
- 产出:`comparison/comparison_{split}.{json,md,xlsx}` × 3 档 + `_manifest.json`。

**评分卡产物(仅 algo=lr)**:
- LR 路径在训练完成后由 `engines/_lr/_model.py:ScoreCardConverter` 生成评分卡表,落盘为 `model/scorecard.csv`,并在 `model/_manifest.json` 标 `has_scorecard: true`。
- 列结构:`[feature, bin, woe, coef, score]`;每行 = 一个特征分箱的 WoE / 系数 / 分数贡献。
- 分数公式:`Score = base_score - factor * ln(odds)`,`factor = pdo / ln(2)`;默认 `base_score=600 / pdo=50 / base_odds=50`(基准 odds 50:1)。
- 非 lr 路径(xgb/dnn)不产 `scorecard.csv`,manifest 标 `has_scorecard: false`。

**model_meta.json**:
- 统一字段:`{algo, feature_names, feature_importance, train_info, params, created_at}`;由 `scripts/stages/model.py:write_model_stage` 落盘。
- xgb: 由引擎 `engines/_xgb/_core/_fit.py` 在 `save_model` 时落盘, 字段 `{feature_names, feature_importance, best_iteration, params}`。
- dnn: 由 `stages/model.py:_write_meta_for_non_xgb` 补写, `feature_importance={}`(MLP 无原生重要性), `train_info={best_epoch, best_val_auc, total_epochs, early_stopped}`。
- lr: 同函数补写, `feature_importance={feat: |coef|}`(取自 `LrPredictor.get_feature_importance()`), `train_info={n_iter, converged, train_auc, val_auc, n_features, regularization, C}`。

### 4.2 对比口径 / version 决议

**对比口径**:
- base 对比 / PSI / 分档走势 / 三段汇总报告由独立 skill 承担,本 skill 不自带。
- 单模型标准化评估:见 `evaluation/{run_name}_{split}_eval.md`(由 `classification-model-evaluation` 产出)。
- 多模型 / 多版本对比:调 `classification-model-comparison` skill。

version 决议规则见「3. 参数说明」末尾。

## 5. 与其他 skill 的关联

| skill / 模块 | 关系 | 用途 |
|---|---|---|
| `feature-analysis` | 上游 | 特征分析(IV/PSI/相关性),报告供人工决定 `features` 列表;按 `model.split` 切 train/test/oot 产 `splits/{train,test,oot}.parquet`(本 skill 直接消费) |
| `classification-model-evaluation` | **评估依赖** | predictions 后调 `eval_single.py` 产标准化三件套(JSON+MD+XLSX),本 skill 不自带评估 |
| `classification-model-comparison` | **对比依赖(可选)** | 配置 `model.baseline_eval_dir` 时调 `compare_models.py` 做 N-way 对比,产 `comparison/` |
| `classification-model-tuning` | 下游 | 读本 skill run 输出,做规则诊断 / Optuna 搜索后落 `-tuned` 新 run |
| `classification-model-recommend` | 下游 | 历史模型清单;训练完成后人工登记模型报告路径 |
| `model-evo/shared`(父目录) | 公共代码 | 跨 skill 共享的配置读写(config_io / fetch_spark 等) |

## 6. 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ 必显式传 features(不装 optbinning) | `model.features` / `feature_list_source` 至少一个;**不回退 optbinning auto_select**,列表为空且未配 `feature_list_source` 时报错终止 |

> 覆盖范围、"不覆盖"清单、何时用、session 决议、迭代报告、输入 yaml 落盘约束、异常处理全表详见 `references/constraints-and-exceptions.md`。

## 7. 异常处理

异常分类与处理方式详见 `references/constraints-and-exceptions.md` 第 9 节。

## 8. 测试

```bash
python -m pytest <skill_dir>/tests/ -q              # 全量(含真训练 slow 用例)
python -m pytest <skill_dir>/tests/ -q -m "not slow"  # 仅快测
```

---

数据来源:输入 `splits/{train,test,oot}.parquet` 由 `feature-analysis` skill 按 `model.split` 切分产出(其上游 sample.parquet 来自 `feature-matching`),本 skill 不取数也不切分;baseline eval JSON 来自 `classification-model-recommend` skill 的 `model-recommend/{model_id}/evaluation/` 目录;调优超参默认值见 `scripts/trainers/tune_train.py:TUNED_PARAMS`。
最后更新:2026-07-07
