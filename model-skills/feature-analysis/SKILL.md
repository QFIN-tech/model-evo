---
name: feature-analysis
description: 建模前对样本宽表里的候选特征做独立分析,产出特征报告(基础统计/单变量预测力 IV+AUC/训练-OOT 稳定性 PSI)。仅产报告供人工决定特征筛选,不自动剔除特征。当用户要"做特征分析/特征IV/特征PSI/建模前看一下特征"时使用。
---

# feature-analysis

建模前的独立特征分析:输入样本宽表 + 特征清单,输出 markdown 报告供人工判断特征质量。
**不自动筛选**,只给数据,筛与不筛由人决定。

## 1. 输入依赖

| 输入 | 必选 | 来源 | 说明 |
|---|:---:|---|---|
| `sample.parquet` | ✅ | 上游 `feature-matching` 产出 / 用户 `--data_path` 指定任意路径 | 样本宽表 |
| `feature_config.yaml` | ✅ | 复制 `feature-analysis/config/feature_config.example.yaml` 到 session 内填写 | 含 `model.{label_col, features/feature_list_source, split, dt_col}` + `analysis.*` |
| 特征清单 | ✅ | `--feature_list_source` / yaml `model.feature_list_source` / yaml `model.features` | 见「6. 执行约束」的特征清单交互约定 |
| `feature-list.csv` | 否 | `feature-matching` 产出,`--cross_validate_csv` 不传时自动从 `--data_path` 同目录推断 | 交叉校验基准 |


## 2. 执行命令

`<skill_dir>` 指本 skill 所在目录(即本文件所在目录),执行时替换为实际绝对路径,不要依赖当前工作目录。

```bash
python <skill_dir>/scripts/run_analysis.py \
    --config <session_dir>/sample-features/feature-analysis/feature_config.yaml \
    --data_path <session_dir>/sample-features/feature-matching/sample.parquet \
    --output_dir <session_dir>/sample-features/feature-analysis/analysis \
    [--feature_list_source model-knowledge/assets/feature-knowledge/feature-list/feature-list-user-operation-v1.csv] \
    [--cross_validate_csv <session_dir>/sample-features/feature-matching/feature-list.csv]
```

配置文件落 session 内(从 `feature-analysis/config/feature_config.example.yaml` 复制),不落 skill 自身 `config/` 目录(模板才放那里),保持 session 自包含,多 session 不互相覆盖。


## 3. 参数说明

| 参数 | 必选 | 默认值 | 说明 |
|---|:---:|---|---|
| `--config` | ✅ | - | `feature_config.yaml` 路径 |
| `--data_path` | ✅ | - | `sample.parquet` 路径(`.parquet` 走 `read_parquet`,否则走 `read_csv`) |
| `--output_dir` | ✅ | - | 报告输出目录 |
| `--feature_list_source` | 否 | `None` | 特征清单文件(.txt 按行 / .csv 取 `feature_name` 列),覆盖 yaml `model.feature_list_source` |
| `--cross_validate_csv` | 否 | `None`(自动从 `--data_path` 同目录推断 `feature-list.csv`) | 用于交叉校验的数据特征清单 |

yaml 内关键字段:

| 字段 | 必填 | 说明 |
|---|:---:|---|
| `model.label_col` | ✅ | 标签列名,仅支持二分类 0/1 |
| `model.features` / `model.feature_list_source` | 见「特征清单交互约定」 | 特征来源(与 CLI `--feature_list_source` 三选一,优先级见约定) |
| `model.split` | ✅ | `train_range` / `test_range` / `oot_range` 三档 pday 区间(8 位 YYYYMMDD,起 ≤ 止,三档时序递增);未配置直接报错 |
| `model.dt_col` | 否 | 默认 `pday`,用于 `model.split` 区间切分 |
| `analysis.iv.n_bins` | 否 | 默认 `10`,IV 等频分箱数 |
| `analysis.psi.n_bins` | 否 | 默认 `10`,PSI 分箱数 |
| `analysis.psi.warn_threshold` | 否 | 默认 `0.10`,须在 `[0,1]` 内,否则报错 |


## 4. 输出产物

主交付为 `report.md`(markdown),同目录另落机器可读产物;切分产物落 `<session_dir>/sample-features/splits/`(即 `output_dir.parent.parent / "splits"`,与 `feature-matching/sample.parquet` 同级)。

```text
<output_dir>/
├── _manifest.json          # schema_version / produced_by / files / overview
├── report.md               # 主交付; 人工阅读
├── report.xlsx             # 多 sheet: overview / feature_profile / feature_quality / woe
├── feature-profile.csv     # 基础统计语义化合并表(stats 全列)
├── feature-quality.csv     # IV + PSI merge 的单变量质量一站表 (feature/iv/auc/n_bins_effective/psi/psi_warn)
├── stats.csv               # 细分: 基础统计 (classification-model-tuning 高缺失率剔除规则消费)
├── iv_table.csv            # 细分: IV  (classification-model-tuning 低 IV 剔除规则消费)
├── woe_table.csv           # 细分: WOE 分桶明细 (feature/bin/cnt/pos/neg/pos_rate/woe/iv_bin)
└── psi_table.csv           # 细分: PSI (classification-model-tuning 高 PSI 剔除规则消费)

<session_dir>/sample-features/splits/
├── train.parquet
├── test.parquet
└── oot.parquet
```

### 4.1 产物内容

| 维度 | 指标/文件 | 说明 |
|---|---|---|
| 基础统计 | `feature-profile.csv` / `stats.csv` | 缺失率/unique/mean/分位数(q25/median/q75)/min/max/dtype,全样本上算 |
| 单变量预测力 | `iv_table.csv` / `woe_table.csv` | IV / 单变量 AUC / 有效分箱数 / WOE 分桶明细;等频分箱,缺失独立分桶;WOE 明细落 `woe_table.csv`,报告内仅展开 IV Top 20。**AUC 口径**:把特征做 WoE 编码后再算 `roc_auc_score(y, woe(x))`,等价于"用 bin 的正样本率排序";支持分类列与缺失列,但 WoE 用了样本内标签信息,数值轻微偏乐观,不是 raw-feature ROC-AUC |
| 稳定性 | `psi_table.csv` | 训练段 vs OOT 段 PSI(默认阈值 0.10,超阈值标 `[PSI_WARN]`) |
| 主报告 | `report.md` | 一、概述 二、基础统计表 三、IV+单变量 AUC 排序 三-bis WOE 分桶明细(IV Top 20) 四、PSI(含 WARN 标记) |
| Excel 全景 | `report.xlsx` | overview/feature_profile/feature_quality/woe sheets,业务/产品离线看 |
| 单变量质量合并 | `feature-quality.csv` | `feature, iv, auc, n_bins_effective, psi, psi_warn`,业务总览 |
| 清单 | `_manifest.json` | 给下游做 schema/版本校验用(`schema_version=1`, `produced_by=skills/feature-analysis`) |

补充说明:

- 合并 csv 与细分 csv 是**冗余但同源**(同次计算);下游可任选一种读法,变更后同步落


## 5. 与其他 skill 的关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `feature-matching` | 产出 `sample.parquet` + `feature-list.csv` |
| 下游 | `classification-model-training` | 产 `splits/{train,test,oot}.parquet` 给 training 直接消费(training 不切分);特征分析报告供人工筛 `features` 列表 |
| 下游 | `classification-model-tuning` | `select_features.py` 直接消费本 skill 的 `stats`/`iv`/`psi` 三份 csv 做自动筛选 |
| 依赖 | `model-evo/shared`(父目录) | 公共配置读写(`config_io`) |


## 6. 执行约束

| 约束 | 说明 |
|---|---|
| ⚠️ 特征清单三档必填 | `--feature_list_source` / yaml `feature_list_source` / yaml `features` 三者都空 → 报错终止,**不默认全量分析**(特征表含 ID/join-key/标签泄漏列,不可不筛) |

> 覆盖范围、取数/切分边界、职责边界、特征清单交互约定详情(来源优先级/获取来源/交叉校验)、异常处理全表详见 `references/constraints-and-exceptions.md`。

## 7. 异常处理

异常分类与处理方式详见 `references/constraints-and-exceptions.md` 第 7 节。


## 8. 测试

```bash
python -m pytest feature-analysis/tests/ -q
```
