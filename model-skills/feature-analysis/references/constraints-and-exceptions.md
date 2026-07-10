# feature-analysis 执行约束与异常处理

> 本文件从 `feature-analysis/SKILL.md` 第 6/7 节抽出,包含覆盖范围、取数/切分边界、职责边界、数据安全红线(项目级通用约定复述)、变更前置流程、特征清单交互约定详情、异常处理全表。SKILL.md 第 6 节只保留红线级约束(特征清单三档必填 / 变更前置),第 7 节保留单行指针指向本文件。

## 1. 覆盖范围与适用范围

用于建模前查看候选特征的分布/缺失/IV/PSI,判断哪些特征不稳定(PSI 高)、哪些预测力弱(IV/AUC 低);不用于建模/训练(走 `classification-model-training`),也不用于历史模型查询(走 `classification-model-recommend`)。

## 2. 取数边界与切分边界

- **取数边界**:`sample.parquet` 由上游 `feature-matching` skill 产出,或用户通过 `--data_path` 指定任意路径;本 skill 不负责取数,也不读上游三档 parquet
- **切分边界**:切分由本 skill 按 `model.split` 内部完成,落 `<session_dir>/sample-features/splits/`,与 `feature-matching/sample.parquet` 同级

## 3. 职责边界

仅产报告供人工决定特征筛选,**不自动剔除特征**、不写 `selected_features.txt`、不修改任何上游配置。

## 4. 数据安全红线(全模式强制)

禁止在配置/where 中硬编码用户 ID/手机号/身份证号。

## 5. 变更前置流程(强制遵循 CLAUDE.md)

修改分析代码前,先输出「变更计划」(一、修改内容 二、预期影响 三、回滚方案)并等确认。

## 6. 特征清单交互约定详情

### 6.1 要素

| 要素 | 写入参数 | 说明 |
|---|---|---|
| 特征清单文件 | `--feature_list_source` / yaml `model.feature_list_source` | .txt 按行 / .csv 取 `feature_name` 列 |
| 特征列表 | yaml `model.features` | 直接在配置里写列表 |
| 交叉校验基准 | `--cross_validate_csv` | feature-matching 产出的 `feature-list.csv`(自动推断) |

### 6.2 来源优先级(CLI → yaml → 报错)

1. `--feature_list_source` 命令行参数:指向 .txt/.csv 文件
2. yaml `model.feature_list_source`:指向特征清单文件(相对路径按 model-skills 根解析)
3. yaml `model.features`:显式特征列表
4. **三者都空 → 报错,不默认全量分析**

### 6.3 获取来源(优先级:task-spec 文档 > feature-knowledge 索引 > 交互询问)

1. **优先从 `classification-model-task-spec` 的规格文档读取**特征筛选要求
2. **feature-knowledge 索引**:`model-knowledge/assets/feature-knowledge/feature-knowledge.md` 登记各业务域的特征表与清单 csv(`feature-list/` 目录);按建模用的特征表(优先)或业务域(兜底)匹配到对应清单
3. **交互询问**:以上都缺失时,用 `AskUserQuestion` 问用户:
   - "用 feature-knowledge 索引识别到的 <清单 csv> 作为特征清单?" → 默认选项(回显识别到的路径)
   - "指定其他特征清单文件"
   - "手动圈定特征列表"

### 6.4 交叉校验(强制)

用户指定的特征清单必须与 feature-matching 产出的 `feature-list.csv` 做交叉校验:

- 每个用户指定的特征名必须在 `feature-list.csv` 中存在
- 不在数据中的特征:打印 warn 并**自动排除**,不阻断分析
- 全部特征都不在数据中 → **报错终止**
- `--cross_validate_csv` 不传时自动从 `--data_path`(sample.parquet)同目录推断 `feature-list.csv`

## 7. 异常处理

| 条件 | 处理方式 |
|---|---|
| `--data_path` 未传 / 文件不存在 | 报错终止(`ValueError` / `FileNotFoundError`) |
| 特征清单三来源(`--feature_list_source` / yaml `feature_list_source` / yaml `features`)均为空 | 报错终止,不默认全量分析 |
| 用户指定特征与 `feature-list.csv` 交叉校验后全部不在数据中 | 报错终止 |
| 用户指定特征部分不在数据中 | 打印 warn 并自动排除,不阻断分析 |
| `cross_validate_csv` 指向路径不存在 | 报错终止(`FileNotFoundError`) |
| `feature-list.csv` 缺 `feature_name` 列 | 报错终止(`ValueError`) |
| `model.split` 未配置 | 报错终止,提示需配 `train_range`/`test_range`/`oot_range` |
| `label_col` 不在样本列中 / 全部为 NaN / 样本量为 0 | 报错终止(`ValueError`) |
| `label_col` 取值非 0/1(含 NaN 之外的其他值) | 报错终止,提示仅支持二分类 0/1 标签 |
| `label_col` 正样本数或负样本数为 0 | 报错终止,无法算 IV/AUC |
| `label_col` 含 NaN(非全部) | 打印 warning 继续执行,IV/AUC 计算时过滤 NaN 行 |
| `label_col` 正样本率 < 0.1% 或 > 99.9% | 打印 warning 继续执行,提示 IV/AUC 可能不稳定 |
| `analysis.psi.warn_threshold` 不在 `[0,1]` 内 | 报错终止(`ValueError`) |
| 缺 `openpyxl` | 跳过 `report.xlsx` 落盘,不阻断其余产物 |
| `train_df` 或 `oot_df` 为空 | PSI 表置空(`psi_table.csv` 仅含空列),不阻断其余分析项 |

---

> 关联:SKILL.md 第 6/7 节。
