# Task 与 Data 协议

本页定义 TaskSpec、loader、split、数据视图和 Profile 产物。完整模板始终以
`modelevo.py init` 或 `config-template` 的输出为准，不从文档手抄 spec。

## 目录

- [TaskSpec](#taskspec)
- [Data loader](#data-loader)
- [Split 与四平面](#split-与四平面)
- [数据视图](#数据视图)
- [特征探索抽样面](#特征探索抽样面)
- [Profile 产物](#profile-产物)
- [Adhoc 分析](#adhoc-分析)
## TaskSpec

最低必备语义：

| 区域 | 必填内容 |
|---|---|
| task | `task_id`、名称、objective、model_type、business_domains |
| target | 列名、正类/数值定义、观察窗、表现窗 |
| identity | 时间列与实体列 |
| data | `data_loader` 路径 |
| metric | primary metric 与可选 secondary metrics |
| split | 类型及 train/validation/shadow/final 窗口或比例 |
| leakage | `features.banned` 与 `features.reparse` 的 reason/evidence/plan |
| budget | trials、generations、wall/compute time、单 Trial timeout |

模型拓扑不在 spec 里授权(`ensemble_enabled` 不生效):trial
默认单模型,级联/融合由用户经 `approve-topology` 批准,见
[template-api.md](template-api.md#模型拓扑)。启动期若已知需要(例如
业务方要求子分级联),在启动问题清单里一次问清,由用户执行
`approve-topology --scope run --kind cascade`。

校验规则：task_id 使用小写/数字/连字符；binary 必须有正类定义；time_based
必须有 train/validation/final_test；uplift 必须有 treatment；指标必须与任务
类型匹配。

任务类型差异：

- binary：score 为正类概率，指标使用 auc/ks/logloss 等；
- regression：建议填写 `value_definition` 与 `unit`，指标使用
  rmse/mae/mape/r2；
- uplift：必填 `treatment.column` 与定义，指标使用 uplift_auc/qini/auuc。

具体口径按 [task-type-conventions.md](task-type-conventions.md) 分发到对应文档。

### 泄漏分类

`features.banned` 表示任何形态都不可进入数据产物；`features.reparse` 表示原始
形态不可用，但允许按明确 plan 解析成新列；`features.leak_confirmation` 记录
**用户对 ambiguous 列的裁决**。旧式 flat `exclude` 统一按 banned
解释，不能与三者混用。

```yaml
features:
  banned:
    - column: req_cnt_t15
      reason: future_window          # 受控枚举，见 BANNED_REASONS
      evidence: 实测包含链 t7 ⊆ t15，违反数 0/300000
  reparse:
    - column: seq_d365
      reason: form_unusable
      plan: 仅解析 k>=1 的分段，派生近端窗口计数与漏斗完成率
  leak_confirmation:                 # 只关闭确认门/撤销封禁，不改变数据面
    kept_ambiguous:
      - offline_inner_maidian_multidim_loan_loan_enter_days_2m
    reinstated_banned:               # 复核后撤销的机械封禁
      - column: seq_d365
        reason: 唯一率0.9976系复合文本字段属性，n_unique 299274 > 实体数 285894
    note: 历史回看口径，窗口不越过决策时点，保留为普通特征
```

Profile 首次调用审计全部列：明显未来窗、label 等价、实体标识列可进入
banned-certain；高基数/复杂 object 可进入 reparse-candidate；与目标同族或
单变量信号过强但语义不明的列进入 ambiguous。若产物显示
`needs_user_confirmation=true`，停止执行、由用户逐列裁决并写回 spec，再重跑同一
`profile`；首次材料化结果不得供 baseline 使用。hard-ban 改为 reparse 等于
协议变化，应新建 run。

裁决有四条合法去向，按语义选择：**泄漏** → `features.banned`；**形态不可用**
→ `features.reparse`（须补 `plan`）；**复核后确认合法、保留为普通特征** →
`features.leak_confirmation.kept_ambiguous`；**机械封禁判错了** →
`features.leak_confirmation.reinstated_banned`。第三、四档是真实场景的常见
裁决（例如"近 N 月申请次数"这类历史回看列看着像代理变量，复核后确认窗口
不越过决策时点；又如逐人唯一的复合文本字段被机械规则判成实体标识）。四档
互斥（同一列只能选一种），改动都会进 `leak_manifest_hash`。

### 机械提案复核清单

`leak-classification.md` 会把 **banned-certain 与 reparse-candidate 单独
列成「机械提案复核清单」**——它们由正则/统计规则产生，**不是人工裁决**。
清单逐列给出 `n_unique` vs **实体数**、长度 CV、分隔符密度、token 词表复用率，
配合两条判据复核：

- **真 ID 的唯一值数不可能超过实体数**——`n_unique > n_entities` 直接否决
  `entity_identifier`；
- 唯一率≈1 的**变长复合文本字段**（事件序列/JSON/编码串：长度 CV 大、
  分隔符密集、token 被语料复用上万次）应判 reparse，而非实体标识。

命中全局经验库「该列/同族列曾有正增益」的机械提案会自动降档为 ambiguous，
强制人工裁决（报告 `## 经验库对账` 小节）。

`reinstated_banned` 的约束：只能撤销**已在 `features.banned` 中**的列，
不得与 `reparse`/`kept_ambiguous` 同时声明（那是改判，该走各自通道），且
**撤销后该列仍保持物理隔离**，直到写下真正的裁决——"撤销"永远不会悄悄
放行一个未重新裁决的列。

## Data loader

`run-{task_id}/data-loader.py` 必须暴露：

```python
def load_data() -> pandas.DataFrame
```

约束：

- 返回完整建模样本，不在 loader 内切分、采样或做 Trial 特征工程；
- 每行含目标、时间、实体以及 uplift treatment（如适用）；
- 可读取 `MODELEVO_TASK_SPEC` 与 `MODELEVO_RUN_DIR`；
- time_based 的时间列必须是 datetime64 或 ISO 字符串；`YYYYMMDD` 先归一化
  为 `YYYY-MM-DD`，否则 split 显式失败。

**loader 可以直接 `import runtime`**：`load_dataframe` 执行 loader 前
把 skill 根注入 `sys.path`，所以分片样本源不必手写串行的
`for f in sorted(glob(...)): 读 → 逐片处理 → pd.concat`：

```python
from runtime.loaders import read_parquet_shards, shrink_object_columns

df = read_parquet_shards(PATHS, threads=4,
                         post=lambda f: shrink_object_columns(
                             f, keep_as_is=("user_no", "pday"),
                             skip_prefixes=("seq_",)))
```

保留原语义（行序=分片名排序、列序=首片 schema、空分片跳过）；`threads` /
`cast_decimals` / `skip_empty` 可覆盖；单文件读同样适用（`read_parquet_shards([path])`）。

**派生列必须声明**：loader 若在原始源之上**派生**列——最常见的是
标签变换（如 `loan_amt_t7_log1p = log1p(loan_amt_t7)`）——在 spec 里声明：

```yaml
data:
  data_loader: data-loader.py
  derived_columns: [loan_amt_t7_log1p]
```

不声明时数据底座指纹会随标签口径变化，同数据上其它任务的经验检索不到。
派生列的原始来源列（如 `loan_amt_t7`）照常进 `features.banned`。

Trial 禁止直接调用 loader；它只消费 Profile 材料化后的授权数据面。

## Split 与四平面

### Split 语义

- time_based 支持 `YYYY-MM`、`YYYY-MM-DD` 和 `A~B`；区间是**闭区间**，
  月端点含整月，日端点含当天，单端点按相同粒度前缀匹配。
- 窗口字段只能写区间，说明文字放 YAML 注释；解析失败会被 validate/profile
  拒绝，不能悄悄降级。
- random/group_based 默认比例 0.6/0.15/0.1/0.15，并固定 seed。
- 窗口外行按协议归入 train 并告警；实体重叠、重复样本和标签漂移由
  leakage audit 复核。

### 四平面材料化

每个 split 的信息被物理分开：

| 路径 | 内容 | 谁可消费 |
|---|---|---|
| `splits/{split}.parquet` | 纯特征；无 target/key/banned/reparse | Trial |
| `splits/labels/{split}.parquet` | target、treatment、对齐键 | 模板访问器 |
| `splits/reparse/{split}.parquet` | 授权解析源与对齐键 | `reparse_frame` |
| `splits/identity/{split}.parquet` | 时间坐标和不可逆样本键 | 仅框架对齐 |

banned 不写入任何产物。材料化后运行 `assert_face_clean`；manifest 记录
`leak_manifest_hash`。分类变化会改变数据协议，旧 Trial 不可混排。

final_test 的 identity 可以材料化，但标签绝不落盘；只有 gate 后 finalizer
可在内存重建一次。

## 数据视图

数据视图允许重建特征面，但不改变样本面。优先用模板：

```bash
python <skill-root>/scripts/modelevo.py write-splits-view \
  --run <run_dir> --view-id <id> \
  --template recency|active|kwin [模板参数] --note "..."
```

复杂视图可提供 `--source views/<id>.py`，脚本暴露
`build_view(df) -> DataFrame`。落盘前必须满足：

- 行数、行序和 sample key 与 v1 逐位置一致；只可增删或替换特征列；
- 输出不得包含 target、ID、时间、treatment 或其明显派生名称；
- reparse 只能消费 spec 授权源，banned 永不可恢复；
- 视图脚本保留在 run 目录，manifest 记录 source path/hash、feature hash、
  parent view 与列数。

视图产物位于 `sample-profile/splits-<view>/`。创建 Trial 时声明
`--action data_view --data-view <id> --heavy`；评估仍使用同一份标签与 sample
key。落盘后先运行 `check-products --spec <spec> --run <run_dir>
--splits-dir <view_path>`。典型用途包括窗口聚合、近因加权、活跃子集、序列
状态和跨表新 join。

### 流式构建与行内约束

- **默认流式**:按 split × 块调用 `build_view` 并追加落盘,内存有界。
  `--chunk-rows`(缺省 `feature_exploration.view_stream_chunk_rows`=200000)控制块行数;
  `--max-mem-gb`(缺省 `view_stream_max_mem_gb`)按行宽估算块大小,RSS 超过上限 85%
  时块减半。`--no-stream` 为全量模式,只适合小数据或非行内脚本。
- **`build_view` 必须是行内变换**:输出第 i 行只依赖输入第 i 行与冻结常量。块内
  `df.index` 是该段在 v1 合并序中的**全局行位置**,与全量模式一致。输出列集合不得
  依赖数据内容(例如按块内取值生成 one-hot 列),不得做跨行的全局统计或逐 split
  参考日。
- **自检**:落盘前在每个 split 的首/末行组各取一段,比对"单段独立调用"与"跨 split
  拼接调用"的输出,要求逐位相等。不等即 `ViewNotRowLocalError`:估算全量峰值 ≤
  预算(`--max-mem-gb`,或 `view_full_fallback_max_gb`=8)时回退全量模式,否则拒绝,
  并提示改写为行内变换或改用 `--from-feature-block`。
- recency 视图脚本须由 `--template recency` 生成(按 `df.index` 查各 split 参考日,
  是行内变换);按整段判定 split 的旧脚本流式自检会失败,重新生成后再落盘。
  active/kwin 本就是行内变换。

### 特征块直接材料化为视图(`--from-feature-block`)

```bash
python <skill-root>/scripts/modelevo.py write-splits-view --run <run_dir> \
  --view-id <id> --from-feature-block <trial>:<block>[,<trial>:<block>...] --note "..."
```

- 把 trial 中已验证的特征块直接追加到父视图特征面,落为新视图:缓存命中就直接用,
  未命中按内核重算。块以 `feature_block` 块名或块函数名指定。不重载原始数据、不经
  `build_view`,与 `--source`/`--template` 互斥。
- 块列与父视图特征或其它块**重名即拒绝**(视图只增列,不替换);落盘带逐值回读与
  缓存抽样重算两道校验。
- 用该视图建的 trial **不要再 `--add-blocks` 同名块**:块列已在视图特征面上,重复
  右接会因列重名报错。

## 特征探索抽样面

```bash
python <skill-root>/scripts/modelevo.py write-splits-sample \
  --run <run_dir> --max-rows 200000 --note "特征探索抽样面"
```

- `sample` 是保留的 view id；train/validation/shadow 按比例、固定 seed 抽样；
- final_test 不抽样也不落盘；manifest 记录原始与抽样行数；
- 落盘后先对 `splits-sample` 执行 `check-products --spec ... --run ...
  --splits-dir ...`；
- 抽样面 Trial 用 `--data-view sample --heavy`；结果只代表抽样证据，不能与
  全量绝对分数直接比较；
- 晋级方向必须回到 v1 全量面复核并做 fresh confirmation;
- 落盘后自动做噪声预检,结果写 `manifest.json.noise_precheck`。
  `recommend_v1=true` 表示抽样面对 MDE 量级没有分辨力,且全量 trial 足够便宜,
  应直接在 v1 上探索。判据见
  [stage-playbook.md](stage-playbook.md#3-adhocbaseline-与抽样面)。

完整 bundle 流程见 [feature-exploration.md](feature-exploration.md)。

## Profile 产物

Profile 不需要 Agent 预分析。spec/loader 已完成且 `validate`、`preflight` 通过后，
主会话直接前台运行 `python <skill-root>/scripts/modelevo.py profile --run
<run_dir>`；失败才使用 `diagnose`，成功后只处理确认门和 critical finding。

`profile` 是长命令且先静默读数据：看到 `profile: 开始读取数据面 …` 这行才开始
计时，更早的无输出不等于卡死，不要 kill。重跑时若数据面未变会复用已有四平面。

| 产物 | Planner 用途 |
|---|---|
| `sample-profile.json/md` | 规模、标签/目标、时间与 split 分布 |
| `leak-classification.md` | phase-1 用户确认门 + **机械提案复核清单** + 经验库对账 |
| `feature-quality.csv` | dtype、coverage、unique、IV/correlation、PSI |
| `feature-woe.csv` | binary/uplift 的分箱、WOE 与 IV |
| `drift-profile.*` | split/月度漂移与稳定性风险 |
| `leakage-audit.md` | group、duplicate、label shift 等数据面问题 |
| `splits-summary` | 行数、目标分布、时间范围与实体重叠 |
| `compression_audit` | 聚合压缩风险与 data_view 候选 |
| `information-inventory.yaml` | reparse/强信号信息源及粒度空间的消费状态 |

IV 或单变量相关性只用于方向排序，异常高信号必须做泄漏复核。critical 数据
问题应先处置或交给用户判断，不能为获得更高指标而忽略。**「没找到问题」
不等于没有问题**：banned-certain/reparse-candidate 是机械判决，复核清单
（尤其 `n_unique` vs 实体数）是防止高价值列被静默封禁的唯一入口。
`leakage-audit` 的 `id_leakage` 只对标识符形态的列告警，判据刻意保守：没报不代表
该列安全，列本身仍按泄漏裁决处理。

## Adhoc 分析

固定 Profile 之后，可围绕特征族、分群、时序、交互、序列、冗余和样本偏移
写一次性分析脚本。脚本先落盘到 run 的 `adhoc/` 或 `feature-lab/` 再执行；
结论必须进入可证伪 Trial，不能把探索图表直接当作晋级证据。
