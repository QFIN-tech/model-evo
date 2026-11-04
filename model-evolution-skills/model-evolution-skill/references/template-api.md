# Trial 模板 API

Generation subagent 只有在编写或修复 Trial code 时读本页。这里定义稳定的
字段与函数契约；操作异常先用 `diagnose`，不要通过阅读 Runtime 源码反推 API。

## 目录

- [文件与常量](#文件与常量)
- [数据访问器](#数据访问器)
- [fit_and_score](#fit_and_score)
- [特征块 trial_feature_blocks](#特征块-trial_feature_blocks)
- [补丁式写码](#补丁式写码)
- [模型拓扑](#模型拓扑)
- [官方 hurdle 容器与子模型缓存](#官方-hurdle-容器与子模型缓存)
- [辅助函数](#辅助函数)
- [HPO](#hpo)
  - [HPO 引擎](#hpo-引擎)
- [Smoke、Quick 与环境变量](#smokequick-与环境变量)
- [快通道与结果解析](#快通道与结果解析)

## 文件与常量

```text
trials/<id>/code/train.py
trials/<id>/config/trial.yaml
```

`new-trial` 生成 marker 段与 body 段。通常只修改 `fit_and_score` 或显式的
transform，不改 `main()`、数据访问器和产物路径；`--from-parent` 则不写代码。

| 常量 | 语义 |
|---|---|
| `RUN_DIR` / `TRIAL_DIR` | run 根与本 Trial 根 |
| `SPLITS` | 当前数据视图的特征面根 |
| `LABELS_DIR` / `REPARSE_DIR` | 标签面与授权解析源根 |
| `TARGET` / `MODEL_TYPE` / `TREATMENT` | TaskSpec 注入；无 treatment 时为空 |
| `BANNED` / `REPARSE_SOURCES` | 冻结的泄漏分类 |
| `SEED` | 当前 Trial 的实际默认 seed |

## 数据访问器

- `load_labels(split) -> DataFrame`：返回目标、treatment 和对齐键；搜索目录
  不存在 final_test label。
- `reparse_frame(split, columns) -> DataFrame`：只接受 `REPARSE_SOURCES`
  白名单；返回请求列，不返回对齐键；解析产物使用新前缀。
- `load_config() -> dict`：读取 `../config/trial.yaml`。
- `exclude_columns(df, cfg=None) -> list[str]`：统一处理 target、ID、时间、
  treatment、banned、reparse、config exclude 和白名单；不要另写 FEATURES。
- `assert_banned_clean(X)`：fit 前终检，默认 main 已调用。

Trial 禁止直接调用 `load_data()`；identity 平面也不是模型信息源。

## fit_and_score

唯一训练与打分入口：

```python
def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None) -> tuple[score, metrics_extra]
```

- smoke 和全量共用本函数，不在 main 中复制第二条训练路径；
- early stopping 默认在 outer validation 上(`_fit_with_eval` 单段拟合:
  100% train 只训一次,`eval_set=[(X_va, y_va)]`);`training.early_stop_source:
  train_tail` 时回退为 train 内部切出的 es_valid。HPO 的**超参选择**始终只用
  train 内切分(lint 规则 7);
- `w_tr/w_va` 仅 uplift 使用；
- 分类 score 为正类概率，回归为预测值，uplift 为 CATE；
- **lint 规则 4**(回归):`.predict_proba(...)`、sigmoid/expit/softmax 的输出
  **实际流入**打分(`DataFrame({"score": …})`、`x["score"] = …`、`dump_*score(…)`、
  `fit_and_score` 返回值首元素)才拒绝;概率 × 另一模型的非常量输出(hurdle `p·μ`)
  不算概率口径;
- `metrics_extra` 必含 `model_num`；
- 训练完成调用 `dump_model(model, X_tr.columns)`；模型对象还要赋给
  `_last_model`，供默认 `leak_scan` 使用；
- 二分类脚手架的 `# lint:scaffold-fit-and-score` 必须被真实实现替换。

**交付契约**:`finalize-retrain` 会在沙箱里再调用一次本函数,并把 `_last_model`
直接作为交付模型(口径见 [stage-playbook.md](stage-playbook.md#交付重训口径))。因此:

- train_tail 口径下早停必须切**传入 X_tr 的尾部**(模板 `_fit_with_es` 与
  `ES_FRACTION` 即如此),不得依赖"行数等于 train"之类的全局假设;
  validation 口径下早停只用传入的 (X_va, y_va),模型宜可 `sklearn.clone`
  (不可 clone 时会回落 train_tail 口径复跑本函数,因此 train_tail 分支也必须可用);
- 返回的模型必须能对新帧 `predict`(回归)/ `predict_proba`(分类)打分;
  **不要求** sklearn `get_params/clone` —— 两段式 hurdle 等复合容器可直接交付,
  但回归容器**不要**暴露 `predict_proba`(交付打分按任务类型分派,回归一律
  `predict`);
- **交付契约自检**:structural 拓扑 trial 在 `new-trial` 时写
  `trial.yaml.deliverability`,只告警不拦截;`deliverable` 为否时收尾交付会失败,
  建议改用官方 `HurdleRegressor`;
- 副作用必须受 `MODELEVO_TRIAL_EXEC` 写保护(模板 dump_* 已具备),交付沙箱
  会把 `TRIAL_DIR` 指向临时目录;
- 加宽 trial 必须 `dump_eval_frame`(交付前的特征面证明读
  `model/validation_input.parquet`)。

## 特征块 trial_feature_blocks

reparse 序列 / JSON / 计数与近因族这类**无训练期拟合状态**的行内变换,一律
写成批内核 + `feature_block`,不要在 main 里手写逐行循环(lint 规则 21:
reparse train ≥ 20 万行时拒绝未走缓存的解析)。

```python
from runtime.feature_cache import feature_block

_SEQ_COLS = [...]                       # 输出列,必须带解析前缀(如 seqd_)

def _seq_kernel(b):                     # b = {"seq_d365": [...], "seq_gap_d365": [...]}
    return [_row(s, g) for s, g in zip(b["seq_d365"], b["seq_gap_d365"])]

def trial_feature_blocks(split: str) -> pd.DataFrame:
    return feature_block("seqd", _seq_kernel, split=split, columns=_SEQ_COLS,
                         sources=["seq_d365", "seq_gap_d365"],
                         splits_dir=SPLITS, n_jobs=n_jobs())
```

契约:

- 内核 `fn(batch) -> (n, len(columns))` 数值矩阵;`batch` 缺省是
  `{列名: python list}`(`batch_format="numpy"|"pandas"` 可选);内核必须是
  batch 内容与模块级常量的纯函数(不读文件、不依赖训练期状态);
- 缓存键含**内核代码指纹**(递归引用的同模块函数与全局常量)与输入数据内容:
  同一内核跨 trial / smoke / 交付重放直接命中,内核或常量任何改动自动失配;
  依赖指纹抓不到的外部状态时用 `version="..."` 换键;
- `plane="features"` 可从特征面取列(如基础列交互),默认 `plane="reparse"`;
- 定义了 `trial_feature_blocks` 后,main 会在 dtype 治理**之后**自动右接
  train/validation/shadow_test 三个 split 的块;`finalize-retrain` 交付重放与
  final_test/shadow 打分走同一函数;
- 子样本(smoke/quick)用 `runtime.feature_cache.align_rows(block, X)` 按
  行位置对齐;行位置不合法时报错,绝不错位拼接;
- `modelevo.py feature-cache --run <run>` 查看缓存(含 `CACHE_INVALIDATION`
  失效预判行),`--clear` 清空;`MODELEVO_FEATURE_CACHE=0` 只算不存(诊断)。

**复用其它 trial 的特征块:`new-trial --add-blocks <trial>:<block>[,...]`**:

- 块标识 = `feature_block` 的块名、包装函数名或 `_<块>_block`。harness 从源 trial
  冻结的 `code/train.py` 求依赖闭包,**字节级**复制进新 trial 的
  `# ==== v2.17.0 add-blocks 段 BEGIN … END ====`,并注册进 `trial_feature_blocks`
  (原有块 + 追加块按列右接,列重名报错)。
- 新 trial 已有 AST 等价的同名定义时跳过;已有但定义不同时保留新 trial 的,并记冲突。
- 创建期比对源与新 trial 的块指纹,不一致 → 拒绝、恢复原代码并给出差异;一致则
  保证缓存命中,stdout 打印 `add-blocks: … 指纹 … 一致`。
- 不可用于 `--action confirm/automl`;`--from-parent` 配 `--add-blocks` 即构成有效
  差异,无需 `--set`。**段内代码勿手改**:改动会让内核指纹漂移、缓存失配,要改语义
  就给块换名或 `version=` 后按新块重算。
- 视图已含同名块列时(`write-splits-view --from-feature-block` 材料化的视图),在该
  视图上不要再追加同一块,特征块列与基础面重名会直接报错。

有训练期拟合状态(分群统计、train 上选词表)的变换继续用 `widen_cached`
(键为 `_widen` 的代码指纹,同实现跨 trial 命中)。**`_widen` / 特征块内核里不得
训练会产出预测的模型**(那是级联,见下节)。

## 补丁式写码

子代相对父代通常只改 3~5%(新块、换内核、改 `fit_and_score`/`build_estimator`)。
**默认用 patch 写子代,不要读写整份 train.py**:

```bash
modelevo.py trial-skeleton --run <run> --trial <父代>      # 只读大纲,替代读 160KB 全文
modelevo.py new-trial --run <run> --from-parent <父代> --patch next.py \
    [--set k=v] [--add-blocks t:blk] --hypothesis ... --changed-modules feature
```

- **大纲** `trial-skeleton`:每个顶层定义的行区间/签名、`[P:原因]` 受保护、`[E]` 典型
  可改、`(被后定义覆盖)`;特征块列表(块名/内核/是否注册/列数与列前缀/sources/依赖
  闭包行数);段标记。`--json` 机读,`--no-probe` 不起探针(列只给静态表达式)。
- **patch 文件**只写**新增或替换**的顶层定义(函数/类/单名常量/import),合并规则:
  - 与父代同名 → 原位替换父代**最后一个**同名绑定(须同类:函数换函数、常量换常量);
    与父代 AST 等价的定义跳过;
  - 新名字 → 按 patch 原文顺序追加进 `# ==== v2.19.0 patch 段 BEGIN … END ====`
    (main 守卫之前,父代全部定义之后);
  - for/assert/增量赋值等其它顶层语句只能作用于本 patch 的新名字;改父代常量写同名整体定义;
  - 原位替换的定义,其定义期求值(常量右值、默认参数、装饰器)不得引用本 patch 新增名字。
- **指令**(字符串字面量列表;其它 `PATCH_*` 名拒绝):
  - `PATCH_REGISTER_BLOCKS = ["_xxx_block"]`:把调用 `feature_block` 的包装函数注册进
    `trial_feature_blocks`(原有块 + 新块按列右接,列重名报错);已注册的拒绝。
    与 `--add-blocks` 共用注册表,两者互相可见;
  - `PATCH_REPLACES = ["fit_and_score", ...]`(可选,推荐):声明应替换的父代名字,与实际
    合并结果不符(拼错 → 被当新增;意外覆盖)即拒绝。
- **受保护符号**(不可替换也不可新增,完整清单列在 skeleton 末尾):数据加载/特征面/泄漏防线
  (`load_labels`/`exclude_columns`/`leak_scan`/`BANNED`/`SEED`…)、评估/产物(`main`/
  `dump_*`/`run_smoke`…)、早停与轮数上限(`_fit_es_on_validation`/`_es_*`/`EARLY_STOP_ROUNDS`…)、
  HPO 入口(`_hpo_fit_inner`/`_build_model`/`_inner_metric`)、设备/线程/seed 并行、特征块与
  加宽缓存基础设施(`trial_feature_blocks`/`widen_cached`)。`fit_and_score`/`build_estimator`/
  `prepare_features`/`_widen`/各 `*_kernel`/`*_block` 可替换。`from runtime.feature_cache
  import feature_block` 是允许的规范导入。
- **校验**:创建前做合并预演(违规直接拒绝、不建目录);合并后做导入探针、注册块探针、
  lint 增量(相对父代**新增**的 lint 错误 → 回滚)。合并后的完整 train.py 照常走 lint、
  拓扑门、冻结与完整性。
- **限制**:必须配 `--from-parent`;不可用于 `--action confirm/automl`;可与 `--set`/
  `--features`/`--add-blocks`(patch 先合并,再追加块)/`--action hpo` 组合;合并后与父代
  零差异拒绝(纯参数变体用 `--set`)。长 trial 运行期间可先起草下一代 patch,
  `finalize-generation` 之后再 `new-trial`。

```python
# next.py —— 新增 seqz 块 + 替换 build_estimator
from runtime.feature_cache import feature_block
PATCH_REGISTER_BLOCKS = ["_seqz_block"]
PATCH_REPLACES = ["build_estimator"]
_SEQZ_COLS = ["seqz_n"]
def _seqz_kernel(b):
    return [[float(len(s or ""))] for s in b["seq_d365"]]
def _seqz_block(split):
    return feature_block("seqz_v1", _seqz_kernel, split=split, columns=_SEQZ_COLS,
                         sources=["seq_d365"], splits_dir=SPLITS, n_jobs=n_jobs())
def build_estimator(n_jobs: int, n_rows: int = 0):
    ...   # 完整新实现(整体替换,不是 diff)
```

## 模型拓扑

每个 trial 默认**只有一个可训练模型**。runner 在执行前(冻结源码之前)对
`code/train.py` 做静态拓扑检测,结果写 `trial.yaml.model_topology`:

| 拓扑 | 含义 | 放行 |
|---|---|---|
| `single` | 一个模型;HPO inner、早停、confirm 的 seed 复现、S-learner 双次打分都不算多模型 | 默认 |
| `structural` | 子模型只吃原始特征,固定公式组合:hurdle `p·μ`(回归)、T-learner `μ1−μ0`(uplift) | 任务类型白名单内放行;白名单外须批准 |
| `cascade` | 模型预测作为另一模型的特征/目标(子分、辅助标签打分、蒸馏、OOF logit),或在 `_widen`/特征块里训练产出预测的模型 | 须人工批准 |
| `ensemble` | 多个模型的预测被平均/加权/累加/rank 组合,多 seed 平均,sklearn Voting/Stacking/Bagging、SeedBag、AutoGluon | 须人工批准 |

被拦时:不训练、不冻结源码、不记失败,trial 保持 CODE_WRITTEN;`run` /
`wait` 输出 `TOPOLOGY_BLOCKED` 并以退出码 5 返回;`run-generation` 跳过该
trial、其余照跑;请求写入 `approvals/requests/` 与 `logs/user-decisions.md`
(PENDING)。两条出路:

1. 改写成单模型后重跑同一 trial(源码尚未冻结);
2. 转告用户,由**用户在自己的交互式终端**执行
   `modelevo.py approve-topology --run <run> --trial <id> --reason "..."`
   (缺省范围 recipe:覆盖该 trial 及其后代,但新增的组合方式须重批;
   `--scope trial` 只认当前代码 sha;`--scope run --kind cascade` 按种类)。
   批准要键入随机确认码;Agent 会话的 Bash 不是终端,guard hook 也拒绝
   Agent 调用该命令或写 `approvals/`。`approve-topology --list` 只读。

声明:`new-trial --topology single|structural:hurdle|structural:t_learner|
cascade|ensemble`(`--from-parent` 缺省继承父代声明)。声明只**增加**需批准
项,检测到的拓扑是底线。未批准的级联/融合冠军 `finalize-retrain` 拒绝交付。

静态检测识别常见写法,不是证明;不要为"过门"改写法规避检测——它和改码
绕过 lint 同属破坏执行边界(铁律 9)。

## 官方 hurdle 容器与子模型缓存

hurdle 一律用官方容器,不要手写 `HurdleXGB` 之类非 sklearn 容器(交付 clone 失败,
且只改一段的子代无法命中子模型缓存)。

**`runtime.hurdle.HurdleRegressor(clf=None, reg=None, *, clf_params=None,
reg_params=None, threshold=0.0, submodel_cache=True, cache_dir=None)`**:

- `predict = P(y>threshold) · max(E[y|y>threshold], 0)`,**不暴露 `predict_proba`**;
  诊断用 `positive_prob(X)` / `conditional_mean(X)`,两段遥测用
  `segment_telemetry()`(各段 best_iteration/cap_bound 等)。
- `BaseEstimator`:可 `sklearn.clone`,可 joblib 持久化,并能在新进程加载。
- 参数:`clf__<键>` / `reg__<键>` 是分段键,无前缀的非容器键两段共用;`hurdle_lr` /
  `hurdle_cap` 是 `learning_rate` / `n_estimators` 的别名。原型为 None 时按
  `clf_params`/`reg_params` 构造 XGBClassifier/XGBRegressor。
- `fit(X, y, eval_set=[(X_es, y_es)])`:两段各自早停,分类段用 `1[y_es>threshold]`,
  回归段用 eval 面的正例行;无 eval_set 时自动关闭早停(冻结轮数重训走这条路径)。

**模板入口 `new-trial --hurdle-template`**(仅 regression、仅 xgboost):把脚手架
`fit_and_score` 换成官方 hurdle 段
`# ==== v2.17.0 官方 hurdle 段 BEGIN … END ====`。段内提供三个入口
`_hurdle_estimators` / `_fit_hurdle` / `fit_and_score`,常量 `HURDLE_LR`、
`N_ESTIMATORS_CAP`、`HURDLE_CLF_PARAMS`、`HURDLE_REG_PARAMS` 可改,并自动注入
`HURDLE_HP_SEGMENT`。

- 隐含 `--topology structural:hurdle`。
- 可与 `--action explore|exploit|crossover|hpo` 组合,不能与 `--from-parent` 同用;
  hurdle 父代的子代直接 `--from-parent`,即继承容器。
- `--set clf__…/reg__…` 与 `--action hpo --from-parent <本 trial>` 直接可用。
- metrics 记 `model_num: 2`、`hurdle`(两段遥测 + 正例率 + p/μ 均值)、`es_info`、
  `submodel_cache`。

**子模型缓存 `runtime.submodel_cache.fit_or_load(name, estimator, X, y, *,
eval_set=None, fit_kwargs=None, run_dir=None, extra_key=None)`**:

- 返回 `(已拟合估计器, info)`。总在 clone 上拟合,`info.status` ∈
  `hit|miss|disabled|uncacheable|error`。
- 键覆盖输入内容、估计器类与参数、设备类型、库版本、seed、fit_kwargs、extra_key。
- **只在结果可确定时缓存**。以下情况只训练不缓存,`status=uncacheable` 并记 reason:
  随机采样参数 < 1 却缺 seed、LightGBM 未开 deterministic、未知估计器有随机性但无
  `random_state`。
- `MODELEVO_SUBMODEL_CACHE=0` 整体关闭;`feature-cache --clear-submodel` 清理。
- `HurdleRegressor` 已集成:只改一段的子代,另一段在同输入、同参数、同 seed 下直接
  命中。摘要写进 `metrics.submodel_cache`(segments/hits/misses/saved_fit_sec/enabled)。

**子模型 HPO**:不提供 `--hpo-target`。`clf__` / `reg__` 分段键已能把搜索限定到单段;
在官方容器下,未搜索那段的参数与输入不变,会命中子模型缓存。

## 辅助函数

| 函数 | 契约 |
|---|---|
| `n_jobs()` | 读取 runner 注入的线程预算；模型不得硬编码线程 |
| `pick_gpu()` | 按调度信号选择设备；main 已调用 |
| `_gpu_allocated(n_rows)` | 仅以 `MODELEVO_GPU` 和行数阈值判断 GPU 意图 |
| `_xgb_cuda_built()` | 判断 XGBoost 是否支持 CUDA |
| `_lgbm_fit_safe(...)` | LightGBM GPU 不可用时安全回退 CPU；LGBM fit 必须走它 |
| `prepare_features(X_tr, X_va)` | 数值化 object、低基数 one-hot、高基数拒绝 |
| `dump_model(model, columns)` | 保存 joblib、原生格式和 feature columns |
| `dump_feature_importance(model, columns)` | 保存 gain 重要性；不支持时跳过 |
| `dump_eval_frame(X_va)` | 保存真实 validation 模型输入；加宽 Trial 必须调用 |
| `dump_shadow_score(score)` | confirm 保存 shadow 分数；普通搜索不反馈 |
| `leak_scan(model, X_tr, y_tr, top_k=50)` | 动态泄漏哨兵，写入 metrics |
| `step(msg)` | 单行阶段进度，长训练可补中间状态 |

未经基数检查禁止 `get_dummies`。transform 必须在 validation、shadow、最终重训
中使用同一逻辑。

**工件写保护**：`dump_model` / `dump_eval_frame` /
`dump_shadow_score` 只在 runner 执行本 trial 时写盘（runner 注入
`MODELEVO_TRIAL_EXEC=<trial 目录名>`）。在 adhoc 脚本、自测或交付期加载
`train.py` 并调用 `fit_and_score` 时它们会打印 `[guard] … 拒写` 并跳过——
这是预期行为，冻结工件因此只读。老模板渲染的 trial 没有这道保护，
**不要**在外部加载它们调用 `fit_and_score`。

shadow 打分实际口径(NaN 原生或降级填 0)记在 `metrics.shadow_scoring_mode`。
multi-seed 变体轮不写工件，`model/` 始终是产生 validation/shadow 分数的
primary 模型（`metrics.artifact_model = primary_seed`）。

## HPO

HPO 在 train 的 inner split 上搜索，best params 再全量重训；outer validation
只评一次。配置至少包含 search space 与 `--budget-configs`。

搜索空间 YAML 顶层是“参数名 → 分布”的映射，只支持三类：`float`/`int`
使用 `low`、`high`，float 可加 `log: true`；`categorical` 使用 `choices`：

```yaml
learning_rate: {type: float, low: 0.01, high: 0.15, log: true}
max_depth: {type: int, low: 3, high: 10}
booster: {type: categorical, choices: [gbtree, dart]}
```

优先直接使用 `plan` 自动生成的
`planner/default-search-space-<algorithm>.yaml`；自定义空间的键必须是目标模型
实际消费的参数名。

有两条创建路径：

1. **模板渲染**：`--action hpo --algorithm <algo> [--hpo-parent <id>]`。
   适合标准特征面；Harness 生成 HPO 四件套并继承父代 config features。
2. **复制父代**：`--action hpo --from-parent <id>`。适合必须保留父代自定义
   transform；父代码缺 HPO 段时 Harness 自动嫁接并运行可导入性探针。

**复制父代路径的 main 改线**:嫁接段的 `fit_and_score` 只有被 `main()` 调用才
生效。baseline 血缘父代的 main 内联 `build_estimator → fit → 打分`,创建期会把
该段机械改线为 `fit_and_score(...)` 调用 + 分数落盘;识别不出该段时**创建即失败**。
Lint 规则 7(hpo 节存在但 main 不调用 fit_and_score)与规则 22(无
`validation_score.parquet` 落盘)在 run 前兜底。

不要创建空 scaffold 只为充当 HPO 载体。嫁接或 import 失败时删除未启动的
误建 Trial 并重建，禁止手改已冻结父代。大数据使用 `--hpo-sample-rows` 做
inner 搜索，best params 仍在全量 train 重训。

模板 HPO 依赖 `fit_and_score`、`_hpo_fit_inner`、`_INNER_METRIC_UP`、
`dump_json` 和正确的 `_HPO_ALGO`。创建探针与 Lint 会检查这些契约。
父代已是嫁接过的 HPO trial 时,旧嫁接段被整段替换而不是叠加。

### HPO 引擎

热启动、边界外扩、lr 收缩等由 Runtime 在一个 trial 内完成 —— **一个预算充足的
HPO trial 代替多轮手工追边界**。开关见 `evolution.hpo.*` 与 `new-trial --hpo-*`:

| 机制 | 行为 | 关闭 |
|---|---|---|
| 热启动 | 接收 run 内 `inner_signature` 完全一致的历史观测,不计入本 trial 的 config 数 | `--hpo-no-warm-start` |
| 基准点 | 父代配方裁进空间后作为第一个评估点 | — |
| 边界自适应 | 前 5 名贴边的参数自动外扩(每参数最多 2 次) | `--hpo-no-expand` |
| lr 收缩 | `lr_mode=auto` 且空间有 lr 维、inner 行数 ≥ 20 万时:搜索固定 `search_lr`,重训降 lr、放大 cap;**显式 `shrink`** 恒收缩(lr 维含 `clf__`/`reg__` 前缀一律移出)。生效值看 `metrics.hpo_lr_mode_effective`,请求≠生效时 WARNING | `--hpo-lr-mode search` |
| 并行 | 同一张已租 GPU 上线程并发;auto = GPU 上的 xgboost 取 2 | `--hpo-parallel 1` |
| 多保真 | 可选:1/9→1/3→1 行比例逐级减半 | 默认关;`--hpo-fidelity successive_halving` 开 |

产物:`hpo-study.json` 记 `inner_signature`、`warm_start`、`space_initial/
space_final/space_expansions`、逐 config 遥测(`best_iteration`/`cap_bound`/
`fit_sec`/`origin`)与 `diagnostics`(`top5_spread`、`tpe_gain_over_random`、
`boundary_hits`、`cap_bound_configs`)。**`best_params` 是重训配方**(lr 收缩
后的 lr、生效的 cap 键 `n_estimators`/`iterations`),`search_best_params`
是搜索原始最优 —— confirm 快通道与 finalize-retrain 都从 `best_params` 取配方。
`evolution.hpo.fidelity_recheck_top_k`(默认 0=关)可把前 K 名放到全量 inner 上
复评后再选 best_params。`metrics.n_features` 记拟合期真实模型列数(含特征块列),
基础列数另记 `n_features_base`。

**早停与 cap**:默认 `_fit_es_on_validation` 单段拟合(早停面 = outer validation,
遥测 `es_source=validation`、`fit_mode=single`);train_tail 口径与 HPO inner 用两段式
`_fit_with_es`(`learning_curve.csv`、`best_iteration` 取第一段)。最优轮数与
`stage1_rounds` 之差 = 早停耐心 ⇒ 正常收敛;`evaluation.complexity.cap_bound`
统一语义为实训轮数 ≥ cap(hurdle 任一段撞即 true),`near_cap` 仅作提示;
外扩后仍撞 cap 时 `es_info.cap_bound_unresolved=true`(评估侧标 `cap_confounded`)。

**容量口径单源**:runner 把 run 配置的 `baseline.n_estimators_cap`/
`early_stopping_rounds` 下发为 `MODELEVO_N_ESTIMATORS_CAP`/
`MODELEVO_EARLY_STOP_ROUNDS`;模板 `_build_model` 优先透传 trial 的
`N_ESTIMATORS_CAP`/`EARLY_STOP_ROUNDS` 模块常量。不要再手补 `n_estimators_cap=`。

**confirm / 参数子代**:`--from-parent <HPO trial>` 剥离 hpo 节后走模板的算法化
非 hpo 分支(`_build_model(hyperparameters) → _fit_with_eval → _model_score`)。

**两道创建期门**:

- 边际递减门:新 HPO 的父代(或父代 confirm 的配方)本身是同面 HPO 且其
  Δ_parent < MDE(量纲感知值)→ 拒绝,需
  `--override-hpo-diminishing "理由"`(附上一轮搜索是否完整)。
- confirm 时机门:父代有 PROPOSED/CODE_WRITTEN/RUNNING 的 HPO 子代 → 拒绝,需
  `--allow-confirm-during-hpo "理由"`。

## Smoke、Quick 与环境变量

| 变量 | 语义 |
|---|---|
| `MODELEVO_SMOKE=0` | 仅纯超参、代码路径未变时可跳过 smoke |
| `MODELEVO_SMOKE_RUN=1` | Harness 内部 smoke 标志；代码不要依赖它 |
| `MODELEVO_QUICK_RATIO` | successive halving 的 train 子样本比例 |
| `MODELEVO_DEADLINE_TS` | HPO 绝对截止时间 |
| `MODELEVO_GPU` | runner 的 GPU 意图；不要从 CUDA_VISIBLE_DEVICES 反推 |
| `MODELEVO_TRIAL_EXEC` | runner 注入的执行凭证（= trial 目录名）；dump 函数据此决定是否写工件，代码不要伪造 |
| `MODELEVO_SEED_VARIANT=1` | 模板内部 multi-seed 变体轮标志；期间 dump 函数不写工件 |
| `MODELEVO_SKIP_LINT=1` | 人工诊断逃生口，不用于正常训练 |
| `MODELEVO_N_ESTIMATORS_CAP` / `MODELEVO_EARLY_STOP_ROUNDS` | runner 按 run 配置下发的 cap/早停耐心(`build_gbm` 默认值) |
| `MODELEVO_CAP_EXTEND_TIMES` / `MODELEVO_CAP_EXTEND_FACTOR` | 撞 cap 时的外扩次数(默认 1)/倍数(默认 2) |
| `MODELEVO_ES_LR_SCALE` | 早停耐心 = max(rounds, ceil(scale/lr)),默认 0.5,0 关闭 |
| `MODELEVO_ES_SOURCE` | outer 拟合早停面,runner 按 `training.early_stop_source` 下发;`validation`(默认,单段)/ `train_tail`(两段式) |
| `MODELEVO_SUBMODEL_CACHE=0` | 关闭子模型缓存(`fit_or_load` 只训练不存取) |
| `MODELEVO_MEMORY_ADMISSION=0` | 关闭内存准入(诊断逃生口,正常训练不用) |

Smoke 使用 train/validation 小样本完整调用 `fit_and_score` 并检查行数、NaN 与
落盘路径；产物写 `smoke/`，不进入反馈。Quick 本身是子样本时自动跳 smoke。

训练分数已落盘但评估被中断时，按 `diagnose` remedy 重入
`run --trial <id>`；runner 会走 evaluate-only，不手改 `evaluation.json`。

## 快通道与结果解析

`new-trial --from-parent <id> --set key=value [--features ...]` 会复制父代 code，
把差异写入 config。仅用于代码路径完全相同的配置差异；改 transform 或模型
结构必须创建普通 child Trial。

确认候选使用 `new-trial --from-parent <id> --action confirm --generation <N>`。
此路径无需 `--set`；Harness 自动换 seed、令 `multi_seed>=3` 并绑定 validation
与 shadow。父代必须已通过代级选择校正且源码完整。

`fit_and_score` 必须读取 `cfg["hyperparameters"]` 或等价表达，确保快通道
覆写真正生效。confirm 默认使用 multi-seed；各 seed 分数保存在
`seed_scores/` 并进入稳定性门。

`run` 终态只解析最后一行，不常规读取日志：

```text
SUMMARY<TAB>trial_id<TAB>status<TAB>verdict<TAB>metric<TAB>delta_parent=...
<TAB>delta_baseline=...<TAB>shadow=...<TAB>failure=...<TAB>runtime=...
```
