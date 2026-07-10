# classification-model-training 执行约束与异常处理

> 本文件从 `classification-model-training/SKILL.md` 第 6/7 节抽出,包含覆盖范围、"不覆盖"清单、何时用、session 约定、迭代报告、输入 yaml 落盘约束、数据安全红线(项目级通用约定复述)、变更前置流程、异常处理全表。SKILL.md 第 6 节只保留红线级约束(必显式传 features / 变更前置),第 7 节保留单行指针指向本文件。

## 1. 覆盖范围

基于 `splits/{train,test,oot}.parquet` 训练 xgboost/dnn/lr 二分类模型,落盘八阶段产物(features/model/evaluation/predictions/explainability/comparison/logs/config);评估委托 `classification-model-evaluation`,对比委托 `classification-model-comparison`;单 run 顶层 `report.md` 整合报告。

## 2. 不覆盖(由其他 skill 负责)

- **取数**:用 `feature-matching`(产 sample.parquet)
- **切分**:用 `feature-analysis`(按 `model.split` 切 train/test/oot,产 `splits/{train,test,oot}.parquet`)
- **特征分析**:IV/PSI/相关性用 `feature-analysis`,报告仅作人工参考
- **特征筛选/调参**:用 `classification-model-tuning`(产 `-feat` / `-tuned` 新 run)
- **历史模型推荐**:用 `classification-model-recommend`
- **会话级横向对比聚合**:用 `classification-model-comparison`

## 3. 何时用

当用户用一份训练数据训练一个新模型时使用;上游建模前的特征分析与切分请走 `feature-analysis` skill。

## 4. Session 决议(本 skill 不负责)

`session_dir` 由上游 `classification-model-development` / `classification-model-orchestration` 传入;无 session 上下文时由调用方先跑 `classification-model-development/scripts/list_sessions.py` 列历史 sessions 并询问用户,本 skill 直接消费 `--output_dir`。

## 5. 迭代报告

每次 run 的顶层整合报告落 `<session_dir>/new-models/{algo}[-suffix]-v{N}/report.md`,session 级项目总报告由 orchestration 侧的 `fill_report.py` 回填到 `<session_dir>/report.md`(见根目录 CLAUDE.md 会话与输出收口约定)。

## 6. 输入 yaml 落盘约束(强制)

`train_config.yaml` 输入 yaml 必须放 `<session_dir>/new-models/{algo}-v{N}/config/` 下(model 内部 config 目录),**严禁落到 session 根目录、`task-spec/`、`sample-features/`、`<skill_dir>/config/` 等位置**。

理由:
1. 输入 yaml 与 run 一一对应,放 model 内部目录天然绑定 run,不串味
2. `write_train_config_yaml` 在该目录原地写 `_manifest.json`(含 `source_yaml` 指向自身),不做副本拷贝,避免冗余
3. 与上游 `task-spec` / `feature-matching` 的 `sample_config.{model_name}.yaml` 落各自子目录的机制对称

`<skill_dir>/config/` 只放 `train_config.example.yaml` 模板,不存实际输入。

## 7. 数据安全红线(全模式强制)

禁止在配置/where 中硬编码用户 ID/手机号/身份证号。

## 8. 变更前置流程(强制遵循 CLAUDE.md)

修改训练/对比代码前,先输出「变更计划」(一、修改内容 二、预期影响 三、回滚方案)并等确认。

## 9. 异常处理

| 异常 | 处理方式 |
|---|---|
| 未传 `--data_dir` 且默认路径 `<output_dir>/sample-features/splits/train.parquet` 不存在 | 停止执行,提示先跑 `feature-analysis` 切分 `splits/{train,test,oot}.parquet` 或显式传 `--data_dir` |
| 配置 yaml 校验失败(`validate_config`) | 停止执行,按报错修正 yaml |
| baseline 默认扫描无命中 | 跳过 comparison 阶段,不影响主流程 |
| `model.baseline_eval_dir` 显式为 `null` | 跳过 comparison 阶段(预期行为,不产 `comparison/` 子目录) |
| version 含非白名单字符 | 自动归一为 `_`,不报错 |
| 特征列表为空且未配 `feature_list_source` | 停止执行;不得回退到 optbinning 自动筛选 |

---

> 关联:SKILL.md 第 6/7 节。
