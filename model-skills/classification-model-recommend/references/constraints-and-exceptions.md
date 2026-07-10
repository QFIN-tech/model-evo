# classification-model-recommend 执行约束与异常处理

> 本文件从 `classification-model-recommend/SKILL.md` 第 6/7 节抽出,包含覆盖范围、职责边界、Step 4/5 强制规则、评估委托边界、PSI 能力边界、三档切分与取数、数据资产边界、需求解析原则、模型报告规范、t-1 滞后 JOIN 模式、数据安全红线(项目级通用约定复述)、异常处理全表。SKILL.md 第 6 节只保留红线级约束(Step 4 强制 / Step 5 强制 / 需求解析原则 / t-1 滞后 JOIN),第 7 节保留单行指针指向本文件。

## 1. 覆盖范围

### 1.1 本 skill 覆盖

解析自然语言需求(客群+用途)→ 读模型台账语义筛选排序 → 展示推荐详情;询问是否在样本上评估 →(可选)生成 `fetch_eval_sample.py` 脚本取数+切分+委托评估;评估委托 `classification-model-evaluation`,产 train/test/oot + all 合并共 4 份标准化三件套。

### 1.2 本 skill 不覆盖

- 训练新模型 → 用 `classification-model-training`
- 特征质量分析(IV/PSI/相关性) → 用 `feature-analysis`
- split 间 PSI / 多版本横向对比 → 用 `classification-model-comparison`
- 会话编排 / 任务路由 → 用 `model-task-routing` / `classification-model-orchestration`

## 2. 职责边界

本 skill 只做历史模型推荐与(可选)评估委托,不训练新模型(走 `classification-model-training`),不做特征分析。

## 3. Step 4 强制(展示详情)

展示详情为强制:不允许只输出 model_id 列表,推荐理由必填。

## 4. Step 5 强制(询问评估)

询问评估为强制:不能跳过、不能默认不评估。

## 5. 评估委托边界

评估逻辑委托 `classification-model-evaluation`:评估阶段调其 `eval_single.py` 产标准化三件套。

## 6. PSI 能力边界

split 间 PSI(如 test vs train 分数分布 PSI)不在本 skill 产出范围内;如需 split 间 PSI 或多版本对比,调 `classification-model-comparison`。

## 7. 三档切分与取数

三档切分复用 task-spec/data-profile 的切分区间,Train/Test/OOT 分别评估,便于观察 OOT 衰减;spark 取数(样本表⋈模型表 JOIN) + 本地 `split_sample.py` 切分,避免 Spark 侧重复扫描。

## 8. 数据安全红线(全模式强制)

脚本仅做分组聚合统计,**不输出任何用户级明细**;严禁在 `eval_config.yaml` 或 `--where` 中硬编码用户 ID/手机号/身份证号。

## 9. 数据资产边界

catalog/reports 是数据资产,不属于 session,不要往 `runs/` 搬;推荐/评估产物一律落 `${session_dir}/model-recommend/{model_id}/`。

## 10. 需求解析原则

解析需求时条件缺失不臆造,留空即不限;解析结果必须回显给用户确认。

## 11. 模型报告规范

模型报告命名 `reports/{model_id}_{模型简称}.md` 并在台账 `模型报告路径` 列登记;上传规范见 `model-knowledge/assets/historical-model-knowledge/reports/README.md`,模板见 `_template_model_report.md`。

## 12. ⚠️ t-1 滞后 JOIN 模式(`--score-lag-day 1`)

启用时:样本表 day t LEFT JOIN 模型分表 day t-1,适用于"当天样本用昨日模型分"场景。

要求:
- `dt_col` 在 `join_keys` 中
- lag=1 时模型分表窗口自动平移为 `[fetch_start-1, fetch_end-1]`
- split 区间仍按样本表 pday 不变

## 13. 异常处理

| 条件 | 处理方式 |
|---|---|
| 台账文件不存在 | 停止执行,提示检查 `model-knowledge/assets/historical-model-knowledge/model_catalog.csv` 路径 |
| 筛选为空 | 说明哪个维度无匹配,给出次优模型或建议(放宽客群/调整关键词/新建模型) |
| 台账字段为空 | 该维度按"不限"匹配,推荐时提示信息缺失 |
| 报告 JSON 缺失(`模型报告路径` 非空但 `.json` 不存在) | 提示"报告 JSON 缺失,可联系负责人",继续推荐流程 |
| `session_dir` 不存在 | `fetch_eval_sample.py` 停止执行,提示检查 `--session-dir` |
| 用户未提供切分区间 | 降级为整体评估(单份 `{model_id}_eval.{json,md,xlsx}`) |
| 两表关联键/日期分区字段不同名 | 停止并提示建视图别名后重试 |
| OOT AUC 较 Train 跌幅 > 0.03 | 不阻断,评估摘要中显著提示模型衰减风险 |

---

> 关联:SKILL.md 第 6/7 节;模型台账字段、工具清单详见 `references/model-catalog-spec.md`。
