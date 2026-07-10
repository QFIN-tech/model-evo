# 公司的业务领域知识

为建模流程提供必要的业务字段知识。下游 skill（classification-model-task-spec / feature-matching / feature-analysis / classification-model-recommend 等）在解析样本表、特征宽表、模型台账时，可参考本文件理解字段语义。

本文档做路由。

| 知识分类 | 触发方式 | 文档位置 | 
|---------|---------|---------|
| 公司业务公共知识 | 默认自动触发 | common-knowledge.md | 
| 用户运营相关任务 | 当任务为用户运营相关任务时触发，比如如下相关模型时：<br>动支意愿模型<br>流失预警模型<br>权益敏感度模型<br>触达敏感度模型<br>Uplift模型 | user-operation-knowledge.md | 

> `classification-model-recommend` 的模型推荐阶段由 Claude 直接读 `model-knowledge/assets/historical-model-knowledge/model_catalog.csv` 做语义筛选与排序，不再依赖额外的规则配置文件。台账字段语义见 `../historical-model-knowledge/historical-model-knowledge.md` 与 `classification-model-recommend/references/model-catalog-spec.md`。
