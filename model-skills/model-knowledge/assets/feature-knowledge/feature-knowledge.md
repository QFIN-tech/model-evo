# 特征知识库

登记各业务域可复用的特征宽表与特征清单，供 feature-matching / feature-analysis 选特征时检索。特征清单 csv 落 `feature-list/` 目录，新增特征表时在下表追加一行。

## 常用特征列表

| 分场景(sub domain) | 触发方式（trigger） | 特征表（feature table) | 可用特征清单(feature list) |
|------|---------|---------|-------|
| 用户运营 | 用户运营/经营相关的建模任务时 |  tmp_db.dm_model_feature | feature-list/feature-list-user-operation-v1.csv |
| 广告投放 | 广告投放相关建模任务时 | user_grouth.user_features_v1 | feature-list/feature-list-user-grouth_v1.csv |
