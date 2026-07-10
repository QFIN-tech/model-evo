# 模型报告存放目录

本目录用于存放每个模型的**完整评估报告**，供业务人员通过 `model-recommend` skill 推荐结果跳转查看。

## 命名规范

报告文件以 `model_id` 开头，保证与台账一一对应：

```
reports/{model_id}_{模型简称}.md
```

示例：`reports/yx_001_模型简称.md`

## 上传步骤

1. 复制 `_template_model_report.md`，按模板填写完整内容。
2. 按上述命名规范保存到本 `reports/` 目录。
3. 回到台账 `../model_catalog.csv`，在对应模型行的 **`模型报告路径`** 列填写相对路径，例如：
   `reports/yx_001_模型简称.md`
4. 填好后，skill 推荐该模型时会自动带出报告链接。

## 报告应包含(对齐项目规范)

- 核心指标：KS、AUC、PSI
- 分档分布(默认 10 档)
- 实验信息：训练样本时间窗、正负样本比、核心超参数
- PSI > 0.1 的特征需标注 [PSI_WARN]

> 模板见同目录 `_template_model_report.md`。
