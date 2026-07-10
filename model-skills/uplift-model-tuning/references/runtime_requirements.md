# Runtime Requirements

`uplift-model-tuning` 的 runner 不自动安装依赖。

必需包：

- `pandas`：读取建模样本并组织 trial 评估表。
- `numpy`：数值计算、分箱和 AUUC 辅助计算。
- `joblib`：读取 baseline 模型包并保存调参 winner 模型包。
- `lightgbm`：训练 S-Learner 的 outcome 模型。
- `scikit-uplift==0.5.1`：使用 `sklift.metrics.uplift_auc_score`、`sklift.metrics.uplift_curve` 计算 AUUC 和曲线。

缺少依赖时，runner 会返回：

- `status`: `failed`
- `issues[].code`: `PYTHON_IMPORT_ERROR`
- `outputs.runtime_requirements_path`: `null`
- `outputs.runtime_requirements_reference`: `SKILL.md#缺依赖处理`
- `issues[].requirements_reference`: `SKILL.md#缺依赖处理`

处理方式：在用户确认后，在测试或运行环境中安装缺失包，再重新运行对应 action。

说明：AUUC 由 `scikit-uplift==0.5.1` 计算；runner 不自动安装或强制导入 `causalml`。
