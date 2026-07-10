# Runtime Requirements

`uplift-model-feature-quality-analysis` 的 runner 不自动安装依赖。

必需包：

- `pandas`：读取 Gate 1 建模样本 CSV，并执行表格统计。
- `numpy`：用于 PSI 分箱和数值计算辅助。

缺少依赖时，runner 会返回：

- `status`: `failed`
- `issues[].code`: `PYTHON_IMPORT_ERROR`
- `outputs.runtime_requirements_path`: `null`
- `outputs.runtime_requirements_reference`: `SKILL.md#缺依赖处理`
- `issues[].requirements_reference`: `SKILL.md#缺依赖处理`

处理方式：在用户确认后，在测试或运行环境中安装缺失包，再重新运行同一个 action。
