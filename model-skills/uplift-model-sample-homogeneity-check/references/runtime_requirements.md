# Runtime Requirements

This skill uses Python standard library modules plus:

- `pandas`: required for reading prepared split CSV files and computing SMD inputs.
- `numpy`: required when AUC diagnostics are executed.
- `scikit-learn`: required when AUC diagnostics are executed.
- `lightgbm`: required when AUC diagnostics are executed.

The runner does not install packages. If an import is missing, stdout/result JSON returns:

```text
status = failed
issues[].code = PYTHON_IMPORT_ERROR
outputs.runtime_requirements_path = null
outputs.runtime_requirements_reference = SKILL.md#缺依赖处理
issues[].requirements_reference = SKILL.md#缺依赖处理
```

Small samples may skip AUC because of the configured minimum treatment/control group size; SMD diagnostics still run when data and covariates are valid.
