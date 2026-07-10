# Runtime Requirements

This skill uses Python standard library modules plus:

- `pandas`: required for `raw_sample_check`, `split_planning`, and `post_split_validation`.

The runner does not install packages. If an import is missing, stdout/result JSON returns:

```text
status = failed
issues[].code = PYTHON_IMPORT_ERROR
outputs.runtime_requirements_path = null
outputs.runtime_requirements_reference = SKILL.md#缺依赖处理
issues[].requirements_reference = SKILL.md#缺依赖处理
```

Use the workspace environment selected for this migration before rerunning the action.
