# DataBar 条件格式规格

> 本文件从 `classification-model-comparison/SKILL.md` 4.1 节 row 6（`对比报告.xlsx`）抽出,包含 Sheet 2 表头合并行规则、4-group 归一化、颜色映射、delta 子表不画条等完整 DataBar 规格。SKILL.md 中保留 1 行摘要 + 指向本文件的指针。

`对比报告.xlsx` 三 Sheet：`1-指标对比`、`2-分桶并排对比`、`3-raw_data`。每个 Sheet 内 oot 和 all 上下堆叠，以 split 标签行分隔。

**Sheet 2 表头**：上方合并行（灰底蓝字）标注指标组名（label率/lift/召回率/累计召回），下方行为各模型全名。

**条件格式（DataBar）**：
- Sheet 2 主表含 DataBar，分四组独立归一化：
  - 绿色 = label率
  - 蓝色 = lift / 召回率 / 累计召回
- 起点 0、统一终点各组 max
- delta 子表不画条
- Sheet 3 无 DataBar

> 关联: `classification-model-comparison/SKILL.md` 4.1 节产物内容 row 6
