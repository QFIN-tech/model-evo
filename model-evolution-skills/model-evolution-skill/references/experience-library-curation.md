# 经验库异步整理协议

整理与 run 解耦：不在任何 run 的收尾里做，由整理 worker（`scripts/curate_worker.py`）或人在终端
逐组执行。合并（`experience-library-merge.md`）处理"新快照里的候选原则"，整理处理"库里已经有的
原则"：去重、归纳跨任务共性、证据变化后的修订与退役、争议重评。

## 流程

```bash
M=python scripts/modelevo.py
$M library curate-plan                         # 哪些主题组需要整理（程序判定，不用 LLM）
$M library curate-next --out ctx.json --model <模型>   # 打开下一组并导出上下文（申请租约）
#   …… 按下文写 decision.json ……
$M library curate-apply --decision decision.json --dry-run --model <模型>   # 只校验
$M library curate-apply --decision decision.json --model <模型>             # 落库
#   无法完成:$M library merge-abort --snapshot <id> --lease-token <token> --reason "…"
#   事后撤销:$M library revert --snapshot <id> --reason "…"
```

`curate-next` 已经持有命名空间合并租约（30 分钟，`library merge-renew` 可续），期间 run 收尾的合并
会等待，所以一组要尽快完成；做不完就 `merge-abort`，不要占着租约。

## 上下文里有什么

- `group`：主题组（组内原则的主主题相同）；`triggers`：程序判定的整理理由：
  | kind | 含义 | 通常的处理 |
  |---|---|---|
  | `evidence_conflict` | 证据置信度为 mixed / contradicted | 看 `decisive_evidence`：结论带条件而被动匹配不知道 → `revise` 补 `condition` / `limitations`；确被推翻 → `retire`（refuted）或 `contest_with` |
  | `generalizable` | 同方向 ≥2 条、来自 ≥2 个任务 | 共性成立就提 `generalized`；只是碰巧同主题就不提 |
  | `near_duplicate` | 同口径同方向、措辞相近 | 说的是同一件事 → `merge_into` + 目标 `revise` |
  | `contest_review` | 争议对双方都积累了证据 | 证据能分出条件 → 两边各自 `revise` 写清条件；分不出 → 保持 |
  | `unused` | Agent 检索中展示 ≥5 次、从未被引用 | 结论太泛 / 不可操作 → `revise` 写成可执行的；过时 → `retire`（obsolete） |
  | `large_group` | 组内 ≥6 条 | 检查重复与可归纳 |
- `principles[]`：每条原则当前版本（`principle`）、所属任务（`tasks`）、证据统计（`stats`：支持 / 反驳
  权重、后验、置信度）、检索反馈（`feedback`）、最有分量的判定性证据（`decisive_evidence`：哪个 run、
  哪个 trial、增益与 CI、trial 的假设）。
- `topics`：主题词表（`revise` / `generalized` 的 topics 必须在表内；整理不能新增主题词）。

## 决议格式

```json
{"schema": "modelevo.curation-decision/1",
 "snapshot_id": "<上下文里的 snapshot_id>",
 "lease_token": "<上下文 lease.token>",
 "actions": [
   {"principle_id": "p-a", "action": "keep", "reason": "证据一致,无需改动"},
   {"principle_id": "p-b", "action": "revise", "reason": "反驳都来自联合面 trial;补适用条件",
    "principle": {"claim": "…", "direction": 0, "topics": ["feature.seq.window_count"],
                  "objects": ["seqp_cnt_"], "scope": {"model_type": "binary_classification",
                  "primary_metric": "auc", "target_transform": "raw"},
                  "condition": {"face": "已含漏斗/近因特征的联合面"},
                  "limitations": ["单独加入时有 +0.004"]},
    "contest_with": []},
   {"principle_id": "p-c", "action": "merge_into", "target": "p-b", "reason": "与 p-b 同一结论"},
   {"principle_id": "p-d", "action": "retire", "retire_kind": "obsolete",
    "reason": "两段式训练已在 v2.16 删除,结论不再适用"}],
 "generalized": [
   {"derived_from": ["p-e", "p-f"], "reason": "二分类与回归两个任务上同面 HPO 收益都随轮次递减",
    "principle": {"claim": "…", "direction": 0, "topics": ["hpo.search"],
                  "scope": {"model_type": "binary_classification", "primary_metric": "*",
                            "target_transform": "*"}}}]}
```

规则（程序校验，不满足即拒绝）：

1. 本组每条原则恰好一个动作，`reason` 必填。原则在打开整理后被别处修改过时整份决议拒绝，重新 `curate-next`。
2. `revise`：给出完整的新版本（同合并协议的 principle 字段）。**口径不得变**；方向由避免 / 无增益改为
   建议时须附 `evidence_trials`（同口径、增益 CI 下界 > 0），否则请用 `contest_with`。
3. `merge_into`：目标必须是本组另一条**同口径、同方向**的原则，且目标的动作必须是 `revise`（写出合并
   后的结论，涵盖被并入原则的数字与条件）。被并入的原则会被 supersede；它的任务指纹由服务端并到目标上。
4. `retire`：`retire_kind=obsolete`（实现、口径或数据已变，不再适用）只需理由；`refuted`（被证据推翻）
   要求证据后验 < 0.4 且证据量 ≥ 1。证据不足但有怀疑时用 `keep` + `contest_with`。
5. `contest_with`：只能附在 `keep` / `revise` 上，对象是另一条现行原则（不能是本决议里下线的）。
6. `generalized`：`derived_from` 是本组 ≥2 条原则，来自 ≥2 个不同任务，方向与上位原则相同，且上位原则的
   口径覆盖每条子原则（跨指标 / 跨变换写 `"*"`）。子原则保留（它们带着具体数字与条件）；不要把同一任务
   里的重复写成上位原则，那是 `merge_into`。任务指纹与 `derived_from` 由服务端写。
7. 业务域用 `runtime/library/business_domains.yaml` 的 code。

## 判断原则

- **宁可少改**。整理的价值在于去掉重复、补上条件、提炼共性；没有把握时 `keep`。
- 被动证据不知道原则的适用条件。看到"反驳"先读 `decisive_evidence` 里 trial 的假设：反驳它的 trial
  是不是在另一个条件下（如联合面、已饱和、不同指标）？是的话改写条件，而不是退役。
- 上位原则要比子原则**更可操作**，而不是更模糊：写清在什么情况下成立、预期幅度、反例。
- 不得编造数字；修订时保留原条目里的数字、trial id 与局限。
