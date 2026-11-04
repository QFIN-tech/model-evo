# 经验库合并协议（合并模型必读）

本文是写**合并决议**的唯一依据。
合并没有人工审核：程序只校验结构与保护规则，**结论质量全靠你**。所有变更写入更新日志并记录你的模型名，可按快照撤销。

## 1. 流程

```bash
M=scripts/modelevo.py
python $M library snapshot --run <run_dir> --model <你的模型名>           # 收尾时:提交本 run 快照
python $M library merge-pending                                         # 列出待合并快照
python $M library merge-context --snapshot <sid> --out <dir>/merge-context.json --model <模型名>
#   → 读上下文,写 <dir>/merge-decision.json
python $M library merge-apply --decision <dir>/merge-decision.json --model <模型名> --dry-run   # 先校验
python $M library merge-apply --decision <dir>/merge-decision.json --model <模型名>
#   → status=rejected 时按 errors 逐条修正后重试;确实无法完成:
python $M library merge-abort --snapshot <sid> --lease-token <token> --reason "<原因>" --model <模型名>
```

- `--model` 填**执行本操作的 LLM 模型名**（如 `claude-opus-5-5`、`deepseek-flash`），不是 trial id / run id；写错时程序忽略该值并告警（回落到环境变量 `MODELEVO_MODEL`），更新日志里就查不到是哪个模型写的。
- `merge-context` 会申请**合并租约**（30 分钟）。同一命名空间同时只能有一个合并；租约过期后 `merge-apply` 会被拒，需要重新取上下文。
  候选很多、预计超过 20 分钟时，在租约过期前续约：`python $M library merge-renew --lease-token <token> --model <模型名>`（每次再延 30 分钟）。
- `merge-apply` 的回执里有 `review_flags`（不阻塞）：`weak_claim`（结论含“可能 / 建议尝试”等弱表述）、`direction_conflict`（同任务、同口径、同主主题下与另一条原则方向相反且未登记 contest）。出现 flag 时应在本次或下次合并里处理：条件不同就 `split_by_scope` 或补 `condition`，结论真冲突就 `contest`。
- 上下文字段：`discoveries`（本快照的候选原则，各带 `related`：库内相近原则 id）、`principles`（这些相近原则的完整内容）、`topics`（当前词表）、`dossiers`（本任务与本数据底座的现有卷宗）、`dossier_scopes`、`lease.token`。

## 2. 决议格式

```json
{
  "schema": "modelevo.merge-decision/1",
  "snapshot_id": "<上下文 snapshot.snapshot_id>",
  "lease_token": "<上下文 lease.token>",
  "topics_added": [
    {"code": "feature.seq.window_count", "definition": "事件 × 时间窗的计数 / 占比网格",
     "aliases": ["seqp_cnt_", "seq_win"]}
  ],
  "topics_aliased": [{"code": "feature.seq.count", "into": "feature.seq.window_count", "reason": "同义"}],
  "actions": [
    {"local_id": "exp-202609-37c98dd2", "action": "add", "reason": "…",
     "principle": {
       "claim": "单独加入事件×窗口计数网格可带来约 +0.004 AUC;叠加在已含漏斗/近因特征的联合面上无独立增量",
       "direction": 0,
       "topics": ["feature.seq.window_count"],
       "objects": ["seqp_cnt_"],
       "scope": {"model_type": "binary_classification", "primary_metric": "auc", "target_transform": "raw"},
       "condition": {"data_modalities": ["event_sequence"], "business_domains": ["credit.drawdown"]},
       "effect": {"magnitude_range": [0.0, 0.0039], "unit": "auc_delta", "n_replications": 2},
       "evidence": {"task_ids": ["req-t7-rand400w"], "trial_ids": ["trial-011"], "summary": "…"},
       "limitations": ["只在 200w_1000w-bc5329dc 底座验证"],
       "confidence": "medium"}},
    {"local_id": "exp-202609-0801", "action": "merge_into", "target": "p-202610-ab12cd34", "reason": "…",
     "principle": {"…": "目标原则合并后的完整新版本"}},
    {"local_id": "…", "action": "drop", "reason": "harness 缺陷记录,已修复,不影响建模决策"}
  ],
  "dossiers": [
    {"scope_type": "task", "scope_id": "<dossier_scopes.task>",
     "sections": {"ceiling": "…", "effective": "…", "falsified": "…", "open_questions": "…", "data_notes": "…"}}
  ]
}
```

**每条 discovery 必须恰好对应一个 action**，每个 action 都要写 `reason`。

## 3. 动作怎么选

| 动作 | 何时用 | 必填 |
|---|---|---|
| `add` | 库里没有同主题、同口径的结论 | `principle` |
| `merge_into` | 与 `target` 说的是同一件事；给出合并后的**完整新版本**（证据合并、效应区间更新） | `target`、`principle`，口径必须相同 |
| `split_by_scope` | 结论在不同口径（raw / log1p、auc / rmse）下不同，拆成多条 | `principles`（≥2，口径两两不同）；可选 `target`（被拆分的旧原则，置为 superseded） |
| `supersede` | 新证据推翻 `target`，旧原则失效 | `target`、`principle`，口径相同；见 §6 保护 |
| `contest` | 与 `target` 结论相反但证据不足以推翻，两条并存并互相标注 | `target`、`principle` |
| `drop_duplicate` | 与已有原则重复，没有新信息 | `target` |
| `drop` | 不是可迁移的建模经验：一次性流水账、已修复的 harness 缺陷、纯操作记录 | `reason` |

判断“同一件事”：主题相同 + 口径相同 + 结论方向和适用条件一致。只是字面相近不算。

## 4. 原则字段

- `claim`：一句可证伪的结论，写清**条件与量级**；不要写“建议尝试”“可能有效”。
- `direction`：`1` 建议做；`-1` 避免（有害）；`0` 无增益 / 边际递减。不确定时宁可 `0`。
- `topics`：1–3 个词表中的 code（别名也可，会自动规范化）。第一个主题决定 `category`。
- `objects`：具体对象（列前缀、信息源，如 `seqp_cnt_`、`offline_inner_`），只在同一数据底座内用于匹配。抽象结论不填。
- `scope`（口径，**必填**）：`model_type`（binary_classification / regression / uplift）、`primary_metric`（auc / rmse …）；回归必须写 `target_transform`（raw / log1p …），分类缺省 raw。
  与指标无关的评估 / 工程类原则（CV 协议、OOM、泄漏对账…）`primary_metric` 写 `"*"`；检索时评估类只要求任务类型一致，工程类不要求。
- `condition`：`data_modalities`、`business_domains` 等适用条件。`business_domains` 用业务域词表（`runtime/library/business_domains.yaml`，`library domains` 可查）的 code，写别名会被自动归一；词表里没有的写法原样保留，但不会与其它业务的同义写法匹配。
- `effect` / `evidence` / `limitations`：保留原条目里的数字与 trial id；`limitations` 写清只在哪些底座 / 口径上验证过。
- `confidence`：可写 high / medium / low；置信度由证据计算，此处只作参考。
- `evidence.abstract_task_ids` **不用写**：合并时服务端自动把本次快照的任务指纹并进去（同任务权重、
  检索首档、避免清单都依赖它）。手写也不会被覆盖，只会去重追加。

## 5. 主题词

- 顶层固定：`feature` / `model` / `hpo` / `eval` / `engineering`，最多 3 层，小写点分（如 `hpo.n_estimators_cap`）。
- 优先复用已有词与别名。确需新增时写进 `topics_added`：父节点必须已存在（或同一决议里先新增父节点），`definition` 必填，新词至少被一条原则使用。
- 与已有兄弟词过于相近会被拒；把常见写法（列前缀、旧 target 值）放进 `aliases`。
- 发现两个词其实同义时用 `topics_aliased` 把旧词并入新词（旧词保留为别名）。
- 主题描述的是**做了哪类事**（如“事件×窗口计数”），不是具体列名；列名放 `objects`。

## 6. 保护规则（程序强制）

- 把 `direction` 为 0 或 -1 的原则推翻成 +1（`supersede`，或 `merge_into` 改了方向）时，必须在该 action 里给 `evidence_trials: [{"run_id": "...", "trial_id": "..."}]`，且至少一个 trial 与原则同指标、增益 CI 下界 > 0。达不到就用 `contest`。
- 口径不同不得 `merge_into` / `supersede`。
- 原则只追加新版本，不删除；`drop` 只丢弃本快照里的候选，不影响库内原则。

## 7. 卷宗

- 每次合并更新 `dossier_scopes` 里的任务卷宗与数据底座卷宗（没有对应 scope 时省略）。在现有卷宗基础上**改写**，不是追加日志。
- 五节固定：`ceiling`（当前天花板与冠军配方）、`effective`（已证实有效的方向）、`falsified`（已证伪 / 无增益方向）、`open_questions`（未决问题）、`data_notes`（泄漏、口径、数据注意事项）。
- 合计 ≤ 3000 字符；写结论与数字，不写过程。数据底座卷宗只写跨任务成立的内容（同一份数据上的特征结论、泄漏列）。
