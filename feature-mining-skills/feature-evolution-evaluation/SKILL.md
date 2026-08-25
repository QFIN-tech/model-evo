---
name: feature-evolution-evaluation
description: 特征进化确定性评估核心(G1~G6关卡/前置诊断/原料矩阵/多seed快筛/全量终判/台账/反馈/导出)。不含任何LLM环节。当用户说"评估这些候选特征""这个特征有没有用""结算这一轮""导出接受的特征"时使用;正常流程由feature-evolution-orchestration调用。
---

# 特征进化确定性评估核心(执行层)

## 1. 定位

本 skill 是特征进化闭环中**唯一的判定权威**:前置诊断、候选安全执行、原料冗余、多seed快筛、全量增益判定、champion更新/回滚全部确定性计算在这里。LLM(编排层)只提候选,不接受 LLM 估算的任何指标。

核心能力(对应 orchestration §0):
- **Stage 0 可挖性诊断**:`probe_roi.py` 开跑前算池内冗余分布+单变量AUC,判ROI
- **原料相关性矩阵**:`probe_roi.py` 算未用特征 vs champion 的\|Spearman\|,产出 whitelist/blacklist
- **多seed Stage A**:`evo_core/gates.py::gate_g4_g5_multiseed` 用 K 个不同种子子样本+bootstrap置信区间,降噪
- **严苛全量 Stage B**:多seed存活的候选上全量帧终判(本任务规模大时启用)
- **场景化G3**:saturated场景 G3 对照=champion全列(严苛),cold_start 对照=本轮已接受(宽松)
- **champion残差反推**:从预测错样本看未用特征异常,产 `champion_residual.md`

评估口径统一(AUC/KS/IV/PSI/十分桶逐行对齐),超参固定(进化比特征增益,调参走 tuning)。

## 2. 输入依赖

| 输入 | 必选 | 来源 |
|---|:---:|---|
| 已初始化 session | ✅ | orchestration prepare_session(含 Stage0 诊断产物) |
| 候选代码+meta | ✅ | 编排层写入 rounds/rXXX/candidates/ |

## 3. 执行命令(无 LLM)

### 3.0 probe_roi.py(Stage 0)

```bash
python <eval>/scripts/probe_roi.py --session-dir <session>
```
开跑前算:池内冗余分布(\|corr\| 直方图)、低冗余+有信号原料清单、ROI 判定。落 `profile/roi_report.md` + `material_*.txt`。

### 3.0b champion_residual.py(champion 残差反推,可选)

```bash
python <eval>/scripts/champion_residual.py --session-dir <session>
```
从 champion 预测错的样本(假阴/假阳)出发,算未用特征在错分 vs 正确子集上的分布差异,产出 `profile/champion_residual.md`(候选生成时优先作交互原料)。需 baseline 预测;饱和场景强烈建议在 Stage 0 后运行。

### 3.1 evaluate_round.py(逐候选过G1~G5,含多seed)

```bash
python <eval>/scripts/evaluate_round.py --session-dir <session> --round <N> [--cids c001,c002]
```

G1过候选声明 PARAM_BOUNDS/DEFAULT_PARAMS 时,先在train档确定性搜常数固化回代码再重过G1(防 train-only 过拟合)。

**大数据量多seed**:配置 `evolution.train_sample_cap`+`evolution.n_seeds` 后,G4/G5在K个不同种子子样本上各训一次,取增益均值+置信区间。多seed均值过门槛才判pass,避免子样本单点噪声误杀/误纳。G2/G3仍全量,G6永远全量。

### 3.2 commit_round.py(G6融合+champion提交/回滚)

```bash
python <eval>/scripts/commit_round.py --session-dir <session> --round <N>
```

### 3.3 build_feedback.py(反馈生成)

```bash
python <eval>/scripts/build_feedback.py --session-dir <session>
```

汇总当前进化状态 -> `evolution/feedback/latest-feedback.md`(下一轮候选生成前,编排层 LLM 必读)。内容包括:champion 状态(baseline vs champion、累计增益、接受数、连续无接受轮数、终止原因)、最近一轮候选判定表(挂卡关卡、差多少、含增益数字)、已接受特征清单(fid + 假设 + 单变量 AUC)、探索引导(未被任何候选用过的原始特征 / 已注册语义特征)、下一轮配额与近几轮指纹黑名单(防重复提案)。退出码:0 成功 / 2 session 状态不可用。

### 3.4 export_features.py(交付包导出)

```bash
python <eval>/scripts/export_features.py --session-dir <session>
```

导出 `evolution/export/accepted-features/` 交付包,供下游训练/生产取用。包内:
- `{fid}.py` + `{fid}.meta.json`:已接受特征代码与元数据(按 fid 顺序应用,后者可见前者)
- `features.parquet`:id_cols + split + label + 全部 fid 列(champion 特征值)
- `model/model.json`(及 meta):融合模型(champion,XGB 固定超参)与特征列清单
- `README.md`:应用方式说明(输入列契约 / 应用顺序 / 截断等注意事项)
- `_manifest.json`:文件清单

退出码:0 成功 / 2 无已接受特征 / 4 运行时失败。

### 3.5 finalize_session.py(收口报告)

```bash
python <eval>/scripts/finalize_session.py --session-dir <session>
```

生成 session 总报告 `report.md` + state 置 `finalized`。报告含:总览(baseline vs champion 三档 AUC/KS、累计 OOT 增益、终止原因)、已接受特征表(fid / 假设 / 单变量 AUC / 提出轮次)、进化史(逐轮 commit/rollback/no_accept 台账汇总)、产物索引(导出包/反馈/画像位置)。退出码:0 成功 / 2 session 状态不可用。

### 3.6 脚本退出码汇总

| 退出码 | 含义 | 适用脚本 |
|:---:|---|---|
| 0 | 成功 | 全部 |
| 2 | session 状态/输入不可用 | 全部 |
| 4 | 运行时失败(已接受特征损坏属严重态) | evaluate_round / export_features |

note:`commit_round` 的 champion 三档评估落 `evolution/accepted/_champion_eval.json`(指标级;分桶明细在 finalize 报告重算)。

## 4. G1~G6 关卡口径

| 关卡 | 判定 | 缺省阈值 |
|:---:|---|---|
| G1 | AST白名单(import math/numpy/pandas/statistics/re;禁双下划线/open/eval/exec)+子进程超时+双跑一致+输出可数值化/非常数/非全NaN | 超时60s |
| G2 | 单特征质量:train方向修正AUC/非NaN覆盖率/train-vs-OOT PSI | AUC>=0.5,覆盖>=0.3,PSI<=0.3 |
| G3 | 冗余 \|Spearman\|。**场景化**:saturated对照=champion全列(cold_start对照=本轮已接受) | saturated<=0.9, cold_start<=0.95 |
| G4 | champion+候选重训,**valid**增益(多seed均值+CI) | >=0 |
| G5 | 同模型 **OOT** 增益(核心门槛) | >=0.0005 |
| G6 | 轮级融合:champion+本轮全部provisional重训,OOT须超上代champion,否则整轮回滚 | >=0.001 |

判定细节:
- G4/G5 多seed:每个候选在 K 个固定种子子样本上各训一次,增益取均值,CI 取 bootstrap。mean_gain>=门槛且 CI下界>=0 才过。救回"单seed略降但多seed均值正"的候选;拒绝"单seed侥幸正但多seed均值负"的候选
- G4/G5 平行世界对比:每个候选各自与当前 champion 比(同帧/同超参/同随机态),候选互不挤占
- G6 防贪心过拟合:单候选都过但合起来 OOT 不涨,整轮回滚(ledger 记 round_rollback)
- 重复提案:假设指纹(base_features排序+hypothesis归一化 sha1)+ **原料指纹**(base_features排序 sha1,防同族反复试)双黑名单
- 候选输入帧不含label/id/dt/既有模型分,从源头防泄漏

## 5. 输出产物

| 产物 | 说明 |
|---|---|
| `profile/roi_report.md` | Stage0可挖性诊断 |
| `profile/material_whitelist.txt` | 低冗余+有信号原料 |
| `profile/material_blacklist.txt` | 高冗余原料 |
| `profile/champion_residual.md` | champion残差反推 |
| `rounds/rXXX/results/cNNN.result.json` | 单候选判定(verdict/gate_failed/reason/各关卡+多seed明细) |
| `rounds/rXXX/round-summary.md` | 本轮判定一览 |
| `evolution/ledger.jsonl` | 全历史(append-only) |
| `evolution/state.json` | champion状态/stop_reason/场景/no_accept_streak |
| `evolution/accepted/f00N.py` | 已接受特征代码(按fid顺序,后者可见前者) |
| `evolution/feedback/latest-feedback.md` | 下一轮必读 |
| `evolution/export/accepted-features/` | 交付包 |
| `report.md` | 总报告 |

## 6. 与其他 skill 关联

| 上下游 | Skill | 关系 |
|---|---|---|
| 上游 | `feature-evolution-orchestration` | 编排层(含Stage0诊断+数据驱动生成) |
| 上游 | `semantic-feature` | 池内榨干时的池外原料 |
| 下游 | 下游训练/生产 | export交付包供取用 |

## 7. 执行约束

1. 零LLM:所有脚本不做模型推理/文本生成
2. 不篡改历史:ledger只追加;result.json写后不重写
3. 阈值不改写:覆盖只能来自 session_config 的 gates 段(留痕 state.config_snapshot)
4. champion只经G6更新:绕过 commit_round 的入库非法
5. 饱和场景必跑Stage0诊断:不开诊断直接evaluate_round是违规

## 8. 异常处理

| 条件 | 处理 |
|---|---|
| 候选执行超时/报错/不确定 | G1拒,reason落result.json |
| 候选引用不存在列 | G1拒(执行失败) |
| 某档样本单类 | 相关AUC置None,G4/G5判"不可计算"拒 |
| state.json不存在 | 退出码2:提示先跑prepare_session |
| 已接受特征代码执行失败 | 退出码4:已接受损坏属严重态,报告用户不静默跳过 |
| 多seed某seed子样本单类 | 该seed增益置None,用剩余seed均值;全None则拒 |
| 无provisional候选的轮 | 正常结算:no_accept_streak+1,进终止判定 |
