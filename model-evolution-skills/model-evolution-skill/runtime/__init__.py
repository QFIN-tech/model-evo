"""ModelEvo V2.12 Immutable Harness runtime package.

本包实现设计方案中的确定性部分。章节号指
核心协议实现位于本包，设计审计见
docs/MODELEVO_FRAMEWORK_AUDIT_AND_OPTIMIZATION.md:

- config      主配置装载与默认值(§21.1)
- spec        TaskSpec contract(§6)
- data        data-loader contract 与 split protocol(§6/§12.1)
- logging_utils 统一 Python 日志格式(§20)
- profile     SampleProfile 固定分析(§7)
- metrics     指标计算与统计工具(§12)
- evaluator   Noise-aware 验收机制(§12)
- trial       Trial recorder 与状态机(§10)
- generation  Generation 记录(§11)
- leaderboard Leaderboard 与 Candidate Archive(§15/§9.4)
- scheduler   资源调度、超时与失败隔离(§16)
- baseline    统一起跑线(§8)
- evolution   Archive-based 演进与停止条件(§9/§17)
- memory      run 内经验:事实与任务总结(§14)
- library     中心化经验库(v2.18.0;run 侧视图见 library_view)
- report      最终交付报告(§18)
- harness     Harness 代码 hash 与协议变更检测(§19.3)

"Agent 可以改模型,但不能改尺子":本包属于 Immutable Harness,
任务工作区(run-{task_id}/)中的代码不得修改本包行为。
"""

__version__ = "2.20.0"  # v2.20.0 = warm-start 确认资格修正 + 先验豁免:root_baseline_only 不再豁免 warm-start(旧序让 C3 预注册 BH 形同虚设)、先验豁免通道(z = Δ_vs_baseline/sqrt(SE²+σ_seed²) ≥ z_required 且 σ_seed 取实测 ⇒ delivery_status=confirmation_waived_by_prior,正常交付位 + final_test 照常打分),见 docs/MODELEVO_V2_20_0_CONFIRM_20261102.md
# v2.19.1 = seed 方差口径 + 收尾整合:σ_seed 规模先验 σ_seed=clip(0.5·(n_val/5万)^0.4,0.5,1.5)·SE(V2_19_1_RUN_CONFIG,仅新 run)、实测值不再被 1×SE 抬高、seed_variance_factor 1→2、经验库按 data_base 复用历史 σ_seed/SE 比值、停止时收尾整合通道(family=1 单次检验,只授予 confirm 资格)、shadow 分数落盘即带样本键,见 docs/MODELEVO_V2_19_0_DEV_20261101.md §10
# v2.19.0 = 证明体系 + 新 run 裁决口径小修 + 数据驱动计划 + 效率:ceiling 天花板估计(runtime/reference.py、runtime/ceiling.py)、benchmark 套件、锚点取现任冠军 / 切片门 √k / SE 计入 σ_seed(V2_19_0_RUN_CONFIG,仅新 run)、平台期与天花板停止信号、画像规则候选与经验生成候选、GPU 感知并发 / HPO 跨卡限时 / confirm seed 分卡 / finalize 复用复现、补丁式写码 new-trial --patch,见 docs/MODELEVO_V2_19_0_DEV_20261101.md
# v2.18.0 = v2.18.0 = 中心化经验库:跨任务经验迁到经验库服务(runtime/library/、runtime/library_view.py),run 固定水位读视图、--basis 依据必填、快照 + LLM 合并;删除 skill 内置 memory/ 与 GlobalExperienceStore,见 docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md
# v2.17.4 = profile 加速批次:分片并行读取(runtime/loaders.py)/ write_splits split 级并发写 / 写盘同步 sha256 / loader 可直接 import runtime,见 docs/MODELEVO_V2_17_4_PROFILE_SPEED_20261026.md
# v2.17.3 = req-t7 drawdown run4 性能批次:搜索代可跳过 shadow / 早停集降采样 / 置换 repeat 5→2 / 特征块缓存跨 run 共享 / GPU 准入门槛 / 交付白名单读面求交,见 docs/MODELEVO_V2_17_3_PERF_20261025.md
# v2.17.2 = run4 req-t7-400w(opus5.5)复盘批次:lr-shrink 重训自适应 + cap_confounded / 冠军撞 cap 提名 lr 修正 / 证伪族与 avoid 经验过滤 / run-generation 不改写在跑 trial / 特征块池 OOM 判定 / confirm seed 并行 / 停止态清理 / holdout 标签守卫 / token 漂移,见 docs/MODELEVO_V2_17_2_FIXES_20261024.md(v2.17.1 = v2.16.3 批次,见 docs/MODELEVO_V2_16_3_FIXES_20261023.md)


def version_at_least(version: "str | None", minimum: str) -> bool:
    """``version >= minimum``(点分数字比较);version 为空(旧 run 未记
    skill_version)视为不满足。用于"只对新 run 启用"的行为门。"""
    def _t(v: str) -> tuple:
        out = []
        for part in str(v).split("."):
            num = "".join(ch for ch in part if ch.isdigit())
            out.append(int(num) if num else 0)
        return tuple(out)
    if not version:
        return False
    return _t(version) >= _t(minimum)

# Harness 协议版本:split/evaluator/recorder/leaderboard 行为变更时递增;
# 递增即开新 epoch,历史 Trial 不再与新协议混排(§19.3)。改动 runtime/ 的
# 登记流程与判定路径见 docs/MAINTENANCE.md §2——不要手改 run-state.json。
#
# v2.18.0(中心化经验库,见 docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md):
# **协议号保持 24,epoch 不变**。split/metrics/evaluator/leaderboard 与裁决数字零改动;
# runtime/ 改动全部在经验通道:新增 library/(存储、证据、检索)与 library_view.py,
# memory.py 删除全局库部分,profile/baseline/evolution/planner/planner_context 改读
# 本 run 固定的经验库视图。v2.18.0 前 init 的未完成 run 不迁移(diagnose 报
# run_predates_experience_library),用切换前的安装完成或新开 run。
#
# v2.17.1(= "v2.16.3 批次",run4 regression loan_amt_t7 第三批;基线在开发
# 期间被 v2.17.0 合并推进,故以 2.17.1 发布,见
# docs/MODELEVO_V2_16_3_FIXES_20261023.md):**协议号保持 24,epoch 不变**。搜索
# trial 的裁决数字(validation/shadow 主指标、Δ、CI/SE、slice/temporal、verdict
# 判据)与 split/metrics/evaluator/leaderboard 零改动。runtime/ 改动:
#   - memory.py:经验条目类型归一(context/condition 非 dict、limitations 非
#     list、evidence 非 dict;读侧 all() 与写入口 _adapt_shorthand/_fill_run_context
#     同源,context 统一为 v2.17.0 的 {"note": 原文} + context_text),
#     TaskExperience.record_final_recipe;
#   - profile.py:泄漏经验对账对非 dict 字段降级取文本(只会把机械封禁降档为
#     ambiguous,不放行任何列);
#   - hpo.py:显式 lr_mode=shrink 不再静默回落 search(搜索机制,创建期/运行期
#     留痕);撞 cap 单一语义 cap_bound_of / unified_cap_bound(遥测,不进裁决);
#   - runner.py:evaluation.complexity.cap_bound 按单一语义重算、trial 记录写
#     hpo.lr_mode_effective(均为遥测);
#   - baseline.py:模板 es_info.cap_bound 改单一语义(遥测);
#   - segment_report.py:回归分桶 Bias 改为 预测−真实(只影响新生成的报表);
#   - report.py:「模型局限」的撞 cap 一条改走单一语义(报告呈现)。
# 在途 run 按 MAINTENANCE §2 路径 2 登记:
#   acknowledge-harness-change --measurement-no-score-impact
#
# v2.17.0(run4 regression.2 优化计划全量修复,见
# docs/MODELEVO_V2_17_0_FIXES_20261023.md):**协议号 23 → 24**,但**在途 run 的
# epoch 不变**。改变裁决口径的四项(锚定晋级 promotion_reference=anchor、
# crossover IUT、量纲感知 MDE、alpha 排程)全部按 run 配置门控:代码默认与
# global-config 保持旧口径,只有 v2.17.0 init 的 run 由 cmd_init 写入新口径
# (runtime/protocol_gates.V2_17_0_RUN_CONFIG)。协议号 24 标记"新 run 的晋级尺"。
# 对在途 run(配置无新键)搜索 trial 的落盘数字零改动;新增的只是诊断字段
# (parent_verdict / regression_diagnostics / anchor 三列仅 anchor 模式)、方向
# 台账的方向感知增益、GI 回归分层抽样、资源准入与缓存键。在途 run 登记:
#   acknowledge-harness-change --measurement-no-score-impact \
#     --verify-against <最近若干 trial>      # v2.17.0:逐项复评证明分数不变
# 特征块缓存 CACHE_SCHEMA_VERSION=2:旧格式条目首次使用时重算一次。
#
# v2.16.2(run4 regression loan_amt_t7 修复批次,见
# docs/MODELEVO_V2_16_2_FIXES_20261023.md):**epoch 保持 23**。搜索 trial 的
# 落盘数字(validation/shadow 主指标、Δ、CI/SE、slice/temporal、verdict 判据)
# 与 split/metrics/leaderboard 零改动。runtime/ 改动:
#   - evaluator.py:回归任务的 multi-seed 变体指标改用变体自己的预测(旧实现
#     恒等于主 seed → σ_seed 退化为 floor、multi_seed_std=0;分类/uplift 不变);
#   - runner.py:group_importance 子样本把抽样行位置交给 GI、confirm 的
#     multi-seed 证据按代码实跑数对账、FAILED 重入时归档旧 result;
#   - group_importance.py:trial 自存评估帧按同一行位置取子集(加宽族不再恒
#     unverifiable;族生死是参考信号,不进 verdict/leaderboard/alpha);
#   - templates.py / baseline.py:leak_scan 单源化并对回归改用 |Spearman|>0.95
#     (哨兵此前对回归从未运行;只会把真泄漏判 invalid,不改其余 trial 分数);
#   - hpo.py:边界外扩识别 hurdle 分段键 clf__/reg__(搜索机制,不产出裁决数字)。
# 在途 run 按 MAINTENANCE §2 路径 2 登记(与 v2.13.3/v2.14.0 同类:修复
# "结构性退化/哨兵失效",不是改尺度):
#   acknowledge-harness-change --measurement-no-score-impact
#
# v2.16.0(早停改在 outer validation、100% train 只训一次,见
# docs/MODELEVO_V2_16_0_VALIDATION_ES_20261022.md):**epoch 23**。trial 的
# outer 拟合(模板 _fit_with_eval / HPO 重训 / confirm 与 seed 变体 / baseline)
# 默认在 validation 上早停,validation 从"只评分"变为"早停面 + 评分面",
# 落盘的 validation 分数口径改变(略偏乐观;同 run 配对 Δ 大部分抵消),
# 与旧 epoch 的 trial 不可混排。在途 run 用
#   acknowledge-harness-change --allow-measurement-impact
# 登记(epoch +1,leaderboard 只保留新 epoch),并在新 epoch 重跑锚点。
# 回退:run 配置 training.early_stop_source: train_tail(v2.15 两段式)。
# 交付:validation 早停冠军自动 refit_mode=frozen_rounds。跨 run 经验/榜单
# 中 v2.16 前后的绝对 AUC 不宜直接比较。
#
# v2.16.0 同批:删除收尾期 `ensemble-final`(跨 trial rank-mean 融合对照)及
# 停止报告里的 ensemble advisory 字段。该条件自 v19 起已不进停止硬门,停止
# 判定与计分代码零改动;它建议的融合交付没有重训/final_test/交付通道,且在
# 选择面(validation)上自评。在途 run 用
#   acknowledge-harness-change --measurement-no-score-impact
# 登记。
#
# v2.16.0 同批:模型拓扑门(runtime/topology.py)。用户裁决 2026-10-22:trial
# 默认单模型,级联(子分作特征)与融合须用户经 approve-topology 人工批准;
# hurdle(回归)/ T-learner(uplift)按任务类型白名单放行。门在 runner 执行前
# 裁决"是否执行",不改任何落盘数字(evaluator/metrics/split/leaderboard
# 零改动);spec.ensemble_enabled 不再授权。在途 run 用
#   acknowledge-harness-change --measurement-no-score-impact
# 登记;存量 trial 不追溯,但 finalize-retrain 对未批准的级联/融合冠军拒绝
# 交付。
#
# v2.15.10(数据底座指纹 + 同数据跨任务检索,见
# docs/MODELEVO_V2_15_10_DATA_BASE_20261022.md):epoch 保持 22。搜索期
# evaluator/metrics/split/leaderboard 计分零改动;改动在 profile 的
# raw_columns_hash(剔除 spec data.derived_columns,未声明时逐位不变)、
# 经验检索(同数据底座 Layer 2b、feature_engineering 跨任务类型放行)、
# spec 校验与导出的 category 留痕。在跑 run 用
# acknowledge-harness-change --measurement-no-score-impact 登记。
#
# v2.15.9(交付重训复跑配方,见 docs/MODELEVO_V2_15_9_DELIVERY_RERUN_20261021.md):
# epoch 保持 22。搜索期 evaluator / metrics / split / leaderboard 计分代码零改动;
# 改动集中在交付链(scripts/_mvo/cmd_deliver.py:refit_mode=rerun_recipe、
# 特征面一致性校验、shadow 预检、按任务类型打分、evaluate-final 的 shadow
# 三方对照与原值分桶)与 report.py 的呈现。在跑 run 用
# acknowledge-harness-change --measurement-no-score-impact 登记。
#
# v2.15.8(经验闭环 + 在途收口,见 docs/MODELEVO_V2_15_8_MEMORY_LOOP_FIX_20261013.md):
# epoch 保持 22。evaluator / metrics / split / leaderboard 的计分代码零改动;
# 改动集中在全局经验库(指纹单源化、分桶检索、回填语义、加锁原子写)、
# 停止判定接线(ROI 停止、inventory has_secondary)、调度(熔断代级失败率、
# wall 余量)与遥测(baseline cap_bound)。在跑 run 用
# acknowledge-harness-change --measurement-no-score-impact 登记。
# v2.15.3~2.15.7 为同批在途改动(全局榜单/固化 baseline/两阶段 gen-00),
# 随本版一并收口。
#
# v2.15.0(HPO 全链路修复,req-t7-rand400w run3 三轮 HPO 复盘回写,见
# docs/MODELEVO_V2_15_0_HPO_20261003.md):epoch 保持 22。evaluator 的落盘
# 数字(AUC/delta/CI/SE/slice/temporal、verdict)算法逐位不变;改动集中在
# trial 侧搜索/拟合机制(runtime/hpo.py 搜索引擎、模板 _fit_with_es 两段式
# 早停遥测与 cap 外扩)、创建期两道门(HPO 边际递减 / confirm 时机)、
# planner 的 HPO 质量感知与方向台账键归一,以及 group_importance_policy
# 新增 auto(只改模型的 trial 继承父代族级置换 —— 族生死参考信号,不进
# verdict/leaderboard/alpha)与 complexity 的 cap 撞顶留痕。在途 run 按
# MAINTENANCE §2 路径 2 登记:
#   acknowledge-harness-change --measurement-no-score-impact
#
# v2.14.0(确认门 + 交付位绑定,req-t7-rand400w run2 事故回写,见
# docs/MODELEVO_CONFIRM_GATE_DELIVERY_FIX_20260930.md):epoch 保持 22。
# 落盘数字(AUC/delta/CI/SE/slice/temporal、搜索 trial 的 verdict)逐位不变;
# 改的是 confirm 的判定(双侧等价 |Δ|≤2·SE → 非劣+保真,margin 含 seed
# 方差;失败记 unconfirmed)与交付位选型(必须绑定搜索冠军配方)。与 v2.13.3
# 同类 —— 修复"结构性不可通过/静默错选",不是改尺度。在途 run 按路径 2
# 登记:acknowledge-harness-change --measurement-no-score-impact。已落盘的
# confirm verdict 保持原样(fresh_confirmation 无 criterion 字段 = 旧判据),
# 但交付位绑定对它们同样生效(陈旧确认不再被静默交付)。
# v2.13(epoch 22):跨代 alpha spending 改动态均摊、HPO 早停机制与 baseline
# 统一、capacity 闸门分期、planner 去重/证伪/量纲、经验回填——裁决与搜索
# 口径均变,旧 run 的 accepted/verdict 不与本 epoch 混排。
# v2.13.5(提速,req-t7-rand400w 23.6h 耗时拆解回写):epoch 保持 22。
# 落盘数字:bootstrap 快路径与 sklearn 逐次口径同一 RNG 抽样序、同一
# 统计量(|Δ|<1e-12,实测 delta/se/p_positive/CI 逐项相等);分组置换
# 默认改在 20 万行分层子样本 + CPU 预测上做 —— 族 delta 数值会变(族生死
# 的参考信号,不进 verdict/leaderboard/alpha),AUC/delta/CI/slice/temporal
# 不变。在途 run 按 MAINTENANCE §2 路径 2 登记:
#   acknowledge-harness-change --measurement-no-score-impact
# 新增 runtime/feature_cache.py(trial 可 import 的特征块缓存,不产出裁决数字)。
#
# v2.13.3:epoch 保持 22 —— 三处改动(hpo 断点续跑/smoke 预算闸、
# evaluator 的 confirm slice 豁免、daemon 会话解耦)均不改落盘数字
# (auc/delta/CI/slice/temporal 数值逐位不变),按 MAINTENANCE §2 路径 2
# 处理:在途 run 加 --measurement-no-score-impact 登记,历史 trial 仍同表
# 可比。唯一的语义变化是 confirm 的 verdict 不再被搜索期护栏短路——那是
# 修复"结构性不可通过",不是改尺度。
#
# v22.1(泄漏分类误判修复):epoch 保持 22 —— 改动只影响 phase-1 的**提案
# 分类**(规则 3 的实体数对齐 + 文本形态判据、reinstated_banned 撤销通道、
# 机械提案与全局经验库对账),不触及 split/evaluator/metrics 的落盘数字:
# AUC/delta/CI/slice/temporal 的算法逐位不变。分类结果变化会改变
# leak_manifest_hash —— 那会触发既有的协议校验报错,正是设计意图(数据面
# 变了,旧 trial 不可复现),不是"尺子"变更。
# v2.19.0:协议号 24 → 25,标记"新 run 的晋级尺"(锚点 incumbent、切片门 √k、
# SE 计入 σ_seed、平台期停止),全部经 runtime.protocol_gates.V2_19_0_RUN_CONFIG
# 只写入新 init 的 run;代码 DEFAULTS 与 global-config 保持旧口径,在途 run 的
# 搜索 trial 落盘数字零改动(登记路径同 v2.17.0:acknowledge-harness-change
# --measurement-no-score-impact --verify-against <最近若干 trial>)。
# v2.19.1:协议号 25 → 26,标记"新 run 的 seed 方差口径与收尾整合"
# (σ_seed 规模先验 + 实测不再抬升 + factor 2 + 停止时收尾整合通道),同样经
# runtime.protocol_gates.V2_19_1_RUN_CONFIG 只写入新 init 的 run。
# v2.20.0:协议号 26 → 27,标记"warm-start 确认资格修正 + 先验豁免"
# (root_baseline_only 不再豁免 warm-start;先验豁免经
# runtime.protocol_gates.V2_20_0_RUN_CONFIG 只写入新 init 的 run,
# 资格修正对 2.20.0 起的 run 生效)。
HARNESS_PROTOCOL_VERSION = 27
