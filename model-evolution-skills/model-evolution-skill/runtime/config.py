"""主配置装载与默认值(§21.1)。

配置来源(优先级从低到高):
1. 代码内默认值(本文件 DEFAULTS;与 assets/global-config.yaml 对齐,
   文件丢失不改变行为);
2. skill 级 assets/global-config.yaml(load 的默认 skill_config,
   显式传 None 可跳过);
3. 运行目录 run-{task_id}/modelevo-config.yaml(快照后不可变)。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from . import __version__
from .logging_utils import load_yaml

# skill 级配置默认路径(load 不传参即应用,杜绝调用点静默跳过 skill 层)
_SKILL_CONFIG = Path(__file__).resolve().parents[1] / "assets" / "global-config.yaml"

#: 全局硬上限:任何 trial 的树数量(n_estimators/iterations/etc)不得超过此值。
#  这是**铁律**,不是可调默认——run 配置、手写 hyperparameters、HPO 搜索
#  空间、lr-shrink 的 cap 外扩、交付重训,全部在写入估计器前经
#  clamp_trees() 钳制。实施背景:历史 run 出现 n_estimators 从 cap 2000
#  经 lr_shrink(×4)外扩到 32000、手写 hyperparameters 直接给 16807 的
#  情形,单 trial 训练成本失控且配方无法跨 run 复用。
MAX_TREES = 8000


def clamp_trees(value: Any) -> int:
    """把任意树数量读数钳到 [1, MAX_TREES]。

    接受 int/float/str(HPO 采样可能给 str)。非法值回退 1(而不是
    MAX_TREES)——宁可欠拟合也不要静默跑到上限把 trial 拖超时。
    """
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return 1
    return max(1, min(n, MAX_TREES))


DEFAULTS: dict[str, Any] = {
    "version": __version__,
    "baseline": {
        # v2.17.0:配置即事实。v2.15 起 cmd_baseline 在内存里把
        # [lightgbm, xgboost] 静默收成 [xgboost](lightgbm 只多花 20~25 min),
        # 配置写一套、实际跑另一套。现改为默认值直接 [xgboost],收缩逻辑删除;
        # 需要 lightgbm 对照时在 run 配置里显式加上,baseline 会全部跑完。
        "algorithms": ["xgboost"],
        # v2.17.0:显式排除(留痕)与锚点算法(缺省:含 xgboost 时取 xgboost)
        "skip_algorithms": [],
        "anchor_algorithm": None,
        "max_trials": 2,
        # v8:learning curve / early stopping(GBM 配 eval_set)
        "n_estimators_cap": 2000,
        "early_stopping_rounds": 50,
        "early_stop_validation_fraction": 0.1,
        # v2.13.5:多算法 baseline 按调度并行(lightgbm CPU ∥ xgboost GPU);
        # 锚点 delta 在全部完成后补评估(只重评估不重训)
        "parallel": True,
        # v2.15.6:跨 run 热启动默认起多少条 warm trial。全局库 archive 保留
        # top-5 配方，gen-00 默认只尝试 rank1/rank2，避免拉高首代算力。
        "warm_rank_limit": 2,
    },
    # v2.16.0:outer 拟合的早停面。validation = 在 outer validation 上早停、
    # 100% train 只训一次(默认);train_tail = v2.15 两段式(train 尾部
    # early_stop_validation_fraction 早停 + 全量重拟合),回退用。
    # runner 以 MODELEVO_ES_SOURCE 下发给 trial;交付按冠军口径自动选
    # refit_mode(validation → frozen_rounds)。
    "training": {
        "early_stop_source": "validation",
        # v2.17.3:早停集行数上限(0 = 不降采样,旧行为)。每轮评估在整个早停
        # 集上打分;44.7 万行 × 2000+ 列的逐轮评估是单 trial wall 的可观部分。
        # 2.55M/450k 实测:full-ES 与 150k-ES 的 best_iter 804 vs 818、全量
        # 验证 AUC 一致到 6 位 —— 只影响"何时停",落盘分数仍全量口径。
        "early_stop_eval_rows": 150000,
    },
    "evolution": {
        "max_generations": 8,
        "max_trials": 80,
        "parallel_trials": "auto",
        "strategy": "archive_evolution",
        # v12(P1-9):模型/超参探索期默认配比 40/40/20(hpo 为默认动作)
        # v18(P2-2):crossover 0.2→0.05——历史全部 run 0 使用、净贡献
        # 为负(早期评审口径:0/3 成功);语义=accepted 特征族并集重训
        "explore_ratio": 0.4,
        "exploit_ratio": 0.4,
        "crossover_ratio": 0.05,
        "archive_size": 12,
        # v12(P2):特征探索最低代数 1→2(唯一护城河只给 1 个超级 trial
        # 的预算是错配:两次 run 中 78–80% 的提升来自特征方向)
        "min_feature_generations": 2,
        # v7:特征重建/聚合重做配额(data_view;与 assets/global-config.yaml 对齐)
        "heavy_trials": {
            "min_per_generation": 1,
            "max_per_generation": 2,
            "timeout_multiplier": 4,
            "runtime_hint_minutes": 30,
        },
        # v11:hpo 动作(inner-split 搜索);default_budget 为
        # --budget-configs 缺省值
        "hpo": {
            "default_budget": 40,
            "inner_fraction": 0.2,
            # v14:hpo trial 不能吃 90 分钟通用超时——40 config 的搜索
            # 必然撞墙(实测 trial-020/021 各 4 次 rc=-15 白跑数小时)
            "timeout_multiplier": 4,
            "max_timeout_minutes": 480,
            # v2.15.0 HPO 引擎(见 runtime/hpo.py 模块 docstring)
            "lr_mode": "auto",
            "search_lr": 0.05,
            "lr_shrink_factor": 0.25,
            "lr_shrink_min_rows": 200000,
            "lr_floor": 0.005,
            "auto_expand": True,
            "warm_start": True,
            "parallel_configs": "auto",
            "fidelity": "single",
            # v2.17.2(A4):抽样面(hpo.sample_rows)搜索后,把前 K 名 config
            # 放到全量 inner 拟合池上复评再选 best_params(0 = 关;opt-in,
            # 每多 1 名多一次全量 inner 拟合)
            "fidelity_recheck_top_k": 0,
            # v2.19.0:搜索墙钟上限(分钟,从搜索开始计;与 deadline 外推取更早
            # 者)。到时不再派发新 config,用已完成 config 中最优者重训;
            # 0/None = 不限。留痕 metrics.hpo_search_capped
            "max_search_minutes": 60,
            # v2.19.0:config 评估跨卡并行。auto = 本 trial 在 GPU 上、搜索前
            # 进程尚未初始化 CUDA、且经 GPU 租约租到额外干净卡时,每个并行
            # worker 一张卡(最多 max_parallel_gpus 张,含本 trial 主卡);
            # off = 旧行为(同卡 parallel_configs)。留痕 metrics.hpo_parallel_gpus
            "multi_gpu": "auto",
            "max_parallel_gpus": 4,
        },
        # v2.14.0:每个配方(confirm 父代)最多完成几次 confirm。确认门
        # 若是抽签,"失败就再抽"会把它变成必过门且只认证中签者(run2 的
        # trial-011 正是 4 次里唯一中签的一次);超额须 new-trial
        # --allow-extra-confirm "<理由>" 显式留痕。
        "max_confirm_attempts_per_recipe": 2,
    },
    # v2.17.2(D2):confirm 的 seed 变体并行。auto = 本 trial 在 GPU 上、能经
    # GPU 租约再租到无外来进程的干净卡、且内存准入(有效可用 − 本 run 未兑现
    # 预留 − 安全余量 ≥ 额外 worker 数 × 单 worker 峰值估计)满足时,主进程与
    # worker 子进程并行训 seed 变体(各写 seed_scores/seed-k.parquet,与串行
    # 同口径);任一条件不满足即串行。off = 恒串行。
    # env MODELEVO_CONFIRM_PARALLEL_SEEDS 优先于本键。
    "confirm": {
        "parallel_seeds": "auto",
        # v2.19.0(#14):Planner 提名 confirm 的时机。final_only = 只在停止
        # 建议已触发 / 剩余预算 ≤1 代 / 未交付增益 ≥ MDE 且无更多高分候选时
        # 对搜索冠军提名一次;per_generation = 旧行为(每代都可提名)。
        "policy": "final_only",
    },
    "execution": {
        "trial_timeout_minutes": 90,
        "kill_grace_period_sec": 30,
        "max_retries_per_trial": 1,
        "max_fix_rounds_per_trial": 2,
        # v14.3:单 trial 独占(plan 未传)时每 trial 线程上限。旧 8 路
        # 保守默认在 solo run 时闲置 ~7/8 CPU;scheduler.plan 的
        # auto 分支在 planned_trials=None 时按 1 路独占分配,线程数
        # 封顶本值(默认 16,与原 min(16,…) 上限一致)。
        "parallel_trials_auto_cap": 24,
        # v2.11:线程下限(高负载期 usable 被压缩时不再把 threads 压到
        # 4——实测 threads=4 的 smoke 1401s、全量进入必然超时区;
        # scheduler.plan 与 runner._downgrade_plan 同口径消费本值)。
        "threads_floor": 8,
        # v2.19.0(#13):GPU trial 按卡分线程。有可用卡且声明了本批 trial 数时,
        # 槽位 = min(可用卡数, 本批 trial 数),线程 = 可用核 / 槽位,钳到
        # [gpu_trial_threads_min, gpu_trial_threads_max];8 卡机器并发可达 6~8
        # (旧口径线程上限 24 → 并发 4~5)。CPU trial 维持旧逻辑。
        "gpu_aware_parallel": True,
        "gpu_trial_threads_min": 8,
        "gpu_trial_threads_max": 24,
        # v2.19.0:内存预估按同类 trial(同视图、共享父代链)最近 k 个实测峰值
        # 的最大值 × 安全系数(旧口径父代峰值 ×1.15 实测高估约 2 倍)
        "memory_peer_window": 3,
        "memory_peer_safety_factor": 1.25,
        # v2.17.3:GPU 准入门槛。实测同卡并发会把 GBDT 训练拖到 CPU 级速度
        # (同卡 2 进程 1.77×、4 进程 3.5×;run4 相邻 trial 单树 0.11 s →
        # 0.46 s 的 2.84× 残差正落在该区间)。require_clean_gpu=true 时
        # "找不到无外来进程的卡"直接降级 CPU,而不是与其它 run 共享算力;
        # min_free_gpu_gb 是显存硬门槛(被大模型占用的卡即便被判定为
        # busy 也可能不抢)。
        "require_clean_gpu": False,
        "min_free_gpu_gb": 20.0,
        "failure_circuit_breaker": {
            "consecutive_failures": 3,
            "generation_failure_rate": 0.5,
        },
        # v2.17.0(run4 regression.2 #11):内存准入。启动训练前要求
        # 有效可用(min(MemAvailable, cgroup 限额−不可回收用量))− 本 run
        # 未兑现预留 − 安全余量 ≥ 本 trial 预估峰值,否则排队;等满
        # admission_max_wait_min → FAILED(admission_timeout,infra 类)。
        # 峰值估计:父代 result.peak_memory_gb×1.15 → 同 lineage 最近 trial
        # ×1.15 → trial.yaml expected_resources.peak_memory_gb → 缺省
        # (memory_estimate_default_gb;数据面存在时取 min(缺省, 数据面×4))。
        # 见 scheduler.MemoryAdmission;日志 logs/admission.jsonl。
        "memory_admission": True,
        "memory_safety_margin_gb": 60,
        "memory_estimate_default_gb": 64,
        "admission_poll_sec": 30,
        "admission_max_wait_min": 120,
    },
    "evaluator": {
        "bootstrap": True,
        "bootstrap_samples": 2000,
        "segment_bins": 10,
        "segment_monotonic_threshold": 0.8,
        "shadow_test_enabled": True,
        "shadow_policy": "confirmation_only",
        # v2.17.3:哪些 search_action 不必产出 shadow 工件(默认空 = 全部产出)。
        # ⚠ 默认必须为空:confirm 硬依赖父代的 shadow_test_score 作独立复现
        # 对照,把搜索代关掉会让之后的 confirm 直接失败。确认这条链在本任务
        # 不适用(例如从不做 fresh confirm)时才在 run 配置里显式列出动作,
        # 可省 44.7 万行上的 transform 重放 + 特征块 + 全量打分
        # (req-t7 run4 冷缓存 557~686 s/trial、热缓存 ~55 s)。
        "shadow_skip_search_actions": [],
        # v8:shadow gap sanity(gap ≤ multiplier×noise;noise 缺失时相对兜底)
        "shadow_gap_multiplier": 2.0,
        "shadow_gap_relative_max": 0.15,
        "final_test_feedback": False,
        "noise_thresholds": {"promising_ratio": 0.5, "validated_ratio": 1.0},
        # v12(P0-3):冠军收缩分 metric − k·noise 的收缩系数
        "selection_shrinkage": 1.0,
        "generation_fdr_q": 0.10,
        "online_alpha_budget": 0.10,
        "minimum_meaningful_improvement": 0.001,
        # v2.17.0(run4 regression.2):以下四键在代码默认与
        # global-config 中保持**旧口径**,新 run 由 cmd_init 按
        # runtime.protocol_gates.V2_17_0_RUN_CONFIG 写入新口径(在途 run
        # 不换尺)。
        # 量纲感知 MDE 的相对系数;None = 只用绝对值(旧)。设置时
        # MDE = max(absolute, relative × |baseline 主指标|),见 runtime/mde.py
        "minimum_meaningful_improvement_relative": None,
        # 晋级对照:parent = 逐步 vs_parent 过代级校正才能晋升(旧);
        # anchor = 代级校正改对"交付锚点"(已确认冠军,无确认时为
        # baseline)计账,vs_parent 仍用于方向台账与归因
        "promotion_reference": "parent",
        # 多父代 crossover 的 vs_parent:first_parent = 沿用旧口径(由调用方
        # cmd_trial/gen_executor 选定的单个父代分数,通常是 validation 最优父代);
        # iut = 对全部父代逐一配对,取 p_better 最低(最保守)者(交并检验)
        "crossover_parent_test": "first_parent",
        # 代级 alpha 排程:uniform = 按剩余代数均摊(旧);lord = 近似
        # batch-LORD++(有发现时回补 alpha,长期无发现时按 γ 序列衰减)
        "alpha_schedule": "uniform",
        # anchor 模式护栏:vs_parent 的 p_better 须 > 该值(不得显著差于父代)
        "anchor_parent_guard_p": 0.05,
        # v2.19.0:以下四键同样保持旧口径,新 run 由
        # runtime.protocol_gates.V2_19_0_RUN_CONFIG 写入新口径。
        # 锚点取值:delivery = 交付位→最优已确认→baseline(旧);incumbent =
        # 上一代结束时同视图最优的 accepted/confirmed trial(含 warm-start)
        "anchor_mode": "delivery",
        # 切片门阈值口径:none = z·SE(旧);sqrt_k = z·√k·SE
        "slice_se_scaling": "none",
        # 配对检验 SE 是否计入 seed 方差(σ_seed 解析同 confirm:本 run 复现组
        # → 比值 / 绝对先验 → floor_ratio×SE);factor 为 σ² 的系数
        "seed_variance_in_se": False,
        "seed_variance_factor": 1.0,
        # v2.19.1:实测 σ_seed 是否仍被 floor_ratio×SE 抬升。True = 旧口径
        # (实测小于 1×SE 时抬到 1×SE);False = 实测值原样使用,floor 只用于
        # 完全没有估计的情形。新 run 由 V2_19_1_RUN_CONFIG 写入 False。
        # 比值先验(scaled/经验库)同样不参与 floor —— 否则 r<1 的下半段恒被
        # 抬到 1×SE,规模先验形同虚设。confirm 门的 floor 另由
        # confirm_seed_sd_floor_ratio 控制,不受此键影响。
        "seed_variance_floor_measured": True,
        # v2.19.1:搜索 trial 配对检验的 σ_seed floor 系数;None = 沿用
        # confirm_seed_sd_floor_ratio(旧口径)
        "seed_variance_floor_ratio": None,
        # v2.19.1:停止时的收尾整合通道 —— 被同代 BH 拦下、但相对现任冠军
        # Δ≥MDE 的最佳候选单独检验一次(family=1,q ≤ 剩余 alpha 并扣费),
        # 通过只授予 confirm 资格(verdict 不变,能否成为冠军由 confirm 裁决);
        # 每个 run 只开一次。旧口径 False(不启用)
        "closing_consolidation": False,
        # v2.19.1:σ_seed 规模先验的形状(σ_seed = clip(base·(n_val/ref_n)^
        # exp, lo, hi)·SE)。None = 用 runtime.confirmation 的常量
        "seed_variance_prior_base": None,
        "seed_variance_prior_ref_n": None,
        "seed_variance_prior_exponent": None,
        "seed_variance_prior_lo": None,
        "seed_variance_prior_hi": None,
        # v2.19.0 停止信号(建议性):冠军与天花板估计差距 ≤ MDE 即建议收尾
        # (ceiling-estimate.json 不存在时不触发)
        "ceiling_stop_enabled": True,
        # 平台期停止:最近 N 个非空代冠军增益都 < ratio×MDE。None = 关闭
        # (旧口径);新 run 由 V2_19_0_RUN_CONFIG 写入 2
        "plateau_stop_generations": None,
        "plateau_stop_ratio": 0.5,
        # v2.19.0:leak_scan 哨兵是否尊重 spec 的泄漏裁决(kept_ambiguous 列及
        # 不强于它们的派生列不再判 invalid,留痕 leakage_suspects_waived)。
        # 旧口径 False;新 run 由 V2_19_0_RUN_CONFIG 写入 True
        "leak_scan_respect_adjudication": False,
        # LORD 视界(γ 序列归一的代数);None = 取 max_generations
        "lord_horizon": None,
        # LORD 初始财富 W0 = ratio × alpha 预算
        "lord_w0_ratio": 0.5,
        # v2.14.0 确认门(runtime/confirmation.py,req-t7-rand400w run2 事故):
        # 旧判据 |Δ|≤2·SE 双侧等价、SE 不含 seed 方差 ⇒ 大样本上可复现配方
        # 单次过门率仅 35~45%,且"比父代好"也判失败。新判据:
        #   非劣 Δ_dir ≥ −z·sd,保真 Δ_dir ≤ +z_fid·sd,
        #   sd = sqrt(SE² + σ_seed²·(1+1/n_child))
        # σ_seed:multi-seed → 同配方复现组(df≥2)→ 先验 → floor_ratio×SE。
        "confirm_margin_z": 2.0,
        "confirm_fidelity_z": 4.0,
        "confirm_seed_sd_floor_ratio": 1.0,
        "confirm_seed_sd_prior": None,
        # v2.19.1:σ_seed = ratio×SE 的比值先验(scaled = 按 n_val 缩放,
        # 见 runtime.confirmation.scaled_sigma_prior;None = 关闭)。
        # 新 run 由 V2_19_1_RUN_CONFIG 写入 "scaled"
        "confirm_seed_sd_prior_ratio": None,
        # confirm 子代 vs 父代的参照(父代当初的比较对象)在 shadow 上不得
        # 显著更差(同一 margin 口径);参照 shadow 分数缺失时跳过并留痕
        "confirm_reference_check": True,
        # confirm 的 multi-seed 稳定性门:seed std > max(2·noise,
        # ratio×MDE) 才判不稳(旧口径只比 2·noise,大样本上 seed 方差天然
        # 大于评测噪声,multi-seed 一旦真正执行就会结构性误杀)
        "confirm_stability_mde_ratio": 0.5,
        # 交付位:陈旧确认(非搜索冠军配方)可替代冠军的最大差距;
        # None = minimum_meaningful_improvement,且不低于 2×noise
        "delivery_divergence_tolerance": None,
        # v2.20.0:先验豁免 —— 满足"换 seed 后增益仍在"的冠军可不做 confirm
        # 直接交付(delivery_status=confirmation_waived_by_prior)。判据见
        # runtime.confirmation.seed_robust_gain:
        #   z = Δ_vs_baseline / sqrt(SE² + σ_seed²) ≥ z_required
        # σ_seed 必须来自实测(复现组 / multi-seed / 经验库同 data_base 比值)。
        # 旧口径 None = 关闭(在途 run 不换尺);新 run 由 V2_20_0_RUN_CONFIG
        # 写入 {"enabled": True, ...}。
        "prior_waiver": None,
        # 先验豁免适用的动作(只有 warm-start 默认在列:它们复用历史配方,
        # 增益主张先天带"这个配方在别的 run 上确实更强"的独立证据)
        "prior_waiver_actions": ["warm_start"],
        # v12(P1-5):切片门多重比较校正(0 = 自动 z_{1-α/k})
        "slice_alpha": 0.05,
        # v11:分组置换重要性(超级 trial 方向裁决;每族置换次数,
        # >1 才报 delta_std,单次置换 std 恒 0 无意义)。
        # v2.17.3:默认 5 → 2。req-t7 run4 实测单 trial 置换段 136~747 s
        # (200k 行 × 54 族 × 5 repeat),占单 trial wall 的 5%~12%,而
        # 置换只做**方向初筛**(晋级/交付裁决走 validation 配对 bootstrap,
        # 与本节无关)。2 repeat 仍能给出 delta_std 的量级,耗时降到 ~40%。
        "group_importance_repeat": 2,
        # v19.4:分组置换按族并行的 worker 数(1 = 串行,与 v18
        # 行为一致;默认 4 = 实测 60 族×5 repeat 340s → 48s。
        # 环境变量 MODELEVO_GI_WORKERS 优先于本键)。快路径不可用
        # (非 GBM/uplift/dtype 异常)时自动回退单线程 DataFrame 路径。
        "group_importance_workers": 4,
        # v19.2:分组置换的评估行数上限(0 = 全量,保持历史行为)。
        # 400w 尺实测:598,955 行 × 55 族 × 5 repeat 的评估段占单 trial
        # 总耗时的 ~60%(~40min),是 trial_timeout/heavy 预算的大头。
        # 设 >0 时在**同一分层口径**的子样本上裁决族生死(跨 trial 可比
        # 性更好);落盘的 validation AUC 始终是全量口径,不受影响。
        # v2.13.5:默认 0 → 200000。req-t7-rand400w 实测 44.7 万行全量置换
        # 单 trial 186~1467s(并发评估时),是评估段最大头;validation 不超过
        # 该行数时仍是全量,行为不变。
        "group_importance_sample_rows": 200000,
        # v2.13.5:分组置换执行策略。all = 每个 trial 都算(历史行为);
        # skip_rejected = verdict=rejected/invalid 的 trial 跳过置换(evaluation
        # 里写 skipped+reason,directions 回退 gain 前缀推断)。默认 all:
        # 被拒 trial 的族 delta 仍是方向证伪的证据,不静默丢弃。
        # v2.15.0:auto = 只改模型的 trial(hpo/confirm/changed_modules=[model])
        # 继承父代的族级置换结论(特征面未变);其余照常重算
        "group_importance_policy": "skip_rejected",  # v2.15.9 默认改为 skip_rejected
        # v2.15: gen-00(baseline + warm trial)跳过 group importance。
        # baseline 只建锚点不提出方向;warm trial 的方向先验已由历史 recipe
        # 编码。400w 上 GI 单 trial ~40min,三个 gen-00 trial 全算 ≈2 GPU 小时
        # 纯浪费。gen-01 起的 explore trial 照常产出。
        "skip_group_importance_on_generation_0": True,
    },
    "gen_executor": {
        # v18/P2-1:successive halving 默认关。机制只在 run-generation
        # 整代 barrier 路径生效,与 SKILL 主推的写码/训练流水线互斥
        # (逐 trial 后台 run 天然绕开它);实测 v17.4 两 run 0 racing
        # 痕迹——保留开关,默认诚实(摆设 > 静默失效)。
        # 历史依据(v12/P1-10):两次 run 机器利用率 1–2%,开启后同代
        # trial 先子样本短评再全量重跑胜者——那是 run-generation 模式
        # 下的正确行为,用 run-generation 时请显式开回 true。
        "halving_enabled": False,
        "halving_ratio": 0.33,
        "halving_budget_fraction": 0.3,
        "halving_min_trials": 4,
        "halving_survive_fraction": 0.5,
    },
    "profile": {
        # v8:信息压缩检测(data_view 候选自动生成)
        "compression_audit_enabled": True,
    },
    # 特征探索默认采用可归因的 bundle beam；super trial 仅作可选初筛，
    # 不再把所有方向揉成一次不可诊断的试验。
    #
    # [v2.13.2] 本块的键此前只有 min_generations 被代码读取，其余全是
    # 摆设（配了不生效，且无任何提示）。现按实现状态逐键标注：
    #   IMPLEMENTED —— 有代码消费，改它有效；
    #   RESERVED    —— 无实现，改它**不会**生效（诚实标注，不是待办）。
    "feature_exploration": {
        "enabled": True,            # RESERVED：特征优先已是唯一流程，
                                    # 旧的"模型/特征混排"路径已从代码移除，
                                    # 本键恒真（关闭它不会回退到旧流程）
        "max_sample_rows": 200000,  # IMPLEMENTED：write-splits-sample
                                    # 的 --max-rows 缺省值（cmd_views.py）
        "min_generations": 2,       # IMPLEMENTED：特征期门（v2.16.0,
                                    # feature_phase.py;Planner 与 new-trial
                                    # 共用;0=关闭);v12:默认 2（原 1）
        "converge_generations": 1,  # RESERVED：对应 feature-exploration.md
                                    # 的 feature_exploration_complete 充分性
                                    # 门，该门同样只有文档、无 runtime 实现
        "super_trial": False,       # IMPLEMENTED（v2.13.2）：true 时
                                    # Planner 产出 screening-only 宽面初筛
                                    # 候选（只产 group_importance 先验，
                                    # 不进 FDR/冠军/Archive/方向台账）
        "bundle_beam_width": 4,     # IMPLEMENTED（v2.13.2）：每代特征
                                    # trial 槽位总数（screening 计入）；门在
                                    # cmd_new_trial，planner 批次留痕
        "conditional_ablation": True,   # RESERVED（v2.13.2）：runtime
                                    # 无任何实现——drop-group retrain 消融
                                    # 目前由 Agent 手建 trial，见
                                    # references/feature-exploration.md
        "ablation_trials": 2,       # RESERVED：同上
        # ---- v2.17.0(WP5b)流式数据视图 / 抽样面噪声预检 ----
        "view_stream_max_mem_gb": None,     # IMPLEMENTED:write-splits-view
                                    # 流式构建内存上限(GB;None=只按块行数,
                                    # --max-mem-gb 覆盖)
        "view_stream_chunk_rows": 200000,   # IMPLEMENTED:流式块行数缺省
        "view_full_fallback_max_gb": 8,     # IMPLEMENTED:build_view 非行内
                                    # 变换时,估算全量内存 ≤ 该值才回退全量模式
        "sample_precheck_max_full_minutes": 30,  # IMPLEMENTED:噪声预检中
                                    # baseline 全量运行时长阈值(分钟)
        "sample_precheck_n_boot": 200,      # IMPLEMENTED:噪声预检 bootstrap 次数
    },
    "ensemble": {"enabled": False},
    # v2.18.0:中心化经验库(docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md
    # §8.3)。客户端键可被 MODELEVO_LIBRARY_URI / _NS / _TEAM 环境变量覆盖。
    "library": {
        # [FIX-2026-10-29] 主动关闭经验库的正式开关。旧版只有"服务不可达 →
        # 退到本机缓存(stale)"的被动路径,禁用意图无处表达:把 uri 指到
        # 不可达地址仍会读到缓存里的真实视图,baseline 照旧创建 warm-start
        # trial(实测 run4 req-t7 run1)。false = 不连服务、不读本机缓存、
        # 不固化视图、不做热启动。环境变量 MODELEVO_LIBRARY_ENABLED 可覆盖。
        "enabled": True,
        "uri": "http://127.0.0.1:18765",
        "namespace": "prod",
        "team": "",
        "timeout_sec": 15,
        "cache_dir": "",            # 空 = ~/.cache/modelevo-library
        "service": {
            "root": "/data/oceanus_share/modelevo-library",
            "db_url": "",           # 空 = sqlite:///{root}/library.db
            "artifacts_url": "",    # 空 = file://{root}/cas
            "host": "127.0.0.1",    # Q44:跨机器接入前只听本机
            "port": 18765,
            "backup_keep_days": 30,
            "python": "",           # 装有 fastapi/uvicorn 的解释器;空 = 当前解释器
            "browse": True,         # 明文镜像 {root}/browse/<namespace>(写入后自动重新生成)
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 override 到 base(返回新 dict,不修改入参)。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """点路径访问的配置对象:c.get('execution.trial_timeout_minutes')。"""

    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def load(
        cls,
        *,
        skill_config: Path | None | bool = True,
        run_config: Path | None = None,
    ) -> "Config":
        # 缺省即应用 skill 级配置(见 _SKILL_CONFIG);显式 False/None 跳过
        if skill_config is True:
            skill_config = _SKILL_CONFIG
        data = copy.deepcopy(DEFAULTS)
        for path in (skill_config, run_config):
            if path and Path(path).exists():
                loaded = load_yaml(Path(path)) or {}
                if not isinstance(loaded, dict):
                    raise ValueError(f"配置必须是映射: {path}")
                data = _deep_merge(data, loaded.get("modelevo", loaded))
        # 运行协议版本只有一个事实来源。旧配置中的 version 仅作历史字段，
        # 不能覆盖当前 harness 版本并造成报告/epoch 口径分裂。
        data["version"] = __version__
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def raw(self) -> dict:
        return self._data

    def snapshot(self) -> dict:
        return copy.deepcopy(self._data)
