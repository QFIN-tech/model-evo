"""v2.17.0:协议行为门——"只对新 run 启用"的统一判定。

run4 regression.2 优化计划里有一批会改变裁决口径的改动(锚定晋级、crossover
IUT、量纲感知 MDE、alpha 排程等)。"尺子不可演进":在途 run 不能中途换尺,
因此这些行为只对 ``run-state.skill_version >= 2.17.0`` 的 run 生效(init 时
写入);旧 run 无该键或版本更低,保持原行为。

与 ``runtime.version_at_least`` 同口径,集中在这里是为了让各模块不再各自
读 run-state、各自写版本常量。
"""

from __future__ import annotations

import json
from pathlib import Path

from . import version_at_least

#: v2.17.0 新协议门(锚定晋级 / crossover IUT / 量纲感知 MDE / alpha 排程)
V2_17_0 = "2.17.0"

#: v2.17.0:新 run 的配置覆写(cmd_init 在配置快照后合并写入 run 的
#: modelevo-config.yaml)。代码 DEFAULTS 与 assets/global-config.yaml 对这些键
#: 保持**旧口径**,因此在途 run(配置快照里没有这些键)行为不变;只有新 init
#: 的 run 才启用新尺子。键的语义见 runtime/config.py DEFAULTS 注释。
V2_17_0_RUN_CONFIG: dict = {
    "evaluator": {
        # 量纲感知 MDE:max(absolute, relative × |baseline 主指标|)
        "minimum_meaningful_improvement_relative": 0.0008,
        # 晋级(冠军/confirm 资格)的对照:parent=逐步对父代(旧),
        # anchor=对当前交付锚点(已确认冠军;无确认时为 baseline)
        "promotion_reference": "anchor",
        # 多父代 crossover 的 vs_parent 口径:first_parent(旧)|iut
        "crossover_parent_test": "iut",
        # 代级 alpha 排程:uniform(旧,按剩余代数均摊)|lord(LORD++ 回补)
        "alpha_schedule": "uniform",
    },
}


#: v2.19.0 新协议门(锚点取现任冠军 / 切片门 √k / SE 计入 seed 方差)
V2_19_0 = "2.19.0"

#: v2.19.0:新 run 的配置覆写(cmd_init 在 V2_17_0_RUN_CONFIG 之后合并)。
#: 动机见 docs/MODELEVO_GOAL_REFLECTION_AND_ROADMAP_20261101.md §4.4(135 个
#: 真实 trial 统计):
#: - 锚点在首次 confirm 前一直是弱 baseline,warm-start 之后每个 trial 的
#:   vs_anchor≈+0.011、p=1,同代 BH 形同虚设(ds-flash trial-009:Δparent=0、
#:   p=0.577 仍判 accepted);
#: - 切片门阈值乘整体 SE,单切片 SE≈√k 倍,零效应误拒率 29~43%;
#: - 配对 SE 只含 bootstrap 抽样噪声,σ_seed 与之同量级,p≥0.95 的 Δ 仍在
#:   seed 噪声以内。
#: 代码 DEFAULTS 保持旧口径,在途 run 不换尺。
V2_19_0_RUN_CONFIG: dict = {
    "evaluator": {
        # 锚点:delivery(旧,交付位→最优已确认→baseline)| incumbent(上一代
        # 结束时同视图最优的 accepted/confirmed trial,含 gen-00 warm-start)
        "anchor_mode": "incumbent",
        # anchor 模式护栏:vs_parent 的 p_better 须 > 该值。旧值 0.05 只拦
        # "显著更差";incumbent 锚点下链式小步仍靠 vs_anchor 累积,护栏只要求
        # 每一步方向为正且有一定把握
        "anchor_parent_guard_p": 0.80,
        # 切片门阈值:none(旧,z·SE)| sqrt_k(z·√k·SE,单切片 SE 口径)
        "slice_se_scaling": "sqrt_k",
        # 配对检验的 SE 计入 seed 方差:se_eff = sqrt(SE² + factor·σ_seed²)
        "seed_variance_in_se": True,
        "seed_variance_factor": 1.0,
        # 平台期停止(建议性):最近 2 个非空代冠军增益都 < 0.5×MDE
        "plateau_stop_generations": 2,
        # leak_scan 哨兵尊重 kept_ambiguous 裁决(benchmark creditcard:合法的强
        # 单变量特征让 21/22 个 trial 判 invalid,run 无法交付)
        "leak_scan_respect_adjudication": True,
    },
}


#: v2.19.1 新协议门(σ_seed 规模先验 / 收尾整合通道)
V2_19_1 = "2.19.1"

#: v2.19.1:新 run 的配置覆写(cmd_init 在 V2_19_0_RUN_CONFIG 之后合并)。
#:
#: 动机(benchmark 8 数据集 + 170 个真实 trial 的落盘指标回放):
#: - **σ_seed 先验的规模敏感**:实测 σ_seed/SE 比值随 validation 行数单调上升
#:   (5 万行中位 0.42,44.7 万行 1.19,400 万级更大)。v2.19.0 的
#:   「σ_seed ≥ 1×SE」在小数据上过保守(零效应误拒 2.9%),在大数据上又缺
#:   刻画比值 >1 的部分(400 万级误放 11.9%)。改为 σ_seed = r(n_val)·SE,
#:   r = clip(0.5·(n_val/5万)^0.4, 0.5, 1.5);实测值(本 run 复现组 /
#:   confirm multi-seed / 经验库同底座历史比值)优先,且不再被 1×SE 抬高。
#: - **factor 1 → 2**:σ² 系数取 2,对应「两个模型的 seed 效应各自独立」的
#:   方差叠加口径(与 confirm margin 的 (1+1/n) 同源)。
#: - **收尾整合通道**:停止建议触发时,把「被同代 BH 拦下的 promising、
#:   质量门有效、同视图、相对现任冠军 Δ≥MDE」的最佳候选单独检验一次
#:   (family=1、q=min(generation_fdr_q, 剩余 alpha) 并扣费),通过只授予
#:   confirm 资格,由 confirm 裁决能否成为冠军。benchmark 复盘:72 个被同代 BH 拦下的正增益
#:   候选里 6 个在测试面上更好(全部 amazon),停损同样会作废它们。
#:
#: 代码 DEFAULTS 保持旧口径,在途 run 不换尺。
V2_19_1_RUN_CONFIG: dict = {
    "evaluator": {
        # σ_seed 规模先验(σ_seed = r(n_val)·SE);None 关闭
        "confirm_seed_sd_prior_ratio": "scaled",
        # 实测 σ_seed 不再被 floor_ratio×SE 抬升(floor 只兜"完全无估计")
        "seed_variance_floor_measured": False,
        # 搜索口径的 σ_seed floor 归零:规模先验 r<1 的下半段不得被抬回 1×SE
        # (否则 5 万行以下的比值先验恒等于旧口径)。confirm 门的 floor 不受影响。
        "seed_variance_floor_ratio": 0.0,
        # se_eff = sqrt(SE² + factor·σ_seed²):factor=2 ≈ 两侧独立 seed 效应
        "seed_variance_factor": 2.0,
        # 停止时的收尾整合通道
        "closing_consolidation": True,
    },
}


#: v2.20.0 新协议门(先验豁免 + warm-start 确认资格修正)
V2_20_0 = "2.20.0"

#: v2.20.0:新 run 的配置覆写(cmd_init 在 V2_19_1_RUN_CONFIG 之后合并)。
#:
#: 动机(2026-11-02 复核):
#: - **warm-start 默认不 confirm 的诉求有实据**:warm trial 复用经验库历史
#:   配方,它的增益主张先天带"这个配方在别的 run 上确实更强"的独立证据;
#:   而 confirm 是等价性检验(非劣 + 保真),margin 恒为 ±z·sd,通过率与
#:   数据无关 ≈ 0.977 —— 对这类配方,confirm 基本是付费盖章(实测 14/14
#:   通过、平均占墙钟 20%)。但**不能**用"confirm 会不会过"作判据(恒真),
#:   改用有判别力的问题:换一个 seed 把分数整体挪 2 个标准差,配方是否仍
#:   优于 baseline。σ_seed 取实测(复现组 / multi-seed / 经验库同 data_base
#:   比值),拿不到实测就不豁免。
#: - **普通 trial 不适用**:它们的 σ_seed 未知,用它去豁免等于循环论证。
#: - 豁免不丢样本外估计:走的是 require_confirmation=false 同一条路径
#:   (final-model/ 正常交付位、final_test/shadow 照常打分),只少一份
#:   独立 seed 复现证明;delivery_status 记 confirmation_waived_by_prior。
#:
#: 代码 DEFAULTS 保持旧口径,在途 run 不换尺。
V2_20_0_RUN_CONFIG: dict = {
    "evaluator": {
        # 先验豁免:满足"换 seed 后增益仍在"的冠军可不做 confirm
        "prior_waiver": {
            "enabled": True,
            # z = Δ_dir / sqrt(SE² + σ_seed²) 的下限
            "z_required": 2.0,
        },
        # 只有 warm-start 在列(普通 trial 的 σ_seed 未知,不豁免)
        "prior_waiver_actions": ["warm_start"],
    },
}


def run_skill_version(run_dir: "Path | str | None") -> "str | None":
    """读 run-state.json 的 skill_version(init 时写入);缺失返回 None。"""
    if not run_dir:
        return None
    path = Path(run_dir) / "run-state.json"
    if not path.exists():
        return None
    try:
        return (json.loads(path.read_text(encoding="utf-8")) or {}).get(
            "skill_version")
    except Exception:  # noqa: BLE001 —— 行为门读失败按旧 run 处理
        return None


def run_at_least(run_dir: "Path | str | None",
                 minimum: str = V2_17_0) -> bool:
    """run 由 ``>= minimum`` 的 skill 初始化 → True(启用新协议行为)。"""
    return version_at_least(run_skill_version(run_dir), minimum)
