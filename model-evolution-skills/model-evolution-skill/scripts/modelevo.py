#!/usr/bin/env python3
"""ModelEvo 确定性 Harness CLI。

命令:
  init      初始化 run-{task_id}/ 工作区(task-spec + data-loader + 目录)
  validate  校验 TaskSpec
  profile   执行 SampleProfile 固定分析(含 splits 落盘)
  baseline  生成并执行 gen-00 baseline Trials
  new-trial 创建 Trial 脚手架(写入 hypothesis/代码模板)
  run       执行一个 Trial(代码必须已落盘)
  wait      确定性等待 Trial 终态(v17.2;输出 SUMMARY 行,替代轮询)
  finalize-generation  Generation 收尾(leaderboard/archive/stop)
  evaluate-final  最终模型一次性 final_test 盲测
  report    生成 16 节最终报告
  leaderboard 重建 Leaderboard
  status    查看 Run 状态
  export-experience  导出经验到全局库
  library   v2.18 中心化经验库:serve/status/stop/import-legacy/dump/load/merge-*
  experience v2.18 按假设主题检索经验库:search/get/evidence/topics
"""

from __future__ import annotations

import inspect
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 让 scripts/modelevo.py 直接可运行:把 skill 根与 scripts/ 加入 sys.path。
# - skill 根:本文件 import runtime.*;
# - scripts/:本文件 import _mvo.*(v2.11.9 起 CLI 实现分片在该包内)。
# 两者都按 __file__ 推导,**不依赖调用方怎么加载本文件**——实测测试用
# importlib.util.spec_from_file_location 装载时不带 sys.path[0],
# 缺 scripts/ 会让 `import _mvo` 直接 ModuleNotFoundError。
_SELF = Path(__file__).resolve()
_SKILL_ROOT = _SELF.parents[1]
for _p in (str(_SKILL_ROOT), str(_SELF.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# v2.17.0:`wait` 轻量快路径。wait 是长时只读轮询,不应为 pandas/numpy/
# runtime.runner 等重模块付常驻内存(run4 regression.2:同机内存争用时 wait
# 两次被 cgroup OOM 杀,exit 137)。仅当以脚本运行且子命令为 wait 时生效;
# 被 import(测试 / 再导出)时照常加载完整 CLI。wait 被杀时原样重调即可。
if __name__ == "__main__" and sys.argv[1:2] == ["wait"]:
    from _mvo.cmd_wait import main as _wait_main

    raise SystemExit(_wait_main(sys.argv[2:]))


from runtime import __version__  # noqa: E402
import argparse
from _mvo import context, render  # noqa: E402
from _mvo.cmd_deliver import cmd_evaluate_final, cmd_finalize_baseline_score, cmd_finalize_retrain, cmd_leaderboard, cmd_release_final_test, cmd_report
from _mvo.cmd_inspect import cmd_acknowledge_harness_change, cmd_assets_config, cmd_benchmark, cmd_capacity_check, cmd_check_products, cmd_delete_trial, cmd_diagnose, cmd_export_experience, cmd_plan, cmd_planner_context, cmd_preflight, cmd_snapshot, cmd_status, cmd_wait
from _mvo.cmd_startup import cmd_baseline, cmd_init, cmd_profile, cmd_validate
from _mvo.cmd_trial import cmd_finalize_generation, cmd_new_trial, cmd_run_generation, cmd_run_trial
from _mvo.cmd_topology import cmd_approve_topology
from _mvo.cmd_library import add_library_parser  # v2.18.0 中心化经验库
from _mvo.cmd_experience import add_experience_parser  # v2.18.0 按假设检索
from _mvo.cmd_wait import WAIT_HELP as _WAIT_HELP  # noqa: E402
from _mvo.cmd_wait import add_wait_arguments as _add_wait_arguments  # noqa: E402
from runtime.topology import DECLARABLE as _TOPOLOGY_DECLARABLE  # noqa: E402
from _mvo.cmd_views import cmd_write_splits_sample, cmd_write_splits_view
from _mvo.cmd_speed import (cmd_contrast, cmd_feature_cache,  # noqa: E402
                            cmd_finalize_all, cmd_timing_report)
from _mvo.cmd_prescreen import cmd_prescreen  # noqa: E402  v2.17.0 WP5b
from _mvo.trial_patch import cmd_trial_skeleton  # noqa: E402  v2.19.0 #15
from _mvo.context import logger

from _mvo.render import (  # noqa: F401  兼容再导出
    _HPO_RENDER_ALGOS,
    _HPO_REQUIRED_SYMBOLS,
    _assert_hpo_capable,
    _copy_parent_trial_code,
    _explicit_face,
    _graft_hpo_section,
    _has_module_level_algo,
    _hpo_cost_gate,
    _hpo_parent_face_risk,
    _inject_hpo_algo_at,
    _inject_hpo_config,
    _load_features_ref,
    _parent_hpo_overridden_symbols,
    _parse_overrides,
    _ref_domain,
    _rollback_trial,
    _validate_grafted_hpo_deps,
)

from _mvo.cmd_startup import (  # noqa: F401  兼容再导出
    _backfill_shadow_scores,
    cmd_baseline,
    cmd_init,
    cmd_profile,
    cmd_validate,
)

from _mvo.cmd_trial import (  # noqa: F401  兼容再导出
    _resolve_features_override,
    cmd_finalize_generation,
    cmd_new_trial,
    cmd_run_generation,
    cmd_run_trial,
)

from _mvo.cmd_deliver import (  # noqa: F401  兼容再导出
    _assert_no_duplicate_columns,
    _finalize_build_estimator,
    _finalize_load_module,
    _record_shadow_waiver,
    _widen_self_contained,
    cmd_evaluate_final,
    cmd_finalize_baseline_score,
    cmd_finalize_retrain,
    cmd_leaderboard,
    cmd_release_final_test,
    cmd_report,
)

from _mvo.cmd_inspect import (  # noqa: F401  兼容再导出
    _diag_product_state,
    _diag_trial_anomalies,
    cmd_acknowledge_harness_change,
    cmd_assets_config,
    cmd_benchmark,
    cmd_capacity_check,
    cmd_check_products,
    cmd_delete_trial,
    cmd_diagnose,
    cmd_export_experience,
    cmd_plan,
    cmd_planner_context,
    cmd_preflight,
    cmd_snapshot,
    cmd_status,
    cmd_wait,
)

from _mvo.cmd_views import (  # noqa: F401  兼容再导出
    cmd_write_splits_sample,
    cmd_write_splits_view,
)

from _mvo.context import (  # noqa: F401  兼容再导出
    PHASE_ORDER,
    _LAST_CREATED_TRIAL_DIR,
    _SKILL_ROOT,
    _WAIT_TERMINAL_STATES,
    _load_best_parent_scores,
    _load_config,
    _load_spec,
    _parse_list,
    _resolve_baseline_id,
    _resolve_splits_dir,
    _run_dir_from,
    _started_at,
    _trial_reached_terminal,
    _trial_summary_line,
    _update_run_state,
    logger,
)


def _add_refit_flags(sp) -> None:
    """交付重训口径开关(v2.15.9;finalize-retrain / finalize-all)。"""
    sp.add_argument(
        "--refit-mode", choices=("rerun_recipe", "frozen_rounds"),
        default=None,
        help="v2.15.9:rerun_recipe(默认)= 复制冠军 code/config,以 "
             "train+validation 重跑冠军配方,早停落在 train+validation 最新"
             "尾部(单段配方只训一次,不全量重拟合;复合容器可用);"
             "frozen_rounds = 旧口径:train "
             "尾部定轮 → sklearn.clone → train+validation 重拟合。缺省读 "
             "config finalize.refit_mode")
    sp.add_argument(
        "--full-reproduction", action="store_true",
        help="v2.15.9:rerun_recipe 下强制在 train 上完整复现冠军并比对 "
             "validation 分数(缺省用特征面一致性校验;冠军无 "
             "model/validation_input.parquet 时自动退回完整复现);"
             "v2.19.0:frozen_rounds 下缺省复用冠军冻结模型(validation "
             "重打分逐位一致即跳过 train 上复现),本开关强制完整复现")


def _add_detach_flags(sp) -> None:
    """给长时训练命令加会话解耦开关(v2.13.3,gen-04 事故回写)。

    默认**关闭**(--no-detach 是缺省值),保持既有调用方行为不变;
    guard hook / SKILL.md 的推荐姿势会带 --detach。理由见
    ``runtime/daemon.py``:trial 挂在 agent 会话的进程树上时,会话边界
    会连带杀掉训练(实测三个 HPO trial 三次同时死亡,空烧 7.9h)。
    """
    sp.add_argument(
        "--detach", action="store_true",
        help="v2.13.3:把本次执行体拉到独立会话(双重 fork + setsid,"
             "stdio 重定向到 logs/detached-*.log)后立即返回。训练不再"
             "随 agent 会话退出而终止;用 `wait`/`status` 追踪终态。"
             "推荐用于 run-generation 与长时 heavy/hpo trial。")
    sp.add_argument(
        "--no-detach", action="store_true",
        help="显式前台执行(detach worker 内部使用;缺省即前台)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modelevo.py",
        description=f"ModelEvo {__version__} 确定性 Harness CLI")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="初始化任务工作区")
    sp.add_argument("--dir", required=True)
    sp.add_argument("--task-id")
    sp.add_argument("--task-name")
    sp.add_argument("--objective")
    sp.add_argument("--target")
    sp.add_argument("--positive-def")
    sp.add_argument("--time-column")
    sp.add_argument("--id-column")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("validate", help="校验 TaskSpec")
    sp.add_argument("--spec", help="task-spec.yaml 路径")
    sp.add_argument("--run")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("preflight", help="依赖预检(required 缺失报错;"
                                          "optional 记录;产 env-manifest)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.set_defaults(func=cmd_preflight)

    sp = sub.add_parser("check-products",
                        help="四产物自消费冒烟(profile 后自动跑;"
                             "write-splits-view/-sample 后手动跑)")
    sp.add_argument("--spec", required=True)
    sp.add_argument("--run", required=True)
    sp.add_argument("--splits-dir", default="sample-profile/splits")
    sp.set_defaults(func=cmd_check_products)

    sp = sub.add_parser("acknowledge-harness-change",
                        help="登记 harness 变更(铁律 10 受支持出口,"
                             "替代手改 run-state.json)")
    sp.add_argument("--run", required=True)
    sp.add_argument("--reason", required=True,
                    help="变更理由,写入 run-state.harness_change_log")
    sp.add_argument("--allow-measurement-impact", action="store_true",
                    help="变更触及 measurement 模块时仍登记;"
                         "harness_epoch +1,新旧 epoch 不混排")
    sp.add_argument("--measurement-no-score-impact", action="store_true",
                    help="v14.2:measurement 模块变更但实测分数口径未变"
                         "(evaluator/metrics 落盘数字不受影响)时降级登记;"
                         "epoch 不 +1,历史 trial 继续同表可比")
    sp.add_argument("--verify-against", default=None,
                    metavar="TRIAL[,TRIAL...]",
                    help="v2.17.0:用当前 evaluator 重评这些 trial 的已落盘"
                         "预测(不重训),与 evaluation.json 的主指标/Δ/CI/SE "
                         "逐项比对;全部一致才放行 "
                         "--measurement-no-score-impact")
    sp.set_defaults(func=cmd_acknowledge_harness_change)

    sp = sub.add_parser("capacity-check",
                        help="天花板诊断(可选,仅供参考,不参与放行):"
                             "样本量/列数曲线+局部标签歧义 → 加行/加列/调参"
                             "排序(抽样面,分钟级)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--splits-dir", default="sample-profile/splits-sample",
                    help="抽样面 splits 相对路径")
    sp.add_argument("--reference-trial",
                    help="提供 feature_importance.csv 的 trial id"
                         "(列数曲线需要)")
    # v2.12.2.9:样本量曲线是主要成本(4 格点 × seeds 次拟合),而 std
    # 只服务"末段增益是否覆盖 0"一个判定 —— 默认可降到 2。
    sp.add_argument("--seeds", type=int, default=2,
                    help="样本量曲线每格点重复 seed 数(默认 2;"
                         "增大可收紧末段增益的 95%% CI,代价是拟合次数"
                         "线性上升)")
    sp.add_argument("--no-cache", action="store_true",
                    help="忽略 capacity-report.json 缓存强制重算"
                         "(默认按抽样面指纹复用已有报告)")
    sp.set_defaults(func=cmd_capacity_check)

    sp = sub.add_parser("profile", help="SampleProfile 固定分析")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--workers", type=int,
                    default=max(4, min(16, (os.cpu_count() or 8) // 8)),
                    help="v2.13.5:缺省按核数取 4~16(旧固定 4)")
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("baseline", help="gen-00 baseline")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("new-trial", help="创建 Trial 脚手架")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--algorithm")
    sp.add_argument("--from-parent", metavar="TRIAL_ID",
                    help="参数 trial 快通道(v10.3):复制该父 trial 的 code/,"
                         "零 LLM 写码;必须配 --set、--features、"
                         "--action hpo 或 --action confirm。语义是"
                         "\"代码/配置载体\",与 --parents"
                         "(血缘声明,供 lineage/crossover)相互独立,可同用")
    sp.add_argument("--allow-code-edit", action="store_true",
                    help="以父代代码作脚手架后编辑；源码会在 run 前冻结，"
                         "不能在执行后改写同一 trial")
    sp.add_argument("--allow-extra-confirm", default=None, metavar="REASON",
                    help="v2.14.0:--action confirm 超出 evolution."
                         "max_confirm_attempts_per_recipe 时的用户裁决理由"
                         "(留痕 trial.yaml.extra_confirm);缺省拒绝,防"
                         "\"失败就再抽\"")
    sp.add_argument("--hpo-parent", metavar="TRIAL_ID",
                    help="[v2.9] --action hpo 无 --from-parent 时的模板渲染"
                         "路径:继承该 trial config 的特征面(features/"
                         "reparse)与 data_view 语义,代码按 --algorithm "
                         "渲染;仅声明特征来源,不构成血缘(parents 不含它)")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="超参覆写(可多次):与 --from-parent 配用,值按 "
                         "YAML 字面量解释(0.03→float);合并进 "
                         "config/trial.yaml 的 hyperparameters")
    sp.add_argument("--hypothesis")
    sp.add_argument("--rationale")
    sp.add_argument("--falsification")
    sp.add_argument("--actions", help="逗号分隔动作描述")
    sp.add_argument("--action", choices=["explore", "exploit", "crossover",
                                         "data_view", "hpo", "confirm",
                                         "automl"],
                    default="explore")
    sp.add_argument("--search-space", metavar="YAML_PATH",
                    help="hpo 动作:搜索空间 YAML(参数名映射到 float/int 的"
                         " low/high 或 categorical 的 choices);plan 会生成"
                         "默认空间;--action hpo 时必填")
    sp.add_argument("--budget-configs", type=int, default=None, metavar="N",
                    help="hpo 动作:inner 搜索的配置评估数上限"
                         "(默认 evolution.hpo.default_budget=40)")
    sp.add_argument("--hpo-sample-rows", type=int, default=None, metavar="N",
                    help="v18/P1-4:hpo 动作 inner 搜索的分层抽样行数上限"
                         "(大数据可行性;best_params 仍全量重训。"
                         "实测 400w run:inner 构建 6217s + 单 config ~900s,"
                         "360min heavy 只装得下 1 个 config——抽样面搜索"
                         "+ 全量 confirm 是唯一可规划路径)")
    sp.add_argument("--force-hpo-budget", action="store_true",
                    help="v16:hpo 成本闸豁免(预估超 heavy 预算仍创建,"
                         "留痕 expected_resources.hpo_force_budget)")
    sp.add_argument("--parents", help="逗号分隔父 trial_id")
    sp.add_argument("--changed-modules",
                    help="逗号分隔改动模块,严格枚举 feature/sample/model/"
                         "loss/data(v2.17.0:未知值直接拒绝),crossover 兼容性"
                         "检查与方向台账用(§13.3)")
    # v2.16.0:--override-capacity 保留为别名(capacity 闸门已由特征期门取代)
    sp.add_argument("--override-feature-phase", "--override-capacity",
                    dest="override_feature_phase", metavar="REASON",
                    help="特征期门豁免理由——已完成探索代数 < "
                         "feature_exploration.min_generations 时,model/"
                         "loss/hpo 方向 trial 需显式给出理由(留痕 "
                         "trial.yaml.feature_phase_override);特征期满后"
                         "无需提供")
    sp.add_argument("--family-hints", metavar="PREFIX[,PREFIX...]",
                    help="v16/P1-3:特征方向的前缀族提示(如 "
                         "\"seq_,ratio_,num_\")——方向台账按族入账,"
                         "UCB 槽位建议落到\"哪个特征族该续投\"粒度")
    sp.add_argument("--experience-ids", metavar="ID[,ID...]",
                    help="旧写法:等价于 --basis <id>:follows(v2.18.0 起用 --basis)")
    sp.add_argument("--basis", metavar="ID:REL[,…]|novel",
                    help="v2.18.0:本 trial 的依据 —— 经验库原则 `<id>:<follows|extends|"
                         "tests|contradicts>`(逗号分隔,关系省略为 follows),或 `novel`。"
                         "run 已固定经验库视图时必填;--from-parent 子代未给时继承父代")
    sp.add_argument("--basis-prefill", metavar="ID:REL[,…]",
                    help="v2.18.6:plan 物化命令自动带上的预填依据(勿手写)。--basis 与之"
                         "相同 → 依据记为程序预填;改写过 → 记为 Agent 自己的判断")
    sp.add_argument("--topics", metavar="TOPIC[,…]",
                    help="v2.18.0:本 trial 的主题(经验库主题词或别名),供检索与证据匹配")
    sp.add_argument("--planner-proposal-id",
                    help="由 plan 生成的 proposal_id，写入 Trial 审计链")
    sp.add_argument("--planner-semantic-hash",
                    help="由 plan 生成的跨代语义 hash，用于候选去重")
    sp.add_argument(
        "--generation", type=int, default=None,
        help="目标代号;漏传时自动取父代代号+1(from-parent)或已落盘"
             "trial 最大代号+1——不提供静默默认值,防止跨代错标"
             "(历史事故:快通道 trial 漏传落 default=1,g01 复现)")
    sp.add_argument("--data-view", default="v1",
                    help="数据视图 id(协议 v7):v1=默认 splits,其他="
                         "splits-{view};data_view 动作必须先 write-splits-view")
    sp.add_argument("--heavy", action="store_true",
                    help="标记 heavy trial(特征重建/聚合重做):超时×"
                         "heavy_trials.timeout_multiplier,占每代 heavy 配额")
    sp.add_argument("--features",
                    metavar="COL[,COL...] | @<ref>",
                    help="特征面白名单(单源写 config):缺省=全列−排除列"
                         "(TaskSpec features 派生,config 只存 exclude);"
                         "显式列名写入本 trial config 的 features.columns。"
                         "引用模式 @<ref> 避免长列表进命令行/上下文:"
                         "@<trial_id>=继承该 trial config 的 features,"
                         "@<trial_id>:feature_importance=按 gain 累计 share"
                         "≈0.95 取头部列,@<trial_id>:group_importance="
                         "按族裁决 families_alive ∪ families_unverifiable"
                         "展开(无证据≠死,v15.2),@<file.yaml>=读取"
                         "features 映射或列名清单(相对路径按 run 目录)")
    # [v2.13.2] 特征 bundle 宽度门与 screening 标记
    sp.add_argument("--override-beam", metavar="REASON",
                    help="v2.13.2:feature_exploration.bundle_beam_width "
                         "槽位豁免理由——本代特征 trial 数已达宽度上限时,"
                         "显式给出理由放行(留痕 trial.yaml.beam_override)")
    sp.add_argument("--screen-only", action="store_true",
                    help="v2.13.2:screening-only trial(宽面初筛)——贡献 "
                         "group_importance 先验,但不参与任何裁决:不进代级 "
                         "FDR/alpha 台账、不计入 accepted、不作冠军/Archive/"
                         "crossover 父代、不入方向台账。仍走完整评估并出现在 "
                         "leaderboard(带 screen_only 列,可审计)")
    # [v2.15.0] HPO 引擎开关(缺省取 run 配置 evolution.hpo.*)与两道门
    sp.add_argument("--hpo-lr-mode", choices=["auto", "search", "shrink"],
                    help="v2.15.0:hpo 的 lr 模式。search=lr 进搜索空间;"
                         "shrink=固定 search_lr 搜结构/正则,重训 lr×"
                         "lr_shrink_factor、cap×1/factor;auto=inner 行数≥"
                         "lr_shrink_min_rows 时 shrink(默认取配置)。v2.16.3:"
                         "显式 shrink 时空间里的 learning_rate 维被移除(不再静默"
                         "回落 search);生效模式写 trial.yaml 的 "
                         "hpo.lr_mode_effective / hpo-study.json / metrics,"
                         "请求≠生效时创建期与运行期 WARNING")
    sp.add_argument("--hpo-parallel", metavar="N|auto",
                    help="v2.15.0:inner 并行 config 数(同一张已租 GPU 上"
                         "线程并发;auto=GPU 上 xgboost 取 2,其余 1)")
    sp.add_argument("--hpo-fidelity", choices=["single", "successive_halving"],
                    help="v2.15.0:多保真逐级减半(1/9→1/3→1 行比例,"
                         "eta=3);默认 single")
    sp.add_argument("--hpo-no-expand", action="store_true",
                    help="v2.15.0:关闭同 trial 内的边界自适应外扩")
    sp.add_argument("--hpo-no-warm-start", action="store_true",
                    help="v2.15.0:不写 config/hpo-warm-start.json(不复用"
                         "签名一致的历史观测)")
    sp.add_argument("--hpo-search-lr", type=float, default=None,
                    metavar="LR",
                    help="v2.16.2:lr 收缩模式下 inner 搜索固定的 lr(缺省取 "
                         "evolution.hpo.search_lr)")
    sp.add_argument("--hpo-lr-shrink-factor", type=float, default=None,
                    metavar="F",
                    help="v2.16.2:lr 收缩模式下重训 lr = search_lr × F、cap × "
                         "1/F(钳到树数上限;缺省取 evolution.hpo."
                         "lr_shrink_factor)")
    sp.add_argument("--override-hpo-diminishing", metavar="REASON",
                    help="v2.15.0:同特征面上一轮 HPO 的 Δ_parent < MDE 时,"
                         "再建同面 HPO 需显式理由(留痕 trial.yaml."
                         "hpo_diminishing_override;等价于把预注册的"
                         "'Δ≤MDE 即证伪'判据交给机器执行)")
    sp.add_argument("--allow-confirm-during-hpo", metavar="REASON",
                    help="v2.15.0:父代仍有在途 HPO 子代(PROPOSED/"
                         "CODE_WRITTEN/RUNNING)时 confirm 需显式理由 —— "
                         "该配方大概率当代即被超越(run3 trial-019/020)")
    sp.add_argument("--topology", default=None,
                    choices=list(_TOPOLOGY_DECLARABLE),
                    help="v2.16.0:声明模型拓扑(缺省:--from-parent 继承父代"
                         "声明,否则 single)。声明只增加需批准项,不能抵消"
                         "运行前的静态检测;cascade/ensemble/白名单外"
                         "structural 须用户在自己的终端 approve-topology")
    sp.add_argument("--hurdle-template", action="store_true",
                    help="v2.17.0:渲染官方 hurdle trial(regression;"
                         "runtime.hurdle.HurdleRegressor:p·max(μ,0)、两段各自"
                         "早停、可 clone、不暴露 predict_proba、子模型缓存),"
                         "隐含 --topology structural:hurdle;自带 hurdle 超参段"
                         "(--set clf__/reg__ 分段键、--action hpo --from-parent"
                         " 直接可用)")
    sp.add_argument("--add-blocks", metavar="TRIAL:BLOCK[,TRIAL:BLOCK...]",
                    help="v2.17.0:从源 trial 冻结的 code/train.py 字节级复制"
                         "特征块段(块名 = feature_block 名 / 包装函数名;含依赖"
                         "的模块级常量与辅助函数)并注册进 trial_feature_blocks;"
                         "创建期校验内核指纹与源 trial 一致(保证缓存命中),"
                         "不一致拒绝并给差异")
    # v2.19.0(#15):补丁式写码
    sp.add_argument("--patch", metavar="PATCH_PY",
                    help="v2.19.0:配 --from-parent。patch 文件只写新增/替换的顶层"
                         "定义,harness 以父代 train.py 为底按 AST 合并(同名替换、"
                         "新名进 patch 段;PATCH_REGISTER_BLOCKS=[...] 注册特征块;"
                         "可选 PATCH_REPLACES=[...] 校验替换面);受保护符号拒绝。"
                         "可与 --set/--add-blocks 组合;confirm/automl 不可用")
    sp.set_defaults(func=cmd_new_trial)

    # v2.19.0(#15):父代大纲(只读),写 patch 前用它代替读整份 train.py
    sp = sub.add_parser(
        "trial-skeleton",
        help="v2.19.0:只读输出 trial code/train.py 大纲(顶层定义/签名/特征块"
             "列前缀与依赖行数/受保护标记/段标记)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--trial", required=True, metavar="TRIAL_ID")
    sp.add_argument("--json", action="store_true", help="输出 JSON")
    sp.add_argument("--no-probe", action="store_true",
                    help="不起探针进程(特征块只给静态 columns 表达式)")
    sp.set_defaults(func=cmd_trial_skeleton)

    sp = sub.add_parser(
        "approve-topology",
        help="v2.16.0:人工批准级联/融合拓扑(仅用户在交互式终端执行;"
             "Agent 会话被 guard hook 拒绝)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--list", action="store_true",
                    help="只读:列出账本与待批准请求(不需要终端)")
    sp.add_argument("--trial", help="批准该 trial 的检测拓扑")
    sp.add_argument("--scope", choices=["trial", "recipe", "run"],
                    help="trial=仅该 trial 且代码不变;recipe(--trial 缺省)"
                         "=该 trial 及其后代,签名不超出;run=本 run 内某类"
                         "拓扑(配 --kind)")
    sp.add_argument("--kind", help="--scope run 时:cascade,ensemble,structural"
                                   "(逗号分隔)")
    sp.add_argument("--revoke", metavar="APPROVAL_ID", help="撤销一条批准")
    sp.add_argument("--reason", help="批准/撤销理由(写入账本)")
    sp.set_defaults(func=cmd_approve_topology)

    sp = sub.add_parser("delete-trial",
                        help="[v2.9] 删除未启动(PROPOSED)的 trial,"
                             "带产物/引用检查与留痕日志——替代契约外 rm -rf")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--trial", required=True, metavar="TRIAL_ID")
    sp.add_argument("--reason", help="删除原因(留痕 delete-log.jsonl)")
    sp.add_argument("--force", action="store_true",
                    help="豁免产物检查(仅限确认误建;引用检查不可豁免)")
    sp.set_defaults(func=cmd_delete_trial)

    sp = sub.add_parser("run", help="执行一个 Trial(阻塞到终态,结尾输出 SUMMARY 一行摘要)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--trial", required=True)
    sp.add_argument("--wait", action="store_true",
                    help="显式阻塞语义(默认已阻塞;保留供脚本/文档引用)")
    sp.add_argument("--summary", action="store_true", default=True,
                    help="输出 SUMMARY\\t<tab分隔>一行结构化摘要"
                         "(trial/status/verdict/metric/delta/shadow/failure)")
    sp.add_argument("--reset-breaker", action="store_true",
                    help="重置失败熔断(需先完成 root-cause reflection)")
    sp.add_argument("--slot-index", type=int, default=0,
                    help="v18:批内槽位序号——手动并行多个 run 时按序号"
                         "绑 scheduler 多卡分配(gpu_indices[slot %% len]);"
                         "单 trial 独占无需关心")
    _add_detach_flags(sp)
    sp.set_defaults(func=cmd_run_trial)

    sp = sub.add_parser(
        "wait",
        help="v17.2:确定性等待 trial 终态(替代 sleep+tail 轮询)"
             "——终态输出与 run 同格式的 SUMMARY 行;v2.11 检测孤儿"
             "RUNNING(锁死+心跳停摆)打印 STALE_RUNNING 并 exit 4;"
             "v2.17.0 轻量快路径。wait 只读不 kill:被杀(exit 137/143)"
             "时原样重调即可,训练不受影响",
        description=_WAIT_HELP)
    # v2.17.0:参数定义单源在 _mvo/cmd_wait.add_wait_arguments(快路径共用)
    _add_wait_arguments(sp)
    sp.set_defaults(func=cmd_wait)

    sp = sub.add_parser("run-generation",
                        help="并行执行一代 Trial(有期限 barrier;"
                             "v19.4 跳过 PROPOSED 并可选等待代码落盘)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--generation", required=True, type=int)
    sp.add_argument("--wait-code", type=float, default=0, metavar="SEC",
                    help="有 PROPOSED trial 时等待代码落盘的秒数"
                         "(默认 0 = 立即跳过未就绪 trial)")
    _add_detach_flags(sp)
    sp.set_defaults(func=cmd_run_generation)

    sp = sub.add_parser(
        "finalize-retrain",
        help="v19.4:冠军代码在 train+validation 重训并打分 final_test/"
             "shadow_test,落盘 final-model/ 全套产物(收尾工具化,"
             "替代手写 retrain 脚本)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--champion", default=None, metavar="TRIAL_ID",
                    help="覆写确定性选型(与 evaluate-final --best-trial "
                         "同款人工入口;缺省 = leaderboard.best_trial)")
    sp.add_argument("--snapshot-only", action="store_true",
                    help="v2.13.4:无 fresh confirmed 冠军时只落证据快照、"
                         "不做未确认重训(默认会对最强合法候选按同口径重训到 "
                         "final-model/unconfirmed/model/)")
    sp.add_argument("--ack-divergence", default=None, metavar="REASON",
                    help="v2.14.0:--champion 指向与搜索冠军配方不同、且落后"
                         "超过容差的已确认 trial 时,写交付位需此用户裁决理由"
                         "(留痕 run-state.delivery_divergence_acks)")
    _add_refit_flags(sp)
    _add_detach_flags(sp)
    sp.set_defaults(func=cmd_finalize_retrain)

    sp = sub.add_parser(
        "finalize-baseline-score",
        help="v19.4:baseline 配方 train-only 重训并对 final_test 打分 → "
             "final-model/baseline_final_test_score.parquet(样本外口径"
             "分母;不进反馈环)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--baseline", default=None, metavar="TRIAL_ID",
                    help="baseline trial id(缺省 = run-state.baseline_id)")
    _add_detach_flags(sp)
    sp.set_defaults(func=cmd_finalize_baseline_score)

    sp = sub.add_parser(
        "finalize-all",
        help="v2.13.5:finalize-retrain ∥ finalize-baseline-score 并行"
             "(两者产物互不相交;串行实测 65+44 min)。不含 evaluate-final")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--champion", default=None, metavar="TRIAL_ID")
    sp.add_argument("--baseline", default=None, metavar="TRIAL_ID")
    sp.add_argument("--snapshot-only", action="store_true")
    sp.add_argument("--ack-divergence", default=None, metavar="REASON",
                    help="透传给 finalize-retrain(见其 --ack-divergence)")
    _add_refit_flags(sp)
    _add_detach_flags(sp)
    sp.set_defaults(func=cmd_finalize_all)

    sp = sub.add_parser(
        "contrast",
        help="v2.13.5:trial 两两(或对 --ref)的 validation 配对 bootstrap "
             "对照(harness 快路径;替代手写 bootstrap 脚本)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--trials", required=True, help="逗号分隔 trial id")
    sp.add_argument("--ref", default=None, help="只做各 trial 对 ref 的对照")
    sp.add_argument("--metric", default=None, help="缺省 spec.primary_metric")
    sp.add_argument("--n-boot", type=int, default=2000)
    sp.add_argument("--seed", type=int, default=20260814)
    sp.add_argument("--out", default=None, help="结果 JSON 路径")
    sp.set_defaults(func=cmd_contrast)

    # v2.17.0(WP5b #16):train 内时间切片预筛(screening 证据,不进 FDR)
    sp = sub.add_parser(
        "prescreen",
        help="v2.17.0:只用 train 面(前段拟合/中段早停/尾段留出)在沙箱里"
             "多 seed 运行候选与对照 trial 代码,输出 Δ/配对 SE;"
             "screening_evidence,不进 FDR/leaderboard,不改状态")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--trial", required=True,
                    help="候选 trial(CODE_WRITTEN 或更后状态)")
    sp.add_argument("--ref", default=None, help="对照 trial(缺省第一父代)")
    sp.add_argument("--seeds", type=int, default=3, help="配对 seed 数 K")
    sp.add_argument("--holdout-frac", type=float, default=0.15)
    sp.add_argument("--es-frac", type=float, default=0.05)
    sp.add_argument("--max-rows", type=int, default=None,
                    help="train 等距抽样上限(缺省全量 train)")
    sp.add_argument("--metric", default=None, help="缺省 spec.primary_metric")
    sp.add_argument("--n-boot", type=int, default=1000)
    sp.add_argument("--threads", type=int, default=None)
    sp.add_argument("--gpu", default=None, help="GPU 序号(缺省 CPU)")
    sp.add_argument("--timeout-min", type=float, default=120.0,
                    help="单次运行超时(分钟)")
    sp.add_argument("--keep-sandbox", action="store_true",
                    help="保留沙箱 trial 副本(诊断用)")
    sp.set_defaults(func=cmd_prescreen)

    sp = sub.add_parser(
        "timing-report",
        help="v2.13.5:run 耗时拆解(训练分项 / 评估分项 / 代际窗口与间隔),"
             "落 logs/timing-report.json")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_timing_report)

    sp = sub.add_parser(
        "feature-cache",
        help="v2.13.5:列出/清空 RUN_DIR/.feature-cache(特征块跨 trial 缓存);"
             "v2.17.0 附失效预判与 .submodel-cache 汇总")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--clear", action="store_true")
    sp.add_argument("--clear-submodel", action="store_true",
                    help="v2.17.0:清空 RUN_DIR/.submodel-cache(子模型缓存)")
    sp.set_defaults(func=cmd_feature_cache)

    sp = sub.add_parser("finalize-generation", help="Generation 收尾")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--generation", required=True, type=int)
    sp.add_argument("--unexplored", help="逗号分隔未探索高潜力方向")
    sp.set_defaults(func=cmd_finalize_generation)

    sp = sub.add_parser("evaluate-final", help="final_test 一次性盲测")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--allow-no-shadow", action="store_true",
                    help="已弃用的兼容参数；不会绕过独立 shadow 确认门")
    sp.add_argument("--best-trial",
                    help="显式指定交付冠军 trial_id；仅改变候选选择，"
                         "不能绕过 confirmed、独立 shadow 确认和完整性门")
    sp.add_argument("--ack-divergence", default=None, metavar="REASON",
                    help="v2.14.0:交付模型与搜索冠军配方分叉且落后超过容差时,"
                         "盲测需此用户裁决理由才放行(留痕;缺省在 gate 前拒绝)")
    sp.add_argument("--ack-shadow-regression", default=None, metavar="REASON",
                    help="v2.15.9:finalize-retrain 的 shadow 预检显示交付模型"
                         "显著劣于冠军 train-only 模型时,盲测需此用户裁决理由"
                         "才放行(留痕;缺省在 gate 前拒绝)")
    sp.add_argument("--segment-inverse-transform", choices=("none", "expm1"),
                    default="none",
                    help="v2.15.9(regression):在同一次内存标签读取中,对 "
                         "final_test/shadow_test 额外产出原值尺度分桶报表 "
                         "final-model/segment-raw/<split>/ 与 evaluation."
                         "raw_scale(目标为 log1p 时用 expm1)")
    sp.set_defaults(func=cmd_evaluate_final)

    sp = sub.add_parser(
        "release-final-test",
        help="将最近一次 final_test touch 标记为中止/失效；不会恢复"
             "盲测次数，再评测必须换独立 holdout 或新 run")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--reason", required=True,
                    help="中止理由(写入 gate 与 run-state 审计记录)")
    sp.set_defaults(func=cmd_release_final_test)

    sp = sub.add_parser("report", help="生成最终报告(report.md + 训练历程 report.html)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--no-html", action="store_true",
                    help="v2.16.1:只写 report.md,不生成训练历程 report.html")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("leaderboard", help="重建并打印 Leaderboard")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.set_defaults(func=cmd_leaderboard)

    sp = sub.add_parser("status", help="Run 状态")
    sp.add_argument("--run")
    sp.add_argument("--spec")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser(
        "plan", help="确定性生成下一代结构化 proposal batch")
    sp.add_argument("--spec")
    sp.add_argument("--run", required=True)
    sp.add_argument("--slots", type=int, default=4,
                    help="批次候选数(受剩余 trial 预算约束,默认 4)")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser(
        "benchmark", help="benchmark:evaluate(缺省,--results 发布质量门)"
                          " | prepare(下载套件+生成 run)| report(S_base/S_mvo/S_ref)")
    from _mvo.cmd_benchmark import add_benchmark_arguments  # v2.19.0
    add_benchmark_arguments(sp)

    sp = sub.add_parser(  # v2.19.0 强参照 + 天花板估计
        "ceiling", help="强参照/天花板估计(validation 面),写 ceiling/ceiling-estimate.json")
    from _mvo.cmd_benchmark import add_ceiling_arguments
    add_ceiling_arguments(sp)

    sp = sub.add_parser("diagnose",
                        help="故障 triage 一键诊断(v17):trial 为何不在"
                             "leaderboard/verdict 异常/产物缺失的确定性"
                             "归因与处置建议(只读;主会话遇异常先跑它)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--trial",
                    help="聚焦单个 trial id;缺省全量扫描")
    sp.set_defaults(func=cmd_diagnose)

    sp = sub.add_parser("write-splits-view",
                        help="数据视图 splits 落盘(协议 v7)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--view-id", required=True,
                    help="视图 id(非 v1);产物 sample-profile/splits-{id}")
    sp.add_argument("--source",
                    help="build_view 脚本路径(先落盘,如 views/{id}.py);"
                         "缺省时须给 --template")
    sp.add_argument("--template", choices=["recency", "active", "kwin"],
                    help="v16/P1-1:视图模板——参数化生成 views/{id}.py "
                         "骨架(recency=近因指数加权/active=活跃子集统计/"
                         "kwin=近 K 期窗口重聚合),保留人工审查点")
    sp.add_argument("--half-life", type=float, default=30.0,
                    help="recency 模板半衰期(天,默认 30)")
    sp.add_argument("--k-periods", type=int, default=3,
                    help="kwin 模板近 K 期(默认 3)")
    sp.add_argument("--active-col", default=None,
                    help="active 模板的活跃指示列名(如登录天数列;"
                         "缺省自动探测含 _days 的最高覆盖列)")
    sp.add_argument("--note", help="视图说明(记入 manifest)")
    # v2.17.0(WP5b #12):流式构建 / 内存上限 / 特征块材料化
    sp.add_argument("--max-mem-gb", type=float, default=None,
                    help="流式构建的内存上限(GB):按行宽估块大小,RSS 逼近"
                         "上限时块减半;缺省读 feature_exploration."
                         "view_stream_max_mem_gb")
    sp.add_argument("--chunk-rows", type=int, default=None,
                    help="流式块行数(缺省 feature_exploration."
                         "view_stream_chunk_rows=200000)")
    sp.add_argument("--no-stream", action="store_true",
                    help="旧全量模式(一次拼出 df_v1 再调 build_view;"
                         "仅小数据/非行内变换脚本)")
    sp.add_argument("--from-feature-block", default=None,
                    metavar="TRIAL:BLOCK[,...]",
                    help="把 trial 的特征块(feature_block 块名或块函数名)"
                         "直接材料化为视图列(缓存命中/按内核重算 + 逐值"
                         "校验),不重载原始数据;与 --source/--template 互斥")
    sp.set_defaults(func=cmd_write_splits_view)

    sp = sub.add_parser(
        "write-splits-sample",
        help="落盘特征探索抽样面(协议 v10:各 split 按比例等距抽样,"
             "final_test 不动;产物 sample-profile/splits-sample)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--max-rows", type=int, default=None,
                    help="抽样上限(默认读 feature_exploration."
                         "max_sample_rows,缺省 20 万;train+validation+"
                         "shadow 合计按比例分摊)。显式传入覆盖配置")
    sp.add_argument("--note", help="抽样说明(记入 manifest)")
    sp.set_defaults(func=cmd_write_splits_sample)

    sp = sub.add_parser("export-experience",
                        help="v2.18.0:提交本 run 的经验快照到经验库(合并见 "
                             "references/experience-library-merge.md)")
    sp.add_argument("--run")
    sp.add_argument("--spec")
    sp.add_argument("--model", default=os.environ.get("MODELEVO_MODEL", ""),
                    help="收尾所用模型名(记入快照与 actor)")
    sp.set_defaults(func=cmd_export_experience)

    sub.add_parser("config-template", help="打印配置模板") \
        .set_defaults(func=cmd_assets_config)
    sp = sub.add_parser(
        "snapshot",
        help="资源快照与并发计划(v2.17.0:含系统内存/cgroup 限额、按内存"
             "准入的建议并发、各卡外来进程与占用;给 --run 时按该 run 最近 "
             "trial 峰值估计)")
    sp.add_argument("--run")
    sp.add_argument("--spec")
    sp.set_defaults(func=cmd_snapshot)
    sp = sub.add_parser(
        "planner-context",
        help="一键组装 Planner 八类上下文(spec/profile/leaderboard/"
             "archive/budget/data_views/经验库视图/反思)")
    sp.add_argument("--spec")
    sp.add_argument("--run")
    sp.add_argument("--top-k", type=int, default=5,
                    help="leaderboard / archive 展示行数")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_planner_context)
    add_library_parser(sub)
    add_experience_parser(sub)
    return p


def main(argv=None) -> int:
    _argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_argv)
    # v2.13.3:detach 需要按原样重建 worker 命令。测试直接调 main(argv)
    # 时 sys.argv 是 pytest 的,不能用;以解析用的那份为准。
    args._raw_argv = _argv
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        # [FIX-2026-09-03] new-trial 失败事务化:非预期异常(模板渲染
        # KeyError 等)时回滚本次创建的 PROPOSED 孤儿目录。SystemExit
        # 失败路径已在 cmd_new_trial 内就地回滚,不经此处。
        if getattr(args, "command", "") == "new-trial" and \
                context._LAST_CREATED_TRIAL_DIR is not None:
            class _T:  # 最小 duck-type:只需 trial_dir
                pass
            _t = _T()
            _t.trial_dir = context._LAST_CREATED_TRIAL_DIR
            render._rollback_trial(_t)
        logger.error("%s 失败: %s", args.command, exc, exc_info=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
