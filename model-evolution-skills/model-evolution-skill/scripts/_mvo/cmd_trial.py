"""演进循环命令:new-trial / run / run-generation / finalize-generation

v2.11.9 从单文件 `scripts/modelevo.py` 拆出(拆分动机见
`_mvo/__init__.py`)。**本文件是机械搬运,未改语义** —— 便于对照原
实现审阅。新增命令请按职责放进对应模块,不要回填到 `modelevo.py`
(那里只留 argparse 与分发)。
"""
from __future__ import annotations

from _mvo.templates import AUTOML_BODY, TRIAL_BODY  # noqa: E402
from runtime.memory import (EvidenceStore,  # noqa: E402
                            TaskExperience)
from runtime.evolution import EvolutionEngine  # noqa: E402
from runtime.feature_phase import (feature_phase_status,  # noqa: E402
                                   is_model_direction, planner_release)
from runtime.leaderboard import Leaderboard  # noqa: E402
from pathlib import Path
from runtime.scheduler import Scheduler, resource_snapshot  # noqa: E402
from runtime.templates import (DATA_LOADER_TEMPLATE, TASK_TEMPLATE,  # noqa: E402
                               TRIAL_TEMPLATE)
from runtime.trial import (TrialFactory,  # noqa: E402
                           confirmation_parent_eligibility)
from runtime.runner import TrialRunner, load_parent_scores  # noqa: E402
from runtime.logging_utils import (dump_yaml, get_logger,  # noqa: E402
                                   load_json, load_yaml, locked_state_write,
                                   now_iso)
import json
import os
import sys
from runtime.harness import protocol_stamp  # noqa: E402
from runtime.data import load_dataframe, resolve_feature_columns  # noqa: E402
import time
from .context import _SKILL_ROOT, _auto_ack_plumbing, _load_best_parent_scores, _load_config, _load_spec, _parse_list, _resolve_baseline_id, _run_dir_from, _started_at, _trial_summary_line, _update_run_state, logger
from .render import _HPO_RENDER_ALGOS, _copy_parent_trial_code, _ensure_hyperparameters_consumed, _hpo_parent_face_risk, _inject_hpo_algo_at, _inject_hpo_config, _load_features_ref, _parse_overrides, _rollback_trial


def _count_feature_trials(factory, generation: int) -> int:
    """本代已创建的特征类 trial 数(bundle_beam_width 槽位口径)。

    [v2.13.2] 计入 changed_modules 含 feature/sample/data 的 explore 与
    data_view trial(它们都在消费特征/信息面),以及 screening trial
    (宽面初筛同样占一个槽位 —— 总槽位数 = bundle_beam_width,不是
    "N 个 bundle 之外再送一个宽 trial")。不含 model/loss/hpo/confirm。
    """
    n = 0
    for t in factory.all_trials():
        if t.generation != generation:
            continue
        d = t.data
        if bool(d.get("screen_only")):
            n += 1
            continue
        if str(d.get("search_action") or "") not in ("explore", "data_view"):
            continue
        mods = {str(m).strip().lower()
                for m in (d.get("changed_modules") or [])}
        if mods & {"feature", "sample", "data"}:
            n += 1
    return n


def _hpo_opts_from_args(args) -> dict:
    """[v2.15.0] new-trial 的 HPO 引擎开关 → hpo 节覆写(缺省取 run 配置)。"""
    o: dict = {}
    if getattr(args, "hpo_lr_mode", None):
        o["lr_mode"] = str(args.hpo_lr_mode)
    _par = getattr(args, "hpo_parallel", None)
    if _par:
        _s = str(_par).strip().lower()
        o["parallel_configs"] = int(_s) if _s.isdigit() else _s
    if getattr(args, "hpo_fidelity", None):
        o["fidelity"] = str(args.hpo_fidelity)
    if getattr(args, "hpo_no_expand", False):
        o["auto_expand"] = False
    if getattr(args, "hpo_no_warm_start", False):
        o["warm_start"] = False
    # v2.16.2:lr 收缩模式的两个旋钮也可按 trial 覆写(此前只能改 run 配置
    # evolution.hpo.search_lr / lr_shrink_factor;hurdle 冠军 lr 0.025 在 run 级
    # 0.05×0.25=0.0125 下重训会撞树数上限,见 render._hurdle_lr_cap_advice)
    if getattr(args, "hpo_search_lr", None) is not None:
        o["search_lr"] = float(args.hpo_search_lr)
    if getattr(args, "hpo_lr_shrink_factor", None) is not None:
        o["lr_shrink_factor"] = float(args.hpo_lr_shrink_factor)
    return o


def _confirm_effective_multi_seed(declared, code: str) -> int:
    """[v2.16.2] confirm 子代 config.multi_seed 应写的值 = 代码实际会跑的 seed 数。

    模板与 baseline 血缘的 multi-seed 段:`multi_seed ≤ 1` 且
    `search_action == "confirm"` 时默认实跑 3 个 seed。旧脚手架写
    max(1, 父代值) = 1,证据对账判 not_declared(run4 regression trial-008:
    declared 1 / found 2)。代码没有 multi-seed 段(不含 seed_scores)时
    保持声明值(创建期已告警,对账会判 missing)。
    """
    try:
        d = int(declared or 0)
    except (TypeError, ValueError):
        d = 0
    if d > 1:
        return d
    return 3 if "seed_scores" in (code or "") else 1


def _hpo_diminishing_check(run_dir: Path, ref_id: str | None,
                           config) -> dict | None:
    """[v2.15.0] 同特征面 HPO 链的边际递减门。

    ref_id 是新 HPO trial 的父代。沿父链找到同面最近一次 HPO(父代本身
    是 hpo,或父代是 hpo 配方的 confirm),若其 Δ_parent < MDE
    (evaluator.minimum_meaningful_improvement)即返回证据 dict。

    依据(req-t7-rand400w run3):第二轮 HPO 预注册"Δ ≤ MDE 则证伪冠军面
    仍有可兑现的模型轴余量",实测 +0.00084 < 0.001 已触发,仍以"衰减但
    非同阶归零"开第三轮 —— +0.00022(¼ MDE)、2.8 GPU 小时 + 1 小时
    confirm,且收尾时又贴住同样的边界。把这条判据交给机器执行。
    """
    if not ref_id:
        return None
    from runtime.trial import TrialRecord as _TR

    tdir = Path(run_dir) / "trials"
    try:
        ref = _TR.load(tdir / ref_id)
    except Exception:  # noqa: BLE001
        return None
    last = None
    if str(ref.data.get("search_action") or "") == "hpo":
        last = ref
    elif str(ref.data.get("search_action") or "") == "confirm" and \
            ref.parent_ids:
        try:
            _p = _TR.load(tdir / ref.parent_ids[0])
            if str(_p.data.get("search_action") or "") == "hpo":
                last = _p
        except Exception:  # noqa: BLE001
            last = None
    if last is None:
        return None
    ev = last.data.get("evaluation") or {}
    delta = (ev.get("delta") or {}).get("vs_parent")
    if not isinstance(delta, (int, float)):
        return None
    # v2.17.0:量纲感知 MDE(evaluator.minimum_meaningful_improvement 绝对值与
    # *_relative × |baseline 主指标| 取大;原值 rmse 回归不再被 0.001 架空)
    from runtime.mde import effective_mde

    mde = float(effective_mde(config, run_dir=run_dir))
    if float(delta) >= mde:
        return None
    # v2.17.2(A1):上一轮 HPO 的小/负 Δ 若是撞 cap 混杂(重训被树数上限截断,
    # run4 trial-017 伪负 −0.00095),不构成"边际递减"证据——否则去混杂后的
    # 同面续搜会被这道门误拦。旧产物无标记时按 metrics.es_info 重算。
    if ev.get("cap_confounded"):
        return None
    try:
        from runtime.evaluator import cap_confounded_detail
        from runtime.metrics import metric_direction

        _m = json.loads((Path(last.trial_dir) / "metrics.json").read_text(
            encoding="utf-8"))
        _pm = (ev.get("primary_metric") or {}).get("name") or "auc"
        if cap_confounded_detail(ev, es_info=_m.get("es_info"),
                                 higher_is_better=metric_direction(_pm)):
            return None
    except Exception:  # noqa: BLE001 旧产物缺 metrics.json 时按原判据
        pass
    try:
        from runtime.hpo import assess_hpo_study

        quality = assess_hpo_study(
            Path(last.trial_dir), default_budget=int(config.get(
                "evolution.hpo.default_budget", 40) or 40))
    except Exception:  # noqa: BLE001
        quality = {}
    return {"last_hpo": last.trial_id, "delta_vs_parent": float(delta),
            "mde": mde, "quality": quality}


def _inflight_hpo_children(factory, parent_id: str) -> list[str]:
    """[v2.15.0] 父代仍在途(未出结果)的 HPO 子代。"""
    out = []
    for t in factory.all_trials():
        if str(t.data.get("search_action") or "") != "hpo":
            continue
        if parent_id not in (t.parent_ids or []):
            continue
        if t.status in ("PROPOSED", "CODE_WRITTEN", "RUNNING"):
            out.append(t.trial_id)
    return out


def _auto_hypothesis(action: str, parent_id: str | None) -> dict | None:
    """[v2.17.0 D5] 可确定性生成假设的动作(当前仅 confirm);其余返回 None。

    confirm = 同配方换 seed 的独立复现,假设/证伪口径与确认门一致
    (validation 非劣 + shadow 保真),不需要人填写。
    """
    if action == "confirm" and parent_id:
        return {
            "statement": (f"确认 {parent_id} 配方在新 seed 与 shadow 上非劣且保真"
                          "(同配方独立复现,不提出新增益)"),
            "rationale": (f"{parent_id} 已过选择校正、尚未通过独立确认;交付位"
                          "必须绑定已确认的冠军配方"),
            "falsification": ("新 seed 在 validation 或独立 shadow 上显著不及父代"
                              "(非劣下界含 seed 方差),即确认失败"),
        }
    return None


BASIS_RELATIONS = ("follows", "extends", "tests", "contradicts")


def _parse_basis(raw, legacy_ids) -> tuple[dict | None, str | None]:
    """--basis / --experience-ids → ({novel, refs:[{id, relation}]}, 错误)。"""
    items = [x.strip() for x in str(raw or "").split(",") if x.strip()]
    if not items and not legacy_ids:
        return None, None
    if [x.lower() for x in items] == ["novel"]:
        if legacy_ids:
            return None, "--basis novel 不能与 --experience-ids 同时使用"
        return {"novel": True, "refs": []}, None
    refs = []
    for x in items:
        if x.lower() == "novel":
            return None, "novel 不能与原则引用混用"
        pid, _, rel = x.partition(":")
        rel = (rel or "follows").strip().lower()
        if rel not in BASIS_RELATIONS:
            return None, f"--basis {x!r} 的关系须为 {BASIS_RELATIONS} 之一"
        refs.append({"id": pid.strip(), "relation": rel})
    for pid in legacy_ids or []:
        if pid not in {r["id"] for r in refs}:
            refs.append({"id": pid, "relation": "follows"})
    return {"novel": False, "refs": refs}, None


def _basis_source(basis: dict, prefill) -> str:
    """依据来源(v2.18.6):``prefilled`` = 原样沿用 plan 预填的依据;``agent`` = Agent
    自己写或改写过。预填是程序按主题命中给的,不代表 Agent 真的依据了这条经验。"""
    pre, _err = _parse_basis(prefill, None)
    if not pre or pre.get("novel"):
        return "agent"
    key = lambda b: sorted((r["id"], r["relation"]) for r in b.get("refs") or [])  # noqa: E731
    return "prefilled" if key(pre) == key(basis) else "agent"


def cmd_new_trial(args) -> int:
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    config = _load_config(run_dir)
    factory = TrialFactory(run_dir, spec=spec, config=config)
    # ensemble 探索(trial 级融合)仍未实现:显式开启时拒绝而不是静默
    # 按单模型执行(§3.2)。v2.16.0 起收尾期 `ensemble-final` 也已删除
    # (它建议的融合交付没有重训/final_test/交付通道,且在选择面上自评),
    # 本版不提供任何融合通道。
    if config.get("ensemble.enabled", False):
        logger.error("ensemble.enabled=true 但本版不支持任何模型融合"
                     "(trial 级融合未实现,收尾期 ensemble-final 已删除);"
                     "请关闭后使用单模型 crossover")
        return 1
    # v2.17.0(#22):--changed-modules 严格枚举。旧实现只在 factory.create 里
    # 告警(run4 regression.2 误填 sample_weight 被放行,只能事后 delete-trial;
    # 方向台账与 crossover 兼容性检查因此拿到无法归类的模块名)。
    from runtime.trial import VALID_CHANGED_MODULES

    _cm_bad = [m for m in _parse_list(getattr(args, "changed_modules", None))
               if str(m).strip().lower() not in VALID_CHANGED_MODULES]
    if _cm_bad:
        logger.error(
            "--changed-modules 含未知值 %s;允许值:%s(样本权重/采样口径属 "
            "sample,特征加工属 feature,模型结构/超参属 model,目标/损失变换属 "
            "loss,数据视图/切分属 data)", _cm_bad,
            ", ".join(VALID_CHANGED_MODULES))
        return 1
    if getattr(args, "add_blocks", None):
        if args.action in ("confirm", "automl"):
            logger.error("--add-blocks 不能用于 --action %s(confirm 必须是父代"
                         "配方的独立重训,不得改特征面)", args.action)
            return 1
        try:
            from .render import parse_add_blocks

            parse_add_blocks(args.add_blocks)
        except SystemExit as exc:
            logger.error("%s", exc)
            return 1
    # v2.19.0(#15):--patch 补丁式写码。与 --add-blocks 同口径不可用于
    # confirm/automl;必须有 --from-parent(以父代 train.py 为底合并)。创建
    # trial 前先对父代代码做一次合并预演,格式/受保护符号违规直接拒绝不落盘。
    if getattr(args, "patch", None):
        if args.action in ("confirm", "automl"):
            logger.error("--patch 不能用于 --action %s(confirm 必须是父代配方"
                         "的独立重训,automl 是零写码参照系)", args.action)
            return 1
        if not getattr(args, "from_parent", None):
            logger.error("--patch 需要 --from-parent <父代>(patch 以父代 "
                         "train.py 为底按 AST 合并)")
            return 1
        try:
            from .trial_patch import dry_run_patch

            dry_run_patch(run_dir, str(args.from_parent),
                          str(args.patch))
        except SystemExit as exc:
            logger.error("%s", exc)
            return 1
    # v2.17.0(#15):官方 hurdle 模板(runtime.hurdle.HurdleRegressor)
    if getattr(args, "hurdle_template", False):
        _ht_err = None
        if str(spec.model_type) != "regression":
            _ht_err = (f"--hurdle-template 只适用于 regression 任务(本任务 "
                       f"{spec.model_type})")
        elif getattr(args, "from_parent", None):
            _ht_err = ("--hurdle-template 是模板渲染路径,不能与 --from-parent 同用"
                       "(hurdle 父代的子代直接 --from-parent 即继承容器)")
        elif args.action not in ("explore", "exploit", "crossover", "hpo"):
            _ht_err = (f"--hurdle-template 不支持 --action {args.action}"
                       "(confirm 用 --from-parent)")
        elif str(getattr(args, "algorithm", None) or "xgboost").lower() \
                != "xgboost":
            _ht_err = "--hurdle-template 当前只提供 xgboost 两段(--algorithm xgboost)"
        elif getattr(args, "topology", None) not in (None, "structural:hurdle"):
            _ht_err = (f"--hurdle-template 与 --topology {args.topology} 冲突"
                       "(hurdle 模板即 structural:hurdle)")
        if _ht_err:
            logger.error(_ht_err)
            return 1
        args.algorithm = "xgboost"
        args.topology = "structural:hurdle"
    # hpo 动作(v11):搜索空间必填;父代两条路径(v2.9):
    # (a) --from-parent:复制父代码,要求父代码自带四件套
    #     (_assert_hpo_capable 校验)——保留给"父代已含定制 widen/
    #     特征逻辑要原样继承"的场景;
    # (b) 无 --from-parent:模板渲染 + 继承父代 config 特征面
    #     (--hpo-parent 指定时)或默认全列;--algorithm 显式指定,
    #     算法化四件套由模板 body 调 runtime.hpo.build_gbm 生成。
    # 旧实现强制 (a),而本 run 全部父代是 baseline 血缘+手写 widen
    # (无四件套)→ 被迫建 scaffold + 手改 7 处代码(gen-03 实测
    # 21 分钟 + 2 次 rm -rf)。两条路径并存,创建期由本分支裁决。
    _hpo_parent = getattr(args, "from_parent", None)
    if args.action == "hpo":
        if not getattr(args, "search_space", None):
            logger.error("--action hpo 需要 --search-space <yaml>"
                         "(参数分布声明)")
            return 1
        if not _hpo_parent:
            algo_l = str(getattr(args, "algorithm", "") or "").strip().lower()
            if algo_l and algo_l not in _HPO_RENDER_ALGOS:
                logger.error(
                    "--action hpo 无 --from-parent 时 --algorithm 仅支持 "
                    "%s(实测算法族);%r 请走 --from-parent 路径",
                    list(_HPO_RENDER_ALGOS), args.algorithm)
                return 1
            if not algo_l:
                logger.error(
                    "--action hpo 需要 --from-parent <trial_id>(在父代特征面"
                    "/代码上搜索),或无父代时显式 --algorithm "
                    "(%s,模板渲染路径:继承 --hpo-parent 的 config 特征面,"
                    "代码由模板按算法渲染,零写码)", list(_HPO_RENDER_ALGOS))
                return 1
    # [v2.15.0] 同特征面 HPO 链的边际递减门(预注册判据机器化):上一轮
    # 同面 HPO 的 Δ_parent < MDE 时,再开一轮需显式理由。
    _hpo_dim_override = None
    if args.action == "hpo":
        _dim = _hpo_diminishing_check(
            run_dir, _hpo_parent or getattr(args, "hpo_parent", None), config)
        if _dim:
            _q = _dim.get("quality") or {}
            _why_incomplete = "; ".join(_q.get("reasons") or []) or "无"
            if not getattr(args, "override_hpo_diminishing", None):
                logger.error(
                    "HPO 边际递减门:同特征面上一轮 HPO %s 的 Δ_parent=%+.6f "
                    "< MDE=%.4f —— 预注册的「Δ≤MDE 即证伪模型轴余量」已触发"
                    "(req-t7-rand400w run3 第三轮:+0.00022、2.8 GPU 小时 + "
                    "confirm 1 小时,收尾时仍贴同样边界)。上一轮搜索完整性:"
                    "%s。确需继续(如上一轮预算不足/最优顶边界/重训撞 cap)时加 "
                    "--override-hpo-diminishing \"理由\"(留痕 trial.yaml);"
                    "v2.15 会热启动复用上一轮全部观测并自动外扩边界,不要重开"
                    "同一空间。", _dim["last_hpo"], _dim["delta_vs_parent"],
                    _dim["mde"], _why_incomplete)
                return 1
            _hpo_dim_override = {
                "reason": str(args.override_hpo_diminishing),
                "last_hpo": _dim["last_hpo"],
                "delta_vs_parent": _dim["delta_vs_parent"],
                "mde": _dim["mde"],
                "previous_search_incomplete": _q.get("reasons") or [],
                "at": __import__("datetime").datetime.now().isoformat()}
    # v12(P1-3):特征面 trial 必须声明 changed_modules(方向台账与
    # crossover 兼容性检查的输入;实测 14 个 trial 中 8–9 个为空,
    # 台账退化成"按动作分四组")
    # v18/P2-3:automl 参照系不在该约束内(单行参照 trial,模板渲染)
    if not _parse_list(getattr(args, "changed_modules", None)) and \
            args.action in ("explore", "exploit", "data_view"):
        logger.error("--action %s 需要 --changed-modules(方向台账与 "
                     "crossover 兼容性检查的输入;缺省时该 trial 无法 "
                     "入账,方向归因退化为按动作分桶)",
                     args.action)
        return 1
    # v18/P2-3:automl 参照系——`--action automl --algorithm <autogluon|flaml>`
    # 零写码渲染外部 AutoML 参照 trial。语义:外部锚点,回答"成熟
    # AutoML 在当前特征面上能到哪",给 capacity 归因第三方证词;
    # **不计入交付候选**(leaderboard 参照行标注 reference_only,
    # best_trial 过滤),不进反馈环,铁律 12 的单模型约束不豁免。
    if args.action == "automl":
        _am = str(getattr(args, "algorithm", "") or "").strip().lower()
        if not _am:
            logger.error("--action automl 需要 --algorithm <autogluon|flaml>")
            return 1
        if _am not in ("autogluon", "flaml"):
            logger.error("--action automl 的 --algorithm 仅支持 "
                         "autogluon|flaml(可选依赖,缺失时 preflight "
                         "记 env-manifest、run 期显式报错)")
            return 1
    # 参数 trial 快通道(v10.3):--from-parent 复制父代代码,--set 注入
    # 超参差异,零 LLM 写码(纯参数网格点不需要读模板/写码/lint 轮)
    overrides = _parse_overrides(getattr(args, "set", None))
    from_parent = getattr(args, "from_parent", None)
    # 特征面白名单(v2.8,可选):显式传入时写进本 trial config 的
    # features.columns(train.py 运行期 exclude_columns 读取);缺省=
    # 全列−排除列(TaskSpec features 派生,仍单源存 config)。
    # 引用模式(@file/@trial_id/@trial_id:裁决产物)在派生特征面解析
    # 之后展开(需要 derived 列表做交集校验),这里先记原始串。
    features_arg = getattr(args, "features", None)
    features_override = None
    if features_arg and features_arg.strip().startswith("@"):
        features_override = features_arg.strip()  # 引用串,延后展开
    elif features_arg:
        features_override = _parse_list(features_arg)
    # hpo 快通道(v11)不需要 --set 覆写(搜索空间走 config.hpo 注入);
    # --features 白名单覆写同样构成有意义的差异(配置注入,代码零改动)
    # v2.13.5:--action data_view 的差异就是数据视图本身(如 sample 面锚点:
    # baseline 代码原样跑在抽样面),不再强制 --set/--features
    # v2.17.0:--add-blocks 同样构成有意义的差异(harness 字节级复制特征块段)
    if from_parent and not overrides and args.action not in (
            "hpo", "confirm", "data_view") \
            and not features_override and not getattr(
                args, "allow_code_edit", False) \
            and not getattr(args, "add_blocks", None) \
            and not getattr(args, "patch", None):
        logger.error("--from-parent 需要至少一个 --set key=value、--features、"
                     "--add-blocks 或 --patch；若要以父代作待编辑脚手架，显式传 "
                     "--allow-code-edit(运行前源码会冻结)")
        return 1
    hypothesis = {
        "statement": args.hypothesis or "(待填写:可证伪假设陈述)",
        "rationale": args.rationale or "",
        "falsification": args.falsification or "",
    }
    # v2.17.0(D5):confirm 的假设是确定性的("同配方换 seed 复现"),缺省时
    # 自动生成;其余动作的假设必须由人/Planner 填写 —— 缺省给清晰 CLI 错误
    # (exit 1、无 traceback),不再让 TrialFactory.create 的 ValueError 冒泡
    # 成 exit 2 + traceback。
    if not (args.hypothesis or "").strip():
        _auto = _auto_hypothesis(args.action, from_parent
                                 or (_parse_list(args.parents) or [None])[0])
        if _auto is None:
            logger.error(
                "--action %s 需要 --hypothesis \"<可证伪假设陈述,≥6 字>\"(只有 "
                "confirm 可自动生成假设;planner 提案的 materialize_command 已带"
                " --hypothesis/--rationale/--falsification,可直接照抄)",
                args.action)
            return 1
        hypothesis.update({k: v for k, v in _auto.items()
                           if not hypothesis.get(k)})
        hypothesis["statement"] = _auto["statement"]
        hypothesis["auto_generated"] = True
        logger.info("--hypothesis 缺省:已为 %s 自动生成假设:%s",
                    args.action, _auto["statement"])
    actions = (args.actions or ["(待填写:具体改动)"])
    changed_modules = _parse_list(args.changed_modules)
    parent_ids = _parse_list(args.parents) or None
    if args.action == "crossover" and parent_ids is not None \
            and len(parent_ids) < 2:
        logger.error("crossover 需要至少两个父 trial_id(--parents a,b)")
        return 1
    # v2.16.0:特征期门(口径见 runtime/feature_phase.py,与 Planner 同一
    # 判定)。取代 v16/P0-3 的 capacity 闸门:后者只在 capacity-report
    # 含 add_rows/new_information 且传了 --changed-modules 时生效(只写
    # --action hpo 即可绕过),且不看代数——Planner 期满已降 advisory,
    # 这里仍要人工 override(本机 76 个放行记录 75 个是人工)。
    _feature_phase_override = None
    if is_model_direction(args.action, changed_modules):
        _fp = feature_phase_status(factory.all_trials(), config)
        _ovr = getattr(args, "override_feature_phase", None)
        if _fp["active"]:
            # [FIX-2026-09-17] Planner 释放名额落地:new-trial 认最新批次
            # 的 released_proposal_id,否则 Planner 放行的候选在这一层仍
            # 被同一道门拦死。
            _rel = None if _ovr else planner_release(
                run_dir, getattr(args, "planner_proposal_id", None))
            if _ovr:
                _feature_phase_override = {"reason": str(_ovr),
                                           "source": "manual"}
                logger.info("特征期门 override: %s", _ovr)
            elif _rel:
                _feature_phase_override = _rel
                logger.info("特征期门:按 Planner 释放名额放行 proposal %s"
                            "(理由:%s)", _rel["proposal_id"],
                            str(_rel.get("reason"))[:160])
            else:
                logger.error(
                    "特征期门:已完成探索代数 %d < feature_exploration."
                    "min_generations=%d,model/loss/hpo 方向 trial 暂不放行"
                    "(协议:先特征、后模型)。可选:继续特征/数据视图方向;"
                    "按 Planner 释放的名额建(--planner-proposal-id);或显式 "
                    "--override-feature-phase \"理由\"(留痕 trial.yaml)",
                    _fp["generations_done"], _fp["min_generations"])
                return 1
        elif _ovr:
            logger.info("特征期已满(已完成 %d 代),--override-feature-phase "
                        "无需提供,已忽略", _fp["generations_done"])
    # 数据视图(协议 v7):非 v1 视图必须已落盘(write-splits-view 先行)
    data_view = (getattr(args, "data_view", None) or "v1").strip() or "v1"
    if data_view != "v1":
        from runtime.data import view_splits_dir

        vd = view_splits_dir(run_dir, data_view)
        if not (vd / "train.parquet").exists():
            logger.error("数据视图 %s 的 splits 不存在(%s):"
                         "先执行 write-splits-view --view-id %s",
                         data_view, vd, data_view)
            return 1
    expected_resources = {}
    if getattr(args, "heavy", False):
        expected_resources["heavy"] = True
        expected_resources["timeout_multiplier"] = float(
            config.get("evolution.heavy_trials.timeout_multiplier", 4))
    # 参数 trial 快通道:算法继承父代;parents 自动指向 from-parent
    algo = args.algorithm
    if from_parent:
        if algo is None:
            from runtime.trial import TrialRecord as _TR

            parent_rec = _TR.load(run_dir / "trials" / from_parent)
            algo = parent_rec.data.get("algorithm") or "LightGBM"
        if parent_ids is None:
            parent_ids = [from_parent]
        # v2.15.8:父代源码冻结前置断言。Agent 误改父代(尤其 baseline)
        # 的 code/ 后,派生出的子代会在 run 期 verify_trial_source 才失败
        # (refactor4 实测:改 trial-003 → trial-004 首跑失败,只能按
        # source-manifest 的 sha256 逆向还原)。在派生前就拦下,并指明
        # 还原方法。父代尚未冻结(无清单)时不查。新 run 拒绝,旧 run 告警。
        _parent_dir = run_dir / "trials" / from_parent
        if (_parent_dir / "source-manifest.json").exists():
            try:
                from runtime.integrity import verify_trial_source

                verify_trial_source(_parent_dir)
            except Exception as exc:  # noqa: BLE001
                from runtime import version_at_least

                _rs = load_json(run_dir / "run-state.json") \
                    if (run_dir / "run-state.json").exists() else {}
                _msg = (f"父代 {from_parent} 的 code/ 与冻结清单不一致({exc}):"
                        "子代会继承被改动的源码,并在运行期源码校验时失败。"
                        "把父代 code/ 还原到 source-manifest.json 记录的版本"
                        "(改动应放进新 child trial,而不是改父代)")
                if version_at_least(_rs.get("skill_version"), "2.15.8"):
                    logger.error(_msg)
                    return 1
                logger.warning(_msg + "(旧 run:仅告警)")
    _extra_confirm = None
    _confirm_during_hpo = None
    if args.action == "confirm":
        if not from_parent or parent_ids != [from_parent]:
            logger.error("confirm 必须且只能用 --from-parent 指定一个父 Trial")
            return 1
        from runtime.trial import TrialRecord as _TR

        _confirm_parent = _TR.load(run_dir / "trials" / from_parent)
        _eligible, _why = confirmation_parent_eligibility(_confirm_parent)
        if not _eligible:
            logger.error(
                "confirm 父代 %s 不具备晋升资格:%s；普通候选必须先完成 "
                "finalize-generation 并通过代级 FDR，根基线需有 "
                "root_baseline_only 证据", from_parent, _why)
            return 1
        try:
            from runtime.integrity import verify_trial_source

            verify_trial_source(run_dir / "trials" / from_parent)
        except Exception as exc:  # noqa: BLE001
            logger.error("confirm 父代源码完整性失败:%s", exc)
            return 1
        # [v2.15.0] 父代仍有在途 HPO 子代 → 该配方大概率当代被超越:
        # run3 的 trial-019(confirm 016)与 trial-020(confirm 018)都与
        # 同代 HPO 并行,确认完即被新冠军取代,合计 1.4 GPU 小时。
        _inflight = _inflight_hpo_children(factory, from_parent)
        if _inflight:
            if not getattr(args, "allow_confirm_during_hpo", None):
                logger.error(
                    "confirm 时机门:父代 %s 仍有在途 HPO 子代 %s —— 确认完"
                    "大概率即被超越(run3 trial-019/020)。等 HPO 出结果后对"
                    "届时的搜索冠军 confirm;确需现在确认(如预算即将耗尽的"
                    "保险式确认)加 --allow-confirm-during-hpo \"理由\"(留痕)",
                    from_parent, ", ".join(_inflight))
                return 1
            _confirm_during_hpo = {
                "reason": str(args.allow_confirm_during_hpo),
                "inflight_hpo_children": _inflight,
                "at": __import__("datetime").datetime.now().isoformat()}
        # v2.14.0:confirm 次数预算。确认门若(曾)是抽签,"失败就再抽"
        # 只会认证中签者 —— run2 对三任冠军配方做了 4 次 confirm,唯一
        # 通过的 trial-011 恰恰是 shadow 上显著更差的那次。超额必须由用户
        # 裁决并留痕,且新判据下失败已是有意义的证据,不应靠重抽抹掉。
        from runtime.confirmation import confirm_attempts

        _cap = int(config.get("evolution.max_confirm_attempts_per_recipe",
                              2) or 0)
        _prev = confirm_attempts(factory.all_trials(), from_parent)
        if _cap > 0 and len(_prev) >= _cap:
            _extra = getattr(args, "allow_extra_confirm", None)
            if not _extra:
                logger.error(
                    "配方 %s 已完成 %d 次 confirm(%s),达上限 "
                    "evolution.max_confirm_attempts_per_recipe=%d。重抽不会"
                    "让结论更可信:先逐项读已有 confirm 的 "
                    "evaluation.json.fresh_confirmation(非劣/保真/参照/"
                    "σ_seed 来源);确需追加请经用户裁决后加 "
                    "--allow-extra-confirm \"<理由>\"(留痕)",
                    from_parent, len(_prev),
                    ", ".join(f"{t.trial_id}={t.verdict}" for t in _prev),
                    _cap)
                return 1
            _extra_confirm = {"reason": str(_extra),
                              "previous_attempts": [t.trial_id for t in _prev],
                              "cap": _cap}
        # multi-seed 证据段:confirm 复制父代代码,若父代(如 baseline 血缘)
        # 没有模板的 multi-seed 段,multi_seed=3 会被静默忽略(run2 实测)。
        # 这里只告警不拒绝 —— 判据会回退到同配方复现组/floor 估 σ_seed,
        # 缺失事实同时写进 evaluation.multi_seed_evidence。
        try:
            _psrc = (run_dir / "trials" / from_parent / "code" /
                     "train.py").read_text(encoding="utf-8")
            if "seed_scores" not in _psrc:
                logger.warning(
                    "父代 %s 的 train.py 不含 multi-seed 段(seed_scores/):"
                    "confirm 声明的 multi_seed 不会执行,σ_seed 将回退到同配方"
                    "复现组/先验/floor。需要 multi-seed 证据时用 "
                    "--allow-code-edit 补入模板的 multi-seed 段", from_parent)
        except OSError:
            pass
    # 代号守卫(防错标):漏传 --generation 时不再落默认值 1,改为
    # 父代代号+1(from-parent)或已落盘 trial 最大代号+1;显式传入
    # 仍优先生效,但低于父代代号时按父代+1 兜底(防跨代回跳错标)。
    gen_requested = getattr(args, "generation", None)
    generation = int(gen_requested) if gen_requested is not None else None
    if from_parent:
        from runtime.trial import TrialRecord as _TR

        parent_rec = _TR.load(run_dir / "trials" / from_parent)
        parent_gen = int(parent_rec.data.get("generation", 0) or 0)
        if generation is None:
            generation = parent_gen + 1
            logger.info("--generation 未传,按父代 %s 代号+1=%d 兜底"
                        "(漏传时不再落默认值 1)", from_parent, generation)
        elif generation < parent_gen:
            logger.warning("--generation %d 低于父代 %s 的代号 %d,"
                           "按父代+1=%d 兜底(防止跨代回跳错标)",
                           generation, from_parent, parent_gen,
                           parent_gen + 1)
            generation = parent_gen + 1
    if generation is None:
        generation = factory.next_generation()
        logger.info("--generation 未传,自动取最大代号+1=%d "
                    "(漏传时不再落默认值 1)", generation)
    # [v2.13.2] 首代动作门。gen-01 是特征探索的起点,此时没有任何
    # 父代证据可"加码"(exploit 的定义)或"组合"(crossover 的定义):
    # 方向台账 directions.json 尚不存在(它在首次 finalize-generation
    # 后才由 DirectionLedger.refresh 落盘),Planner 的
    # _direction_candidates 因此返回 [] —— 同代不会提名这两类候选。
    # 实测两个 run 的 gen-01 各有一个 Agent 手建的 exploit "prune 对照臂"
    # (example-seq-v3-v2.12.2.4 trial-007/trial-016),净收益 +0.00022
    # (promising),代价是一个槽位。判据放在这里而不是创建口更早的位置,
    # 是因为 --generation 可能被上方兜底改写,只有解析完成后才可判定。
    if args.action in ("exploit", "crossover") and generation <= 1:
        logger.error(
            "gen-%02d 禁止 --action %s:首代无父代证据可加码/组合"
            "(方向台账 directions.json 尚不存在,Planner 同代也不会"
            "提名这两类候选)。首代请用 --action explore 或 data_view "
            "建立可归因证据,exploit/crossover 留到 gen-02 起",
            generation, args.action)
        return 1
    # [v2.13.2] 特征 bundle 槽位门(feature_exploration.bundle_beam_width)。
    # 该配置键此前零引用(纯摆设):gen-01 的实际宽度由调用方手传的
    # `plan --slots N` 决定,而实测 0/15 个 gen-01 trial 带
    # planner_proposal_id —— Agent 全部手建,Planner 侧任何约束都咬不到。
    # 因此把门放在创建口,它才是真正生效的那一道。
    if args.action in ("explore", "data_view") and changed_modules:
        _is_feature_trial = "feature" in changed_modules or \
            "sample" in changed_modules or "data" in changed_modules
        _beam = int(config.get("feature_exploration.bundle_beam_width", 4))
        if _is_feature_trial and _beam > 0:
            _used = _count_feature_trials(factory, generation)
            if _used >= _beam and not getattr(args, "override_beam", None):
                logger.error(
                    "本代(gen-%02d)特征 bundle 槽位已满:"
                    "bundle_beam_width=%d,已建 %d 个。"
                    "提高 feature_exploration.bundle_beam_width,或把该"
                    "方向排到下一代;确需超宽时用 --override-beam \"理由\" "
                    "放行(留痕 trial.yaml.beam_override)",
                    generation, _beam, _used)
                return 1
    try:
        trial = factory.create(
            algorithm=algo or "LightGBM",
            hypothesis=hypothesis,
            execution={"action_description": actions,
                       "code_entry": "train.py",
                       # cwd 为 code/,配置在上级 config/(绝对定位见 load_config)
                       "command": "python train.py --config ../config/trial.yaml",
                       "changed_modules": changed_modules},
            search_action=args.action,
            parent_trial_ids=parent_ids,
            generation=generation,
            expected_resources=expected_resources or None,
        )
    except ValueError as exc:
        # v2.17.0(D5):创建口校验失败(假设占位/过短等)是用户输入问题 ——
        # 清晰报错 exit 1,不打 traceback(create 在落盘前校验,无孤儿目录)
        logger.error("new-trial 被拒绝: %s", exc)
        return 1
    # [FIX-2026-09-03] 事务登记:create 成功即记录目录,后续任何失败
    # 路径(SystemExit 分支就地回滚 / 非预期异常由 main() 安全网回滚)
    # v2.11.9:跨模块接力改走 context 登记点(原模块级 global 拆分后
    # 只会绑到本模块,main() 的异常安全网读不到)
    from . import context as _ctx
    _ctx._LAST_CREATED_TRIAL_DIR = Path(trial.trial_dir)
    trial.data["data_view"] = data_view
    # v2.16.0:拓扑声明。缺省继承快通道父代的声明(confirm/hpo 子代沿用
    # 父代结构);声明只增加需批准项,运行前的静态检测是底线
    _decl = getattr(args, "topology", None)
    if not _decl and getattr(args, "from_parent", None):
        _pdir = run_dir / "trials" / str(args.from_parent)
        if (_pdir / "trial.yaml").exists():
            from runtime.trial import TrialRecord as _TRp

            _decl = _TRp.load(_pdir).data.get("declared_topology")
    trial.data["declared_topology"] = _decl or "single"
    if trial.data["declared_topology"] != "single":
        from runtime.topology import STRUCTURAL_WHITELIST

        _form = trial.data["declared_topology"].split(":", 1)[-1]
        if _form not in STRUCTURAL_WHITELIST.get(str(spec.model_type), ()):
            print(f"  注意: 声明拓扑 {trial.data['declared_topology']} 需要"
                  "用户人工批准后才能 run(写完代码后 run 会登记批准请求;"
                  "Agent 不得自行批准)")
    if _hpo_dim_override:
        trial.data["hpo_diminishing_override"] = _hpo_dim_override
    if _confirm_during_hpo:
        trial.data["confirm_during_hpo"] = _confirm_during_hpo
    if getattr(args, "planner_proposal_id", None):
        trial.data["planner_proposal_id"] = str(args.planner_proposal_id)
    if getattr(args, "planner_semantic_hash", None):
        trial.data["planner_semantic_hash"] = str(
            args.planner_semantic_hash)
    # v16/P1-3:family-hints 落 trial.yaml(方向台账前缀族归因)
    _fh = getattr(args, "family_hints", None)
    if _fh:
        trial.data["family_hints"] = [h.strip() for h in
                                      str(_fh).split(",") if h.strip()]
    # v2.16.0:特征期门放行留痕(来源可辨:manual = 人工 override;
    # planner_feature_phase_release = Planner 确定性收敛判定)
    if _feature_phase_override:
        trial.data["feature_phase_override"] = dict(
            _feature_phase_override,
            at=__import__("datetime").datetime.now().isoformat())
    # [v2.13.2] beam 超宽放行留痕(与 feature_phase_override 同形:理由可读、
    # 不静默;report 与反思据此区分"按宽度纪律建代"与"人工放行超宽")
    if getattr(args, "override_beam", None):
        trial.data["beam_override"] = {
            "reason": str(args.override_beam),
            "bundle_beam_width": int(config.get(
                "feature_exploration.bundle_beam_width", 4)),
            "at": __import__("datetime").datetime.now().isoformat(),
        }
    # [v2.13.2] screening-only 标记:宽面初筛 trial 贡献 group_importance
    # 先验,但不参与任何裁决(见 runtime/trial.py:trial_is_screening
    # 的调用点清单)。
    if getattr(args, "screen_only", False):
        trial.data["screen_only"] = True
    # v2.18.0(Q18):依据。--basis "<原则id>:<关系>[,…]" 或 "novel";关系 ∈
    # follows / extends / tests / contradicts(省略 = follows);--experience-ids
    # 为旧写法(等价 follows)。run 已固定经验库视图时必填,--from-parent 子代
    # 未给时继承父代依据;引用的 id 必须在本 run 固定水位的视图中存在。
    # --topics:本 trial 的主题(经验库主题词或别名),供检索与证据匹配。
    from runtime import library_view as _lv
    _view = _lv.load(run_dir)
    _basis, _berr = _parse_basis(getattr(args, "basis", None),
                                 _parse_list(getattr(args, "experience_ids", None)))
    if _basis is None and from_parent:
        try:
            from runtime.trial import TrialRecord as _TRb
            _pb = _TRb.load(run_dir / "trials" / from_parent).data.get("basis")
            if _pb:
                _basis = {**_pb, "inherited_from": from_parent}
        except Exception:  # noqa: BLE001
            pass
    if _berr is None and _basis is None and _view.pinned:
        _berr = ("必须写 --basis:引用经验库原则 `<id>:<follows|extends|tests|contradicts>`,"
                 "或明确 `--basis novel`。先用 `experience search --run <run> --topics …` "
                 "检索同主题原则")
    if _berr is None and _basis and not _basis.get("novel") and not _view.pinned:
        # 经验库视图不可用(服务不可达且无缓存):无法核验 id,照记并标注(Q26 不阻塞)
        _basis["unverified"] = True
    elif _berr is None and _basis and not _basis.get("novel"):
        _missing = [r["id"] for r in _basis["refs"] if not _view.exists(r["id"])]
        if _missing:
            _berr = (f"--basis 引用的原则不在本 run 固定水位(水位 {_view.watermark})的经验库"
                     f"视图中: {_missing};用 `experience search` 查可用 id")
    if _berr:
        logger.error("new-trial 依据校验失败: %s", _berr)
        _rollback_trial(trial)
        return 1
    if _basis is not None and _basis.get("refs") and "inherited_from" not in _basis:
        _basis["source"] = _basis_source(_basis, getattr(args, "basis_prefill", None))
    if _basis is not None:
        trial.data["basis"] = _basis
        trial.data["experience_ids"] = [r["id"] for r in _basis.get("refs") or []]
    _topics_raw = _parse_list(getattr(args, "topics", None))
    if _topics_raw:
        _res = _view.resolver() if _view.data.get("topics") else {}
        _known = [_res.get(t.lower(), t) for t in _topics_raw if t.lower() in _res]
        _unknown = [t for t in _topics_raw if t.lower() not in _res]
        if _unknown and _view.data.get("topics"):
            logger.warning("--topics 中不在经验库词表的写法: %s(保留原文;合并时可新增主题词)",
                           _unknown)
        trial.data["topics"] = list(dict.fromkeys(_known + _unknown))
    trial.flush()
    # 参数 trial 快通道:复制父代代码 + 注入超参,跳过模板渲染。
    # 引用模式(@...)在快通道下先展开成列再覆写(复用模板路径同一
    # _load_features_ref),列清单只落 config,不进命令行/上下文
    if from_parent and isinstance(features_override, str):
        import pandas as _pd_q

        splits_dir_q = run_dir / "sample-profile" / "splits"
        derived_q: list[str] = []
        if (splits_dir_q / "train.parquet").exists():
            derived_q = resolve_feature_columns(
                _pd_q.read_parquet(splits_dir_q / "train.parquet"), spec)
        resolved_q = _load_features_ref(features_override, run_dir, derived_q)
        if not resolved_q:
            logger.error("特征面引用 %s 展开为空(与派生特征面无交集)",
                         features_override)
            _rollback_trial(trial)
            return 1
        features_override = _resolve_features_override(
            resolved_q, spec, derived_q)
    if from_parent:
        try:
            _copy_parent_trial_code(run_dir, trial, from_parent, overrides,
                                    features_override=features_override,
                                    action=args.action)
            # v2.16.2:--set 消费检查前移到创建期(lint 规则 11 同口径);hurdle
            # 父代自动注入 hurdle 超参段作消费点(D-008 gen-04 的手补脚本)
            _hp_notes = _ensure_hyperparameters_consumed(
                run_dir, trial, from_parent, args.action,
                allow_code_edit=bool(getattr(args, "allow_code_edit", False)))
            if _hp_notes:
                trial.data.setdefault("execution", {})[
                    "hyperparameters_consumer"] = _hp_notes
                trial.flush()
        except SystemExit:
            # [FIX-2026-09-03] 失败事务化:父代码校验/复制失败时回滚
            # 本次创建的孤儿目录(旧实现残留 PROPOSED 目录,伪装成
            # "半成品 trial",gen-03 实测 005/006 孤儿引发误判与 rm -rf)
            _rollback_trial(trial)
            raise
        _retarget_splits_to_view(run_dir, trial, data_view)
        if getattr(args, "allow_code_edit", False):
            trial.data.setdefault("execution", {})["changed_modules"] = \
                changed_modules
            trial.data["execution"]["action_description"] = actions
            trial.flush()
        if args.action == "confirm":
            # confirmation 必须是独立重训，而非同 seed 的逐字复算。
            _cfg_path = Path(trial.trial_dir) / "config" / "trial.yaml"
            _confirm_cfg = load_yaml(_cfg_path) or {}
            _parent_seed = int(_confirm_cfg.get("seed", 20260814))
            _confirm_seed = _parent_seed + 1009 + generation
            _confirm_cfg["seed"] = _confirm_seed
            # v2.16.2(run4 regression D-006):写**代码实际会跑**的 seed 数。
            # 模板与 baseline 血缘的 multi-seed 段对 confirm 在 config ≤1 时
            # 默认实跑 3 个 seed;旧实现写 max(1, 父代值)=1 → 证据对账判
            # not_declared(declared 1 / found 2)。无 multi-seed 段的代码保持
            # 声明值(上方已告警,对账会判 missing)。
            try:
                _code_ms = (Path(trial.trial_dir) / "code" / "train.py"
                            ).read_text(encoding="utf-8")
            except OSError:
                _code_ms = ""
            _confirm_cfg["multi_seed"] = _confirm_effective_multi_seed(
                _confirm_cfg.get("multi_seed"), _code_ms)
            _confirm_cfg["confirmation"] = {
                "reference_trial": from_parent,
                "selection_split": "validation",
                "independent_split": "shadow_test",
            }
            dump_yaml(_cfg_path, _confirm_cfg)
            if _extra_confirm:
                trial.data["extra_confirm"] = _extra_confirm
                trial.flush()
            # baseline 血缘把 seed 编译进模块常量 SEED，main() 不读
            # config.seed；只改 YAML 会产生“审计上新 seed、实际仍旧 seed”
            # 的伪独立确认。模板血缘会 cfg.get('seed')，但同步常量也让
            # 所有路径的 fallback 与配置一致。
            import re as _re

            _src_path = Path(trial.trial_dir) / "code" / "train.py"
            _src = _src_path.read_text(encoding="utf-8")
            _patched, _n_seed = _re.subn(
                r"(?m)^SEED\s*=\s*[-+]?\d+\s*$",
                f"SEED = {_confirm_seed}", _src, count=1)
            if _n_seed:
                _src_path.write_text(_patched, encoding="utf-8")
            elif "cfg.get(\"seed\"" not in _src and \
                    "cfg.get('seed'" not in _src:
                logger.error(
                    "confirm 代码既无可同步的 SEED 常量，也未消费 "
                    "config.seed；无法证明独立重训")
                _rollback_trial(trial)
                return 1
        if args.action == "hpo":
            # hpo 快通道:复制父代码后注入搜索空间与预算到 config
            if getattr(args, "force_hpo_budget", False):
                # v16:成本闸豁免留痕(见 _hpo_cost_gate)
                er = dict(trial.data.get("expected_resources") or {})
                er["hpo_force_budget"] = True
                trial.data["expected_resources"] = er
                trial.flush()
            try:
                _inject_hpo_config(run_dir, trial, args.search_space,
                                   int(args.budget_configs or config.get(
                                       "evolution.hpo.default_budget", 40)),
                                   sample_rows=getattr(
                                       args, "hpo_sample_rows", None),
                                   hpo_opts=_hpo_opts_from_args(args))
            except SystemExit:
                _rollback_trial(trial)
                raise
        # v2.17.0:创建期后处理(add-blocks / 回归单分支 / 交付契约自检)
        if _post_create_code(run_dir, spec, trial, args) != 0:
            return 1
        (Path(trial.trial_dir) / "hypothesis.md").write_text(
            f"# Hypothesis\n\n{hypothesis['statement']}\n\n"
            f"## Rationale\n\n{hypothesis.get('rationale') or '—'}\n\n"
            f"## Falsification\n\n"
            f"{hypothesis.get('falsification') or '—'}\n",
            encoding="utf-8")
        print(trial.trial_id)
        print(f"  目录: {trial.trial_dir}")
        print(f"  参数覆写: {overrides}(代码复制自 {from_parent})")
        print(f"  下一步: modelevo.py run --run {run_dir} "
              f"--trial {trial.trial_id}(无需编辑代码)")
        return 0
    # [FIX-2026-09-03] hpo 模板渲染路径(v2.9):无 --from-parent 时
    # 不再拒绝创建,而是按 --algorithm 渲染含算法化四件套的可执行
    # trial;特征面按 --hpo-parent 继承(其 config features 单源)或
    # 默认全列。--set 覆写同样生效(非 hpo 分支被 hpo 分支遮蔽,但
    # best_params + _cfg_hp 的叠加消费口径已在模板 hpo 分支实现)。
    if args.action == "hpo":
        _hpo_feat_parent = getattr(args, "hpo_parent", None)
        if _hpo_feat_parent:
            from runtime.logging_utils import load_yaml as _ly

            _p_cfg = _ly(Path(run_dir) / "trials" / _hpo_feat_parent /
                         "config" / "trial.yaml") or {}
            _p_feats = _p_cfg.get("features") or {}
            # 继承父代特征声明(exclude/banned/reparse 整体继承;
            # columns 白名单存在则原样继承——特征面单源搬运)
            if isinstance(_p_feats, dict) and _p_feats:
                spec_features = {
                    "exclude": sorted(set(_p_feats.get("exclude") or [])),
                    "banned": sorted(set(_p_feats.get("banned") or [])),
                    "reparse": sorted(set(_p_feats.get("reparse") or [])),
                }
                if _p_feats.get("columns"):
                    spec_features["columns"] = list(_p_feats["columns"])
                    _feat_cols = list(_p_feats["columns"])
                else:
                    _feat_cols = None
                features_spec_override = spec_features
            else:
                features_spec_override = None
                _feat_cols = None
            logger.info("hpo 模板渲染路径:特征面继承自 %s config"
                        "(columns=%s)", _hpo_feat_parent,
                        "继承" if _feat_cols else "默认全列−排除列")
        else:
            features_spec_override = None
        _algo_l = str(args.algorithm or "lightgbm").strip().lower()
        if _algo_l not in _HPO_RENDER_ALGOS:
            logger.error("hpo 模板渲染路径不支持算法 %r(支持 %s)",
                         args.algorithm, list(_HPO_RENDER_ALGOS))
            _rollback_trial(trial)
            return 1
        # [v2.13.4] 特征面一致性守卫:--hpo-parent 的"继承"是**声明面级**
        # 的(只搬 config.features 的 exclude/banned/reparse/columns),
        # 父代的 transform 加宽列(运行时产物)拿不到 —— 于是"在冠军
        # 特征面上做 HPO"会静默降级为基础面。实测事故
        # (req-t7-drawdown-v21331):同一份 planner 命令,本 run trial-018
        # 跑在 1500 列基础面、先行 run trial-014 跑在 1887 列加宽面,
        # 差别只在"Agent 手工换成了 --from-parent"。两次都"成功"返回,
        # 没有任何信号提示特征面不等价。
        # 判据:父代 metrics.json 的 widen_stats.n_new_cols > 0(transform
        # 加宽的确定性标记)。命中即拒绝,并给出可复制的替代命令 ——
        # 不静默降级、也不让操作者自己猜路径。
        _face_risk = _hpo_parent_face_risk(trial, _hpo_feat_parent)
        if _face_risk:
            logger.error(
                "hpo 特征面守卫:--hpo-parent 无法复现父代的加宽特征面。%s\n"
                "  改用(零写码,--from-parent 会自动嫁接缺失的 hpo 段):\n"
                "    python scripts/modelevo.py new-trial --run %s "
                "--generation %s --action hpo --algorithm %s "
                "--from-parent %s --parents %s --changed-modules model "
                "--search-space %s --budget-configs %s",
                _face_risk, run_dir, generation, _algo_l, _hpo_feat_parent,
                _hpo_feat_parent, args.search_space,
                args.budget_configs or 40)
            _rollback_trial(trial)
            return 1
        trial.data["algorithm"] = _algo_l
        trial.data.setdefault("execution", {})[
            "action_description"] = actions
        trial.data["execution"]["hpo_render_path"] = {
            "algorithm": _algo_l,
            "features_parent": _hpo_feat_parent,
        }
        trial.flush()
    # crossover:创建口做兼容性检查 + 登记父代组合(§13.3)。
    # v8:父代从 archive candidate 取(带 changed_modules),冲突时拒绝创建
    if args.action == "crossover" and parent_ids:
        from runtime.evolution import (EvolutionEngine,
                                       check_crossover_compatibility)

        try:
            engine = EvolutionEngine(run_dir, spec, config)
            by_id = {c.get("trial_id"): c for c
                     in engine.archive.state.get("candidates", [])}
            parents = [by_id.get(pid) or {"trial_id": pid}
                       for pid in parent_ids]
            conflicts = check_crossover_compatibility(
                parents, engine.state.get("crossover_combos", []))
            if conflicts:
                logger.error("crossover 兼容性检查未通过,拒绝创建:")
                for c in conflicts:
                    logger.error("  - %s", c)
                return 1
            engine.register_crossover(parent_ids)
        except Exception as exc:  # noqa: BLE001
            logger.error("crossover 组合登记失败: %s", exc)
            return 1
    # 代码模板
    import pandas as pd

    splits_dir = run_dir / "sample-profile" / "splits"
    feature_cols = []
    if (splits_dir / "train.parquet").exists():
        feature_cols = resolve_feature_columns(
            pd.read_parquet(splits_dir / "train.parquet"), spec)
    # [FIX-2026-09-03] hpo 模板渲染路径的特征面:--hpo-parent 继承时,
    # columns 白名单直接来自父代 config(它可能是加宽列,不在 splits
    # parquet 派生面里——引用模式 @trial 的交集校验会静默丢列,此处
    # 绕开),其余(exclude/banned/reparse)由 features_spec_override
    # 在下方 config 写盘时替换。
    if args.action == "hpo" and features_spec_override and \
            features_spec_override.get("columns"):
        feature_cols = list(features_spec_override["columns"])
    if isinstance(features_override, str) and features_override.startswith("@"):
        # 引用模式(@file/@trial_id/@trial_id:裁决产物):以派生特征面为
        # 基准展开(交集校验),列表本身不进命令行/上下文
        resolved = _load_features_ref(features_override, run_dir, feature_cols)
        if not resolved:
            logger.error("特征面引用 %s 展开为空(与派生特征面无交集)",
                         features_override)
            _rollback_trial(trial)
            return 1
        features_override = resolved
        feature_cols = _resolve_features_override(
            features_override, spec, feature_cols)
    elif features_override:
        # 显式特征面白名单:校验列存在后替换派生结果(写入本 trial
        # config 的 features.columns;仍自动剔除目标/时间/ID/排除列)
        missing = [c for c in features_override if c not in feature_cols
                   and c not in (spec.target_name, spec.time_column,
                                 spec.id_column)]
        if missing:
            logger.error("特征白名单含不存在列: %s", missing[:10])
            _rollback_trial(trial)
            return 1
        feature_cols = _resolve_features_override(
            features_override, spec, feature_cols)
    # v19.3 修复(算法口径错):搜索空间(space-*.yaml)按父代算法命名键,
    # 而 _build_model 在 `_HPO_ALGO` 缺失时**回退 lightgbm**。嫁接路径
    # (--from-parent 且父代码缺四件套)此前从不注入该常量,于是 trial_id
    # 声明 xgboost、实际却在训 lightgbm,整轮搜索口径作废(gen-03
    # trial-008 实测:operator 手工补 `_HPO_ALGO='xgboost'` 才纠正)。
    # 这里把 hpo 动作的**全部路径**(模板渲染 + from-parent 快通道 + 嫁接)
    # 统一交给 _inject_hpo_algo_at,注入点与模板路径一致(SEED 锚),幂等。
    # v18/P2-3:automl 参照系 body——零写码渲染外部 AutoML 参照 trial
    # (fit_and_score 内 try-import autogluon/flaml,缺失时显式报错
    # 而非静默降级;产物与常规 trial 同契约,score 语义 = 正类概率)
    body = TRIAL_BODY
    if args.action == "automl":
        body = AUTOML_BODY
        args.actions = ["automl 参照系(零写码渲染)"]
    _hurdle_tpl = bool(getattr(args, "hurdle_template", False))
    if _hurdle_tpl:
        # v2.17.0(#15):官方 hurdle 模板(脚手架 fit_and_score → hurdle 三入口)
        from .templates import hurdle_trial_body

        body = hurdle_trial_body()
    # v12/P0-8:非特征列按 spec 注入(替换硬编码 pday/user_no)。
    # v13:BANNED/REPARSE_SOURCES 按泄漏分类注入(毒性轴);banned 的
    # 主防御在数据面物理隔离,模板里的集合做 fit 前终检 + lint 9a。
    # 注意:marker 段(exclude_columns/BANNED 等)来自 TRIAL_TEMPLATE 而
    # 非 body,替换必须在 format 之后作用于 code(旧序 replace(body)
    # 是空操作,注入静默失效——本测试即由此发现)。
    _nonfeature = sorted({c for c in (spec.time_column, spec.id_column,
                                      spec.treatment_column) if c})
    code = TRIAL_TEMPLATE.format(
        trial_id=trial.trial_id,
        hypothesis=hypothesis["statement"],
        actions="; ".join(actions),
        run_dir=str(run_dir.resolve()),
        target=spec.target_name,
        model_type=spec.model_type,
        treatment=spec.treatment_column or "",
        seed=20260814,
        # v14:trial 子进程需要 skill 根在 sys.path 才能 import runtime.hpo
        skill_root=str(_SKILL_ROOT),
        body=body,
    )
    code = code.replace(
        "BANNED: frozenset[str] = frozenset()",
        f"BANNED: frozenset[str] = frozenset({sorted(set(spec.banned_columns))!r})")
    code = code.replace(
        "REPARSE_SOURCES: frozenset[str] = frozenset()",
        f"REPARSE_SOURCES: frozenset[str] = frozenset({sorted(set(spec.reparse_columns))!r})")
    code = code.replace(
        "_SPEC_NONFEATURE: set[str] = set()",
        f"_SPEC_NONFEATURE: set[str] = set({_nonfeature!r})")
    # [FIX-2026-09-03] hpo 模板渲染路径:注入算法常量(算法化四件套);
    # hpo config 节在下方 trial_cfg 写盘后追加(_inject_hpo_config
    # 读取-改写 config,必须落在 config 存在之后)。
    # v19.3:改走 _inject_hpo_algo_at 统一入口(与 --from-parent 快通道
    # /嫁接路径同一实现,三处口径不再各自漂移)。
    if args.action == "hpo" and not _hurdle_tpl:
        try:
            _code2 = _inject_hpo_algo_at(code, _algo_l)
        except SystemExit as _exc:
            logger.error("%s", _exc)
            _rollback_trial(trial)
            return 1
        if _code2 is not None:
            code = _code2
    # v2.17.0(D6):回归脚手架遵从 --algorithm。旧版不论 --algorithm 都落到
    # DummyRegressor(中位数常数预测),且 lint 规则 10 因脚手架标记拒跑 ——
    # "--algorithm xgboost" 形同虚设。可渲染算法(xgboost/lightgbm/catboost)
    # 注入算法常量,走模板算法化分支(runtime.hpo.build_gbm 构造回归器 +
    # validation 早停,与 confirm/--set 子代同一路径;config.hyperparameters
    # 经 _cfg_hp 覆写);其余算法保留占位并在创建期明确提示。
    elif args.action not in ("hpo", "automl") and not _hurdle_tpl \
            and str(spec.model_type) == "regression":
        _scaffold_algo = str(args.algorithm or "").strip().lower()
        if _scaffold_algo in _HPO_RENDER_ALGOS:
            try:
                _code2 = _inject_hpo_algo_at(code, _scaffold_algo)
            except SystemExit as _exc:
                logger.error("%s", _exc)
                _rollback_trial(trial)
                return 1
            if _code2 is not None:
                code = _code2
                trial.data["algorithm"] = _scaffold_algo
                trial.data.setdefault("execution", {})[
                    "scaffold_algorithm"] = {
                        "algorithm": _scaffold_algo,
                        "path": "template_algorithmic_branch",
                        "note": "回归脚手架按 --algorithm 构造真实回归器"
                                "(build_gbm);模型改造请改算法化分支/"
                                "_build_model,DummyRegressor 分支不可达"}
                trial.flush()
        else:
            logger.warning(
                "回归脚手架:--algorithm %r 不在可渲染算法 %s 内,fit_and_score "
                "保留 DummyRegressor 占位(lint 规则 10 会拒跑未改造的脚手架);"
                "请实现真实回归器后再 run,或改用 --algorithm xgboost|lightgbm",
                args.algorithm, list(_HPO_RENDER_ALGOS))
    # v2.17.0(#7):回归任务按 model_type 渲染单一打分分支(分类不变)
    from runtime.templates import narrow_model_type_dispatch

    code, _narrow_notes = narrow_model_type_dispatch(code, str(spec.model_type))
    # 数据视图(协议 v7):模板 SPLITS 指向视图目录(非 v1 时)
    if data_view != "v1":
        from runtime.data import view_splits_dir

        view_dir = view_splits_dir(run_dir.resolve(), data_view)
        code = code.replace(
            'SPLITS = RUN_DIR / "sample-profile" / "splits"',
            f'SPLITS = RUN_DIR / {str(view_dir.relative_to(run_dir.resolve()))!r}')
    code_dir = Path(trial.trial_dir) / "code"
    code_dir.mkdir(exist_ok=True)
    (code_dir / "train.py").write_text(code, encoding="utf-8")
    # 可复现配置(§10.3):features/目标/seed/超参落盘到 config/trial.yaml。
    # 特征面单源存储:config 的 features(exclude 恒存;--features 显式
    # 白名单时写 columns),train.py 运行期经 exclude_columns() 读取,
    # 代码不再内置 FEATURES 常量(去双写漂移)。
    feats_cfg: dict = {"exclude": sorted(set(spec.excluded_columns)),
                       "banned": sorted(set(spec.banned_columns)),
                       "reparse": sorted(set(spec.reparse_columns))}
    if features_override:
        feats_cfg["columns"] = feature_cols
    # [FIX-2026-09-03] hpo 模板渲染路径 + --hpo-parent:特征声明整体
    # 来自父代 config(features_spec_override),非列部分替换默认派生
    if args.action == "hpo" and features_spec_override:
        feats_cfg = features_spec_override
        if features_override and feature_cols:
            feats_cfg["columns"] = feature_cols
    trial_cfg = {
        "trial_id": trial.trial_id,
        # v2.12.2.7:algorithm 与 trial.yaml/trial_id 同源。模板渲染路径
        # 旧实现不写该字段(实测 g01 的 10 条 trial config 里缺 algorithm),
        # 快通道则继承父代的值(5 条 HPO 子代误标 lightgbm);两处都补齐,
        # 让 config 单独可读即可还原真实算法。
        "algorithm": str(trial.data.get("algorithm") or "").strip().lower(),
        "features": feats_cfg,
        "target": spec.target_name,
        "model_type": spec.model_type,
        "seed": 20260814,
        "hyperparameters": {},
        # v18:search_action 进 config——模板 main() 据此对 confirm
        # trial 默认 multi_seed=3(交付候选稳定性证据,无需 Agent 记得)
        "search_action": args.action,
        "data_view": data_view,
        "notes": "特征面单源:本文件 features(exclude/banned/reparse/columns)"
                 "即完整声明,code/train.py 的 exclude_columns() 运行期读取;"
                 "banned 列已从数据面物理剔除(features.banned 仅作终检参考);"
                 "Agent 修改超参时更新 hyperparameters,代码只保留默认值",
    }
    dump_yaml(Path(trial.trial_dir) / "config" / "trial.yaml", trial_cfg)
    # [FIX-2026-09-03] hpo 模板渲染路径:config 已落盘,追加 hpo 节
    # (search_space/budget/inner_split;与快通道同一消费契约)
    if args.action == "hpo":
        try:
            _inject_hpo_config(run_dir, trial, args.search_space,
                               int(args.budget_configs or config.get(
                                   "evolution.hpo.default_budget", 40)),
                               skip=True,
                               sample_rows=getattr(
                                   args, "hpo_sample_rows", None),
                               hpo_opts=_hpo_opts_from_args(args))
        except SystemExit:
            _rollback_trial(trial)
            raise
    if _hurdle_tpl:
        # hurdle 超参段(HPO / --set 分段键 / confirm 与交付复跑同一消费口径),
        # 与 --from-parent hurdle 父代注入的是同一段(幂等替换)
        from .render import _import_probe, _inject_hurdle_hp_segment

        try:
            _h_notes = _inject_hurdle_hp_segment(code_dir / "train.py")
            _bad = _import_probe(Path(trial.trial_dir))
            if _bad:
                raise SystemExit(f"hurdle 模板可导入性探针失败:{_bad}")
        except SystemExit as _exc:
            logger.error("%s", _exc)
            _rollback_trial(trial)
            return 1
        trial.data.setdefault("execution", {})["hurdle_template"] = {
            "container": "runtime.hurdle.HurdleRegressor",
            "hp_segment": _h_notes}
        if args.action == "hpo":
            trial.data["execution"]["hpo_topology"] = "structural:hurdle"
        trial.flush()
    # v2.17.0:创建期后处理(add-blocks / 回归单分支 / 交付契约自检)
    if _post_create_code(run_dir, spec, trial, args) != 0:
        return 1
    (Path(trial.trial_dir) / "hypothesis.md").write_text(
        f"# Hypothesis\n\n{hypothesis['statement']}\n\n"
        f"## Rationale\n\n{hypothesis.get('rationale') or '—'}\n\n"
        f"## Falsification\n\n{hypothesis.get('falsification') or '—'}\n",
        encoding="utf-8")
    print(trial.trial_id)
    print(f"  目录: {trial.trial_dir}")
    if _hurdle_tpl:
        print("  hurdle 模板:runtime.hurdle.HurdleRegressor(两段各自早停、子模型"
              "缓存、可 clone);超参改常量 HURDLE_* 或 --from-parent --set "
              "clf__/reg__<键>;HPO 用 --action hpo --from-parent <本 trial>")
    print(f"  下一步: 编辑 code/train.py 后执行 "
          f"modelevo.py run --run {run_dir} --trial {trial.trial_id}")
    return 0


def _post_create_code(run_dir: Path, spec, trial, args) -> int:
    """[v2.17.0] new-trial 两条路径共用的创建期后处理;失败回滚并返回 1。

    1. `--add-blocks`:字节级复制源 trial 的特征块段 + 指纹校验(#16);
    2. 回归任务的打分分派折叠为单分支(#7;嫁接/升级/注入后的最终代码再过一遍,
       幂等);
    3. structural 拓扑的交付契约自检(#5 第 4 点):只告警 + 写 trial.yaml
       的 deliverability,不拦截。
    """
    from .render import (_narrow_dispatch_file, apply_add_blocks,
                         deliverability_probe)

    tp = Path(trial.trial_dir) / "code" / "train.py"
    # v2.19.0(#15):--patch 先于 --add-blocks 合并进父代副本;校验(导入/注册块
    # 探针/lint 增量)放在全部代码改写之后,对最终落盘源码做
    _patch_rep = None
    if getattr(args, "patch", None):
        from .trial_patch import apply_patch

        try:
            _patch_rep, _ = apply_patch(trial, str(args.patch))
        except SystemExit as exc:
            logger.error("%s", exc)
            _rollback_trial(trial)
            return 1
    if getattr(args, "add_blocks", None):
        try:
            rep = apply_add_blocks(run_dir, trial, args.add_blocks)
        except SystemExit as exc:
            logger.error("%s", exc)
            _rollback_trial(trial)
            return 1
        trial.data["added_blocks"] = [
            {k: f.get(k) for k in ("source_trial", "wrapper",
                                   "feature_block_name", "code_fingerprint",
                                   "n_columns", "train_block_key",
                                   "cache_present", "match")}
            for f in rep.get("fingerprints") or []]
        trial.data.setdefault("execution", {})["add_blocks"] = {
            "spec": str(args.add_blocks),
            "copied": rep.get("copied") or [],
            "skipped_identical": rep.get("skipped_identical") or [],
            "conflicts_kept_new_trial": [c["names"] for c in
                                         rep.get("conflicts") or []]}
        trial.flush()
        for f in rep.get("fingerprints") or []:
            print(f"  add-blocks: {f['source_trial']}:{f['wrapper']} "
                  f"(块 {f.get('feature_block_name')}, {f.get('n_columns')} 列) "
                  f"指纹 {f.get('code_fingerprint')} 一致"
                  + ("" if f.get("cache_present") is None else
                     f";train 缓存{'已在' if f.get('cache_present') else '未在'}盘上"))
    try:
        _n = _narrow_dispatch_file(tp, str(spec.model_type))
    except Exception as exc:  # noqa: BLE001  折叠失败不阻断(lint 兜底)
        _n = 0
        logger.warning("回归单分支渲染跳过:%s", exc)
    if _n:
        trial.data.setdefault("execution", {})["regression_single_branch"] = _n
        trial.flush()
    if _patch_rep is not None:
        from .trial_patch import trial_patch_record, verify_patched_trial

        try:
            verify_patched_trial(run_dir, trial, str(args.from_parent), _patch_rep)
        except SystemExit as exc:
            logger.error("%s", exc)
            _rollback_trial(trial)
            return 1
        # trial.yaml 留痕:合并结果才是被冻结的源码,这里只记 patch 来源与差异面
        trial.data["patch"] = trial_patch_record(_patch_rep)
        trial.flush()
        print(f"  patch: 替换 {_patch_rep['replaced'] or '无'};新增 "
              f"{_patch_rep['added'] or '无'};注册块 "
              f"{[b['wrapper'] for b in _patch_rep['registered_blocks']] or '无'}"
              f"(sha {_patch_rep['source_sha'][:12]})")
    _decl = str(trial.data.get("declared_topology") or "single")
    _structural = _decl.startswith("structural:")
    if not _structural:
        try:
            from runtime.topology import analyze_code

            _structural = analyze_code(tp.read_text(encoding="utf-8")).get(
                "kind") == "structural"
        except Exception:  # noqa: BLE001
            _structural = False
    if _structural:
        deliv = deliverability_probe(Path(trial.trial_dir), str(spec.model_type))
        trial.data["deliverability"] = deliv
        trial.flush()
        if not deliv.get("deliverable") or deliv.get("issues") or \
                deliv.get("clone_ok") is False:
            logger.warning(
                "交付契约自检(structural,不拦截):clone_ok=%s rerun_recipe_ok=%s "
                "issues=%s —— 不可 clone 的容器交付只能走 rerun_recipe;两者皆不"
                "满足时收尾交付会失败,建议改用 runtime.hurdle.HurdleRegressor"
                "(new-trial --hurdle-template)", deliv.get("clone_ok"),
                deliv.get("rerun_recipe_ok"), deliv.get("issues"))
        print(f"  交付契约自检: deliverable={deliv.get('deliverable')} "
              f"clone_ok={deliv.get('clone_ok')} "
              f"rerun_recipe_ok={deliv.get('rerun_recipe_ok')} "
              f"preferred_refit={deliv.get('preferred_refit')}")
    return 0


def _resolve_features_override(cols: list[str], spec, derived: list[str]) -> list[str]:
    """把显式白名单转成最终特征面:按 derived 顺序、剔除目标/时间/ID/
    排除列(与 resolve_feature_columns 同口径)。空白名单 = 无特征(拒绝)。
    """
    drop = {spec.target_name, spec.time_column, spec.id_column}
    drop.discard(None)
    if spec.treatment_column:
        drop.add(spec.treatment_column)
    drop.update(spec.excluded_columns)
    ordered = [c for c in derived if c in cols] or [c for c in cols
                                                    if c not in drop]
    if not ordered:
        raise ValueError("特征面白名单为空(全部被排除列剔除)")
    return ordered


def _retarget_splits_to_view(run_dir: Path, trial, data_view: str) -> None:
    """[v2.13.5] `--from-parent` 复制的代码按子 trial 的 data_view 改写 SPLITS。

    模板渲染路径早就会改写(见 cmd_new_trial 的数据视图段),快通道复制
    路径没有:子 trial 声明 data_view=sample 却读父代的 v1 面,或反过来
    (req-t7-rand400w gen-01:sample → v1 重建时 SPLITS 仍指向
    splits-sample,"声明 v1 实读 sample",靠人工 grep 才发现)。这里把
    模块级 `SPLITS = RUN_DIR / ...` 一行改写为本 trial 视图目录;
    找不到该行时只告警(手写代码的其它取数方式不在此保证范围)。
    """
    import re as _re

    from runtime.data import view_splits_dir

    code_path = Path(trial.trial_dir) / "code" / "train.py"
    if not code_path.exists():
        return
    code = code_path.read_text(encoding="utf-8")
    rel = str(view_splits_dir(run_dir.resolve(), data_view or "v1")
              .relative_to(run_dir.resolve()))
    target = (f'SPLITS = RUN_DIR / "sample-profile" / "splits"'
              if rel == "sample-profile/splits"
              else f"SPLITS = RUN_DIR / {rel!r}")
    pat = _re.compile(r"(?m)^SPLITS = RUN_DIR / .*$")
    m = pat.search(code)
    if not m:
        logger.warning("trial %s: 未找到模块级 `SPLITS = RUN_DIR / ...`,"
                       "无法按 data_view=%s 改写取数目录(手写代码请自行核对)",
                       trial.trial_id, data_view)
        return
    if m.group(0).strip() == target:
        return
    code_path.write_text(code[:m.start()] + target + code[m.end():],
                         encoding="utf-8")
    logger.info("trial %s: SPLITS 按 data_view=%s 改写:%s → %s",
                trial.trial_id, data_view, m.group(0).strip(), target)


def _maybe_detach(args, run_dir: Path, tag: str) -> int | None:
    """``--detach`` 时把本次执行体拉到独立会话,父进程立即返回。

    返回 None 表示"未 detach,继续在本次调用里执行";返回 int 表示
    "已 detach,直接以该返回码退出"(打印 pid 供调用方/wait 追踪)。

    设计见 ``runtime/daemon.py`` 的模块 docstring(gen-04 三个 HPO trial
    三次随 agent 会话同时死亡的实测)。要点:worker 是**同一个 CLI
    命令**,只把 --detach 换成 --no-detach,因此 run-generation 的
    并行度、超时、资源计划等语义完全不变——变的只是它挂在哪个会话下。
    """
    if getattr(args, "no_detach", False) or not getattr(args, "detach", False):
        return None
    from runtime.daemon import (build_worker_argv, is_detached_child,
                                spawn_detached, write_pid_file)

    if is_detached_child():
        return None  # 已经是分离进程,不再套娃

    _raw = list(getattr(args, "_raw_argv", None) or sys.argv[1:])
    if not _raw:
        logger.error("detach 失败:无法重建 worker 命令(缺原始参数)")
        return 2
    argv = [sys.executable, str(_SKILL_ROOT / "scripts" / "modelevo.py"),
            *build_worker_argv(_raw)]
    log_path = run_dir / "logs" / f"detached-{tag}.log"
    pid = spawn_detached(argv, log_path=log_path, cwd=Path.cwd())
    if pid <= 0:
        logger.error("detach 失败:未取到 worker pid(见 %s)", log_path)
        return 2
    pid_file = run_dir / "logs" / f"detached-{tag}.pid"
    write_pid_file(pid_file, pid)
    logger.info(
        "已 detach:worker pid=%s(日志 %s,pid 文件 %s)。"
        "训练不再随本会话退出而终止;用 `wait` 或 `status` 追踪终态。",
        pid, log_path, pid_file)
    print(f"DETACHED\t{tag}\tpid={pid}\tlog={log_path}")
    return 0


def _topology_precheck(run_dir: Path, spec, trial) -> dict:
    """v2.16.0:前台拓扑门预检(detach 之前)。

    detached 子进程里被拦,父进程早已返回、`wait --trial` 只会空等到超时;
    所以在派生前用同一裁决函数先判一次,被拦即以退出码 5 返回。
    """
    from runtime.topology import evaluate_gate, gate_message, record_request

    if trial.status not in ("PROPOSED", "CODE_WRITTEN", "FAILED"):
        return {"allowed": True}
    gate = evaluate_gate(run_dir, trial, spec.model_type)
    if not gate.get("allowed", True):
        trial.data["model_topology"] = gate
        trial.flush()
        record_request(run_dir, trial, gate)
        logger.error("%s", gate_message(trial.trial_id, gate))
    return gate


def cmd_run_trial(args) -> int:
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    _tdir0 = run_dir / "trials" / args.trial
    if (_tdir0 / "trial.yaml").exists():
        from runtime.topology import EXIT_APPROVAL_REQUIRED
        from runtime.trial import TrialRecord as _TR

        _t0 = _TR.load(_tdir0)
        if not _topology_precheck(run_dir, spec, _t0).get("allowed", True):
            print(f"TOPOLOGY_BLOCKED\t{_t0.trial_id}\t"
                  f"{(_t0.data.get('model_topology') or {}).get('kind')}")
            return EXIT_APPROVAL_REQUIRED
    rc = _maybe_detach(args, run_dir, f"run-{args.trial}")
    if rc is not None:
        return rc
    config = _load_config(run_dir)
    factory = TrialFactory(run_dir, spec=spec, config=config)
    trial_dir = run_dir / "trials" / args.trial
    if not trial_dir.exists():
        logger.error("Trial 不存在: %s", trial_dir)
        return 1
    from runtime.trial import TrialRecord

    trial = TrialRecord.load(trial_dir)

    runner = TrialRunner(run_dir, spec, config)
    if getattr(args, "reset_breaker", False):
        runner.reset_breaker()
        logger.info("熔断状态已重置(root-cause reflection 应已写入 Task Experience)")
    # ---- v14:Trial Lint 硬闸(执行前,失败即拒跑)----
    # 旧实现 lint_trial.py 从不被 CLI 调用(init 只复制进工作区),
    # 全部规则(含泄漏 9a/9b/9c)都是建议性的,依赖 Agent 自觉。
    # 快通道 --from-parent 零写码路径尤其需要这道闸:代码未经人眼。
    # MODELEVO_SKIP_LINT=1 逃生口(诊断场景),默认强制。
    if os.environ.get("MODELEVO_SKIP_LINT") != "1":
        lint_path = run_dir / "lint_trial.py"
        if not lint_path.exists():  # 旧工作区兜底:从 skill 复制
            lint_path = _SKILL_ROOT / "assets" / "lint_trial.py"
        if lint_path.exists():
            import importlib.util as _ilu

            _lspec = _ilu.spec_from_file_location("modelevo_lint",
                                                  lint_path)
            _lm = _ilu.module_from_spec(_lspec)
            _lspec.loader.exec_module(_lm)
            # v2.16.0:规则 8 按早停面口径生效 —— 把 run 配置(含 skill 级
            # global-config)的 training.early_stop_source 交给 lint;已显式
            # 设置的 MODELEVO_ES_SOURCE 优先
            _es_prev = os.environ.get("MODELEVO_ES_SOURCE")
            if not _es_prev:
                os.environ["MODELEVO_ES_SOURCE"] = str(config.get(
                    "training.early_stop_source", "validation") or "validation")
            # v2.16.0:规则 2(融合正则)只对经人工批准的 cascade/ensemble
            # 拓扑放行;spec.ensemble_enabled 由 Agent 起草,不再授权
            from runtime.topology import composite_approved, evaluate_gate

            _composite_ok = composite_approved(
                evaluate_gate(run_dir, trial, spec.model_type))
            try:
                _lint_errors = _lm.lint_trial(
                    run_dir, trial_dir, _composite_ok)
            finally:
                if not _es_prev:
                    os.environ.pop("MODELEVO_ES_SOURCE", None)
            if _lint_errors:
                for _e in _lint_errors:
                    logger.error("Trial Lint: %s", _e)
                logger.error("lint 未通过(可用 MODELEVO_SKIP_LINT=1 "
                             "跳过,仅限诊断;修复 code/ 后重试)")
                return 1
    parents = trial.parent_ids
    parent_scores = _load_best_parent_scores(run_dir, spec, parents)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    baseline_id = _resolve_baseline_id(run_dir, spec, factory)
    baseline_score = None
    baseline_view = None
    if baseline_id:
        bl = load_parent_scores(run_dir, baseline_id)
        baseline_score = bl.get("validation") if bl else None
        # v2.13.2:视图随分数一起传(跨视图不可配对,见 runner 侧门)
        baseline_view = bl.get("data_view") if bl else None
    slice_cols = [c for c in (spec.time_column, spec.id_column) if c]

    # ---- 运行期 Harness 校验(§19.3):hash 不一致视为新协议 ----
    # v14:唯一受支持出口是 acknowledge-harness-change(登记 diff+理由,
    # epoch+1,新旧 epoch 由 leaderboard 拒绝混排)——手改 run-state.json
    # 会清空铁律 10 的审计留痕(实测一次运行被手改三次)
    stamp = protocol_stamp(_SKILL_ROOT / "runtime")
    recorded = st.get("harness_hash")
    # v2.17.2(B8):变更全部落在 plumbing 模块时自动登记,不再拦截
    if recorded and recorded != stamp["harness_hash"] \
            and _auto_ack_plumbing(run_dir, st, stamp) is not None:
        recorded = stamp["harness_hash"]
    if recorded and recorded != stamp["harness_hash"]:
        logger.error(
            "Harness 代码已变更(记录 %s,当前 %s):视为新协议,"
            "历史 Trial 不与新协议混排。确属非测量面改动时用"
            " `modelevo.py acknowledge-harness-change --run %s "
            "--reason ...` 登记(留痕于 harness_change_log);"
            "measurement 改动但实测分数口径未变时加"
            " --measurement-no-score-impact(epoch 不变,历史仍可比),"
            "不要手改 run-state.json",
            recorded, stamp["harness_hash"], run_dir)
        return 1
    if not recorded:
        _update_run_state(run_dir, **stamp)
    # ---- v13 泄漏分类冻结校验:分类变更 = 新协议(与 harness_hash
    # 同口径)。banned/reparse 任一方向变更(含收紧)都会改变数据面,
    # 历史 trial 的分数不再可比,必须重新 baseline。hard-ban→reparse
    # 尤其危险(等于给泄漏列解封),只在 spec 重写(新任务)时发生。
    leak_hash = spec.leak_manifest_hash
    recorded_leak = st.get("leak_manifest_hash")
    if recorded_leak and recorded_leak != leak_hash:
        logger.error(
            "泄漏分类已变更(记录 %s,当前 %s):数据面定义变了,"
            "视为新协议,必须重新 baseline;hard-ban→reparse 的解封"
            "需要用户显式确认后重写 spec",
            recorded_leak, leak_hash)
        return 1
    if not recorded_leak:
        _update_run_state(run_dir, leak_manifest_hash=leak_hash)
    # ---- 环境变化记账(v20,替代 v11 的"env 变更=新协议判死")--
    # 装 optional 包不改已跑 trial 的任何字节(用了什么算法记在 trial
    # 的 algorithm 字段),用 run 级 kill switch 处理 per-trial 属性会
    # 把合法演进动作判成违规;真正让历史 trial 不可复现的是已装
    # required 包的版本变化,那是**记录 + 归因**的事,不是叫停运行的事
    # (用户决策 2026-09:不拦 pip,只记账)。harness 与泄漏分类仍判死。
    if st.get("phase") in ("BASELINED", "EVOLVING") and st.get(
            "env_fingerprint"):
        from runtime.preflight import DRIFT_KINDS, check_env_drift, record_env_events

        drift = check_env_drift(st, _SKILL_ROOT)
        if drift:
            record_env_events(run_dir, drift["events"],
                              phase=st.get("phase"),
                              generation=trial.data.get("generation"))
            logger.warning(
                "环境已变化(运行继续;已在 %s 记账):%s",
                run_dir / "env-ledger.jsonl",
                "; ".join(f"{k}={DRIFT_KINDS.get(k, k)}"
                          for k in drift["kinds"]))
        # 指纹刷新:ledger 已留下"变化前"的基线,这里同步成当前值,
        # 免得同一次变化在每个 trial 上重复记账
        cur_fp = (drift or {}).get("current")
        if cur_fp and cur_fp != st.get("env_fingerprint"):
            from runtime.preflight import run_preflight as _rp

            _manifest = _rp(_SKILL_ROOT)
            _update_run_state(run_dir, env_fingerprint=cur_fp,
                              env_manifest=_manifest)
            st["env_fingerprint"] = cur_fp
            st["env_manifest"] = _manifest

    # ---- 运行期 harness hash 写入 trial(§19.3)----
    trial.data["harness_hash"] = stamp["harness_hash"]
    # v14:epoch 记到 trial 上,leaderboard 据此拒绝跨 epoch 混排
    trial.data["harness_epoch"] = int(st.get("harness_epoch", 0))
    # v20:trial 盖上当时的 env 指纹。漂移只记录不阻断,若不在 trial
    # 上留痕,"哪个 trial 在哪个环境下实测的"就无从回答(只剩漂移之后
    # 跑的那批能确定)。
    trial.data["env_fingerprint"] = st.get("env_fingerprint")
    trial.flush()

    # ---- heavy trial 超时放大(协议 v7):特征重建需要更长预算 ----
    # v14:hpo trial 由 _inject_hpo_config 自动打 heavy 标记,同机制放大
    timeout_minutes = None
    heavy = (trial.data.get("expected_resources") or {}).get("heavy")
    if heavy:
        timeout_minutes = runner.resolve_timeout_minutes() * float(
            trial.data["expected_resources"].get("timeout_multiplier",
                                                 config.get(
                                                     "evolution.heavy_trials"
                                                     ".timeout_multiplier",
                                                     4)))
        # v14:上限封顶(hpo space 大时 4×90=360 正常,但不许无界)
        cap = float(config.get("evolution.hpo.max_timeout_minutes", 480))
        timeout_minutes = min(timeout_minutes, cap)
        logger.info("heavy trial:超时放大至 %.1f 分钟(上限 %.0f)",
                    timeout_minutes, cap)

    runner.run(trial, parent_scores=parent_scores,
               baseline_score=baseline_score, baseline_view=baseline_view,
               slice_columns=slice_cols,
               timeout_minutes=timeout_minutes,
               slot_index=int(getattr(args, "slot_index", 0) or 0))
    # evidence 记录
    EvidenceStore(run_dir).record_trial(trial)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    lb = Leaderboard(run_dir, spec)
    lb.build(factory.all_trials(), baseline_id=st.get("baseline_id"))
    ev = trial.data.get("evaluation") or {}
    verdict = ev.get("verdict")
    print(f"{trial.trial_id}: status={trial.status} verdict={verdict}")
    if ev.get("primary_metric"):
        print(f"  {spec.primary_metric}="
              f"{ev['primary_metric'].get('value')}")
    if verdict and verdict != "invalid":
        # REFLECTED 需要 reflection.md 落盘(§10.3);Agent 未写时保持
        # EVALUATED 并提示,不虚推状态
        if trial.status == "EVALUATED":
            if (trial_dir / "reflection.md").exists():
                trial.transition("REFLECTED")
                trial.flush()
            else:
                print("  提示: 写入 trials/…/reflection.md 后再次 run 同一 "
                      "Trial 可推进到 REFLECTED;或由 finalize-generation 汇总")
    # v10.3:结构化摘要(--summary 或默认输出)——一行机器可读结果,
    # 替代 sleep+tail 轮询读日志:Generation Agent 只需解析这一行即可
    # 拿到 verdict/指标/delta/失败原因,原始日志无需进 LLM 上下文。
    # v17.2:格式单源 _trial_summary_line(cmd_wait 复用同一格式)。
    if getattr(args, "summary", True):
        print(_trial_summary_line(trial, spec))
    return 0


def _topology_filter(run_dir: Path, spec, trials: list) -> tuple[list, list]:
    """v2.16.0:整代拓扑预检——被拦的 trial 不进批(不训练、不冻结、不记
    失败),其余照常执行(运行中的裁决不阻塞算力)。"""
    ok, blocked = [], []
    for t in trials:
        try:
            g = _topology_precheck(run_dir, spec, t)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 拓扑预检异常(放行,runner 侧会再判): %s",
                           t.trial_id, exc)
            g = {"allowed": True}
        (ok if g.get("allowed", True) else blocked).append(t)
    if blocked:
        logger.warning(
            "generation 有 %d 个 trial 因模型拓扑未获人工批准被跳过: %s"
            "——已写入 logs/user-decisions.md;批准只能由用户在自己的终端"
            "执行 approve-topology,Agent 继续其它方向",
            len(blocked), ", ".join(t.trial_id for t in blocked))
    return ok, blocked


def _split_already_running(trials) -> tuple[list, dict]:
    """v2.17.2(B2):(可进批的 trial, {已有活 runner 的 trial_id: pid})。

    判据与 wait 快路径同源(runtime.daemon.runner_lock_alive:trial 目录
    `.runner.lock` 的 pid 存活)。只读探测,不加载 trial 以外的任何东西。
    """
    from runtime.daemon import runner_lock_alive

    keep, running = [], {}
    for t in trials:
        pid = None
        try:
            pid = runner_lock_alive(Path(t.trial_dir))
        except Exception:  # noqa: BLE001  探针失败按无锁处理(旧行为)
            pid = None
        if pid is not None and pid != os.getpid():
            running[t.trial_id] = int(pid)
        else:
            keep.append(t)
    return keep, running


def cmd_run_generation(args) -> int:
    """并行执行一代 Trial,施加有期限 barrier(§11.3/§16.5.2)。"""
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    if getattr(args, "detach", False):
        # 前台先判拓扑门:detached worker 里被拦,调用方只会看到 GEN_DONE
        from runtime.topology import EXIT_APPROVAL_REQUIRED

        _gid0 = int(args.generation)
        _cands = [t for t in TrialFactory(run_dir, spec=spec).all_trials()
                  if t.generation == _gid0
                  and (Path(t.trial_dir) / "code" / "train.py").is_file()
                  and t.status in ("PROPOSED", "CODE_WRITTEN", "FAILED")]
        _ok0, _blocked0 = _topology_filter(run_dir, spec, _cands)
        for _t in _blocked0:
            print(f"TOPOLOGY_BLOCKED\t{_t.trial_id}\t"
                  f"{(_t.data.get('model_topology') or {}).get('kind')}")
        if _cands and not _ok0:
            return EXIT_APPROVAL_REQUIRED
    rc = _maybe_detach(args, run_dir, f"gen-{int(args.generation):02d}")
    if rc is not None:
        return rc
    config = _load_config(run_dir)
    factory = TrialFactory(run_dir, spec=spec, config=config)
    runner = TrialRunner(run_dir, spec, config)
    from runtime.gen_executor import GenerationExecutor

    gid = int(args.generation)
    # v19.4:状态过滤——PROPOSED(代码未写)的 trial 不进批。旧实现
    # 无过滤,提前调用(流水线期)会把空 code/ 目录的 trial 打成
    # FAILED(infra_error),毒化状态机;现在显式跳过并提示,配合
    # --wait-code 可等代码落盘(见下方 wait 循环)。
    all_gen = [t for t in factory.all_trials() if t.generation == gid]

    def _code_present(t) -> bool:
        """v2.10.1/F4:以 code/train.py 实际存在且非空为就绪真值。

        旧实现只看 status 字段:--from-parent 快通道复制父代 code/ 但
        不推进状态机,于是这些 trial 永远停在 PROPOSED,被本命令整代
        跳过(实测连续两代,avg_parallelism ≈0.84)。看文件而非状态,
        既不误跳已写好的 trial,也仍然挡住空 code/ 目录。"""
        p = run_dir / "trials" / t.trial_id / "code" / "train.py"
        try:
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False

    runnable = [t for t in all_gen
                if t.status != "PROPOSED" or _code_present(t)]
    skipped = [t.trial_id for t in all_gen if t not in runnable]
    # v2.17.2(B2):已有活 runner(`run --detach` 正在跑)的 trial 不进批。
    # 旧实现只看代码是否落盘:runner.run 对它抛"并发 run"→ 批路径吞异常标
    # FAILED(infra_error),barrier 兜底再把 RUNNING 改 timeout;下方盖
    # harness_hash 的 flush 还会用旧快照覆写它的 trial.yaml(run4 opus5.5)。
    runnable, already_running = _split_already_running(runnable)
    if already_running:
        logger.warning(
            "generation %d 有 %d 个 trial 已有活 runner,本批跳过且不改写"
            "其状态: %s(用 wait --trial 等它们自己的终态)",
            gid, len(already_running),
            ", ".join(f"{tid}(pid={pid})"
                      for tid, pid in already_running.items()))
    if skipped:
        logger.warning(
            "generation %d 有 %d 个 PROPOSED trial(代码未落盘)被跳过:"
            " %s —— 全部写完后再跑一次本命令,或用 --wait-code 等待",
            gid, len(skipped), ", ".join(skipped))
    wait_code = float(getattr(args, "wait_code", 0) or 0)
    if wait_code > 0 and skipped:
        import time as _wt

        _deadline = _wt.time() + wait_code
        while _wt.time() < _deadline:
            _new = {t.trial_id: t for t in factory.all_trials()
                    if t.generation == gid}
            skipped = [tid for tid in skipped
                       if (_new.get(tid).status == "PROPOSED")]
            runnable = [t for t in all_gen
                        if t.trial_id not in skipped
                        and t.trial_id not in already_running]
            if not skipped:
                break
            _wt.sleep(min(15.0, max(1.0, _deadline - _wt.time())))
        if skipped:
            logger.error(
                "--wait-code 超时,仍有 PROPOSED trial 未落盘代码: %s;"
                "仅执行已就绪的 %d 个", ", ".join(skipped), len(runnable))
    trials = runnable
    if not trials:
        if already_running:
            # v2.17.2(B2):全部在跑 ≠ 失败;只报告,不动它们
            print(json.dumps({"executed": [], "n_succeeded": 0,
                              "n_failed": 0,
                              "already_running": sorted(already_running),
                              "already_running_pids": already_running},
                             ensure_ascii=False, indent=2))
            return 0
        logger.error("generation %d 无就绪 Trial(全部 PROPOSED 或为空)",
                     gid)
        return 1
    # v2.16.0 拓扑门:未获批准的级联/融合 trial 不进批
    _gate_cands = [t for t in trials
                   if t.status in ("PROPOSED", "CODE_WRITTEN", "FAILED")]
    _, _blocked = _topology_filter(run_dir, spec, _gate_cands)
    if _blocked:
        _bids = {t.trial_id for t in _blocked}
        trials = [t for t in trials if t.trial_id not in _bids]
        if not trials:
            from runtime.topology import EXIT_APPROVAL_REQUIRED

            logger.error("generation %d 的就绪 trial 全部等待拓扑批准", gid)
            return EXIT_APPROVAL_REQUIRED
    sched = Scheduler(config)
    # 按本批实际 trial 数收窄并发槽位,线程反比放大(§16.3 不变式)
    plan = sched.plan(planned_trials=len(trials))
    # v2.17.0:内存口径并发低于 CPU 口径时,按内存口径重排线程——否则
    # 准入只允许 2 路同时训练,每路却只分到 8 路口径的线程(run4
    # regression.2 实测:可用 370 GB、单 trial 152 GB → 2 路,CPU 口径 8 路)
    try:
        from runtime.scheduler import resource_admission_view

        _adm = resource_admission_view(run_dir, config,
                                       max_parallel=len(trials))
        _mem_par = _adm.get("suggested_parallel")
        if _adm.get("enabled") and isinstance(_mem_par, int) \
                and 1 <= _mem_par < plan.n_parallel_trials:
            plan = sched.plan(planned_trials=_mem_par)
            logger.info(
                "run-generation:内存口径并发 %d < CPU 口径,按 %d 路重排线程"
                "(单 trial 估计 %.0f GB,可用 %.0f GB);其余 trial 由准入排队",
                _mem_par, plan.n_parallel_trials,
                float(_adm.get("per_trial_estimate_gb") or 0),
                float(_adm.get("available_gb") or 0))
    except Exception as exc:  # noqa: BLE001 —— 预估失败退回 CPU 口径
        logger.debug("内存口径并发预估失败(沿用 CPU 口径): %s", exc)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}

    # 运行期 Harness 校验(§19.3)
    # v14:唯一受支持出口是 acknowledge-harness-change(见 cmd_run_trial)
    stamp = protocol_stamp(_SKILL_ROOT / "runtime")
    # v2.17.2(B8):变更全部落在 plumbing 模块时自动登记,不再拦截
    if st.get("harness_hash") and st["harness_hash"] != stamp["harness_hash"] \
            and _auto_ack_plumbing(run_dir, st, stamp) is not None:
        st = load_json(run_dir / "run-state.json")
    if st.get("harness_hash") and st["harness_hash"] != stamp["harness_hash"]:
        logger.error(
            "Harness 代码已变更(%s → %s):视为新协议。确属非测量面改动"
            "时用 `modelevo.py acknowledge-harness-change --run %s "
            "--reason ...` 登记,不要手改 run-state.json",
            st["harness_hash"], stamp["harness_hash"], run_dir)
        return 1
    for t in trials:
        t.data["harness_hash"] = stamp["harness_hash"]
        # v14:epoch 记到 trial 上,leaderboard 据此拒绝跨 epoch 混排
        t.data["harness_epoch"] = int(st.get("harness_epoch", 0))
        t.flush()

    # wall-time 预算(TaskSpec 优先)
    wall_budget = spec.resource_budget.get("max_wall_time_hours") \
        or config.get("execution.max_wall_time_hours", 24)
    # v2.8.1:baseline_id 兜底解析(缺失时取 gen-00 首个 trial 并回写)
    baseline_id = _resolve_baseline_id(run_dir, spec, factory)
    # v2.15.8:wall 余量 fail-closed 预检接线(GenerationExecutor 早有该
    # 参数,但没有任何调用方传 —— 余量不够跑完最短单 trial 时,整代照样
    # 启动、到点被杀,算力白烧)。只对新 run 启用(旧 run 中途接入可能
    # 让一个本来还能跑完的尾代被直接拒绝)。
    from runtime import version_at_least
    _wall_remaining_sec = None
    _rs = load_json(run_dir / "run-state.json") \
        if (run_dir / "run-state.json").exists() else {}
    if version_at_least(_rs.get("skill_version"), "2.15.8") \
            and _rs.get("started_at_epoch"):
        _wall_remaining_sec = (float(wall_budget) * 3600.0
                               - (time.time() - float(_rs["started_at_epoch"])))
    summary = GenerationExecutor(runner).execute_generation(
        trials, plan=plan, wall_time_budget_hours=float(wall_budget),
        run_dir=run_dir, baseline_id=baseline_id,
        wall_remaining_sec=_wall_remaining_sec)
    # v2.17.2(B2):批前探测与执行器内复查到的在跑 trial 合并呈现;它们的
    # 记录是旧快照,不进 evidence(由其自身 runner 收尾)
    if already_running:
        _ar = dict(summary.get("already_running_pids") or {})
        _ar.update(already_running)
        summary["already_running"] = sorted(_ar)
        summary["already_running_pids"] = dict(sorted(_ar.items()))
    _skip_ids = set(summary.get("already_running") or [])
    es = EvidenceStore(run_dir)
    for t in trials:
        if t.trial_id in _skip_ids:
            continue
        es.record_trial(t)
    lb = Leaderboard(run_dir, spec)
    lb.build(factory.all_trials(), baseline_id=baseline_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def _basis_summary(run_dir: Path, gid: int, trials) -> dict:
    """本代 trial 的依据统计 + 全 run "展示过却从未引用"的原则数(v2.18.0)。"""
    import json as _json

    gen_trials = [t for t in trials if t.data.get("generation") == gid
                  and t.data.get("search_action") not in ("warm_start",)]
    cited, novel, missing, prefilled, refs = 0, 0, 0, 0, {}
    for t in gen_trials:
        b = t.data.get("basis")
        if not b:
            missing += 1
        elif b.get("novel"):
            novel += 1
        else:
            cited += 1
            prefilled += b.get("source") == "prefilled"
            for r in b.get("refs") or []:
                refs[r["id"]] = refs.get(r["id"], 0) + 1
    all_cited = {r["id"] for t in trials for r in ((t.data.get("basis") or {}).get("refs") or [])}
    shown: set = set()
    tel = run_dir / "experience-retrieval-telemetry.jsonl"
    if tel.exists():
        for line in tel.read_text(encoding="utf-8").splitlines():
            try:
                shown.update(_json.loads(line).get("shown_ids") or [])
            except ValueError:
                continue
    return {"trials": len(gen_trials), "cited": cited, "prefilled_kept": prefilled,
            "novel": novel, "missing": missing,
            "cited_principles": refs,
            "shown_never_cited": len(shown - all_cited),
            "shown_never_cited_ids": sorted(shown - all_cited)[:20]}


def cmd_finalize_generation(args) -> int:
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    config = _load_config(run_dir)
    engine = EvolutionEngine(run_dir, spec, config)
    factory = TrialFactory(run_dir, spec=spec, config=config)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    baseline_metric = None
    bid = st.get("baseline_id")
    for t in factory.all_trials():
        if t.trial_id == bid:
            baseline_metric = (t.data.get("evaluation") or {}).get(
                "primary_metric", {}).get("value")
    wall = (time.time() - _started_at(run_dir)) / 3600
    # v16/P0-2:compute 口径(全部已启动 trial 的 runtime_sec 之和)
    # 传给 finalize → stop_report 双轨呈现;max_compute_hours 可配
    # (spec resource_budget 或 evolution.max_compute_hours)时参与
    # 停止判定,wall 保留为绝对上限护栏。
    # v2.17.2(B5):走单源 compute_hours_summary——在 trial Σruntime 之上
    # 计入收尾(finalize-all / evaluate-final)用时(finalize_hours 分列)
    from runtime.trial import compute_hours_summary

    compute_h = compute_hours_summary(
        run_dir, factory.all_trials())["compute_hours"]

    reflection = {}
    gid = int(args.generation)
    # 反思产物校验(§10.3):reflection.md 由 Agent 写;缺失时告警并保留占位
    refl_path = run_dir / "generations" / f"gen-{gid:02d}" / "reflection.md"
    if refl_path.exists():
        try:
            import yaml as _y

            reflection = _y.safe_load(refl_path.read_text(encoding="utf-8")) or {}
            if not isinstance(reflection, dict):
                reflection = {"raw": str(reflection)}
        except Exception:  # noqa: BLE001  非 yaml 反思按原文记录
            reflection = {"raw": refl_path.read_text(encoding="utf-8")[:2000]}
    else:
        logger.warning("generations/gen-%02d/reflection.md 缺失:"
                       "反思应由 Agent 在 finalize 前写入(§10.3)", gid)
    planned = [t.trial_id for t in factory.all_trials() if t.generation == gid]
    # v16/P2-4:空 trial 目录清理——创建半途终止的 trial(仅 trial.yaml,
    # 无 code/config,PROPOSED 未启动,如 120w run 的 trial-010)在
    # finalize 时标注 abandoned(不删目录,留痕;leaderboard 天然不含)
    for t in factory.all_trials():
        if t.generation == gid and not (
                Path(t.trial_dir) / "code" / "train.py").exists():
            t.data.setdefault("notes", "")
            if "abandoned" not in str(t.data.get("notes")):
                t.data["notes"] = (str(t.data.get("notes") or "")
                                   + ";v16:创建半途终止(无 code),"
                                   "标记 abandoned")
                t.flush()
                logger.info("空 trial 目录标记 abandoned: %s", t.trial_id)
    out = engine.finalize_generation(
        gid, planned, reflection=reflection,
        unexplored_high_potential=_parse_list(args.unexplored),
        wall_time_hours=wall, baseline_metric=baseline_metric,
        compute_hours=compute_h)
    # v18/P1-1:信息源清单刷新(finalize 节奏)——重建消费状态,保留
    # Agent 回填的粒度状态;unconsumed 计数进 stop 出口对账
    try:
        from runtime.inventory import refresh_inventory

        refresh_inventory(run_dir, spec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("information-inventory 刷新失败(不阻塞): %s", exc)
    EvidenceStore(run_dir).record_generation({
        "generation_id": f"gen-{gid:02d}",
        "planned_trials": planned,
        "best_metric": out.get("best_metric"),
        "archive_update": out.get("archive_update"),
    })
    # v18/P2-4:experience_hits 进 generation.yaml——检索遥测只写
    # 不读的两处断点补上 finalize 侧。telemetry 是 planner-context
    # 写的"本代检索命中了哪些经验卡";本代命中 = 文件里时间落在
    # 本代开始之后的所有记录(按 evolution-state.started_at 分代,
    # 近似口径可接受——反思要的是"命中什么、是否改变 plan")。
    try:
        _tel = run_dir / "experience-retrieval-telemetry.jsonl"
        if _tel.exists() and (run_dir / "generations" /
                              f"gen-{gid:02d}" / "generation.yaml").exists():
            import json as _json2

            _hits: list[str] = []
            _evo = load_json(run_dir / "evolution-state.json") or {}
            _evo_started = _evo.get("started_at") or ""
            _prev_finalized = set(_evo.get("finalized_generations") or [])
            _prev_finalized.discard(gid)  # 当前这代 finalize 中
            for _line in _tel.read_text(encoding="utf-8").splitlines():
                if not _line.strip():
                    continue
                try:
                    _rec = _json2.loads(_line)
                except Exception:  # noqa: BLE001
                    continue
                _hits.extend(str(h) for h in (_rec.get("hit_ids") or []))
            _gen_path = (run_dir / "generations" / f"gen-{gid:02d}"
                         / "generation.yaml")
            from runtime.logging_utils import load_yaml as _ly2, dump_yaml \
                as _dy2

            _gen_yaml = _ly2(_gen_path) or {}
            _gen_yaml["experience_hits"] = sorted(set(_hits))
            _gen_yaml["experience_hits_note"] = (
                "本代 planner-context 检索命中的经验卡(全史去重);"
                "反思必答:命中经验是否改变了本代 plan")
            _dy2(_gen_path, _gen_yaml)
    except Exception as exc:  # noqa: BLE001
        logger.warning("experience_hits 写入失败(不阻塞 finalize): %s", exc)
    # v2.18.0:经验使用情况汇总(只写本 run 的 generation.yaml,不写经验库)。
    # 证据关联(显式依据 + 按主题被动匹配)在收尾提交 run 快照时由经验库统一建立
    # (Q19),热启动 trial 作为配方复测进快照 —— 不再逐代回写全局库。
    try:
        _fb = _basis_summary(run_dir, gid, factory.all_trials())
        _gen_path = run_dir / "generations" / f"gen-{gid:02d}" \
            / "generation.yaml"
        if _gen_path.exists():
            _gen_yaml = load_yaml(_gen_path) or {}
            _gen_yaml["experience_feedback"] = _fb
            _gen_yaml["experience_feedback_note"] = (
                "本代 trial 的依据:cited = 引用经验库原则,novel = 明确为新方向,"
                "missing = 未写依据(旧 run 或未固定经验库视图);证据关联在收尾"
                "提交快照时由经验库建立。shown_never_cited = 本 run 展示 / 检索过"
                "但从未被任何 trial 引用的原则数")
            dump_yaml(_gen_path, _gen_yaml)
    except Exception as exc:  # noqa: BLE001
        logger.warning("experience_feedback 汇总失败(不阻塞 finalize): %s", exc)
    # Task Experience 增量更新(§14.1 L2)
    te = TaskExperience(run_dir)
    data = te.load() or te.auto_summarize(factory.all_trials(), spec)
    merged = te.auto_summarize(factory.all_trials(), spec)
    merged.update({k: v for k, v in data.items()
                   if k in ("key_findings", "reusable_experiences",
                            "task_specific_experiences", "final_recipe")
                   and data.get(k)})
    te.save(merged)
    _fin_out = {"archive_update": out["archive_update"],
                "stop_report": out["stop_report"] is not None,
                "stop_reason": (out["stop_report"] or {}).get("stop_reason"),
                "generation_delta": out.get("generation_delta"),
                "data_views_tried": out.get("data_views_tried"),
                "heavy_trials_in_generation":
                    out.get("heavy_trials_in_generation")}
    # v2.17.0(D3):代际 delta 口径与空代告警(空代 delta=None、不触发停止)
    if out.get("generation_delta_protocol"):
        _fin_out["generation_delta_protocol"] = out[
            "generation_delta_protocol"]
    if out.get("empty_generation"):
        _fin_out["empty_generation"] = True
        _fin_out["warning"] = out.get("empty_generation_warning")
    # v2.17.2(B7):上一代停止报告被本代取代(已归档进 stop_history)
    if out.get("stop_superseded"):
        _fin_out["stop_superseded"] = out["stop_superseded"]
    print(json.dumps(_fin_out, ensure_ascii=False, indent=2))
    return 0
