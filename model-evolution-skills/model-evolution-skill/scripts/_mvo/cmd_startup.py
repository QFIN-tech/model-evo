"""启动路径命令:init / validate / profile / baseline

v2.11.9 从单文件 `scripts/modelevo.py` 拆出(拆分动机见
`_mvo/__init__.py`)。**本文件是机械搬运,未改语义** —— 便于对照原
实现审阅。新增命令请按职责放进对应模块,不要回填到 `modelevo.py`
(那里只留 argparse 与分发)。
"""
from __future__ import annotations

from runtime.baseline import BaselineRunner  # noqa: E402
from runtime.templates import (DATA_LOADER_TEMPLATE, TASK_TEMPLATE,  # noqa: E402
                               TRIAL_TEMPLATE)
from runtime.generation import GenerationRecord  # noqa: E402
from runtime.leaderboard import Leaderboard  # noqa: E402
from pathlib import Path
from runtime.profile import SampleProfile  # noqa: E402
from runtime.scheduler import Scheduler, resource_snapshot  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialFactory, TrialRecord  # noqa: E402
from runtime.runner import TrialRunner, load_parent_scores  # noqa: E402
from runtime.logging_utils import (dump_json, dump_yaml, get_logger,  # noqa: E402
                                   load_json, load_yaml, locked_state_write,
                                   now_iso)
import json
from runtime.data import load_dataframe, resolve_feature_columns  # noqa: E402
from runtime.harness import protocol_stamp  # noqa: E402
from runtime.safe_dtype import safe_to_numeric  # noqa: E402
import time
from .context import _SKILL_ROOT, _load_config, _load_spec, _resolve_baseline_id, _run_dir_from, _update_run_state, logger


def cmd_init(args) -> int:
    target = Path(args.dir)
    target.mkdir(parents=True, exist_ok=True)
    task_id = args.task_id or "new-task"
    spec_path = target / "task-spec.yaml"
    if spec_path.exists() and not args.force:
        logger.error("已存在 %s;使用 --force 覆盖", spec_path)
        return 1
    spec_text = TASK_TEMPLATE.format(
        task_id=task_id,
        task_name=args.task_name or f"{task_id} 建模任务",
        objective_description=args.objective or "(待填写:业务目标一句话)",
        target_name=args.target or "label",
        positive_definition=args.positive_def or "(待填写:正样本定义)",
        time_column=args.time_column or "dt",
        id_column=args.id_column or "user_id",
    )
    spec_path.write_text(spec_text, encoding="utf-8")
    if not (target / "data-loader.py").exists():
        (target / "data-loader.py").write_text(
            DATA_LOADER_TEMPLATE, encoding="utf-8")
    # 目录骨架(§20)
    for sub in ("sample-profile", "generations", "trials", "leaderboard",
                "experience", "final-model/model", "report/assets", "logs",
                "adhoc", "tools"):
        (target / sub).mkdir(parents=True, exist_ok=True)
    # v2.8.1:init 时自动复制工作区标准工具(lint_trial + backfill_baseline)
    tools_dir = target / "tools"
    for tool_name in ("lint_trial.py", "backfill_baseline.py"):
        src = _SKILL_ROOT / "assets" / tool_name
        if src.exists() and not (tools_dir / tool_name).exists():
            import shutil as _sh
            _sh.copy2(src, tools_dir / tool_name)
            print(f"  tools/{tool_name} 已复制")
    # v2.17.0:新 run 合并新协议口径(在途 run 的配置快照不含这些键,
    # 行为不变;见 runtime/protocol_gates.V2_17_0_RUN_CONFIG)
    from runtime.config import _deep_merge as _merge
    from runtime.protocol_gates import V2_17_0_RUN_CONFIG as _V217
    from runtime.protocol_gates import V2_19_0_RUN_CONFIG as _V219
    from runtime.protocol_gates import V2_19_1_RUN_CONFIG as _V2191
    from runtime.protocol_gates import V2_20_0_RUN_CONFIG as _V220
    dump_yaml(target / "modelevo-config.yaml",
              {"modelevo": _merge(_merge(_merge(_merge(
                  _load_config(target).snapshot(), _V217), _V219), _V2191),
                  _V220)})
    # v2.15.8:skill_version 是"只对新 run 启用"行为门的标记(ROI 停止、
    # 代级失败率熔断、父代源码前置断言);旧 run 无此键,保持原行为
    from runtime import __version__ as _skill_version
    _update_run_state(target, created_at=now_iso(), started_at_epoch=time.time(),
                      status="INITED", skill_version=_skill_version,
                      **protocol_stamp())
    # v19:guard hook 自动安装(铁律 13 的硬边界此前只有文档指引,
    # 实测 run 全程无 .claude/settings.json——hook 从未生效,主会话
    # 的源码拦截/演进循环拦截全部依赖 LLM 自觉)。init 时 merge 进
    # 工作区 settings.json(存在则合并 hooks,不动用户其他配置);
    # 失败不阻塞 init,打印醒目告警。
    try:
        import json as _json

        _guard_src = _SKILL_ROOT / "assets" / "guard_main_session.py"
        _settings_dir = target / ".claude"
        _settings_dir.mkdir(exist_ok=True)
        _settings_path = _settings_dir / "settings.json"
        _settings = (load_json(_settings_path)
                     if _settings_path.exists() else {})
        _hooks = _settings.setdefault("hooks", {}).setdefault(
            "PreToolUse", [])
        # v2.16.0:matcher 加 Write/Edit —— 拓扑批准账本 approvals/ 只能
        # 由用户经 approve-topology 写入,Agent 的文件写工具同样要过 hook
        _matcher = "Read|Grep|Glob|Bash|Write|Edit|MultiEdit|NotebookEdit"
        _hook_entry = {
            "matcher": _matcher,
            "hooks": [{"type": "command",
                       "command": f"python {_guard_src}"}],
        }
        _existing = [e for e in _hooks
                     if any("guard_main_session" in h.get("command", "")
                            for h in e.get("hooks", []))]
        if not _existing:
            _hooks.append(_hook_entry)
            _settings_path.write_text(
                _json.dumps(_settings, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"  .claude/settings.json:guard_main_session hook 已安装"
                  f"(铁律 13 硬边界)")
        elif any(e.get("matcher") != _matcher for e in _existing):
            for e in _existing:
                e["matcher"] = _matcher
            _settings_path.write_text(
                _json.dumps(_settings, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print("  .claude/settings.json:guard_main_session hook matcher "
                  "已升级(覆盖 Write/Edit,v2.16.0 拓扑批准边界)")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "guard hook 自动安装失败(不阻塞 init;请按 SKILL.md "
            "「guard_main_session.py 安装」手工配置): %s", exc)
    print(f"初始化完成: {target}")
    print("下一步: 1) 编辑 task-spec.yaml 与 data-loader.py"
          " 2) modelevo.py validate 3) modelevo.py profile")
    return 0


def cmd_validate(args) -> int:
    spec = _load_spec(args)
    errors = spec.validate()
    if errors:
        print("TaskSpec 校验失败:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"TaskSpec 校验通过: {spec.task_id} ({spec.model_type}, "
          f"primary={spec.primary_metric})")
    return 0


def cmd_profile(args) -> int:
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    # [FIX-2026-10-29] 旧版在这里静默 4 min(4M 行 × 1513 列、24.6 GB 分片
    # 的 loader 读取),stdout 与 sample-profile/logs/profile.log 都还没有任何
    # 输出 —— 实测被主会话误判成卡死并 kill 掉,白跑一次完整 profile。现在
    # 读取前后各打一行并强制 flush,同时把开始时刻写进 run-state 便于排查。
    loader_name0 = (spec.data.get("data_loader") or "data-loader.py")
    print(f"profile: 开始读取数据面(loader={loader_name0}, run={run_dir}) ...",
          flush=True)
    _update_run_state(run_dir, profile_loading_started_at=now_iso())
    df = load_dataframe(spec, run_dir)
    n = len(df)
    print(f"profile: 数据面已载入 {n} 行 × {df.shape[1]} 列,开始 profiling ...",
          flush=True)
    out_dir = run_dir / "sample-profile"
    # 信息压缩检测(v8):定位 data-loader 源码供扫描
    loader_name = (spec.data.get("data_loader") or "data-loader.py")
    loader_path = run_dir / loader_name
    profile = SampleProfile(df, spec, out_dir,
                            loader_path=loader_path if loader_path.exists()
                            else None)
    result = profile.run(max_workers=int(args.workers))
    # ---- v14:产物自消费冒烟 ----
    # profile 刚写完四产物,立刻用每个下游消费方的取数路径读一遍。
    # v13 上线时那六次 KeyError 全部落在本检查覆盖范围内,每次代价是
    # 一个失败的 trial/generation,而这里的代价是几秒钟。
    from runtime.preflight import check_consumers, check_data_products

    issues = (check_data_products(run_dir, spec)
              + check_consumers(run_dir, spec))
    if issues:
        for msg in issues:
            logger.error("产物自消费冒烟失败: %s", msg)
        logger.error("四产物无法被 harness 自身消费,不落 PROFILED;"
                     "修好 write_splits/消费方后重跑 profile")
        return 1
    _update_run_state(run_dir, status="PROFILED", rows=n,
                      feature_count=result["n_features"])
    # v2.15：profiling 完成后推导 abstract_task_id 并写回 task-spec.yaml
    # 格式：{model_type}-{target_name}-{sample_scale}-{col_hash8}
    # col_hash8 来自 df_full.columns（data_loader 读入的原始列集，
    # 已由 profile.run() 写入 sample-profile.json["raw_columns_hash"]）。
    # 这样同一份数据底座的所有 run（不论建模路线）得到相同的 task_id，
    # 全局经验库的 top_trial_baseline 条目能被正确检索和复用。
    try:
        derived = spec.resolve_abstract_task_id(run_dir)
        # v2.15.8:只有可信来源(原生全量面 hash)才写回 task_id ——
        # resolve 已把可信值持久化进 run-state;splits 兜底不持久化,
        # 也不该覆盖 task_id(7580b665 事故:task_id 被写成 splits 指纹)
        _st_now = load_json(run_dir / "run-state.json") \
            if (run_dir / "run-state.json").exists() else {}
        _trusted = _st_now.get("abstract_task_id") == derived
        if "pending" not in derived and _trusted:
            old_id = spec.data.get("task_id", "")
            if derived != old_id:
                spec.data["task_id"] = derived
                spec.save(run_dir / "task-spec.yaml")
                print(f"  task_id 自动推导: {old_id!r} → {derived!r}")
            else:
                print(f"  task_id 已是规范格式: {derived!r}")
        elif "pending" not in derived:
            print(f"  [warn] abstract_task_id {derived!r} 来自 splits 兜底(原生面"
                  " hash 不可得),不写回 task_id")
        # v2.18.0:拿到可信任务指纹后,在本 run 已固定的库水位上补取任务相关的经验
        # (卷宗、方向表、榜单、固化 baseline 与热启动配方)
        if "pending" not in derived and _trusted:
            from runtime import library_view
            # [FIX-2026-10-29] 传 run 级 config:旧版 ensure 不带 config 时
            # client_settings 退回 skill 级配置,run 级 library.* 不生效
            _vcfg = _load_config(run_dir)
            print("  " + library_view.describe_resolution(
                library_view.client_settings(_vcfg, run_dir=run_dir), run_dir))
            _v = library_view.ensure(run_dir, atid=derived, spec=spec,
                                     config=_vcfg)
            print(f"  经验库视图: 水位 {_v.watermark}"
                  f"{'(本机缓存,stale)' if _v.stale else ''},原则 {len(_v.principles)} 条,"
                  f"榜单 {len(_v.leaderboard)} 条")
    except Exception as exc:
        logger.warning("abstract_task_id 推导失败(不阻塞): %s", exc)
    print(f"SampleProfile 完成: {n} 行 × {result['n_features']} 特征")
    leak = result["leakage"]["severity_counts"]
    print(f"  leakage: critical={leak['critical']} warning={leak['warning']}")
    ca = result.get("compression_audit")
    if ca:
        print(f"  压缩检测: risk={ca.get('risk_level')}, "
              f"data_view 候选 {len(ca.get('data_view_candidates') or [])} 条")
    print(f"  产物: {out_dir}")
    return 0


def cmd_baseline(args) -> int:
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    config = _load_config(run_dir)
    sched = Scheduler(config)
    # v2.17.0:配置即事实——删除 v2.15 的内存收缩([lightgbm, xgboost] →
    # [xgboost])。默认值已改为 [xgboost](lightgbm 只多花 20~25 min);
    # 配置显式含 lightgbm 时照跑,算法计划由 BaselineRunner 打印留痕。
    # baseline 串行执行:单 trial 独占线程预算(usable//1,cap 16)
    plan = sched.plan(planned_trials=1)
    br = BaselineRunner(spec, config, run_dir)
    algorithms = br.plan_baseline_trials(plan, force_all=True)

    df = None
    splits_dir = run_dir / "sample-profile" / "splits"
    if not (splits_dir / "train.parquet").exists():
        raise SystemExit("sample-profile 未执行;先运行 profile")

    factory = TrialFactory(run_dir, spec=spec, config=config)
    runner = TrialRunner(run_dir, spec, config)
    gen_dir = GenerationRecord(run_dir, 0)
    results = []
    baseline_id = None
    best_metric, direction_best = None, None
    # v19.2:baseline 幂等续跑。此前每次调用都无条件 create+run 全部
    # 算法,进程被外部 kill(后台命令 60min 上限、timeout、人工中断)
    # 后重跑会再建一整批新 trial——实测一次 400w baseline 连跑 4 轮、
    # 落盘 6 个重复 trial(其中 2 个已 EVALUATED 的 AUC 完全相同),
    # 编排上表现为"一直停在 gen-00"。已完成(SUCCEEDED/EVALUATED)的
    # 算法现在直接复用,只补跑缺失/失败的那个。
    done: dict[str, "TrialRecord"] = {}
    for t in factory.all_trials():
        if t.data.get("generation") != 0:
            continue
        # v2.15.7:warm trial 与 baseline 同算法(xgboost),不能被当成
        # baseline 复用,否则 warm trial 会顶替 trial-001 成为锚点。
        if t.data.get("search_action") == "warm_start" or str(
                (t.data.get("hypothesis") or {}).get("statement", "")
        ).startswith("[warmstart"):
            continue
        if t.status in ("SUCCEEDED", "EVALUATED"):
            done.setdefault(str(t.data.get("algorithm")), t)

    # v2.15: 提前收集 warm trial，与 baseline 一起并行跑。原实现放在
    # baseline 循环结束后串行跑，每个 warm trial 额外耗费一个完整训练周期。
    # 现在把它们加入同一个并发队列，gen-00 总耗时 ≈ max(所有并发项)。
    # v2.15.6: 全局库按配方去重后保留 top-5 作为 archive（rank1..rank5），
    # 但 gen-00 默认只起 rank1/rank2 两个 warm trial（活跃尝试集）。
    # 这样 archive 里保留更多历史最优配方供人工/后续代调用，而 gen-00
    # 的算力开销可控。可用 config baseline.warm_rank_limit 调整上限。
    # v2.15.7:baseline 算法 trial 必须第一个创建(拿到 seq=001)。先创建所有
    # baseline 算法的 trial,再创建 warm trial,确保 trial-001 是 baseline。
    # warm trial 从 002 起。baseline_id 在 baseline 创建后立即锁定。
    # v2.15.7:baseline 配方固化到全局榜单 top-trials/<atid>/baseline/。
    # 首次运行该任务时渲染并固化;之后同一 abstract_task_id 的 run 一律
    # 复用固化的 train.py/trial.yaml,保证多次运行的 baseline 配方相同、
    # vs_baseline 跨 run 可比。固化目录不存在或缺文件时退回现场渲染。
    # v2.18.0:固化 baseline 与热启动配方都来自本 run 固定水位的经验库视图
    # (旧实现读写 memory/top-trials/<atid>/{baseline,rank*})。视图不可用时
    # baseline 现场渲染、不做热启动。
    _view, _atid0 = None, None
    try:
        _atid0 = spec.resolve_abstract_task_id(run_dir)
        if _atid0 and "pending" not in _atid0:
            from runtime import library_view
            # [FIX-2026-10-29] 传 config:否则 run 级 library.* 不生效,禁用
            # 经验库后 baseline 仍会创建 warm-start trial
            _view = library_view.ensure(run_dir, atid=_atid0, spec=spec,
                                        config=config)
            _lcfg = library_view.client_settings(config, run_dir=run_dir)
            if _view.principles or _view.leaderboard or _view.baselines:
                print("  " + library_view.describe_resolution(_lcfg, run_dir))
                if _view.leaderboard:
                    print(f"  经验库榜单 {len(_view.leaderboard)} 条 → "
                          f"gen-00 将创建最多 "
                          f"{int(config.get('baseline.warm_rank_limit', 2) or 2)} 个 "
                          "warm-start trial")
            else:
                print("  " + library_view.describe_resolution(_lcfg, run_dir)
                      + f" reason={_view.data.get('reason')} → 不创建 warm-start trial")
    except Exception as _e:  # noqa: BLE001
        logger.warning("经验库视图不可用(baseline 现场渲染、不做热启动): %s", _e)
    _baseline_trials = []
    for algo in algorithms:
        if algo not in done:
            _t = factory.create(
                algorithm=algo,
                hypothesis={
                    "statement": f"{algo} 默认参数 baseline(统一起跑线)",
                    "rationale": "稳定、便宜、可复现,不做 HPO(§8)",
                },
                execution={
                    "action_description": ["固定默认参数", "全特征输入"],
                    "code_entry": "train.py",
                    "command": "python train.py",
                },
                search_action="explore",
                generation=0,
            )
            _frozen = (_view.baselines.get(algo) if _view is not None else None)
            if _frozen and _apply_library_baseline(_view, _frozen, _t, run_dir):
                _t.data.setdefault("execution", {})["baseline_recipe"] = {
                    "source": "library", "recipe_id": _frozen["recipe_id"],
                    "source_run_id": _frozen.get("source_run_id"),
                    "source_trial_id": _frozen.get("source_trial_id")}
                logger.info("baseline %s 复用经验库固化配方 %s", algo,
                            _frozen["recipe_id"][:12])
            else:
                br.write_baseline_code(_t.trial_dir, algo, [])
                _t.data.setdefault("execution", {})["baseline_recipe"] = {
                    "source": "rendered"}
            _t.flush()
            _baseline_trials.append((algo, _t))
        else:
            _baseline_trials.append((algo, done[algo]))

    # v2.15.7:baseline_id 锁定为第一个 baseline trial(trial-001),不等跑完、
    # 失败也不换(后续 trial 的 delta_vs_baseline 为空,表示基线不可用)。
    # 续跑时 run-state 已有的 baseline_id 优先,保证整个 run 只有一个锚点。
    _st0 = load_json(run_dir / "run-state.json") \
        if (run_dir / "run-state.json").exists() else {}
    baseline_id = _st0.get("baseline_id") or (
        _baseline_trials[0][1].trial_id if _baseline_trials else None)
    if baseline_id and baseline_id != _st0.get("baseline_id"):
        _update_run_state(run_dir, baseline_id=baseline_id)
    if baseline_id:
        logger.info("baseline 锚点已锁定: %s(失败也不更换)", baseline_id)
        if not str(baseline_id).startswith("trial-001-"):
            logger.warning(
                "baseline 锚点 %s 不是 trial-001(run 内先于 baseline 已有 "
                "trial,或沿用旧 run-state)", baseline_id)

    # warm trial 在 baseline 之后创建(拿到 seq >= 002)
    _warm_created: list[tuple[str, "TrialRecord"]] = []
    # v2.18.4:语义指纹去重(见下方 loop 内注释)
    _warm_fp_owner: dict[str, int] = {}
    _warm_skipped_dup: list[dict] = []
    try:
        import shutil as _sh
        _atid = _atid0
        if _view is not None and _atid and "pending" not in _atid:
            _warm_limit = int(config.get("baseline.warm_rank_limit", 2) or 2)
            _existing_hyps = {
                str(t.data.get("hypothesis", {}).get("statement", ""))[:60]
                for t in factory.all_trials()
                if t.data.get("generation") == 0
            }
            for _rank, _lb in enumerate(_view.leaderboard, 1):
                # v2.18.4:名额按"语义去重后的配方数"计 —— 被判重复跳过的
                # rank 不占名额,顺延到下一名(其它跳过原因照旧占名额)
                if _rank - len(_warm_skipped_dup) > _warm_limit:
                    break
                # v2.18.0:配方从本机缓存渲染(RUN_DIR / SKILL_ROOT 填回本 run 与本
                # skill),不再复制带源 run 绝对路径的文件
                _rid = _lb["recipe_id"]
                _rdir = run_dir / "experience" / "warm-recipes" / _rid[:12]
                _merrs = _view.materialize(_rid, _rdir, run_dir=run_dir,
                                           skill_root=_SKILL_ROOT)
                _src_code = _rdir / "train.py"
                _src_cfg  = _rdir / "trial.yaml"
                if _merrs or not _src_code.exists() or not _src_cfg.exists():
                    logger.warning("热启动配方 rank-%d(%s)不可用,跳过: %s", _rank,
                                   _rid[:12], "; ".join(_merrs) or "缺 train.py / trial.yaml")
                    continue
                # v2.18.4:语义级去重。榜单上不同 recipe_id 可能是**训练行为
                # 完全相同**的配方(实测 rank-1/rank-2 逐字节差 435 行,而
                # 特征列/模型工件的 sha256 全都相同、val AUC 逐位相同),
                # 各烧 69 min × 2 卡却产出 0 比特新信息。主修复在经验库榜单
                # 源头(library.recipes.recipe_family 按同一语义规范形归组);
                # 这里是兜底 —— 旧版经验库服务仍可能返回重复。
                try:
                    # v2.19.0:指纹 = 代码语义 + 训练 config 语义 + 附带模块
                    # (scripts/_mvo/warmstart.warm_recipe_fingerprint)
                    from .warmstart import warm_recipe_fingerprint

                    _fp = warm_recipe_fingerprint(_rdir)
                    if _fp != "unparsable" and _fp in _warm_fp_owner:
                        logger.warning(
                            "热启动配方 rank-%d(%s)与本 run 已建的 rank-%s 训练"
                            "语义相同(指纹 %s),跳过以免重复计费",
                            _rank, _rid[:12], _warm_fp_owner[_fp], _fp[:12])
                        print(f"[warm-start] 跳过 rank-{_rank}({_rid[:12]}):与 "
                              f"rank-{_warm_fp_owner[_fp]} 代码+config 语义指纹相同"
                              f"({_fp[:12]}),不重复创建 trial", flush=True)
                        _warm_skipped_dup.append(
                            {"rank": _rank, "recipe_id": _rid,
                             "same_as_rank": _warm_fp_owner[_fp],
                             "fingerprint": _fp})
                        continue
                    if _fp != "unparsable":
                        _warm_fp_owner[_fp] = _rank
                except Exception as _fp_exc:  # noqa: BLE001 — 指纹失败不阻塞
                    logger.debug("热启动语义指纹失败(忽略): %s", _fp_exc)
                _src_td = load_yaml(_src_cfg) or {}
                if _src_td.get("data_view") not in (None, "", "v1"):
                    # 本 run 尚未构建该数据视图(构建脚本已随配方入库:data_view/)
                    logger.warning("热启动配方 rank-%d(%s)依赖数据视图 %s,本 run 未构建,跳过",
                                   _rank, _rid[:12], _src_td.get("data_view"))
                    continue
                _hyp_md = _rdir / "hypothesis.md"
                _src_hyp = (_hyp_md.read_text(encoding="utf-8").strip().splitlines() or [""])[0] \
                    if _hyp_md.exists() else str(_lb.get("source_trial_id") or "")
                _hyp = (
                    f"[warmstart rank-{_rank}] {_atid}: "
                    + _src_hyp.lstrip("# ")[:120]
                )
                if _hyp[:60] in _existing_hyps:
                    logger.info("warm trial rank-%d 已存在,跳过", _rank)
                    # 幂等：已完成的 warm trial 进 done
                    for _t in factory.all_trials():
                        if str(_t.data.get("hypothesis", {}).get(
                                "statement", ""))[:60] == _hyp[:60] \
                                and _t.status in ("SUCCEEDED", "EVALUATED"):
                            done[f"warmstart-rank{_rank}"] = _t
                    continue
                _wt = factory.create(
                    algorithm=str(_src_td.get("algorithm") or "xgboost"),
                    hypothesis={
                        "statement": _hyp,
                        "rationale": (
                            f"直接复用经验库 {_atid} 榜单 rank-{_rank} 的历史最优配方"
                            f"({_rid[:12]}),跳过重新发现已知增益。"
                        ),
                    },
                    execution={
                        "action_description": [
                            f"渲染经验库榜单 rank-{_rank} 配方 {_rid[:12]} 的 train.py "
                            "和 trial.yaml"],
                        "code_entry": "train.py",
                        "command": "python train.py",
                    },
                    search_action="warm_start",  # v2.15.6: warm_start 语义区分于 explore
                    generation=0,
                )
                # v2.15.8:记来源并自动绑定经验。来源按 source_trial_id 记
                # (榜单每次导出都重排,rankN 的 id 会指向别的配方);
                # finalize-generation 据 experience_bindings 把本 trial 的
                # vs_baseline 结果回填进该配方条目的后验。
                _src_tid = _lb.get("source_trial_id") or \
                    (str(_src_td.get("trial_id") or "") or None)
                _meas = _lb.get("measurement") or {}
                _wt.data.setdefault("execution", {})["warm_start"] = {
                    "rank": _rank, "abstract_task_id": _atid,
                    "source_trial_id": _src_tid,
                    "source_run_id": _lb.get("source_run_id"),
                    "recipe_id": _rid, "library_watermark": _view.watermark,
                    "reference_auc": _meas.get("value"),
                    "reference_gain": _meas.get("gain"),
                }
                # 热启动 trial 收尾时作为该配方的一次复测进快照(warm_remeasure)
                _wt.data["experience_bindings"] = [{
                    "recipe_id": _rid, "source_trial_id": _src_tid}]
                _wt.flush()
                _sh.copy2(_src_code, _wt.trial_dir / "code" / "train.py")
                _copy_recipe_code_modules(_rdir, _wt.trial_dir)
                # v2.15.2: 修正 RUN_DIR — memory/top-trials 里保存的 train.py
                # 的 RUN_DIR 指向来源 run（相对式或绝对式），直接复制会让
                # warmstart trial 读外部 run 的 splits（行数不对齐），评估时
                # 必然失败。改写 + **结果校验**由 scripts/_mvo/warmstart.py
                # 单源实现(放在 scripts 侧而非 runtime/,避免 harness_hash
                # 变动——本修复不碰任何评分/落盘口径)。
                #
                # v2.15.0 的事故：旧实现用
                #   re.subn(r'^RUN_DIR\s*=\s*pathlib\.Path\([^\)]*\)', ...)
                # 的 `[^\)]*` 在 `pathlib.Path(__file__` 处遇首个右括号即停，
                # 之后的 `.resolve().parents[3]` **残留在行尾**，把新 run 路径
                # 又往上跳 3 层 → SPLITS 指向 <share>/sample-profile/splits、
                # 两个 warm trial 同因 infra_error 失败；且只要 subn 命中就打
                # "已改写"info 日志，错误完全不可见。改为整行替换 + 硬校验：
                # 校验不过就**放弃该 warm trial**（落 FAILED + 专用失败原因），
                # 绝不带着错路径进训练。
                try:
                    from .warmstart import rewrite_recipe_run_dir
                    _dst_code = _wt.trial_dir / "code" / "train.py"
                    _new_text, _rw_status = rewrite_recipe_run_dir(
                        _dst_code.read_text(encoding="utf-8"), run_dir)
                    if _rw_status == "verified":
                        _dst_code.write_text(_new_text, encoding="utf-8")
                        logger.info("warm trial rank-%d: RUN_DIR 已改写并校验通过", _rank)
                        # v2.16.0:历史配方的早停口径升级到 outer validation
                        # 单段拟合(全局库口径:任何口径都只训一次)
                        from .render import _apply_es_source_upgrade
                        _es_notes = _apply_es_source_upgrade(_dst_code,
                                                              harden=True)
                        if _es_notes:
                            logger.info("warm trial rank-%d: %s", _rank,
                                        ";".join(_es_notes))
                    else:
                        _reason = "warm_start_recipe_incompatible"
                        logger.error(
                            "warm trial rank-%d: RUN_DIR 改写未通过校验"
                            "(status=%s)，放弃该 warm trial(绝不带错路径进训练)",
                            _rank, _rw_status)
                        try:
                            _wt.data["failure_reason"] = _reason
                            _wt.transition("FAILED")
                            _wt.flush()
                        except Exception:  # noqa: BLE001 — 留痕失败不阻塞 baseline
                            pass
                        continue
                except Exception as _rw_exc:
                    logger.error("warm trial rank-%d: RUN_DIR 改写异常，"
                                 "放弃该 warm trial: %s", _rank, _rw_exc)
                    try:
                        _wt.data["failure_reason"] = "warm_start_recipe_incompatible"
                        _wt.transition("FAILED")
                        _wt.flush()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                _sh.copy2(_src_cfg,  _wt.trial_dir / "config" / "trial.yaml")
                # gen-00 warm trial: multi_seed > 1 收紧为 1。
                # 历史 rank trial 可能带 multi_seed=3（确认门需要），但 gen-00
                # 的目标只是拿到指标读数，多 seed 开销不必要。
                _cfg_path = _wt.trial_dir / "config" / "trial.yaml"
                try:
                    _cfg_data = load_yaml(_cfg_path) or {}
                    _cfg_changed = False
                    if int(_cfg_data.get("multi_seed", 1) or 1) > 1:
                        _cfg_data["multi_seed"] = 1
                        _cfg_changed = True
                    # v2.15.1: 来源 trial.yaml 的 search_action 是来源 run 里的角色
                    # (通常是 confirm)，在本 run 里这是无父代的根 trial，必须改为 explore。
                    if _cfg_data.get("search_action") != "explore":
                        _cfg_data["search_action"] = "explore"
                        _cfg_changed = True
                    # parent_trial_ids 对 explore 根 trial 应为空
                    _cfg_data["parent_trial_ids"] = []
                    _cfg_changed = True
                    if _cfg_changed:
                        dump_yaml(_cfg_path, _cfg_data)
                except Exception:
                    pass
                _wt.flush()
                _warm_created.append((f"warmstart-rank{_rank}", _wt))
    except Exception as _wexc:  # noqa: BLE001
        logger.warning("warm trial 创建失败(不阻塞 baseline): %s", _wexc)

    # v2.15.7:两阶段执行。旧实现把 baseline 与 warm trial 放进同一并发池,
    # 而 warm 的 baseline_score 在提交前就要读出 —— 那时 trial-001 尚未
    # 产出分数,warm trial 的 vs_baseline 恒空。现在:
    #   阶段 1:baseline 算法 trial(trial-001 起)先跑完,得到锚点分数;
    #   阶段 2:warm trial 带 trial-001 的分数并行执行。
    def _anchor_scores():
        if not baseline_id:
            return None, None
        _bl = load_parent_scores(run_dir, baseline_id)
        return ((_bl or {}).get("validation"), (_bl or {}).get("data_view"))

    # ---- 阶段 1:baseline 算法 ----
    for algo_i, (algo, trial) in enumerate(_baseline_trials):
        if trial.trial_id == baseline_id:
            # trial-001 自身即锚点,不与自己比较
            _bs, _bv = None, None
        else:
            _bs, _bv = _anchor_scores()
        if algo in done:
            logger.info("baseline 复用已完成 trial:%s(%s)", trial.trial_id, algo)
            print(f"[reuse] {algo}: {trial.trial_id} status={trial.status}",
                  flush=True)
            _ev = (trial.data.get("evaluation") or {})
            if trial.status == "EVALUATED" and _bs is not None \
                    and not (_ev.get("delta") or {}).get("vs_baseline"):
                runner.run(trial, baseline_score=_bs, baseline_view=_bv)
        else:
            runner.run(trial, plan=plan, slot_index=algo_i,
                       baseline_score=_bs, baseline_view=_bv)
        results.append(trial)
    _bs, _bv = _anchor_scores()
    if baseline_id and _bs is None:
        logger.warning(
            "baseline 锚点 %s 无 validation 分数(失败或未完成):本 run 全部 "
            "trial 的 delta_vs_baseline 将为空。锚点不更换(v2.15.7)。",
            baseline_id)
    elif _bs is not None:
        logger.info("baseline 锚点分数就绪: %s", baseline_id)
    # v2.18.0:首次运行该任务时,现场渲染的 baseline 训练评估完成后固化进经验库
    # (配方 + 测量 + pin_baseline),同 atid 的后续 run 复用,Δ_baseline 跨 run 可比
    if _atid0 and "pending" not in _atid0:
        _freeze_rendered_baselines(run_dir, _baseline_trials, _atid0, config)

    # ---- 阶段 2:warm trial(带 trial-001 分数) ----
    _warm_todo = [(label, t) for label, t in _warm_created
                  if t.status not in ("SUCCEEDED", "EVALUATED", "FAILED")]
    if _warm_todo:
        plan_par = sched.plan(planned_trials=max(len(_warm_todo), 1))
        _par = int(getattr(plan_par, "n_parallel_trials", 1) or 1)
        if bool(config.get("baseline.parallel", True)) \
                and _par >= 2 and len(_warm_todo) >= 2:
            import concurrent.futures as _cf
            logger.info("warm trial 并行执行 %d 个(%s),threads/trial=%s",
                        len(_warm_todo), [l for l, _ in _warm_todo],
                        getattr(plan_par, "threads_per_trial", "?"))
            with _cf.ThreadPoolExecutor(
                    max_workers=min(len(_warm_todo), _par)) as _ex:
                _futs = {
                    _ex.submit(runner.run, _t, plan=plan_par, slot_index=_i,
                               baseline_score=_bs, baseline_view=_bv):
                        label
                    for _i, (label, _t) in enumerate(_warm_todo)}
                for _f in _cf.as_completed(_futs):
                    try:
                        _f.result()
                    except Exception as _exc:  # noqa: BLE001
                        logger.error("warm %s 执行异常:%s", _futs[_f], _exc)
        else:
            for _i, (label, _t) in enumerate(_warm_todo):
                try:
                    runner.run(_t, plan=plan, slot_index=_i,
                               baseline_score=_bs, baseline_view=_bv)
                except Exception as _exc:  # noqa: BLE001
                    logger.error("warm %s 执行异常:%s", label, _exc)
    if _warm_skipped_dup:
        logger.warning("warm-start 语义去重跳过 %d 个重复配方:%s",
                       len(_warm_skipped_dup),
                       "; ".join(f"rank-{d['rank']}(={d['same_as_rank']})"
                                 for d in _warm_skipped_dup))
    # 已完成的 warm trial 若当时缺锚点分数,补评估
    _by_id = {t.trial_id: t for t in factory.all_trials()}
    for label, _wt in _warm_created:
        _wt2 = _by_id.get(_wt.trial_id, _wt)
        _ev = (_wt2.data.get("evaluation") or {})
        if _wt2.status == "EVALUATED" and _bs is not None \
                and not (_ev.get("delta") or {}).get("vs_baseline"):
            runner.run(_wt2, baseline_score=_bs, baseline_view=_bv)
            _wt2 = TrialRecord.load(Path(_wt2.trial_dir))
        results.append(_wt2)
    # 续跑:已完成的 warm trial 走"已存在,跳过"分支进了 done,也要计入结果
    _seen = {t.trial_id for t in results}
    for _k, _t in sorted(done.items()):
        if _k.startswith("warmstart-") and _t.trial_id not in _seen:
            _ev = (_t.data.get("evaluation") or {})
            if _t.status == "EVALUATED" and _bs is not None \
                    and not (_ev.get("delta") or {}).get("vs_baseline"):
                runner.run(_t, baseline_score=_bs, baseline_view=_bv)
            results.append(_t)
            _seen.add(_t.trial_id)

    gen_dir.write_plan({"trials": [t.trial_id for t in results],
                        "kind": "baseline"})
    # v10.3:baseline shadow 补打分——模板只产 validation 分数,shadow
    # sanity(evaluator.shadow_test_enabled)依赖 shadow_test_score.parquet。
    # [WORKAROUND-2026-08-31] runner.evaluate_only 会把 run-state phase
    # 从 BASELINED 覆盖为 EVOLVING(EVALUATED 状态映射的既有行为),导致
    # baseline 完成后 phase 停在 EVOLVING。shadow 补打分在 cmd_run_trial /
    # run-generation 阶段会由 evaluator 正常补上(ephemeral score),故此处
    # 跳过后置重评,只做 leaderboard + BASELINED 收尾。
    if config.get("evaluator.shadow_test_enabled", True) \
            and (splits_dir / "shadow_test.parquet").exists():
        for t in results:
            _backfill_shadow_scores(run_dir, spec, t)
    # leaderboard + 设定 baseline
    lb = Leaderboard(run_dir, spec)
    lb.build(factory.all_trials(), baseline_id=baseline_id)
    # [FIX-2026-08-31] 收尾显式把 phase 钉在 BASELINED,不回退(幂等)
    _update_run_state(run_dir, status="BASELINED", baseline_id=baseline_id)
    _update_run_state(run_dir, phase="BASELINED")
    print("baseline 完成:")
    for t in results:
        pm = ((t.data.get("evaluation") or {}).get("primary_metric") or {})
        print(f"  {t.trial_id}: status={t.status} "
              f"{spec.primary_metric}={pm.get('value')}")
    return 0


def _apply_library_baseline(view, frozen: dict, trial, run_dir: Path) -> bool:
    """复用经验库固化的 baseline 配方:从本机缓存渲染到本 trial。

    渲染失败(配方不在缓存、.py 校验不过)返回 False,调用方退回现场渲染;
    渲染成功后沿用 :func:`_apply_frozen_baseline` 的 RUN_DIR 硬校验与早停口径升级。
    """
    rid = frozen["recipe_id"]
    tmp = Path(run_dir) / "experience" / "baseline-recipes" / rid[:12]
    errors = view.materialize(rid, tmp, run_dir=run_dir, skill_root=_SKILL_ROOT)
    if errors or not (tmp / "train.py").exists() or not (tmp / "trial.yaml").exists():
        logger.warning("固化 baseline 配方 %s 不可用,退回现场渲染: %s", rid[:12],
                       "; ".join(errors) or "缺 train.py / trial.yaml")
        return False
    _apply_frozen_baseline(tmp, trial, run_dir)
    return True


def _freeze_rendered_baselines(run_dir: Path, baseline_trials, atid: str, config) -> None:
    """现场渲染且评估成功的 baseline trial 固化进经验库;失败只告警。"""
    from runtime import library_view
    from .library_snapshot import freeze_baseline

    settings = library_view.client_settings(config)
    for algo, t in baseline_trials:
        src = ((t.data.get("execution") or {}).get("baseline_recipe") or {}).get("source")
        if src != "rendered" or t.status != "EVALUATED":
            continue
        try:
            res = freeze_baseline(run_dir, t, atid=atid, settings=settings)
            if res.get("status") == "frozen":
                logger.info("baseline %s 配方已固化进经验库: %s", algo, res["recipe_id"][:12])
        except Exception as exc:  # noqa: BLE001  固化失败不阻塞 baseline
            logger.warning("baseline %s 固化进经验库失败(不阻塞,下次 baseline 再试): %s",
                           algo, exc)


def _copy_recipe_code_modules(src: Path, trial_dir: Path) -> list[str]:
    """配方 bundle 的 ``code/`` 下除 train.py 外的模块(如 deep_seq.py)拷进新 trial 的
    code/;train.py 由调用方单独改写 RUN_DIR 后写入。"""
    import shutil as _shutil
    out = []
    code_src = Path(src) / "code"
    if not code_src.is_dir():
        return out
    for f in sorted(code_src.rglob("*")):
        if f.is_file() and f.name != "train.py":
            dst = Path(trial_dir) / "code" / f.relative_to(code_src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            _shutil.copy2(f, dst)
            out.append(str(f.relative_to(code_src)))
    return out


def _apply_frozen_baseline(src: Path, trial, run_dir: Path) -> None:
    """复用固化的 baseline 配方:拷贝 train.py/trial.yaml 到新 trial。

    train.py 的 ``RUN_DIR`` 行改写为本 run 路径并做解析级校验(与
    warm-start 同一实现);校验不过视为固化配方损坏,抛错拒绝开跑,
    不能带着错误路径训练。trial.yaml 的 trial_id 改为本 trial。
    """
    from .warmstart import rewrite_recipe_run_dir
    src = Path(src)
    code = (src / "train.py").read_text(encoding="utf-8")
    new_code, status = rewrite_recipe_run_dir(code, run_dir)
    if status != "verified":
        raise RuntimeError(
            f"固化 baseline 配方 RUN_DIR 改写未通过校验(status={status}):"
            f"{src}/train.py。删除该固化目录可回退到现场渲染")
    (Path(trial.trial_dir) / "code").mkdir(parents=True, exist_ok=True)
    (Path(trial.trial_dir) / "code" / "train.py").write_text(
        new_code, encoding="utf-8")
    _copy_recipe_code_modules(src, Path(trial.trial_dir))
    # v2.16.0:固化于 v2.16 前的 baseline 配方按 train 尾部早停,在本 run 的
    # 副本上升级为 outer validation 早停(与 warm-start 同为全局库单段口径)
    from .render import _apply_es_source_upgrade
    _apply_es_source_upgrade(Path(trial.trial_dir) / "code" / "train.py",
                             harden=True)
    cfg = load_yaml(src / "trial.yaml") or {}
    cfg["trial_id"] = trial.trial_id
    (Path(trial.trial_dir) / "config").mkdir(parents=True, exist_ok=True)
    dump_yaml(Path(trial.trial_dir) / "config" / "trial.yaml", cfg)


def _backfill_shadow_scores(run_dir: Path, spec: TaskSpec, trial) -> None:
    """用 trial 落盘的 model/model.joblib 对 shadow_test 打分。

    仅当模型工件与 shadow split 都存在且 trial 尚未产出 shadow 分数时
    执行;失败只告警不阻塞 baseline(模型不可加载/特征不齐等留给
    特征探索阶段的 Agent 深挖)。

    v12 修复:特征列经 one-hot 后(cat_a/cat_b/…)在原始 shadow 帧上
    不存在——先对 shadow 做与训练同口径的 dtype 治理(get_dummies +
    reindex 对齐 feature_columns),补打分才能在含类别特征的 run 上
    生效(旧行为:特征列不齐直接跳过,shadow sanity 恒空)。
    """
    tdir = Path(trial.trial_dir)
    model_path = tdir / "model" / "model.joblib"
    shadow_path = tdir / "shadow_test_score.parquet"
    splits_dir = Path(run_dir) / "sample-profile" / "splits"
    if not model_path.exists() or shadow_path.exists():
        return
    try:
        import joblib
        import pandas as pd

        model = joblib.load(model_path)
        cols = json.loads(
            (tdir / "model" / "feature_columns.json").read_text())
        shadow = pd.read_parquet(splits_dir / "shadow_test.parquet")
        raw_cols = list(shadow.columns)  # v13:特征面已无目标列
        X = shadow[raw_cols].copy()
        # 与训练同口径的 dtype 治理:可数值化 object → float32,
        # 低基数类别 → one-hot。
        # v2.12.2.8:safe_to_numeric 替下 pd.to_numeric(长非数字列段错误;
        # 十六进制键列被部分解析后误判为可数值化)。
        cat_cols = []
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                continue
            conv = safe_to_numeric(X[c])
            if conv is not None:
                X[c] = conv.astype("float32")
            elif X[c].nunique() <= 100:
                cat_cols.append(c)
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            logger.warning("shadow 补打分 %s:特征列不齐(缺 %d 列如 %s),"
                           "跳过", trial.trial_id, len(missing), missing[:3])
            return
        X = X[cols].fillna(0).astype("float32")
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)[:, 1]
        else:
            score = model.predict(X)
        pd.DataFrame({"score": score}).to_parquet(shadow_path, index=False)
        logger.info("shadow 补打分完成: %s (%d 行)",
                    trial.trial_id, len(score))
    except Exception as exc:  # noqa: BLE001
        logger.warning("shadow 补打分失败 %s: %s(不阻塞 baseline)",
                       trial.trial_id, exc)
