"""v2.12.2.2 修复批次单元验证(2026-09-17,由 drawdown/t7 run 的实测事故驱动)。

四条修复,每条都对应一次真实 run 的可观测损失:

A. HPO 嫁接依赖闭包只查 body 前置段 → `_stratified_head_labels`
   定义在 body 后方(run_smoke 段),嫁接后父代码里不存在 →
   fit_and_score 的 `except Exception` 把它吞成一句 [warn] 后
   **静默退回全量 inner 搜索**(用户以为限制了抽样,实际烧全量;
   事后从 metrics 也查不出)。修复:依赖查找放宽到整个 body +
   依赖名单加入该 helper + 抽样失败写进 metrics。
   (实测:drawdown/t7 gen-03 trial-016,inner 1433s,
   --hpo-sample-rows 120000 完全未生效)

B. 快通道 `--from-parent <hpo_trial>` 剥离父代 hpo 节时**丢失
   best_params** → 子代跑默认超参。`--action confirm --from-parent
   <hpo_trial>` 于是静默确认了一个与父代配方**不同**的模型
   (审计上确认了冠军,实际确认的是另一个模型)。修复:剥离前从
   父代 metrics.json.best_params 捞回 hyperparameters。
   (实测:drawdown/t7 gen-03,confirm 只能靠手写 child trial 绕过)

C. capacity 闸门单向锁:capacity-check 只在 baseline 后跑一次、
   报告永不重算,而 add_rows/new_information 在单数据视图任务是
   恒定结论 → model/HPO 被永久拒绝,直到人工 --override-capacity。
   Planner 每代提名同一对 HPO 候选、每代被同一原因拒;停止报告
   又把"被闸门拒"误报成"无未执行高潜力方向"。修复:特征侧收敛
   (信息清单清账 / 连续两代无新 accepted)时,每代确定性释放
   **一个**模型方向名额,理由落盘；报告显式列出被拒方向。
   (实测:drawdown/t7 gen-01/02/03 三代同一拒绝;HPO 是唯一被
   官方证实有效的剩余杠杆,却只在用户显式要求后才执行,run
   一度无 confirmed 冠军、无法交付)

D. 报告 §15 把被闸门拒绝的方向漏出"未执行的高潜力方向"统计。

运行:python tests/test_v212dot2dot2_fixes.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

_sys_path = Path(__file__).resolve().parent
sys.path.insert(0, str(_sys_path))
from _common import SKILL as REPO  # noqa: E402

sys.path.insert(0, str(REPO.parent))

ok = fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        fail += 1
        print(f"  [FAIL] {name}")


def _mk_trial(fac, algorithm="lightgbm", generation=0):
    """建一个最小可用的 TrialRecord(够 _copy_parent_trial_code 用)。"""
    return fac.create(
        algorithm=algorithm,
        hypothesis={"statement": "测试用最小假设陈述",
                    "rationale": "r", "falsification": "f"},
        execution={"action_description": ["a"], "code_entry": "train.py",
                   "command": "python train.py", "changed_modules": []},
        search_action="explore", generation=generation)


def main() -> int:
    # -----------------------------------------------------------------
    print("\n== A: hpo 嫁接依赖闭包(全 body 查找 + 依赖名单) ==")
    render_src = (REPO / "scripts" / "_mvo" / "render.py").read_text(
        encoding="utf-8")
    check("依赖名单含 _stratified_head_labels",
          '"_stratified_head_labels"' in render_src)
    check("依赖查找覆写 _src(前置段优先、全 body 兜底)",
          "_src = prefix" not in render_src and "_src = _prefix" in render_src
          and "_src = body_src" in render_src)
    check("missing_deps 报错文案已更新",
          "不是仅前置段" in render_src)

    # 行为验证:复现抽取 + 依赖闭包,断言 helper 真被搬进来
    from _mvo.templates import TRIAL_BODY  # noqa: E402
    import re

    body = TRIAL_BODY
    i0 = body.find("\ndef dump_json(")
    i1 = body.find("\ndef step(", i0)
    fs0 = body.find("\ndef fit_and_score(", i1)
    fs1 = body.find("\n\n\ndef dump_eval_frame(", fs0)
    check("模板锚点仍可定位", i0 > 0 and i1 > i0 and fs0 > 0 and fs1 > fs0)
    hpo_section, fs_section = body[i0:i1], body[fs0:fs1]
    check("fs_section 引用 _stratified_head_labels",
          "_stratified_head_labels" in fs_section)
    _prefix = body[:i0]
    check("该 helper 不在前置段(故旧实现必然漏搬)",
          f"\ndef _stratified_head_labels(" not in _prefix)
    _called = set(re.findall(r"\b(_?[A-Za-z]\w*)\s*\(", hpo_section)) | \
        set(re.findall(r"\b(_?[A-Za-z]\w*)\s*\(", fs_section))
    fake_code = "import pandas as pd\n\ndef main():\n    pass\n"
    deps, missing = "", []
    for dep in ("_lgbm_fit_safe", "_gpu_allocated", "n_jobs",
                "_stratified_head_labels"):
        if dep not in _called or dep in fake_code:
            continue
        d = _prefix.find(f"\ndef {dep}(")
        src = _prefix
        if d < 0:
            d = body.find(f"\ndef {dep}(")
            src = body
        if d < 0:
            missing.append(dep)
            continue
        dend = src.find("\ndef ", d + 1)
        dend = len(src) if dend < 0 else dend
        deps += src[d:dend]
    check("嫁接后 missing 为空", missing == [])
    check("四个依赖全部被搬运",
          all(f"\ndef {d}(" in deps for d in
              ("_lgbm_fit_safe", "_gpu_allocated", "n_jobs",
               "_stratified_head_labels")))
    # 搬进来的 helper 必须可执行(不是注释/残片)
    _ns: dict = {}
    import pandas as _pd
    import numpy as _np
    exec(compile(deps, "<graft-deps>", "exec"),
         {"pd": _pd, "np": _np}, _ns)
    check("搬运的 _stratified_head_labels 可执行且行为正确",
          callable(_ns.get("_stratified_head_labels")))
    if callable(_ns.get("_stratified_head_labels")):
        y = _pd.Series([1] * 800 + [0] * 200)
        sel = _ns["_stratified_head_labels"](y, 100)
        _rate = float(y.loc[sel].mean())
        check(f"分层抽样正例率≈0.8(实测 {_rate:.3f})",
              abs(_rate - 0.8) < 0.06)
        check("分层抽样不超上限", len(sel) <= 100)

    # 抽样失败必须留痕(不再静默)
    tpl = (REPO / "scripts" / "_mvo" / "templates.py").read_text(
        encoding="utf-8")
    check("hpo 抽样失败写入 metrics.hpo_sample_fallback",
          'metrics_extra["hpo_sample_fallback"]' in tpl)
    check("hpo 抽样落 inner 行数",
          'metrics_extra["hpo_inner_rows"]' in tpl)

    # -----------------------------------------------------------------
    print("\n== B: 快通道继承 HPO 父代 best_params ==")
    check("剥离 hpo 节前读取父代 best_params",
          "best_params" in render_src)
    check("继承结果写回 config.hyperparameters",
          "_merged" in render_src and
          'cfg["hyperparameters"] = _merged' in render_src)
    check("best_params 缺失时有明确告警(confirm 无效)",
          "该确认无效" in render_src)

    # 行为验证:构造父 trial(metrics.json 带 best_params)+ 子 trial,
    # 直接调用 _copy_parent_trial_code,断言超参被继承且 overrides 优先。
    import tempfile
    from runtime.logging_utils import dump_yaml
    from runtime.trial import TrialFactory
    from runtime.spec import TaskSpec
    from runtime.config import Config
    from _mvo.render import _copy_parent_trial_code

    spec = TaskSpec(data={
        "task_id": "t", "task_name": "t",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {}, "resource_budget": {"max_trials": 10},
    })
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        cfg = Config(run_dir)
        fac = TrialFactory(run_dir, spec=spec, config=cfg)
        parent = _mk_trial(fac, generation=1)
        (parent.trial_dir / "code").mkdir(parents=True, exist_ok=True)
        (parent.trial_dir / "code" / "train.py").write_text(
            "SEED = 1\n", encoding="utf-8")
        (parent.trial_dir / "config").mkdir(parents=True, exist_ok=True)
        dump_yaml(parent.trial_dir / "config" / "trial.yaml",
                  {"hpo": {"search_space": {"a": 1}},
                   "hyperparameters": {},
                   "features": {"columns": ["x"]}})
        best = {"learning_rate": 0.0267, "num_leaves": 100,
                "reg_lambda": 8.047}
        (parent.trial_dir / "metrics.json").write_text(
            json.dumps({"best_params": best}), encoding="utf-8")
        child = _mk_trial(fac, generation=2)
        _copy_parent_trial_code(run_dir, child, parent.trial_id, {},
                                action="confirm")
        out = (Path(child.trial_dir) / "config" / "trial.yaml").read_text(
            encoding="utf-8")
        out_y = json.loads(json.dumps(
            __import__("yaml").safe_load(out) or {}))
        check("子代 hyperparameters 继承父代 best_params",
              out_y.get("hyperparameters") == best)
        check("子代 config 不含 hpo 节",
              not out_y.get("hpo"))
        check("继承动作写入 notes(best_params 字样)",
              "best_params" in str(out_y.get("notes") or ""))
        # overrides 优先级:显式 --set 覆盖继承值
        child2 = _mk_trial(fac, generation=2)
        _copy_parent_trial_code(run_dir, child2, parent.trial_id,
                                {"num_leaves": 7}, action="exploit")
        y2 = __import__("yaml").safe_load(
            (Path(child2.trial_dir) / "config" / "trial.yaml").read_text(
                encoding="utf-8")) or {}
        check("显式 --set 覆盖继承的冠军参数",
              y2.get("hyperparameters", {}).get("num_leaves") == 7)
        check("未被覆写的继承参数保留",
              y2.get("hyperparameters", {}).get("learning_rate")
              == best["learning_rate"])
        # 父代无 best_params 时不静默
        (parent.trial_dir / "metrics.json").write_text(
            json.dumps({"hpo_n_configs": 3}), encoding="utf-8")
        child3 = _mk_trial(fac, generation=2)
        _copy_parent_trial_code(run_dir, child3, parent.trial_id, {},
                                action="confirm")
        y3 = __import__("yaml").safe_load(
            (Path(child3.trial_dir) / "config" / "trial.yaml").read_text(
                encoding="utf-8")) or {}
        check("父代 best_params 缺失时 notes 标注(confirm 无效)",
              "best_params 缺失" in str(y3.get("notes") or ""))

    # -----------------------------------------------------------------
    # v2.16.0:收敛释放通道从 capacity 闸门迁到特征期门
    # (runtime/feature_phase.py),不再依赖 capacity-report.json。
    print("\n== C: 特征期门收敛释放 ==")
    plan_src = (REPO / "runtime" / "planner.py").read_text(encoding="utf-8")
    check("存在收敛判定 _exploration_converged",
          "def _exploration_converged" in plan_src)
    check("每代至多释放一个名额(_release_id = min)",
          "_release_id = min(_modelish)" in plan_src)
    check("释放理由落盘 batch.feature_phase_gate",
          "FEATURE_PHASE_KEY" in plan_src and "release_reason" in plan_src)
    check("被门拒绝的候选 id 落盘(不再静默)",
          "blocked_proposal_ids" in plan_src)
    check("_validate 接受 release_id 参数",
          "release_id: str | None = None" in plan_src)

    # 行为:已收敛的信息清单(无 capacity 报告),断言恰好放行一个。
    # min_generations=3:两代已评估后仍在特征期内。
    from runtime.planner import AutonomousPlanner  # noqa: E402
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        cfg = Config({"feature_exploration": {"min_generations": 3}})
        fac = TrialFactory(run_dir, spec=spec, config=cfg)
        base = _mk_trial(fac, generation=0)
        # 两代已评估 trial、均非 accepted → 满足 (c)
        for g in (1, 2):
            t = _mk_trial(fac, generation=g)
            t.data["evaluation"] = {"verdict": "rejected",
                                    "trial_id": t.trial_id}
            t.flush()
        pl = AutonomousPlanner(run_dir, spec, cfg)
        converged, why = pl._exploration_converged()
        check(f"连续两代无新 accepted 判为收敛(why={why})",
              converged and "two_generations_without_new_accepted" in why)
        # 未收敛(有 accepted)时不应释放
        t = _mk_trial(fac, generation=2)
        t.data["evaluation"] = {"verdict": "accepted", "trial_id": t.trial_id}
        t.flush()
        pl2 = AutonomousPlanner(run_dir, spec, cfg)
        conv2, _ = pl2._exploration_converged()
        check("存在近两代 accepted 时不判收敛", not conv2)
        # validate:带释放 id 的 hpo 候选放行,不带则仍拒
        battr = {"action": "hpo", "family": "model:hpo",
                 "parent_ids": [base.trial_id] if False else [],
                 "algorithm": "lightgbm", "data_view": "v1",
                 "changed_modules": ["model"],
                 "hypothesis": "h", "falsification": "f",
                 "proposal_id": "prop-x"}
        errs_no = pl2._validate(dict(battr))
        errs_yes = pl2._validate(dict(battr), release_id="prop-x")
        check("无释放名额时仍被特征期门拒",
              "feature_phase_model_locked" in errs_no)
        check("带释放名额时放行(不再报门错)",
              "feature_phase_model_locked" not in errs_yes)
        _q = dict(battr)
        pl2._validate(_q, release_id="prop-x")
        check("放行候选留痕 feature_phase_release.granted",
              (_q.get("feature_phase_release") or {}).get("granted") is True)
        _p = dict(battr)
        pl2._validate(_p, release_id="prop-x")
        check("放行候选携带可读理由",
              bool((_p.get("feature_phase_release") or {}).get("reason")))

    # cmd_trial 认这个标记
    ct_src = (REPO / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    fp_src = (REPO / "runtime" / "feature_phase.py").read_text(
        encoding="utf-8")
    check("new-trial 认 planner 释放名额并落 trial.yaml",
          "planner_release(" in ct_src
          and "feature_phase_override" in ct_src
          and "planner_feature_phase_release" in fp_src)

    # -----------------------------------------------------------------
    print("\n== D: 报告列出被特征期门拒绝的方向 ==")
    rep_src = (REPO / "runtime" / "report.py").read_text(encoding="utf-8")
    check("报告读 planner.feature_phase_gate",
          "_FP_KEY" in rep_src and "planner/latest.json" in rep_src)
    check("报告显式说明'被拒 ≠ 已探索过'",
          "被拒方向不等于" in rep_src)

    # -----------------------------------------------------------------
    print(f"\n{'=' * 52}\n{ok} passed / {fail} failed\n{'=' * 52}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
