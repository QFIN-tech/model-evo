"""v2.12.2.10 修复批次回归验证。

依据:example-seq-v3-v2.12.2.9 / run-jt-draw-t7(30w × 1513 列)的实测排障。
四项缺陷,每项都有可复现的现场证据:

- F1 planner 的 inventory 候选把"未消费的 reparse 源"排在最后:
  两个 unconsumed 的 seq 源与 20 个 partially_consumed 的 base_face 源
  同分同 novelty,只有 cost_hours 不同(reparse 2.5h → cost_penalty 0.375),
  排序后落在 valid 列表第 21/22 位(共 23),被 slots 截断**全部丢弃**。
  实测:gen-01 落盘的 8 个候选里 seq 出现 0 次,7 个是同一算法的
  近似重复实验。
- F2 `_exploration_converged()` 读 `inv["unconsumed"]` /
  `inv["sources_total"]["unconsumed"]`,而 build_inventory() 返回的是
  `{"sources":..., "summary":{"unconsumed":...}}` → 两个键都不存在,
  函数**永远**返回 False,每代一个的模型方向名额从未生效。
- F3 信息源清单只有读取端与升格端,**没有写入端**:refresh_inventory
  的注释说粒度状态由 subagent 回填(refresh),但代码里没有任何一条
  路径会写 implemented,`unconsumed` 永远清不了零。
- F4 代码生成模板的未定义名缺陷(实测:baseline 模板的 dtype 治理段
  调用 safe_to_numeric 却未定义/未 import)在 preflight 与 lint 都不拦,
  只在 trial 真正跑起来时炸;失败 trial 按协议不可删除。

运行:python tests/test_v2_12_2_10_fixes.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

import yaml  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}: {detail}")


#: 真实 run(现场证据来源);不在环境时相关用例跳过
LIVE_RUN = Path(
    "/data/oceanus_ctr/j-wangyaoxuan-jk/modelevo/model-evo-V2.8/"
    "example-subs/example-seq-v3-v2.12.2.9/run-jt-draw-t7")


def _make_fake_run(tmp: Path, *, seq_unconsumed: bool = True) -> Path:
    """构造最小 run:清单里有两个 unconsumed reparse 源 + 若干
    partially_consumed 的 base_face 源,外加一个已落盘的 planner 批次。"""
    (tmp / "sample-profile").mkdir(parents=True, exist_ok=True)
    (tmp / "planner").mkdir(exist_ok=True)
    (tmp / "trials").mkdir(exist_ok=True)
    sources = []
    for col in ("seq_d365", "seq_gap_d365"):
        sources.append({
            "source_type": "reparse", "column": col,
            "consumption_status": "unconsumed" if seq_unconsumed
            else "fully_consumed",
            "granularity_space": [
                {"dimension": "window", "hint": "多窗计数(7/14/30/…)", "state":
                 "unexplored"},
                {"dimension": "recency", "hint": "距最近 K 次事件的衰减", "state":
                 "unexplored"},
                {"dimension": "aggregation", "hint": "计数/求和/趋势", "state":
                 "unexplored"},
            ],
        })
    for i in range(6):
        sources.append({
            "source_type": "base_face_family",
            "column": f"offline_inner_fam{i}_cnt_1m",
            "consumption_status": "partially_consumed",
            "note": "强 IV 族",
        })
    n_un = sum(1 for s in sources
               if s["consumption_status"] == "unconsumed")
    inv = {
        "created_at_version": "v18",
        "run_dir": str(tmp),
        "sources": sources,
        "summary": {"n_sources": len(sources), "unconsumed": n_un,
                    "partially_consumed": len(sources) - n_un,
                    "fully_consumed": 0},
    }
    (tmp / "sample-profile" / "information-inventory.yaml").write_text(
        yaml.safe_dump(inv, allow_unicode=True), encoding="utf-8")
    return tmp


# ---------------------------------------------------------------------------
# F1: 未消费 reparse 源必须进批次(而不是被 base_face 源挤掉)
# ---------------------------------------------------------------------------
def test_f1_unconsumed_priority() -> None:
    print("== F1 planner 候选优先级与粒度维度展开 ==")
    from runtime.config import Config
    from runtime.planner import AutonomousPlanner
    from runtime.spec import TaskSpec

    tmp = Path(tempfile.mkdtemp())
    try:
        _make_fake_run(tmp)
        (tmp / "task-spec.yaml").write_text(
            yaml.safe_dump({
                "task_id": "t", "task_name": "t", "objective_description": "t",
                "model_type": "binary_classification",
                "business_domains": ["user_operation"],
                "target": {"name": "y", "positive_definition": "y==1",
                           "observation_window": "p", "performance_window": "p+7"},
                "time": {"column": "pday"}, "entity": {"id_column": "uid"},
                "data": {"data_loader": "data-loader.py"},
                "primary_metric": "auc",
                "split_policy": {"type": "time_based", "train": "2026-01",
                                 "validation": "2026-02",
                                 "final_test": "2026-03"},
                "features": {"banned": [], "reparse": ["seq_d365",
                                                       "seq_gap_d365"]},
                "resource_budget": {"max_trials": 80, "max_generations": 8},
                "ensemble_enabled": False,
            }, allow_unicode=True), encoding="utf-8")
        spec = TaskSpec.load(tmp / "task-spec.yaml")
        eng = AutonomousPlanner(tmp, spec, Config.load(skill_config=False))

        cands = eng._inventory_candidates(None)
        seq = [c for c in cands if "seq_d365" in str(c.get("family"))]
        check("unconsumed reparse 源产出候选", len(seq) >= 1,
              f"seq 候选数={len(seq)}")
        check("同一源的粒度维度展开(>1 个候选)",
              len(seq) >= 2, f"seq_d365 候选数={len(seq)}")
        base = [c for c in cands if "offline_inner" in str(c.get("family"))]
        if seq and base:
            check("unconsumed 源的信息增益高于 partially 源",
                  seq[0]["acquisition"]["information_gain"]
                  > base[0]["acquisition"]["information_gain"],
                  f"{seq[0]['acquisition']['information_gain']} vs "
                  f"{base[0]['acquisition']['information_gain']}")
            check("unconsumed 源的 acquisition score 更高",
                  seq[0]["acquisition"]["score"]
                  > base[0]["acquisition"]["score"],
                  f"{seq[0]['acquisition']['score']} vs "
                  f"{base[0]['acquisition']['score']}")
        # 端到端:plan() 选出的候选里必须有 seq
        batch = eng.plan(slots=6)
        picked = [p["family"] for p in batch["proposals"]]
        check("plan() 批次里出现未消费源",
              any("seq" in f for f in picked),
              f"选中={picked}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_f1_live_run_batch_contains_seq() -> None:
    print("== F1b 现场复核:真实 run 重规划后 seq 必须进批次 ==")
    if not LIVE_RUN.exists():
        check("现场 run 不在环境(跳过)", True)
        return
    from runtime.config import Config
    from runtime.planner import AutonomousPlanner
    from runtime.spec import TaskSpec

    spec = TaskSpec.load(LIVE_RUN / "task-spec.yaml")
    cfg = Config.load(run_config=LIVE_RUN / "modelevo-config.yaml")
    eng = AutonomousPlanner(LIVE_RUN, spec, cfg)
    parent = eng._parent()
    gen = (eng._confirmation_candidates()
           + eng._inventory_candidates(parent)
           + eng._direction_candidates(parent, 8)
           + eng._model_candidates(parent))
    seq = [g for g in gen if "seq_d365" in str(g.get("family"))]
    check("真实 run:seq 候选已生成", len(seq) >= 2, f"n={len(seq)}")
    if seq:
        best = max(g["acquisition"]["score"] for g in gen
                   if "seq" in str(g.get("family")))
        others = [g["acquisition"]["score"] for g in gen
                  if "offline_inner" in str(g.get("family"))]
        check("真实 run:seq 分数高于 base_face 家族",
              (not others) or best > max(others),
              f"seq={best} others_max={max(others) if others else None}")


# ---------------------------------------------------------------------------
# F2: 收敛检测器必须真的读得到清单状态
# ---------------------------------------------------------------------------
def test_f2_convergence_detector() -> None:
    print("== F2 _exploration_converged 读对清单结构 ==")
    from runtime.config import Config
    from runtime.planner import AutonomousPlanner
    from runtime.spec import TaskSpec
    import runtime.inventory as invmod

    tmp = Path(tempfile.mkdtemp())
    orig = invmod.build_inventory
    try:
        _make_fake_run(tmp, seq_unconsumed=True)
        (tmp / "task-spec.yaml").write_text(
            (LIVE_RUN / "task-spec.yaml").read_text(encoding="utf-8")
            if LIVE_RUN.exists() else "", encoding="utf-8")
        if not (tmp / "task-spec.yaml").read_text().strip():
            check("缺少真实 spec 模板(跳过)", True)
            return
        spec = TaskSpec.load(tmp / "task-spec.yaml")
        cfg = Config.load(skill_config=False)

        # (a) 清单里还有 unconsumed → 未收敛
        conv, why = AutonomousPlanner(tmp, spec, cfg)._exploration_converged()
        check("unconsumed>0 时判未收敛", conv is False, f"({conv}, {why})")

        # (b) 把清单整体置为 fully_consumed → 必须判收敛
        def _all_consumed(run_dir, spec_):
            d = orig(run_dir, spec_)
            for s in d["sources"]:
                s["consumption_status"] = "fully_consumed"
            d["summary"] = {"n_sources": len(d["sources"]), "unconsumed": 0,
                            "partially_consumed": 0,
                            "fully_consumed": len(d["sources"])}
            return d

        invmod.build_inventory = _all_consumed
        conv2, why2 = AutonomousPlanner(
            tmp, spec, cfg)._exploration_converged()
        check("unconsumed==0 时判收敛(旧实现恒 False)",
              conv2 is True, f"({conv2}, {why2})")

        # (c) 旧结构(顶层 unconsumed)兼容
        def _legacy(run_dir, spec_):
            d = orig(run_dir, spec_)
            d.pop("summary", None)
            d["unconsumed"] = 0
            return d

        invmod.build_inventory = _legacy
        conv3, _ = AutonomousPlanner(tmp, spec, cfg)._exploration_converged()
        check("兼容顶层 unconsumed 的旧结构", conv3 is True, f"{conv3}")
    finally:
        invmod.build_inventory = orig
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# F3: 清单必须有写入端(v2.17.2 D1:只认 trial 证据 → partial;规划不算)
# ---------------------------------------------------------------------------
def test_f3_inventory_writeback() -> None:
    print("== F3 清单状态回流:trial 证据 → partial,规划不算,Agent 结论粘性 ==")
    from runtime.inventory import refresh_inventory
    from runtime.spec import TaskSpec

    if not LIVE_RUN.exists():
        check("现场 run 不在环境(跳过)", True)
        return
    spec = TaskSpec.load(LIVE_RUN / "task-spec.yaml")
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "sample-profile").mkdir()
        (tmp / "planner").mkdir()
        shutil.copy(LIVE_RUN / "sample-profile" / "feature-quality.csv",
                    tmp / "sample-profile" / "feature-quality.csv")
        (tmp / "trials").mkdir()
        (tmp / "planner" / "proposal-batch-gen-01.yaml").write_text(
            yaml.safe_dump({"proposals": [
                {"family": "info:seq_d365:window"},
                {"family": "info:seq_d365:recency"}]}, allow_unicode=True),
            encoding="utf-8")

        # v2.17.2(D1):只有规划、没有 trial 证据 → 不标 partial(防自反馈)
        r0 = refresh_inventory(tmp, spec)
        s0 = next(x for x in r0["sources"] if x["column"] == "seq_d365")
        st0 = {g["dimension"]: g["state"] for g in s0["granularity_space"]}
        check("仅规划 → 粒度仍 unexplored",
              st0.get("window") == "unexplored", f"{st0}")

        _td = tmp / "trials" / "trial-002-g01-explore-xgboost-1"
        _td.mkdir()
        (_td / "trial.yaml").write_text(yaml.safe_dump({
            "trial_id": _td.name, "status": "EVALUATED",
            "family_hints": ["info:seq_d365:window", "info:seq_d365:recency"],
            "evaluation": {"verdict": "promising"}}, allow_unicode=True),
            encoding="utf-8")
        r1 = refresh_inventory(tmp, spec)
        s1 = next(x for x in r1["sources"] if x["column"] == "seq_d365")
        states = {g["dimension"]: g["state"] for g in s1["granularity_space"]}
        check("trial 证据 → 对应粒度标 partial",
              states.get("window") == "partial", f"{states}")
        check("trial 证据 → 源升 partially_consumed",
              s1["consumption_status"] == "partially_consumed",
              s1["consumption_status"])
        check("未命中的粒度仍 unexplored",
              states.get("aggregation") == "unexplored", f"{states}")

        # 粘性:Agent 写 implemented 后刷新不得降级
        for s in r1["sources"]:
            for g in s.get("granularity_space") or []:
                if s["column"] == "seq_d365" and g["dimension"] == "window":
                    g["state"] = "implemented"
                    g["state_source"] = "agent"
        (tmp / "sample-profile" / "information-inventory.yaml").write_text(
            yaml.safe_dump(r1, allow_unicode=True), encoding="utf-8")
        r2 = refresh_inventory(tmp, spec)
        s2 = next(x for x in r2["sources"] if x["column"] == "seq_d365")
        st2 = {g["dimension"]: g["state"] for g in s2["granularity_space"]}
        check("Agent 的 implemented 不被刷新降级",
              st2.get("window") == "implemented", f"{st2}")

        # trial 证据派生的 partial 不粘:证据消失即回落
        shutil.rmtree(_td)
        r3 = refresh_inventory(tmp, spec)
        s3 = next(x for x in r3["sources"] if x["column"] == "seq_d365")
        st3 = {g["dimension"]: g["state"] for g in s3["granularity_space"]}
        check("trial 派生的 partial 不粘(证据消失即回落)",
              st3.get("recency") == "unexplored", f"{st3}")
        # seq_gap_d365 从未有 trial 证据,应保持 unconsumed
        s4 = next(x for x in r3["sources"] if x["column"] == "seq_gap_d365")
        check("从未被触达的源保持 unconsumed",
              s4["consumption_status"] == "unconsumed",
              s4["consumption_status"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# F4: 生成代码自洽性 —— 模板缺陷必须在写盘前被拦
# ---------------------------------------------------------------------------
def test_f4_code_audit() -> None:
    print("== F4 生成代码自洽性检查 ==")
    from runtime.baseline import (BASELINE_TEMPLATE, ESTIMATOR_CODE,
                                  FIT_CODE)
    from runtime.code_audit import check_generated_code

    def render(algo: str = "lightgbm") -> str:
        return BASELINE_TEMPLATE.format(
            algorithm=algo, target="__t__", model_type="binary_classification",
            treatment="", seed=0, n_estimators_cap=2000,
            early_stopping_rounds=50, es_fraction=0.1,
            estimator_code=ESTIMATOR_CODE[algo], fit_code=FIT_CODE[algo])

    for algo in sorted(ESTIMATOR_CODE):
        check(f"当前 BASELINE_TEMPLATE[{algo}] 自洽",
              check_generated_code(render(algo)) == [],
              str(check_generated_code(render(algo))[:2]))

    # 回退历史修复 → 必须被抓
    code = render("lightgbm")
    i = code.find("def safe_to_numeric")
    j = code.find("def load_config")
    if i > 0 and j > i:
        problems = check_generated_code(code[:i] + code[j:])
        check("safe_to_numeric 缺失被抓(现场事故复现)",
              any("safe_to_numeric" in p for p in problems), f"{problems}")
    # 常量缺失(非调用形式)也必须被抓
    k = code.find("_NUM_STR_RE = re.compile")
    if k > 0:
        k2 = code.find("\n\n", k)
        problems = check_generated_code(code[:k] + code[k2:])
        check("常量缺失被抓(只检测调用名会漏)",
              any("_NUM_STR_RE" in p for p in problems), f"{problems}")
    # 局部 import 被删(roc_auc_score 型风险)
    s = code.find("from sklearn.metrics import roc_auc_score")
    if s > 0:
        problems = check_generated_code(
            code[:s] + code[s + len("from sklearn.metrics import "
                                    "roc_auc_score"):])
        check("roc_auc_score 型缺陷被抓(删局部 import)",
              any("roc_auc_score" in p for p in problems), f"{problems}")
    check("语法错误被判为问题",
          check_generated_code("def f(:\n pass") != [])

    # preflight 接入
    from runtime.preflight import check_generated_templates
    check("preflight.check_generated_templates 对现状返回空",
          check_generated_templates() == [],
          str(check_generated_templates()[:2]))


def test_f4_live_failed_trials_detected() -> None:
    print("== F4b 现场复核:已知失败 trial 必须被规则 19 抓到 ==")
    if not LIVE_RUN.exists():
        check("现场 run 不在环境(跳过)", True)
        return
    import importlib.util as ilu

    lint_path = SKILL / "assets" / "lint_trial.py"
    spec = ilu.spec_from_file_location("modelevo_lint_v2122", lint_path)
    lm = ilu.module_from_spec(spec)
    spec.loader.exec_module(lm)
    broken = LIVE_RUN / "trials" / "trial-001-g00-lightgbm-1824"
    ok = LIVE_RUN / "trials" / "trial-003-g00-lightgbm-1841"
    if broken.exists():
        errs = lm.lint_trial(LIVE_RUN, broken, False)
        check("历史失败 trial(NameError)safe_to_numeric 被规则 19 抓到",
              any("safe_to_numeric" in e for e in errs),
              str([e for e in errs if "规则 19" in e][:2]))
    if ok.exists():
        errs = lm.lint_trial(LIVE_RUN, ok, False)
        check("正常 trial 不被规则 19 误报",
              not any("规则 19" in e for e in errs),
              str([e for e in errs if "规则 19" in e][:2]))


def test_protocol() -> None:
    print("== 协议与版本 ==")
    from runtime import HARNESS_PROTOCOL_VERSION, __version__
    from runtime.harness import harness_hash

    check("版本号已推进(>= 2.12.2.10)",
          tuple(int(x) for x in __version__.split(".")) >= (2, 12, 2, 10),
          __version__)
    check("HARNESS_PROTOCOL_VERSION >= 21", HARNESS_PROTOCOL_VERSION >= 21)
    check("harness_hash 可计算且为 16 位",
          len(harness_hash()) == 16)


def main() -> int:
    test_f1_unconsumed_priority()
    test_f1_live_run_batch_contains_seq()
    test_f2_convergence_detector()
    test_f3_inventory_writeback()
    test_f4_code_audit()
    test_f4_live_failed_trials_detected()
    test_protocol()
    print(f"=== v2.12.2.10: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
