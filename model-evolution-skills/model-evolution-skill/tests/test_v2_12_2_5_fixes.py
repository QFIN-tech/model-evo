"""v2.12.2.5 修复批次回归测试(实测事故:drawdown-t7-sv24 run)。

覆盖 8 项修复中的 6 项(另 2 项为文档/口径说明):

A. widen 缓存键含 RUN_DIR(跨 run 同名 trial 撞键 → 静默拿到别人的列)
B. transition(SUCCEEDED) 清空残留 failure_reason
C. trial_currently_failed 取代裸读 failure_reason 作门槛
D. lint 规则 16(_widen_face_key 缺 RUN_DIR 即拒)
E. report §11/§12/§13/§14 在无冠军时回落到最优非冠军 Trial
F. finalize-retrain 无冠军时落"未确认候选快照"(不冒充冠军产物)
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("[OK]   " if cond else "[FAIL] ") + name
          + ("" if cond else f"  <- {detail}"))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
# A. widen 缓存键含 RUN_DIR
# ---------------------------------------------------------------------------
def test_widen_key_includes_run_dir() -> None:
    print("== A widen 缓存键含 RUN_DIR ==")
    from _mvo.templates import TRIAL_BODY  # noqa: E402

    seg = TRIAL_BODY[TRIAL_BODY.find("def _widen_face_key"):
                     TRIAL_BODY.find("def widen_cached(")]
    check("键体引用 RUN_DIR", "RUN_DIR" in seg, seg[-200:])
    check("键串含 _run_sig", "_run_sig" in seg)

    ns: dict = {}
    prelude = (
        "import hashlib, pathlib\n"
        "TRIAL_DIR = pathlib.Path('/tmp/_k')\n"
        "RUN_DIR = pathlib.Path('/tmp/_runA')\n")
    exec(prelude + seg, ns)
    kA = ns["_widen_face_key"]("main", 100, ["a"], 50, ["a"])
    ns2: dict = {}
    exec(prelude.replace("_runA", "_runB") + seg, ns2)
    kB = ns2["_widen_face_key"]("main", 100, ["a"], 50, ["a"])
    check("同 trial 名/tag/帧、异 RUN_DIR → 键不同", kA != kB,
          f"{kA} vs {kB}")
    check("同 RUN_DIR → 键稳定", ns["_widen_face_key"](
        "main", 100, ["a"], 50, ["a"]) == kA)


# ---------------------------------------------------------------------------
# B/C. failure_reason 残留
# ---------------------------------------------------------------------------
class _FakeTrial:
    def __init__(self, data: dict):
        self.data = data


def test_failure_reason_residue() -> None:
    print("== B/C 残留 failure_reason 不再挡住成功 Trial ==")
    from runtime.trial import (confirmation_parent_eligibility,
                               trial_currently_failed)

    # 典型残留:先失败 → 同 trial 重跑成功(accepted, integrity 有效)
    residue = _FakeTrial({
        "status": "EVALUATED",
        "result": {"status": "succeeded"},
        "failure_reason": "code_error",
        "evaluation": {"verdict": "accepted",
                       "integrity": {"valid": True},
                       "selection_correction": {"passed": True}},
    })
    check("残留失败字段但当前 succeeded → 不算失败",
          not trial_currently_failed(residue))
    ok, why = confirmation_parent_eligibility(residue)
    check("残留失败字段不否决 confirm 父代资格", ok, why)

    # 真失败
    real = _FakeTrial({"status": "FAILED", "result": {"status": "failed"},
                       "failure_reason": "code_error",
                       "evaluation": {"verdict": "accepted"}})
    check("当前 FAILED → 判失败", trial_currently_failed(real))
    ok2, why2 = confirmation_parent_eligibility(real)
    check("当前 FAILED 否决父代资格", (not ok2) and why2 == "parent_has_failure",
          why2)

    # 旧记录(无 status/result)保守拒绝
    legacy = _FakeTrial({"failure_reason": "code_error",
                         "evaluation": {"verdict": "accepted"}})
    check("旧记录无 status → 保守判失败", trial_currently_failed(legacy))

    # transition(SUCCEEDED) 清字段
    from runtime.trial import TrialRecord
    src = pathlib.Path(TrialRecord.__module__ and
                       __import__("inspect").getsourcefile(TrialRecord))
    body = src.read_text(encoding="utf-8")
    check("transition 在 SUCCEEDED 时清除 failure_reason",
          'new_state == "SUCCEEDED"' in body
          and 'pop("failure_reason"' in body)


# ---------------------------------------------------------------------------
# D. lint 规则 16
# ---------------------------------------------------------------------------
def test_lint_rule16() -> None:
    print("== D lint 规则 16 ==")
    sys.path.insert(0, str(SKILL / "assets"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_lint16", SKILL / "assets" / "lint_trial.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = mod.check_widen_cache_provenance

    bad = ("def _widen_face_key(tag, b, tr_rows, tr_cols, va_rows=()):\n"
           "    import hashlib\n"
           "    return hashlib.sha256("
           "f'{TRIAL_DIR.name}|{tag}'.encode()).hexdigest()[:16]\n")
    errs = fn(bad)
    check("缺 RUN_DIR 的键函数被判违规", len(errs) == 1, str(errs))

    good = bad.replace("f'{TRIAL_DIR.name}|{tag}'",
                       "f'{RUN_DIR.resolve()}|{TRIAL_DIR.name}|{tag}'")
    check("含 RUN_DIR 的键函数通过", fn(good) == [], str(fn(good)))

    check("无 _widen_face_key 的代码不触发", fn("def main():\n    pass\n") == [])
    check("豁免注释生效",
          fn("# lint:skip-widen-cache-provenance\n" + bad) == [])
    # 已注册进 lint_trial 主流程
    lint_src = (SKILL / "assets" / "lint_trial.py").read_text(encoding="utf-8")
    check("规则 16 已注册进 lint_trial()",
          "errors.extend(check_widen_cache_provenance(code))" in lint_src)


# ---------------------------------------------------------------------------
# E/F. 无冠军时的回落呈现与快照
# ---------------------------------------------------------------------------
def test_delivery_fallback_sources() -> None:
    print("== E/F 无冠军回落路径存在 ==")
    rep = (SKILL / "runtime" / "report.py").read_text(encoding="utf-8")
    check("report 有 _best_nonchampion 回落选型",
          "def _best_nonchampion" in rep)
    check("§11 回落带非交付配方声明", "不是交付配方" in rep)
    check("§12 回落标注非交付结论", "回落口径(非交付结论)" in rep)
    check("§13 回落读 trial 级 evaluation", "回落 Trial" in rep)
    check("§14 回落给出确认门归因",
          "fresh_confirmation" in rep and "确认门失败不等于候选能力不足" in rep)

    deliv = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    check("finalize-retrain 有降级快照函数",
          "def _stage_unconfirmed_snapshot" in deliv)
    # v2.12.2.6:降级快照必须从**两条**无冠军路径都可达。实测事故:
    # `best_trial(delivery=True) is None` 分支旧实现直接 `return 1`,
    # 于是 "_stage_unconfirmed_snapshot" 成了死代码,"无冠军自动降级
    # 落盘"永不触发(只在显式 --champion 未确认候选时才可达)。
    # 注:断言前先剥注释/文档字符串 —— 修复说明里会出现 "return 1"、
    # "recipe.yaml" 等字样,裸子串匹配会误判(我第一版就是这么写错的)。
    def _code_only(src: str) -> str:
        import re as _re
        out = _re.sub(r'""".*?"""', "", src, flags=_re.S)
        out = _re.sub(r"'''.*?'''", "", out, flags=_re.S)
        return "\n".join(_re.sub(r"#[^\n]*", "", ln)
                         for ln in out.splitlines())

    _best_none_branch = _code_only(
        deliv.split("best = lb.best_trial(")[1]
        .split("champ_id = best.trial_id")[0])
    check("best_trial None 分支调用降级快照(不是死代码)",
          "_stage_unconfirmed_snapshot" in _best_none_branch,
          _best_none_branch)
    check("best_trial None 分支不再直接 return 1",
          "return 1" not in _best_none_branch, _best_none_branch)
    # 快照的**拷贝白名单**里不得出现冠军位产物(那两个位置会被
    # evaluate-final / 交付校验当成冠军产物)。
    # v2.13.4:白名单提为模块常量 _UNCONFIRMED_EVIDENCE,且合约反转——
    # 旧合约要求"含 model.joblib/feature_columns.json",恰恰是把 trial 的
    # train-only 搜索模型拷进 final-model/ 下的陷阱(实测 req-t7-drawdown
    # 因此交付了未重训模型)。未确认路径的模型改由 finalize-retrain 同口径
    # 重训写入 unconfirmed/model/;证据白名单**不得**含任何模型二进制。
    from _mvo.cmd_deliver import _UNCONFIRMED_EVIDENCE
    _for_rel = " ".join(_UNCONFIRMED_EVIDENCE)
    check("快照拷贝白名单不含 recipe.yaml",
          "recipe" not in _for_rel, _for_rel)
    check("快照拷贝白名单不含 model_meta.json",
          "model_meta" not in _for_rel, _for_rel)
    check("快照拷贝白名单不含任何模型二进制(train-only 搜索工件)",
          not any(k in _for_rel for k in (
              "model.joblib", "model.json", "model.txt", "model.bin",
              "model.pt", "validation_input.parquet")), _for_rel)
    _snap_raw = deliv.split("def _write_unconfirmed_evidence")[1] \
        .split("def cmd_finalize_retrain")[0]
    check("快照只落 final-model/unconfirmed/",
          '"unconfirmed"' in _snap_raw)
    # v2.12.2.6:回落段不得把 .parquet 交给默认 JSON loader —— 实测
    # `_trial_artifact(id, "shadow_test_score.parquet")` 抛
    # UnicodeDecodeError,整份 report 在写盘前崩溃(回落逻辑虽在,落地为零)。
    _fb_seg = rep.split("if fallback and not final_eval:")[1].split(
        "if _sd")[0]
    check("§12 回落未用 JSON loader 读 parquet",
          "_trial_artifact" not in _fb_seg,
          _fb_seg[-300:])
    check("快照声明未 acquire final_test gate",
          "未对 final_test 打分" in deliv or "未 acquire gate" in deliv)
    check("无合法候选时仍保持旧行为(报错+空)",
          "无合法候选" in deliv)


def main() -> int:
    test_widen_key_includes_run_dir()
    test_failure_reason_residue()
    test_lint_rule16()
    test_delivery_fallback_sources()
    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败: {FAILS}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
