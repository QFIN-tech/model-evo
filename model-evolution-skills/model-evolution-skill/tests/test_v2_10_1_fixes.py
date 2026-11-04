"""ModelEvo V2.10.1 回归测试:drawdown-req-t7-v20 run 回写的 7 个 harness 缺陷。

证据基线(run:`example-seq-v3-4-2.8.6-v20`,2026-09-14):
  F1 交付打分 fillna(0)        → final_test 0.7842→0.7706,提升 +0.0098→-0.0038(反号)
  F2 prepare_features 硬解4元组 → finalize-retrain ValueError 崩溃
  F3 白名单只在基础 header 求交 → 528 个 transform 派生列交付期静默丢失
  F4 run-generation 只看 status → --from-parent 的整代 trial 被跳过(avg_parallelism≈0.84×3 代)
  F5 model_axis_tried 恒 True   → 0 个 model 轴 trial 的 run 被据此判穷尽停工
  F7 best_trial 跨视图裸比 AUC  → sample 面行连续三代被选成交付冠军
  F8 报告缺验收目标/口径修正提示 → §12 输出反号结论且无提示

跑法:python tests/test_v2_10_1_fixes.py [--skill-root <path>]
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cli_source  # noqa: E402

PASS = 0
FAIL = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    # v2.11.9:默认值相对本文件推导,不再写死本机绝对路径
    ap.add_argument("--skill-root",
                    default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args()
    root = Path(a.skill_root).resolve()
    sys.path.insert(0, str(root))

    cli_src = cli_source()
    gen_src = (root / "runtime" / "generation.py").read_text(encoding="utf-8")
    evo_src = (root / "runtime" / "evolution.py").read_text(encoding="utf-8")
    lb_src = (root / "runtime" / "leaderboard.py").read_text(encoding="utf-8")
    rep_src = (root / "runtime" / "report.py").read_text(encoding="utf-8")

    def fn_body(src: str, name: str) -> str:
        """取函数体;嵌套(缩进)函数按缩进边界截断,避免吞掉后文同名串。"""
        i = src.find(f"def {name}(")
        if i < 0:
            return ""
        line_start = src.rfind("\n", 0, i) + 1
        indent = i - line_start
        j = i + 1
        while True:
            j = src.find("\ndef ", j)
            if j < 0:
                return src[i:]
            ls = j + 1
            if (ls - src.rfind("\n", 0, ls) - 1) <= indent:
                return src[i:j]
            j = ls

    print("== F1 交付打分口径:不得 fillna ==")
    body = fn_body(cli_src, "_score_split")
    # 静态检查必须先剥注释:修复说明里会引用被删掉的旧写法(如
    # "原为 ...fillna(0)"),否则解释性注释会把断言本身打假。
    code_only = re.sub(r"#.*", "", "\n".join(body.splitlines()))
    check("_score_split 源码可定位", bool(body))
    check("不再出现 .fillna(0) 抹平缺失",
          ".fillna(0)" not in code_only and "fillna(0)" not in code_only)
    check("缺列时显式报错而非静默填 0",
          "missing" in code_only and "RuntimeError" in code_only)
    check("保留 NaN 原生语义(reindex 不带 fill_value=0)",
          "fill_value=0" not in code_only)

    print("\n== F2 prepare_features 返回值形态兼容 ==")
    # v2.11:锚定 F2 调用点本身("_pf = champ.prepare_features(")。
    # 旧锚 find("champ.prepare_features(") 会命中 P7 加宽重放分支的
    # 调用(先于 F2 出现),1200 字符窗口装不下 F2 的显式失败分支
    # (距离 ~1476),检查随插入顺序漂移。锚定唯一前缀消除该脆弱性。
    i = cli_src.find("_pf = champ.prepare_features(")
    seg = cli_src[i - 200:i + 1400] if i > 0 else ""
    check("兼容 4 元组分支", "len(_pf)" in seg and "== 4" in seg)
    check("兼容 2 元组分支", "== 2" in seg)
    check("形态不可识别时显式失败", "无法识别" in seg or "return 1" in seg)

    print("\n== F3 特征白名单解析域并入 config 声明列 ==")
    j = cli_src.find('header = pd.read_parquet(splits / "train.parquet").columns')
    seg3 = cli_src[j:j + 1200] if j > 0 else ""
    check("解析域含 header ∪ declared", "_domain" in seg3 and "declared" in seg3)
    check("exclude_columns 消费并集域", "columns=_domain" in seg3)

    print("\n== F4 run-generation 以代码存在为就绪真值 ==")
    k = cli_src.find("def cmd_run_generation(")
    seg4 = cli_src[k:k + 3000] if k > 0 else ""
    check("_code_present 判据存在", "_code_present" in seg4)
    check("就绪判定不再只看 status",
          "t.status != \"PROPOSED\" or _code_present(t)" in seg4)

    print("\n== F5 model_axis_tried 实测(不再恒 True) ==")
    check("generation.py 不再硬编码 True",
          'conditions["model_axis_tried"] = True' not in gen_src)
    check("改为消费传入值",
          'conditions["model_axis_tried"] = bool(model_axis_tried)' in gen_src)
    check("check() 暴露 model_axis_tried 形参",
          "model_axis_tried: bool | None = None" in gen_src)
    check("evolution 实测实现存在", "_model_axis_tried" in evo_src)

    # 行为测试:直接跑实测判据
    from runtime.evolution import EvolutionEngine as _EC

    class _T:
        def __init__(self, **kw):
            self.trial_id = kw.pop("trial_id", "t")
            self.parent_ids = kw.pop("parent_ids", [])
            self.data = kw

    cases = [
        ("feature-only 的 run 必须判 False(本次事故场景)",
         [_T(status="EVALUATED", changed_modules=["feature"],
             search_action="data_view")], False),
        ("changed_modules 含 model → True",
         [_T(status="EVALUATED", changed_modules=["model"],
             search_action="hpo")], True),
        ("search_action=hpo → True",
         [_T(status="EVALUATED", changed_modules=[], search_action="hpo")],
         True),
        ("PROPOSED 的 model trial 不算试过",
         [_T(status="PROPOSED", changed_modules=["model"],
             search_action="hpo")], False),
        ("FAILED 的 model trial 不算试过",
         [_T(status="FAILED", changed_modules=["model"],
             search_action="hpo")], False),
        ("空 trial 列表 → False(未知不等于已试)",
         [], False),
    ]
    for name, trials, want in cases:
        try:
            got = _EC._model_axis_tried(trials)
        except Exception as e:  # noqa: BLE001
            got = f"EXC:{type(e).__name__}"
        check(name, got == want, f"got={got} want={want}")

    print("\n== F7 冠军选型限定在交付可比视图 ==")
    check("_remember_baseline_view 存在", "_remember_baseline_view" in lb_src)
    check("build() 调用它",
          "_remember_baseline_view(trials, baseline_id)" in lb_src)
    check("best_trial 消费 _baseline_view",
          "_delivery_view" in lb_src and "getattr(self, \"_baseline_view\"" in lb_src)

    from runtime.leaderboard import Leaderboard

    class _Spec:
        primary_metric = "auc"
        data = {"secondary_metrics": []}

    class _Cfg:
        def get(self, k, d=None):
            return d

    lb = Leaderboard.__new__(Leaderboard)
    lb.dir = Path(".")
    lb.spec = _Spec()
    lb.config = _Cfg()
    lb.direction = True

    t_sample = _T(trial_id="t-sample", data_view="sample", generation=1)
    t_v1 = _T(trial_id="t-v1", data_view="v1", generation=2)
    lb._remember_baseline_view([t_sample, t_v1], "t-v1")
    check("基线视图被记为 v1", getattr(lb, "_baseline_view", None) == "v1",
          f"got={getattr(lb, '_baseline_view', None)}")
    lb._remember_baseline_view([t_sample, t_v1], None)
    check("无基线时退回非抽样视图",
          getattr(lb, "_baseline_view", None) == "v1",
          f"got={getattr(lb, '_baseline_view', None)}")

    print("\n== F9 --features 引用模式不得静默丢弃派生列 ==")
    check("_ref_domain(父代权威特征面)存在", "_ref_domain" in cli_src)
    check("_explicit_face(显式清单不求交)存在", "_explicit_face" in cli_src)
    i9 = cli_src.find("def _load_features_ref(")
    seg9 = cli_src[i9:i9 + 8000] if i9 > 0 else ""
    # 三条展开路径都必须改口径:文件引用 / 父代 config / 裁决产物
    n_explicit = seg9.count("_explicit_face(")
    check("文件引用 + 父代 config 均走显式清单口径(≥3 处调用)",
          n_explicit >= 3, f"got={n_explicit}")
    check("裁决产物展开走 _ref_domain(≥2 处)",
          seg9.count("_ref_domain(") >= 2,
          f"got={seg9.count('_ref_domain(')}")
    check("不再与基础帧 derived 直接求交(旧写法已清除)",
          "return [c for c in derived if c in set(keep)]" not in seg9
          and "return [c for c in derived if c in set(data)] or None"
          not in seg9)

    # 行为测试:真实 run 上的引用展开必须能带出 transform 派生列
    # v2.11.9:示例 run 相对 root 定位,不写死绝对路径
    _run = root.parent / "example-seq-v3-4-2.8.6-v20"
    _champ = _run / "trials" / "trial-005-g02-xgboost-1226"
    if (_champ / "model" / "feature_columns.json").exists():
        import importlib.util
        _sp = importlib.util.spec_from_file_location(
            "_mv_probe", root / "scripts" / "modelevo.py")
        _mv = importlib.util.module_from_spec(_sp)
        _sp.loader.exec_module(_mv)
        _base = ["a", "b", "c"]          # 基础面故意不含派生列
        _dom = _mv._ref_domain(_champ, _base)
        check("_ref_domain 取到父代权威特征面(含派生列)", len(_dom) > 100,
              f"got={len(_dom)}")
        check("_ref_domain 含 transform 派生列",
              any(str(c).startswith("seq_") for c in _dom))
    else:
        check("父代权威特征面在档(跳过行为验证)", False,
              "缺少 model/feature_columns.json")

    print("\n== F8 报告:验收目标 + 口径修正提示 ==")
    check("新增验收目标/缺口节", "验收目标与信息量天花板" in rep_src)
    check("缺口以噪声倍数呈现", "× 噪声" in rep_src)
    check("surface scoring_contract_note",
          "scoring_contract_note" in rep_src
          and "交付打分口径修正" in rep_src)
    check("支持交付口径依赖记述位", "delivery-caveats.md" in rep_src)
    # 内联片段的标题降级(独立可测的纯函数;此前两次栽在
    # "漏了源文件自身的 # 首行"与"把引用行误降成 H1")
    from runtime.report import _demote_headings as _dh

    _src = ("# 交付口径依赖(drawdown-req-t7-v20)\n\n"
            "> 引用说明\n> 第二行\n\n"
            "## 1. 子节\n- 列表\n\n### 1.1 更深\n| a | b |\n")
    _out = _dh(_src)
    check("内联片段无残留 H1", not any(l.startswith("# ")
                                      for l in _out.splitlines()))
    check("源文件自身的 # 首行被降级",
          "### 交付口径依赖(drawdown-req-t7-v20)" in _out)
    check("## / ### 各降两级",
          "#### 1. 子节" in _out and "##### 1.1 更深" in _out)
    check("引用行 / 列表 / 表格原样保留",
          "> 引用说明" in _out and "- 列表" in _out and "| a | b |" in _out)
    # drop_leading_title:源首行 H1 与调用方包装标题重复,内联时去掉
    _out2 = _dh(_src, drop_leading_title=True)
    check("去重模式丢弃源首行 H1",
          "交付口径依赖(drawdown-req-t7-v20)" not in _out2
          and "#### 1. 子节" in _out2)
    check("去重模式保留下方全部内容",
          "> 引用说明" in _out2 and "- 列表" in _out2)

    print(f"\n{'='*60}\nv2.10.1 fixes: {PASS} passed, {FAIL} failed")
    if FAILED:
        print("FAILED: " + "; ".join(FAILED))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
