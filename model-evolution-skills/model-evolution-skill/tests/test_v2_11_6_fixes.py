"""ModelEvo v2.11.6 回归测试:drawdown-req-t7-v211 run 交付期回写的 3 个 harness 缺陷。

证据基线(2026-09-15,run = example-subs/example-v2.11,17 trial / 5 代 /
final_test AUC 0.7758):
  D1 widen 缓存键单边         → _widen_face_key 只哈希**第一侧**帧,而
                               shadow 重放的调用形态是「train 帧当 raw_tr、
                               shadow 帧当 raw_va」——第一侧与训练轮完全相同,
                               键碰撞命中训练缓存,返回 6624 行 validation
                               加宽帧;该帧被当作 shadow 预测落盘,与 6578 行
                               shadow 标签错配。实测 gen-01/gen-02 共 7 个
                               加宽 trial 的 shadow_test_score.parquet 与
                               validation_score.parquet **逐位相同**,而
                               gap sanity 报「shadow 评测失败」被当成
                               split 缺标签的假象(交付链第二把尺子静默污染)
  D2 reparse_frame 零列硬失败 → spec 声明 reparse=[] 时 profile 不落盘 reparse
                               目录(整目录跳过),但访问器对 columns=[] 仍去
                               读盘 ⇒ 每次空请求抛 FileNotFoundError。
                               finalize-retrain 对无 reparse 任务无条件调用它
                               ⇒ **交付链直接阻断**(该 run 只能靠进程内兼容
                               垫片绕过,并留下往 reparse/ 写零信息量对齐键
                               文件的变通产物)
  D3 leaderboard.json 基线锚空 → build() 把**调用方原始参数**写进 JSON,而
                               evolution.finalize() 只传 trials 不传 baseline_id
                               ⇒ JSON 的 baseline_id 恒为 null。report §5 /
                               planner-context / 一切读 JSON 而非 CSV 的消费方
                               拿不到基线锚,report 直接打出与事实相反的
                               "baseline 未完成"(CSV/MD 投影一直是对的)

三者同族前的共同形态:**下游消费方拿到的产物与上游意图不一致,且不报错**。
本文件把一致性固化成断言。

跑法:python tests/test_v2_11_6_fixes.py [--skill-root <path>]
"""
from __future__ import annotations

import argparse
import ast
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL, trial_body, cli_source  # noqa: E402

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


def _load_body_template(cli_src: str) -> str:
    """按 harness 同款锚点抽取渲染 body 字面量。"""
    # v2.11.9:模板已是 _mvo/templates.TRIAL_BODY 模块常量,直接取。
    # 旧实现从 scripts/modelevo.py 源码找赋值锚点 + ast.literal_eval
    # 还原字面量;模板抽出后源码里已无该字面量。
    del cli_src                              # 保持旧签名(调用方仍传)
    return trial_body()


# ---------------------------------------------------------------------------
# D1 缓存键两侧指纹
# ---------------------------------------------------------------------------
def test_widen_key_two_sided() -> None:
    print("== D1 widen 缓存键含两侧帧指纹 ==")
    cli_src = cli_source()
    body = _load_body_template(cli_src)
    seg = body[body.find("def _widen_face_key"):
               body.find("def widen_cached(")]
    check("_widen_face_key 接受两侧帧参数(raw_va/va_cols)",
          "va_rows" in seg and "va_cols" in seg,
          f"签名片段={seg.splitlines()[0][:80]}")
    check("键串含第二侧指纹(va_rows|va_sig)",
          "va_rows" in seg.split("sha256")[-1]
          or "{va_rows}" in seg or "va_sig" in seg)
    # v2.12.2.5:RUN_DIR 必须入键(跨 run 同名 trial + 同 tag + 同帧指纹
    # 会算出同一 key,命中别的 run 实现的列 —— drawdown-t7-sv24 实测)
    check("键串含 RUN_DIR(跨 run 不碰撞)",
          "_run_sig" in seg.split("sha256")[-1]
          or "{_run_sig}" in seg or "RUN_DIR" in seg)
    check("widen_cached 传第二侧指纹",
          "list(raw_va.columns)" in body or "raw_va.columns" in body)

    # 行为断言:同 tag / 同第一侧 / **不同第二侧** → 必须是不同键,
    # 否则 shadow 重放会命中训练缓存(本 run 的实际事故形态)。
    # 另:同 trial 名 / 同 tag / 同两帧、**不同 RUN_DIR** → 必须不同键,
    # 否则跨 run 缓存串味(同一数据面复制出的多 run 会撞键)。
    ns: dict = {}
    prelude = """
import hashlib, pathlib
TRIAL_DIR = pathlib.Path('/tmp/_k')
RUN_DIR = pathlib.Path('/tmp/_run')
"""
    body_exec = body[body.find("def _widen_face_key"):
                     body.find("def widen_cached(")]
    exec(prelude + body_exec, ns)
    key_fn = ns["_widen_face_key"]
    cols = ["a", "b"]
    k_train = key_fn("main", 33862, cols, 6624, cols)     # 训练轮
    k_shadow = key_fn("main", 33862, cols, 6578, cols)    # shadow 重放
    check("同第一侧 / 异第二侧 → 键不同(不再碰撞)",
          k_train != k_shadow, f"{k_train} vs {k_shadow}")
    check("同两侧 → 键稳定(缓存仍可命中)",
          key_fn("main", 33862, cols, 6624, cols) == k_train)
    # 跨 run:RUN_DIR 变 → 键必须变
    ns2: dict = {}
    exec(prelude.replace("/tmp/_run", "/tmp/_run_other") + body_exec, ns2)
    check("同 trial 名/tag/帧、异 RUN_DIR → 键不同(不跨 run 串味)",
          ns2["_widen_face_key"]("main", 33862, cols, 6624, cols)
          != k_train)
    check("第二侧缺省(-1)仍有确定键",
          isinstance(key_fn("main", 33862, cols), str))


# ---------------------------------------------------------------------------
# D2 reparse_frame 零列短路
# ---------------------------------------------------------------------------
def test_reparse_empty_columns() -> None:
    print("== D2 reparse_frame(columns=[]) 不读盘 ==")
    from runtime.templates import TRIAL_TEMPLATE
    cli_src = cli_source()
    body = _load_body_template(cli_src)
    code = TRIAL_TEMPLATE.format(
        trial_id="trial-T", hypothesis="h", actions="a", run_dir="/tmp/nonexist",
        target="y", model_type="binary_classification", treatment="",
        seed=1, skill_root="/tmp/s", body=body)
    ns: dict = {}
    with tempfile.TemporaryDirectory() as td:
        prelude = f"""
import pathlib, pandas as pd
RUN_DIR = pathlib.Path({td!r})
REPARSE_SOURCES = frozenset()
REPARSE_DIR = RUN_DIR / "sample-profile" / "splits" / "reparse"
"""
        start = code.find("def reparse_frame(")
        end = code.find("def assert_banned_clean(")
        segment = code[start:end]
        # 只取函数本身(避免执行整份渲染产物)
        seg_only = segment[:segment.find("\n\n\n")] if "\n\n\n" in segment \
            else segment
        try:
            exec(prelude + seg_only, ns)
        except Exception as exc:  # noqa: BLE001
            check("reparse_frame 段可独立执行", False, str(exc)[:120])
            return
    fn = ns.get("reparse_frame")
    check("reparse_frame 已渲染", callable(fn))
    if not callable(fn):
        return
    try:
        out = fn("train", [])
    except FileNotFoundError as exc:
        check("空列请求不抛 FileNotFoundError", False, str(exc)[:120])
        return
    except Exception as exc:  # noqa: BLE001
        check("空列请求不抛异常", False, f"{type(exc).__name__}: {exc}"[:120])
        return
    check("空列请求返回空 frame", hasattr(out, "empty") and out.empty,
          f"type={type(out).__name__}")
    # 非空列仍须走白名单 + 读盘(隔离性没有被短路削弱)
    try:
        fn("train", ["some_banned_col"])
    except ValueError:
        check("非空列仍受 REPARSE_SOURCES 白名单约束", True)
    except Exception as exc:  # noqa: BLE001
        check("非空列仍受 REPARSE_SOURCES 白名单约束", False,
              f"{type(exc).__name__}: {exc}"[:120])
    else:
        check("非空列仍受 REPARSE_SOURCES 白名单约束", False, "未拒绝非法列")


# ---------------------------------------------------------------------------
# D3 leaderboard.json 基线锚
# ---------------------------------------------------------------------------
def test_leaderboard_json_baseline() -> None:
    print("== D3 leaderboard.json 落盘解析后的 baseline_id ==")
    from runtime.leaderboard import Leaderboard
    src = (SKILL / "runtime" / "leaderboard.py").read_text(encoding="utf-8")
    check("_remember_baseline_view 返回解析值",
          "-> str | None:" in src or "return baseline_id" in src)
    check("build() 接收返回值并回填",
          "_resolved_baseline_id" in src and "resolved_bid" in src)
    check("JSON dump 读回填值",
          '"_resolved_baseline_id"' in src
          or "_resolved_baseline_id," in src
          or "_resolved_baseline_id" in src.split("dump_json")[-1])

    # 行为:无显式 baseline_id 时,从 run-state 解析并写进 JSON
    class _T:
        def __init__(self, tid, view="v1"):
            self.trial_id = tid
            self.data = {"data_view": view, "evaluation": {}}

    with tempfile.TemporaryDirectory() as td:
        run = Path(td)
        (run / "leaderboard").mkdir(parents=True)
        (run / "run-state.json").write_text(
            '{"baseline_id": "trial-001-base"}', encoding="utf-8")
        lb = Leaderboard.__new__(Leaderboard)
        lb.run_dir = run
        lb.dir = run / "leaderboard"
        trials = [_T("trial-001-base"), _T("trial-002-x")]
        resolved = lb._remember_baseline_view(trials, None)
        check("无显式参数时从 run-state 解析基线",
              resolved == "trial-001-base", f"resolved={resolved}")
        check("解析结果记录基线视图",
              lb._baseline_view == "v1", f"view={lb._baseline_view}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-root", default=str(Path(__file__).parent.parent))
    args = ap.parse_args()
    if not (SKILL / "scripts" / "modelevo.py").exists():
        print(f"[FATAL] skill root 无效: {SKILL}")
        return 2
    test_widen_key_two_sided()
    test_reparse_empty_columns()
    test_leaderboard_json_baseline()
    print()
    print("=" * 60)
    print(f"v2.11.6 fixes: {PASS} passed, {FAIL} failed")
    for f in FAILED:
        print(f"FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
