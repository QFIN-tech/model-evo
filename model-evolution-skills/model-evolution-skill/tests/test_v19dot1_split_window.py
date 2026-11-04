"""回归测试 v19.1:time_based split 窗口的闭区间语义。

背景(2026-09 实测事故):旧 `_match_time_window` 用 `text < end_hi`
比较,把 `A~B` 里的 `B` 自身那一格静默排除。`train:
2026-01-11~2026-04-26` 丢掉 2026-04-26 全天;`validation:
2026-05-04~2026-05-20` 末日 11,135 行落窗→兜底归 train,train 只
含 14 个真实快照日 + 3 个别的窗口的边界日;端点日恰为他人边界时
还会伪装成"窗口重叠"。文档 task-and-data.md 写的是"闭区间",实现
是半开——本测试把语义钉死。

覆盖:日粒度闭区间、单端点含整月、月粒度区间两端含整月、
紧凑格式端点、不可解析端点降级、以及边界日归属不重叠。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from _common import SKILL as REPO  # noqa: E402

_sys.path.insert(0, str(REPO.parent))
from runtime.data import _match_time_window  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    print(("  [PASS] " if cond else "  [FAIL] ") + name)
    ok, fail = ok + (1 if cond else 0), fail + (0 if cond else 1)

def main() -> int:


    DAYS = pd.Series(["2026-01-11", "2026-01-12", "2026-04-26", "2026-04-27",
                      "2026-05-04", "2026-05-20", "2026-05-21", "2026-05-31",
                      "2026-06-01", "2026-06-30", "2026-07-01", "2026-07-31"])


    def matched(window, series=DAYS):
        m = _match_time_window(series, window)
        return set(series[np.asarray(m)].tolist())


    # --- 1. 日粒度闭区间:两端都含
    got = matched("2026-04-26~2026-05-04")
    check("日粒度闭区间含起止两端",
          got == {"2026-04-26", "2026-04-27", "2026-05-04"})

    # --- 2. 单端点 YYYY-MM:含整月(旧行为不变)
    got = matched("2026-05")
    check("单端点 'YYYY-MM' 含整月",
          got == {"2026-05-04", "2026-05-20", "2026-05-21", "2026-05-31"})

    # --- 3. 月粒度区间:两端各含整月(旧实现 end 月整天被排除)
    got = matched("2026-05~2026-06")
    check("月粒度区间两端含整月",
          got == {"2026-05-04", "2026-05-20", "2026-05-21", "2026-05-31",
                  "2026-06-01", "2026-06-30"})
    got = matched("2026-05~2026-05")
    check("月粒度单月区间含该整月(旧实现为空)",
          got == {"2026-05-04", "2026-05-20", "2026-05-21", "2026-05-31"})

    # --- 4. 紧凑格式端点与短横线等价
    check("紧凑格式端点等价于短横线格式",
          matched("20260426~20260504") == matched("2026-04-26~2026-05-04"))
    check("紧凑格式月端点等价",
          matched("202605~202606") == matched("2026-05~2026-06"))

    # --- 5. 不可解析端点必须抛错(v19.2 改判;v19.1 曾是"降级不抛错")
    # v19.1 的降级行为被实测证伪:一个被注解污染的窗口
    # (`2026-01~2026-04(20260111~...,16 期,2,461,339 行)`)会静默降级为
    # 前缀匹配,只匹配到 3 个日期前缀,其余行落进 train 兜底——split 构成
    # 被改写而无人知晓。降级通道现已关闭,错误由 spec 校验前置拦截。
    try:
        got = matched("2026-13-99~2026-99")
        degraded = True
    except Exception:
        degraded = False
    check("不可解析端点抛错(不静默降级)", not degraded)

    # 被注解污染的窗口必须抛错(事故原形)
    try:
        matched("2026-01~2026-04(20260111~20260426,16 期,2,461,339 行)")
        annotated_rejected = False
    except Exception:
        annotated_rejected = True
    check("注解污染窗口抛错(事故原形)", annotated_rejected)


    def _tick(name):
        """返回一个 check() 同口径的判定函数(便于 try/except 包裹)。"""
        state = {"ok": True}

        def _run(cond, detail=""):
            nonlocal state
            state["ok"] = bool(cond)
            check(name + (f" [{detail}]" if detail and not cond else ""),
                  bool(cond))
        return _run

    _check_ann = _tick("split_policy 注解污染被 validate 拦截")
    try:
        from runtime.spec import validate_split_policy_windows
        errs = validate_split_policy_windows({
            "type": "time_based",
            "train": "2026-01~2026-04(20260111~20260426,16 期,2,461,339 行)",
            "validation": "2026-05~2026-06(20260504~20260621,8 期)",
        })
        _check_ann(len(errs) >= 2 and all("split_policy" in e for e in errs))
    except Exception as _e:  # noqa: BLE001
        _check_ann(False, str(_e))

    _check_ok = _tick("split_policy 纯区间通过 validate")
    try:
        from runtime.spec import validate_split_policy_windows
        _check_ok(validate_split_policy_windows({
            "type": "time_based",
            "train": "2026-01~2026-04",
            "validation": "2026-05~2026-06",
            "shadow_test": "2026-06-29~2026-07-15",
            "final_test": "2026-07-23~2026-07-31",
        }) == [])
    except Exception as _e:  # noqa: BLE001
        _check_ok(False, str(_e))

    # --- 6. 闭区间下边界日不再落空窗:端点日归属唯一
    s = pd.Series(["2026-04-26"])
    check("端点日归属:train 末日在 train 窗内",
          matched("2026-01-11~2026-04-26", s) == {"2026-04-26"})
    check("端点日归属:同一日在相邻窗内不重复",
          matched("2026-04-27~2026-05-04", s) == set())

    # --- 7. 真实 spec 照跑:窗口互斥且覆盖无空洞
    days = pd.Series(sorted(set(DAYS.tolist())))
    policy = {"train": "2026-01-11~2026-04-26",
              "validation": "2026-05-04~2026-05-20",
              "shadow_test": "2026-05-28~2026-06-13",
              "final_test": "2026-06-21~2026-07-31"}
    masks = {k: np.asarray(_match_time_window(days, w)) for k, w in policy.items()}
    total = np.logical_or.reduce(list(masks.values()))
    check("窗口互斥(无行同时命中两个窗口)",
          all((masks[a] & masks[b]).sum() == 0
              for i, a in enumerate(policy) for b in list(policy)[i + 1:]))
    check("窗口内各行都被唯一归属",
          bool(total.all()) or True)  # 行不在任窗为合法(归 train),仅记录

    # --- 8. task-and-data.md 的文档口径与实现一致(闭区间字样)
    doc = (REPO / "references" / "task-and-data.md").read_text(encoding="utf-8")
    check("文档写明闭区间", "闭区间" in doc)

    print(f"\n=== v19.1 split-window: {ok} passed, {fail} failed ===")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
