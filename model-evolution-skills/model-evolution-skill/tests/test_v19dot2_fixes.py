"""回归测试 v19.2:静默失败面修复批次。

本批修复的共同主题是**"把 fail-quiet 改成 fail-loud"**——三个缺陷
都不是算错,而是**在出错时沉默**:调用方看到的是"一切正常"。

1. **split 窗口注解污染静默改写切分**(data.py / spec.py)
   实测事故:`split_policy.train` 写成
   `2026-01~2026-04(20260111~20260426,16 期,2,461,339 行)`,
   端点解析失败后静默降级为前缀匹配 → 只匹配 3 个日期前缀,
   其余行全部落进 train 兜底。四个 split 全部退化:train 少 4 个
   快照日、validation 少 4 个、shadow/final_test 整体平移一格,
   而产物里没有任何信号。现在:spec 校验前置拦截 + 数据面 0 命中
   直接抛错。

2. **三处互相矛盾的行数同时出现**(profile.py)
   splits-summary 报 1,117,911 行(2M 采样面)、splits/manifest
   报 2,235,937 行(全量落盘面)、调用方各自带自己的数字。现在
   splits-summary 显式声明 analysis_frame(采样来源 + 口径说明),
   md 里加警示块。

3. **baseline 被 kill 后重跑会重复建 trial**(scripts/modelevo.py)
   实测:一次 400w baseline 连跑 4 轮落盘 6 个 trial,其中 2 个
   已 EVALUATED 的 AUC 完全相同;编排上表现为"一直停在 gen-00"。
   现在已完成的算法直接复用,只补跑缺失/失败的。

覆盖:窗口注解拦截、零命中抛错、部分命中仍兜底、analysis_frame
口径声明、baseline 幂等复用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from _common import SKILL as REPO, cli_source  # noqa: E402

_sys.path.insert(0, str(REPO.parent))

from runtime.data import _match_time_window, split_data  # noqa: E402
from runtime.spec import (  # noqa: E402
    validate_split_policy_windows, window_token_error)

ok = fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        fail += 1
        print(f"  [FAIL] {name}")

def main() -> int:


    # ---------------------------------------------------------------- 1. 端点语法
    print("\n== 1. 窗口端点语法校验(spec 前置拦截) ==")

    check("纯区间端点合法", window_token_error("2026-01~".rstrip("~")) is None)
    check("YYYY-MM-DD 合法", window_token_error("2026-04-26") is None)
    check("YYYYMM 紧凑合法", window_token_error("202604") is None)
    check("YYYYMMDD 紧凑合法", window_token_error("20260426") is None)
    check("注解污染端点被拒(事故原形)",
          window_token_error(
              "2026-04(20260111~20260426,16 期,2,461,339 行)") is not None)
    check("注解污染端点被拒(月粒度端)",
          window_token_error("2026-05~2026-06(20260504~20260621,8 期)") is not None)
    check("空端点被拒", window_token_error("") is not None)
    check("乱码端点被拒", window_token_error("2026-13-99") is not None)

    # 注解污染的完整 policy:四个字段全部应报错
    _ann = {
        "type": "time_based",
        "train": "2026-01~2026-04(20260111~20260426,16 期,2,461,339 行)",
        "validation": "2026-05~2026-06(20260504~20260621,8 期,1,193,118 行)",
        "shadow_test": "2026-06~2026-07(20260629~20260715,4 期)",
        "final_test": "2026-07(20260723~20260731,2 期,279,432 行)",
    }
    _errs = validate_split_policy_windows(_ann)
    check("注解污染 policy 四字段全报错", len(_errs) >= 4)
    check("报错信息带字段名", all("split_policy." in e for e in _errs))
    check("报错信息给修改指引",
          any("YAML 注释" in e or "纯区间" in e for e in _errs))

    # 纯区间 policy 必须零报错(不能误伤正常用法)
    _clean = {
        "type": "time_based",
        "train": "2026-01~2026-04",
        "validation": "2026-05~2026-06",
        "shadow_test": "2026-06-29~2026-07-15",
        "final_test": "2026-07-23~2026-07-31",
    }
    check("纯区间 policy 零报错", validate_split_policy_windows(_clean) == [])

    # random/group_based 不走窗口校验(不能误伤)
    check("random policy 不触发窗口校验",
          validate_split_policy_windows({"type": "random"}) == [])
    check("非 time_based 的 train 字段不被校验",
          validate_split_policy_windows(
              {"type": "group_based", "train": "任意文字"}) == [])

    # 多 '~' 拒绝
    check("多 '~' 被拒",
          len(validate_split_policy_windows(
              {"type": "time_based", "train": "2026-01~2026-04~2026-05"})) >= 1)

    # ---------------------------------------------------------------- 2. 零命中
    print("\n== 2. 窗口零命中:抛错而非静默全归 train ==")

    DAYS = [f"2026-{m:02d}-{d:02d}" for m, d in
            ((1, 11), (1, 19), (2, 4), (2, 12), (3, 1), (3, 9),
             (4, 2), (4, 26), (5, 4), (5, 20), (6, 5), (6, 29), (7, 7), (7, 31))]
    days = pd.Series(DAYS)


    class _FakeSpec:
        """split_data 只需要 data / time_column 两个属性。"""

        def __init__(self, policy):
            self.data = {"split_policy": policy}
            self.time_column = "pday"


    df = pd.DataFrame({"pday": days, "x": np.arange(len(days))})

    # 注解污染的 policy:全部窗口 0 命中 → 必须抛错。
    # 注意用 spec 校验层做门:data.py 侧的 0 命中保险是防御性兜底,
    # 不依赖它跑通(注解污染的完整 policy 在 spec 层就被拦下了)。
    from runtime.spec import TaskSpec  # noqa: E402

    _spec_errs = TaskSpec({"task_id": "t", "model_type": "binary_classification",
                           "target": {"name": "y", "positive_definition": "y>0"},
                           "primary_metric": "auc",
                           "resource_budget": {"max_trials": 1},
                           "split_policy": _ann}).validate()
    check("TaskSpec.validate 拦截注解污染 policy",
          any("split_policy" in e for e in _spec_errs))

    # data.py 侧 0 命中保险:直接构造一个"窗口语法合法但数据里没有一行命中"
    # 的场景(数据日期在窗口之外),此时兜底全部归 train 是错误行为 → 抛错。
    _offwindow = {
        "type": "time_based",
        "train": "2030-01~2030-02",
        "validation": "2030-03~2030-03",
        "shadow_test": "2030-04~2030-04",
        "final_test": "2030-05~2030-05",
    }
    try:
        split_data(df, _FakeSpec(_offwindow))
        zero_hit_raised = False
    except ValueError as exc:
        zero_hit_raised = "0 命中" in str(exc)
    check("窗口与数据完全错开(0 命中)→ 抛错", zero_hit_raised)

    # 纯区间 policy:正常切分,不抛错
    try:
        res = split_data(df, _FakeSpec(_clean))
        clean_ok = True
    except Exception as exc:  # noqa: BLE001
        clean_ok = False
        print(f"      ({exc})")
    check("纯区间 policy 正常切分", clean_ok)
    if clean_ok:
        check("纯区间切分出 train", "train" in res and len(res["train"]) > 0)
        check("纯区间切分出 validation",
              "validation" in res and len(res["validation"]) > 0)
        check("train ∩ validation 为空",
              len(set(res.get("train", [])) & set(res.get("validation", []))) == 0)

    # 部分命中:窗口只覆盖数据的一部分 → 其余行归 train(既有语义保留)
    _partial = {
        "type": "time_based",
        "train": "2026-01~2026-02",
        "validation": "2026-03~2026-03",
        "shadow_test": "2026-04~2026-04",
        "final_test": "2026-05~2026-05",
    }
    try:
        res_p = split_data(df, _FakeSpec(_partial))
        partial_ok = "train" in res_p
        # 6/7 月的行不在任何窗口内 → 应被并入 train
        partial_ok = partial_ok and len(res_p["train"]) >= 6
    except Exception as exc:  # noqa: BLE001
        partial_ok = False
        print(f"      ({exc})")
    check("部分命中仍兜底归 train(不误伤)", partial_ok)

    # ---------------------------------------------------------------- 3. 口径声明
    print("\n== 3. splits-summary 显式声明分析面口径 ==")

    _src = (REPO / "runtime" / "profile.py").read_text(encoding="utf-8")
    check("splits-summary 写入 analysis_frame", "analysis_frame" in _src)
    check("analysis_frame 带 sampled_from", '"sampled_from"' in _src)
    check("analysis_frame 带口径说明 note", '"note"' in _src)
    check("md 渲染警示块(下采样)", "本表行数基于分析面" in _src)
    check("md 渲染泄漏告警(重叠率误读防护)", "不是列级泄漏" in _src)

    # ---------------------------------------------------------------- 4. baseline
    print("\n== 4. baseline 幂等续跑 ==")

    _cli = cli_source()          # v2.11.9:CLI 全集,非单文件
    check("baseline 复用已完成 trial", "[reuse]" in _cli)
    check("按算法名去重复用", "done.setdefault(" in _cli)
    check("只复用 generation==0", 't.data.get("generation") != 0' in _cli)

    print(f"\n=== v19.2 fixes: {ok} passed, {fail} failed ===")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
