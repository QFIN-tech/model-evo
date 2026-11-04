"""v2.20.1 修复批次：并行 seed worker 配方/训练面丢失。

依据：run-req-t7-clf gen-04（2026-11-03）confirm trial 实测事故。

## 事故

`confirm` 默认 `parallel_seeds: auto`。baseline 血缘 trial 的 multi-seed
separation 走 `runtime.seed_parallel.start_seed_workers`：主进程把拟合面
pickle 到 /dev/shm，子进程 `python -m runtime.seed_parallel --seed-worker`
把 trial 脚本当模块导入（`__name__ != "__main__"`，main() 不执行）后调用
脚本自身的 `_fit_seed_variant`。

问题：`_fit_seed_variant`（嫁接版，由 `scripts/_mvo/templates.py` 的
`TRIAL_BODY` 定义）依赖三个**模块级全局**：

    _PRIMARY_FIT_PARAMS   主轮 HPO / config 配方
    _PRIMARY_FIT_XY       主轮训练面 (X_tr, y_tr)
    _LAST_EVAL_XY         主轮早停面 (X_va, y_va)

它们只在**主进程** `fit_and_score` 内被赋值。worker 是独立进程，三个都是
None，于是落进兜底分支：

    else:
        X_all, y_all = X_fit, y_fit
        if eval_set:
            X_all = concat([X_fit, eval_set[0][0]])   # ← validation 进训练集
    _params = dict(globals().get("_PRIMARY_FIT_PARAMS") or {})  # ← 默认超参

v2.16.0 起早停面默认是 outer validation（`_ES_MAIN_ANCHOR` 让
`X_es, y_es = X_va, y_va`），所以那个"把 eval_set 拼回去"的兜底从
train_tail 时代的正确行为变成了**验证集泄漏**。

实测（trial-014-g04-confirm-xgboost-2351）：

    primary  0.802235   主进程
    seed-1   0.802367   主进程      ← 真实 seed 噪声 1.3e-4
    seed-2   0.837398   并行 worker ← +0.035，270×

    |primary-seed1| 均值 0.0092 / |seed1-seed2| 0.0213（2.31×）
    Spearman: primary-seed1 0.9980 / primary-seed2 0.9895
    top-1% 重叠: 0.918 / 0.811

退化路径可定位：`_PRIMARY_FIT_PARAMS` 为空 → `_build_model({}, ...)` →
`build_gbm` 默认参数 + `n_estimators_cap`（本 run 2000），是个 2000 轮弱
模型 + 224 万含验证集的行，所以只虚高 +0.035 而非冲到 0.99。

后果：`delta_used` 用了被污染的三 seed 均值（+0.0116，真值约 -2e-4），
把非劣/保真 margin 撑到 0.047（形同虚设）；同时稳定性门被同一个坏 seed
拦下（multi_seed_std 0.0175 > 0.0005）。两个判据都被同一处污染带歪，
最终记为 unconfirmed —— 很容易被误读成"配方对 seed 敏感"。sigma_seed 是
confirm 判据的核心输入，受影响的是确认门本身。

## 修复（三处）

1. `runtime/seed_parallel.py`
   - `start_seed_workers` 新增 `primary_params` / `primary_xy` 关键字参数，
     随 pickle payload 下发；
   - `_worker_main` 装配 `mod._PRIMARY_FIT_PARAMS / _PRIMARY_FIT_XY /
     _LAST_EVAL_XY`，缺参时 return 2（调用方按 rc!=0 在主进程串行补齐）。

2. `runtime/baseline.py`
   - 调用点传 `primary_params=globals().get("_PRIMARY_FIT_PARAMS")` 与
     `primary_xy=globals().get("_PRIMARY_FIT_XY")`。当前血缘没有这两个名字
     （get 得 None），对 baseline 自身无副作用；对嫁接版后代才有值。

3. `scripts/_mvo/templates.py`
   - `_fit_seed_variant` 删除"把 eval_set 拼进训练集 + 静默退回默认超参"
     的兜底，改为缺 `_PRIMARY_FIT_XY` / `_PRIMARY_FIT_PARAMS` 时抛
     RuntimeError。

## 判别门槛（关键）

只有**嫁接版** `_fit_seed_variant` 声明了 `_PRIMARY_FIT_PARAMS/_PRIMARY_FIT_XY`；
baseline 血缘自带的版本（`build_estimator` + 全局 `SEED`）根本没有这两个名字，
如果对它们也要求 primary_*，会把本来正常的并行路径整条打成串行。
因此 worker 侧用 `hasattr(mod, "_PRIMARY_FIT_XY")` 区分。

运行：python tests/test_v20dot1_seed_variant.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILS.append(msg)


def _trial_body() -> str:
    src = (SKILL / "scripts" / "_mvo" / "templates.py").read_text()
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", "") == "TRIAL_BODY"):
            return node.value.value
    raise AssertionError("TRIAL_BODY 未找到")


def test_template_hardened() -> None:
    print("\n[1] 模板不再把 eval_set 拼进训练集")
    body = _trial_body()
    check("concat([X_fit, eval_set[0][0]])" not in body,
          "旧 concat([X_fit, eval_set[0][0]]) 已移除")
    check("禁止把 eval_set 拼进训练集" in body,
          "缺 _PRIMARY_FIT_XY 时抛 RuntimeError")
    check("_PRIMARY_FIT_PARAMS(主轮配方未赋值)" in body,
          "缺 _PRIMARY_FIT_PARAMS 时也抛错（不再静默用默认超参）")

    # 嫁接段必须可独立解析（render._upgrade_hpo_lineage_code 按锚点切片）
    i = body.find("\n#: [v2.15.0] 主轮 fit_and_score 的重训配方")
    j = body.find("\ndef step(", i)
    check(i >= 0 and j > i, "嫁接段锚点存在")
    seg = body[i:j]
    ast.parse(seg)
    check(True, "嫁接段可独立 AST 解析")
    check("_PRIMARY_FIT_XY = None" in body, "幂等判据仍存在")


def test_worker_assembly() -> None:
    print("\n[2] worker 装配主轮配方/训练面")
    src = (SKILL / "runtime" / "seed_parallel.py").read_text()
    check("primary_params: dict | None = None" in src, "签名含 primary_params")
    check("primary_xy: Any = None" in src, "签名含 primary_xy")
    check('"primary_params": primary_params' in src, "payload 携带 primary_params")
    check('"primary_xy": primary_xy' in src, "payload 携带 primary_xy")
    check("mod._PRIMARY_FIT_XY = (_pxy[0], _pxy[1])" in src,
          "worker 装配 _PRIMARY_FIT_XY")
    check("mod._PRIMARY_FIT_PARAMS = dict(_pp)" in src,
          "worker 装配 _PRIMARY_FIT_PARAMS")
    check("mod._LAST_EVAL_XY = (d[\"eval_set\"][0][0], d[\"eval_set\"][0][1])" in src,
          "worker 装配 _LAST_EVAL_XY（早停面）")
    check("return 2" in src, "缺参时返回非零码（触发串行补齐）")

    print("\n[3] 判别门槛：不误伤 baseline 血缘")
    check('_needs_primary = hasattr(mod, "_PRIMARY_FIT_XY")' in src,
          "按模块是否声明 _PRIMARY_FIT_XY 区分嫁接版/基线版")
    i = src.find("_needs_primary")
    j = src.find("os.environ[\"MODELEVO_SEED_VARIANT\"]", i)
    guard = src[i:j]
    depth0 = guard.find("if _needs_primary:")
    check(depth0 >= 0, "装配逻辑在 if _needs_primary 内")
    plain = guard[:depth0]
    check("mod._PRIMARY_FIT_XY =" not in plain,
          "基线版不装配 _PRIMARY_FIT_XY（不会被打成串行）")


def test_call_site() -> None:
    print("\n[4] 调用点传递")
    src = (SKILL / "runtime" / "baseline.py").read_text()
    check("primary_params=globals().get(\"_PRIMARY_FIT_PARAMS\")" in src,
          "baseline.py 传 primary_params")
    check("primary_xy=globals().get(\"_PRIMARY_FIT_XY\")" in src,
          "baseline.py 传 primary_xy")


def test_graft_end_to_end() -> None:
    print("\n[5] 端到端：注入 _HPO_ALGO 后嫁接，新代码进入产物")
    from _mvo import render as R

    # SKILL = <proj>/.claude/skills/modelevo → parents[2] = <proj>
    run = SKILL.parents[2] / "run-req-t7-clf"
    cand = run / "trials" / "trial-006-g02-explore-xgboost-1933" / "code" / "train.py"
    if not cand.exists():
        print("  skip 真实 run 目录不可用")
        return
    code = cand.read_text()
    inj = R._inject_hpo_algo_at(code, "xgboost")
    check(bool(inj), "注入模块级 _HPO_ALGO")
    out, notes = R._upgrade_hpo_lineage_code(inj)
    ast.parse(out)
    check(out.count("def _fit_seed_variant") == 2, "补齐第二份 _fit_seed_variant")
    check("禁止把 eval_set 拼进训练集" in out, "嫁接产物含 hardening")
    check("concat([X_fit, eval_set[0][0]])" not in out, "嫁接产物不含旧 concat")

    print("\n[6] 幂等")
    out2, notes2 = R._upgrade_hpo_lineage_code(out)
    check(out2.count("def _fit_seed_variant") == 2, "重复嫁接不叠加定义")
    check(out2.count("禁止把 eval_set 拼进训练集")
          == out.count("禁止把 eval_set 拼进训练集"), "重复嫁接种硬化段"+"数不变")


def main() -> int:
    test_template_hardened()
    test_worker_assembly()
    test_call_site()
    test_graft_end_to_end()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED {len(FAILS)} 项：")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
