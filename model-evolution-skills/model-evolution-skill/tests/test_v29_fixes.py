"""v2.9 修复批次单元验证(new-trial 事务化 / hpo 模板渲染 / delete-trial)。

背景(2026-09-03 drawdown-t7 gen-03 事故回写):hpo 快通道强制
--from-parent 且父代码必须自带四件套,而 run 内父代全是 baseline
血缘 + 手写 widen → subagent 被迫建 scaffold + 手改 7 处代码,
8 次创建/2 次契约外 rm -rf(孤儿 trial-005/006 伪装成半成品被误判
"设计缺陷",trial-007 被删后 trial-008 lineage 断链)。本批次:

1. new-trial 失败事务化:任何创建期失败(SystemExit / 模板渲染异常)
   回滚本次 PROPOSED 目录,消灭孤儿;失败输出显式提示"已回滚";
2. hpo 两条路径:无 --from-parent 时按 --algorithm 渲染(四件套经
   runtime.hpo.build_gbm 分发,支持 lightgbm/xgboost/catboost),
   特征面经 --hpo-parent 继承(加宽列白名单单源搬运,不经派生面
   交集校验);--from-parent 复制路径保留 _assert_hpo_capable;
3. delete-trial 子命令:仅 PROPOSED、产物存在拒绝(--force 豁免)、
   被现存 trial 声明为父代时拒绝,删除留痕 delete-log.jsonl;
4. 占位词黑名单扩展:reserved/cleanup-check/scaffold/占位 拦截在
   创建口(旧词表漏拦 "RESERVED trial-010" 型占位)。

运行:python tests/test_v29_fixes.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import yaml

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _load_cli():
    """v2.11.9:CLI 已拆成 `modelevo.py` + `_mvo/` 包。

    `spec_from_file_location` 单文件装载只执行入口,`_mvo` 包会以顶层
    模块被导入 → 入口里 `from _mvo.cmd_* import …` 的再导出**不会**
    落到本模块对象上(实测 TrialFactory/Config 取不到)。这里装完补
    别名,与运行时 CLI 的属性面一致。
    """
    spec = importlib.util.spec_from_file_location(
        "modelevo_cli_v29", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    from runtime.config import Config
    from runtime.trial import TrialFactory
    cli.TrialFactory = TrialFactory
    cli.Config = Config
    return cli


def _make_run(td: Path, *, with_splits: bool = True) -> Path:
    """最小 run 工作区:task-spec + run-state + 空 trials/。"""
    (td / "trials").mkdir(parents=True)
    (td / "leaderboard").mkdir()
    (td / "task-spec.yaml").write_text(yaml.safe_dump({
        "task_id": "v29-test",
        "task_name": "v2.9 修复单测",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "y=1"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based",
                         "train": "2025-01~2025-06",
                         "validation": "2025-07~2025-08",
                         "final_test": "2025-09~2025-09"},
    }), encoding="utf-8")
    (td / "run-state.json").write_text(json.dumps({
        "phase": "EVOLVING", "harness_epoch": 0,
        "harness_hash": "fake-hash",
    }), encoding="utf-8")
    if with_splits:
        # 最小 splits(train.parquet 两列足够 resolve_feature_columns)
        import pandas as pd

        sp = td / "sample-profile" / "splits"
        sp.mkdir(parents=True)
        pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [1, 2, 3]}).to_parquet(
            sp / "train.parquet")
        pd.DataFrame({"f1": [0.4], "f2": [4]}).to_parquet(
            sp / "validation.parquet")
        (sp / "labels").mkdir()
        pd.DataFrame({"label": [0, 1, 0]}).to_parquet(sp / "labels" /
                                                      "train.parquet")
        pd.DataFrame({"label": [1]}).to_parquet(sp / "labels" /
                                                "validation.parquet")
    return td


def _make_space(td: Path) -> str:
    sp = td / "space.yaml"
    sp.write_text(yaml.safe_dump({
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.2,
                          "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 8},
    }), encoding="utf-8")
    return str(sp)


def _args(ns_dict):
    """argparse.Namespace 等价物(模拟 CLI parse 后的 args)。"""
    class _A:
        pass
    a = _A()
    for k, v in ns_dict.items():
        setattr(a, k, v)
    a.command = getattr(a, "command", "new-trial")
    return a


# ---------------------------------------------------------------------------
# 1. new-trial 失败事务化(hpo 复制路径父代码缺四件套 → 回滚孤儿)
# ---------------------------------------------------------------------------

def test_new_trial_rollback() -> None:
    print("== new-trial 失败事务化 ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run = _make_run(td)
        space = _make_space(td)
        # 先建一个 baseline 血缘父代(无 hpo 四件套)
        t0 = cli.TrialFactory(run, spec=None, config=cli.Config.load())
        parent = t0.create(algorithm="xgboost",
                           hypothesis={"statement": "baseline 血缘父代,无 hpo 分支",
                                       "rationale": "单测", "falsification": "n/a"},
                           execution={"action_description": ["单测"]},
                           search_action="explore", generation=1)
        (parent.trial_dir / "code" / "train.py").write_text(
            "import pandas\n", encoding="utf-8")
        (parent.trial_dir / "config" / "trial.yaml").write_text(
            yaml.safe_dump({"trial_id": parent.trial_id,
                            "hyperparameters": {},
                            "data_view": "v1"}), encoding="utf-8")
        args = _args({
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "algorithm": "xgboost",
            "from_parent": parent.trial_id,
            "set": None, "hypothesis": "hpo 复制路径资格不足",
            "rationale": "r", "falsification": "f", "actions": "hpo",
            "action": "hpo", "search_space": space,
            "budget_configs": 5, "force_hpo_budget": False,
            "parents": None, "changed_modules": "model",
            "override_feature_phase": "单测豁免", "family_hints": None,
            "generation": 2, "data_view": "v1", "heavy": False,
            "features": None,
        })
        # 期望:v19 起复制路径资格不足**自动嫁接**模板 hpo 段(不再是
        # SystemExit 拒绝)——gen-03 实测"冠军加宽面 + HPO"两条路都
        # 到不了的事故回写。本用例改为验证:创建成功、code 同时含父
        # 标记与四件套、嫁接段标注。
        rolled_back = None
        cli.cmd_new_trial(args)
        children = [p for p in (run / "trials").iterdir()
                    if p.name != parent.trial_id]
        check("hpo 复制路径资格不足 → 自动嫁接并创建成功",
              len(children) == 1,
              f"children={[p.name for p in children]}")
        if children:
            _code = (children[0] / "code" / "train.py").read_text()
            check("嫁接后 code 含 hpo 四件套",
                  all(s in _code for s in
                      ("fit_and_score", "_hpo_fit_inner",
                       "_INNER_METRIC_UP", "dump_json")),
                  "缺四件套")
            check("嫁接段留痕(v19 hpo 嫁接段)",
                  "v19 hpo 嫁接段" in _code, "无嫁接标注")
        # 编号无泄漏:下一编号 = 现有 max+1(嫁接成功路径不再触发回滚
        # 分支;保留原连续性断言)
        t2 = t0.create(algorithm="xgboost",
                       hypothesis={"statement": "回滚后编号连续性验证",
                                   "rationale": "r", "falsification": "f"},
                       execution={"action_description": ["单测"]},
                       search_action="explore", generation=1)
        import re
        m0 = re.match(r"trial-(\d+)", parent.trial_id)
        m2 = re.match(r"trial-(\d+)", t2.trial_id)
        # v19:嫁接成功路径不再回滚删除,hpo 子代占用了 parent+1 的
        # 编号 → 下一编号 = parent+2(连续性语义从"回滚复用"变为
        # "无空洞跳号")
        check("编号连续(无空洞,嫁接路径占用 parent+1)",
              int(m2.group(1)) == int(m0.group(1)) + 2,
              f"parent={parent.trial_id} next={t2.trial_id}")


# ---------------------------------------------------------------------------
# 2. hpo 模板渲染路径(无 --from-parent,--algorithm + --hpo-parent)
# ---------------------------------------------------------------------------

def test_hpo_render_path() -> None:
    print("== hpo 模板渲染路径 ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run = _make_run(td)
        space = _make_space(td)
        # 特征面父代:config 带 features.columns(模拟加宽列)
        factory = cli.TrialFactory(run, spec=None, config=cli.Config.load())
        widen_parent = factory.create(
            algorithm="xgboost",
            hypothesis={"statement": "手写加宽父代(模拟 1703 列面)",
                        "rationale": "r", "falsification": "f"},
            execution={"action_description": ["单测"]},
            search_action="data_view", generation=1)
        (widen_parent.trial_dir / "config").mkdir(exist_ok=True)
        (widen_parent.trial_dir / "config" / "trial.yaml").write_text(
            yaml.safe_dump({"features": {"exclude": ["f1"],
                                         "columns": ["f2", "widen_a",
                                                     "widen_b"]}}),
            encoding="utf-8")
        args = _args({
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "algorithm": "xgboost", "from_parent": None,
            "hpo_parent": widen_parent.trial_id,
            "set": None, "hypothesis": "模板渲染路径 hpo,继承加宽面",
            "rationale": "r", "falsification": "f", "actions": "hpo",
            "action": "hpo", "search_space": space,
            "budget_configs": 5, "force_hpo_budget": False,
            "parents": None, "changed_modules": "model",
            "override_feature_phase": "单测豁免", "family_hints": None,
            "generation": 2, "data_view": "v1", "heavy": False,
            "features": None,
        })
        try:
            rc = cli.cmd_new_trial(args)
            ok = (rc == 0)
        except SystemExit as e:
            ok = False
            rc = e.code
        check("hpo 模板渲染路径创建成功", ok, f"rc={rc}")
        new_dirs = [p for p in (run / "trials").iterdir()
                    if p.name != widen_parent.trial_id]
        check("恰好创建 1 个 trial", len(new_dirs) == 1,
              f"实际 {len(new_dirs)}")
        if len(new_dirs) == 1:
            tdir = new_dirs[0]
            code = (tdir / "code" / "train.py").read_text(encoding="utf-8")
            cfg = yaml.safe_load((tdir / "config" / "trial.yaml")
                                 .read_text(encoding="utf-8"))
            check("渲染代码注入 _HPO_ALGO=xgboost",
                  "_HPO_ALGO = 'xgboost'" in code)
            check("渲染代码四件套在位(_hpo_fit_inner)",
                  "_hpo_fit_inner" in code)
            check("config.hpo 节注入(search_space/budget)",
                  isinstance(cfg.get("hpo"), dict)
                  and "search_space" in cfg["hpo"])
            check("特征面继承加宽列白名单(不经派生面交集校验)",
                  set(cfg.get("features", {}).get("columns") or [])
                  >= {"f2", "widen_a", "widen_b"},
                  f"columns={cfg.get('features', {}).get('columns')}")
            check("algorithm 归一化为 xgboost",
                  (yaml.safe_load((tdir / "trial.yaml")
                                  .read_text(encoding="utf-8"))
                   .get("algorithm")) == "xgboost")
        # 不支持的算法 → 拒绝且回滚
        args2 = _args({
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "algorithm": "torch_mlp", "from_parent": None,
            "hpo_parent": None, "set": None,
            "hypothesis": "不支持算法应被拒绝",
            "rationale": "r", "falsification": "f", "actions": "hpo",
            "action": "hpo", "search_space": space,
            "budget_configs": 5, "force_hpo_budget": False,
            "parents": None, "changed_modules": "model",
            "override_feature_phase": "单测豁免", "family_hints": None,
            "generation": 2, "data_view": "v1", "heavy": False,
            "features": None,
        })
        n_before = len(list((run / "trials").iterdir()))
        try:
            rc2 = cli.cmd_new_trial(args2)
            rejected = (rc2 == 1)
        except SystemExit:
            rejected = True
        n_after = len(list((run / "trials").iterdir()))
        check("不支持算法 → 拒绝(rc=1)", rejected)
        check("不支持算法 → 目录已回滚", n_after == n_before,
              f"before={n_before} after={n_after}")


# ---------------------------------------------------------------------------
# 3. delete-trial:PROPOSED 可删 / 有产物拒绝 / 被引用拒绝 / 留痕
# ---------------------------------------------------------------------------

def test_delete_trial() -> None:
    print("== delete-trial ==")
    cli = _load_cli()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        run = _make_run(td, with_splits=False)
        factory = cli.TrialFactory(run, spec=None, config=cli.Config.load())
        # case A: 纯脚手架 PROPOSED → 可删
        a = factory.create(algorithm="lightgbm",
                           hypothesis={"statement": "误建的纯脚手架 trial",
                                       "rationale": "r", "falsification": "f"},
                           execution={"action_description": ["单测"]},
                           search_action="explore", generation=1)
        args = _args({"spec": str(run / "task-spec.yaml"), "run": str(run),
                      "trial": a.trial_id, "reason": "单测清理",
                      "force": False})
        rc = cli.cmd_delete_trial(args)
        check("PROPOSED 纯脚手架 → 删除成功", rc == 0)
        check("目录已删除", not a.trial_dir.exists())
        log = run / "delete-log.jsonl"
        check("留痕 delete-log.jsonl", log.exists())
        if log.exists():
            rec = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
            check("留痕含 trial_id 与 reason",
                  rec.get("trial_id") == a.trial_id
                  and rec.get("reason") == "单测清理")
        # case B: 有产物(validation_score)→ 拒绝;--force 豁免
        b = factory.create(algorithm="lightgbm",
                           hypothesis={"statement": "带产物的 PROPOSED 异常态",
                                       "rationale": "r", "falsification": "f"},
                           execution={"action_description": ["单测"]},
                           search_action="explore", generation=1)
        (b.trial_dir / "validation_score.parquet").write_bytes(b"x")
        rc = cli.cmd_delete_trial(_args({
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "trial": b.trial_id, "reason": "r2", "force": False}))
        check("有产物 → 拒绝", rc == 1 and b.trial_dir.exists())
        rc = cli.cmd_delete_trial(_args({
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "trial": b.trial_id, "reason": "r3", "force": True}))
        check("有产物 --force → 删除成功", rc == 0
              and not b.trial_dir.exists())
        # case C: 被现存 trial 声明为父代 → 拒绝(force 也不可豁免)
        c = factory.create(algorithm="lightgbm",
                           hypothesis={"statement": "被引用的父代 trial",
                                       "rationale": "r", "falsification": "f"},
                           execution={"action_description": ["单测"]},
                           search_action="explore", generation=1)
        d = factory.create(algorithm="lightgbm",
                           hypothesis={"statement": "子代引用 c 为父代",
                                       "rationale": "r", "falsification": "f"},
                           execution={"action_description": ["单测"]},
                           search_action="explore",
                           parent_trial_ids=[c.trial_id], generation=2)
        rc = cli.cmd_delete_trial(_args({
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "trial": c.trial_id, "reason": "r4", "force": True}))
        check("被现存子代引用 → 拒绝(force 不豁免)",
              rc == 1 and c.trial_dir.exists())
        # case D: 非 PROPOSED → 拒绝
        d.data["status"] = "EVALUATED"
        d.flush()
        rc = cli.cmd_delete_trial(_args({
            "spec": str(run / "task-spec.yaml"), "run": str(run),
            "trial": d.trial_id, "reason": "r5", "force": False}))
        check("EVALUATED 状态 → 拒绝", rc == 1 and d.trial_dir.exists())


# ---------------------------------------------------------------------------
# 4. 占位词黑名单扩展
# ---------------------------------------------------------------------------

def test_placeholder_blacklist() -> None:
    print("== 占位词黑名单扩展 ==")
    from runtime.trial import TrialFactory

    cli = _load_cli()
    with tempfile.TemporaryDirectory() as td:
        run = _make_run(Path(td), with_splits=False)
        factory = TrialFactory(run, spec=None, config=cli.Config.load())
        for bad in ("RESERVED trial-010: HPO on v1 full",
                    "RESERVED-CLEANUP: re-create from proper parent",
                    "scaffold parent creation (hpo machinery carrier)",
                    "占位 trial"):
            try:
                factory.create(algorithm="lightgbm",
                               hypothesis={"statement": bad,
                                           "rationale": "r",
                                           "falsification": "f"},
                               execution={"action_description": ["x"]},
                               search_action="explore", generation=1)
                caught = False
            except ValueError:
                caught = True
            check(f"占位拦截: {bad[:36]!r}", caught)
        # 合法 hypothesis 不受影响
        try:
            factory.create(algorithm="lightgbm",
                           hypothesis={"statement":
                                       "xgboost HPO inner 30 configs 搜索",
                                       "rationale": "r", "falsification": "f"},
                           execution={"action_description": ["x"]},
                           search_action="explore", generation=1)
            ok = True
        except ValueError:
            ok = False
        check("合法 hypothesis 不误伤", ok)


def main() -> int:
    test_new_trial_rollback()
    test_hpo_render_path()
    test_delete_trial()
    test_placeholder_blacklist()
    print(f"\n{'=' * 50}\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
