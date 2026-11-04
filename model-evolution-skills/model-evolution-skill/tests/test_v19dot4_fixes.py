"""v19.4 修复批次单元验证(耗时归因优化,依据
docs/ModelEvo_V2.10_训练探索耗时归因与优化方案_20260911.md)。

P0-1:run-generation 状态过滤(不打 FAILED)/ finalize 并行度统计;
P0-2:分组置换向量化+并行的位级一致性 / DataFrame 回退等价;
P0-3:嫁接可导入性探针 / lint 规则 14(runtime bootstrap);
P1-2:finalize-retrain / finalize-baseline-score CLI 注册;
P1-3:guard hook Bash 源码直读拦截 / CLI 正则误伤修复;
lint 规则 2 探测豁免(baseline 渲染 trial 的 _probe import)。

运行:python tests/test_v19dot4_fixes.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from _common import SKILL as REPO, cli_source  # noqa: E402

_sys.path.insert(0, str(REPO.parent))

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


    # ---------------------------------------------------------------------
    print("\n== P0-2: 分组置换向量化 + 按族并行 ==")

    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402

    from runtime.group_importance import (  # noqa: E402
        _FastPredict, group_permutation_importance, infer_families)


    class _FakeGBM:
        """确定性伪模型:score = sigmoid(列均值加权和)。
        有 predict_proba(走 DataFrame 慢路径的对照组)。"""

        def predict_proba(self, X):
            v = X.to_numpy(dtype=np.float64)
            s = 1.0 / (1.0 + np.exp(-(v @ self.w[: v.shape[1]])))
            return np.stack([1 - s, s], axis=1)

        # 对称小区间权重(防 sigmoid 饱和 → y 全 1 → AUC=NaN)
        w = np.linspace(-2, 2, 5000, dtype=np.float64)


    def _y_from_scores(s):
        return (s > np.median(s)).astype(int)


    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.random((2000, 40)).astype("float32"),
        columns=[f"{f}_{i}" for f in ("aaa", "bbb", "ccc", "ddd")
                 for i in range(10)])
    # 权重取对称区间防 sigmoid 饱和(y 全 1 → AUC=NaN);
    # 概率化标签让 AUC 对置换敏感(排序敏感)
    _W = np.linspace(-2, 2, 40)
    _score = 1.0 / (1.0 + np.exp(-(X.to_numpy() @ _W)))
    y = (rng.random(len(X)) < _score).astype(int)
    fams = infer_families(list(X.columns))
    model = _FakeGBM()


    def score_fn(Xf):
        return model.predict_proba(Xf)[:, 1]


    # 1) 位级一致:同一 seed 下 串行(1 worker) vs 并行(4 workers)
    import copy  # noqa: E402
    import os  # noqa: E402

    os.environ["MODELEVO_GI_WORKERS"] = "1"
    d1 = group_permutation_importance(
        score_fn, X, y, copy.deepcopy(fams), "auc", n_repeat=3, seed=20260814,
        model=None)  # 伪模型无 booster → DataFrame 回退路径
    os.environ["MODELEVO_GI_WORKERS"] = "4"
    d2 = group_permutation_importance(
        score_fn, X, y, copy.deepcopy(fams), "auc", n_repeat=3, seed=20260814,
        model=None)
    check("DataFrame 回退路径 1w vs 4w 位级一致(非快路径也确定性)",
          d1 is not None and d2 is not None
          and len(d1) == len(d2)
          and all(np.array_equal(d1[c].to_numpy(), d2[c].to_numpy())
                  for c in d1.columns))

    # 2) delta 确实非零(置换真的在改数据)
    check("置换产生非零 delta(语义未塌)",
          d1 is not None and float(d1["delta_mean"].abs().max()) > 1e-6)

    # 3) _FastPredict 无 booster 模型时正确回退(ok=False)
    fp = _FastPredict(score_fn, model, X)
    check("_FastPredict 非 GBM 模型 → fast 不可用(回退)",
          fp.ok is False)

    # 4) xgboost 快路径位级一致(真模型;若环境缺 xgboost 则跳过)
    try:
        import joblib  # noqa: E402
        import xgboost as xgb  # noqa: E402

        R = REPO.parent / ('example-seq-v3-4-2.8.6-v19.2')
        tdir = R / 'trials' / 'trial-004-g02-xgboost-1821'
        Xr = pd.read_parquet(tdir / 'model' / 'validation_input.parquet')
        mr = joblib.load(tdir / 'model' / 'model.joblib')

        def sfn(Xf):
            return mr.predict_proba(Xf)[:, 1]

        fpr = _FastPredict(sfn, mr, Xr)
        check("xgboost _FastPredict 探测通过(与 predict_proba 位级一致)",
              fpr.ok is True
              and np.array_equal(fpr._fast_predict(fpr.V), sfn(Xr)))
        # 抽 2 族 × 2 repeat 验证置换路径位级一致
        fr = infer_families(list(Xr.columns))
        sel = {k: fr[k] for k in list(fr)[:2]}
        _yr = _y_from_scores(sfn(Xr))  # 真模型分数排序造标签(二值化)
        os.environ["MODELEVO_GI_WORKERS"] = "1"
        da = group_permutation_importance(
            sfn, Xr, _yr, sel, "auc",
            n_repeat=2, seed=7, model=mr)
        db = group_permutation_importance(
            sfn, Xr, _yr, copy.deepcopy(sel), "auc",
            n_repeat=2, seed=7, model=None)  # DataFrame 路径对照
        check("xgboost 快路径 vs DataFrame 路径 delta 位级一致",
              da is not None and db is not None and np.array_equal(
                  da["delta_mean"].to_numpy(), db["delta_mean"].to_numpy()))
    except ImportError:
        print("  [SKIP] xgboost 真模型位级一致(环境缺库)")

    # ---------------------------------------------------------------------
    print("\n== P0-1: run-generation 状态过滤 / 并行度统计 ==")

    # 状态过滤逻辑在 cmd_run_generation 内联;用语法级断言 + 行为可测部分:
    cli_src = cli_source()
    check("run-generation 过滤 PROPOSED(源码含过滤逻辑)",
          't.status != "PROPOSED"' in cli_src
          and 'PROPOSED trial(代码未落盘)被跳过' in cli_src)
    check("run-generation --wait-code 已注册",
          '"--wait-code"' in cli_src)

    # finalize parallelism:用纯逻辑复算验证口径(首终态,不含重评污染)
    import datetime as dt  # noqa: E402

    from runtime.trial import TrialFactory  # noqa: E402
    from runtime.config import Config  # noqa: E402
    from runtime.spec import TaskSpec  # noqa: E402

    R = REPO.parent / 'example-seq-v3-4-2.8.6-v19.2'
    spec = TaskSpec.load(R / 'task-spec.yaml')
    cfg = Config.load(run_config=None)
    fac = TrialFactory(R, spec=spec, config=cfg)


    def _parallelism(gid):
        spans = []
        for t in fac.all_trials():
            if t.generation != gid:
                continue
            hist = t.data.get('status_history') or []
            t0 = next((h['at'] for h in hist if h.get('state') == 'RUNNING'),
                      None)
            t1 = next((h['at'] for h in hist if h.get('state') in
                       ('EVALUATED', 'FAILED', 'TIMEOUT')), None)
            if t0 and t1:
                d0, d1 = dt.datetime.fromisoformat(t0), \
                    dt.datetime.fromisoformat(t1)
                spans.append((d0, d1))
        if len(spans) < 2:
            return None
        wall = (max(d1 for _, d1 in spans)
                - min(d0 for d0, _ in spans)).total_seconds()
        tot = sum((d1 - d0).total_seconds() for d0, d1 in spans)
        return tot / wall


    check("gen-02(实测串行代)avg_parallelism ≈ 1",
          0.8 <= _parallelism(2) <= 1.2)
    check("gen-01(单 trial 代)不计入(返回 None)",
          _parallelism(1) is None)

    ev_src = (REPO / 'runtime' / 'evolution.py').read_text(encoding='utf-8')
    check("finalize 含 parallelism 统计与 <1.3 告警",
          'avg_parallelism' in ev_src and '< 1.3' in ev_src)
    check("并行度用首终态(防 backfill 重评污染)",
          '# 首个终态(原始终态)' in ev_src)

    runner_src = (REPO / 'runtime' / 'runner.py').read_text(encoding='utf-8')
    check("手动并行路径 GPU slot 错开(实测撞卡修复)",
          '手动并行 slot=' in runner_src
          and 'def _count_running_trials(self) -> tuple[int, int]:' in runner_src)

    # ---------------------------------------------------------------------
    print("\n== P0-3: 嫁接可导入性探针 / lint 规则 14 ==")

    check("嫁接尾部含可导入性探针",
          '可导入性探针' in cli_src and 'PROBE_FAIL' in cli_src)
    check("探针 cwd=code/(复刻 runner 环境)",
          'cwd=str(Path(trial.trial_dir) / "code")' in cli_src)

    import importlib.util as _ilu  # noqa: E402

    _lspec = _ilu.spec_from_file_location('lt_v194', REPO / 'assets' /
                                          'lint_trial.py')
    LT = _ilu.module_from_spec(_lspec)
    _lspec.loader.exec_module(LT)

    check("规则 14:无 runtime import 放行",
          LT.check_runtime_import_bootstrap('import numpy\n') == [])
    check("规则 14:runtime import + bootstrap 放行",
          LT.check_runtime_import_bootstrap(
              "import sys\nSKILL_ROOT='/x'\n"
              "if SKILL_ROOT not in sys.path:\n"
              "    sys.path.insert(0, SKILL_ROOT)\n"
              "from runtime.hpo import build_gbm\n") == [])
    r14 = LT.check_runtime_import_bootstrap(
        'from runtime.hpo import build_gbm\n')
    check("规则 14:runtime import 无 bootstrap 拒绝", bool(r14))
    check("规则 14:诊断豁免标记放行",
          LT.check_runtime_import_bootstrap(
              'from runtime.hpo import x\n'
              '# lint:skip-runtime-bootstrap\n') == [])

    # 规则 2 探测豁免(误伤修复)
    code_probe = (
        "import xgboost as xgb\n"
        "def _lgbm_cuda_ok():\n"
        "    try:\n"
        "        import lightgbm as _lgb_probe\n"
        "        _lgb_probe.LGBMClassifier().fit(None, None)\n"
        "    except Exception:\n"
        "        pass\n")
    check("规则 2:try 块内 _probe 后缀 import 豁免(baseline 渲染形态)",
          LT.check_no_ensemble(code_probe) == [])
    check("规则 2:模块级双库仍拒绝",
          any('多个 GBM 库' in e for e in LT.check_no_ensemble(
              'import lightgbm\nimport xgboost\n')))

    # ---------------------------------------------------------------------
    print("\n== P1-2: finalize-retrain / finalize-baseline-score CLI ==")

    check("finalize-retrain 已注册并实现",
          'def cmd_finalize_retrain' in cli_src
          and '"finalize-retrain"' in cli_src)
    check("finalize-baseline-score 已注册并实现",
          'def cmd_finalize_baseline_score' in cli_src
          and '"finalize-baseline-score"' in cli_src)
    check("finalize-retrain 默认确定性选型(可 --champion 覆写)",
          '"--champion"' in cli_src
          # v2.11.9 订正:旧断言写死 `lb.best_trial(trials)`,而代码早已
          # 演进为 `lb.best_trial(trials, shadow_recheck=True)`(v2.11.7
          # 起与 evaluate-final 的 gate 口径对齐),字符串断言随之过期。
          # 断言"存在以 trials 为入参的确定性选型调用"即可,不锁定 kwarg。
          and re.search(r"lb\.best_trial\(\s*trials", cli_src) is not None)

    # ---------------------------------------------------------------------
    print("\n== P1-3: guard hook Bash 拦截 / CLI 正则收紧 ==")

    import json  # noqa: E402
    import subprocess  # noqa: E402

    GPY = REPO / 'assets' / 'guard_main_session.py'


    def _hook(tool, payload, sub=False):
        inp = json.dumps({
            'tool_name': tool, 'tool_input': payload,
            'transcript_path': '/tmp/fake.jsonl'})
        env = dict(os.environ)
        env['CLAUDE_PROJECT_DIR'] = '/tmp/fakeproject'
        env.pop('MODELEVO_ALLOW_SOURCE_READ', None)
        env.pop('MODELEVO_ALLOW_MAIN_EXEC', None)
        r = subprocess.run([sys.executable, str(GPY)], input=inp,
                           capture_output=True, text=True, env=env, timeout=60)
        return r.returncode


    check("guard:Bash sed 直读 harness 源码被拦",
          _hook('Bash', {'command': "sed -n '1,40p' .claude/skills/"
                                    "modelevo/runtime/runner.py"}) == 2)
    check("guard:grep -n 直读被拦",
          _hook('Bash', {'command': 'grep -n "def x" .claude/skills/'
                                    'modelevo/scripts/modelevo.py'}) == 2)
    check("guard:合法 CLI(status)+ tail 不误伤",
          _hook('Bash', {'command': 'python .claude/skills/modelevo/scripts/'
                                    'modelevo.py status 2>&1 | tail -40'}) == 0)
    check("guard:日志内 grep 'modelevo.py run' 字符串不误伤(正则收紧)",
          _hook('Bash', {'command': "cat x.log | grep 'modelevo.py run "
                                    "--trial' | head"}) == 0)
    check("guard:工作区 tools/ 副本不拦",
          _hook('Bash', {'command': 'cat tools/lint_trial.py | head -5'}) == 0)
    check("guard:MODELEVO_ALLOW_SOURCE_READ=1 豁免",
          subprocess.run(
              [sys.executable, str(GPY)],
              input=json.dumps({'tool_name': 'Bash', 'tool_input': {
                  'command': 'head -5 x/runtime/runner.py'}}),
              capture_output=True, text=True, timeout=60,
              env={**os.environ,
                   'CLAUDE_PROJECT_DIR': '/tmp/fakeproject',
                   'MODELEVO_ALLOW_SOURCE_READ': '1'}).returncode == 0)

    # ---------------------------------------------------------------------
    print("\n== steering 文档同步 ==")

    digest = (REPO / 'references' / 'protocol-digest.md').read_text(
        encoding='utf-8')
    skill = (REPO / 'SKILL.md').read_text(encoding='utf-8')
    check("digest:run-generation 反转为默认路径",
          '反转为' in digest and '默认路径' in digest)
    check("digest:签名表含 finalize-retrain/--wait-code",
          '`finalize-retrain`' in digest and '--wait-code' in digest)
    check("SKILL.md:批量并行优先 + 写码纪律(禁微调循环)",
          '批量并行优先' in skill and '禁微调循环' in skill)
    check("SKILL.md:收尾用 finalize-retrain(禁手写脚本)",
          '不写脚本' in skill)

    print()
    print(f"== v19.4 测试结果: {ok} pass / {fail} fail ==")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
