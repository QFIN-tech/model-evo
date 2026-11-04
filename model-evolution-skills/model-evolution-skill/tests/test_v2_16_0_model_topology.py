"""v2.16.0 模型拓扑门:trial 默认单模型,级联/融合须人工批准。

用户裁决(2026-10-22):默认每个 trial 只有单一模型,不支持"子分作特征"的级联
与模型融合,突破需人介入确认;hurdle(回归)与 T-learner(uplift)归为
structural,按任务类型白名单放行。

本文件锁定(全部离线):

  A. 静态检测:单模型 / 级联(特征、目标、跨函数、特征管线内训练)/ 融合
     (加权、均值、循环累积、容器)/ structural(hurdle、T-learner)/ 死分支
     折叠 / 诊断差分与 HPO 循环不误报;
  B. 现行模板与全局库配方:二分类/回归/多分类模板单模型,uplift 模板
     T-learner,回归库 hurdle,库内无级联;
  C. 门与白名单:hurdle 在回归放行、在二分类拦截;声明只增不减;
  D. 批准账本:recipe 覆盖后代、签名越界重批、trial 范围绑 sha、run 范围按
     种类、撤销、哈希链篡改即全部作废;
  E. runner 接线:被拦 trial 不冻结、不迁移状态、不记失败,留痕请求与
     user-decisions;
  F. CLI:approve-topology 非 TTY 拒绝;guard hook 拒绝 Agent 批准/写账本。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL, tmp_run_dir  # noqa: E402

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

from runtime.config import Config  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.topology import (analyze_code, append_ledger,  # noqa: E402
                              evaluate_gate, ledger_path, record_request)
from runtime.trial import TrialRecord  # noqa: E402

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


_HEAD = ("import numpy as np\nimport lightgbm as lgb\nimport xgboost as xgb\n"
         "def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,"
         " w_tr=None, aux=None):\n")

SINGLE = _HEAD + """    m = lgb.LGBMClassifier()
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    return m.predict_proba(X_va)[:, 1], {}
"""
CASCADE = _HEAD + """    a = lgb.LGBMClassifier().fit(X_tr, y_tr)
    X_tr = X_tr.assign(sub=a.predict_proba(X_tr)[:, 1])
    X_va = X_va.assign(sub=a.predict_proba(X_va)[:, 1])
    b = xgb.XGBClassifier()
    b.fit(X_tr, y_tr)
    return b.predict_proba(X_va)[:, 1], {}
"""
CASCADE_AUX = _HEAD + """    aux_m = lgb.LGBMRegressor().fit(X_tr, aux)
    X_tr["aux_score"] = aux_m.predict(X_tr)
    m = lgb.LGBMClassifier()
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_va)[:, 1], {}
"""
CASCADE_TARGET = _HEAD + """    t = lgb.LGBMClassifier().fit(X_tr, y_tr)
    s = lgb.LGBMRegressor().fit(X_tr, t.predict_proba(X_tr)[:, 1])
    return s.predict(X_va), {}
"""
CASCADE_INTERPROC = """import lightgbm as lgb
def _sub(X, y, Xo):
    m = lgb.LGBMClassifier().fit(X, y)
    return m.predict_proba(Xo)[:, 1]
def _final(X, y):
    m = lgb.LGBMClassifier()
    m.fit(X, y)
    return m
def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    f = _sub(X_tr, y_tr, X_tr)
    X2 = X_tr.copy(); X2["f"] = f
    m = _final(X2, y_tr)
    return m.predict_proba(X_va)[:, 1], {}
"""
CASCADE_DL_WIDEN = """import torch
def _dl_fit(x, y):
    net = torch.nn.Linear(3, 1)
    loss = net(x).sum(); loss.backward()
    return net
def _dl_oof(x, y):
    return _dl_fit(x, y)(x).detach().numpy()
def _widen(raw_tr, re_tr, raw_va, re_va):
    return _dl_oof(raw_tr, re_tr), None, {}, {}
"""
FEATURE_SELECT = """import lightgbm as lgb
def _widen(raw_tr, re_tr, raw_va, re_va):
    m = lgb.LGBMClassifier().fit(raw_tr, re_tr)
    keep = [c for c, g in zip(raw_tr.columns, m.feature_importances_) if g]
    return raw_tr[keep], raw_va[keep], {}, {}
"""
ENSEMBLE_WEIGHTED = _HEAD + """    a = lgb.LGBMClassifier().fit(X_tr, y_tr)
    b = xgb.XGBClassifier().fit(X_tr, y_tr)
    s = 0.7 * a.predict_proba(X_va)[:, 1] + 0.3 * b.predict_proba(X_va)[:, 1]
    return s, {}
"""
ENSEMBLE_SEEDS = _HEAD + """    preds = []
    for sd in range(3):
        m = lgb.LGBMClassifier(random_state=sd).fit(X_tr, y_tr)
        preds.append(m.predict_proba(X_va)[:, 1])
    return np.mean(preds, axis=0), {}
"""
ENSEMBLE_ACCUM = _HEAD + """    s = 0
    for k in range(5):
        m = lgb.LGBMClassifier(random_state=k).fit(X_tr, y_tr)
        s += m.predict_proba(X_va)[:, 1] / 5
    return s, {}
"""
ENSEMBLE_COMP = """import numpy as np
class Bag:
    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)
"""
ENSEMBLE_CONTAINER = """from sklearn.ensemble import StackingClassifier
def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    m = StackingClassifier([]).fit(X_tr, y_tr)
    return m.predict_proba(X_va)[:, 1], {}
"""
HURDLE = """import xgboost as xgb
class Hurdle:
    def __init__(self, c, r): self.clf_, self.reg_ = c, r
    def predict(self, X):
        return self.clf_.predict_proba(X)[:, 1] * self.reg_.predict(X)
def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    c = xgb.XGBClassifier(); c.fit(X_tr, y_tr > 0)
    r = xgb.XGBRegressor(); r.fit(X_tr[y_tr > 0], y_tr[y_tr > 0])
    return Hurdle(c, r).predict(X_va), {}
"""
T_LEARNER = """import lightgbm as _lgb
MODEL_TYPE = '{mt}'
def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None, w_tr=None):
    if MODEL_TYPE == "uplift":
        _m_t = _lgb.LGBMClassifier().fit(X_tr[w_tr == 1], y_tr[w_tr == 1])
        _m_c = _lgb.LGBMClassifier().fit(X_tr[w_tr == 0], y_tr[w_tr == 0])
        score = (_m_t.predict_proba(X_va)[:, 1]
                 - _m_c.predict_proba(X_va)[:, 1])
    else:
        m = _lgb.LGBMClassifier().fit(X_tr, y_tr)
        score = m.predict_proba(X_va)[:, 1]
    return score, {{}}
"""
S_LEARNER = """import lightgbm as lgb
def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None):
    m = lgb.LGBMClassifier().fit(X_tr, y_tr)
    return (m.predict_proba(X_va.assign(w=1))[:, 1]
            - m.predict_proba(X_va.assign(w=0))[:, 1]), {}
"""
DIAG = _HEAD + """    from sklearn.metrics import roc_auc_score
    m = lgb.LGBMClassifier().fit(X_tr, y_tr)
    s = m.predict_proba(X_va)[:, 1]
    gap = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1]) - roc_auc_score(y_va, s)
    return s - s.mean(), {"gap": gap}
"""
HPO_LOOP = _HEAD + """    best, best_auc = None, -1
    for p in [7, 15, 31]:
        m = lgb.LGBMClassifier(num_leaves=p).fit(X_tr[:100], y_tr[:100])
        a = roc_auc(y_tr[100:], m.predict_proba(X_tr[100:])[:, 1])
        if a > best_auc:
            best, best_auc = p, a
    m = lgb.LGBMClassifier(num_leaves=best).fit(X_tr, y_tr)
    return m.predict_proba(X_va)[:, 1], {}
"""


def test_a_detector() -> None:
    print("== A. 静态检测 ==")
    cases = [
        ("单模型", SINGLE, "single", None),
        ("级联:子分作特征", CASCADE, "cascade", "prediction_as_feature"),
        ("级联:辅助标签子分", CASCADE_AUX, "cascade", "prediction_as_feature"),
        ("级联:预测作目标(蒸馏)", CASCADE_TARGET, "cascade",
         "prediction_as_target"),
        ("级联:跨函数", CASCADE_INTERPROC, "cascade", "prediction_as_feature"),
        ("级联:_widen 内训练 DL(seqtab_logit 形态)", CASCADE_DL_WIDEN,
         "cascade", "model_trained_in_feature_pipeline"),
        ("融合:加权和", ENSEMBLE_WEIGHTED, "ensemble", "combine_add"),
        ("融合:多 seed 均值", ENSEMBLE_SEEDS, "ensemble", "combine_aggregate"),
        ("融合:循环累积", ENSEMBLE_ACCUM, "ensemble", "combine_accumulate"),
        ("融合:成员推导式均值", ENSEMBLE_COMP, "ensemble",
         "combine_aggregate"),
        ("融合:sklearn 容器", ENSEMBLE_CONTAINER, "ensemble",
         "ensemble_container"),
        ("structural:hurdle", HURDLE, "structural", "hurdle"),
        ("structural:T-learner(uplift)", T_LEARNER.format(mt="uplift"),
         "structural", "t_learner"),
        ("死分支:二分类模板里的 T-learner 分支",
         T_LEARNER.format(mt="binary_classification"), "single", None),
        ("S-learner(同一模型两次打分)", S_LEARNER, "single", None),
        ("诊断差分/去均值不算组合", DIAG, "single", None),
        ("HPO 选择循环不算融合", HPO_LOOP, "single", None),
        ("特征选择(只 fit 不预测)", FEATURE_SELECT, "single", None),
    ]
    for name, code, kind, rule in cases:
        r = analyze_code(code)
        got_rules = {s["rule"] for s in r["signals"]}
        ok = (kind in r["kind"].split("+") if kind != "single"
              else r["kind"] == "single")
        if rule:
            ok = ok and rule in got_rules
        check(name, ok, f"kind={r['kind']} rules={sorted(got_rules)}")
    r = analyze_code(FEATURE_SELECT)
    check("特征选择留下提示 note", any(
        n["rule"] == "model_fit_in_feature_pipeline" for n in r["notes"]))
    r = analyze_code("def f(:\n")
    check("语法错误不抛异常", r["parse_error"] is not None)


def test_b_templates_and_library() -> None:
    print("== B. 现行模板与全局库 ==")
    from _mvo.templates import TRIAL_BODY
    from runtime.templates import TRIAL_TEMPLATE

    for mt, expect in (("binary_classification", "single"),
                       ("regression", "single"), ("multiclass", "single"),
                       ("uplift", "structural")):
        code = TRIAL_TEMPLATE.format(
            trial_id="t", hypothesis="h", actions="a", run_dir="/tmp/r",
            target="y", model_type=mt, treatment="w", seed=1,
            skill_root="/tmp", body=TRIAL_BODY)
        r = analyze_code(code)
        check(f"模板 {mt} → {expect}", r["kind"] == expect,
              f"kind={r['kind']} {r['signals'][:2]}")
    lib = SKILL / "memory" / "top-trials"
    files = sorted(lib.glob("*/rank*/train.py")) + sorted(
        lib.glob("*/baseline/*/train.py"))
    if not files:
        check("全局库配方(不在环境,跳过)", True)
        return
    kinds = {}
    for f in files:
        kinds.setdefault(analyze_code(f.read_text(encoding="utf-8"))["kind"],
                         []).append(f)
    check("全局库无级联/融合配方",
          not any(k for k in kinds if k not in ("single", "structural")),
          f"{ {k: len(v) for k, v in kinds.items()} }")
    check("库内 structural 全是回归 hurdle", all(
        "reg-" in str(f.parent.parent.name)
        for f in kinds.get("structural", [])))


def _spec(mt: str = "binary_classification") -> TaskSpec:
    return TaskSpec(data={
        "task_id": "topo", "task_name": "topo", "model_type": mt,
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc" if mt != "regression" else "rmse",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "final_test": "2026-04"},
        "data": {},
    })


def _mk(run: Path, tid: str, code: str, parents=(), **extra) -> TrialRecord:
    d = run / "trials" / tid
    (d / "code").mkdir(parents=True, exist_ok=True)
    (d / "config").mkdir(exist_ok=True)
    (d / "code" / "train.py").write_text(code, encoding="utf-8")
    data = {"trial_id": tid, "status": "CODE_WRITTEN", "generation": 1,
            "parent_trial_ids": list(parents), "search_action": "explore",
            "hypothesis": {"statement": f"h-{tid}"},
            "execution": {"code_entry": "train.py",
                          "command": "python train.py"},
            "status_history": [{"state": "CODE_WRITTEN"}]}
    data.update(extra)
    import yaml
    (d / "trial.yaml").write_text(yaml.safe_dump(data, allow_unicode=True),
                                  encoding="utf-8")
    return TrialRecord.load(d)


def _approve(run: Path, **rec) -> dict:
    return append_ledger(run, {"action": "approve", "approver": "tester",
                               "host": "h", "tty": "/dev/pts/0",
                               "reason": "test", **rec})


def test_c_whitelist_and_declare() -> None:
    print("== C. 白名单与声明 ==")
    run = tmp_run_dir()
    try:
        t = _mk(run, "trial-001-g01-xgboost-1", HURDLE)
        g = evaluate_gate(run, t, "regression")
        check("hurdle @ regression → whitelisted",
              g["allowed"] and g["decision"] == "whitelisted", str(g["needs"]))
        g = evaluate_gate(run, t, "binary_classification")
        check("hurdle @ binary → blocked(structural:hurdle 需批准)",
              not g["allowed"] and g["needs"] == ["structural:hurdle"])
        t2 = _mk(run, "trial-002-g01-lgbm-1", T_LEARNER.format(mt="uplift"))
        check("T-learner @ uplift → whitelisted",
              evaluate_gate(run, t2, "uplift")["decision"] == "whitelisted")
        t3 = _mk(run, "trial-003-g01-lgbm-1", SINGLE,
                 declared_topology="cascade")
        g = evaluate_gate(run, t3, "binary_classification")
        check("声明 cascade 的单模型代码仍需批准(声明只增不减)",
              not g["allowed"] and g["needs"] == ["cascade"])
        t4 = _mk(run, "trial-004-g01-lgbm-1", CASCADE,
                 declared_topology="single")
        check("声明 single 不能抵消检测到的级联",
              not evaluate_gate(run, t4, "binary_classification")["allowed"])
        b = _mk(run, "trial-005-g00-lgbm-1", CASCADE,
                execution={"baseline_recipe": {"source": "rendered"},
                           "code_entry": "train.py"})
        g = evaluate_gate(run, b, "binary_classification")
        check("baseline(Runtime 渲染)只留痕不拦",
              g["allowed"] and g["decision"] == "exempt_baseline")
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_d_ledger() -> None:
    print("== D. 批准账本 ==")
    run = tmp_run_dir()
    try:
        bt = "binary_classification"
        p = _mk(run, "trial-001-g01-xgboost-1", CASCADE)
        g = evaluate_gate(run, p, bt)
        check("未批准级联 → blocked", not g["allowed"])
        a1 = _approve(run, scope="recipe", trial_id=p.trial_id,
                      code_sha256=g["code_sha256"], signature=g["signature"],
                      kinds=["cascade"])
        g = evaluate_gate(run, p, bt)
        check("recipe 批准后放行", g["allowed"] and g["decision"] == "approved"
              and g["approvals"] == [a1["approval_id"]])
        c = _mk(run, "trial-002-g02-xgboost-1", CASCADE + "\n# confirm\n",
                parents=[p.trial_id])
        check("recipe 覆盖后代(同签名,代码不同)",
              evaluate_gate(run, c, bt)["allowed"])
        c2 = _mk(run, "trial-003-g02-xgboost-1", CASCADE + ENSEMBLE_SEEDS
                 .replace("def fit_and_score", "def other"),
                 parents=[p.trial_id])
        check("后代新增组合方式(签名越界)须重批",
              not evaluate_gate(run, c2, bt)["allowed"])
        u = _mk(run, "trial-004-g02-xgboost-1", CASCADE)
        check("非后代不被 recipe 覆盖", not evaluate_gate(run, u, bt)["allowed"])
        g_u = evaluate_gate(run, u, bt)
        _approve(run, scope="trial", trial_id=u.trial_id,
                 code_sha256=g_u["code_sha256"], signature=g_u["signature"],
                 kinds=["cascade"])
        check("trial 范围批准放行", evaluate_gate(run, u, bt)["allowed"])
        (Path(u.trial_dir) / "code" / "train.py").write_text(
            CASCADE + "\n# edited\n", encoding="utf-8")
        check("trial 范围:改码即失效",
              not evaluate_gate(run, u, bt)["allowed"])
        e = _mk(run, "trial-005-g02-lgbm-1", ENSEMBLE_WEIGHTED)
        check("run 范围 cascade 不覆盖 ensemble",
              not evaluate_gate(run, e, bt)["allowed"])
        a_run = _approve(run, scope="run", kinds=["ensemble"])
        check("run 范围 ensemble 放行", evaluate_gate(run, e, bt)["allowed"])
        append_ledger(run, {"action": "revoke", "revokes":
                            a_run["approval_id"], "reason": "r",
                            "approver": "tester"})
        check("撤销后重新拦截", not evaluate_gate(run, e, bt)["allowed"])
        # 篡改:手工追加一条 run 级批准 → 链断裂 → 全部作废
        led = ledger_path(run)
        fake = {"action": "approve", "scope": "run", "kinds": ["ensemble",
                "cascade"], "approval_id": "apr-999", "seq": 999,
                "prev_sha": "x", "sha": "y"}
        with open(led, "a", encoding="utf-8") as f:
            f.write(json.dumps(fake) + "\n")
        g = evaluate_gate(run, p, bt)
        check("哈希链篡改 → 全部批准作废", not g["allowed"]
              and g["ledger_valid"] is False)
        try:
            _approve(run, scope="run", kinds=["cascade"])
            check("链断裂后拒绝追加", False)
        except RuntimeError:
            check("链断裂后拒绝追加", True)
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_e_runner() -> None:
    print("== E. runner 接线 ==")
    from runtime.runner import TrialRunner

    run = tmp_run_dir()
    try:
        t = _mk(run, "trial-001-g01-xgboost-1", CASCADE)
        runner = TrialRunner(run, _spec(), Config.load(skill_config=False))
        out = runner.run(t)
        t2 = TrialRecord.load(Path(t.trial_dir))
        check("被拦 trial 不执行、状态不变",
              out.status == "CODE_WRITTEN" and t2.status == "CODE_WRITTEN")
        check("不记失败", not t2.data.get("failure_reason"))
        check("未冻结源码(仍可改写)",
              not (t2.data.get("integrity") or {}).get(
                  "source_manifest_sha256"))
        topo = t2.data.get("model_topology") or {}
        check("trial.yaml 记录拓扑裁决", topo.get("decision") == "blocked"
              and topo.get("kind") == "cascade")
        req = run / "approvals" / "requests" / f"{t.trial_id}.json"
        ud = run / "logs" / "user-decisions.md"
        check("写入批准请求", req.exists())
        txt = ud.read_text(encoding="utf-8") if ud.exists() else ""
        check("user-decisions.md 记 PENDING", "[PENDING] 模型拓扑批准" in txt)
        record_request(run, t2, topo)
        check("同 trial 同签名请求幂等", ud.read_text(encoding="utf-8")
              .count(f"模型拓扑批准:{t.trial_id}") == 1)
    finally:
        shutil.rmtree(run, ignore_errors=True)


def test_f_cli_and_hook() -> None:
    print("== F. CLI 与 hook ==")
    run = tmp_run_dir()
    try:
        _mk(run, "trial-001-g01-xgboost-1", CASCADE)
        (run / "task-spec.yaml").write_text(
            "\n".join(["task_id: topo", "task_name: topo",
                       "model_type: binary_classification",
                       "target: {name: label, positive_definition: '1'}",
                       "primary_metric: auc", "data: {}"]), encoding="utf-8")
        cli = [sys.executable, str(SKILL / "scripts" / "modelevo.py"),
               "approve-topology", "--run", str(run)]
        r = subprocess.run(cli + ["--trial", "trial-001-g01-xgboost-1",
                                  "--reason", "x"], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True)
        check("非交互终端拒绝批准(rc=2)", r.returncode == 2,
              r.stderr[-300:])
        check("拒绝后账本未写", not ledger_path(run).exists())
        r = subprocess.run(cli + ["--list"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        check("--list 只读可用", r.returncode == 0, r.stderr[-300:])
    finally:
        shutil.rmtree(run, ignore_errors=True)

    hook = SKILL / "assets" / "guard_main_session.py"

    def _decide(payload: dict, extra_env: dict | None = None) -> str:
        env = dict(os.environ, CLAUDE_PROJECT_DIR="/x", **(extra_env or {}))
        p = subprocess.run([sys.executable, str(hook)],
                           input=json.dumps(payload), capture_output=True,
                           text=True, env=env)
        return "deny" if '"deny"' in p.stdout else "allow"

    approve = {"tool_name": "Bash", "tool_input": {
        "command": "python /s/scripts/modelevo.py approve-topology --run /r"
                   " --trial t --reason x"}}
    check("hook:主会话批准 → deny", _decide(approve) == "deny")
    check("hook:subagent + MODELEVO_ALLOW_MAIN_EXEC 批准 → 仍 deny",
          _decide(dict(approve, agent_id="a"),
                  {"MODELEVO_ALLOW_MAIN_EXEC": "1"}) == "deny")
    check("hook:--list → allow", _decide({"tool_name": "Bash", "tool_input": {
        "command": "python /s/scripts/modelevo.py approve-topology --run /r"
                   " --list"}}) == "allow")
    check("hook:heredoc/python 里拼出批准命令 → deny", _decide({
        "agent_id": "a", "tool_name": "Bash", "tool_input": {
            "command": "python3 - <<'EOF'\nimport os\nos.execvp('python', "
                       "['python', 'scripts/modelevo.py', 'approve-topology',"
                       " '--trial', 't'])\nEOF"}}) == "deny")
    check("hook:直接调账本写入函数 → deny", _decide({
        "agent_id": "a", "tool_name": "Bash", "tool_input": {
            "command": "python -c 'from runtime.topology import append_ledger'"
        }}) == "deny")
    check("hook:Write 账本 → deny", _decide({
        "agent_id": "a", "tool_name": "Write", "tool_input": {
            "file_path": "/r/approvals/model-topology.jsonl"}}) == "deny")
    check("hook:Bash 重定向写账本 → deny", _decide({
        "agent_id": "a", "tool_name": "Bash", "tool_input": {
            "command": "echo {} >> /r/approvals/model-topology.jsonl"}})
          == "deny")
    check("hook:cat 账本 → allow", _decide({
        "tool_name": "Bash", "tool_input": {
            "command": "cat /r/approvals/model-topology.jsonl"}}) == "allow")
    check("hook:写 trial 代码 → allow", _decide({
        "agent_id": "a", "tool_name": "Write", "tool_input": {
            "file_path": "/r/trials/t/code/train.py"}}) == "allow")


if __name__ == "__main__":
    test_a_detector()
    test_b_templates_and_library()
    test_c_whitelist_and_declare()
    test_d_ledger()
    test_e_runner()
    test_f_cli_and_hook()
    print(f"\n=== v2.16.0 topology: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
