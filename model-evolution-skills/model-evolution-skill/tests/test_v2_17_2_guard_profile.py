"""v2.17.2 回归:E1 holdout 标签面守卫 + E2 reparse/文本列 token 级漂移。

事故来源(run4-v2.16.0-2.opus5.5 / req-t7-400w):
- E1:gen-06 的 5 个 adhoc/gen06_cb_*probe.py 直接
  pd.read_parquet(".../sample-profile/splits/labels/validation.parquet")
  在 validation 标签上调 CatBoost——guard hook 只拦 harness 源码;
- E2:LOAN_MARKTING 事件在 2026-02 之后断档,drift-profile 只有数值列 PSI,
  靠 gen-03 手写探针(adhoc/gen03_probe_event_coverage.py)才发现。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))

GUARD = SKILL / "assets" / "guard_main_session.py"
_EXEMPT_VARS = ("MODELEVO_ALLOW_SOURCE_READ", "MODELEVO_ALLOW_MAIN_EXEC",
                "MODELEVO_ALLOW_HOLDOUT_READ")


# ===========================================================================
# E1 holdout 标签面守卫
# ===========================================================================
def _guard(payload: dict, env: dict | None = None) -> tuple[str, str]:
    e = {k: v for k, v in os.environ.items() if k not in _EXEMPT_VARS}
    e.update(CLAUDE_PROJECT_DIR="/x", **(env or {}))
    p = subprocess.run([sys.executable, str(GUARD)],
                       input=json.dumps(payload), capture_output=True,
                       text=True, env=e, timeout=60)
    return ("deny" if '"deny"' in p.stdout else "allow"), p.stdout


def _bash(cmd: str, *, sub: bool = True, cwd: str | None = None) -> dict:
    d = {"tool_name": "Bash", "tool_input": {"command": cmd}}
    if sub:
        d["agent_id"] = "a"
    if cwd:
        d["cwd"] = cwd
    return d


_R = "/r/run-x"
_VAL = f"{_R}/sample-profile/splits/labels/validation.parquet"
_PROBE = (  # 事故原文(gen06_cb_gpu_probe.py 头部)
    'import pandas as pd\nR = "/r/run-x"\n'
    'X = pd.read_parquet(f"{R}/trials/t-018/model/validation_input.parquet")\n'
    'y = pd.read_parquet(f"{R}/sample-profile/splits/labels/validation.parquet")\n')


@pytest.fixture()
def run_dir(tmp_path):
    run = tmp_path / "run"
    for d in ("adhoc", "feature-lab"):
        (run / d).mkdir(parents=True)
    (run / "adhoc" / "probe_val.py").write_text(_PROBE, encoding="utf-8")
    (run / "adhoc" / "probe_join.py").write_text(
        "from pathlib import Path\nimport pandas as pd\n"
        "SPL = Path(__file__).parents[1] / 'sample-profile' / 'splits'\n"
        "y = pd.read_parquet(SPL / 'labels' / 'shadow_test.parquet')\n",
        encoding="utf-8")
    (run / "feature-lab" / "loop.py").write_text(
        "import pandas as pd\nfor s in ['train', 'validation']:\n"
        "    y = pd.read_parquet(f'{SPL}/labels/{s}.parquet')\n",
        encoding="utf-8")
    (run / "adhoc" / "train_only.py").write_text(
        "import pandas as pd\n"
        "y = pd.read_parquet('sample-profile/splits/labels/train.parquet')\n"
        "X = pd.read_parquet('sample-profile/splits/validation.parquet')\n"
        "from runtime.data import load_labels\n"
        "lab = load_labels(R, splits=('train',))\n", encoding="utf-8")
    (run / "adhoc" / "feat_only.py").write_text(
        "import pandas as pd\n"
        "X = pd.read_parquet('sample-profile/splits/validation.parquet')\n"
        "print(X['seq_d365'].str.len().describe())\n", encoding="utf-8")
    return run


@pytest.mark.parametrize("payload", [
    {"tool_name": "Read", "tool_input": {"file_path": _VAL}},
    {"agent_id": "a", "tool_name": "Read", "tool_input": {
        "file_path": f"{_R}/sample-profile/splits-sample/labels/"
                     "shadow_test.parquet"}},
    {"tool_name": "Grep", "tool_input": {
        "pattern": "x", "path": f"{_R}/sample-profile/splits/labels/"
                               "final_test.parquet"}},
    _bash("python -c \"import pandas as pd; print(pd.read_parquet('"
          + _VAL + "').mean())\""),
    _bash("python3 - <<'EOF'\nimport pandas as pd\n"
          "y = pd.read_parquet('" + _VAL + "')\nEOF", sub=False),
    _bash("cd /r/run-x/sample-profile/splits && python -c \"import pandas as "
          "pd; pd.read_parquet('labels/validation.parquet')\""),
    _bash("duckdb -c \"select avg(y) from '" + _VAL + "'\""),
    _bash("python -c \"import os, pandas as pd; pd.read_parquet("
          "os.path.join(S, 'labels', 'validation.parquet'))\""),
    _bash("python -c \"from runtime.data import load_labels; "
          "lab = load_labels(R)\""),
    _bash("python -c \"from runtime.data import load_labels; "
          "lab = load_labels(R, splits=('validation',))\""),
    # 官方 CLI 子句豁免不能顺带放行同一命令里的其他子句
    _bash("python /s/scripts/modelevo.py status --run /r && python -c "
          "\"import pandas as pd; pd.read_parquet('" + _VAL + "')\""),
    # 用 Bash 写 adhoc 脚本(heredoc)
    _bash("cat > adhoc/p.py <<'EOF'\n" + _PROBE + "EOF"),
    # prescreen 路径用 .. 回跳到真实 splits 不算沙箱
    {"tool_name": "Read", "tool_input": {
        "file_path": f"{_R}/prescreen/../sample-profile/splits/labels/"
                     "validation.parquet"}},
])
def test_e1_denies_direct_holdout_label_reads(payload):
    decision, out = _guard(payload)
    assert decision == "deny", payload
    assert "holdout 标签面边界" in out and "prescreen" in out \
        and "contrast" in out
    # 主会话/subagent 的编排与源码豁免都不放行
    assert _guard(payload, {"MODELEVO_ALLOW_MAIN_EXEC": "1",
                            "MODELEVO_ALLOW_SOURCE_READ": "1"})[0] == "deny"
    # 唯一豁免:用户在会话环境显式设 MODELEVO_ALLOW_HOLDOUT_READ=1
    assert _guard(payload, {"MODELEVO_ALLOW_HOLDOUT_READ": "1"})[0] == "allow"


@pytest.mark.parametrize("payload", [
    {"tool_name": "Read", "tool_input": {
        "file_path": f"{_R}/sample-profile/splits/labels/train.parquet"}},
    {"tool_name": "Read", "tool_input": {
        "file_path": f"{_R}/sample-profile/splits/validation.parquet"}},
    {"tool_name": "Glob", "tool_input": {
        "pattern": "sample-profile/splits/labels/*.parquet"}},
    {"agent_id": "a", "tool_name": "Read", "tool_input": {
        "file_path": f"{_R}/prescreen/_sandbox/v1-abc/sample-profile/splits/"
                     "labels/validation.parquet"}},
    _bash("python -c \"import pandas as pd; pd.read_parquet('"
          f"{_R}/sample-profile/splits/labels/train.parquet')\""),
    _bash("python -c \"from runtime.data import load_labels; "
          "lab = load_labels(R, splits=('train',))\""),
    # 官方 CLI(含 harness 内部读取 validation 标签的子命令)
    _bash("python /s/scripts/modelevo.py contrast --run /r --trials a,b"),
    _bash("python /s/scripts/modelevo.py prescreen --run /r --trial t "
          "--seeds 3"),
    _bash("MODELEVO_X=1 nohup python3 /s/scripts/modelevo.py run --run /r "
          "--trial t > /r/logs/run.log 2>&1 &"),
    # 只看元数据 / 文本检索
    _bash(f"ls -la {_R}/sample-profile/splits/labels/"),
    _bash(f"grep -rn 'labels/validation.parquet' {_R}/adhoc/"),
    # df["labels"] 列访问不是标签目录
    _bash("python -c \"df = load(); print(df['labels'].mean(), "
          "'validation')\""),
])
def test_e1_allows_train_cli_and_sandbox(payload):
    assert _guard(payload)[0] == "allow", payload


def test_e1_inspects_executed_lab_scripts(run_dir):
    run = str(run_dir)
    # 事故形态:subagent 在 run 根执行 adhoc 探针
    assert _guard(_bash("python adhoc/probe_val.py", cwd=run))[0] == "deny"
    # 主会话同规;cd 进 run 后执行 / 带解释器开关与重定向
    assert _guard(_bash(f"cd {run} && python -u adhoc/probe_join.py > "
                        "adhoc/o.txt 2>&1", sub=False, cwd="/"))[0] == "deny"
    # split 名来自循环变量 + holdout 字面量
    assert _guard(_bash(f"timeout 600 python3 {run}/feature-lab/loop.py",
                        cwd="/"))[0] == "deny"
    # 只读 train 标签 + validation 特征面:放行
    assert _guard(_bash("python adhoc/train_only.py", cwd=run))[0] == "allow"
    assert _guard(_bash("python adhoc/feat_only.py", cwd=run))[0] == "allow"
    # 只 cat 脚本(不执行)不检查脚本内容
    assert _guard(_bash("cat adhoc/probe_val.py", cwd=run))[0] == "allow"
    # 找不到脚本文件 → 无法检查,放行(fail-open)
    assert _guard(_bash("python adhoc/missing.py", cwd=run))[0] == "allow"


def test_e1_write_edit_lab_scripts(run_dir):
    run = run_dir
    w = {"agent_id": "a", "tool_name": "Write", "tool_input": {
        "file_path": str(run / "adhoc" / "new_probe.py"), "content": _PROBE}}
    assert _guard(w)[0] == "deny"
    # 同内容写到 adhoc/feature-lab 之外(如 trial 代码)不在本规则范围
    w_trial = {"agent_id": "a", "tool_name": "Write", "tool_input": {
        "file_path": str(run / "trials" / "t" / "code" / "train.py"),
        "content": _PROBE}}
    assert _guard(w_trial)[0] == "allow"
    # 非脚本文件(笔记)提到路径不拦
    w_md = {"tool_name": "Write", "tool_input": {
        "file_path": str(run / "adhoc" / "notes.md"), "content": _PROBE}}
    assert _guard(w_md)[0] == "allow"
    # Edit:新片段单独看无害,与原文件合成后读 holdout → 拦
    e = {"agent_id": "a", "tool_name": "Edit", "tool_input": {
        "file_path": str(run / "adhoc" / "train_only.py"),
        "old_string": "splits=('train',)",
        "new_string": "splits=('train', 'validation')"}}
    assert _guard(e)[0] == "deny"
    e_ok = {"agent_id": "a", "tool_name": "Edit", "tool_input": {
        "file_path": str(run / "adhoc" / "feat_only.py"),
        "old_string": "describe()", "new_string": "mean()"}}
    assert _guard(e_ok)[0] == "allow"
    me = {"agent_id": "a", "tool_name": "MultiEdit", "tool_input": {
        "file_path": str(run / "feature-lab" / "x.py"), "edits": [
            {"old_string": "", "new_string":
                "y = pd.read_parquet(SPL / 'labels' / 'final_test.parquet')"}]}}
    assert _guard(me)[0] == "deny"


def test_e1_existing_guards_unchanged():
    # 源码拦截与拓扑批准仍按旧规则
    assert _guard({"tool_name": "Read", "tool_input": {
        "file_path": "/h/.claude/skills/modelevo/runtime/runner.py"}})[0] \
        == "deny"
    assert _guard(_bash("python /s/scripts/modelevo.py approve-topology "
                        "--run /r --trial t --reason x"))[0] == "deny"
    # 非 Claude Code 上下文(无 CLAUDE_PROJECT_DIR)整体放行
    e = {k: v for k, v in os.environ.items()
         if k not in _EXEMPT_VARS + ("CLAUDE_PROJECT_DIR",)}
    p = subprocess.run([sys.executable, str(GUARD)], input=json.dumps(
        {"tool_name": "Read", "tool_input": {"file_path": _VAL}}),
        capture_output=True, text=True, env=e, timeout=60)
    assert p.returncode == 0 and '"deny"' not in p.stdout


# ===========================================================================
# E2 token 级漂移
# ===========================================================================
_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


def _token_spec(reparse: bool = True):
    from runtime.spec import TaskSpec

    d = {
        "task_id": "tok-drift", "task_name": "token 漂移回归",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "primary_metric": "auc",
        "time": {"column": "pday"}, "entity": {"id_column": "uid"},
        "split_policy": {"type": "time_based", "train": "2026-01~2026-04",
                         "validation": "2026-05", "shadow_test": "2026-06"},
    }
    if reparse:
        d["features"] = {"reparse": [{
            "column": "seq", "reason": "form_unusable",
            "plan": "逗号分隔事件序列,按事件名计数"}]}
    return TaskSpec(data=d)


def _token_frame(n_per_month: int = 300, *, with_seq: bool = True,
                 seed: int = 0) -> pd.DataFrame:
    """GONE 事件只在前两个月出现(≈50% 行),之后断档;A/B/C 全程稳定。"""
    rng = np.random.default_rng(seed)
    recs = []
    for mi, m in enumerate(_MONTHS):
        for i in range(n_per_month):
            day = 1 + i % 28
            toks = [f"@DB_{rng.integers(0, 30)}", "A@HR_noon"]
            if rng.random() < 0.6:
                toks.append("B@HR_night@AMT_lt1k")
            if rng.random() < 0.3:
                toks.append("C")
            if mi < 2 and rng.random() < 0.5:
                toks += ["GONE@HR_noon", f"@DB_{rng.integers(0, 30)}",
                         "GONE@HR_late@CALL_AUTO"]
            rng.shuffle(toks)
            rec = {"uid": f"U{mi}{i:05d}", "pday": f"{m}-{day:02d}",
                   "label": int(rng.random() < 0.3),
                   "f1": float(rng.normal()), "city": ["bj", "sh", "gz"][i % 3]}
            if with_seq:
                rec["seq"] = ",".join(toks)
            recs.append(rec)
    return pd.DataFrame(recs)


def _run_profile(tmp_path, df, spec):
    from runtime.profile import SampleProfile

    out = tmp_path / "sample-profile"
    prof = SampleProfile(df, spec, out)
    res = prof.run(max_workers=2)
    return res, out


def test_e2_collapsed_token_flagged_and_stable_tokens_not(tmp_path):
    res, out = _run_profile(tmp_path, _token_frame(), _token_spec())
    dp = pd.read_csv(out / "drift-profile.csv")
    pq_dp = pd.read_parquet(out / "drift-profile.parquet")
    assert len(pq_dp) == len(dp)
    # 数值行与既有列保持不变
    num = dp[dp["kind"] != "token"]
    assert {"feature", "comparison", "psi", "kind"} <= set(dp.columns)
    assert set(num["kind"]) <= {"split", "date_first", "date_adjacent"}
    assert (num["feature"] == "f1").all() and num["psi"].notna().any()
    assert res["drift_summary"]["n_comparisons"] == len(num)
    # token 行:reparse 列 × top-K token × 月
    tok = dp[dp["kind"] == "token"]
    assert set(tok["feature"]) == {"seq"}
    assert set(tok["period"]) == set(_MONTHS)
    names = set(tok["token"])
    assert {"A", "B", "C", "GONE", "__vocab__"} <= names
    assert not any(str(t).startswith("@") for t in names)  # 位置标记不计
    gone = tok[(tok["token"] == "GONE")].set_index("period")
    assert gone.loc["2026-01", "coverage"] > 0.3
    assert gone.loc["2026-03", "coverage"] == 0
    flagged = gone["flag"].fillna("").str.split("|")
    assert set(gone.index[flagged.map(lambda f: "collapse" in f)]) \
        == set(_MONTHS[2:])
    for t in ("A", "B", "C"):
        assert (tok.loc[tok["token"] == t, "flag"].fillna("") == "").all(), t
    voc = tok[tok["token"] == "__vocab__"].set_index("period")
    assert voc["tokens_per_row"].notna().all() and voc["jsd"].notna().all()
    # GONE 断档后:消失份额 > 0、分布 JSD 上升
    assert voc.loc["2026-05", "vanished_token_share"] > 0
    assert voc.loc["2026-05", "jsd"] > voc.loc["2026-01", "jsd"]
    # 摘要 / leakage 告警 / md
    ds = res["drift_summary"]
    assert ds["tokens_coverage_collapsed"] == ["seq:GONE"]
    c = ds["token_drift"]["collapsed_tokens"][0]
    assert c["scope"] == "row" and c["token"] == "GONE"
    assert c["collapse_month"] == "2026-03" and c["persistent"] is True
    assert c["min_coverage_ratio"] == 0
    assert {x["token"] for x in ds["token_drift"]["collapsed_tokens"]} \
        == {"GONE"}
    finds = [f for f in res["leakage"]["findings"]
             if f["kind"] == "token_drift"]
    assert len(finds) == 1 and finds[0]["severity"] == "warning" \
        and finds[0]["columns"] == ["seq"] and "GONE" in finds[0]["detail"]
    md = (out / "sample-profile.md").read_text(encoding="utf-8")
    assert "token 漂移" in md and "| seq | GONE |" in md
    assert "token_drift" in (out / "leakage-audit.md").read_text(
        encoding="utf-8")
    # feature-quality 单表视图不受 token 行影响
    fq = pd.read_csv(out / "feature-quality.csv")
    assert "seq" not in set(fq["feature"])


def test_e2_no_text_columns_no_token_rows(tmp_path):
    res, out = _run_profile(tmp_path, _token_frame(with_seq=False),
                            _token_spec(reparse=False))
    dp = pd.read_csv(out / "drift-profile.csv")
    assert "token" not in set(dp["kind"])
    assert list(dp.columns) == ["feature", "comparison", "psi", "kind"]
    td = res["drift_summary"]["token_drift"]
    assert td["columns"] == [] and td["skipped"] == "no_text_columns"
    assert "tokens_coverage_collapsed" not in res["drift_summary"]
    assert not [f for f in res["leakage"]["findings"]
                if f["kind"] == "token_drift"]


def test_e2_stable_sequences_not_flagged_and_extra_text_feature(tmp_path):
    # 无断档;且序列列不声明 reparse(作为普通文本特征列也被纳入)
    df = _token_frame(seed=1)
    df["seq"] = df["seq"].str.replace(r"GONE@[^,]*,?", "", regex=True)
    res, out = _run_profile(tmp_path, df, _token_spec(reparse=False))
    td = res["drift_summary"]["token_drift"]
    assert td["columns"] == ["seq"]
    assert td["n_collapsed"] == 0 and td["collapsed_tokens"] == []
    dp = pd.read_csv(out / "drift-profile.csv")
    assert (dp.loc[dp["kind"] == "token", "flag"].fillna("") == "").all()
    # 普通类别列(无分隔符)不纳入
    assert "city" not in td["columns"]


def _stub_profile(df: pd.DataFrame, train_mask: np.ndarray, cfg: dict | None
                  = None, feature_cols: list | None = None):
    """不走完整 profile:只挂 _token_drift 需要的属性(大样本/成本用例)。"""
    from runtime.profile import SampleProfile

    prof = object.__new__(SampleProfile)
    prof.df, prof.spec = df, _token_spec()
    prof.feature_cols = feature_cols or []
    prof.split_index = {"train": df.index[train_mask]}

    class _C:
        @staticmethod
        def get(key, default=None):
            return (cfg or {}).get(key, default)

    prof.config = _C()
    return prof


def test_e2_lookback_logging_gap_flagged_by_recent_window():
    """事故形态:365 天回溯序列(旧→新),MKT 事件 2026-03-01 起停止落日志。

    整行覆盖率只缓慢下降(用户一年内总有 MKT)→ 整行口径不塌缩;序列尾窗
    (近端 10% token)陡降 → collapse_tail 命中;稳定事件 ENTER 不命中。
    """
    import datetime as dt

    rng = np.random.default_rng(7)
    gap = dt.date(2026, 3, 1)
    recs = []
    for m in _MONTHS:
        y, mo = map(int, m.split("-"))
        for i in range(200):
            pday = dt.date(y, mo, 1 + i % 28)
            toks = []
            for k in range(364, -1, -1):
                if rng.random() >= 0.3:
                    continue
                day = pday - dt.timedelta(days=k)
                toks += [f"@DB_{k}", "ENTER@HR_x"]
                if day < gap and rng.random() < 0.5:
                    toks.append("MKT@HR_x@AMT_lt1k")
            recs.append({"pday": pday.isoformat(), "seq": ",".join(toks)})
    df = pd.DataFrame(recs)
    tok = _stub_profile(df, (df["pday"] < "2026-05").to_numpy())._token_drift()
    coll = tok["summary"]["collapsed_tokens"]
    assert {(c["token"], c["scope"]) for c in coll} == {("MKT", "tail")}
    c = coll[0]
    assert c["collapse_month"] in ("2026-03", "2026-04") and c["persistent"]
    rows = pd.DataFrame(tok["rows"])
    mkt = rows[rows["token"] == "MKT"].set_index("period")
    assert mkt.loc["2026-06", "coverage"] > 0.9            # 整行几乎不变
    assert mkt.loc["2026-06", "tail_coverage"] < 0.05      # 近端窗口断档
    assert "collapse_tail" in mkt.loc["2026-06", "flag"]
    assert "collapse" not in mkt.loc["2026-06", "flag"].split("|")
    # 关闭近端窗口(edge_frac=0)→ 回到纯整行口径,不再命中
    tok0 = _stub_profile(df, (df["pday"] < "2026-05").to_numpy(),
                         {"profile.token_drift_edge_frac": 0})._token_drift()
    assert tok0["summary"]["n_collapsed"] == 0
    assert "tail_coverage" not in pd.DataFrame(tok0["rows"]).columns


def test_e2_unit_tokenizer_and_cost_bounded():
    import time

    from runtime.profile import _token_head_regex

    rx = _token_head_regex()
    assert [h for h in rx.findall(
        "@DB_3,LOAN_MARKTING@HR_noon@AMT_lt1k,,X,@DB_1,Y@a") if h] \
        == ["LOAN_MARKTING", "X", "Y"]
    assert [h for h in _token_head_regex(",;", "").findall("a@1;b,c")
            if h] == ["a@1", "b", "c"]
    # 1M 行 × ~20KB 字符串:每月抽样,成本与总行数无关
    rng = np.random.default_rng(0)
    pool = []
    for k in range(40):
        toks = [f"EV{rng.integers(0, 120)}@HR_x@AMT_{rng.integers(0, 9)}"
                for _ in range(1300)]
        pool.append(",".join(toks))
    assert min(len(s) for s in pool) > 18_000
    n = 1_000_000
    seq = np.empty(n, dtype=object)
    seq[:] = [pool[i % 40] for i in range(n)]
    months = np.repeat(_MONTHS, n // len(_MONTHS) + 1)[:n]
    df = pd.DataFrame({"pday": [f"{m}-01" for m in months], "seq": seq})
    prof = _stub_profile(df, months <= "2026-04",
                         {"profile.token_drift_rows_per_month": 2000})
    t0 = time.time()
    tok = prof._token_drift()
    elapsed = time.time() - t0
    assert tok["summary"]["columns"] == ["seq"]
    assert tok["summary"]["n_collapsed"] == 0
    assert elapsed < 60, elapsed
