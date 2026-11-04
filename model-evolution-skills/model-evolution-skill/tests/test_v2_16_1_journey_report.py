"""v2.16.1:交付时默认输出训练历程 HTML 报告(report/report.html)。

`report` 命令在 report.md 之外,从落盘产物确定性生成单文件 HTML(左侧目录、
冠军路径、进化地图、全部尝试、证伪方向、最终效果、过程、裁决、经验),说明文字
优先取可选的 report/journey-narrative.yaml。本文件用合成 run 覆盖:冠军路径与
阶梯加总、指标方向、叙述覆盖与回退、缺失产物降级、注入转义、CLI 开关、
final_test 标签零读取,以及(有 node 时)模板脚本的语法与 DOM 冒烟。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

from runtime.config import Config  # noqa: E402
from runtime.data import SAMPLE_KEY_COLUMN  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from _mvo.report_html import (build_journey_html, champion_path,  # noqa: E402
                              collect_journey_data, render_html, short_label,
                              step_category)

NODE = shutil.which("node")
STUB = SKILL / "tests" / "_journey_dom_stub.js"


def _tid(n, g, act):
    return f"trial-{n:03d}-g{g:02d}-{act}-xgboost-1"


def _make_run(tmp_path, *, metric="rmse", final=True, extras=True):
    """合成 run:基线 → 特征 accepted → 拒绝臂 → 模型 accepted → confirm 冠军。"""
    hib = metric == "auc"
    run = tmp_path / "run"
    (run / "leaderboard").mkdir(parents=True)
    spec = {"task_id": "demo-task", "task_name": "演示任务", "model_type":
            "binary_classification" if hib else "regression",
            "target": {"name": "y", "value_definition": "演示目标"},
            "primary_metric": metric, "secondary_metrics": ["mae"] if not hib else ["ks"],
            "split_policy": {"type": "time_based", "train": "2026-01-01~2026-03-31",
                             "validation": "2026-04", "shadow_test": "2026-05",
                             "final_test": "2026-06"}}
    (run / "task-spec.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
    sgn = 1 if hib else -1          # 改进方向
    base = 0.70 if hib else 2.80
    ids = {"base": _tid(2, 0, "explore"), "feat": _tid(3, 1, "explore"),
           "rej": _tid(4, 1, "explore"), "model": _tid(5, 2, "hpo"),
           "conf": _tid(6, 3, "confirm")}
    m_feat, m_model = base + sgn * 0.010, base + sgn * 0.016
    specs = [
        (ids["base"], 0, "explore", "", None, base, None, None, "accepted"),
        (ids["feat"], 1, "explore", "feature", ids["base"], m_feat, sgn * 0.010, [sgn * 0.012, sgn * 0.008], "accepted"),
        (ids["rej"], 1, "explore", "feature", ids["base"], base - sgn * 0.002, -sgn * 0.002, [-sgn * 0.003, -sgn * 0.001], "rejected"),
        (ids["model"], 2, "hpo", "model", f"{ids['feat']};{ids['rej']}", m_model, sgn * 0.006, [sgn * 0.007, sgn * 0.005], "accepted"),
        (ids["conf"], 3, "confirm", "", ids["model"], m_model + sgn * 0.0001, sgn * 0.0001, [-0.0005, 0.0006], "confirmed"),
    ]
    rows = []
    for tid, g, act, mods, parent, m, dp, ci, v in specs:
        rows.append({"trial_id": tid, "generation": g, "search_action": act, "changed_modules": mods or None,
                     "parent": parent, metric: m, ("mae" if not hib else "ks"): m / 2,
                     "delta_vs_parent": dp, "delta_ci": (f"[{min(ci):+.5f},{max(ci):+.5f}]" if ci else None),
                     "p_better": 0.99 if v == "accepted" else 0.2, "verdict": v, "algorithm": "xgboost",
                     "runtime_sec": 60.0})
        td = run / "trials" / tid
        td.mkdir(parents=True)
        (td / "trial.yaml").write_text(yaml.safe_dump({
            "trial_id": tid, "hypothesis": {"statement": f"包1 {act}:{tid} 的假设,详细说明",
                                             "rationale": f"{tid} 的理由", "falsification": "Δ≥0 即证伪"}},
            allow_unicode=True), encoding="utf-8")
        if act == "confirm":
            (td / "evaluation.json").write_text(json.dumps({"fresh_confirmation": {
                "passed": True, "reference": {"reference_trial": ids["feat"], "delta": sgn * 0.006}}}))
    rows.append({"trial_id": _tid(1, 0, "explore"), "generation": 0, metric: float("nan")})
    (run / "leaderboard" / "leaderboard.json").write_text(json.dumps({
        "primary_metric": metric, "baseline_id": ids["base"], "rows": rows,
        "delivery_resolution": {"status": "bound", "search_champion": ids["conf"], "tolerance": 0.001}}))
    if final:
        fm = run / "final-model"
        fm.mkdir()
        sec = "mae" if not hib else "ks"
        (fm / "evaluation.json").write_text(json.dumps({
            "final_test": {metric: m_model, sec: 1.0}, "baseline_final_test": {metric: base, sec: 1.1},
            "shadow_test": {metric: m_model, sec: 1.0}, "baseline_shadow_test": {metric: base, sec: 1.1},
            "shadow_test_search_model": {metric: m_model, sec: 1.0}, "shadow_test_source": "delivery_model",
            "segment_evaluation": {"bin_means": [0.1 * i for i in range(10)]}}))
        (fm / "model_meta.json").write_text(json.dumps({"refit_mode": "rerun_recipe", "n_validation_rows": 10}))
        # shadow 分桶输入:标签 + 交付/基线分数
        rng = np.random.default_rng(0)
        n = 400
        keys = np.array([f"k{i}" for i in range(n)])
        y = rng.random(n) if not hib else (rng.random(n) < 0.3).astype(float)
        labd = run / "sample-profile" / "splits" / "labels"
        labd.mkdir(parents=True)
        pd.DataFrame({"y": y, SAMPLE_KEY_COLUMN: keys}).to_parquet(labd / "shadow_test.parquet")
        pd.DataFrame({"score": y + rng.normal(0, 0.2, n), SAMPLE_KEY_COLUMN: keys}).to_parquet(
            fm / "shadow_test_score.parquet")
        pd.DataFrame({"score": y + rng.normal(0, 0.6, n), SAMPLE_KEY_COLUMN: keys}).to_parquet(
            run / "trials" / ids["base"] / "shadow_test_score.parquet")
    if extras:
        (run / "logs").mkdir()
        (run / "logs" / "timing-report.json").write_text(json.dumps({
            "trial_hours": 1.5, "phase_share": {"train:fit": 0.6, "eval": 0.4},
            "generations": [{"generation": g, "wall_min": 10.0 + g} for g in range(4)]}))
        (run / "logs" / "user-decisions.md").write_text(
            "# 用户待决事项\n\n## D-002 第二个裁决 — RESOLVED(用户 2026-10-22)\n\n- x\n\n"
            "## D-001 第一个裁决 — PENDING\n", encoding="utf-8")
        gen = run / "generations" / "gen-01"
        gen.mkdir(parents=True)
        (gen / "generation.yaml").write_text(yaml.safe_dump({"stop_report": {"stop_reason": "single_data_view_insufficient"}}))
        (run / "experience").mkdir()
        (run / "experience" / "exported-global-experiences.yaml").write_text(yaml.safe_dump(
            [{"claim": "特征 X 有效", "outcome": "positive"}, {"claim": "loss Y 无效", "outcome": "negative"}],
            allow_unicode=True), encoding="utf-8")
    return run, ids, base, m_model


def _collect(run):
    spec = TaskSpec(data=yaml.safe_load((run / "task-spec.yaml").read_text(encoding="utf-8")))
    return collect_journey_data(run, spec, Config.load(skill_config=False)), spec


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", ["rmse", "auc"])
def test_champion_path_and_ladder_sum(tmp_path, metric):
    run, ids, base, m_model = _make_run(tmp_path, metric=metric)
    d, _ = _collect(run)
    assert d["path"] == [ids["base"], ids["feat"], ids["model"], ids["conf"]], "沿第一父代回溯"
    assert [s["id"] for s in d["steps"]] == [ids["feat"], ids["model"]], "confirm 不计入阶梯步"
    total = sum(s["display_gain"] for s in d["steps"])
    assert total == pytest.approx(0.016, abs=1e-9), "方向归一化增益恒为正"
    assert [s["category"] for s in d["steps"]] == ["feature", "model"]
    assert d["shares"]["feature"] == pytest.approx(0.010) and d["shares"]["model"] == pytest.approx(0.006)
    rej = next(t for t in d["trials"] if t["id"] == ids["rej"])
    assert rej["gain"] == pytest.approx(-0.002), "被拒臂增益为负(变差)"
    assert d["meta"]["higher_is_better"] is (metric == "auc")
    assert d["n_trials_failed"] == 1 and len(d["trials"]) == 5
    assert d["confirmations"] == {"n": 1, "passed": 1}
    assert [f["id"] for f in d["falsified"]] == [ids["rej"]]


def test_helpers():
    rows = {"a": {"parent": None}, "b": {"parent": "a"}, "c": {"parent": "b;x"}, "x": {"parent": "a"}}
    assert champion_path(rows, "c") == ["a", "b", "c"]
    assert champion_path(rows, "missing") == []
    assert step_category({"changed_modules": "feature"}) == "feature"
    assert step_category({"search_action": "hpo"}) == "model"
    assert step_category({"changed_modules": "loss"}) == "model"
    assert step_category({"search_action": "data_view"}) == "data"
    assert short_label("包1 seqd_:在 baseline 配方不变的前提下,右接...") == "seqd_"


def test_narrative_overrides_and_fallback(tmp_path):
    run, ids, *_ = _make_run(tmp_path)
    d0, _ = _collect(run)
    s0 = d0["steps"][0]
    assert s0["why"] == f"{ids['feat']} 的理由", "缺 narrative 时 why 取 rationale"
    assert any("CI" in e for e in s0["evidence"]), "证据自动生成"
    assert [i["text"] for i in d0["insights"]] == ["特征 X 有效", "loss Y 无效"], "insights 回退到经验卡"
    (run / "report").mkdir(exist_ok=True)
    (run / "report" / "journey-narrative.yaml").write_text(yaml.safe_dump({
        "headline": "一句话结论",
        "steps": {ids["feat"]: {"title": "解析序列", "why": "因为 A", "evidence": ["证据 1"]}},
        "falsified": {"004": {"title": "裁剪", "why": "无增益"}},
        "labels": {ids["model"]: "HPO"},
        "insights": [{"title": "要点", "text": "内容"}],
        "notes": ["业务提醒"]}, allow_unicode=True), encoding="utf-8")
    d, _ = _collect(run)
    s = d["steps"][0]
    assert (s["title"], s["why"], s["evidence"]) == ("解析序列", "因为 A", ["证据 1"])
    assert d["steps"][1]["title"] == "HPO", "labels 作为图上与步骤的短标签"
    assert d["falsified"][0]["title"] == "裁剪", "短 id 也能匹配"
    assert d["headline"] == "一句话结论" and d["notes"] == ["业务提醒"]
    assert d["insights"] == [{"title": "要点", "text": "内容", "kind": "insight"}]


def test_missing_optional_inputs_degrade(tmp_path):
    run, *_ = _make_run(tmp_path, final=False, extras=False)
    d, spec = _collect(run)
    assert d["final"]["evaluated"] is False and d["shadow_seg"] is None
    assert d["decisions"] == [] and d["phase_share"] == [] and d["insights"] == []
    assert all(g["wall_min"] is None for g in d["generations"])
    html = build_journey_html(run, spec, Config.load(skill_config=False))
    assert html.exists() and (run / "report" / "journey-data.json").exists()


def test_shadow_segments_and_decisions(tmp_path):
    run, *_ = _make_run(tmp_path)
    d, _ = _collect(run)
    ss = d["shadow_seg"]
    assert ss and len(ss["deliv"]["cum"]) == 10 and ss["deliv"]["cum"][-1] == pytest.approx(1.0)
    assert ss["deliv"]["top"]["10%"] >= ss["base"]["top"]["10%"], "交付分数噪声更小,头部捕获更高"
    assert [x["id"] for x in d["decisions"]] == ["D-002", "D-001"]
    assert any(e["kind"] == "stop" for e in d["events"])


def test_html_structure_toc_and_escape(tmp_path):
    run, ids, *_ = _make_run(tmp_path)
    (run / "report").mkdir(exist_ok=True)
    (run / "report" / "journey-narrative.yaml").write_text(yaml.safe_dump(
        {"headline": "危险 </script><script>alert(1)</script>"}), encoding="utf-8")
    d, _ = _collect(run)
    html = render_html(d)
    assert '<aside id="toc">' in html and 'id="toc-nav"' in html, "左侧目录"
    for sid in ("s-overview", "s-ladder", "s-evo", "s-path", "s-forest", "s-fals",
                "s-final", "s-process", "s-events", "s-insights"):
        assert f'id="{sid}"' in html
    assert "</script><script>alert" not in html, "注入的 </ 已转义"
    blob = html.split('id="journey-data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(blob.replace("<\\/", "</"))["headline"].startswith("危险")
    assert "__JOURNEY_DATA__" not in html and "__JOURNEY_TITLE__" not in html


def test_generator_never_reads_final_test_labels():
    src = (SKILL / "scripts" / "_mvo" / "report_html.py").read_text(encoding="utf-8")
    assert "load_final_test_labels" not in src
    assert "labels/final_test" not in src and "final_test.parquet" not in src


def test_cmd_report_writes_html_by_default_and_respects_flag(tmp_path, monkeypatch):
    from _mvo import cmd_deliver
    run, *_ = _make_run(tmp_path)

    class FakeRB:
        def __init__(self, run_dir, spec, config):
            self.run_dir = run_dir

        def build(self):
            p = self.run_dir / "report" / "report.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# report\n", encoding="utf-8")
            return p

    monkeypatch.setattr(cmd_deliver, "ReportBuilder", FakeRB)
    args = types.SimpleNamespace(run=str(run), spec=None, no_html=False)
    assert cmd_deliver.cmd_report(args) == 0
    assert (run / "report" / "report.html").exists()
    assert "report.html" in (run / "report" / "report.md").read_text(encoding="utf-8")
    (run / "report" / "report.html").unlink()
    assert cmd_deliver.cmd_report(types.SimpleNamespace(run=str(run), spec=None, no_html=True)) == 0
    assert not (run / "report" / "report.html").exists()

    import _mvo.report_html as rh
    monkeypatch.setattr(rh, "build_journey_html", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cmd_deliver.cmd_report(args) == 0, "HTML 失败不影响 report 退出码"


def test_cli_flag_parsed():
    sys.path.insert(0, str(SKILL / "scripts"))
    import modelevo as cli
    ns = cli.build_parser().parse_args(["report", "--run", "x", "--no-html"])
    assert ns.no_html is True


@pytest.mark.skipif(NODE is None, reason="node 不可用")
@pytest.mark.parametrize("metric,final", [("rmse", True), ("auc", True), ("rmse", False)])
def test_template_executes_in_dom_stub(tmp_path, metric, final):
    run, *_ = _make_run(tmp_path, metric=metric, final=final, extras=final)
    d, _ = _collect(run)
    p = tmp_path / "r.html"
    p.write_text(render_html(d), encoding="utf-8")
    js = p.read_text(encoding="utf-8").split("<script>", 1)[1].rsplit("</script>", 1)[0]
    (tmp_path / "t.js").write_text(js, encoding="utf-8")
    assert subprocess.run([NODE, "--check", str(tmp_path / "t.js")]).returncode == 0
    r = subprocess.run([NODE, str(STUB), str(p)], capture_output=True, text=True, timeout=60)
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["error"] is None, out["error"]
    assert out["ok"], out["bad_attrs"][:5]
    for sec in ("c-ladder", "c-evo", "c-path", "c-forest", "toc-nav"):
        assert out["sections"][sec] > 0, sec
    if not final:
        assert "s-insights" in out["hidden"] and out["sections"]["c-cap"] == 0
