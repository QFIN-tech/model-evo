"""v15.2 修复批次单元验证:分组置换零填充伪影(unverifiable 语义)。

覆盖三处修复:
1. group_importance:评估帧缺列时,全缺失族判 families_unverifiable
   (无证据,非死),部分缺失族标注 zero_fill_share;coverage 按真实
   交集计(不再被 0 填充虚报 1.0);trial 自存
   model/validation_input.parquet 时优先消费(eval_frame_source);
2. lint 规则 6b:检测到特征加宽(assign/insert/concat/df["col"]=)但
   未调用 dump_eval_frame → 拒绝;基础特征面 trial 不触发;
3. CLI 引用展开 @trial:group_importance:families_alive ∪
   families_unverifiable(unverifiable=无证据≠死,不得被白名单筛掉)。

实测背景(gen-02 回写):v14.1 对评估帧缺失列补 0,常数列打乱不掉分
→ delta≡0 → 8 个加宽族全部伪判 dead 而 coverage 报 1.0;真帧重算
seq_ delta=+0.0049(alive),方向生死裁决被系统性污染。
运行:python tests/test_v15dot2_fixes.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))

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


class _FakeModel:
    """可 pickle 的常数模型:score = f(可用列的线性组合)。

    用于验证置换 delta 的方向性:对承载信号的列打乱 → AUC 掉;
    对全 0 常数列打乱 → delta 恒 0(v14.1 伪影的根源)。
    """

    def __init__(self, weights: dict[str, float]):
        self.weights = weights
        self.feature_columns = list(weights)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        s = np.zeros(len(X))
        for c, w in self.weights.items():
            if c in X.columns:
                s = s + w * np.nan_to_num(
                    pd.to_numeric(X[c], errors="coerce").fillna(0).values)
        return s

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # sklearn 口:返回 (n,2),列 1 为正类概率(用 sigmoid 近似)
        z = self.predict(X)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _make_trial(tmp: Path, tid: str, weights: dict[str, float],
                feature_cols: list[str],
                eval_frame: pd.DataFrame | None = None) -> Path:
    """构造最小 trial 目录:model.joblib + feature_columns.json(+
    可选 validation_input.parquet)。"""
    import joblib

    tdir = tmp / "trials" / tid
    mdir = tdir / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(_FakeModel(weights), mdir / "model.joblib")
    (mdir / "feature_columns.json").write_text(
        json.dumps(feature_cols), encoding="utf-8")
    if eval_frame is not None:
        eval_frame.to_parquet(mdir / "validation_input.parquet",
                              index=False)
    return tdir


# ---------------------------------------------------------------------------
# 1) group_importance:unverifiable / zero_fill_share / coverage / 自存帧
# ---------------------------------------------------------------------------

def test_group_importance_unverifiable():
    print("v15.2 group_importance 零填充伪影:")
    tmp = Path(tempfile.mkdtemp())
    try:
        from runtime.group_importance import run_group_importance

        rng = np.random.default_rng(7)
        n = 800
        # 信号列 f_sig:与标签强相关;widened 族两列承载信号;
        # dead 族一列纯噪声
        f_sig = rng.normal(size=n)
        y = (f_sig + rng.normal(scale=0.8, size=n) > 0).astype(int)
        base = pd.DataFrame({
            "f_sig": f_sig,
            "f_noise": rng.normal(size=n),
        })
        widened = pd.DataFrame({
            "w_a": f_sig * 0.5 + rng.normal(scale=0.2, size=n),
            "w_b": f_sig * 0.3 + rng.normal(scale=0.3, size=n),
            "d_x": rng.normal(size=n),
        })
        weights = {"f_sig": 1.0, "f_noise": 0.05,
                   "w_a": 0.8, "w_b": 0.5, "d_x": 0.03}
        fc = list(weights)

        # -- 场景 A:基础帧(缺全部 w_/d_ 列)—— 旧行为补 0 伪判 dead
        tdir_a = _make_trial(tmp, "trial-a", weights, fc)
        summary = run_group_importance(
            tdir_a, base.copy(), y, "auc", n_repeat=3)
        check("A: 全缺失族判 unverifiable(非 dead)",
              summary is not None
              and set(summary["families_unverifiable"]) == {"w", "d"},
              str(summary and summary["families_unverifiable"]))
        check("A: unverifiable 不进 families_dead",
              "w" not in summary["families_dead"]
              and "d" not in summary["families_dead"])
        check("A: coverage 按真实交集计(2/5,不虚报 1.0)",
              abs(summary["feature_coverage"] - 0.4) < 1e-6,
              str(summary["feature_coverage"]))
        check("A: filled_constant_columns 记录填充列数",
              summary["filled_constant_columns"] == 3)
        check("A: eval_frame_source 标注基础帧",
              summary["eval_frame_source"] == "splits_base")
        check("A: 基础族正常裁决(f alive)",
              "f" in summary["families_alive"],
              str(summary["families_alive"]))

        # -- 场景 B:trial 自存真实评估帧 → 加宽族得到真实裁决
        eval_frame = pd.concat([base, widened], axis=1)
        tdir_b = _make_trial(tmp, "trial-b", weights, fc,
                             eval_frame=eval_frame)
        summary_b = run_group_importance(
            tdir_b, base.copy(), y, "auc", n_repeat=3)
        check("B: 自存帧被消费(eval_frame_source)",
              summary_b["eval_frame_source"]
              == "trial_saved_validation_input",
              str(summary_b["eval_frame_source"]))
        check("B: 加宽族 w 判 alive(真帧上的真实信号)",
              "w" in summary_b["families_alive"],
              str(summary_b["families_alive"]))
        check("B: unverifiable 为空",
              summary_b["families_unverifiable"] == [])
        check("B: coverage = 1.0(全部列真实在场)",
              abs(summary_b["feature_coverage"] - 1.0) < 1e-6)
        check("B: CSV 落盘且含全部族",
              (tdir_b / "feature" / "group_importance.csv").exists()
              and len(summary_b["families_alive"])
              + len(summary_b["families_dead"]) == 3)

        # -- 场景 C:自存帧行数不符 → 弃用回退基础帧(加宽族 unverifiable)
        tdir_c = _make_trial(tmp, "trial-c", weights, fc,
                             eval_frame=eval_frame.iloc[:100])
        summary_c = run_group_importance(
            tdir_c, base.copy(), y, "auc", n_repeat=3)
        check("C: 行数不符的自存帧被弃用",
              summary_c["eval_frame_source"] == "splits_base"
              and "w" in summary_c["families_unverifiable"],
              str(summary_c["eval_frame_source"]))

        # -- 场景 D:部分缺失(帧里有 w_a 无 w_b)→ zero_fill_share 标注
        base_d = pd.concat([base, widened[["w_a"]]], axis=1)
        tdir_d = _make_trial(tmp, "trial-d", weights, fc)
        summary_d = run_group_importance(
            tdir_d, base_d, y, "auc", n_repeat=3)
        check("D: 部分缺失族不被移出裁决(仍可对照)",
              "w" not in summary_d["families_unverifiable"])
        csv = pd.read_csv(tdir_d / "feature" / "group_importance.csv")
        w_row = csv[csv["family"] == "w"]
        check("D: zero_fill_share 按族标注(0.5)",
              len(w_row) == 1 and abs(float(w_row["zero_fill_share"].iloc[0])
                                     - 0.5) < 1e-6,
              str(w_row.to_dict(orient="records")))
        check("D: d 族(全缺失)仍 unverifiable",
              "d" in summary_d["families_unverifiable"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2) lint 规则 6b:加宽 trial 必须落盘评估帧
# ---------------------------------------------------------------------------

def _load_lint():
    p = SKILL / "assets" / "lint_trial.py"
    spec_m = importlib.util.spec_from_file_location("lintm", str(p))
    m = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(m)
    return m


def test_lint_rule_6b():
    print("v15.2 lint 规则 6b(加宽 trial 评估帧门):")
    m = _load_lint()
    # 加宽(assign)且无 dump_eval_frame → 违规
    errs = m.check_model_persistence(
        "def fit(X_tr, X_va):\n"
        "    X_tr = X_tr.assign(seq_cnt=X_tr['a'] + 1)\n"
        "    X_va = X_va.assign(seq_cnt=X_va['a'] + 1)\n"
        "    dump_model(model, X_tr.columns)\n"
        "    return model.predict(X_va)\n")
    check("assign 加宽无评估帧 → 规则 6b 拒绝",
          any("6b" in e for e in errs), str(errs))
    # 加宽但有 dump_eval_frame → 通过
    errs = m.check_model_persistence(
        "def fit(X_tr, X_va):\n"
        "    X_va = X_va.assign(seq_cnt=X_va['a'] + 1)\n"
        "    dump_model(model, X_tr.columns)\n"
        "    dump_eval_frame(X_va)\n"
        "    return model.predict(X_va)\n")
    check("assign 加宽 + dump_eval_frame → 通过",
          not any("6b" in e for e in errs), str(errs))
    # 基础特征面(无加宽)→ 不触发 6b
    errs = m.check_model_persistence(
        "def fit(X_tr, X_va):\n"
        "    model.fit(X_tr)\n"
        "    dump_model(model, X_tr.columns)\n"
        "    return model.predict(X_va)\n")
    check("基础特征面 trial → 不触发 6b", not errs, str(errs))
    # 真实历史事故形态:insert + concat(axis=1)
    errs = m.check_model_persistence(
        "def transform_split(face, rep):\n"
        "    out = face.copy()\n"
        "    out.insert(0, 'seq_cnt_7', v)\n"
        "    out = pd.concat([out, gap_df], axis=1)\n"
        "    return out\n"
        "def fit(X_tr, X_va):\n"
        "    dump_model(model, X_tr.columns)\n"
        "    return model.predict(X_va)\n")
    check("insert/concat 加宽(gen-02 事故形态)→ 6b 拒绝",
          any("6b" in e for e in errs), str(errs))
    # 豁免标记
    errs = m.check_model_persistence(
        "def fit(X_tr, X_va):  # lint:skip-model-persist\n"
        "    X_va = X_va.assign(a2=X_va['a'] ** 2)\n"
        "    return model.predict(X_va)\n")
    check("豁免标记沿用(lint:skip-model-persist)",
          not errs, str(errs))
    # 模板渲染产物本体(含 dump_eval_frame 定义与调用)→ 通过
    from runtime.templates import TRIAL_TEMPLATE
    errs = m.check_model_persistence(TRIAL_TEMPLATE.format(
        trial_id="t", hypothesis="h", actions="a", run_dir="/tmp",
        target="y", model_type="binary_classification", treatment="",
        seed=1, skill_root="", body="pass\n"))
    check("TRIAL_TEMPLATE 渲染产物不误报 6b",
          not any("6b" in e for e in errs), str(errs))


# ---------------------------------------------------------------------------
# 3) CLI 引用展开:alive ∪ unverifiable
# ---------------------------------------------------------------------------

def test_cli_reference_expand():
    print("v15.2 CLI @trial:group_importance 展开口径:")
    m_spec = importlib.util.spec_from_file_location(
        "mcli", str(SKILL / "scripts" / "modelevo.py"))
    m = importlib.util.module_from_spec(m_spec)
    m_spec.loader.exec_module(m)
    tmp = Path(tempfile.mkdtemp())
    try:
        run = tmp / "run-r"
        tdir = run / "trials" / "trial-001"
        fdir = tdir / "feature"
        fdir.mkdir(parents=True)
        # trial 目录需含 config/trial.yaml 才会被识别为 trial 引用
        cdir = tdir / "config"
        cdir.mkdir()
        (cdir / "trial.yaml").write_text("trial_id: trial-001\n",
                                         encoding="utf-8")
        # 旧伪影形态:families_alive 只剩 base,加宽族全在 dead
        # (v14.1 0 填充伪判),但 families_unverifiable 记录了真因
        gi = {
            "families_alive": ["base"],
            "families_dead": ["w"],
            "families_unverifiable": ["w2"],
        }
        (fdir / "group_importance.json").write_text(
            json.dumps(gi), encoding="utf-8")
        derived = ["base_a", "base_b", "w_x", "w2_y"]
        got = m._load_features_ref("@trial-001:group_importance",
                                   run, derived)
        check("unverifiable 族并入白名单(无证据≠死)",
              got is not None and set(got) == {"base_a", "base_b", "w2_y"},
              str(got))
        # 全空(alive 与 unverifiable 均空)→ 引用失败报错
        gi2 = {"families_alive": [], "families_unverifiable": []}
        (fdir / "group_importance.json").write_text(
            json.dumps(gi2), encoding="utf-8")
        try:
            m._load_features_ref("@trial-001:group_importance",
                                 run, derived)
            check("全空引用显式报错", False, "no SystemExit")
        except SystemExit:
            check("全空引用显式报错", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_group_importance_unverifiable()
    test_lint_rule_6b()
    test_cli_reference_expand()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
