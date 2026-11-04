"""v2.13.4 修复批次回归测试(harness bug 修复)。

事故来源:run `example-seq-v3-v2.13.3.1` / `req-t7-drawdown-v21331`
(2026-09-22 实测)。该 run 在**已经确认的最强特征面上从未做过模型轴
探索**,Agent 据此判定"模型方向无潜力"并进入收尾交付 —— 交付的模型
停在 baseline 默认超参上。逐层归因后发现四处独立缺陷,本测试逐一
锁定。

1. Bug A  HPO 创建路径丢失父代 transform 加宽面
   `planner._materialize_command` 对 hpo 动作发 `--hpo-parent <id>`,
   该路径(模板渲染)只继承父代 **config.features 声明**;父代的加宽列
   是**代码 transform 的运行时产物** —— 于是"在冠军特征面上做 HPO"
   静默跑在基础面上。实测对照:

     | trial                | 创建路径      | n_features   |
     |----------------------|---------------|--------------|
     | v2.13.3.1 trial-018  | --hpo-parent  | 1500(基础)  |
     | v2.13.0   trial-014  | --from-parent | 1887(加宽)  |

   同一份 planner 命令,先行 run 之所以拿到加宽面,是靠 Agent 手工
   偏离命令 —— 正确性依赖操作者,这就是缺陷本身。
   修法:创建期加特征面守卫(`_hpo_parent_face_risk`),父代含
   transform 加宽列时拒绝 `--hpo-parent` 并给出 `--from-parent` 替代
   命令;命令生成侧保持 `--hpo-parent`(它仍是合法路径),由守卫决定
   拒绝。

2. Bug B  acquisition 评分量纲二次错配
   D3 只把 `expected_gain/uncertainty` 除以 MDE,`information_gain/
   novelty` 未除 —— 归一化做了一半,常数项被放大 1000 倍:

     信息源候选: 1.2 + 0.8 + 0.5×500.0 = 252.0
     HPO 候选:   0.3 + 0.0 + 0.5×300.0 = 150.1

   `uncertainty=0.5`(手设常数)被当作 500×MDE 的"实测不确定性",于是
   任何信息源候选恒压过任何模型候选,模型/HPO 轴在整个搜索期实际
   不可达。修法:实测值与常数先验分字段(`uncertainty` vs
   `uncertainty_prior`),各自用同源刻度入 score;`_base` 另加量纲
   守卫兜底(`uncertainty_gate_mde`)。

3. Bug C  冠军已消费的信息源被反复提名
   冠军 trial-008 已是"seq_d365/seq_gap_d365 全宽解析"(sq_ 374 +
   sqgap_ 13),gen-04 仍提名 `info:seq_d365:window` /
   `info:seq_gap_d365:recency`,复制父代代码后列集完全相同 →
   Δparent 恒为 0,两个槽位烧 15 分钟零产出。信息清单把粒度记为
   `partial`(因为 planner 自己提名过),形成提名→记 partial→继续
   提名的自反馈。修法:按冠军模型输入列面判定源是否已消费
   (`_source_already_consumed_by_champion`),已消费则该候选
   information_gain 归零。

4. Bug D  HPO inner 搜索与构造期早停参数冲突
   harness 对**任何** hpo 路径注入 `_HPO_ALGO` → `build_gbm` 对
   xgboost **构造期**注入 `early_stopping_rounds`;而"自带四件套"的
   冠军代码其 `_hpo_fit_inner` 做裸 `model.fit(X, y)`(无 eval_set)
   → `ValueError: Must have at least 1 validation dataset for early
   stopping`,trial-019 实测 25 秒 FAILED。修法:父代码自带 hpo 符号
   时也走模板段整段嫁接(与交付同机制),嫁接后自检
   (`_validate_grafted_hpo_deps`)不一致则创建期拒绝。
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import pytest

SKILL = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

from runtime.config import Config  # noqa: E402
from runtime.planner import AutonomousPlanner  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402


def _spec() -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v2134-fixes", "task_name": "v2134-fixes",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
        "resource_budget": {"max_trials": 30},
    })


# ---------------------------------------------------------------------------
# Bug B: acquisition 量纲 —— 实测与常数先验分刻度
# ---------------------------------------------------------------------------
def test_b_uncertainty_scales_are_not_mixed(tmp_path: Path):
    """手设常数不得以 MDE 单位进入 score。

    事故形态:inventory 候选 `information_gain=1.2, novelty=0.8,
    uncertainty=0.5` 得 252.0 分,而冠军面 HPO 候选(实测正先验
    0.75×MDE)只得 150.1 分 —— 常数 0.5 被当作 500×MDE 的实测值。
    """
    cfg = Config.load(skill_config=False)
    mde = float(cfg.get("evaluator.minimum_meaningful_improvement", 0.001))
    planner = AutonomousPlanner(tmp_path, _spec(), cfg)

    # 信息源候选的形态:三个手设常数 + 零实测
    inventory_like = planner._base(
        action="explore", family="info:src:window",
        hypothesis="h", rationale="r", falsification="f",
        information_gain=1.2, novelty=0.8,
        uncertainty=0.0, uncertainty_prior=0.5,
        cost_hours=2.5)
    # 冠军面 HPO 的形态:实测正先验 + 小常数
    hpo_like = planner._base(
        action="hpo", family="model:hpo",
        hypothesis="h", rationale="r", falsification="f",
        information_gain=0.3, novelty=0.0,
        expected_gain=0.75 * mde,
        uncertainty=0.0, uncertainty_prior=0.3,
        cost_hours=2.0)

    a_inv = inventory_like["acquisition"]
    a_hpo = hpo_like["acquisition"]

    # 常数不许被换算成 MDE 单位:prior 档的 uncertainty_mde 存在但不入分
    assert a_inv["uncertainty_scale"] == "measured"
    assert a_inv["uncertainty_prior"] == pytest.approx(0.5)
    assert a_inv["uncertainty_mde"] == pytest.approx(0.0)

    # 手算复核(v2.15.x 公式):
    # score = gain_mde + const_score × roi_mult - cost_penalty
    # inv: gain=0 → gain_mde=0, gain_lcb=0 → roi_mult=sigmoid(0)=0.5
    #   const_score = info+novelty+prior+0.5*u = 1.2+0.8+0.5 = 2.5
    #   score = 0 + 2.5×0.5 - 2.5×0.15 = 1.25 - 0.375 = 0.875
    # hpo: gain=0.75×MDE → gain_mde=0.75, gain_lcb=0.75×MDE → roi_mult=sigmoid(0.75)≈0.679
    #   const_score = info+novelty+prior = 0.3+0+0.3 = 0.6
    #   score = 0.75 + 0.6×roi_mult - 2.0×0.15
    # 容差 1e-5:roi_multiplier 落盘时 round(6) 与中间值有微小差
    assert a_inv["score"] == pytest.approx(
        a_inv["expected_gain_mde"]
        + (1.2 + 0.8 + 0.5) * a_inv["roi_multiplier"] - 2.5 * 0.15,
        abs=1e-5)
    assert a_hpo["score"] == pytest.approx(
        a_hpo["expected_gain_mde"]
        + (0.3 + 0.0 + 0.3) * a_hpo["roi_multiplier"] - 2.0 * 0.15,
        abs=1e-5)
    # 关键不变量:两者必须落在**同一量级**(O(1))。
    # 修复前:inventory=252.0 / hpo=150.1 —— 常数 0.5 被当作 500×MDE
    # 的实测不确定性,整个模型轴被永久压在信息源候选之下;两个分数
    # 虽"都很大",但排序语义完全由被误放大的常数决定。
    assert a_inv["score"] < 10.0
    assert a_hpo["score"] < 10.0
    # 且实测正先验确实进了分(0.75×MDE → 0.75 分)
    assert a_hpo["expected_gain_mde"] == pytest.approx(0.75)
    # 两者差距从 102 分缩到 1.1 分以内 —— 模型轴重新变得可选
    assert abs(a_inv["score"] - a_hpo["score"]) < 1.2


def test_b_gate_demotes_constant_passed_as_uncertainty(tmp_path: Path):
    """量纲守卫:误传到 `uncertainty` 的常数被降为 prior 档。

    契约必须可执行 —— 只靠文档约定"uncertainty 传实测值"不够,旧调用
    方(或第三方)传 0.5 时不能让整条模型轴被永久压制。
    """
    cfg = Config.load(skill_config=False)
    planner = AutonomousPlanner(tmp_path, _spec(), cfg)
    # 误传:把 0.5 这个常数塞进 uncertainty(实测通道)
    bad = planner._base(
        action="explore", family="feat:x",
        hypothesis="h", rationale="r", falsification="f",
        information_gain=1.2, novelty=0.8,
        uncertainty=0.5, cost_hours=0.5)
    a = bad["acquisition"]
    assert a["uncertainty_scale"] == "prior"
    # v2.15.x 乘性公式:prior 档 uncertainty 不进 measured 通道
    # gain=0,std=0 → gain_lcb=0 → roi_mult=sigmoid(0)=0.5
    # base_score = info+novelty+prior = 1.2+0.8+0.5 = 2.5
    # score = 2.5 * 0.5 - 0.5*0.15 = 1.25 - 0.075 = 1.175
    assert a["score"] == pytest.approx(1.175, abs=1e-5)
    assert "prior" in a["uncertainty_scale"]

    # 真正的实测值(量级 ~0.5×MDE)仍走 measured 档
    good = planner._base(
        action="exploit", family="feat:y",
        hypothesis="h", rationale="r", falsification="f",
        expected_gain=0.0004, uncertainty=0.0004,
        cost_hours=0.5)
    b = good["acquisition"]
    assert b["uncertainty_scale"] == "measured"
    assert b["uncertainty_mde"] == pytest.approx(0.4)
    # v2.15.x 公式: score = gain_mde + const*roi_mult - cost_penalty
    # gain=0.0004,std=0.0004 → gain_lcb=0.0004-1.28*0.0004=-1.12e-4
    # roi_arg=-0.112 → roi_mult=sigmoid(-0.112)≈0.472
    # const = 0.5*u_mde = 0.5*0.4 = 0.2; gain_mde=0.4
    # score = 0.4 + 0.2*roi_mult - 0.5*0.15
    assert b["score"] == pytest.approx(
        b["expected_gain_mde"]
        + (0.5 * b["uncertainty_mde"]) * b["roi_multiplier"] - 0.5 * 0.15,
        abs=1e-5)


def test_b_real_run_candidates_are_comparable(tmp_path: Path):
    """跨类可比:信息源候选与冠军面 HPO 候选必须能与信息常数同台比较。

    这是事故的直接形态 —— 修复前 inventory=252.0 / model:hpo=150.1;
    修复后两者应落在同一量级(个位数)。
    """
    cfg = Config.load(skill_config=False)
    planner = AutonomousPlanner(tmp_path, _spec(), cfg)
    inv = planner._base(
        action="explore", family="info:seq:window",
        hypothesis="h", rationale="r", falsification="f",
        information_gain=1.2, novelty=0.8,
        uncertainty=0.0, uncertainty_prior=0.5, cost_hours=2.5)
    hpo = planner._base(
        action="hpo", family="model:hpo",
        hypothesis="h", rationale="r", falsification="f",
        information_gain=0.3, uncertainty=0.0, uncertainty_prior=0.3,
        expected_gain=0.00075, cost_hours=2.0)
    assert inv["acquisition"]["score"] < 10.0
    assert hpo["acquisition"]["score"] < 10.0
    assert inv["acquisition"]["score"] > 0
    assert hpo["acquisition"]["score"] > 0


# ---------------------------------------------------------------------------
# Bug C:冠军已消费的信息源不再被反复提名
# ---------------------------------------------------------------------------
def _write_champion(tmp_path: pathlib.Path, *,
                    code: str = "", n_features: int = 1887,
                    new_cols: int = 387,
                    family_sizes: dict | None = None,
                    feature_columns: list | None = None,
                    algorithm: str = "xgboost",
                    trial_id: str = "trial-008-champion",
                    parent_columns: list | None = None) -> None:
    """落一个最小的"冠军 trial"桩(planner 只读 code/metrics/model)。

    v2.19.0:`parent_columns` 给定时同时落一个父代桩(代码不引用任何源列),
    冠军的 parent_trial_ids 指向它 —— "已消费"判定改为 trial 产物口径
    (代码字面量首次引用源列 + 相对父代新增的列仍在冠军输入),需要父代列集。
    """
    parents = []
    if parent_columns is not None:
        pid = "trial-007-parent"
        pdir = tmp_path / "trials" / pid
        (pdir / "code").mkdir(parents=True, exist_ok=True)
        (pdir / "model").mkdir(parents=True, exist_ok=True)
        (pdir / "code" / "train.py").write_text("def main():\n    pass\n",
                                                encoding="utf-8")
        (pdir / "model" / "feature_columns.json").write_text(
            json.dumps(parent_columns), encoding="utf-8")
        (pdir / "trial.yaml").write_text(json.dumps({
            "trial_id": pid, "algorithm": algorithm,
            "search_action": "explore", "generation": 1,
            "data_view": "v1", "changed_modules": ["feature"],
            "parent_trial_ids": [], "status": "SUCCEEDED",
            "evaluation": {"verdict": "accepted",
                           "integrity": {"valid": True},
                           "primary_metric": {"name": "auc",
                                              "value": 0.7700}},
        }), encoding="utf-8")
        parents = [pid]
    tdir = tmp_path / "trials" / trial_id
    (tdir / "code").mkdir(parents=True, exist_ok=True)
    (tdir / "model").mkdir(parents=True, exist_ok=True)
    (tdir / "code" / "train.py").write_text(code, encoding="utf-8")
    (tdir / "metrics.json").write_text(json.dumps({
        "n_features": n_features,
        "widen_stats": {"n_new_cols": new_cols,
                        "family_sizes": family_sizes or
                        {"sq_": 374, "sqgap_": 13}},
    }), encoding="utf-8")
    if feature_columns is not None:
        (tdir / "model" / "feature_columns.json").write_text(
            json.dumps(feature_columns), encoding="utf-8")
    (tdir / "trial.yaml").write_text(json.dumps({
        "trial_id": trial_id, "algorithm": algorithm,
        "search_action": "data_view", "generation": 2,
        "data_view": "v1", "changed_modules": ["feature"],
        "parent_trial_ids": parents,
        "status": "SUCCEEDED",
        "evaluation": {"verdict": "accepted", "integrity": {"valid": True},
                       "primary_metric": {"name": "auc", "value": 0.7818}},
    }), encoding="utf-8")


def test_c_source_consumed_detection(tmp_path: Path):
    """冠军代码首次引用某源、且其相对父代新增的列在模型输入里 ⇒ 该源已消费。

    v2.19.0 语义更新:旧版按 `{seq_d365: sq_}` 前缀表判定(数据集特定,已删);
    现按 trial 产物判定(代码字面量 + 列集差集),派生列叫什么都无所谓。
    """
    _write_champion(
        tmp_path,
        code=("import pathlib\nSRC = ['seq_d365', 'seq_gap_d365']\n"
              "X_tr, X_va, _s, _i = widen_cached('t', a, b, c, d)\n"),
        feature_columns=["sq_ev_LOAN_ENTER_c365", "sq_db_ndays",
                         "sqgap_ev_mean", "offline_inner_x"],
        parent_columns=["offline_inner_x"])
    cfg = Config.load(skill_config=False)
    planner = AutonomousPlanner(tmp_path, _spec(), cfg)
    assert planner._source_already_consumed_by_champion("seq_d365", "window")
    assert planner._source_already_consumed_by_champion(
        "seq_gap_d365", "recency")
    # 代码没引用的源不算已消费(不再按前缀猜)
    assert not planner._source_already_consumed_by_champion(
        "prime_cust_mob6", "aggregation")


def test_c_source_not_consumed_when_champion_has_no_widen(tmp_path: Path):
    """冠军没有加宽 transform(基础面冠军)⇒ 不得判"已消费"。"""
    _write_champion(tmp_path, code="def main():\n    pass\n",
                    n_features=1500, new_cols=0, family_sizes={},
                    feature_columns=["offline_inner_x", "dw_0006_used_amt"])
    cfg = Config.load(skill_config=False)
    planner = AutonomousPlanner(tmp_path, _spec(), cfg)
    assert not planner._source_already_consumed_by_champion(
        "seq_d365", "window")


def test_c_consumed_source_loses_information_gain(tmp_path: Path):
    """已消费源的候选 information_gain 归零(仍可竞争,但不再虚高)。"""
    (tmp_path / "sample-profile").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sample-profile" / "information-inventory.yaml").write_text(
        "sources:\n"
        "- source_type: reparse\n"
        "  column: seq_d365\n"
        "  consumption_status: partially_consumed\n"
        "  granularity_space:\n"
        "  - {dimension: window, hint: 多窗计数, state: partial}\n",
        encoding="utf-8")
    _write_champion(
        tmp_path,
        code=("raw = df['seq_d365']\n"
              "X_tr, X_va, _s, _i = widen_cached('t', a, b, c, d)\n"),
        feature_columns=["sq_ev_LOAN_ENTER_c365", "sqgap_ev_mean"],
        parent_columns=["offline_inner_x"])
    cfg = Config.load(skill_config=False)
    planner = AutonomousPlanner(tmp_path, _spec(), cfg)
    cands = planner._inventory_candidates(planner._parent())
    assert cands, "应产出信息源候选"
    for c in cands:
        assert c["acquisition"]["information_gain"] == 0.0
        assert "已被现有冠军" in c["acquisition"]["prior_note"]


# ---------------------------------------------------------------------------
# Bug D / A:HPO 机制与特征面守卫
# ---------------------------------------------------------------------------
def test_d_grafted_deps_detects_bare_fit(tmp_path: Path):
    """`_hpo_fit_inner` 仍是裸 fit 时,自检必须报出来。

    事故形态:冠军的 `_hpo_fit_inner` 调 `_lgbm_fit_safe`(裸 fit),
    而 `build_gbm` 对 xgboost 注入构造期早停 → inner 第一棵树即抛
    "Must have at least 1 validation dataset for early stopping"。
    """
    import modelevo as M

    tdir = tmp_path / "code"
    tdir.mkdir(parents=True)
    (tdir / "train.py").write_text(
        "def _hpo_fit_inner(params, X_fit, y_fit, X_in, y_in, seed):\n"
        "    m = _build_model(params, seed, len(X_fit))\n"
        "    _lgbm_fit_safe(m, X_fit, y_fit)\n"
        "    return 0\n\n"
        "def _build_model(params, seed, n=0):\n"
        "    from runtime.hpo import build_gbm\n"
        "    return build_gbm('xgboost', params, seed, 'x', n, False)\n",
        encoding="utf-8")

    class _T:
        trial_dir = str(tmp_path)
        trial_id = "trial-900-g09-xgboost-9999"
        data = {"algorithm": "xgboost"}

    gap = M._validate_grafted_hpo_deps(_T())
    assert gap, "裸 fit + 构造期早停必须被自检拦下"
    assert any("_fit_with_es" in g for g in gap)


def test_d_aligned_inner_passes_selfcheck(tmp_path: Path):
    """`_hpo_fit_inner` 走 `_fit_with_es` 时自检通过(修好的形态)。"""
    import modelevo as M

    tdir = tmp_path / "code"
    tdir.mkdir(parents=True)
    (tdir / "train.py").write_text(
        "def _hpo_fit_inner(params, X_fit, y_fit, X_in, y_in, seed):\n"
        "    m = _build_model(params, seed, len(X_fit))\n"
        "    m = _fit_with_es(m, X_fit, y_fit)\n"
        "    return 0\n\n"
        "def _fit_with_es(model, X, y, es_fraction=0.1, rounds=None):\n"
        "    return model\n",
        encoding="utf-8")

    class _T:
        trial_dir = str(tmp_path)
        trial_id = "trial-901-g09-xgboost-9999"
        data = {"algorithm": "xgboost"}

    assert M._validate_grafted_hpo_deps(_T()) == []


def test_a_hpo_parent_face_risk_flags_widened_parent():
    """`--hpo-parent` 指向加宽父代时必须报出特征面不等价风险。

    这是 Bug A 的核心判据:父代 metrics 的 widen_stats.n_new_cols>0
    就是"config 声明面表达不了它的特征面"的确定性标记。
    """
    import modelevo as M

    tmp = pathlib.Path(tempfile.mkdtemp())
    tdir = tmp / "trials" / "trial-800-g02-xgboost-0001"
    tdir.mkdir(parents=True)
    (tdir / "metrics.json").write_text(json.dumps({
        "n_features": 1887,
        "widen_stats": {"n_new_cols": 387,
                        "family_sizes": {"sq_": 374, "sqgap_": 13}},
    }), encoding="utf-8")

    class _T:
        trial_dir = str(tmp / "trials" / "trial-801-g05-xgboost-0002")
        trial_id = "trial-801-g05-xgboost-0002"
        data = {"algorithm": "xgboost"}

    risk = M._hpo_parent_face_risk(_T(), "trial-800-g02-xgboost-0001")
    assert risk, "加宽父代 + --hpo-parent 必须报风险"
    assert "387" in risk
    assert "--from-parent" in risk


def test_a_hpo_parent_face_risk_silent_for_flat_parent():
    """基础面父代(`n_new_cols=0`)不报风险 —— 该路径本来适配。"""
    import modelevo as M

    tmp = pathlib.Path(tempfile.mkdtemp())
    tdir = tmp / "trials" / "trial-802-g02-lightgbm-0001"
    tdir.mkdir(parents=True)
    (tdir / "metrics.json").write_text(json.dumps({
        "n_features": 1493, "widen_stats": {"n_new_cols": 0},
    }), encoding="utf-8")

    class _T:
        trial_dir = str(tmp / "trials" / "trial-803-g05-xgboost-0002")
        trial_id = "trial-803-g05-xgboost-0002"
        data = {"algorithm": "xgboost"}

    assert M._hpo_parent_face_risk(
        _T(), "trial-802-g02-lightgbm-0001") is None


def test_a_hpo_parent_face_risk_silent_when_metrics_missing():
    """父代 metrics 读不到时不阻断(保守:宁可多跑一次)。"""
    import modelevo as M

    tmp = pathlib.Path(tempfile.mkdtemp())

    class _T:
        trial_dir = str(tmp / "trials" / "trial-804-g05-xgboost-0002")
        trial_id = "trial-804-g05-xgboost-0002"
        data = {"algorithm": "xgboost"}

    assert M._hpo_parent_face_risk(_T(), "trial-999-missing") is None


# ---------------------------------------------------------------------------
# 交叉:plan() 不得再让模型轴无声消失
# ---------------------------------------------------------------------------
def test_plan_records_model_axis_when_cut_off(tmp_path: Path):
    """模型轴候选被 slots 截断时必须落盘留痕。

    "被截断 ≠ 已探索":事故里 Agent 只看批次内容,读不出"模型轴候选
    存在但没进批次",于是判定模型方向无潜力。
    夹具:足够多的信息源候选填满唯一槽位(默认 `_spec()` 无 inventory
    文件,故手工落一份多源清单),使 model 轴必被截断。
    """
    (tmp_path / "sample-profile").mkdir(parents=True, exist_ok=True)
    (tmp_path / "sample-profile" / "information-inventory.yaml").write_text(
        "sources:\n"
        "- source_type: reparse\n"
        "  column: seq_d365\n"
        "  consumption_status: partially_consumed\n"
        "  granularity_space:\n"
        "  - {dimension: window, hint: w, state: partial}\n"
        "  - {dimension: recency, hint: r, state: partial}\n",
        encoding="utf-8")
    cfg = Config.load(skill_config=False)
    planner = AutonomousPlanner(tmp_path, _spec(), cfg)
    batch = planner.plan(slots=1)
    assert "model_axis_not_selected" in batch, (
        f"模型轴被截断却无留痕: {sorted(batch)}")
    note = batch["model_axis_not_selected"]
    assert note["families"], "应记录被截断的模型族"
    assert "截断" in note["reason"]
