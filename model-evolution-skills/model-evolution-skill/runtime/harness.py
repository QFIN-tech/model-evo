"""Harness 协议 hash(§19.3):Harness 一旦变更即视为新协议。

trial recorder 在每个 Trial 记录 runtime/ 代码 hash;
若 hash 与 Run 初始化时不一致,告警并要求重新 baseline。

v14:铁律 10 的执行门从"拒绝运行"改为"标记 epoch、拒绝跨 epoch 比较"。
旧实现只有一条出路——手改 run-state.json 的 harness_hash(实测一次
运行被手改三次,改了什么/为什么改零记录,铁律名存实亡)。新增
`acknowledge-harness-change` 命令登记变更(diff 分类 + 理由 + epoch),
gate 的报错信息指向该命令。hash 本身保持全量 glob(白名单漏列会让
协议 hash 对行为变更不敏感,v12 的教训),逐文件 hash 只用于分类。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import HARNESS_PROTOCOL_VERSION

#: 纯管道模块:不产出任何落盘数字,改动不影响任何 trial 的可比性。
#: 本集合**不参与 harness_hash 计算**(v12 教训:白名单漏列=hash 盲区),
#: 只用于 acknowledge-harness-change 时给变更分类。
#: v2.11.9 订正:config.py **不在**本集合。它持有 DEFAULTS 里的评分口径
#: 常量(bootstrap_samples / noise_thresholds / shadow_gap_multiplier /
#: selection_shrinkage / slice_alpha / group_importance_repeat …);把
#: 它算作纯管道,会让「改 noise 阈值」被判成 plumbing → 按路径 1 直接登
#: 记、epoch 不变 → 新旧口径 trial 混排(正是本文件注释里「宁可误报不
#: 可漏报」要防的那一类)。config.py 的改动一律按 measurement 处理。
#: v2.17.2(B8):planner.py / planner_context.py / report.py 归入本集合。三者
#: 只产出"下一代提什么"的建议与人读报告,不写任何 trial 的落盘分数/裁决
#: (evaluator/metrics 不读它们),改动不影响历史 trial 的可比性;run4
#: opus5.5 每修一次 planner 都要手工 ack 分数影响。config.py 仍不在本集合。
#: 变更全部落在本集合时,run / run-generation 的 hash 门自动登记
#: (auto_acknowledgeable;context._auto_ack_plumbing),无需手工 ack。
#: v2.19.0:reference.py / ceiling.py / benchmark.py(等预算参照、天花板估计
#: 与 benchmark 汇总:只测量参照模型、读写 ceiling-estimate.json 与报告,不写
#: 任何 trial 的分数/裁决)与 profile_rules.py(只供 planner 产出候选建议)
#: 归入本集合。天花板估计会进 stop report 的建议性停止码,与 planner 同属
#: "建议"层,不改历史 trial 的可比性。
PLUMBING_MODULES = frozenset({
    "harness.py", "logging_utils.py", "preflight.py",
    "planner.py", "planner_context.py", "report.py",
    "reference.py", "ceiling.py", "benchmark.py", "profile_rules.py",
})


def harness_hash(runtime_dir: Path | None = None) -> str:
    """对 runtime/ 全部 .py 源码计算稳定 hash(内容级,与路径无关)。

    v12(P2):不再维护 _RUNTIME_FILES 白名单(旧列表引用了不存在的
    archive.py,且新增模块漏列会让协议 hash 对行为变更不敏感)——
    glob 全量 .py 即协议面本身。注释级改动会改变 hash;属已知行为,
    行为变更必须重新 baseline,宁可误报不可漏报。
    """
    runtime_dir = Path(runtime_dir) if runtime_dir else Path(__file__).parent
    digest = hashlib.sha256()
    files = sorted(p for p in runtime_dir.glob("*.py")
                   if p.name != "__pycache__")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def harness_file_hashes(runtime_dir: Path | None = None) -> dict[str, str]:
    """逐文件内容 hash;acknowledge-harness-change 据此算差异清单。

    旧 run-state 只有一个 16 位串,人工比对无从下手,只能手改 JSON;
    逐文件 hash 让"改了哪些文件"可分类,而不是一个不透明的整体。
    """
    runtime_dir = Path(runtime_dir) if runtime_dir else Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(runtime_dir.glob("*.py"))
            if p.name != "__pycache__"}


def classify_harness_change(before: dict, after: dict) -> dict:
    """文件级差异 → measurement / plumbing 两类。

    before 为空(旧 run 无 harness_files 键)时返回 unknown=True,
    调用方按"全部算 measurement"处理——宁可误报不可漏报。
    """
    if not before:
        return {"unknown": True, "changed": sorted(after),
                "measurement": sorted(after), "plumbing": []}
    names = set(before) | set(after)
    changed = [n for n in sorted(names) if before.get(n) != after.get(n)]
    return {"unknown": False, "changed": changed,
            "measurement": [n for n in changed
                            if n not in PLUMBING_MODULES],
            "plumbing": [n for n in changed if n in PLUMBING_MODULES]}


def auto_acknowledgeable(before: dict, after: dict) -> dict | None:
    """v2.17.2(B8):变更全部落在 PLUMBING_MODULES 时返回 diff(可自动登记),
    否则 None(旧 run 无逐文件 hash / 触及 measurement / 无差异)。"""
    diff = classify_harness_change(before, after)
    if diff.get("unknown") or diff["measurement"] or not diff["changed"]:
        return None
    return diff


def protocol_stamp(runtime_dir: Path | None = None) -> dict:
    return {
        "harness_protocol_version": HARNESS_PROTOCOL_VERSION,
        "harness_hash": harness_hash(runtime_dir),
        # v14:逐文件 hash 落 run-state,acknowledge 时才知道"改了哪些
        # 文件";旧行为只有一个 16 位串,人工比对无从下手,只能手改 JSON
        "harness_files": harness_file_hashes(runtime_dir),
    }


# ---------------------------------------------------------------------------
# v2.17.0(#22):acknowledge-harness-change --verify-against
# ---------------------------------------------------------------------------
#: 逐项比对的 evaluation.json 字段(主指标 / Δ / CI / SE)。存档里缺的字段
#: 不比(旧 trial 没有);存档有、重算没有 = 差异。
VERIFY_FIELDS = (
    "primary_metric.value",
    "delta.vs_parent",
    "delta.vs_baseline",
    "delta.vs_parent_paired_bootstrap.delta_mean",
    "delta.vs_parent_paired_bootstrap.delta_ci",
    "delta.vs_parent_paired_bootstrap.se",
    "delta.vs_parent_paired_bootstrap.z_score",
    "delta.vs_parent_paired_bootstrap.p_positive",
    "delta.vs_baseline_paired_bootstrap.delta_mean",
    "delta.vs_baseline_paired_bootstrap.delta_ci",
    "delta.vs_baseline_paired_bootstrap.se",
    "delta.vs_baseline_paired_bootstrap.z_score",
    "noise",
    "stability.bootstrap_ci",
    "stability.multi_seed_std",
)

_MISSING = object()


def _dig(d, dotted: str):
    node = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _same(a, b, rel_tol: float, abs_tol: float) -> bool:
    import math

    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) \
                or len(a) != len(b):
            return False
        return all(_same(x, y, rel_tol, abs_tol) for x, y in zip(a, b))
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    return math.isclose(fa, fb, rel_tol=rel_tol, abs_tol=abs_tol)


def compare_evaluations(stored: dict, fresh: dict, *,
                        rel_tol: float = 1e-9,
                        abs_tol: float = 1e-12) -> tuple[list[str], list[dict]]:
    """(参与比对的字段, 差异清单)。"""
    compared, diffs = [], []
    for f in VERIFY_FIELDS:
        a = _dig(stored, f)
        if a is _MISSING:
            continue
        compared.append(f)
        b = _dig(fresh, f)
        if b is _MISSING or not _same(a, b, rel_tol, abs_tol):
            diffs.append({"field": f, "stored": a,
                          "recomputed": None if b is _MISSING else b})
    return compared, diffs


def verify_trial_evaluation(run_dir: Path, spec, config, trial_id: str, *,
                            rel_tol: float = 1e-9,
                            abs_tol: float = 1e-12) -> dict:
    """用**当前** evaluator 重新评估已落盘预测(不重训),与 evaluation.json
    的主指标 / Δ / CI / SE 逐项比对。

    输入与 runner 评估段同源:视图标签面 validation、trial 的
    validation_score.parquet、父代(多父代逐个尝试,取与存档一致者)与
    baseline 的已落盘预测(完整性工件校验同 runner)、seed_scores/。
    只读:不写 evaluation.json、不改 trial 状态。
    """
    import numpy as np
    import pandas as pd

    from .data import SAMPLE_KEY_COLUMN, view_splits_dir
    from .evaluator import Evaluator
    from .logging_utils import load_json, load_yaml
    from .metrics import align_score_by_keys
    from .runner import baseline_view_compatible, load_parent_scores

    run_dir = Path(run_dir)
    tdir = run_dir / "trials" / trial_id
    out: dict = {"trial": trial_id, "consistent": False}
    try:
        rec = load_yaml(tdir / "trial.yaml") or {}
        stored = load_json(tdir / "evaluation.json")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"trial 记录/evaluation.json 不可读: {exc}"
        return out
    try:
        view = str(rec.get("data_view") or "v1")
        lab = pd.read_parquet(view_splits_dir(run_dir, view) / "labels"
                              / "validation.parquet")
        score_df = pd.read_parquet(tdir / "validation_score.parquet")
        if len(score_df) != len(lab):
            raise ValueError(f"预测 {len(score_df)} 行 != validation "
                             f"{len(lab)} 行")
        if SAMPLE_KEY_COLUMN in score_df.columns and \
                SAMPLE_KEY_COLUMN in lab.columns and not np.array_equal(
                    score_df[SAMPLE_KEY_COLUMN].astype(str).values,
                    lab[SAMPLE_KEY_COLUMN].astype(str).values):
            raise ValueError("预测样本键与 validation 标签面不一致")
        y = lab[spec.target_name].values
        score = score_df["score"].values
        tcol = spec.treatment_column
        treatment = lab[tcol].values if tcol and tcol in lab.columns else None
        parents = [str(p) for p in (rec.get("parent_trial_ids") or [])]
        st = load_json(run_dir / "run-state.json") if (
            run_dir / "run-state.json").exists() else {}
        bid = st.get("baseline_id")
        base_pack = load_parent_scores(run_dir, bid) if bid else None
        seed_dir = tdir / "seed_scores"
        ms = [pd.read_parquet(p)["score"].values
              for p in sorted(seed_dir.glob("*.parquet"))] \
            if seed_dir.is_dir() else None
        keys = score_df[SAMPLE_KEY_COLUMN].astype(str).values \
            if SAMPLE_KEY_COLUMN in score_df.columns else None
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"评估输入不可重建: {exc}"
        return out

    def _aligned(pack):
        """(score, comparable):与 runner 同口径按样本键对齐。"""
        if not pack or not isinstance(pack.get("validation"), pd.DataFrame):
            return None, True
        if not baseline_view_compatible(view, pack.get("data_view")):
            return pack["validation"]["score"].values, False
        df = pack["validation"]
        s = df["score"].values
        if keys is not None and SAMPLE_KEY_COLUMN in df.columns:
            s = align_score_by_keys(s, keys,
                                    df[SAMPLE_KEY_COLUMN].astype(str).values)
        return s, True

    base_score, base_ok = (None, True)
    if bid and bid != trial_id:
        base_score, base_ok = _aligned(base_pack)
    # v2.19.0:seed_variance_in_se 的 run 复评复用落盘的 σ_seed(它取决于
    # 评估当时的同配方复现组,事后重算会随新 trial 漂移)
    _ssd = None
    _dl = (stored or {}).get("delta") or {}
    for _k in ("vs_parent_paired_bootstrap", "vs_baseline_paired_bootstrap",
               "vs_anchor_paired_bootstrap"):
        _sig = (_dl.get(_k) or {}).get("sigma_seed")
        if _sig is not None:
            _ssd = {"sigma_seed_override": _sig,
                    "sigma_seed_source": (_dl.get(_k) or {}).get(
                        "sigma_seed_source"),
                    "sigma_seed_estimate": (_dl.get(_k) or {}).get(
                        "sigma_seed_estimate"),
                    "sigma_seed_estimate_source": (_dl.get(_k) or {}).get(
                        "sigma_seed_estimate_source")}
            break
    best = None
    evaluator = Evaluator(spec, config)
    for pid in (parents or [None]):
        try:
            pack = load_parent_scores(run_dir, pid) if pid else None
            p_score, p_ok = _aligned(pack) if pid else (None, True)
            fresh = evaluator.evaluate(
                y, score, treatment=treatment,
                parent_score=p_score if p_ok else None,
                baseline_score=base_score,
                baseline_comparable=base_ok,
                is_root_trial=not parents,
                confirm_trial=False,
                multi_seed_scores=ms,
                seed_sd_context=_ssd)
        except Exception as exc:  # noqa: BLE001
            cand = (pid, [{"field": "<evaluate>", "stored": None,
                           "recomputed": f"重算失败: {exc}"}], [])
        else:
            compared, diffs = compare_evaluations(
                stored, fresh, rel_tol=rel_tol, abs_tol=abs_tol)
            cand = (pid, diffs, compared)
        if best is None or len(cand[1]) < len(best[1]):
            best = cand
        if not cand[1]:
            break
    pid, diffs, compared = best
    out.update({"consistent": not diffs, "parent_used": pid,
                "data_view": view, "compared_fields": compared,
                "diffs": diffs})
    if not compared:
        out["consistent"] = False
        out["error"] = "evaluation.json 无可比对字段"
    return out
