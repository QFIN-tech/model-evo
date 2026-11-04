"""分桶报表(§12 附加交付):按 score 分位的全量分桶表,Excel + MD。

三种口径(由 spec.model_type 裁决):
- classification:分桶人数、样本占比、正样本数/占比、lift、累计正样本数、
  累计准确率、累计 lift、累计召回率、KS;
- regression:预测平均值、真实均值、Bias(预测−真实)、MAE、RMSE、中位数、
  最大值、最小值、label 求和、label 占比、累计 label、累计 label 占比;
- uplift:人群占比、T/C 人数、T/C 转化率、实际与预测 Uplift、增量人数、
  累计 Uplift、累计增量人数、95% CI。

产物(落盘到调用方目录):
- segment-report.xlsx  多 sheet(分桶表 + 汇总;openpyxl 缺失时降级 csv);
- segment-report.md    人类可读版(便于直接阅读/粘贴周报)。

排序口径:一律按 score 降序(桶 1 = 最高分段),与风险/增益报表惯例一致;
回归同样降序(高预测值人群在前,便于按头部圈人)。分箱沿用
segment_ranking 的 rank 均分口径,两套产物可互查。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import get_logger, now_iso
from .spec import TaskSpec

logger = get_logger(__name__)


def _bin_by_score(score: np.ndarray, n_bins: int) -> np.ndarray | None:
    """score 分位分箱(rank 化均分,并列值不跨段),返回桶号(0=最低段)。

    样本不足(段均 <10,与 segment_ranking 同口径)或 score 退化
    (唯一值 < n_bins)时返回 None。
    """
    score = np.asarray(score, dtype=float)
    n = len(score)
    if n_bins < 2 or n < n_bins * 10:
        return None
    if len(np.unique(score)) < n_bins:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    return (ranks * n_bins / n).astype(int).clip(0, n_bins - 1)


def _score_range(score: np.ndarray, mask: np.ndarray) -> str:
    """桶内 score 区间(降序展示:high~low;退化单值只显示一个数)。"""
    if not mask.any():
        return "—"
    lo, hi = float(score[mask].min()), float(score[mask].max())
    return f"{hi:.4f}" if abs(hi - lo) < 1e-12 else f"{hi:.4f}~{lo:.4f}"


def _fmt_md_table(df: pd.DataFrame, decimals: int = 4) -> list[str]:
    """DataFrame → Markdown 表格行(数值保留 decimals 位,异常值 —)。"""
    def _cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(_cell(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return [header, sep, *rows]


def classification_segment_table(y_true: np.ndarray, score: np.ndarray,
                                 n_bins: int = 10) -> pd.DataFrame | None:
    """分类分桶表(桶 1 = score 最高段,降序)。"""
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    bins = _bin_by_score(score, n_bins)
    if bins is None:
        return None
    n = len(y)
    total_pos = float(y.sum())
    base = total_pos / n if n else np.nan
    # 逐桶累计 KS:按 score 降序累积,KS = |累计正样本召回 − 累计负样本
    # "召回"|,分母是全量正/负样本数(CDF 差口径,与 metrics.ks_stat 一致);
    # 若误用截断分母(前 k 个样本)会退化成 |2×累计准确率−1|,
    # 单调且与全局 KS 不符
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    cum_pos_all = np.cumsum(ys)
    cum_neg_all = np.cumsum(1 - ys)
    if total_pos <= 0 or total_pos >= n:
        return None  # 单一类别无 KS 可言,整表不产出
    ks_by_rank = np.abs(cum_pos_all / total_pos
                       - cum_neg_all / (n - total_pos))
    # ks_at[k] = 取降序前 k 个样本的累计 KS;k=0 → 0
    ks_at = np.concatenate([[0.0], ks_by_rank])

    rows = []
    cum_n = cum_pos = 0
    seq = 0
    for b in range(n_bins - 1, -1, -1):
        m = bins == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        pos = float(y[m].sum())
        cum_n += cnt
        cum_pos += pos
        rate = pos / cnt
        seq += 1
        rows.append({
            "分桶": seq,
            "分数区间": _score_range(score, m),
            "分桶人数": cnt,
            "样本占比": cnt / n,
            "正样本数": int(round(pos)),
            "正样本占比": rate,
            "lift": rate / base if base else np.nan,
            "累计样本数": cum_n,
            "累计正样本数": int(round(cum_pos)),
            "累计准确率": cum_pos / cum_n,
            "累计lift": (cum_pos / cum_n / base) if base else np.nan,
            "累计召回率": cum_pos / total_pos if total_pos else np.nan,
            # KS:截至本桶(含)的降序累计 CDF 差;末桶 = 全局 KS
            "KS": float(ks_at[cum_n]),
        })
    return pd.DataFrame(rows)


def regression_segment_table(y_true: np.ndarray, y_pred: np.ndarray,
                             n_bins: int = 10) -> pd.DataFrame | None:
    """回归分桶表(按预测值降序;label 系列列名与交付口径一致)。

    **Bias 口径(v2.16.3,run4 regression D-014 表内第 4 条)**:``Bias`` 列
    = 桶内 **预测−真实** 的均值,与 ``evaluation.raw_scale.mean_bias``、
    ``report`` 的"均值偏置"同一符号(正 = 高估)。旧实现是 ``真实−预测``,
    同一份交付物里两处 Bias 符号相反(``segment-report.md`` 显示 −x,
    ``evaluation.raw_scale.mean_bias`` 显示 +x)。MAE/RMSE 不受影响(取绝对值
    /平方)。列名保留 ``Bias``(下游按列名解析,xlsx/md 表头一致),口径写在
    references/stage-playbook.md 与 task-type-regression.md。
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    bins = _bin_by_score(p, n_bins)
    if bins is None:
        return None
    n = len(y)
    total_label = float(y.sum())
    rows = []
    cum_label = 0.0
    cum_n = 0
    seq = 0
    for b in range(n_bins - 1, -1, -1):
        m = bins == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        yb, pb = y[m], p[m]
        d = yb - pb        # 残差(真实−预测):MAE/RMSE 用,与指标口径同号
        bias = pb - yb     # v2.16.3:报表 Bias 统一为 **预测−真实**
        lbl = float(yb.sum())
        cum_label += lbl
        cum_n += cnt
        seq += 1
        rows.append({
            "分桶": seq,
            "预测区间": _score_range(p, m),
            "分桶人数": cnt,
            "样本占比": cnt / n,
            "预测平均值": float(pb.mean()),
            "真实均值": float(yb.mean()),
            "Bias": float(bias.mean()),
            "MAE": float(np.abs(d).mean()),
            "RMSE": float(np.sqrt(np.mean(d ** 2))),
            "中位数": float(np.median(yb)),
            "最大值": float(yb.max()),
            "最小值": float(yb.min()),
            "label求和": lbl,
            "label占比": lbl / total_label if total_label else np.nan,
            "累计label": cum_label,
            "累计label占比": (cum_label / total_label)
            if total_label else np.nan,
        })
    return pd.DataFrame(rows)


def _uplift_ci_halfwidth(y: np.ndarray, w: np.ndarray, m: np.ndarray,
                         n_boot: int = 200,
                         seed: int = 20260814) -> float | None:
    """桶内实际 uplift 的 bootstrap 95% CI 半宽([est±hw] 即 95% CI)。"""
    yb, wb = np.asarray(y)[m], np.asarray(w)[m]
    if len(yb) < 30:
        return None
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(yb), len(yb))
        yt, wt = yb[idx], wb[idx]
        t, c = wt == 1, wt == 0
        vals[b] = yt[t].mean() - yt[c].mean() \
            if t.sum() and c.sum() else np.nan
    vals = vals[np.isfinite(vals)]
    if len(vals) < n_boot * 0.5:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float((hi - lo) / 2)


def uplift_segment_table(y_true: np.ndarray, score: np.ndarray,
                         treatment: np.ndarray,
                         n_bins: int = 10) -> pd.DataFrame | None:
    """uplift 分桶表(桶 1 = CATE 最高段,降序)。

    增量人数 = 桶内实际 uplift × 桶人数(处置组增量外推到整桶);
    累计增量人数 = 累计实际 uplift × 累计人数。桶内缺处置/对照组时
    该桶 uplift 无意义 → 整表不产出(None),与 segment_ranking 口径一致。
    """
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    w = np.asarray(treatment)
    bins = _bin_by_score(score, n_bins)
    if bins is None:
        return None
    n = len(y)
    t_all, c_all = w == 1, w == 0
    if not t_all.any() or not c_all.any():
        return None
    ate = float(y[t_all].mean() - y[c_all].mean())
    # 每桶 T/C 计数与 y 和(供累计口径增量计算)
    bin_n = np.zeros(n_bins, dtype=int)
    bin_nt = np.zeros(n_bins, dtype=int)
    bin_st = np.zeros(n_bins, dtype=float)
    bin_sc = np.zeros(n_bins, dtype=float)
    for b in range(n_bins):
        m = bins == b
        bin_n[b] = int(m.sum())
        mt, mc = m & t_all, m & c_all
        bin_nt[b] = int(mt.sum())
        bin_st[b] = float(y[mt].sum())
        bin_sc[b] = float(y[mc].sum())
    rows = []
    cum_n = cum_nt = 0
    cum_st = cum_sc = 0.0
    seq = 0
    for b in range(n_bins - 1, -1, -1):
        cnt, n_t = int(bin_n[b]), int(bin_nt[b])
        n_c = cnt - n_t
        if cnt == 0:
            continue
        if n_t == 0 or n_c == 0:
            return None
        rate_t = bin_st[b] / n_t
        rate_c = bin_sc[b] / n_c
        actual = rate_t - rate_c
        cum_n += cnt
        cum_nt += n_t
        cum_st += bin_st[b]
        cum_sc += bin_sc[b]
        cum_uplift = (cum_st / cum_nt) - (cum_sc / (cum_n - cum_nt)) \
            if cum_nt and (cum_n - cum_nt) else np.nan
        seq += 1
        rows.append({
            "分桶": seq,
            "预测区间": _score_range(score, bins == b),
            "人群占比": cnt / n,
            "分桶人数": cnt,
            "T人数": n_t,
            "C人数": n_c,
            "T转化率": rate_t,
            "C转化率": rate_c,
            "实际Uplift": actual,
            "预测Uplift": float(score[bins == b].mean()),
            "增量人数": actual * cnt,
            "累计Uplift": cum_uplift,
            "累计增量人数": cum_uplift * cum_n
            if np.isfinite(cum_uplift) else np.nan,
            "95%CI": _uplift_ci_halfwidth(y, w, bins == b),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 汇总 sheet + 落盘
# ---------------------------------------------------------------------------

def _summary_frame(spec: TaskSpec, n: int, n_bins: int,
                   extra: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("task_id", spec.task_id),
        ("model_type", spec.model_type),
        ("primary_metric", spec.primary_metric),
        ("样本量", n),
        ("分桶数", n_bins),
        ("排序口径", "score 降序(桶 1 = 最高分段)"),
        ("生成时间", now_iso()),
    ]
    rows.extend(extra.items())
    return pd.DataFrame(rows, columns=["项目", "值"])


def build_segment_report(
    y_true: np.ndarray,
    score: np.ndarray,
    spec: TaskSpec,
    *,
    treatment: np.ndarray | None = None,
    n_bins: int = 10,
    y_pred: np.ndarray | None = None,
    out_dir: Path,
    extra_summary: dict[str, Any] | None = None,
    label: str = "",
) -> tuple[pd.DataFrame, Path, Path] | None:
    """构造分桶表并落盘 xlsx + md;返回 (df, xlsx_path, md_path)。

    口径由 spec.model_type 裁决:uplift 必须传 treatment(缺失抛 ValueError,
    与 evaluator 同样不静默降级);regression 的 y_pred 缺省回退 score
    (validation_score.parquet 的 score 列即预测值);其余走分类口径。
    样本不足/score 退化时返回 None(不产出空表)。
    """
    y = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(y)
    mt = spec.model_type
    kind = "uplift" if mt == "uplift" else \
        "regression" if mt == "regression" else "classification"
    if kind == "uplift" and treatment is None:
        raise ValueError("uplift 分桶报表需要 treatment 数组")
    if kind == "regression" and y_pred is None:
        y_pred = score

    if kind == "uplift":
        df = uplift_segment_table(y, score, np.asarray(treatment), n_bins)
    elif kind == "regression":
        df = regression_segment_table(y, y_pred, n_bins)
    else:
        df = classification_segment_table(y, score, n_bins)
    if df is None or not len(df):
        logger.warning("分桶报表未产出(样本不足或 score 退化,n=%d)", n)
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "segment-report.xlsx"
    md_path = out_dir / "segment-report.md"
    extra = dict(extra_summary or {})
    # 核心指标:primary + spec 声明的 secondary(与本表同一数据面现算,
    # 汇总区即看即得,不必回 evaluation.json 对照)
    from .metrics import compute_metric

    try:
        primary = compute_metric(
            spec.primary_metric, y, y_pred if kind == "regression" else None,
            score=score, treatment=treatment)
        extra[spec.primary_metric] = round(float(primary), 6) \
            if primary is not None and np.isfinite(primary) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("汇总区 primary_metric 计算失败: %s", exc)
    for name in (spec.data.get("secondary_metrics") or []):
        if name == spec.primary_metric:
            continue
        try:
            v = compute_metric(
                name, y, y_pred if kind == "regression" else None,
                score=score, treatment=treatment)
            extra[name] = round(float(v), 6) \
                if v is not None and np.isfinite(v) else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("汇总区 secondary %s 计算失败: %s", name, exc)
    # 分类:汇总额外附头部圈人 lift_top10/top30(圈人交付口径)
    if kind == "classification":
        from .metrics import top_lift

        for name, pct in (("lift_top10", 10), ("lift_top30", 30)):
            extra[name] = top_lift(y, score, pct)
    # v2.17.0(#20):回归汇总附整体水位(预测均值 / 真实均值 / 水位比),
    # 与 evaluation.regression_diagnostics 同一口径(runtime.regression_diagnostics)
    if kind == "regression":
        try:
            from .regression_diagnostics import level_stats

            _lv = level_stats(y, y_pred)
            for _k, _name in (("pred_mean", "预测均值"), ("y_mean", "真实均值"),
                              ("level_ratio", "水位比(pred/y)")):
                _v = _lv.get(_k)
                extra[_name] = round(float(_v), 6) if _v is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("汇总区水位统计失败: %s", exc)
    summary = _summary_frame(spec, n, n_bins, extra)
    _write_xlsx(xlsx_path, df, summary)
    _write_md(md_path, df, summary, kind, label)
    return df, xlsx_path, md_path


def _write_xlsx(path: Path, df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Excel 多 sheet(分桶表 + 汇总);openpyxl 缺失时降级 csv 副本。"""
    try:
        import openpyxl  # noqa: F401

        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="分桶表", index=False)
            summary.to_excel(xw, sheet_name="汇总", index=False)
    except ImportError:
        logger.warning("openpyxl 未安装:xlsx 降级为 csv(%s)",
                       path.with_suffix(".csv"))
        df.to_csv(path.with_suffix(".csv"), index=False)
        path.with_suffix(".summary.csv") \
            .write_text(summary.to_csv(index=False), encoding="utf-8")


def _write_md(path: Path, df: pd.DataFrame, summary: pd.DataFrame,
              kind: str, label: str) -> None:
    title = {"classification": "分类分桶表", "regression": "回归分桶表",
             "uplift": "Uplift 分桶表"}[kind]
    lines = [f"# {title}{(' — ' + label) if label else ''}", ""]
    lines += _fmt_md_table(summary)
    lines += ["", f"共 {len(df)} 桶(桶 1 = score 最高段,降序)。", ""]
    lines += _fmt_md_table(df)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
