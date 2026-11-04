"""v2.19.0(#10):画像规则表 —— 让 sample-profile 进入 Planner 的候选生成。

背景(路线文档 §4.1):planner 只读 information-inventory.yaml,画像里的漂移、
缺失、基数、正样本率、样本量等统计量只在 planner-context 里展示给 LLM;换一份
数据,Planner 给出的候选结构基本相同。本模块把画像产物按"形态"匹配到规则,
每条命中产出:

- ``evidence``:触发命中的数值证据(阈值与实测值并列,可审计);
- ``suggestions``:建议的候选(family / action / changed_modules / operator);
- ``prior_gain_mde``:以 MDE 为单位的**保守先验**(模块常量,待 benchmark 校准);
- ``cost_hours``:成本估计。

只读输入(全部来自 profile 产物,不读任何标签文件):
``sample-profile/sample-profile.json``、``feature-quality.csv``、``feature-woe.csv``
(缺失箱的 WOE 是 profile 在 train 面算好的统计量)、``splits/train.parquet`` 与
``splits/reparse/train.parquet`` 的**特征值**(仅文本列前 N 行,做形态统计)。

序列/编码串列按**形态**判定(分隔符密度、token 复用率、长度 CV、唯一率),不看列名;
形态统计的判据与 ``runtime.profile.SampleProfile._text_shape_evidence`` 同口径、
同阈值(直接复用其常量)。任何读取失败都静默降级为"无命中",不阻断规划。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 形态判据(与 runtime.profile 同口径;导入失败时用同值兜底)
# ---------------------------------------------------------------------------
try:  # noqa: SIM105
    from .profile import (TEXT_SHAPE_DELIM_MIN, TEXT_SHAPE_LEN_CV_MIN,
                          TEXT_SHAPE_VOCAB_REUSE_MIN)
except Exception:  # noqa: BLE001 —— profile 依赖缺失时保持同值
    TEXT_SHAPE_LEN_CV_MIN = 0.25
    TEXT_SHAPE_VOCAB_REUSE_MIN = 50
    TEXT_SHAPE_DELIM_MIN = 0.5

#: 序列/编码串:平均每行至少 1 个分隔符(≥2 个 token)。TEXT_SHAPE_DELIM_MIN
#: (0.5)是"疑似复合文本"的宽口径,这里要的是"可按 token 做算子"的列,取 1.0。
SEQ_DELIM_MIN = 1.0
#: 数值 token 占比 ≥ 0.9 视为数值序列(间隔/金额/时长串),不要求词表复用
NUMERIC_TOKEN_SHARE_MIN = 0.9
#: 参与形态统计的文本列上限(逐行字符串运算较贵;reparse 源优先)
MAX_SHAPE_COLUMNS = 40
#: 现算形态时每列扫描的行数。profile 的 TEXT_SHAPE_SCAN_ROWS=20000 在 500
#: token/行的序列列上要 ~12s(run4 实测);形态四项在 5000 行上已稳定,
#: 且结果按产物 mtime 落盘缓存(planner/profile-shapes.json),每 run 只算一次。
SHAPE_SCAN_ROWS = 5000
SHAPE_CACHE_REL = Path("planner") / "profile-shapes.json"
#: 形态字段集合的版本(新增统计量时 +1,使旧缓存失效)
SHAPE_SCHEMA = 2

# ---------------------------------------------------------------------------
# 规则阈值
# ---------------------------------------------------------------------------
HIGH_CARD_MIN_UNIQUE = 50          # 类别列 n_unique ≥ 50 视为高基数
HIGH_CARD_MAX_UNIQUE_RATIO = 0.5   # 唯一率过高的是 ID / 自由文本,不算类别
IMBALANCE_POS_RATE = 0.05          # 正样本率 < 5%
DRIFT_PSI = 0.25                   # PSI > 0.25 视为强漂移
DRIFT_SHARE_MIN = 0.05             # 强漂移列占比 ≥ 5% 才命中
DRIFT_MIN_COLS = 3                 # 且至少 3 列(防极窄表的偶然命中)
SMALL_N = 20_000                   # n_train < 2 万
LARGE_N = 1_000_000                # n_train > 100 万
MISSING_WOE_ABS = 0.2              # 缺失箱 |WOE| ≥ 0.2 视为缺失与标签相关
MISSING_BIN_SHARE_MIN = 0.01       # 缺失箱样本占比 ≥ 1%(太稀的箱 WOE 不稳)
MISSING_INFORMATIVE_MIN_COLS = 3   # 命中列数 ≥ 3 才建议(组级缺失模式需要多列)
MISSING_BIN_IV_MIN = 0.005         # 缺失箱 IV 分量 ≥ 0.005(feature-woe.csv 有 iv 列时)

#: 各建议的先验增益(单位:MDE)。**待 benchmark 校准** —— 初值刻意保守
#: (≤1×MDE):先验只用来让有画像依据的候选在冷启动时不至于 expected_gain=0,
#: 一旦同 family 在本 run 有实测(方向台账/已评估 trial),Planner 用后验替代。
PRIOR_GAIN_MDE = {
    "seq_ops": 1.0,                 # 序列算子族宽面(多源经验:最强信息通道)
    "freq_encoding": 0.3,
    "target_encoding_oof": 0.4,
    "catboost_native_cat": 0.3,
    "class_weight": 0.2,
    "drift_column_drop": 0.3,
    "recency_weighting": 0.3,
    "small_n_cv_strong_reg": 0.5,
    "missing_indicator": 0.2,
    "large_n_subsample_screen": 0.0,  # 效率动作,不产增益
}
#: 成本估计(小时;与 planner 其它候选同量纲),待实测校准
COST_HOURS = {
    "seq_ops": 2.0,
    "freq_encoding": 0.75,
    "target_encoding_oof": 1.0,
    "catboost_native_cat": 1.5,
    "class_weight": 0.75,
    "drift_column_drop": 0.75,
    "recency_weighting": 0.75,
    "small_n_cv_strong_reg": 0.5,
    "missing_indicator": 0.75,
    "large_n_subsample_screen": 0.0,
}

_SEP_RE = r"[,;|]"


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return []


def _spec_attr(spec, name: str, default=None):
    try:
        v = getattr(spec, name)
    except Exception:  # noqa: BLE001
        return default
    return default if v is None else v


# ---------------------------------------------------------------------------
# 文本列形态统计
# ---------------------------------------------------------------------------
_SHAPE_CACHE: dict = {}


def _shape_of_series(s) -> dict:
    """与 SampleProfile._text_shape_evidence 同口径,另补 n_unique/unique_ratio。"""
    out: dict = {}
    txt = s.dropna().astype(str)
    if not len(txt):
        return out
    if len(txt) > SHAPE_SCAN_ROWS:
        txt = txt.iloc[:SHAPE_SCAN_ROWS]
    lens = txt.str.len()
    mean_len = float(lens.mean())
    out["len_cv"] = round(float(lens.std(ddof=0)) / mean_len, 3) \
        if mean_len > 0 else 0.0
    out["delim_density"] = round(float(txt.str.count(_SEP_RE).mean()), 2)
    tokens = txt.str.split(_SEP_RE).explode()
    vocab = int(tokens.nunique())
    out["token_vocab"] = vocab
    out["token_reuse"] = round(len(tokens) / vocab, 1) if vocab else 0.0
    # 数值 token 占比:数值序列(间隔/金额串)的词表随样本量增长,复用倍数
    # 不稳定(run4 seq_gap_d365:2 万行 118x、5 千行 <50x),单独认定
    import pandas as pd

    head = tokens.iloc[:50000]
    out["numeric_token_share"] = round(float(
        pd.to_numeric(head, errors="coerce").notna().mean()), 3) \
        if len(head) else 0.0
    out["n_unique"] = int(txt.nunique())
    out["unique_ratio"] = round(out["n_unique"] / max(len(txt), 1), 4)
    out["rows_scanned"] = int(len(txt))
    return out


def _text_columns_of(path: Path, wanted: "set[str] | None") -> list[str]:
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(str(path))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for name, typ in zip(schema.names, schema.types):
        t = str(typ)
        if wanted is not None and name not in wanted:
            continue
        if t in ("string", "large_string") or t.startswith("dictionary"):
            out.append(str(name))
    return out


def _scan_shapes(path: Path, cols: list[str]) -> dict:
    if not cols:
        return {}
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        batch = next(pf.iter_batches(batch_size=SHAPE_SCAN_ROWS,
                                     columns=cols), None)
        if batch is None:
            return {}
        df = batch.to_pandas()
    except Exception as exc:  # noqa: BLE001
        logger.debug("形态扫描失败 %s: %s", path, exc)
        return {}
    out = {}
    for c in cols:
        try:
            sh = _shape_of_series(df[c])
        except Exception:  # noqa: BLE001
            sh = {}
        if sh:
            sh["source"] = f"scan:{path.parent.name}/{path.name}"
            out[c] = sh
    return out


def column_shapes(run_dir, spec=None) -> dict:
    """{列名: 形态统计}(文本列)。来源优先级:

    1. ``sample-profile.json`` 的 ``leak_classification.proposals``(profile 已逐列
       落盘的 len_cv / delim_density / token_reuse / unique_ratio);
    2. ``splits/reparse/train.parquet``(spec reparse 列)与 ``splits/train.parquet``
       (特征面里的文本列)前 SHAPE_SCAN_ROWS 行现算(结果落盘缓存);
    3. ``drift-profile.csv`` 的 token ``__vocab__`` 行(只有 tokens_per_row,
       记为 delim_density≈tokens_per_row−1,token_reuse 缺失)。

    时间列、实体 ID 列、目标列不参与。结果按 run 目录 + 产物 mtime 进程内缓存。
    """
    run = Path(run_dir)
    sp = run / "sample-profile"
    paths = [sp / "sample-profile.json", sp / "splits" / "train.parquet",
             sp / "splits" / "reparse" / "train.parquet",
             sp / "drift-profile.csv"]
    key = (str(run.resolve()), SHAPE_SCHEMA) + tuple(
        (p.stat().st_mtime if p.exists() else 0.0) for p in paths)
    if key in _SHAPE_CACHE:
        return _SHAPE_CACHE[key]
    disk = run / SHAPE_CACHE_REL
    cached = _load_json(disk) if disk.is_file() else {}
    if cached.get("key") == list(key[1:]) and \
            isinstance(cached.get("shapes"), dict):
        _SHAPE_CACHE.clear()
        _SHAPE_CACHE[key] = cached["shapes"]
        return cached["shapes"]
    skip = {c for c in (_spec_attr(spec, "time_column"),
                        _spec_attr(spec, "id_column"),
                        _spec_attr(spec, "target_name")) if c}
    shapes: dict = {}
    prof = _load_json(paths[0])
    for p in ((prof.get("leak_classification") or {}).get("proposals")
              or []):
        col = p.get("column")
        if not col or col in skip:
            continue
        sh = {k: p.get(k) for k in ("len_cv", "delim_density", "token_vocab",
                                    "token_reuse", "unique_ratio",
                                    "n_unique") if p.get(k) is not None}
        if {"delim_density", "token_reuse"} <= set(sh):
            sh["source"] = "sample-profile.leak_classification.proposals"
            shapes[str(col)] = sh
    reparse = [c for c in (_spec_attr(spec, "reparse_columns", []) or [])
               if c not in skip and c not in shapes]
    if not reparse:
        # spec 不可用时,以 manifest / leak_classification 记录的 reparse 列为准
        man = (prof.get("splits_manifest") or {}).get("reparse_columns") or \
            ((prof.get("leak_classification") or {}).get("confirmed") or {}
             ).get("reparse") or []
        reparse = [c for c in man if c not in skip and c not in shapes]
    if reparse and paths[2].is_file():
        cols = _text_columns_of(paths[2], set(reparse))[:MAX_SHAPE_COLUMNS]
        shapes.update(_scan_shapes(paths[2], cols))
    room = MAX_SHAPE_COLUMNS - len(shapes)
    if room > 0 and paths[1].is_file():
        cols = [c for c in _text_columns_of(paths[1], None)
                if c not in skip and c not in shapes][:room]
        shapes.update(_scan_shapes(paths[1], cols))
    # 兜底:token 漂移 __vocab__ 行(profile 只对形态像序列的文本列做 token 漂移)
    for row in _read_csv(paths[3]):
        if row.get("kind") != "token" or row.get("token") != "__vocab__":
            continue
        col = row.get("feature")
        if not col or col in shapes or col in skip:
            continue
        tpr = _f(row.get("baseline_tokens_per_row")) or \
            _f(row.get("tokens_per_row"))
        if tpr is None:
            continue
        shapes[col] = {"delim_density": round(max(tpr - 1.0, 0.0), 2),
                       "tokens_per_row": round(tpr, 2),
                       "source": "drift-profile.token_vocab"}
    _SHAPE_CACHE.clear()
    _SHAPE_CACHE[key] = shapes
    if shapes:
        try:
            disk.parent.mkdir(parents=True, exist_ok=True)
            tmp = disk.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"key": list(key[1:]), "shapes": shapes},
                                      ensure_ascii=False, default=str),
                           encoding="utf-8")
            tmp.replace(disk)
        except OSError:  # 只读 run 目录(审计/测试)时不缓存
            pass
    return shapes


def sequence_kind(shape: "dict | None") -> "str | None":
    """形态 → ``variable_length_sequence`` / ``encoded_string`` / ``""``(非序列);
    形态统计缺失时返回 None(调用方据此走兜底)。

    判据:平均分隔符 ≥ SEQ_DELIM_MIN 且(token 复用倍数 > TEXT_SHAPE_VOCAB_REUSE_MIN
    —— token 被大量复用 = 事件/编码词表,而不是自由文本或 ID —— 或数值 token
    占比 ≥ NUMERIC_TOKEN_SHARE_MIN,即数值序列);长度 CV >
    TEXT_SHAPE_LEN_CV_MIN 为变长事件序列,否则为定长编码串。只有 __vocab__ 兜底
    (无 token_reuse)时按分隔符密度判定:profile 只对形态像 token 序列的列做
    token 漂移,本身已过复用判据。
    """
    if not shape:
        return None
    delim = _f(shape.get("delim_density"))
    if delim is None:
        return None
    reuse = _f(shape.get("token_reuse"))
    num_share = _f(shape.get("numeric_token_share"))
    if delim < SEQ_DELIM_MIN:
        return ""
    if num_share is not None and num_share >= NUMERIC_TOKEN_SHARE_MIN:
        pass                    # 数值序列:不看词表复用
    elif reuse is None:
        if shape.get("source") != "drift-profile.token_vocab":
            return None
    elif reuse <= TEXT_SHAPE_VOCAB_REUSE_MIN:
        return ""
    cv = _f(shape.get("len_cv"))
    if cv is not None and cv <= TEXT_SHAPE_LEN_CV_MIN:
        return "encoded_string"
    return "variable_length_sequence"


def is_sequence_shape(shape: "dict | None") -> "bool | None":
    """形态是否为序列/编码串;统计缺失返回 None。"""
    k = sequence_kind(shape)
    return None if k is None else bool(k)


# ---------------------------------------------------------------------------
# 画像事实
# ---------------------------------------------------------------------------
def _n_train(prof: dict) -> int | None:
    for sp in ((prof.get("splits_summary") or {}).get("splits") or []):
        if isinstance(sp, dict) and sp.get("split") == "train" and \
                _f(sp.get("rows")):
            return int(sp["rows"])
    tr = (((prof.get("splits_manifest") or {}).get("splits") or {})
          .get("train") or {})
    if _f(tr.get("rows")):
        return int(tr["rows"])
    tr = (((prof.get("sample_summary") or {}).get("by_split") or {})
          .get("train") or {})
    if _f(tr.get("rows")):
        return int(tr["rows"])
    return int(prof["n_rows"]) if _f(prof.get("n_rows")) else None


def _pos_rate(prof: dict) -> float | None:
    for sp in ((prof.get("splits_summary") or {}).get("splits") or []):
        if isinstance(sp, dict) and sp.get("split") == "train" and \
                _f(sp.get("positive_rate")) is not None:
            return float(sp["positive_rate"])
    return _f((prof.get("sample_summary") or {}).get("label_rate"))


def _is_binary(spec, prof: dict) -> bool:
    mt = str(_spec_attr(spec, "model_type", "") or "").lower()
    if mt:
        return "binary" in mt or mt == "classification"
    return _pos_rate(prof) is not None


def profile_facts(run_dir, spec=None) -> dict:
    run = Path(run_dir)
    prof = _load_json(run / "sample-profile" / "sample-profile.json")
    return {"available": bool(prof), "n_train": _n_train(prof),
            "positive_rate": _pos_rate(prof) if _is_binary(spec, prof)
            else None,
            "n_features": prof.get("n_features"),
            "model_type": _spec_attr(spec, "model_type")}


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------
def _sugg(key: str, *, family: str, action: str, modules: list[str],
          hint: str, **extra) -> dict:
    return {"key": key, "family": family, "action": action,
            "changed_modules": modules, "hint": hint,
            "prior_gain_mde": PRIOR_GAIN_MDE[key],
            "cost_hours": COST_HOURS[key], **extra}


def _rule_sequence(shapes: dict) -> list[dict]:
    hits = []
    for col in sorted(shapes):
        kind = sequence_kind(shapes[col])
        if not kind:
            continue
        sh = shapes[col]
        hits.append({
            "rule": "sequence_column", "id": f"sequence:{col}",
            "evidence": {"column": col, "kind": kind,
                         **{k: sh.get(k) for k in (
                             "delim_density", "token_reuse", "len_cv",
                             "numeric_token_share", "unique_ratio",
                             "tokens_per_row", "source")
                            if sh.get(k) is not None},
                         "thresholds": {"delim_density>=": SEQ_DELIM_MIN,
                                        "token_reuse>": TEXT_SHAPE_VOCAB_REUSE_MIN,
                                        "len_cv>": TEXT_SHAPE_LEN_CV_MIN}},
            "suggestions": [_sugg(
                "seq_ops", family=f"profile:seq_ops:{col}", action="explore",
                modules=["feature"], source_column=col,
                operator="sequence_operator_family_wide",
                hint=(f"{col} 形态为{'变长事件序列' if kind == 'variable_length_sequence' else '编码串'}"
                      "(按分隔符切 token):一次加宽序列算子族(多窗计数/近因衰减/"
                      "转移对与熵/间隔分布/token 频次),train 内拟合"))]})
    return hits


def _rule_high_cardinality(quality: list[dict], shapes: dict) -> list[dict]:
    cols = []
    for r in quality:
        dtype = str(r.get("dtype") or "").lower()
        if any(t in dtype for t in ("int", "float", "bool", "datetime")):
            continue
        n_u = _f(r.get("n_unique"))
        ur = _f(r.get("unique_ratio"))
        if n_u is None or n_u < HIGH_CARD_MIN_UNIQUE:
            continue
        if str(r.get("id_like")).lower() == "true":
            continue
        if ur is not None and ur > HIGH_CARD_MAX_UNIQUE_RATIO:
            continue
        if sequence_kind(shapes.get(r.get("feature"))):
            continue
        cols.append((r["feature"], int(n_u), ur))
    for col, sh in shapes.items():
        if any(c[0] == col for c in cols) or sequence_kind(sh):
            continue
        n_u, ur = _f(sh.get("n_unique")), _f(sh.get("unique_ratio"))
        if n_u is not None and n_u >= HIGH_CARD_MIN_UNIQUE and \
                (ur is None or ur <= HIGH_CARD_MAX_UNIQUE_RATIO):
            cols.append((col, int(n_u), ur))
    if not cols:
        return []
    cols.sort(key=lambda c: -c[1])
    names = [c[0] for c in cols[:20]]
    return [{
        "rule": "high_cardinality_categorical", "id": "high_cardinality",
        "evidence": {"n_columns": len(cols),
                     "top": [{"column": c, "n_unique": n, "unique_ratio": u}
                             for c, n, u in cols[:5]],
                     "thresholds": {"n_unique>=": HIGH_CARD_MIN_UNIQUE,
                                    "unique_ratio<=": HIGH_CARD_MAX_UNIQUE_RATIO}},
        "suggestions": [
            _sugg("freq_encoding", family="profile:freq_encoding",
                  action="explore", modules=["feature"], columns=names,
                  hint="高基数类别列做频次编码(train 内计数)"),
            _sugg("target_encoding_oof", family="profile:target_encoding_oof",
                  action="explore", modules=["feature"], columns=names,
                  hint="折内(out-of-fold)目标编码,带平滑;禁止用 validation 标签"),
            _sugg("catboost_native_cat", family="profile:catboost_native_cat",
                  action="explore", modules=["model"], columns=names,
                  algorithm="catboost",
                  hint="CatBoost 原生类别特征(cat_features),有序目标统计"),
        ]}]


def _rule_imbalance(facts: dict) -> list[dict]:
    pr = facts.get("positive_rate")
    if pr is None or pr >= IMBALANCE_POS_RATE:
        return []
    return [{
        "rule": "class_imbalance", "id": "class_imbalance",
        "evidence": {"positive_rate": round(pr, 6),
                     "thresholds": {"positive_rate<": IMBALANCE_POS_RATE}},
        "advisory": "核对 AUC 与 PR-AUC 的结论是否一致(稀有正类下 AUC 可能掩盖头部排序退化)",
        "suggestions": [_sugg(
            "class_weight", family="profile:class_weight", action="explore",
            modules=["sample"],
            hint=("类别加权(scale_pos_weight / class_weight≈neg/pos 的平方根档),"
                  "同时报告 PR-AUC"))]}]


def _rule_drift(quality: list[dict], prof: dict) -> list[dict]:
    vals = []
    for r in quality:
        v = _f(r.get("psi_train_vs_validation"))
        adj = _f(r.get("psi_max_adjacent_months"))
        m = max([x for x in (v, adj) if x is not None], default=None)
        if m is not None:
            vals.append((r.get("feature"), m))
    if not vals:
        return []
    strong = sorted([x for x in vals if x[1] > DRIFT_PSI], key=lambda x: -x[1])
    share = len(strong) / len(vals)
    if len(strong) < DRIFT_MIN_COLS or share < DRIFT_SHARE_MIN:
        return []
    cols = [c for c, _ in strong[:50]]
    return [{
        "rule": "strong_drift", "id": "strong_drift",
        "evidence": {"n_psi_gt": len(strong), "n_columns": len(vals),
                     "share": round(share, 4),
                     "max_psi": round(strong[0][1], 4),
                     "top": [{"column": c, "psi": round(p, 4)}
                             for c, p in strong[:5]],
                     "thresholds": {"psi>": DRIFT_PSI,
                                    "share>=": DRIFT_SHARE_MIN}},
        "suggestions": [
            _sugg("drift_column_drop", family="profile:drift_column_drop",
                  action="explore", modules=["feature"], columns=cols,
                  hint="剔除强漂移列(PSI>0.25)做对照,看 validation 是否改善"),
            _sugg("recency_weighting", family="profile:recency_weighting",
                  action="explore", modules=["sample"],
                  hint="近期样本加权(按时间指数衰减的 sample_weight,仅 train 内)"),
        ]}]


def _rule_sample_size(facts: dict) -> list[dict]:
    n = facts.get("n_train")
    if not n:
        return []
    if n < SMALL_N:
        return [{
            "rule": "small_sample", "id": "small_sample",
            "evidence": {"n_train": n, "thresholds": {"n_train<": SMALL_N}},
            "advisory": "单次 validation 噪声大:用 CV 评估候选、强正则、小样本模型族",
            "suggestions": [_sugg(
                "small_n_cv_strong_reg", family="profile:small_n_cv_strong_reg",
                action="explore", modules=["model"],
                hint="强正则(更浅树/更大 min_child/更强 L2/更低 lr)并在 train 内 CV 评估")]}]
    if n > LARGE_N:
        return [{
            "rule": "large_sample", "id": "large_sample",
            "evidence": {"n_train": n, "thresholds": {"n_train>": LARGE_N}},
            "advisory": ("大样本:特征/超参先在抽样面筛选,只把胜出者放到全量复核;"
                         "同面 HPO 默认降权(见 planner HPO 先验)"),
            # 效率动作不物化为候选(prior=0;由 HPO 多保真 / 抽样面承担)
            "suggestions": [_sugg(
                "large_n_subsample_screen",
                family="profile:large_n_subsample_screen", action="explore",
                modules=["sample"], materialize=False,
                hint="抽样面筛选 + 全量复核")]}]
    return []


def _rule_missing(woe_rows: list[dict]) -> list[dict]:
    totals: dict[str, float] = {}
    nan_rows: dict[str, tuple] = {}
    for r in woe_rows:
        col = r.get("feature")
        cnt = _f(r.get("count")) or 0.0
        if not col:
            continue
        totals[col] = totals.get(col, 0.0) + cnt
        if str(r.get("bin")) in ("__nan__", "nan", "NaN", "missing"):
            nan_rows[col] = (cnt, _f(r.get("woe")), _f(r.get("iv")))
    hits = []
    for col, (cnt, woe, iv) in nan_rows.items():
        tot = totals.get(col) or 0.0
        if woe is None or not tot:
            continue
        if iv is not None and iv < MISSING_BIN_IV_MIN:
            continue
        share = cnt / tot
        if share >= MISSING_BIN_SHARE_MIN and abs(woe) >= MISSING_WOE_ABS:
            hits.append((col, round(share, 4), round(woe, 4)))
    if len(hits) < MISSING_INFORMATIVE_MIN_COLS:
        return []
    hits.sort(key=lambda x: -abs(x[2]))
    return [{
        "rule": "informative_missing", "id": "informative_missing",
        "evidence": {"n_columns": len(hits),
                     "top": [{"column": c, "missing_share": s, "missing_woe": w}
                             for c, s, w in hits[:5]],
                     "thresholds": {"|missing_woe|>=": MISSING_WOE_ABS,
                                    "missing_share>=": MISSING_BIN_SHARE_MIN,
                                    "missing_bin_iv>=": MISSING_BIN_IV_MIN,
                                    "n_columns>=": MISSING_INFORMATIVE_MIN_COLS}},
        "suggestions": [_sugg(
            "missing_indicator", family="profile:missing_indicator",
            action="explore", modules=["feature"],
            columns=[c for c, _, _ in hits[:50]],
            hint="缺失指示列 + 组级缺失模式(同源列组的缺失计数/缺失模式哈希)")]}]


def evaluate(run_dir, spec=None) -> dict:
    """读取 profile 产物,返回 {facts, shapes, hits};任何异常降级为空命中。"""
    run = Path(run_dir)
    sp = run / "sample-profile"
    out: dict = {"facts": {}, "shapes": {}, "hits": []}
    try:
        prof = _load_json(sp / "sample-profile.json")
        facts = profile_facts(run, spec)
        shapes = column_shapes(run, spec)
        quality = _read_csv(sp / "feature-quality.csv")
        hits = []
        for fn, args in ((_rule_sequence, (shapes,)),
                         (_rule_high_cardinality, (quality, shapes)),
                         (_rule_imbalance, (facts,)),
                         (_rule_drift, (quality, prof)),
                         (_rule_sample_size, (facts,)),
                         (_rule_missing, (_read_csv(sp / "feature-woe.csv"),))):
            try:
                hits += fn(*args)
            except Exception as exc:  # noqa: BLE001 —— 单条规则失败不影响其它
                logger.warning("画像规则 %s 失败: %s", fn.__name__, exc)
        out.update(facts=facts, shapes=shapes, hits=hits)
    except Exception as exc:  # noqa: BLE001
        logger.warning("画像规则评估失败(按无命中处理): %s", exc)
    return out


def summary_lines(result: dict, limit: int = 12) -> list[str]:
    """planner-context 用的一行一条命中摘要。"""
    lines = []
    facts = result.get("facts") or {}
    if facts.get("available"):
        pr = facts.get("positive_rate")
        lines.append(f"- 画像事实:n_train={facts.get('n_train')}"
                     + (f",正样本率={pr:.4f}" if pr is not None else "")
                     + f",n_features={facts.get('n_features')}")
    for h in (result.get("hits") or [])[:limit]:
        ev = {k: v for k, v in (h.get("evidence") or {}).items()
              if k not in ("thresholds", "top")}
        fams = ", ".join(f"{s['family']}(prior {s['prior_gain_mde']}×MDE)"
                         for s in h.get("suggestions") or []
                         if s.get("materialize", True))
        line = f"- [{h['rule']}] {json.dumps(ev, ensure_ascii=False, default=str)}"
        if fams:
            line += f" → {fams}"
        if h.get("advisory"):
            line += f";提示:{h['advisory']}"
        lines.append(line)
    if not (result.get("hits") or []):
        lines.append("- (无画像规则命中)")
    return lines


__all__ = ["evaluate", "column_shapes", "sequence_kind", "is_sequence_shape",
           "profile_facts", "summary_lines", "PRIOR_GAIN_MDE", "COST_HOURS"]

