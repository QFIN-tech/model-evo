#!/usr/bin/env python
"""v2.19.0 benchmark 准备:下载 OpenML 数据 + 生成可直接 validate/profile 的 run 目录。

    python scripts/benchmark/prepare.py --suite assets/benchmark-suite.yaml \
        --datasets-root <dir> --runs-root <dir> [--tier fast] [--only a,b] [--force]

也可经 CLI:``modelevo.py benchmark prepare --suite ... --runs-root ...``。

每个数据集:
1. 下载 OpenML parquet 到 ``<datasets-root>/<name>/raw.parquet``(已存在跳过);
2. 按套件口径处理成 ``data.parquet``:目标编码(二分类 1=positive_class;回归
   可选 log1p)、丢弃泄漏列、合成时间列(真实时间)或分层切分伪时间列
   ``bench_split_date``、补行 ID;
3. ``modelevo.py init`` 建 run 目录,再写 task-spec.yaml / data-loader.py,并把
   modelevo-config.yaml 设为冷启动(``library.enabled: false``)与套件预算。

网络:OpenML 需要代理时设置 http(s)_proxy;下载文件只当数据读取(pandas
解析 parquet),不执行其中任何内容。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SKILL = Path(__file__).resolve().parents[2]
MODELEVO = [sys.executable, str(SKILL / "scripts" / "modelevo.py")]
SPLIT_NAMES = ("train", "validation", "shadow_test", "final_test")
#: 分层切分的伪时间:四个月份 = 四个 split
PSEUDO_MONTHS = {"train": "2000-01", "validation": "2000-02",
                 "shadow_test": "2000-03", "final_test": "2000-04"}
PSEUDO_TIME_COL = "bench_split_date"
ROW_ID_COL = "bench_row_id"


def load_suite(path: "Path | str") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def select_datasets(suite: dict, tier: str | None = None,
                    only: list[str] | None = None) -> list[dict]:
    out = []
    for d in suite.get("datasets") or []:
        if tier and tier not in (d.get("tiers") or []):
            continue
        if only and d["name"] not in only:
            continue
        out.append(d)
    return out


def openml_parquet_url(data_id: int) -> str:
    return (f"https://data.openml.org/datasets/{data_id // 10000:04d}/"
            f"{data_id:04d}/dataset_{data_id}.pq")


def download(ds: dict, root: Path, *, force: bool = False) -> Path:
    d = root / ds["name"]
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "raw.parquet"
    # v2.19.0:本地来源(如 HDFS 上的多表数据经 builder 拼成单表)——不下载,
    # raw.parquet 必须已由 ds["builder"] 生成
    if ds.get("source") == "local":
        if not raw.exists():
            raise FileNotFoundError(
                f"{ds['name']}: 本地来源缺 {raw},先运行 {ds.get('builder')}")
        return raw
    if raw.exists() and not force:
        return raw
    url = openml_parquet_url(int(ds["openml_id"]))
    tmp = raw.with_suffix(".part")
    # 代理/CDN 会在 6~9 MB 处静默断流(实测 creditcard 73 MB 每次只收到
    # 5~9 MB,urlopen 不报错):按 Content-Length 校验,断了就用 HTTP Range
    # 从已收字节续传;最后核对 parquet 尾部魔数
    # 已有的 .part 是上次中断留下的进度:先拿到总长度,再从它续传
    expect, last_err = 0, None
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, method="HEAD"), timeout=120) as r:
            expect = int(r.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001 —— 拿不到长度就从头下
        tmp.unlink(missing_ok=True)
    if expect and tmp.exists() and tmp.stat().st_size > expect:
        tmp.unlink()
    for attempt in range(200):
        have = tmp.stat().st_size if tmp.exists() else 0
        if expect and have >= expect:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if not have:
                    expect = int(r.headers.get("Content-Length") or 0)
                elif r.status != 206:  # 服务端不支持续传:从头再来
                    tmp.unlink(missing_ok=True)
                with open(tmp, "ab") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            if not expect:
                break  # 无长度信息:只能信任本次流结束
        except Exception as exc:  # noqa: BLE001 —— 续传重试
            last_err = exc
        got = tmp.stat().st_size if tmp.exists() else 0
        if attempt % 10 == 0 and expect and got < expect:
            print(f"[download] {ds['name']} 续传中 {got}/{expect}", flush=True)
    got = tmp.stat().st_size if tmp.exists() else 0
    if expect and got != expect:
        raise IOError(f"{ds['name']} 下载失败 {got}/{expect}: {last_err}")
    with open(tmp, "rb") as f:
        f.seek(-4, os.SEEK_END)
        if f.read(4) != b"PAR1":
            raise IOError(f"{ds['name']} 不是完整的 parquet(缺尾部魔数)")
    os.replace(tmp, raw)
    (d / "source.json").write_text(json.dumps(
        {"openml_id": ds["openml_id"], "url": url}, indent=1),
        encoding="utf-8")
    return raw


def _stratified_assign(y: pd.Series, model_type: str, ratios: dict,
                       seed: int) -> np.ndarray:
    """分层随机切分:返回每行的 split 名(二分类按标签,回归按 10 分位桶)。"""
    rng = np.random.default_rng(seed)
    if model_type == "regression":
        strata = pd.qcut(y.rank(method="first"), 10, labels=False).to_numpy()
    else:
        strata = y.to_numpy()
    fr = np.array([float(ratios.get(k, 0)) for k in SPLIT_NAMES])
    fr = fr / fr.sum()
    out = np.empty(len(y), dtype=object)
    for s in pd.unique(strata):
        pos = np.flatnonzero(strata == s)
        rng.shuffle(pos)
        bounds = np.round(np.cumsum(fr) * len(pos)).astype(int)
        lo = 0
        for name, hi in zip(SPLIT_NAMES, bounds):
            out[pos[lo:hi]] = name
            lo = hi
    return out


def process(ds: dict, raw: Path, *, ratios: dict, seed: int) -> tuple[
        pd.DataFrame, dict]:
    """原始表 → 建模样本;返回 (df, spec 片段)。"""
    df = pd.read_parquet(raw)
    # xgboost 特征名不得含 [ ] <:baseline 模板对低基数类别做 one-hot,列名 =
    # 原列名_取值(credit-g 的 checking_status 取值 '<0' 实测让 baseline 崩)。
    # 列名与类别取值统一替换为安全记号;只改记号,不改信息。
    _safe = {"<": "lt", ">": "gt", "[": "(", "]": ")"}

    def _clean(v):
        if not isinstance(v, str):
            return v
        for a, b in _safe.items():
            v = v.replace(a, b)
        return v

    df.columns = [_clean(str(c)) for c in df.columns]
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
        # bool 列 → int8:profile 的 feature_quality 对 bool 走数值分支后
        # describe() 无 min/max(KeyError,bike_sharing 实测)
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("int8")
        if df[c].dtype == object and c != _clean(ds["target"]):
            df[c] = df[c].map(_clean)
    tgt = ds["target"]
    mtype = ds["model_type"]
    df = df.drop(columns=[c for c in ds.get("drop_columns") or []
                          if c in df.columns])
    derived = []
    if mtype == "binary_classification":
        pos = str(ds["positive_class"])
        df[tgt] = (df[tgt].astype(str) == pos).astype("int64")
        target_name = tgt
    else:
        y = pd.to_numeric(df[tgt], errors="coerce").astype("float64")
        if ds.get("target_transform") == "log1p":
            target_name = f"{tgt}_log1p"
            df[target_name] = np.log1p(y)
            df = df.drop(columns=[tgt])
            derived.append(target_name)
        else:
            target_name = tgt
            df[tgt] = y
    # 时间列:真实(derive_time)或分层伪时间
    dt = ds.get("derive_time") or {}
    if ds.get("split") == "time":
        col = dt["column"]
        if dt.get("from") == "year_month":
            yb = int(dt.get("year_base", 0))
            df[col] = [f"{yb + int(a):04d}-{int(b):02d}-01"
                       for a, b in zip(df["year"], df["month"])]
        elif dt.get("from") == "date_string":
            s = df["date"].astype(str).str.slice(0, 8)
            df[col] = s.str.slice(0, 4) + "-" + s.str.slice(4, 6) + "-" \
                + s.str.slice(6, 8)
            df = df.drop(columns=["date"])
        else:
            raise ValueError(f"{ds['name']}: 未知 derive_time.from")
        time_col = col
        windows = dict(ds["time_windows"])
        split_desc = "time_based(真实时间窗口)"
    else:
        assign = _stratified_assign(df[target_name], mtype, ratios, seed)
        df[PSEUDO_TIME_COL] = [f"{PSEUDO_MONTHS[a]}-01" for a in assign]
        time_col = PSEUDO_TIME_COL
        windows = dict(PSEUDO_MONTHS)
        split_desc = (f"stratified random(seed={seed},比例 {ratios});"
                      f"{PSEUDO_TIME_COL} 为切分归属伪时间")
    id_col = ds.get("id_column")
    if not id_col:
        id_col = ROW_ID_COL
        df[id_col] = np.arange(len(df), dtype="int64")
    frag = {"target_name": target_name, "time_column": time_col,
            "id_column": id_col, "windows": windows, "derived": derived,
            "split_desc": split_desc, "n_rows": int(len(df)),
            "n_columns": int(df.shape[1])}
    return df, frag


def _spec_yaml(ds: dict, frag: dict, suite: dict, tier: str) -> str:
    mtype = ds["model_type"]
    budget = (suite.get("budget") or {}).get(tier) or {}
    if mtype == "binary_classification":
        tgt = {"name": frag["target_name"],
               "positive_definition": f"原始 {ds['target']} == "
                                      f"{ds['positive_class']!s}(编码为 1)",
               "observation_window": "OpenML 公开数据,全部字段",
               "performance_window": "N/A(静态标签)"}
        secondary = ["ks", "logloss"]
    else:
        tgt = {"name": frag["target_name"],
               "value_definition": (f"log1p({ds['target']})"
                                    if frag["derived"] else ds["target"]),
               "unit": "log" if frag["derived"] else "count",
               "observation_window": "OpenML 公开数据,全部字段",
               "performance_window": "N/A(静态标签)"}
        secondary = ["mae", "r2"] if ds["primary_metric"] == "rmse" \
            else ["rmse", "r2"]
    data = {"data_loader": "data-loader.py"}
    if frag["derived"]:
        data["derived_columns"] = list(frag["derived"])
    spec = {
        "task_id": "bench-" + ds["name"].replace("_", "-"),
        "task_name": f"benchmark {ds['name']}("
                     + (f"OpenML {ds['openml_id']}" if ds.get("openml_id")
                        else ds.get("source_desc") or "local") + ")",
        "objective_description": (
            f"v2.19.0 benchmark 冷启动任务:{', '.join(ds.get('shape_tags') or [])}"
            + (f"。{ds['description']}" if ds.get("description") else "")),
        "model_type": mtype,
        "business_domains": list(ds.get("business_domains") or ["user_operation"]),
        "target": tgt,
        "time": {"column": frag["time_column"]},
        "entity": {"id_column": frag["id_column"]},
        "data": data,
        "primary_metric": ds["primary_metric"],
        "secondary_metrics": secondary,
        "split_policy": {"type": "time_based", **frag["windows"]},
        "features": {"exclude": []},
        "resource_budget": {
            "max_trials": int(budget.get("max_trials", 10)),
            "max_generations": int(budget.get("max_generations", 3)),
            "max_wall_time_hours": float(budget.get("wall_hours", 1.0)),
            "max_compute_hours": float(budget.get("wall_hours", 1.0)),
            "trial_timeout_minutes": int(budget.get(
                "trial_timeout_minutes", 30)),
        },
    }
    head = (f"# v2.19.0 benchmark run(scripts/benchmark/prepare.py 生成)\n"
            f"# 数据:{('OpenML ' + str(ds['openml_id'])) if ds.get('openml_id') else ds.get('source_desc', 'local')} {ds['name']};切分:"
            f"{frag['split_desc']}\n")
    return head + yaml.safe_dump(spec, allow_unicode=True, sort_keys=False)


LOADER_TEMPLATE = '''"""benchmark data-loader(v2.19.0 prepare 生成):读取已处理的建模样本。"""
import pathlib

import pandas as pd

DATA_PATH = pathlib.Path({path!r})


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)
'''


def _patch_config(cfg_path: Path, suite: dict, tier: str) -> None:
    """冷启动(library.enabled=false)+ 套件预算写进 run 配置。"""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    root = cfg.setdefault("modelevo", {})
    root.setdefault("library", {})["enabled"] = False
    budget = (suite.get("budget") or {}).get(tier) or {}
    evo = root.setdefault("evolution", {})
    if budget.get("max_generations"):
        evo["max_generations"] = int(budget["max_generations"])
    if budget.get("max_trials"):
        evo["max_trials"] = int(budget["max_trials"])
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True,
                                       sort_keys=False), encoding="utf-8")


def prepare_dataset(ds: dict, suite: dict, *, datasets_root: Path,
                    runs_root: Path, tier: str = "fast",
                    force: bool = False, download_fn=download) -> dict:
    raw = download_fn(ds, datasets_root, force=False)
    ratios = suite.get("ratios") or {"train": 0.6, "validation": 0.15,
                                     "shadow_test": 0.1, "final_test": 0.15}
    df, frag = process(ds, raw, ratios=ratios, seed=int(suite.get("seed", 0)))
    data_path = (datasets_root / ds["name"] / "data.parquet").resolve()
    df.to_parquet(data_path, index=False)
    run = (runs_root / ds["name"]).resolve()
    if (run / "task-spec.yaml").exists() and not force:
        return {"name": ds["name"], "run_dir": str(run), "status": "exists",
                **{k: frag[k] for k in ("n_rows", "n_columns", "split_desc")}}
    cmd = [*MODELEVO, "init", "--dir", str(run),
           "--task-id", "bench-" + ds["name"].replace("_", "-"),
           "--target", frag["target_name"],
           "--time-column", frag["time_column"],
           "--id-column", frag["id_column"]]
    if force:
        cmd.append("--force")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"init 失败({ds['name']}): {proc.stderr[-800:]}")
    (run / "task-spec.yaml").write_text(_spec_yaml(ds, frag, suite, tier),
                                        encoding="utf-8")
    (run / "data-loader.py").write_text(
        LOADER_TEMPLATE.format(path=str(data_path)), encoding="utf-8")
    # v2.19.0:数据字典(序列列的字段顺序等)随 run 下发,供 Agent 理解业务字段
    dd = datasets_root / ds["name"] / "DATA_DICTIONARY.md"
    if dd.exists():
        import shutil as _sh
        _sh.copy2(dd, run / "DATA_DICTIONARY.md")
    _patch_config(run / "modelevo-config.yaml", suite, tier)
    (run / "benchmark-task.json").write_text(json.dumps(
        {"suite_id": suite.get("suite_id"), "tier": tier, "dataset": ds,
         "data_path": str(data_path), **frag}, ensure_ascii=False, indent=1,
        default=str), encoding="utf-8")
    return {"name": ds["name"], "run_dir": str(run), "status": "prepared",
            **{k: frag[k] for k in ("n_rows", "n_columns", "split_desc")}}


def prepare_suite(suite_path: "Path | str", *, datasets_root: "Path | str",
                  runs_root: "Path | str", tier: str = "fast",
                  only: list[str] | None = None,
                  force: bool = False) -> list[dict]:
    suite = load_suite(suite_path)
    datasets_root, runs_root = Path(datasets_root), Path(runs_root)
    datasets_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    out = []
    for ds in select_datasets(suite, tier, only):
        try:
            out.append(prepare_dataset(ds, suite, datasets_root=datasets_root,
                                       runs_root=runs_root, tier=tier,
                                       force=force))
        except Exception as exc:  # noqa: BLE001 —— 单个数据集失败不阻塞其余
            out.append({"name": ds["name"], "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", default=str(SKILL / "assets" / "benchmark-suite.yaml"))
    ap.add_argument("--datasets-root", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--only", default=None, help="逗号分隔的数据集名")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 run 模板")
    a = ap.parse_args(argv)
    res = prepare_suite(a.suite, datasets_root=a.datasets_root,
                        runs_root=a.runs_root, tier=a.tier,
                        only=[x for x in (a.only or "").split(",") if x] or None,
                        force=a.force)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if all(r["status"] != "error" for r in res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
