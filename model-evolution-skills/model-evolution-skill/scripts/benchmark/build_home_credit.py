#!/usr/bin/env python
"""v2.19.0 benchmark:Home Credit Default Risk 多表 → 单表(主表列 + 历史明细序列串列)。

    python scripts/benchmark/build_home_credit.py --raw-dir <7 张 csv 所在目录> \
        --out-dir <datasets-root>/home_credit

为什么拼成序列串:ModelEvo 的输入是单表,行为历史以变长序列串列存在(与
req-t7 的 seq_d365 同形态)。这里**只做数据整形,不做任何聚合或特征构造**:每个
申请人(SK_ID_CURR)的每类历史明细按时间升序逐条序列化,原始字段原样保留;
解析与特征挖掘留给框架,这正是 benchmark 要测的能力。

产物:
- ``raw.parquet``:application_train 全部列(含 TARGET)+ 5 个序列列;
- ``DATA_DICTIONARY.md``:每个序列列的记录分隔符、字段顺序与官方字段说明。

序列格式:记录之间用 ``;``,记录内字段用 ``|``,字段顺序见数据字典;缺失值为空串;
字符串字段里的 ``|``/``;`` 替换为 ``/``。bureau 记录额外带 ``BB_STATUS_SEQ``:
该笔征信贷款在 bureau_balance 中按月(由远到近)的状态字符拼接(0~5=逾期档,
C=结清,X=未知)。application_test 无标签,不使用。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ID = "SK_ID_CURR"

#: 序列列 → (源表, 排序键, 序列化字段)。字段 = 原表除 SK_ID_CURR 外的全部列
#: (bureau 去掉纯 ID SK_ID_BUREAU、加 BB_STATUS_SEQ);SK_ID_PREV 保留,供框架
#: 在 prev/pos/installments/credit_card 之间按贷款关联。
TABLES = {
    "hist_bureau": ("bureau", ["DAYS_CREDIT"]),
    "hist_prev_app": ("previous_application", ["DAYS_DECISION", "SK_ID_PREV"]),
    "hist_pos_cash": ("POS_CASH_balance", ["MONTHS_BALANCE", "SK_ID_PREV"]),
    "hist_installments": ("installments_payments",
                          ["DAYS_INSTALMENT", "SK_ID_PREV",
                           "NUM_INSTALMENT_NUMBER"]),
    "hist_credit_card": ("credit_card_balance",
                         ["MONTHS_BALANCE", "SK_ID_PREV"]),
}


def _fmt(s: pd.Series) -> pd.Series:
    """向量化转字符串:整数原样,浮点保留 2 位(去掉尾随 .0),缺失 → 空串。"""
    if pd.api.types.is_float_dtype(s):
        if s.dropna().mod(1).eq(0).all():
            out = s.astype("Int64").astype(str)
        else:
            out = s.round(2).astype(str)
        return out.replace({"<NA>": "", "nan": ""})
    if pd.api.types.is_integer_dtype(s):
        return s.astype(str)
    return s.astype(str).replace({"nan": "", "None": ""}).str.replace(
        r"[|;]", "/", regex=True)


def _serialize(df: pd.DataFrame, fields: list[str], sort_keys: list[str],
               name: str) -> pd.Series:
    t0 = time.time()
    df = df.sort_values([ID, *sort_keys], kind="mergesort")
    rec = _fmt(df[fields[0]])
    for f in fields[1:]:
        rec = rec.str.cat(_fmt(df[f]), sep="|")
    out = rec.groupby(df[ID].to_numpy(), sort=False).agg(";".join)
    print(f"[home_credit] {name}: {len(df)} 条 → {len(out)} 个申请人 "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def build(raw_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    app = pd.read_csv(raw_dir / "application_train.csv")
    print(f"[home_credit] application_train {app.shape}", flush=True)
    fields_doc: dict[str, list[str]] = {}

    bb = pd.read_csv(raw_dir / "bureau_balance.csv")
    bb = bb.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"], kind="mergesort")
    bb_seq = bb.groupby("SK_ID_BUREAU", sort=False)["STATUS"].agg(
        lambda x: "".join(x.astype(str)))
    del bb
    for col, (table, keys) in TABLES.items():
        df = pd.read_csv(raw_dir / f"{table}.csv")
        if table == "bureau":
            df["BB_STATUS_SEQ"] = df["SK_ID_BUREAU"].map(bb_seq)
            df = df.drop(columns=["SK_ID_BUREAU"])
        fields = [c for c in df.columns if c != ID]
        fields_doc[col] = fields
        ser = _serialize(df, fields, keys, col)
        app[col] = app[ID].map(ser)
        del df, ser
    path = out_dir / "raw.parquet"
    app.to_parquet(path, index=False)
    _write_dictionary(raw_dir, out_dir, fields_doc)
    print(f"[home_credit] 写出 {path} {app.shape}", flush=True)
    return path


def _write_dictionary(raw_dir: Path, out_dir: Path,
                      fields_doc: dict[str, list[str]]) -> None:
    desc = {}
    try:
        d = pd.read_csv(raw_dir / "HomeCredit_columns_description.csv",
                        encoding="latin1")
        for _, r in d.iterrows():
            desc.setdefault(str(r["Row"]), str(r["Description"]))
    except Exception:  # noqa: BLE001 —— 说明文件缺失不影响数据
        pass
    lines = ["# Home Credit 序列列数据字典(v2.19.0 benchmark)", "",
             "记录之间 `;`,记录内字段 `|`,缺失为空串;每个申请人的记录按时间"
             "升序(DAYS_*/MONTHS_BALANCE 为相对申请日的负数,越接近 0 越近)。",
             ""]
    for col, fields in fields_doc.items():
        lines += [f"## {col}({TABLES[col][0]},排序键 "
                  f"{', '.join(TABLES[col][1])})", "",
                  "| # | 字段 | 说明 |", "|---|---|---|"]
        for i, f in enumerate(fields):
            note = desc.get(f, "")
            if f == "BB_STATUS_SEQ":
                note = ("bureau_balance 按月(由远到近)状态字符拼接:0=无逾期,"
                        "1~5=逾期档(1:1-30 天 … 5:>120 天或核销),C=结清,"
                        "X=未知")
            lines.append(f"| {i} | {f} | {note.replace('|', '/')[:160]} |")
        lines.append("")
    (out_dir / "DATA_DICTIONARY.md").write_text("\n".join(lines),
                                                encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    build(Path(a.raw_dir), Path(a.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
