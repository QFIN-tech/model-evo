"""v2.13 修复批次回归测试:`write-splits-view` 在 gate 保护型 run 上 100% 失败。

事故来源:run-draw-req-t7(run-1)与 run-draw-req-t7-v2(2026-09-20)实测。
两个 run 的 `sample-profile/splits/labels/final_test.parquet` 均**不存在**
(v2.13 的 final_test 标签不落盘,`labels_materialized=false`),
于是 `write-splits-view` **在任何视图脚本下都落不了盘** —— 实测 no-op 探针、
`--template kwin` 骨架、自定义 seqc 脚本三条路径同一行报错:

    InvalidIndexError: Reindexing only valid with uniquely valued Index objects

根因(`cmd_write_splits_view`,拼 `_parts` 段):

    for n in SPLIT_NAMES:
        _cols = [_face]
        if n in labels:   _cols.append(labels[n])      # ← train/val/shadow 走这
        elif n == "final_test": _cols.append(_stub)    # ← final_test 走这
        ...
        _parts.append(pd.concat(_cols, axis=1))        # ← 此处就抛
    df_v1 = pd.concat(_parts, ignore_index=True, copy=False)

`_stub` 只建 `target`(binary 且无 treatment 时**只此一列**),而真实标签面
是 `[target, time_column, id_column, SAMPLE_KEY_COLUMN]` 四列。两侧同一
`pd.concat(objs, axis=1)` 调用的列多重集因此不同:

    part(train,..)      = 20 列, 且 `__mvo_sample_key` 出现 2 次
    part(final_test)    = 18 列, 且 `__mvo_sample_key` 出现 1 次

**关键机制**:`pd.concat` 对双参数形态的 fast path 条件是
`all(obj.columns.equals(objs[0].columns) for obj in objs)` —— 列**数值相等**
即走快路径(不再做 get_indexer)。真正抛错的是 **outer concat 的列对齐**:
因为两个 part 的列多重集不同(pandas 2.3 把索引化的 `get_indexer` 从
`need_unique` 放宽为 `allow_duplicates`,而两处 objs[0] 多重集不同),
`concat.py` 的 `obj_labels.get_indexer(new_labels)` 抛 `InvalidIndexError`。

最小合成复现:
    A=[x,x,a], B=[x,b]      -> FAIL
    A=[x,x,a], B=[x,x,a]    -> OK   (多重集一致)

修法:`_stub` 补齐真实标签面的**全部列名**(顺序同 labels 面),使两侧一致。
final_test 绝不引入真标签 —— 缺的列填 NaN(stub 语义),`_identity` 的
`__mvo_sample_key` 仍在下方 append,不受影响。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

from runtime.data import (SAMPLE_KEY_COLUMN, IDENTITY_SUBDIR, SPLIT_NAMES,
                          load_identities, load_labels, load_split_frames,
                          write_splits)


def _spec():
    from runtime.spec import TaskSpec
    return TaskSpec(data={
        "task_id": "v213view", "task_name": "v213view",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
    })


def _make_run(tmp_path: Path):
    """造一个 v2.13 形态的 run:final_test 标签**不落盘**。"""
    df = pd.DataFrame({
        "dt": [f"2026-0{m}-01" for m in range(1, 5) for _ in range(4)],
        "uid": list(range(16)), "x": np.arange(16, dtype=float),
        "label": [0, 1, 0, 1] * 4,
    })
    run = tmp_path / "run"
    base = run / "sample-profile" / "splits"
    write_splits(df, _spec(), base)
    return run, base


def _rebuild_df_v1_as_cli_does(base: Path, stub_all_columns: bool):
    """复刻 `cmd_write_splits_view` 里 `_parts` -> `df_v1` 的那段。

    `stub_all_columns`:
      False = 事故当时的写法(final_test stub 只建 target 列)
      True  = 修复后的写法(stub 补齐真实标签面全部列名)
    """
    frames = load_split_frames(base.parent.parent)
    labels = load_labels(base.parent.parent)
    identities = load_identities(base.parent.parent)
    spec = _spec()
    parts = []
    for n in SPLIT_NAMES:
        if n not in frames:
            continue
        _face = frames[n].reset_index(drop=True)
        _identity = identities[n].reset_index(drop=True)
        _cols = [_face]
        if n in labels:
            _cols.append(labels[n].reset_index(drop=True))
        else:
            assert n == "final_test"
            if stub_all_columns:
                # 修复后:stub 与真实 labels 面**同列同序**(含
                # SAMPLE_KEY_COLUMN)。这样 final_test 与可搜索 split 的
                # part 列多重集完全一致,outer concat 不再抛
                # InvalidIndexError;且去重(keep="first")保留的
                # SAMPLE_KEY_COLUMN 就是 identity 的真实键。
                _stub = pd.DataFrame(index=range(len(_face)))
                for c in labels["validation"].columns:
                    if c == spec.target_name:
                        _stub[c] = float("nan")
                    elif c == spec.time_column:
                        _stub[c] = pd.NaT
                    elif c == SAMPLE_KEY_COLUMN:
                        _stub[c] = _identity[SAMPLE_KEY_COLUMN].astype(
                            str).values
                    else:
                        _stub[c] = _identity[SAMPLE_KEY_COLUMN].astype(
                            str).values
            else:
                _stub = pd.DataFrame(index=range(len(_face)))
                _stub[spec.target_name] = float("nan")
                if spec.id_column:
                    _stub[spec.id_column] = _identity[
                        SAMPLE_KEY_COLUMN].astype(str).values
            _cols.append(_stub)
        _cols.append(_identity)
        parts.append(pd.concat(_cols, axis=1))
    return pd.concat(parts, ignore_index=True, copy=False)


def test_cli_stub_uses_real_time_column():
    """静态守卫:final_test stub 的时间列必须取 identity 面的真实值。

    stub 在 `_cols` 里排在 identity 之前,而 df_v1 按"首次出现优先"
    去重 —— 填 NaT 会把真实时间坐标顶掉。实测后果:final_test 视图的
    identity 面 dt 全为 NaT,`_sort_split_frame` 随之失序,视图
    final_test 的行序与父视图不再逐位置一致(实测 42685/42687 行错位)。
    """
    import ast
    src = (SKILL / "scripts" / "_mvo" / "cmd_views.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "cmd_write_splits_view")
    seg = ast.get_source_segment(src, fn)
    assert "_identity[_c].values" in seg, (
        "final_test stub 的时间列未取 identity 面真实值 —— "
        "会让视图 final_test 行序失序")
    assert "_stub[_c] = pd.NaT" not in seg, (
        "检测到用 NaT 填时间列的旧形态")


def test_old_stub_shape_reproduces_invalid_index_error(tmp_path: Path):
    """锁定事故:final_test stub 列集与可搜索 split 不同 => concat 抛错。"""
    run, base = _make_run(tmp_path)
    try:
        _rebuild_df_v1_as_cli_does(base, stub_all_columns=False)
    except pd.errors.InvalidIndexError as exc:
        assert "uniquely valued" in str(exc)
    else:
        raise AssertionError(
            "事故未复现 —— 说明 pandas 行为已变或 stub 形态已改动,"
            "请重新核对根因")


def test_full_column_stub_concatenates_cleanly(tmp_path: Path):
    """锁定修复:stub 补齐真实标签面全部列后,拼装不再抛错。"""
    run, base = _make_run(tmp_path)
    df_v1 = _rebuild_df_v1_as_cli_does(base, stub_all_columns=True)
    assert len(df_v1) == 16
    # CLI 在拼装后按"首次出现优先"去重对齐键(见 cmd_write_splits_view)。
    # 本测试复刻该步,再核验行序与 SPLIT_NAMES 拼接序一致。
    df_v1 = df_v1.loc[:, ~df_v1.columns.duplicated(keep="first")].copy()
    assert not df_v1.columns.duplicated().any()
    keys = pd.concat(
        [load_identities(run)[n][SAMPLE_KEY_COLUMN] for n in SPLIT_NAMES],
        ignore_index=True).astype(str)
    assert list(df_v1[SAMPLE_KEY_COLUMN].astype(str)) == list(keys)


def test_cli_stub_builds_full_label_columns():
    """静态守卫:CLI 的 final_test 分支必须补齐真实标签面的全部列名。

    只建 target 的旧写法会让 final_test 的列多重集与可搜索 split 不同,
    `df_v1` 的 outer concat 必抛 InvalidIndexError。
    """
    import ast
    src = (SKILL / "scripts" / "_mvo" / "cmd_views.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "cmd_write_splits_view")
    seg = ast.get_source_segment(src, fn)
    assert 'n == "final_test"' in seg, "final_test 分支锚点未命中"
    # 修复后:必须从已有 labels 面取列集(而不是只写 target 一个列名),
    # 并且要按该列集循环补齐。
    assert "_ref = next(" in seg and "_ref.columns" in seg, (
        "final_test stub 未按真实标签面列集补齐 —— "
        "这正是 InvalidIndexError 的根因")
    assert "for _c in _stub_cols" in seg, "stub 未按列集循环"
    # 且不得退回"只建 target 列"的形态(target 赋值必须是循环体的一部分)
    assert 'stub[spec.target_name] = float("nan")\n' not in seg, (
        "检测到旧 stub 形态(直接给 _stub[target] 赋值)")


def test_reparse_present_still_concatenates(tmp_path: Path):
    """reparse 源存在时(四产物齐)也必须能拼装,不能因 stub 形态再炸。"""
    run, base = _make_run(tmp_path)
    rp = base / "reparse"
    rp.mkdir(parents=True, exist_ok=True)
    for n in SPLIT_NAMES:
        ids = load_identities(run)[n]
        pd.DataFrame({"seq_x": [""] * len(ids), "dt": ids["dt"].values,
                      "uid": ids[SAMPLE_KEY_COLUMN].astype(str).values}
                     ).to_parquet(rp / f"{n}.parquet", index=False)
    assert (base / "reparse" / "final_test.parquet").exists()
    df_v1 = _rebuild_df_v1_as_cli_does(base, stub_all_columns=True)
    assert len(df_v1) == 16


# ---------------------------------------------------------------------------
# Bug#2:视图消费 reparse 源后剔除原文 => write_view_splits 曾硬报错
# ---------------------------------------------------------------------------
def _spec_with_reparse():
    from runtime.spec import TaskSpec
    return TaskSpec(data={
        "task_id": "v213view2", "task_name": "v213view2",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
        "features": {"reparse": [
            {"column": "seq_raw", "reason": "form_unusable",
             "plan": "按 ',' 切分派生计数列,原文不入面"}]},
    })


def test_view_consuming_reparse_is_accepted(tmp_path: Path):
    """视图脚本消费 reparse 源并剔除原文时,落盘必须成功。

    契约(task-and-data.md「数据视图」):reparse 只能消费 spec 授权源,
    视图产物中 banned 永不可恢复。真实视图**总是**把解析原文剔掉(只
    留下派生列),否则等于把长文本原样搬进特征面。修复前
    `write_view_splits` 走 `_split_face_frames` 的"声明了 reparse 却
    不在数据中"硬错误分支 —— 任何真实消费 reparse 的视图都落不了盘。
    """
    from runtime.data import write_view_splits
    spec = _spec_with_reparse()
    df = pd.DataFrame({
        "dt": [f"2026-0{m}-01" for m in range(1, 5) for _ in range(4)],
        "uid": list(range(16)), "x": np.arange(16, dtype=float),
        "seq_raw": ["a,b"] * 16, "label": [0, 1, 0, 1] * 4,
    })
    run = tmp_path / "run"
    base = run / "sample-profile" / "splits"
    write_splits(df, spec, base)
    identities = load_identities(run)
    labels = load_labels(run)
    parts = []
    for name in SPLIT_NAMES:
        ids = identities[name]
        face = pd.read_parquet(base / f"{name}.parquet")
        frame = face.copy()
        if name in labels:
            frame["label"] = labels[name]["label"].values
        else:
            frame["label"] = float("nan")
        frame["uid"] = ids[SAMPLE_KEY_COLUMN].astype(str).values
        frame["dt"] = ids["dt"].values
        frame["__mvo_sample_key"] = ids[SAMPLE_KEY_COLUMN].astype(str).values
        # reparse 面按契约拼回(CLI 会这么做),视图消费后剔除原文
        rep = pd.read_parquet(base / "reparse" / f"{name}.parquet")
        frame["seq_raw"] = rep["seq_raw"].values
        frame["seq_len"] = 2.0
        frame = frame.drop(columns=["seq_raw"])
        parts.append(frame)
    view = pd.concat(parts, ignore_index=True)
    manifest = write_view_splits(view, spec, run, "vrep")
    assert manifest["n_features"] >= 1
    out = run / "sample-profile" / "splits-vrep"
    assert (out / "train.parquet").exists()
    assert "seq_raw" not in pd.read_parquet(out / "train.parquet").columns


def test_profile_path_still_rejects_missing_reparse(tmp_path: Path):
    """守住第二道:profile 路径下 reparse 列缺失仍是硬错误(不静默落空)。"""
    from runtime.data import _split_face_frames
    spec = _spec_with_reparse()
    chunk = pd.DataFrame({"dt": ["2026-01-01"], "uid": ["u"], "x": [1.0],
                          "label": [0]})
    try:
        _split_face_frames(chunk, spec)
    except ValueError as exc:
        assert "不在数据中" in str(exc)
    else:
        raise AssertionError("profile 路径丢失了 reparse 缺失的硬报错")
