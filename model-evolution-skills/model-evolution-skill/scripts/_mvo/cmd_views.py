"""数据视图命令:write-splits-view / write-splits-sample

v2.11.9 从单文件 `scripts/modelevo.py` 拆出(拆分动机见
`_mvo/__init__.py`)。新增命令请按职责放进对应模块,不要回填到
`modelevo.py`(那里只留 argparse 与分发)。

v2.17.0(WP5b):write-splits-view 默认流式构建(--max-mem-gb /
--chunk-rows / --no-stream)、--from-feature-block;write-splits-sample
落盘后做抽样面噪声预检(manifest.noise_precheck)。
"""
from __future__ import annotations

from pathlib import Path
from runtime.data import load_dataframe, resolve_feature_columns  # noqa: E402
from .context import _load_spec, _run_dir_from, logger


def cmd_write_splits_view(args) -> int:
    """数据视图 splits 落盘:build_view(df_v1) -> df_view,沿用 v1 行归属。

    契约(references/task-and-data.md「数据视图」节):
    - --source 脚本暴露 build_view(df) -> DataFrame;
    - 输入为 v1 splits 合并序的完整样本；可搜索 split 从标签面拼回
      目标/ID，final_test 只提供空目标占位与哈希 ID；四产物下特征面
      物理不含这些保护列;
    - 输出行数/行序/样本键必须与输入完全一致,只允许替换特征列;
    - 脚本必须先落盘(通常放 run 目录 views/{view_id}.py)再引用。
    """
    # v19.3 修复(作用域陷阱):SPLIT_NAMES/load_labels/load_split_frames
    # 必须在本函数**最开头**绑定。旧实现把它们 import 在函数中后段,
    # 而骨架字面量注入段(更早执行)已引用 SPLIT_NAMES —— Python 因
    # 该名字在本函数内被赋值而判其为局部变量,前段引用抛
    # UnboundLocalError,又被注入段的 `except Exception` 静默吞掉:
    # `_SPLIT_ROWS`/`_PROTECTED_COLS` 永远为空占位(实测 gen-03 的
    # recency/kwin 骨架两个占位都没注入,而 recency 模板的加权泄漏
    # 防线正是靠 _PROTECTED_COLS —— 空值 = 标签列被加权成 AUC≈1 伪特征)。
    from runtime.data import (IDENTITY_SUBDIR, SAMPLE_KEY_COLUMN, SPLIT_NAMES,
                              load_identities, load_labels,
                              load_split_frames, write_view_splits)

    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    view_id = (args.view_id or "").strip()
    if not view_id or view_id in ("v1", "default"):
        logger.error("--view-id 不能为空或保留字 v1/default")
        return 1
    # v2.17.0(#12):已缓存/可重算的特征块直接材料化为视图列(不重载原始数据、
    # 不经 build_view);与 --source/--template 互斥
    if getattr(args, "from_feature_block", None):
        if getattr(args, "source", None) or getattr(args, "template", None):
            logger.error("--from-feature-block 与 --source/--template 互斥")
            return 2
        return _cmd_view_from_feature_blocks(args, spec, run_dir, view_id)
    # v16/P1-1:模板通道——--template 参数化生成 views/{id}.py 骨架,
    # 把"换聚合口径"从"LLM 从零写构建脚本"降为模板 + 参数(14 个
    # 历史 run 只用过 3 次真实 data_view 的摩擦根因)。骨架保留
    # 人工审查点(生成后打印路径,审查再落盘 splits)。
    _template = getattr(args, "template", None)
    if _template and not getattr(args, "source", None):
        import pandas as pd

        from runtime.view_templates import render_view_template

        views_dir = run_dir / "views"
        views_dir.mkdir(exist_ok=True)
        src_path = views_dir / f"{view_id}.py"
        _rendered = render_view_template(
            _template, view_id=view_id,
            half_life=float(getattr(args, "half_life", 30.0) or 30.0),
            k_periods=int(getattr(args, "k_periods", 3) or 3),
            active_col=getattr(args, "active_col", None),
            time_column=spec.time_column,
            target_name=spec.target_name,
        )
        # v19(req-t7-drawdown gen-04 审查回写):向骨架注入两个
        # 字面量占位——(a) 各 split 行数(recency 逐段参考日的切段
        # 依据;旧实现全局 max 参考日,train 段衰减尺度跨 split 漂移);
        # (b) 保护列清单(target/ID/treatment/banned/reparse,防
        # 标签列被加权成 w60_<target> 的 AUC≈1 伪特征)。
        # v19.3:注入不再"失败即静默回落"——占位注入是泄漏防线的一部分
        # (见 cmd_write_splits_view 头注),静默失败曾使 _PROTECTED_COLS
        # 恒为空,标签加权列直达数据面。此处失败一律 abort 生成。
        try:
            from runtime.data import load_split_frames as _lsf  # noqa: F401
        except ImportError as _exc:  # noqa: BLE001
            logger.error("骨架注入依赖的 runtime.data 不可导入: %s", _exc)
            return 1

        try:
            # v2.17.0:行数取 parquet 元数据、参考日取 identity 面时间列 ——
            # 旧实现 _lsf(run_dir) 为数行数全量加载 v1 四个 split 的特征面
            # (4M×1513 列即数十 GB)
            import pyarrow.parquet as _pq

            _v1 = run_dir / "sample-profile" / "splits"
            _names = [n for n in SPLIT_NAMES
                      if (_v1 / f"{n}.parquet").exists()]
            _rows_lit = [int(_pq.ParquetFile(_v1 / f"{n}.parquet")
                             .metadata.num_rows) for n in _names]
            _ids = load_identities(run_dir)
            _refs_lit = []
            for n in _names:
                _tc = spec.time_column
                if _tc and n in _ids and _tc in _ids[n].columns:
                    _mx = pd.to_datetime(_ids[n][_tc], errors="coerce").max()
                    _refs_lit.append(str(_mx))
                else:
                    _refs_lit = []
                    break
            _prot = [spec.target_name]
            for _c in (spec.id_column, spec.treatment_column):
                if _c:
                    _prot.append(_c)
            _prot += list(spec.banned_columns)
            _prot += list(spec.reparse_columns)
            _prot.append(SAMPLE_KEY_COLUMN)
            _rows_tok = repr([int(x) for x in _rows_lit])
            _prot_tok = repr(sorted(set(_prot)))
        except Exception as exc:  # noqa: BLE001
            logger.error("骨架字面量计算失败,拒绝生成无防护的视图骨架"
                         "(旧实现在此静默回落占位 —— 保护列为空等于"
                         "关闭 recency 的标签泄漏防线): %s", exc)
            return 1

        _row_pat = "_SPLIT_ROWS = []  # @SPLIT_ROWS@"
        _prot_pat = "_PROTECTED_COLS = []  # @PROTECTED_COLS@"
        if _row_pat not in _rendered or _prot_pat not in _rendered:
            logger.error("骨架占位锚点未命中(模板结构变更,请同步 "
                         "cmd_write_splits_view): rows=%s prot=%s",
                         _row_pat in _rendered, _prot_pat in _rendered)
            return 1
        _rendered = _rendered.replace(
            _row_pat, f"_SPLIT_ROWS = {_rows_tok}  # @SPLIT_ROWS@"
        ).replace(
            _prot_pat, f"_PROTECTED_COLS = {_prot_tok}  # @PROTECTED_COLS@"
        ).replace(
            "_SPLIT_REF_DATES = []  # @SPLIT_REF_DATES@",
            f"_SPLIT_REF_DATES = {_refs_lit!r}  # @SPLIT_REF_DATES@")
        # 替换后自检:占位必须是"非空字面量 + 注释尾巴仍在"(注释尾巴
        # 是人工审查点的可读性锚;字面量非空是防线生效的必要条件)。
        if "@SPLIT_ROWS@" not in _rendered or "@PROTECTED_COLS@" \
                not in _rendered:
            logger.error("骨架注入自检失败:占位标记丢失")
            return 1
        logger.info("骨架注入: split 行数 %s,保护列 %d 个", _rows_lit,
                    len(set(_prot)))
        src_path.write_text(_rendered, encoding="utf-8")
        logger.info("视图模板 %s 已生成: %s(人工审查后以 --source "
                    "该路径落盘 splits)", _template, src_path)
        args.source = str(src_path)
    source = Path(args.source) if getattr(args, "source", None) else None
    if source is None or not source.exists():
        logger.error("视图构建脚本不存在: %s(先落盘再执行,铁律 2;"
                     "或用 --template 生成骨架)", source)
        return 1

    import importlib.util

    import pandas as pd

    mod_spec = importlib.util.spec_from_file_location(
        f"modelevo_view_{view_id}", source)
    module = importlib.util.module_from_spec(mod_spec)
    try:
        mod_spec.loader.exec_module(module)
        _build_view = module.build_view
    except Exception as exc:  # noqa: BLE001
        logger.error("build_view 加载失败(%s): %s", source, exc)
        return 1

    # v2.17.0(#12):默认流式构建(按 split × 分块,内存有界)。build_view
    # 非行内变换(自检失败)时,估算全量内存 ≤ 预算才回退旧全量模式,否则拒绝。
    if not getattr(args, "no_stream", False):
        from runtime.data import (ViewNotRowLocalError, _ParentPlan,
                                  write_view_splits_streaming)

        _max_gb, _chunk, _fb_gb = _view_stream_budget(args, run_dir)
        try:
            manifest = write_view_splits_streaming(
                _build_view, spec, run_dir, view_id, chunk_rows=_chunk,
                max_mem_gb=_max_gb, note=args.note or "",
                source_script=source)
        except ViewNotRowLocalError as exc:
            try:
                _est = _ParentPlan(spec, run_dir).estimate_full_gb()
            except Exception:  # noqa: BLE001
                _est = float("inf")
            _limit = _max_gb or _fb_gb
            if _est <= _limit:
                logger.warning("%s;估算全量模式峰值 %.2f GB ≤ 预算 %.1f GB,"
                               "回退全量模式", exc, _est, _limit)
            else:
                logger.error(
                    "%s。估算全量模式峰值 %.1f GB > 预算 %.1f GB,拒绝回退"
                    "(会把共享机吃满)。请把 build_view 改写为行内变换(输出"
                    "第 i 行只依赖输入第 i 行,见 references/task-and-data.md"
                    "「数据视图」),或用 --from-feature-block 把已验证的特征块"
                    "直接材料化为视图", exc, _est, _limit)
                return 1
        except Exception as exc:  # noqa: BLE001
            logger.error("流式视图落盘失败(%s): %s", source, exc)
            return 1
        else:
            return _print_view_manifest(run_dir, view_id, manifest)

    else:
        try:
            from runtime.data import _ParentPlan

            _est = _ParentPlan(spec, run_dir).estimate_full_gb()
            if _est > 32:
                logger.warning("--no-stream 全量模式:估算峰值内存 %.0f GB"
                               "(共享机慎用;行内变换请去掉 --no-stream)", _est)
        except Exception:  # noqa: BLE001
            pass
    # v19.3:原在此处的 `from runtime.data import (SPLIT_NAMES, ...)` 已
    # 上移到函数开头(见函数文档尾注),此处不再重复 import —— 重复
    # 绑定正是 UnboundLocalError 作用于前段引用的成因。
    frames = load_split_frames(run_dir)
    if "train" not in frames:
        logger.error("v1 splits 不存在:先执行 profile")
        return 1
    # v21 四产物:build_view 的输入需要保护列,而特征面物理不含它们。
    # 可搜索 split 拼回真实标签面；final_test 只拼 identity 面、空目标
    # 占位和哈希 ID，占位只满足视图函数的结构契约，绝不落标签面。
    # v19(req-t7-drawdown gen-04 实测回写):reparse 列同样拼回——
    # 视图框架此前只拼 face+labels,spec 声明 reparse 时
    # _split_face_frames 在落盘期显式 raise("reparse 列不在数据中"),
    # recency/kwin 视图在声明 reparse 的任务上必然失败,只能 trial
    # 内手拼。拼回后视图的 reparse/ 产物由 write_view_splits 正常切出。
    labels = load_labels(run_dir)
    identities = load_identities(run_dir)
    reparse_dir = Path(run_dir) / "sample-profile" / "splits" / "reparse"
    reparse_frames = {}
    if spec.reparse_columns and reparse_dir.is_dir():
        for n in SPLIT_NAMES:
            _rp = reparse_dir / f"{n}.parquet"
            if _rp.exists():
                reparse_frames[n] = pd.read_parquet(_rp)
    _parts = []
    for n in SPLIT_NAMES:
        if n not in frames:
            continue
        if n not in identities:
            logger.error("identity 面缺失 %s/%s.parquet；请重新 profile",
                         IDENTITY_SUBDIR, n)
            return 1
        _face = frames[n].reset_index(drop=True)
        _identity = identities[n].reset_index(drop=True)
        if len(_face) != len(_identity):
            logger.error("%s 的特征面与 identity 面行数不一致", n)
            return 1
        _cols = [_face]
        if n in labels:
            _cols.append(labels[n].reset_index(drop=True))
        elif n == "final_test":
            # v2.13.1 修复:final_test 标签**不落盘**(v21 起
            # labels_materialized=false,仅 finalizer 过 gate 后内存重建)。
            # 此前 stub 只建 target(+treatment/id)列,列多重集与可搜索
            # split 的真实标签面 [target, time, id, SAMPLE_KEY] 不同 ——
            # 于是下方 df_v1 的 outer concat 在列对齐时抛
            # `InvalidIndexError: Reindexing only valid with uniquely valued
            # Index objects`(pandas 2.x:concat.py `obj_labels.get_indexer`;
            # 两侧 objs[0] 多重集不同时不再走 allow_duplicates 路径)。
            # 实测 run-draw-req-t7 / -v2(2026-09-20)的 write-splits-view
            # 因此**在任何视图脚本下都失败**(no-op 探针、--template kwin
            # 骨架、自定义脚本同一行报错)。修法:stub 补齐真实标签面的
            # **全部列名与顺序**,缺的填 NaN/NaT/哈希键 —— final_test
            # 绝不引入真标签,只满足 build_view 的结构契约。
            _ref = next((labels[s] for s in SPLIT_NAMES if s in labels), None)
            _stub = pd.DataFrame(index=range(len(_face)))
            _stub_cols = list(_ref.columns) if _ref is not None else [
                c for c in (spec.target_name, spec.treatment_column,
                            spec.id_column) if c]
            _key_vals = _identity[SAMPLE_KEY_COLUMN].astype(str).values
            for _c in _stub_cols:
                if _c in (spec.target_name, spec.treatment_column):
                    _stub[_c] = float("nan")
                elif _c == spec.time_column:
                    # 时间列必须用 **identity 面的真实值**:stub 在
                    # `_columns` 里排在 _identity 之前,而下方按
                    # "首次出现优先"去重,填 NaT 会把真实时间坐标顶掉
                    # —— 实测后果是 final_test 视图的 identity 面 dt
                    # 全为 NaT,`_sort_split_frame` 随之失序,视图
                    # final_test 的行序与父视图不再逐位置一致。
                    _stub[_c] = (_identity[_c].values
                                 if _c in _identity.columns
                                 else pd.NaT)
                else:
                    # id_column / SAMPLE_KEY_COLUMN 等标识列:沿用旧语义,
                    # 用不可逆哈希样本键占位(不泄露真实实体 ID)
                    _stub[_c] = _key_vals
            _cols.append(_stub)
        else:
            logger.error("标签面缺失: %s/%s.parquet(先跑 profile)",
                         run_dir / "sample-profile/splits/labels", n)
            return 1
        _cols.append(_identity)
        if n in reparse_frames:
            _cols.append(reparse_frames[n].reset_index(drop=True))
        _parts.append(pd.concat(_cols, axis=1))
    if not _parts:
        logger.error("splits 数据产物缺失: %s(先跑 profile)",
                     run_dir / "sample-profile/splits")
        return 1
    df_v1 = pd.concat(_parts, ignore_index=True, copy=False)
    # v19.3 修复(broadcast 崩溃根因):多个数据面共含对齐键。labels 面是
    # [target, pday, user_no],reparse 面是 [seq_*, pday, user_no] ——
    # 逐 split concat(axis=1) 后 pday/user_no 各出现两次,df_v1 列数
    # 1500 但不同名仅 1498。后果两处必炸:
    #   (a) 视图脚本对 df["pday"] 取列得 DataFrame → pd.to_datetime
    #       抛 "cannot assemble with duplicate keys";
    #   (b) write_view_splits 的 ID 对齐守卫 pv=parent_ids...values 拿到
    #       (300000, 2) → "operands could not be broadcast together with
    #       shapes (300000,) (300000,2)"(gen-03 §10.1 升级到人的那条)。
    # 修法:拼回后按"首次出现优先"去重 —— 键列语义两处完全同源
    # (同 split、同序、同值),保留 labels 面那一份即可,特征列不受影响。
    _dup = [c for c in df_v1.columns[df_v1.columns.duplicated()]]
    if _dup:
        df_v1 = df_v1.loc[:, ~df_v1.columns.duplicated(keep="first")].copy()
        logger.info("df_v1 对齐列去重: %s(labels/reparse 面同源同名)",
                    sorted(set(_dup)))
    # 去重后仍必须"无重名列"——守住第二道:任何残留重名都会让下游
    # 取列语义变得依赖 pandas 版本,不静默放行。
    if df_v1.columns.duplicated().any():
        logger.error("df_v1 仍存在重名列,拒绝落盘(视图输入必须单义): %s",
                     sorted(set(df_v1.columns[df_v1.columns.duplicated()])))
        return 1

    try:
        df_view = _build_view(df_v1)
    except Exception as exc:  # noqa: BLE001
        logger.error("build_view 执行失败(%s): %s", source, exc)
        return 1
    if not isinstance(df_view, pd.DataFrame):
        logger.error("build_view 必须返回 DataFrame,得到 %s", type(df_view))
        return 1

    manifest = write_view_splits(df_view, spec, run_dir, view_id,
                                 note=args.note or "",
                                 source_script=source)
    try:
        from runtime.data import BUILD_INFO_FILE
        from runtime.logging_utils import dump_json as _dj, now_iso as _ni

        _dj(run_dir / "sample-profile" / f"splits-{view_id}" / BUILD_INFO_FILE,
            {"mode": "full", "created_at": _ni(),
             "note": "全量模式(--no-stream 或流式自检失败后的回退)"})
    except Exception:  # noqa: BLE001
        pass
    return _print_view_manifest(run_dir, view_id, manifest)


def _view_stream_budget(args, run_dir: Path):
    """(max_mem_gb, chunk_rows, full_fallback_max_gb):CLI 显式 > run 配置。"""
    from .context import _load_config

    try:
        cfg = _load_config(run_dir)
    except Exception:  # noqa: BLE001
        cfg = None

    def _cfg(key, default):
        return cfg.get(key, default) if cfg is not None else default

    max_gb = getattr(args, "max_mem_gb", None)
    if max_gb is None:
        max_gb = _cfg("feature_exploration.view_stream_max_mem_gb", None)
    chunk = getattr(args, "chunk_rows", None) or _cfg(
        "feature_exploration.view_stream_chunk_rows", 200000)
    fb = float(_cfg("feature_exploration.view_full_fallback_max_gb", 8) or 8)
    return (float(max_gb) if max_gb else None), int(chunk), fb


def _print_view_manifest(run_dir: Path, view_id: str, manifest: dict) -> int:
    out_dir = run_dir / "sample-profile" / f"splits-{view_id}"
    print(f"数据视图 {view_id} 已落盘: {out_dir}")
    print(f"  features={manifest.get('n_features')} "
          f"feature_sha256={manifest.get('feature_sha256')}")
    if manifest.get("source_script"):
        print(f"  构建脚本: {manifest['source_script']}"
              f"(sha256={manifest.get('source_script_sha256')},"
              "请保留在 run 目录,视图凭它可复现)")
    for name, entry in manifest["splits"].items():
        print(f"  {name}: rows={entry['rows']}")
    try:
        from runtime.data import BUILD_INFO_FILE
        from runtime.logging_utils import load_json as _lj

        _bi = _lj(out_dir / BUILD_INFO_FILE)
        print(f"  构建方式: {_bi.get('mode')}"
              + (f" chunks={_bi.get('n_chunks')} chunk_rows="
                 f"{_bi.get('chunk_rows_final')} peak_rss="
                 f"{_bi.get('peak_rss_gb')}GB" if _bi.get('mode') == 'stream'
                 else ""))
        if _bi.get("value_check"):
            print(f"  逐值一致性: {_bi['value_check']}")
    except Exception:  # noqa: BLE001
        pass
    print(f"  下一步: new-trial --action data_view --data-view {view_id} --heavy")
    return 0


def _cmd_view_from_feature_blocks(args, spec, run_dir: Path,
                                  view_id: str) -> int:
    """v2.17.0:--from-feature-block <trial>:<block>[,...]。"""
    from runtime.data import write_view_from_feature_blocks

    specs = []
    for tok in str(args.from_feature_block).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            logger.error("--from-feature-block 需形如 <trial>:<block>,得到 %r",
                         tok)
            return 2
        tid, blk = tok.split(":", 1)
        tdir = run_dir / "trials" / tid.strip()
        if not (tdir / "code").is_dir():
            logger.error("trial 不存在或无 code/: %s", tdir)
            return 1
        specs.append((tdir, blk.strip()))
    if not specs:
        logger.error("--from-feature-block 为空")
        return 2
    _max_gb, _chunk, _ = _view_stream_budget(args, run_dir)
    try:
        manifest = write_view_from_feature_blocks(
            specs, spec, run_dir, view_id, chunk_rows=_chunk,
            max_mem_gb=_max_gb, note=args.note or "")
    except Exception as exc:  # noqa: BLE001
        logger.error("--from-feature-block 材料化失败: %s", exc)
        return 1
    return _print_view_manifest(run_dir, view_id, manifest)


def cmd_write_splits_sample(args) -> int:
    """特征探索抽样面落盘(协议 v10,feature-exploration.md)。

    从 data-loader 重载全量样本,各 split 按比例等距抽样到 ≤ max_rows
    (默认 20 万,固定 seed 可复现);final_test 不抽样不落盘(盲测面
    永远不动)。产物 sample-profile/splits-sample/。
    """
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    from runtime.data import write_sample_splits

    # [v2.13.2] --max-rows 缺省时读 feature_exploration.max_sample_rows
    # (该配置键此前零引用)。命令行显式传入仍优先 —— 配置只是缺省值,
    # 不改变"显式参数覆盖配置"的既有语义。
    if getattr(args, "max_rows", None) is not None:
        max_rows = int(args.max_rows)
    else:
        from .context import _load_config
        max_rows = int(_load_config(run_dir).get(
            "feature_exploration.max_sample_rows", 200000))
    try:
        df = load_dataframe(spec, run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("重载全量样本失败: %s(data-loader 必须仍可执行)", exc)
        return 1
    try:
        manifest = write_sample_splits(
            df, spec, run_dir, max_rows=max_rows, note=args.note or "")
    except Exception as exc:  # noqa: BLE001
        logger.error("抽样面落盘失败: %s", exc)
        return 1
    out_dir = run_dir / "sample-profile" / "splits-sample"
    print(f"特征探索抽样面已落盘: {out_dir}")
    print(f"  原始行数={manifest.get('n_rows_original')} "
          f"max_rows={manifest.get('max_rows')} seed={manifest.get('seed')}")
    for name, entry in manifest["splits"].items():
        rate = entry.get("positive_rate")
        extra = f" positive_rate={rate:.4f}" if rate is not None else ""
        print(f"  {name}: rows={entry['rows']} "
              f"(原 {entry['rows_original']}){extra}")
    # v2.17.0(WP5b #19):抽样面噪声预检 → manifest.noise_precheck
    _noise_precheck(spec, run_dir, out_dir)
    # v2.13.5:同视图锚点。sample 面 trial 的父代若在 v1 面,vs_parent 与
    # vs_baseline 都结构性不可比 —— 配对检验全部缺席,sample 面筛选失去
    # 裁决能力(req-t7-rand400w gen-01:4 个 sample trial 跑完才发现,删掉
    # 重建到 v1,多花 ~25 min 并引入一次 SPLITS 口径错位)。先在 sample 面
    # 复制一份 baseline 作锚点,后续 sample trial 以它为父代即同视图可比。
    _bl = None
    try:
        from runtime.logging_utils import load_json as _lj

        _bl = (_lj(run_dir / "run-state.json") or {}).get("baseline_id")
    except Exception:  # noqa: BLE001
        _bl = None
    _bl_s = _bl or "<baseline_id>"
    print("  下一步(v2.13.5,先建同视图锚点):")
    print(f"    1) new-trial --from-parent {_bl_s} --action data_view "
          "--data-view sample   # baseline 代码原样跑在抽样面(分钟级),"
          "SPLITS 自动改写到 splits-sample")
    print("    2) run --trial <上一步 trial> ;之后的 sample 面 trial 一律 "
          "--from-parent <该锚点> --data-view sample,vs_parent 同视图可比")
    print("    (sample 面没有 vs_baseline 是预期行为;晋级方向必须回 v1 全量复核)")
    return 0


def _noise_precheck(spec, run_dir: Path, out_dir: Path) -> dict | None:
    """v2.17.0(#19):写 splits-sample/manifest.json 的 noise_precheck 并打印。

    失败不阻塞抽样面落盘(预检是建议,不是门)。"""
    try:
        from runtime.data import sample_noise_precheck
        from runtime.logging_utils import dump_json, load_json

        from .context import _load_config

        cfg = _load_config(run_dir)
        res = sample_noise_precheck(
            run_dir, spec, cfg,
            n_boot=int(cfg.get("feature_exploration.sample_precheck_n_boot",
                               200) or 200))
        man_p = out_dir / "manifest.json"
        man = load_json(man_p)
        man["noise_precheck"] = res
        dump_json(man_p, man)
    except Exception as exc:  # noqa: BLE001
        logger.warning("抽样面噪声预检失败(不阻塞): %s", exc)
        return None
    if res.get("skipped"):
        print(f"  噪声预检: 跳过 —— {res.get('reason')}")
    else:
        print(f"  噪声预检: recommend_v1={str(res['recommend_v1']).lower()} "
              f"se_sample={res['se_sample']:.4g} se_full={res['se_full']:.4g} "
              f"se_ratio={res['se_ratio']} mde={res['mde']} "
              f"(basis={res.get('basis')})")
        print(f"    理由: {res['reason']}")
    return res
