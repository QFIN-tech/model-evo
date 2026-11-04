"""数据视图模板(v16/P1-1):参数化生成 views/{id}.py 骨架。

背景(反思文档 §3 前提 1):data_view 换聚合口径是"唯一可能翻动
+0.01 量级的通道"(v3-5 run v2 视图 +0.0009 accepted 实证有效),
但 14 个历史 run 只用过 3 次——摩擦根因是每个视图都要 LLM 从零写
构建脚本 + 行对齐校验。本模块把最常见的三类口径变换模板化:

- recency:近因指数加权——对数值列按时间衰减加权(半衰期参数),
  生成 {col}_w{half_life} 前缀新列;
- active:活跃子集统计——近 K 期活跃行为子集上的计数/均值,
  生成 act_{col} 前缀新列;
- kwin:近 K 期窗口重聚合——把全窗聚合列重做近 K 期口径,
  生成 kw{k}_{col} 前缀新列。

契约(write-splits-view 消费):暴露 build_view(df) -> DataFrame;
输出入行数/行序/ID 逐位置一致,只增特征列不替换。骨架包含人工
审查点注释;生成后 Agent/用户审查再落盘 splits。

v2.17.0(流式视图):write-splits-view 默认按 split × 分块调用 build_view,
要求它是**行内变换**(输出第 i 行只依赖输入第 i 行与冻结常量;df.index 是
v1 合并序的全局行位置)。active/kwin 天然行内;recency 的"逐 split 参考日"
改为由 CLI 注入 _SPLIT_REF_DATES 字面量、按 df.index 查表,同样行内。
"""

from __future__ import annotations


def render_view_template(template: str, *, view_id: str, half_life: float,
                         k_periods: int, active_col: str | None,
                         time_column: str | None, target_name: str) -> str:
    """渲染 views/{id}.py 骨架源码(纯文本生成,不执行)。"""
    tcol = time_column or ""
    common_head = f'''"""数据视图 {view_id}(模板: {template})。

由 write-splits-view --template {template} 生成(v16/P1-1)。
build_view 契约:输入 v1 splits 合并序完整样本(含目标/时间列),
输出行数/行序逐位置一致,只增前缀新列,不替换不删行。
[人工审查点] 生成后必须审查:(1) 时间列解析是否适配本任务
(时间列: {tcol or "无——recency/kwin 需要时间列,否则显式证伪该模板"});
(2) 数值列过滤是否合理(object 列已跳过);(3) 落盘前跑一次
行数断言。审查通过后由 write-splits-view --source 本文件落盘。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n_in = len(out)
    tcol = "{tcol}"
    # v19:CLI 侧注入的两个占位(write-splits-view 落盘前以实际字面量
    # 替换;模板独立运行时保持空值=回退全局参考日/不过滤,行为同旧版):
    # _SPLIT_ROWS      —— 各 split 行数(train/validation/shadow/final
    #                     拼接序),recency 逐段参考日的切段依据;
    # _PROTECTED_COLS  —— 禁止加权的目标/ID/treatment/banned/reparse
    #                     列(泄漏防线;加权标签列 = AUC≈1 伪特征)。
    _SPLIT_ROWS = []  # @SPLIT_ROWS@
    _PROTECTED_COLS = []  # @PROTECTED_COLS@
    # v2.17.0:各 split 时间列最大值(ISO 字面量,CLI 注入)。recency 按
    # df.index(v1 合并序的全局行位置)查表取参考日 → 行内变换,
    # write-splits-view 流式分块构建与全量模式逐位一致。
    _SPLIT_REF_DATES = []  # @SPLIT_REF_DATES@
    # v19.3:保护列为空时是否放行(默认 False = 拒绝生成加权列)。
    # 纯特征面任务(无 target/ID 可保护)可在脚本内改为 True 并写明理由。
    _ALLOW_UNPROTECTED = False
'''
    if template == "recency":
        body = f'''    # ---- recency:近因指数加权(half-life={half_life} 天)----
    # 对每个数值特征列 x 生成 x_w{int(half_life)} = x * exp(-ln2 * days_ago / {half_life})
    # [审查点] days_ago 的口径:v19 起按 split 逐段参考日(各 split
    # 用本段内 max 日期)——旧实现全局 max(=final_test 末日)使
    # train 段 days_ago 系统性偏大、跨 split 衰减尺度漂移。分段边界
    # 按 build_view 输入的 SPLIT_NAMES 拼接序 train/validation/
    # shadow_test/final_test 行数切分(与 write_view_splits 的切分
    # 一致;行数注入由 CLI 传入 _SPLIT_ROWS)。
    # v19 泄漏修复:加权列只对特征列生成——target/ID/treatment/
    # banned/reparse 一律排除(旧实现 _num_cols 只排时间列,标签列
    # 被卷入 w60_{{target}},train 单变量 AUC≈1)。
    if not tcol or tcol not in df.columns:
        raise ValueError("recency 模板需要时间列 {tcol}")
    _t = pd.to_datetime(df[tcol], errors="coerce")
    # 逐 split 参考日:按调用方注入的 _SPLIT_ROWS 切段,段内 max 为
    # 参考日(无注入时回退全局 max,与旧行为一致——但 CLI 侧总会注入)
    _ref = pd.Series(_t.max(), index=df.index)
    # v19.3 修复(读取错名字空间):_SPLIT_ROWS/_PROTECTED_COLS 是
    # build_view 的**局部变量**,旧实现用 globals().get(...) 读取 ——
    # 永远拿不到(返回 None → `or []` 静默退化为空),即 CLI 侧注入
    # 再成功也无效。改为直接读局部变量(注入与否都必须可观测)。
    _rows = _SPLIT_ROWS
    _refs = _SPLIT_REF_DATES
    _pos = np.asarray(df.index)
    if (_rows and _refs and len(_refs) == len(_rows) and len(_pos)
            and _pos.dtype.kind in "iu" and _pos.min() >= 0
            and _pos.max() < sum(_rows)):
        # v2.17.0 行内口径:全局行位置 → 所属 split → 该 split 参考日
        _segi = np.searchsorted(np.cumsum(_rows), _pos, side="right")
        _ref = pd.Series(
            pd.to_datetime(pd.Series(_refs).iloc[_segi].values),
            index=df.index)
    elif _rows and sum(_rows) == len(df):
        _start = 0
        for _n in _rows:
            if _n <= 0:
                continue
            _seg = _t.iloc[_start:_start + _n]
            _ref.iloc[_start:_start + _n] = _seg.max()
            _start += _n
    _days_ago = (_ref - _t).dt.days.clip(lower=0).astype(float)
    _decay = np.exp(-np.log(2) * _days_ago / {half_life})
    # v19.3:直读局部变量(旧实现 globals().get(...) 在此永远取不到值,
    # 保护列形同虚设 —— 见上方 _rows 处同一修复说明)。
    _exclude = set(_PROTECTED_COLS)
    # v19.3 硬防线:保护列为空的 recency 视图等于关闭泄漏防线(标签列
    # 会被加权成 w{{hl}}_{{target}},train 单变量 AUC≈1)。CLI 注入失败
    # 已在生成期 abort,此处再兜一层:运行时为空 → 显式 raise,不产出
    # 伪特征。允许显式豁免(纯特征面任务确实无 target/ID 可保护时)。
    if not _exclude and not _ALLOW_UNPROTECTED:
        raise ValueError(
            "recency 视图保护列为空:拒绝生成加权列(否则标签/ID 列会被"
            "加权成 w{int(half_life)}_<列名>,AUC≈1 的伪特征直入"
            "数据面)。请经 write-splits-view --template 重新生成骨架"
            "(CLI 会注入 _PROTECTED_COLS),或在脚本内显式设 "
            "_ALLOW_UNPROTECTED = True 并写明理由。")
    _num_cols = [c for c in df.columns
                 if c != tcol and c not in _exclude
                 and pd.api.types.is_numeric_dtype(df[c])]
    for c in _num_cols:
        out[f"w{int(half_life)}_{{c}}"] = df[c].fillna(0) * _decay
    assert len(out) == n_in, "行数漂移"
    return out
'''
    elif template == "active":
        body = f'''    # ---- active:活跃子集统计(近 K 期活跃人群口径)----
    # 以 active_col(活跃指示,如近 30 日登录天数)定义活跃子集,
    # 对数值列生成 act_{{c}} = 全体值 × 活跃指示(0/1 化)——
    # 等价于"只在活跃人群上取值的列",让模型可分开学两类口径。
    # [审查点] active_col 缺省自动探测;探测错则显式传列名重生成
    _ac = "{active_col or ''}"
    if not _ac:
        _cands = [c for c in df.columns
                  if ("login" in c.lower() or "active" in c.lower()
                      or "days" in c.lower())
                  and pd.api.types.is_numeric_dtype(df[c])]
        if not _cands:
            raise ValueError("active 模板未探测到活跃指示列;"
                             "显式传 --active-col 后重生成")
        _ac = _cands[0]
    _act = (df[_ac].fillna(0) > 0).astype(float)
    # v19.3 硬防线:与 recency 同口径 —— 保护列为空时拒绝生成加权列
    # (active 同样是"数值列 × 系数",不排除目标列会造出
    # act_<target> = target × 活跃指示,AUC≈1 的伪特征直入数据面)。
    _exclude = set(_PROTECTED_COLS)
    if not _exclude and not _ALLOW_UNPROTECTED:
        raise ValueError(
            "active 视图保护列为空:拒绝生成加权列(否则目标/ID 列会被"
            "加权成 act_<列名> 的伪特征直入数据面)。请经 "
            "write-splits-view --template 重新生成骨架(CLI 会注入 "
            "_PROTECTED_COLS),或在脚本内显式设 "
            "_ALLOW_UNPROTECTED = True 并写明理由。")
    _num_cols = [c for c in df.columns
                 if c not in (tcol, _ac) and c not in _exclude
                 and pd.api.types.is_numeric_dtype(df[c])]
    for c in _num_cols:
        out[f"act_{{c}}"] = df[c].fillna(0) * _act
    out["act_indicator"] = _act
    assert len(out) == n_in, "行数漂移"
    return out
'''
    elif template == "kwin":
        body = f'''    # ---- kwin:近 K 期窗口重聚合(K={k_periods})----
    # 对带窗口后缀(_1m/_3m/_6m/_12m)的列,重做"近 K 期"口径的
    # 派生:kw{k_periods}_{{c}} = 近窗口值 / 全窗值(占比)。
    # 让"最近 K 期占全窗的比例"成为独立可学的口径(实测 ratio 族
    # 是单列最大增量源,本模板把该口径系统化)。
    import re as _re

    _win_suffix = _re.compile(r"_(1m|3m|6m|12m)$")
    _pairs = {{}}
    for c in df.columns:
        m = _win_suffix.search(c)
        if m and pd.api.types.is_numeric_dtype(df[c]):
            _pairs.setdefault(m.group(1), []).append(c)
    _rank = {{"1m": 1, "3m": 2, "6m": 3, "12m": 4}}
    _near = [w for w in sorted(_pairs, key=lambda w: _rank.get(w, 9))
             if _rank.get(w, 9) <= {k_periods}]
    _full = "12m" if "12m" in _pairs else (
        max(_pairs, key=lambda w: _rank.get(w, 9)) if _pairs else None)
    if _near and _full:
        _stem = lambda c: _win_suffix.sub("", c)  # noqa: E731
        _full_map = {{_stem(c): c for c in _pairs[_full]}}
        for w in _near:
            for c in _pairs[w]:
                s = _stem(c)
                if s in _full_map and _full_map[s] != c:
                    fv = df[_full_map[s]].replace(0, np.nan)
                    out[f"kw{k_periods}_{{s}}_{{w}}_ratio"] = (
                        df[c] / fv).fillna(0)
    assert len(out) == n_in, "行数漂移"
    return out
'''
    else:
        raise ValueError(f"未知视图模板: {template}")
    return common_head + body
