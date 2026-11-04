"""配方语义指纹(v2.18.4):训练行为相同的配方应被识别为同一份。

**事故**(req-t7 400w,`run4-v2.18.0-3.ds-flash` gen-00):经验库榜单 rank-1 与
rank-2 是两份 `recipe_id` 不同、训练行为**完全相同**的配方 —— 逐字节 diff 显示
435 行差异(语句顺序、注释排版),但最终特征列(1596 列,含 18 个 `seph_`)、
超参、`model.json` / `model.joblib` / `validation_input.parquet` 的 sha256
**全部相同**,validation AUC 也逐位相同(`0.8122654617328389`)。两个 trial
各烧约 69 min × 2 张卡,产出 0 比特新信息。

逐字节 diff 抓不到这种重复(排版/顺序不同),所以这里按 **AST 规范化**:
顶层语句统一 dump 后排序拼接再 sha256 —— 语句顺序、注释、空行与字符串以外的
空白都不影响;`_G2_SLOT` / `_G2_N_BLOCK_COLS` 这类只承载标签/计数的顶层常量,
以及 `SEED` / `RUN_DIR` / `SKILL_ROOT` 这类只表示"用哪个 seed / 配方在哪儿"的
赋值也剔除。

纯 plumbing:不参与任何评分或落盘数字,用于
- 经验库榜单去重(`library.recipes.recipe_family` 的 .py 部分)—— 同一配方
  在**榜单源头**只占一个名次,这是主修复;
- `cmd_startup` 建 warm trial 前的兜底去重(旧版经验库服务仍可能返回重复);
- run 内 leaderboard 的 `recipe_fingerprint` 列与方向台账的复测判定。
"""

from __future__ import annotations

import ast
import hashlib

__all__ = ["VOLATILE_TOP_CONSTS", "semantic_normal_form",
           "recipe_semantic_fingerprint"]

#: 规范化时忽略的顶层赋值:标签/计数常量,以及 seed 与路径(换 seed 复跑、
#: 换 run / 换机器渲染都不改变配方本身)
VOLATILE_TOP_CONSTS = frozenset({"_G2_SLOT", "_G2_N_BLOCK_COLS",
                                 "SEED", "RUN_DIR", "SKILL_ROOT"})

_UNPARSABLE = "unparsable"


def _assigned_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 \
            and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def semantic_normal_form(code_text: str) -> str | None:
    """配方源码的顶层规范形(排序后的 AST dump);无法解析返回 None。"""
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return None
    nodes = [ast.dump(node) for node in tree.body
             if _assigned_name(node) not in VOLATILE_TOP_CONSTS]
    return "\n".join(sorted(nodes))


def recipe_semantic_fingerprint(code_text: str) -> str:
    """配方**语义**指纹;源码无法解析时返回 ``"unparsable"``。

    只做顶层语句级规范化,不进函数体 —— 函数体内的差异通常会改变训练
    行为,噪声(局部变量名、注释)远小于顶层;进函数体会让指纹对"只改了
    一个无关注释"的配方产生假区分,反而退回逐字节 diff 的缺点。

    已知取舍:顶层语句**排序**后比较,依赖顶层执行顺序的写法
    (``FEATURES = [...]`` 后接两条次序不同的 ``FEATURES += ...``)会被
    判为相同 —— 代价是少跑一个 warm,不影响评分;模块/函数 docstring
    属于 AST,docstring 不同的配方判为不同(偏保守)。
    """
    norm = semantic_normal_form(code_text)
    if norm is None:
        return _UNPARSABLE
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()
