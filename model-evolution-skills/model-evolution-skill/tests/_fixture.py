"""真实 run 的「瘦身夹具」——让依赖示例 run 的测试不再拷贝 GB 级数据。

**背景(v2.11.9)**:`tests/test_v19_gate_fix.py` 原先对 8.3GB 的示例 run
`shutil.copytree` **三次**(三个场景),`tests/test_v19dot3_fixes.py` 更是
直接往真实 run 目录 `write_view_splits(..., RUN, "v193-t3")` 写盘——测试
既慢又污染用户数据,还会因为并发/目录状态变化而 flaky(已实测复现)。

**做法**:只拷元数据(小文件原样),parquet 只保留前 `rows` 行(保持
schema 与列名不变);模型二进制 / 备选视图 / 特征重要性等**与断言无关的
大件**按 `SKIP_GLOBS` 跳过。交付链校验关心的是**行数一致性 + 文件存在性
+ 元数据内容**,截断后语义等价,体积从 GB 降到 MB。

**示例 run 的定位**:相对 skill 根推导(`<skill>/../example-*`),不再写死
绝对路径;缺失时由调用方**显式跳过**(`find_run` 返回 None),不得静默通过。
"""
from __future__ import annotations

import fnmatch
import pathlib
import shutil

from _common import SKILL

#: 瘦身后每个 parquet 保留的行数(够撑行数一致性校验即可)
DEFAULT_ROWS = 200

#: 与断言无关的大件(相对 run 根的 glob);命中即整棵跳过。
SKIP_GLOBS: tuple[str, ...] = (
    "trials/*/model", "trials/*/model/*",
    "trials/*/feature", "trials/*/feature/*",
    "trials/*/seed_scores", "trials/*/seed_scores/*",
    "trials/*/metrics.json.bak*",
    "final-model/model", "final-model/model/*",
    "sample-profile/splits-sample", "sample-profile/splits-sample/*",
    "sample-profile/splits-v-*", "sample-profile/splits-v-*/*",
    "sample-profile/splits-reparse", "sample-profile/splits-reparse/*",
    "data", "data/*",
    "adhoc", "adhoc/*",
    "logs", "logs/*",
)


def _skipped(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, g) for g in SKIP_GLOBS)


def find_run(*candidates: str) -> pathlib.Path | None:
    """在 skill 的同级目录下找示例 run;找不到返回 None(调用方须跳过)。

    `candidates` 是相对 `<skill>/../` 的路径,例如
    `"example-seq-v3-4-2.8.6-v19.2"`。
    """
    base = SKILL.parent
    for c in candidates:
        p = base / c
        if (p / "task-spec.yaml").exists():
            return p
    return None


def slim_run(src: pathlib.Path, dst: pathlib.Path, *,
             rows: int = DEFAULT_ROWS) -> pathlib.Path:
    """把 `src` run 目录瘦身复制到 `dst`,返回 `dst`。

    - `SKIP_GLOBS` 命中的子树整棵跳过(模型二进制/备选视图/日志);
    - 非 parquet 文件原样复制(元数据/JSON/YAML/CSV 都很小);
    - parquet 只保留前 `rows` 行(schema 与列名不变);
    - `*.lock` / `__pycache__` / `*.pyc` 跳过。
    """
    import pandas as pd

    src, dst = pathlib.Path(src), pathlib.Path(dst)
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        rels = str(rel)
        if any(part == "__pycache__" for part in rel.parts):
            continue
        if p.suffix in (".lock", ".pyc"):
            continue
        if _skipped(rels):
            continue
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".parquet":
            try:
                pd.read_parquet(p).head(rows).to_parquet(target, index=False)
            except Exception:  # noqa: BLE001  读不了就退化成空表占位
                pd.DataFrame({"score": []}).to_parquet(target, index=False)
        else:
            shutil.copy2(p, target)
    return dst


def reset_gate(run_dir: pathlib.Path) -> None:
    """把 run 目录还原成「未盲测」初始态(gate 清零 + 删 evaluation)。"""
    for rel in ("final-test-gate.json", "final-model/evaluation.json"):
        (run_dir / rel).unlink(missing_ok=True)


def declare_recipe_champion(run_dir: pathlib.Path, trial_id: str) -> None:
    """把 recipe.yaml 的 best_trial 改写成指定值(制造/消除选型分裂)。"""
    import re

    recipe = run_dir / "final-model" / "recipe.yaml"
    if not recipe.exists():
        recipe.parent.mkdir(parents=True, exist_ok=True)
        recipe.write_text(f"best_trial: {trial_id}\n", encoding="utf-8")
        return
    txt = recipe.read_text(encoding="utf-8")
    if re.search(r"^best_trial:", txt, re.M):
        txt = re.sub(r"^best_trial:.*$", f"best_trial: {trial_id}", txt,
                     flags=re.M)
    else:
        txt += f"\nbest_trial: {trial_id}\n"
    recipe.write_text(txt, encoding="utf-8")


__all__ = ["find_run", "slim_run", "reset_gate", "declare_recipe_champion",
           "DEFAULT_ROWS", "SKIP_GLOBS"]
