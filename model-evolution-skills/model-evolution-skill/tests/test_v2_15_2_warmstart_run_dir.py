"""v2.15.2 修复批次单元验证:warm-start 配方 RUN_DIR 改写。

背景(v2.15.0 实测事故,req-t7-rand400w run4-memory-refactor4):
`memory/top-trials/<task_id>/rank<N>/train.py` 里保存的配方行是**相对式**
`RUN_DIR = pathlib.Path(__file__).resolve().parents[3]`。gen-00 的 warm-start
把它复制进新 run 时,旧实现用

    re.subn(r'(?m)^RUN_DIR\\s*=\\s*pathlib\\.Path\\([^\\)]*\\)',
            f"RUN_DIR = pathlib.Path({run_dir!r})", text)

改写:`[^\\)]*` 在 `pathlib.Path(__file__` 处遇**第一个右括号即停**,匹配区间
不含其后的 `.resolve().parents[3]`;而替换串只插入新路径、不删除匹配区间内的
原文 ⇒ 结果行是

    RUN_DIR = pathlib.Path('/…/run-x').resolve().parents[3]

新 run 路径**又被上跳 3 层** → `SPLITS` 指向 `<share>/sample-profile/splits/`
→ 两个 warm trial 同因 `FileNotFoundError` 失败(runtime 0.48s,infra_error);
更糟的是只要 subn 命中就打 "RUN_DIR 已改写" info 日志,错误完全不可见。

本测试覆盖修复后的 `scripts/_mvo/warmstart.rewrite_recipe_run_dir`:
1. 相对式(`__file__` + `parents[3]`)→ 整行替换,无残留,解析 == 本 run;
2. 绝对式(旧 run 导出的配方)→ 同样正确改写;
3. 等号两侧空白可变;
4. 改不到(非 `pathlib.Path(...)` 形态)→ 返回 not_found,**绝不返回错文本**;
5. 校验器 `run_dir_resolves_in_source` 能识破"带残留"的错误改写(回归哨兵)。

运行:python tests/test_v2_15_2_warmstart_run_dir.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._mvo.warmstart import (  # noqa: E402
    rewrite_recipe_run_dir,
    run_dir_resolves_in_source,
)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


#: 事故当时的**错误**输出形态(残留 `.resolve().parents[3]`),用作回归哨兵:
#: 修复后的校验器必须判它 false。
_BUGGY_OUTPUT = (
    "RUN_DIR = pathlib.Path('/tmp/wsv2-run/run-x').resolve().parents[3]\n"
    "SPLITS = RUN_DIR / 'sample-profile' / 'splits'\n"
)

_REL_RECIPE = (
    "import pathlib\n"
    "\n"
    "RUN_DIR = pathlib.Path(__file__).resolve().parents[3]\n"
    "SPLITS = RUN_DIR / 'sample-profile' / 'splits'\n"
    "TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent\n"
)

_ABS_RECIPE = (
    "RUN_DIR = pathlib.Path("
    "'/data/oceanus_share/j-wangyaoxuan-jk/modelevo-v1-run/run3')\n"
    "SPLITS = RUN_DIR / 'sample-profile' / 'splits'\n"
)


def test_relative_recipe():
    print("1) 相对式配方(top-trials 真实形态)改写:")
    run_dir = Path(tempfile.mkdtemp()) / "run-x"
    run_dir.mkdir(parents=True, exist_ok=True)
    out, status = rewrite_recipe_run_dir(_REL_RECIPE, run_dir)
    check("status == verified", status == "verified", f"(got {status})")
    head = out.split("SPLITS")[0]
    check("无 __file__ 残留", "__file__" not in head)
    check("无 .parents[ 残留", ".parents[" not in head)
    check("解析指向本 run",
          run_dir_resolves_in_source(out, run_dir))
    check("保留其它行(SPLITS 行未被吃掉)",
          "sample-profile" in out and "TRIAL_DIR" in out)
    check("行数不变", len(out.splitlines()) == len(_REL_RECIPE.splitlines()))


def test_absolute_recipe():
    print("2) 绝对式配方(旧 run 导出)改写:")
    run_dir = Path(tempfile.mkdtemp()) / "run-y"
    run_dir.mkdir(parents=True, exist_ok=True)
    out, status = rewrite_recipe_run_dir(_ABS_RECIPE, run_dir)
    check("status == verified", status == "verified", f"(got {status})")
    check("解析指向本 run", run_dir_resolves_in_source(out, run_dir))
    check("不含来源 run 路径", "run3" not in out)


def test_whitespace_tolerant():
    print("3) 等号两侧空白可变:")
    run_dir = Path(tempfile.mkdtemp()) / "run-z"
    run_dir.mkdir(parents=True, exist_ok=True)
    src = "RUN_DIR  =  pathlib.Path(__file__).resolve().parents[3]\n"
    out, status = rewrite_recipe_run_dir(src, run_dir)
    check("status == verified", status == "verified", f"(got {status})")
    check("解析指向本 run", run_dir_resolves_in_source(out, run_dir))


def test_unrewritable():
    print("4) 非 pathlib.Path 形态 → not_found(绝不返回错文本):")
    run_dir = Path(tempfile.mkdtemp()) / "run-w"
    run_dir.mkdir(parents=True, exist_ok=True)
    src = "RUN_DIR = Path(os.environ['MODELEVO_RUN_DIR'])\n"
    out, status = rewrite_recipe_run_dir(src, run_dir)
    check("status == not_found", status == "not_found", f"(got {status})")
    check("文本原样返回", out == src)


def test_buggy_output_detected():
    print("5) 回归哨兵:v2.15.0 的错误改写必须被判 false:")
    run_dir = Path("/tmp/wsv2-run/run-x")
    check("带 .resolve().parents[3] 残留 → false",
          not run_dir_resolves_in_source(_BUGGY_OUTPUT, run_dir))
    check("未改写(仍是 __file__ 相对式) → false",
          not run_dir_resolves_in_source(_REL_RECIPE, run_dir))
    check("指向别的 run 的绝对式 → false",
          not run_dir_resolves_in_source(
              "RUN_DIR = pathlib.Path('/somewhere/else')\n",
              Path("/tmp/wsv2-run/run-x")))


if __name__ == "__main__":
    test_relative_recipe()
    test_absolute_recipe()
    test_whitespace_tolerant()
    test_unrewritable()
    test_buggy_output_detected()
    print(f"\n{'='*50}")
    print(f"PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
