"""v2.19.0(#15)补丁式写码:new-trial --from-parent --patch 与 trial-skeleton。

父代 = 模板渲染的真实结构 train.py(new-trial explore xgboost,约 2500 行)。
覆盖:新增块并注册、替换同名函数、拒绝受保护段、与 --set 组合、合并结果过 lint、
trial.yaml 留痕、skeleton 输出、confirm 拒绝、PATCH_REPLACES 校验、链式 patch。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

for _p in (str(SKILL), str(SKILL / "scripts"), str(SKILL / "assets")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("xgboost")

import lint_trial as L  # noqa: E402
import test_v2_17_0_trial_templates as T  # noqa: E402  夹具复用
from _mvo import trial_patch as TP  # noqa: E402
from runtime.logging_utils import load_yaml  # noqa: E402

_PATCH_ADD = '''"""patch:新增 seqz 特征块 + 替换 prepare_features。"""
from runtime.feature_cache import feature_block

PATCH_REGISTER_BLOCKS = ["_seqz_block"]
PATCH_REPLACES = ["prepare_features"]

# seqz 事件词表
_SEQZ_HEADS = ("E0", "E1", "E2")
_SEQZ_COLS = [f"seqz_{h}" for h in _SEQZ_HEADS] + ["seqz_len"]


def _seqz_row(s):
    toks = [t for t in (s or "").split(",") if t]
    return [float(sum(1 for t in toks if t == h)) for h in _SEQZ_HEADS] + \\
        [float(len(toks))]


def _seqz_kernel(b):
    return [_seqz_row(s) for s in b["seq"]]


def _seqz_block(split):
    return feature_block("seqz_v1", _seqz_kernel, split=split,
                         columns=_SEQZ_COLS, sources=["seq"],
                         splits_dir=SPLITS, n_jobs=n_jobs())


def prepare_features(X_tr: pd.DataFrame, X_va: pd.DataFrame) -> tuple:
    # v2.19 patch 单测:全部列先转 float32(无类别列的小夹具)
    X_tr = X_tr.astype("float32")
    X_va = X_va.astype("float32")
    return X_tr, X_va
'''


@pytest.fixture(autouse=True)
def _isolate_feature_memo():
    """本文件在进程内算过的块不得以 memo 命中泄漏给其它测试文件(否则它们的
    "缓存已落盘"断言因同内容同键而 memo 命中、不写盘)。"""
    import runtime.feature_cache as fc

    yield
    fc._MEMO.clear()


@pytest.fixture
def logs():
    """收集 _mvo logger 的消息(该 logger 不走 caplog)。"""
    import logging

    from _mvo.context import logger as _lg

    seen: list = []

    class _H(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    h = _H()
    _lg.addHandler(h)
    yield seen
    _lg.removeHandler(h)


def _parent(tmp_path, model_type="regression"):
    run = T._mk_run(tmp_path, model_type)
    rc, p = T._new_trial(run)
    assert rc == 0 and p is not None
    return run, p


def _write(tmp_path, text, name="p.py") -> str:
    f = tmp_path / name
    f.write_text(text, encoding="utf-8")
    return str(f)


def _child(run, parent, **over):
    over.setdefault("generation", None)
    return T._new_trial(run, from_parent=parent.name, **over)


def test_patch_adds_registered_block_and_replaces_function(tmp_path):
    run, p = _parent(tmp_path)
    parent_code = (p / "code" / "train.py").read_text(encoding="utf-8")
    rc, c = _child(run, p, patch=_write(tmp_path, _PATCH_ADD),
                   set=["max_depth=4"])                     # 与 --set 组合
    assert rc == 0 and c is not None
    code = (c / "code" / "train.py").read_text(encoding="utf-8")
    # 替换:原位,父代旧实现消失,新实现在
    assert "v2.19 patch 单测:全部列先转 float32" in code
    assert "直接 get_dummies 会按唯一值膨胀" not in code
    assert code.count("def prepare_features(") == 1
    assert code.index("def prepare_features(") < code.index("def fit_and_score(")
    # 新增:进 patch 段(main 守卫之前),块已注册
    assert TP.PATCH_BEGIN in code and TP.PATCH_END in code
    seg = code[code.index(TP.PATCH_BEGIN):code.index(TP.PATCH_END)]
    for s in ("# seqz 事件词表\n_SEQZ_HEADS", "def _seqz_kernel(b):",
              "_ADDED_FEATURE_BLOCKS_P1", "def trial_feature_blocks("):
        assert s in seg, s
    assert code.index(TP.PATCH_END) < code.rindex('if __name__ == "__main__"')
    # 其余父代代码逐字不变(只多了 patch 段与替换处)
    tail = parent_code[parent_code.index("def fit_and_score("):
                       parent_code.index("def main(")]
    assert tail in code
    # trial.yaml 留痕
    tr = load_yaml(c / "trial.yaml")
    pr = tr["patch"]
    assert pr["replaced"] == ["prepare_features"]
    assert set(pr["added"]) >= {"_SEQZ_HEADS", "_SEQZ_COLS", "_seqz_row",
                                "_seqz_kernel", "_seqz_block"}
    assert pr["imports_added"] == ["feature_block"]
    assert len(pr["source_sha"]) == 64 and pr["lint"]["new_errors"] == 0
    rb = pr["registered_blocks"][0]
    assert rb["wrapper"] == "_seqz_block" and rb["n_columns"] == 4
    assert rb["feature_block_name"] == "seqz_v1" and rb["code_fingerprint"]
    cfg = load_yaml(c / "config" / "trial.yaml")
    assert cfg["hyperparameters"]["max_depth"] == 4
    # 合并结果通过 lint;块真实可算
    assert L.lint_trial(run, c, False) == []
    m = T._load(c / "code" / "train.py")
    W = m.trial_feature_blocks("train")
    assert list(W.columns) == ["seqz_E0", "seqz_E1", "seqz_E2", "seqz_len"]
    assert len(W) == 60 and np.isfinite(W.to_numpy()).all()


def test_patch_chain_on_patched_parent_and_skeleton(tmp_path, capsys):
    run, p = _parent(tmp_path)
    rc, c1 = _child(run, p, patch=_write(tmp_path, _PATCH_ADD))
    assert rc == 0
    # 第二代:在已打补丁的父代上再加一个块 + 改常量(原位替换)
    p2 = '''
_SEQZ_HEADS = ("E0", "E1", "E2", "E3")

_SEQW_COLS = ["seqw_n"]


def _seqw_kernel(b):
    return [[float(len(s or ""))] for s in b["seq"]]


def _seqw_block(split):
    return feature_block("seqw_v1", _seqw_kernel, split=split,
                         columns=_SEQW_COLS, sources=["seq"],
                         splits_dir=SPLITS, n_jobs=n_jobs())


PATCH_REGISTER_BLOCKS = ["_seqw_block"]
'''
    rc2, c2 = _child(run, c1, patch=_write(tmp_path, p2, "p2.py"))
    assert rc2 == 0
    code = (c2 / "code" / "train.py").read_text(encoding="utf-8")
    assert code.count(TP.PATCH_BEGIN) == 2 and "_ADDED_FEATURE_BLOCKS_P2" in code
    assert '_SEQZ_HEADS = ("E0", "E1", "E2", "E3")' in code
    assert '_SEQZ_HEADS = ("E0", "E1", "E2")\n' not in code
    pr = load_yaml(c2 / "trial.yaml")["patch"]
    assert pr["replaced"] == ["_SEQZ_HEADS"] and "_seqw_block" in pr["added"]
    m = T._load(c2 / "code" / "train.py")
    W = m.trial_feature_blocks("validation")
    assert list(W.columns) == ["seqz_E0", "seqz_E1", "seqz_E2", "seqz_E3",
                               "seqz_len", "seqw_n"]
    assert L.lint_trial(run, c2, False) == []
    # skeleton(文本 + JSON)
    cli = T._cli()
    capsys.readouterr()
    assert cli.main(["trial-skeleton", "--run", str(run), "--trial",
                     c2.name]) == 0
    out = capsys.readouterr().out
    assert "## 特征块" in out and "_seqz_block → 块 seqz_v1" in out
    assert "5 列 前缀 {'seqz_': 5}" in out and "1 列 前缀 {'seqw_': 1}" in out
    assert "[P:评估/产物/打分口径]" in out and "[E]" in out
    assert "(被后定义覆盖)" in out          # 多次定义的 trial_feature_blocks
    sk = TP.build_skeleton(code)            # 无探针:静态 columns 表达式
    b = {x["wrapper"]: x for x in sk["feature_blocks"]}
    assert b["_seqw_block"]["columns_expr"] == "_SEQW_COLS"
    assert b["_seqw_block"]["registered"] and sk["patch_segments"] == 2
    assert len(out) < len(code) / 3          # 大纲远小于全文


@pytest.mark.parametrize("bad, why", [
    ("def leak_scan(*a, **k):\n    return None\n", "leak_scan 是受保护符号"),
    ("def _fit_es_on_validation(*a, **k):\n    return None\n", "早停"),
    ("def _hpo_fit_inner(*a, **k):\n    return None\n", "HPO 入口"),
    ("def trial_feature_blocks(split):\n    return None\n",
     "PATCH_REGISTER_BLOCKS 注册"),
    ("def load_labels(split):\n    return None\n", "数据加载"),
    ("SEED = 1\n", "SEED 是受保护符号"),
    ("PATCH_REGISTER = ['x']\n", "未知指令"),
    ("_MAX_TREES += 1\n", "改写了父代已有名字"),
    ("import json as main\n", "受保护符号"),
    ("def step(m):\n    print(m)\n", "step 是受保护符号"),
])
def test_patch_rejects_protected_and_malformed(tmp_path, logs, bad, why):
    run, p = _parent(tmp_path)
    before = sorted(d.name for d in (run / "trials").iterdir())
    logs.clear()
    rc, c = _child(run, p, patch=_write(tmp_path, bad))
    assert rc == 1 and c is None                         # 预演拒绝,不落盘
    assert sorted(d.name for d in (run / "trials").iterdir()) == before
    err = "\n".join(logs)
    assert why in err, err


def test_patch_contract_checks(tmp_path, logs):
    run, p = _parent(tmp_path)
    code = (p / "code" / "train.py").read_text(encoding="utf-8")
    # PATCH_REPLACES 拼错 → 被当新增 → 拒绝
    with pytest.raises(SystemExit, match="PATCH_REPLACES 与合并结果不符"):
        TP.merge_patch(code, "PATCH_REPLACES = ['prepare_feature']\n"
                             "def prepare_feature(a, b):\n    return a, b\n")
    # 注册非 feature_block 包装函数 → 拒绝
    with pytest.raises(SystemExit, match="不是调用 feature_block"):
        TP.merge_patch(code, "PATCH_REGISTER_BLOCKS = ['_x_block']\n"
                             "def _x_block(split):\n    return None\n")
    # 原位替换的常量定义期依赖本 patch 新增名字 → 拒绝
    with pytest.raises(SystemExit, match="定义期引用了本 patch 新增"):
        TP.merge_patch(code, "_NEWC = 3\n"
                             "def prepare_features(a, b, k=_NEWC):\n"
                             "    return a, b\n")
    # 与父代 AST 等价 → 零差异拒绝
    import ast
    src = ast.get_source_segment(code, next(
        s for s in ast.parse(code).body
        if getattr(s, "name", "") == "prepare_features"))
    with pytest.raises(SystemExit, match="无差异"):
        TP.merge_patch(code, src + "\n")
    # confirm / automl 不可用;无 --from-parent 不可用
    pf = _write(tmp_path, _PATCH_ADD)
    logs.clear()
    assert _child(run, p, patch=pf, action="confirm")[0] == 1
    assert "--patch 不能用于 --action confirm" in "\n".join(logs)
    assert T._new_trial(run, patch=pf)[0] == 1
    assert "--patch 需要 --from-parent" in "\n".join(logs)


def test_patch_new_lint_error_rolls_back(tmp_path, logs):
    """合并结果相对父代新增 lint 错误(这里:fit_and_score 直接 load_data)→ 回滚。"""
    run, p = _parent(tmp_path)
    bad = ("def prepare_features(X_tr, X_va):\n"
           "    from runtime.data import load_data\n"
           "    load_data('x')\n"
           "    return X_tr, X_va\n")
    before = sorted(d.name for d in (run / "trials").iterdir())
    logs.clear()
    rc, c = _child(run, p, patch=_write(tmp_path, bad))
    assert rc == 1
    assert sorted(d.name for d in (run / "trials").iterdir()) == before
    assert "新增 lint 错误" in "\n".join(logs)


def test_patch_with_add_blocks(tmp_path):
    run, p = _parent(tmp_path)
    src = T._mk_source_trial(run, "trial-090-g00-explore-xgboost-1")
    rc, c = _child(run, p, patch=_write(tmp_path, _PATCH_ADD),
                   add_blocks=f"{src.name}:seqx_v1")
    assert rc == 0
    tr = load_yaml(c / "trial.yaml")
    assert tr["added_blocks"][0]["match"] is True and tr["patch"]["replaced"]
    m = T._load(c / "code" / "train.py")
    cols = list(m.trial_feature_blocks("train").columns)
    assert cols[:4] == ["seqz_E0", "seqz_E1", "seqz_E2", "seqz_len"]
    assert "seqx_len" in cols
    assert L.lint_trial(run, c, False) == []
