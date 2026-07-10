from __future__ import annotations

import ast
from pathlib import Path


SKILLS_ROOT = Path(__file__).resolve().parents[2]
S_RUNNER = SKILLS_ROOT / "uplift-model-s-learner-modeling" / "scripts" / "run.py"
T_RUNNER = SKILLS_ROOT / "uplift-model-t-learner-modeling" / "scripts" / "run.py"


def _run_train_split_order(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_train":
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.For)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == "split"
                    and isinstance(child.iter, ast.Tuple)
                ):
                    return tuple(
                        item.value
                        for item in child.iter.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    )
    raise AssertionError(f"run_train split loop not found in {path}")


def test_s_learner_split_evaluations_include_valid_like_t_learner() -> None:
    expected = ("train", "valid", "test", "oot")

    assert _run_train_split_order(S_RUNNER) == expected
    assert _run_train_split_order(S_RUNNER) == _run_train_split_order(T_RUNNER)
