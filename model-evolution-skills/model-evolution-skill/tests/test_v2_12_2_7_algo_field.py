"""v2.12.2.7 修复:快通道复制的 config 必须写**本 trial 的真实算法**。

实测事故(drawdown-t7-sv24):
`--action hpo --from-parent <lightgbm_trial> --algorithm xgboost` 建出的
trial,`trial.yaml.algorithm=xgboost`、`code/train.py:_HPO_ALGO='xgboost'`、
落盘模型是 `XGBClassifier`,但 `config/trial.yaml` 里写着
`algorithm: lightgbm`(整份 cfg 从父代拷贝而来,`--algorithm` 只进了命名与
trial.yaml)。该 run 的 trial-030~034 五条全中;普查 `.2`/`.3` 两个 run
零条不一致 —— 因为那两个 run 没跑过"Hpo + --from-parent"组合。

影响面:`config.algorithm` 无运行期消费点(实际算法由 trial.yaml 与 code
决定),属**审计字段**缺陷 —— 不影响 AUC,但按 config 单独复现会拿到错算法,
且随血缘一路传播。

本文件用**两个假 trial 对**验证修复逻辑本身(不依赖真实 run,故总能跑):
构造"父=lightgbm / 子=xgboost"的 config 复制场景,断言产出 config 的
algorithm 等于子代真实算法,且 notes 与其余字段仍按原契约继承。
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL, tmp_run_dir  # noqa: E402

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("[OK]   " if cond else "[FAIL] ") + name
          + ("" if cond else f"  <- {detail}"))
    if not cond:
        FAILS.append(name)


class _FakeTrial:
    """最小 Trial 替身:只需要 trial_id / trial_dir / data / flush。"""

    def __init__(self, trial_id: str, trial_dir: pathlib.Path,
                 algorithm: str):
        self.trial_id = trial_id
        self.trial_dir = str(trial_dir)
        self.data = {"algorithm": algorithm,
                     "execution": {}, "search_action": "explore"}
        self.flushed = 0

    def flush(self) -> None:
        self.flushed += 1


def _write_parent_config(run_dir: pathlib.Path, parent_id: str,
                         algorithm: str) -> None:
    from runtime.logging_utils import dump_yaml
    p = run_dir / "trials" / parent_id / "config" / "trial.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(p, {
        "trial_id": parent_id,
        "algorithm": algorithm,
        "features": {"exclude": ["a"], "banned": [], "reparse": []},
        "target": "label",
        "model_type": "binary_classification",
        "seed": 20260814,
        "hyperparameters": {},
        "notes": "父代 notes",
    })
    code = run_dir / "trials" / parent_id / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "train.py").write_text("SEED = 1\n", encoding="utf-8")


def test_fast_path_config_algorithm() -> None:
    print("== 快通道 config.algorithm 写子代真实算法 ==")
    from runtime.logging_utils import load_yaml
    from _mvo.render import _copy_parent_trial_code  # noqa: E402

    tmp = tmp_run_dir("algo-fix-")
    run_dir = pathlib.Path(tmp)
    parent_id = "trial-001-g00-lightgbm-1010"
    child_id = "trial-002-g01-xgboost-2030"
    _write_parent_config(run_dir, parent_id, "lightgbm")
    child_dir = run_dir / "trials" / child_id
    (child_dir / "config").mkdir(parents=True, exist_ok=True)
    (child_dir / "code").mkdir(parents=True, exist_ok=True)

    child = _FakeTrial(child_id, child_dir, "xgboost")
    _copy_parent_trial_code(run_dir, child, parent_id, {"max_depth": 7},
                            action="explore")

    cfg = load_yaml(child_dir / "config" / "trial.yaml") or {}
    check("config.algorithm == 子代真实算法(xgboost)",
          str(cfg.get("algorithm")) == "xgboost", repr(cfg.get("algorithm")))
    check("父代其余配置仍按契约继承(features.exclude)",
          (cfg.get("features") or {}).get("exclude") == ["a"],
          repr(cfg.get("features")))
    check("超参覆写仍生效", cfg.get("hyperparameters") == {"max_depth": 7},
          repr(cfg.get("hyperparameters")))
    check("notes 保留父代留痕",
          "父代 notes" in str(cfg.get("notes") or ""),
          repr(cfg.get("notes")))
    check("flush 被调用", child.flushed >= 1)


def test_algorithm_falls_back_to_trial_id() -> None:
    print("== data.algorithm 缺失时退回 trial_id 算法段 ==")
    from runtime.logging_utils import load_yaml
    from _mvo.render import _copy_parent_trial_code  # noqa: E402

    tmp = tmp_run_dir("algo-fix2-")
    run_dir = pathlib.Path(tmp)
    parent_id = "trial-001-g00-lightgbm-1011"
    child_id = "trial-003-g01-catboost-2040"
    _write_parent_config(run_dir, parent_id, "lightgbm")
    child_dir = run_dir / "trials" / child_id
    (child_dir / "config").mkdir(parents=True, exist_ok=True)
    (child_dir / "code").mkdir(parents=True, exist_ok=True)

    child = _FakeTrial(child_id, child_dir, "")
    child.data.pop("algorithm", None)
    _copy_parent_trial_code(run_dir, child, parent_id, {},
                            action="explore")
    cfg = load_yaml(child_dir / "config" / "trial.yaml") or {}
    check("缺 data.algorithm 时退回命名段 catboost",
          str(cfg.get("algorithm")) == "catboost", repr(cfg.get("algorithm")))


def test_template_path_writes_algorithm() -> None:
    print("== 模板渲染路径也写 algorithm(旧实现整段缺失) ==")
    src = (SKILL / "scripts" / "_mvo" / "cmd_trial.py").read_text(
        encoding="utf-8")
    seg = src.split("trial_cfg = {")[1].split("dump_yaml(")[0]
    check("trial_cfg 含 algorithm 键", '"algorithm"' in seg, seg[:300])
    check("algorithm 取自 trial.data(与 trial.yaml 同源)",
          'trial.data.get("algorithm")' in seg)


def main() -> int:
    test_fast_path_config_algorithm()
    test_algorithm_falls_back_to_trial_id()
    test_template_path_writes_algorithm()
    print()
    if FAILS:
        print(f"{len(FAILS)} 项失败: {FAILS}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
