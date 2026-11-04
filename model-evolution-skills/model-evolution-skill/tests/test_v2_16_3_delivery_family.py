"""v2.16.3:finalize-retrain 算法族校验对 trial 自定义复合容器(run4 regression D-013)。

搜索期 HurdleXGB 定义在 `__main__`,交付沙箱从配方副本以 `champ_train` 加载;
旧 `_model_family` 取模块根名,把同一配方判成换族并拒绝落盘。
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402,F401  hermetic import bootstrap

import xgboost as xgb  # noqa: E402

from scripts._mvo.cmd_deliver import _model_family  # noqa: E402


def _container_cls(module_name: str, cls_name: str = "HurdleXGB"):
    mod = types.ModuleType(module_name)

    def __init__(self, clf, reg, stage1=None):
        self.clf_ = clf
        self.reg_ = reg
        self.stage1_ = stage1

    cls = type(cls_name, (), {"__init__": __init__, "__module__": module_name})
    cls.__qualname__ = cls_name
    setattr(mod, cls_name, cls)
    return cls


def test_library_models_keep_library_family():
    assert _model_family(xgb.XGBRegressor()) == "xgboost"
    assert _model_family(xgb.XGBClassifier()) == "xgboost"


def test_custom_container_same_family_across_modules():
    search = _container_cls("__main__")(xgb.XGBClassifier(), xgb.XGBRegressor(),
                                        {"clf_best_iteration": 3})
    delivery = _container_cls("champ_train")(xgb.XGBClassifier(), xgb.XGBRegressor())
    fam = _model_family(search)
    assert fam == _model_family(delivery) == "custom:HurdleXGB[xgboost]"


def test_custom_container_differs_from_library_and_other_containers():
    hurdle = _container_cls("__main__")(xgb.XGBClassifier(), xgb.XGBRegressor())
    other = _container_cls("__main__", "TLearner")(xgb.XGBRegressor(), xgb.XGBRegressor())
    assert _model_family(hurdle) != "xgboost"
    assert _model_family(hurdle) != _model_family(other)
