# -*- coding: utf-8 -*-
"""config_io 通用校验单测。"""
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from config_io import load_config, validate_common, validate_split_ranges


def _base_cfg():
    """返回一份合法的最小通用配置字典(name/sample_table/dt_col/label/fetch_dt/features)。"""
    return {
        "spark": {"app_name": "t", "master": "local[*]"},
        "model": {
            "name": "m", "sample_table": "db.t",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "fetch_dt": ["20250101", "20250131"], "where": None,
            "features": ["f0", "f1"],
        },
    }


def test_validate_ok():
    """合法配置应通过校验。"""
    validate_common(_base_cfg())


def test_validate_missing_features():
    """features 为空必须报错(auto_select 关闭,引擎不会自挑列)。"""
    cfg = _base_cfg()
    cfg["model"]["features"] = []
    with pytest.raises(ValueError, match="features"):
        validate_common(cfg)


def test_validate_missing_label():
    """label_col 与 label_expr 都缺必须报错。"""
    cfg = _base_cfg()
    cfg["model"].pop("label_col")
    with pytest.raises(ValueError, match="label"):
        validate_common(cfg)


def test_validate_rejects_hardcoded_id_in_where():
    """where 里出现疑似身份证/手机号必须报错(数据安全红线)。"""
    cfg = _base_cfg()
    cfg["model"]["where"] = "id_card='110101199001011234'"
    with pytest.raises(ValueError, match="安全红线|敏感"):
        validate_common(cfg)


def test_load_config_reads_yaml(tmp_path):
    """load_config 读 yaml 并返回 dict。"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_base_cfg(), allow_unicode=True), encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["model"]["name"] == "m"


def test_validate_rejects_hardcoded_phone_in_where():
    """where 里出现疑似手机号必须报错(数据安全红线)。"""
    cfg = _base_cfg()
    cfg["model"]["where"] = "mobile='13800138000'"
    with pytest.raises(ValueError, match="安全红线|敏感"):
        validate_common(cfg)


def test_validate_rejects_sensitive_in_sample_table():
    """sample_table 里出现疑似敏感信息也必须报错(校验覆盖 where 与 sample_table)。"""
    cfg = _base_cfg()
    cfg["model"]["sample_table"] = "db.13800138000"
    with pytest.raises(ValueError, match="安全红线|敏感"):
        validate_common(cfg)


def test_validate_bad_fetch_dt_shape():
    """fetch_dt 非两元素列表必须报错。"""
    cfg = _base_cfg()
    cfg["model"]["fetch_dt"] = ["20250101"]
    with pytest.raises(ValueError, match="fetch_dt"):
        validate_common(cfg)


def test_validate_features_file(tmp_path):
    """features_file 指向外部文件时,应加载为 features 列表。"""
    feats_path = tmp_path / "feats.txt"
    feats_path.write_text("a\nb\nc\n", encoding="utf-8")
    cfg = _base_cfg()
    cfg["model"].pop("features")
    cfg["model"]["features_file"] = str(feats_path)
    validate_common(cfg)
    assert cfg["model"]["features"] == ["a", "b", "c"]


# ---- validate_split_ranges: 可选 model.split 时间划分校验 ----

def _split_model():
    """返回含合法 model.split 的 model 段。"""
    return {
        "fetch_dt": ["20260312", "20260524"],
        "split": {
            "train_range": ["20260312", "20260430"],
            "test_range": ["20260501", "20260516"],
            "oot_range": ["20260517", "20260524"],
        },
    }


def test_split_ranges_none_noop():
    """无 model.split 时直接返回, 不报错。"""
    validate_split_ranges({"fetch_dt": ["20260312", "20260524"]})


def test_split_ranges_ok_list_and_str():
    """合法三档(列表)通过; 字符串 '起,止' 也支持。"""
    validate_split_ranges(_split_model())
    m = _split_model()
    m["split"]["train_range"] = "20260312,20260430"
    validate_split_ranges(m)


def test_split_ranges_missing_tier():
    """三档缺一报错。"""
    m = _split_model()
    m["split"].pop("oot_range")
    with pytest.raises(ValueError, match="oot_range"):
        validate_split_ranges(m)


def test_split_ranges_bad_date():
    """非 8 位日期报错。"""
    m = _split_model()
    m["split"]["train_range"] = ["2026031", "20260430"]
    with pytest.raises(ValueError, match="8 位"):
        validate_split_ranges(m)


def test_split_ranges_start_gt_end():
    """单档起 > 止报错。"""
    m = _split_model()
    m["split"]["test_range"] = ["20260516", "20260501"]
    with pytest.raises(ValueError, match="不应大于"):
        validate_split_ranges(m)


@pytest.mark.parametrize("test_range", [
    ["20260420", "20260516"],   # 与 train 重叠
    ["20260430", "20260516"],   # 与 train 同日(前档结束日=后档开始日, 视为重叠)
])
def test_split_ranges_overlap(test_range):
    """三档重叠/同日报错(允许相邻: 前档结束日次日=后档开始日)。"""
    m = _split_model()
    m["split"]["test_range"] = test_range
    with pytest.raises(ValueError, match="重叠或逆序"):
        validate_split_ranges(m)


def test_split_ranges_adjacent_ok():
    """相邻间隔(前档结束日次日=后档开始日)允许通过。"""
    m = _split_model()
    # train 20260312~20260430, test 次日 20260501 起
    m["split"]["test_range"] = ["20260501", "20260516"]
    # test 20260501~20260516, oot 次日 20260517 起
    m["split"]["oot_range"] = ["20260517", "20260524"]
    validate_split_ranges(m)  # 不报错


def test_split_ranges_exceed_fetch_dt():
    """划分并集超出 fetch_dt 报错。"""
    m = _split_model()
    m["split"]["oot_range"] = ["20260517", "20260601"]  # 超出 fetch_dt 末端
    with pytest.raises(ValueError, match="超出取数窗口"):
        validate_split_ranges(m)
