# -*- coding: utf-8 -*-
"""model-evo/_modelevo-shared 公共配置读写: yaml 加载 + 通用必填校验 + 数据安全红线。

各 skill(feature-analysis / classification-model-training / classification-model-tuning)在本模块之上叠加自己的专有校验。
"""
from __future__ import annotations

import csv
import os
import re
import sys

import yaml

# 定位 model-skills 根(用于注入 feature-matching/scripts 到 sys.path、解析 feature_list_source 相对路径)。
# 兼容三种部署形态:
#   ① 仓库源:     model-evo/_modelevo-shared              → 父目录(model-evo)下有 model-skills/
#   ② 仓库软链接: model-skills/_modelevo-shared           → 父目录 basename == "model-skills"
#   ③ 安装后:     SKILL_ROOT/_modelevo-shared             → 各 skill 直接平铺在父目录下(无 model-skills 层)
# model-knowledge 在 model-skills/ 下; 相对路径(如 feature_list_source: model-knowledge/assets/.../
# xxx.csv)按 model-skills 根解析, 与 gen_feature_list.load_feature_list 一致, 不依赖 yaml 文件位置。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELEVO_SHARED_DIR = os.path.dirname(_THIS_DIR)
_PARENT = os.path.dirname(_MODELEVO_SHARED_DIR)
if os.path.basename(_PARENT) == "model-skills":
    _MODEL_SKILLS_ROOT = _PARENT
elif os.path.isdir(os.path.join(_PARENT, "model-skills")):
    _MODEL_SKILLS_ROOT = os.path.join(_PARENT, "model-skills")
else:
    _MODEL_SKILLS_ROOT = _PARENT

# 注入 feature-matching/scripts 到 sys.path 以复用 load_feature_list 的 CSV/TXT 解析
_FM_SCRIPTS = os.path.join(_MODEL_SKILLS_ROOT, "feature-matching", "scripts")
if _FM_SCRIPTS not in sys.path:
    sys.path.insert(0, _FM_SCRIPTS)

# 疑似敏感信息正则: 18位身份证 / 11位手机号
_SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{17}[\dxX]\b"),   # 身份证
    re.compile(r"\b1[3-9]\d{9}\b"),    # 手机号
]


def _load_feature_list(fpath: str) -> list:
    """加载特征清单, 复用 feature-matching/scripts/gen_feature_list 的解析逻辑。

    与各 skill _bootstrap 同款注入 sys.path 后 import gen_feature_list;
    .csv 取 feature_name 列(跳过表头), .txt 按行(跳过 # 注释), 去重保序。
    复用而非重写, 保证「特征清单如何解析」只有一处真相。

    Args:
        fpath: 特征清单文件绝对路径

    Returns:
        去重保序的 feature 名列表
    """
    from gen_feature_list import load_feature_list

    return load_feature_list(fpath)



def load_config(path: str) -> dict:
    """读取 yaml 配置文件并返回字典。

    Args:
        path: 配置 yaml 路径

    Returns:
        配置字典(含 _config_dir 用于解析相对路径; feature_list_source 已按 repo 根解析)
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def check_sensitive(text: str) -> None:
    """检查字符串是否含疑似用户ID/手机号/身份证号,命中即抛错。

    Args:
        text: 待检查文本(如 where 条件)

    Raises:
        ValueError: 命中敏感信息红线
    """
    if not text:
        return
    for pat in _SENSITIVE_PATTERNS:
        if pat.search(text):
            raise ValueError(f"触碰数据安全红线: 配置中疑似硬编码敏感信息: {text!r}")


def validate_common(cfg: dict) -> None:
    """通用配置校验: 必填字段 + features 加载 + 敏感信息红线。

    各 skill 在此基础上追加专有校验(如 model-training 校验 base_score_col)。

    Args:
        cfg: load_config 返回的字典

    Raises:
        ValueError: 缺必填项 / features 为空 / 命中敏感信息
    """
    model = cfg.get("model") or {}
    mode = (model.get("mode") or "spark").lower()
    is_local = mode == "local_file"

    # spark 模式必填 sample_table + fetch_dt; local_file 模式靠本地 parquet, 不需要取数表
    required = ["name", "dt_col"] if is_local else ["name", "sample_table", "dt_col", "fetch_dt"]

    has_label = bool(model.get("label_col")) or bool(model.get("label_expr"))

    for key in required:
        if not model.get(key):
            raise ValueError(f"配置 model.{key} 缺失或为空 (mode={mode})")
    if not has_label:
        raise ValueError("配置 model.label_col 与 model.label_expr 必须至少填一个")

    # features: 列表直接填,或通过外部文件加载(features_file 或 feature_list_source)
    # 相对路径解析顺序: ① yaml 所在目录(直觉行为, 支持 session 内相对引用)
    #                 ② model-skills 根(向后兼容 model-knowledge/... 风格)
    features_file = model.get("features_file") or model.get("feature_list_source")
    if features_file:
        if os.path.isabs(features_file):
            fpath = features_file
        else:
            cfg_dir = cfg.get("_config_dir")
            candidates = []
            if cfg_dir:
                candidates.append(os.path.join(cfg_dir, features_file))
            candidates.append(os.path.join(_MODEL_SKILLS_ROOT, features_file))
            fpath = next((c for c in candidates if os.path.exists(c)), candidates[-1])
        # 复用 feature-matching/scripts/gen_feature_list.load_feature_list 正确解析
        # (.csv 取 feature_name 列 / .txt 按行 / 跳过 # 注释 / 去重保序),
        # 避免朴素按行读取把 CSV 表头 feature_name 当成特征名。
        # 透传 _config_dir 让 load_feature_list 的相对路径基准与本函数一致。
        if cfg.get("_config_dir"):
            os.environ["_CONFIG_DIR"] = cfg["_config_dir"]
        model["features"] = _load_feature_list(fpath)
    # local_file 模式允许 features 为空: 视为"用本地 parquet 全部列(除 id/label/dt)"
    if not model.get("features") and not model.get("feature_table") and not is_local:
        raise ValueError("model.features 必填(auto_select 关闭); 或填 model.features_file / feature_list_source 从文件加载; 或填 model.feature_table 走特征表全列模式")

    fetch_dt = model.get("fetch_dt")
    if is_local:
        # local_file 模式 fetch_dt 不强求; 若填仍按列表两元素校验
        if fetch_dt is not None and not (isinstance(fetch_dt, list) and len(fetch_dt) == 2):
            raise ValueError("model.fetch_dt 须为 [起始, 结束] 两元素列表")
    else:
        if not (isinstance(fetch_dt, list) and len(fetch_dt) == 2):
            raise ValueError("model.fetch_dt 须为 [起始, 结束] 两元素列表")

    check_sensitive(model.get("where") or "")
    check_sensitive(model.get("sample_table") or "")


def _parse_range_pair(name: str, value) -> tuple:
    """把单档区间值规整成 (起, 止) 并校验 8 位 YYYYMMDD、起 ≤ 止。

    Args:
        name: 档名(train/test/oot), 仅用于报错信息
        value: 形如 ["20260312", "20260430"] 或 "20260312,20260430"

    Returns:
        (start, end) 两个 8 位日期字符串

    Raises:
        ValueError: 非两元素 / 非 8 位数字 / 起 > 止
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    else:
        raise ValueError("split.%s_range 须为 [起, 止] 列表或 '起,止' 字符串" % name)
    if len(parts) != 2:
        raise ValueError("split.%s_range 须为两元素 [起, 止], 当前 %r" % (name, value))
    start, end = parts
    for d in (start, end):
        if not (d.isdigit() and len(d) == 8):
            raise ValueError("split.%s_range 日期须为 8 位 YYYYMMDD, 当前 %r" % (name, d))
    if start > end:
        raise ValueError("split.%s_range 起始 %s 不应大于结束 %s" % (name, start, end))
    return start, end


def validate_split_ranges(model: dict) -> None:
    """校验 model.split(可选): train/test/oot 三档 pday 区间的合法性。

    仅当 model.split 存在时触发。约束(间隔逻辑):
      1. 三档齐全(train_range/test_range/oot_range 缺一报错)
      2. 每档 [起, 止] 为 8 位 YYYYMMDD 且起 ≤ 止
      3. 三档时序递增(train ≤ test ≤ oot), 允许相邻(前档结束日次日后档开始日),
         仅真正重叠或逆序才报错
      4. 三档并集 ⊆ model.fetch_dt(划分范围不得超出取数窗口);
         local_file 模式不强制 fetch_dt, 若未填则跳过本条

    Args:
        model: 配置 model 段

    Raises:
        ValueError: 任一约束不满足
    """
    split = model.get("split")
    if not split:
        return
    ranges = {}
    for name in ("train", "test", "oot"):
        key = "%s_range" % name
        if not split.get(key):
            raise ValueError("model.split 须三档齐全, 缺 %s" % key)
        ranges[name] = _parse_range_pair(name, split[key])

    ordered = [("train", ranges["train"]), ("test", ranges["test"]), ("oot", ranges["oot"])]
    for (n1, r1), (n2, r2) in zip(ordered, ordered[1:]):
        # 前档结束日必须早于后档开始日: 允许相邻(前档结束日次日 = 后档开始日),
        # 仅当前档结束日 >= 后档开始日时视为重叠或逆序
        if r1[1] >= r2[0]:
            raise ValueError(
                "split.%s_range [%s,%s] 与 split.%s_range [%s,%s] 重叠或逆序, "
                "要求 train<test<oot(允许相邻, 间隔≥1天)" % (n1, r1[0], r1[1], n2, r2[0], r2[1])
            )

    fetch_dt = model.get("fetch_dt")
    if isinstance(fetch_dt, list) and len(fetch_dt) == 2:
        f_start, f_end = str(fetch_dt[0]), str(fetch_dt[1])
        union_start = ranges["train"][0]
        union_end = ranges["oot"][1]
        # local_file 模式不强制 fetch_dt(本地 parquet 无取数窗口概念);
        # fetch_dt 为空字符串或占位时跳过本条校验
        if f_start and f_end and f_start.isdigit() and f_end.isdigit():
            if union_start < f_start or union_end > f_end:
                raise ValueError(
                    "train/test/oot 划分并集 [%s,%s] 超出取数窗口 fetch_dt [%s,%s]"
                    % (union_start, union_end, f_start, f_end)
                )
