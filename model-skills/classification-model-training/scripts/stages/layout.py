# -*- coding: utf-8 -*-
"""model-training 单次训练的目录布局与 manifest 落盘。

每次训练 run 落到:
  <output_dir>/new-models/{algo}[-suffix]-v{N}/
      ├── config.json
      ├── config/
      │   ├── train_config.yaml       # 入参 yaml 副本(可独立复现)
      │   └── _manifest.json
      ├── features/
      ├── model/
      ├── evaluation/
      ├── predictions/
      ├── explainability/
      └── logs/

命名规则:
  - baseline:  xgb-v1
  - 特征选择:  xgb-feat-v1
  - 调参:      xgb-tuned-v1
  - 同类再跑:  xgb-v2 / xgb-feat-v2 / xgb-tuned-v2 (version 自增)

各子目录由对应模块产出 + 写 `_manifest.json`,由 RunLayout 统一管理路径与 schema。
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_MANIFEST_SCHEMA_VERSION = "1"
_DEFAULT_PRODUCED_BY = "skills/model-training"
_VERSION_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
_STAGES = ("features", "model", "evaluation", "predictions", "explainability", "logs", "config")
_LAZY_STAGES = ("comparison",)  # 仅在有产物时由调用方 mkdir, 避免预创建空目录

_DIR_PATTERN = re.compile(r"^(?P<algo>[a-z0-9]+)(?P<suffix>-[a-z0-9]+)?-v(?P<n>\d+)$")

# algo 与 suffix 保留字: version 标识里若出现这些 token 会与目录命名规则冲突,
# 导致 {algo}{suffix}-{version} 出现重复前缀 (如 xgb-xgb-v1 / xgb-tuned-tuned-v1)。
# 列表来源: 内置 algo (xgb/dnn/lr) + 已知变体 (lgb) + tuning suffix 关键字 (tuned/feat)。
_RESERVED_VERSION_TOKENS = frozenset({
    "xgb", "dnn", "lr", "lgb", "gbm", "rf", "lightgbm",
    "tuned", "feat",
})


def validate_version_label(version: Optional[str]) -> None:
    """校验 version 标识不携带 algo/suffix 保留字前缀。

    目录命名规则为 `{algo}{suffix}-{version}`(如 `xgb-v1` / `xgb-feat-v1` /
    `xgb-tuned-v2`)。若 `version` 形如 `xgb-v1` / `tuned-v1` / `feat`,与 algo
    或 suffix 叠加后会出现 `xgb-xgb-v1` / `xgb-tuned-tuned-v1` / `xgb-feat`
    这类重复前缀或缺版本号的目录名。

    Args:
        version: 待校验的 version 标识(已 trim)。None 视为合法(走自动自增)。

    Raises:
        ValueError: version 含保留字 token,或格式不在白名单内。
    """
    if version is None:
        return
    s = (version or "").strip()
    if not s:
        return  # 空串走自动自增, 不在此拦截
    lower = s.lower()
    # 按 - / _ / . 切 token, 任一 token 命中保留字即拒绝
    tokens = {t for t in re.split(r"[-_.]", lower) if t}
    bad = tokens & _RESERVED_VERSION_TOKENS
    if bad:
        raise ValueError(
            f"version 标识 {version!r} 含算法/后缀保留字 {sorted(bad)}, "
            "会与目录命名规则 `{algo}{suffix}-{version}` 叠加产生重复前缀 "
            "(如 xgb-xgb-v1 / xgb-tuned-tuned-v1 / xgb-feat)。"
            "version 应为纯版本号(如 v1 / v2 / custom-tag), 不要带 algo 或 suffix 前缀。"
        )


def normalize_version(version: str) -> str:
    """约束 version 字符集: 字母/数字/下划线/中划线/点; 其他字符替换为 '_'。

    Args:
        version: 用户输入的版本标识(如 v1 / v2 / custom-tag)

    Returns:
        归一化后的 version;空串回退为 'v1'
    """
    s = (version or "").strip()
    if not s:
        return "v1"
    return "".join(c if c in _VERSION_OK else "_" for c in s)


def next_version(output_dir: str, algo: str, suffix: str = "") -> str:
    """扫描 <output_dir>/new-models/ 下同 algo+suffix 的目录,返回自增的 v{N+1}。

    匹配规则: 目录名形如 `{algo}{suffix}-v{N}`(如 `xgb-v1` / `xgb-feat-v2`),
    找最大 N,返回 `v{N+1}`;无匹配则返回 `v1`。

    Args:
        output_dir: model-training skill 的输出根(通常 <session_dir>)
        algo: 算法标识 xgb|dnn|lr
        suffix: 后缀("" / "-feat" / "-tuned")

    Returns:
        形如 "v2" 的版本号字符串
    """
    new_models_dir = Path(output_dir) / "new-models"
    if not new_models_dir.exists():
        return "v1"
    prefix = f"{algo}{suffix}-v"
    max_n = 0
    for entry in new_models_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith(prefix):
            continue
        n_str = name[len(prefix):]
        if n_str.isdigit():
            max_n = max(max_n, int(n_str))
    return "v%d" % (max_n + 1)


@dataclass(frozen=True)
class RunLayout:
    """单次训练 run 的目录骨架(只读路径容器)。

    用 `RunLayout.create(output_dir, algo, suffix, version)` 工厂方法构造,会立即 mkdir 全部子目录,
    使各阶段产出代码可以无脑写入。
    """

    run_dir: Path
    algo: str
    suffix: str
    version: str
    timestamp: str
    config_json: Path
    config_dir: Path
    features_dir: Path
    model_dir: Path
    evaluation_dir: Path
    predictions_dir: Path
    explainability_dir: Path
    logs_dir: Path
    comparison_dir: Path

    @classmethod
    def create(
        cls,
        output_dir: str,
        algo: str,
        suffix: str = "",
        version: Optional[str] = None,
        when: Optional[datetime] = None,
    ) -> "RunLayout":
        """在 `<output_dir>/new-models/` 下新建本 run 的目录骨架。

        目录名: `{algo}{suffix}-{version}`(如 xgb-v1 / xgb-feat-v1 / xgb-tuned-v2)。

        Args:
            output_dir: model-training skill 的输出根(通常 <session_dir>)
            algo: 算法标识 xgb|dnn|lr
            suffix: 后缀(""=baseline / "-feat" / "-tuned");默认 ""
            version: 版本号(如 v1 / v2);None 或空时自动调 next_version 自增
            when: 时间戳(测试注入);None=当前,仅供 config.json 记录,不入目录名

        Returns:
            RunLayout 实例(各子目录已 mkdir)
        """
        algo = (algo or "xgb").lower()
        if suffix and not suffix.startswith("-"):
            suffix = "-" + suffix
        version = normalize_version(version) if version else next_version(output_dir, algo, suffix)
        ts = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
        run_dir = Path(output_dir) / "new-models" / f"{algo}{suffix}-{version}"
        run_dir.mkdir(parents=True, exist_ok=True)
        subdirs = {name: run_dir / name for name in _STAGES}
        for p in subdirs.values():
            p.mkdir(parents=True, exist_ok=True)
        # comparison_dir 延迟创建: 仅在 invoke/comparison.py 实际产 comparison_* 时 mkdir
        comparison_dir = run_dir / "comparison"
        return cls(
            run_dir=run_dir,
            algo=algo,
            suffix=suffix,
            version=version,
            timestamp=ts,
            config_json=run_dir / "config.json",
            config_dir=subdirs["config"],
            features_dir=subdirs["features"],
            model_dir=subdirs["model"],
            evaluation_dir=subdirs["evaluation"],
            predictions_dir=subdirs["predictions"],
            explainability_dir=subdirs["explainability"],
            logs_dir=subdirs["logs"],
            comparison_dir=comparison_dir,
        )

    @property
    def label(self) -> str:
        """返回 version(layout 的 version 标识)。"""
        return self.version

    @property
    def run_name(self) -> str:
        """目录名 = run name(供模型报告/日志标题用)。"""
        return self.run_dir.name


def _file_entry(path: Path) -> Dict[str, Any]:
    """构造 manifest 单文件条目(name + 字节数;不存在的文件标 status=missing)。"""
    if not path.exists():
        return {"name": path.name, "status": "missing"}
    return {"name": path.name, "size": path.stat().st_size}


def write_manifest(
    stage_dir: Path,
    stage: str,
    files: List[Path],
    extra: Optional[Dict[str, Any]] = None,
    produced_by: Optional[str] = None,
) -> Path:
    """把当前阶段的产物清单写到 stage_dir/_manifest.json。

    Args:
        stage_dir: 阶段目录(如 RunLayout.model_dir)
        stage: 阶段名(features/model/evaluation/predictions/explainability/logs)
        files: 该阶段产出的文件列表;不存在的文件会标 status=missing(可见性更高)
        extra: 任意补充字段(算法/超参概要/skipped 原因等),并入 manifest 顶层
        produced_by: 写入 manifest 的来源标识;None 时默认 "skills/model-training"
            (model-tuning 等下游 skill 通过传值标识自己)

    Returns:
        _manifest.json 的绝对路径
    """
    payload: Dict[str, Any] = {
        "stage": stage,
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "produced_by": produced_by or _DEFAULT_PRODUCED_BY,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": [_file_entry(Path(f)) for f in files],
    }
    if extra:
        payload.update(extra)
    out = stage_dir / "_manifest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_config_snapshot(
    layout: RunLayout,
    cfg: dict,
    data_dir: str,
    extra: Optional[Dict[str, Any]] = None,
    produced_by: Optional[str] = None,
) -> Path:
    """落 config.json 快照: 运行时配置 + 关键路径 + version/algo/timestamp。

    剔除 _config_dir(load_config 注入,不可移植)等私有字段。

    Args:
        layout: 本次 run 的 layout
        cfg: load_config 返回的 dict(已 validate)
        data_dir: 上游 sample.parquet 所在目录
        extra: 训练时机/超参等附加信息
        produced_by: 标识谁生成了这个快照;None 时默认 "skills/model-training"

    Returns:
        config.json 绝对路径
    """
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    # pre-split 模式: 三档路径指 <data_dir>/splits/{train,test,oot}.parquet
    # 与 run_build._load_pre_split_data 实际读取路径一致, 供下游 select_features / run_tuning 复用
    splits_dir = os.path.join(data_dir, "splits")
    snapshot: Dict[str, Any] = {
        "run_name": layout.run_name,
        "algo": layout.algo,
        "suffix": layout.suffix,
        "version": layout.version,
        "label": layout.version,
        "timestamp": layout.timestamp,
        "produced_by": produced_by or _DEFAULT_PRODUCED_BY,
        "input": {
            "data_dir": os.path.abspath(data_dir),
            "sample_path": os.path.abspath(os.path.join(data_dir, "sample.parquet")),
            "train_path": os.path.abspath(os.path.join(splits_dir, "train.parquet")),
            "test_path": os.path.abspath(os.path.join(splits_dir, "test.parquet")),
            "oot_path": os.path.abspath(os.path.join(splits_dir, "oot.parquet")),
        },
        "output": {"run_dir": str(layout.run_dir.resolve())},
        "config": clean_cfg,
    }
    if extra:
        snapshot["runtime"] = extra
    layout.config_json.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return layout.config_json


def write_train_config_yaml(
    layout: RunLayout,
    source_yaml_path: Optional[str],
    produced_by: Optional[str] = None,
) -> Optional[Path]:
    """登记 run_dir/config/ 下的 train_config.yaml, 写 _manifest.json。

    输入 yaml 应已落在 layout.config_dir/train_config.yaml (model 内部 config 目录,
    SKILL.md §6 强制约束)。本函数不做副本拷贝, 只写 _manifest.json 溯源。
    若 source_yaml_path 指向 config_dir 之外的文件, 作为兜底 copyfile 进来, 并打 info
    提示输入 yaml 应直接放 <run_dir>/config/train_config.yaml。

    Args:
        layout: 本次 run 的 layout
        source_yaml_path: 入参 yaml 路径;None 时跳过
        produced_by: 标识来源;None 时默认 "skills/model-training"

    Returns:
        落盘的 train_config.yaml 路径;跳过时返回 None
    """
    if not source_yaml_path:
        return None
    src = Path(source_yaml_path)
    if not src.exists():
        return None
    dst = layout.config_dir / "train_config.yaml"
    try:
        src_abs = src.resolve()
        dst_abs = dst.resolve()
    except OSError:
        src_abs, dst_abs = src, dst
    if src_abs != dst_abs:
        # 兜底: 输入 yaml 不在 run_dir/config/ 下, copyfile 进来并提示
        shutil.copyfile(str(src), str(dst))
        print(
            f"[write_train_config_yaml] info: 输入 yaml 从 {src} 拷贝到 {dst}。"
            "输入 yaml 应直接放 <run_dir>/config/train_config.yaml, "
            "详见 SKILL.md §6 输入 yaml 落盘约束。"
        )
    write_manifest(
        stage_dir=layout.config_dir,
        stage="config",
        files=[dst],
        extra={"source_yaml": str(src.resolve())},
        produced_by=produced_by or _DEFAULT_PRODUCED_BY,
    )
    return dst
