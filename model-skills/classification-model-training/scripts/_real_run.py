# -*- coding: utf-8 -*-
"""一次性驱动: 用真实 config + sample.parquet 跑单一 algo(每进程一种, 模块隔离)。

用法: python scripts/_real_run.py <algo> <data_dir> <output_dir> <config_yaml>
algo ∈ xgb|dnn|lr。config_yaml 指向输入 yaml(应已落 <session_dir>/new-models/{algo}-v{N}/config/train_config.yaml)。
load_config + validate + 覆盖 model.algo, 调 run。
data_dir 需含 sample.parquet(由 feature-matching 产出); 切分由本 skill 按 model.split 内部完成。
"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from validate_config import load_config, validate_config
from run_build import run


def main() -> None:
    """读配置 -> 覆盖 algo -> 真训练+对比, 打印关键指标。"""
    algo = sys.argv[1]
    data_dir = sys.argv[2]
    out_dir = sys.argv[3]
    config_yaml = sys.argv[4] if len(sys.argv) > 4 else str(
        _SCRIPTS.parent / "config" / "train_config.yaml"
    )
    cfg = load_config(config_yaml)
    validate_config(cfg)
    cfg["model"]["algo"] = algo
    print(f"[real_run] algo={algo} data_dir={data_dir} out={out_dir} cfg={config_yaml}", flush=True)
    res = run(
        cfg, data_dir, out_dir,
        version="real_run",
        source_yaml_path=config_yaml,
    )
    print(f"[real_run] DONE algo={algo}: {res}", flush=True)


if __name__ == "__main__":
    main()
