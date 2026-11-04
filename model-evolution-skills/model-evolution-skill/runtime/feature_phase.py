"""特征期门(v2.16.0):协议「先特征、后模型」的唯一执行点。

背景:此前这条协议规则没有独立实现,一直由 capacity 闸门兼职——
`capacity-report.json` 的 action_ranking 含 add_rows/new_information
时拒绝 model/loss/hpo。而这两个信号在抽样面上几乎恒成立(本机 44 份
去重报告局部歧义指数全在 0.946–0.961、阈值 0.10,44/44 触发),闸门
实际输出是常量;76 个带放行记录的 trial 里 75 个是人工 override,
理由大多是同一句"add_rows 无源/信息源已消费完"。同时 Planner 与
new-trial 两层口径不一致(Planner 期满降 advisory,new-trial 不看代数
仍要 override;不带 --changed-modules 的 hpo 绕过 new-trial 门),
报告不存在时整条规则又完全不生效。

现口径(Planner 与 new-trial 共用本模块,capacity-check 只作参考诊断,
不再参与放行):
- 模型方向 = action == "hpo" 或 changed_modules 含 model/loss;
  confirm 不算(复现父代配方,不是新的模型方向投资)。
- 特征期 = 已完成(有 evaluation)的探索代数 <
  feature_exploration.min_generations(默认 2;gen-00 baseline 不计)。
  设为 0 即关闭本门。
- 特征期内模型方向默认拒绝,两条放行通道都留痕
  trial.yaml.feature_phase_override:
  (a) Planner 在特征侧已收敛时每代确定性释放一个名额
      (planner/latest.json 的 feature_phase_gate.released_proposal_id);
  (b) 人工 `new-trial --override-feature-phase "理由"`。
- 特征期满后不拦。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .logging_utils import load_json

MODEL_MODULES = frozenset({"model", "loss"})
#: Planner 拒绝码与 batch 键(report / new-trial 按此读取)
ERROR_CODE = "feature_phase_model_locked"
BATCH_KEY = "feature_phase_gate"


def is_model_direction(action: str | None,
                       changed_modules: Iterable[str] | None) -> bool:
    if str(action or "") == "confirm":
        return False
    mods = {str(m).strip().lower() for m in (changed_modules or [])}
    return str(action or "") == "hpo" or bool(mods & MODEL_MODULES)


def feature_generations_done(trials) -> int:
    """已完成(有 evaluation)的探索代数;gen-00 baseline 不计。"""
    return len({t.generation for t in trials
                if t.data.get("evaluation")} - {0})


def feature_phase_status(trials, config) -> dict:
    done = feature_generations_done(trials)
    need = int(config.get("feature_exploration.min_generations", 2))
    return {"active": done < need, "generations_done": done,
            "min_generations": need}


def planner_release(run_dir: Path, proposal_id: str | None) -> dict | None:
    """proposal_id 是最新批次释放的名额时,返回可落 trial.yaml 的留痕。"""
    if not proposal_id:
        return None
    try:
        batch = load_json(Path(run_dir) / "planner" / "latest.json") or {}
    except Exception:  # noqa: BLE001 — 批次不可读即视为未释放
        return None
    gate = batch.get(BATCH_KEY) or {}
    if not gate.get("released_proposal_id") or \
            gate.get("released_proposal_id") != proposal_id:
        return None
    return {"reason": gate.get("release_reason") or "",
            "proposal_id": proposal_id,
            "source": "planner_feature_phase_release"}
