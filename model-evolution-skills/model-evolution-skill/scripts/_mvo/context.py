"""CLI 共享底座:路径/状态/装载 helper 与工作集常量

v2.11.9 从单文件 `scripts/modelevo.py` 拆出(拆分动机见
`_mvo/__init__.py`)。**本文件是机械搬运,未改语义** —— 便于对照原
实现审阅。新增命令请按职责放进对应模块,不要回填到 `modelevo.py`
(那里只留 argparse 与分发)。
"""
from __future__ import annotations

from . import SKILL_ROOT
from runtime.config import Config  # noqa: E402
from pathlib import Path
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialFactory  # noqa: E402
from runtime.logging_utils import (dump_json, dump_yaml, get_logger,  # noqa: E402
                                   load_json, load_yaml, locked_state_write,
                                   now_iso)
# v2.17.0:runtime.runner(numpy/pandas/evaluator)改为惰性导入——本模块是
# `wait` 轻量快路径的依赖,wait 进程不能为它付出 ~80MB 的重模块常驻
# (run4 regression.2:同机内存争用时 wait 两次被 cgroup OOM 杀)。旧的
# `from .context import TrialRunner / load_parent_scores` 经 __getattr__ 兼容。
import os
import time


def __getattr__(name: str):
    if name in ("TrialRunner", "load_parent_scores"):
        from runtime import runner as _runner

        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#: skill 根(等价于 `_mvo.SKILL_ROOT`;本模块内的短名)
_SKILL_ROOT = SKILL_ROOT

#: CLI 统一 logger(各命令模块 from .context import logger)
logger = get_logger("modelevo.cli")


def _run_dir_from(args) -> Path:
    # 缺省解析顺序:--run > --spec 所在目录 > MODELEVO_RUN_DIR > cwd。
    # v17.1 修复:status/leaderboard 在 run 目录内裸调(不带 --run/
    # --spec)时,旧实现直接 Path(args.spec).parent → Path(None) 崩溃
    # (TypeError: expected str, bytes or os.PathLike object, not
    # NoneType)。SKILL.md 的命令速查标榜"在 run 目录执行时可省",
    # 实际省了就炸——本修复让缺省真正可用。
    # v2.17.2(B6):一律返回绝对路径(--run/--spec/env 均 resolve;--spec
    # 只 resolve 目录,不跟随 spec 文件本身的符号链接)。相对 --run 在
    # detach worker / 子进程 / 换 cwd 的消费方里会拼出错误路径。
    if getattr(args, "run", None):
        rd = Path(args.run).resolve()
    elif getattr(args, "spec", None):
        rd = Path(args.spec).absolute().parent.resolve()
    else:
        env_run = os.environ.get("MODELEVO_RUN_DIR")
        rd = Path(env_run).resolve() if env_run else Path.cwd()
    _drop_stale_prescreen_root(rd)
    return rd


#: v2.17.2(B6):prescreen 沙箱环境变量(runtime.data.PRESCREEN_ROOT_ENV 同值;
#: 此处不 import runtime.data —— 本模块是 wait 轻量快路径的依赖)
_PRESCREEN_ROOT_ENV = "MODELEVO_PRESCREEN_ROOT"


def _drop_stale_prescreen_root(run_dir: Path) -> None:
    """CLI 进程里残留的 MODELEVO_PRESCREEN_ROOT 一律失效(v2.17.2 B6)。

    该变量只由 prescreen 注入**沙箱内 train.py 子进程**的环境;CLI 命令
    (check-products / status / diagnose / run …)从不在沙箱内执行。shell
    里残留它时,runtime.data 的读函数把真实 run 重映射进沙箱(沙箱无
    validation 标签面)→ "evaluator: validation 特征面/标签面缺失"。目标
    run 不在该沙箱内即视为残留:从本进程环境移除并告警(子进程随之不继承)。
    """
    root = os.environ.get(_PRESCREEN_ROOT_ENV)
    if not root:
        return
    try:
        root_p = Path(root).resolve()
        rd = Path(run_dir).resolve()
        inside = rd == root_p or root_p in rd.parents
    except OSError:
        inside = False
    if inside:
        return
    os.environ.pop(_PRESCREEN_ROOT_ENV, None)
    logger.warning(
        "忽略残留环境变量 %s=%s(只用于 prescreen 沙箱内的 trial 子进程;"
        "目标 run %s 不在该沙箱内)。建议在 shell 中 unset 它",
        _PRESCREEN_ROOT_ENV, root, run_dir)


def _resolve_splits_arg(run_dir: Path, splits_dir: str | None,
                        default: str = "sample-profile/splits") -> str:
    """--splits-dir → 绝对路径字符串(v2.17.2 B6)。

    相对路径沿用"相对 run 目录"的旧口径;run 下不存在而相对 cwd 存在时
    取后者(用户按 shell 位置传参的常见形态)。下游 `run_dir / splits_dir`
    对绝对路径天然成立。"""
    p = Path(splits_dir or default)
    if p.is_absolute():
        return str(p.resolve())
    cand = Path(run_dir) / p
    if not cand.exists() and p.resolve().exists():
        return str(p.resolve())
    return str(cand.resolve())


def _load_spec(args) -> TaskSpec:
    spec_path = Path(args.spec).absolute() if getattr(args, "spec", None) \
        else _run_dir_from(args) / "task-spec.yaml"
    if not spec_path.exists():
        # v17.1:给出可行动的报错(裸调不在 run 目录、或 spec 缺失时,
        # 旧实现抛 FileNotFoundError 裸栈)。status/leaderboard 这类
        # 读状态命令会先于 profile 产出 task-spec 被调用吗?不会——
        # 但 init 前误跑、或 cwd 不是 run 目录时会;提示 --spec/--run
        # 用法比栈追踪省一轮 triage。
        raise FileNotFoundError(
            f"task-spec.yaml 未找到(尝试: {spec_path})。"
            "请在 run 目录内执行,或用 --spec <path> / --run <dir> 指定"
            "(也可设环境变量 MODELEVO_RUN_DIR)。")
    spec = TaskSpec.load(spec_path)
    spec._spec_path = spec_path  # data-loader 上下文用
    return spec


def _load_config(run_dir: Path) -> Config:
    # skill 层缺省即应用(见 Config.load);显式 None 跳过
    return Config.load(run_config=run_dir / "modelevo-config.yaml")


def _started_at(run_dir: Path) -> float:
    st = run_dir / "run-state.json"
    if st.exists():
        return load_json(st).get("started_at_epoch", time.time())
    return time.time()


#: v2.17.2(B7):STOPPED 排在 FINAL_EVALUATED 之前——停止是建议性状态,
#: 之后的 evaluate-final 必须能把 phase 推进到 FINAL_EVALUATED(旧序 STOPPED
#: 最高,收尾后 phase 永远停在 STOPPED,status 字段还残留 PROFILED)。
#: 与 runtime.runner.TrialRunner.PHASE_ORDER 同序。
PHASE_ORDER = ["INITED", "PROFILED", "BASELINED", "EVOLVING",
               "STOPPED", "FINAL_EVALUATED"]


def _update_run_state(run_dir: Path, **kw) -> dict:
    st = run_dir / "run-state.json"
    # flock 临界区:与并行 trial 收尾/STOPPED 写入互斥,读-改-写不丢更新
    with locked_state_write(st):
        state = load_json(st) if st.exists() else {}
        # phase 只前进不后退:status 传入更高阶段才覆盖
        new_status = kw.get("status")
        if new_status in PHASE_ORDER:
            cur = state.get("phase") or state.get("status")
            if cur not in PHASE_ORDER or \
                    PHASE_ORDER.index(new_status) > PHASE_ORDER.index(cur):
                kw["phase"] = new_status
            else:
                kw.pop("status")  # 不回退;phase 保持
                kw["phase"] = cur
        state.update(kw)
        dump_json(st, state)
        return state


def _auto_ack_plumbing(run_dir: Path, st: dict, stamp: dict) -> dict | None:
    """v2.17.2(B8):harness 变更全部落在 plumbing 模块时自动登记。

    与 `acknowledge-harness-change` 的路径 1(只动 plumbing → 直接重新盖章、
    epoch 不变)同口径,只是由 run / run-generation 的 hash 门直接执行,不再
    要求 Agent 手工 ack。旧 run(无 harness_files)或触及 measurement 模块
    时返回 None,调用方照旧拒绝。返回写入 harness_change_log 的条目。
    """
    from runtime.harness import auto_acknowledgeable

    diff = auto_acknowledgeable(st.get("harness_files") or {},
                                stamp.get("harness_files") or {})
    if diff is None:
        return None
    epoch = int(st.get("harness_epoch", 0))
    log = list(st.get("harness_change_log") or [])
    entry = {
        "at": now_iso(),
        "from_hash": st.get("harness_hash"),
        "to_hash": stamp.get("harness_hash"),
        "changed": diff["changed"],
        "measurement_impact": False,
        "harness_epoch": epoch,
        "reason": "auto: 变更仅涉及 plumbing 模块(v2.17.2 自动登记)",
        "auto": True,
    }
    log.append(entry)
    _update_run_state(run_dir, harness_epoch=epoch,
                      harness_change_log=log, **stamp)
    logger.info("harness 变更仅涉及 plumbing 模块 %s:已自动登记"
                "(epoch=%d 不变,历史 trial 仍可比)", diff["changed"], epoch)
    return entry


_LAST_CREATED_TRIAL_DIR: Path | None = None


def _parse_list(text: str | None) -> list:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _resolve_baseline_id(run_dir: Path, spec: TaskSpec,
                         factory: TrialFactory) -> str | None:
    """解析 baseline_id:run-state 已设直接用;缺失时兜底取 gen-00 序号最小的
    trial(v2.15.7:baseline 固定为 trial-001,即使它失败也不换)。兜底结果回写
    run-state,防 baseline 命令被 kill 后 baseline_id 未落盘导致全部 trial
    的 delta_vs_baseline 恒空。
    """
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    bid = st.get("baseline_id")
    if bid:
        return bid
    # v2.15.7:兜底取 gen-00 序号最小的非 warm-start trial(即 trial-001
    # baseline)。不看是否已有分数:锚点按序号固定,失败也不换。
    gen0 = [t for t in factory.all_trials()
            if t.generation == 0
            and t.data.get("search_action") != "warm_start"
            and not str((t.data.get("hypothesis") or {}).get(
                "statement", "")).startswith("[warmstart")]
    if not gen0:
        return None
    baseline = min(gen0, key=lambda t: t.trial_id)
    logger.warning(
        "[baseline_id 缺失] run-state 未设 baseline_id,"
        "兜底取 gen-00 最小序号 trial %s 作为基线锚定(应确认与预期一致)",
        baseline.trial_id)
    _update_run_state(run_dir, baseline_id=baseline.trial_id)
    return baseline.trial_id


def _load_best_parent_scores(run_dir: Path, spec: TaskSpec,
                             parents: list[str]):
    """加载父代分数:crossover 多父代时取指标最优者作为对比基准(§12.2)。

    单父代直接返回;找不到任何父代分数返回 None(根 Trial 语义)。
    v12(P1-11,O4):uplift 的 compute_metric 必须传 treatment——
    漏传时 ValueError 被吞 → 视作根 trial → 全部误判 accepted。
    """
    if not parents:
        return None
    from runtime.runner import load_parent_scores

    if len(parents) == 1:
        return load_parent_scores(run_dir, parents[0])
    from runtime.metrics import compute_metric, metric_direction

    best_pid, best_val = None, None
    from runtime.data import load_labels, load_split_frames

    frames = load_split_frames(run_dir)
    valid = frames.get("validation")
    # v21 四平面:目标/treatment 在 splits/labels/,特征面物理不含;
    # 旧实现直接 valid[target] 必 KeyError(未进 try,整个 crossover
    # 创建路径崩),treatment 则恒 None——uplift 回归 P1-11 的洞
    _lab = load_labels(run_dir, splits=("validation",)).get("validation")
    treatment = None
    tcol = spec.treatment_column
    if _lab is not None and tcol is not None and tcol in _lab.columns:
        treatment = _lab[tcol].values
    for pid in parents:
        pack = load_parent_scores(run_dir, pid)
        if not pack or valid is None or _lab is None:
            continue
        import pandas as pd

        s = pack["validation"]["score"].values
        y = _lab[spec.target_name].values
        try:
            val = compute_metric(spec.primary_metric, y, s, score=s,
                                 treatment=treatment)
        except Exception:  # noqa: BLE001
            continue
        if best_val is None or (val > best_val if metric_direction(
                spec.primary_metric) else val < best_val):
            best_pid, best_val = pid, val
    if best_pid is None:
        return None
    return load_parent_scores(run_dir, best_pid)


def _resolve_splits_dir(run_dir: Path, cfg: dict) -> Path:
    """特征面根:config data_view 声明视图目录则用之,否则基础 splits。"""
    view = str(cfg.get("data_view") or "v1")
    if view and view not in ("v1", "base"):
        cand = run_dir / "sample-profile" / f"splits-{view}"
        if cand.exists():
            return cand
    return run_dir / "sample-profile" / "splits"


def _trial_summary_line(trial, spec) -> str:
    """SUMMARY 行格式单源(v17.2):cmd_run_trial 与 cmd_wait 共用。

    Generation Agent 只需解析这一行即可拿到 verdict/指标/delta/
    失败原因——两处输出口径必须一致,否则 wait 等来的行与 run 打出
    的行字段错位,Agent 解析会静默错读。"""
    ev = trial.data.get("evaluation") or {}
    pm = (ev.get("primary_metric") or {}).get("value")
    _delta = ev.get("delta") or {}
    dp = _delta.get("vs_parent")
    db = _delta.get("vs_baseline")
    shadow = ev.get("shadow_test") or {}
    return ("SUMMARY\t{}\t{}\t{}\t{}\tdelta_parent={}\tdelta_baseline={}"
            "\tshadow={}\tfailure={}\truntime={}".format(
                trial.trial_id, trial.status, ev.get("verdict") or "-",
                f"{pm:.6f}" if isinstance(pm, (int, float)) else "-",
                f"{dp:+.6f}" if isinstance(dp, (int, float)) else "-",
                f"{db:+.6f}" if isinstance(db, (int, float)) else "-",
                (f"{shadow.get(spec.primary_metric):.6f}"
                 if isinstance(shadow.get(spec.primary_metric),
                               (int, float)) else "-"),
                trial.data.get("failure_reason") or "-",
                (trial.data.get("result") or {}).get("runtime_sec", "-")))


_WAIT_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "EVALUATED", "REFLECTED"}


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _read_pid(path: Path) -> int:
    try:
        return int((Path(path).read_text(encoding="utf-8") or "0").strip()
                   or 0)
    except (OSError, ValueError):
        return 0


def _last_terminal_ts(trial) -> float | None:
    from datetime import datetime

    for ent in reversed(trial.data.get("status_history") or []):
        if ent.get("state") in _WAIT_TERMINAL_STATES:
            try:
                return datetime.fromisoformat(str(ent.get("at"))).timestamp()
            except (TypeError, ValueError):
                return None
    return None


def _reentry_pending(trial, run_dir) -> bool:
    """[v2.16.2] 本 trial 是否有"当前 attempt"正在路上(wait 不能认旧终态)。

    run4 regression D-006:FAILED trial 经 `run --trial <id> --reset-breaker
    --detach` 重入,父进程立即返回;worker 还没把状态改回 CODE_WRITTEN 时,
    `wait --trial` 读到上一轮 FAILED 立即返回。判据(任一成立即"在路上"):
      1. trial 目录有活 runner 锁(`.runner.lock` 的 pid 存活);
      2. `logs/detached-run-<id>.pid` 的 worker 存活,且 pid 文件写于最近一次
         终态之后(重入发生在该终态之后;旧 pid 文件 / pid 复用不误判)。
    """
    tid = getattr(trial, "trial_id", None)
    tdir = getattr(trial, "trial_dir", None)
    if not tid or run_dir is None:
        return False
    tdir = Path(tdir) if tdir else Path(run_dir) / "trials" / str(tid)
    if _pid_alive(_read_pid(tdir / ".runner.lock")):
        return True
    pidf = Path(run_dir) / "logs" / f"detached-run-{tid}.pid"
    if pidf.exists() and _pid_alive(_read_pid(pidf)):
        last = _last_terminal_ts(trial)
        try:
            started = pidf.stat().st_mtime
        except OSError:
            return False
        return last is None or started >= last - 1.0
    return False


def _trial_reached_terminal(trial, run_dir=None) -> bool:
    status = trial.status
    # v2.16.2:RUNNING 一律是活动态(孤儿由 wait 的 stale 探针处理)。旧实现对
    # 非终态都看 status_history,重入后 RUNNING 的 trial 因历史里有上一轮
    # FAILED 被判终态,wait 立即返回(run4 regression D-006)。
    if status == "RUNNING":
        return False
    # 当前 attempt 在路上(活 runner 锁 / 终态之后启动的 detached worker):
    # 不认上一轮终态(需 run_dir;旧调用方不传时行为不变)
    if run_dir is not None and _reentry_pending(trial, run_dir):
        return False
    if status in _WAIT_TERMINAL_STATES:
        return True
    # PROPOSED/CODE_WRITTEN 视作尚未启动——但若
    # status_history 表明它曾进入过终态又被回退(force_rerun 会把
    # EVALUATED 回写 CODE_WRITTEN),以历史为准立即返回,避免等一个
    # 由调用方自己负责的进程。
    history = [s.get("state") for s in trial.data.get("status_history", [])]
    return bool(set(history) & _WAIT_TERMINAL_STATES)
