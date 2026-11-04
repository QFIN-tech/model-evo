"""依赖预检(v11):三层依赖策略的确定性执行,修「不得安装任何包」
与 requirements.txt/铁律 11 允许值的自相矛盾。

分层:
- required:runtime/requirements.txt 声明,缺失 → preflight 报错并
  阻塞(这是 setup 阶段,允许阻塞用户);
- optional:optuna/catboost/torch 等增强依赖,try-import 记录
  import_ok,缺失不报错(功能回退),但记 env-manifest 供追溯;
- 环境记录(v20):BASELINED 之后的环境变化**只记录、不阻塞**
  (铁律 10 的环境侧因此不再判死)。env_fingerprint 与 harness_hash
  并列写入 run-state,env_signature/classify_env_drift 给出结构化
  归因,事件落 run 目录 env-ledger.jsonl。

v20 背景(实测三处失配,详见 docs/MAINTENANCE.md):

1. 装 optional 包(catboost/torch)不改任何已跑 trial 的字节 ——
   「用了什么算法」本就记在 trial 的 algorithm 字段。用 run 级
   kill switch 处理 per-trial 属性,把合法演进动作判成违规。
2. 旧指纹(`if p["import_ok"]`)不含**缺失**的 required 包,而
   cmd_preflight 先写指纹再报 missing —— 用户按提示装好缺失的包,
   签名从「没有」变「有」→ 漂移 → 之后所有 run 全炸。修好环境 =
   作废整轮运行。故本模块把「缺失→已装」(修复)与「版本变化」
   (真正让历史 trial 不可复现)分成不同 kind。
3. 指纹里没有任何解释器标识,而 trial 是另起进程跑的(v14 实测
   「CLI 全绿、10 个 baseline 全灭」)。v20 起 sys.executable /
   Python 版本 / trial 解释器路径一并入签名。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as _md
import json
import os
import sys
from pathlib import Path
from typing import Any

from .logging_utils import append_jsonl, dump_json, get_logger, now_iso

logger = get_logger(__name__)

# optional tier:模块名 → 用途说明(import_ok=false 时功能降级路径)
# v20:openpyxl 从本表移除 —— requirements.txt:15 已按 required 声明,
# 两处并存使旧指纹里它被探两次(签名重复一份,且 its tier 语义含混)。
OPTIONAL_PACKAGES: dict[str, str] = {
    "optuna": "hpo trial 的 TPE 采样器(缺失回退随机搜索)",
    "catboost": "算法族(缺失则该算法不可用,铁律 11 声明值跳过)",
    "torch": "mlp/nn 深度学习 trial(缺失则回退 GBM)",
    "sentence_transformers": "经验检索 embedding(缺失回退 n-gram)",
    # v18/P2-3:automl 参照系引擎(--action automl;缺失时该 trial
    # 显式报错而非静默降级,参照系缺席好过假参照)
    "autogluon": "automl 参照系引擎(--action automl;缺失则不可用)",
    "flaml": "automl 参照系引擎备选(--action automl --algorithm flaml)",
}

#: import 名 → 分发名(仅 importlib.metadata 回退取版本时用)
_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "sentence_transformers": "sentence-transformers",
}

#: 环境台账文件名(run 目录下,与 delete-log.jsonl 同层)
ENV_LEDGER = "env-ledger.jsonl"

#: 漂移 kind → 语义(供 ledger 与 diagnose 措辞复用)
DRIFT_KINDS: dict[str, str] = {
    "required_version_change": "已装 required 包版本变化(历史 trial 不可复现)",
    "required_appeared": "缺失的 required 包装上(修复动作,非违规)",
    "required_disappeared": "required 包消失(环境损坏)",
    "optional_change": "optional 依赖集合变化(装/卸/版本;纯加法不影响既有 trial)",
    "interpreter_change": "解释器身份变化(路径或 Python 版本)",
    "unknown_baseline": "run 的环境基线早于逐包台账(仅指纹串,无从归因)",
}


def _version_of(mod: Any) -> str | None:
    v = getattr(mod, "__version__", None)
    return str(v) if v else None


def _dist_version(dists: list[str]) -> str | None:
    """`__version__` 缺失时回退 importlib.metadata(v20)。

    实测:pynvml 装了但无 `__version__`,旧实现记 None → 签名里成
    `'?'`,版本升降完全测不出(只能测装没装)。
    """
    for d in dists:
        try:
            return str(_md.version(d))
        except Exception:  # noqa: BLE001  该分发名不存在
            continue
    return None


def _probe(import_name: str, tier: str, note: str = "",
           dists: list[str] | None = None) -> dict:
    """探一个包:**装与没装都记**(v20)。

    旧实现只把 import_ok 的包放进签名,导致「缺失 → 已装」也算漂移
    —— 用户按 preflight 提示装好缺失的 required 包,反而把整轮运行
    判死。新版签名对 absent 也留状态位,让修复与版本变化可区分
    (归因见 classify_env_drift)。
    """
    dists = list(dists or [_IMPORT_TO_DIST.get(import_name, import_name)])
    try:
        mod = importlib.import_module(import_name)
    except Exception as exc:  # noqa: BLE001
        return {"name": import_name, "tier": tier, "import_ok": False,
                "version": None, "dists": dists,
                "error": str(exc)[:120], "note": note}
    return {"name": import_name, "tier": tier, "import_ok": True,
            "version": _version_of(mod) or _dist_version(dists),
            "dists": dists, "note": note}


# pip 名 → import 名映射(仅两例不一致,其余同名)
_PIP_TO_IMPORT = {"scikit-learn": "sklearn", "pyyaml": "yaml"}


def _read_required(skill_root: Path) -> list[tuple[str, str]]:
    """requirements.txt → [(分发名, import 名)];重复 import 名已归并。

    v20:同一 import 名被多处声明(实测 openpyxl 同时在 required 与
    optional 表)时只保留一条,分发名并入 dists 供版本回退用。
    """
    req_path = Path(skill_root) / "runtime" / "requirements.txt"
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    if not req_path.exists():
        return out
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        # "pandas>=1.5" → pip 名,再映射 import 名
        pip = line.split(">=")[0].split("==")[0].split("<")[0].strip()
        imp = _PIP_TO_IMPORT.get(pip, pip)
        if imp in seen:
            continue
        seen.add(imp)
        out.append((pip, imp))
    return out


def probe_gbm_backends(python_exe: str | None = None) -> dict:
    """探测 GBM 后端的 **CUDA 编译位**(v2.12.2.9)。

    动机:lightgbm 的 PyPI wheel 默认**不带 CUDA**(需要
    `cmake -DUSE_CUDA=1` 自编),而调度器只要机器上有卡就注入
    ``MODELEVO_GPU=1``。于是 trial 模板每次都先试 ``device=cuda``、
    由 ``_lgbm_cuda_ok()`` 探针捕获 LightGBM 抛出的 Fatal 再回退 CPU。
    这条 Fatal 由 C++ 层直写 fd 2,会落进 ``trials/<id>/stderr.log``,
    **看起来像 trial 崩溃**(实测一次排查即被误导)。

    在 preflight 阶段把「本机 GBM 能不能上卡」变成一条可查的环境事实,
    而不是翻 stderr 才能发现。探测在**子进程**里做(与 trial 同口径),
    并且子进程内已重定向 fd 2,不会污染 preflight 自身的输出。

    返回结构落 ``env-manifest.json.gbm_backends``,供 run/report 归因。
    """
    import subprocess

    exe = python_exe or sys.executable
    code = (
        "import json, os, sys\n"
        "sys.path.insert(0, os.environ.get('MODELEVO_SKILL_ROOT', ''))\n"
        "import contextlib, io\n"
        "out = {}\n"
        "try:\n"
        "    import lightgbm, numpy as np\n"
        "    out['lightgbm_version'] = lightgbm.__version__\n"
        "    X = np.random.RandomState(0).rand(64, 4).astype('float32')\n"
        "    y = (np.random.RandomState(1).rand(64) > 0.5).astype(int)\n"
        "    _saved = os.dup(2)\n"
        "    _null = os.open(os.devnull, os.O_WRONLY)\n"
        "    os.dup2(_null, 2); os.close(_null)\n"
        "    try:\n"
        "        lightgbm.LGBMClassifier(device='cuda', n_estimators=1,\n"
        "                                verbosity=-1,\n"
        "                                random_state=0).fit(X, y)\n"
        "        out['lightgbm_cuda'] = True\n"
        "    except Exception as e:\n"
        "        out['lightgbm_cuda'] = False\n"
        "        out['lightgbm_cuda_error'] = str(e)[:120]\n"
        "    finally:\n"
        "        os.dup2(_saved, 2); os.close(_saved)\n"
        "except Exception as e:\n"
        "    out['lightgbm_error'] = str(e)[:120]\n"
        "try:\n"
        "    import xgboost\n"
        "    out['xgboost_version'] = xgboost.__version__\n"
        "    out['xgboost_cuda'] = bool(\n"
        "        (xgboost.build_info() or {}).get('USE_CUDA'))\n"
        "except Exception as e:\n"
        "    out['xgboost_error'] = str(e)[:120]\n"
        "print(json.dumps(out))\n")
    env = dict(os.environ)
    env["MODELEVO_SKILL_ROOT"] = str(Path(__file__).resolve().parent.parent)
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True,
                           text=True, timeout=120, env=env)
        probe = json.loads(r.stdout.strip() or "{}")
    except Exception as exc:  # noqa: BLE001  探测失败不阻塞 preflight
        probe = {"error": str(exc)[:120]}
    if probe.get("lightgbm_cuda") is False:
        probe["note"] = (
            "lightgbm wheel 非 CUDA 编译(通常是 PyPI wheel;CUDA 版需 "
            "cmake -DUSE_CUDA=1 自编)——该算法族一律 CPU 训练,这是"
            "预期的降级而非故障")
    return probe


def _probe_all(skill_root: Path) -> list[dict]:
    """required + optional 一次探完,按 import 名归并(required 优先)。"""
    merged: dict[str, dict] = {}
    for pip, imp in _read_required(skill_root):
        merged[imp] = _probe(imp, "required", dists=[pip])
    for name, note in OPTIONAL_PACKAGES.items():
        if name in merged:  # required 已声明 → 保留 required 语义
            continue
        merged[name] = _probe(name, "optional", note)
    return list(merged.values())


def interpreter_identity(trial_interp: dict | None = None) -> dict:
    """解释器身份(v20):**trial 是另起进程跑的**,故两个都记。

    旧指纹里零解释器标识 —— 换个 venv 而版本恰好相同即指纹一致、
    无告警,而 v14 实测过「CLI 解释器全绿、10 个 baseline trial 全灭」。
    """
    return {
        "cli_executable": sys.executable,
        "cli_python": "%d.%d.%d" % sys.version_info[:3],
        "trial_executable": (trial_interp or {}).get("interpreter"),
        "trial_python": (trial_interp or {}).get("python_version"),
    }


def env_signature(packages: list[dict], interp: dict) -> str:
    """包状态(含 absent)+ 解释器身份 → sha256 前 16 位。

    与旧签名的差别:absent 的包也占位(`name=absent`),解释器入签。
    """
    parts = []
    for p in sorted(packages, key=lambda x: x["name"]):
        if not p.get("import_ok"):
            state = "absent"
        else:
            state = p.get("version") or "unknown"
        parts.append(f"{p['name']}={state}")
    parts.append("cli=%s|%s" % (interp.get("cli_executable"),
                                interp.get("cli_python")))
    parts.append("trial=%s|%s" % (interp.get("trial_executable"),
                                  interp.get("trial_python")))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def run_preflight(skill_root: Path,
                  output_path: Path | None = None) -> dict:
    """扫描 required + optional 依赖并返回环境清单。

    默认纯读取，不改 skill 仓库。CLI 显式传任务 run 内的 output_path，
    避免测试/任务执行污染受版本控制的 assets/env-manifest.json。
    """
    skill_root = Path(skill_root)
    packages = _probe_all(skill_root)
    manifest = {
        "packages": packages,
        "required_missing": [p["name"] for p in packages
                             if p["tier"] == "required"
                             and not p["import_ok"]],
        "optional_available": [p["name"] for p in packages
                               if p["tier"] == "optional"
                               and p["import_ok"]],
    }
    # v14:trial 子进程解释器单独探一遍(_probe 的 importlib 跑在 CLI
    # 解释器里,而 trial 经 scheduler.run_trial_process 另起进程;
    # 两者不是同一个 python 时,preflight 全绿而 baseline 全灭——
    # 实测事故:10 个 baseline trial 全部 ModuleNotFoundError)
    manifest["trial_interpreter"] = probe_trial_interpreter()
    # v2.12.2.9:GBM 后端 CUDA 编译位(把"能不能上卡"变成可查事实)
    manifest["gbm_backends"] = probe_gbm_backends(
        manifest["trial_interpreter"].get("interpreter"))
    manifest["required_missing"] += [
        f"{m}@trial-interpreter"
        for m in manifest["trial_interpreter"].get("missing") or []]
    # 环境指纹(v20):包状态(含 absent)+ 解释器身份,漂移只记录不阻塞
    manifest["interpreter"] = interpreter_identity(manifest["trial_interpreter"])
    manifest["signature"] = env_signature(packages, manifest["interpreter"])
    # v20 前向兼容:env_signature 即原 fingerprint 位的继任者。保留同名键
    # 让旧消费方(run-state.env_fingerprint)与旧值格式的归因路径一致。
    manifest["fingerprint"] = manifest["signature"]
    if output_path is not None:
        dump_json(Path(output_path), manifest)
    return manifest


def probe_trial_interpreter(python_exe: str | None = None) -> dict:
    """在**子进程**里探 required 依赖(与 runner._execute_once 同口径)。

    _probe 的 importlib 跑在 CLI 解释器里,而 trial 是另起进程跑的
    (scheduler.run_trial_process,命令首位的 `python` 会被替换为
    sys.executable)。两者不是同一个 python 时(PATH 里的 python 无
    pandas),preflight 全绿而 10 个 baseline trial 全部
    ModuleNotFoundError —— 实测事故。这里按 runner 的实际口径
    (sys.executable)再探一次。

    v20:顺带回传子进程的 python 版本,供 interpreter_identity 入签名。
    """
    import subprocess

    exe = python_exe or sys.executable
    code = ("import json, sys, importlib.util as u; "
            "print(json.dumps({'v': '%d.%d.%d' % sys.version_info[:3], "
            "'m': {m: u.find_spec(m) is not None for m in "
            "['pandas','numpy','sklearn','yaml','lightgbm','pyarrow']}}))")
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True,
                           text=True, timeout=60)
        payload = json.loads(r.stdout.strip() or "{}")
        found = payload.get("m") or {}
        pv = payload.get("v")
    except Exception as exc:  # noqa: BLE001
        return {"interpreter": exe, "probe_ok": False,
                "python_version": None,
                "error": str(exc)[:120], "missing": ["<探测失败>"]}
    return {"interpreter": exe, "probe_ok": True, "python_version": pv,
            "missing": sorted(m for m, ok in found.items() if not ok)}


def _pkg_state(packages: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """[{name,tier,import_ok,version}] → ({name: state}, {name: tier})。

    absent 也是一个状态位(v20):缺失的 required 包装上时,旧实现因
    签名里根本没有它而「漂移」,新实现能认出这是**修复**而非版本变化。
    """
    states: dict[str, str] = {}
    tiers: dict[str, str] = {}
    for p in packages:
        name = p["name"]
        states[name] = (("v:" + str(p["version"]))
                        if p.get("import_ok") else "absent")
        tiers[name] = p.get("tier") or "required"
    return states, tiers


def classify_env_drift(recorded: dict | None,
                       current: dict) -> dict:
    """环境变化结构化归因(v20);纯函数,不碰盘。

    仿 `runtime/harness.py::classify_harness_change` 的形态。返回
    `{changed, kinds, events}`:
    - kinds 为去重后的分类名(见 DRIFT_KINDS);
    - events 逐条 `{kind, name, from, to}` 供 ledger 落盘。

    **关键区分**:`required_appeared`(缺失→已装)是**修复动作**,
    `required_version_change` 才是真正让历史 trial 不可复现的变化。
    旧实现把两者混在同一个哈希里,导致「按 preflight 提示装好缺失
    依赖 = 整轮运行判死」。

    recorded 为空(旧 run 无记录)时返回 unknown=True + 全量事件,
    调用方按「宁可误报」处理。
    """
    rec_pkgs = (recorded or {}).get("packages")
    if not rec_pkgs:
        # v20 之前的 run:run-state 只有 16 位指纹串,没有逐包快照 ——
        # 无从归因,记一条 unknown_baseline 说明「此 run 的环境基线
        # 早于逐包台账」,而不是伪造成某个具体包的变化。
        return {"unknown": True, "changed": True,
                "kinds": ["unknown_baseline"],
                "events": [{"kind": "unknown_baseline",
                            "name": "packages",
                            "from": "pre-v20(仅指纹串)",
                            "to": "逐包快照已建立"}]}
    before, tier_of = _pkg_state(rec_pkgs)
    after, tier_after = _pkg_state(current.get("packages") or [])
    tier_of.update(tier_after)
    events: list[dict] = []
    kinds: list[str] = []
    for name in sorted(set(before) | set(after)):
        b, a = before.get(name), after.get(name)
        if b == a:
            continue
        if b is not None and a is None:
            continue                       # 被移出声明表,不算环境事件
        # 声明表新增的探测项(两侧都无记录)→ 按 optional 装配记账
        is_required = tier_of.get(name) == "required"
        if b is None:
            kind = "optional_change"
        elif b == "absent" and a != "absent":
            kind = "required_appeared" if is_required else "optional_change"
        elif b != "absent" and a == "absent":
            kind = "required_disappeared" if is_required else "optional_change"
        elif is_required:
            kind = "required_version_change"
        else:
            kind = "optional_change"
        events.append({"kind": kind, "name": name, "from": b, "to": a})
        if kind not in kinds:
            kinds.append(kind)
    # 解释器身份:路径或版本任一变化即记(两个解释器版本恰好相同、
    # 只是换了 venv 的情形,旧指纹一个字都不变)
    rb = (recorded or {}).get("interpreter") or {}
    ra = current.get("interpreter") or {}
    if rb and ra and rb != ra:
        events.append({"kind": "interpreter_change", "name": "interpreter",
                       "from": rb, "to": ra})
        kinds.append("interpreter_change")
    return {"unknown": False, "changed": bool(events),
            "kinds": kinds, "events": events}


def check_env_drift(run_state: dict, skill_root: Path) -> dict | None:
    """BASELINED 后的环境变化归因(v20);无变化返回 None。

    返回 `{recorded, current, unknown, kinds, events}`。**不再自带
    「需重新 baseline」措辞** —— 环境变化由调用方记录到 env-ledger,
    不阻塞运行(v20:铁律 10 的环境侧不判死;harness 与泄漏分类仍判死)。
    """
    recorded = run_state.get("env_manifest") or {}
    if not recorded.get("packages"):
        # 只有指纹串(旧 run):无从逐包归因,退化成整体比对
        fp_rec = run_state.get("env_fingerprint")
        if not fp_rec:
            return None
        cur_fp = run_preflight(Path(skill_root)).get("fingerprint")
        if cur_fp == fp_rec:
            return None
        return {"recorded": fp_rec, "current": cur_fp, "unknown": True,
                "kinds": ["unknown_baseline"],
                "events": [{"kind": "unknown_baseline",
                            "name": "fingerprint",
                            "from": fp_rec, "to": cur_fp}]}
    current = run_preflight(Path(skill_root))
    diff = classify_env_drift(recorded, current)
    if not diff["changed"]:
        return None
    return {"recorded": recorded.get("fingerprint"),
            "current": current.get("fingerprint"),
            **diff}


def record_env_events(run_dir: Path, events: list[dict], *,
                      phase: str | None = None,
                      generation: int | None = None) -> Path | None:
    """环境事件追加到 run 目录 `env-ledger.jsonl`(v20)。

    与 delete-log.jsonl 同层同样式:一行一条 JSON,谁在哪个阶段因为
    什么改了环境,复盘可见。**台账是记录,不是许可** —— 不改变任何
    控制流,写失败也不影响调用方(v20:环境变化永不阻塞运行)。
    """
    if not events:
        return None
    path = Path(run_dir) / ENV_LEDGER
    for ev in events:
        append_jsonl(path, {"at": now_iso(), "phase": phase,
                            "generation": generation, **ev})
    return path


# ---------------------------------------------------------------------------
# v14/v21:四产物自消费冒烟(profile 后强制;视图/抽样面后手动)
# ---------------------------------------------------------------------------

#: 冒烟读取行数(只验"能不能读通",不做统计)。
#:
#: v2.12.2.8:1000 → 4096。旧值**低于** pandas 递归解析的爆栈阈值
#: (实测 1978 行起 `pd.to_numeric(errors="coerce")` 在长非数字 object
#: 列上段错误),于是"冒烟通过"对真实数据规模毫无预测力 —— 实测
#: req-cnt-t7:1000 行冒烟一路绿灯,44638 行的 evaluation 段 SIGSEGV,
#: 还伴随 16GB core dump。4096 > 阈值,让这类崩溃在几秒级的冒烟里
#: 就暴露,而不是烧掉一代算力。
SMOKE_ROWS = 4096


def check_data_products(run_dir: Path, spec, *,
                        splits_dir: str = "sample-profile/splits",
                        n_rows: int = SMOKE_ROWS) -> list[str]:
    """四产物结构冒烟(v14/v21):harness 能否吃下自己刚写出的东西。

    v13 数据面物理隔离上线后,capacity / runner / gen_executor /
    baseline / segment 五处消费方各自 KeyError 一次才被逐个手补(见
    memory/v13-splits-labels-fix)—— 因为 write_splits 只断言了 banned
    列不在产物里(_assert_view_clean),没有任何一处验证"下游拿得到
    它要的东西"。返回违规清单(空 = 通过)。
    """
    import pandas as pd
    import pyarrow.parquet as pq

    from .data import (IDENTITY_SUBDIR, LABELS_SUBDIR, REPARSE_SUBDIR,
                       SAMPLE_KEY_COLUMN, assert_face_clean)

    base = Path(run_dir) / splits_dir
    errs: list[str] = []
    # 1) 特征面无 banned 列(复用既有 schema 级断言)
    errs += assert_face_clean(base, spec)

    for name in ("train", "validation", "shadow_test", "final_test"):
        face_p = base / f"{name}.parquet"
        if not face_p.exists():
            continue
        identity_p = base / IDENTITY_SUBDIR / f"{name}.parquet"
        if not identity_p.exists():
            errs.append(
                f"{name}: 缺无标签对齐面 {IDENTITY_SUBDIR}/{name}.parquet")
        else:
            identity_names = set(pq.read_schema(identity_p).names)
            if SAMPLE_KEY_COLUMN not in identity_names:
                errs.append(f"{name}: identity 面缺 {SAMPLE_KEY_COLUMN}")
            if pq.read_metadata(identity_p).num_rows != \
                    pq.read_metadata(face_p).num_rows:
                errs.append(f"{name}: identity 面与特征面行数不一致")
        lab_p = base / LABELS_SUBDIR / f"{name}.parquet"
        # v21:final_test 标签是 finalizer-only capability，搜索目录中
        # 存在该文件反而是严重协议违规。
        if name == "final_test":
            if lab_p.exists():
                errs.append(
                    "final_test: 标签面不应物化在搜索目录"
                    f"({LABELS_SUBDIR}/final_test.parquet);请重新 profile")
            continue
        # 2) 标签面存在且含目标列
        if not lab_p.exists():
            errs.append(f"{name}: 缺标签面 {LABELS_SUBDIR}/{name}.parquet")
            continue
        lab_names = set(pq.read_schema(lab_p).names)
        if spec.target_name not in lab_names:
            errs.append(f"{name}: 标签面无目标列 {spec.target_name}"
                        f"(实有 {sorted(lab_names)[:8]})")
        if spec.treatment_column and spec.treatment_column not in lab_names:
            errs.append(f"{name}: 标签面无 treatment 列 "
                        f"{spec.treatment_column}")
        # 3) 行数一致 —— 下游 join 是按位置对齐(labels.set_index(
        #    frame.index)),行数不等即错位且**不报错**,只出错数字
        n_face = pq.read_metadata(face_p).num_rows
        n_lab = pq.read_metadata(lab_p).num_rows
        if n_face != n_lab:
            errs.append(f"{name}: 特征面 {n_face} 行 ≠ 标签面 {n_lab} 行"
                        "(按位置对齐 → 标签会接到错的行上)")
            continue
        # 4) 行序对齐:v13 下特征面与标签面**无共享列**(时间/ID 只在
        #    标签面),无法直接交叉比对——改验标签面自身的时间序:
        #    write_splits 落盘前按 time_column 排序,时间列非单调
        #    即说明两产物之一被落盘后改动/错位(标签会接到错的行上)
        face_names = set(pq.read_schema(face_p).names)
        shared = [c for c in (spec.id_column, spec.time_column)
                  if c and c in lab_names and c in face_names]
        if shared:
            f = pd.read_parquet(face_p, columns=shared).head(n_rows)
            l = pd.read_parquet(lab_p, columns=shared).head(n_rows)
            if not f.equals(l):
                errs.append(f"{name}: 特征面与标签面前 {n_rows} 行的对齐键 "
                            f"{shared} 不一致(行序错位 = 标签接错行)")
        elif spec.time_column and spec.time_column in lab_names:
            tvals = pd.read_parquet(lab_p, columns=[spec.time_column])[
                spec.time_column].astype(str)
            if len(tvals) > 2 and not tvals.is_monotonic_increasing:
                errs.append(
                    f"{name}: 标签面时间列 {spec.time_column} 非单调"
                    "(write_splits 落盘前按时间排序;非单调说明标签面"
                    "与特征面行序已错位 → 标签接错行,静默出错数字)")
        # 5) reparse 目录当且仅当 spec 声明了 reparse 列时存在
        rep_p = base / REPARSE_SUBDIR / f"{name}.parquet"
        if spec.reparse_columns and not rep_p.exists():
            errs.append(f"{name}: spec 声明 reparse 列 "
                        f"{spec.reparse_columns} 但无 {REPARSE_SUBDIR}/")
        if rep_p.exists():
            if not spec.reparse_columns:
                errs.append(f"{name}: spec 无 reparse 列却落了 "
                            f"{REPARSE_SUBDIR}/{name}.parquet")
            else:
                rep_names = set(pq.read_schema(rep_p).names)
                miss = [c for c in spec.reparse_columns
                        if c not in rep_names]
                if miss:
                    errs.append(f"{name}: reparse 源缺列 {miss}")
                if pq.read_metadata(rep_p).num_rows != n_face:
                    errs.append(f"{name}: reparse 源行数与特征面不等"
                                "(解析产物会与特征面错位)")
    # 6) [FIX-2026-09-18] 代码生成模板自洽性(渲染 + AST 未定义名)。
    #    历史事故:runtime/baseline.py 的 BASELINE_TEMPLATE 里 dtype 治理
    #    段调用 safe_to_numeric,而模板体既未定义也未 import 它。模板文件
    #    自身 import 正常、既有测试全绿(被调用名只出现在字符串里),缺陷
    #    只在 gen-00 跑起来时炸:两个 trial 各 3 秒 NameError,且按协议
    #    失败 trial 不可删除,永久留在 leaderboard 上污染审计面。
    #    这里在 preflight 阶段渲染一遍再做 AST 扫描,挡在起跑线之前。
    errs.extend(check_generated_templates())
    return errs


def check_generated_templates() -> list[str]:
    """渲染 baseline/trial 模板并做未定义名审计(见 code_audit)。"""
    from .baseline import BASELINE_TEMPLATE, ESTIMATOR_CODE, FIT_CODE
    from .code_audit import check_generated_code

    errs: list[str] = []
    for algorithm in sorted(ESTIMATOR_CODE):
        try:
            code = BASELINE_TEMPLATE.format(
                algorithm=algorithm, target="__target__",
                model_type="binary_classification", treatment="", seed=0,
                n_estimators_cap=2000, early_stopping_rounds=50,
                es_fraction=0.1,
                estimator_code=ESTIMATOR_CODE[algorithm],
                fit_code=FIT_CODE[algorithm],
            )
        except Exception as exc:  # noqa: BLE001
            errs.append(f"BASELINE_TEMPLATE[{algorithm}] 渲染失败: {exc}")
            continue
        for problem in check_generated_code(code):
            errs.append(
                f"BASELINE_TEMPLATE[{algorithm}] 生成代码不自洽: {problem}"
                "(模板缺少对应定义/import;这类缺陷只会在 trial 真正跑起来"
                "时才炸,且失败 trial 不可删除)")
    return errs


def check_consumers(run_dir: Path, spec, *,
                    splits_dir: str = "sample-profile/splits",
                    n_rows: int = SMOKE_ROWS) -> list[str]:
    """下游消费方取数路径冒烟:每个消费方按它**自己的**读法取一遍。

    只验"取得到 + dtype 能进模型",不验数值。任何一个 KeyError /
    LightGBMError 在这里暴露代价是几秒;在 trial 里暴露代价是一代算力。
    """
    import numpy as np
    import pandas as pd

    from .data import (load_labels, load_split_frames,
                       resolve_feature_columns)

    base = Path(run_dir) / splits_dir
    errs: list[str] = []
    # (a) runner._evaluate_trial / gen_executor 口径:特征面 + 标签面
    try:
        frames = load_split_frames(run_dir, splits_dir)
        labs = load_labels(run_dir, splits_dir)
        if "validation" not in frames or "validation" not in labs:
            errs.append("evaluator: validation 特征面/标签面缺失")
            return errs
        va = frames["validation"].head(n_rows)
        y = labs["validation"][spec.target_name].values[:n_rows]
        feats = resolve_feature_columns(frames["train"].head(n_rows), spec)
    except Exception as exc:  # noqa: BLE001
        errs.append(f"evaluator 取数路径失败: {exc}")
        return errs
    if len(y) != len(va):
        errs.append("evaluator: 标签与特征面长度不等")
    # (b) capacity._prep:诊断侧 dtype 治理必须能吃下特征面
    try:
        from .capacity import _prep

        _prep(frames["train"].head(n_rows)[feats])
    except Exception as exc:  # noqa: BLE001
        errs.append(f"capacity 取数/dtype 治理失败: {exc}")
    # (c) group_importance:原始帧的 object 列直接进 LightGBM 是历史
    #     事故(p5_can_use_total_amt: object);可数值化列经 _coerce_numeric
    #     已安全,这里只报**既非数值也无法数值化**的真类别列(低基数
    #     类别在训练侧走 one-hot,特征名在 feature_columns.json 里有
    #     对应 dummies;permutation 在可用列交集上裁决,属预期降覆盖)
    #
    #     v2.12.2.8:判据换 safe_to_numeric。旧实现
    #     `pd.to_numeric(...).isna().all()` 在**长非数字 object 列**上
    #     会段错误 —— 冒烟检查自己就是崩溃点,profile 无法落 PROFILED
    #     (实测 req-cnt-t7 三次 profile 全部卡在这里,产物只有提案面)。
    try:
        from .safe_dtype import safe_to_numeric

        bad = [c for c in feats
               if not pd.api.types.is_numeric_dtype(va[c])
               and safe_to_numeric(va[c]) is None
               and va[c].nunique(dropna=True) > 100]
        if bad:
            errs.append(f"group_importance: {len(bad)} 列既非数值也无法"
                        f"数值化(如 {bad[:3]}),置换打分会崩")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"group_importance 预检失败: {exc}")
    # (d) error_analysis:用伪分数验签名与分桶路径
    try:
        from .error_analysis import build_error_analysis

        rng = np.random.default_rng(0)
        build_error_analysis(y, rng.random(len(y)), None, va,
                             model_type=spec.model_type, n_bins=5,
                             label="__smoke__")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"error_analysis 取数路径失败: {exc}")
    del base
    return errs
