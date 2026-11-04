"""人工批准命令:approve-topology(v2.16.0 模型拓扑门)。

trial 默认单模型;级联 / 融合 / 白名单外的结构型组合必须由**用户本人**
批准。本命令是写入 approvals/model-topology.jsonl 的唯一入口:

- 必须在交互式终端执行(stdin/stdout 都是 TTY)——Agent 会话的 Bash 不是
  TTY;guard hook 另外拒绝 Agent 调用本命令或改写账本;
- 展示的是 Harness 检测到的事实(信号、代码 sha),不是 Agent 的描述;
- 需要键入屏幕上的随机确认码;
- 批准绑定到检测签名(trial 范围还绑定代码 sha),改码即失效。

`--list` 只读,不需要终端。
"""
from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

from .context import _load_spec, _run_dir_from, logger


def _interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def _confirm(prompt_head: str) -> bool:
    code = secrets.token_hex(3).upper()
    print(prompt_head)
    try:
        typed = input(f"请输入确认码 {code} 以继续(其它输入取消): ").strip()
    except EOFError:
        return False
    return typed.upper() == code


def _list(run_dir: Path) -> int:
    from runtime.topology import REQUESTS_REL, active_approvals, load_ledger

    led = load_ledger(run_dir)
    active, _ = active_approvals(run_dir)
    print(f"账本: {led['path']}  链完整: {led['chain_valid']}"
          + (f"({led['error']})" if led["error"] else ""))
    for r in led["records"]:
        tag = "撤销" if r.get("action") == "revoke" else (
            "有效" if r in active else "已撤销")
        print(f"  [{tag}] {r.get('approval_id') or '-'} seq={r.get('seq')} "
              f"scope={r.get('scope')} trial={r.get('trial_id')} "
              f"kinds={r.get('kinds')} by={r.get('approver')}@"
              f"{r.get('host')} at={r.get('at')} "
              f"{'revokes=' + str(r.get('revokes')) if r.get('revokes') else ''}"
              f" reason={r.get('reason')}")
    req_dir = run_dir / REQUESTS_REL
    reqs = sorted(req_dir.glob("*.json")) if req_dir.exists() else []
    if reqs:
        print("待批准请求:")
        for p in reqs:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            g = d.get("gate") or {}
            print(f"  {d.get('trial_id')}: {g.get('kind')} needs="
                  f"{g.get('needs')} at={d.get('requested_at')}")
    return 0


def cmd_approve_topology(args) -> int:
    from runtime.topology import (APPROVAL_SCOPES, REQUESTS_REL, append_ledger,
                                  describe_for_human, env_identity,
                                  evaluate_gate)
    from runtime.trial import TrialRecord

    run_dir = _run_dir_from(args)
    if getattr(args, "list", False):
        return _list(run_dir)
    if not _interactive():
        logger.error(
            "approve-topology 只能由用户在交互式终端执行(本进程 stdin/stdout"
            " 不是 TTY)。Agent 不得代为批准;请把 approvals/requests/ 或"
            " logs/user-decisions.md 中的请求转告用户。")
        return 2
    ident = env_identity()
    if getattr(args, "revoke", None):
        if not _confirm(f"撤销批准 {args.revoke}(run={run_dir})"):
            print("已取消")
            return 1
        reason = args.reason or input("撤销理由: ").strip()
        rec = append_ledger(run_dir, {"action": "revoke",
                                      "revokes": args.revoke,
                                      "reason": reason, **ident})
        print(f"已撤销 {args.revoke}(seq={rec['seq']})")
        return 0

    # --trial 缺省 recipe:批准的是"这种结构",confirm/hpo 子代不必逐个再批
    scope = args.scope or ("recipe" if args.trial else None)
    if scope not in APPROVAL_SCOPES:
        logger.error("需要 --trial <id>(范围 trial/recipe)或 --scope run "
                     "--kind cascade[,ensemble,structural]")
        return 2
    record: dict = {"action": "approve", "scope": scope, **ident}
    if scope == "run":
        kinds = sorted({k.strip() for k in str(args.kind or "").split(",")
                        if k.strip()})
        bad = [k for k in kinds if k not in ("cascade", "ensemble",
                                             "structural")]
        if not kinds or bad:
            logger.error("--scope run 需要 --kind,取值 cascade/ensemble/"
                         "structural(逗号分隔),非法: %s", bad)
            return 2
        head = (f"批准整个 run 内的 {kinds} 拓扑(run={run_dir})。\n"
                "该范围对本 run 后续所有 trial 生效,直到撤销。")
        record["kinds"] = kinds
    else:
        if not args.trial:
            logger.error("--scope %s 需要 --trial <id>", scope)
            return 2
        tdir = run_dir / "trials" / args.trial
        if not (tdir / "trial.yaml").exists():
            logger.error("Trial 不存在: %s", tdir)
            return 1
        spec = _load_spec(args)
        trial = TrialRecord.load(tdir)
        gate = evaluate_gate(run_dir, trial, spec.model_type)
        if not gate.get("needs"):
            print(f"{trial.trial_id} 检测为 {gate.get('kind')},"
                  "无需批准(单模型或白名单内结构型)")
            return 0
        hyp = (trial.data.get("hypothesis") or {}).get("statement", "")
        head = (f"Trial {trial.trial_id}\n假设: {hyp}\n"
                f"代码: {tdir / 'code'}\n" + describe_for_human(gate)
                + ("\n范围 recipe:同时覆盖其后代(签名不超出本次检测)"
                   if scope == "recipe" else
                   "\n范围 trial:仅本 trial,改码即失效"))
        record.update({"trial_id": trial.trial_id,
                       "code_sha256": gate["code_sha256"],
                       "signature": gate["signature"],
                       "kinds": sorted({n.split(":")[0]
                                        for n in gate["needs"]}),
                       "needs": gate["needs"]})
    if not _confirm(head):
        print("已取消")
        return 1
    reason = args.reason or input("批准理由(写入账本): ").strip()
    if not reason:
        print("理由不能为空,已取消")
        return 1
    record["reason"] = reason
    rec = append_ledger(run_dir, record)
    print(f"已批准 {rec['approval_id']}(scope={scope})")
    if record.get("trial_id"):
        req = run_dir / REQUESTS_REL / f"{record['trial_id']}.json"
        if req.exists():
            req.rename(req.with_suffix(f".{rec['approval_id']}.done"))
    ud = run_dir / "logs" / "user-decisions.md"
    ud.parent.mkdir(parents=True, exist_ok=True)
    with open(ud, "a", encoding="utf-8") as f:
        f.write(f"\n## [APPROVED] {rec['approval_id']} 模型拓扑 "
                f"{record.get('trial_id') or 'run'} {record.get('kinds')}\n"
                f"- 批准人:{rec.get('approver')}@{rec.get('host')},"
                f"{rec.get('at')}\n- 理由:{reason}\n")
    return 0
