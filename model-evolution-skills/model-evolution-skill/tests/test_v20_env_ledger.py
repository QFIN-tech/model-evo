"""v20 环境台账回归测试(铁律 10 环境侧:判死 → 记录 + 归因)。

事故来源:2026-09 会话核实的三处失配(详见 docs/MAINTENANCE.md §6)。
本测试锁定 v20 的修复面:

1. **旧指纹不含缺失包** —— `if p["import_ok"]` 使 absent 的 required
   包不在签名里,而 cmd_preflight 先写指纹再报 missing。用户按提示
   装好缺失的包 → 签名「没有」变「有」→ 漂移 → 之后所有 run 全炸。
   **修好环境 = 作废整轮运行。** v20 把 required_appeared 单列一类。
2. **解释器不在指纹里** —— trial 是另起进程跑的(v14 实测「CLI 全绿、
   10 个 baseline 全灭」)。换 venv 而版本恰好相同即指纹一致、无告警。
3. **run/diagnose/preflight 三处判死 + 全零留痕** —— run 直接 return 1;
   diagnose 记 critical 并 exit 1;preflight 的无条件静默覆盖指纹是
   官方的手改通道,无 reason 无留痕。v20 三处全改为记 env-ledger.jsonl。

另锁定:装 optional 包记 optional_change(不是违规)、ledger 幂等、
旧 run(只有指纹串)归因为 unknown_baseline。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from _common import SKILL, cli_source  # noqa: E402

_sys.path.insert(0, str(SKILL))

from runtime.preflight import (  # noqa: E402
    OPTIONAL_PACKAGES, check_env_drift, classify_env_drift, env_signature,
    interpreter_identity, record_env_events, run_preflight)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def _pk(name: str, version, tier: str = "required") -> dict:
    return {"name": name, "import_ok": version is not None,
            "version": version, "tier": tier}


def _snap(pkgs: list[dict], cli: str = "/a/python",
          cli_py: str = "3.9.16") -> dict:
    interp = interpreter_identity({"interpreter": cli, "python_version": cli_py})
    return {"packages": pkgs, "interpreter": interp,
            "fingerprint": env_signature(pkgs, interp)}


# ---------------------------------------------------------------------------
# 1. 指纹构造:absent 占位 + 解释器入签
# ---------------------------------------------------------------------------
def test_signature_covers_absent() -> None:
    print("== 1. 指纹含缺失包与解释器(旧实现两个都不含)==")
    present = [_pk("lightgbm", "4.6.0"), _pk("pandas", "2.3.3")]
    missing = [_pk("lightgbm", None), _pk("pandas", "2.3.3")]
    i = interpreter_identity({"interpreter": "/a/python",
                              "python_version": "3.9.16"})
    check("缺失 required 改变签名(修复动作可被识别)",
          env_signature(present, i) != env_signature(missing, i),
          f"{env_signature(present, i)} vs {env_signature(missing, i)}")
    # 旧实现:缺失包不进签名 → 装上后签名不变(这是"修好环境=作废整轮"
    # 的反面:其实旧实现是缺失时签名里没有它,装上后多出一个条目 →
    # 签名变了。这里钉住"变"的方向,并确认归因是 appeared 而非版本变化)
    i2 = interpreter_identity({"interpreter": "/b/python",
                               "python_version": "3.9.16"})
    check("解释器路径变化改变签名(版本号可以完全相同)",
          env_signature(present, i) != env_signature(present, i2))
    i3 = interpreter_identity({"interpreter": "/a/python",
                               "python_version": "3.10.0"})
    check("Python 版本变化改变签名",
          env_signature(present, i) != env_signature(present, i3))
    # 签名必须对包顺序不敏感(稳定排序)
    check("签名与包声明顺序无关",
          env_signature(list(reversed(present)), i)
          == env_signature(present, i))


def test_probe_dedup_and_version() -> None:
    print("== 2. 声明表去重 + 版本回退 ==")
    m = run_preflight(SKILL)
    names = [p["name"] for p in m["packages"]]
    check("无重复探测项(openpyxl 曾在 required/optional 各一份)",
          len(names) == len(set(names)),
          f"{len(names)} 项 / {len(set(names))} 唯一")
    check("openpyxl 只有一份且归 required",
          names.count("openpyxl") == 1
          and next(p["tier"] for p in m["packages"]
                   if p["name"] == "openpyxl") == "required")
    check("shap 不在探测表(全仓零消费方)",
          "shap" not in names and "shap" not in OPTIONAL_PACKAGES)
    # pynvml 无 __version__,旧实现记 None → 签名成 '?' 测不出版本
    pyn = next((p for p in m["packages"] if p["name"] == "pynvml"), None)
    if pyn and pyn["import_ok"]:
        check("pynvml 版本经 importlib.metadata 回退取得(旧为 None)",
              pyn["version"] not in (None, "?"), str(pyn["version"]))
    else:
        print("  [SKIP] pynvml 未安装")
    check("manifest 带 interpreter 身份块",
          isinstance(m.get("interpreter"), dict)
          and m["interpreter"].get("cli_executable"))
    check("signature 与 fingerprint 一致(前向兼容旧消费方)",
          m["signature"] == m["fingerprint"])


# ---------------------------------------------------------------------------
# 2. classify_env_drift:四类 kind 与"修复 ≠ 漂移"
# ---------------------------------------------------------------------------
def test_classify_kinds() -> None:
    print("== 3. classify_env_drift 分类 ==")
    base = _snap([_pk("lightgbm", "4.6.0"), _pk("pandas", "2.3.3"),
                  _pk("catboost", None, "optional")])
    check("无变化 → changed=False", not classify_env_drift(base, base)["changed"])

    v = _snap([_pk("lightgbm", "4.7.0"), _pk("pandas", "2.3.3"),
               _pk("catboost", None, "optional")])
    d = classify_env_drift(base, v)
    check("required 版本变化 → required_version_change",
          d["kinds"] == ["required_version_change"], str(d["kinds"]))

    # 事故 1 的正面面:缺失包被装上,必须是 appeared(修复)不是版本变化
    before = _snap([_pk("lightgbm", None), _pk("pandas", "2.3.3")])
    after = _snap([_pk("lightgbm", "4.6.0"), _pk("pandas", "2.3.3")])
    d = classify_env_drift(before, after)
    check("缺失 required 装上 → required_appeared(修复动作)",
          d["kinds"] == ["required_appeared"], str(d["kinds"]))
    check("appeared 事件的 from 是 absent",
          d["events"][0]["from"] == "absent"
          and d["events"][0]["to"] == "v:4.6.0")

    opt = _snap([_pk("lightgbm", "4.6.0"), _pk("pandas", "2.3.3"),
                 _pk("catboost", "1.2.8", "optional")])
    d = classify_env_drift(base, opt)
    check("optional 包装上 → optional_change(不是违规)",
          d["kinds"] == ["optional_change"], str(d["kinds"]))

    interp = dict(base)
    interp["interpreter"] = interpreter_identity(
        {"interpreter": "/b/python", "python_version": "3.9.16"})
    interp["fingerprint"] = env_signature(interp["packages"],
                                          interp["interpreter"])
    d = classify_env_drift(base, interp)
    check("换 venv 而版本号相同 → interpreter_change",
          "interpreter_change" in d["kinds"], str(d["kinds"]))

    d = classify_env_drift(None, base)
    check("旧 run(无逐包快照)→ unknown_baseline",
          d["unknown"] and d["kinds"] == ["unknown_baseline"])


# ---------------------------------------------------------------------------
# 3. ledger 落盘:记录 + 幂等 + 不抛
# ---------------------------------------------------------------------------
def test_ledger_recording() -> None:
    print("== 4. env-ledger.jsonl 记账 ==")
    td = pathlib.Path(tempfile.mkdtemp())
    try:
        p = record_env_events(
            td, [{"kind": "optional_change", "name": "catboost",
                  "from": "absent", "to": "v:1.2.8"}],
            phase="EVOLVING", generation=3)
        check("台账落在 run 目录 env-ledger.jsonl",
              p is not None and p.name == "env-ledger.jsonl" and p.exists())
        rows = [json.loads(x) for x in
                p.read_text(encoding="utf-8").splitlines() if x.strip()]
        check("一条事件一行 JSON,带 phase/generation/at",
              len(rows) == 1 and rows[0]["phase"] == "EVOLVING"
              and rows[0]["generation"] == 3 and rows[0].get("at")
              and rows[0]["kind"] == "optional_change")
        check("空事件列表不建文件",
              record_env_events(td, []) is None)
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. check_env_drift 的返回形态与控制流(不再判死)
# ---------------------------------------------------------------------------
def test_no_death_paths() -> None:
    print("== 5. 三处判死已移除 ==")
    src = cli_source()
    # run 闸门段落:同一段里不得再出现 `return 1`
    anchor = src.index("环境变化记账(v20")
    seg = src[anchor:anchor + 2200]
    check("run 环境段无 return 1(不再判死)", "return 1" not in seg)
    check("run 环境段写 ledger",
          "record_env_events" in seg and "env-ledger" in seg)
    check("run 环境段刷新基线(防同一次变化重复记账)",
          "env_fingerprint=cur_fp" in seg)
    # trial 盖 env 戳
    check("trial 记录带 env_fingerprint",
          'trial.data["env_fingerprint"]' in src)
    # diagnose:severity 非 critical
    dseg = src[src.index('"code": "env_fingerprint_drift"'):][:600]
    check("diagnose 的 env 漂移 severity=info(不再 exit 1)",
          '"severity": "info"' in dseg and '"severity": "critical"' not in dseg)
    check("diagnose remedy 指向 ledger 而非重新 baseline",
          "env-ledger.jsonl" in dseg and "重新 baseline" not in dseg)
    # preflight:覆盖前记账
    pseg = src[src.index("def cmd_preflight"):]
    pseg = pseg[:pseg.index("def cmd_acknowledge_harness_change")]
    check("cmd_preflight 覆盖指纹前写 ledger",
          "classify_env_drift" in pseg and "record_env_events" in pseg)
    check("required 缺失的措辞不再说「不得自行安装」",
          "不得自行安装" not in pseg)
    # diagnose 的 exit 1 判定仍认 critical(harness/leak/gate 不受影响)
    check("diagnose 仍按 critical 判 exit 1",
          'any(f.get("severity") == "critical"' in src)


def test_drift_ok_still_reported() -> None:
    print("== 6. 无变化时可观测(不再「静默 = 没检查」)==")
    src = cli_source()
    check("diagnose 无漂移时输出 env_fingerprint_ok",
          '"env_fingerprint_ok"' in src)
    check("diagnose 输出带 kinds_meaning 人话解释",
          "kinds_meaning" in src and "DRIFT_KINDS" in src)


def main() -> int:
    for t in (test_signature_covers_absent, test_probe_dedup_and_version,
              test_classify_kinds, test_ledger_recording,
              test_no_death_paths, test_drift_ok_still_reported):
        t()
        print()
    print("=" * 60)
    print(f"v20 env ledger: {len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
