"""v2.18.3 修复批次:经验库解析 + profile 重跑。

三个复现自 run4 req-t7 run1(2026-10-29)的缺陷,每个用例对应一条:

1. **A(`test_library_resolution_*`)**:run 级 ``modelevo-config.yaml`` 的
   ``library.*`` 被静默忽略。旧版 ``library_view.client_settings(None)``
   退回 ``Config.load()``(只读 skill 级),于是把 run config 的 uri 指到不可达
   地址后,profile 仍连上真实服务、固化真实库视图(watermark 243)、baseline
   照旧创建 warm-start trial。另外旧版**没有**主动关闭经验库的开关:服务不可达
   会退到本机缓存(stale),禁用意图无处表达。

2. **B(`test_profile_reports_loading_phase`)**:``cmd_profile`` 在
   ``load_dataframe`` 之前没有任何输出(4M×1513、24.6 GB 分片约 4 min),被
   误判成卡死并 kill,白跑一次完整 profile。

3. **C(`test_splits_reusable_*`)**:``leak_confirmation`` 按协议"只关闭确认门、
   不改数据面",旧版仍无条件 ``write_splits``,把 20 GB 四平面原样重写
   (402~406 s)。

运行:python tests/test_v2_18_3_library_and_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import tmp_run_dir  # noqa: E402,F401  hermetic import bootstrap

from runtime import library_view  # noqa: E402
from runtime.config import Config  # noqa: E402
from runtime.data import split_data, splits_reusable, write_splits  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _spec(**features) -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v2183", "task_name": "v2183",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02",
                         "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "features": dict(features),
        "data": {},
    })


def _df(n: int = 16) -> pd.DataFrame:
    return pd.DataFrame({
        "dt": [f"2026-0{m}-01" for m in range(1, 5) for _ in range(n // 4)],
        "uid": list(range(n)), "x": np.arange(n),
        "label": [0, 1, 0, 1] * (n // 4),
    })


def _write_run_config(run_dir: Path, uri: str, *, cache: str | None = None,
                      enabled: str | None = None) -> None:
    lines = ["modelevo:", "  library:", f"    uri: {uri}"]
    if cache is not None:
        lines.append(f"    cache_dir: {cache}")
    if enabled is not None:
        lines.append(f"    enabled: {enabled}")
    (run_dir / "modelevo-config.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# A. run 级 library.* 必须生效
# ---------------------------------------------------------------------------

def test_library_resolution_uses_run_config():
    print("A1 run 级 library.uri 覆盖 skill 级(旧版被静默忽略):")
    run = tmp_run_dir("mevo-v2183-lib-")
    try:
        _write_run_config(run, "http://127.0.0.1:9",
                          cache=str(run / "empty-cache"))
        cfg = Config.load(run_config=run / "modelevo-config.yaml")
        check("Config 合并层本身是生效的",
              cfg.get("library.uri") == "http://127.0.0.1:9",
              str(cfg.get("library.uri")))
        s = library_view.client_settings(run_dir=run)
        check("client_settings(run_dir=...) 用 run 级 uri",
              s["uri"] == "http://127.0.0.1:9", str(s["uri"]))
        # tests/_common.py 默认设了 MODELEVO_LIBRARY_CACHE(env > 配置),
        # 验证 run 级 cache_dir 必须先摘掉它
        _old_cache = library_view.os.environ.pop("MODELEVO_LIBRARY_CACHE", None)
        try:
            s_cache = library_view.client_settings(run_dir=run)
        finally:
            if _old_cache is not None:
                library_view.os.environ["MODELEVO_LIBRARY_CACHE"] = _old_cache
        check("cache_dir 也随 run 级配置走",
              str(s_cache["cache_dir"]) == str(run / "empty-cache"),
              str(s_cache["cache_dir"]))
        # 显式 config 的形式(baseline/planner_context 的调用姿势)
        s2 = library_view.client_settings(cfg, run_dir=run)
        check("client_settings(config=...) 同样生效",
              s2["uri"] == "http://127.0.0.1:9", str(s2["uri"]))
    finally:
        import shutil

        shutil.rmtree(run, ignore_errors=True)


def test_library_resolution_env_wins():
    print("A2 环境变量优先级高于 run 级配置:")
    run = tmp_run_dir("mevo-v2183-lib-env-")
    try:
        _write_run_config(run, "http://127.0.0.1:9")
        old = library_view.os.environ.get("MODELEVO_LIBRARY_URI")
        library_view.os.environ["MODELEVO_LIBRARY_URI"] = "http://127.0.0.1:7"
        try:
            s = library_view.client_settings(run_dir=run)
            check("env 覆盖 run 配置",
                  s["uri"] == "http://127.0.0.1:7", str(s["uri"]))
        finally:
            if old is None:
                library_view.os.environ.pop("MODELEVO_LIBRARY_URI", None)
            else:
                library_view.os.environ["MODELEVO_LIBRARY_URI"] = old
    finally:
        import shutil

        shutil.rmtree(run, ignore_errors=True)


def test_library_disabled_is_explicit():
    print("A3 library.enabled=false 是显式禁用(不读缓存、不固化真实视图):")
    run = tmp_run_dir("mevo-v2183-lib-off-")
    try:
        _write_run_config(run, "http://127.0.0.1:9", enabled="false")
        s = library_view.client_settings(run_dir=run)
        check("enabled=false 解析为 False", s["enabled"] is False, str(s))
        # 造一份"缓存里有真实视图"的现场:若禁用无效就会读到它
        cache_ns = Path(s["cache_dir"]) / s["namespace"].replace("/", "__")
        view_dir = cache_ns / "views" / "atid-x"
        view_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "999.json").write_text(
            '{"schema": "modelevo.run-view/1", "watermark": 999, '
            '"principles": [{"principle_id": "p-leak", "claim": "x"}], '
            '"leaderboard": [{"recipe_id": "r-deadbeef"}]}', encoding="utf-8")
        (run / "run-state.json").write_text('{"phase": "INITED"}',
                                            encoding="utf-8")
        view = library_view.pin(run, atid="atid-x",
                                config=Config.load(run_config=run / "modelevo-config.yaml"))
        check("禁用时不读本机缓存(watermark 为空,未变成 999)",
              view.watermark is None, str(view.watermark))
        check("禁用时不带出任何原则/榜单 → baseline 不会建 warm trial",
              not view.principles and not view.leaderboard and not view.baselines)
        check("视图 reason=library_disabled",
              view.data.get("reason") == "library_disabled",
              str(view.data.get("reason")))
        import json as _json

        st = _json.loads((run / "run-state.json").read_text(encoding="utf-8"))
        check("run-state.library.source=disabled",
              (st.get("library") or {}).get("source") == "disabled",
              str(st.get("library")))
        check("ensure 复用时也认得 library_disabled(不会重新 pin)",
              library_view.ensure(run).data.get("reason") == "library_disabled")
    finally:
        import shutil

        shutil.rmtree(run, ignore_errors=True)


def test_library_enabled_by_default():
    print("A4 不写 enabled 时保持默认开启(不改变既有行为):")
    run = tmp_run_dir("mevo-v2183-lib-default-")
    try:
        _write_run_config(run, "http://127.0.0.1:9")
        s = library_view.client_settings(run_dir=run)
        check("缺省 enabled=True", s["enabled"] is True, str(s.get("enabled")))
    finally:
        import shutil

        shutil.rmtree(run, ignore_errors=True)


# ---------------------------------------------------------------------------
# B. profile 必须尽早输出
# ---------------------------------------------------------------------------

def test_profile_reports_loading_phase():
    print("B1 cmd_profile 在读取数据面前后有输出与 run-state 留痕:")
    src = (Path(__file__).resolve().parents[1] / "scripts" / "_mvo"
           / "cmd_startup.py").read_text(encoding="utf-8")
    body = src.split("def cmd_profile(args)", 1)[1].split("\ndef ", 1)[0]
    load_at = body.find("df = load_dataframe(")
    check("cmd_profile 里存在 load_dataframe", load_at > 0)
    before = body[:load_at]
    check("读取前打印进度",
          "开始读取数据面" in before, before[-200:])
    check("读取前把时刻写进 run-state",
          "profile_loading_started_at" in before)
    after = body[load_at:]
    check("读取后打印行数列数", "数据面已载入" in after, after[:200])
    check("打印带 flush=True(否则管道下不落地)",
          before.count("flush=True") >= 1)


# ---------------------------------------------------------------------------
# C. 数据面没变就不重写四平面
# ---------------------------------------------------------------------------

def test_splits_reusable_hit_and_reasons():
    print("C1 splits_reusable:命中/未命中的判定:")
    tmp = tmp_run_dir("mevo-v2183-reuse-")
    try:
        df = _df()
        spec = _spec()
        out = tmp / "splits"
        sidx = split_data(df, spec)
        write_splits(df, spec, out, sidx)
        ok, why = splits_reusable(out, spec, sidx, df=df)
        check("刚写完即可复用", ok, why)

        # 只有 leak_confirmation 变化(裁决后重跑 profile 的真实场景)
        spec_kept = _spec(leak_confirmation={"kept_ambiguous": ["x"]})
        ok2, why2 = splits_reusable(out, spec_kept, sidx, df=df)
        check("kept_ambiguous 变化不改数据面 → 仍可复用", ok2, why2)
        check("两档指纹确实不同(审计用)",
              spec.leak_manifest_hash != spec_kept.leak_manifest_hash)
        check("data_face_hash 保持一致",
              spec.data_face_hash == spec_kept.data_face_hash)

        # banned 变化 → 必须重写
        ok3, why3 = splits_reusable(
            out, _spec(banned=[{"column": "x", "reason": "future_window"}]),
            sidx, df=df)
        check("banned 变化 → 判为不可复用", not ok3, why3)
        check("原因里说明是数据面指纹", "数据面指纹" in why3, why3)

        # 产物被截断(sha 不符) → 必须重写
        face = out / "train.parquet"
        face.write_bytes(face.read_bytes()[:64])
        ok4, why4 = splits_reusable(out, spec, sidx, df=df)
        check("特征面 sha256 不符 → 不可复用", not ok4, why4)

        # 切分吸附的行数变了 → 必须重写
        write_splits(df, spec, out, sidx)
        df_bigger = pd.concat([df, _df().assign(uid=lambda d: d.uid + 100)],
                              ignore_index=True)
        sidx_big = split_data(df_bigger, spec)
        ok5, why5 = splits_reusable(out, spec, sidx_big, df=df_bigger)
        check("切分行数变化 → 不可复用", not ok5, why5)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


def test_profile_skips_rewrite_when_reused():
    print("C2 SampleProfile.run 在可复用时跳过 write_splits:")
    src = (Path(__file__).resolve().parents[1] / "runtime"
           / "profile.py").read_text(encoding="utf-8")
    check("run() 调用了 splits_reusable",
          "splits_reusable(" in src)
    check("命中时走 write_splits_reuse 分支",
          "write_splits_reuse" in src)
    check("未命中时仍调用 write_splits",
          "_timed, \"write_splits\", write_splits" in src)


def test_sample_profile_reads_run_config(tmp_path: Path):
    print("A5 SampleProfile 读 run 级配置(泄漏对账走同一开关):")
    run = tmp_path / "run-x"
    (run / "sample-profile").mkdir(parents=True)
    _write_run_config(run, "http://127.0.0.1:9", enabled="false")
    df = _df()
    from runtime.profile import SampleProfile

    sp = SampleProfile(df, _spec(), run / "sample-profile")
    check("SampleProfile.config 读到 run 级 enabled=false",
          library_view.client_settings(sp.config, run_dir=run)["enabled"] is False)
    check("run 级 uri 也随之一致",
          library_view.client_settings(sp.config, run_dir=run)["uri"]
          == "http://127.0.0.1:9")
    # 没有 run 级配置时退回 skill 级(不改变既有行为)
    run2 = tmp_path / "run-y"
    (run2 / "sample-profile").mkdir(parents=True)
    sp2 = SampleProfile(df, _spec(), run2 / "sample-profile")
    check("无 run 配置时不报错(退回 skill 级)",
          sp2.config.get("library.enabled", True) is True)


def test_profile_runs_before_config_change():
    print("A6 profile 内部不再绕过 run 配置(源码断言):")
    src = (Path(__file__).resolve().parents[1] / "runtime"
           / "profile.py").read_text(encoding="utf-8")
    check("SampleProfile 构造时带 run_config",
          'run_config=_run_cfg' in src)
    check("泄漏对账把 self.config 传给 ensure",
          "config=getattr(self, \"config\", None)" in src)


def main() -> int:
    tests = [
        test_library_resolution_uses_run_config,
        test_library_resolution_env_wins,
        test_library_disabled_is_explicit,
        test_library_enabled_by_default,
        test_sample_profile_reads_run_config,
        test_profile_runs_before_config_change,
        test_profile_reports_loading_phase,
        test_splits_reusable_hit_and_reasons,
        test_profile_skips_rewrite_when_reused,
    ]
    for t in tests:
        t()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
