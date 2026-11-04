"""脚手架模板:TaskSpec / data-loader / Trial 代码(init 用;配置模板见 assets/global-config.yaml)。"""

from __future__ import annotations

TASK_TEMPLATE = '''# TaskSpec:任务规格(§6)— 除 data 段外均为必填示例
task_id: {task_id}
task_name: {task_name}
objective_description: {objective_description}

model_type: binary_classification          # binary_classification | regression | multiclass | ordinal | uplift
business_domains:                          # 取值见 runtime/library/business_domains.yaml(代码、别名或中文名)
  - user_operation

target:
  name: {target_name}
  positive_definition: {positive_definition}   # regression 改为 value_definition + unit
  observation_window: 特征可用的全部历史周期
  performance_window: T0 ~ T+30

# uplift 任务额外需要 treatment 段(1=处置 0=对照,整数列):
#   treatment:
#     column: is_treated
#     definition: 名单内下发优惠券的用户为 1,未下发为 0
#   primary_metric/secondary_metrics 只能 uplift_auc/qini/auuc

time:
  column: {time_column}                    # 时间切分依据;无时间列可删除本段并改用 random split
entity:
  id_column: {id_column}                   # 实体 ID(group leakage 审计用)

data:
  data_loader: data-loader.py              # 必须暴露 load_data() -> DataFrame

primary_metric: auc                        # 主指标(决策口径);regression 可用 rmse/mae/mape/r2;uplift 只能 uplift_auc/qini/auuc
secondary_metrics:
  - ks
  - logloss

split_policy:
  type: time_based
  train: 2026-01~2026-04
  validation: 2026-05
  shadow_test: 2026-06
  final_test: 2026-07                      # 一次性盲测,不进反馈环

features:
  exclude: []                              # 显式剔除列(如泄漏列)
  # columns: []                            # 白名单;缺省=除排除列外全部

resource_budget:
  max_trials: 80
  max_generations: 8
  max_wall_time_hours: 24
  # v18/P0-2:compute 口径默认启用(Σ trial runtime,含评估段)。
  # 墙钟含编排空闲/过夜/中断,不再是预算主口径;wall 保留为绝对护栏。
  # 任务需要更长有效计算时按需上调,注释掉即回退"只按 wall 判停"。
  max_compute_hours: 24
  trial_timeout_minutes: 90

# performance_target:                      # 可选:业务预设目标(§17.2)
#   auc: 0.80
#   improvement_vs_baseline: 0.01

# 模型拓扑(v2.16.0):trial 默认单模型;级联(子分作特征)/融合/白名单外的
# 组合须用户在自己的终端 `approve-topology` 人工批准。旧键 ensemble_enabled
# 已不再授权,无需填写。
'''

DATA_LOADER_TEMPLATE = '''"""data-loader:任务工作区中唯一针对任务定制的 Harness 输入。

契约:暴露 load_data() -> pandas.DataFrame。
约束:不得在此做切分/采样/特征工程(由 Harness 与 Trial 代码负责)。

【分片单文件(推荐先从这改)】
    df = pd.read_parquet(DATA_PATH)
若上游是**多分片 parquet**,不要手写 "for f in sorted(glob(...)): 读 →
逐片处理 → pd.concat"。那样是纯串行,实测 400 万行 × 1513 列(8 片,
24.6GB)要 329 s,其中读 229 s、拼装 59 s。runtime 里有现成的并行实现:

    from runtime.loaders import read_parquet_shards, shrink_object_columns

    df = read_parquet_shards(PATHS, threads=4,
                             post=lambda f: shrink_object_columns(
                                 f, keep_as_is=KEEP_AS_IS,
                                 skip_prefixes=("seq_",)))

它做四件事(全部保留原语义:行序=分片名排序、列序=首片 schema):
  1. 多线程读分片(8 片实测 229 s -> 50 s);
  2. 在 **Arrow 层**把 decimal128 列 cast 成 float64,省掉 pandas
     object -> to_numeric 的往返(16 列实测省 4.8 s/片);
  3. ``pa.concat_tables`` 零拷贝拼装,替代 ``pd.concat`` 大帧(省 59 s);
  4. 一次 ``to_pandas``(分块转换再拼会更慢,已实测)。
同数据整体 329 s -> 174 s。

【如果是单文件、且列里有 decimal】
    pq.read_table(path).to_pandas() 会让 decimal 退化成 object;用
    ``read_parquet_shards([path])`` 走同一条 Arrow cast 路径即可。
"""
import os
import pathlib

import pandas as pd

# 示例:读取本地 parquet;可替换为内部表查询,但必须返回完整建模样本
DATA_PATH = pathlib.Path(os.environ.get("MODELEVO_DATA_PATH", "data/sample.parquet"))


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # 必须包含:目标列、时间列、实体 ID 列(如 TaskSpec 声明)
    return df
'''

#: [v2.16.2] 训练后泄漏哨兵 leak_scan 的单源实现(trial 模板、baseline 模板、
#: --from-parent / warm-start 副本升级共用;D-004:回归目标改用 |Spearman ρ|)。
#: 原文是可执行 Python;嵌入 .format 模板时经 _fmt_escape 转义花括号。
LEAK_SCAN_SOURCE = r'''def leak_scan(model, X_tr: pd.DataFrame, y_tr, top_k: int = 50) -> dict:
    """训练后泄漏哨兵(v13;v2.16.2 按任务类型分派)。

    取模型 top-K gain 特征,只扫 train(不触裁决面):
    - 分类 / uplift:单变量 AUC(方向修正)> 0.9 记入 suspects;
    - 回归(MODEL_TYPE == "regression",或未声明且目标取值 > 2 种):单变量
      |Spearman ρ| > 0.95 记入 suspects —— 统计量与阈值同 profile 的回归泄漏
      规则(单特征 |Spearman corr| > 0.95 判 critical)。旧实现对连续目标算
      AUC 必抛 "multi_class must be in ('ovo','ovr')",哨兵从未运行(run4
      regression D-004)。大样本按固定种子抽 ≤500,000 行算秩相关。
    metrics.json 的 leak_scan 由 runner 消费:suspects 非空 → verdict 强制
    invalid(leakage_suspect),升级到人。**这不是唯一防线**:经聚合搬运的
    弱代理靠数据面物理隔离防,哨兵只兜"从合法列重建极端未来信息"的动态形态。
    """
    out = {"n_scanned": 0, "suspects": []}
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is None and hasattr(model, "booster_"):
            imp = model.booster_.feature_importance(importance_type="gain")
        if imp is None:
            return out
        order = np.argsort(-np.asarray(imp))[:top_k]
        cols = list(X_tr.columns)
        _mt = str(globals().get("MODEL_TYPE") or "")
        _reg = _mt == "regression" or (
            not _mt and pd.Series(np.asarray(y_tr)).nunique() > 2)
        if _reg:
            y_f = pd.to_numeric(pd.Series(np.asarray(y_tr)),
                                errors="coerce").to_numpy(dtype="float64")
            idx = np.arange(len(y_f))
            if len(idx) > 500000:
                idx = np.sort(np.random.default_rng(20260814).choice(
                    len(idx), 500000, replace=False))
            y_s = y_f[idx]
            y_ok = np.isfinite(y_s)
            y_rank = pd.Series(y_s[y_ok]).rank().to_numpy()
            suspects = []
            for i in order:
                c = cols[i]
                v = X_tr[c]
                if not pd.api.types.is_numeric_dtype(v):
                    continue
                s = pd.to_numeric(v, errors="coerce").to_numpy(
                    dtype="float64")[idx]
                ok = np.isfinite(s) & y_ok
                if ok.sum() < 10 or np.unique(s[ok]).size < 2:
                    continue
                if ok[y_ok].all():
                    yr = y_rank
                    sr = pd.Series(s[y_ok]).rank().to_numpy()
                else:
                    yr = pd.Series(y_s[ok]).rank().to_numpy()
                    sr = pd.Series(s[ok]).rank().to_numpy()
                rho = float(np.corrcoef(sr, yr)[0, 1])
                if not np.isfinite(rho):
                    continue
                if abs(rho) > 0.95:
                    suspects.append({"feature": c,
                                     "train_abs_spearman": round(abs(rho), 4)})
            return {"n_scanned": int(len(order)), "suspects": suspects,
                    "statistic": "abs_spearman", "threshold": 0.95,
                    "n_rows_scanned": int(len(idx))}
        y_arr = np.asarray(y_tr).astype(int)
        from sklearn.metrics import roc_auc_score
        suspects = []
        for i in order:
            c = cols[i]
            v = X_tr[c]
            if not pd.api.types.is_numeric_dtype(v):
                continue
            s = pd.to_numeric(v, errors="coerce").fillna(0).values
            if np.unique(s).size < 2:
                continue
            a = roc_auc_score(y_arr, s)
            a = max(a, 1 - a)
            if a > 0.9:
                suspects.append({"feature": c, "train_auc": round(a, 4)})
        out = {"n_scanned": int(len(order)), "suspects": suspects}
    except Exception as _e:  # noqa: BLE001 哨兵失败不阻塞训练
        out["error"] = str(_e)[:120]
    return out'''


def _fmt_escape(src: str) -> str:
    """把可执行源码嵌入 str.format 模板(花括号转义)。"""
    return src.replace("{", "{{").replace("}", "}}")


TRIAL_TEMPLATE = '''"""Trial {trial_id} 实验代码。

假设:{hypothesis}
动作:{actions}

约定(必须遵守):
1. 读取 <run_dir>/sample-profile/splits/ 下的 {{train,validation}}.parquet
   (**特征面**;不含目标/banned/reparse 列——四平面物理隔离,v21);
   标签从 splits-labels/(load_labels 访问器)、解析源从
   splits-reparse/(reparse_frame 访问器)取;identity 仅供框架对齐，
   Trial 不消费;禁止直接调用 load_data() 重建特征(绕过数据面,
   lint 规则 9c 拦截);
2. 只在 train 上拟合;禁止读取 final_test 标签;
3. 早停/调参验证集必须从 train 内部切(尾部时间序,如 es_valid/
   inner-validation),**禁止把 outer validation 用作 eval_set**——
   validation 同时是主指标/delta/noise gate/榜单排名/最终选型的
   裁决面,在它上面做早停会引入与搜索方向相关的系统性乐观偏差
   (v12/P0-4;lint 规则 8 拦截);
4. 输出以下文件到 Trial 目录上级(即 trial 目录本身):
   - validation_score.parquet      列名 score。
                                  分类=正类概率;regression=预测值本身
                                  (rmse/mae/r2 直接以 score 为预测计算);
                                  uplift=CATE 估计(可正可负,S-learner 为
                                  p1-p0 差值)
   - metrics.json                  任意可读指标(n_features 等);集成/多模型
                                  交付必须声明 model_num=实际模型数(单模型=1),
                                  Leaderboard 据此累计交付模型数;leak_scan
                                  的结果也写在这里(模板内置)
   - segment-report.xlsx/.md       (自动)harness 评估时按 score 分桶生成,
                                  无需 Trial 代码产出
   - feature/feature_importance.csv         (强烈建议)列 feature/importance,gain 口径;
                                          反思与经验沉淀依赖它;模型无原生重要性时跳过
   - model/model.joblib(与原生格式+feature_columns.json)
                                          (必须)dump_model(model, columns) 在
                                          fit_and_score 训练完成后调用——模型工件
                                          自动落盘,供事后复用/上线;smoke 轮自动跳过
   - model/validation_input.parquet       (加宽特征 trial 必须)dump_eval_frame(
                                          X_va) 落盘模型输入级的 validation 帧,
                                          行序与 validation split 严格一致。分组
                                          置换裁决(方向生死)依赖它:transform 造出
                                          的新列只存在于 trial 代码里,harness 无法
                                          重放;缺失时新列族判 unverifiable(实测
                                          gen-02:8 个加宽族被伪判 dead)。无 transform
                                          的 trial 不需要(基础帧即模型输入)
5. 多 seed 稳定性(可选):seed_scores/ 下每个 seed 一个 parquet
6. 可复现配置在 ../config/trial.yaml;命令行的 --config 参数仅为提示,
   实际以 load_config() 的固定相对定位读取。特征面以 config 的 features
   为唯一存储(features.exclude 恒生效;features.columns 白名单存在时
   限定特征面),代码运行期经 exclude_columns() 读取——不在代码里再落
   FEATURES 常量,避免双写漂移;超参/seed 同样优先从 config 读,代码内
   只放默认值。**banned 列物理隔离是数据面保证的(不在任何 parquet 里),
   exclude_columns 只做第二道名字过滤**
7. 进度日志(必须):print(..., flush=True) 输出分阶段步骤到 stdout,
   阶段至少包含 数据加载 → 特征工程 → 训练(含折/轮次粒度)→ 评估打分;
   每阶段一行,形如 [step] load_data rows=647287 cols=1503 (12.3s)。
   长训练(GBM 每 100 轮 / NN 每 epoch)再输出中间指标;
   harness 每分钟在 trial.log 追加 runtime/cpu/mem 心跳,与步骤日志互补
8. 线程数(必须):estimator 的 n_jobs / thread_count 一律取脚手架的
   n_jobs() 返回值(读 scheduler 注入的 OMP_NUM_THREADS,与本批并发数
   反比联动),禁止硬编码固定值——固定值会短路调度器的
   "并发×线程≈可用核"分配,批次不满时 CPU 闲置。
9. **泄漏防御(v13)**:特征面物理无 banned 列;transform 不得以任何
   形态重建它们(名字检查不做数,信息流检查才做数);fit 前终检
   assert_banned_clean(X);训练后调用 leak_scan()(top-K gain 特征的
   train 单变量 AUC,>0.9 记入 metrics.json.leak_scan——评估器据此
   强制 invalid)。reparse 解析走 reparse_frame(),产物必须带解析前缀。
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

RUN_DIR = pathlib.Path({run_dir!r})
TRIAL_DIR = pathlib.Path(__file__).resolve().parent.parent
# v14:skill 根入 sys.path。trial 以 cwd=code/ 起进程,`from runtime.hpo
# import …` 找不到包(实测 ModuleNotFoundError: No module named
# 'runtime',operator 手工补了 sys.path.insert 才跑通)。模板本就
# import sys 却从未用过——这里补上。
SKILL_ROOT = {skill_root!r}
if SKILL_ROOT and SKILL_ROOT not in sys.path:
    sys.path.insert(0, SKILL_ROOT)
SPLITS = RUN_DIR / "sample-profile" / "splits"
LABELS_DIR = SPLITS / "labels"
REPARSE_DIR = SPLITS / "reparse"

TARGET = {target!r}
MODEL_TYPE = {model_type!r}
TREATMENT = {treatment!r}
SEED = {seed}


# v2.11:load_config 不在模板头部定义——渲染 body(scripts/modelevo.py
# 主模板)已带同名实现,旧序两处定义后定义覆盖前定义,产物里出现
# 两个 load_config(trial-003/008 实测 AST 级重复定义)。单源 = body。
#
# TaskSpec 泄漏分类(harness 注入,spec 单源;v13 毒性轴)
# banned = 信息有毒:任何形态不得进任何数据产物(数据面已物理隔离,
#   此集合用于 fit 前终检与 lint 规则 9a 的名字防御)
# REPARSE_SOURCES = 形态不可用但信息合法:仅可经 reparse_frame() 消费
BANNED: frozenset[str] = frozenset()
REPARSE_SOURCES: frozenset[str] = frozenset()
#: 非特征列(time/id/treatment;v12/P0-8 由 harness 按 spec 注入,
#: 替换硬编码的 "pday"/"user_no"——换任务时硬编码列名会让时间列
#: 直接进特征面,构成时间穿越泄漏)
_SPEC_NONFEATURE: set[str] = set()


def exclude_columns(df: pd.DataFrame, cfg=None) -> list[str]:
    # 特征面 = 全列 − 目标/时间/ID/treatment − banned/reparse −
    # features.exclude(config)− features.columns 白名单过滤。
    # 防呆:一律以本函数解析特征面,不要手写 FEATURES 列表(历史教训:
    # 手写漏 exclude 会把 label 同族窗口列带回模型,得到 AUC=1.0 假结果)。
    # v13:banned 的主防御在数据面(物理不落盘),这里只是第二道过滤。
    cfg = cfg if cfg is not None else load_config()
    feats_cfg = cfg.get("features") or {{}}
    drop = ({{TARGET}} | _SPEC_NONFEATURE | set(BANNED)
            | set(REPARSE_SOURCES))
    if isinstance(feats_cfg, dict):
        drop |= set(feats_cfg.get("exclude") or [])
    cols = [c for c in df.columns if c not in drop]
    whitelist = (feats_cfg.get("columns")
                 if isinstance(feats_cfg, dict) else None)
    if whitelist:
        cols = [c for c in whitelist if c in cols]
    return cols


def load_labels(split: str) -> pd.DataFrame:
    """标签面访问器:目标 + treatment(存在时)+ 时间/ID 对齐键。

    标签与特征面物理分文件(四平面隔离),从这里取 y/w——
    从特征面找不到目标列,手写特征面带回 label 同族列的事故
    从根上不可能。
    """
    path = LABELS_DIR / f"{{split}}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"标签面缺失: {{path}}(先跑 profile)")
    return pd.read_parquet(path)


def reparse_frame(split: str, columns: list[str]) -> pd.DataFrame:
    """解析源访问器(v13):reparse 列的唯一合法消费入口。

    - 仅限 REPARSE_SOURCES(白名单断言;列名不在其中即拒绝);
    - 返回列 = 请求列 + 对齐键(时间/ID,行序与特征面严格一致);
    - 解析产物必须带解析前缀(seq_* 等),以口径前缀可追溯。
    特征面物理没有这些列——除本访问器外无处可取。
    """
    allow = set(REPARSE_SOURCES)
    illegal = [c for c in columns if c not in allow]
    if illegal:
        raise ValueError(
            f"列 {{illegal}} 不在 REPARSE_SOURCES 白名单({{sorted(allow)}});"
            "banned/普通列不得经 reparse 通道取数")
    if not columns:
        # v2.11.6:零列请求直接返回空帧。spec 声明 reparse=[] 时 profile
        # 不落盘 reparse 目录(整目录跳过),旧实现仍去读盘 ⇒ 每次空
        # 请求都抛 FileNotFoundError(实测:finalize-retrain 对无 reparse
        # 任务无条件调用本访问器,交付链被阻断)。零列本就无须读盘。
        return pd.DataFrame()
    path = REPARSE_DIR / f"{{split}}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"解析源缺失: {{path}}(spec 无 reparse 列或 profile 未落盘)")
    frame = pd.read_parquet(path)
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"reparse 源缺列 {{missing}}: 可用 {{list(frame.columns)}}")
    return frame[columns]


def assert_banned_clean(X: pd.DataFrame) -> None:
    """fit 前终检(零成本,防绕行):特征面不得含 banned 列。

    数据面物理隔离是主防御;本断言把"绕过数据面重建特征"的
    意外提前到训练前报错,而不是训练后靠 leak_scan 兜底。
    """
    hit = BANNED & set(X.columns)
    if hit:
        raise ValueError(f"泄漏列渗入特征面: {{sorted(hit)}}"
                         "(banned 列任何形态不得进模型)")


''' + _fmt_escape(LEAK_SCAN_SOURCE) + '''


def dump_feature_importance(model, columns) -> None:
    """尽力输出 gain 重要性;模型不提供时静默跳过。

    落盘到 TRIAL_DIR/feature/feature_importance.csv。
    """
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "booster_"):
        try:
            imp = model.booster_.feature_importance(importance_type="gain")
        except Exception:
            imp = None
    if imp is None:
        return
    out_dir = TRIAL_DIR / "feature"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({{"feature": list(columns),
                   "importance": list(imp)}}).sort_values(
        "importance", ascending=False).to_csv(
        out_dir / "feature_importance.csv", index=False)


def dump_model(model, columns) -> None:
    """持久化模型工件到 TRIAL_DIR/model/(尽力而为,不阻塞 trial):
    model.joblib(sklearn 口 wrapper)+ 原生格式(lightgbm .txt /
    xgboost .json / catboost .bin / torch .pt)+ feature_columns.json
    (推理列对齐用)。smoke 预检轮(MODELEVO_SMOKE_RUN=1)跳过,
    防止小样本模型覆盖全量模型。Agent 改造 fit_and_score 时必须保留
    该调用(Trial Lint 规则 6 拦截删除)。

    v2.13.4 写保护:仅 runner 执行本 trial 时(MODELEVO_TRIAL_EXEC ==
    TRIAL_DIR.name)才写盘。外部脚本/交付链/自测加载本文件调用
    fit_and_score 时拒写,冻结工件只读。"""
    import os
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return
    # multi-seed 变体轮不覆写 primary 工件(打分模型 == 落盘模型)
    if os.environ.get("MODELEVO_SEED_VARIANT") == "1":
        return
    if os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name:
        print("[guard] dump_model 拒写: 非 runner 执行上下文"
              "(MODELEVO_TRIAL_EXEC 不匹配),trial 工件只读", flush=True)
        return
    try:
        import joblib
        mdir = TRIAL_DIR / "model"
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, mdir / "model.joblib")
        mod = type(model).__module__
        if hasattr(model, "booster_"):
            model.booster_.save_model(str(mdir / "model.txt"))
        elif mod.startswith("xgboost") and hasattr(model, "save_model"):
            model.save_model(str(mdir / "model.json"))
        elif mod.startswith("catboost") and hasattr(model, "save_model"):
            model.save_model(str(mdir / "model.bin"))
        elif hasattr(model, "state_dict"):
            import torch
            torch.save(model.state_dict(), mdir / "model.pt")
        elif hasattr(model, "model") and hasattr(model.model, "state_dict"):
            import torch
            torch.save(model.model.state_dict(), mdir / "model.pt")
        with open(mdir / "feature_columns.json", "w",
                  encoding="utf-8") as fh:
            json.dump(list(columns), fh)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] dump_model skipped: {{_e}}", flush=True)


# v2.11:dump_eval_frame 亦单源到渲染 body(旧序与 body 各定义一次,
# 后定义覆盖,产物重复定义;v2.10.1 run 实测)。规则 6b 的检测消费它
# 的调用,定义在哪一侧不影响。
def dump_shadow_score(score) -> None:
    """shadow_test 预测落盘，供未来独立 confirm 作父代对照。

    普通搜索只绑定样本键，不读取 shadow 目标、不计算指标；只有新 seed
    的 confirm trial 才能消费标签并形成 fresh_confirmation。
    smoke/quick 轮跳过(小样本分数不得覆盖全量帧)。"""
    import os
    if os.environ.get("MODELEVO_SMOKE_RUN") == "1":
        return
    if os.environ.get("MODELEVO_QUICK_RATIO", "0") not in ("", "0"):
        return
    if os.environ.get("MODELEVO_TRIAL_EXEC") != TRIAL_DIR.name:
        print("[guard] dump_shadow_score 拒写: 非 runner 执行上下文"
              "(MODELEVO_TRIAL_EXEC 不匹配),trial 工件只读", flush=True)
        return
    try:
        import pandas as _pd
        _frame = _pd.DataFrame({{"score": score}})
        # v2.19.1:落盘即带样本键。旧实现只写 score 列,键由 runner 评估阶段
        # 的 bind_prediction_artifact 绑定;trial 若在评估前失败(如 amazon
        # trial-016 catboost 重跑覆盖),留下的 shadow 工件**没有键**,benchmark
        # 的对齐评分器与 confirm 的父代对照都无法消费。这里只投影读取键列
        # (不读目标列),与 runner 的无标签身份绑定同源。
        _lab = LABELS_DIR / "shadow_test.parquet"
        if _lab.exists() and len(_frame) > 0:
            try:
                _keys = _pd.read_parquet(
                    _lab, columns=["__mvo_sample_key"])["__mvo_sample_key"]
                if len(_keys) == len(_frame):
                    _frame["__mvo_sample_key"] = _keys.to_numpy()
                else:
                    print(f"[warn] dump_shadow_score 不加样本键: 分数 {{len(_frame)}} 行"
                          f" != 标签面 {{len(_keys)}} 行", flush=True)
            except Exception as _ke:  # noqa: BLE001  键缺失不阻塞落盘
                print(f"[warn] dump_shadow_score 读样本键失败(不写入键列): {{_ke}}",
                      flush=True)
        _frame.to_parquet(TRIAL_DIR / "shadow_test_score.parquet", index=False)
    except Exception as _e:  # noqa: BLE001
        print(f"[warn] dump_shadow_score skipped: {{_e}}", flush=True)


{body}
'''


# ===========================================================================
# v2.17.0:按 spec.model_type 渲染单一打分分支(lint 规则 4 误报的根治)
# ===========================================================================
# run4 regression.2:baseline 渲染代码与 `--action hpo` 嫁接的模板段都带
# `predict_proba(...)[:, 1]` 通用打分分派(分类 / uplift / hasattr 分派),对
# 回归任务全是不可达分支,却每代都要 adhoc/narrow_reg_dispatch_g03.py 收窄才
# 过 lint 规则 4。这里在**生成期**把回归任务的打分分派折叠成单分支:
#   - 只折叠"本身或其分支含 predict_proba / sigmoid / expit"的 if 链(打分
#     分派),其余 MODEL_TYPE 分支(落盘口径、指标 sanity)原样保留 —— 模板
#     锚点(render 的嫁接/升级依赖的 `elif MODEL_TYPE == "regression":` 等)
#     不受影响;
#   - 判定只用静态可证的条件:`MODEL_TYPE ==/!=/in/not in <字面量>` 及其
#     and/or/not 组合;`hasattr(<x>, "predict_proba")` 在回归任务下视为 False
#     (回归估计器与官方 HurdleRegressor 均不暴露 predict_proba),**但**模块里
#     有类自定义 predict_proba 时不折叠该条件(不改变自定义容器的语义);
#   - 分类 / uplift / 多分类任务原样返回(逐字节不变)。
# 折叠是纯文本行级改写(保留注释与原始格式),改写后必须能 ast.parse,
# 否则整体放弃、返回原文。
_DISPATCH_TEXT_RE = None


def _module_model_type(tree) -> str | None:
    import ast as _ast

    for node in tree.body:
        if isinstance(node, _ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], _ast.Name) and \
                node.targets[0].id == "MODEL_TYPE" and \
                isinstance(node.value, _ast.Constant) and \
                isinstance(node.value.value, str):
            return node.value.value
    return None


def _defines_predict_proba(tree) -> bool:
    import ast as _ast

    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (_ast.FunctionDef, _ast.AsyncFunctionDef)) \
                        and sub.name == "predict_proba":
                    return True
    return False


def static_model_type_truth(test, model_type: str, *,
                            proba_absent: bool = False):
    """条件表达式在给定 model_type 下的静态真值:True / False / None(不可判)。

    支持 `MODEL_TYPE ==/!=/in/not in 字面量`(含反向写法)、and/or/not 组合;
    proba_absent=True 时 `hasattr(<x>, "predict_proba")` 判 False。
    """
    import ast as _ast

    def lit(node):
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, (_ast.Tuple, _ast.List, _ast.Set)) and all(
                isinstance(e, _ast.Constant) and isinstance(e.value, str)
                for e in node.elts):
            return {e.value for e in node.elts}
        return None

    def ev(node):
        if isinstance(node, _ast.Compare) and len(node.ops) == 1:
            a, b, op = node.left, node.comparators[0], node.ops[0]
            if isinstance(b, _ast.Name) and b.id == "MODEL_TYPE" and \
                    isinstance(op, (_ast.Eq, _ast.NotEq)):
                a, b = b, a
            if not (isinstance(a, _ast.Name) and a.id == "MODEL_TYPE"):
                return None
            vals = lit(b)
            if vals is None:
                return None
            if isinstance(op, _ast.Eq):
                return len(vals) == 1 and model_type in vals
            if isinstance(op, _ast.NotEq):
                return not (len(vals) == 1 and model_type in vals)
            if isinstance(op, _ast.In):
                return model_type in vals
            if isinstance(op, _ast.NotIn):
                return model_type not in vals
            return None
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) \
                and node.func.id == "hasattr" and len(node.args) == 2 and \
                isinstance(node.args[1], _ast.Constant) and \
                node.args[1].value == "predict_proba":
            return False if proba_absent else None
        if isinstance(node, _ast.BoolOp):
            vals = [ev(v) for v in node.values]
            if isinstance(node.op, _ast.And):
                if any(v is False for v in vals):
                    return False
                return True if all(v is True for v in vals) else None
            if any(v is True for v in vals):
                return True
            return False if all(v is False for v in vals) else None
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.Not):
            v = ev(node.operand)
            return None if v is None else (not v)
        return None

    return ev(test)


def _indent_of(line: str) -> str:
    return line[:len(line) - len(line.lstrip(" \t"))]


def _multiline_literal_in(node) -> bool:
    import ast as _ast

    for sub in _ast.walk(node):
        if isinstance(sub, (_ast.Constant, _ast.JoinedStr)) and \
                getattr(sub, "end_lineno", sub.lineno) > sub.lineno:
            return True
    return False


def _dedent_block(block: list[str], shift: int) -> list[str] | None:
    out = []
    for ln in block:
        if not ln.strip():
            out.append("\n")
            continue
        ind = len(_indent_of(ln))
        if ind < shift or "\t" in ln[:ind]:
            return None
        out.append(ln[shift:])
    return out


def _rewrite_if(lines: list[str], node, truth: bool, parent_len: int
                ) -> list[str] | None:
    """返回替换 lines[node.lineno-1 : node.end_lineno] 的新行;不可安全改写返回 None。"""
    import ast as _ast

    head = lines[node.lineno - 1]
    hind = _indent_of(head)
    is_elif = head.lstrip().startswith("elif")
    if not node.body or node.body[0].lineno <= node.test.end_lineno:
        return None                                   # 单行 if x: y
    body_lo = node.test.end_lineno                    # 0-based 起始行(header 下一行)
    body_hi = node.body[-1].end_lineno                # 0-based 终止(不含)
    body_lines = lines[body_lo:body_hi]
    bind = node.body[0].col_offset
    orelse = node.orelse
    else_kind, else_lines = "none", []
    if orelse:
        first = lines[orelse[0].lineno - 1].lstrip()
        if len(orelse) == 1 and isinstance(orelse[0], _ast.If) and \
                first.startswith("elif"):
            else_kind = "elif"
            else_lines = lines[orelse[0].lineno - 1:node.end_lineno]
        else:
            else_at = None
            for k in range(body_hi, orelse[0].lineno - 1):
                s = lines[k].strip()
                if s.startswith("else") and s[4:].lstrip().startswith(":") \
                        and _indent_of(lines[k]) == hind:
                    else_at = k
            if else_at is None or orelse[0].lineno - 1 <= else_at:
                return None
            if lines[else_at].strip().split("#", 1)[0].strip() != "else:":
                return None                           # else: stmt 同行
            else_kind = "else"
            else_lines = lines[else_at + 1:node.end_lineno]
    note = (f"{hind}# [v2.17.0] MODEL_TYPE={'regression'} 单分支渲染"
            "(分类/uplift 打分分派不生成)\n")
    if truth:
        if is_elif:
            return [f"{hind}else:\n"] + body_lines
        if any(_multiline_literal_in(s) for s in node.body):
            return None
        ded = _dedent_block(body_lines, bind - len(hind))
        return None if ded is None else [note] + ded
    # truth False
    if else_kind == "none":
        if is_elif:
            return []
        return [] if parent_len > 1 else [note, f"{hind}pass\n"]
    if else_kind == "elif":
        if is_elif:
            return else_lines
        first = else_lines[0]
        i = first.find("elif")
        return [first[:i] + first[i + 2:]] + else_lines[1:]
    # else 块
    if is_elif:
        return [f"{hind}else:\n"] + else_lines
    if any(_multiline_literal_in(s) for s in orelse):
        return None
    eind = orelse[0].col_offset
    ded = _dedent_block(else_lines, eind - len(hind))
    return None if ded is None else [note] + ded


def narrow_model_type_dispatch(code: str, model_type: str | None = None
                               ) -> tuple[str, list[str]]:
    """把回归任务代码里的分类/uplift 打分分派折叠为单分支(见上方说明)。

    model_type 缺省时取模块级 `MODEL_TYPE = '...'` 常量;非 regression 原样
    返回。返回 (新代码, 改写说明列表)。幂等:再次调用无可折叠项时原样返回。
    """
    import ast as _ast
    import re as _re

    global _DISPATCH_TEXT_RE
    if _DISPATCH_TEXT_RE is None:
        _DISPATCH_TEXT_RE = _re.compile(r"predict_proba|\bsigmoid\b|\bexpit\b")
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return code, []
    mt = model_type or _module_model_type(tree)
    if mt != "regression":
        return code, []
    original, notes = code, []
    for _ in range(400):
        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return original, []
        proba_absent = not _defines_predict_proba(tree)
        lines = code.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        parents: dict[int, int] = {}
        for par in _ast.walk(tree):
            for field in ("body", "orelse", "finalbody"):
                seq = getattr(par, field, None)
                if isinstance(seq, list):
                    for ch in seq:
                        parents[id(ch)] = len(seq)
            for h in getattr(par, "handlers", None) or []:
                for ch in h.body:
                    parents[id(ch)] = len(h.body)
        cands = []
        for n in _ast.walk(tree):
            if not isinstance(n, _ast.If):
                continue
            v = static_model_type_truth(n.test, "regression",
                                        proba_absent=proba_absent)
            if v is None:
                continue
            seg = "".join(lines[n.lineno - 1:n.end_lineno])
            if not _DISPATCH_TEXT_RE.search(seg):
                continue
            cands.append((n.lineno, -n.end_lineno, id(n), n, v))
        if not cands:
            break
        cands.sort(key=lambda c: (c[0], c[1]))
        done = False
        for _l, _e, _i, n, v in cands:
            rep = _rewrite_if(lines, n, v, parents.get(id(n), 1))
            if rep is None:
                continue
            new = lines[:n.lineno - 1] + rep + lines[n.end_lineno:]
            new_code = "".join(new)
            try:
                _ast.parse(new_code)
            except SyntaxError:
                continue
            notes.append(f"L{n.lineno}: `{lines[n.lineno - 1].strip()[:60]}` "
                         f"→ {'保留本分支' if v else '删除本分支'}")
            code = new_code
            done = True
            break
        if not done:
            break
    if code != original and not original.endswith("\n") and code.endswith("\n"):
        code = code[:-1]
    return code, notes
