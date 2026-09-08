# 候选特征可用算子目录（references/operator-catalog.md）

> 每轮提出候选前必读。核心用法：
> **先用多个字段交互算出一个数值，再把这个数值代入非线性复合算子**；
> 分段函数必须结合复合算子使用，不要写裸的分段常数。

## 1. 非线性复合算子（作用于交互结果的第二层）

| 家族 | 常用形式（numpy / math，均在 import 白名单内） | 典型假设场景 |
|---|---|---|
| 指数/对数 | `np.log1p(x)` / `np.expm1(x)` / `np.exp(-x)` | 金额/次数右偏分布压尾 |
| 根函数 | `np.sqrt(x)` / `np.cbrt(x)` / `x ** 0.25` | 压缩量纲且保留 0 点语义 |
| 多项式 | `x ** 2` / `x ** 3` / `(a - b) ** 2` | U 型关系 / 偏离度放大 |
| 三角/双曲 | `np.tanh(x)` / `np.sin(x / period)` | 饱和映射 / 周期性行为 |
| 分式 | `a / (b + 1e-6)` / `a / (a + b + 1e-6)` | 比率型结构（占比/压力/浓度） |
| 误差函数 | `math.erf(x / s)`（配 `np.vectorize` 或向量化写法） | 围绕阈值的平滑过渡 |
| 高斯核 | `np.exp(-((x - mu) / sigma) ** 2)` | 距某个业务中心越近越异常 |
| 分段非线性 | `np.where(cond, 复合算子A, 复合算子B)` | 不同取值区间不同机制（勿用裸常数分段） |

## 2. 多字段交互构造（复合算子的输入层）

- **差分/组合**：`a - b`、`a + k * b`（负债-收入型压力）
- **比率/占比**：`a / (b + eps)`、`a / (a + b + eps)`（还款率、额度使用率）
- **排名/相对位置**：`s.rank(pct=True)`（全体中的相对位置，免量纲）
- **分箱衍生**：`pd.cut` / `pd.qcut` 得到等频分箱序号后再进复合算子
- **语义 × 数值**：`sem_*` 语义列（score 或 pca 低维）与金额/行为特征组合——语义列需先由 semantic-feature skill 注册
- **已接受特征加工**：fid 列（如 `f001`）可与新原料再交互（exploit 方向）

## 2b. 文本/类别字段先转数值

候选输入帧里文本列与类别列**原样可见**（未被删除），可直接在 transform 里加工；重语义理解走 semantic-feature skill，轻量文本算子直接写：

- **文本长度**：`s.str.len()` / 词数 `s.str.split().str.len()` / 平均词长
- **关键词/标点统计**：`s.str.count("逾期")`、`s.str.count(r"[!?]")`
- **正则匹配**：`import re` 后 `s.str.extract(r"(\d+年)")` / `s.fillna("").map(lambda t: len(re.findall(r"电话|手机", t)))`
- **类别列编码**：频次 `s.map(s.value_counts())`（全量统计，无标签参与）、序号 `s.astype("category").cat.codes`、`pd.factorize`
- 组合套路：文本派生数值（长度/关键词计数）-> 再进第 1 节复合算子；或与 `sem_*` 语义打分交互（如"语义分 × 文本长度"）

注意：NaN 文本先 `s.fillna("")` 兜底；**禁止任何用到 label 的统计**（目标编码不可用，G1 输入帧里也没有 label）。

## 2c. 可选常数精调约定（thresholds/带宽类常数交给系统搜）

分箱边界、阈值、带宽（高斯核 sigma）、幂次这类**非单调常数**手写往往粗糙；声明参数约定后，评估系统会在 **train 档**做确定性搜索（随机探针 + 坐标细化，约 60~120 次评估），把最优常数固化回代码尾部，再重过全部门禁--test/OOT 全程不参与搜索，保持干净：

```python
import numpy as np
import pandas as pd

PARAM_BOUNDS = {"center": (0.0, 4.0), "sigma": (0.1, 2.0)}  # 待精调常数及上下界(宜宽)
DEFAULT_PARAMS = {"center": 2.0, "sigma": 0.5}               # 初始值(= 手写值), 键与上表一致

def transform(df, params=None):
    p = dict(DEFAULT_PARAMS) if params is None else dict(params)
    d = (df["f1"] - p["center"]).abs()
    return pd.Series(np.exp(-(d ** 2) / p["sigma"]))
```

适用判断：常数改变会**非单调地**改变特征形状（分界点/带宽/幂次）才值得声明；单纯的比例系数（`f1 * k`）对下游 XGB 无意义，不必声明。

## 3. 进化原则

1. **只增不改**：已接受特征代码冻结只读；想修正某个已接受特征的不合理之处，通过**新增候选**叠加实现（引用其 fid 列），不修改原代码。
2. **一个候选一个假设**：假设必须具体可证伪（"X 与 Y 的比率反映 Z，正类显著更低"），不许"综合多个特征"。
3. **从数据出发**：先看 case-batch 中正负样本的取值差异，再决定用什么算子，而不是反过来套算子。
4. **失败路径回避**：上一轮反馈（latest-feedback.md）里被拒的原因和近轮黑名单方向不要重复尝试。
5. **NaN 显式处理**：`fillna` / `np.where` / `clip` 兜底；非 NaN 覆盖率 < 0.3 会被 G2 直接拒绝。
6. **泛化优先**：宁可单变量增益小而稳（Test/OOT 同向），不可过拟合 train；G5/G6 只认 OOT 稳定增益。

## 4. 硬性契约（G1 强制，违反即拒）

- 模块级 `def transform(df: pd.DataFrame) -> pd.Series`，返回一列
- import 仅限 `math / numpy / pandas / statistics / re`
- 确定性：禁随机、禁时间、禁环境读取；双跑不一致即拒
- 输入帧不含 label/id/dt/既有模型分（如配置 base_model_score_col），引用不存在的列会执行失败
