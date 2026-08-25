# 候选特征代码契约

> 候选生成前必读. 契约核心是 transform(df)->Series, 并强调与前置诊断联动.

## 1. 硬性契约(G1 强制, 违反即拒)

- 模块级 `def transform(df: pd.DataFrame) -> pd.Series`, 返回一列
- import 仅限 `math / numpy / pandas / statistics / re`
- 确定性: 禁随机、禁时间、禁环境读取; 双跑不一致即拒
- 输入帧不含 label/id/dt/既有模型分(防泄漏), 引用不存在的列会执行失败

## 2. 候选生成约束(与诊断联动)

### 2.1 原料只从白名单选

候选 `base_features` **必须**来自 `profile/material_whitelist.txt`(低冗余+有信号原料).
`material_blacklist.txt` 里的高冗余原料**禁止**使用--赌中高相关原料的候选
必然栽在 G3 冗余. 前置诊断把这类候选在生成阶段就拦掉.

### 2.2 优先用残差异常原料

`profile/champion_residual.md` 列出 champion 预测错的样本(假阴/假阳)在哪些未用特征上
分布异常. 这些是"模型缺的信号"的数据驱动答案, 优先作交互原料.

### 2.3 算子套路(见 operator-catalog)

先用多字段交互算一个数值(差分/比率/排名/分箱/语义×数值), 再代入非线性复合算子
(log1p/sqrt/多项式/tanh/分式/高斯核/erf/分段复合). 分段函数必须结合复合算子, 不写裸分段.

## 3. meta.json

```json
{
  "cid": "c001",
  "hypothesis": "具体可证伪的一句话假设(不许'综合多个特征')",
  "category": "explore|exploit",
  "base_features": ["白名单里的特征名"]
}
```

## 4. 失败路径回避(读 latest-feedback.md)

上一轮被拒的原因和原料指纹黑名单不要重复尝试. **原料指纹**黑名单
(base_features 排序 sha1), 同族原料近几轮已试则直接判 `material_duplicate`,
逼 LLM 换方向.

## 5. 场景适配

- **cold_start**(baseline 从零训): 低垂果实多, 单特征交互即可涨 AUC
- **saturated**(给已有强模型加特征): 难度高, 优先残差异常原料 × 非线性复合,
  避免与 champion 强特征直接线性交互(直接差分往往与原特征 \|corr\|>0.99, 必挂 G3)
