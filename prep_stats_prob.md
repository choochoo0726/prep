# Quant 面试完全指南：概率论 · 回归分析 · 机器学习

> **适用：** 量化研究员、量化交易员、风险分析师、数据科学家面试备考
> **内容：** 62道高频面试题，含完整数学推导、几何直觉与实战 Insights
> **格式：** 所有公式使用 LaTeX（支持 Obsidian / Typora / VS Code 渲染）

---

## 目录

- [第一部分：概率论（20题）](#第一部分概率论)
- [第二部分：回归分析（26题）](#第二部分回归分析)
- [第三部分：机器学习（16题）](#第三部分机器学习)

---

# 第一部分：概率论

---

## P1. 条件概率与贝叶斯定理

**难度：** ⭐ 基础 | **频率：** 极高

### 定义

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**贝叶斯定理：**

$$\boxed{P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}}$$

其中分母由全概率公式展开：

$$P(B) = P(B \mid A)\,P(A) + P(B \mid A^c)\,P(A^c)$$

### 完整推导

由联合概率的对称性 $P(A \cap B) = P(B \mid A)\,P(A) = P(A \mid B)\,P(B)$，直接相除即得贝叶斯定理。

**贝叶斯更新的迭代形式：**

$$P(A \mid B_1, B_2) = \frac{P(B_2 \mid A, B_1)\,P(A \mid B_1)}{P(B_2 \mid B_1)}$$

每次观测到新证据，后验概率成为下一轮先验——这是贝叶斯滤波（Kalman Filter 的离散前身）的核心。

### 经典例题：疾病检测

已知：患病率 $P(D)=0.01$，检测灵敏度 $P(+\mid D)=0.99$，假阳率 $P(+\mid D^c)=0.05$。

$$P(D \mid +) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} \approx 16.7\%$$

> **Insight：** 即使测试精度99%，检测阳性后真正患病的概率只有16.7%！这是**基率忽视（base rate neglect）**的经典案例。量化中对应：即使信号精度高，若基础胜率（base rate）低，策略实际盈利概率仍可能很低。

---

## P2. 独立事件与互斥事件

**难度：** ⭐ 基础 | **频率：** 高

### 定义对比

| 概念 | 定义 | 直觉 |
|------|------|------|
| **独立** | $P(A \cap B) = P(A) \cdot P(B)$ | A的发生不影响B的概率 |
| **互斥** | $P(A \cap B) = 0$ | A和B不能同时发生 |

### 关键关系

**互斥事件一般不独立（除非某事件概率为0）：**

若 $A, B$ 互斥且 $P(A) > 0, P(B) > 0$，则：

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)} = 0 \neq P(A)$$

即 B 发生后，A 的概率变为 0——强相关，绝非独立。

> **Insight（量化应用）：** 在极端市场崩溃时，"本应独立"的资产会同时下跌（尾部相关性激增）。多元高斯 Copula 假设各资产独立尾部，严重低估了 2008 年的联合违约概率。

---

## P3. 期望与方差的线性性质

**难度：** ⭐⭐ 中等 | **频率：** 极高

### 核心公式

**期望（无论是否独立，永远成立）：**

$$E[aX + bY] = a\,E[X] + b\,E[Y]$$

**方差：**

$$\text{Var}(aX + bY) = a^2\,\text{Var}(X) + b^2\,\text{Var}(Y) + 2ab\,\text{Cov}(X, Y)$$

### 推导

$$\text{Var}(X) = E[X^2] - (E[X])^2$$

$$\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$$

独立时 $E[XY] = E[X]E[Y]$，故 $\text{Cov}=0$，方差可加。

**$n$ 个独立同分布变量之和：**

$$E\left[\sum_{i=1}^n X_i\right] = n\mu, \quad \text{Var}\left(\sum_{i=1}^n X_i\right) = n\sigma^2, \quad \text{SD}(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

### 投资组合应用

权重向量 $\mathbf{w}$，资产收益向量 $\mathbf{r}$，协方差矩阵 $\Sigma$：

$$\text{Var}(\mathbf{w}^\top \mathbf{r}) = \mathbf{w}^\top \Sigma\, \mathbf{w}$$

> **Insight：** 分散化的数学本质是利用 $\text{Cov}$ 项。若所有资产相关系数 $\rho < 1$，增加资产数可以降低组合方差，但有下界 $\rho \sigma^2$（系统风险，无法消除）。

---

## P4. 中心极限定理（CLT）

**难度：** ⭐⭐ 中等 | **频率：** 极高

### 定理表述

设 $X_1, X_2, \ldots, X_n$ 独立同分布，$E[X_i]=\mu$，$\text{Var}(X_i)=\sigma^2 < \infty$，则：

$$\sqrt{n}\,\frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} N(0, 1) \quad (n \to \infty)$$

等价地：$\bar{X}_n \overset{\text{approx}}{\sim} N\!\left(\mu,\, \frac{\sigma^2}{n}\right)$

### 直觉证明（MGF方法）

设 $Z_i = (X_i - \mu)/\sigma$，$S_n = \frac{1}{\sqrt{n}}\sum Z_i$。$Z_i$ 的 MGF 为 $M(t) = 1 + 0 \cdot t + \frac{1}{2}t^2 + O(t^3)$（Taylor展开，利用 $E[Z_i]=0, E[Z_i^2]=1$）。

$$M_{S_n}(t) = \left[M\!\left(\frac{t}{\sqrt{n}}\right)\right]^n = \left[1 + \frac{t^2}{2n} + O(n^{-3/2})\right]^n \to e^{t^2/2}$$

而 $e^{t^2/2}$ 正是 $N(0,1)$ 的 MGF。

### 适用条件与速度

| 分布形态 | 需要的 $n$ |
|---------|-----------|
| 对称分布 | $n \geq 10$ 通常足够 |
| 轻度偏斜 | $n \geq 30$ |
| 重尾/强偏斜 | $n \geq 100+$ |

> **Insight：** 金融收益率具有厚尾，CLT 收敛极慢。用 CLT 近似计算极端分位点（VaR 99.9%）会严重低估尾部风险——这是为什么极端风险管理要用 EVT（极值理论）而非正态近似。

---

## P5. 大数定律（LLN）

**难度：** ⭐⭐ 中等 | **频率：** 高

### 两种形式

**弱大数定律（WLLN，Chebyshev）：**

$$\bar{X}_n \xrightarrow{p} \mu \quad \text{即} \quad \lim_{n\to\infty} P(|\bar{X}_n - \mu| > \varepsilon) = 0 \; \forall\varepsilon > 0$$

**强大数定律（SLLN，Kolmogorov）：**

$$\bar{X}_n \xrightarrow{a.s.} \mu \quad \text{即} \quad P\!\left(\lim_{n\to\infty} \bar{X}_n = \mu\right) = 1$$

### WLLN 的 Chebyshev 证明

$$P(|\bar{X}_n - \mu| \geq \varepsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2} \to 0$$

### LLN vs CLT 的本质区别

| | LLN | CLT |
|--|-----|-----|
| 描述 | 均值收敛到哪里 | 收敛的速率和形状 |
| 结果 | $\bar{X}_n \to \mu$（点收敛） | $\sqrt{n}(\bar{X}_n - \mu) \to N(0,\sigma^2)$（分布收敛） |
| 前提 | 有限均值即可 | 需要有限方差 |

> **Insight（量化应用）：** LLN 是回测的理论基础——足够多的交易后，样本胜率收敛到真实胜率。但 CLT 告诉我们"足够多"到底需要多少次，以及估计误差的置信区间。

---

## P6. 矩母函数（MGF）

**难度：** ⭐⭐ 中等 | **频率：** 中

### 定义与矩的提取

$$M_X(t) = E[e^{tX}] = \sum_{k=0}^{\infty} \frac{E[X^k]}{k!} t^k$$

第 $k$ 阶矩：$E[X^k] = M_X^{(k)}(0)$（在 $t=0$ 处的 $k$ 阶导数）

### 独立变量之和的MGF

若 $X, Y$ 独立：

$$M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$$

**应用：证明独立正态之和仍为正态**

$$X \sim N(\mu_1, \sigma_1^2),\; M_X(t) = e^{\mu_1 t + \frac{1}{2}\sigma_1^2 t^2}$$

$$M_{X+Y}(t) = e^{(\mu_1+\mu_2)t + \frac{1}{2}(\sigma_1^2+\sigma_2^2)t^2} \implies X+Y \sim N(\mu_1+\mu_2,\, \sigma_1^2+\sigma_2^2)$$

### 常见分布的MGF

| 分布 | MGF $M(t)$ |
|------|-----------|
| $N(\mu, \sigma^2)$ | $e^{\mu t + \frac{1}{2}\sigma^2 t^2}$ |
| $\text{Poisson}(\lambda)$ | $e^{\lambda(e^t - 1)}$ |
| $\text{Exp}(\lambda)$ | $\frac{\lambda}{\lambda - t},\; t < \lambda$ |
| $\text{Bernoulli}(p)$ | $1 - p + pe^t$ |

> **Insight：** MGF 不仅能算矩，更重要的是其唯一性定理——若两个分布的 MGF 在某邻域内相同，则分布完全相同。这是很多分布推导的核心工具。

---

## P7. 马尔可夫链与平稳分布

**难度：** ⭐⭐⭐ 进阶 | **频率：** 中高

### 定义

**马尔可夫性：** $P(X_{t+1} = j \mid X_t = i, X_{t-1}, \ldots) = P(X_{t+1} = j \mid X_t = i) = p_{ij}$

转移矩阵 $P$，满足 $p_{ij} \geq 0$ 且每行之和为 1（随机矩阵）。

### 平稳分布的求解

平稳分布 $\pi$ 满足：

$$\pi P = \pi, \quad \sum_i \pi_i = 1, \quad \pi_i \geq 0$$

**例：2状态链**

$$P = \begin{pmatrix} 1-\alpha & \alpha \\ \beta & 1-\beta \end{pmatrix}$$

解 $\pi P = \pi$：

$$\pi_1 \alpha = \pi_2 \beta \implies \frac{\pi_1}{\pi_2} = \frac{\beta}{\alpha}$$

$$\boxed{\pi_1 = \frac{\beta}{\alpha+\beta}, \quad \pi_2 = \frac{\alpha}{\alpha+\beta}}$$

### 收敛速度

$n$ 步转移矩阵 $P^n \to \mathbf{1}\pi^\top$（行收敛到平稳分布）。收敛速度由第二大特征值 $|\lambda_2|$ 决定：

$$\|P^n_{i\cdot} - \pi\|_{TV} \leq C \cdot |\lambda_2|^n$$

混合时间（mixing time）$\tau \sim \frac{1}{1-|\lambda_2|}$。

> **Insight（量化应用）：** 市场状态（牛/熊/震荡）可建模为马尔可夫链，平稳分布给出长期各状态占比。Hamilton (1989) 的区制转换模型（Regime-Switching）正是此思路，被广泛用于宏观择时。

---

## P8. 正态分布与对数正态分布

**难度：** ⭐⭐ 中等 | **频率：** 极高

### 核心关系

若 $X \sim N(\mu, \sigma^2)$，则 $S = e^X$ 服从对数正态分布 $\text{LN}(\mu, \sigma^2)$。

**对数正态的矩：**

$$E[S] = e^{\mu + \frac{1}{2}\sigma^2}, \quad \text{Var}(S) = e^{2\mu+\sigma^2}(e^{\sigma^2}-1)$$

**推导** $E[e^X]$（正态 MGF 令 $t=1$）：

$$E[e^X] = M_X(1) = e^{\mu \cdot 1 + \frac{1}{2}\sigma^2 \cdot 1^2} = e^{\mu + \frac{1}{2}\sigma^2}$$

### Black-Scholes 中的应用

股票价格 $S_t = S_0 \exp\!\left[(\mu - \tfrac{1}{2}\sigma^2)t + \sigma W_t\right]$（GBM）

- 对数收益率 $\ln(S_t/S_0) \sim N((\mu-\tfrac{1}{2}\sigma^2)t,\; \sigma^2 t)$
- 漂移修正项 $-\tfrac{1}{2}\sigma^2$：来自伊藤引理（$d(\ln S)$比$dS/S$少这一项）

> **Insight：** 用对数正态建模价格的关键优点是**价格非负性**（$e^X > 0$）。但其厚尾（skewness > 0，右偏）与实际金融收益的**左偏厚尾**不符，这是 BS 模型低估深度虚值看跌期权价值的根本原因（波动率微笑）。

---

## P9. 泊松分布与泊松过程

**难度：** ⭐⭐ 中等 | **频率：** 中高

### 分布定义

$$P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k = 0,1,2,\ldots$$

$$E[X] = \text{Var}(X) = \lambda$$

**关键性质：叠加性**

若 $X_1 \sim \text{Pois}(\lambda_1)$，$X_2 \sim \text{Pois}(\lambda_2)$ 独立，则 $X_1+X_2 \sim \text{Pois}(\lambda_1+\lambda_2)$。

### 泊松过程

计数过程 $\{N(t), t \geq 0\}$ 是强度为 $\lambda$ 的泊松过程，若：

1. $N(0) = 0$
2. 独立增量
3. $N(t+s) - N(t) \sim \text{Pois}(\lambda s)$

**事件间隔：** $T_k \sim \text{Exp}(\lambda)$（i.i.d.）

### 复合泊松过程（跳扩散模型）

$$S_t = S_0 e^{\sigma W_t + (r-\frac{1}{2}\sigma^2)t} \cdot \prod_{k=1}^{N(t)} J_k$$

其中 $N(t)$ 是泊松过程，$J_k$ 是跳跃幅度（Merton 跳扩散模型）。

> **Insight：** 标准 BS 假设连续路径，无法解释闪崩（Flash Crash）等突发事件。跳扩散模型通过泊松过程加入跳跃项，更好地刻画尾部风险，但期权定价变得不完全（无法完全对冲跳跃风险）。

---

## P10. 协方差矩阵与正定性

**难度：** ⭐⭐⭐ 进阶 | **频率：** 高

### 定义与性质

$$\Sigma_{ij} = \text{Cov}(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]$$

**协方差矩阵是半正定的：** 对任意向量 $\mathbf{x}$，

$$\mathbf{x}^\top \Sigma \mathbf{x} = \text{Var}(\mathbf{x}^\top \mathbf{X}) \geq 0$$

### 正定性的等价条件

1. $\mathbf{x}^\top \Sigma \mathbf{x} > 0$，$\forall \mathbf{x} \neq \mathbf{0}$
2. 所有特征值 $\lambda_i > 0$
3. 所有顺序主子式 $> 0$（Sylvester 准则）
4. 存在 Cholesky 分解 $\Sigma = LL^\top$，$L$ 下三角且对角元素 $> 0$

### 实际估计中的正定性问题

样本协方差矩阵 $\hat{\Sigma} = \frac{1}{n-1}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top$

当 $p > n$（变量数超过样本量），$\hat{\Sigma}$ 必然奇异（秩最多为 $n-1$）。

**修正方法（Ledoit-Wolf 收缩估计）：**

$$\hat{\Sigma}_{\text{shrink}} = (1-\alpha)\hat{\Sigma} + \alpha T$$

其中 $T$ 是目标矩阵（通常取对角阵），$\alpha \in [0,1]$ 通过最优化选取。

> **Insight（量化风险管理）：** 高维情形（500只股票，3年日度数据约750个观测）中，历史协方差矩阵的最小特征值会极度低估真实最小特征值（Marcenko-Pastur 定律），导致均值方差优化组合极度集中。Ledoit-Wolf 收缩是标准的工业级解决方案。

---

## P11. 置信区间

**难度：** ⭐ 基础 | **频率：** 高

### 定义（频率派）

**95% 置信区间**：如果重复无穷多次抽样，按相同方法构建的区间中，约 95% 会包含真实参数值 $\theta$。

**不是：** "此特定区间包含 $\theta$ 的概率是 95%"（$\theta$ 是固定的，区间包含或不包含，无概率可言）

### 常用置信区间公式

已知 $\sigma$：$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$，其中 $z_{0.025} = 1.96$

未知 $\sigma$：$\bar{X} \pm t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}}$

**双侧 95% CI 的快速记法：** $\bar{X} \pm 2\,\frac{s}{\sqrt{n}}$（近似）

### 影响 CI 宽度的因素

$$\text{宽度} = 2 z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

| 因素 | 变化 | CI 宽度 |
|------|------|---------|
| 样本量 $n$ 增大 | ↑ | 变窄 ($1/\sqrt{n}$) |
| 总体波动 $\sigma$ 增大 | ↑ | 变宽 |
| 置信水平提高（95%→99%） | ↑ | 变宽 |

> **Insight：** 贝叶斯 vs 频率派对 CI 的解读：贝叶斯的**可信区间（Credible Interval）**才是"参数落在区间内的概率是95%"，需要先验分布。面试中要明确区分这两个概念。

---

## P12. p值与假设检验

**难度：** ⭐⭐ 中等 | **频率：** 极高

### 精确定义

$$p\text{-value} = P(\text{检验统计量} \geq T_{\text{obs}} \mid H_0 \text{ 为真})$$

**p 值不是：**
- "H₀ 为真的概率"
- "犯错的概率"  
- 与效应大小无关（大样本下微小效应也可 $p < 0.001$）

### 多重检验问题

检验 $m$ 个假设，每个 $\alpha=0.05$，至少一个假阳性的概率：

$$1 - (1-\alpha)^m \approx 1 - e^{-m\alpha}$$

$m=100$ 时：$1-(0.95)^{100} \approx 99.4\%$——几乎必然出现假阳性！

**校正方法：**

| 方法 | 控制目标 | 调整后 $\alpha$ |
|------|---------|--------------|
| Bonferroni | FWER（族错误率） | $\alpha/m$ |
| Benjamini-Hochberg | FDR（错误发现率） | 更宽松 |

> **Insight（量化回测的"p值黑客"）：** Harvey, Liu & Zhu (2016) 指出：150年来金融学已发现 300+ 个"显著"因子，若考虑多重检验，真正的显著性阈值应为 $t > 3.0$（$p < 0.003$），而非传统的 $t > 2.0$。**因子挖掘中的数据窥探偏差（data snooping）是 p 值黑客的典型场景。**

---

## P13. I型错误与II型错误

**难度：** ⭐⭐ 中等 | **频率：** 高

### 混淆矩阵

|  | H₀ 为真 | H₀ 为假 |
|--|---------|---------|
| **拒绝 H₀** | I型错误（$\alpha$，假阳性） | 正确（检验功效 $1-\beta$） |
| **不拒绝 H₀** | 正确（$1-\alpha$） | II型错误（$\beta$，假阴性） |

### 功效分析

检验功效 $1-\beta = P(\text{拒绝} H_0 \mid H_0 \text{为假})$，与效应量 $\delta$、样本量 $n$、$\alpha$ 有关：

$$1 - \beta \approx \Phi\!\left(\frac{|\delta|\sqrt{n}}{\sigma} - z_{\alpha/2}\right)$$

**所需样本量（给定功效 $1-\beta$）：**

$$n \geq \frac{(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}$$

例：$\alpha=0.05$，$1-\beta=0.80$，$z_{0.025}=1.96$，$z_{0.20}=0.84$，则 $n \geq 7.85\,\sigma^2/\delta^2$。

> **Insight（量化策略评估）：** 策略开发中 $\alpha = $ "策略无效但以为有效"（浪费资本），$\beta = $ "策略有效但未发现"（错失机会）。高频策略样本多、功效高；低频策略样本少、功效极低——这是低频策略难以统计显著的根本原因。

---

## P14. 鞅（Martingale）

**难度：** ⭐⭐⭐ 进阶 | **频率：** 高（衍生品岗）

### 定义

随机过程 $\{M_t, t \geq 0\}$ 是关于过滤 $\{\mathcal{F}_t\}$ 的**鞅**，若：

1. $M_t$ 是 $\mathcal{F}_t$-可测的（适应性）
2. $E[|M_t|] < \infty$
3. $E[M_{t+s} \mid \mathcal{F}_t] = M_t$，$\forall s > 0$（无预期漂移）

### 重要例子

| 过程 | 是否鞅 | 说明 |
|------|--------|------|
| $W_t$（标准布朗运动） | ✅ 鞅 | $E[W_{t+s}\mid\mathcal{F}_t]=W_t$ |
| $W_t^2 - t$ | ✅ 鞅 | $E[W_{t+s}^2\mid\mathcal{F}_t]=W_t^2+s$，减去 $s$ 后成鞅 |
| $e^{\sigma W_t - \frac{1}{2}\sigma^2 t}$（Girsanov 核） | ✅ 鞅 | 等价鞅测度变换的核心 |
| $S_t = S_0 e^{\mu t + \sigma W_t}$ | ❌（上鞅若$\mu<0$，下鞅若$\mu>0$） | 有漂移 |
| $\tilde{S}_t = e^{-rt}S_t$（风险中性贴现价格） | ✅ 鞅 | 无套利定价的核心 |

### 鞅与无套利定价

**第一基本定理：** 市场无套利 $\Leftrightarrow$ 存在等价鞅测度 $Q$，使得所有贴现资产价格在 $Q$ 下是鞅。

$$V_0 = e^{-rT} E^Q[V_T]$$

> **Insight：** 风险中性定价的本质：在真实测度 $P$ 下资产有正漂移（风险溢价），通过 Girsanov 定理变换到 $Q$ 测度，消除漂移，使贴现价格成鞅。**整个衍生品定价体系建立在鞅理论上。**

---

## P15. 伊藤引理（Itô's Lemma）

**难度：** ⭐⭐⭐ 进阶 | **频率：** 极高（衍生品岗）

### 定理

设 $X_t$ 满足 SDE：$dX_t = \mu(X_t,t)\,dt + \sigma(X_t,t)\,dW_t$

$f(X_t, t)$ 二阶连续可微，则：

$$\boxed{df = \left(\frac{\partial f}{\partial t} + \mu\frac{\partial f}{\partial X} + \frac{1}{2}\sigma^2\frac{\partial^2 f}{\partial X^2}\right)dt + \sigma\frac{\partial f}{\partial X}\,dW_t}$$

额外项 $\frac{1}{2}\sigma^2 f_{XX}$ 来自布朗运动的**二阶变差**：$(dW_t)^2 = dt$（非零！）

### 关键推导：$d(\ln S_t)$

设 $S_t$ 满足 GBM：$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$，令 $f = \ln S$：

$$\frac{\partial f}{\partial S} = \frac{1}{S}, \quad \frac{\partial^2 f}{\partial S^2} = -\frac{1}{S^2}, \quad \frac{\partial f}{\partial t} = 0$$

$$d(\ln S_t) = \frac{1}{S_t}\cdot \mu S_t\,dt + \frac{1}{2}\sigma^2 S_t^2 \cdot\left(-\frac{1}{S_t^2}\right)dt + \frac{1}{S_t}\cdot\sigma S_t\,dW_t$$

$$= \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma\,dW_t$$

所以 $\ln(S_T/S_0) \sim N\!\left((\mu-\tfrac{1}{2}\sigma^2)T,\;\sigma^2 T\right)$

### 推导 Black-Scholes PDE

设期权价格 $V(S,t)$，由伊藤引理：$dV = \left(V_t + \mu S V_S + \frac{1}{2}\sigma^2 S^2 V_{SS}\right)dt + \sigma S V_S\,dW_t$

构建 Delta 对冲组合 $\Pi = V - \Delta S$，选 $\Delta = V_S$ 消去随机项：

$$d\Pi = \left(V_t + \frac{1}{2}\sigma^2 S^2 V_{SS}\right)dt$$

无套利要求 $d\Pi = r\Pi\,dt$，得到 **Black-Scholes PDE**：

$$V_t + \frac{1}{2}\sigma^2 S^2 V_{SS} + rS V_S - rV = 0$$

> **Insight：** BS PDE 的推导完全依赖伊藤引理——如果没有 $\frac{1}{2}\sigma^2 S^2 V_{SS}$ 项，Delta 对冲将无法消去风险项，期权定价体系就不存在。伊藤引理是量化金融最重要的数学工具之一。

---

## P16. 蒙特卡洛模拟

**难度：** ⭐⭐ 中等 | **频率：** 高

### 欧式期权定价算法

```
1. 生成 N 条价格路径：
   S_T^{(i)} = S_0 · exp[(r - σ²/2)T + σ√T · Z_i], Z_i ~ N(0,1)

2. 计算每条路径的到期收益：
   C^{(i)} = max(S_T^{(i)} - K, 0)

3. 期权价格 = e^{-rT} · (1/N) · ΣC^{(i)}
```

**误差：** $\varepsilon \sim \frac{\sigma_C}{\sqrt{N}}$（$N=10000$ 时误差约为 $\sigma_C/100$）

### 方差缩减技术

**1. 对偶变量法（Antithetic Variables）：**
用 $Z_i$ 和 $-Z_i$ 成对模拟，利用负相关性降低方差：

$$\hat{C} = \frac{1}{2N}\sum_{i=1}^N [C(Z_i) + C(-Z_i)]$$

若 $\text{Cov}(C(Z), C(-Z)) < 0$，方差可减少至原来的 $\frac{1}{2}(1 + \rho)$

**2. 控制变量法（Control Variates）：**
用有解析解的类似期权（如欧式期权）校正蒙特卡洛估计：

$$\hat{C}_{\text{CV}} = \hat{C}_{\text{MC}} + \beta(C_{\text{exact}} - \hat{C}_{CV,\text{MC}})$$

最优 $\beta = -\text{Cov}(C, C_{CV})/\text{Var}(C_{CV})$

> **Insight：** 蒙特卡洛的最大优势是处理**路径依赖**期权（亚式期权、障碍期权、回望期权），这些期权没有解析解，PDE 方法在高维时也失效（维数灾难）。

---

## P17. VaR 与 CVaR

**难度：** ⭐⭐ 中等 | **频率：** 极高（风险管理岗）

### 定义

**VaR（Value at Risk）：** 置信水平 $1-\alpha$，持有期 $T$ 下：

$$\text{VaR}_\alpha = \inf\{l : P(L > l) \leq \alpha\} = F_L^{-1}(1-\alpha)$$

即亏损超过 VaR 的概率不超过 $\alpha$。

**CVaR/ES（Expected Shortfall）：**

$$\text{CVaR}_\alpha = E[L \mid L > \text{VaR}_\alpha] = \frac{1}{\alpha}\int_{\text{VaR}_\alpha}^{\infty} l\,f_L(l)\,dl$$

### VaR 的三种计算方法

| 方法 | 假设 | 优点 | 缺点 |
|------|------|------|------|
| 历史模拟 | 未来 = 历史 | 简单，无分布假设 | 样本依赖，无法外推 |
| 参数法（正态） | 收益正态分布 | 解析、快速 | 低估尾部风险 |
| 蒙特卡洛 | 指定模型 | 灵活 | 计算量大，模型风险 |

**正态假设下：** $\text{VaR}_\alpha = \mu + z_{1-\alpha}\,\sigma$（$z_{0.99} = 2.326$）

### VaR 的致命缺陷：非次可加性

$$\text{VaR}(A + B) > \text{VaR}(A) + \text{VaR}(B) \text{ 可能成立！}$$

这违反了风险分散化原则，而 CVaR 满足次可加性（Artzner 一致性风险度量的4条公理之一）。

> **Insight：** 2008年危机期间，银行持有大量 CDO，VaR 模型显示风险极低（历史数据无违约先例），但实际尾部损失是灾难性的。**CVaR 不仅告诉你损失超过阈值的概率，还告诉你平均会亏多少——这正是风险管理真正关心的。**

---

## P18. 偏度与峰度

**难度：** ⭐⭐ 中等 | **频率：** 高

### 定义

$$\text{Skewness} = \frac{E[(X-\mu)^3]}{\sigma^3}, \quad \text{Kurtosis} = \frac{E[(X-\mu)^4]}{\sigma^4}$$

**超额峰度（Excess Kurtosis）** = 峰度 $- 3$（正态分布峰度 $= 3$）

### 金融收益率的典型特征

| 特征 | 描述 | 含义 |
|------|------|------|
| 左偏（负偏度） | 左尾比正态更厚 | 极端亏损比正态预期更可能 |
| 超额峰度 $> 0$（尖峰厚尾） | 中间峰更高、尾部更厚 | 极端事件（黑天鹅）比正态分布更频繁 |

**Jarque-Bera 正态性检验：**

$$JB = \frac{n}{6}\left[\text{Sk}^2 + \frac{(\text{Ex.Kurt})^2}{4}\right] \sim \chi^2(2) \text{ under } H_0$$

> **Insight：** 标普500日收益率（1928-2020）超额峰度约 13，偏度约 $-0.5$。用正态分布预测的"百年一遇"下跌事件，实际上每20年就发生一次。**厚尾是所有风险模型必须认真对待的现实，不是可以简化的细节。**

---

## P19. Copula 与相依结构

**难度：** ⭐⭐⭐ 进阶 | **频率：** 中高（结构性产品岗）

### Sklar 定理

对任意联合分布 $F(x,y)$，存在 Copula $C$ 使得：

$$F(x, y) = C(F_X(x), F_Y(y))$$

即联合分布 = Copula（相依结构）$\times$ 边缘分布（分离的）。

### 常用 Copula 及其尾部相关性

| Copula | 上尾相关 $\lambda_U$ | 下尾相关 $\lambda_L$ | 适用场景 |
|--------|---------------------|---------------------|---------|
| 高斯 | 0 | 0 | 风险低估！ |
| t-Copula | $> 0$ | $> 0$ | 危机场景 |
| Gumbel（阿基米德） | $> 0$ | 0 | 正向尾部依赖 |
| Clayton | 0 | $> 0$ | 下行联动（危机） |

**尾部相关性定义：**

$$\lambda_U = \lim_{u \to 1} P(V > u \mid U > u), \quad \lambda_L = \lim_{u \to 0} P(V < u \mid U < u)$$

> **Insight（2008年的教训）：** Li (2000) 用高斯 Copula 定价 CDO，其尾部相关性为零——意味着极端情况下违约事件独立！但现实中金融危机时所有资产同步崩盘，下尾相关性接近1。**t-Copula 或 Clayton Copula 对危机场景的刻画要准确得多，代价是定价变得复杂。**

---

## P20. 最大似然估计（MLE）

**难度：** ⭐⭐ 中等 | **频率：** 高

### 定义

$$\hat{\theta}_{\text{MLE}} = \argmax_\theta L(\theta; \mathbf{x}) = \argmax_\theta \sum_{i=1}^n \ln f(x_i; \theta)$$

（取对数将乘积变为求和，不改变最优解）

### 正态分布的 MLE

对数似然：$\ell(\mu,\sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i-\mu)^2$

对 $\mu$ 求导：$\hat{\mu} = \bar{x}$（与 OLS 一致）

对 $\sigma^2$ 求导：$\hat{\sigma}^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$（有偏！分母为 $n$，非 $n-1$）

### MLE 的渐进性质

1. **一致性：** $\hat{\theta}_n \xrightarrow{p} \theta_0$
2. **渐进正态性：** $\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, \mathcal{I}(\theta_0)^{-1})$
3. **渐进有效性：** 达到 Cramér-Rao 下界，即 $\text{Var}(\hat{\theta}) \geq \frac{1}{n\mathcal{I}(\theta)}$

其中 Fisher 信息 $\mathcal{I}(\theta) = -E\!\left[\frac{\partial^2 \ln f}{\partial \theta^2}\right]$

> **Insight（连接 OLS 与 MLE）：** 当误差 $\varepsilon_i \sim \text{i.i.d.}\,N(0,\sigma^2)$ 时，线性回归的 MLE 完全等价于 OLS。对数似然最大化 $\Leftrightarrow$ 最小化残差平方和。这解释了为什么 OLS 在正态假设下是最优的。

---

# 第二部分：回归分析

---

## R1. OLS 的 Gauss-Markov 假设与 BLUE

**难度：** ⭐ 基础 | **频率：** 极高

### 五个假设

| # | 假设 | 表达式 | 违反后果 |
|---|------|--------|---------|
| GM1 | 线性模型 | $Y = X\beta + \varepsilon$ | 模型设定错误 |
| GM2 | 随机抽样 | 样本来自总体 | 选择偏差 |
| GM3 | 无完全多重共线性 | $\text{rank}(X) = k+1$ | $(X^\top X)$ 不可逆 |
| GM4 | 严格外生性 | $E[\varepsilon \mid X] = 0$ | **系数有偏且不一致** |
| GM5 | 同方差性 | $\text{Var}(\varepsilon \mid X) = \sigma^2 I$ | 标准误有偏，推断失效 |

**Gauss-Markov 定理：** 满足 GM1-5 $\Rightarrow$ OLS 是 BLUE（Best Linear Unbiased Estimator）。

**"Best"的精确含义：** 在所有线性无偏估计量中，OLS 的方差最小（Cramér-Rao 的线性版本）。

### 各假设的严重性排序

```
GM4（内生性）> GM3（完全多重共线）> GM5（异方差）> GM1（非线性）> GM2
     最严重                                                        较易修复
     偏误无法                                                       （大样本OK）
     靠n修复
```

> **Insight：** GM5 违反只影响推断（标准误错误），系数本身仍无偏；GM4 违反则连系数都是错的——这是本质区别。面试中必须能说清楚每个假设违反后的后果。

---

## R2. OLS 估计量的推导

**难度：** ⭐⭐ 中等 | **频率：** 极高

### 矩阵形式推导

**目标：** 最小化 $\text{RSS}(\beta) = (Y - X\beta)^\top(Y - X\beta)$

展开：

$$\text{RSS} = Y^\top Y - 2\beta^\top X^\top Y + \beta^\top X^\top X \beta$$

对 $\beta$ 求导并令为零：

$$\frac{\partial \text{RSS}}{\partial \beta} = -2X^\top Y + 2X^\top X \beta = 0$$

$$\Rightarrow \boxed{\hat{\beta} = (X^\top X)^{-1} X^\top Y}$$

### 几何解释

$\hat{Y} = X\hat{\beta} = X(X^\top X)^{-1}X^\top Y = H Y$，其中 $H = X(X^\top X)^{-1}X^\top$ 是**投影矩阵**（帽子矩阵）。

OLS 将 $Y$ 正交投影到 $X$ 的列空间，残差 $\hat{\varepsilon} = Y - \hat{Y} = (I-H)Y$ 与列空间正交：

$$X^\top \hat{\varepsilon} = X^\top(I-H)Y = X^\top Y - X^\top HY = X^\top Y - X^\top Y = 0$$

```
     Y
    /|
   / |  ε̂ ⊥ col(X)
  /  |
 /   |
Ŷ---+--------> col(X)
     投影
```

### 系数的统计性质

$$E[\hat{\beta}] = \beta \quad \text{（无偏）}$$

$$\text{Var}(\hat{\beta} \mid X) = \sigma^2 (X^\top X)^{-1} \quad \text{（GM1-5下）}$$

用 $\hat{\sigma}^2 = \frac{\text{RSS}}{n-k-1}$ 估计 $\sigma^2$（$n-k-1$ 为自由度，除以 $n-k-1$ 使其无偏）。

> **Insight：** $(X^\top X)^{-1}$ 的对角元素越大，对应系数的方差越大，估计越不精确。多重共线性使 $(X^\top X)$ 接近奇异，导致 $(X^\top X)^{-1}$ 的对角元素爆炸——这是多重共线性危害的代数本质。

---

## R3. R² 与调整 R²

**难度：** ⭐ 基础 | **频率：** 极高

### 定义

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{ESS}}{\text{TSS}}$$

其中 $\text{TSS} = \sum(y_i-\bar{y})^2$，$\text{ESS} = \sum(\hat{y}_i-\bar{y})^2$，$\text{RSS} = \sum\hat{\varepsilon}_i^2$

**调整 R²（惩罚变量数）：**

$$\bar{R}^2 = 1 - \frac{\text{RSS}/(n-k-1)}{\text{TSS}/(n-1)} = 1 - (1-R^2)\frac{n-1}{n-k-1}$$

### R² 的数学性质

**加入任何变量，R² 不会减少（即使变量无关）：**

设模型 M1（$k$ 个变量）和 M2（M1 加入变量 $z$），则 $\text{RSS}_{M2} \leq \text{RSS}_{M1}$（因为 OLS 总能令 $z$ 的系数为 0），故 $R^2_{M2} \geq R^2_{M1}$。

**调整 R² 的增减条件：** 加入变量后 $\bar{R}^2$ 增加 $\Leftrightarrow$ 新变量的 F 统计量 $> 1$，等价地，当新变量的 $t$ 统计量 $|t| > 1$。

### 样本内 vs 样本外 R²

$$R^2_{\text{OOS}} = 1 - \frac{\sum_{t \in \text{test}} (y_t - \hat{y}_t)^2}{\sum_{t \in \text{test}} (y_t - \bar{y}_{\text{train}})^2}$$

注意：样本外 $R^2$ **可以为负**（模型预测还不如直接用均值）！

> **Insight（量化警告）：** 金融预测模型的样本内 $R^2$ 往往很高（0.8+），但样本外 $R^2$ 可能接近零甚至为负。Goyal & Welch (2008) 发现，大量"显著"的股票收益预测变量，样本外预测能力约等于零。**高样本内 $R^2$ 是过拟合的信号，不是模型质量的保证。**

---

## R4. 多重共线性

**难度：** ⭐⭐ 中等 | **频率：** 高

### 后果的代数分析

简单两变量情形：$Y = \beta_1 X_1 + \beta_2 X_2 + \varepsilon$，设 $r_{12}$ 是 $X_1, X_2$ 的相关系数。

$$\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{\sum x_{1i}^2} \cdot \frac{1}{1-r_{12}^2}$$

**方差膨胀因子（VIF）：**

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

其中 $R_j^2$ 是 $X_j$ 对其他所有变量回归的 $R^2$。

| VIF | 解释 |
|-----|------|
| 1 | 无共线性 |
| 1–5 | 轻微 |
| 5–10 | 中度，需关注 |
| $> 10$ | 严重，标准误膨胀 $> 3$ 倍 |

### 处理方法的比较

| 方法 | 原理 | 代价 |
|------|------|------|
| 删除冗余变量 | 减少变量数 | 可能导致遗漏变量偏误 |
| Ridge 回归（L2） | 约束系数大小 | 引入偏误，换取方差降低 |
| PCA 降维 | 正交化特征 | 失去系数的直接解释性 |
| 增加样本量 | 提高信噪比 | 并不能根本解决问题 |

> **Insight：** 多重共线性的一个反直觉之处：**预测精度不受影响，只是系数估计不稳定**。如果目的是预测而非解释系数，多重共线性问题不那么严重。但若要解读"固定其他变量，$X_j$ 对 $Y$ 的独立效应"，则共线性会使这种解读完全失效。

---

## R5. 异方差性

**难度：** ⭐⭐ 中等 | **频率：** 高

### 后果的精确表述

异方差下，OLS 估计量 $\hat{\beta}$ 仍然无偏（因 GM4 未违反），但：

$$\text{Var}(\hat{\beta} \mid X) = (X^\top X)^{-1} \left(\sum_i \sigma_i^2 x_i x_i^\top\right) (X^\top X)^{-1} \neq \sigma^2(X^\top X)^{-1}$$

用标准公式 $\hat{\sigma}^2(X^\top X)^{-1}$ 估计方差是**错的**，导致 $t$ 和 $F$ 检验失效。

### White 稳健标准误（HC3）

$$\widehat{\text{Var}}_{\text{White}}(\hat{\beta}) = (X^\top X)^{-1} \left(\sum_i \hat{\varepsilon}_i^2 x_i x_i^\top\right) (X^\top X)^{-1}$$

**HC3 版本**（小样本修正）：用 $\frac{\hat{\varepsilon}_i^2}{(1-h_{ii})^2}$ 替代 $\hat{\varepsilon}_i^2$，其中 $h_{ii}$ 是帽子矩阵对角元素。

### 检验方法

**Breusch-Pagan 检验：** 将 $\hat{\varepsilon}_i^2$ 对 $X$ 回归，检验整体显著性（$F$ 或 $LM$ 检验）。

**White 检验：** 将 $\hat{\varepsilon}_i^2$ 对 $X$、$X^2$ 和所有交叉项回归（更一般）。

> **Insight（实践建议）：** 工业界标准做法是**直接使用稳健标准误，不做预检验**。因为：(1) 检验本身有犯错的概率；(2) 稳健标准误在同方差下也渐进有效；(3) 样本量够大时，稳健 SE 与 OLS SE 几乎一致。**没有理由不用稳健标准误。**

---

## R6. 序列相关（Autocorrelation）

**难度：** ⭐⭐ 中等 | **频率：** 高

### 定义与后果

$\text{Cov}(\varepsilon_t, \varepsilon_s) \neq 0$（$t \neq s$）

AR(1) 误差：$\varepsilon_t = \rho \varepsilon_{t-1} + u_t$，$|ρ| < 1$

**后果：** OLS 无偏，但**不再是 BLUE**，且标准 SE 低估真实 SE：

$$\text{Var}(\hat{\beta}_{OLS}) \approx \frac{\sigma^2}{n S_{xx}} \cdot \frac{1+\rho}{1-\rho}$$

当 $\rho > 0$ 时，真实方差比标准公式大 $\frac{1+\rho}{1-\rho}$ 倍——严重低估不确定性！

### Durbin-Watson 统计量

$$DW = \frac{\sum_{t=2}^n (\hat{\varepsilon}_t - \hat{\varepsilon}_{t-1})^2}{\sum_{t=1}^n \hat{\varepsilon}_t^2} \approx 2(1-\hat{\rho})$$

| DW 值 | 含义 |
|-------|------|
| $\approx 2$ | 无序列相关 |
| $< 2$（趋向0） | 正序列相关 |
| $> 2$（趋向4） | 负序列相关 |

### Newey-West HAC 标准误

$$\widehat{\text{Var}}_{\text{NW}}(\hat{\beta}) = (X^\top X)^{-1}\hat{\Omega}_{\text{NW}}(X^\top X)^{-1}$$

$$\hat{\Omega}_{\text{NW}} = \hat{\Gamma}_0 + \sum_{l=1}^{L} \left(1-\frac{l}{L+1}\right)(\hat{\Gamma}_l + \hat{\Gamma}_l^\top)$$

其中 $\hat{\Gamma}_l = \frac{1}{n}\sum_{t=l+1}^n \hat{\varepsilon}_t \hat{\varepsilon}_{t-l} x_t x_{t-l}^\top$，$L \approx T^{1/3}$ 或 $T^{2/5}$

> **Insight：** 时间序列金融数据几乎必然存在序列相关（收益率有动量，波动率有聚集）。**Newey-West SE 是时序回归的标配，不用 NW SE 就报 t 统计量是不专业的行为。** 在量化研究报告中，永远要注明标准误的类型。

---

## R7. 内生性（Endogeneity）

**难度：** ⭐⭐⭐ 进阶 | **频率：** 极高

### 三大来源与后果

**来源：**

| 来源 | 描述 | 金融例子 |
|------|------|---------|
| 遗漏变量 | 遗漏了与 $X$ 和 $Y$ 都相关的变量 | 公司治理影响杠杆和收益，遗漏治理变量 |
| 反向因果 | $X$ 影响 $Y$，同时 $Y$ 也影响 $X$ | 股价影响融资成本，融资成本影响股价 |
| 测量误差 | $X$ 被噪声测量，$X^* = X + \eta$ | Beta 估计有误差 |

**后果（OLS 系数偏误方向）：**

$$\text{plim}\,\hat{\beta}_1 = \beta_1 + \frac{\text{Cov}(X_1, \varepsilon)}{\text{Var}(X_1)}$$

遗漏变量 $Z$（对 $X_1$ 系数 $\pi$，对 $Y$ 系数 $\gamma$）：偏误 $= \pi \gamma$

**测量误差（衰减偏误）：**

$$\text{plim}\,\hat{\beta} = \beta \cdot \frac{\sigma_X^2}{\sigma_X^2 + \sigma_\eta^2} = \beta \cdot \frac{1}{1 + \sigma_\eta^2/\sigma_X^2} < \beta$$

系数被"衰减"（attenuation bias）向零——**Beta 估计值比真实 Beta 更小**。

### 工具变量（IV）

**条件：**
1. **相关性（Relevance）：** $\text{Cov}(Z, X) \neq 0$（可检验：第一阶段 F > 10）
2. **外生性（Exogeneity）：** $\text{Cov}(Z, \varepsilon) = 0$（不可检验，依赖经济假设）

**2SLS 步骤：**

第一阶段：$X = Z\pi + \nu$，得 $\hat{X} = Z\hat{\pi}$

第二阶段：$Y = \hat{X}\beta + \varepsilon$，得 $\hat{\beta}_{IV} = (\hat{X}^\top X)^{-1}\hat{X}^\top Y$

**IV 估计量的一致性：**

$$\text{plim}\,\hat{\beta}_{IV} = \beta + \frac{\text{Cov}(Z, \varepsilon)}{\text{Cov}(Z, X)} = \beta \quad (\text{若 } \text{Cov}(Z,\varepsilon)=0)$$

> **Insight（弱工具问题）：** 即使工具变量满足外生性，若相关性很弱（第一阶段 F < 10），IV 估计量有严重的有限样本偏误，且方差极大。**弱工具下 IV 估计甚至比有内生性的 OLS 更差**——这是 IV 使用中最常见的陷阱。

---

## R8. Ridge 回归与 Lasso 回归

**难度：** ⭐⭐⭐ 进阶 | **频率：** 极高

### 优化目标对比

$$\hat{\beta}_{\text{Ridge}} = \argmin_\beta \underbrace{\|Y - X\beta\|^2}_{\text{RSS}} + \lambda\underbrace{\|\beta\|_2^2}_{\text{L2惩罚}}$$

$$\hat{\beta}_{\text{Lasso}} = \argmin_\beta \|Y - X\beta\|^2 + \lambda\|\beta\|_1$$

### Ridge 的解析解

$$\hat{\beta}_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top Y$$

注意：即使 $X^\top X$ 奇异，加上 $\lambda I$ 后总可逆！这是 Ridge 的一大优势。

**系数收缩：** 若 $X$ 正交化，设 $X^\top X = I$，则 $\hat{\beta}_j^{\text{Ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1+\lambda}$——均匀压缩

### Lasso 的稀疏性（几何直觉）

```
Lasso（L1球，正方形）:        Ridge（L2球，圆形）:

    β₂                           β₂
     |  *                         |    *
     |   \  椭圆 (RSS等高线)       |   /
     |    \                       |  /
  *--+-----*  L1约束区域          --+---  L2约束区域
     |    /   (菱形，有角)         |  \   (圆形，无角)
     |   /                        |   \
     |  *                         |    *
     +--------β₁                  +--------β₁
  ↑                             ↑
解落在角上→β₁=0（稀疏！）       解通常不在轴上→无稀疏性
```

**为何 L1 产生稀疏性：** L1 约束区域（菱形/超正方形）在坐标轴上有尖点，椭圆等高线最容易在尖点处相切，使得某些系数精确为零。

### 超参数 $\lambda$ 的选择

用 K 折交叉验证最小化样本外预测误差：

$$\lambda^* = \argmin_\lambda \frac{1}{K}\sum_{k=1}^K \text{MSE}_k(\lambda)$$

**规律：**
- $\lambda \to 0$：趋向 OLS
- $\lambda \to \infty$：所有系数趋向 0

> **Insight（贝叶斯解释）：** Ridge = MAP 估计，先验为 $\beta_j \sim N(0, \sigma^2/\lambda)$（高斯先验，鼓励小系数）；Lasso = MAP 估计，先验为 $\beta_j \sim \text{Laplace}(0, 1/\lambda)$（拉普拉斯先验，尖峰导致稀疏性）。**正则化等价于给参数加先验——从频率派角度是收缩，从贝叶斯角度是信息融入。**

---

## R9. 系数显著性检验（t 检验）

**难度：** ⭐ 基础 | **频率：** 极高

### 检验统计量

$H_0: \beta_j = 0$ vs $H_1: \beta_j \neq 0$

$$t = \frac{\hat{\beta}_j}{\widehat{SE}(\hat{\beta}_j)} \sim t(n-k-1) \quad \text{under } H_0$$

$$\widehat{SE}(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 [(X^\top X)^{-1}]_{jj}}, \quad \hat{\sigma}^2 = \frac{\text{RSS}}{n-k-1}$$

### 统计显著性 vs 经济显著性

**统计显著性** 依赖样本量，大样本下微小效应也显著：

$$t \propto \frac{\hat{\beta}_j}{SE} \approx \frac{\hat{\beta}_j}{\sigma/\sqrt{n}} = \frac{\hat{\beta}_j \sqrt{n}}{\sigma}$$

$n$ 足够大时任何非零效应都显著。

**经济显著性** 看效应大小：系数改变1个标准差的 $X$ 对 $Y$ 的影响是否实际重要？

### 量化实践建议

| 指标 | 统计显著性门槛 | 量化实践建议 |
|------|--------------|-------------|
| $t$ 统计量 | $> 2.0$ | Harvey et al. (2016) 建议 $> 3.0$ |
| 样本外 Sharpe | — | $> 0.5$ 有实用价值 |
| 最大回撤 | — | $<$ 期望年化收益 |

> **Insight：** 在量化因子研究中，仅报告 $t > 2$ 的因子是不够的。应该同时报告：样本外表现、在不同市场状态下的稳健性、经济机制（为什么这个因子应该被定价？），以及在 Bonferroni 校正后是否仍显著。

---

## R10. F 检验

**难度：** ⭐⭐ 中等 | **频率：** 高

### 联合假设检验

$H_0: R\beta = r$（$q$ 个线性约束）

$$F = \frac{(R\hat{\beta} - r)^\top [R(X^\top X)^{-1}R^\top]^{-1} (R\hat{\beta} - r)/q}{\hat{\sigma}^2} \sim F(q, n-k-1) \quad \text{under } H_0$$

**整体回归显著性：** $H_0: \beta_1 = \beta_2 = \cdots = \beta_k = 0$

$$F = \frac{\text{ESS}/k}{\text{RSS}/(n-k-1)} = \frac{R^2/k}{(1-R^2)/(n-k-1)}$$

### F 与 t 的关系

对单个系数 $H_0: \beta_j = 0$，$F = t^2$（自由度为 $(1, n-k-1)$ 的 F 分布等于自由度为 $n-k-1$ 的 $t^2$）。

### Chow 检验（结构突变）

将样本分为两段（分裂点 $\tau$），检验系数是否跨期稳定：

$$F_{\text{Chow}} = \frac{(\text{RSS}_{\text{pooled}} - \text{RSS}_1 - \text{RSS}_2)/(k+1)}{(\text{RSS}_1 + \text{RSS}_2)/(n - 2k - 2)}$$

> **Insight（多因子模型）：** 当几个因子之间高度共线时，各因子的 t 检验可能都不显著，但联合 F 检验可能高度显著——说明这些因子作为一组是有预测力的，但无法分清各自的贡献。这正是因子降维（PCA）的动机。

---

## R11. 过拟合与信息准则

**难度：** ⭐⭐ 中等 | **频率：** 高

### 信息准则

$$\text{AIC} = 2k - 2\ln\hat{L}$$

$$\text{BIC} = k\ln n - 2\ln\hat{L}$$

（$k$ = 参数数，$n$ = 样本量，$\hat{L}$ = 最大化的似然值）

**BIC 对参数的惩罚更重**（当 $n > e^2 \approx 7.4$ 时，$\ln n > 2$，即 BIC 的单位惩罚超过 AIC），倾向于选择更简洁的模型，一致性强（当 $n \to \infty$ 时选择真实模型）。

### 偏差-方差分解视角

期望预测误差（测试误差）= 偏差² + 方差 + 不可约噪声

```
误差
  |         过拟合区域
  |  *          测试误差
  |    *      *
  |      *  *
  |        * ← 最优复杂度
  |      训练误差
  |    * * *
  +-------------------------> 模型复杂度（参数数量）
```

| 模型选择方法 | 估计测试误差方式 |
|------------|---------------|
| AIC | 渐进等于交叉验证（对大样本） |
| BIC | 贝叶斯模型选择，偏向简洁 |
| K 折 CV | 直接估计测试误差 |
| 样本外测试集 | 最可靠（但浪费数据） |

> **Insight（量化最佳实践）：** 模型选择的黄金标准是**样本外表现**，而非样本内指标。在因子选择中，将数据分为训练期（用于因子发现）和测试期（用于验证），且测试期必须在未来，避免前视偏差。**任何用了全部数据调参的模型，其样本内表现必然乐观。**

---

## R12. Fama-MacBeth 回归

**难度：** ⭐⭐⭐ 进阶 | **频率：** 极高（资产定价岗）

### 动机

资产定价检验中，同期资产收益高度相关（截面相关），直接 OLS 会低估标准误（高估 t 统计量）。

### 两步骤

**第一步（时序回归）：** 对每支股票 $i$，用全样本时序回归估计因子暴露（Beta）：

$$r_{it} = \alpha_i + \beta_i' f_t + \varepsilon_{it}, \quad t = 1, \ldots, T$$

**第二步（截面回归）：** 对每个时间点 $t$，将截面收益对 Beta 回归，得到因子风险溢价 $\hat{\lambda}_t$：

$$r_{it} = \gamma_{0t} + \hat{\beta}_i'\lambda_t + \eta_{it}, \quad i = 1, \ldots, N$$

**因子溢价估计：** 取时间序列均值

$$\hat{\lambda} = \frac{1}{T}\sum_{t=1}^T \hat{\lambda}_t$$

**t 统计量：** 用 $\hat{\lambda}_t$ 的时序标准误，**自动处理截面相关**：

$$t = \frac{\hat{\lambda}}{s(\hat{\lambda}_t)/\sqrt{T}}$$

### 与 GLS 的关系

当截面相关结构已知时，GLS 更有效；但 FM 回归的优势在于对截面相关结构**无需假设**，是半参数方法。

> **Insight：** Fama-French (1992, 1993) 正是用此方法发现了规模效应和价值效应。**FM 的本质是：用时序重复"同一个截面实验"，用重复实验的波动性来衡量估计的不确定性**，因此天然处理了截面相关——这个思路非常优雅。

---

## R13. 固定效应 vs 随机效应（面板数据）

**难度：** ⭐⭐⭐ 进阶 | **频率：** 高

### 面板数据模型

$$y_{it} = \alpha_i + X_{it}'\beta + \varepsilon_{it}$$

**固定效应（FE）：** $\alpha_i$ 是固定但未知的常数，与 $X_{it}$ 可能相关

**随机效应（RE）：** $\alpha_i \sim \text{i.i.d.}(0, \sigma_\alpha^2)$，与 $X_{it}$ 不相关

### FE 的 Within 变换

去除个体均值：$\tilde{y}_{it} = y_{it} - \bar{y}_i$，$\tilde{X}_{it} = X_{it} - \bar{X}_i$

$$\tilde{y}_{it} = \tilde{X}_{it}'\beta + \tilde{\varepsilon}_{it}$$

个体效应 $\alpha_i$ 被完全消除，估计 $\beta$ 无需关于 $\alpha_i$ 的假设。

**代价：** 无法估计时不变变量（性别、国家等）的系数（已被差掉）。

### Hausman 检验

$H_0: \text{Cov}(\alpha_i, X_{it}) = 0$（RE 一致且有效）

$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})^\top [\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE}) \sim \chi^2(k)$$

若拒绝 $H_0$：用 FE（即使效率低，至少一致）；若不拒绝：用 RE（更有效）。

> **Insight（量化面板应用）：** 股票面板数据（每只股票每月数据）中，FE 控制了公司固定特征（商业模式、行业）的影响，相当于每只股票自己做控制组——比跨股票比较更干净。但若研究的变量本身变化很慢（如书值/市值），within 变换会放大测量误差问题。

---

## R14. 分位数回归

**难度：** ⭐⭐⭐ 进阶 | **频率：** 中高

### 与 OLS 的根本区别

OLS：$\hat{\beta} = \argmin_\beta E[(Y - X\beta)^2]$，估计条件均值 $E[Y \mid X]$

分位数回归（$\tau$ 分位）：

$$\hat{\beta}(\tau) = \argmin_\beta E[\rho_\tau(Y - X\beta)]$$

其中 $\rho_\tau(u) = u(\tau - \mathbf{1}_{u<0})$（检验函数/折叠损失）

直观：$\rho_{0.5}(u) = |u|/2$（等价于中位数回归，对异常值更稳健）

### 几何理解

```
Y的分布（给定X）：
           ┌──────────────┐
           │   ▓▓▓▓│▓▓░░ │
           │ ▓▓▓▓▓▓│▓░░░ │
           │▓▓▓▓▓▓▓│░░░░ │
           └──────────────┘
OLS 拟合中位数/均值    ↑
分位数回归拟合 τ=0.9  ────

不同τ的斜率β(τ)可以不同！
→ 揭示 X 对 Y 分布形状的影响，不只是均值
```

### 量化金融应用

**尾部 VaR 估计：** $\text{VaR}_\alpha = Q_\alpha(L) = X\hat{\beta}(1-\alpha)$（线性 CVaR 模型）

**"超额收益与因子"分析：** 若因子对收益率中位数无影响，但对尾部（$\tau=0.1$）有强烈影响，均值回归会漏掉这个关系。

> **Insight：** 分位数回归的一个重要发现：**波动性高的股票（高 $\sigma$）在下行市场（$\tau=0.1$）跌得更惨，但在上行市场（$\tau=0.9$）涨得不多**——这在均值回归中表现为"高 Beta 低 Alpha"，但分位数回归能更清晰地刻画这种不对称性。

---

## R15. 协整（Cointegration）

**难度：** ⭐⭐⭐ 进阶 | **频率：** 高（统计套利岗）

### 单位根与非平稳

随机游走 $X_t = X_{t-1} + \varepsilon_t$ 是单位根过程（$I(1)$），非平稳：

$$X_t = X_0 + \sum_{s=1}^t \varepsilon_s, \quad \text{Var}(X_t) = t\sigma^2 \to \infty$$

**ADF（Augmented Dickey-Fuller）检验：** 检验单位根 $H_0: \rho = 1$

$$\Delta X_t = \alpha + (\rho-1)X_{t-1} + \sum_{j=1}^p \beta_j \Delta X_{t-j} + \varepsilon_t$$

### 协整的定义

$X_t \sim I(1)$，$Y_t \sim I(1)$，但存在 $\alpha$ 使得 $Y_t - \alpha X_t \sim I(0)$（平稳），则称 $X, Y$ 协整。

**误差修正模型（ECM）：**

$$\Delta Y_t = \gamma_0 + \gamma_1 (Y_{t-1} - \alpha X_{t-1}) + \gamma_2 \Delta X_{t-1} + u_t$$

误差修正项 $\gamma_1 < 0$：当偏离长期均衡时，以速度 $|\gamma_1|$ 修正回归。

### 配对交易（统计套利）流程

```
1. 寻找协整对：
   - 选择同行业股票（基本面逻辑）
   - ADF检验价差的平稳性（统计验证）

2. 建立交易信号：
   price_spread = Y_t - β̂X_t
   z_score = (spread - μ) / σ
   
3. 交易规则：
   z > +2 → 做空Y，做多X（价差回归）
   z < -2 → 做多Y，做空X
   z ∈ (-0.5, 0.5) → 平仓
```

> **Insight：** 协整关系会**衰减**——经济结构变化、竞争格局改变都会破坏长期均衡。对冲基金通常用滚动窗口不断重新估计协整参数，并设置止损防范协整关系突破风险。**均值回归是挣的协整维持的钱，方向交易是挣的协整破裂的钱——两者对立。**

---

## R16. 逻辑回归系数解释

**难度：** ⭐⭐ 中等 | **频率：** 高

### 模型设定

$$\ln\frac{P(Y=1\mid X)}{1-P(Y=1\mid X)} = \beta_0 + \beta_1 X_1 + \cdots + \beta_k X_k$$

$$P(Y=1\mid X) = \frac{1}{1+e^{-(\beta_0+X\beta)}} = \sigma(\beta_0+X\beta)$$

### 系数的正确解释

$\beta_j$ 是**对数几率（log-odds）**的斜率：

- $X_j$ 增加1单位 → $\log\text{-odds}$ 增加 $\beta_j$
- $X_j$ 增加1单位 → **几率（odds）乘以 $e^{\beta_j}$**（Odds Ratio）
- $X_j$ 增加1单位 → **概率的变化量取决于当前概率**（非线性）

**边际效应（在样本均值处）：**

$$\frac{\partial P}{\partial X_j} = \beta_j \cdot P(1-P) \bigg|_{X=\bar{X}}$$

### 与线性概率模型（LPM）的比较

| | 逻辑回归 | 线性概率模型 $P = X\beta$ |
|--|---------|----------------------|
| 预测概率范围 | $(0,1)$ | 可超出 $[0,1]$ |
| 系数解释 | 对数几率，非线性 | 直接是概率变化量，线性 |
| 异方差 | 自然解决 | 必然异方差（需稳健SE）|
| 估计方法 | MLE | OLS |

> **Insight（量化应用）：** 信用违约模型、欺诈检测、事件驱动策略中常用逻辑回归。**最常见的错误是把系数直接解释为"概率变化"，而非"对数几率变化"**。在汇报结果时，应转化为边际效应（AME 或 MEM）以获得直观的概率解释。

---

## R17. Newey-West 标准误

**难度：** ⭐⭐⭐ 进阶 | **频率：** 高

### HAC 估计量的构建

设 OLS 残差为 $\hat{\varepsilon}_t$，自协方差估计：

$$\hat{\Gamma}_l = \frac{1}{n}\sum_{t=l+1}^n \hat{\varepsilon}_t \hat{\varepsilon}_{t-l} x_t x_{t-l}^\top$$

Newey-West 估计量（Bartlett 核）：

$$\hat{\Omega}_{NW} = \hat{\Gamma}_0 + \sum_{l=1}^{L} w_l (\hat{\Gamma}_l + \hat{\Gamma}_l^\top), \quad w_l = 1 - \frac{l}{L+1}$$

**滞后截断阶数 $L$ 的选择：**
- 经验规则：$L = \lfloor 4(n/100)^{2/9} \rfloor$ 或 $L = \lfloor n^{1/3} \rfloor$
- Andrews (1991) 自适应方法（数据驱动）

### 为什么权重递减

Bartlett 核的权重 $w_l = 1-l/(L+1)$ 确保 $\hat{\Omega}_{NW}$ 为正定矩阵（必须正定才是合法的协方差矩阵估计）。简单截断核（$w_l=1$）可能产生非正定矩阵。

> **Insight（实践标准）：** 在因子模型月度回归中（T约 600 月），典型的 NW 滞后阶数约为 6–12 个月。**NW SE 通常比 OLS SE 大 20–50%，这意味着很多"显著"的因子实际上并不显著**——这是量化研究中夸大因子效应的一大来源。

---

## R18. 主成分回归（PCR）vs PLS

**难度：** ⭐⭐⭐ 进阶 | **频率：** 中

### PCA 降维

对 $X$（$n \times p$）做 SVD：$X = U\Sigma V^\top$

前 $k$ 个主成分：$T = XV_k$（$n \times k$ 的得分矩阵）

**方差解释比：** 第 $j$ 个主成分解释的方差比 $= \lambda_j/\sum \lambda_i$（$\lambda_j$ 是第 $j$ 大特征值）

PCR：$\hat{Y} = T\hat{\gamma}$，其中 $\hat{\gamma} = (T^\top T)^{-1}T^\top Y$

### PLS vs PCR 的方向

```
原始特征空间 X:
  PC₁方向 = X方差最大方向（PCR）
  PLS₁方向 = X对Y协方差最大方向（PLS）
  
两者一般不同！
→ PCR 的主成分可能与 Y 无关
→ PLS 的成分专注于预测 Y
```

### 何时选哪个

| 场景 | 推荐 |
|------|------|
| 仅关心降噪，Y 未知 | PCR |
| 需要预测 Y | PLS（通常更好）|
| 需要解释（因子归因） | PCR（PC 正交，含义清晰）|
| 变量极多，样本少 | Lasso（可变量选择）|

> **Insight（宏观预测）：** Stock & Watson (2002) 用大量宏观指标的主成分（扩散指数 DFM）预测 GDP、通胀，PCR 效果好于包含全部变量的回归。但后续研究发现，PLS 在金融变量预测中通常比 PCR 表现更好，因为宏观因子的真正预测力往往集中在少数与目标高度相关的成分上。

---

## R19. 虚假回归

**难度：** ⭐⭐ 中等 | **频率：** 高

### 理论结果（Granger & Newbold, 1974）

两个独立随机游走 $X_t$, $Y_t$：
$$X_t = X_{t-1} + \varepsilon_t, \quad Y_t = Y_{t-1} + u_t, \quad \text{Cov}(\varepsilon_t, u_t) = 0$$

理论上 $\hat{\beta}$ 应趋于 0，但实际上：

- $t$ 统计量 **不收敛**，随样本量增大而发散
- $R^2$ 趋向非零值（甚至接近1）
- DW 统计量趋向 0（强正序列相关）

**根本原因：** $I(1)$ 过程的回归不满足 CLT 条件，$t$ 统计量的极限分布不是 $t$ 分布，而是 Brownian Motion 的连续函数（Spurious Regression 分布）。

### 诊断清单

```
时间序列回归前必做：
□ 1. 对每个变量做 ADF 检验（单位根检验）
□ 2. 若 I(1)：检验是否协整（Engle-Granger 或 Johansen）
□ 3. 协整：用 ECM 模型；非协整：对差分后的序列回归
□ 4. 检查残差平稳性（如残差有单位根 → 虚假回归）
□ 5. 报告 DW 统计量（接近0是警报）
```

> **Insight：** 这是金融实证中极其常见的错误。**"GDP 增长与股市收益显著正相关"（两者都有趋势），"M2 增速与房价显著正相关"（两者都在增长）**——这些"显著"结果很可能是虚假回归。在声称发现任何宏观变量与金融变量的关系前，必须先检验平稳性。

---

## R20. 帽子矩阵与杠杆点

**难度：** ⭐⭐ 中等 | **频率：** 中

### 帽子矩阵的性质

$H = X(X^\top X)^{-1}X^\top$，对称幂等：$H^2 = H$, $H^\top = H$

- $\hat{Y} = HY$（预测值）
- $\hat{\varepsilon} = (I-H)Y$（残差）
- $0 \leq h_{ii} \leq 1$，$\sum h_{ii} = k+1$（迹 = 秩）

### 诊断指标

**杠杆值（Leverage）：** $h_{ii}$——观测点在 $X$ 空间中的极端程度

**标准化残差：** $r_i = \frac{\hat{\varepsilon}_i}{\hat{\sigma}\sqrt{1-h_{ii}}}$（消除异方差影响）

**Cook's Distance（影响力）：**

$$D_i = \frac{(\hat{\beta} - \hat{\beta}_{(-i)})^\top X^\top X (\hat{\beta} - \hat{\beta}_{(-i)})}{(k+1)\hat{\sigma}^2} = \frac{r_i^2}{k+1}\cdot\frac{h_{ii}}{1-h_{ii}}$$

$D_i > 4/n$ 或 $D_i > 1$ 通常认为是有影响力的观测点。

```
四种特殊观测点：
         高杠杆
            ↑
          × │ ×
 低Cook's D─┤─高Cook's D
          × │ ×
            ↓
          低杠杆

左上：高杠杆，X极端但在回归线上（低影响）
右上：高杠杆且高Cook's D（真正危险！）
右下：低杠杆，Y方向离群（异常值，影响较小）
```

> **Insight（量化数据清洗）：** 在因子回归中，个别月份的极端公司（刚上市的小市值股票，财务数据异常的公司）可能是高影响力观测点，严重扭曲系数估计。**标准的量化数据清洗流程包括：缩尾处理（winsorize）、剔除 $|r_i| > 3$ 的观测点，并报告去除异常值前后的系数稳健性。**

---

## R21. 截距项的作用

**难度：** ⭐ 基础 | **频率：** 中

### 截距的统计角色

有截距时，OLS 保证残差均值为零：$\sum_i \hat{\varepsilon}_i = 0$（由 $X^\top \hat{\varepsilon} = 0$ 的截距列得出）

无截距时此性质不成立，$R^2$ 的定义也发生变化（不再有 $R^2 \in [0,1]$ 的保证）。

### Jensen's Alpha

CAPM 回归：$r_i - r_f = \alpha + \beta(r_m - r_f) + \varepsilon$

截距 $\alpha$（Jensen's Alpha）= 超额收益，衡量主动管理的价值：
- $\alpha > 0$：组合跑赢风险调整后的基准
- $\alpha = 0$：CAPM 成立
- $\alpha < 0$：跑输

> **Insight：** 强制无截距（通过原点）回归在资产定价中有理论动机（若 CAPM 完全成立，$\alpha$ 应为0），但这是对模型的强烈约束。实践中，**始终保留截距**——错误省略截距比错误包含截距造成的损失大得多。

---

## R22. 交互项

**难度：** ⭐⭐ 中等 | **频率：** 中高

### 模型与解释

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \varepsilon$$

**$X_1$ 的边际效应（随 $X_2$ 变化）：**

$$\frac{\partial E[Y \mid X_1, X_2]}{\partial X_1} = \beta_1 + \beta_3 X_2$$

这是 $X_2$ 的线性函数——说明 $X_2$ 调节了 $X_1$ 对 $Y$ 的影响（调节变量/moderator）。

### 中心化的重要性

若不中心化，截距和主效应系数的解释变得极为奇怪：
- $\beta_1$ = $X_2 = 0$ 时 $X_1$ 的效应（$X_2=0$ 可能是样本范围外的无意义值）

中心化后（$\tilde{X}_j = X_j - \bar{X}_j$）：
- $\beta_1$ = $X_2$ 在均值处时 $X_1$ 的效应（有意义）
- 大幅降低 $X_1$ 与 $X_1 X_2$ 之间的多重共线性

### 量化因子交互

市场 Beta × VIX 水平：研究"在高波动市场中，高 Beta 股票是否跌得更多（下行 Beta > 上行 Beta）"。

$$r_{it} = \alpha + \beta_1 r_{mt} + \beta_2 \text{VIX}_t + \beta_3 r_{mt} \cdot \text{VIX}_t + \varepsilon_{it}$$

$\beta_3 > 0$：高 VIX 时，市场 Beta 的影响放大（系统性风险放大效应）。

> **Insight：** 交互项是捕捉**非线性效应**和**异质性效应**的最简单线性工具。在因子投资中，研究因子溢价在不同市场状态下的差异（regime-dependent alpha）本质上就是加入了"因子 × 状态"交互项。

---

## R23. 工具变量（IV）与 2SLS

**难度：** ⭐⭐⭐ 进阶 | **频率：** 高

### IV 估计量的方差

$$\text{Var}(\hat{\beta}_{IV}) = \sigma^2 (Z^\top X)^{-1} Z^\top Z (X^\top Z)^{-1}$$

与 OLS 相比：

$$\text{Var}(\hat{\beta}_{IV}) = \text{Var}(\hat{\beta}_{OLS}) \cdot \frac{1}{\rho_{ZX}^2}$$

其中 $\rho_{ZX}$ 是 $Z$ 与 $X$ 的相关系数。$\rho_{ZX}^2 < 1$ → IV 方差 > OLS 方差。

**弱工具（$\rho_{ZX} \approx 0$）：** 方差趋于无穷大！

### 检验与诊断

| 检验 | 目的 | 判断标准 |
|------|------|---------|
| 第一阶段 F 检验 | 检验工具相关性 | $F > 10$（Staiger & Stock 1997） |
| Sargan/Hansen J 检验 | 过度识别检验（工具有效性） | $p > 0.05$（无法拒绝外生性） |
| Durbin-Wu-Hausman | 检验是否真的有内生性 | 若不拒绝，OLS 更好 |

### 自然实验（Quasi-IV）

量化金融中难以找到完美 IV，常用"自然实验"：

- 指数成分调整作为机构持股的 IV（被动基金被迫买卖）
- 同行业其他公司的平均薪酬作为高管薪酬的 IV
- 随机分配的处理（A/B 测试）= 完美 IV

> **Insight：** IV 的逻辑是：用 $Z$ 的外生变动来"识别" $X$ 对 $Y$ 的因果效应。**IV 估计的是局部平均处理效应（LATE）**——只对那些因 $Z$ 变化而改变 $X$ 的个体（compliers）的处理效应，不一定代表整体平均处理效应（ATE）。这个细节在因果推断文献中极为重要。

---

## R24. Bootstrap 回归

**难度：** ⭐⭐ 中等 | **频率：** 中

### 两种 Bootstrap 方案

**1. 残差 Bootstrap（Parametric Bootstrap）：**
```
对 b = 1, ..., B:
  从 {ε̂₁, ..., ε̂ₙ} 有放回地抽取 {ε*₁, ..., ε*ₙ}
  构造 Y*ᵢ = X̂ᵢβ̂ + ε*ᵢ
  重新估计 β̂* = (X'X)⁻¹X'Y*
报告 β̂* 的经验分布
```

**2. 配对 Bootstrap（Wild Bootstrap）：**
```
对 b = 1, ..., B:
  从 {(x₁,y₁), ..., (xₙ,yₙ)} 有放回抽取 n 对
  重新估计 β̂*
```

配对 Bootstrap 对模型误设更稳健（不假设 $\varepsilon \sim$ 任何特定分布）。

### Bootstrap vs 解析标准误

| | 解析 SE | Bootstrap SE |
|--|---------|-------------|
| 需要分布假设 | 是（t分布） | 否 |
| 计算速度 | 快 | 慢（需 B 次估计）|
| 非标准统计量 | 不适用 | 适用 |
| 小样本 | 可能不准 | 通常更准 |

> **Insight（量化应用）：** Sharpe 比率的置信区间是量化中的典型 Bootstrap 应用——Sharpe 的精确抽样分布复杂（与收益分布有关），Bootstrap 可以直接估计。**月度数据 T=60 月，Bootstrap 95% CI 通常比解析 CI 宽 30-50%**，反映了真实的参数不确定性。

---

## R25. 样本选择偏差与 Heckman 校正

**难度：** ⭐⭐⭐ 进阶 | **频率：** 中

### 选择偏差的数学形式

若只观测到 $Y$ 当 $S_i = 1$（选择进入样本），且 $\text{Cov}(S, \varepsilon) \neq 0$，则：

$$E[Y_i \mid X_i, S_i=1] = X_i\beta + E[\varepsilon_i \mid S_i=1] = X_i\beta + \sigma_{\varepsilon\eta}\lambda(\alpha_0 + X_i\alpha_1)$$

其中 $\lambda(\cdot)$ 是**逆 Mills 比率（IMR）**，$\lambda(z) = \phi(z)/\Phi(z)$（$\phi, \Phi$ 分别是标准正态密度和分布函数）

### Heckman 两步法

**第一步（选择方程，Probit）：**

$$P(S_i=1 \mid Z_i) = \Phi(Z_i\alpha)$$

估计 $\hat{\alpha}$，计算 $\hat{\lambda}_i = \phi(Z_i\hat{\alpha})/\Phi(Z_i\hat{\alpha})$

**第二步（结果方程，OLS）：**

$$Y_i = X_i\beta + \rho\sigma_\varepsilon\hat{\lambda}_i + u_i$$

将 $\hat{\lambda}_i$ 作为额外控制变量，校正选择偏差。

> **Insight（量化中的存活偏差）：** 只研究现存基金（Survivorship Bias）会高估基金平均收益约 1-3%/年——失败的基金已从数据库中消失。**CRSP 数据库包含了退市公司，是避免存活偏差的标准选择；而 Morningstar 等商业数据库往往只有存活基金，不适合做表现研究。**

---

## R26. 虚假变量（Dummy Variables）与差断回归

**难度：** ⭐⭐ 中等 | **频率：** 高

### 虚假变量陷阱

$k$ 个类别需要 $k-1$ 个虚假变量（否则完全多重共线，矩阵不可逆）。

**不加截距时可以放 $k$ 个虚假变量**（此时每个系数是该类别的条件均值）。

### 差断回归（RDD, Regression Discontinuity Design）

**局部因果推断：** 若处理分配由阈值决定（$T_i = \mathbf{1}(X_i \geq c)$），则阈值两侧的个体除了处理状态外几乎相同，可以估计局部平均处理效应：

$$\hat{\tau}_{RDD} = \lim_{x \to c^+} E[Y \mid X=x] - \lim_{x \to c^-} E[Y \mid X=x]$$

```
     Y
      │           /  右侧（T=1）
      │          /
      │    跳跃│ ↑ 处理效应
      │  /     │
      │ /  左侧│（T=0）
      +─────────────> X
              阈值 c
```

> **Insight（量化事件研究）：** RDD 的思想在金融中对应：研究进入/退出指数成分（以市值或评分为阈值）对股票表现的影响——恰好在阈值两侧的股票提供了接近随机分配的自然实验。这是目前因果推断最严格的准实验方法之一。


---

# 第三部分：机器学习

---

## M1. 偏差-方差权衡

**难度：** ⭐ 基础 | **频率：** 极高

### 分解推导

设真实函数 $f(x)$，学习到的模型 $\hat{f}(x)$，噪声 $\varepsilon \sim (0, \sigma^2)$：

$$E[(y - \hat{f})^2] = \underbrace{[E[\hat{f}] - f]^2}_{\text{Bias}^2} + \underbrace{E[(\hat{f} - E[\hat{f}])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{不可约噪声}}$$

**推导：**

$$E[(y-\hat{f})^2] = E[(f+\varepsilon-\hat{f})^2] = E[(f-\hat{f})^2] + \sigma^2$$

$$E[(f-\hat{f})^2] = (f - E[\hat{f}])^2 + E[(\hat{f}-E[\hat{f}])^2] = \text{Bias}^2 + \text{Var}$$

### 各算法的偏差-方差位置

```
高方差          低方差
|               |
深度决策树   浅层决策树   线性回归（强正则）
KNN(k=1)     KNN(k=large)
神经网络(过拟合) 神经网络(正则化) 常数模型
|               |
低偏差          高偏差
```

### 集成方法的原理

**Bagging（降低方差）：** $B$ 个独立同分布模型，方差 $\sigma^2/B$；但若相关系数 $\rho > 0$：

$$\text{Var}_{\text{avg}} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

当 $B \to \infty$，极限是 $\rho\sigma^2$——降不到零，所以需要随机特征选择来降低 $\rho$（随机森林的动机）。

**Boosting（降低偏差）：** 每步拟合当前模型的残差（梯度方向），逐步降低偏差，但可能引入高方差（需要早停）。

> **Insight（量化模型选择）：** 金融数据信噪比极低（$R^2$ 通常 < 5%），主要挑战是**高方差（过拟合）而非高偏差**。因此，量化中倾向于用简单模型（线性、浅层树）配合强正则化，而非追求复杂的低偏差模型。**Gu et al. (2020) 发现，在资产定价中，经过强正则化的神经网络比随机森林好，但简单线性 Lasso 已经能抓住大部分可预测性。**

---

## M2. 交叉验证

**难度：** ⭐ 基础 | **频率：** 极高

### K 折 CV 的算法

```python
# K折CV伪代码
data = [(x₁,y₁), ..., (xₙ,yₙ)]
folds = split(data, K)
cv_errors = []

for k in range(K):
    train = folds except fold k
    val   = fold k
    model.fit(train)
    cv_errors.append(MSE(model.predict(val), val.y))

cv_score = mean(cv_errors)  # 模型选择标准
```

**偏差-方差分析：**
- $K=n$（LOOCV）：高方差（每次训练集差异小，预测高度相关），低偏差
- $K=5$ 或 $10$：偏差-方差的好平衡（经验推荐）

### 时间序列的时序 CV（Walk-Forward）

```
训练  ←────────────────────────→ 测试
[====================][────────]  第1折
[=======================][────]  第2折
[==========================][──]  第3折
                           ↑
                     只用历史数据，严格前向
```

**Gap（间隔）的必要性：** 在训练集末尾和测试集起始之间保留若干时间步，防止测试标签"泄漏"（例如用 t+1 的数据计算 t 时刻的特征）。

> **Insight：** 时序 CV 中，早期折的测试集误差通常低于晚期折（模型在早期市场环境更好拟合）——这是**非平稳性（nonstationarity）**的表现。若所有折误差相近，说明模型泛化性好；若有明显趋势，说明市场结构在变化，需要重新训练机制。

---

## M3. 随机森林

**难度：** ⭐⭐ 中等 | **频率：** 高

### 算法细节

```
随机森林训练算法：
for b = 1 to B:
  1. Bootstrap 采样: 从 n 个样本有放回地抽取 n 个样本
  2. 生长决策树（每次分裂时）:
     - 随机选 m 个特征（通常 m = √p 分类，m = p/3 回归）
     - 从 m 个特征中选最优分裂点
  3. 树生长至最大深度（或满足停止条件）

预测: 回归取平均，分类取投票
```

### OOB（Out-of-Bag）估计

每棵树约 $1/e \approx 36.8\%$ 的样本未被选中（OOB 样本）。

用 OOB 样本评估模型，无需额外验证集：

$$\text{OOB Error} = \frac{1}{n}\sum_{i=1}^n L(y_i, \hat{y}_i^{\text{OOB}})$$

其中 $\hat{y}_i^{\text{OOB}}$ 是所有未包含 $i$ 的树的预测均值。

### 特征重要性的陷阱

**不纯度重要性（MDI）的偏差：** 高基数特征（连续变量、高类别变量）会系统性地获得更高的 MDI 重要性，即使它们实际上没有预测力。

**排列重要性（MDA）** 更可靠但更慢：

$$\text{Imp}_j = \frac{1}{B}\sum_b [e_b - e_b^{\text{perm}_j}]$$

打乱特征 $j$ 后 OOB 误差的增量。

> **Insight（量化特征工程中）：** 使用 MDI 重要性筛选因子时，连续型宏观变量（利率水平）会比二元虚假变量（市场状态）显得更重要，但这可能是假象。**在量化中推荐使用 SHAP 值（同时具有正确的方向性和交互效应分解），而非单纯的特征重要性排名。**

---

## M4. XGBoost 与梯度提升

**难度：** ⭐⭐ 中等 | **频率：** 高

### 梯度提升的数学推导

目标：最小化 $\sum_i L(y_i, \hat{y}_i^{(t)})$，其中 $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$

在 $\hat{y}^{(t-1)}$ 处对损失函数做 Taylor 展开：

$$L^{(t)} \approx \sum_i [L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

其中 $g_i = \partial_{\hat{y}} L(y_i, \hat{y})$（一阶梯度），$h_i = \partial^2_{\hat{y}} L$（Hessian）

**XGBoost 的树结构正则化：**

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2$$

（$T$ = 叶子数，$w$ = 叶子权重，$\gamma, \lambda$ 是惩罚参数）

每个叶子节点的最优权重：$w_j^* = -\frac{G_j}{H_j + \lambda}$，增益：$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{G^2}{H+\lambda}\right] - \gamma$

### 关键超参数

| 超参数 | 作用 | 典型范围 |
|--------|------|---------|
| `n_estimators` | 树的数量（配合 early stopping） | 100-2000 |
| `learning_rate` | 每棵树的收缩系数 | 0.01-0.3 |
| `max_depth` | 树的最大深度（主要过拟合控制） | 3-8 |
| `subsample` | 样本子采样比例 | 0.5-1.0 |
| `colsample_bytree` | 特征子采样比例 | 0.5-1.0 |
| `min_child_weight` | 最小叶子权重和（控制最小样本） | 1-10 |

> **Insight（调参策略）：** 金融数据调参的反直觉经验：**浅树（max_depth=3-4）+ 大量树（n_estimators=500+）+ 小学习率（lr=0.01）** 通常优于深树。原因：金融信号微弱，深树会过拟合；浅树是弱学习器，需要更多棵；小学习率提高了集成的多样性。

---

## M5. 正则化方法对比

**难度：** ⭐⭐ 中等 | **频率：** 高

### 贝叶斯视角的统一

| 正则化 | 对应先验 | 几何约束 |
|--------|---------|---------|
| L2 (Ridge) | $\beta_j \sim N(0, 1/\lambda)$，高斯先验 | L2 球（无角，光滑） |
| L1 (Lasso) | $\beta_j \sim \text{Laplace}(0, 1/\lambda)$，拉普拉斯先验 | L1 球（有角，稀疏） |
| L0 | 均匀稀疏先验 | 组合约束（NP难） |
| Elastic Net | L1+L2 先验混合 | 介于两者之间 |

### Dropout 的集成解释

Dropout 等价于在每次前向传播时随机采样一个"子网络"。训练了 $T$ 步 $\approx$ 隐式训练了 $2^p$ 个共享权重的子网络的集成（$p$ = 神经元数）。

**测试时的 MC Dropout（不确定性估计）：** 保持 Dropout 开启，多次前向传播，用预测方差估计模型不确定性——对量化中的**预测置信度**很有价值。

> **Insight：** 深度学习中还有一种重要的正则化是**数据增强（Data Augmentation）**——对输入加随机扰动、旋转、噪声，等价于增加了训练样本多样性。对金融时序数据，可以：随机时间窗口偏移、加高斯噪声、用 SMOTE 平衡正负样本（用于分类任务）。**金融数据稀缺，数据增强是减少过拟合的重要手段。**

---

## M6. 支持向量机（SVM）

**难度：** ⭐⭐ 中等 | **频率：** 中

### 最大间隔分类器（硬间隔）

$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.}\; y_i(w^\top x_i + b) \geq 1, \; \forall i$$

等价对偶问题（凸 QP）：

$$\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^\top x_j \quad \text{s.t.}\; \alpha_i \geq 0, \; \sum_i \alpha_i y_i = 0$$

**KKT 条件：** $\alpha_i > 0 \Leftrightarrow$ 点在间隔边界上（支持向量）；其他点不影响决策边界。

### 核技巧

将 $x_i^\top x_j$ 替换为 $K(x_i, x_j) = \phi(x_i)^\top\phi(x_j)$，无需显式计算高维映射 $\phi$：

| 核函数 | 表达式 | 特点 |
|--------|--------|------|
| 线性 | $x_i^\top x_j$ | 线性决策边界 |
| 多项式 | $(x_i^\top x_j + c)^d$ | 多项式边界 |
| RBF/高斯 | $\exp(-\gamma\|x_i-x_j\|^2)$ | 局部，无限维特征空间 |
| Sigmoid | $\tanh(\alpha x_i^\top x_j + c)$ | 类似神经网络 |

> **Insight：** SVM 的核心思想——**对偶性 + 核技巧**——比 SVM 本身更重要。核方法（Gaussian Process、Kernel PCA）都利用了同样的思路：在高维空间中线性操作，通过核函数避免显式计算。在深度学习兴起前，核方法是非线性函数估计的主流工具。

---

## M7. 数据泄漏与时序特征工程

**难度：** ⭐⭐⭐ 进阶 | **频率：** 极高（量化必考）

### 数据泄漏的三种类型

**1. 标签泄漏（Target Leakage）：** 特征中包含了标签的未来信息

```python
# 错误示例：用T+1的信息预测T时刻的收益
X['next_day_volume'] = df['volume'].shift(-1)  # 未来数据！

# 正确：只用T时刻及之前的数据
X['prev_day_volume'] = df['volume'].shift(1)
```

**2. 训练-测试污染：** 用全样本数据做预处理

```python
# 错误：用全样本归一化
scaler = StandardScaler().fit(X_all)  # 泄漏了测试集统计量！

# 正确：只用训练集归一化
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # 用训练集的均值/标准差
```

**3. 前视偏差（Look-ahead Bias）：**

```
公司 T 季度的财务数据通常在 T+2 个月才公布！
用 T 时刻的 P/E（用了T季报，实际T+2才可用）预测 T 时刻收益 → 前视偏差

解决：使用"点时间（point-in-time）"数据，记录每个数据何时实际可获取
```

### 时序 Pipeline 设计

```
原始数据流:
  ...t-2, t-1, t, t+1, t+2...

特征工程（只用 ≤ t 的数据）:
  moving_avg_20d(t) = mean(price[t-19:t])   # OK
  rolling_beta(t)   = β fitted on [t-60:t]  # OK

标签:
  label(t) = return(t+1)/return(t+5)        # 必须在预测时刻之后

训练/验证分割:
  train: [t₀, t_split)
  gap:   [t_split, t_split + gap)            # 防止信息泄漏
  test:  [t_split + gap, t_end)
```

> **Insight（工业实践）：** Point-in-time 数据管理是量化机构的核心基础设施。错误地用了"后复权"价格（而非当时的实际价格）、未考虑财务数据发布延迟、用了后来才知道的行业分类——**这些错误在回测中可能产生 Sharpe 3+，实盘中立刻亏损。数据泄漏是量化研究中最昂贵的错误之一。**

---

## M8. 特征重要性与 SHAP

**难度：** ⭐⭐ 中等 | **频率：** 高

### SHAP 值的博弈论基础

Shapley 值：将特征的预测贡献分配给每个特征，满足：

1. **有效性：** $\sum_j \phi_j = f(x) - E[f]$（贡献之和等于预测值减基线）
2. **对称性：** 相同贡献的特征获得相同 Shapley 值
3. **虚特征：** 无贡献特征的 Shapley 值为零
4. **线性：** 两个模型的和的 Shapley 值等于各自 Shapley 值的和

**计算公式：**

$$\phi_j = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(|\mathcal{F}|-|S|-1)!}{|\mathcal{F}|!} [f_{S\cup\{j\}}(x) - f_S(x)]$$

（穷举所有不包含 $j$ 的特征子集 $S$，计算加入 $j$ 的边际贡献加权平均）

### SHAP 的可视化工具

| 图表类型 | 信息 |
|---------|------|
| SHAP 瀑布图 | 单个预测的特征贡献分解 |
| SHAP 蜂群图（beeswarm） | 所有样本的 SHAP 分布（显示非线性） |
| SHAP 依赖图 | 特定特征的 SHAP 值 vs 特征值（检测交互）|
| SHAP 全局重要性 | $\text{mean}(|\phi_j|)$（无方向性的全局重要性）|

> **Insight（量化因子归因）：** 在多因子模型中，SHAP 可以对**每笔交易**分解"哪个信号贡献了这笔交易的超额收益"。这对风险归因（attribution）极有价值：基金经理可以看到动量信号 vs 价值信号各自贡献了多少收益，而不只是整体 Alpha。**SHAP 是连接复杂模型与可解释因子框架的桥梁。**

---

## M9. 梯度消失与爆炸

**难度：** ⭐⭐⭐ 进阶 | **频率：** 中高

### 数学根源

$L$ 层网络，反向传播：

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_L} \cdot \prod_{l=2}^{L} \frac{\partial a_l}{\partial a_{l-1}}$$

每层贡献 $\frac{\partial a_l}{\partial a_{l-1}} = W_l \odot \sigma'(z_{l-1})$

**Sigmoid 激活的消失问题：** $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$

当 $L=100$：$\prod_l \sigma'_l \leq 0.25^{100} \approx 10^{-60}$ → 梯度消失至数值零

**ReLU 激活：** $\text{ReLU}'(z) = 1 \text{ (if } z>0\text{), else } 0$ → 正区间梯度不消失

**Dying ReLU 问题：** 若某神经元始终在 $z \leq 0$ 区域，梯度永远为零，该神经元"死亡"。Leaky ReLU/ELU 可修复。

### 解决方案汇总

| 方法 | 机制 | 主要解决 |
|------|------|---------|
| BatchNorm | 规范化激活值分布，防止饱和 | 消失 |
| 残差连接（ResNet） | 添加跳跃连接，梯度直接反传 | 消失 |
| 梯度裁剪（clip norm） | 限制梯度向量的L2范数 | 爆炸 |
| Xavier/He 初始化 | 合理初始化权重方差 | 两者 |
| LSTM/GRU 门控 | 门控机制维持长程梯度通路 | 时序消失 |

> **Insight：** ResNet 的残差连接 $y = F(x) + x$ 提供了"高速公路"：梯度可以直接从深层传回浅层，使训练超过100层的网络成为可能。**这一结构创新比任何激活函数或优化算法的改进影响都大**——它解决了深度网络训练的根本障碍，是深度学习近十年最重要的结构贡献之一。

---

## M10. K 均值聚类

**难度：** ⭐⭐ 中等 | **频率：** 中

### 算法与收敛性

```
K-means 算法:
初始化: 随机选 k 个质心 μ₁, ..., μₖ
重复直到收敛:
  E步（分配）: zᵢ = argmin_k ||xᵢ - μₖ||²
  M步（更新）: μₖ = mean({xᵢ: zᵢ = k})
```

**收敛保证：** 每次迭代目标函数（总组内平方和 WCSS）单调不增，且有下界，因此必然收敛。但可能收敛到局部最优。

**初始化策略——K-means++：**

1. 随机选第一个质心
2. 按距离已有质心的平方距离概率选下一个质心（远点更容易被选）
3. 重复直到选够 k 个

K-means++ 的期望目标函数值在 $O(\log k)$ 内接近最优（比随机初始化好得多）。

### 选择 K 的方法

**Elbow 方法：** 画 WCSS vs K 的曲线，找"肘部"（边际改善急剧下降处）

**轮廓系数（Silhouette Score）：**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$

$a(i)$：点 $i$ 到同簇其他点的平均距离；$b(i)$：点 $i$ 到最近其他簇的平均距离。

$s(i)$ 越接近 1 越好（簇内紧密，簇间分离）。

> **Insight（量化应用：市场状态识别）：** K-means 将市场分为 K 个"状态"（如牛市/熊市/震荡），但 K-means 假设球形簇。金融收益分布是椭球形（因为协方差矩阵非单位阵）。**高斯混合模型（GMM）= 软性 K-means，允许椭球形簇，Hamilton 区制转换模型正是 GMM 的时序版本**，在宏观策略和风险管理中被广泛使用。

---

## M11. 强化学习与量化交易

**难度：** ⭐⭐⭐ 进阶 | **频率：** 中（前沿岗位）

### RL 框架

$$\text{智能体} \xrightarrow{\text{动作 } a_t} \text{环境} \xrightarrow{\text{状态 } s_{t+1},\text{奖励 } r_t} \text{智能体}$$

**目标：** 最大化累计折扣奖励 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$

**Bellman 方程：** $Q^*(s,a) = r + \gamma \max_{a'} Q^*(s', a')$

### 量化交易的 RL 应用

| 应用 | 状态空间 | 动作空间 | 奖励函数 |
|------|---------|---------|---------|
| 最优执行 | 剩余持仓、时间、市场状态 | 每时刻执行量 | 实现价格 - VWAP |
| 动态对冲 | Delta、Gamma、Vega 暴露 | 对冲调整量 | 对冲误差 + 交易成本 |
| 投资组合管理 | 因子暴露、市场状态 | 目标权重向量 | Sharpe 或 $r - \lambda \cdot \text{turnover}$ |

### 金融 RL 的特殊挑战

1. **非平稳性：** 市场规律随时间变化，策略需要持续适应
2. **奖励延迟：** 今天的决策可能要几个月后才知道好坏
3. **部分可观测性：** 市场状态无法完全观测（POMDP 而非 MDP）
4. **历史数据有限：** 金融数据量与图像/游戏相比极小，样本效率是核心约束

> **Insight：** RL 在**最优执行**领域最为成功（Almgren-Chriss 模型的 RL 改进），因为执行问题的定义清晰、奖励明确、可以大量模拟。在更广泛的选股/配置应用中，RL 目前仍难以超越经典因子模型——主要障碍是金融市场的极低信噪比和有限样本量，而不是算法本身的局限。

---

## M12. 分类模型评估指标

**难度：** ⭐⭐ 中等 | **频率：** 高

### 混淆矩阵与衍生指标

|  | 预测正 | 预测负 |
|--|--------|--------|
| **实际正** | TP | FN |
| **实际负** | FP | TN |

$$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P+R}$$

$$\text{Accuracy} = \frac{TP+TN}{N} \quad \text{（类别不平衡时无意义）}$$

### ROC 曲线与 AUC

**ROC 曲线：** 以不同分类阈值 $\tau \in [0,1]$，画 FPR vs TPR：

$$\text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau)+TN(\tau)}, \quad \text{TPR}(\tau) = \text{Recall}(\tau) = \frac{TP(\tau)}{TP(\tau)+FN(\tau)}$$

**AUC 的概率解释：** $\text{AUC} = P(\hat{p}_\text{pos} > \hat{p}_\text{neg})$（随机取一个正例和一个负例，模型正确排序的概率）

### 类别不平衡时的正确指标

金融中违约/欺诈事件通常是 1% 的稀有事件，此时：

- **不要用 Accuracy**（永远预测"不违约"就有 99% accuracy）
- **用 PR-AUC**（Precision-Recall 曲线下面积），对少数类更敏感
- **用 F1 score** 或 **F-beta**（$\beta>1$ 更强调 Recall）

> **Insight（量化信号评估）：** 预测股票涨跌（50/50 平衡），AUC=0.52 看似微弱，但在 Sharpe 比率上可能对应 0.5 以上的年化超额收益。**在量化中，AUC 的绝对值没有 P&L 重要——0.52 的 AUC 配合好的执行策略也能盈利，0.60 的 AUC 若没有考虑交易成本可能实盘亏损。**

---

## M13. ML 因子挖掘

**难度：** ⭐⭐⭐ 进阶 | **频率：** 高（研究岗）

### 传统因子 vs ML 因子

**传统线性因子模型：**

$$r_{it} = \alpha_i + \sum_k \beta_{ik} f_{kt} + \varepsilon_{it}$$

因子 $f_k$ 手工设计（市场、规模、价值、动量……），线性叠加

**ML 方法的贡献（Gu, Kelly, Xiu 2020）：**

$$r_{it+1} = g(z_{it}) + \varepsilon_{it+1}$$

用神经网络/随机森林学习 $g(\cdot)$（允许非线性交互），在美国股市超额月收益预测中：

| 方法 | 月均超额收益（多空） |
|------|---------------------|
| OLS（线性） | ~0.3% |
| PLS | ~0.4% |
| Lasso/Ridge | ~0.4% |
| 随机森林 | ~0.5% |
| **神经网络（强正则）** | **~0.6%** |

### ML 因子的衰减问题

```
因子发现 → 学术发表 → 机构套利 → 因子衰减
    ↑__________________________|
         信息套利循环

典型衰减时间：
  低换手率价值因子：数年
  中频技术因子：6-18个月
  高频微结构因子：数周-数月
```

> **Insight：** Gu et al. (2020) 的关键发现：**超额收益主要来自非线性特征交互**（价值 × 动量 × 质量的复杂组合），而非任何单一因子的非线性变换。这说明因子组合的协同效应比单因子的精细建模更重要。但文章也承认，大部分可预测性发生在小市值股票——实际容量有限，高换手成本后超额收益大幅压缩。

---

## M14. 降维方法对比

**难度：** ⭐⭐ 中等 | **频率：** 中

### PCA 的数学推导

**目标：** 找方向 $w_1$（单位向量），最大化投影方差：

$$w_1 = \argmax_{\|w\|=1} \text{Var}(Xw) = \argmax_{\|w\|=1} w^\top \Sigma w$$

由 Lagrange 乘子法：$\Sigma w = \lambda w$，即 $w_1$ 是 $\Sigma$ 的最大特征向量。

**Scree Plot（碎石图）：** 画特征值 $\lambda_1 \geq \lambda_2 \geq \ldots$ 的折线，选"肘部"前的主成分数。

**Kaiser 准则：** 保留 $\lambda_j > 1$（方差大于均值的主成分）

### 各方法的适用场景

| 方法 | 类型 | 保留结构 | 适用场景 |
|------|------|---------|---------|
| PCA | 线性 | 全局方差 | 因子降维、风险归因 |
| Kernel PCA | 非线性 | 核空间方差 | 非线性结构 |
| Autoencoder | 非线性 | 重建误差 | 另类数据压缩 |
| t-SNE | 非线性 | 局部邻居结构 | 可视化（2D/3D） |
| UMAP | 非线性 | 局部+全局 | 快速可视化 |
| NMF | 非负线性 | 加法分解 | 话题模型、文本 |

> **Insight（量化风险归因）：** 股票收益协方差矩阵的 PCA，通常第一主成分（市场因子）解释 30-50% 的总方差，前5个主成分解释 50-70%。**这些主成分对应 Fama-French 因子（规模、价值）等，PCA 提供了"数据驱动的因子提取"，不依赖先验理论**。在无法直接观测因子载荷时，PCA 是统计因子模型的标准方法。

---

## M15. 神经网络架构基础

**难度：** ⭐⭐ 中等 | **频率：** 中高

### 前馈网络（MLP）的表达能力

**通用近似定理：** 具有一个隐藏层的 MLP，使用足够多的神经元，可以以任意精度逼近任何连续函数。

但"足够多"可能是指数级的神经元数——这是为什么需要深层网络（指数压缩需要的神经元数）。

### 量化中的时序模型

**LSTM 的门控机制：**

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{遗忘门}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{输入门}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c)$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{输出门}$$

遗忘门 $f_t$ 决定保留多少历史记忆，解决了 RNN 的梯度消失问题。

**Transformer（注意力机制）：**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

对金融时序：注意力权重可以自适应地关注最相关的历史时间步（如财报发布日、联储会议）。

> **Insight：** 深度学习在量化中的现实局限：**金融时序数据量（~20年月度，~240条）远小于 CV/NLP 中的数百万样本**。简单 LSTM/Transformer 在小数据集上比 XGBoost 表现差，需要极强正则化（Dropout, early stopping, L2）。深度学习在量化中的真正价值是**处理另类数据**（卫星图像、财报文本、新闻语义），这些数据量大、原始特征高维，正是深度学习的优势所在。

---

## M16. 时间序列预测模型

**难度：** ⭐⭐ 中等 | **频率：** 高

### ARIMA 模型

$ARIMA(p, d, q)$：

$$\Phi(L)(1-L)^d y_t = \theta(L)\varepsilon_t$$

- $p$：AR 阶数（自回归，用 PACF 确定）
- $d$：差分阶数（使序列平稳）
- $q$：MA 阶数（移动平均，用 ACF 确定）

**Box-Jenkins 建模步骤：**

```
1. 平稳性检验（ADF）
   → 非平稳则差分至平稳
2. ACF/PACF 图确定 (p, q)
   → AR(p): PACF截尾，ACF拖尾
   → MA(q): ACF截尾，PACF拖尾
3. 估计参数（MLE）
4. 残差诊断（白噪声检验：Ljung-Box）
5. 预测
```

### GARCH 模型（波动率聚集）

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

条件：$\alpha + \beta < 1$（平稳），$\alpha + \beta$ 越接近1，波动率冲击越持久。

**重要推论：** $\sum_{k=0}^{\infty}$ GARCH 的波动率长期预测收敛到无条件方差 $\bar{\sigma}^2 = \omega/(1-\alpha-\beta)$，即均值回归。

> **Insight：** ARIMA 解决均值建模（收益率），GARCH 解决方差建模（波动率）。**现实中，收益率几乎不可预测（ARIMA 几乎无用），但波动率高度可预测（GARCH 效果好）**——这是市场效率的直接体现。VIX 隐含波动率模型、风险平价策略都依赖波动率预测的可靠性。

---

## 附录：量化面试常见数学工具速查

### 常用分布

| 分布 | 记号 | E[X] | Var(X) | 金融应用 |
|------|------|------|---------|---------|
| 正态 | $N(\mu,\sigma^2)$ | $\mu$ | $\sigma^2$ | 收益率、因子 |
| 对数正态 | $LN(\mu,\sigma^2)$ | $e^{\mu+\sigma^2/2}$ | $(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$ | 股票价格 |
| 泊松 | $\text{Pois}(\lambda)$ | $\lambda$ | $\lambda$ | 违约事件、订单到达 |
| 指数 | $\text{Exp}(\lambda)$ | $1/\lambda$ | $1/\lambda^2$ | 事件间隔 |
| Beta | $\text{Beta}(\alpha,\beta)$ | $\frac{\alpha}{\alpha+\beta}$ | 复杂 | 次序统计量、贝叶斯 |
| $\chi^2(k)$ | | $k$ | $2k$ | 卡方检验 |
| $t(k)$ | | 0 | $k/(k-2)$ | t检验（小样本） |
| $F(d_1,d_2)$ | | 复杂 | 复杂 | F检验、回归 |

### 关键不等式

$$P(|X-\mu| \geq k\sigma) \leq \frac{1}{k^2} \quad \text{（Chebyshev，不需分布假设）}$$

$$P(X \geq t) \leq \frac{E[X]}{t} \quad \text{（Markov，X≥0）}$$

$$P\left(\left|\frac{1}{n}\sum X_i - \mu\right| \geq \varepsilon\right) \leq \frac{\sigma^2}{n\varepsilon^2} \quad \text{（WLLN证明）}$$

### 矩阵求导速查

| 形式 | 结果 |
|------|------|
| $\partial (Ax)/\partial x$ | $A^\top$ |
| $\partial (x^\top A)/\partial x$ | $A$ |
| $\partial (x^\top Ax)/\partial x$ | $(A+A^\top)x$（对称时为 $2Ax$）|
| $\partial \ln\det(A)/\partial A$ | $A^{-\top} = (A^{-1})^\top$ |
| $\partial \text{tr}(AB)/\partial A$ | $B^\top$ |

---

*整理完毕。共 62 道题，含完整数学推导、几何直觉与量化金融 Insights。*

*建议学习顺序：概率论（P1-P10）→ 回归基础（R1-R4, R9-R11）→ 回归进阶（R5-R8, R12-R15）→ 机器学习（M1-M6）→ 量化专项（P11-P20, R16-R26, M7-M16）*
