# Quant 面试：条件期望计算完全指南

> **核心考察点：** 条件期望、Tower Law、矩计算、顺序统计量、鞅与停时
> **适用岗位：** Quant Researcher、Quant Trader、Risk Analyst、Stochastic Calculus 相关岗位
> **公式渲染：** 需要支持 LaTeX 的编辑器（Obsidian / Typora / VS Code + Markdown Preview Enhanced）

---

## 核心工具速查

在开始刷题前，先把这几个核心工具牢记于心：

### 工具 1：联合正态的条件期望

若 $(X, Y)$ 联合正态，则：
$$E[X \mid Y = y] = \mu_X + \frac{\text{Cov}(X, Y)}{\text{Var}(Y)}(y - \mu_Y)$$
$$X \mid Y=y \sim N\!\left(\mu_X + \frac{\rho\sigma_X}{\sigma_Y}(y-\mu_Y),\; \sigma_X^2(1-\rho^2)\right)$$

### 工具 2：Tower Law（迭代期望定律）

$$\boxed{E[X] = E\bigl[E[X \mid Y]\bigr]}$$

推广（方差分解，Eve's Law）：
$$\text{Var}(X) = E[\text{Var}(X \mid Y)] + \text{Var}(E[X \mid Y])$$

### 工具 3：逆 Mills 比率（截断正态）

若 $Z \sim N(0,1)$：
$$E[Z \mid Z > a] = \frac{\phi(a)}{1 - \Phi(a)} \equiv \lambda(a)$$
$$E[Z \mid Z < a] = -\frac{\phi(a)}{\Phi(a)}$$

### 工具 4：矩母函数（MGF）

$$M_X(t) = E[e^{tX}], \quad E[X^n] = M_X^{(n)}(0)$$

正态 $N(\mu, \sigma^2)$：$M(t) = e^{\mu t + \sigma^2 t^2/2}$

### 工具 5：顺序统计量期望

$X_1, \ldots, X_n \stackrel{iid}{\sim} U[0,1]$，第 $k$ 小的次序统计量：
$$E[X_{(k)}] = \frac{k}{n+1}$$

### 工具 6：鞅与 Optional Stopping

若 $M_t$ 是鞅，$\tau$ 是满足 Doob 条件的停时：
$$E[M_\tau] = E[M_0]$$

常用鞅：$W_t$（标准 BM），$W_t^2 - t$，$e^{\theta W_t - \theta^2 t/2}$

---

## 第一章：高斯分布相关期望

---

### 题目 G1 ⭐（经典必备）

**题：** $X, Y \overset{iid}{\sim} N(0,1)$，求 $E[X \mid X+Y = s]$

**解：**

令 $S = X + Y$，则 $(X, S)$ 是联合正态（线性变换保持正态性）：
- $E[X] = 0,\quad E[S] = 0$
- $\text{Var}(X) = 1,\quad \text{Var}(S) = \text{Var}(X) + \text{Var}(Y) = 2$
- $\text{Cov}(X, S) = \text{Cov}(X, X+Y) = \text{Var}(X) + \underbrace{\text{Cov}(X,Y)}_{=0} = 1$

代入条件期望公式：
$$E[X \mid S = s] = E[X] + \frac{\text{Cov}(X,S)}{\text{Var}(S)}(s - E[S]) = 0 + \frac{1}{2}(s - 0)$$

$$\boxed{E[X \mid X+Y = s] = \frac{s}{2}}$$

**直觉：** $X$ 和 $Y$ 对称，给定总和为 $s$，各自期望贡献一半。取 $s=3$ 得 $E[X \mid X+Y=3] = 3/2$。

**扩展：条件分布**

$$X \mid S=s \sim N\!\left(\frac{s}{2},\; \frac{1}{2}\right)$$

其中条件方差 $= \text{Var}(X)\left(1 - \frac{\text{Cov}(X,S)^2}{\text{Var}(X)\text{Var}(S)}\right) = 1 \cdot \left(1 - \frac{1}{2}\right) = \frac{1}{2}$

---

### 题目 G2

**题：** $X, Y \overset{iid}{\sim} N(0,1)$，求 $E[X \mid X + 2Y = 4]$

**解：**

令 $S = X + 2Y$：
- $\text{Var}(S) = 1 + 4 = 5$
- $\text{Cov}(X, S) = \text{Cov}(X, X+2Y) = \text{Var}(X) = 1$

$$E[X \mid S=4] = 0 + \frac{1}{5}(4 - 0) = \boxed{\frac{4}{5}}$$

**一般公式：** 若 $S = aX + bY$（$X, Y$ i.i.d. $N(0,1)$，独立）：
$$E[X \mid S=s] = \frac{a}{a^2 + b^2} \cdot s$$

---

### 题目 G3

**题：** $X \sim N(\mu, \sigma^2)$，$Y \sim N(\nu, \tau^2)$ 独立，求 $E[X \mid X+Y = s]$

**解：**

$$\text{Cov}(X, X+Y) = \sigma^2, \quad \text{Var}(X+Y) = \sigma^2 + \tau^2$$

$$E[X \mid X+Y=s] = \mu + \frac{\sigma^2}{\sigma^2+\tau^2}(s - \mu - \nu)$$

令权重 $w = \dfrac{\sigma^2}{\sigma^2 + \tau^2}$，则：

$$\boxed{E[X \mid X+Y=s] = (1-w)\mu + w(s - \nu)}$$

**解释：** 权重 $w$ 与 $X$ 的方差成正比——方差越大的变量，在总和已知时，对其均值的"贡献"也越大（不确定性大的变量更"灵活"）。

---

### 题目 G4 ⭐（卡尔曼滤波原型）

**题：** 信号 $X \sim N(0,1)$，噪声 $\varepsilon \sim N(0, \sigma^2)$ 独立，观测 $Y = X + \varepsilon$，求 $E[X \mid Y = y]$

**解：**

$$\text{Cov}(X, Y) = \text{Var}(X) = 1, \quad \text{Var}(Y) = 1 + \sigma^2$$

$$\boxed{E[X \mid Y=y] = \frac{1}{1+\sigma^2} \cdot y}$$

条件分布：$X \mid Y=y \sim N\!\left(\dfrac{y}{1+\sigma^2},\; \dfrac{\sigma^2}{1+\sigma^2}\right)$

**信噪比视角：**

| 噪声 $\sigma^2$ | $E[X \mid Y=y]$ | 含义 |
|----------------|-----------------|------|
| $\to 0$ | $\to y$ | 观测完全可靠，$E[X] = y$ |
| $= 1$ | $y/2$ | 信噪比 = 1，均值回归一半 |
| $\to \infty$ | $\to 0$ | 观测无信息，回归先验均值 $E[X]=0$ |

**量化应用：** 这是卡尔曼滤波一步更新的核心，也是贝叶斯估计的原型。信号强度决定估计向观测值倾斜的程度，这与 Vasicek Beta 调整、Ledoit-Wolf 协方差收缩的思路完全相同。

---

### 题目 G5

**题：** $(X,Y)$ 服从二元标准正态，相关系数 $\rho$，求 $E[X^2 \mid Y=y]$

**解：**

已知条件分布 $X \mid Y=y \sim N(\rho y, 1-\rho^2)$，利用公式 $E[Z^2] = \text{Var}(Z) + [E[Z]]^2$：

$$E[X^2 \mid Y=y] = \underbrace{(1-\rho^2)}_{\text{条件方差}} + \underbrace{(\rho y)^2}_{\text{条件均值}^2}$$

$$\boxed{E[X^2 \mid Y=y] = (1-\rho^2) + \rho^2 y^2}$$

**当 $y = 0$：** $E[X^2 \mid Y=0] = 1 - \rho^2$（知道 $Y=0$ 减少了 $X^2$ 的期望——信息降低不确定性）

**当 $|\rho| = 1$：** $E[X^2 \mid Y=y] = y^2$（完全相关时，$X = \pm y$ 确定性地）

---

### 题目 G6 ⭐（高频难题）

**题：** $X, Y \overset{iid}{\sim} N(0,1)$，求 $E[X \mid X > Y]$

**解：**

**关键思路：** 令 $S = X+Y$，$D = X-Y$。由于 $X, Y$ 独立正态，$S$ 和 $D$ 独立（协方差为零且联合正态）：
$$S \sim N(0,2), \quad D \sim N(0,2), \quad \text{Cov}(S,D) = \text{Var}(X) - \text{Var}(Y) = 0$$

用 $X = (S+D)/2$ 改写：
$$E[X \mid X>Y] = E\!\left[\frac{S+D}{2} \,\middle|\, D>0\right] = \frac{1}{2}E[S \mid D>0] + \frac{1}{2}E[D \mid D>0]$$

由 $S \perp D$：$E[S \mid D>0] = E[S] = 0$

$D \mid D>0$：$D$ 的分布截断到正半轴，$D \sim N(0,2)$：

$$E[D \mid D>0] = \sqrt{2} \cdot E[Z \mid Z>0] = \sqrt{2} \cdot \sqrt{\frac{2}{\pi}} = \frac{2}{\sqrt{\pi}}$$

（其中 $E[Z \mid Z>0] = \sqrt{2/\pi}$ 对 $Z \sim N(0,1)$）

$$E[X \mid X>Y] = 0 + \frac{1}{2} \cdot \frac{2}{\sqrt{\pi}} = \boxed{\frac{1}{\sqrt{\pi}} \approx 0.5642}$$

**半正态期望公式（必背）：**

若 $Z \sim N(0, \sigma^2)$：
$$E[Z \mid Z>0] = \sigma\sqrt{\frac{2}{\pi}}, \quad E[|Z|] = \sigma\sqrt{\frac{2}{\pi}}$$

---

### 题目 G7

**题：** $X, Y \overset{iid}{\sim} N(0,1)$，求 $E[X \mid X^2 + Y^2 = r^2]$

**解：**

**旋转对称性论证：** $N(0,1)$ 是旋转不变的——$(X,Y)$ 的联合分布在 $\mathbb{R}^2$ 中是圆对称的。

给定 $X^2 + Y^2 = r^2$（点在半径 $r$ 的圆上），由旋转对称性，$(X,Y) \mid X^2+Y^2=r^2$ 均匀分布在圆周上：
$$X = r\cos\theta, \quad Y = r\sin\theta, \quad \theta \sim U[0, 2\pi)$$

$$E[X \mid X^2+Y^2=r^2] = r \cdot E[\cos\theta] = r \cdot 0 = \boxed{0}$$

**扩展：** $E[X^2 \mid X^2+Y^2=r^2] = r^2/2$（由对称性，$X^2$ 和 $Y^2$ 各承担一半）

---

### 题目 G8 ⭐

**题：** $X, Y \overset{iid}{\sim} N(0,1)$，求 $E[\max(X,Y)]$

**解（恒等式法）：**

利用恒等式：$\max(a,b) = \dfrac{a+b}{2} + \dfrac{|a-b|}{2}$

$$E[\max(X,Y)] = \underbrace{\frac{E[X]+E[Y]}{2}}_{=0} + \frac{E[|X-Y|]}{2}$$

$X - Y \sim N(0, 2)$，令 $D = X-Y$：
$$E[|D|] = \sqrt{2} \cdot E[|Z|] = \sqrt{2} \cdot \sqrt{\frac{2}{\pi}} = \frac{2}{\sqrt{\pi}}$$

$$E[\max(X,Y)] = \frac{1}{2} \cdot \frac{2}{\sqrt{\pi}} = \boxed{\frac{1}{\sqrt{\pi}} \approx 0.5642}$$

**有趣：** $E[\max(X,Y)] = E[X \mid X>Y] = 1/\sqrt{\pi}$，两者恰好相同（因为 $\max(X,Y) = X \cdot \mathbf{1}_{X>Y} + Y \cdot \mathbf{1}_{Y>X}$，期望等于 $E[X \mid X>Y] \cdot P(X>Y) + E[Y \mid Y>X] \cdot P(Y>X) = 1/\sqrt{\pi} \cdot 1/2 + 1/\sqrt{\pi} \cdot 1/2$）。

**推广到 $n$ 个变量的最大值（次序统计量）：**

$X_1,\ldots,X_n \overset{iid}{\sim} N(0,1)$，$E[X_{(n)}] \approx \sqrt{2\ln n}$（大 $n$ 渐进）

---

### 题目 G9

**题：** $X, Y \overset{iid}{\sim} N(0,1)$，求 $E[XY \mid X+Y = s]$

**解：**

**方法：利用 $Y = s - X$（在条件 $X+Y=s$ 下）**

$$E[XY \mid X+Y=s] = E[X(s-X) \mid S=s] = s \cdot E[X \mid S=s] - E[X^2 \mid S=s]$$

已知 $X \mid S=s \sim N(s/2,\; 1/2)$（见题目 G1）：
- $E[X \mid S=s] = s/2$
- $E[X^2 \mid S=s] = \text{Var}(X \mid S=s) + [E[X \mid S=s]]^2 = \dfrac{1}{2} + \dfrac{s^2}{4}$

$$E[XY \mid S=s] = s \cdot \frac{s}{2} - \left(\frac{1}{2} + \frac{s^2}{4}\right) = \frac{s^2}{2} - \frac{1}{2} - \frac{s^2}{4} = \boxed{\frac{s^2}{4} - \frac{1}{2}}$$

**验证（$s=0$）：** $E[XY \mid X+Y=0] = -1/2$。当 $X+Y=0$ 时 $Y=-X$，$XY = -X^2$，$E[-X^2 \mid S=0] = -E[X^2 \mid X \sim N(0,1/2)] = -1/2$ ✓

---

### 题目 G10 ⭐（期权定价必备）

**题：** $X \sim N(0,1)$，求 $E[e^X]$，$E[e^{aX}]$，$E[e^{aX+b}]$

**解（矩母函数法）：**

正态分布的 MGF：$M_X(t) = E[e^{tX}] = e^{\mu t + \sigma^2 t^2/2}$

对 $X \sim N(0,1)$，$\mu=0$，$\sigma^2=1$：

$$E[e^{tX}] = e^{t^2/2}$$

代入不同 $t$：

$$E[e^X] = e^{1/2}, \quad E[e^{aX}] = e^{a^2/2}, \quad E[e^{aX+b}] = e^b \cdot e^{a^2/2} = e^{b + a^2/2}$$

**推广（一般正态）：** 若 $X \sim N(\mu, \sigma^2)$：
$$E[e^X] = e^{\mu + \sigma^2/2} \quad \text{（对数正态的期望）}$$

**期权定价应用：**

在风险中性测度下 $\ln(S_T) \sim N(\ln S_0 + (r - \sigma^2/2)T,\; \sigma^2 T)$，则：
$$E[S_T] = S_0 e^{rT} \quad \text{（由上式，令 $\mu = \ln S_0 + (r-\sigma^2/2)T$，$\sigma^2 \to \sigma^2 T$）}$$

---

## 第二章：条件概率与条件期望

---

### 题目 C1

**题：** $X \sim \text{Exp}(\lambda)$，求 $E[X \mid X > a]$

**解（无记忆性）：**

指数分布的无记忆性：$P(X > a+t \mid X > a) = P(X > t)$

等价地：$(X - a) \mid X > a \sim \text{Exp}(\lambda)$（与原分布相同）

$$E[X \mid X > a] = a + E[X - a \mid X > a] = a + E[X] = a + \frac{1}{\lambda}$$

$$\boxed{E[X \mid X > a] = a + \frac{1}{\lambda}}$$

**直接验证：**
$$E[X \cdot \mathbf{1}_{X>a}] = \int_a^\infty x \lambda e^{-\lambda x}\,dx = \left(a + \frac{1}{\lambda}\right)e^{-\lambda a}$$
$$P(X > a) = e^{-\lambda a}$$
$$E[X \mid X>a] = \frac{(a+1/\lambda)e^{-\lambda a}}{e^{-\lambda a}} = a + \frac{1}{\lambda} \checkmark$$

---

### 题目 C2 ⭐（高频）

**题：** $X, Y \overset{iid}{\sim} \text{Exp}(\lambda)$，$\text{Exp}(\mu)$ 分别（独立），求 $E[X \mid X < Y]$

**解：**

$$P(X < Y) = \frac{\lambda}{\lambda + \mu}$$

计算 $E[X \cdot \mathbf{1}_{X<Y}]$：

$$E[X \cdot \mathbf{1}_{X<Y}] = \int_0^\infty \int_x^\infty x \lambda e^{-\lambda x} \mu e^{-\mu y}\,dy\,dx = \int_0^\infty x \lambda e^{-\lambda x} e^{-\mu x}\,dx$$

$$= \lambda \int_0^\infty x e^{-(\lambda+\mu)x}\,dx = \frac{\lambda}{(\lambda+\mu)^2}$$

$$E[X \mid X < Y] = \frac{\lambda/(\lambda+\mu)^2}{\lambda/(\lambda+\mu)} = \boxed{\frac{1}{\lambda+\mu}}$$

**直觉：** 给定 $X < Y$，$X = \min(X,Y)$，而 $\min(\text{Exp}(\lambda), \text{Exp}(\mu)) \sim \text{Exp}(\lambda+\mu)$，期望 $= 1/(\lambda+\mu)$。

---

### 题目 C3

**题：** $N \sim \text{Poisson}(\lambda)$，求 $E[N \mid N \geq 1]$

**解：**

$$E[N \cdot \mathbf{1}_{N\geq 1}] = E[N] - 0 \cdot P(N=0) = \lambda$$

$$P(N \geq 1) = 1 - e^{-\lambda}$$

$$\boxed{E[N \mid N \geq 1] = \frac{\lambda}{1 - e^{-\lambda}}}$$

**极限情况：**
- $\lambda \to 0$：$E[N \mid N \geq 1] \to 1$（给定至少发生一次，期望恰好发生一次）
- $\lambda \to \infty$：$E[N \mid N \geq 1] \to \lambda$（截断几乎没有影响）

---

### 题目 C4

**题：** $X, Y \overset{iid}{\sim} U(0,1)$，求 $E[X \mid X + Y > 1]$

**解（几何方法）：**

区域 $\{X+Y > 1\}$ 是单位正方形内的上三角形，面积 $= 1/2$，故 $P(X+Y>1) = 1/2$。

$$E[X \cdot \mathbf{1}_{X+Y>1}] = \int_0^1\int_{1-x}^1 x\,dy\,dx = \int_0^1 x \cdot x\,dx = \int_0^1 x^2\,dx = \frac{1}{3}$$

$$E[X \mid X+Y > 1] = \frac{1/3}{1/2} = \boxed{\frac{2}{3}}$$

**直觉：** 在 $X+Y>1$ 的三角形内，$X$ 的条件密度为 $f(x \mid X+Y>1) = 2x$（在 $[0,1]$），是倒 U 形——$X$ 越大越有可能满足条件，因此条件均值 $2/3 > 1/2$（无条件均值）。

---

### 题目 C5 ⭐（欧式期权核心积分）

**题：** $X \sim N(\mu, \sigma^2)$，求 $E[(X-c)^+] = E[\max(X-c, 0)]$

**解：**

$$E[(X-c)^+] = \int_c^\infty (x-c) \frac{1}{\sigma}\phi\!\left(\frac{x-\mu}{\sigma}\right)dx$$

令 $z = (x-\mu)/\sigma$，$x=c$ 对应 $z_0 = (c-\mu)/\sigma$，令 $d = (\mu-c)/\sigma = -z_0$：

$$= \int_{-d}^\infty (\sigma z + \mu - c)\phi(z)\,dz = \sigma\int_{-d}^\infty z\phi(z)\,dz + (\mu-c)\int_{-d}^\infty \phi(z)\,dz$$

利用 $\int_a^\infty z\phi(z)\,dz = \phi(a)$（因为 $-d(\phi(z))/dz = z\phi(z)$）：

$$= \sigma\phi(d) + (\mu-c)\Phi(d)$$

$$\boxed{E[(X-c)^+] = \sigma\phi(d) + (\mu-c)\Phi(d), \quad d = \frac{\mu-c}{\sigma}}$$

**Black-Scholes 公式的来源：**

在 BS 模型中，风险中性测度下 $\ln S_T \sim N(\ln S_0 + (r-\sigma^2/2)T,\; \sigma^2 T)$。

欧式看涨期权：$C = e^{-rT}E[(S_T-K)^+]$

将 $S_T = e^X$，$X \sim N(\mu, \sigma^2 T)$ 代入，利用上述公式（需要额外处理 $e^X$ 的积分），最终得到 BS 公式 $C = S_0\Phi(d_1) - Ke^{-rT}\Phi(d_2)$。

---

### 题目 C6 ⭐（CVaR 计算）

**题：** $X \sim N(0,1)$，求 $E[X \mid X > z_\alpha]$（其中 $z_\alpha = \Phi^{-1}(\alpha)$）

**解：**

$$E[X \cdot \mathbf{1}_{X > z_\alpha}] = \int_{z_\alpha}^\infty x\phi(x)\,dx = \int_{z_\alpha}^\infty (-\phi'(x))\,dx = \phi(z_\alpha)$$

（利用 $x\phi(x) = -\phi'(x)$，因为 $\phi'(x) = -x\phi(x)$）

$$P(X > z_\alpha) = 1 - \Phi(z_\alpha) = 1 - \alpha$$

$$\boxed{E[X \mid X > z_\alpha] = \frac{\phi(z_\alpha)}{1 - \Phi(z_\alpha)} = \lambda(z_\alpha)}$$

其中 $\lambda(a) = \phi(a)/(1-\Phi(a))$ 是**逆 Mills 比率**。

**CVaR 应用：**

对正态分布，CVaR 在置信水平 $\alpha$ 处（即条件于损失超过 VaR）：

$$\text{CVaR}_\alpha = E[X \mid X > \text{VaR}_\alpha] = \frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha}$$

数值示例（$\alpha = 95\%$，$z_{0.95} = 1.645$）：
$$\text{CVaR}_{95\%} = \frac{\phi(1.645)}{0.05} = \frac{0.1031}{0.05} \approx 2.063\sigma$$

---

## 第三章：Tower Law 与递推计算

---

### 题目 T1 ⭐（复合分布）

**题：** $N \sim \text{Poisson}(\lambda)$，给定 $N=n$，$X_1,\ldots,X_n \overset{iid}{\sim} \text{Bernoulli}(p)$，$S = \sum_{i=1}^N X_i$，求 $E[S]$，$\text{Var}(S)$，以及 $S$ 的分布

**解：**

**期望（Tower Law）：**
$$E[S] = E\bigl[E[S \mid N]\bigr] = E[Np] = p \cdot E[N] = p\lambda$$

**方差（Eve's Law）：**
$$\text{Var}(S) = E[\text{Var}(S \mid N)] + \text{Var}(E[S \mid N])$$
$$= E\bigl[Np(1-p)\bigr] + \text{Var}(Np) = p(1-p)\lambda + p^2\lambda = p\lambda\bigl[(1-p) + p\bigr] = p\lambda$$

**分布：** $S \sim \text{Poisson}(p\lambda)$

这是**泊松稀疏化（Thinning）定理**：每个事件以概率 $p$ 保留，保留数服从 $\text{Poisson}(p\lambda)$。

**量化应用：** 每天的交易事件数 $\sim \text{Poisson}(\lambda)$，其中有信号意义的事件比例为 $p$，则有效信号数 $\sim \text{Poisson}(p\lambda)$。

---

### 题目 T2

**题：** $X \sim N(0,1)$，$Y \mid X = x \sim N(x, 1)$，求 $E[Y]$，$\text{Var}(Y)$，$Y$ 的边缘分布

**解：**

**期望：**
$$E[Y] = E\bigl[E[Y \mid X]\bigr] = E[X] = 0$$

**方差（Eve's Law）：**
$$\text{Var}(Y) = \underbrace{E[\text{Var}(Y \mid X)]}_{=E[1]=1} + \underbrace{\text{Var}(E[Y \mid X])}_{=\text{Var}(X)=1} = 1 + 1 = 2$$

**边缘分布：** $Y = X + \varepsilon$（$\varepsilon \sim N(0,1)$ 独立），故 $Y \sim N(0,2)$

**Eve's Law 的直觉（方差分解）：**

```
Var(Y) = 2
        ├── E[Var(Y|X)] = 1  ← "within" 方差（给定X后Y仍有不确定性）
        └── Var(E[Y|X]) = 1  ← "between" 方差（X变化导致Y均值变化）
```

这正是 ANOVA（方差分析）的基础：总方差 = 组内方差 + 组间方差。

---

### 题目 T3 ⭐（Wald 恒等式）

**题：** $X_1, X_2, \ldots \overset{iid}{\sim} F$，$E[X]=\mu$，$\text{Var}(X)=\sigma^2$，$N$ 为与 $X_i$ 独立的随机变量，$S_N = \sum_{i=1}^N X_i$，求 $E[S_N]$ 和 $\text{Var}(S_N)$

**解（Wald 恒等式）：**

**第一 Wald 恒等式（期望）：**
$$E[S_N] = E\bigl[E[S_N \mid N]\bigr] = E[N\mu] = \mu \cdot E[N]$$

**第二 Wald 恒等式（方差）：**
$$\text{Var}(S_N) = E[\text{Var}(S_N \mid N)] + \text{Var}(E[S_N \mid N])$$
$$= E[N\sigma^2] + \text{Var}(N\mu) = \sigma^2 E[N] + \mu^2 \text{Var}(N)$$

$$\boxed{E[S_N] = \mu E[N], \quad \text{Var}(S_N) = \sigma^2 E[N] + \mu^2 \text{Var}(N)}$$

**保险精算应用：** 年度总理赔 $S_N$，$N$ 是理赔次数（Poisson），$X_i$ 是单次金额（对数正态）。Wald 公式直接给出总理赔的均值和方差，用于保费定价。

---

### 题目 T4

**题：** 公平硬币，$N$ 是投掷次数直到正面出现，$N \sim \text{Geom}(1/2)$，递推法求 $E[N]$

**解（条件期望递推）：**

$$E[N] = E[N \mid \text{第一次正面}] \cdot P(\text{正面}) + E[N \mid \text{第一次反面}] \cdot P(\text{反面})$$

$$= 1 \cdot \frac{1}{2} + (1 + E[N]) \cdot \frac{1}{2}$$

$$E[N] = \frac{1}{2} + \frac{1}{2} + \frac{1}{2}E[N] = 1 + \frac{1}{2}E[N]$$

$$\frac{1}{2}E[N] = 1 \implies E[N] = 2$$

一般地，$N \sim \text{Geom}(p)$：$E[N] = 1/p$。

---

### 题目 T5 ⭐（鞅 + OST）

**题：** 随机游走 $X_0=0$，每步 $\pm 1$ 各以 $1/2$，$\tau = \min\{n : X_n = a \text{ 或 } X_n = -b\}$（$a, b > 0$），求 $P(X_\tau = a)$ 和 $E[\tau]$

**解：**

**求概率（用鞅 $M_n = X_n$）：**

$X_n$ 是鞅，由 OST：$E[X_\tau] = X_0 = 0$

设 $p = P(X_\tau = a)$：
$$p \cdot a + (1-p) \cdot (-b) = 0 \implies p = \frac{b}{a+b}$$

$$\boxed{P(X_\tau = a) = \frac{b}{a+b}}$$

**求停时期望（用鞅 $M_n = X_n^2 - n$）：**

$X_n^2 - n$ 是鞅（验证：$E[X_{n+1}^2 - (n+1) \mid X_n] = E[X_n^2 \pm 2X_n + 1 - n - 1 \mid X_n] = X_n^2 - n$），由 OST：

$$E[X_\tau^2 - \tau] = X_0^2 - 0 = 0 \implies E[\tau] = E[X_\tau^2]$$

$$E[X_\tau^2] = p \cdot a^2 + (1-p) \cdot b^2 = \frac{b}{a+b}a^2 + \frac{a}{a+b}b^2 = \frac{ab(a+b)}{a+b} = ab$$

$$\boxed{E[\tau] = ab}$$

**例：** $a = b = 1$（双侧边界在 $\pm 1$）：$P(X_\tau=1) = 1/2$，$E[\tau] = 1$。

---

### 题目 T6 ⭐（布朗运动首达时）

**题：** 标准布朗运动 $W_t$，$\tau = \min\{t : W_t = a \text{ 或 } W_t = -b\}$，求 $P(W_\tau = a)$ 和 $E[\tau]$

**解（连续时间鞅 + OST）：**

与随机游走类比（连续时间版本）：

**概率：** $W_t$ 是鞅，$E[W_\tau] = 0 \implies P(W_\tau = a) = \dfrac{b}{a+b}$

**期望停时：** $W_t^2 - t$ 是鞅，$E[W_\tau^2] = E[\tau] \implies E[\tau] = P(W_\tau=a) \cdot a^2 + P(W_\tau=-b) \cdot b^2$

$$E[\tau] = \frac{b}{a+b}a^2 + \frac{a}{a+b}b^2 = ab$$

$$\boxed{P(W_\tau = a) = \frac{b}{a+b}, \quad E[\tau] = ab}$$

**应用：** 期权的行权时间（障碍期权的首达时），KO 期权的定价，风险管理中的"破产时间"。

---

## 第四章：矩计算

---

### 题目 M1 ⭐（正态高阶矩）

**题：** $X \sim N(0,1)$，求所有阶次的矩 $E[X^n]$

**解：**

**奇数阶矩：** $E[X^{2k+1}] = 0$（奇函数在对称分布下积分为零）

**偶数阶矩递推：** 分部积分 $E[X^n] = (n-1)E[X^{n-2}]$

$$E[X^{2n}] = (2n-1)!! = 1 \cdot 3 \cdot 5 \cdots (2n-1)$$

| $n$ | $E[X^n]$ |
|-----|----------|
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |
| 4 | 3 |
| 5 | 0 |
| 6 | 15 |
| 8 | 105 |

**量化应用：** 峰度 $= E[X^4]/(E[X^2])^2 = 3$（正态特征）。金融收益率的超额峰度 $> 0$ 说明厚尾，$E[X^4] > 3$（$t$ 分布等）。

---

### 题目 M2

**题：** $X \sim \text{Exp}(\lambda)$，求 $E[X^n]$（所有正整数阶矩）

**解（MGF 法）：**

$$M_X(t) = \frac{\lambda}{\lambda - t} = \lambda \sum_{k=0}^\infty \left(\frac{t}{\lambda}\right)^k \quad (|t| < \lambda)$$

$$E[X^n] = M_X^{(n)}(0) = \frac{n!}{\lambda^n}$$

**递推推导：** 分部积分 $E[X^n] = \dfrac{n}{\lambda} E[X^{n-1}]$，初始 $E[X^0]=1$，故 $E[X^n] = n!/\lambda^n$

**记忆：** 令 $\lambda=1$，$E[X^n] = n! = \Gamma(n+1)$（Gamma 函数），这正是 $\text{Gamma}(n, 1)$ 分布的均值。

---

### 题目 M3

**题：** $X, Y \overset{iid}{\sim} U(0,1)$，求 $E[|X-Y|]$

**解（三种方法）：**

**方法一（对称性）：**

$E[|X-Y|] = 2E[(X-Y)^+] = 2E[X-Y \mid X>Y] \cdot P(X>Y)$

$= 2 \cdot E[X-Y \mid X>Y] \cdot \frac{1}{2}$

给定 $X>Y$，$E[X-Y] = \int_0^1 \int_0^x (x-y)\, dy\, dx \cdot 2 / 1 = \ldots$

**方法二（次序统计量）：**

$|X-Y| = X_{(2)} - X_{(1)} = \max(X,Y) - \min(X,Y)$

$E[\max] = \dfrac{2}{3}$，$E[\min] = \dfrac{1}{3}$

$$E[|X-Y|] = \frac{2}{3} - \frac{1}{3} = \boxed{\frac{1}{3}}$$

**方法三（直接积分）：**

$$E[|X-Y|] = 2\int_0^1\int_0^x (x-y)\,dy\,dx = 2\int_0^1 \frac{x^2}{2}\,dx = \int_0^1 x^2\,dx = \frac{1}{3}$$

---

### 题目 M4 ⭐（Delta 方法）

**题：** $X, Y$ 独立，均值分别为 $\mu_X, \mu_Y$，方差 $\sigma_X^2, \sigma_Y^2$，用 Delta 方法近似 $E[X/(X+Y)]$

**解：**

令 $g(x,y) = x/(x+y)$，在 $(\mu_X, \mu_Y)$ 处 Taylor 展开至二阶：

$$g(x,y) \approx g(\mu) + g_x \cdot (x-\mu_X) + g_y \cdot (y-\mu_Y) + \frac{1}{2}g_{xx}(x-\mu_X)^2 + \frac{1}{2}g_{yy}(y-\mu_Y)^2 + \ldots$$

偏导数（令 $\mu = \mu_X + \mu_Y$）：
$$g_x = \frac{\mu_Y}{\mu^2}, \quad g_y = -\frac{\mu_X}{\mu^2}, \quad g_{xx} = -\frac{2\mu_Y}{\mu^3}, \quad g_{yy} = \frac{2\mu_X}{\mu^3}$$

取期望（一阶项消失）：

$$E\!\left[\frac{X}{X+Y}\right] \approx \frac{\mu_X}{\mu} + \frac{1}{2}\left(-\frac{2\mu_Y}{\mu^3}\right)\sigma_X^2 + \frac{1}{2}\left(\frac{2\mu_X}{\mu^3}\right)\sigma_Y^2$$

$$\boxed{E\!\left[\frac{X}{X+Y}\right] \approx \frac{\mu_X}{\mu_X+\mu_Y} + \frac{\mu_X\sigma_Y^2 - \mu_Y\sigma_X^2}{(\mu_X+\mu_Y)^3}}$$

**Jensen 不等式给出方向：** $x/(x+y)$ 关于 $x$ 是凹函数（$g_{xx} < 0$），故 $E[g] < g(\mu)$（当 $Y$ 固定时）。

---

### 题目 M5

**题：** $X \sim N(0,1)$，求 $E[\Phi(X)]$，其中 $\Phi$ 是标准正态 CDF

**解（引入独立副本）：**

$$E[\Phi(X)] = E[P(Y \leq X \mid X)] = P(Y \leq X)$$

其中 $Y \sim N(0,1)$ 独立于 $X$（因为 $\Phi(x) = P(Y \leq x)$）。

$$P(Y \leq X) = P(X - Y \geq 0)$$

$X - Y \sim N(0, 2)$，由对称性：

$$P(X-Y \geq 0) = \frac{1}{2}$$

$$\boxed{E[\Phi(X)] = \frac{1}{2}}$$

**技巧：** "引入独立副本"是期望计算的强大技巧——$E[F(X)] = P(Y \leq X)$，将函数的期望转化为联合事件的概率。

---

## 第五章：顺序统计量与尾部期望

---

### 题目 O1 ⭐

**题：** $X_1, \ldots, X_n \overset{iid}{\sim} U[0,1]$，求 $E[X_{(k)}]$（第 $k$ 小次序统计量）

**解：**

$X_{(k)}$ 的密度：
$$f_{(k)}(x) = \frac{n!}{(k-1)!(n-k)!} x^{k-1}(1-x)^{n-k}, \quad x \in [0,1]$$

这是 $\text{Beta}(k, n-k+1)$ 分布，其期望：

$$E[X_{(k)}] = \frac{k}{k + (n-k+1)} = \frac{k}{n+1}$$

$$\boxed{E[X_{(k)}] = \frac{k}{n+1}}$$

**直觉（等间距论证）：** $n$ 个均匀随机点将 $[0,1]$ 划分为 $n+1$ 段，由对称性每段期望长度相等（$= 1/(n+1)$），第 $k$ 个点平均位于 $k/(n+1)$。

| $k$ | $E[X_{(k)}]$ |
|-----|-------------|
| $1$（min） | $1/(n+1)$ |
| $\lfloor(n+1)/2\rfloor$（中位数） | $\approx 1/2$ |
| $n$（max） | $n/(n+1)$ |

---

### 题目 O2 ⭐（优惠券收集）

**题：** $X_1, \ldots, X_n \overset{iid}{\sim} \text{Exp}(1)$，求 $E[X_{(n)}]$（最大值期望）

**解（间距分解）：**

指数分布次序统计量有特殊的**间距独立性**：

令 $D_k = X_{(k)} - X_{(k-1)}$（$X_{(0)} = 0$），则 $D_1, D_2, \ldots, D_n$ 独立，且：

$$D_k \sim \text{Exp}(n-k+1)$$

因此：
$$E[X_{(n)}] = \sum_{k=1}^n E[D_k] = \sum_{k=1}^n \frac{1}{n-k+1} = \sum_{j=1}^n \frac{1}{j} = H_n$$

$$\boxed{E[X_{(n)}] = H_n = 1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n} \approx \ln n + \gamma}$$

（$\gamma \approx 0.5772$ 是 Euler-Mascheroni 常数）

**优惠券收集连接：** 收集 $n$ 种优惠券（每次等概率），已收集 $k-1$ 种时，等待新券的时间 $\sim \text{Geom}(1 - (k-1)/n) \approx \text{Exp}(n/(n-k+1))$，总期望时间 $= n H_n$。

---

### 题目 O3

**题：** $X \sim \text{Pareto}(\alpha, x_m)$，$P(X>x) = (x_m/x)^\alpha$（$x \geq x_m$），求 $E[X]$，$\text{Var}(X)$，并分析矩的存在条件

**解：**

PDF：$f(x) = \alpha x_m^\alpha / x^{\alpha+1}$，$x \geq x_m$

$$E[X] = \int_{x_m}^\infty x \cdot \frac{\alpha x_m^\alpha}{x^{\alpha+1}}\,dx = \alpha x_m^\alpha \int_{x_m}^\infty x^{-\alpha}\,dx$$

当 $\alpha > 1$：
$$E[X] = \alpha x_m^\alpha \cdot \frac{x_m^{1-\alpha}}{\alpha-1} = \frac{\alpha x_m}{\alpha-1}$$

当 $\alpha \leq 1$：积分发散，$E[X] = \infty$

**矩存在条件：**

| 条件 | 存在的矩 |
|------|---------|
| $\alpha > 1$ | $E[X]$ 有限 |
| $\alpha > 2$ | $\text{Var}(X)$ 有限 |
| $\alpha > k$ | $E[X^k]$ 有限 |

**金融厚尾的含义：** 若 $1 < \alpha \leq 2$，均值有限但方差无穷——CLT 不适用，极端损失无法用正态分布建模！金融危机时的损失分布常具有 $\alpha \approx 2$-$3$ 的 Pareto 尾部，这是使用正态 VaR 会严重低估风险的数学根源。

---

### 题目 O4 ⭐（连接线问题）

**题：** 碗中有 $2n$ 个面条端头，随机配对形成环。第一次配对：随机取一端，再随机取另一端。若配到同一根面条则形成环，否则合并成更长的面条（仍有 $2n-2$ 个端头）。求期望环数 $E[C_n]$。

**解（Tower Law 递推）：**

当有 $2k$ 个端头时，取出一端后，另一端是同一根面条的概率 $= 1/(2k-1)$（与此端配对形成环），否则概率 $= (2k-2)/(2k-1)$ 合并成长面条（剩 $2k-2$ 端头）。

$$E[C_k] = \frac{1}{2k-1} \cdot (1 + E[C_{k-1}]) + \frac{2k-2}{2k-1} \cdot E[C_{k-1}]$$

$$= E[C_{k-1}] + \frac{1}{2k-1}$$

初始 $E[C_0] = 0$，递推得：

$$E[C_n] = \sum_{k=1}^n \frac{1}{2k-1} = 1 + \frac{1}{3} + \frac{1}{5} + \cdots + \frac{1}{2n-1}$$

$$\approx \frac{1}{2}\ln(2n) + \frac{\gamma}{2} \approx \frac{\ln(2n)}{2}$$

对 $n = 100$：$E[C_{100}] \approx \ln(200)/2 \approx 2.65$ 个环。

---

## 附录：高频公式速查卡

### 正态分布条件期望公式

若 $(X_1, X_2)$ 联合正态：

$$E[X_1 \mid X_2 = x_2] = \mu_1 + \frac{\sigma_{12}}{\sigma_{22}}(x_2 - \mu_2) = \mu_1 + \rho\frac{\sigma_1}{\sigma_2}(x_2-\mu_2)$$

$$\text{Var}(X_1 \mid X_2) = \sigma_{11} - \frac{\sigma_{12}^2}{\sigma_{22}} = \sigma_1^2(1-\rho^2)$$

### 截断正态期望（逆 Mills 比率）

$$E[X \mid X > a] = \mu + \sigma \cdot \frac{\phi((a-\mu)/\sigma)}{1-\Phi((a-\mu)/\sigma)}$$

$$E[X \mid X < a] = \mu - \sigma \cdot \frac{\phi((a-\mu)/\sigma)}{\Phi((a-\mu)/\sigma)}$$

### 无记忆性（指数 / 几何分布）

$$E[X \mid X > a] = a + \frac{1}{\lambda} \quad (X \sim \text{Exp}(\lambda))$$

$$E[N \mid N > k] = k + E[N] = k + \frac{1}{p} \quad (N \sim \text{Geom}(p))$$

### 次序统计量期望

$$E[X_{(k)}] = \frac{k}{n+1} \quad \text{（}X_i \overset{iid}{\sim} U[0,1]\text{）}$$

$$E[X_{(n)}] = H_n \quad \text{（}X_i \overset{iid}{\sim} \text{Exp}(1)\text{）}$$

### 半正态与截断正态

$$E[|Z|] = \sqrt{\frac{2}{\pi}} \approx 0.7979 \quad (Z \sim N(0,1))$$

$$E[Z \mid Z > 0] = \sqrt{\frac{2}{\pi}} \approx 0.7979 \quad (Z \sim N(0,1))$$

$$E[\max(X,Y)] = \frac{1}{\sqrt{\pi}} \approx 0.5642 \quad (X,Y \overset{iid}{\sim} N(0,1))$$

### Tower Law 与方差分解（Eve's Law）

$$E[X] = E[E[X\mid Y]]$$

$$\text{Var}(X) = E[\text{Var}(X\mid Y)] + \text{Var}(E[X\mid Y])$$

$$\text{Cov}(X,Y) = E[\text{Cov}(X,Y\mid Z)] + \text{Cov}(E[X\mid Z], E[Y\mid Z])$$

### 鞅期望（OST 应用）

布朗运动 $W_t$，双侧边界 $a > 0 > -b$，$\tau$ = 首达时：

$$P(W_\tau = a) = \frac{b}{a+b}, \quad E[\tau] = ab$$

带漂移 BM：$dX_t = \mu\,dt + \sigma\,dW_t$，单侧边界 $a$（$X_0=0, \mu>0$）：

$$E[\tau_a] = \frac{a}{\mu}, \quad P(\text{到达 }a) = 1$$

---

*整理完毕。共 40 道核心题目，按高斯、条件概率、Tower Law、矩计算、顺序统计量分类，含完整推导与量化金融 Insights。*

*建议学习路径：G1→G4→G8（正态基础）→ C1→C2（指数/无记忆）→ T1→T3（Tower Law）→ M3→M5（矩技巧）→ T5→T6（鞅+OST）→ G5→C5→C6（高阶）*
