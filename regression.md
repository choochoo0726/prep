# 回归分析变换问题完全指南

> **核心考察点：** 对 OLS 几何结构、代数性质、统计量计算的深度理解
> **题型特征：** "如果把 X/Y 做某种变换，$\hat{\beta}$、$R^2$、$t$ 统计量会如何变化？"
> **适用场景：** Quant Research、统计面试、数据科学岗位
> **格式说明：** 所有公式使用 LaTeX（Obsidian / Typora / VS Code 可渲染）

---

## 总览速查表

| # | 变换操作 | $\hat{\beta}$ | $R^2$ | $t$ 统计量 | SE |
|---|---------|--------------|-------|-----------|-----|
| 1 | 交换 X 和 Y | 完全改变 | **不变** | 改变 | 改变 |
| 2 | X 和 Y 都乘以 2（复制数据集） | **不变** | **不变** | **增大 $\sqrt{2}$** | **减小 $\sqrt{2}$** |
| 3 | 仅 Y 乘以常数 $c$ | 乘以 $c$ | **不变** | **不变** | 乘以 $c$ |
| 4 | 仅 X 乘以常数 $c$ | 除以 $c$ | **不变** | **不变** | 除以 $c$ |
| 5 | X 加常数（平移） | 截距变，斜率**不变** | **不变** | 斜率 $t$ **不变** | **不变** |
| 6 | Y 加常数（平移） | 截距变，斜率**不变** | **不变** | 斜率 $t$ **不变** | **不变** |
| 7 | 复制观测（每行重复 $k$ 次） | **不变** | **不变** | **增大 $\sqrt{k}$** | **减小 $\sqrt{k}$** |
| 8 | 增加一个与 Y 无关的 X | 其他系数**不变**（正交时） | 增大或不变 | 新变量 $t\approx0$ | — |
| 9 | X 标准化（z-score） | 变为标准化系数 $\beta^*$ | **不变** | **不变** | 改变 |
| 10 | 对 Y 取对数 | 完全改变（解释改变） | 一般改变 | 改变 | 改变 |
| 11 | X 中加入 Y 的噪声副本 | 衰减（attenuation） | 降低 | 降低 | 增大 |
| 12 | 加入完全多重共线变量 | 不唯一（矩阵奇异） | 不变 | 无意义 | $\to\infty$ |
| 13 | 回归 $Y$ 对 $X$ 再对残差回归 | Frisch-Waugh 定理 | — | — | — |
| 14 | 对 $X$ 和 $Y$ 同时中心化 | 斜率**不变**，截距=0 | **不变** | 斜率 $t$ **不变** | **不变** |
| 15 | 加入 $X^2$ 项 | 原系数一般改变 | 增大 | 改变 | 改变 |

---

## 核心设定

**基础模型：** $Y = X\beta + \varepsilon$，OLS 估计量：

$$\hat{\beta} = (X^\top X)^{-1} X^\top Y$$

**简单回归（$n$ 个观测，1 个自变量）：**

$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

$$R^2 = \frac{S_{xy}^2}{S_{xx} S_{yy}} = \rho_{xy}^2, \quad SE(\hat{\beta}_1) = \frac{s}{\sqrt{S_{xx}}}, \quad t = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)}$$

其中 $s^2 = \frac{\text{RSS}}{n-2}$，$S_{xx} = \sum(x_i-\bar{x})^2$，$S_{yy} = \sum(y_i-\bar{y})^2$，$S_{xy} = \sum(x_i-\bar{x})(y_i-\bar{y})$。

---

## 问题 1：交换 X 和 Y 的位置

### 问题

在简单线性回归 $Y = \beta_0 + \beta_1 X + \varepsilon$ 中，如果改为回归 $X = \alpha_0 + \alpha_1 Y + \varepsilon'$，$\hat{\alpha}_1$、$R^2$、$t$ 统计量分别如何变化？

### 推导

**原始回归（Y on X）：**

$$\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}}$$

**交换后（X on Y）：**

$$\hat{\alpha}_1 = \frac{S_{xy}}{S_{yy}}$$

**两者关系：**

$$\hat{\alpha}_1 = \frac{S_{xy}}{S_{yy}} = \hat{\beta}_1 \cdot \frac{S_{xx}}{S_{yy}}$$

一般情况下 $S_{xx} \neq S_{yy}$，故 $\hat{\alpha}_1 \neq 1/\hat{\beta}_1$（**注意：不是倒数！**）

**$R^2$ 的计算：**

$$R^2_{Y \text{ on } X} = \frac{S_{xy}^2}{S_{xx} S_{yy}}, \quad R^2_{X \text{ on } Y} = \frac{S_{xy}^2}{S_{yy} S_{xx}}$$

两者相等！$R^2 = \rho_{xy}^2$ 与回归方向无关。

**几何直觉：**

```
          Y
          |        /← 回归 Y on X（最小化垂直距离到水平线）
      ●   |   ●  /
        ● | ●  /
    ●   ● |● /
          |/
──────────●──────────── X
         /|
        / |   ← 回归 X on Y（最小化水平距离到垂直线）
       /  |

两条回归线不同！除非 |ρ| = 1（完全相关），
两条线只在 (x̄, ȳ) 处相交。
```

**两条回归线的夹角：** 当 $\rho \to 0$，两线趋于垂直（一条水平，一条竖直）；当 $|\rho| \to 1$，两线趋于重合。

**$t$ 统计量：**

$$t_{Y \text{ on } X} = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)} = \hat{\beta}_1 \cdot \frac{\sqrt{S_{xx}}}{s_Y}, \quad t_{X \text{ on } Y} = \frac{\hat{\alpha}_1}{SE(\hat{\alpha}_1)} = \hat{\alpha}_1 \cdot \frac{\sqrt{S_{yy}}}{s_X}$$

可以证明两个 $t$ 统计量**数值相等**（只是方向含义不同）：

$$t_{Y \text{ on } X}^2 = t_{X \text{ on } Y}^2 = \frac{S_{xy}^2}{s^2 S_{xx} S_{yy}} \cdot \frac{(n-2)}{1}$$

### 结论汇总

| 统计量 | 变化 |
|--------|------|
| $\hat{\beta}_1$ | $\frac{S_{xy}}{S_{xx}} \to \frac{S_{xy}}{S_{yy}}$，**完全不同** |
| $\hat{\beta}_1 \cdot \hat{\alpha}_1$ | $= \frac{S_{xy}^2}{S_{xx}S_{yy}} = R^2$（乘积等于 $R^2$！）|
| $R^2$ | **不变**（= $\rho_{xy}^2$，与方向无关）|
| $t$ 统计量绝对值 | **不变**（两个回归的 $t$ 值相等）|

> **Insight：** $\hat{\beta}_1 \cdot \hat{\alpha}_1 = R^2$ 是一个优美的恒等式。只有在 $R^2 = 1$（完美线性关系）时，$\hat{\alpha}_1 = 1/\hat{\beta}_1$。这说明**因果方向的假设不能由数据本身确定**——数据无法区分"X 导致 Y"和"Y 导致 X"，这正是内生性问题的核心困境。

---

## 问题 2：将数据集整体复制（Duplicate the Dataset）

### 问题

将原始数据集 $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ 完整复制一份，变成 $2n$ 个观测（每行出现两次），$\hat{\beta}$、$R^2$、$t$ 统计量如何变化？

### 推导

设原始数据：$S_{xx}, S_{yy}, S_{xy}, \bar{x}, \bar{y}, n$

复制后的新数据集（$2n$ 个观测）：

$$\bar{x}' = \bar{x}, \quad \bar{y}' = \bar{y}$$（均值不变，因为每个值出现两次）

$$S'_{xx} = \sum_{i=1}^{2n}(x_i' - \bar{x}')^2 = 2\sum_{i=1}^{n}(x_i - \bar{x})^2 = 2S_{xx}$$

同理 $S'_{yy} = 2S_{yy}$，$S'_{xy} = 2S_{xy}$

**系数：**

$$\hat{\beta}_1' = \frac{S'_{xy}}{S'_{xx}} = \frac{2S_{xy}}{2S_{xx}} = \hat{\beta}_1 \quad \checkmark$$

**$R^2$：**

$$R'^2 = \frac{(S'_{xy})^2}{S'_{xx} S'_{yy}} = \frac{(2S_{xy})^2}{2S_{xx} \cdot 2S_{yy}} = \frac{S_{xy}^2}{S_{xx} S_{yy}} = R^2 \quad \checkmark$$

**残差方差 $s^2$：**

$$s'^2 = \frac{\text{RSS}'}{2n - 2} = \frac{2 \cdot \text{RSS}}{2(n-1)} = \frac{\text{RSS}}{n-1}$$

原始：$s^2 = \frac{\text{RSS}}{n-2}$

**注意：** 复制后自由度变为 $2n-2$，而不是原来的 $n-2$。

$$\frac{s'^2}{s^2} = \frac{n-2}{n-1} \approx 1 \text{（对大 } n\text{）}$$

对于中等样本量：$s'^2 \approx s^2$（略小，因分母更大）

**标准误：**

$$SE'(\hat{\beta}_1) = \frac{s'}{\sqrt{S'_{xx}}} = \frac{s'}{\sqrt{2S_{xx}}} \approx \frac{s}{\sqrt{2} \cdot \sqrt{S_{xx}}} = \frac{SE(\hat{\beta}_1)}{\sqrt{2}}$$

标准误缩小为原来的 $\approx 1/\sqrt{2}$！

**$t$ 统计量：**

$$t' = \frac{\hat{\beta}_1'}{SE'(\hat{\beta}_1)} \approx \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)/\sqrt{2}} = \sqrt{2} \cdot t$$

$t$ 统计量增大为原来的 $\approx \sqrt{2}$ 倍，**显著性增加**！

### 直觉解释

```
原始数据（n个点）:
  ●    ●
    ●●    ●
  ●    ●

复制后（2n个点，完全重叠）:
  ⊕    ⊕          ← 每个点现在是两个重叠的点
    ⊕⊕    ⊕
  ⊕    ⊕

信息量没有增加，但统计量"以为"样本量翻倍！
→ 系数不变（正确）
→ 标准误减半√2（错误！虚假精度）
→ t 统计量增大√2（虚假显著性！）
```

### 结论汇总

| 统计量 | 变化 | 原因 |
|--------|------|------|
| $\hat{\beta}$ | **不变** | $2S_{xy}/2S_{xx}$ |
| $R^2$ | **不变** | 相关系数不变 |
| $s^2$（残差方差） | 略减小（$\approx$不变） | 自由度 $2n-2$ |
| $SE(\hat{\beta})$ | **减小 $\approx\sqrt{2}$ 倍** | $\sqrt{S_{xx}}$ 增大 $\sqrt{2}$ |
| $t$ 统计量 | **增大 $\approx\sqrt{2}$ 倍** | SE 减小，系数不变 |
| p 值 | **减小（更显著）** | $t$ 增大 |

> **警告（数据伦理）：** 这正是一种**数据操纵**的方式——通过重复数据使结果看起来更显著，但实际信息量完全没有增加。量化回测中，如果不小心重复加载了数据或用了重叠的训练集，会产生完全相同的效果：结果看起来更稳健，实际上是假象。

---

## 问题 3：仅对 Y 乘以常数 $c$

### 问题

将 $Y$ 替换为 $Y^* = cY$（$c > 0$ 为常数），各统计量如何变化？

### 推导

$$S^*_{xy} = \sum(x_i - \bar{x})(cy_i - c\bar{y}) = c \cdot S_{xy}$$

$$S^*_{yy} = c^2 S_{yy}, \quad S^*_{xx} = S_{xx} \text{（X未变）}$$

**系数：**

$$\hat{\beta}_1^* = \frac{c S_{xy}}{S_{xx}} = c \hat{\beta}_1$$

$$\hat{\beta}_0^* = c\bar{y} - c\hat{\beta}_1 \bar{x} = c(\bar{y} - \hat{\beta}_1 \bar{x}) = c\hat{\beta}_0$$

所有系数均乘以 $c$（单位换算！）

**$R^2$：**

$$R^{*2} = \frac{(cS_{xy})^2}{S_{xx} \cdot c^2 S_{yy}} = \frac{S_{xy}^2}{S_{xx} S_{yy}} = R^2$$

**残差：** $\hat{\varepsilon}_i^* = cy_i - c\hat{y}_i = c\hat{\varepsilon}_i$

$$\text{RSS}^* = c^2 \text{RSS}, \quad s^{*2} = c^2 s^2$$

**标准误：**

$$SE^*(\hat{\beta}_1^*) = \frac{cs}{\sqrt{S_{xx}}} = c \cdot SE(\hat{\beta}_1)$$

**$t$ 统计量：**

$$t^* = \frac{c\hat{\beta}_1}{c \cdot SE(\hat{\beta}_1)} = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)} = t \quad \checkmark$$

### 单位换算应用

若 $Y$ 是美元收益，$Y^* = Y/100$ 是百元收益：
- $\hat{\beta}_1^*$ = 原来的 $1/100$（正确，单位一致）
- $t$ 统计量、$R^2$ 不变（统计显著性与单位无关）

> **Insight：** $t$ 统计量和 $R^2$ 是**无量纲**的，不受单位缩放影响。这是为什么我们用标准化系数（$\beta^*$，见问题 9）比较不同变量"哪个更重要"——消除了单位的影响。

---

## 问题 4：仅对 X 乘以常数 $c$

### 问题

将 $X$ 替换为 $X^* = cX$，各统计量如何变化？

### 推导

$$S^*_{xy} = c S_{xy}, \quad S^*_{xx} = c^2 S_{xx}, \quad S^*_{yy} = S_{yy}$$

**系数：**

$$\hat{\beta}_1^* = \frac{cS_{xy}}{c^2 S_{xx}} = \frac{\hat{\beta}_1}{c}$$

（直觉：$X$ 单位变大 $c$ 倍，斜率变小 $c$ 倍，乘积 $\hat{\beta}_1^* \cdot X^* = \hat{\beta}_1 \cdot X$ 不变）

$$\hat{\beta}_0^* = \bar{y} - \frac{\hat{\beta}_1}{c} \cdot c\bar{x} = \bar{y} - \hat{\beta}_1\bar{x} = \hat{\beta}_0$$

截距**不变**（因为预测线仍过 $(\bar{x}, \bar{y})$）

**$R^2$：** 不变（同上）

**标准误：**

$$SE^*(\hat{\beta}_1^*) = \frac{s}{\sqrt{c^2 S_{xx}}} = \frac{s}{c\sqrt{S_{xx}}} = \frac{SE(\hat{\beta}_1)}{c}$$

**$t$ 统计量：**

$$t^* = \frac{\hat{\beta}_1/c}{SE(\hat{\beta}_1)/c} = t \quad \checkmark$$

### 结论

| | Y 乘以 $c$ | X 乘以 $c$ |
|--|-----------|-----------|
| $\hat{\beta}_1$ | $\times c$ | $\div c$ |
| $\hat{\beta}_0$ | $\times c$ | 不变 |
| $R^2$ | 不变 | 不变 |
| $t$ | 不变 | 不变 |

---

## 问题 5：对 X 加常数（平移）

### 问题

将 $X$ 替换为 $X^* = X + a$（$a$ 为常数），各统计量如何变化？

### 推导

$$x_i^* - \bar{x}^* = (x_i + a) - (\bar{x} + a) = x_i - \bar{x}$$

所有离差不变！

$$S^*_{xy} = S_{xy}, \quad S^*_{xx} = S_{xx}, \quad S^*_{yy} = S_{yy}$$

**斜率：** $\hat{\beta}_1^* = \hat{\beta}_1$（不变）

**截距：**

$$\hat{\beta}_0^* = \bar{y} - \hat{\beta}_1^* \bar{x}^* = \bar{y} - \hat{\beta}_1(\bar{x} + a) = \hat{\beta}_0 - a\hat{\beta}_1$$

截距改变了！（但预测值 $\hat{y}_i = \hat{\beta}_0^* + \hat{\beta}_1^* x_i^* = \hat{\beta}_0 + \hat{\beta}_1 x_i$ 不变）

**所有其他量：** $R^2$、$s^2$、$SE(\hat{\beta}_1)$、$t$ 统计量——全部不变。

### 中心化的作用

将 $X$ 中心化（$X^* = X - \bar{x}$，即 $a = -\bar{x}$）：

$$\hat{\beta}_0^* = \bar{y} - \hat{\beta}_1 \cdot 0 = \bar{y}$$

截距变为 $Y$ 的均值，斜率不变，**多重共线性（$X$ 与截距列的相关性）降低**，数值更稳定。

> **Insight（交互项中心化）：** 在模型 $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2$ 中，若不中心化，$X_1$ 与 $X_1 X_2$ 高度相关（因为 $X_1 X_2 \approx X_1 \cdot \text{const}$ 当 $X_2$ 均值不为零）。中心化后，$\tilde{X}_1 \tilde{X}_2$ 与 $\tilde{X}_1$、$\tilde{X}_2$ 的相关性大幅降低，使主效应系数的解释更清晰。

---

## 问题 6：对 Y 加常数（平移）

### 问题

将 $Y$ 替换为 $Y^* = Y + b$，各统计量如何变化？

### 推导

$$y_i^* - \bar{y}^* = y_i - \bar{y}$$（离差不变）

$$S^*_{xy} = S_{xy}, \quad S^*_{xx} = S_{xx}, \quad S^*_{yy} = S_{yy}$$

**斜率：** $\hat{\beta}_1^* = \hat{\beta}_1$（不变）

**截距：** $\hat{\beta}_0^* = (\bar{y}+b) - \hat{\beta}_1\bar{x} = \hat{\beta}_0 + b$（增加 $b$）

**所有其他量不变：** $R^2$、$SE(\hat{\beta}_1)$、$t$ 统计量。

### 直觉

对 $Y$ 加常数相当于将整个散点图**整体上移 $b$**，回归线也上移 $b$，但斜率和相关性完全不变。

---

## 问题 7：重复每个观测 $k$ 次

### 问题

将数据集中的每个观测 $(x_i, y_i)$ 重复 $k$ 次，变成 $kn$ 个观测，各统计量如何变化？

### 推导

与问题 2（$k=2$）的推广：

$$S'_{xy} = k S_{xy}, \quad S'_{xx} = k S_{xx}, \quad S'_{yy} = k S_{yy}$$

**系数：** 不变

**$R^2$：** 不变

**残差方差：**

$$s'^2 = \frac{k \cdot \text{RSS}}{kn - 2} \approx \frac{\text{RSS}}{n} \cdot \frac{n-2}{n} \cdot \frac{k(n-2)}{kn-2}$$

对大 $n$，$s'^2 \approx s^2 \cdot \frac{n-2}{n} \cdot \frac{k}{k} \approx s^2$（近似不变，分母 $kn-2 \approx k(n-2)$）

**标准误：**

$$SE'(\hat{\beta}_1) = \frac{s'}{\sqrt{k S_{xx}}} \approx \frac{s}{\sqrt{k} \cdot \sqrt{S_{xx}}} = \frac{SE(\hat{\beta}_1)}{\sqrt{k}}$$

**$t$ 统计量：** 增大 $\sqrt{k}$ 倍

### 一般化结论

$$t' \approx \sqrt{k} \cdot t, \quad SE' \approx \frac{SE}{\sqrt{k}}$$

这与"样本量从 $n$ 增加到 $kn$"的效果相同，但**没有增加任何信息**——是虚假的精度提升。

> **Insight（时间序列重叠问题）：** 在金融研究中，若用月度收益数据但以「周度」频率报告，或用重叠窗口计算滚动指标，会产生**人为序列相关**，使 $t$ 统计量虚增。这是因子研究中常见的数据处理错误。Newey-West 标准误部分校正了此问题。

---

## 问题 8：对 X 和 Y 同时乘以不同常数

### 问题

令 $X^* = aX$，$Y^* = bY$，系数如何变化？

### 推导

综合问题 3 和 4：

$$\hat{\beta}_1^* = \frac{bS_{xy}}{a^2 S_{xx}} \cdot a = \frac{b}{a}\hat{\beta}_1$$

等价地：

$$\hat{\beta}_1^* = \frac{S^*_{xy}}{S^*_{xx}} = \frac{ab S_{xy}}{a^2 S_{xx}} = \frac{b}{a}\hat{\beta}_1$$

**$R^2$：** 不变（如前证明）

**$t$ 统计量：** 不变

### 标准化系数（Beta 系数）

令 $a = \text{SD}(X) = \sqrt{S_{xx}/(n-1)}$，$b = \text{SD}(Y) = \sqrt{S_{yy}/(n-1)}$：

$$\hat{\beta}_1^* = \frac{b}{a}\hat{\beta}_1 = \frac{\text{SD}(Y)}{\text{SD}(X)}\hat{\beta}_1 = \hat{\beta}_1 \cdot \frac{\sqrt{S_{xx}}}{\sqrt{S_{yy}}} \cdot \frac{\sqrt{n-1}}{\sqrt{n-1}}$$

化简得：

$$\hat{\beta}_1^* = \frac{S_{xy}}{\sqrt{S_{xx} S_{yy}}} = \rho_{xy}$$

**标准化回归系数 = 相关系数！**（简单回归情形）

在多元回归中，标准化系数 $\beta_j^*$ 表示其他变量固定时，$X_j$ 每增加 1 个标准差，$Y$ 增加 $\beta_j^*$ 个标准差，用于比较不同 $X$ 变量的相对重要性。

---

## 问题 9：对 X 和 Y 同时标准化（Z-score）

### 问题

令 $\tilde{X} = (X - \bar{x})/s_x$，$\tilde{Y} = (Y - \bar{y})/s_y$，回归 $\tilde{Y}$ 对 $\tilde{X}$，结果如何？

### 推导

标准化是先中心化（问题 5 和 6 的组合）再缩放（问题 3 和 4）。

标准化后 $\sum \tilde{x}_i^2 = n-1$，$\sum \tilde{y}_i^2 = n-1$，$\sum \tilde{x}_i \tilde{y}_i = (n-1)\rho_{xy}$

$$\hat{\tilde{\beta}}_1 = \frac{\sum \tilde{x}_i \tilde{y}_i}{\sum \tilde{x}_i^2} = \frac{(n-1)\rho_{xy}}{n-1} = \rho_{xy}$$

$$\hat{\tilde{\beta}}_0 = \bar{\tilde{y}} - \hat{\tilde{\beta}}_1 \bar{\tilde{x}} = 0 - \rho_{xy} \cdot 0 = 0$$

**截距自动为 0！斜率等于相关系数！**

**$R^2$：** 不变（$= \rho_{xy}^2$，与变换无关）

**$t$ 统计量：** 不变

### 重要恒等式

$$\text{标准化回归斜率} = \rho_{xy} = \sqrt{R^2} \cdot \text{sign}(\hat{\beta}_1)$$

---

## 问题 10：对 Y 取对数

### 问题

将 $Y$ 替换为 $\ln Y$（要求 $Y > 0$），回归 $\ln Y$ 对 $X$，如何解释结果？与原回归的关系是什么？

### 推导与解释

设 $\ln Y = \beta_0 + \beta_1 X + \varepsilon$

**系数解释（弹性/半弹性）：**

$$\frac{d(\ln Y)}{dX} = \beta_1 \implies \frac{dY/Y}{dX} = \beta_1$$

$X$ 增加 1 单位，$Y$ 变化约 $100\beta_1\%$（**半弹性**）

**$R^2$ 的不可比性：**

- 原模型的 $R^2$ 度量解释 $Y$ 方差的比例
- 对数模型的 $R^2$ 度量解释 $\ln Y$ 方差的比例
- **两者不可直接比较！**（不同的因变量）

**正确的比较方法：** 将对数模型的预测值转换回原始尺度，计算在 $Y$ 尺度上的预测误差。

注意：$E[\hat{Y}] = e^{\hat{\mu} + s^2/2}$（需要对数正态修正，而非 $e^{\hat{\mu}}$）。

### 四种对数模型的系数解释

| 模型 | 斜率解释 |
|------|---------|
| $Y = \beta_0 + \beta_1 X$ | $X$ +1，$Y$ 变化 $\beta_1$（线性） |
| $\ln Y = \beta_0 + \beta_1 X$ | $X$ +1，$Y$ 变化约 $100\beta_1\%$（半弹性）|
| $Y = \beta_0 + \beta_1 \ln X$ | $X$ +1%，$Y$ 变化约 $\beta_1/100$（半弹性）|
| $\ln Y = \beta_0 + \beta_1 \ln X$ | $X$ +1%，$Y$ 变化约 $\beta_1\%$（弹性）|

> **Insight（量化中的对数变换）：** 用 $\ln(\text{价格})$ 而非价格回归有几个好处：使收益率（差分对数）近似正态；消除量级差异（苹果股价 vs 便士股）；使乘法关系变为加法（方便估计）。但 $R^2$ 不可跨模型比较是一个容易忽略的陷阱。

---

## 问题 11：X 的测量误差（Classical Measurement Error）

### 问题

真实变量 $X^*$ 不可观测，只能观测到带噪声的 $X = X^* + \eta$（$\eta \sim (0, \sigma_\eta^2)$，与 $X^*, \varepsilon$ 独立）。用 $X$ 代替 $X^*$ 回归，系数如何变化？

### 推导（衰减偏误）

设真实模型：$Y = \beta_0 + \beta_1 X^* + \varepsilon$

用含误差的 $X = X^* + \eta$ 回归：

$$\text{plim}\,\hat{\beta}_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = \frac{\text{Cov}(X^*+\eta, \beta_1 X^*+\varepsilon)}{\text{Var}(X^*+\eta)} = \frac{\beta_1 \text{Var}(X^*)}{\text{Var}(X^*)+\sigma_\eta^2}$$

$$\boxed{\text{plim}\,\hat{\beta}_1 = \beta_1 \cdot \underbrace{\frac{\text{Var}(X^*)}{\text{Var}(X^*)+\sigma_\eta^2}}_{\lambda \in (0,1)} < \beta_1}$$

$\lambda = \text{Var}(X^*)/\text{Var}(X)$ 称为**信噪比**，$\hat{\beta}_1$ 被向零**衰减**（attenuation bias）。

**衰减的直觉：**

```
真实关系 (Y vs X*):          观测到的 (Y vs X = X* + noise):
     ●                             ●  ●
  ●    ●                         ●      ●
    ●●    ●    →    噪声使点     ● ●  ●● ●●
  ●    ●               模糊→         ●    ●  ●
     ●                           ●    ●●
                                        ●

斜率变小（更平）！
```

**影响 $R^2$：**

$$R^{*2} = \rho_{XY}^2 = \lambda^2 \cdot \rho_{X^*Y}^2 < R^2_{\text{true}}$$

测量误差同时降低系数估计和 $R^2$（两者都被"稀释"）。

> **Insight（量化中的 Beta 衰减）：** 用历史数据估计股票 Beta 时，估计值 $\hat{\beta}$ 受到估计误差的影响，会向均值（$\beta=1$）衰减——高 Beta 股票的真实 Beta 可能被低估，低 Beta 股票被高估。**Vasicek 调整** 正是对此的修正：$\tilde{\beta} = w\hat{\beta} + (1-w)\bar{\beta}$，$w$ 与测量误差大小成反比。

---

## 问题 12：加入完全多重共线变量

### 问题

在模型 $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$ 中，若 $X_2 = 2X_1$（完全共线），会发生什么？

### 推导

设计矩阵 $X = [1, X_1, X_2] = [1, X_1, 2X_1]$，第3列 = 2 × 第2列。

$$X^\top X = \begin{pmatrix} n & \sum x_{1i} & 2\sum x_{1i} \\ \sum x_{1i} & S_{11} & 2S_{11} \\ 2\sum x_{1i} & 2S_{11} & 4S_{11} \end{pmatrix}$$

$\det(X^\top X) = 0$（第3行 = 2 × 第2行），**矩阵奇异，$\hat{\beta}$ 无唯一解**！

任意 $\gamma$ 下，$(\hat{\beta}_1 + 2\gamma, \hat{\beta}_2 - \gamma)$ 都给出相同的预测值：

$$\hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 = (\hat{\beta}_1 + 2\gamma)x_1 + (\hat{\beta}_2 - \gamma)(2x_1) = \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2$$

**系数不唯一，但预测值唯一。**

| 统计量 | 结果 |
|--------|------|
| $\hat{\beta}$ | 不唯一（无穷多解）|
| 预测值 $\hat{Y}$ | **唯一** |
| $R^2$ | **唯一** |
| $SE(\hat{\beta})$ | $\to \infty$（标准误爆炸）|

---

## 问题 13：Frisch-Waugh-Lovell（偏回归）定理

### 问题

在多元回归 $Y = X_1\beta_1 + X_2\beta_2 + \varepsilon$ 中，$\hat{\beta}_2$ 等于什么？与单独对 $Y$ 回归 $X_2$ 有何区别？

### 定理

**FWL 定理：** $\hat{\beta}_2$ 等于将 $M_1 Y$（$Y$ 对 $X_1$ 回归的残差）对 $M_1 X_2$（$X_2$ 对 $X_1$ 回归的残差）做单变量回归的斜率。

其中 $M_1 = I - X_1(X_1^\top X_1)^{-1}X_1^\top$ 是将向量投影到 $X_1$ 的列空间的**残差化算子**。

**证明要点：**

设 $\tilde{Y} = M_1 Y$，$\tilde{X}_2 = M_1 X_2$，则：

$$\hat{\beta}_2^{FWL} = (\tilde{X}_2^\top \tilde{X}_2)^{-1}\tilde{X}_2^\top \tilde{Y}$$

可验证这等于完整多元回归中 $\beta_2$ 的 OLS 估计。

### 可视化理解

```
原始空间:                    去除 X₁ 影响后的残差空间:

      Y                           M₁Y  (Y的残差)
    / |                            |
   /  |  X₂                        |    M₁X₂ (X₂的残差)
  /   | /                          |   /
 /    |/                           |  /
X₁---+                            +

FWL: β̂₂ = 回归(M₁Y 对 M₁X₂)
含义: β̂₂ 衡量"去除X₁的影响后"，X₂对Y的纯粹效应
```

### 应用：固定效应的理解

面板数据固定效应（within 估计量）等价于：先对所有变量做个体去均值，再跑 OLS——这正是 FWL 定理的应用（$X_1$ = 个体虚假变量组）。

> **Insight：** FWL 定理的深刻含义：**多元回归系数衡量的是"净效应"——控制其他变量后X的边际贡献**。这是回归作为"条件均值"工具的几何基础，也是为什么多元回归不能直接将单变量回归结果叠加（系数会因控制变量而改变）。

---

## 问题 14：加入与 Y 线性相关的 X

### 问题

在回归 $Y = \beta_0 + \beta_1 X_1 + \varepsilon$ 中，加入 $X_2 = Y + \delta$（$\delta$ 是随机噪声），会发生什么？

### 分析

这是**因变量用作自变量**的情形，导致严重内生性（$\text{Cov}(X_2, \varepsilon) \neq 0$）。

$$\hat{\beta}_2 \text{ 的 OLS 不一致（} E[\varepsilon | X_2] \neq 0\text{）}$$

若 $\delta = 0$（即 $X_2 = Y$ 完全相同）：

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 Y$$

则 $(1-\beta_2) Y = \beta_0 + \beta_1 X_1$，完全的内生性问题，OLS 无意义。

> **Insight：** 这是**同期变量（contemporaneous variable）**作为控制变量的错误。例如，研究"利率对 GDP 的影响"，如果把"当期通胀"（本身受 GDP 影响）作为控制变量，会引入内生性，使利率系数偏误。正确做法是使用**滞后变量**作为控制变量。

---

## 问题 15：加入常数（全为1的）列 vs 移除截距

### 问题

比较以下两个模型：
- 模型A：$Y = \beta_0 + \beta_1 X + \varepsilon$（有截距）
- 模型B：$Y = \beta_1 X + \varepsilon$（无截距，强制通过原点）

它们的 $\hat{\beta}_1$ 是否相同？$R^2$ 是否可比？

### 推导

**模型A（有截距）：**

$$\hat{\beta}_1^A = \frac{S_{xy}}{S_{xx}} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sum(x_i-\bar{x})^2}$$

**模型B（无截距）：**

$$\hat{\beta}_1^B = \frac{\sum x_i y_i}{\sum x_i^2}$$

**两者不同！** 只有当 $\bar{x} = 0$ 或 $\bar{y} = 0$ 时才相等。

**$R^2$ 的不可比性：**

- 有截距时：$R^2 = 1 - \text{RSS}/\text{TSS}$，其中 $\text{TSS} = \sum(y_i-\bar{y})^2$
- 无截距时：$R^2 = 1 - \text{RSS}/\sum y_i^2$（分母不同！）

无截距模型的 $R^2$ 通常更高，但这是定义不同造成的，**不能说明无截距模型更好**。

### 关键后果

去掉截距后，**残差均值 $\neq 0$**（OLS 不再保证此性质），可能导致：
- 系数偏误（若真实截距 $\neq 0$）
- 残差和 $\neq 0$，违反 OLS 基本性质

> **Insight：** 金融中唯一有理论依据省略截距的场景：CAPM 下的 Beta 估计（$r_i - r_f = \beta(r_m - r_f)$），理论上 $\alpha = 0$，但实践中**仍应保留截距**——因为 $\alpha$ 正是想检验 CAPM 是否成立的证据。强制 $\alpha = 0$ 是循环论证。

---

## 问题 16：加入完全无关的随机变量 $Z$

### 问题

在 $Y = \beta_0 + \beta_1 X + \varepsilon$ 中，加入一个与 $Y, X$ 完全无关的 $Z$（$Z \perp X, Z \perp Y$），原系数 $\hat{\beta}_1$ 会改变吗？

### 理论结果

在总体（无限样本）中，$Z$ 与 $X$ 正交，FWL 定理告诉我们：

$$\hat{\beta}_1 \text{ (多元)} = \hat{\beta}_1 \text{ (简单)}$$

即加入正交的无关变量，原系数**不变**。

**但在有限样本中：** 由于抽样误差，$Z$ 与 $X$ 不会完全正交，故系数会有轻微变化（标准误增大，因为多耗一个自由度）。

**$R^2$ 的变化：**

$$R^2 \geq R^2_{\text{原}} \text{（非负增加，因 OLS 总可令 Z 系数=0）}$$

**调整 $R^2$** 可能下降（因为加入无关变量浪费自由度）。

**$t$ 统计量（Z 的系数）：** 约服从 $t(n-3)$，期望为 0，但有约 5% 的概率 $|t| > 2$（I型错误）。

> **Insight（数据窥探警告）：** 若你在 100 个候选变量中搜索显著因子，即使所有变量都无效，也期望有 5 个（Bonferroni 前）显示 $p < 0.05$。**这是量化因子挖掘中"多重检验"问题的统计学基础**——不经 Bonferroni 校正报告的因子，很可能只是随机噪声。

---

## 问题 17：两步回归（Auxiliary Regression）

### 问题

先回归 $Y$ 对 $X_1$ 得残差 $e_1$，再回归 $e_1$ 对 $X_2$，得到的 $X_2$ 系数与直接多元回归是否相同？

### 推导

**第一步：** $e_1 = M_1 Y$（$Y$ 对 $X_1$ 的残差）

**第二步：** 回归 $e_1$ 对 $X_2$，系数为：

$$\hat{\gamma}_2 = \frac{X_2^\top e_1}{X_2^\top X_2} = \frac{X_2^\top M_1 Y}{X_2^\top X_2}$$

**完整多元回归的 $X_2$ 系数（FWL）：**

$$\hat{\beta}_2 = \frac{(M_1 X_2)^\top M_1 Y}{(M_1 X_2)^\top M_1 X_2} = \frac{X_2^\top M_1 Y}{X_2^\top M_1 X_2}$$

**两者相差分母：** $X_2^\top X_2 \neq X_2^\top M_1 X_2$

**结论：两步回归的系数与多元回归不同！**

正确的两步做法（等价于多元回归）：

1. 回归 $Y$ 对 $X_1$ 得 $e_Y$
2. 回归 $X_2$ 对 $X_1$ 得 $e_{X_2}$（**这步常被忘记！**）
3. 回归 $e_Y$ 对 $e_{X_2}$——这才等价于多元回归

> **Insight：** 这正是 Frisch-Waugh 定理的完整表述——必须同时对 $Y$ 和 $X_2$ 都去除 $X_1$ 的影响，才能得到正确的偏系数。**只对 $Y$ 做残差化，但不对 $X_2$ 做，是常见的两步回归错误。**

---

## 问题 18：Y 对 X 的回归 vs X 的排名对 Y 的排名的回归

### 问题

用 $Y$ 的排名（rank）对 $X$ 的排名做 OLS 回归，与直接用原始数据的 OLS 回归有何关系？

### 分析

设排名变量 $R_X = \text{rank}(X)$，$R_Y = \text{rank}(Y)$（均匀分布在 $1, 2, \ldots, n$）。

标准化排名：$\tilde{R}_X = (R_X - (n+1)/2)/s_{R}$，则 $\text{Corr}(\tilde{R}_X, \tilde{R}_Y) = \rho_s$（Spearman 相关系数）

**排名回归的斜率 = Spearman 相关系数**（标准化后）

**与 Pearson 相关的比较：**

| | Pearson | Spearman |
|--|---------|---------|
| 基于 | 原始值 | 排名 |
| 对异常值 | 敏感 | 稳健 |
| 线性关系 | 最优 | 单调关系即可 |
| 适用场景 | 正态数据 | 非正态/厚尾 |

> **Insight（量化信号评估）：** 量化因子研究中，**IC（信息系数）**通常定义为因子值与未来收益的 Spearman 相关——因为金融收益厚尾，Pearson 相关对极端值（公司破产、特殊事件）极为敏感，Spearman 更稳健。**Rank IC > 0.05 通常认为有统计意义。**

---

## 问题 19：加入二次项（多项式回归）

### 问题

在 $Y = \beta_0 + \beta_1 X + \varepsilon$ 中，加入 $X^2$ 项变为 $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \varepsilon$，原系数 $\hat{\beta}_1$ 会改变吗？

### 分析

加入 $X^2$ 后，由 FWL 定理：

$$\hat{\beta}_1^{\text{new}} = \text{回归}(M_2 Y \text{ 对 } M_2 X)$$

其中 $M_2$ 是去除 $X^2$ 影响的残差化算子。

由于 $X$ 与 $X^2$ 通常相关（除非 $X$ 是零均值对称的），$M_2 X \neq X$，故：

$$\hat{\beta}_1^{\text{new}} \neq \hat{\beta}_1^{\text{original}}$$

**特殊情况（$X$ 先中心化）：**

若 $X$ 均值为 0，则 $\text{Cov}(X, X^2) = E[X^3] - E[X]E[X^2] = E[X^3]$。

若 $X$ 对称分布（如正态），$E[X^3] = 0$，即 $X$ 和 $X^2$ 正交，此时 $\hat{\beta}_1^{\text{new}} = \hat{\beta}_1^{\text{original}}$。

> **Insight：** 加入非线性项（$X^2, X^3, \ldots$）会改变线性项的系数，除非先中心化。**因此，在报告多项式回归时，必须对 $X$ 先中心化**，使线性系数有清晰解释（在 $X=0$，即 $\bar{X}$ 处的斜率），否则线性项的系数无法单独解释。

---

## 问题 20：删除一个观测点（Leave-One-Out）

### 问题

删除第 $i$ 个观测点后，回归系数变化多少？

### Sherman-Morrison 更新公式

设全样本估计量 $\hat{\beta}$，删除第 $i$ 个观测后的估计量 $\hat{\beta}_{(-i)}$：

$$\hat{\beta}_{(-i)} = \hat{\beta} - (X^\top X)^{-1} x_i \frac{\hat{\varepsilon}_i}{1 - h_{ii}}$$

其中 $h_{ii}$ 是帽子矩阵的第 $i$ 个对角元素，$\hat{\varepsilon}_i$ 是第 $i$ 个残差。

**删除后的预测误差（LOOCV 误差）：**

$$\hat{\varepsilon}_i^{(-i)} = y_i - x_i^\top \hat{\beta}_{(-i)} = \frac{\hat{\varepsilon}_i}{1 - h_{ii}}$$

称为**学生化残差**（Studentized residual），是异常值检测的标准工具。

**系数变化量：**

$$\|\hat{\beta} - \hat{\beta}_{(-i)}\| \propto \frac{|\hat{\varepsilon}_i| \cdot h_{ii}^{1/2}}{1 - h_{ii}} \sim \sqrt{\text{Cook's }D_i}$$

> **Insight（LOOCV 的计算捷径）：** 注意 $\hat{\varepsilon}_i^{(-i)} = \hat{\varepsilon}_i/(1-h_{ii})$ 意味着 **LOOCV 误差可以从全样本拟合直接计算，无需真的重新拟合 $n$ 次！** 这是线性回归的一个强大性质，使 LOOCV 比 $k$-fold CV 快得多（对线性模型而言）。

---

## 问题 21：加入 X 的噪声复制（Noisy Duplicate）

### 问题

在模型 $Y = \beta_0 + \beta_1 X + \varepsilon$ 中，加入 $X$ 的噪声版本 $\tilde{X} = X + \eta$（$\eta \sim N(0, \sigma_\eta^2)$ 与其他变量独立），结果如何？

### 分析

回归 $Y$ 对 $X$ 和 $\tilde{X} = X + \eta$：

设计矩阵中 $X$ 与 $\tilde{X}$ 高度相关（$\rho = \text{Var}(X)/\sqrt{\text{Var}(X)\text{Var}(\tilde{X})}$），产生多重共线性。

**极端情况 $\sigma_\eta \to 0$（$\tilde{X} \to X$）：**

完全多重共线性，系数不唯一。

**小噪声（$\sigma_\eta$ 小）：**

VIF 极大，$SE(\hat{\beta}_1), SE(\hat{\beta}_2) \to \infty$，系数不稳定但预测值稳定。

**实际影响：**

$$\hat{\beta}_1 + \hat{\beta}_2 \approx \beta_1 \text{（总效应稳定，但分配不稳定）}$$

> **Insight（多因子模型的多重共线性）：** 在 Fama-French 因子模型中，价值（HML）和盈利能力（RMW）因子高度相关（r≈0.6），导致各自的系数估计不稳定——系数可能为负，但经济上意义不明。这是为什么多因子模型要用正交化（GRS检验）或 Ridge 回归来稳定系数估计。

---

## 问题 22：混合两个子样本

### 问题

有两个子样本：$A$（$n_A$ 个观测，$\hat{\beta}_A$）和 $B$（$n_B$ 个观测，$\hat{\beta}_B$）。合并后的回归系数 $\hat{\beta}_{AB}$ 是 $\hat{\beta}_A$ 和 $\hat{\beta}_B$ 的加权平均吗？

### 推导

**结论：一般不是！** 存在**辛普森悖论（Simpson's Paradox）**的可能性。

$$\hat{\beta}_{AB} \neq \frac{n_A \hat{\beta}_A + n_B \hat{\beta}_B}{n_A + n_B}$$

真正的合并估计量：

$$\hat{\beta}_{AB} = \frac{S_{xy}^A + S_{xy}^B}{S_{xx}^A + S_{xx}^B}$$

而非按 $n_A, n_B$ 加权。权重实际上按 $S_{xx}^A, S_{xx}^B$（X的方差，即"信息量"）加权：

$$\hat{\beta}_{AB} = \frac{S_{xx}^A}{S_{xx}^A + S_{xx}^B}\hat{\beta}_A + \frac{S_{xx}^B}{S_{xx}^A + S_{xx}^B}\hat{\beta}_B$$

**辛普森悖论示例：**

```
子样本A（女性）: X(教育年) 对 Y(工资) 正相关，β_A = +0.5
子样本B（男性）: X(教育年) 对 Y(工资) 正相关，β_B = +0.4

合并后：β_AB 可能是负数？！
→ 若男性平均教育高但工资低（不同基础），
  合并数据中"高教育"样本(男性)反而工资低，
  导致负相关——辛普森悖论！
```

> **Insight（金融中的组合效应）：** 在多国资产定价研究中，价值因子在各国单独回归均显著，但合并回归可能不显著——因为各国均值不同，合并后的"组内"效应被"组间"差异掩盖。**这是使用固定效应（控制国家均值）的动机之一。**

---

## 问题 23：用预测值 $\hat{Y}$ 代替 $Y$ 再做回归

### 问题

先回归 $Y$ 对 $X_1$ 得 $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1$，再回归 $\hat{Y}$ 对 $X_1$，结果如何？

### 推导

$\hat{Y}$ 已经是 $X_1$ 的线性函数，再对 $X_1$ 做回归：

$$\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1 + 0$$

$$R^2 = 1 \text{（完美拟合！）}, \quad \text{残差} = 0$$

**$t$ 统计量：** 趋于无穷大（$SE \to 0$，因残差为0），毫无意义。

> **Insight：** 这是回归中的"循环论证"问题，常出现在不规范的分析流程中（如用"平滑后的 Y"代替"原始 Y"）。更普遍的版本：用 $\hat{Y}$ 代替 $Y$ 分析任何东西，都会夸大显著性，因为消除了测量误差和随机噪声。

---

## 问题 24：$X$ 为二元变量（0/1）

### 问题

若 $X$ 是虚假变量（$X \in \{0, 1\}$），OLS 回归的含义是什么？系数如何解释？

### 推导

设 $n_1$ 个 $X=1$，$n_0$ 个 $X=0$，$n = n_0 + n_1$

$$\bar{x} = \frac{n_1}{n} = \hat{p}（X=1\text{的样本比例}）$$

$$S_{xx} = \sum(x_i - \bar{x})^2 = n_1(1-\hat{p})^2 + n_0(0-\hat{p})^2 = n\hat{p}(1-\hat{p})$$

$$S_{xy} = \sum(x_i - \bar{x})(y_i - \bar{y}) = n_1(\bar{y}_1 - \bar{y})(1-\hat{p}) + n_0(\bar{y}_0 - \bar{y})(0-\hat{p})$$

化简（其中 $\bar{y} = \hat{p}\bar{y}_1 + (1-\hat{p})\bar{y}_0$）：

$$S_{xy} = n\hat{p}(1-\hat{p})(\bar{y}_1 - \bar{y}_0)$$

$$\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} = \bar{y}_1 - \bar{y}_0$$

**结论：虚假变量的 OLS 斜率 = 两组均值之差！**

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x} = \hat{p}\bar{y}_1 + (1-\hat{p})\bar{y}_0 - (\bar{y}_1 - \bar{y}_0)\hat{p} = \bar{y}_0$$

**截距 = $X=0$ 组的均值！**

> **Insight：** OLS 与 $t$ 检验的统一：**虚假变量回归的 $t$ 统计量等于等方差假设下两样本 $t$ 检验的统计量**（完全等价）。这说明 $t$ 检验是线性回归的特例，整个假设检验体系在 OLS 框架内是统一的。

---

## 附录：常见变换效果的记忆口诀

```
"乘以常数看方向，R方t值不用变"
  → X或Y乘以常数：R²和t不变，系数按比例变

"平移截距会改变，斜率t值不影响"
  → X或Y加常数：截距变，斜率/t/R²不变

"复制数据系数稳，t值随之增根n"
  → 重复k次：系数R²不变，t值增√k（虚假显著）

"交换XY系数变，R方t值均相同"
  → X↔Y：系数完全不同，R²和t值不变
  → 补充：β₁·α₁ = R²（乘积恒等式）

"去掉截距系数偏，R方定义要注意"
  → 强制无截距：系数变，R²定义不同，不可比

"噪声复制共线性，SE增大系数缩"
  → 加噪声副本：多重共线性，SE增大，系数衰减
```

---

## 延伸阅读与深度问题

### 更高阶的变换问题

1. **Box-Cox 变换：** $Y^{(\lambda)} = (Y^\lambda - 1)/\lambda$（$\lambda \to 0$ 时趋向对数），如何用 Profile MLE 选择最优 $\lambda$？

2. **分数阶微分（Fractional Differentiation）：** Lopez de Prado (2018) 提出，使用分数阶差分 $(1-L)^d Y$（$0 < d < 1$），在保留最多记忆的同时使序列平稳。系数如何随 $d$ 变化？

3. **Demeaning vs Differencing：** 面板数据中，个体去均值（FE）vs 一阶差分（FD），在 $T=2$ 时等价；在 $T > 2$ 时效率不同，何时选哪个？

4. **回归稀释（Regression Dilution）：** $X$ 中有测量误差时，$Y$ 对 $X$ 的回归系数被衰减，但 $X$ 对 $Y$ 的（逆）回归系数被**放大**，两者的几何均值估计无衰减的真实系数：$\hat{\beta}^{\text{true}} \approx \sqrt{\hat{\beta}_{Y \text{ on } X} \cdot (1/\hat{\alpha}_{X \text{ on } Y})}$

5. **加权 OLS 变换：** WLS 等价于将 $Y, X$ 都乘以权重 $\sqrt{w_i}$，再做 OLS。这是"数据变换"的另一形式，但只有当权重反映真实方差结构时才提升效率。

---

*整理完毕。共 24 道核心变换问题，含完整推导、几何直觉和量化金融 Insights。*

*建议掌握顺序：问题 1-2（面试最高频）→ 问题 3-9（缩放与标准化）→ 问题 13、17（FWL 定理，理论核心）→ 问题 11、21（测量误差）→ 其余（情境题）*
