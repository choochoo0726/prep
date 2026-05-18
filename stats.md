# Statistics Questions Log / 统计学题目记录

---

## Q1: IC Range of OLS Fitted Values / OLS 预测值的 IC 范围

**Setup / 已知条件：**

- IC(X1, y) = Corr(X1, y) = 0.1
- IC(X2, y) = Corr(X2, y) = 0.2
- OLS regression / OLS 回归: ŷ = β₀ + β₁X₁ + β₂X₂

**Question / 问题：** What is the range of IC(ŷ, y)? / y^ 与 y 之间的 IC 处于什么范围？

---

### Key Insight / 核心思路

In OLS, the fitted value ŷ is the projection of y onto the column space of [1, X1, X2]. Therefore:

在 OLS 回归中，预测值 ŷ 是 y 在 [1, X1, X2] 列空间上的投影，因此：

```
IC(ŷ, y) = Corr(ŷ, y) = R   (the multiple correlation coefficient / 多元相关系数)
```

---

### Formula for R² with Two Predictors / 双预测变量的 R² 公式

```
R² = (r₁² + r₂² − 2·r₁·r₂·r₁₂) / (1 − r₁₂²)
```

where r₁ = 0.1, r₂ = 0.2, and r₁₂ = Corr(X1, X2) is unknown.

其中 r₁ = 0.1，r₂ = 0.2，r₁₂ = Corr(X1, X2) 未知。

---

### Lower Bound / 下界推导

Differentiate R² with respect to r₁₂ and set to zero:

对 R² 关于 r₁₂ 求导并令其为零：

```
d/dr₁₂ [(r₁² + r₂² − 2r₁r₂r₁₂)/(1 − r₁₂²)] = 0
```

Solving / 解得: r₁₂* = r₁/r₂ = 0.1/0.2 = **0.5**

At r₁₂ = 0.5 / 当 r₁₂ = 0.5 时：
```
R² = (0.01 + 0.04 − 2·0.1·0.2·0.5) / (1 − 0.25)
   = (0.05 − 0.02) / 0.75
   = 0.03 / 0.75
   = 0.04
```
So R_min = **0.2** / 故 R 的最小值为 **0.2**。

**Intuition / 直觉理解：** When X1 and X2 are correlated at exactly r₁/r₂ = 0.5, X1 contributes no incremental predictive power beyond X2. The combined model reduces to the best single predictor.

当 X1 与 X2 的相关系数恰好等于 r₁/r₂ = 0.5 时，X1 在 X2 已知的情况下不再提供任何额外的预测信息，联合模型退化为仅用最优单因子 X2 的结果。

This also follows from the general OLS property: adding more predictors never decreases R², so:

这也可以从 OLS 的一般性质推导：增加预测变量不会降低 R²，因此：

```
R ≥ max(|r₁|, |r₂|) = 0.2
```

---

### Upper Bound / 上界推导

The 3×3 correlation matrix Σ of (X1, X2, y) must be positive semi-definite. Its determinant must satisfy det(Σ) ≥ 0:

(X1, X2, y) 的 3×3 相关矩阵 Σ 必须是半正定的，即行列式满足 det(Σ) ≥ 0：

```
det(Σ) = 0.95 − r₁₂² + 0.04·r₁₂ ≥ 0
```

At the boundary (det = 0), the variables become linearly dependent, and the OLS fit becomes exact: **R = 1**.

在边界处（det = 0），三个变量线性相关，OLS 拟合完美，此时 **R = 1**。

The two boundary values of r₁₂ are / r₁₂ 的两个边界值为：
```
r₁₂ = (0.04 ± √(0.0016 + 3.8)) / 2 ≈ 0.9949  or / 或  −0.9549
```

Both give R² = 1. All intermediate values of r₁₂ produce R ∈ (0.2, 1).

两个边界值均对应 R² = 1，其余中间值对应 R ∈ (0.2, 1)。

---

### Answer / 结论

```
IC(ŷ, y) ∈ [0.2, 1]
```

| r₁₂ (Corr X1,X2) | R² | IC(ŷ, y) |
|---:|---:|---:|
| −0.9549 (boundary / 边界) | 1.00 | 1.000 |
| −0.50 | 0.093 | 0.305 |
| 0.00 | 0.050 | 0.224 |
| **0.50 (minimum / 最小值点)** | **0.040** | **0.200** |
| 0.90 | 0.074 | 0.271 |
| 0.9949 (boundary / 边界) | 1.00 | 1.000 |

- **Lower bound 0.2 / 下界 0.2**：achieved when r₁₂ = 0.5 (X1 is fully redundant given X2) / 当 r₁₂ = 0.5 时取到，此时 X1 对 X2 完全冗余。
- **Upper bound 1 / 上界 1**：approached as the correlation matrix becomes singular (X1, X2, y are linearly dependent) / 当相关矩阵趋于奇异（X1、X2、y 线性相关）时趋近。
- **In practice / 实践中** (non-degenerate data / 非退化数据): IC(ŷ, y) ∈ [0.2, 1), with the minimum equal to the stronger single-factor IC / 下界等于单因子中较强的 IC 值。

---
