# Statistics Questions Log

---

## Q1: IC Range of OLS Fitted Values

**Setup:**

- IC(X1, y) = Corr(X1, y) = 0.1
- IC(X2, y) = Corr(X2, y) = 0.2
- OLS regression: ŷ = β₀ + β₁X₁ + β₂X₂

**Question:** What is the range of IC(ŷ, y)?

---

### Key Insight

In OLS, the fitted value ŷ is the projection of y onto the column space of [1, X1, X2]. Therefore:

```
IC(ŷ, y) = Corr(ŷ, y) = R   (the multiple correlation coefficient)
```

### Formula for R² with Two Predictors

```
R² = (r₁² + r₂² − 2·r₁·r₂·r₁₂) / (1 − r₁₂²)
```

where r₁ = 0.1, r₂ = 0.2, and r₁₂ = Corr(X1, X2) is unknown.

### Lower Bound

Differentiate R² with respect to r₁₂ and set to zero:

```
d/dr₁₂ [(r₁² + r₂² − 2r₁r₂r₁₂)/(1 − r₁₂²)] = 0
```

Solving: r₁₂* = r₁/r₂ = 0.1/0.2 = **0.5**

At r₁₂ = 0.5:
```
R² = (0.01 + 0.04 − 2·0.1·0.2·0.5) / (1 − 0.25)
   = (0.05 − 0.02) / 0.75
   = 0.03 / 0.75
   = 0.04
```
So R_min = **0.2**.

**Intuition:** When X1 and X2 are correlated at exactly r₁/r₂ = 0.5, X1 contributes no incremental predictive power beyond X2. The combined model reduces to the best single predictor.

This also follows from the general OLS property: adding more predictors never decreases R², so

```
R ≥ max(|r₁|, |r₂|) = 0.2
```

### Upper Bound

The 3×3 correlation matrix Σ of (X1, X2, y) must be positive semi-definite. Its determinant must satisfy det(Σ) ≥ 0:

```
det(Σ) = 0.95 − r₁₂² + 0.04·r₁₂ ≥ 0
```

At the boundary (det = 0), the variables become linearly dependent, and the OLS fit becomes exact: **R = 1**.

The two boundary values of r₁₂ are:
```
r₁₂ = (0.04 ± √(0.0016 + 3.8)) / 2 ≈ 0.9949  or  −0.9549
```

Both give R² = 1. All intermediate values of r₁₂ produce R ∈ (0.2, 1).

### Answer

```
IC(ŷ, y) ∈ [0.2, 1]
```

| r₁₂ (Corr X1,X2) | R² | IC(ŷ, y) |
|---:|---:|---:|
| −0.9549 (boundary) | 1.00 | 1.000 |
| −0.50 | 0.093 | 0.305 |
| 0.00 | 0.050 | 0.224 |
| **0.50 (minimum)** | **0.040** | **0.200** |
| 0.90 | 0.074 | 0.271 |
| 0.9949 (boundary) | 1.00 | 1.000 |

- **Lower bound 0.2** is achieved when r₁₂ = 0.5 (X1 is fully redundant given X2).
- **Upper bound 1** is approached as the correlation matrix becomes singular (X1, X2, y are linearly dependent).
- In practice (non-degenerate data): **IC(ŷ, y) ∈ [0.2, 1)**, with the minimum equal to the stronger single-factor IC.

---
