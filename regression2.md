# Regression Questions for Quantitative Finance Interviews

---

## 1. Levels vs. Cumulative Sum Regression

**Question:** You have two variables $X$ and $Y$, both i.i.d. Gaussian. Your model is $Y = \beta X + \varepsilon$. Compare:

- **Model A:** Regress $Y_t$ on $X_t$ (levels regression)
- **Model B:** Regress $\sum_{i=1}^{t} Y_i$ on $\sum_{i=1}^{t} X_i$ (cumsum regression)

What are the differences? Which is better?

**Answer:**

**Model A (levels):** Standard OLS on stationary i.i.d. data.
- Residuals are i.i.d., OLS is BLUE (Gauss-Markov).
- $\hat{\beta}$ is unbiased, consistent, and asymptotically normal.
- $R^2$ reflects the true signal-to-noise ratio.

**Model B (cumsums):** Both $S_t^X = \sum_{i=1}^t X_i$ and $S_t^Y = \sum_{i=1}^t Y_i$ are random walks (unit-root processes).
- Regressing one random walk on another **without a true cointegrating relationship** produces **spurious regression**.
- $R^2 \to 1$ and $t$-statistics diverge as $n \to \infty$, even if $X \perp Y$.
- The Durbin-Watson statistic $\to 0$, indicating severe serial correlation in residuals.
- Standard OLS inference ($t$-tests, $F$-tests) is **invalid**; critical values are non-standard.

**Which is better?** **Model A**, unless you have economic reason to believe $S^X$ and $S^Y$ are cointegrated (i.e., they share a common stochastic trend). If cointegration exists, use the Engle-Granger or Johansen procedure, not plain OLS on cumulative sums.

**Key takeaway:** Differencing restores stationarity. When in doubt, test for unit roots (ADF test) before choosing the regression form.

---

## 2. Multicollinearity: Detection and Remedies

**Question:** What is multicollinearity? How do you detect it and what do you do about it?

**Answer:**

Multicollinearity occurs when predictors are highly linearly correlated, making $X^TX$ nearly singular.

**Effects:**
- $\hat{\beta}$ is still unbiased but has inflated variance — wide confidence intervals, unstable coefficients.
- Individual $t$-statistics become insignificant even when the joint $F$-test is significant.

**Detection:**
- **Variance Inflation Factor (VIF):** $\text{VIF}_j = \frac{1}{1 - R_j^2}$ where $R_j^2$ is from regressing $X_j$ on all other predictors. $\text{VIF} > 10$ is a common concern threshold.
- **Condition number** of $X^TX$: $> 30$ suggests multicollinearity.
- Pairwise correlation matrix.

**Remedies:**
1. Drop one of the collinear variables (if redundant economically).
2. Ridge regression: adds $\lambda I$ to $X^TX$, shrinking coefficients.
3. PCA regression: project onto orthogonal principal components.
4. Collect more data to reduce variance.

---

## 3. Heteroskedasticity

**Question:** What is heteroskedasticity? How do you test for it and how do you handle it?

**Answer:**

Heteroskedasticity means $\text{Var}(\varepsilon_i | X_i) = \sigma_i^2$ varies across observations (violating constant-variance assumption).

**Effects:** OLS $\hat{\beta}$ remains unbiased but is no longer BLUE; standard errors are wrong, invalidating inference.

**Tests:**
- **Breusch-Pagan test:** Regress squared residuals $\hat{\varepsilon}^2$ on $X$; test joint significance.
- **White test:** More general version including cross-terms.
- **Goldfeld-Quandt test:** Split sample and compare residual variances.
- Visual: plot $\hat{\varepsilon}^2$ vs. $\hat{Y}$ — look for fanning pattern.

**Remedies:**
1. **White (HC) standard errors:** Robust SEs that are valid under heteroskedasticity without changing $\hat{\beta}$.
2. **WLS (Weighted Least Squares):** Weight observations by $1/\sigma_i^2$ if variance structure is known.
3. **Transform the dependent variable:** e.g., $\log Y$ often stabilizes variance.

---

## 4. Serial Correlation (Autocorrelation)

**Question:** How do you detect and handle serial correlation in regression residuals?

**Answer:**

Serial correlation means $\text{Cov}(\varepsilon_t, \varepsilon_{t-k}) \neq 0$. Common in time series financial data.

**Effects:** OLS $\hat{\beta}$ is unbiased but inefficient; standard errors are biased (usually underestimated), inflating $t$-statistics.

**Detection:**
- **Durbin-Watson statistic:** $DW = \frac{\sum_{t=2}^n (\hat{\varepsilon}_t - \hat{\varepsilon}_{t-1})^2}{\sum \hat{\varepsilon}_t^2}$. $DW \approx 2$ means no autocorrelation; $DW < 2$ suggests positive autocorrelation.
- **Ljung-Box Q test:** Tests joint significance of autocorrelations up to lag $k$.
- ACF/PACF plots of residuals.

**Remedies:**
1. **Newey-West standard errors:** HAC (heteroskedasticity and autocorrelation consistent) SEs.
2. **GLS/FGLS:** Model the error structure explicitly (e.g., AR(1) errors).
3. Include lagged dependent variable or relevant omitted variables.

---

## 5. Omitted Variable Bias

**Question:** What happens if you omit a relevant variable from a regression?

**Answer:**

If the true model is $Y = \beta_1 X_1 + \beta_2 X_2 + \varepsilon$ but you regress $Y$ on $X_1$ only:

$$\text{Bias}(\hat{\beta}_1) = \beta_2 \cdot \frac{\text{Cov}(X_1, X_2)}{\text{Var}(X_1)}$$

The bias is zero only if $\beta_2 = 0$ (irrelevant variable) or $\text{Cov}(X_1, X_2) = 0$ (orthogonal regressors).

In finance: omitting a risk factor (e.g., market beta) when estimating alpha leads to biased alpha estimates.

---

## 6. Interpreting $R^2$ and Adjusted $R^2$

**Question:** What is $R^2$, and when is it misleading?

**Answer:**

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$

Measures fraction of variance in $Y$ explained by the model.

**Adjusted $R^2$:**
$$\bar{R}^2 = 1 - \frac{SS_{\text{res}} / (n-k-1)}{SS_{\text{tot}} / (n-1)}$$

Penalizes for adding irrelevant predictors. Use for model comparison with different numbers of regressors.

**When $R^2$ is misleading:**
- Spurious regressions (trending/non-stationary data): $R^2$ is high but meaningless.
- Adding variables always increases $R^2$ even if they're noise.
- Low $R^2$ doesn't mean the model is useless if the goal is inference on $\hat{\beta}$, not prediction.
- In factor models, a low $R^2$ on individual assets is expected and acceptable.

---

## 7. Ridge vs. Lasso vs. OLS

**Question:** Compare OLS, Ridge, and Lasso regression.

**Answer:**

| Property | OLS | Ridge ($L_2$) | Lasso ($L_1$) |
|---|---|---|---|
| Penalty | None | $\lambda \sum \beta_j^2$ | $\lambda \sum |\beta_j|$ |
| Bias | 0 | Positive | Positive |
| Variance | High (collinear) | Reduced | Reduced |
| Feature selection | No | No (shrinks all) | Yes (zeros out) |
| Closed form | Yes | Yes | No |
| Handles multicollinearity | Poorly | Well | Variable |

**In finance:** Lasso is useful for sparse factor models (many candidate factors, few true ones). Ridge is preferred when all factors contribute but you want to stabilize estimates.

---

## 8. Endogeneity and Instrumental Variables

**Question:** What is endogeneity? How do you address it?

**Answer:**

Endogeneity: $\text{Cov}(X, \varepsilon) \neq 0$, meaning the regressor is correlated with the error term.

**Causes:**
- Omitted variable bias
- Simultaneity (reverse causality): e.g., price and demand affect each other
- Measurement error in $X$

**Effects:** OLS $\hat{\beta}$ is biased and inconsistent.

**Remedy — Instrumental Variables (IV):**
Find an instrument $Z$ such that:
1. **Relevance:** $\text{Cov}(Z, X) \neq 0$
2. **Exclusion restriction:** $\text{Cov}(Z, \varepsilon) = 0$ (Z affects Y only through X)

Two-Stage Least Squares (2SLS):
1. Regress $X$ on $Z$, get $\hat{X}$
2. Regress $Y$ on $\hat{X}$

**Finance application:** Using lagged values as instruments, or using cost shifters as instruments for price in demand estimation.

---

## 9. The Frisch-Waugh-Lovell (FWL) Theorem

**Question:** State and explain the FWL theorem. Why is it useful in finance?

**Answer:**

Given $Y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$, the OLS estimate $\hat{\beta}_1$ equals the OLS estimate from regressing $M_2 Y$ on $M_2 X_1$, where $M_2 = I - X_2(X_2^T X_2)^{-1} X_2^T$ is the residual-maker (annihilator) of $X_2$.

**Interpretation:** $\hat{\beta}_1$ measures the effect of $X_1$ after partialling out the influence of $X_2$ from both $X_1$ and $Y$.

**Finance use cases:**
- **Alpha estimation:** Partial out benchmark returns to isolate manager skill.
- **Panel regressions:** Demeaning to remove fixed effects is equivalent to including dummy variables (FWL).
- **Factor attribution:** Decompose returns controlling for known risk factors.

---

## 10. Robust Standard Errors in Practice

**Question:** When should you use robust standard errors in a financial regression?

**Answer:**

Use robust (HC or HAC) standard errors when:
- **HC (heteroskedasticity-consistent, White):** Cross-sectional regressions where variance may vary with firm size, leverage, etc.
- **HAC (Newey-West):** Time-series regressions where residuals are autocorrelated (e.g., monthly return data).
- **Clustered SE:** Panel data where observations within a group (firm, industry) are correlated but independent across groups.

**Caveat:** Robust SEs fix inference but don't improve efficiency. If the heteroskedasticity structure is known, WLS is more efficient.

In factor model regressions (e.g., Fama-MacBeth), the cross-sectional standard deviation of time-series slopes provides an alternative to robust SEs.

---

## 11. Fama-MacBeth Regression

**Question:** Describe the Fama-MacBeth two-pass procedure and its purpose.

**Answer:**

**Goal:** Estimate risk premia for factor exposures (betas) while accounting for cross-sectional correlation.

**Two passes:**
1. **Time-series pass:** For each asset $i$, regress $R_{it}$ on factors $F_t$ over time to estimate $\hat{\beta}_i$ (factor loadings).
2. **Cross-sectional pass:** For each time period $t$, regress $R_{it}$ on $\hat{\beta}_i$ across assets to estimate the factor risk premium $\hat{\lambda}_t$.

**Inference:** Average $\hat{\lambda}_t$ over time; use the time-series standard deviation to compute $t$-statistics. This accounts for cross-sectional correlation of returns (which OLS would ignore).

**Limitations:**
- Errors-in-variables (EIV) problem: $\hat{\beta}_i$ is estimated with error, biasing $\hat{\lambda}$.
- Assumes stable betas over time.

---
