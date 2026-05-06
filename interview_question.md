# Interview Question:

## Question 1.  Predicting Rest-of-Day (ROD) Volume
Consider we have minutely volume data for all the stocks in the US, HK and Europe market in the past 5 years. The goal is to predict the Rest-of-Day (ROD) volume. How would you model the volume for a stock at any given time in the day?

## Response

ROD volume prediction is a core problem in algorithmic execution — it underpins VWAP scheduling, POV targeting, and optimal liquidation. The key insight is that ROD volume is never predicted from scratch: part of the day has already been observed, so the problem is a **real-time Bayesian update** of a prior forecast.

### 1. Problem Formulation

Let $\tau \in \{1, \ldots, T\}$ index minutes within a trading day ($T$ = total minutes in the session). At the current minute $\tau$, cumulative observed volume $V_{1:\tau} = \sum_{t=1}^{\tau} v_t$ is known. The target is:

$$\hat{V}_{\text{ROD}}(\tau) = E\!\left[\sum_{t=\tau+1}^{T} v_t \;\middle|\; \mathcal{F}_\tau\right]$$

where $\mathcal{F}_\tau$ is all information available at time $\tau$ (observed volumes, prices, spreads, news flags, etc.).

Since $V_{\text{total}} = V_{1:\tau} + V_{\text{ROD}}$ and $V_{1:\tau}$ is already observed:

$$\hat{V}_{\text{ROD}}(\tau) = \hat{V}_{\text{total}}(\tau) - V_{1:\tau}$$

So the problem reduces to **nowcasting total daily volume** $\hat{V}_{\text{total}}$ given partial intraday evidence.

---

### 2. The Multiplicative Decomposition

A principled starting point is the multiplicative model:

$$V_{\text{total}}(i, d) = \underbrace{\text{ADV}_i}_{\text{level}} \times \underbrace{\eta_{i,d}}_{\text{daily surprise}}$$

where $\text{ADV}_i$ is the rolling $N$-day average daily volume (prior), and $\eta_{i,d}$ is the unknown daily multiplier with $E[\eta] = 1$.

At the intraday level, volume in minute $t$ is:

$$v_{i,t} = V_{\text{total}}(i,d) \times f_i(t) \times \epsilon_{i,t}$$

where $f_i(t)$ is the **intraday volume fraction** (the U-shaped profile), satisfying $\sum_{t=1}^{T} f_i(t) = 1$, and $\epsilon_{i,t}$ is idiosyncratic minute-level noise with $E[\epsilon] = 1$.

Cumulative volume through $\tau$ is therefore:

$$V_{1:\tau} = V_{\text{total}} \cdot F_i(\tau) \cdot \bar{\epsilon}_{1:\tau}$$

where $F_i(\tau) = \sum_{t=1}^{\tau} f_i(t)$ is the **cumulative fraction** traded by time $\tau$, and $\bar{\epsilon}$ absorbs the noise aggregate. This gives an immediate naive estimate:

$$\hat{V}_{\text{total}}^{\text{naive}} = \frac{V_{1:\tau}}{F_i(\tau)}, \qquad \hat{V}_{\text{ROD}}^{\text{naive}} = V_{1:\tau} \cdot \frac{1 - F_i(\tau)}{F_i(\tau)}$$

This is a useful baseline — it simply scales up observed volume using the historical profile — but it ignores all other available signals.

---

### 3. Estimating the Intraday Profile $f_i(\tau)$

The profile $f_i(\tau)$ is not a single universal curve. It must be conditioned on context:

**a) Static baseline:** compute the historical mean fraction per minute bin, separately by stock, session, and day-type:
$$\hat{f}_i(\tau) = E\!\left[\frac{v_{i,\tau}}{V_{\text{total},i,d}}\right]$$

A U-shape emerges universally: high at open (auction / information release), low at midday, high at close (index rebalancing, hedging).

**b) Dynamic profile conditioning:** the shape shifts based on:
- **Day of week** (Friday close is heavier; Monday open has gap news)
- **Volatility regime** (high-vol days front-load volume — more trading at open)
- **Event flag** (earnings days have a spike at the announcement minute; index rebalancing days are back-loaded)
- **Time remaining to a known catalyst** (e.g., Fed decision at 2pm ET compresses pre-event volume and explodes post-event)

In practice, maintain a library of profile templates and select/blend based on the current day's context features.

---

### 4. Real-Time Nowcasting: Bayesian Update

The naive estimate ignores cross-sectional and intraday signals. A richer approach treats $\log \eta_{i,d}$ as a latent variable and updates it sequentially.

**Prior:** before the open, form a prior using pre-market signals:
$$\log \hat{\eta}_{i,d}^{(0)} = \alpha_0 + \alpha_1 \cdot \log(\text{pre-market volume}) + \alpha_2 \cdot \text{overnight return} + \alpha_3 \cdot \hat{\sigma}_d + \cdots$$

**Likelihood:** each observed minute $t \leq \tau$ contributes:
$$\log v_{i,t} - \log(\text{ADV}_i) - \log f_i(t) = \log \eta_{i,d} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_\epsilon^2)$$

**Posterior (Kalman/Gaussian update):** assuming $\log \eta_{i,d} \sim \mathcal{N}(\mu_0, \sigma_0^2)$, the posterior after $\tau$ observations is:

$$\hat{\eta}_{i,d}(\tau) = \frac{\sigma_\epsilon^2 \mu_0 + \sigma_0^2 \sum_{t=1}^{\tau} z_t}{\sigma_\epsilon^2 + \tau \sigma_0^2}, \qquad z_t = \log v_{i,t} - \log(\text{ADV}_i) - \log f_i(t)$$

The posterior variance $\sigma^2(\tau) = \frac{\sigma_0^2 \sigma_\epsilon^2}{\sigma_\epsilon^2 + \tau \sigma_0^2}$ decreases monotonically — confidence increases as more of the day is observed. The ROD prediction then becomes:

$$\hat{V}_{\text{ROD}}(\tau) = \text{ADV}_i \cdot e^{\hat{\eta}_{i,d}(\tau) + \frac{1}{2}\sigma^2(\tau)} \cdot (1 - F_i(\tau))$$

(the $\frac{1}{2}\sigma^2$ term is the log-normal bias correction).

---

### 5. Cross-Sectional Signals

Volume is highly correlated cross-sectionally — a market-wide news event lifts all stocks. Exploit this by decomposing $\eta$ into market, sector, and idiosyncratic components:

$$\log \eta_{i,d} = \beta_i^M \cdot \log \eta_d^{\text{market}} + \beta_i^S \cdot \log \eta_d^{\text{sector}} + \log \eta_d^{\text{idio}}$$

- $\log \eta_d^{\text{market}}$ is estimated from the full cross-section (e.g., ETF volume or average universe volume ratio)
- $\log \eta_d^{\text{sector}}$ is estimated from the stock's peer group
- Individual stock's $\eta$ borrows strength from these common factors, especially useful early in the day when few idiosyncratic observations exist

---

### 6. Feature Engineering Summary

| Feature | Signal |
| :--- | :--- |
| Volume surprise ratio $V_{1:\tau} / (\text{ADV} \cdot F(\tau))$ | Primary real-time signal; persistence of volume shock |
| Market/sector volume ratio | Cross-sectional commonality |
| Realized intraday volatility $\hat{\sigma}_{1:\tau}$ | High vol → elevated remaining volume |
| Bid-ask spread level and change | Wider spreads suppress participation |
| Pre-market volume (US/Europe) | Strong predictor of day's total volume |
| Auction volume at open | Reveals institutional order imbalance |
| Event flags: earnings, index rebal, macro | Shift profile shape and level |
| Day of week / month-end | Systematic seasonality in ADV multiplier |
| Time remaining $(1 - \tau/T)$ | Determines how much weight prior vs. observed gets |

---

### 7. Market-Specific Nuances

**US:**
- Pre-market trading (4–9:30am ET) provides a leading signal — model it separately.
- The **opening auction** volume is a separate observation with high signal content.
- Closing auction (MOC/LOC orders) creates a discrete spike at 4pm — model it as a point mass, not a smooth bin.
- Options expiry Fridays and index rebalancing dates (quarterly) are structural outlier days.

**HK:**
- Explicitly zero-out the **lunch break** (12:00–1:00pm HKT) in $f(\tau)$. There are effectively two sub-sessions; treat them as separate profiles.
- The afternoon re-open has a mini-auction spike similar to the morning open.
- Southbound/Northbound Stock Connect flows (China-HK link) create correlated volume shocks.

**Europe:**
- The **US market open** (2:30–3:30pm CET) is a structural feature: European stocks with US listings or US revenue exposure spike when US opens. Add a "US open" dummy and interaction with beta-to-US.
- European close (5:30pm CET) has a closing auction in most markets — model separately.
- Fragmented liquidity across venues (LSE, Xetra, Euronext) — aggregate across venues before modeling.

**Cross-listing / dual-listed stocks:**
- Volume in the primary market leads volume in the secondary. A spike in HSBC London predicts elevated HSBC HK volume in the next session overlap window.

---

### 8. Model Architecture Options

| Architecture | Best For | Trade-off |
| :--- | :--- | :--- |
| Naive profile scaling | Baseline, interpretable | Ignores real-time signals |
| Kalman filter on log-volume | Online updates, principled uncertainty | Assumes log-normality; linear |
| Ridge/LASSO regression | Many features, inference on coefficients | Static, no sequential structure |
| Gradient boosting (GBDT) | Non-linear interactions (e.g., event × volatility) | Needs offline retraining; no uncertainty |
| LSTM / Transformer | Capture complex intraday temporal patterns | Data-hungry; less interpretable |

In practice: use the **Kalman-based nowcaster** as the core engine (online, principled) and feed its output as a feature into a **GBDT** that handles non-linear interactions with event flags and cross-sectional signals.

---

### 9. Evaluation

- **Primary metric:** Mean Absolute Percentage Error (MAPE) of $\hat{V}_{\text{ROD}}$ vs. realized $V_{\text{ROD}}$, evaluated at multiple intraday horizons (e.g., after 10%, 25%, 50%, 75% of the session).
- **Bias check:** the model should be unbiased at each time $\tau$ — check $E[\hat{V}_{\text{ROD}} - V_{\text{ROD}}] \approx 0$ across days.
- **Event-day performance:** separately evaluate on earnings days, rebalancing days, and macro event days — these are the hardest cases and the most consequential for execution.
- **Execution P&L impact:** the ultimate test is whether using the model's ROD forecast reduces VWAP slippage vs. the naive profile baseline.


## Question 2: Investment Opportunity Evaluation

Given the following investment opportunities and probability matrices:

**Opportunities:**
| Opportunity | Bull | Base | Bear | Horizon (Yrs) |
| :--- | :--- | :--- | :--- | :--- |
| A | 12% | 6% | -8% | 2 |
| B | 30% | 15% | -20% | 1 |
| C | 6% | 4% | 0% | 3 |

**Probability Matrices:**
| Matrix | Bull | Base | Bear |
| :--- | :--- | :--- | :--- |
| A | 20% | 60% | 20% |
| B | 10% | 50% | 40% |
| C | 30% | 40% | 30% |

Please evaluate these opportunities.

## Response

### 1. Expected Return Calculation

For each opportunity, the expected return is the probability-weighted average of the scenario returns:

$$E[R] = p_{Bull} \cdot R_{Bull} + p_{Base} \cdot R_{Base} + p_{Bear} \cdot R_{Bear}$$

| Opportunity | Calculation | Expected Return (Total) | Horizon | Annualized |
| :--- | :--- | :--- | :--- | :--- |
| A | 0.20×12% + 0.60×6% + 0.20×(−8%) | **4.40%** | 2 yr | ~2.18% |
| B | 0.10×30% + 0.50×15% + 0.40×(−20%) | **2.50%** | 1 yr | 2.50% |
| C | 0.30×6% + 0.40×4% + 0.30×0% | **3.40%** | 3 yr | ~1.12% |

> Annualized return approximated as $(1 + E[R])^{1/\text{Horizon}} - 1$.

### 2. Risk Analysis

Standard deviation quantifies the dispersion of outcomes around the expected return:

$$\sigma_{\text{total}} = \sqrt{\sum_s p_s \cdot (R_s - E[R])^2}$$

To compare across different horizons, annualize volatility using the standard i.i.d. scaling:

$$\sigma_{\text{annual}} = \frac{\sigma_{\text{total}}}{\sqrt{T}}$$

| Opportunity | $\sigma_{\text{total}}$ | Horizon | $\sigma_{\text{annual}}$ | P(Loss) | Worst Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A | 6.62% | 2 yr | **4.68%** | 20% | −8% |
| B | 18.88% | 1 yr | **18.88%** | 40% | −20% |
| C | 2.37% | 3 yr | **1.37%** | 0% | 0% |

**Key observations:**
- **B** has by far the highest risk: 40% probability of a −20% loss, giving the largest $\sigma$.
- **C** has zero probability of loss — all downside scenarios are at worst flat.
- **A** is the middle ground.

### 3. Risk-Adjusted Comparison

Using annualized return-to-risk ratio (analogous to an information ratio, assuming zero risk-free rate for simplicity):

| Opportunity | $E[R]_{\text{annual}}$ | $\sigma_{\text{annual}}$ | Return / Risk |
| :--- | :--- | :--- | :--- |
| A | 2.18% | 4.68% | **0.47** |
| B | 2.50% | 18.88% | **0.13** |
| C | 1.12% | 1.37% | **0.82** |

**C dominates on a risk-adjusted basis** despite having the lowest annualized return.

### 4. Recommendation

The ranking depends on investor mandate, but the core insight is:

| Profile | Preferred | Rationale |
| :--- | :--- | :--- |
| Capital-preserving / low-risk | **C** | Zero loss probability; best risk-adjusted ratio |
| Balanced | **A** | Reasonable return with moderate risk; 2-yr horizon allows some recovery |
| High-risk / speculative | **B** | Highest upside (30%), but the 40% bear probability at −20% makes this unattractive unless the investor has a strong bull view |

**Avoid B** unless you have a specific conviction that bull/base probabilities are understated — its expected return (2.5%) barely compensates for a 40% chance of a −20% drawdown.

**Additional considerations in practice:**
- **Correlation to existing portfolio:** If the portfolio is already long-equity-beta, C's low-volatility profile provides diversification.
- **Liquidity:** A 3-year horizon for C requires capital to be locked up longer.
- **Asymmetry / skew:** B has high positive skew (bull upside of 30%), which some strategies explicitly seek (e.g., options-like payoffs).

---

## Question 3: Investment Decision Criteria

When you have an investment opportunity, how do you evaluate it and decide whether to take it? What are the criteria?

## Response

Evaluating an investment opportunity is a four-stage process: assess it standalone, assess it in portfolio context, size the position, then apply practical filters.

### 1. Standalone Attractiveness

The first gate is whether the opportunity has positive risk-adjusted expected value on its own.

**a) Expected return vs. hurdle rate**

$$E[R] = \sum_s p_s \cdot R_s > r_f + \text{risk premium}$$

An opportunity must clear a minimum hurdle (risk-free rate + compensation for risk taken). If $E[R]$ is below the hurdle, it doesn't justify the capital.

**b) Risk-adjusted return (Sharpe / Information Ratio)**

$$\text{IR} = \frac{E[R] - r_f}{\sigma}$$

A higher IR means more return per unit of risk. As a rough benchmark, an IR > 0.5 annualized is attractive; > 1.0 is exceptional.

**c) Downside risk**
- **Probability of loss:** How likely is a negative outcome?
- **Expected shortfall (CVaR):** What is the average loss in the worst $\alpha$% of scenarios? More informative than just σ when return distributions are skewed or fat-tailed.
- **Max drawdown:** What is the worst peak-to-trough loss under stress scenarios?

**d) Conviction in the inputs**

Probability estimates are rarely objective. Ask: *what is the source of edge?* If the bull/base/bear probabilities come from your own research that the market has not priced in, that is alpha. If they mirror consensus, the opportunity may already be fully valued.

### 2. Portfolio Fit (Marginal Contribution)

A good standalone opportunity can still be a bad addition to the portfolio if it increases concentration risk. Evaluate the *marginal* impact:

**a) Correlation to existing holdings**

$$\rho_{i,P} = \frac{\text{Cov}(R_i, R_P)}{\sigma_i \cdot \sigma_P}$$

A low or negative $\rho_{i,P}$ means the new position diversifies the portfolio. A high $\rho$ means it mainly adds correlated risk.

**b) Marginal Sharpe contribution**

Adding asset $i$ improves portfolio Sharpe if:

$$\frac{E[R_i] - r_f}{\sigma_i} > \text{SR}_P \cdot \rho_{i,P}$$

In words: the new asset's Sharpe must exceed the existing portfolio Sharpe scaled by its correlation to the portfolio. An asset with a lower standalone Sharpe can still improve the portfolio if it is sufficiently uncorrelated.

**c) Factor exposure**

Does this opportunity add or hedge existing factor tilts (market beta, sector, duration, currency)? Unintended factor concentration is a common risk.

### 3. Position Sizing: Kelly Criterion

Once an opportunity passes the quality filters, sizing is determined optimally by the **Kelly Criterion**, which maximizes the long-run growth rate of capital:

$$f^* = \frac{E[R] - r_f}{\sigma^2}$$

For a discrete scenario set (as in Question 2):

$$f^* = \frac{bp - q}{b}$$

where $b$ is the net odds (upside/downside ratio), $p$ is the win probability, and $q = 1 - p$.

**In practice, use fractional Kelly** (e.g., half-Kelly) to reduce variance, since probability estimates are never perfectly calibrated. Full Kelly is theoretically optimal but highly sensitive to estimation error and produces large drawdowns.

### 4. Practical Filters

Even after passing steps 1–3, several real-world constraints can kill an otherwise attractive opportunity:

| Filter | Question to Ask |
| :--- | :--- |
| **Liquidity** | Can you enter and exit within the investment horizon without moving the market? |
| **Capacity** | Does the strategy remain profitable at the capital size you intend to deploy? |
| **Transaction costs** | After bid-ask spread, commissions, and market impact, is $E[R]$ still positive? |
| **Drawdown limits** | Does the worst-case scenario breach your portfolio's drawdown mandate? |
| **Horizon match** | Does the investment horizon align with your liquidity needs or fund lock-up? |
| **Regulatory / mandate** | Is the instrument within the fund's permitted universe? |

### 5. Summary Decision Framework

```
E[R] > hurdle rate?          → No  → Reject
IR acceptable?               → No  → Reject
Downside/CVaR within limits? → No  → Reject
Marginal Sharpe positive?    → No  → Reject (or hedge the correlated risk first)
Liquidity / capacity OK?     → No  → Reject or reduce size
→ Yes to all: size via (fractional) Kelly, enter position
```

The most common mistake is evaluating an opportunity in isolation. Portfolio fit and position sizing are just as important as the standalone return — a high-IR opportunity that is 0.95 correlated with your existing book adds almost no value.

---

## Question 4: Multicollinearity

What is multicollinearity? How do you detect it? How do you handle it in the modeling process?

## Response

### 1. What Is Multicollinearity?

Multicollinearity occurs when two or more predictor variables in a regression model are highly linearly correlated with each other. It comes in two forms:

- **Perfect multicollinearity:** one predictor is an exact linear combination of others (e.g., the dummy variable trap — including all category dummies plus an intercept). OLS cannot be estimated at all; $X'X$ is singular.
- **Near (imperfect) multicollinearity:** predictors are highly but not perfectly correlated. OLS estimates still exist but become unreliable.

**Effect on OLS estimates:**

OLS coefficients remain **unbiased** under multicollinearity — the Gauss-Markov theorem still holds. The damage is to **variance**:

$$\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{SST_j(1 - R^2_j)}$$

where $R^2_j$ is the R² from regressing $X_j$ on all other predictors. As $R^2_j \to 1$ (perfect collinearity), $\text{Var}(\hat{\beta}_j) \to \infty$.

Practical consequences:
- Standard errors inflate → t-statistics shrink → predictors appear insignificant even when they are not
- Coefficient estimates become unstable: small changes in the data cause large swings in $\hat{\beta}$
- The model's overall F-statistic can be highly significant while all individual t-stats are not — a classic warning sign
- Coefficient signs can flip and become economically nonsensical

> **Key distinction:** if the goal is **prediction**, multicollinearity is less damaging — the model can still fit well. If the goal is **inference** (understanding individual coefficient effects), multicollinearity is a serious problem.

### 2. Detection

#### a) Correlation Matrix
Inspect pairwise correlations between all predictors. A rule of thumb: $|\rho| > 0.8$ is concerning, but this only catches pairwise relationships — it misses cases where three or more variables jointly collinear.

#### b) Variance Inflation Factor (VIF)
The most widely used diagnostic. For each predictor $j$, run the **auxiliary regression**: treat $x_j$ as the dependent variable and regress it on all remaining predictors $X_{-j}$ (the original response $y$ plays no role):

$$x_j = \gamma_0 + \gamma_1 x_1 + \cdots + \gamma_{j-1}x_{j-1} + \gamma_{j+1}x_{j+1} + \cdots + \gamma_p x_p + \tilde{x}_j$$

or in matrix form: $x_j = X_{-j}\hat{\gamma} + \tilde{x}_j$

This is repeated $p$ times in total, once per predictor. The residual $\tilde{x}_j$ captures the part of $x_j$ that is **not** linearly explained by the other predictors — its unique variation. Let $R^2_j$ be the $R^2$ of this auxiliary regression:
- $R^2_j \approx 0$: $x_j$ is nearly orthogonal to the others — no multicollinearity
- $R^2_j \approx 1$: $x_j$ is almost a linear combination of the others — severe multicollinearity

Then:

$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$

| VIF Range | Interpretation |
| :--- | :--- |
| 1 | No multicollinearity |
| 1 – 5 | Moderate; usually acceptable |
| 5 – 10 | High; warrants investigation |
| > 10 | Severe multicollinearity |

Tolerance = $1 / \text{VIF}_j$ is the equivalent inverse metric (low tolerance = high collinearity).

#### c) Condition Number
Based on the eigenvalues of $X'X$:

$$\kappa = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}$$

| Condition Number | Interpretation |
| :--- | :--- |
| < 10 | No issue |
| 10 – 30 | Moderate multicollinearity |
| > 30 | Severe multicollinearity |

This is more comprehensive than VIF because it detects multicollinearity involving combinations of more than two variables.

#### d) Behavioral Red Flags in the Model
- Large standard errors relative to coefficient magnitudes
- Coefficients with unexpected signs (e.g., price has a positive sign in a demand model)
- Coefficients change dramatically when one variable is added or removed

### 3. Handling Multicollinearity

The right remedy depends on whether the goal is inference or prediction, and how severe the collinearity is.

#### a) Drop or Merge Collinear Variables
The simplest fix: use domain knowledge to remove one of two near-duplicate predictors, or combine them into a single meaningful feature (e.g., replace two correlated macro indicators with their average or a composite index).

#### b) Regularization (Best General-Purpose Fix)

**Ridge Regression (L2):**

$$\hat{\beta}_{\text{ridge}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|^2 + \lambda\|\beta\|^2 \right] = (X'X + \lambda I)^{-1} X'y$$

Adding $\lambda I$ ensures the matrix is always invertible, even under perfect collinearity. Ridge shrinks all coefficients toward zero but retains all predictors — it trades variance for bias. Best when all predictors are potentially relevant.

**Lasso Regression (L1):**

$$\hat{\beta}_{\text{lasso}} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|^2 + \lambda\|\beta\|_1 \right]$$

L1 penalty produces sparse solutions — some coefficients shrink to exactly zero. When a group of variables is mutually collinear, Lasso tends to pick one and zero out the rest. Best when you suspect only a subset of predictors are truly relevant.

**Elastic Net (L1 + L2):**

$$\hat{\beta} = \underset{\beta}{\arg\min} \left[ \|y - X\beta\|^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|^2 \right]$$

Combines the sparsity of Lasso with the grouping behavior of Ridge. Best when there are groups of correlated predictors and you want to either select the whole group or none.

#### c) Dimensionality Reduction

**Principal Component Regression (PCR):**
Transform the correlated predictors into orthogonal principal components (PCA), then regress on the top $k$ components. Multicollinearity is eliminated by construction since PCs are orthogonal. The trade-off is interpretability — PCs are linear combinations of original features, not directly meaningful.

**Partial Least Squares (PLS):**
Similar to PCR, but constructs latent components that maximize covariance with the response variable, not just variance in $X$. Often outperforms PCR when the directions of maximum variance in $X$ are not aligned with predictive directions for $y$.

#### d) Collect More Data
More observations reduce $\text{Var}(\hat{\beta})$ directly (via larger $SST_j$). Not always feasible but worth considering when the collinearity is mild.

### 4. Summary

| Situation | Recommended Approach |
| :--- | :--- |
| Two predictors are near-identical | Drop one; use domain knowledge |
| Many correlated predictors, inference goal | Ridge (stabilizes all coefficients) |
| Many correlated predictors, sparsity expected | Lasso or Elastic Net |
| High-dimensional features, prediction goal | PCR or PLS |
| Dummy variable trap (perfect collinearity) | Drop one dummy category (reference group) |

The most important point: **multicollinearity does not bias OLS, it inflates variance.** If you only care about prediction accuracy, moderate multicollinearity is often tolerable. If you care about interpreting individual coefficients — for factor attribution, risk decomposition, or causal inference — it must be addressed.

### 5. Proof: Why Multicollinearity Inflates Variance

**Setup.** The standard linear model:
$$y = X\beta + \epsilon, \quad E[\epsilon] = 0, \quad \text{Var}(\epsilon) = \sigma^2 I$$

where $X$ is $n \times p$, full column rank assumed for now.

---

**Step 1 — OLS estimator and unbiasedness.**

$$\hat{\beta} = (X'X)^{-1}X'y$$

Substituting $y = X\beta + \epsilon$:

$$\hat{\beta} = (X'X)^{-1}X'(X\beta + \epsilon) = \beta + (X'X)^{-1}X'\epsilon$$

Taking expectations ($E[\epsilon] = 0$):

$$E[\hat{\beta}] = \beta \quad \checkmark$$

Multicollinearity does not enter here — **OLS is unbiased regardless of collinearity** (as long as $X'X$ is invertible).

---

**Step 2 — Variance-covariance matrix of $\hat{\beta}$.**

$$\text{Var}(\hat{\beta}) = \text{Var}\!\left[(X'X)^{-1}X'\epsilon\right] = (X'X)^{-1}X' \cdot \sigma^2 I \cdot X(X'X)^{-1} = \sigma^2(X'X)^{-1}$$

So $\text{Var}(\hat{\beta}_j) = \sigma^2 \left[(X'X)^{-1}\right]_{jj}$, the $j$-th diagonal element. The goal is to evaluate this element.

---

**Step 3 — Block matrix inverse (Schur complement).**

Partition $X = [X_{-j} \;\; x_j]$, so:

$$X'X = \begin{bmatrix} X_{-j}'X_{-j} & X_{-j}'x_j \\ x_j'X_{-j} & x_j'x_j \end{bmatrix}$$

By the **block matrix inverse formula**, the $(j,j)$ scalar element of $(X'X)^{-1}$ is the reciprocal of the Schur complement of $X_{-j}'X_{-j}$:

$$\left[(X'X)^{-1}\right]_{jj} = \frac{1}{x_j'x_j - x_j'X_{-j}(X_{-j}'X_{-j})^{-1}X_{-j}'x_j}$$

Define the **annihilator matrix** $M_{-j} = I - X_{-j}(X_{-j}'X_{-j})^{-1}X_{-j}'$, which projects onto the orthogonal complement of $\text{col}(X_{-j})$. Then:

$$\left[(X'X)^{-1}\right]_{jj} = \frac{1}{x_j'M_{-j}x_j}$$

Let $\tilde{x}_j = M_{-j}x_j$ — the **residual from regressing $x_j$ on all other predictors**. Then:

$$\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{\tilde{x}_j'\tilde{x}_j}$$

---

**Step 4 — Connect $\tilde{x}_j'\tilde{x}_j$ to $R_j^2$.**

$\tilde{x}_j'\tilde{x}_j$ is the residual sum of squares (RSS) from the **auxiliary regression** of $x_j$ on all other predictors $X_{-j}$:

$$x_j = X_{-j}\hat{\gamma} + \tilde{x}_j$$

Here $R_j^2$ is the $R^2$ of this auxiliary regression — **not** the $R^2$ of the main regression of $y$ on $X$. It measures how much of $x_j$'s variation is linearly explained by the other predictors. A high $R_j^2$ means $x_j$ is nearly redundant given the rest, which is precisely the definition of multicollinearity.

$$R_j^2 = 1 - \frac{\tilde{x}_j'\tilde{x}_j}{SST_j} \implies \tilde{x}_j'\tilde{x}_j = SST_j(1 - R_j^2)$$

where $SST_j = \sum_i(x_{ji} - \bar{x}_j)^2$ is the total variation in $x_j$.

Substituting:

$$\boxed{\text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{SST_j(1 - R_j^2)} = \frac{\sigma^2}{SST_j} \cdot \text{VIF}_j}$$

---

**Interpretation.**

| Term | Role |
| :--- | :--- |
| $\sigma^2$ | Irreducible noise in $y$ |
| $SST_j$ | Variation in $x_j$ — more data or a wider range of $x_j$ reduces variance |
| $1 - R_j^2$ | The "unique" variation in $x_j$ not explained by other predictors |

As $R_j^2 \to 1$ (i.e., $x_j$ is nearly a linear combination of the other predictors), $(1 - R_j^2) \to 0$, so $\text{VIF}_j \to \infty$ and $\text{Var}(\hat{\beta}_j) \to \infty$. The coefficient estimate exists but is arbitrarily noisy.

At the limit of **perfect multicollinearity** ($R_j^2 = 1$), $\tilde{x}_j'\tilde{x}_j = 0$, $X'X$ becomes singular, and the OLS estimator no longer exists — consistent with the formula blowing up.