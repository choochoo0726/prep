# Alternative Data Preparation and Evaluation

This document outlines key considerations for evaluating, preparing, and handling alternative datasets in a quantitative research environment.

---

## I. Vendor Evaluation Questions: The Due Diligence Phase

Before touching the data, you must evaluate the structural integrity and legal safety of the dataset.

### 1. Data Integrity and Lineage
*   **What is the primary source of the data?** Is it direct-from-source (e.g., credit card exhaust from a bank), web-scraped, or aggregated from third-party apps? Aggregated data is highly vulnerable to "inorganic" panel shocks if the vendor loses a sub-supplier.
    *   *💡 Real-World Example:* A vendor aggregating credit card data from multiple banks suddenly loses a contract with Chase. If you don't know the lineage, your model will mistakenly interpret the massive drop in transaction volume as a severe macroeconomic recession.
*   **Compliance and PII (Personally Identifiable Information):** Does the dataset contain toxic PII? Has it been properly anonymized? If it’s web-scraped, does it violate the target site's Terms of Service? Trading on non-compliant data introduces massive legal and reputational risk.
*   **Restatements and Revisions:** If the vendor changes a data point post-facto (e.g., a merchant reclassifies a transaction 30 days later), how is that flagged? You must ensure your backtest only uses data that was *actually known* on that historical date.
    *   *💡 Real-World Example:* A vendor restates an entire quarter of retail sales because a major retailer changed how they report returns. If your backtest uses the restated data, you have "Look-Ahead Bias," trading on information that was impossible to know at the time.
*   **Point-in-Time (PiT) vs. Event-Time:** Does the timestamp reflect when the event occurred (Event-Time) or when the data was actually published to a subscriber's FTP/API (PiT)? Using Event-Time in a backtest creates fatal **Look-Ahead Bias**.

### 2. Methodology and Pre-processing
*   **Pre-Smoothing:** Are they applying outlier detection or normalization before you get the data? You need to know if they are clipping "noise" that might actually be a valid macroeconomic signal.
*   **Survivorship Bias:** Does the dataset include companies that have gone bankrupt or been acquired, or is it only currently active tickers? If bankrupt companies are dropped, historical performance will look artificially inflated.
    *   *💡 Real-World Example:* If you are testing a signal on App Store downloads and the vendor deletes historical data for apps that were banned last year, your model will falsely conclude that every app in the historical dataset was a long-term winner.
*   **Mapping and Corporate Actions:** How do they map raw data to tradable entities (e.g., mapping a specific subsidiary's brand name to a parent company's FIGI/Ticker)? Furthermore, do they automatically adjust historical data for M&A, spin-offs, or ticker changes?

---

## II. Data Preparation Techniques: Cleaning the Signal

### 1. Handling Panel Instability
Alternative data panels rarely represent a stable population. Users churn, and vendors sign new partnerships.

*   **Cohort-Based Filtering:** Create a "fixed cohort" of users active for a minimum continuous period. This removes the noise of users onboarding or abandoning the platform.
    *   **Formula:** $C = \{i : \text{active}(i, \tau) \in \text{Panel}, \forall \tau \in [t-k, t]\}$
    *   *💡 Real-World Example:* Instead of looking at "Total Amazon Purchases" (which naturally grows as the vendor adds more users to the panel), you look only at the subset of 50,000 users who have been in the panel for exactly 24 continuous months, tracking whether *their* spending behavior is changing.
*   **Panel Normalization (Market-Share Method):** Divide the metric of interest by a "control" metric to get a ratio that is independent of panel size fluctuations.
    *   **Formula:** $M_{i,t} = \frac{X_{i,t}}{\sum_{j \in \text{Panel}} X_{j,t}}$
    *   *💡 Real-World Example:* If the vendor's total panel size drops by 20%, both Uber and Lyft recorded rides will instantly drop. But if you track the ratio `Uber Rides / (Uber Rides + Lyft Rides)`, this "market share" metric remains stable despite the panel volatility.

### 2. Removing Geographical and Demographic Bias
*   **Reweighting via Ground Truth:** Compare your panel’s distribution to a ground-truth source (like the U.S. Census or BLS data) and apply corrective weights.
    *   **Formula:** $w_g = \frac{P(\text{Group}_g | \text{Census})}{P(\text{Group}_g | \text{Panel})}$
    *   *💡 Real-World Example:* If your mobile location panel is 40% from New York and California (but they are only ~20% of the actual US population), you must downweight those coastal pings to accurately forecast a national retailer like Walmart.
*   **Regional De-averaging:** Build separate regional sub-signals and aggregate them based on actual economic weighting, rather than letting coastal urban centers dominate the signal.
    *   **Formula:** $\hat{S}_{\text{aggregate}} = \sum_{r} \beta_r \cdot \text{Normalize}(S_r)$
*   **Propensity Score Matching (PSM):** Run a logistic regression to calculate the probability ($e$) that a person is in your dataset given their traits ($X$). Use this to select a subset of users that perfectly mirrors the national average.
    *   **Formula:** $e(X) = P(D=1 | X)$

---

## III. Risks of Reweighting Alternative Datasets

Reweighting is a dangerous mathematical hammer. If pushed too hard, you destroy alpha.

### 1. Variance Inflation and Effective Sample Size (ESS)
*   **The Math:** $\text{Var}(\hat{Y}) = \sum w_i^2 \text{Var}(y_i)$. Large weights $w_i$ (applied to severely under-represented groups) exponentially increase the estimator's variance.
*   **The Diagnostic (ESS):** You must calculate the Effective Sample Size to know if you've destroyed your data. 
    *   **Formula:** $\text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}$
    *   *Rule of Thumb:* If your ESS drops below 30% of your actual panel size after reweighting, your weights are too extreme. Your signal is now driven by a tiny handful of heavily-weighted users.
    *   *💡 Real-World Example (The "Small Cell" Problem):* Your panel has only 5 people from Rural North Dakota, but they need a $50x$ weight to match the Census. If one of those 5 people buys a new tractor on their credit card, your model, multiplying that by 50, might falsely predict a massive national surge in John Deere sales.

### 2. Dimension Interdependency (The "Whack-a-Mole" Effect)
*   **The Conflict:** Reweighting for Geography may inadvertently worsen Income bias, because those variables are highly correlated.
    *   *💡 Real-World Example:* You reweight your panel to have more 18-24 year olds to match the Census. But in your specific dataset, the 18-24 year olds happen to be wealthy software engineers. By "fixing" the Age bias, you accidentally created a massive Income bias.
*   **The Solution (Raking):** Quants use **Iterative Proportional Fitting (Raking)** to iteratively adjust weights until multiple marginal distributions (e.g., Age, Income, Geo) simultaneously match the Census. However, this dramatically increases the risk of over-fitting.

---

## IV. Seasonality Removal Frameworks: A Practitioner's Guide

When removing seasonality from alternative data, you are making a fundamental bet on what is "normal" and what is "alpha."

### 1. Year-Over-Year (YoY) Differencing
The "Retail Heuristic" and most common industry baseline.
*   **How to Use it:** Apply log-differencing ($\log(Y_t) - \log(Y_{t-s})$). Always use a "Base Check": If $Y_{t-s}$ is an outlier (check its Z-score), flag the resulting growth rate as unreliable.
*   **Why it is Good:** Zero parameterization, no convergence issues, and stakeholder alignment.
*   **Best Problem Profile:** Short histories (< 24 months) and Sanity Checks ("Floor" model).
    *   *💡 Real-World Example:* Home Depot saw a massive, one-time spike in sales in April 2020 due to pandemic lockdowns. In April 2021, the YoY growth rate was deeply negative. If you didn't check the "Base Effect" of 2020, you would have falsely assumed Home Depot's business was collapsing in 2021.

### 2. STL Decomposition (Seasonal-Trend using LOESS)
The "Visualizer’s Choice" for slowly evolving seasonal patterns.
*   **How to Use it:** Tweak the `seasonal.window` (LOESS filter width). Use `robust=True` to ensure shocks don't bleed into the trend.
*   **Why it is Good:** Flexibility to handle "evolving" peaks (e.g., e-commerce shifts) and handles additive/multiplicative seamlessly via log transforms.
*   **Best Problem Profile:** Long-Term Thematic Trading and Visual Alpha Discovery.
    *   *💡 Real-World Example:* Ten years ago, the peak holiday shopping week was strictly Black Friday. Today, it has shifted earlier into early November due to "Cyber Month" promotions. STL with a small `seasonal.window` dynamically adapts to this slow multi-year shift, whereas YoY assumes the pattern is perfectly static.

### 3. X-13ARIMA-SEATS
The "Econometric Standard" for high-precision macro and banking data.
*   **How to Use it:** Specify "Trading Day" regressors and run "Sliding Spans" diagnostics.
*   **Why it is Good:** Calendar Precision (adjusts for the number of weekends in a month) and Benchmark Fidelity (matches government methodology).
*   **Best Problem Profile:** High-Volume Transaction Data and Macro-Leading Indicators.
    *   *💡 Real-World Example:* March 2024 has 5 Fridays, 5 Saturdays, and 5 Sundays. March 2023 had only 4 of each. A restaurant chain's credit card transactions will look artificially high in 2024 simply due to the calendar. X-13ARIMA applies a "Trading Day Penalty" to mathematically neutralize this artifact.

### 4. Prophet (Generalized Additive Model)
The "Data Scientist’s Tool" for messy, high-frequency, real-world data.
*   **How to Use it:** Add custom `holidays` dataframes and tune the `changepoint_prior_scale`.
*   **Why it is Good:** Native Gap Handling and Multi-Frequency modeling (daily/weekly/yearly simultaneously).
*   **Best Problem Profile:** Messy web/app traffic data and Event-Driven Trading.
    *   *💡 Real-World Example:* Predicting daily server traffic for a streaming service. You know traffic drops every Tuesday, spikes every Friday, plummets on July 4th, and you have 3 random days of missing data from a server outage last month. Prophet handles the multiple cycles, the known holiday, and the missing data natively.

---

## V. Isolating and Trading on "Innovation"

By removing trend and seasonality, you perform a **Whitening Transformation** to isolate the **Innovation** (Shock).

### 1. Mathematical Definition
$$Y_t = \text{Known}_t + \text{Innovation}_t$$
The Innovation ($\epsilon_t$) is the unexpected "Information Flow" not yet priced in by the market.

### 2. Transient vs. Structural Innovation
*   **Transient:** Residual spikes. These mean-revert quickly.
    *   *💡 Real-World Example:* A severe winter storm freezes a major logistics hub for a week. Shipments plummet. This is a transient shock—the packages will simply be shipped next week, causing a positive spike that cancels the negative one out.
*   **Structural:** Trend acceleration (the second derivative of the trend). This suggests a fundamental regime shift in market share or consumer behavior.
    *   *💡 Real-World Example:* A major competitor files for bankruptcy. Your target company's shipments jump 10% and stay there permanently. This is a structural innovation—a permanent regime shift.

### 3. Advanced Extraction (Removing Autocorrelation)
Even after removing seasonality, time-series data often has "momentum" (autocorrelation). You must remove this to find the *true* surprise.
*   **AR Filtering:** Fit an Autoregressive model of order $p$:
    $$\epsilon_t = Y_t - (\phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p})$$
*   **Choosing $p$:** Use the Partial Autocorrelation Function (PACF) or minimize the Akaike Information Criterion (AIC) to select how many lag periods to subtract. The remaining $\epsilon_t$ is pure, unadulterated alpha.

### 4. Dynamic Extraction via Kalman Filters (State-Space Innovation)
The **Kalman Filter** is the premier tool for quantitative, real-time extraction of innovations. Unlike STL or moving averages, which are "backward-looking," a Kalman Filter is a recursive algorithm that maintains an internal "state" of the system and continuously updates its beliefs as new data arrives.

*   **The Core Assumption:** The underlying "true" trend ($x_t$) is unobservable (hidden) and evolves over time with some system noise. What we actually observe ($y_t$) is this true trend corrupted by measurement noise (e.g., scraping errors, daily jitter).
*   **The Mathematical Formulation:**
    *   **State Equation (The Hidden Reality):** $x_t = A \cdot x_{t-1} + w_t$ (where $w_t \sim N(0, Q)$ is the process noise).
    *   **Measurement Equation (What We See):** $y_t = H \cdot x_t + v_t$ (where $v_t \sim N(0, R)$ is the measurement noise).
*   **The Innovation ($\epsilon_t$):** The Kalman Filter makes a prediction for tomorrow ($\hat{y}_t$). When tomorrow's actual data ($y_t$) arrives, the difference is the **Innovation** (or Measurement Residual):
    $$\epsilon_t = y_t - \hat{y}_t$$
    *If this innovation is consistently positive over several days, the Kalman Filter automatically adjusts its hidden state ($x_t$) upward, realizing this is a structural shift, not just noise.*
*   **Best Problem Profile:** High-frequency, real-time trading environments (MFT/HFT) where you need to detect structural regime shifts *before* a traditional rolling average catches up.
    *   *💡 Real-World Example:* You track hourly satellite data of parking lot cars at Walmart. The data is extremely noisy (clouds, sensor glitches). A Kalman Filter maintains a "true expected volume." If a massive promotion drives a sudden 30% spike in cars, the Kalman Filter's prediction error ($\epsilon_t$) spikes instantly, generating a tradable signal on day one, whereas a 14-day rolling average would barely move.

*   **Python Implementation (`pykalman`):**
```python
from pykalman import KalmanFilter
import numpy as np
import pandas as pd

def extract_kalman_innovation(series):
    """
    Uses a 1D Kalman Filter to smooth the series and extract the Innovation.
    """
    # Initialize a basic random walk Kalman Filter
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01 # Tuning parameter: lower = smoother trend
    )
    
    # Optional: Use the EM algorithm to estimate the optimal noise covariances
    # kf = kf.em(series.values, n_iter=5)
    
    # Filter the data
    state_means, _ = kf.filter(series.values)
    
    # The 'state_means' is the Kalman Filter's estimate of the "Known" trend
    # The Innovation is the Actual minus the Estimated Trend
    innovations = series.values - state_means.flatten()
    
    return pd.Series(innovations, index=series.index)

# Example Usage:
# df['kalman_innovation'] = extract_kalman_innovation(df['raw_shipments'])
```

---

## VI. Normalizing Innovation: The Z-Score Workflow

To convert raw innovations into tradable signals, you must normalize them.

### 1. The Core Formula (Rolling/Dynamic Z-score)
Because alternative data volatility regimes shift over time, a static Z-score is useless.
$$Z_t = \frac{\epsilon_t - \mu_{\epsilon(t, w)}}{\sigma_{\epsilon(t, w)}}$$

### 2. Cross-Sectional vs. Time-Series Z-Scoring
*   **Time-Series Z-Score (Self-Relative):** Compares Target's innovation today against Target's own historical innovations. Excellent for directional, absolute-return trading.
*   **Cross-Sectional Z-Score (Peer-Relative):** At time $t$, calculates the Z-score of Target's innovation relative to the distribution of innovations across its peer group (e.g., Target vs. Walmart vs. Costco). This neutralizes sector-wide macro shocks and isolates idiosyncratic company alpha. Essential for Market-Neutral / Statistical Arbitrage strategies.
    *   *💡 Real-World Example:* Target has a Time-Series Z-score of $+1.5$ today (a solid positive surprise compared to its own history). However, you look at the Cross-Sectional Z-score for the Retail Sector, and Walmart, Costco, and Kroger are all at $+3.0$ due to a new government stimulus check. Cross-sectionally, Target is actually underperforming its peers and might be a *short* target in a pair trade.

### 3. Robust Z-scores and EWMA (Handling Fat Tails)
*   **Exponentially Weighted Moving Average (EWMA):** Instead of a simple rolling window, calculate $\mu$ and $\sigma$ using EWMA ($\mu_t = \alpha x_t + (1-\alpha)\mu_{t-1}$) to make the Z-score react faster to new volatility regimes.
*   **Median Absolute Deviation (MAD):** Standard Z-scores are broken by massive outliers. Use MAD for a robust estimation:
    $$Z_{modified} = \frac{0.6745 \times (\epsilon_t - \text{Median})}{\text{MAD}}$$

### 4. Python Implementation: Rolling Robust Z-Score
```python
import pandas as pd
import numpy as np

def robust_zscore(series, window=21):
    rolling_median = series.rolling(window=window).median()
    def get_mad(x):
        return np.median(np.abs(x - np.median(x)))
    rolling_mad = series.rolling(window=window).apply(get_mad, raw=True)
    return (0.6745 * (series - rolling_median)) / rolling_mad
```

---

## VII. Summary of Frequency & Model Fit

| Frequency | Recommended Framework | Why? | Risk |
| :--- | :--- | :--- | :--- |
| **Monthly** | X-13ARIMA-SEATS | Best-in-class for Trading Day adjustment. | High setup complexity. |
| **Weekly** | Prophet | Handles "53rd week" and holiday drift natively. | Can over-smooth signals. |
| **Daily** | MSTL or Prophet | Decomposes Weekly + Yearly cycles. | High computational cost. |
| **Sparse** | Prophet | Robust to irregular sampling and gaps. | High risk of over-fitting noise. |
| **Real-Time**| Kalman Filter | Recursive/Online extraction of Innovations. | Requires precise state-space tuning. |
