# Feature Generation for Alpha Signal Generation (Alternative Data)
# 另类数据 Alpha 信号特征生成完整指南

---

## 0. Vendor Due Diligence: Before Touching the Data
## 0. 数据供应商尽职调查：接触数据之前

Before any feature engineering, the structural integrity and legal safety of the dataset must be evaluated.

*在任何特征工程之前，必须评估数据集的结构完整性和法律合规性。*

### 0.1 Data Integrity and Lineage
### 0.1 数据完整性与来源追溯

**What is the primary source?** Is it direct-from-source (e.g., credit card exhaust from a bank), web-scraped, or aggregated from third-party apps? Aggregated data is highly vulnerable to "inorganic" panel shocks if the vendor loses a sub-supplier.

*【数据主要来源是什么？】是直接来源（如银行信用卡流水）、网络爬取，还是从第三方App聚合？聚合数据极易受到"非有机"面板冲击——若供应商失去某个子供应商合同，数据会无故暴跌。*

> **Real-World Example:** A vendor aggregating credit card data from multiple banks suddenly loses a contract with Chase. Without knowing the lineage, your model will interpret the massive drop in transaction volume as a severe macroeconomic recession.
>
> *【真实案例】某供应商汇聚多家银行信用卡数据，突然失去与摩根大通的合同。若不了解数据来源，模型会将交易量的大幅下降误判为严重的宏观衰退。*

**Compliance and PII (Personally Identifiable Information):** Does the dataset contain toxic PII? Has it been properly anonymized? If web-scraped, does it violate the target site's Terms of Service? Trading on non-compliant data introduces massive legal and reputational risk.

*【合规与个人身份信息（PII）】数据集是否含有敏感PII？是否经过适当匿名化？若为爬取数据，是否违反目标网站的服务条款？基于不合规数据交易会带来巨大的法律和声誉风险。*

**Restatements and Revisions:** If the vendor changes a data point post-facto, how is it flagged? Your backtest must only use data that was *actually known* on that historical date.

*【数据修订与重述】若供应商事后更改数据点，如何标记？回测必须只使用在历史日期上**实际已知**的数据。*

> **Real-World Example:** A vendor restates an entire quarter of retail sales because a major retailer changed how they report returns. A backtest using restated data has **Look-Ahead Bias** — trading on information that was impossible to know at the time.
>
> *【真实案例】某供应商因一家大型零售商改变了退货报告方式而重述整个季度的零售销售数据。使用重述数据的回测存在**前视偏差**——基于当时不可能知道的信息交易。*

**Point-in-Time (PiT) vs. Event-Time:** Does the timestamp reflect when the event occurred (Event-Time) or when the data was actually published to a subscriber's API (PiT)? Using Event-Time in a backtest creates fatal Look-Ahead Bias.

*【时点数据 vs. 事件时间】时间戳反映的是事件发生时间（事件时间）还是数据实际发布给订阅者的时间（时点数据）？在回测中使用事件时间会产生致命的前视偏差。*

### 0.2 Methodology and Pre-processing by Vendor
### 0.2 供应商的方法论与预处理

**Pre-Smoothing:** Is the vendor applying outlier detection or normalization before delivering the data? You need to know if they are clipping "noise" that might actually be a valid macroeconomic signal.

*【预平滑】供应商在交付数据前是否做了异常值检测或归一化？你需要知道他们是否裁剪了可能实际上是有效宏观信号的"噪声"。*

**Survivorship Bias:** Does the dataset include companies that have gone bankrupt or been acquired, or only currently active tickers? If bankrupt companies are dropped, historical performance will look artificially inflated.

*【幸存者偏差】数据集是否包含已破产或被收购的公司，还是只有当前活跃的股票代码？若剔除破产公司，历史表现将被人为夸大。*

> **Real-World Example:** Testing a signal on App Store downloads where the vendor deletes historical data for apps banned last year — the model falsely concludes every historical app was a long-term winner.
>
> *【真实案例】在App Store下载量上测试信号，但供应商删除了去年被下架App的历史数据——模型错误地认为历史数据集中的每个App都是长期赢家。*

**Mapping and Corporate Actions:** How do they map raw data to tradable entities (e.g., subsidiary brand name → parent company ticker)? Do they adjust historical data for M&A, spin-offs, or ticker changes?

*【映射与公司行动】如何将原始数据映射到可交易主体（如子公司品牌名→母公司股票代码）？是否对并购、分拆或股票代码变更调整历史数据？*

---

## 1. Data Quality Control, Panel Stability & Reweighting
## 1. 数据质量控制、面板稳定性与重新加权

### 1.1 Handling Panel Instability
### 1.1 处理面板不稳定性

Alternative data panels rarely represent a stable population. Users churn, and vendors sign new partnerships.

*另类数据面板很少代表稳定的总体。用户流失，供应商不断签署新合作。*

**Cohort-Based Filtering:** Create a "fixed cohort" of users active for a minimum continuous period. This removes the noise of users onboarding or abandoning the platform.

*【基于队列的过滤】创建一个在最短连续期间内均活跃的"固定队列"用户集合。这消除了用户入驻或离开平台的噪声。*

$$C = \{i : \text{active}(i, \tau) \in \text{Panel}, \forall \tau \in [t-k, t]\}$$

> **Real-World Example:** Instead of "Total Amazon Purchases" (which grows as the vendor adds more users), track only the 50,000 users who have been in the panel for exactly 24 continuous months — watch whether *their* spending behavior is changing.
>
> *【真实案例】不看"亚马逊总购买量"（随供应商增加用户而自然增长），而只追踪在面板中连续24个月都存在的5万用户——观察**他们的**消费行为是否在变化。*

**Panel Normalization (Market-Share Method):** Divide the metric of interest by a "control" metric to get a ratio independent of panel size fluctuations.

*【面板归一化（市场份额法）】将关注指标除以"控制"指标，得到与面板规模波动无关的比率。*

$$M_{i,t} = \frac{X_{i,t}}{\sum_{j \in \text{Panel}} X_{j,t}}$$

> **Real-World Example:** If the panel size drops by 20%, both Uber and Lyft recorded rides drop. But `Uber Rides / (Uber Rides + Lyft Rides)` remains stable despite the panel volatility.
>
> *【真实案例】若面板规模下降20%，Uber和Lyft的记录行程数量均下降。但 `Uber行程 / (Uber行程 + Lyft行程)` 在面板波动中依然稳定。*

### 1.2 Removing Geographical and Demographic Bias
### 1.2 去除地理与人口偏差

**Reweighting via Ground Truth:** Compare the panel's distribution to a ground-truth source (U.S. Census, BLS) and apply corrective weights.

*【基于基准数据重新加权】将面板分布与真实来源（美国人口普查、劳工统计局）比较，施加校正权重。*

$$w_g = \frac{P(\text{Group}_g | \text{Census})}{P(\text{Group}_g | \text{Panel})}$$

> **Real-World Example:** If a mobile location panel is 40% from New York and California (but they are only ~20% of the US population), you must downweight those coastal pings to accurately forecast a national retailer like Walmart.
>
> *【真实案例】若移动位置面板40%来自纽约和加州（但其实际人口仅占美国约20%），必须降低这些沿海数据的权重，以准确预测沃尔玛等全国性零售商。*

**Regional De-averaging:** Build separate regional sub-signals and aggregate based on actual economic weighting rather than letting coastal urban centers dominate.

*【区域去平均化】构建独立的区域子信号，按实际经济权重聚合，而非让沿海都市中心主导信号。*

$$\hat{S}_{\text{aggregate}} = \sum_{r} \beta_r \cdot \text{Normalize}(S_r)$$

**Propensity Score Matching (PSM):** Run a logistic regression to calculate the probability that a person is in your dataset given their traits. Use this to select a subset of users that mirrors the national average.

*【倾向得分匹配（PSM）】通过逻辑回归计算某人基于其特征进入数据集的概率，用于筛选镜像全国平均水平的用户子集。*

$$e(X) = P(D=1 | X)$$

### 1.3 Risks of Reweighting
### 1.3 重新加权的风险

Reweighting is a dangerous mathematical hammer. Pushed too hard, it destroys alpha.

*重新加权是一把危险的数学锤子。力度过大，会破坏Alpha。*

**Variance Inflation and Effective Sample Size (ESS):**

*【方差膨胀与有效样本量（ESS）：】*

$$\text{Var}(\hat{Y}) = \sum w_i^2 \text{Var}(y_i) \qquad \text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}$$

*Rule of thumb: if ESS drops below 30% of actual panel size after reweighting, the weights are too extreme.*

*经验法则：重新加权后ESS低于实际面板规模30%，则权重过于极端。*

> **Real-World Example (The "Small Cell" Problem):** 5 people from Rural North Dakota need a 50× weight to match the Census. If one buys a new tractor, the model — multiplying by 50 — falsely predicts a massive national surge in John Deere sales.
>
> *【真实案例（"小单元"问题）】北达科他州农村5个人需要50倍权重以匹配人口普查。若其中一人购买了一台拖拉机，模型乘以50后，会错误预测约翰迪尔销售全国大幅增长。*

**Dimension Interdependency (The "Whack-a-Mole" Effect):** Reweighting for Geography may inadvertently worsen Income bias because those variables are correlated.

*【维度相互依赖（"打地鼠"效应）】对地理重新加权可能无意间加剧收入偏差，因为这些变量高度相关。*

**Solution (Raking / Iterative Proportional Fitting):** Iteratively adjust weights until multiple marginal distributions (Age, Income, Geo) simultaneously match the Census. However, this dramatically increases overfitting risk.

*【解决方案（Raking / 迭代比例拟合）】迭代调整权重，直到多个边际分布（年龄、收入、地理）同时匹配人口普查。但这会大幅增加过拟合风险。*

### 1.4 Missing Data Strategies
### 1.4 缺失值处理策略

| Strategy (策略) | When to Use (适用场景) |
|---|---|
| Forward fill (carry last value) (前向填充) | Metric is a stock/level that doesn't change until updated, e.g., reported revenue (指标是存量/水平值，直到更新前不变，如已披露营收) |
| Interpolation (插值) | Smooth underlying process, e.g., web traffic, estimated daily spend (底层过程平滑，如网络流量、估计日消费) |
| Flag as missing — no imputation (标记为缺失，不插补) | Data absence is itself informative, e.g., company stopped transacting (数据缺失本身有信息含义，如公司停止交易) |
| Cross-sectional median imputation (截面中位数插补) | Missing data is random — coverage gap, not structural (缺失是随机的——覆盖缺口，非结构性) |

**Stale data detection:** If an entity's data hasn't updated in N periods, flag or drop it. A "frozen" signal is worse than no signal because it looks valid but is not.

*【过时数据检测】若某主体数据在N期内未更新，标记或丢弃。"冻结"信号比没有信号更危险，因为它看起来有效但实际上已经失效。*

---

## 2. Base Signals: Growth Features
## 2. 基础信号：增长类特征

Starting point for most alternative data signals. Compute raw metric growth across multiple horizons:

*大多数另类数据信号的起点。对原始指标计算不同时间窗口的增长率：*

| Horizon (时间窗口) | Example (示例) |
|---|---|
| Weekly YoY (周度同比) | `metric(week_t) / metric(week_{t-52}) - 1` |
| Monthly YoY (月度同比) | `metric(month_t) / metric(month_{t-12}) - 1` |
| Quarterly YoY (季度同比) | `metric(quarter_t) / metric(quarter_{t-4}) - 1` |
| Sequential — MoM, QoQ (环比，月/季) | `metric(t) / metric(t-1) - 1` |

**Why YoY?** Removes seasonality natively. Sequential growth retains seasonality — useful when you want to capture within-year acceleration.

*【为什么用同比？】同比天然消除季节性因素。环比则保留季节性——当你想捕捉年内加速/减速时环比更有用。*

---

## 3. Additional Transformations
## 3. 其他特征变换

### 3.1 Growth of Growth (Acceleration / Deceleration)
### 3.1 增长率的增长（加速/减速）

```
growth_accel(t) = growth(t) - growth(t-1)
```

**Intuition:** First derivative tells you the level of growth; second derivative tells you whether the business is accelerating or slowing. Markets often react more to the *change in rate* than the rate itself (i.e., beat-vs-miss dynamics).

*【直觉】一阶导数反映增长水平；二阶导数反映增长是在加速还是减速。市场往往对增速的变化（超预期/低于预期）比增速本身反应更强烈。*

**Pros:** Captures momentum inflection points early; aligns with analyst revision logic.

*【优点】能较早捕捉动量拐点；与分析师预期修正逻辑一致。*

**Cons:** Noisier — amplifies data volatility; requires more historical depth to be stable.

*【缺点】噪音更大——放大数据波动；需要更长的历史数据才能稳定。*

---

### 3.2 Benchmark Comparison (Relative Signals)
### 3.2 基准比较（相对信号）

Rather than using the raw growth, compute deviation from a reference:

*与其使用原始增长率，不如计算相对于参考基准的偏差：*

| Benchmark Type (基准类型) | Formula (公式) | When to Use (适用场景) |
|---|---|---|
| Rolling historical mean (滚动历史均值) | `growth(t) - mean(growth[t-N:t-1])` | Mean-reverting signals (均值回归信号) |
| Consensus / sell-side estimate (卖方一致预期) | `growth(t) - consensus(t)` | Fundamental surprise signals (基本面超预期信号) |
| Sector / industry median (行业中位数) | `growth(t) - median(sector peers at t)` | Isolate idiosyncratic component (隔离特质性成分) |
| Index / macro baseline (指数/宏观基准) | `growth(t) - macro_growth(t)` | Remove systematic risk (去除系统性风险) |

**Intuition:** Raw growth conflates the macro environment with idiosyncratic firm behavior. Subtracting a benchmark isolates the residual that is company-specific and more likely to be alpha-generating.

*【直觉】原始增长率将宏观环境与公司特质混在一起。减去基准后，剩余部分是公司特有残差，更可能产生Alpha。*

**Pros:** Removes common factors; surfaces relative winners/losers; more stationary than absolute levels.

*【优点】去除共同因子；区分相对赢家和输家；比绝对水平更平稳。*

**Cons:** Benchmark choice carries its own biases; consensus data has coverage gaps; industry classification may be imprecise.

*【缺点】基准选择带有自身偏差；卖方一致预期存在覆盖缺口；行业分类可能不够精确。*

---

### 3.3 Volatility-Adjusted Signal (Signal-to-Noise Ratio)
### 3.3 波动率调整后的信号（信噪比）

```
vol_adj_signal(t) = growth(t) / rolling_std(growth[t-N:t])
```

Conceptually a Sharpe ratio applied to the feature itself.

*概念上相当于对特征本身计算夏普比率。*

**Intuition:** A 10% growth figure means something very different for a company whose growth typically oscillates ±1% vs ±15%. Scaling by the entity's own volatility gives a cleaner read on "how significant is this reading."

*【直觉】10%的增长率对增长波动±1%与±15%的公司意义截然不同。除以自身波动率后，能更清晰地判断"这个读数有多显著"。*

**Pros:** Naturally risk-adjusts the signal; reduces influence of high-volatility entities.

*【优点】天然对信号做风险调整；降低高波动率公司的主导影响。*

**Cons:** Rolling std is noisy with short windows; can inflate signal for low-volatility entities with sparse data.

*【缺点】短窗口下滚动标准差噪音大；在数据稀疏情况下可能对低波动率公司产生虚高信号。*

---

### 3.4 Percentile Rank
### 3.4 百分位排名

```
rank_signal(t) = percentile_rank(signal(t), historical_distribution or cross-section)
```

Non-parametric alternative to z-score.

*Z分数的非参数替代方案。*

**Intuition:** Map the signal to its empirical rank rather than assuming a Gaussian distribution. Robust to outliers.

*【直觉】将信号映射到其经验排名位置，而非假设正态分布。对异常值鲁棒。*

**Pros:** No distributional assumption; bounded [0,1]; handles fat tails naturally.

*【优点】无分布假设；有界[0,1]；天然处理厚尾问题。*

**Cons:** Loses magnitude information at the extremes; depends on universe composition.

*【缺点】丢失极端值的幅度信息；依赖于投资域构成。*

---

### 3.5 Trend Deviation
### 3.5 趋势偏差

Fit a simple trend (linear regression, EMA) to the signal history and compute the residual:

*对信号历史拟合简单趋势（线性回归、指数移动平均），计算残差：*

```
trend_deviation(t) = signal(t) - fitted_trend(t)
```

**Intuition:** Separates the secular trend from cyclical deviation. The *surprise* relative to trend is what moves prices, not the trend itself.

*【直觉】将长期趋势与周期性偏差分离。相对于趋势的"意外"才是推动价格变动的因素。*

**Pros:** Decomposes level from momentum; can be combined with acceleration features.

*【优点】将水平与动量分解；可与加速度特征结合。*

**Cons:** Trend model choice (window, functional form) matters a lot; sensitive to regime breaks.

*【缺点】趋势模型的选择影响很大；对市场状态转换敏感。*

---

## 4. Seasonality Removal Frameworks
## 4. 季节性去除框架

When removing seasonality from alternative data, you are making a fundamental bet on what is "normal" and what is "alpha."

*从另类数据中去除季节性，本质上是对什么是"正常"、什么是"Alpha"做出基本押注。*

### 4.1 Year-Over-Year (YoY) Differencing
### 4.1 同比差分（YoY）

The "Retail Heuristic" and most common industry baseline.

*"零售行业启发法"，最常用的行业基准。*

**How to use:** Apply log-differencing: `log(Y_t) - log(Y_{t-s})`. Always apply a "Base Check": if `Y_{t-s}` is an outlier (check its Z-score), flag the resulting growth rate as unreliable.

*【如何使用】应用对数差分：`log(Y_t) - log(Y_{t-s})`。始终做"基期检查"：若基期值是异常值（检查其Z分数），将对应增长率标记为不可靠。*

**Best for:** Short histories (< 24 months); sanity checks.

*【最适合】历史较短（< 24个月）；合理性检验。*

> **Real-World Example:** Home Depot saw a massive spike in sales in April 2020 due to pandemic lockdowns. In April 2021, the YoY growth rate was deeply negative. Without checking the "Base Effect" of 2020, you would falsely assume Home Depot's business was collapsing.
>
> *【真实案例】家得宝在2020年4月因疫情封锁出现销售额大幅飙升。2021年4月同比增长率深度为负。若不检查2020年的"基期效应"，会错误地认为家得宝业务在崩溃。*

**Pros:** Zero parameterization; no convergence issues; stakeholder alignment.

*【优点】零参数化；无收敛问题；易于与利益相关方对齐。*

**Cons:** One extreme base period contaminates an entire year of signals; only removes annual seasonality.

*【缺点】一个极端基期会污染整年信号；只去除年度季节性。*

---

### 4.2 STL Decomposition (Seasonal-Trend using LOESS)
### 4.2 STL 分解（基于LOESS的季节-趋势分解）

The "Visualizer's Choice" for slowly evolving seasonal patterns.

*"可视化分析师的选择"，适用于缓慢演变的季节性规律。*

**How to use:** Tune the `seasonal.window` (LOESS filter width). Use `robust=True` to prevent shocks from bleeding into the trend.

*【如何使用】调整 `seasonal.window`（LOESS滤波宽度）。使用 `robust=True` 防止冲击渗入趋势成分。*

**Best for:** Long-term thematic trading; visual alpha discovery.

*【最适合】长期主题交易；视觉Alpha挖掘。*

> **Real-World Example:** Ten years ago, peak holiday shopping was strictly Black Friday. Today it has shifted to early November due to "Cyber Month" promotions. STL with a small `seasonal.window` dynamically adapts to this multi-year shift; YoY assumes the pattern is perfectly static.
>
> *【真实案例】十年前，节日购物旺季严格集中于黑色星期五。如今由于"网购月"促销，已提前至11月初。小 `seasonal.window` 的STL能动态适应这种多年期转变；而同比法假设季节性规律完全静态。*

**Pros:** Flexibility to handle "evolving" seasonal peaks; handles additive/multiplicative structures via log transforms.

*【优点】灵活处理"演变"的季节性峰值；通过对数变换处理加法/乘法结构。*

**Cons:** Requires long history to estimate stable components; can over-smooth short-term signals.

*【缺点】需要较长历史来估计稳定成分；可能过度平滑短期信号。*

---

### 4.3 X-13ARIMA-SEATS
### 4.3 X-13ARIMA-SEATS

The "Econometric Standard" for high-precision macro and banking data.

*"计量经济学标准"，适用于高精度宏观和银行数据。*

**How to use:** Specify "Trading Day" regressors and run "Sliding Spans" diagnostics.

*【如何使用】指定"交易日"回归量，运行"滑动跨度"诊断。*

**Best for:** High-volume transaction data; macro-leading indicators.

*【最适合】高频交易量数据；宏观领先指标。*

> **Real-World Example:** March 2024 has 5 Fridays, 5 Saturdays, and 5 Sundays; March 2023 had only 4 of each. A restaurant chain's credit card transactions look artificially high in 2024 simply due to the calendar. X-13ARIMA applies a "Trading Day Penalty" to neutralize this artifact.
>
> *【真实案例】2024年3月有5个星期五、5个星期六、5个星期日；2023年3月只有4个。某连锁餐厅的信用卡交易额在2024年仅因日历因素就显得人为偏高。X-13ARIMA通过"交易日惩罚"在数学上消除这一现象。*

**Pros:** Best-in-class calendar precision (adjusts for trading day composition); matches government methodology.

*【优点】一流的日历精度（调整交易日构成）；与政府统计方法保持一致。*

**Cons:** High setup complexity; requires specialized software.

*【缺点】设置复杂度高；需要专业软件。*

---

### 4.4 Prophet (Generalized Additive Model)
### 4.4 Prophet（广义加法模型）

The "Data Scientist's Tool" for messy, high-frequency, real-world data.

*"数据科学家的工具"，适用于杂乱、高频的真实数据。*

**How to use:** Add custom `holidays` dataframes and tune the `changepoint_prior_scale`.

*【如何使用】添加自定义 `holidays` 数据框，调整 `changepoint_prior_scale`。*

**Best for:** Messy web/app traffic data; event-driven trading.

*【最适合】杂乱的网络/App流量数据；事件驱动交易。*

> **Real-World Example:** Predicting daily server traffic for a streaming service: traffic drops every Tuesday, spikes every Friday, plummets on July 4th, and 3 random days are missing from a server outage. Prophet handles multiple cycles, known holidays, and missing data natively.
>
> *【真实案例】预测流媒体服务的每日服务器流量：每周二下降，每周五飙升，7月4日骤降，另有3天因服务器故障数据缺失。Prophet天然处理多重周期、已知节假日和缺失数据。*

**Pros:** Native gap handling; multi-frequency modeling (daily/weekly/yearly simultaneously).

*【优点】天然处理数据缺口；多频率建模（同时处理日/周/年周期）。*

**Cons:** Can over-smooth signals; high risk of overfitting noise with many custom changepoints.

*【缺点】可能过度平滑信号；自定义变点过多时过拟合风险高。*

---

### 4.5 Frequency and Model Selection Guide
### 4.5 频率与模型选择指南

| Data Frequency (数据频率) | Recommended Framework (推荐框架) | Why (原因) | Risk (风险) |
|---|---|---|---|
| Monthly (月度) | X-13ARIMA-SEATS | Best-in-class Trading Day adjustment (最佳交易日调整) | High setup complexity (设置复杂) |
| Weekly (周度) | Prophet | Handles "53rd week" and holiday drift (处理"第53周"和节假日漂移) | Can over-smooth (可能过度平滑) |
| Daily (日度) | MSTL or Prophet | Decomposes Weekly + Yearly cycles (分解周+年周期) | High computational cost (计算成本高) |
| Sparse (稀疏) | Prophet | Robust to irregular sampling and gaps (对不规则采样和缺口鲁棒) | High overfitting risk (高过拟合风险) |
| Real-Time (实时) | Kalman Filter | Recursive online extraction of Innovations (递归在线提取创新量) | Requires precise tuning (需要精确调参) |

---

## 5. Innovation Extraction: Trading the Unexpected
## 5. 创新量提取：交易意外变动

By removing trend and seasonality, you perform a **Whitening Transformation** to isolate the **Innovation** (Shock) — the unexpected "information flow" not yet priced in by the market.

*通过去除趋势和季节性，你执行了一次**白化变换**，将**创新量**（冲击）隔离出来——市场尚未定价的意外"信息流"。*

$$Y_t = \text{Known}_t + \text{Innovation}_t$$

### 5.1 Transient vs. Structural Innovation
### 5.1 暂时性创新 vs. 结构性创新

**Transient Innovation:** Residual spikes that mean-revert quickly.

*【暂时性创新】残差尖峰，快速均值回归。*

> **Example:** A severe winter storm freezes a major logistics hub for a week. Shipments plummet. This is transient — the packages will be shipped next week, creating a positive spike that cancels the negative one.
>
> *【示例】严重冬季风暴冻结主要物流枢纽一周，货运量骤降。这是暂时性的——包裹下周就会发出，形成正向尖峰抵消负向尖峰。*

**Structural Innovation:** Trend acceleration (the second derivative of the trend) — a fundamental regime shift in market share or consumer behavior.

*【结构性创新】趋势加速度（趋势的二阶导数）——市场份额或消费行为的基本性状态转换。*

> **Example:** A major competitor files for bankruptcy. Your target company's shipments jump 10% and stay there permanently. This is structural — a permanent regime shift.
>
> *【示例】主要竞争对手申请破产，目标公司货运量永久性上升10%。这是结构性的——永久性状态转换。*

### 5.2 Removing Autocorrelation (AR Filtering)
### 5.2 去除自相关（AR滤波）

Even after removing seasonality, time-series data often has "momentum" (autocorrelation). Remove this to find the *true* surprise:

*即使去除季节性后，时间序列数据通常仍有"动量"（自相关）。去除它以找到**真正的**意外：*

$$\epsilon_t = Y_t - (\phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p})$$

**Choosing order $p$:** Use the Partial Autocorrelation Function (PACF) or minimize the Akaike Information Criterion (AIC). The remaining $\epsilon_t$ is pure alpha.

*【选择阶数 $p$】使用偏自相关函数（PACF）或最小化AIC准则。剩余的 $\epsilon_t$ 是纯粹的Alpha。*

### 5.3 Kalman Filter (State-Space Innovation)
### 5.3 卡尔曼滤波器（状态空间创新量）

The Kalman Filter is the premier tool for real-time extraction of innovations. Unlike STL or moving averages (backward-looking), a Kalman Filter is a recursive algorithm that maintains an internal "state" and continuously updates beliefs as new data arrives.

*卡尔曼滤波器是实时提取创新量的顶级工具。与STL或移动平均（向后看）不同，卡尔曼滤波器是一种递归算法，维护系统的内部"状态"，并随新数据到达持续更新信念。*

**Core assumption:** The underlying "true" trend ($x_t$) is unobservable and evolves over time. What we observe ($y_t$) is this true trend corrupted by measurement noise.

*【核心假设】底层"真实"趋势（$x_t$）不可观测且随时间演变。我们观测到的（$y_t$）是被测量噪声污染的真实趋势。*

**State Equation (Hidden Reality):** $x_t = A \cdot x_{t-1} + w_t$ where $w_t \sim N(0, Q)$

**Measurement Equation (What We See):** $y_t = H \cdot x_t + v_t$ where $v_t \sim N(0, R)$

**The Innovation:** The Kalman Filter predicts tomorrow ($\hat{y}_t$). When actual data ($y_t$) arrives, the difference is the Innovation:

*【创新量】卡尔曼滤波器预测明天（$\hat{y}_t$）。当实际数据（$y_t$）到达时，差值即为创新量：*

$$\epsilon_t = y_t - \hat{y}_t$$

*If this innovation is consistently positive over several days, the filter automatically adjusts its hidden state upward, recognizing a structural shift.*

*若创新量连续多天为正，滤波器自动向上调整隐藏状态，识别出结构性转变。*

**Best for:** High-frequency, real-time environments (MFT/HFT) where you need to detect structural regime shifts *before* a traditional rolling average catches up.

*【最适合】需要在传统滚动平均反应之前检测结构性状态转变的高频实时环境（MFT/HFT）。*

> **Real-World Example:** Tracking hourly satellite data of parking lot cars at Walmart. Data is extremely noisy (clouds, sensor glitches). A Kalman Filter maintains a "true expected volume." A massive 30% spike drives the filter's prediction error instantly, generating a tradable signal on day one, whereas a 14-day rolling average would barely move.
>
> *【真实案例】追踪沃尔玛停车场的每小时卫星数据。数据极度嘈杂（云层、传感器故障）。卡尔曼滤波器维护"真实预期量"。大型促销活动带来30%的停车量飙升，滤波器的预测误差立即反应，在第一天就产生可交易信号，而14天滚动平均几乎不动。*

```python
from pykalman import KalmanFilter
import numpy as np
import pandas as pd

def extract_kalman_innovation(series):
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01  # lower = smoother trend
    )
    state_means, _ = kf.filter(series.values)
    innovations = series.values - state_means.flatten()
    return pd.Series(innovations, index=series.index)
```

---

## 6. Z-Score Transformations: Three Scenarios
## 6. Z分数变换：三种场景

### 6.1 Scenario 1: Time-Series Z-Score
### 6.1 场景一：时间序列 Z 分数

**Setup:** You have 5 years of transaction date history up to `query_date`. Compute the signal at each point, z-score across the time dimension, keep the most recent observation.

*【设置】截至 `query_date`，你有5年的交易历史。对每个时间点计算信号，在时间维度上做Z标准化，取最新观测值。*

```
ts_zscore(t) = (signal(t) - mean(signal[t-N:t])) / std(signal[t-N:t])
```

**Intuition:** "Is this company's signal at an unusually high or low level *relative to its own history*?" Removes entity-level absolute scale differences.

*【直觉】"相对于该公司自身历史，当前信号是否处于异常水平？"消除主体间绝对水平差异。*

**Pros:** Removes cross-sectional heterogeneity; stable even if universe changes; useful for mean-reverting signals.

*【优点】去除截面异质性；即使投资域变化也保持稳定；适用于均值回归信号。*

**Cons:** Requires deep per-entity history; fixed window may include stale regimes; no cross-sectional information; sensitive to structural breaks.

*【缺点】需要每个主体的深度历史；固定窗口可能包含过时状态；无截面信息；对结构性变化敏感。*

---

### 6.2 Scenario 2: Cross-Sectional Z-Score
### 6.2 场景二：截面 Z 分数

**Setup:** As of `query_date`, z-score the signal across all entities in the universe at that single point in time.

*【设置】在 `query_date` 时点，对投资域内所有主体的信号做截面Z标准化。*

```
xs_zscore(i, t) = (signal(i, t) - mean_j(signal(j, t))) / std_j(signal(j, t))
```

**Intuition:** "How does this entity compare to its peers *right now*?" Natural framing for relative value long-short strategies.

*【直觉】"这家公司此刻与同行相比如何？"相对价值多空策略的自然框架。*

**Pros:** Directly captures relative competitive positioning; naturally market-neutral; removes macro common factors; directly usable in long-short construction.

*【优点】直接捕捉相对竞争地位；天然市场中性；去除宏观共同因子；可直接用于多空组合构建。*

**Cons:** Sensitive to universe composition; outliers compress others — prefer robust z-score (median/MAD); no historical context; sector composition matters.

*【缺点】对投资域构成敏感；异常值压缩他人——优先使用稳健Z分数（中位数/MAD）；无历史背景；行业构成影响显著。*

**Practical note:** Often best applied *within* sector/industry groups, then recombined.

*【实践注意】通常最好在行业/板块内部分别计算，再合并。*

---

### 6.3 Scenario 3: Z-Score Across Query Dates
### 6.3 场景三：跨查询日期的 Z 分数

**Setup:** Z-score across the `query_date` observation timestamps in your backtest or panel.

*【设置】在回测或面板数据的 query_date 时间戳维度上做Z标准化。*

```
qd_zscore(i, t) = (signal(i, t) - mean_tau(signal(i, tau))) / std_tau(signal(i, tau))
```

**Intuition:** "Given all moments in my backtest at which I observed this entity, is today's reading extreme?" Standardizes across your investment calendar to ensure signal stationarity across regimes.

*【直觉】"在我回测期间所有观测时刻中，今天的读数是否极端？"在投资日历上做标准化，确保信号跨市场状态的平稳性。*

**Pros:** Ensures signal stationarity; useful for combining signals across regimes; implementable with expanding window.

*【优点】确保信号平稳性；便于跨市场状态合并信号；可用扩展窗口实施。*

**Cons:** Lookahead bias if using full-sample fixed window (only valid for research, not live use); unreliable with short panel; distribution shifts with universe changes.

*【缺点】使用全样本固定窗口有前视偏差（仅研究有效，不可实盘使用）；面板较短时不可靠；投资域变化时分布漂移。*

---

### 6.4 Robust Z-Score (MAD-Based)
### 6.4 稳健 Z 分数（基于MAD）

Standard z-scores are broken by massive outliers. Use Median Absolute Deviation for robustness:

*标准Z分数被大异常值破坏。使用中位数绝对偏差（MAD）提高鲁棒性：*

$$Z_{modified} = \frac{0.6745 \times (\epsilon_t - \text{Median})}{\text{MAD}}$$

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

**When to use:** Any alt data series with known fat tails or intermittent spikes (credit card spend, satellite data, web traffic).

*【何时使用】任何具有已知厚尾或间歇性尖峰的另类数据序列（信用卡消费、卫星数据、网络流量）。*

---

## 7. Comparing the Three Z-Score Scenarios
## 7. 三种 Z 分数场景对比

| Dimension (维度) | TS Z-Score / S1 (时间序列Z) | Cross-Sectional Z-Score / S2 (截面Z) | Query-Date Z-Score / S3 (跨查询日期Z) |
|---|---|---|---|
| **Axis (标准化轴)** | Entity's own time series (主体自身时间序列) | Across entities at a point in time (某时间点上的截面) | Across investment timestamps (主体的投资日历时间轴) |
| **Question (回答问题)** | Extreme vs. own past? (相对自身历史极端吗？) | Extreme vs. peers today? (相对同期同行极端吗？) | Extreme vs. all prior snapshots? (相对历史快照极端吗？) |
| **Market neutral (市场中性)** | No (否) | Yes — by construction (是，构造上天然) | No (否) |
| **Lookahead bias (前视偏差)** | Window choice only (仅来自窗口选择) | None (无) | Yes if full-sample; use expanding window (全样本有；用扩展窗口规避) |
| **Best for (适用场景)** | Mean-reversion, fundamental anchoring (均值回归、基本面锚定) | Long-short, relative value (多空、相对价值) | Signal stationarity across regimes (跨市场状态的信号平稳性) |

---

## 8. Combining Transformations (Double Z-Score)
## 8. 组合变换（双重 Z 标准化）

A common production pattern:

*一种常见的生产级处理范式：*

1. Compute signal (e.g., YoY growth)  →  *计算信号（如同比增长率）*
2. Apply **time-series z-score** (remove entity-level absolute differences)  →  *做**时间序列Z标准化**（消除主体间绝对水平差异）*
3. Apply **cross-sectional z-score** (make the result comparable across the universe)  →  *做**截面Z标准化**（使结果在投资域内可比）*

Step 2 handles entity heterogeneity; Step 3 makes the result directly usable in portfolio construction. This double-standardization is standard in systematic quant factors.

*第2步处理主体异质性；第3步使结果可直接用于组合构建。双重标准化在系统化量化因子中是行业标准。*

---

## 9. Level Features vs. Growth Features
## 9. 水平类特征 vs. 增长类特征

### Why Levels Matter
### 为什么水平值很重要

Converting to growth removes size effects and seasonality, but **discards absolute scale information** that can be independently predictive.

*转换为增长率去除了规模效应和季节性，但**丢弃了绝对规模信息**，而后者本身可以具有预测力。*

**Example — Crowding:** Large-cap companies naturally have higher absolute crowding. YoY-differencing crowding tells you direction but loses the signal that a company is *structurally* heavily crowded vs. structurally under-owned. The level of crowding (how much institutional concentration exists now) carries information the growth rate does not.

*【示例——拥挤度】大盘股天然有更高的绝对拥挤度。同比差分拥挤度只能捕捉方向，却丢失了公司**结构性**高度拥挤还是结构性持有不足的信号。*

Other cases where level dominates: market share / competitive dominance; network effects data; balance sheet-like metrics (cash, debt).

*水平值更重要的其他场景：市场份额/竞争主导地位；网络效应数据；类资产负债表指标（现金、债务）。*

**The challenge:** Raw levels are not cross-sectionally comparable. The goal is to transform levels into something bounded, scale-invariant, and economically meaningful.

*【挑战】原始水平值在截面上不可比。目标是将水平值转换为有界、规模不变且经济上有意义的形式。*

### Transformations to Preserve Level Information
### 保留水平信息的变换方法

**9.1 Market Share** — `metric(i,t) / sum_j(metric(j,t))` — Bounded [0,1]. Preserves competitive position meaningfully. Macro-robust: if all companies grow equally, market share stays constant.

*【9.1 市场份额】有界[0,1]。以有意义的方式保留竞争地位。宏观鲁棒：所有公司等比例增长时，市场份额不变。*

*Cons: Denominator is dataset, not real economy; dominated by very large companies; requires stable universe.*

*【缺点】分母是数据集而非真实经济；被超大市值公司主导；需要稳定的投资域。*

**9.2 Log Transformation** — `log(metric(i,t))` — Not bounded, but converts right-skewed distribution to symmetric. Makes YoY growth a linear difference. Standard in finance (log-returns).

*【9.2 对数变换】无界，但将右偏分布转换为对称分布。使同比增长变为线性差分。金融标准假设（对数收益率）。*

*Cons: Undefined for zeros (use `log(1+x)`); still requires z-scoring for model use.*

*【缺点】零值无定义（使用 `log(1+x)`）；模型使用仍需Z标准化。*

**9.3 Percentile Rank** — Non-parametric, bounded [0,1]. Robust to any distributional shape and fat tails.

*【9.3 百分位排名】非参数，有界[0,1]。对任何分布形状和厚尾均鲁棒。*

*Cons: Loses magnitude information at extremes; depends on universe composition.*

*【缺点】丢失极端值幅度信息；依赖于投资域构成。*

**9.4 Float-Adjusted Ownership %** — `shares_held / shares_outstanding` — Bounded [0,1] by construction. Natural size normalization for ownership/crowding data.

*【9.4 流通股本调整持股比例】构造上有界[0,1]。持股/拥挤度数据的天然规模标准化。*

**9.5 Min-Max Scaling** — Bounded [0,1] but sensitive to outliers. **Not recommended** without winsorization first.

*【9.5 最小-最大归一化】有界[0,1]但对异常值敏感。未先做缩尾处理时**不推荐**。*

### Summary: When to Use Level vs. Growth
### 小结：何时使用水平 vs. 增长类特征

| Signal Type (信号类型) | Prefer Level (偏好水平) | Prefer Growth (偏好增长) |
|---|---|---|
| Competitive positioning (竞争定位) | ✓ Market share (市场份额) | |
| Crowding / ownership (拥挤度/持股) | ✓ Ownership % (持股比例) | |
| Business momentum (业务动量) | | ✓ YoY growth (同比增长) |
| Inflection detection (拐点检测) | | ✓ Acceleration (加速度) |
| Network / scale effects (网络/规模效应) | ✓ Log(level) (对数水平) | |
| Macro-correlated metrics (宏观相关指标) | | ✓ Growth removes macro (增长去除宏观) |

In practice: **generate both and let the model decide** (or use a combination factor).

*实践中：**两者都生成，让模型决定**（或构建组合因子）。*

---

## 10. Normalizing Alternative Data by External Denominators
## 10. 用外部分母对另类数据进行归一化

Different denominators answer different economic questions. The choice of normalizer encodes a hypothesis about what dimension of variation is signal vs. noise.

*不同的分母回答不同的经济问题。归一化器的选择隐含了一个假设：哪个维度的变化是信号，哪个是噪声。*

### 10.1 Divide by Total Dataset Sales → Market Share
### 10.1 除以数据集总销售额 → 市场份额

**Economic question:** "What fraction of total measured economic activity does this entity capture?"

*【经济问题】"该主体占测量到的总经济活动的比例是多少？"*

**Rationale:** Macro-neutral by construction. Signal reflects idiosyncratic competitive dynamics.

*【逻辑】构造上宏观中性。信号反映特质性竞争动态。*

**Cons:** Denominator is dataset, not real economy; dominated by large players; requires stable universe.

*【缺点】分母是数据集而非真实经济；被大市值公司主导；需要稳定、定义明确的数据域。*

---

### 10.2 Divide by Market Cap → Sales Intensity (P/S Proxy)
### 10.2 除以市值 → 销售强度（市销率代理）

**Economic question:** "How much revenue/activity does this company generate per dollar of market valuation?"

*【经济问题】"该公司每单位市值能产生多少另类数据销售额/活动？"*

**Rationale:** Creates an alt-data-based analog to Price/Sales. Connects to fundamental valuation.

*【逻辑】构建基于另类数据的市销率类比指标。与基本面估值挂钩。*

**Cons:** Market cap volatile — injects price noise; endogeneity risk; cross-sector comparability limited.

*【缺点】市值波动大——注入价格噪声；内生性风险；跨行业可比性有限。*

---

### 10.3 Divide by ADV → Liquidity-Adjusted Signal
### 10.3 除以日均成交额 → 流动性调整信号

**Economic question:** "How strong is this signal per unit of capacity I can deploy?"

*【经济问题】"每单位可部署容量对应的信号强度是多少？"*

**Rationale:** Position size is constrained by liquidity. Normalizing by ADV puts signals on a "per tradeable dollar" basis.

*【逻辑】仓位大小受流动性约束。除以ADV将信号转换为"每可交易美元"单位。*

**Cons:** ADV itself is noisy; can create circular dependency with price momentum; less economically intuitive.

*【缺点】ADV本身有噪声；可能与价格动量产生循环依赖；经济直觉弱。*

---

### 10.4 Divide by Reported Revenue → Coverage / Penetration Ratio
### 10.4 除以已披露营收 → 覆盖率/渗透率

**Economic question:** "What fraction of the company's actual revenue does this dataset capture?"

*【经济问题】"该数据集捕捉到了该公司实际营收的多大比例？"*

**Rationale:** Measures data quality per company. Not primarily a signal — useful for filtering and reliability scoring.

*【逻辑】衡量每家公司的数据质量。主要不是信号，而是用于过滤和可靠性评分。*

**Cons:** Reported revenue is lagged (quarterly); requires fundamental data infrastructure; coverage ratio may drift.

*【缺点】已披露营收有滞后（季报）；需要基本面数据基础设施；覆盖率可能漂移。*

---

### 10.5 Divide by Sector Total → Sector Market Share
### 10.5 除以行业总量 → 行业市场份额

**Economic question:** "What is this company's competitive position within its own industry?"

*【经济问题】"该公司在其所在行业内的竞争地位如何？"*

**Rationale:** More refined than universe-wide market share. A 5% share in retail is very different from 5% in cloud infrastructure.

*【逻辑】比全域市场份额更精细。零售业5%的份额与云基础设施5%意义截然不同。*

**Cons:** Requires well-defined sector classification; small sectors produce unreliable estimates.

*【缺点】需要定义明确的行业分类；公司数量少的小行业产生不可靠的市场份额。*

---

### Normalizer Comparison Summary
### 归一化器对比总结

| Normalizer (归一化器) | Removes (去除) | Retains (保留) | Best Use Case (最佳场景) |
|---|---|---|---|
| Total dataset (全数据集) | Macro, size (宏观、规模) | Competitive position (竞争地位) | Broad competitive signal (广泛竞争信号) |
| Sector total (行业总量) | Macro, sector, size (宏观、行业、规模) | Intra-sector competition (行业内竞争) | Cross-sector portfolios (跨行业组合) |
| Market cap (市值) | Size (规模) | Valuation linkage (估值关联) | Fundamental quant (基本面量化) |
| ADV (日均成交额) | Size (规模) | Tradeable capacity (可交易容量) | Capacity-constrained strategies (容量受限策略) |
| Reported revenue (已披露营收) | Size (规模) | Data quality signal (数据质量信号) | Signal reliability scoring (信号可靠性评分) |

---

## 11. Rolling Average vs. EWMA
## 11. 简单滚动平均 vs. 指数加权移动平均（EWMA）

The core argument for EWMA: **recent data is more relevant than old data**. SMA treats all observations in the window equally; EWMA does not.

*EWMA的核心论点：**近期数据比历史数据更相关**。SMA对窗口内所有观测等权；EWMA则不然。*

| Dimension (维度) | SMA (简单滚动平均) | EWMA (指数加权移动平均) |
|---|---|---|
| **Weighting (权重)** | Uniform (均匀) | Exponential decay (指数衰减) |
| **"Ghost" effect ("幽灵效应")** | Yes — spike drops out abruptly (有——尖峰突然脱离) | No — old data fades smoothly (无——旧数据平滑淡出) |
| **Lag (滞后)** | Higher (较高) | Lower (较低) |
| **Stability (稳定性)** | More stable (更稳定) | More reactive (更灵敏) |
| **Parameter (参数)** | Window length, integer (窗口长度，整数) | Halflife, continuous (半衰期，连续) |

**The ghost effect** is the most important practical objection to SMA: a one-time spike 12 months ago counts equally today, and when it falls off the window, you get a sharp downward move in your SMA with no corresponding change in the underlying data.

*【幽灵效应】是反对SMA最重要的实践理由：12个月前的一次性尖峰今天仍被等权计算，当它脱离窗口时，SMA急剧下降，而底层数据没有任何对应变化。*

### Choosing the Halflife
### 如何选择半衰期

Halflife $h$ means weight decays to 50% after $h$ periods. Decay factor: $\lambda = 2^{-1/h}$.

*半衰期 $h$ 意味着权重在 $h$ 期后衰减到50%。衰减因子：$\lambda = 2^{-1/h}$。*

| Halflife — monthly (半衰期 — 月度) | Effective memory (有效记忆) | When appropriate (适用场景) |
|---|---|---|
| 1–2 months (1–2个月) | Very short (极短) | High-frequency alt data, e.g., weekly credit card spend (高频另类数据，如每周信用卡消费) |
| 3–6 months (3–6个月) | Medium (适中) | Most alt data signals, earnings revision momentum (大多数另类数据信号、盈利预期修正动量) |
| 9–12 months (9–12个月) | Long (较长) | Fundamental/value-oriented signals (基本面/价值导向信号) |
| 12–24 months (12–24个月) | Very long (很长) | Structural trend detection (结构性趋势检测) |

**How to choose:**

*【选择方法：】*

1. **Information decay analysis** — regress future returns on lagged signal values at different lags; halflife is where the coefficient drops to half its peak. *【信息衰减分析】将未来收益对不同滞后期信号值做回归；系数降至峰值一半处即为半衰期。*
2. **Backtest grid search** — pick best out-of-sample IR; use held-out test period to avoid overfitting. *【回测网格搜索】选择样本外IR最优者；使用保留测试期避免过拟合。*
3. **Domain intuition** — match to data's natural refresh rate and your return horizon. *【领域直觉】与数据的自然刷新率和收益预测周期匹配。*
4. **Rule of thumb** — halflife ≈ 1/3 to 1/2 of signal horizon. *【经验法则】半衰期 ≈ 信号预测周期的1/3到1/2。*

---

## 12. Full Alpha Signal Generation Pipeline
## 12. 完整 Alpha 信号生成流程

Beyond the individual transformations, a production-grade process follows a structured pipeline. Each step exists for a specific reason and order matters.

*除单个变换外，生产级流程遵循结构化管线。每个步骤都有其特定原因，且顺序至关重要。*

### Step 1: Vendor Due Diligence & Data Acquisition
### 步骤一：供应商尽职调查与数据获取

(See Section 0) — PiT timestamps, PII compliance, survivorship bias, restatement handling.

*（见第0节）——时点时间戳、PII合规、幸存者偏差、数据修订处理。*

### Step 2: Data QC, Panel Stability & Reweighting
### 步骤二：数据质控、面板稳定性与重新加权

(See Section 1) — Cohort filtering, demographic reweighting (check ESS!), missing data handling, stale data detection, coverage filtering.

*（见第1节）——队列过滤、人口统计重新加权（检查ESS！）、缺失值处理、过时数据检测、覆盖率过滤。*

### Step 3: Seasonality Removal
### 步骤三：季节性去除

(See Section 4) — Choose YoY, STL, X-13ARIMA, or Prophet based on data frequency and history length. Apply base-effect checks for YoY.

*（见第4节）——根据数据频率和历史长度选择YoY、STL、X-13ARIMA或Prophet。对YoY应用基期效应检查。*

### Step 4: Feature Construction
### 步骤四：特征构建

Growth, level, acceleration, benchmark difference, vol-adjusted, trend deviation, innovation extraction (AR filter or Kalman filter).

*增长率、水平、加速度、基准差异、波动率调整、趋势偏差、创新量提取（AR滤波或卡尔曼滤波器）。*

### Step 5: Normalization by External Denominator
### 步骤五：外部分母归一化

(See Section 10) — Market share, market cap, ADV, etc. Choose based on the economic question you want the signal to answer.

*（见第10节）——市场份额、市值、ADV等。根据信号需要回答的经济问题选择。*

### Step 6: Winsorization
### 步骤六：缩尾处理

**Always winsorize before z-scoring.** Clip at [1%, 99%] cross-sectionally at each query_date.

*【在Z标准化之前必须做缩尾处理。】在每个 query_date 截面上，裁剪至[1%, 99%]。*

Clipping preserves the observation as "extreme but real" while bounding its influence — unlike removal, which loses the observation entirely.

*裁剪将观测保留为"极端但真实"，同时限制其影响——与删除不同，删除完全丢失观测。*

### Step 7: Z-Score (Double Z-Score)
### 步骤七：Z标准化（双重Z标准化）

Time-series z-score (entity heterogeneity) → cross-sectional z-score (portfolio-ready). Or use robust z-score (MAD-based) for fat-tailed distributions.

*时间序列Z标准化（消除主体异质性）→截面Z标准化（组合就绪）。或对厚尾分布使用稳健Z分数（基于MAD）。*

### Step 8: Factor Neutralization (Orthogonalization)
### 步骤八：因子中性化（正交化）

Remove exposure to known systematic factors via cross-sectional regression:

*通过截面回归去除对已知系统性因子的暴露：*

```
signal_neutralized(i, t) = residual from regressing signal(i, t) on [factor_exposures(i, t)]
```

Common factors to neutralize: sector/industry, size (log market cap), momentum (12-1m return), value (B/P), volatility.

*常见中性化因子：行业/板块、规模（对数市值）、动量（12-1月收益）、价值（账面市值比）、波动率。*

*Intuition: If your alt data signal is correlated with market cap, the raw signal is just a size factor in disguise. After neutralizing, the residual captures something genuinely beyond size.*

*【直觉】若另类数据信号与市值相关，原始信号实质上是规模因子的变装。中性化后，残差捕捉的是超越规模的额外信息。*

### Step 9: Signal Decay and IC Analysis
### 步骤九：信号衰减与 IC 分析

**Information Coefficient (IC):** `spearman_corr(signal(t), return(t → t+h))`

**IC decay curve:** Plot IC(h) for h = 1 week, 2 weeks, 1 month, 3 months... Reveals:
- Peak IC horizon (optimal holding period)
- IC half-life (guides EWMA halflife choice)
- Sign flips (mean-reverting signal)

*【IC衰减曲线】绘制不同 h 下的IC(h)，揭示：IC峰值周期（最优持仓周期）；IC半衰期（指导EWMA半衰期选择）；符号翻转（均值回归信号）。*

**ICIR:** `mean(IC) / std(IC)` — ICIR > 0.5 is usable; > 1.0 is strong.

*【ICIR】ICIR > 0.5 通常可用；> 1.0 则很强。*

### Step 10: Turnover Control and Signal Smoothing
### 步骤十：换手率控制与信号平滑

Apply EWMA to the **final factor score** (after all transformations) to reduce noise and control turnover:

*对最终因子得分（所有变换后）应用EWMA以降低噪声和控制换手率：*

```
signal_smooth(t) = alpha * signal(t) + (1 - alpha) * signal_smooth(t-1)
```

*Rule of thumb: turnover ≤ 2× the IC peak horizon's frequency.*

*经验法则：换手率 ≤ IC峰值周期频率的2倍。*

### Step 11: Signal Combination and Interaction Features
### 步骤十一：信号组合与交互特征

| Combination Type (组合类型) | Intuition (直觉) |
|---|---|
| Growth × Market share (增长率 × 市场份额) | Fast-growing dominant companies — winner-take-all convexity (快速增长且占主导地位的公司——赢者通吃的凸性) |
| IC-weighted composite (IC加权合成) | Adaptively weight by recent predictive power (按近期预测力自适应加权) |
| Surprise × Price momentum (超预期 × 价格动量) | Double confirmation — data and price agree (双重确认——数据和价格一致) |
| PCA decomposition (PCA分解) | Extract orthogonal components from correlated alt datasets (从相关另类数据集中提取正交成分) |

### Pipeline Order Summary
### 管线顺序总结

```
Raw alt data → Vendor QC (PiT, PII, survivorship)
原始另类数据 → 供应商质检（时点、PII、幸存者偏差）
    ↓
Panel stability: cohort filter, reweighting, ESS check
面板稳定性：队列过滤、重新加权、ESS检查
    ↓
Seasonality removal (YoY / STL / X-13ARIMA / Prophet)
季节性去除（YoY / STL / X-13ARIMA / Prophet）
    ↓
Feature construction (growth, level, acceleration, innovation)
特征构建（增长率、水平、加速度、创新量）
    ↓
External denominator normalization (market share, market cap, ADV...)
外部分母归一化（市场份额、市值、ADV…）
    ↓
Winsorization [1%, 99%] — BEFORE z-scoring
缩尾处理 [1%, 99%] — 在Z标准化之前
    ↓
Double Z-score (TS → XS) or robust z-score (MAD)
双重Z标准化（时序→截面）或稳健Z分数（MAD）
    ↓
Factor neutralization (sector, size, momentum...)
因子中性化（行业、规模、动量…）
    ↓
Signal smoothing (EWMA on factor score)
信号平滑（对因子得分做EWMA）
    ↓
IC / ICIR analysis & decay curve validation
IC/ICIR分析与衰减曲线验证
    ↓
Signal combination / interaction features / PCA
信号组合/交互特征/PCA
    ↓
Alpha signal → Portfolio construction
Alpha信号 → 组合构建
```

**Why order matters:**
- Winsorize *before* z-scoring, or one outlier controls the mean and std
- Neutralize *after* z-scoring, so the regression is well-conditioned
- Smooth *after* neutralization, so you're smoothing the final clean signal

*【为什么顺序重要】在Z标准化**之前**缩尾；在Z标准化**之后**中性化；在中性化**之后**平滑——确保平滑的是最终干净的信号。*

---

## 13. Summary Decision Framework
## 13. 总结：决策框架

```
Signal type               → Transformation priority
信号类型                    → 变换优先级

Trending metric           → Growth YoY, then acceleration, then XS z-score
趋势型指标                  → 同比增长，再做加速度，再做截面Z分数

Mean-reverting            → TS z-score, then deviation from benchmark
均值回归型                  → 时间序列Z分数，再做相对基准偏差

High outlier risk         → Percentile rank or robust z-score (MAD)
高异常值风险                → 百分位排名或稳健Z分数（MAD）

Cross-entity compare      → Cross-sectional z-score (within sector)
跨主体比较                  → 截面Z分数（行业内）

Regime-stable signal      → Query-date z-score (expanding window)
跨市场状态平稳信号           → 查询日期Z分数（扩展窗口）

Noisy/volatile data       → Vol-adjusted signal, then z-score
噪声/高波动数据              → 波动率调整信号，再做Z分数

Competitive positioning   → Market share (level feature)
竞争定位信号                → 市场份额（水平类特征）

Crowding / ownership      → Float-adjusted ownership % (level feature)
拥挤度/持股信号              → 流通股本调整持股比例（水平类特征）

Real-time regime shift    → Kalman Filter innovation extraction
实时状态转换检测             → 卡尔曼滤波器创新量提取

Messy high-frequency data → Prophet seasonality adjustment
杂乱高频数据                → Prophet季节性调整

Calendar-precise monthly  → X-13ARIMA-SEATS
高精度月度日历调整            → X-13ARIMA-SEATS
```
