# Alternative Data Preparation and Evaluation
# 另类数据的准备与评估

This document outlines key considerations for evaluating, preparing, and handling alternative datasets in a quantitative research environment.
本文概述了在量化研究环境中评估、准备和处理另类数据集的关键注意事项。

---

## I. Vendor Evaluation Questions: The Due Diligence Phase
## 一、 供应商评估问题：尽职调查阶段

Before touching the data, you must evaluate the structural integrity and legal safety of the dataset.
在接触数据之前，必须评估数据集的结构完整性和法律安全性。

### 1. Data Integrity and Lineage
### 1. 数据完整性与溯源
*   **What is the primary source of the data?** Is it direct-from-source (e.g., credit card exhaust from a bank), web-scraped, or aggregated from third-party apps? Aggregated data is highly vulnerable to "inorganic" panel shocks if the vendor loses a sub-supplier.
    *   **数据的核心来源是什么？** 是直接源数据（例如银行的信用卡流水）、网页抓取，还是从第三方 App 汇总的？如果供应商失去了一个子供应商，汇总型数据非常容易受到“非自然”样本池冲击的影响。
    *   *💡 Real-World Example:* A vendor aggregating credit card data from multiple banks suddenly loses a contract with Chase. If you don't know the lineage, your model will mistakenly interpret the massive drop in transaction volume as a severe macroeconomic recession.
    *   *💡 现实案例：* 一个汇总多家银行信用卡数据的供应商突然失去了与大通银行（Chase）的合同。如果你不知道数据溯源，你的模型会错误地将交易量的巨大下降解释为严重的宏观经济衰退。
*   **Compliance and PII (Personally Identifiable Information):** Does the dataset contain toxic PII? Has it been properly anonymized? If it’s web-scraped, does it violate the target site's Terms of Service? Trading on non-compliant data introduces massive legal and reputational risk.
    *   **合规性与 PII（个人身份信息）：** 数据集是否包含敏感的 PII？是否经过了妥善的去标识化处理？如果是网页抓取，是否违反了目标网站的服务条款？基于不合规数据进行交易会带来巨大的法律和声誉风险。
*   **Restatements and Revisions:** If the vendor changes a data point post-facto (e.g., a merchant reclassifies a transaction 30 days later), how is that flagged? You must ensure your backtest only uses data that was *actually known* on that historical date.
    *   **重述与修订：** 如果供应商事后更改了某个数据点（例如，商户在 30 天后对某笔交易进行了重新分类），该如何标记？你必须确保你的回测只使用在该历史日期*实际上已知*的数据。
    *   *💡 Real-World Example:* A vendor restates an entire quarter of retail sales because a major retailer changed how they report returns. If your backtest uses the restated data, you have "Look-Ahead Bias," trading on information that was impossible to know at the time.
    *   *💡 现实案例：* 由于一家主要零售商更改了退货报告方式，供应商重述了整个季度的零售额。如果你的回测使用了重述后的数据，你就产生了“未来函数（Look-Ahead Bias）”，即利用了当时不可能知道的信息进行交易。
*   **Point-in-Time (PiT) vs. Event-Time:** Does the timestamp reflect when the event occurred (Event-Time) or when the data was actually published to a subscriber's FTP/API (PiT)? Using Event-Time in a backtest creates fatal **Look-Ahead Bias**.
    *   **时间点 (PiT) vs. 事件时间：** 时间戳反映的是事件发生的时间（事件时间），还是数据实际发布到订阅者 FTP/API 的时间（PiT）？在回测中使用事件时间会导致致命的**未来函数偏差**。

### 2. Methodology and Pre-processing
### 2. 方法论与预处理
*   **Pre-Smoothing:** Are they applying outlier detection or normalization before you get the data? You need to know if they are clipping "noise" that might actually be a valid macroeconomic signal.
    *   **预平滑处理：** 在你拿到数据之前，供应商是否应用了离群值检测或归一化？你需要知道他们是否剪掉了可能实际上是有效宏观信号的“噪音”。
*   **Survivorship Bias:** Does the dataset include companies that have gone bankrupt or been acquired, or is it only currently active tickers? If bankrupt companies are dropped, historical performance will look artificially inflated.
    *   **生存者偏差：** 数据集是否包含已破产或被收购的公司，还是仅包含当前活跃的股票？如果剔除了破产公司，历史表现看起来会虚高。
    *   *💡 Real-World Example:* If you are testing a signal on App Store downloads and the vendor deletes historical data for apps that were banned last year, your model will falsely conclude that every app in the historical dataset was a long-term winner.
    *   *💡 现实案例：* 如果你在测试 App Store 下载量信号，而供应商删除了去年被封禁的 App 的历史数据，你的模型会错误地得出结论，认为历史数据集中的每个 App 都是长期的赢家。
*   **Mapping and Corporate Actions:** How do they map raw data to tradable entities (e.g., mapping a specific subsidiary's brand name to a parent company's FIGI/Ticker)? Furthermore, do they automatically adjust historical data for M&A, spin-offs, or ticker changes?
    *   **映射与公司行为：** 他们如何将原始数据映射到可交易实体（例如，将特定子公司的品牌名称映射到母公司的 FIGI/股票代码）？此外，他们是否会自动根据并购（M&A）、拆分或更名调整历史数据？

---

## II. Data Preparation Techniques: Cleaning the Signal
## 二、 数据准备技术：清洗信号

### 1. Handling Panel Instability
### 1. 处理样本池（Panel）的不稳定性
Alternative data panels rarely represent a stable population. Users churn, and vendors sign new partnerships.
另类数据的样本池很少能代表一个稳定的群体。用户会流失，供应商也会签署新的合作伙伴。

*   **Cohort-Based Filtering:** Create a "fixed cohort" of users active for a minimum continuous period. This removes the noise of users onboarding or abandoning the platform.
    *   **基于同类群组（Cohort）的过滤：** 创建一个在最短连续时间内保持活跃的“固定群组”。这可以消除用户加入或退出平台带来的噪音。
    *   **Formula:** $C = \{i : \text{active}(i, \tau) \in \text{Panel}, \forall \tau \in [t-k, t]\}$
    *   *💡 Real-World Example:* Instead of looking at "Total Amazon Purchases" (which naturally grows as the vendor adds more users to the panel), you look only at the subset of 50,000 users who have been in the panel for exactly 24 continuous months, tracking whether *their* spending behavior is changing.
    *   *💡 现实案例：* 与其观察“亚马逊总购买量”（这会随着供应商向样本池添加更多用户而自然增长），不如只观察连续 24 个月都在样本池中的 50,000 名用户的子集，跟踪*他们的*消费行为是否在发生变化。
*   **Panel Normalization (Market-Share Method):** Divide the metric of interest by a "control" metric to get a ratio that is independent of panel size fluctuations.
    *   **样本池归一化（市场份额法）：** 将关注的指标除以一个“对照”指标，得到一个独立于样本池规模波动的比率。
    *   **Formula:** $M_{i,t} = \frac{X_{i,t}}{\sum_{j \in \text{Panel}} X_{j,t}}$
    *   *💡 Real-World Example:* If the vendor's total panel size drops by 20%, both Uber and Lyft recorded rides will instantly drop. But if you track the ratio `Uber Rides / (Uber Rides + Lyft Rides)`, this "market share" metric remains stable despite the panel volatility.
    *   *💡 现实案例：* 如果供应商的总样本池规模下降了 20%，Uber 和 Lyft 记录的行程都会立即下降。但如果你追踪 `Uber 行程 / (Uber 行程 + Lyft 行程)` 这一比例，尽管样本池波动，这个“市场份额”指标仍会保持稳定。

### 2. Removing Geographical and Demographic Bias
### 2. 消除地理和人口统计偏差
*   **Reweighting via Ground Truth:** Compare your panel’s distribution to a ground-truth source (like the U.S. Census or BLS data) and apply corrective weights.
    *   **通过基准事实（Ground Truth）重新加权：** 将你的样本池分布与基准事实源（如美国人口普查或劳工统计局数据）进行比较，并应用修正权重。
    *   **Formula:** $w_g = \frac{P(\text{Group}_g | \text{Census})}{P(\text{Group}_g | \text{Panel})}$
    *   *💡 Real-World Example:* If your mobile location panel is 40% from New York and California (but they are only ~20% of the actual US population), you must downweight those coastal pings to accurately forecast a national retailer like Walmart.
    *   *💡 现实案例：* 如果你的移动定位样本池中有 40% 来自纽约和加利福尼亚（但他们仅占美国实际人口的约 20%），你必须降低这些沿海地区的权重，才能准确预测像沃尔玛这样的全国性零售商。
*   **Regional De-averaging:** Build separate regional sub-signals and aggregate them based on actual economic weighting, rather than letting coastal urban centers dominate the signal.
    *   **区域去平均化：** 构建独立的区域子信号，并根据实际经济权重进行汇总，而不是让沿海城市中心主导信号。
    *   **Formula:** $\hat{S}_{\text{aggregate}} = \sum_{r} \beta_r \cdot \text{Normalize}(S_r)$
*   **Propensity Score Matching (PSM):** Run a logistic regression to calculate the probability ($e$) that a person is in your dataset given their traits ($X$). Use this to select a subset of users that perfectly mirrors the national average.
    *   **倾向评分匹配 (PSM)：** 运行逻辑回归以计算在给定特征 ($X$) 的情况下，一个人出现在你的数据集中的概率 ($e$)。利用这一概率选择一个能完美反映全国平均水平的用户子集。
    *   **Formula:** $e(X) = P(D=1 | X)$

---

## III. Risks of Reweighting Alternative Datasets
## 三、 另类数据重新加权的风险

Reweighting is a dangerous mathematical hammer. If pushed too hard, you destroy alpha.
重新加权是一把危险的数学之锤。如果用力过猛，你会摧毁 Alpha。

### 1. Variance Inflation and Effective Sample Size (ESS)
### 1. 方差膨胀与有效样本量 (ESS)
*   **The Math:** $\text{Var}(\hat{Y}) = \sum w_i^2 \text{Var}(y_i)$. Large weights $w_i$ (applied to severely under-represented groups) exponentially increase the estimator's variance.
    *   **数学原理：** $\text{Var}(\hat{Y}) = \sum w_i^2 \text{Var}(y_i)$。较大的权重 $w_i$（应用于代表性严重不足的群体）会呈指数级增加估计量的方差。
*   **The Diagnostic (ESS):** You must calculate the Effective Sample Size to know if you've destroyed your data. 
    *   **诊断 (ESS)：** 你必须计算有效样本量，以确定你是否破坏了数据。
    *   **Formula:** $\text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}$
    *   *Rule of Thumb:* If your ESS drops below 30% of your actual panel size after reweighting, your weights are too extreme. Your signal is now driven by a tiny handful of heavily-weighted users.
    *   *经验法则：* 如果重新加权后你的 ESS 降至实际样本池规模的 30% 以下，说明你的权重过于极端。你的信号现在仅由极少数权重极高的用户驱动。
    *   *💡 Real-World Example (The "Small Cell" Problem):* Your panel has only 5 people from Rural North Dakota, but they need a $50x$ weight to match the Census. If one of those 5 people buys a new tractor on their credit card, your model, multiplying that by 50, might falsely predict a massive national surge in John Deere sales.
    *   *💡 现实案例（“小单元”问题）：* 你的样本池中只有 5 人来自北达科他州农村，但他们需要 50 倍的权重才能匹配人口普查数据。如果这 5 人中有一个人用信用卡买了一台新拖拉机，你的模型将其乘以 50，可能会错误地预测约翰迪尔（John Deere）在全国范围内的销量将出现巨大增长。

### 2. Dimension Interdependency (The "Whack-a-Mole" Effect)
### 2. 维度间依赖（“打地鼠”效应）
*   **The Conflict:** Reweighting for Geography may inadvertently worsen Income bias, because those variables are highly correlated.
    *   **冲突：** 为地理因素重新加权可能会无意中加剧收入偏差，因为这些变量高度相关。
    *   *💡 Real-World Example:* You reweight your panel to have more 18-24 year olds to match the Census. But in your specific dataset, the 18-24 year olds happen to be wealthy software engineers. By "fixing" the Age bias, you accidentally created a massive Income bias.
    *   *💡 现实案例：* 你为了匹配人口普查数据而重新加权样本池以增加 18-24 岁的人数。但在你的特定数据集中，18-24 岁的人恰好是富裕的软件工程师。通过“修复”年龄偏差，你意外地造成了巨大的收入偏差。
*   **The Solution (Raking):** Quants use **Iterative Proportional Fitting (Raking)** to iteratively adjust weights until multiple marginal distributions (e.g., Age, Income, Geo) simultaneously match the Census. However, this dramatically increases the risk of over-fitting.
    *   **解决方案 (Raking)：** 量化研究员使用**迭代比例拟合（Raking）**来迭代调整权重，直到多个边缘分布（如年龄、收入、地理）同时匹配人口普查数据。然而，这会显著增加过拟合的风险。

---

## IV. Seasonality Removal Frameworks: A Practitioner's Guide
## 四、 季节性剔除框架：实践指南

When removing seasonality from alternative data, you are making a fundamental bet on what is "normal" and what is "alpha."
在从另类数据中剔除季节性时，你实际上是在对什么是“常态”、什么是“Alpha”进行根本性的博弈。

### 1. Year-Over-Year (YoY) Differencing
### 1. 同比 (YoY) 差分
The "Retail Heuristic" and most common industry baseline.
“零售启发法”，也是最常见的行业基准。
*   **How to Use it:** Apply log-differencing ($\log(Y_t) - \log(Y_{t-s})$). Always use a "Base Check": If $Y_{t-s}$ is an outlier (check its Z-score), flag the resulting growth rate as unreliable.
    *   **使用方法：** 应用对数差分 ($\log(Y_t) - \log(Y_{t-s})$)。务必进行“基数检查”：如果 $Y_{t-s}$ 是离群值（检查其 Z-score），则将得出的增长率标记为不可靠。
*   **Why it is Good:** Zero parameterization, no convergence issues, and stakeholder alignment.
    *   **优点：** 无参数化、无收敛问题，且易于与利益相关者达成共识。
*   **Best Problem Profile:** Short histories (< 24 months) and Sanity Checks ("Floor" model).
    *   **最佳应用场景：** 历史数据较短（< 24 个月）和合理性检查（“底线”模型）。
    *   *💡 Real-World Example:* Home Depot saw a massive, one-time spike in sales in April 2020 due to pandemic lockdowns. In April 2021, the YoY growth rate was deeply negative. If you didn't check the "Base Effect" of 2020, you would have falsely assumed Home Depot's business was collapsing in 2021.
    *   *💡 现实案例：* 由于疫情封锁，家得宝（Home Depot）在 2020 年 4 月出现了大规模的一次性销售激增。在 2021 年 4 月，其同比增长率为深度负值。如果你不检查 2020 年的“基数效应”，你可能会错误地认为家得宝的业务在 2021 年正在崩溃。

### 2. STL Decomposition (Seasonal-Trend using LOESS)
### 2. STL 分解 (基于 LOESS 的季节趋势分解)
The "Visualizer’s Choice" for slowly evolving seasonal patterns.
处理缓慢演变的季节性模式的“可视化首选”。
*   **How to Use it:** Tweak the `seasonal.window` (LOESS filter width). Use `robust=True` to ensure shocks don't bleed into the trend.
    *   **使用方法：** 调整 `seasonal.window`（LOESS 过滤器宽度）。使用 `robust=True` 以确保冲击不会渗入趋势中。
*   **Why it is Good:** Flexibility to handle "evolving" peaks (e.g., e-commerce shifts) and handles additive/multiplicative seamlessly via log transforms.
    *   **优点：** 能够灵活处理“演变中”的峰值（例如电子商务的转变），并通过对数变换无缝处理加法/乘法模型。
*   **Best Problem Profile:** Long-Term Thematic Trading and Visual Alpha Discovery.
    *   **最佳应用场景：** 长期主题交易和可视化 Alpha 发现。
    *   *💡 Real-World Example:* Ten years ago, the peak holiday shopping week was strictly Black Friday. Today, it has shifted earlier into early November due to "Cyber Month" promotions. STL with a small `seasonal.window` dynamically adapts to this slow multi-year shift, whereas YoY assumes the pattern is perfectly static.
    *   *💡 现实案例：* 十年前，假日的购物高峰周严格限定在“黑色星期五”。如今，由于“网络月”促销活动，这一高峰已提前至 11 月初。带有较小 `seasonal.window` 的 STL 能够动态适应这种缓慢的跨年转变，而 YoY 则假设这种模式是完全静态的。

### 3. X-13ARIMA-SEATS
The "Econometric Standard" for high-precision macro and banking data.
高精度宏观和银行数据的“计量经济学标准”。
*   **How to Use it:** Specify "Trading Day" regressors and run "Sliding Spans" diagnostics.
    *   **使用方法：** 指定“交易日”回归量并运行“滑动跨度（Sliding Spans）”诊断。
*   **Why it is Good:** Calendar Precision (adjusts for the number of weekends in a month) and Benchmark Fidelity (matches government methodology).
    *   **优点：** 日历精确度（调整一个月中周末的数量）和基准忠实度（符合政府统计方法）。
*   **Best Problem Profile:** High-Volume Transaction Data and Macro-Leading Indicators.
    *   **最佳应用场景：** 大量交易数据和宏观领先指标。
    *   *💡 Real-World Example:* March 2024 has 5 Fridays, 5 Saturdays, and 5 Sundays. March 2023 had only 4 of each. A restaurant chain's credit card transactions will look artificially high in 2024 simply due to the calendar. X-13ARIMA applies a "Trading Day Penalty" to mathematically neutralize this artifact.
    *   *💡 现实案例：* 2024 年 3 月有 5 个周五、5 个周六和 5 个周日。而 2023 年 3 月每种只有 4 个。仅仅因为日历原因，一家连锁餐厅在 2024 年的信用卡交易额看起来会虚高。X-13ARIMA 会应用“交易日惩罚”来从数学上抵消这种人工痕迹。

### 4. Prophet (Generalized Additive Model)
### 4. Prophet (广义相加模型)
The "Data Scientist’s Tool" for messy, high-frequency, real-world data.
处理杂乱、高频、现实世界数据的“数据科学家工具”。
*   **How to Use it:** Add custom `holidays` dataframes and tune the `changepoint_prior_scale`.
    *   **使用方法：** 添加自定义 `holidays` 数据框并调整 `changepoint_prior_scale`。
*   **Why it is Good:** Native Gap Handling and Multi-Frequency modeling (daily/weekly/yearly simultaneously).
    *   **优点：** 原生支持缺失值处理，以及多频率建模（同时处理日/周/年周期）。
*   **Best Problem Profile:** Messy web/app traffic data and Event-Driven Trading.
    *   **最佳应用场景：** 杂乱的网页/App 流量数据和事件驱动交易。
    *   *💡 Real-World Example:* Predicting daily server traffic for a streaming service. You know traffic drops every Tuesday, spikes every Friday, plummets on July 4th, and you have 3 random days of missing data from a server outage last month. Prophet handles the multiple cycles, the known holiday, and the missing data natively.
    *   *💡 现实案例：* 预测流媒体服务的每日服务器流量。你已知流量每周二下降，每周五飙升，7 月 4 日骤降，并且上个月由于服务器故障有 3 天数据随机丢失。Prophet 可以原生处理多个周期、已知假日和缺失数据。

---

## V. Isolating and Trading on "Innovation"
## 五、 隔离“创新（Innovation）”并据此交易

By removing trend and seasonality, you perform a **Whitening Transformation** to isolate the **Innovation** (Shock).
通过剔除趋势和季节性，你执行了**白化变换（Whitening Transformation）**，以隔离出“创新”（即冲击）。

### 1. Mathematical Definition
### 1. 数学定义
$$Y_t = \text{Known}_t + \text{Innovation}_t$$
The Innovation ($\epsilon_t$) is the unexpected "Information Flow" not yet priced in by the market.
“创新” ($\epsilon_t$) 是尚未被市场定价的意外“信息流”。

### 2. Transient vs. Structural Innovation
### 2. 瞬态 vs. 结构性创新
*   **Transient:** Residual spikes. These mean-revert quickly.
    *   **瞬态：** 残余的尖峰。这些会迅速均值回归。
    *   *💡 Real-World Example:* A severe winter storm freezes a major logistics hub for a week. Shipments plummet. This is a transient shock—the packages will simply be shipped next week, causing a positive spike that cancels the negative one out.
    *   *💡 现实案例：* 一场严重的暴风雪导致主要的物流枢纽停摆一周。出货量骤降。这是一个瞬态冲击——包裹只需在下周发出，产生的正向激增会抵消之前的负向冲击。
*   **Structural:** Trend acceleration (the second derivative of the trend). This suggests a fundamental regime shift in market share or consumer behavior.
    *   **结构性：** 趋势加速（趋势的二阶导数）。这暗示了市场份额或消费者行为发生了根本性的系统性转变（Regime Shift）。
    *   *💡 Real-World Example:* A major competitor files for bankruptcy. Your target company's shipments jump 10% and stay there permanently. This is a structural innovation—a permanent regime shift.
    *   *💡 现实案例：* 一个主要竞争对手申请破产。你的目标公司的出货量跳升了 10% 并永久保持在那里。这是一个结构性创新——永久性的系统转变。

### 3. Advanced Extraction (Removing Autocorrelation)
### 3. 高级提取（剔除自相关）
Even after removing seasonality, time-series data often has "momentum" (autocorrelation). You must remove this to find the *true* surprise.
即使剔除了季节性，时间序列数据通常仍具有“动量”（自相关）。你必须剔除这一点才能找到*真正的*惊喜。
*   **AR Filtering:** Fit an Autoregressive model of order $p$:
    $$\epsilon_t = Y_t - (\phi_1 Y_{t-1} + \dots + \phi_p Y_{t-p})$$
*   **Choosing $p$:** Use the Partial Autocorrelation Function (PACF) or minimize the Akaike Information Criterion (AIC) to select how many lag periods to subtract. The remaining $\epsilon_t$ is pure, unadulterated alpha.
    *   **选择 $p$：** 使用偏自相关函数（PACF）或最小化赤池信息准则（AIC）来选择要减去多少个滞后周期。剩下的 $\epsilon_t$ 就是纯净、未被掺杂的 Alpha。

### 4. Dynamic Extraction via Kalman Filters (State-Space Innovation)
### 4. 通过卡尔曼滤波进行动态提取（状态空间创新）
The **Kalman Filter** is the premier tool for quantitative, real-time extraction of innovations. Unlike STL or moving averages, which are "backward-looking," a Kalman Filter is a recursive algorithm that maintains an internal "state" of the system and continuously updates its beliefs as new data arrives.
**卡尔曼滤波（Kalman Filter）**是实时量化提取创新的首选工具。与 STL 或移动平均线这种“滞后”的方法不同，卡尔曼滤波是一种递归算法，它维护系统的内部“状态”，并随着新数据的到来不断更新其信念。

*   **The Core Assumption:** The underlying "true" trend ($x_t$) is unobservable (hidden) and evolves over time with some system noise. What we actually observe ($y_t$) is this true trend corrupted by measurement noise (e.g., scraping errors, daily jitter).
    *   **核心假设：** 底层的“真实”趋势 ($x_t$) 是不可观测的（隐藏的），并随着某些系统噪声而演变。我们实际观察到的是被测量噪声（如抓取错误、每日波动）干扰后的真实趋势 ($y_t$)。
*   **The Mathematical Formulation:**
    *   **State Equation (The Hidden Reality):** $x_t = A \cdot x_{t-1} + w_t$ (where $w_t \sim N(0, Q)$ is the process noise).
    *   **Measurement Equation (What We See):** $y_t = H \cdot x_t + v_t$ (where $v_t \sim N(0, R)$ is the measurement noise).
*   **The Innovation ($\epsilon_t$):** The Kalman Filter makes a prediction for tomorrow ($\hat{y}_t$). When tomorrow's actual data ($y_t$) arrives, the difference is the **Innovation** (or Measurement Residual):
    $$\epsilon_t = y_t - \hat{y}_t$$
    *If this innovation is consistently positive over several days, the Kalman Filter automatically adjusts its hidden state ($x_t$) upward, realizing this is a structural shift, not just noise.*
    *   **创新 ($\epsilon_t$)：** 卡尔曼滤波会对明天做出预测 ($\hat{y}_t$)。当明天实际数据 ($y_t$) 到来时，两者的差值就是**创新**（或测量残差）：$\epsilon_t = y_t - \hat{y}_t$。*如果这种创新连续几天均为正值，卡尔曼滤波会自动向上调整其隐藏状态 ($x_t$)，意识到这是一个结构性转变，而非仅仅是噪声。*
*   **Best Problem Profile:** High-frequency, real-time trading environments (MFT/HFT) where you need to detect structural regime shifts *before* a traditional rolling average catches up.
    *   **最佳应用场景：** 高频、实时交易环境（MFT/HFT），在这种环境下，你需要在传统的滚动平均线反应过来*之前*检测到结构性系统转变。
    *   *💡 Real-World Example:* You track hourly satellite data of parking lot cars at Walmart. The data is extremely noisy (clouds, sensor glitches). A Kalman Filter maintains a "true expected volume." If a massive promotion drives a sudden 30% spike in cars, the Kalman Filter's prediction error ($\epsilon_t$) spikes instantly, generating a tradable signal on day one, whereas a 14-day rolling average would barely move.
    *   *💡 现实案例：* 你追踪沃尔玛停车场汽车的每小时卫星数据。数据噪音极大（云层、传感器故障）。卡尔曼滤波维护一个“真实预期量”。如果一场大规模促销活动导致汽车数量突然激增 30%，卡尔曼滤波的预测误差 ($\epsilon_t$) 会立即飙升，在第一天就产生可交易的信号，而 14 天滚动平均线几乎不会波动。

*   **Python Implementation (`pykalman`):**
```python
from pykalman import KalmanFilter
import numpy as np
import pandas as pd

def extract_kalman_innovation(series):
    """
    使用一维卡尔曼滤波平滑序列并提取创新项。
    """
    # 初始化一个基础的随机游走卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.01 # 微调参数：越小则趋势越平滑
    )
    
    # 可选：使用 EM 算法估计最佳噪声协方差
    # kf = kf.em(series.values, n_iter=5)
    
    # 过滤数据
    state_means, _ = kf.filter(series.values)
    
    # 'state_means' 是卡尔曼滤波对“已知”趋势的估计
    # 创新项是实际值减去估计趋势
    innovations = series.values - state_means.flatten()
    
    return pd.Series(innovations, index=series.index)

# 使用示例:
# df['kalman_innovation'] = extract_kalman_innovation(df['raw_shipments'])
```

---

## VI. Normalizing Innovation: The Z-Score Workflow
## 六、 创新归一化：Z-Score 工作流

To convert raw innovations into tradable signals, you must normalize them.
为了将原始创新转化为可交易信号，必须对其进行归一化。

### 1. The Core Formula (Rolling/Dynamic Z-score)
### 1. 核心公式（滚动/动态 Z-score）
Because alternative data volatility regimes shift over time, a static Z-score is useless.
由于另类数据的波动性状态会随时间变化，静态 Z-score 是无用的。
$$Z_t = \frac{\epsilon_t - \mu_{\epsilon(t, w)}}{\sigma_{\epsilon(t, w)}}$$

### 2. Cross-Sectional vs. Time-Series Z-Scoring
### 2. 截面 vs. 时间序列 Z-Score
*   **Time-Series Z-Score (Self-Relative):** Compares Target's innovation today against Target's own historical innovations. Excellent for directional, absolute-return trading.
    *   **时间序列 Z-Score (自身相对)：** 将目标公司今天的创新与其自身历史创新进行比较。非常适合方向性、绝对收益交易。
*   **Cross-Sectional Z-Score (Peer-Relative):** At time $t$, calculates the Z-score of Target's innovation relative to the distribution of innovations across its peer group (e.g., Target vs. Walmart vs. Costco). This neutralizes sector-wide macro shocks and isolates idiosyncratic company alpha. Essential for Market-Neutral / Statistical Arbitrage strategies.
    *   **截面 Z-Score (同业相对)：** 在时间 $t$，计算目标公司创新相对于其同业群体（如 Target vs. 沃尔玛 vs. 好市多）创新分布的 Z-score。这可以中和行业范围内的宏观冲击，并隔离出公司特有的 Alpha。对于市场中性/统计套利策略至关重要。
    *   *💡 Real-World Example:* Target has a Time-Series Z-score of $+1.5$ today (a solid positive surprise compared to its own history). However, you look at the Cross-Sectional Z-score for the Retail Sector, and Walmart, Costco, and Kroger are all at $+3.0$ due to a new government stimulus check. Cross-sectionally, Target is actually underperforming its peers and might be a *short* target in a pair trade.
    *   *💡 现实案例：* Target 今天的时序 Z-score 为 +1.5（与其自身历史相比是稳健的正向惊喜）。然而，你观察零售行业的截面 Z-score，由于政府发放了新的刺激支票，沃尔玛、好市多和克罗格的 Z-score 都在 +3.0。从截面来看，Target 实际上表现不如同业，在配对交易中可能是一个*做空*目标。

### 3. Robust Z-scores and EWMA (Handling Fat Tails)
### 3. 稳健 Z-score 与 EWMA（处理肥尾）
*   **Exponentially Weighted Moving Average (EWMA):** Instead of a simple rolling window, calculate $\mu$ and $\sigma$ using EWMA ($\mu_t = \alpha x_t + (1-\alpha)\mu_{t-1}$) to make the Z-score react faster to new volatility regimes.
    *   **指数加权移动平均 (EWMA)：** 与其使用简单的滚动窗口，不如使用 EWMA 计算 $\mu$ 和 $\sigma$，使 Z-score 对新的波动状态反应更快。
*   **Median Absolute Deviation (MAD):** Standard Z-scores are broken by massive outliers. Use MAD for a robust estimation:
    *   **绝对中位差 (MAD)：** 标准 Z-score 会被巨大的离群值破坏。使用 MAD 进行稳健估计：
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
## 七、 频率与模型拟合总结

| Frequency (频率) | Recommended Framework (推荐框架) | Why? (原因) | Risk (风险) |
| :--- | :--- | :--- | :--- |
| **Monthly (月度)** | X-13ARIMA-SEATS | Best-in-class for Trading Day adjustment. (交易日调整的最优方案) | High setup complexity. (配置复杂) |
| **Weekly (周度)** | Prophet | Handles "53rd week" and holiday drift natively. (原生处理第 53 周和假日漂移) | Can over-smooth signals. (可能过度平滑信号) |
| **Daily (日度)** | MSTL or Prophet | Decomposes Weekly + Yearly cycles. (分解周周期 + 年周期) | High computational cost. (计算成本高) |
| **Sparse (稀疏)** | Prophet | Robust to irregular sampling and gaps. (对不规则采样和缺失值稳健) | High risk of over-fitting noise. (过拟合噪声的风险高) |
| **Real-Time (实时)**| Kalman Filter | Recursive/Online extraction of Innovations. (递归/在线提取创新项) | Requires precise state-space tuning. (需要精确的状态空间微调) |
