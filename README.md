# Washout & Burst Strategy (A-Share)

This is a quantitative trading system designed to capture the **"Washout and Rebound"** (or "Washout & Burst") pattern in the A-share market. By analyzing price-volume interactions, it identifies stocks undergoing a "washout" phase, where weak holders are shaken out and is often followed by a sharp "burst" driven by smart money.

## 1. Quantitative Methodology

The core philosophy is to identify assets with specific "Price Action" and "Volume" characteristics. The mathematical formulation of the features and targets is defined as follows.

### 1.1 Feature Engineering (Factors)

We construct a multi-dimensional feature vector $X_t$ for each stock at day $t$ using **Microsoft Qlib** expressions.

#### **A. Price Action (Washout Signals)**
We look for high amplitude and long lower shadows, indicating strong support or manipulation.

* **Amplitude ($A_t$)**: Measures daily price intensity.
    $$A_t = \frac{H_t - L_t}{C_{t-1}}$$
* **Lower Shadow Ratio ($S_{low}$)**: Captures the "V-shape" intraday rebound.
    $$S_{low} = \frac{\min(O_t, C_t) - L_t}{C_t}$$

#### **B. Volume Dynamics (Liquidity)**
Detecting volume shrinkage (washout) or anomalies.

* **Turnover Ratio ($TR_{5d}$)**: Current activity relative to the weekly average.
    $$TR_{5d} = \frac{\text{Turnover}_t}{\text{Mean}(\text{Turnover}, 5)}$$
* **Volume Shrinkage ($V_{shrink}$)**: Identifies if selling pressure is exhausting.
    $$V_{shrink} = \frac{V_t}{\text{Mean}(V, 20)}$$

#### **C. Trend & Momentum**
* **Volatility ($\sigma_{20d}$)**: Normalized standard deviation to find active stocks.
    $$\sigma_{20d} = \frac{\text{Std}(C, 20)}{\text{Mean}(C, 20)}$$
* **Momentum ($R_{20d}$)**: Monthly return to ensure the stock is in an active trend.
    $$R_{20d} = \frac{C_t}{C_{t-20}} - 1$$
* **Simplified RSI**: Probability of Up-days in the past 2 weeks.
    $$RSI_{sim} = \frac{1}{14} \sum_{i=0}^{13} \mathbb{I}(C_{t-i} > O_{t-i})$$

> Notation: $O,H,L,C,V$ represent Open, High, Low, Close, and Volume respectively. $\mathbb{I}$ is the indicator function.

---

### 1.2 Modeling & Labeling

The strategy treats the prediction as a **Learning to Rank** problem.

* **Prediction Target ($Y$)**:
    We use different labels for training (classification/ranking) and backtesting. The primary goal is to capture the **Intraday Burst**.
    $$Y_{train} = \frac{H_{t+1}}{C_t} - 1 \quad (\text{Next Day Max Return})$$

* **Binary Classification Target** (used in Research):

$$
Y_{class} =
\begin{cases}
1, & \text{if } Y_{train} > 0.04 \\
0, & \text{otherwise}
\end{cases}
$$

- **Model Architecture**:
  - **Algorithm**: LightGBM (Gradient Boosting Decision Tree)
  - **Loss Function**: Rank Loss (LambdaRank) or MSE with `CSRankNorm` (Cross-Sectional Rank Normalization).
  - **Ranking**: $`\text{Score}_t = f_{\theta}(X_t)`$.
    Stocks are ranked daily by $\text{Score}_t$, and the Top $K$ are selected.

---

## 2. Architecture & Workflow

### System Stack
| Component | Technology | Description |
| :--- | :--- | :--- |
| **Data Source** | `AkShare` | East Money API for daily OHLCV & Turnover. |
| **Storage** | `ClickHouse` | High-performance OLAP database for full-market storage. |
| **Framework** | `Microsoft Qlib` | Data pipeline, feature calculation, and backtesting engine. |
| **Model** | `LightGBM` | GBDT model optimized for tabular financial data. |

### Pipeline
1.  **Ingestion**: Fetch daily data via AkShare $\rightarrow$ ClickHouse.
2.  **ETL**: ClickHouse $\rightarrow$ `.bin` (Qlib format).
3.  **Training**: Train LightGBM to predict $Y_{train}$ using features $X_t$.
4.  **Inference**: Generate daily scores for all stocks.
5.  **Strategy**:
    * **Select**: Top 5 stocks with highest scores.
    * **Filter**: Exclude stocks with "Limit-up at Open".
    * **Rotate**: Daily rebalancing (Hold 1 day).

---

## 3. Performance (Backtest)

**Period**: 2024.07.01 - 2025.12.30

The strategy demonstrates significant alpha during bull markets (High Beta).

| Metric | Strategy (Washout) | Benchmark (CSI 300) |
| :--- | :--- | :--- |
| **Annualized Return** | **1284.47%** | 22.49% |
| **Total Return** | **4493.12%** | 34.37% |
| **Sharpe Ratio** | **4.66** | 1.13 |
| **Max Drawdown** | -31.85% | -15.66% |

![Equity Curve](backtest/strategy_comparison.png)



## Analysis

The current strategy behaves as a typical "High-Beta Bull Market Amplifier."

**Pros**: It achieves astonishing excess returns when market sentiment is high, aggressively capturing upward momentum.

**Cons**: It suffers massive drawdowns during market downturns due to a lack of defensive mechanisms. It is currently a high-risk, high-reward system.


## Future Investigation & Assessment

Based on the deep attribution analysis, future development will focus on three core areas: Drawdown Control, Market Timing, and resolving Data Bias.


**Risk Management & Timing**

The biggest risk currently is "stubbornly holding top performers," operating at a full position even during market crashes.

  Market Regime Filter: Introduce the CSI 300 index as a market wind vane. If the index falls below the 20-day MA or MACD shows a death cross, the system will forcibly reduce positions or clear positions (go to cash).

  Stop-Loss Logic: Implement individual stock-level stop-loss mechanisms. If a single-day drop exceeds 5% or breaks key support, sell immediately intraday or at the next open.
  

**Model Robustness**

The current model is trained on data from 2021 to 2024, which may be overfit to recent market trends.

  Rolling Walk-forward: Adopt a "Sliding Window" training approach (e.g., train on the past 3 years to predict the next 6 months) to ensure the model adapts to changing market styles (Concept Drift).

  Extreme Market Stress Test: Backfill data for the 2015 Crash, 2018 Trade War, and Early 2024 Liquidity Crisis to verify the strategy's survival capability in bear markets.

**Factor Mining**

  Microstructure: Introduce minute-level data to calculate higher-frequency factors, such as "Capital Flow in the first 30 mins."

  Sentiment Factors: Combine price action with Sector Heat to filter out isolated pump-and-dump stocks and focus on legitimate sector leaders.
