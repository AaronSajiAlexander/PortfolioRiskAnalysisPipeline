# Financial Risk Metrics Documentation

This document outlines the core financial risk metrics used in the project, detailing their calculation, purpose, and critical application in both **Rule-Based Risk Rating** and **Machine Learning Models** for risk prediction.

---

## 1. Core Risk Metrics and Calculations

| Metric | Purpose | Calculation (Conceptual) | Python Snippet (Numpy) |
| :--- | :--- | :--- | :--- |
| **Volatility** | Measures price fluctuation (unpredictability). High volatility $\rightarrow$ high risk. | Annualized standard deviation of daily returns. | `volatility = np.std(returns) * np.sqrt(252)` |
| **Maximum Drawdown (MDD)** | Largest percentage drop from a peak to a subsequent trough. Worst-case loss scenario. | Largest peak-to-trough decline in cumulative value. | `max_drawdown = np.min(drawdown)` |
| **Sharpe Ratio** | Risk-adjusted return. Measures excess return per unit of volatility (assuming 2% risk-free rate). | (Annualized Excess Return) / Volatility | `sharpe_ratio = excess_returns / volatility` |
| **RSI (Relative Strength Index)** | Momentum indicator determining if an asset is overbought ($\text{RSI}>70$) or oversold ($\text{RSI}<30$). | Ratio of average gains to average losses over 14 periods. | $\text{RSI} = 100 - \left( \frac{100}{1 + \text{RS}} \right)$ |
| **Beta ($\beta$)** | Measures an asset's volatility relative to the overall market. **Systematic risk** indicator. | Covariance of asset and market returns divided by market variance. | $\beta = \frac{\text{Covariance}}{\text{Market Variance}}$ |

---

## 2. Application in Project

These metrics are fundamental to the project's **dual-approach risk assessment strategy**—combining deterministic rules with predictive modeling.

### 🎯 Rule-Based Risk Calculator

The metrics feed into a **hierarchical flag system** to assign an asset a $\text{GREEN}$, $\text{YELLOW}$, or $\text{RED}$ risk rating.

| Metric | Rule-Based Flag Trigger | Consequence | Rating Impact |
| :--- | :--- | :--- | :--- |
| **Maximum Drawdown** | **Critical Flag: extreme\_drawdown** if MDD $< -20\%$ | Asset has experienced severe losses. | **Immediate RED Rating** 🚨 |
| **Volatility** | **Warning Flag: high\_volatility** if Annualized Volatility $> 40\%$ | Price swings are highly unpredictable. | Contributes to $\text{YELLOW}$ or $\text{RED}$ rating. |
| **Sharpe Ratio** | **Warning Flag: poor\_risk\_return** if Sharpe Ratio $< -0.5$ | Returns do not adequately compensate for the risk taken. | Contributes to $\text{YELLOW}$ or $\text{RED}$ rating. |
| **RSI & Beta** | Not directly used for flag calculation. | Provide context for momentum and market sensitivity. | N/A |

#### Final Rating Logic

* **RED** 🚨: Triggered by any **Critical Flag** (e.g., *extreme\_drawdown*, *volume\_collapse*, *severe\_decline*) or a high accumulation of warning flags.
* **YELLOW** ⚠️: Triggered if $\mathbf{2}$ or more **Warning Flags** (e.g., *high\_volatility*, *poor\_risk\_return*, *extended\_decline*) are active.
* **GREEN** ✅: No Critical or concerning Warning Flags triggered.

---

### 🧠 Machine Learning Models

The five core metrics, alongside four other features (e.g., price momentum and volume decline), form the complete **9-feature set** used by the two predictive models.

| Metric | Role as an ML Feature | Models Using the Feature |
| :--- | :--- | :--- |
| **Volatility** | Identifies assets with **unstable price behavior** and high risk. | Isolation Forest, Random Forest Classifier |
| **Maximum Drawdown** | Critical for detecting assets experiencing **severe, non-recoverable losses**. | Isolation Forest, Random Forest Classifier |
| **Sharpe Ratio** | Helps models identify **unfavorable risk-return profiles**. | Isolation Forest, Random Forest Classifier |
| **RSI (Relative Strength Index)** | Allows models to detect **momentum patterns** and predict future risk changes. | Isolation Forest, Random Forest Classifier |
| **Beta** | Provides context on **systematic risk** (market sensitivity). | Isolation Forest, Random Forest Classifier |

#### Model Objectives and Details

#### 1. Isolation Forest (Anomaly Detection)

| Attribute | Detail |
| :--- | :--- |
| **Core Function** | **Detecting Outliers/Anomalies.** Identifies assets whose combined feature values deviate significantly from the rest of the asset population. |
| **Mechanism** | It works by randomly isolating samples using few splits in a tree structure. Anomalies are easier to isolate using fewer splits near the root. |
| **Usage in Project** | **Risk Identification:** Flags assets with **unusual risk profiles** that may not be caught by simple rules. The output anomaly score informs the overall risk rating. |
| **Strength** | Highly efficient and ideal for checking the **consistency and coherence** of the 9 risk features. |

#### 2. Random Forest Classifier (Risk Prediction)

| Attribute | Detail |
| :--- | :--- |
| **Core Function** | **Multi-Class Classification.** Predicts whether an asset's risk rating will **maintain**, **improve**, or **deteriorate** over the next period. |
| **Mechanism** | An **ensemble learning method** that builds multiple decision trees to create a robust, accurate prediction and avoid overfitting. |
| **Usage in Project** | **Predictive Rating:** Uses the current 9 features to classify the asset's **future risk trend**, acting as a forward-looking alert. |
| **Key Output** | **Feature Importance:** Ranks the 9 features based on their predictive power, revealing which financial metric is currently **driving the most risk** in the portfolio. |