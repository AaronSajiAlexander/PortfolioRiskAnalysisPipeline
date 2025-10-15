# Portfolio Risk Analysis Pipeline

## Overview
This project is a multi-stage portfolio risk analysis application built with Streamlit. It simulates a Bloomberg data integration pipeline to analyze financial portfolios through five sequential stages: (1) Data Ingestion, (2) Core Risk Analysis, (3) ML Analysis (Anomaly Detection & Risk Prediction), (4) Sentiment Analysis for high-risk assets, and (5) Comprehensive PDF Report Generation. The application uses mock data to simulate real-world financial feeds, applying time-series analysis, rule-based risk scoring, advanced machine learning, and NLP-based sentiment analysis to provide actionable investment insights. The primary goal is to empower users with a powerful tool for understanding and mitigating portfolio risks.

## User Preferences
Preferred communication style: Simple, everyday language.

## AI & Machine Learning Explained (For Beginners)

This application uses artificial intelligence (AI) and machine learning (ML) to help analyze investment portfolios. If you're new to these concepts, here's what they mean and how they work in this application:

### What is AI and Machine Learning?

**Artificial Intelligence (AI)** is like giving computers the ability to make smart decisions, similar to how humans think and learn. Instead of following strict rules, AI can recognize patterns and make predictions.

**Machine Learning (ML)** is a type of AI where computers learn from examples. Think of it like teaching a child to recognize animals - you show them many pictures of cats and dogs, and eventually they learn to tell the difference on their own. ML works the same way with data.

### The Three ML Techniques We Use

#### 1. Anomaly Detection (Finding the Unusual)

**What it does:** Identifies investments that are behaving strangely or differently from the rest of your portfolio.

**How it works:** We use a technique called "Isolation Forest" - imagine you have a forest of trees, and you're trying to find which fruit is different from the others. The unusual fruits are easier to isolate (separate) from the group because they don't fit the normal pattern.

**In practice:** The system looks at things like price swings, trading volume, and returns for all your investments. If one stock is behaving very differently - maybe dropping dramatically while others are stable, or showing unusually high volatility - it flags it as an "anomaly" (something unusual).

**What you get:**
- **Anomaly Score (0-100)**: Higher scores mean more unusual behavior
- **Severity Level**: LOW, MEDIUM, HIGH, or CRITICAL
- **Contributing Factors**: Which specific measurements made it unusual (e.g., extreme price drops, unusual trading volume)
- **Recommendations**: What action to take based on the severity

#### 2. Risk Prediction (Forecasting the Future)

**What it does:** Predicts whether an investment's risk level will stay the same, get better, or get worse.

**How it works:** We use "Random Forest Classifier" - imagine you're asking advice from 100 different financial experts. Each expert looks at the investment data and makes their own prediction. Then you take a vote: whatever most experts agree on becomes the final prediction. This "wisdom of the crowd" approach makes predictions more reliable.

**In practice:** The system trains itself by looking at patterns in your current portfolio data. It learns which combinations of metrics (like volatility, price trends, and trading patterns) typically lead to high-risk vs. low-risk ratings. Then it applies this knowledge to predict future risk levels.

**What you get:**
- **Predicted Risk Rating**: GREEN (low risk), YELLOW (moderate), or RED (high risk)
- **Confidence Score**: How sure the system is about its prediction (0-100%)
- **Risk Trend**: Whether risk is IMPROVING, DETERIORATING, or STABLE
- **Key Risk Drivers**: Which factors are most important in determining risk

#### 3. Sentiment Analysis (Reading the News)

**What it does:** Reads financial news headlines and determines if they're positive, negative, or neutral about your investments.

**How it works:** We use "TextBlob" - a tool that understands language context. It's like having someone read all the news for you and tell you whether the overall tone is good news, bad news, or neutral.

**In practice:** For high-risk (RED-flagged) investments, the system gathers recent news articles and analyzes the language used. Words like "soaring," "breakthrough," or "record profits" suggest positive sentiment. Words like "plummeting," "investigation," or "losses" suggest negative sentiment.

**What you get:**
- **Sentiment Score (-1 to +1)**: Negative to positive scale
- **Sentiment Label**: NEGATIVE, NEUTRAL, or POSITIVE
- **Sentiment Trend**: Whether news is getting better or worse over time
- **Key Themes**: What topics are being discussed (earnings, regulatory issues, management changes, etc.)
- **Confidence Score**: How reliable the sentiment analysis is (based on number of articles and consistency)

### Why Use Machine Learning?

1. **Pattern Recognition**: ML can spot complex patterns in large amounts of data that humans might miss.

2. **Speed**: It can analyze thousands of data points across many investments in seconds.

3. **Consistency**: Unlike humans, ML doesn't get tired or emotional - it applies the same analysis standards every time.

4. **Early Warning System**: By detecting anomalies and predicting trends, ML can alert you to potential problems before they become serious.

5. **Data-Driven Decisions**: Instead of relying on gut feelings, ML provides objective insights based on actual data patterns.

### Important Limitations to Understand

- **Historical Data**: ML learns from past patterns. It can't predict completely new situations or "black swan" events (unexpected market crashes).

- **Not 100% Accurate**: The predictions are probabilities, not certainties. A 90% confidence score means there's still a 10% chance the prediction is wrong.

- **Requires Good Data**: ML is only as good as the data it learns from. In this application, we use simulated (mock) data for demonstration purposes.

- **Human Judgment Still Matters**: ML is a tool to help you make better decisions, not to make decisions for you. Always combine ML insights with your own research and judgment.

### How to Interpret the Results

When you see ML analysis results:

1. **High anomaly scores (60+)** deserve immediate attention - investigate why the investment is unusual.

2. **Risk predictions showing DETERIORATING trends** are early warnings to review those positions.

3. **Negative sentiment with high confidence** suggests you should check what's happening with that company.

4. **Look at the "Key Drivers" and "Contributing Factors"** - they tell you *why* the system flagged something, which is often more valuable than the flag itself.

Remember: This application is designed for learning and demonstration. In real-world scenarios, always consult with financial professionals before making investment decisions.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with a single-page design and sidebar controls.
- **State Management**: Streamlit session state for pipeline results and timings.
- **UI/UX**: Interactive elements (sliders, buttons), wide layout for visualizations.
- **Visualization**: Matplotlib for charts, ReportLab for PDF generation.
- **Downloads**: PDF report, portfolio CSV, and risk analysis CSV download options.

### Backend Architecture
- **Pipeline Pattern**: A five-stage sequential processing pipeline:
  1. **Stage 1 - Data Ingestion**: Simulates Bloomberg API data fetching.
  2. **Stage 2 - Core Analysis**: Performs time-series and rule-based risk scoring.
  3. **Stage 3 - ML Analysis**: Conducts anomaly detection (Isolation Forest) and risk prediction (Random Forest Classifier).
  4. **Stage 4 - Sentiment Analysis**: NLP-based analysis exclusively on RED-flagged assets.
  5. **Stage 5 - Report Generation**: Creates PDF reports with visualizations and ML insights.
- **Modular Design**: Each pipeline stage is encapsulated in its own engine class.
- **Data Flow**: Linear progression with staged transformations.
- **Risk Score Calculation**: Based on 7 distinct risk flags (e.g., high_volatility, extreme_drawdown).
- **Risk Rating Classification**: Assets categorized as RED (High), YELLOW (Moderate), or GREEN (Low) based on hierarchical flag system.
- **Sentiment Analysis**: Calculates weighted average sentiment, classifies sentiment (NEGATIVE, NEUTRAL, POSITIVE), analyzes trends, and provides a confidence score. Applied selectively to RED-flagged assets.
- **Machine Learning**:
    - **Anomaly Detection**: Isolation Forest identifies unusual behaviors, providing a score, severity (LOW, MEDIUM, HIGH, CRITICAL), and contributing factors.
    - **Risk Rating Prediction**: Random Forest Classifier predicts future risk ratings (GREEN, YELLOW, RED), identifies key drivers, provides prediction confidence, and analyzes risk trends (IMPROVING, DETERIORATING, STABLE).
- **ML Validation System**: Automated checks validate anomaly detection, risk prediction, feature quality, and feature importance.

### Data Storage Solutions
- **In-Memory**: Pandas DataFrames for all data manipulation.
- **File System**: Reports saved to `/reports`, charts to `/charts`.
- **Session State**: Streamlit session state for managing pipeline results.
- **No Database**: Application is stateless.

### Key Technologies
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn (Isolation Forest, Random Forest Classifier)
- **NLP**: TextBlob
- **Visualization**: Matplotlib
- **PDF Generation**: ReportLab
- **Web Framework**: Streamlit

## External Dependencies

### Third-party Libraries
- **streamlit**: Web application framework.
- **pandas**: Data manipulation.
- **numpy**: Numerical operations.
- **scikit-learn**: Machine learning algorithms.
- **matplotlib**: Charting.
- **reportlab**: PDF generation.
- **textblob**: Sentiment analysis.

### Simulated External Services
- **Bloomberg API**: Simulated via `MockBloombergData` class for financial data (asset pricing, historical series, volume data, market capitalization, Bloomberg IDs).

### Data Sources (Simulated)
- **News Sources**: Mock news feeds from various financial outlets (Reuters, Bloomberg News, Financial Times, WSJ, etc.).
- **Market Data**: Mock time-series data for prices, volumes, and technical indicators.

### File System Dependencies
- Requires write access to `/reports` and `/charts` directories.