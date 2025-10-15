# Portfolio Risk Analysis Pipeline

## Overview
This project is a multi-stage portfolio risk analysis application built with Streamlit. It simulates a Bloomberg data integration pipeline to analyze financial portfolios through five sequential stages: Data Ingestion, Core Risk Analysis, ML Analysis (Anomaly Detection & Risk Prediction), Sentiment Analysis for high-risk assets, and Comprehensive PDF Report Generation. The application uses mock data to simulate real-world financial feeds, applying time-series analysis, rule-based risk scoring, advanced machine learning, and NLP-based sentiment analysis to provide actionable investment insights. The primary goal is to empower users with a powerful tool for understanding and mitigating portfolio risks.

## User Preferences
Preferred communication style: Simple, everyday language.
Target audience: Users new to AI and ML - requires beginner-friendly explanations.

## Backend Pipeline - Simple Overview

For users new to AI and ML, here's a simple explanation of what the 5-stage pipeline does:

### Stage 1: Data Ingestion (Gathering Information)
**What it does:** Collects all the raw data about your investments - stock prices, trading volumes, historical data, etc.
**Think of it like:** Organizing your filing cabinet before you can analyze anything.

### Stage 2: Core Analysis (Traditional Health Check)
**What it does:** Examines each investment using traditional financial methods that professional investors have used for decades.
**Checks 8 vital signs:** Volatility, Maximum Drawdown, Volume, Price Momentum, Sharpe Ratio, RSI, Beta, and Returns.
**Creates 7 warning flags:** High Volatility, Extreme Drawdown, Volume Collapse, Severe Decline, Extended Decline, Poor Risk/Return, Momentum Breakdown.
**Assigns ratings:** 🔴 RED (high risk), 🟡 YELLOW (moderate risk), or 🟢 GREEN (low risk).
**Think of it like:** Taking your car to a mechanic for a complete inspection.

### Stage 3: ML Analysis (AI-Powered Pattern Detection)
**What it does:** Uses artificial intelligence to find hidden patterns that traditional analysis might miss.
**Part 1 - Anomaly Detection:** Finds investments behaving unusually (like spotting someone wearing a space suit in a business meeting).
**Part 2 - Risk Prediction:** Predicts whether investments will get riskier or safer (like a weather forecast for your portfolio).
**Part 3 - Validation:** Checks its own work to ensure results are reliable.
**Think of it like:** Having a detective who spots unusual patterns and connections that aren't obvious.

### Stage 4: Sentiment Analysis (Reading the News)
**What it does:** Reads and analyzes financial news about your high-risk (RED) investments.
**Analyzes:** Sentiment scores, trends, themes, and confidence levels from multiple news sources.
**Only for RED items:** Focuses on investments that need attention most urgently.
**Think of it like:** Having someone read 50 news articles and summarize whether they're good news, bad news, or neutral.

### Stage 5: Report Generation (Pulling It All Together)
**What it does:** Creates professional reports and downloadable files from all the analysis.
**Generates:** PDF report (executive summary), Portfolio CSV (all data), and Risk Analysis CSV (detailed risk metrics).
**Think of it like:** A chef assembling all prepared ingredients into a complete meal.

**📘 For detailed beginner-friendly explanations of each stage, see `BACKEND_GUIDE_FOR_BEGINNERS.md`**

**🔧 For detailed explanation of how test/mock data is created, see `MOCK_DATA_GUIDE.md`**

**📰 For detailed technical guide on sentiment analysis (Stage 4), see `SENTIMENT_ANALYSIS_GUIDE.md`**

## System Architecture

### UI/UX Decisions
- **Frontend Framework**: Streamlit web application with a single-page design and sidebar controls.
- **State Management**: Streamlit session state for pipeline results and timings.
- **Visualization**: Matplotlib for charts, ReportLab for PDF generation.
- **Downloads**: Offers PDF report, portfolio CSV, and risk analysis CSV download options.

### Technical Implementations
- **Pipeline Pattern**: A five-stage sequential processing pipeline where each stage builds on the previous one:
  1. **Data Ingestion**: Simulates Bloomberg API data fetching, collecting basic details, current snapshot, historical data, and trading activity.
  2. **Core Analysis**: Performs time-series and rule-based risk scoring based on 8 financial vital signs (Volatility, Max Drawdown, Volume Analysis, Price Momentum, Sharpe Ratio, RSI, Beta, Returns) leading to 7 warning flags and a GREEN/YELLOW/RED rating system.
  3. **ML Analysis**: Conducts anomaly detection (Isolation Forest) and risk prediction (Random Forest Classifier) using 9 key measurements. Includes ML validation and feature importance analysis.
  4. **Sentiment Analysis**: NLP-based analysis (TextBlob) on financial news exclusively for RED-flagged assets, providing sentiment scores, trends, themes (e.g., Earnings, Regulatory), and confidence levels.
  5. **Report Generation**: Creates comprehensive PDF reports with visualizations, executive summaries, and ML insights, along with detailed CSV exports.
- **Modular Design**: Each pipeline stage is encapsulated in its own engine class.
- **Data Flow**: Linear progression with staged transformations.
- **ML Validation System**: Automated checks for anomaly detection, risk prediction, feature quality, and feature importance.

### System Design Choices
- **Data Storage**: Primarily uses Pandas DataFrames for in-memory data manipulation.
- **File System**: Reports and charts are saved to dedicated `/reports` and `/charts` directories.
- **Stateless Application**: The application does not use a persistent database.

## External Dependencies

### Third-party Libraries
- **streamlit**: For building the web application.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **scikit-learn**: For machine learning algorithms (Isolation Forest, Random Forest Classifier).
- **matplotlib**: For generating plots and charts.
- **reportlab**: For creating PDF documents.
- **textblob**: For performing sentiment analysis.

### Simulated External Services
- **Bloomberg API**: Simulated via a `MockBloombergData` class to provide financial data such as asset pricing, historical series, volume data, and market capitalization.

### Data Sources (Simulated)
- **News Sources**: Mock news feeds from various financial outlets (e.g., Reuters, Bloomberg News, Financial Times, WSJ).
- **Market Data**: Mock time-series data for prices, volumes, and technical indicators.