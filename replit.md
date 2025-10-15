# Portfolio Risk Analysis Pipeline

## Overview
This project is a multi-stage portfolio risk analysis application built with Streamlit. It simulates a Bloomberg data integration pipeline to analyze financial portfolios through five key stages: data ingestion, core risk analysis, machine learning analysis (anomaly detection & risk prediction), sentiment analysis for high-risk assets, and comprehensive PDF report generation. The application uses mock data to simulate real-world financial feeds, applying time-series analysis, rule-based risk scoring, advanced machine learning, and NLP-based sentiment analysis to provide actionable investment insights. The primary goal is to empower users with a powerful tool for understanding and mitigating portfolio risks.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application, designed as a single-page application with sidebar controls.
- **State Management**: Streamlit session state is used for managing pipeline results and execution timings.
- **UI/UX**: Interactive sliders, buttons for staged execution, and a wide layout for data visualization.
- **Visualization**: Matplotlib for charts and ReportLab for PDF generation.

### Backend Architecture
- **Pipeline Pattern**: A five-stage sequential processing pipeline:
  1. **Data Ingestion**: Simulates Bloomberg API data fetching.
  2. **Core Analysis**: Performs time-series and rule-based risk scoring.
  3. **ML Analysis**: Conducts anomaly detection (Isolation Forest) and risk prediction (Random Forest Classifier).
  4. **Sentiment Analysis**: NLP-based analysis exclusively on RED-flagged assets.
  5. **Report Generation**: Creates PDF reports incorporating visualizations and ML insights.
- **Modular Design**: Each pipeline stage is encapsulated within its own engine class, ensuring clear separation of concerns.
- **Data Flow**: Linear progression of data with staged transformations across the pipeline.
- **Risk Score Calculation**: Based on the count of triggered risk flags, with 7 distinct flags (e.g., high_volatility, extreme_drawdown).
- **Risk Rating Classification**: Assets are categorized as RED (High Risk), YELLOW (Moderate Risk), or GREEN (Low Risk) using a hierarchical flag system.
- **Sentiment Analysis**: Calculates a weighted average sentiment score from news articles, classifies sentiment (NEGATIVE, NEUTRAL, POSITIVE), analyzes sentiment trends, and provides a confidence score based on article count and consistency. Sentiment analysis is selectively applied only to RED-flagged assets.
- **Machine Learning**:
    - **Anomaly Detection**: Uses Isolation Forest to identify unusual asset behaviors, providing an anomaly score and severity classification (LOW, MEDIUM, HIGH, CRITICAL) along with contributing factors.
    - **Risk Rating Prediction**: Employs a Random Forest Classifier to predict future risk ratings (GREEN, YELLOW, RED) and identify key risk drivers. It provides prediction confidence and analyzes risk trends (IMPROVING, DETERIORATING, STABLE). The model dynamically adapts to the number of risk classes present and includes safeguards for insufficient samples or class imbalance.

### Data Storage Solutions
- **In-Memory**: Pandas DataFrames for all data manipulation during execution.
- **File System**: Reports are stored in the `/reports` directory and charts in the `/charts` directory.
- **Session State**: Streamlit session state manages pipeline results between runs.
- **No Database**: The application is stateless with no persistent database.

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
- **Bloomberg API**: Simulated via `MockBloombergData` class to provide realistic financial data (asset pricing, historical series, volume data, market capitalization, Bloomberg IDs).

### Data Sources (Simulated)
- **News Sources**: Mock news feeds from various financial outlets (Reuters, Bloomberg News, Financial Times, WSJ, etc.).
- **Market Data**: Mock time-series data for prices, volumes, and technical indicators.

### File System Dependencies
- Requires write access to `/reports` and `/charts` directories.

## PDF Report Structure

The generated PDF reports include comprehensive analysis across multiple sections:

### Report Sections
1. **Title Page**: Report metadata, portfolio size, and risk assessment summary
2. **Executive Summary**: Key findings, risk concentration, and sentiment overview
3. **Portfolio Overview**: Sector allocation with market cap breakdown
4. **Risk Analysis Results**: Risk distribution and high-risk asset details
5. **Machine Learning Analysis**: Anomaly detection results, risk predictions, feature importance, and ML validation results
6. **Sentiment Analysis**: Sentiment scores, trends, and key themes for RED-flagged assets
7. **Detailed Asset Analysis**: In-depth analysis of top 5 highest risk assets
8. **Recommendations**: Actionable recommendations based on risk analysis
9. **Appendix**: Methodology, risk thresholds, and report generation details
10. **Test Data Section**: Complete portfolio and analysis data for further analysis

### Test Data Section
The Test Data section (added at the end of reports) provides comprehensive raw data for validation and further analysis:
- **Complete Portfolio Data**: All assets with symbols, companies, sectors, prices, market caps, P/E ratios, and dividend yields
- **Complete Risk Analysis Results**: All risk metrics including ratings (color-coded), volatility, max drawdown, beta, Sharpe ratio, and RSI
- **Performance Metrics Data**: Returns (1M, 3M, 6M), volume decline, and Sharpe ratios
- **Risk Flags Details**: Detailed breakdown of which risk flags are triggered for each asset (high volatility, extreme drawdown, volume collapse, etc.)
- **Data Summary**: Metadata about the test data generation and analysis period

This section enables users to perform custom analysis, verify results, and conduct detailed reviews of all portfolio data.

## ML Validation System

To ensure ML results are accurate and reliable, an automated validation system runs after ML analysis completes.

### Validation Checks

**1. Anomaly Detection Validation**:
- Score range check: Validates all anomaly scores are 0-100
- Severity consistency: Confirms severity matches score thresholds (CRITICAL ≥80, HIGH ≥60, MEDIUM ≥40, LOW <40)
- Anomaly rate verification: Checks rate is reasonable (~15% expected, warns if >30% or <5%)
- Critical anomaly validation: Ensures critical anomalies have scores ≥80

**2. Risk Prediction Validation**:
- Confidence score validation: Verifies all confidence scores are 0-100%
- Model accuracy check: Flags if accuracy <50% (too low) or >99% (possible overfitting)
- Trend consistency: Validates trend direction matches rating changes
- Probability validation: Ensures risk probabilities sum to ~100%

**3. Feature Quality Validation**:
- NaN detection: Identifies missing values in features
- Infinite value check: Detects infinite values that could break models
- Variance analysis: Warns about features with very low variance (<0.0001)

**4. Feature Importance Validation**:
- Sum validation: Checks that importance scores sum to ~100%
- Negative value check: Ensures no negative importance values
- Dominance detection: Warns if single feature >60% importance

### Validation Status Levels

- **PASS**: All validation checks successful, results are reliable
- **WARNING**: Minor issues detected, results usable but review recommended
- **FAIL**: Critical issues found, results may be unreliable

### Output

- Overall validation status (PASS/WARNING/FAIL)
- Detailed check results with metrics
- List of warnings and issues detected
- Displayed in UI (ML Analysis tab) and PDF report

### Example Validation Scenarios

- Detects potential overfitting when model accuracy reaches 100%
- Identifies data quality issues (NaN values, infinite values)
- Catches inconsistencies in trend analysis
- Validates severity classifications are correct
