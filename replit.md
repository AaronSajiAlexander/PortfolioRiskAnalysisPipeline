# Portfolio Risk Analysis Pipeline

## Overview
This project is a multi-stage portfolio risk analysis application built with Streamlit. It simulates a Bloomberg data integration pipeline to analyze financial portfolios through five key stages: data ingestion, core risk analysis, machine learning analysis (anomaly detection & risk prediction), sentiment analysis for high-risk assets, and comprehensive PDF report generation. The application uses mock data to simulate real-world financial feeds, applying time-series analysis, rule-based risk scoring, advanced machine learning, and NLP-based sentiment analysis to provide actionable investment insights. The primary goal is to empower users with a powerful tool for understanding and mitigating portfolio risks.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with a single-page design and sidebar controls.
- **State Management**: Streamlit session state for pipeline results and timings.
- **UI/UX**: Interactive elements (sliders, buttons), wide layout for visualizations.
- **Visualization**: Matplotlib for charts, ReportLab for PDF generation.
- **Downloads**: PDF report, portfolio CSV, and risk analysis CSV download options.

### Backend Architecture
- **Pipeline Pattern**: A five-stage sequential processing pipeline:
  1. **Data Ingestion**: Simulates Bloomberg API data fetching.
  2. **Core Analysis**: Performs time-series and rule-based risk scoring.
  3. **ML Analysis**: Conducts anomaly detection (Isolation Forest) and risk prediction (Random Forest Classifier).
  4. **Sentiment Analysis**: NLP-based analysis exclusively on RED-flagged assets.
  5. **Report Generation**: Creates PDF reports with visualizations and ML insights.
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