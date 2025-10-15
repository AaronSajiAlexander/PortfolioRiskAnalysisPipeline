# Portfolio Risk Analysis Pipeline

## Overview
This project is a multi-stage portfolio risk analysis application built with Streamlit. It simulates a Bloomberg data integration pipeline to analyze financial portfolios through five sequential stages: Data Ingestion, Core Risk Analysis, ML Analysis (Anomaly Detection & Risk Prediction), Sentiment Analysis for high-risk assets, and Comprehensive PDF Report Generation. The application uses mock data to simulate real-world financial feeds, applying time-series analysis, rule-based risk scoring, advanced machine learning, and NLP-based sentiment analysis to provide actionable investment insights. The primary goal is to empower users with a powerful tool for understanding and mitigating portfolio risks.

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
  1. **Stage 1 - Data Ingestion**: Simulates Bloomberg API data fetching.
  2. **Stage 2 - Core Analysis**: Performs time-series and rule-based risk scoring with 7 risk flags and a GREEN/YELLOW/RED rating system.
  3. **Stage 3 - ML Analysis**: Conducts anomaly detection (Isolation Forest) and risk prediction (Random Forest Classifier). Includes ML validation and feature importance analysis.
  4. **Stage 4 - Sentiment Analysis**: NLP-based analysis (TextBlob) exclusively on RED-flagged assets, providing sentiment scores, trends, themes, and confidence.
  5. **Stage 5 - Report Generation**: Creates PDF reports with visualizations and ML insights.
- **Modular Design**: Each pipeline stage is encapsulated in its own engine class.
- **Data Flow**: Linear progression with staged transformations.
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