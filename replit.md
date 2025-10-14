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