# Portfolio Risk Analysis Pipeline

## Overview

This is a multi-stage portfolio risk analysis application built with Streamlit that simulates a Bloomberg data integration pipeline. The system analyzes financial portfolios through five sequential stages: data ingestion, core risk analysis, machine learning analysis (anomaly detection & risk prediction), sentiment analysis for high-risk assets, and comprehensive PDF report generation. The application uses mock data to simulate real-world financial data feeds and applies time-series analysis, rule-based risk scoring, advanced machine learning techniques, and NLP-based sentiment analysis to generate actionable investment insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application
- **Design Pattern**: Single-page application with sidebar controls
- **State Management**: Streamlit session state for pipeline results and execution timing
- **UI Components**: Interactive sliders, buttons for staged execution, and wide layout for data visualization
- **Visualization**: Matplotlib for charts, ReportLab for PDF generation

### Backend Architecture
- **Pipeline Pattern**: Five-stage sequential processing pipeline
  - Stage 1: Data Ingestion - Simulates Bloomberg API data fetching
  - Stage 2: Core Analysis - Time-series and rule-based risk scoring
  - Stage 2.5: ML Analysis - Anomaly detection and risk prediction using machine learning
  - Stage 3: Sentiment Analysis - NLP analysis on RED-flagged assets only
  - Stage 4: Report Generation - PDF report creation with visualizations and ML insights
- **Modular Design**: Each pipeline stage is encapsulated in its own engine class
- **Data Flow**: Linear pipeline with staged transformations (ingestion → analysis → ML analysis → sentiment → reporting)

### Risk Analysis Logic
- **Threshold-based Classification**: Uses volatility, drawdown, and volume metrics to assign GREEN/YELLOW/RED ratings
- **Selective Processing**: Sentiment analysis only runs on RED-flagged assets for efficiency
- **Portfolio-level Metrics**: Calculates portfolio weights and correlation analysis

### Data Storage Solutions
- **In-Memory**: Uses pandas DataFrames for data manipulation
- **File System**: 
  - Reports stored in `/reports` directory
  - Charts stored in `/charts` directory
- **Session State**: Streamlit session state stores pipeline results between reruns
- **No Database**: Application is stateless with no persistent database

### Key Technologies
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: Scikit-learn (Isolation Forest, Random Forest Classifier)
- **NLP**: TextBlob for sentiment analysis
- **Visualization**: Matplotlib for chart generation
- **PDF Generation**: ReportLab for comprehensive report creation
- **Web Framework**: Streamlit for interactive UI

### Machine Learning Analysis
- **Anomaly Detection (Isolation Forest)**: 
  - Identifies unusual asset behavior patterns
  - Contamination rate: 15% (expects ~15% anomalies)
  - Anomaly scores: 0-100 scale (higher = more anomalous)
  - Severity levels: LOW, MEDIUM, HIGH, CRITICAL
  - Edge case handling: Guards against division by zero when all scores are identical
- **Risk Prediction (Random Forest Classifier)**:
  - Predicts future risk ratings (GREEN, YELLOW, RED)
  - Model accuracy typically 80-90%
  - Provides prediction confidence and trend analysis (IMPROVING, DETERIORATING, STABLE)
  - Dynamically handles portfolios with 2 or 3 risk classes
  - Includes feature importance ranking to identify key risk drivers
- **Bug Fixes Applied**:
  - Division by zero protection in anomaly score normalization
  - Index bounds handling for portfolios with fewer than 3 risk classes

## External Dependencies

### Third-party Libraries
- **streamlit**: Web application framework and UI components
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and array operations
- **scikit-learn**: Machine learning library (Isolation Forest, Random Forest, preprocessing)
- **matplotlib**: Chart and graph generation
- **reportlab**: PDF document generation with tables and charts
- **textblob**: Natural language processing for sentiment analysis

### Simulated External Services
- **Bloomberg API**: Simulated via `MockBloombergData` class - generates realistic financial data
  - Asset pricing data
  - Historical price series (252 trading days)
  - Trading volume data
  - Market capitalization
  - Bloomberg IDs (BBG identifiers)

### Data Sources (Simulated)
- **News Sources**: Simulated news feeds from Reuters, Bloomberg News, Financial Times, WSJ, MarketWatch, Yahoo Finance, CNBC, Seeking Alpha
- **Market Data**: Mock time-series data for prices, volumes, and technical indicators

### File System Dependencies
- Requires write access for:
  - `/reports` directory for PDF storage
  - `/charts` directory for visualization images
- No external database or cloud storage configured