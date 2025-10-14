# Portfolio Risk Analysis Pipeline

## Overview

This is a multi-stage portfolio risk analysis application built with Streamlit that simulates a Bloomberg data integration pipeline. The system analyzes financial portfolios through four sequential stages: data ingestion, core risk analysis, sentiment analysis for high-risk assets, and comprehensive PDF report generation. The application uses mock data to simulate real-world financial data feeds and applies time-series analysis, rule-based risk scoring, and NLP-based sentiment analysis to generate actionable investment insights.

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
- **Pipeline Pattern**: Four-stage sequential processing pipeline
  - Stage 1: Data Ingestion - Simulates Bloomberg API data fetching
  - Stage 2: Core Analysis - Time-series and rule-based risk scoring
  - Stage 3: Sentiment Analysis - NLP analysis on RED-flagged assets only
  - Stage 4: Report Generation - PDF report creation with visualizations
- **Modular Design**: Each pipeline stage is encapsulated in its own engine class
- **Data Flow**: Linear pipeline with staged transformations (ingestion → analysis → sentiment → reporting)

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
- **NLP**: TextBlob for sentiment analysis
- **Visualization**: Matplotlib for chart generation
- **PDF Generation**: ReportLab for comprehensive report creation
- **Web Framework**: Streamlit for interactive UI

## External Dependencies

### Third-party Libraries
- **streamlit**: Web application framework and UI components
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and array operations
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