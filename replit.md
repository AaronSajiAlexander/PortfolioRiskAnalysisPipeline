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

#### Stage 2.5: ML Analysis Pipeline

The ML analysis engine (`pipeline/ml_analysis.py`) implements two complementary machine learning techniques to enhance portfolio risk assessment:

##### 1. Anomaly Detection using Isolation Forest

**Purpose**: Identifies assets exhibiting unusual behavioral patterns that deviate from typical portfolio behavior.

**Algorithm**: Isolation Forest
- **How it works**: Constructs random decision trees that isolate observations. Anomalies require fewer splits to isolate, resulting in shorter path lengths.
- **Contamination rate**: 0.15 (15%) - expects approximately 15% of assets to be anomalous
- **Input features**: Volatility, maximum drawdown, Sharpe ratio, volume volatility, RSI
- **Random state**: 42 (for reproducibility)

**Output Metrics**:
- **Anomaly Score**: Normalized 0-100 scale (higher = more anomalous)
  - Raw scores from Isolation Forest are normalized using: `(score - min) / (max - min) * 100`
  - Edge case: If all scores identical (max == min), defaults to 30.0 to prevent division by zero
- **Severity Classification**:
  - LOW: Score < 40 (minor deviations)
  - MEDIUM: 40 ≤ Score < 60 (moderate anomalies)
  - HIGH: 60 ≤ Score < 80 (significant anomalies)
  - CRITICAL: Score ≥ 80 (extreme anomalies requiring immediate attention)
- **Contributing Factors**: Top 3 metrics driving anomaly score for each asset

**Use Cases**:
- Detect hidden risks not captured by traditional volatility metrics
- Identify assets with unusual correlation patterns
- Flag potential data quality issues or market anomalies

##### 2. Risk Rating Prediction using Random Forest Classifier

**Purpose**: Predicts future risk ratings and identifies key risk drivers through supervised learning.

**Algorithm**: Random Forest Classifier
- **How it works**: Ensemble of 100 decision trees that vote on risk classification
- **Input features**: Volatility, maximum drawdown, Sharpe ratio, volume volatility, RSI
- **Target variable**: Current risk rating (GREEN, YELLOW, RED)
- **Train-test split**: 80% training, 20% testing with stratified sampling
- **Random state**: 42 (for reproducibility)

**Model Training Process**:
1. Extracts feature matrix (X) and target labels (y) from portfolio data
2. Splits data using stratified sampling to maintain class distribution
3. Trains Random Forest on training set
4. Evaluates accuracy on test set
5. Dynamically adapts to number of risk classes present (2 or 3 classes)

**Output Metrics**:
- **Model Accuracy**: Percentage of correct predictions on test set (typically 80-90%)
- **Predicted Risk Rating**: GREEN, YELLOW, or RED classification
- **Prediction Confidence**: Probability score (0-100%) for the predicted class
- **Risk Trend Analysis**:
  - IMPROVING: Predicted rating better than current (e.g., RED → YELLOW)
  - DETERIORATING: Predicted rating worse than current (e.g., GREEN → YELLOW)
  - STABLE: Predicted rating matches current rating
- **Feature Importance**: Ranked list of metrics by predictive power
  - Shows which metrics (volatility, drawdown, etc.) most influence risk ratings
  - Useful for understanding portfolio risk drivers

**Edge Case Handling**:
- **Insufficient samples**: Requires minimum samples for train-test split
- **Limited risk classes**: Dynamically handles portfolios with only 2 risk classes (e.g., only GREEN and RED assets, no YELLOW)
  - Uses `model.classes_` to identify actual classes present
  - Prevents index out of bounds errors when accessing prediction probabilities
- **Class imbalance**: Stratified sampling maintains representative class distribution

##### Integration in Pipeline

**Execution Flow**:
1. Receives analyzed portfolio data from Stage 2 (Core Analysis)
2. Extracts ML features: volatility, max_drawdown, sharpe_ratio, volume_volatility, RSI
3. Runs Isolation Forest for anomaly detection
4. Trains Random Forest classifier for risk prediction
5. Generates ML insights payload with all results
6. Updates progress (50% → 70%)
7. Passes enriched data to Stage 3 (Sentiment Analysis)

**Data Structure Output**:
```python
{
    'anomaly_detection': {
        'total_anomalies': int,
        'critical_count': int,
        'high_count': int,
        'results': [
            {
                'ticker': str,
                'score': float,
                'severity': str,
                'top_factors': [str, str, str]
            }
        ]
    },
    'risk_prediction': {
        'model_accuracy': float,
        'predictions': [
            {
                'ticker': str,
                'current_rating': str,
                'predicted_rating': str,
                'confidence': float,
                'trend': str
            }
        ],
        'feature_importance': [
            {'feature': str, 'importance': float}
        ]
    }
}
```

##### Performance Considerations

- **Computation time**: ~1-3 seconds for portfolios of 10-100 assets
- **Memory usage**: Minimal (operates on feature matrices)
- **Scalability**: Linear with portfolio size

##### Known Limitations

1. **Training data dependency**: Model accuracy depends on portfolio diversity
2. **Historical bias**: Predictions based on current portfolio characteristics only
3. **Class imbalance**: Very small portfolios may have insufficient samples for robust training
4. **Feature engineering**: Uses only 5 core metrics; additional features could improve accuracy

##### Bug Fixes & Safeguards

1. **Division by Zero Protection** (Line ~150):
   - Issue: When all anomaly scores identical, normalization fails
   - Fix: Check if `max_score == min_score`, default to 30.0
   - Impact: Prevents NaN propagation in severity calculations

2. **Index Out of Bounds Protection** (Line ~225):
   - Issue: Accessing `prediction_proba[:, 2]` fails when only 2 risk classes exist
   - Fix: Use `model.classes_` to get actual number of classes, access indices dynamically
   - Impact: Handles portfolios with GREEN+YELLOW, GREEN+RED, or YELLOW+RED combinations

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