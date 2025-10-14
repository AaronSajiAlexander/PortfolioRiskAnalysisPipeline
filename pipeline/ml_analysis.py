import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MLAnalysisEngine:
    """
    Machine Learning Analysis Engine
    Performs anomaly detection and risk prediction on portfolio assets
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = None
        self.risk_predictor = None
        self.feature_columns = [
            'volatility', 'max_drawdown', 'volume_decline', 'sharpe_ratio', 
            'beta', 'rsi', 'price_change_1m', 'price_change_3m', 'price_change_6m'
        ]
    
    def analyze_portfolio_ml(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive ML analysis on portfolio
        
        Args:
            analysis_results: Results from core analysis engine
            
        Returns:
            Dictionary containing ML analysis results
        """
        # Prepare feature matrix
        features_df = self._prepare_features(analysis_results)
        
        # Perform anomaly detection
        anomaly_results = self._detect_anomalies(features_df, analysis_results)
        
        # Perform risk prediction
        risk_prediction_results = self._predict_risk_ratings(features_df, analysis_results)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features_df, analysis_results)
        
        # Generate ML insights summary
        ml_summary = self._generate_ml_summary(
            anomaly_results, risk_prediction_results, feature_importance
        )
        
        return {
            'anomaly_detection': anomaly_results,
            'risk_prediction': risk_prediction_results,
            'feature_importance': feature_importance,
            'ml_summary': ml_summary,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _prepare_features(self, analysis_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare feature matrix for ML analysis
        
        Args:
            analysis_results: Analysis results from core engine
            
        Returns:
            DataFrame with features for ML
        """
        features_data = []
        
        for asset in analysis_results:
            feature_row = {
                'symbol': asset['symbol'],
                'sector': asset['sector'],
                'current_price': asset['current_price'],
                'market_cap': asset['market_cap'],
                'risk_rating': asset['risk_rating']
            }
            
            # Add numerical features
            for feature in self.feature_columns:
                feature_row[feature] = asset.get(feature, 0)
            
            # Add correlation features if available
            feature_row['avg_correlation'] = asset.get('avg_correlation', 0)
            feature_row['max_correlation'] = asset.get('max_correlation', 0)
            
            features_data.append(feature_row)
        
        return pd.DataFrame(features_data)
    
    def _detect_anomalies(self, features_df: pd.DataFrame, 
                         analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            features_df: Feature DataFrame
            analysis_results: Original analysis results
            
        Returns:
            List of anomaly detection results
        """
        # Select features for anomaly detection
        X = features_df[self.feature_columns].fillna(0)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.15,  # Expect ~15% anomalies
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        anomaly_labels = self.anomaly_detector.fit_predict(X_scaled)
        
        # Get anomaly scores (lower score = more anomalous)
        anomaly_scores = self.anomaly_detector.score_samples(X_scaled)
        
        # Normalize scores to 0-100 scale (higher = more anomalous)
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        
        # Guard against division by zero when all scores are identical
        if max_score == min_score:
            # All assets have same anomaly score - treat as low anomaly
            normalized_scores = np.full(len(anomaly_scores), 30.0)
        else:
            normalized_scores = 100 * (1 - (anomaly_scores - min_score) / (max_score - min_score))
        
        # Compile results
        anomaly_results = []
        for i, asset in enumerate(analysis_results):
            is_anomaly = anomaly_labels[i] == -1
            anomaly_score = normalized_scores[i]
            
            # Determine anomaly severity
            if anomaly_score >= 80:
                severity = 'CRITICAL'
            elif anomaly_score >= 60:
                severity = 'HIGH'
            elif anomaly_score >= 40:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'
            
            # Identify which features contribute most to anomaly
            contributing_features = self._identify_anomalous_features(
                features_df.iloc[i], features_df
            )
            
            anomaly_results.append({
                'symbol': asset['symbol'],
                'sector': asset['sector'],
                'is_anomaly': is_anomaly,
                'anomaly_score': round(anomaly_score, 2),
                'severity': severity,
                'risk_rating': asset['risk_rating'],
                'contributing_features': contributing_features,
                'recommendation': self._get_anomaly_recommendation(is_anomaly, anomaly_score)
            })
        
        return sorted(anomaly_results, key=lambda x: x['anomaly_score'], reverse=True)
    
    def _identify_anomalous_features(self, asset_features: pd.Series, 
                                    all_features: pd.DataFrame) -> List[str]:
        """
        Identify which features make an asset anomalous
        
        Args:
            asset_features: Features for single asset
            all_features: Features for all assets
            
        Returns:
            List of anomalous feature names
        """
        anomalous_features = []
        
        for feature in self.feature_columns:
            asset_value = asset_features[feature]
            mean_value = all_features[feature].mean()
            std_value = all_features[feature].std()
            
            # Check if value is more than 2 standard deviations from mean
            if std_value > 0:
                z_score = abs((asset_value - mean_value) / std_value)
                if z_score > 2:
                    anomalous_features.append(feature)
        
        return anomalous_features[:3]  # Return top 3
    
    def _predict_risk_ratings(self, features_df: pd.DataFrame, 
                             analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict future risk ratings using ML classifier
        
        Args:
            features_df: Feature DataFrame
            analysis_results: Original analysis results
            
        Returns:
            Risk prediction results
        """
        # Prepare training data
        X = features_df[self.feature_columns].fillna(0)
        y = features_df['risk_rating'].map({'GREEN': 0, 'YELLOW': 1, 'RED': 2})
        
        # Check if we have enough data and diversity
        if len(X) < 10 or len(y.unique()) < 2:
            return {
                'model_trained': False,
                'predictions': [],
                'accuracy': 0.0,
                'message': 'Insufficient data or risk diversity for prediction model'
            }
        
        # Split data for validation
        if len(X) >= 15:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train Random Forest Classifier
        self.risk_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.risk_predictor.fit(X_train, y_train)
        
        # Get the actual classes the model learned
        actual_classes = self.risk_predictor.classes_
        reverse_label_map = {0: 'GREEN', 1: 'YELLOW', 2: 'RED'}
        class_to_label = {i: reverse_label_map[actual_classes[i]] for i in range(len(actual_classes))}
        
        # Calculate accuracy
        train_accuracy = self.risk_predictor.score(X_train, y_train)
        test_accuracy = self.risk_predictor.score(X_test, y_test) if len(X) >= 15 else train_accuracy
        
        # Make predictions with probability
        predictions = self.risk_predictor.predict(X)
        prediction_proba = self.risk_predictor.predict_proba(X)
        
        # Compile prediction results
        prediction_results = []
        
        for i, asset in enumerate(analysis_results):
            # Map prediction index to actual class label
            predicted_class_idx = np.where(actual_classes == predictions[i])[0][0]
            predicted_rating = class_to_label[predicted_class_idx]
            actual_rating = asset['risk_rating']
            
            # Get confidence (probability of predicted class)
            confidence = prediction_proba[i][predicted_class_idx] * 100
            
            # Check if prediction differs from current rating
            rating_change = predicted_rating != actual_rating
            
            # Determine if trend is improving or deteriorating
            rating_order = {'GREEN': 0, 'YELLOW': 1, 'RED': 2}
            if rating_change:
                if rating_order[predicted_rating] > rating_order[actual_rating]:
                    trend = 'DETERIORATING'
                else:
                    trend = 'IMPROVING'
            else:
                trend = 'STABLE'
            
            # Build risk probabilities dict based on actual classes present
            risk_probabilities = {}
            for class_idx, class_label in class_to_label.items():
                risk_probabilities[class_label] = round(prediction_proba[i][class_idx] * 100, 1)
            
            # Fill in missing classes with 0
            for label in ['GREEN', 'YELLOW', 'RED']:
                if label not in risk_probabilities:
                    risk_probabilities[label] = 0.0
            
            prediction_results.append({
                'symbol': asset['symbol'],
                'sector': asset['sector'],
                'current_rating': actual_rating,
                'predicted_rating': predicted_rating,
                'confidence': round(confidence, 1),
                'rating_change': rating_change,
                'trend': trend,
                'risk_probabilities': risk_probabilities
            })
        
        return {
            'model_trained': True,
            'predictions': prediction_results,
            'train_accuracy': round(train_accuracy * 100, 1),
            'test_accuracy': round(test_accuracy * 100, 1),
            'total_assets_analyzed': len(prediction_results),
            'rating_changes_predicted': sum(1 for p in prediction_results if p['rating_change'])
        }
    
    def _calculate_feature_importance(self, features_df: pd.DataFrame, 
                                     analysis_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate feature importance from risk prediction model
        
        Args:
            features_df: Feature DataFrame
            analysis_results: Original analysis results
            
        Returns:
            List of feature importance scores
        """
        if self.risk_predictor is None:
            return []
        
        # Get feature importances
        importances = self.risk_predictor.feature_importances_
        
        # Create feature importance list
        feature_importance = []
        for feature, importance in zip(self.feature_columns, importances):
            feature_importance.append({
                'feature': feature.replace('_', ' ').title(),
                'importance': round(importance * 100, 2),
                'rank': 0  # Will be set after sorting
            })
        
        # Sort by importance and assign ranks
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        for i, feature in enumerate(feature_importance):
            feature['rank'] = i + 1
        
        return feature_importance
    
    def _get_anomaly_recommendation(self, is_anomaly: bool, anomaly_score: float) -> str:
        """
        Generate recommendation based on anomaly detection
        
        Args:
            is_anomaly: Whether asset is flagged as anomaly
            anomaly_score: Anomaly score (0-100)
            
        Returns:
            Recommendation string
        """
        if not is_anomaly or anomaly_score < 40:
            return "Normal behavior pattern - Continue monitoring"
        elif anomaly_score < 60:
            return "Moderate anomaly detected - Review underlying fundamentals"
        elif anomaly_score < 80:
            return "Significant anomaly - Conduct thorough due diligence"
        else:
            return "Critical anomaly - Consider immediate position review"
    
    def _generate_ml_summary(self, anomaly_results: List[Dict], 
                           risk_predictions: Dict, 
                           feature_importance: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary of ML analysis
        
        Args:
            anomaly_results: Anomaly detection results
            risk_predictions: Risk prediction results
            feature_importance: Feature importance scores
            
        Returns:
            ML summary dictionary
        """
        # Anomaly summary
        total_anomalies = sum(1 for a in anomaly_results if a['is_anomaly'])
        critical_anomalies = sum(1 for a in anomaly_results if a['severity'] == 'CRITICAL')
        high_anomalies = sum(1 for a in anomaly_results if a['severity'] == 'HIGH')
        
        # Risk prediction summary
        rating_changes = 0
        deteriorating_count = 0
        improving_count = 0
        
        if risk_predictions.get('model_trained'):
            predictions = risk_predictions['predictions']
            rating_changes = sum(1 for p in predictions if p['rating_change'])
            deteriorating_count = sum(1 for p in predictions if p['trend'] == 'DETERIORATING')
            improving_count = sum(1 for p in predictions if p['trend'] == 'IMPROVING')
        
        # Top risk factors
        top_risk_factors = [f['feature'] for f in feature_importance[:3]] if feature_importance else []
        
        return {
            'total_assets_analyzed': len(anomaly_results),
            'anomaly_summary': {
                'total_anomalies': total_anomalies,
                'critical_anomalies': critical_anomalies,
                'high_anomalies': high_anomalies,
                'anomaly_rate': round(total_anomalies / len(anomaly_results) * 100, 1) if anomaly_results else 0
            },
            'prediction_summary': {
                'model_trained': risk_predictions.get('model_trained', False),
                'rating_changes_predicted': rating_changes,
                'deteriorating_assets': deteriorating_count,
                'improving_assets': improving_count,
                'model_accuracy': risk_predictions.get('test_accuracy', 0)
            },
            'top_risk_factors': top_risk_factors,
            'key_insights': self._generate_key_insights(
                total_anomalies, critical_anomalies, deteriorating_count, top_risk_factors
            )
        }
    
    def _generate_key_insights(self, total_anomalies: int, critical_anomalies: int,
                               deteriorating_count: int, top_factors: List[str]) -> List[str]:
        """
        Generate key insights from ML analysis
        
        Args:
            total_anomalies: Number of anomalies detected
            critical_anomalies: Number of critical anomalies
            deteriorating_count: Number of deteriorating assets
            top_factors: Top risk factors
            
        Returns:
            List of insight strings
        """
        insights = []
        
        if critical_anomalies > 0:
            insights.append(f"{critical_anomalies} assets show critical anomalous behavior requiring immediate review")
        
        if deteriorating_count > 0:
            insights.append(f"{deteriorating_count} assets predicted to deteriorate in risk rating")
        
        if total_anomalies > 0:
            insights.append(f"Anomaly detection identified {total_anomalies} assets with unusual patterns")
        
        if top_factors:
            insights.append(f"Key risk drivers: {', '.join(top_factors)}")
        
        if not insights:
            insights.append("Portfolio shows stable risk patterns with no critical ML alerts")
        
        return insights
