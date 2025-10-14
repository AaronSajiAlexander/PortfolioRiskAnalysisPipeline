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
        
        # Validate ML results
        validation_results = self._validate_ml_results(
            anomaly_results, risk_prediction_results, feature_importance, features_df
        )
        
        return {
            'anomaly_detection': anomaly_results,
            'risk_prediction': risk_prediction_results,
            'feature_importance': feature_importance,
            'ml_summary': ml_summary,
            'validation': validation_results,
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
    
    def _validate_ml_results(self, anomaly_results: List[Dict], 
                            risk_predictions: Dict, 
                            feature_importance: List[Dict],
                            features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate ML results for quality and consistency
        
        Args:
            anomaly_results: Anomaly detection results
            risk_predictions: Risk prediction results
            feature_importance: Feature importance scores
            features_df: Feature DataFrame
            
        Returns:
            Validation results with checks and warnings
        """
        validation_checks = []
        warnings = []
        overall_status = 'PASS'
        
        # 1. Validate Anomaly Detection Results
        anomaly_check = self._validate_anomaly_detection(anomaly_results)
        validation_checks.append(anomaly_check)
        if anomaly_check['status'] == 'WARNING':
            warnings.extend(anomaly_check['issues'])
        if anomaly_check['status'] == 'FAIL':
            overall_status = 'FAIL'
        
        # 2. Validate Risk Prediction Results
        prediction_check = self._validate_risk_predictions(risk_predictions)
        validation_checks.append(prediction_check)
        if prediction_check['status'] == 'WARNING':
            warnings.extend(prediction_check['issues'])
        if prediction_check['status'] == 'FAIL':
            overall_status = 'FAIL'
        
        # 3. Validate Feature Quality
        feature_check = self._validate_feature_quality(features_df)
        validation_checks.append(feature_check)
        if feature_check['status'] == 'WARNING':
            warnings.extend(feature_check['issues'])
        if feature_check['status'] == 'FAIL':
            overall_status = 'FAIL'
        
        # 4. Validate Feature Importance
        importance_check = self._validate_feature_importance(feature_importance)
        validation_checks.append(importance_check)
        if importance_check['status'] == 'WARNING':
            warnings.extend(importance_check['issues'])
        
        # Set overall status based on checks
        if overall_status != 'FAIL' and warnings:
            overall_status = 'WARNING'
        
        return {
            'overall_status': overall_status,
            'validation_checks': validation_checks,
            'warnings': warnings,
            'total_checks': len(validation_checks),
            'passed_checks': sum(1 for c in validation_checks if c['status'] == 'PASS'),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _validate_anomaly_detection(self, anomaly_results: List[Dict]) -> Dict[str, Any]:
        """Validate anomaly detection results"""
        issues = []
        
        # Check anomaly scores are in valid range
        for result in anomaly_results:
            score = result['anomaly_score']
            if not 0 <= score <= 100:
                issues.append(f"Invalid anomaly score {score} for {result['symbol']}")
        
        # Validate severity classifications
        for result in anomaly_results:
            score = result['anomaly_score']
            severity = result['severity']
            
            expected_severity = None
            if score >= 80:
                expected_severity = 'CRITICAL'
            elif score >= 60:
                expected_severity = 'HIGH'
            elif score >= 40:
                expected_severity = 'MEDIUM'
            else:
                expected_severity = 'LOW'
            
            if severity != expected_severity:
                issues.append(f"Severity mismatch for {result['symbol']}: score {score} has severity {severity}, expected {expected_severity}")
        
        # Check anomaly rate is reasonable (expected ~15%)
        anomaly_count = sum(1 for r in anomaly_results if r['is_anomaly'])
        anomaly_rate = anomaly_count / len(anomaly_results) * 100 if anomaly_results else 0
        
        if anomaly_rate > 30:
            issues.append(f"High anomaly rate: {anomaly_rate:.1f}% (expected ~15%)")
        elif anomaly_rate < 5 and len(anomaly_results) >= 20:
            issues.append(f"Low anomaly rate: {anomaly_rate:.1f}% (expected ~15%)")
        
        # Validate critical anomalies have high scores
        critical_anomalies = [r for r in anomaly_results if r['severity'] == 'CRITICAL']
        for result in critical_anomalies:
            if result['anomaly_score'] < 80:
                issues.append(f"Critical anomaly {result['symbol']} has score below 80: {result['anomaly_score']}")
        
        status = 'FAIL' if any('Invalid' in i for i in issues) else 'WARNING' if issues else 'PASS'
        
        return {
            'check_name': 'Anomaly Detection Validation',
            'status': status,
            'issues': issues,
            'metrics': {
                'total_assets': len(anomaly_results),
                'anomalies_detected': anomaly_count,
                'anomaly_rate': round(anomaly_rate, 1),
                'critical_count': len(critical_anomalies)
            }
        }
    
    def _validate_risk_predictions(self, risk_predictions: Dict) -> Dict[str, Any]:
        """Validate risk prediction results"""
        issues = []
        
        if not risk_predictions.get('model_trained'):
            return {
                'check_name': 'Risk Prediction Validation',
                'status': 'WARNING',
                'issues': ['Risk prediction model not trained - insufficient data'],
                'metrics': {}
            }
        
        predictions = risk_predictions.get('predictions', [])
        
        # Validate confidence scores
        for pred in predictions:
            confidence = pred['confidence']
            if not 0 <= confidence <= 100:
                issues.append(f"Invalid confidence score {confidence} for {pred['symbol']}")
        
        # Check model accuracy is reasonable
        test_accuracy = risk_predictions.get('test_accuracy', 0)
        if test_accuracy < 50:
            issues.append(f"Low model accuracy: {test_accuracy}% (below 50%)")
        elif test_accuracy > 99:
            issues.append(f"Suspiciously high accuracy: {test_accuracy}% (possible overfitting)")
        
        # Validate trend analysis consistency
        for pred in predictions:
            current = pred['current_rating']
            predicted = pred['predicted_rating']
            trend = pred['trend']
            
            rating_order = {'GREEN': 0, 'YELLOW': 1, 'RED': 2}
            
            if current == predicted and trend != 'STABLE':
                issues.append(f"Trend inconsistency for {pred['symbol']}: same rating but trend is {trend}")
            elif rating_order[predicted] > rating_order[current] and trend != 'DETERIORATING':
                issues.append(f"Trend inconsistency for {pred['symbol']}: worsening rating but trend is {trend}")
            elif rating_order[predicted] < rating_order[current] and trend != 'IMPROVING':
                issues.append(f"Trend inconsistency for {pred['symbol']}: improving rating but trend is {trend}")
        
        # Check risk probabilities sum to ~100%
        for pred in predictions:
            probs = pred['risk_probabilities']
            total_prob = sum(probs.values())
            if not 99 <= total_prob <= 101:
                issues.append(f"Risk probabilities for {pred['symbol']} sum to {total_prob}% (should be ~100%)")
        
        status = 'FAIL' if any('Invalid' in i for i in issues) else 'WARNING' if issues else 'PASS'
        
        return {
            'check_name': 'Risk Prediction Validation',
            'status': status,
            'issues': issues,
            'metrics': {
                'model_accuracy': test_accuracy,
                'predictions_made': len(predictions),
                'rating_changes': sum(1 for p in predictions if p['rating_change'])
            }
        }
    
    def _validate_feature_quality(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature data quality"""
        issues = []
        
        # Check for NaN values
        nan_counts = features_df[self.feature_columns].isna().sum()
        for feature, count in nan_counts.items():
            if count > 0:
                issues.append(f"Feature '{feature}' has {count} NaN values")
        
        # Check for infinite values
        for feature in self.feature_columns:
            if np.isinf(features_df[feature]).any():
                issues.append(f"Feature '{feature}' contains infinite values")
        
        # Check feature variance (low variance may indicate data quality issues)
        for feature in self.feature_columns:
            variance = features_df[feature].var()
            if variance < 0.0001:
                issues.append(f"Feature '{feature}' has very low variance ({variance:.6f})")
        
        status = 'FAIL' if any('NaN' in i or 'infinite' in i for i in issues) else 'WARNING' if issues else 'PASS'
        
        return {
            'check_name': 'Feature Quality Validation',
            'status': status,
            'issues': issues,
            'metrics': {
                'features_checked': len(self.feature_columns),
                'nan_features': int(nan_counts.sum()),
                'total_samples': len(features_df)
            }
        }
    
    def _validate_feature_importance(self, feature_importance: List[Dict]) -> Dict[str, Any]:
        """Validate feature importance results"""
        issues = []
        
        if not feature_importance:
            return {
                'check_name': 'Feature Importance Validation',
                'status': 'WARNING',
                'issues': ['No feature importance calculated'],
                'metrics': {}
            }
        
        # Check importance values sum to ~100%
        total_importance = sum(f['importance'] for f in feature_importance)
        if not 99 <= total_importance <= 101:
            issues.append(f"Feature importances sum to {total_importance}% (should be ~100%)")
        
        # Check for negative importance
        for feature in feature_importance:
            if feature['importance'] < 0:
                issues.append(f"Negative importance for {feature['feature']}: {feature['importance']}")
        
        # Warn if one feature dominates (>60%)
        top_importance = feature_importance[0]['importance'] if feature_importance else 0
        if top_importance > 60:
            issues.append(f"Single feature dominates: {feature_importance[0]['feature']} ({top_importance}%)")
        
        status = 'WARNING' if issues else 'PASS'
        
        return {
            'check_name': 'Feature Importance Validation',
            'status': status,
            'issues': issues,
            'metrics': {
                'total_importance': round(total_importance, 1),
                'top_feature': feature_importance[0]['feature'] if feature_importance else None,
                'top_importance': round(top_importance, 1)
            }
        }
