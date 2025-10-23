import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class CoreAnalysisEngine:
    """
    Stage 2: Core Analysis Engine
    Performs time-series and rule-based analysis to determine Asset Quality Ratings
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'volatility_red': 0.4,      # 40% annualized volatility
            'volatility_yellow': 0.25,   # 25% annualized volatility
            'drawdown_red': -0.2,        # -20% maximum drawdown
            'drawdown_yellow': -0.1,     # -10% maximum drawdown
            'volume_decline_red': -0.5,  # -50% volume decline
            'volume_decline_yellow': -0.3, # -30% volume decline
            'correlation_threshold': 0.8  # High correlation threshold
        }
    
    def analyze_portfolio(self, portfolio_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze entire portfolio and generate risk ratings
        
        Args:
            portfolio_data: List of asset data dictionaries
            
        Returns:
            List of analysis results with risk ratings
        """
        analysis_results = []
        
        # Convert to DataFrame for easier analysis
        portfolio_df = pd.DataFrame(portfolio_data)
        
        for asset in portfolio_data:
            asset_analysis = self.analyze_single_asset(asset)
            
            # Add portfolio-level metrics
            asset_analysis.update({
                'portfolio_weight': asset['market_cap'] / portfolio_df['market_cap'].sum(),
                'analysis_timestamp': datetime.now().isoformat()
            })
            
            analysis_results.append(asset_analysis)
        
        # Add correlation analysis
        self.add_correlation_analysis(analysis_results, portfolio_data)
        
        return analysis_results
    
    def analyze_single_asset(self, asset_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a single asset
        
        Args:
            asset_data: Dictionary containing asset information
            
        Returns:
            Analysis results for the asset
        """
        prices = np.array(asset_data['historical_prices'])
        volumes = np.array(asset_data['trading_volume_history'])
        
        # Safety check for empty or single-element price arrays
        if len(prices) < 2:
            # Insufficient data - return zero metrics
            returns = np.array([])
            volatility = 0.0
            max_drawdown = 0.0
        else:
            # Calculate key metrics
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(52)  # Annualized volatility for weekly data
            
            # Maximum drawdown calculation
            if len(returns) > 0:
                cumulative_returns = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            else:
                max_drawdown = 0.0
        
        # Volume analysis
        if len(volumes) > 0:
            volume_ma_short = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            volume_ma_long = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
            volume_decline = (volume_ma_short - volume_ma_long) / volume_ma_long if volume_ma_long != 0 else 0
        else:
            volume_ma_short = 0
            volume_ma_long = 0
            volume_decline = 0
        
        # Price momentum
        price_change_1m = (prices[-1] - prices[-21]) / prices[-21] if len(prices) >= 21 else 0
        price_change_3m = (prices[-1] - prices[-63]) / prices[-63] if len(prices) >= 63 else 0
        price_change_6m = (prices[-1] - prices[-126]) / prices[-126] if len(prices) >= 126 else 0
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        if len(returns) > 0:
            excess_returns = np.mean(returns) * 52 - risk_free_rate  # Annualize using 52 weeks
            sharpe_ratio = excess_returns / volatility if volatility != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Technical indicators
        rsi = self.calculate_rsi(prices)
        beta = self.calculate_beta(returns)
        
        # Risk flag calculation
        risk_flags = self.calculate_risk_flags(
            volatility, max_drawdown, volume_decline, 
            price_change_1m, price_change_3m, sharpe_ratio
        )
        
        # Overall risk rating
        risk_rating = self.determine_risk_rating(risk_flags)
        
        # Debug logging for first few stocks to understand data quality
        if len(prices) < 10:
            print(f"⚠️  {asset_data['symbol']}: Only {len(prices)} price points - insufficient data")
        
        # Log stocks that should be risky but aren't flagged
        expected_risky = ['TSLA', 'NVDA', 'AMD', 'GME', 'AMC', 'BYND', 'NKLA', 'RIVN', 'SPCE']
        if asset_data['symbol'] in expected_risky:
            print(f"🔍 {asset_data['symbol']}: vol={volatility:.3f}, drawdown={max_drawdown:.3f}, "
                  f"rating={risk_rating}, flags={sum(risk_flags.values())}, prices={len(prices)}")
        
        return {
            'symbol': asset_data['symbol'],
            'sector': asset_data['sector'],
            'current_price': asset_data['current_price'],
            'market_cap': asset_data['market_cap'],
            
            # Risk metrics
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'volume_decline': volume_decline,
            'sharpe_ratio': sharpe_ratio,
            'beta': beta,
            'rsi': rsi,
            
            # Performance metrics
            'price_change_1m': price_change_1m,
            'price_change_3m': price_change_3m,
            'price_change_6m': price_change_6m,
            
            # Risk assessment
            'risk_flags': risk_flags,
            'risk_rating': risk_rating,
            'risk_score': len([f for f in risk_flags.values() if f])
        }
    
    def calculate_risk_flags(self, volatility: float, max_drawdown: float, 
                           volume_decline: float, price_change_1m: float, 
                           price_change_3m: float, sharpe_ratio: float) -> Dict[str, bool]:
        """
        Calculate individual risk flags based on thresholds
        
        Args:
            volatility: Annualized volatility
            max_drawdown: Maximum drawdown
            volume_decline: Volume decline ratio
            price_change_1m: 1-month price change
            price_change_3m: 3-month price change
            sharpe_ratio: Risk-adjusted return ratio
            
        Returns:
            Dictionary of risk flags
        """
        return {
            'high_volatility': volatility > self.risk_thresholds['volatility_red'],
            'extreme_drawdown': max_drawdown < self.risk_thresholds['drawdown_red'],
            'volume_collapse': volume_decline < self.risk_thresholds['volume_decline_red'],
            'severe_decline': price_change_1m < -0.15,  # -15% in 1 month
            'extended_decline': price_change_3m < -0.25,  # -25% in 3 months
            'poor_risk_return': sharpe_ratio < -0.5,
            'momentum_breakdown': price_change_1m < -0.1 and price_change_3m < -0.1
        }
    
    def determine_risk_rating(self, risk_flags: Dict[str, bool]) -> str:
        """
        Determine overall risk rating based on flags
        
        Args:
            risk_flags: Dictionary of risk flags
            
        Returns:
            Risk rating: 'RED', 'YELLOW', or 'GREEN'
        """
        critical_flags = ['extreme_drawdown', 'volume_collapse', 'severe_decline']
        warning_flags = ['high_volatility', 'extended_decline', 'poor_risk_return']
        
        # RED rating if any critical flag is true
        if any(risk_flags[flag] for flag in critical_flags):
            return 'RED'
        
        # YELLOW rating if 2 or more warning flags
        warning_count = sum(risk_flags[flag] for flag in warning_flags)
        if warning_count >= 2:
            return 'YELLOW'
        
        # Additional YELLOW conditions
        if risk_flags['momentum_breakdown'] and warning_count >= 1:
            return 'YELLOW'
        
        return 'GREEN'
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index
        
        Args:
            prices: Array of historical prices
            period: RSI calculation period
            
        Returns:
            RSI value
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_beta(self, asset_returns: np.ndarray) -> float:
        """
        Calculate beta relative to market (simulated market returns)
        
        Args:
            asset_returns: Array of asset returns
            
        Returns:
            Beta coefficient
        """
        # Generate mock market returns for beta calculation
        market_returns = np.random.normal(0.0008, 0.012, len(asset_returns))  # ~8% annual return, 12% vol
        
        if len(asset_returns) < 30:  # Need sufficient data points
            return 1.0
        
        covariance = np.cov(asset_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        return beta
    
    def add_correlation_analysis(self, analysis_results: List[Dict], portfolio_data: List[Dict]):
        """
        Add correlation analysis between assets
        
        Args:
            analysis_results: List of analysis results to update
            portfolio_data: Original portfolio data
        """
        # Build correlation matrix
        price_data = {}
        for asset in portfolio_data:
            symbol = asset['symbol']
            prices = np.array(asset['historical_prices'])
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                price_data[symbol] = returns
        
        # Calculate pairwise correlations for each asset
        for i, result in enumerate(analysis_results):
            symbol = result['symbol']
            if symbol in price_data:
                correlations = []
                for other_symbol, other_returns in price_data.items():
                    if other_symbol != symbol:
                        if len(price_data[symbol]) == len(other_returns):
                            corr = np.corrcoef(price_data[symbol], other_returns)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                
                if correlations:
                    result['avg_correlation'] = np.mean(correlations)
                    result['max_correlation'] = np.max(correlations)
                    result['high_correlation_flag'] = result['max_correlation'] > self.risk_thresholds['correlation_threshold']
                else:
                    result['avg_correlation'] = 0.0
                    result['max_correlation'] = 0.0
                    result['high_correlation_flag'] = False
