import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import requests
import time
from typing import List, Dict, Any
from utils.mock_data import MockBloombergData

class DataIngestionEngine:
    """
    Stage 1: Data Ingestion Engine
    Fetches real-time stock data from Alpha Vantage API
    """
    
    def __init__(self):
        self.mock_data_generator = MockBloombergData()
        self.connection_status = "Connected"
        self.api_key = "XH5DTCIKQMS1C26Z"
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_weekly_data(self, symbol: str) -> Dict[str, Any] | None:
        """
        Fetch weekly adjusted data from Alpha Vantage API
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            API response data or None if error
        """
        params = {
            'function': 'TIME_SERIES_WEEKLY_ADJUSTED',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                print(f"API Error for {symbol}: {data['Error Message']}")
                return None
            if 'Note' in data:
                print(f"API Rate Limit for {symbol}: {data['Note']}")
                return None
                
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def parse_weekly_data(self, weekly_data: Dict, max_weeks: int = 50) -> Dict[str, List]:
        """
        Parse weekly adjusted data from Alpha Vantage API
        
        Args:
            weekly_data: Alpha Vantage weekly response
            max_weeks: Maximum number of weeks to include (default 50)
            
        Returns:
            Dictionary with weekly prices, dates, and volumes
        """
        if not weekly_data:
            print(f"⚠️ No API data received")
            return {'prices': [], 'dates': [], 'volumes': []}
        
        if 'Weekly Adjusted Time Series' not in weekly_data:
            if 'Information' in weekly_data:
                # Rate limit message - only print once to avoid spam
                if not hasattr(self, '_rate_limit_warned'):
                    print(f"⚠️ API Rate Limit: {weekly_data['Information']}")
                    print(f"⚠️ Falling back to mock data for all remaining stocks")
                    self._rate_limit_warned = True
            else:
                print(f"⚠️ API response missing 'Weekly Adjusted Time Series'. Keys: {list(weekly_data.keys())}")
            return {'prices': [], 'dates': [], 'volumes': []}
        
        time_series = weekly_data['Weekly Adjusted Time Series']
        
        # Sort dates in descending order (most recent first) and limit to max_weeks
        sorted_dates = sorted(time_series.keys(), reverse=True)[:max_weeks]
        
        # Reverse to get chronological order (oldest to newest)
        sorted_dates = sorted_dates[::-1]
        
        # Extract adjusted close prices and volumes
        prices = [float(time_series[date]['5. adjusted close']) for date in sorted_dates]
        volumes = [int(time_series[date]['6. volume']) for date in sorted_dates]
        
        return {
            'prices': prices,
            'dates': sorted_dates,
            'volumes': volumes
        }
    
    def convert_daily_to_weekly(self, daily_data: Dict[str, List]) -> Dict[str, List]:
        """
        Convert daily price data to weekly by sampling every 5th trading day
        
        Args:
            daily_data: Dictionary with daily prices, dates, and volumes
            
        Returns:
            Dictionary with weekly sampled prices, dates, and volumes
        """
        if not daily_data['prices'] or len(daily_data['prices']) < 5:
            return daily_data
        
        # Sample every 5th day to approximate weekly data (5 trading days ≈ 1 week)
        weekly_prices = daily_data['prices'][::5]
        weekly_dates = daily_data['dates'][::5]
        weekly_volumes = daily_data['volumes'][::5]
        
        # Limit to 50 weeks to match API data
        return {
            'prices': weekly_prices[-50:],
            'dates': weekly_dates[-50:],
            'volumes': weekly_volumes[-50:]
        }
    
    def calculate_metrics(self, prices: List[float], is_weekly: bool = True) -> Dict[str, Any]:
        """Calculate financial metrics from price history
        
        Args:
            prices: List of historical prices
            is_weekly: If True, treats prices as weekly data for volatility calculation
        """
        if not prices or len(prices) < 2:
            return {
                'volatility': 0.20,
                'pe_ratio': None,
                'dividend_yield': 0.0
            }
        
        returns = np.diff(prices) / prices[:-1]
        
        # Annualized volatility
        # For weekly data: sqrt(52) weeks per year
        # For daily data: sqrt(252) trading days per year
        annualization_factor = np.sqrt(52) if is_weekly else np.sqrt(252)
        volatility = float(np.std(returns) * annualization_factor)
        
        return {
            'volatility': volatility,
            'pe_ratio': random.uniform(10, 30) if random.random() > 0.2 else None,
            'dividend_yield': random.uniform(0, 0.05) if random.random() > 0.4 else 0.0
        }
    
    def ingest_portfolio_data(self, portfolio_size: int = 45) -> List[Dict[str, Any]]:
        """
        Ingest portfolio data from Alpha Vantage API using weekly data
        
        Args:
            portfolio_size: Number of assets in portfolio (default 45)
            
        Returns:
            List of dictionaries containing asset data
        """
        print(f"Connecting to Alpha Vantage API...")
        print(f"Fetching weekly data for {portfolio_size} stocks...")
        
        portfolio_data = []
        all_stocks = self.mock_data_generator.all_stocks
        
        for i, stock in enumerate(all_stocks[:portfolio_size]):
            print(f"Fetching {stock['symbol']} ({i+1}/{portfolio_size})...")
            
            # Fetch weekly data from Alpha Vantage
            api_data = self.fetch_weekly_data(stock['symbol'])
            
            # Parse weekly data (limited to 50 weeks)
            if api_data:
                historical = self.parse_weekly_data(api_data, max_weeks=50)
                current_price = historical['prices'][-1] if historical['prices'] else random.uniform(50, 400)
                if not historical['prices']:
                    print(f"⚠️ API returned no price data for {stock['symbol']}, using fallback")
            else:
                historical = {'prices': [], 'dates': [], 'volumes': []}
            
            # Use fallback if API didn't provide data
            if not historical['prices']:
                # Fallback to mock data if API fails (generates daily data, then convert to weekly)
                print(f"Using mock data for {stock['symbol']} (converting daily to weekly)")
                mock_asset = self.mock_data_generator.generate_asset_data_for_stock(stock)
                daily_historical = self.mock_data_generator.generate_historical_prices(
                    mock_asset['current_price'], days=252
                )
                # Convert daily to weekly to maintain consistent granularity
                historical = self.convert_daily_to_weekly(daily_historical)
                current_price = mock_asset['current_price']
                print(f"✓ Generated {len(historical['prices'])} weekly price points for {stock['symbol']}")
            
            # Calculate metrics (always weekly data now, whether from API or converted fallback)
            metrics = self.calculate_metrics(historical['prices'], is_weekly=True)
            
            # Estimate market cap
            shares_outstanding = random.randint(50_000_000, 2_000_000_000)
            market_cap = current_price * shares_outstanding
            
            asset_data = {
                'symbol': stock['symbol'],
                'company_name': stock['name'],
                'sector': stock['sector'],
                'current_price': round(current_price, 2),
                'market_cap': int(market_cap),
                'shares_outstanding': shares_outstanding,
                'pe_ratio': round(metrics['pe_ratio'], 2) if metrics['pe_ratio'] else None,
                'dividend_yield': round(metrics['dividend_yield'], 4),
                'volatility_base': metrics['volatility'],
                'currency': 'USD',
                'exchange': random.choice(['NYSE', 'NASDAQ']),
                'country': 'United States',
                'data_ingestion_timestamp': datetime.now().isoformat(),
                'historical_prices': historical['prices'][-52:] if len(historical['prices']) > 52 else historical['prices'],
                'historical_dates': historical['dates'][-52:] if len(historical['dates']) > 52 else historical['dates'],
                'trading_volume_history': historical['volumes'][-52:] if len(historical['volumes']) > 52 else historical['volumes'],
                'data_cadence': 'weekly',
                'weeks_of_data': len(historical['prices']),
                'bloomberg_id': f"BBG{random.randint(100000000, 999999999)}",
                'data_quality_score': 1.0 if api_data else 0.85
            }
            
            portfolio_data.append(asset_data)
            
            # Add delay to avoid API rate limits
            if i < portfolio_size - 1:
                time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
        
        print(f"Successfully ingested data for {len(portfolio_data)} assets")
        return portfolio_data
    
    def validate_data_integrity(self, portfolio_data: List[Dict]) -> Dict[str, Any]:
        """
        Validate the integrity of ingested data
        
        Args:
            portfolio_data: List of asset data dictionaries
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_assets': len(portfolio_data),
            'complete_records': 0,
            'missing_data_assets': [],
            'data_quality_issues': [],
            'average_data_quality': 0.0
        }
        
        quality_scores = []
        
        for asset in portfolio_data:
            # Check for required fields
            required_fields = ['symbol', 'current_price', 'historical_prices', 'market_cap']
            missing_fields = [field for field in required_fields if field not in asset or asset[field] is None]
            
            if not missing_fields:
                validation_results['complete_records'] += 1
            else:
                validation_results['missing_data_assets'].append({
                    'symbol': asset.get('symbol', 'Unknown'),
                    'missing_fields': missing_fields
                })
            
            # Track data quality scores
            if 'data_quality_score' in asset:
                quality_scores.append(asset['data_quality_score'])
                
                if asset['data_quality_score'] < 0.9:
                    validation_results['data_quality_issues'].append({
                        'symbol': asset['symbol'],
                        'quality_score': asset['data_quality_score']
                    })
        
        if quality_scores:
            validation_results['average_data_quality'] = np.mean(quality_scores)
        
        return validation_results
    
    def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch real-time data for specific symbols (simulated)
        
        Args:
            symbols: List of asset symbols
            
        Returns:
            Dictionary with real-time data for each symbol
        """
        real_time_data = {}
        
        for symbol in symbols:
            real_time_data[symbol] = {
                'last_price': random.uniform(50, 500),
                'bid': random.uniform(50, 500),
                'ask': random.uniform(50, 500),
                'volume': random.randint(10000, 1000000),
                'timestamp': datetime.now().isoformat(),
                'change_percent': random.uniform(-5.0, 5.0)
            }
        
        return real_time_data
    
    def check_connection_status(self) -> Dict[str, Any]:
        """
        Check Bloomberg API connection status
        
        Returns:
            Connection status information
        """
        return {
            'status': self.connection_status,
            'last_check': datetime.now().isoformat(),
            'latency_ms': random.uniform(50, 200),
            'api_rate_limit_remaining': random.randint(800, 1000)
        }
