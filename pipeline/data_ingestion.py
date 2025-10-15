import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
from utils.mock_data import MockBloombergData

class DataIngestionEngine:
    """
    Stage 1: Data Ingestion Engine
    Simulates Bloomberg data ingestion with realistic portfolio data
    """
    
    def __init__(self):
        self.mock_data_generator = MockBloombergData()
        self.connection_status = "Connected"
    
    def ingest_portfolio_data(self, portfolio_size: int = 20) -> List[Dict[str, Any]]:
        """
        Ingest portfolio data from Bloomberg API (simulated)
        
        Args:
            portfolio_size: Number of assets in portfolio
            
        Returns:
            List of dictionaries containing asset data
        """
        print(f"Connecting to Bloomberg API...")
        print(f"Fetching data for {portfolio_size} assets...")
        
        # Generate mock portfolio data with unique stocks
        portfolio_data = []
        selected_stocks = random.sample(self.mock_data_generator.all_stocks, min(portfolio_size, len(self.mock_data_generator.all_stocks)))
        
        for stock in selected_stocks:
            asset_data = self.mock_data_generator.generate_asset_data_for_stock(stock)
            
            # Add historical price data (252 trading days = 1 year)
            historical_data = self.mock_data_generator.generate_historical_prices(
                asset_data['current_price'], 
                days=252
            )
            
            asset_data.update({
                'data_ingestion_timestamp': datetime.now().isoformat(),
                'historical_prices': historical_data['prices'],
                'historical_dates': historical_data['dates'],
                'trading_volume_history': historical_data['volumes'],
                'bloomberg_id': f"BBG{random.randint(100000000, 999999999)}",
                'data_quality_score': random.uniform(0.85, 1.0)
            })
            
            portfolio_data.append(asset_data)
        
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
