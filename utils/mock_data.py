import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class MockBloombergData:
    """
    Mock data generator to simulate Bloomberg data feeds
    Generates realistic portfolio and market data for demonstration
    """
    
    def __init__(self):
        self.sectors = [
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary',
            'Industrial', 'Energy', 'Materials', 'Utilities', 'Real Estate', 'Consumer Staples'
        ]
        
        self.company_prefixes = [
            'Global', 'Advanced', 'United', 'American', 'International', 'Pacific',
            'National', 'Premier', 'Superior', 'Dynamic', 'Strategic', 'Innovative',
            'First', 'Capital', 'Metro', 'Regional', 'Consolidated', 'Alliance'
        ]
        
        self.company_suffixes = [
            'Corp', 'Inc', 'LLC', 'Group', 'Holdings', 'Systems', 'Technologies',
            'Solutions', 'Industries', 'Enterprises', 'Partners', 'Capital',
            'Resources', 'Services', 'International', 'Global'
        ]
        
        self.symbol_prefixes = ['A', 'B', 'C', 'D', 'G', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z']
    
    def generate_asset_data(self) -> Dict[str, Any]:
        """
        Generate mock data for a single asset
        
        Returns:
            Dictionary containing asset information
        """
        # Generate company name and symbol
        prefix = random.choice(self.company_prefixes)
        suffix = random.choice(self.company_suffixes)
        company_name = f"{prefix} {suffix}"
        
        # Generate symbol (2-4 characters)
        symbol_length = random.choice([2, 3, 3, 4])  # Bias toward 3-character symbols
        symbol = random.choice(self.symbol_prefixes)
        for _ in range(symbol_length - 1):
            symbol += random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        # Sector and basic info
        sector = random.choice(self.sectors)
        
        # Price data - realistic ranges based on sector
        if sector in ['Technology', 'Healthcare']:
            base_price = random.uniform(50, 400)
            volatility_base = random.uniform(0.15, 0.45)
        elif sector in ['Financial Services', 'Energy']:
            base_price = random.uniform(30, 150)
            volatility_base = random.uniform(0.20, 0.50)
        elif sector in ['Utilities', 'Consumer Staples']:
            base_price = random.uniform(40, 120)
            volatility_base = random.uniform(0.10, 0.25)
        else:
            base_price = random.uniform(35, 200)
            volatility_base = random.uniform(0.15, 0.35)
        
        # Market cap based on price and shares outstanding
        shares_outstanding = random.randint(50_000_000, 2_000_000_000)
        market_cap = base_price * shares_outstanding
        
        # Additional financial metrics
        pe_ratio = random.uniform(8, 35) if random.random() > 0.1 else None  # 10% chance of no P/E
        dividend_yield = random.uniform(0, 0.06) if random.random() > 0.3 else 0  # 70% pay dividends
        
        return {
            'symbol': symbol,
            'company_name': company_name,
            'sector': sector,
            'current_price': round(base_price, 2),
            'market_cap': int(market_cap),
            'shares_outstanding': shares_outstanding,
            'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
            'dividend_yield': round(dividend_yield, 4),
            'volatility_base': volatility_base,  # Used for historical price generation
            'currency': 'USD',
            'exchange': random.choice(['NYSE', 'NASDAQ', 'NYSE', 'NASDAQ']),  # Bias toward major exchanges
            'country': 'United States'
        }
    
    def generate_historical_prices(self, current_price: float, days: int = 252) -> Dict[str, List]:
        """
        Generate realistic historical price data using geometric Brownian motion
        
        Args:
            current_price: Current/ending price
            days: Number of historical days to generate
            
        Returns:
            Dictionary with prices, dates, and volumes
        """
        # Parameters for price simulation
        dt = 1/252  # Daily time step
        mu = random.uniform(-0.05, 0.15)  # Annual drift (return)
        sigma = random.uniform(0.15, 0.45)  # Annual volatility
        
        # Generate price path working backwards from current price
        prices = [current_price]
        
        for i in range(days - 1):
            # Random walk component
            dW = np.random.normal(0, np.sqrt(dt))
            
            # Geometric Brownian motion
            prev_price = prices[-1]
            new_price = prev_price * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            
            # Add some mean reversion to keep prices reasonable
            if new_price > current_price * 2:
                new_price = current_price * 2 * random.uniform(0.9, 1.0)
            elif new_price < current_price * 0.3:
                new_price = current_price * 0.3 * random.uniform(1.0, 1.1)
            
            prices.append(max(0.01, new_price))  # Ensure positive prices
        
        # Reverse to get chronological order
        prices.reverse()
        
        # Generate corresponding dates
        end_date = datetime.now().date()
        dates = []
        for i in range(days):
            date = end_date - timedelta(days=days-1-i)
            dates.append(date.isoformat())
        
        # Generate trading volumes (correlated with price movements)
        volumes = []
        base_volume = random.randint(100_000, 10_000_000)
        
        for i in range(len(prices)):
            # Volume tends to be higher on days with large price movements
            if i > 0:
                price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                volume_multiplier = 1 + price_change * 3  # Higher volume on volatile days
            else:
                volume_multiplier = 1
            
            daily_volume = int(base_volume * volume_multiplier * random.uniform(0.5, 2.0))
            volumes.append(daily_volume)
        
        return {
            'prices': [round(p, 2) for p in prices],
            'dates': dates,
            'volumes': volumes
        }
    
    def generate_market_indices(self, days: int = 252) -> Dict[str, Dict]:
        """
        Generate mock market index data for correlation analysis
        
        Args:
            days: Number of historical days
            
        Returns:
            Dictionary with market index data
        """
        indices = {
            'S&P 500': {'current_level': 4200, 'volatility': 0.16},
            'NASDAQ': {'current_level': 13000, 'volatility': 0.20},
            'DOW JONES': {'current_level': 34000, 'volatility': 0.15}
        }
        
        market_data = {}
        
        for index_name, params in indices.items():
            historical_data = self.generate_historical_prices(
                params['current_level'], days
            )
            market_data[index_name] = historical_data
        
        return market_data
    
    def generate_economic_indicators(self) -> Dict[str, Any]:
        """
        Generate mock economic indicators
        
        Returns:
            Dictionary with economic indicators
        """
        return {
            'interest_rates': {
                'fed_funds_rate': round(random.uniform(0.25, 5.5), 2),
                '10_year_treasury': round(random.uniform(1.5, 6.0), 2),
                '30_year_treasury': round(random.uniform(2.0, 6.5), 2)
            },
            'economic_metrics': {
                'gdp_growth': round(random.uniform(-2.0, 4.0), 1),
                'unemployment_rate': round(random.uniform(3.5, 8.0), 1),
                'inflation_rate': round(random.uniform(1.0, 6.0), 1),
                'consumer_confidence': round(random.uniform(80, 140), 1)
            },
            'market_metrics': {
                'vix': round(random.uniform(12, 35), 1),
                'dollar_index': round(random.uniform(95, 110), 2),
                'oil_price': round(random.uniform(60, 120), 2),
                'gold_price': round(random.uniform(1800, 2100), 2)
            }
        }
    
    def generate_news_headlines(self, symbol: str, sentiment_bias: str = 'mixed') -> List[Dict[str, Any]]:
        """
        Generate mock news headlines for sentiment analysis
        
        Args:
            symbol: Asset symbol
            sentiment_bias: 'positive', 'negative', or 'mixed'
            
        Returns:
            List of news headline dictionaries
        """
        positive_templates = [
            f"{symbol} reports strong quarterly earnings, beats estimates",
            f"{symbol} announces major partnership with industry leader",
            f"Analysts upgrade {symbol} target price following positive outlook",
            f"{symbol} receives FDA approval for breakthrough product",
            f"{symbol} CEO outlines ambitious growth strategy at investor day"
        ]
        
        negative_templates = [
            f"{symbol} misses quarterly expectations, guidance lowered",
            f"Regulatory concerns weigh on {symbol} stock price",
            f"{symbol} faces increased competition in core markets",
            f"Credit rating agency places {symbol} on negative watch",
            f"{symbol} announces restructuring, job cuts expected"
        ]
        
        neutral_templates = [
            f"{symbol} announces quarterly dividend payment",
            f"{symbol} schedules earnings call for next week",
            f"Trading volume in {symbol} remains elevated",
            f"{symbol} stock included in new ESG index",
            f"Analyst maintains neutral rating on {symbol} shares"
        ]
        
        headlines = []
        num_headlines = random.randint(10, 25)
        
        for _ in range(num_headlines):
            if sentiment_bias == 'positive':
                template_choice = random.choices(
                    [positive_templates, neutral_templates, negative_templates],
                    weights=[0.6, 0.3, 0.1]
                )[0]
            elif sentiment_bias == 'negative':
                template_choice = random.choices(
                    [positive_templates, neutral_templates, negative_templates],
                    weights=[0.1, 0.2, 0.7]
                )[0]
            else:  # mixed
                template_choice = random.choices(
                    [positive_templates, neutral_templates, negative_templates],
                    weights=[0.3, 0.4, 0.3]
                )[0]
            
            headline = random.choice(template_choice)
            
            # Generate publication date
            days_ago = random.randint(1, 365)
            pub_date = datetime.now() - timedelta(days=days_ago)
            
            headlines.append({
                'headline': headline,
                'publication_date': pub_date.isoformat(),
                'source': random.choice([
                    'Reuters', 'Bloomberg', 'MarketWatch', 'Yahoo Finance',
                    'CNBC', 'Financial Times', 'Wall Street Journal'
                ])
            })
        
        return headlines
