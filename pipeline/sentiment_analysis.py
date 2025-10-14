import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
from textblob import TextBlob
import re

class SentimentAnalysisEngine:
    """
    Stage 3: Sentiment Analysis Engine
    Analyzes news sentiment for RED-flagged assets only
    """
    
    def __init__(self):
        self.sentiment_threshold_negative = -0.3
        self.sentiment_threshold_positive = 0.3
        self.news_sources = [
            "Reuters", "Bloomberg News", "Financial Times", "Wall Street Journal", 
            "MarketWatch", "Yahoo Finance", "CNBC", "Seeking Alpha"
        ]
    
    def analyze_sentiment(self, red_flagged_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for RED-flagged assets only
        
        Args:
            red_flagged_assets: List of RED-flagged assets from core analysis
            
        Returns:
            List of sentiment analysis results
        """
        sentiment_results = []
        
        for asset in red_flagged_assets:
            # Fetch news for this asset (simulated)
            news_articles = self.fetch_news_for_asset(asset['symbol'])
            
            # Perform sentiment analysis
            asset_sentiment = self.analyze_asset_sentiment(asset, news_articles)
            sentiment_results.append(asset_sentiment)
        
        return sentiment_results
    
    def fetch_news_for_asset(self, symbol: str, days_back: int = 365) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a specific asset (simulated)
        
        Args:
            symbol: Asset symbol
            days_back: Number of days to look back for news
            
        Returns:
            List of news articles
        """
        # Generate mock news articles with realistic financial content
        news_templates = [
            f"{symbol} reports quarterly earnings miss, revenue down {random.randint(5, 25)}%",
            f"Analyst downgrades {symbol} citing regulatory concerns and market headwinds",
            f"{symbol} faces investigation over accounting practices, shares tumble",
            f"CEO of {symbol} resigns amid strategic disagreements with board",
            f"{symbol} announces major restructuring, plans to cut {random.randint(1000, 5000)} jobs",
            f"Credit rating agency downgrades {symbol} debt to junk status",
            f"{symbol} misses guidance for third consecutive quarter",
            f"Regulatory approval delayed for {symbol}'s key product launch",
            f"{symbol} competitor gains market share, pressure on margins continues",
            f"Institutional investors reduce {symbol} holdings amid volatility concerns",
            f"{symbol} explores strategic alternatives including potential sale",
            f"Supply chain disruptions impact {symbol} production forecasts",
            f"{symbol} settles lawsuit for ${random.randint(50, 500)} million",
            f"Moody's places {symbol} on review for possible downgrade",
            f"{symbol} withdraws full-year guidance citing economic uncertainty"
        ]
        
        positive_templates = [
            f"{symbol} beats earnings expectations, raises full-year guidance",
            f"New partnership announced between {symbol} and major tech company",
            f"{symbol} receives FDA approval for breakthrough therapy",
            f"Activist investor takes stake in {symbol}, pushes for changes",
            f"{symbol} announces share buyback program worth ${random.randint(100, 1000)}M"
        ]
        
        # Generate articles with bias toward negative news for RED-flagged assets
        news_articles = []
        num_articles = random.randint(15, 30)
        
        for i in range(num_articles):
            # 70% negative, 20% neutral, 10% positive for RED-flagged assets
            sentiment_bias = random.random()
            
            if sentiment_bias < 0.7:  # Negative news
                headline = random.choice(news_templates)
                sentiment_score = random.uniform(-0.8, -0.2)
            elif sentiment_bias < 0.9:  # Neutral news
                headline = f"{symbol} trading volume increases amid sector rotation"
                sentiment_score = random.uniform(-0.1, 0.1)
            else:  # Positive news
                headline = random.choice(positive_templates)
                sentiment_score = random.uniform(0.2, 0.6)
            
            # Generate article date
            days_ago = random.randint(1, days_back)
            article_date = datetime.now() - timedelta(days=days_ago)
            
            news_articles.append({
                'headline': headline,
                'source': random.choice(self.news_sources),
                'published_date': article_date.isoformat(),
                'url': f"https://example-news.com/{symbol.lower()}-{random.randint(1000, 9999)}",
                'sentiment_score': sentiment_score,
                'relevance_score': random.uniform(0.6, 1.0)
            })
        
        return sorted(news_articles, key=lambda x: x['published_date'], reverse=True)
    
    def analyze_asset_sentiment(self, asset: Dict[str, Any], news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform sentiment analysis for a specific asset
        
        Args:
            asset: Asset information
            news_articles: List of news articles for the asset
            
        Returns:
            Comprehensive sentiment analysis results
        """
        if not news_articles:
            return {
                'symbol': asset['symbol'],
                'news_count': 0,
                'sentiment_score': 0.0,
                'sentiment_label': 'NEUTRAL',
                'confidence': 0.0,
                'recent_news': [],
                'sentiment_trend': 'STABLE'
            }
        
        # Calculate overall sentiment metrics
        sentiment_scores = [article['sentiment_score'] for article in news_articles]
        relevance_scores = [article['relevance_score'] for article in news_articles]
        
        # Weighted average sentiment (by relevance)
        weighted_sentiment = np.average(sentiment_scores, weights=relevance_scores)
        
        # Recent trend analysis (last 30 days vs. older news)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_articles = [a for a in news_articles if datetime.fromisoformat(a['published_date'].replace('Z', '+00:00')).replace(tzinfo=None) > recent_cutoff]
        older_articles = [a for a in news_articles if datetime.fromisoformat(a['published_date'].replace('Z', '+00:00')).replace(tzinfo=None) <= recent_cutoff]
        
        recent_sentiment = np.mean([a['sentiment_score'] for a in recent_articles]) if recent_articles else 0
        older_sentiment = np.mean([a['sentiment_score'] for a in older_articles]) if older_articles else 0
        
        # Determine sentiment trend
        if recent_sentiment > older_sentiment + 0.1:
            sentiment_trend = 'IMPROVING'
        elif recent_sentiment < older_sentiment - 0.1:
            sentiment_trend = 'DETERIORATING'
        else:
            sentiment_trend = 'STABLE'
        
        # Sentiment label
        if weighted_sentiment <= self.sentiment_threshold_negative:
            sentiment_label = 'NEGATIVE'
        elif weighted_sentiment >= self.sentiment_threshold_positive:
            sentiment_label = 'POSITIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        # Calculate confidence based on number of articles and consistency
        sentiment_std = np.std(sentiment_scores)
        confidence = min(1.0, len(news_articles) / 20.0) * (1.0 - min(1.0, sentiment_std))
        
        # Key themes extraction (simulated)
        key_themes = self.extract_key_themes(news_articles)
        
        # Recent significant news
        recent_significant = sorted(
            [a for a in recent_articles if abs(a['sentiment_score']) > 0.4],
            key=lambda x: abs(x['sentiment_score']),
            reverse=True
        )[:5]
        
        return {
            'symbol': asset['symbol'],
            'sector': asset['sector'],
            'risk_rating': asset['risk_rating'],
            
            # Sentiment metrics
            'sentiment_score': weighted_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'sentiment_trend': sentiment_trend,
            
            # News analysis
            'news_count': len(news_articles),
            'recent_news_count': len(recent_articles),
            'negative_news_ratio': len([a for a in news_articles if a['sentiment_score'] < -0.2]) / len(news_articles),
            
            # Key insights
            'key_themes': key_themes,
            'recent_significant_news': recent_significant,
            
            # Time-based analysis
            'recent_sentiment': recent_sentiment,
            'older_sentiment': older_sentiment,
            'sentiment_volatility': sentiment_std,
            
            # Analysis timestamp
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def extract_key_themes(self, news_articles: List[Dict[str, Any]]) -> List[str]:
        """
        Extract key themes from news headlines (simplified approach)
        
        Args:
            news_articles: List of news articles
            
        Returns:
            List of key themes
        """
        # Common financial themes/keywords
        theme_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'guidance'],
            'regulatory': ['investigation', 'regulatory', 'compliance', 'lawsuit', 'settlement'],
            'management': ['CEO', 'resign', 'leadership', 'board', 'management'],
            'market_share': ['competitor', 'market share', 'competitive'],
            'financial_health': ['debt', 'credit', 'rating', 'downgrade', 'bankruptcy'],
            'operations': ['restructuring', 'layoffs', 'production', 'supply chain'],
            'growth': ['expansion', 'partnership', 'acquisition', 'buyback', 'investment']
        }
        
        # Count theme occurrences
        theme_counts = {theme: 0 for theme in theme_keywords}
        
        for article in news_articles:
            headline_lower = article['headline'].lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in headline_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        # Return top themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes if count > 0][:5]
    
    def calculate_sentiment_impact_score(self, sentiment_result: Dict[str, Any], 
                                       risk_metrics: Dict[str, Any]) -> float:
        """
        Calculate combined sentiment-risk impact score
        
        Args:
            sentiment_result: Sentiment analysis results
            risk_metrics: Risk metrics from core analysis
            
        Returns:
            Combined impact score (0-1, higher = more concerning)
        """
        # Base sentiment impact
        sentiment_impact = abs(sentiment_result['sentiment_score']) * sentiment_result['confidence']
        
        # Adjust for trend
        if sentiment_result['sentiment_trend'] == 'DETERIORATING':
            sentiment_impact *= 1.2
        elif sentiment_result['sentiment_trend'] == 'IMPROVING':
            sentiment_impact *= 0.8
        
        # Combine with risk metrics
        risk_score = risk_metrics.get('risk_score', 0) / 7.0  # Normalize to 0-1
        volatility_score = min(1.0, risk_metrics.get('volatility', 0) / 0.5)  # Cap at 50% vol
        
        # Weighted combination
        combined_score = (
            0.4 * sentiment_impact +
            0.3 * risk_score +
            0.3 * volatility_score
        )
        
        return min(1.0, combined_score)
