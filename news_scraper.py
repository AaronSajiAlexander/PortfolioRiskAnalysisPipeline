"""
Standalone News Intelligence Scraper
Fetches financial news from legal RSS feeds and consolidates using LLM
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from typing import List, Dict, Any
import time
import os


class NewsScraperLLM:
    """
    Scrapes financial news from legal RSS feeds and uses LLM to consolidate
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize news scraper
        
        Args:
            openai_api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        # RSS feed sources (legal public feeds)
        self.rss_feeds = {
            'Reuters Business (via Google)': 'https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com+business+markets&ceid=US:en&hl=en-US&gl=US',
            'Reuters Finance (via Google)': 'https://news.google.com/rss/search?q=when:24h+allinurl:reuters.com+finance+stocks&ceid=US:en&hl=en-US&gl=US',
            'CNBC Top News': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'CNBC Finance': 'https://www.cnbc.com/id/10001147/device/rss/rss.html',
            'CNBC US Markets': 'https://www.cnbc.com/id/15839135/device/rss/rss.html',
            'WSJ Markets (via Google)': 'https://news.google.com/rss/search?q=when:24h+allinurl:wsj.com+markets+stocks&ceid=US:en&hl=en-US&gl=US',
            'Bloomberg (via Google)': 'https://news.google.com/rss/search?q=when:24h+allinurl:bloomberg.com+markets+finance&ceid=US:en&hl=en-US&gl=US',
            'Finextra Headlines': 'https://www.finextra.com/rss/headlines.aspx',
            'Finextra News': 'https://www.finextra.com/rss/news.aspx'
        }
        
        # Stock symbols to track (from mock_data.py)
        self.stock_symbols = self._load_stock_symbols()
        
    def _load_stock_symbols(self) -> List[str]:
        """Load stock symbols from mock_data.py"""
        # Import from mock_data
        try:
            from utils.mock_data import MockBloombergData
            mock_data = MockBloombergData()
            symbols = [stock['symbol'] for stock in mock_data.all_stocks]
            return symbols
        except Exception as e:
            print(f"Warning: Could not load symbols from mock_data.py: {e}")
            # Fallback to hardcoded list
            return [
                'AAPL', 'MSFT', 'JNJ', 'PG', 'V', 'WMT', 'XOM',  # GREEN
                'CRM', 'SBUX', 'BA', 'UBER', 'SNAP', 'SHOP', 'SQ',  # YELLOW
                'TSLA', 'NVDA', 'GME', 'AMC', 'RIVN', 'SPCE', 'BLNK'  # RED
            ]
    
    def fetch_rss_feeds(self) -> List[Dict[str, Any]]:
        """
        Fetch all RSS feeds and parse articles
        
        Returns:
            List of raw articles with metadata
        """
        all_articles = []
        
        print(f"📰 Fetching news from {len(self.rss_feeds)} RSS feeds...")
        
        for feed_name, feed_url in self.rss_feeds.items():
            try:
                print(f"  • Fetching {feed_name}...")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    article = {
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'summary': entry.get('summary', entry.get('description', '')),
                        'published': entry.get('published', ''),
                        'source': feed_name,
                        'raw_entry': entry
                    }
                    all_articles.append(article)
                
                print(f"    ✓ Found {len(feed.entries)} articles")
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"    ⚠️ Error fetching {feed_name}: {e}")
        
        print(f"\n✓ Total articles fetched: {len(all_articles)}")
        return all_articles
    
    def filter_articles_by_stocks(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Filter articles by stock symbols
        
        Args:
            articles: List of all articles
            
        Returns:
            Dictionary mapping stock symbols to relevant articles
        """
        print(f"\n🔍 Filtering articles for {len(self.stock_symbols)} stocks...")
        
        stock_news = {symbol: [] for symbol in self.stock_symbols}
        
        for article in articles:
            # Combine title and summary for matching
            text = f"{article['title']} {article['summary']}".upper()
            
            for symbol in self.stock_symbols:
                # Match stock symbol (with word boundaries to avoid false matches)
                # e.g., "AAPL" should match "AAPL stock" but not "AAPPLE"
                pattern = r'\b' + re.escape(symbol) + r'\b'
                if re.search(pattern, text):
                    stock_news[symbol].append(article)
        
        # Count stocks with news
        stocks_with_news = {k: v for k, v in stock_news.items() if len(v) > 0}
        print(f"✓ Found news for {len(stocks_with_news)}/{len(self.stock_symbols)} stocks")
        
        for symbol, articles in stocks_with_news.items():
            print(f"  • {symbol}: {len(articles)} articles")
        
        return stock_news
    
    def extract_full_article(self, url: str) -> str:
        """
        Extract full article text from URL using BeautifulSoup
        
        Args:
            url: Article URL
            
        Returns:
            Full article text (or empty string if failed)
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0; +http://example.com/bot)'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            return article_text[:2000]  # Limit to 2000 chars to save LLM tokens
        
        except Exception as e:
            return ""
    
    def consolidate_with_llm(self, stock_news: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Use OpenAI LLM to consolidate and structure news data
        
        Args:
            stock_news: Dictionary mapping symbols to articles
            
        Returns:
            Consolidated JSON structure
        """
        print(f"\n🤖 Using LLM to consolidate news data...")
        
        if not self.openai_api_key:
            print("⚠️ Warning: No OpenAI API key provided. Skipping LLM consolidation.")
            print("Set OPENAI_API_KEY environment variable to enable LLM processing.")
            return self._simple_consolidation(stock_news)
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            consolidated = {}
            
            # Process each stock with news
            stocks_with_news = {k: v for k, v in stock_news.items() if len(v) > 0}
            
            for i, (symbol, articles) in enumerate(stocks_with_news.items(), 1):
                print(f"  • Processing {symbol} ({i}/{len(stocks_with_news)})...")
                
                # Prepare article summaries for LLM
                article_summaries = []
                for j, article in enumerate(articles[:10], 1):  # Limit to 10 articles per stock
                    article_summaries.append(f"{j}. {article['title']} - {article['summary'][:200]}")
                
                # Create LLM prompt
                prompt = f"""You are a financial news analyst. Given the following news articles about {symbol}, 
extract and consolidate them into a structured format.

News Articles:
{chr(10).join(article_summaries)}

Return ONLY a JSON array of objects with this structure:
[
  {{
    "headline": "Clear, concise headline",
    "body": "Summary of the article body (2-3 sentences)",
    "source": "Source name",
    "date": "Publication date if available"
  }}
]

Be concise. Return valid JSON only."""

                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a financial news analyst. Return only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    
                    # Parse LLM response
                    llm_output = response.choices[0].message.content.strip()
                    
                    # Extract JSON from response (handle markdown code blocks)
                    if '```json' in llm_output:
                        llm_output = llm_output.split('```json')[1].split('```')[0].strip()
                    elif '```' in llm_output:
                        llm_output = llm_output.split('```')[1].split('```')[0].strip()
                    
                    news_list = json.loads(llm_output)
                    consolidated[symbol] = news_list
                    
                    time.sleep(1)  # Rate limiting
                    
                except json.JSONDecodeError as e:
                    print(f"    ⚠️ JSON parse error for {symbol}: {e}")
                    # Fallback to simple consolidation
                    consolidated[symbol] = self._simple_consolidation_for_stock(symbol, articles)
                
                except Exception as e:
                    print(f"    ⚠️ LLM error for {symbol}: {e}")
                    consolidated[symbol] = self._simple_consolidation_for_stock(symbol, articles)
            
            print(f"✓ Consolidated news for {len(consolidated)} stocks")
            return consolidated
        
        except Exception as e:
            print(f"⚠️ LLM initialization error: {e}")
            return self._simple_consolidation(stock_news)
    
    def _simple_consolidation(self, stock_news: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Fallback: Simple consolidation without LLM"""
        consolidated = {}
        
        for symbol, articles in stock_news.items():
            if len(articles) > 0:
                consolidated[symbol] = self._simple_consolidation_for_stock(symbol, articles)
        
        return consolidated
    
    def _simple_consolidation_for_stock(self, symbol: str, articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Simple consolidation for a single stock"""
        news_list = []
        
        for article in articles[:10]:  # Limit to 10 articles
            news_list.append({
                'headline': article['title'],
                'body': article['summary'][:300],  # Truncate summary
                'source': article['source'],
                'date': article['published']
            })
        
        return news_list
    
    def scrape_and_consolidate(self) -> Dict[str, Any]:
        """
        Main method: Scrape RSS feeds and consolidate with LLM
        
        Returns:
            Dictionary mapping stock symbols to news articles
        """
        print("=" * 80)
        print("🚀 NEWS SCRAPER LLM - Starting...")
        print("=" * 80)
        
        # Step 1: Fetch RSS feeds
        all_articles = self.fetch_rss_feeds()
        
        # Step 2: Filter by stock symbols
        stock_news = self.filter_articles_by_stocks(all_articles)
        
        # Step 3: Consolidate with LLM
        consolidated = self.consolidate_with_llm(stock_news)
        
        print("\n" + "=" * 80)
        print("✅ NEWS SCRAPER COMPLETE")
        print("=" * 80)
        
        return consolidated


def main():
    """
    Main entry point for standalone execution
    """
    # Initialize scraper
    scraper = NewsScraperLLM()
    
    # Scrape and consolidate
    result = scraper.scrape_and_consolidate()
    
    # Print JSON output
    print("\n" + "=" * 80)
    print("📊 FINAL JSON OUTPUT")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    
    # Save to file
    output_file = 'news_scraper_output.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Output saved to: {output_file}")


if __name__ == "__main__":
    main()
