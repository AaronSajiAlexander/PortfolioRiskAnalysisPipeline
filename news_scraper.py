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
        
    def _load_stock_symbols(self) -> Dict[str, str]:
        """
        Load stock symbols and company names from mock_data.py
        
        Returns:
            Dictionary mapping symbols to company names
        """
        # Import from mock_data
        try:
            from utils.mock_data import MockBloombergData
            mock_data = MockBloombergData()
            symbol_map = {stock['symbol']: stock['name'] for stock in mock_data.all_stocks}
            return symbol_map
        except Exception as e:
            print(f"Warning: Could not load symbols from mock_data.py: {e}")
            # Fallback to hardcoded list with company names
            return {
                'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'JNJ': 'Johnson & Johnson',
                'PG': 'Procter & Gamble Co.', 'V': 'Visa Inc.', 'WMT': 'Walmart Inc.',
                'XOM': 'Exxon Mobil Corporation', 'CRM': 'Salesforce Inc.',
                'SBUX': 'Starbucks Corporation', 'BA': 'Boeing Co.',
                'UBER': 'Uber Technologies Inc.', 'SNAP': 'Snap Inc.',
                'SHOP': 'Shopify Inc.', 'SQ': 'Block Inc.', 'TSLA': 'Tesla Inc.',
                'NVDA': 'NVIDIA Corporation', 'GME': 'GameStop Corp.',
                'AMC': 'AMC Entertainment Holdings', 'RIVN': 'Rivian Automotive Inc.',
                'SPCE': 'Virgin Galactic Holdings', 'BLNK': 'Blink Charging Co.'
            }
    
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
        Filter articles by stock symbols and company names
        
        Args:
            articles: List of all articles
            
        Returns:
            Dictionary mapping stock symbols to relevant articles
        """
        print(f"\n🔍 Filtering articles for {len(self.stock_symbols)} stocks...")
        
        stock_news = {symbol: [] for symbol in self.stock_symbols.keys()}
        
        # Problematic symbols that cause false matches (with context)
        false_match_patterns = {
            'SNAP': r'\bSNAP\s+(benefits|program|food|assistance)',  # SNAP benefits program
            'AMC': r'\bAMC\s+(theatre|theaters|cinema)',  # Could be generic AMC
            'BA': r'\bBA\s+(degree|bachelor)',  # BA degree
            'SQ': r'\bSQ\s+(ft|feet|meter)',  # Square feet/meter
            'V': r'\bV\s+(shaped|neck|sign)'  # V-shaped, V-neck, etc.
        }
        
        for article in articles:
            # Combine title and summary for matching
            text = f"{article['title']} {article['summary']}"
            text_upper = text.upper()
            
            for symbol, company_name in self.stock_symbols.items():
                matched = False
                
                # Check for false positive patterns first
                if symbol in false_match_patterns:
                    if re.search(false_match_patterns[symbol], text, re.IGNORECASE):
                        continue  # Skip this article for this symbol
                
                # Match 1: Stock symbol with context (e.g., "AAPL stock", "$AAPL", "AAPL shares")
                symbol_pattern = r'(\$' + re.escape(symbol) + r'\b|\b' + re.escape(symbol) + r'\s+(STOCK|SHARES|TICKER|INC|CORP|CORPORATION))'
                if re.search(symbol_pattern, text_upper):
                    matched = True
                
                # Match 1b: Stock symbol standalone in financial context (high confidence match)
                # This catches headlines like "MSFT up 5%" or "BA stock slides"
                standalone_pattern = r'\b' + re.escape(symbol) + r'\b'
                if re.search(standalone_pattern, text_upper):
                    # Additional validation: ensure it's in a financial/stock context
                    financial_keywords = ['STOCK', 'SHARES', 'EARNINGS', 'REVENUE', 'PRICE', 'MARKET', 
                                         'INVESTORS', 'TRADING', 'QUARTERLY', 'PROFIT', 'LOSS', 'GAINS',
                                         'FALLS', 'RISES', 'SLIDES', 'SURGES', 'PLUNGES', 'RALLIES']
                    if any(keyword in text_upper for keyword in financial_keywords):
                        matched = True
                
                # Match 2: Company name (extract main company name before "Inc", "Corp", etc.)
                company_base = company_name.split(' Inc')[0].split(' Corp')[0].split(' Co.')[0]
                # Only match if company name is substantial (3+ chars to avoid false matches)
                if len(company_base) >= 3:
                    company_pattern = r'\b' + re.escape(company_base) + r'\b'
                    if re.search(company_pattern, text, re.IGNORECASE):
                        matched = True
                
                if matched:
                    # Deduplicate: check if article already added
                    article_links = [a['link'] for a in stock_news[symbol]]
                    if article['link'] not in article_links:
                        stock_news[symbol].append(article)
        
        # Count stocks with news
        stocks_with_news = {k: v for k, v in stock_news.items() if len(v) > 0}
        print(f"✓ Found news for {len(stocks_with_news)}/{len(self.stock_symbols)} stocks")
        
        for symbol, articles in stocks_with_news.items():
            company_name = self.stock_symbols[symbol]
            print(f"  • {symbol} ({company_name}): {len(articles)} articles")
        
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
