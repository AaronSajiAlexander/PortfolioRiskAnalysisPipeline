# News Article JSON Structure

This document defines the required JSON structure for news articles used in sentiment analysis.

## Overview

The sentiment analysis engine now uses **both headlines and article content** for more accurate sentiment scoring:
- **Headlines**: 40% weight in sentiment calculation
- **Content**: 60% weight in sentiment calculation

This provides deeper, more nuanced sentiment analysis compared to headlines alone.

---

## Complete JSON Structure

```json
{
  "AAPL": [
    {
      "headline": "Apple announces record quarterly earnings, beats expectations",
      "content": "Apple Inc. reported record-breaking quarterly earnings today, surpassing analyst expectations across all key metrics. The tech giant's revenue grew 15% year-over-year, driven by strong iPhone sales and services growth. CEO Tim Cook highlighted the company's continued innovation and market expansion as key drivers of success. Investors responded enthusiastically to the positive developments. The company's strategic initiatives are showing strong results and momentum. Analysts have raised their price targets following this announcement.",
      "source": "Reuters",
      "published_date": "2025-10-15T14:30:00",
      "url": "https://reuters.com/article/apple-earnings-12345",
      "sentiment_score": 0.65,
      "relevance_score": 0.95
    },
    {
      "headline": "Apple faces regulatory scrutiny in EU over App Store practices",
      "content": "Apple faces regulatory scrutiny in EU over App Store practices. European regulators have opened a formal investigation into Apple's App Store policies, citing concerns about anti-competitive behavior. The company faces mounting pressure to address these issues quickly. Industry analysts warn this could lead to significant fines and required changes to business practices. Competitors are seizing this opportunity to gain market advantage.",
      "source": "Financial Times",
      "published_date": "2025-10-14T09:15:00",
      "url": "https://ft.com/content/apple-eu-investigation",
      "sentiment_score": -0.45,
      "relevance_score": 0.88
    }
  ],
  "MSFT": [
    {
      "headline": "Microsoft expands AI capabilities with new Azure OpenAI features",
      "content": "Microsoft expands AI capabilities with new Azure OpenAI features. This marks a significant milestone for the company's growth trajectory in the AI sector. The new features enable enterprise customers to build more sophisticated AI applications with enhanced security and compliance. Industry observers note the competitive advantages being established. The market outlook remains favorable as the company executes its vision for AI-powered cloud computing.",
      "source": "Bloomberg News",
      "published_date": "2025-10-15T10:00:00",
      "url": "https://bloomberg.com/microsoft-ai-expansion",
      "sentiment_score": 0.72,
      "relevance_score": 0.91
    }
  ],
  "SERV": [
    {
      "headline": "Small-cap tech firm SERV faces cash flow challenges",
      "content": "Small-cap tech firm SERV faces cash flow challenges. Market analysts express concerns about the company's future prospects given recent operational difficulties. The recent developments have raised questions among investors about long-term viability. Industry experts warn that these challenges could persist throughout the fiscal year, potentially requiring additional financing or strategic alternatives.",
      "source": "MarketWatch",
      "published_date": "2025-10-13T11:00:00",
      "url": "https://marketwatch.com/serv-challenges",
      "sentiment_score": -0.58,
      "relevance_score": 0.87
    }
  ]
}
```

---

## Field Specifications

### Required Fields

| Field | Type | Format | Description |
|-------|------|--------|-------------|
| `headline` | string | Text | Article headline/title (concise, typically 10-20 words) |
| `content` | string | Text | **NEW**: Full article content or summary (200-500 words recommended) |
| `source` | string | Text | News source name (e.g., "Reuters", "Bloomberg News", "Financial Times") |
| `published_date` | string | ISO 8601 | Publication timestamp: `YYYY-MM-DDTHH:MM:SS` or `YYYY-MM-DDTHH:MM:SS.sssZ` |
| `url` | string | URL | Full URL to the original article |
| `sentiment_score` | float | -1.0 to 1.0 | Pre-calculated sentiment score (optional, will be recalculated) |
| `relevance_score` | float | 0.0 to 1.0 | How relevant the article is to the stock symbol |

### Sentiment Score Scale

- **-1.0 to -0.5**: Very negative (major concerns, crises, disasters)
- **-0.5 to -0.3**: Negative (challenges, setbacks, warnings)
- **-0.3 to -0.1**: Slightly negative (minor concerns, cautious outlook)
- **-0.1 to +0.1**: Neutral (factual reporting, mixed signals)
- **+0.1 to +0.3**: Slightly positive (modest gains, stable performance)
- **+0.3 to +0.5**: Positive (growth, achievements, opportunities)
- **+0.5 to +1.0**: Very positive (major breakthroughs, exceptional results)

### Relevance Score Guide

- **0.9 - 1.0**: Directly about the company (earnings, product launches, CEO changes)
- **0.7 - 0.9**: Significantly impacts the company (industry regulations, major competitor moves)
- **0.5 - 0.7**: Moderately relevant (sector trends, market conditions)
- **0.3 - 0.5**: Tangentially relevant (broader economic news, related industries)
- **0.0 - 0.3**: Minimally relevant (general news mentioning the company)

---

## Content Quality Guidelines

### Good Content Examples ✅

**Detailed and Informative**:
```
"Tesla reported Q3 earnings that exceeded Wall Street expectations, with revenue 
reaching $23.4 billion, up 9% year-over-year. The electric vehicle maker delivered 
435,000 vehicles during the quarter, setting a new company record. CEO Elon Musk 
announced plans to expand production capacity in both Texas and Berlin facilities. 
Gross margins improved to 17.9%, beating analyst estimates of 16.8%. The positive 
results come amid growing competition in the EV sector and ongoing supply chain 
challenges that have affected the broader automotive industry."
```

**Balanced Reporting**:
```
"Amazon's cloud computing division AWS showed mixed results in the latest quarter. 
While revenue grew 12% to $23.1 billion, this marked a slowdown from previous 
quarters. The company cited increased competition from Microsoft Azure and Google 
Cloud as factors impacting growth rates. However, AWS maintained its market-leading 
position with 32% share of the cloud infrastructure market. Management expressed 
confidence in the division's long-term prospects, highlighting new AI services 
and enterprise customer wins."
```

### Poor Content Examples ❌

**Too Short**:
```
"Stock went up today."
```

**Too Vague**:
```
"The company had some news today about things."
```

**Missing Context**:
```
"Shares fell 5%."  
(Missing: Why? What caused it? What's the impact?)
```

---

## Sentiment Calculation Method

The system uses **TextBlob** for sentiment analysis with weighted combination:

```python
# Headline sentiment: 40% weight
headline_sentiment = TextBlob(headline).sentiment.polarity

# Content sentiment: 60% weight  
content_sentiment = TextBlob(content).sentiment.polarity

# Combined score
final_sentiment = (0.4 * headline_sentiment) + (0.6 * content_sentiment)
```

**Why this weighting?**
- Headlines are often sensationalized for clicks
- Article content provides more balanced, detailed information
- Content better reflects the true impact and context
- 60/40 split balances both while emphasizing substance

---

## Integration with Langflow

### Step 1: Web Scraping
Your Langflow agent should:
1. Scrape news from trusted sources (Reuters, Bloomberg, FT, WSJ, etc.)
2. Extract both **headline** and **full article text**
3. Filter for relevance to the 20 stock symbols

### Step 2: Processing
For each article:
1. Extract headline, content, source, date, URL
2. Calculate relevance score (keyword matching, NER)
3. Optionally pre-calculate sentiment (will be recalculated anyway)

### Step 3: Output
Save to `news_data.json` in the structure shown above.

### Step 4: Integration
The sentiment analysis pipeline will:
1. Load `news_data.json`
2. Recalculate sentiment using both headline + content
3. Perform deep theme extraction from full content
4. Generate comprehensive sentiment reports

---

## Trusted News Sources

Recommended sources for financial news scraping:

**Tier 1 (Highest Quality)**:
- Reuters
- Bloomberg News
- Financial Times
- Wall Street Journal

**Tier 2 (Quality)**:
- MarketWatch
- CNBC
- Yahoo Finance
- Barron's

**Tier 3 (Specialized)**:
- Seeking Alpha
- The Motley Fool
- Investor's Business Daily
- TechCrunch (for tech stocks)

---

## Example Implementation

### Minimal Article
```json
{
  "headline": "NVDA shares rise on AI chip demand",
  "content": "NVIDIA shares climbed 3.2% today following reports of strong demand for its latest AI accelerator chips. Data center customers are increasing orders to support growing artificial intelligence workloads.",
  "source": "Reuters",
  "published_date": "2025-10-15T15:30:00",
  "url": "https://reuters.com/nvda-ai-chips",
  "sentiment_score": 0.45,
  "relevance_score": 0.92
}
```

### Comprehensive Article
```json
{
  "headline": "JPMorgan Chase reports strong Q3 results despite economic headwinds",
  "content": "JPMorgan Chase & Co. posted third-quarter earnings of $13.2 billion, exceeding analyst expectations despite a challenging macroeconomic environment. The nation's largest bank by assets reported revenue of $39.9 billion, up 7% from the same period last year. Net interest income rose 30% to $21.9 billion, benefiting from higher interest rates. However, the bank set aside $1.5 billion for credit losses, reflecting caution about the economic outlook. CEO Jamie Dimon noted that while the consumer remains resilient, the bank is prepared for potential economic softness ahead. Investment banking fees declined 43% amid a slowdown in mergers and acquisitions. The results demonstrate JPMorgan's diversified business model, with strength in traditional banking offsetting weakness in capital markets activities.",
  "source": "Bloomberg News",
  "published_date": "2025-10-13T07:00:00",
  "url": "https://bloomberg.com/news/jpmorgan-q3-earnings",
  "sentiment_score": 0.28,
  "relevance_score": 0.98
}
```

---

## Questions?

For integration support or questions about the structure, refer to:
- `pipeline/sentiment_analysis.py` - Sentiment analysis implementation
- `MOCK_DATA_GUIDE.md` - Overview of data systems
- This file - News article structure reference
