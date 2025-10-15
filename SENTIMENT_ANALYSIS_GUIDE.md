# Sentiment Analysis System - Detailed Technical Guide

This guide provides a comprehensive explanation of the Sentiment Analysis Engine (Stage 4 of the pipeline), written for users who want to understand exactly how the system works.

---

## Table of Contents

1. [What Is Sentiment Analysis?](#what-is-sentiment-analysis)
2. [Why Only RED-Flagged Assets?](#why-only-red-flagged-assets)
3. [The SentimentAnalysisEngine Class](#the-sentimentanalysisengine-class)
4. [Step-by-Step Process](#step-by-step-process)
5. [News Article Generation](#news-article-generation)
6. [Sentiment Score Calculation](#sentiment-score-calculation)
7. [Trend Analysis](#trend-analysis)
8. [Key Themes Extraction](#key-themes-extraction)
9. [Confidence Calculation](#confidence-calculation)
10. [Impact Score Calculation](#impact-score-calculation)
11. [Real-World Examples](#real-world-examples)
12. [Technical Implementation Details](#technical-implementation-details)

---

## What Is Sentiment Analysis?

### Definition

**Sentiment Analysis** (also called opinion mining) is the process of using natural language processing (NLP) to determine the emotional tone behind text. In finance, it means reading news articles and determining if they're:

- **Positive** (good news): earnings beat, partnerships, approvals
- **Negative** (bad news): losses, investigations, downgrades
- **Neutral** (just facts): routine announcements, filings

### Simple Analogy

Imagine you're reading restaurant reviews to decide where to eat:
- "Amazing food, best service ever!" → Positive (5 stars)
- "Terrible experience, never going back!" → Negative (1 star)
- "Restaurant serves Italian food" → Neutral (just a fact)

Sentiment analysis does the same thing with financial news, but using algorithms instead of human judgment.

### Why It Matters for Investing

**Context for Decisions:**
- Numbers tell you WHAT is happening (stock down 20%)
- News tells you WHY it's happening (SEC investigation announced)

**Early Warning System:**
- Negative sentiment often appears before price drops
- News spreads faster than prices adjust
- Gives you time to react

**Validation:**
- If Stage 2 says "RED" but news is positive, maybe the worst is over
- If Stage 2 says "RED" and news is terrible, definitely time to worry

---

## Why Only RED-Flagged Assets?

### The Strategic Focus

The system only analyzes sentiment for assets rated **RED** (high risk) in Stage 2. Here's why:

**Resource Efficiency:**
- Sentiment analysis requires fetching and processing news (computationally expensive)
- No point analyzing news for healthy (GREEN) investments
- Focus computational resources where they matter most

**Urgency Principle:**
- RED assets need immediate attention
- Understanding WHY they're risky helps you decide what to do
- GREEN/YELLOW assets can wait

**Information Value:**
- For RED assets: News provides critical context
- For GREEN assets: News is less urgent (they're doing fine)
- For YELLOW assets: May be worth checking, but less critical

### Real-World Example

**Portfolio of 30 investments:**
- 5 are RED (high risk) ← Analyze sentiment for these 5
- 8 are YELLOW (moderate risk) ← Skip for now
- 17 are GREEN (low risk) ← Skip for now

**Result:** Only process 5 assets instead of 30 (83% time savings)

---

## The SentimentAnalysisEngine Class

### Class Overview

**Location:** `pipeline/sentiment_analysis.py`

**Purpose:** Analyzes news sentiment for RED-flagged assets to provide context and early warnings.

### Class Initialization

```python
class SentimentAnalysisEngine:
    def __init__(self):
        self.sentiment_threshold_negative = -0.3
        self.sentiment_threshold_positive = 0.3
        self.news_sources = [
            "Reuters", "Bloomberg News", "Financial Times", 
            "Wall Street Journal", "MarketWatch", "Yahoo Finance", 
            "CNBC", "Seeking Alpha"
        ]
```

**What This Sets Up:**

**1. Sentiment Thresholds:**
- Scores **≤ -0.3**: Classified as NEGATIVE
- Scores **≥ +0.3**: Classified as POSITIVE
- Scores **between -0.3 and +0.3**: Classified as NEUTRAL

**Why these numbers?**
- Based on TextBlob's sentiment polarity scale (-1.0 to +1.0)
- -0.3/+0.3 are industry-standard cutoffs
- Creates a "dead zone" for truly neutral news

**2. News Sources List:**
- Simulates real financial news outlets
- Used to assign source attribution to articles
- Mirrors real-world news ecosystem

### Main Methods

The class has 6 key methods:

1. **`analyze_sentiment()`** - Main entry point
2. **`fetch_news_for_asset()`** - Retrieves news articles
3. **`analyze_asset_sentiment()`** - Performs sentiment analysis
4. **`extract_key_themes()`** - Identifies main topics
5. **`calculate_sentiment_impact_score()`** - Combines sentiment with risk
6. Helper methods for processing

---

## Step-by-Step Process

### High-Level Flow

```
1. Receive RED-flagged assets from Stage 2
2. For each RED asset:
   a. Fetch news articles (15-30 per asset)
   b. Calculate sentiment scores
   c. Analyze sentiment trends
   d. Extract key themes
   e. Determine confidence level
3. Return comprehensive sentiment results
```

### Detailed Walkthrough

#### Step 1: Entry Point

```python
def analyze_sentiment(self, red_flagged_assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sentiment_results = []
    
    for asset in red_flagged_assets:
        # Fetch news for this asset
        news_articles = self.fetch_news_for_asset(asset['symbol'])
        
        # Perform sentiment analysis
        asset_sentiment = self.analyze_asset_sentiment(asset, news_articles)
        sentiment_results.append(asset_sentiment)
    
    return sentiment_results
```

**What Happens:**
1. Loop through each RED asset
2. Get news articles for that asset
3. Analyze sentiment for that asset
4. Collect all results
5. Return complete analysis

**Input Example:**
```python
red_flagged_assets = [
    {'symbol': 'ACME', 'risk_rating': 'RED', ...},
    {'symbol': 'TECH', 'risk_rating': 'RED', ...},
    {'symbol': 'FINC', 'risk_rating': 'RED', ...}
]
```

**Output Example:**
```python
sentiment_results = [
    {'symbol': 'ACME', 'sentiment_score': -0.52, ...},
    {'symbol': 'TECH', 'sentiment_score': -0.38, ...},
    {'symbol': 'FINC', 'sentiment_score': -0.61, ...}
]
```

---

## News Article Generation

### The `fetch_news_for_asset()` Method

This method generates realistic mock news articles for an asset.

```python
def fetch_news_for_asset(self, symbol: str, days_back: int = 365) -> List[Dict[str, Any]]:
```

**Parameters:**
- `symbol`: Stock symbol (e.g., "ACME")
- `days_back`: How far back to look (default: 365 days)

**Returns:** List of news article dictionaries

### News Template Categories

The system has three types of news templates:

#### 1. Negative Templates (For RED Assets)

```python
news_templates = [
    f"{symbol} reports quarterly earnings miss, revenue down {random.randint(5, 25)}%",
    f"Analyst downgrades {symbol} citing regulatory concerns and market headwinds",
    f"{symbol} faces investigation over accounting practices, shares tumble",
    f"CEO of {symbol} resigns amid strategic disagreements with board",
    f"{symbol} announces major restructuring, plans to cut {random.randint(1000, 5000)} jobs",
    f"Credit rating agency downgrades {symbol} debt to junk status",
    # ... 15 total negative templates
]
```

**Why These Templates:**
- Based on real financial news patterns
- Cover major risk categories (earnings, regulatory, management, debt)
- Include dynamic elements (random percentages, numbers)
- Sound authentic

#### 2. Positive Templates

```python
positive_templates = [
    f"{symbol} beats earnings expectations, raises full-year guidance",
    f"New partnership announced between {symbol} and major tech company",
    f"{symbol} receives FDA approval for breakthrough therapy",
    f"Activist investor takes stake in {symbol}, pushes for changes",
    f"{symbol} announces share buyback program worth ${random.randint(100, 1000)}M"
]
```

**Why Fewer Positive Templates:**
- RED assets are risky, so less likely to have good news
- Still need some positive news for realism
- Used sparingly (10% of articles)

#### 3. Neutral Templates

```python
headline = f"{symbol} trading volume increases amid sector rotation"
```

**Generic Statements:**
- Factual observations
- No clear positive or negative tone
- Used as filler (20% of articles)

### Article Generation Process

#### Step 1: Determine Number of Articles

```python
num_articles = random.randint(15, 30)
```

**Why 15-30:**
- Realistic for a year of coverage
- Major stocks get ~30 articles/year
- Smaller stocks get ~15 articles/year
- Provides statistical significance

#### Step 2: Sentiment Bias (RED Assets)

```python
for i in range(num_articles):
    sentiment_bias = random.random()  # 0.0 to 1.0
    
    if sentiment_bias < 0.7:  # 70% chance
        # Negative news
        headline = random.choice(news_templates)
        sentiment_score = random.uniform(-0.8, -0.2)
    
    elif sentiment_bias < 0.9:  # 20% chance (0.7 to 0.9)
        # Neutral news
        headline = f"{symbol} trading volume increases amid sector rotation"
        sentiment_score = random.uniform(-0.1, 0.1)
    
    else:  # 10% chance (0.9 to 1.0)
        # Positive news
        headline = random.choice(positive_templates)
        sentiment_score = random.uniform(0.2, 0.6)
```

**Distribution for RED Assets:**
- 70% negative articles
- 20% neutral articles
- 10% positive articles

**Why This Mix:**
- RED assets have problems, so mostly bad news
- But not 100% negative (unrealistic)
- Some neutral/positive for balance

#### Step 3: Add Metadata

```python
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
```

**Each Article Contains:**
- **headline**: The news text
- **source**: Reuters, Bloomberg, etc.
- **published_date**: Random date in past year
- **url**: Mock URL (for display purposes)
- **sentiment_score**: Pre-assigned sentiment (-1 to +1)
- **relevance_score**: How relevant to the stock (0.6-1.0)

#### Step 4: Sort by Date

```python
return sorted(news_articles, key=lambda x: x['published_date'], reverse=True)
```

**Result:** Most recent articles first (just like real news feeds)

### Example Output

**For Symbol "ACME" (RED-flagged):**

```python
[
    {
        'headline': 'ACME reports quarterly earnings miss, revenue down 18%',
        'source': 'Reuters',
        'published_date': '2024-10-01T14:30:00',
        'url': 'https://example-news.com/acme-7823',
        'sentiment_score': -0.72,
        'relevance_score': 0.95
    },
    {
        'headline': 'CEO of ACME resigns amid strategic disagreements with board',
        'source': 'Financial Times',
        'published_date': '2024-09-15T09:20:00',
        'sentiment_score': -0.68,
        'relevance_score': 0.88
    },
    {
        'headline': 'ACME trading volume increases amid sector rotation',
        'source': 'MarketWatch',
        'published_date': '2024-08-22T11:45:00',
        'sentiment_score': 0.03,
        'relevance_score': 0.72
    },
    # ... 15-27 more articles
]
```

---

## Sentiment Score Calculation

### The `analyze_asset_sentiment()` Method

This is the core analysis method that processes all news articles.

```python
def analyze_asset_sentiment(self, asset: Dict[str, Any], news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
```

### Step 1: Handle Empty News

```python
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
```

**Edge Case Handling:**
- If no news found, return neutral result
- Prevents errors from missing data
- Confidence set to 0 (unreliable)

### Step 2: Extract Scores

```python
sentiment_scores = [article['sentiment_score'] for article in news_articles]
relevance_scores = [article['relevance_score'] for article in news_articles]
```

**Creates Two Lists:**

**Sentiment Scores Example:**
```python
[-0.72, -0.68, 0.03, -0.45, -0.52, 0.21, -0.38, ...]
```

**Relevance Scores Example:**
```python
[0.95, 0.88, 0.72, 0.91, 0.84, 0.67, 0.93, ...]
```

### Step 3: Calculate Weighted Average

```python
weighted_sentiment = np.average(sentiment_scores, weights=relevance_scores)
```

**Why Weighted Average?**

Not all news is equally important:
- CEO resignation: High relevance (0.95) → Counts more
- Routine filing: Low relevance (0.67) → Counts less

**Formula:**
```
Weighted Average = Σ(sentiment × relevance) / Σ(relevance)
```

**Example Calculation:**

| Headline | Sentiment | Relevance | Weighted Value |
|----------|-----------|-----------|----------------|
| CEO resigns | -0.70 | 0.95 | -0.665 |
| Earnings miss | -0.72 | 0.93 | -0.670 |
| Volume increases | 0.05 | 0.70 | 0.035 |
| Restructuring | -0.50 | 0.88 | -0.440 |

```
Sum of weighted values: -0.665 + -0.670 + 0.035 + -0.440 = -1.74
Sum of relevance: 0.95 + 0.93 + 0.70 + 0.88 = 3.46
Weighted sentiment: -1.74 / 3.46 = -0.50
```

**Result:** Overall sentiment score of **-0.50** (moderately negative)

### Step 4: Determine Sentiment Label

```python
if weighted_sentiment <= self.sentiment_threshold_negative:  # ≤ -0.3
    sentiment_label = 'NEGATIVE'
elif weighted_sentiment >= self.sentiment_threshold_positive:  # ≥ +0.3
    sentiment_label = 'POSITIVE'
else:  # -0.3 to +0.3
    sentiment_label = 'NEUTRAL'
```

**Classification:**

| Score Range | Label | Interpretation |
|-------------|-------|----------------|
| ≤ -0.3 | NEGATIVE | Bad news dominates |
| -0.3 to +0.3 | NEUTRAL | Mixed or neutral news |
| ≥ +0.3 | POSITIVE | Good news dominates |

**Our Example:**
- Score: -0.50
- Since -0.50 ≤ -0.3 → Label: **NEGATIVE**

---

## Trend Analysis

### Purpose

Sentiment can change over time. Trend analysis tells you:
- Is the situation **IMPROVING**? (Bad news, but getting better)
- Is it **DETERIORATING**? (Getting worse)
- Is it **STABLE**? (Consistently bad/good)

### The Process

#### Step 1: Split Articles by Age

```python
recent_cutoff = datetime.now() - timedelta(days=30)

recent_articles = [
    a for a in news_articles 
    if datetime.fromisoformat(a['published_date'].replace('Z', '+00:00')).replace(tzinfo=None) > recent_cutoff
]

older_articles = [
    a for a in news_articles 
    if datetime.fromisoformat(a['published_date'].replace('Z', '+00:00')).replace(tzinfo=None) <= recent_cutoff
]
```

**What This Does:**
- **Recent articles**: Last 30 days
- **Older articles**: Everything before that (31-365 days ago)

**Example:**
- Today: October 15, 2024
- Recent cutoff: September 15, 2024
- Recent articles: Sep 16 - Oct 15 (30 days)
- Older articles: Oct 15, 2023 - Sep 15, 2024 (335 days)

#### Step 2: Calculate Average Sentiment for Each Period

```python
recent_sentiment = np.mean([a['sentiment_score'] for a in recent_articles]) if recent_articles else 0
older_sentiment = np.mean([a['sentiment_score'] for a in older_articles]) if older_articles else 0
```

**Example:**

**Recent Articles (last 30 days):**
```python
[-0.72, -0.45, -0.38, -0.52, 0.10]
recent_sentiment = mean([-0.72, -0.45, -0.38, -0.52, 0.10]) = -0.39
```

**Older Articles (31-365 days ago):**
```python
[-0.65, -0.70, -0.68, -0.55, -0.60, -0.63, ...]
older_sentiment = mean([...]) = -0.62
```

#### Step 3: Determine Trend

```python
if recent_sentiment > older_sentiment + 0.1:
    sentiment_trend = 'IMPROVING'
elif recent_sentiment < older_sentiment - 0.1:
    sentiment_trend = 'DETERIORATING'
else:
    sentiment_trend = 'STABLE'
```

**Logic:**

| Condition | Trend | Meaning |
|-----------|-------|---------|
| Recent > Older + 0.1 | IMPROVING | Getting noticeably better |
| Recent < Older - 0.1 | DETERIORATING | Getting noticeably worse |
| Otherwise | STABLE | No significant change |

**Our Example:**
- Recent sentiment: -0.39
- Older sentiment: -0.62
- Difference: -0.39 - (-0.62) = +0.23
- Since +0.23 > +0.1 → Trend: **IMPROVING**

**Interpretation:**
- Still negative overall
- But recent news is less bad than older news
- Situation may be stabilizing

**Another Example (Deteriorating):**
- Recent sentiment: -0.75
- Older sentiment: -0.40
- Difference: -0.75 - (-0.40) = -0.35
- Since -0.35 < -0.1 → Trend: **DETERIORATING**
- Getting much worse!

---

## Key Themes Extraction

### Purpose

Identify **what** people are talking about. Is it:
- Earnings problems?
- Regulatory issues?
- Management changes?
- Debt concerns?

### The `extract_key_themes()` Method

```python
def extract_key_themes(self, news_articles: List[Dict[str, Any]]) -> List[str]:
```

### Theme Categories and Keywords

The system looks for 7 predefined themes:

```python
theme_keywords = {
    'earnings': ['earnings', 'revenue', 'profit', 'loss', 'guidance'],
    'regulatory': ['investigation', 'regulatory', 'compliance', 'lawsuit', 'settlement'],
    'management': ['CEO', 'resign', 'leadership', 'board', 'management'],
    'market_share': ['competitor', 'market share', 'competitive'],
    'financial_health': ['debt', 'credit', 'rating', 'downgrade', 'bankruptcy'],
    'operations': ['restructuring', 'layoffs', 'production', 'supply chain'],
    'growth': ['expansion', 'partnership', 'acquisition', 'buyback', 'investment']
}
```

### Processing Steps

#### Step 1: Initialize Theme Counters

```python
theme_counts = {theme: 0 for theme in theme_keywords}
# Result: {'earnings': 0, 'regulatory': 0, 'management': 0, ...}
```

#### Step 2: Count Keyword Matches

```python
for article in news_articles:
    headline_lower = article['headline'].lower()
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in headline_lower for keyword in keywords):
            theme_counts[theme] += 1
```

**What This Does:**
- Loop through each article
- Check if headline contains any keywords for each theme
- Increment counter if match found

#### Step 3: Sort and Return Top Themes

```python
sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
return [theme for theme, count in sorted_themes if count > 0][:5]
```

**Returns:** Top 5 themes (or fewer if less than 5 found)

### Example

**Headlines:**
1. "ACME reports quarterly earnings miss, revenue down 18%" ← **earnings**
2. "CEO of ACME resigns amid strategic disagreements with board" ← **management**
3. "ACME faces investigation over accounting practices" ← **regulatory**
4. "Credit rating agency downgrades ACME debt to junk status" ← **financial_health**
5. "ACME announces major restructuring, plans to cut 3000 jobs" ← **operations**
6. "Analyst downgrades ACME citing regulatory concerns" ← **regulatory**
7. "ACME misses guidance for third consecutive quarter" ← **earnings**

**Theme Counts:**
```python
{
    'earnings': 2,           # Headlines 1, 7
    'regulatory': 2,         # Headlines 3, 6
    'management': 1,         # Headline 2
    'financial_health': 1,   # Headline 4
    'operations': 1,         # Headline 5
    'market_share': 0,
    'growth': 0
}
```

**Sorted (descending):**
```python
[
    ('earnings', 2),
    ('regulatory', 2),
    ('financial_health', 1),
    ('management', 1),
    ('operations', 1),
    ('market_share', 0),
    ('growth', 0)
]
```

**Final Result:**
```python
['earnings', 'regulatory', 'financial_health', 'management', 'operations']
```

**Interpretation:**
- Top concerns: Earnings and regulatory issues (tied)
- Also notable: Financial health, management, operations
- No issues with market share or growth

---

## Confidence Calculation

### Purpose

**How confident should you be in the sentiment score?**

High confidence = Trust this score
Low confidence = Take with grain of salt

### The Formula

```python
sentiment_std = np.std(sentiment_scores)
confidence = min(1.0, len(news_articles) / 20.0) * (1.0 - min(1.0, sentiment_std))
```

### Breaking It Down

#### Component 1: Article Count Factor

```python
article_count_factor = min(1.0, len(news_articles) / 20.0)
```

**Logic:**
- More articles = More confidence
- 20+ articles = Maximum confidence from count (1.0)
- Fewer articles = Lower confidence

**Examples:**
| Articles | Calculation | Factor |
|----------|-------------|--------|
| 5 | min(1.0, 5/20) | 0.25 |
| 10 | min(1.0, 10/20) | 0.50 |
| 20 | min(1.0, 20/20) | 1.00 |
| 30 | min(1.0, 30/20) | 1.00 (capped) |

#### Component 2: Consistency Factor

```python
sentiment_std = np.std(sentiment_scores)
consistency_factor = 1.0 - min(1.0, sentiment_std)
```

**Standard Deviation (std):**
- Measures how spread out the sentiment scores are
- Low std = Articles agree with each other (consistent)
- High std = Articles disagree (inconsistent)

**Consistency Factor:**
- If std = 0 (all identical): consistency = 1.0 - 0 = 1.0 (perfect)
- If std = 0.5 (moderate spread): consistency = 1.0 - 0.5 = 0.5
- If std ≥ 1.0 (very spread): consistency = 1.0 - 1.0 = 0.0 (no confidence)

**Examples:**

**Consistent Sentiment:**
```python
sentiment_scores = [-0.70, -0.72, -0.68, -0.71, -0.69]
std = 0.015  # Very low (all similar)
consistency = 1.0 - 0.015 = 0.985
```

**Inconsistent Sentiment:**
```python
sentiment_scores = [-0.80, 0.50, -0.30, 0.60, -0.90, 0.40]
std = 0.62  # High (very different)
consistency = 1.0 - 0.62 = 0.38
```

#### Final Confidence

```python
confidence = article_count_factor × consistency_factor
```

### Complete Examples

#### Example 1: High Confidence

**Scenario:**
- 25 articles (plenty of data)
- Sentiment scores: [-0.70, -0.72, -0.68, -0.71, -0.69, ...]
- Standard deviation: 0.02 (very consistent)

**Calculation:**
```
Article factor: min(1.0, 25/20) = 1.0
Consistency: 1.0 - 0.02 = 0.98
Confidence: 1.0 × 0.98 = 0.98 (98%)
```

**Interpretation:** Very reliable - lots of articles, all saying the same thing

#### Example 2: Moderate Confidence

**Scenario:**
- 15 articles (decent amount)
- Sentiment scores: [-0.65, -0.50, -0.30, -0.70, 0.10, ...]
- Standard deviation: 0.35 (moderate variation)

**Calculation:**
```
Article factor: min(1.0, 15/20) = 0.75
Consistency: 1.0 - 0.35 = 0.65
Confidence: 0.75 × 0.65 = 0.49 (49%)
```

**Interpretation:** Somewhat reliable - reasonable data, but mixed messages

#### Example 3: Low Confidence

**Scenario:**
- 8 articles (limited data)
- Sentiment scores: [-0.80, 0.60, -0.40, 0.50, -0.90, 0.30, -0.20, 0.70]
- Standard deviation: 0.58 (high variation)

**Calculation:**
```
Article factor: min(1.0, 8/20) = 0.40
Consistency: 1.0 - 0.58 = 0.42
Confidence: 0.40 × 0.42 = 0.17 (17%)
```

**Interpretation:** Unreliable - few articles and they contradict each other

### Confidence Interpretation Guide

| Confidence | Interpretation | Action |
|------------|----------------|--------|
| 80-100% | Very High | Trust this signal strongly |
| 60-79% | High | Reliable, use for decisions |
| 40-59% | Moderate | Consider, but verify |
| 20-39% | Low | Use with caution |
| 0-19% | Very Low | Don't rely on this |

---

## Impact Score Calculation

### The `calculate_sentiment_impact_score()` Method

Combines sentiment analysis with risk metrics to create a unified impact score.

```python
def calculate_sentiment_impact_score(self, sentiment_result: Dict[str, Any], 
                                   risk_metrics: Dict[str, Any]) -> float:
```

### Purpose

**Question:** How concerning is this investment overall?

**Combines:**
- Sentiment data (news analysis)
- Risk metrics (financial analysis from Stage 2)

**Output:** Single score from 0-1
- 0 = Not concerning at all
- 1 = Extremely concerning

### Step 1: Base Sentiment Impact

```python
sentiment_impact = abs(sentiment_result['sentiment_score']) * sentiment_result['confidence']
```

**Formula:** Impact = |Sentiment Score| × Confidence

**Why Absolute Value?**
- Both very negative (-0.8) and very positive (+0.8) indicate strong sentiment
- For risk, strong sentiment of any kind matters
- Absolute value treats both equally

**Examples:**

**Strong Negative Sentiment (High Impact):**
```python
sentiment_score = -0.75
confidence = 0.90
impact = abs(-0.75) × 0.90 = 0.675
```

**Weak Negative Sentiment (Low Impact):**
```python
sentiment_score = -0.25
confidence = 0.40
impact = abs(-0.25) × 0.40 = 0.10
```

**Strong Positive Sentiment (High Impact):**
```python
sentiment_score = 0.80
confidence = 0.85
impact = abs(0.80) × 0.85 = 0.68
```

### Step 2: Trend Adjustment

```python
if sentiment_result['sentiment_trend'] == 'DETERIORATING':
    sentiment_impact *= 1.2  # Increase by 20%
elif sentiment_result['sentiment_trend'] == 'IMPROVING':
    sentiment_impact *= 0.8  # Decrease by 20%
```

**Logic:**
- **DETERIORATING**: Getting worse → More concerning → +20%
- **IMPROVING**: Getting better → Less concerning → -20%
- **STABLE**: No change → Keep as is

**Example:**

**Base impact: 0.60**

**If DETERIORATING:**
```python
adjusted_impact = 0.60 × 1.2 = 0.72
```

**If IMPROVING:**
```python
adjusted_impact = 0.60 × 0.8 = 0.48
```

### Step 3: Normalize Risk Metrics

```python
risk_score = risk_metrics.get('risk_score', 0) / 7.0  # Normalize to 0-1
volatility_score = min(1.0, risk_metrics.get('volatility', 0) / 0.5)  # Cap at 50% vol
```

**Risk Score Normalization:**
- Risk score ranges from 0-7 (number of flags triggered)
- Divide by 7 to get 0-1 scale
- Example: 5 flags → 5/7 = 0.71

**Volatility Score Normalization:**
- Volatility is annual percentage (e.g., 0.35 = 35%)
- Divide by 0.5 (50%) to normalize
- Cap at 1.0 if exceeds 50%
- Examples:
  - 20% volatility → 0.20/0.50 = 0.40
  - 35% volatility → 0.35/0.50 = 0.70
  - 60% volatility → min(1.0, 0.60/0.50) = 1.00 (capped)

### Step 4: Weighted Combination

```python
combined_score = (
    0.4 * sentiment_impact +
    0.3 * risk_score +
    0.3 * volatility_score
)
```

**Weights:**
- Sentiment Impact: **40%** (most important - provides context)
- Risk Score: **30%** (number of red flags)
- Volatility Score: **30%** (price instability)

**Why These Weights:**
- Sentiment provides forward-looking information (40%)
- Risk flags show current problems (30%)
- Volatility shows instability (30%)
- Total: 100%

### Step 5: Cap at 1.0

```python
return min(1.0, combined_score)
```

Ensures score never exceeds 1.0 (even if calculation would)

### Complete Example

**Investment: ACME Corp**

**Sentiment Data:**
- Sentiment score: -0.65 (negative)
- Confidence: 0.85 (high)
- Trend: DETERIORATING

**Risk Metrics:**
- Risk score: 5 (out of 7 flags)
- Volatility: 38% (0.38)

**Calculation:**

**Step 1: Base Sentiment Impact**
```
sentiment_impact = abs(-0.65) × 0.85 = 0.5525
```

**Step 2: Trend Adjustment (DETERIORATING)**
```
sentiment_impact = 0.5525 × 1.2 = 0.663
```

**Step 3: Normalize Risk Metrics**
```
risk_score = 5 / 7 = 0.714
volatility_score = 0.38 / 0.50 = 0.76
```

**Step 4: Weighted Combination**
```
combined_score = (0.4 × 0.663) + (0.3 × 0.714) + (0.3 × 0.76)
               = 0.2652 + 0.2142 + 0.228
               = 0.7074
```

**Step 5: Cap at 1.0**
```
final_score = min(1.0, 0.7074) = 0.7074
```

**Result: Impact Score = 0.71 (71% - Highly Concerning)**

### Impact Score Interpretation

| Score | Level | Interpretation | Action |
|-------|-------|----------------|--------|
| 0.80-1.00 | Critical | Severe concerns, multiple red flags | Consider immediate exit |
| 0.60-0.79 | High | Significant concerns | Review closely, prepare to sell |
| 0.40-0.59 | Moderate | Some concerns | Monitor situation |
| 0.20-0.39 | Low | Minor concerns | Keep on watchlist |
| 0.00-0.19 | Minimal | Few/no concerns | Likely safe to hold |

---

## Real-World Examples

### Example 1: Severe Case

**Stock: TECH Inc (RED-flagged)**

**Sentiment Analysis Results:**
```python
{
    'symbol': 'TECH',
    'sector': 'Technology',
    'risk_rating': 'RED',
    
    # Sentiment metrics
    'sentiment_score': -0.68,
    'sentiment_label': 'NEGATIVE',
    'confidence': 0.88,
    'sentiment_trend': 'DETERIORATING',
    
    # News analysis
    'news_count': 24,
    'recent_news_count': 8,
    'negative_news_ratio': 0.79,
    
    # Key insights
    'key_themes': ['regulatory', 'earnings', 'management', 'financial_health'],
    'recent_sentiment': -0.75,
    'older_sentiment': -0.55,
    
    # Sample headlines
    'recent_significant_news': [
        {
            'headline': 'TECH faces SEC investigation over accounting practices',
            'sentiment_score': -0.85,
            'source': 'Reuters',
            'published_date': '2024-10-10'
        },
        {
            'headline': 'CEO of TECH resigns amid board disagreements',
            'sentiment_score': -0.78,
            'source': 'Financial Times',
            'published_date': '2024-10-05'
        },
        {
            'headline': 'TECH misses earnings for third consecutive quarter',
            'sentiment_score': -0.70,
            'source': 'Bloomberg',
            'published_date': '2024-09-28'
        }
    ]
}
```

**Interpretation:**

**Overall Assessment:** VERY CONCERNING

**Red Flags:**
- ✗ Sentiment score: -0.68 (clearly negative)
- ✗ High confidence: 88% (reliable signal)
- ✗ Trend: DETERIORATING (getting worse)
- ✗ Recent sentiment (-0.75) worse than older (-0.55)
- ✗ 79% of news is negative
- ✗ Top themes: regulatory, earnings, management, financial health (all problems)

**Key Issues:**
1. SEC investigation (legal/regulatory risk)
2. CEO resignation (leadership crisis)
3. Three consecutive earnings misses (fundamental problems)

**Recommendation:** Strong sell candidate - multiple severe issues, deteriorating situation

### Example 2: Improving Situation

**Stock: FINC Corp (RED-flagged, but recovering)**

**Sentiment Analysis Results:**
```python
{
    'symbol': 'FINC',
    'risk_rating': 'RED',
    
    'sentiment_score': -0.42,
    'sentiment_label': 'NEGATIVE',
    'confidence': 0.65,
    'sentiment_trend': 'IMPROVING',
    
    'news_count': 18,
    'recent_news_count': 6,
    'negative_news_ratio': 0.55,
    
    'key_themes': ['financial_health', 'operations', 'earnings'],
    'recent_sentiment': -0.25,
    'older_sentiment': -0.52,
    
    'recent_significant_news': [
        {
            'headline': 'FINC announces debt restructuring plan',
            'sentiment_score': -0.15,
            'source': 'Wall Street Journal',
            'published_date': '2024-10-08'
        },
        {
            'headline': 'New CEO of FINC outlines turnaround strategy',
            'sentiment_score': 0.35,
            'source': 'Bloomberg',
            'published_date': '2024-10-01'
        }
    ]
}
```

**Interpretation:**

**Overall Assessment:** CONCERNING BUT IMPROVING

**Mixed Signals:**
- ✗ Still negative sentiment (-0.42)
- ✓ Trend is IMPROVING (recent: -0.25 vs older: -0.52)
- ✓ Recent news less negative
- ~ Moderate confidence (65%)

**Key Issues:**
1. Still has financial health problems (debt)
2. Operations and earnings issues
3. But: New CEO with turnaround plan
4. But: Debt restructuring underway

**Recommendation:** 
- Not a buy yet (still RED)
- But monitor closely - could be bottoming out
- If next quarter shows improvement, might upgrade to YELLOW
- Hold decision depends on risk tolerance

### Example 3: Low Confidence Case

**Stock: SMCO Inc (RED-flagged)**

**Sentiment Analysis Results:**
```python
{
    'symbol': 'SMCO',
    'risk_rating': 'RED',
    
    'sentiment_score': -0.35,
    'sentiment_label': 'NEGATIVE',
    'confidence': 0.22,  # LOW CONFIDENCE
    'sentiment_trend': 'STABLE',
    
    'news_count': 7,  # Few articles
    'recent_news_count': 2,
    'negative_news_ratio': 0.57,
    
    'key_themes': ['operations'],
    'sentiment_volatility': 0.68  # High variation
}
```

**Interpretation:**

**Overall Assessment:** UNCERTAIN

**Issues:**
- Only 7 articles (low sample size)
- High sentiment volatility (0.68) - articles disagree
- Low confidence (22%) - don't trust this signal
- Trend: STABLE (no clear direction)

**Recommendation:**
- Can't rely on sentiment analysis here
- Need more information
- Look at other data sources
- Consider fundamental analysis more heavily
- Sentiment inconclusive

---

## Technical Implementation Details

### Dependencies

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
from textblob import TextBlob
import re
```

**Key Libraries:**
- **pandas**: Data manipulation
- **numpy**: Statistical calculations (mean, std, average)
- **datetime**: Time-based analysis
- **textblob**: NLP sentiment analysis (not actively used in mock version)
- **typing**: Type hints for code clarity

### TextBlob Integration

The code imports TextBlob but doesn't actively use it in the mock version. In a real implementation:

```python
# Real implementation (not in current code):
def get_real_sentiment(headline: str) -> float:
    blob = TextBlob(headline)
    return blob.sentiment.polarity  # -1.0 to +1.0
```

**Current Mock Version:**
- Pre-assigns sentiment scores when generating articles
- Faster for demonstration
- In production, would use TextBlob to analyze real headlines

### Data Structures

**Article Dictionary:**
```python
{
    'headline': str,           # The news text
    'source': str,             # News outlet
    'published_date': str,     # ISO format datetime
    'url': str,                # Article URL
    'sentiment_score': float,  # -1.0 to +1.0
    'relevance_score': float   # 0.0 to 1.0
}
```

**Sentiment Result Dictionary:**
```python
{
    'symbol': str,
    'sector': str,
    'risk_rating': str,
    'sentiment_score': float,
    'sentiment_label': str,
    'confidence': float,
    'sentiment_trend': str,
    'news_count': int,
    'recent_news_count': int,
    'negative_news_ratio': float,
    'key_themes': List[str],
    'recent_significant_news': List[Dict],
    'recent_sentiment': float,
    'older_sentiment': float,
    'sentiment_volatility': float,
    'analysis_timestamp': str
}
```

### Performance Considerations

**Time Complexity:**
- Article generation: O(n) where n = number of articles
- Sentiment calculation: O(n) - single pass through articles
- Theme extraction: O(n × m) where m = number of themes (constant 7)
- Overall: O(n) - linear time

**For 5 RED assets with 20 articles each:**
- Total articles: 100
- Processing time: ~1-2 seconds
- Negligible compared to ML analysis (Stage 3)

### Edge Cases Handled

1. **No articles found**: Returns neutral result with 0 confidence
2. **All articles same date**: Recent/older split still works
3. **Empty recent or older**: Uses 0 for missing period
4. **High volatility**: Capped at 1.0 for consistency calculation
5. **Very high article count**: Article factor capped at 1.0

---

## Summary

### What Sentiment Analysis Provides

**Context:**
- WHY an asset is risky (not just that it is)
- What people are talking about
- Whether situation is improving or worsening

**Early Warnings:**
- News often precedes price changes
- Sentiment deterioration signals future problems
- Improving sentiment hints at recovery

**Confidence Levels:**
- Not all sentiment signals are equal
- High confidence = trust it
- Low confidence = verify elsewhere

**Actionable Insights:**
- Combined with risk metrics (impact score)
- Specific themes to investigate
- Trend direction for timing

### Key Metrics Explained

1. **Sentiment Score (-1 to +1)**: Overall news tone
2. **Sentiment Label**: NEGATIVE/NEUTRAL/POSITIVE classification
3. **Confidence (0-1)**: How reliable the signal is
4. **Sentiment Trend**: IMPROVING/DETERIORATING/STABLE
5. **Key Themes**: What issues dominate the news
6. **Impact Score (0-1)**: Combined risk assessment

### Best Practices for Using Results

**High Confidence (>0.8):**
- Trust the sentiment signal
- Use for decision-making
- Act on deteriorating trends

**Moderate Confidence (0.5-0.8):**
- Consider the signal
- Verify with other sources
- Use as one input among many

**Low Confidence (<0.5):**
- Don't rely on sentiment alone
- Gather more information
- Focus on fundamental analysis

**Always:**
- Combine with Stage 2 (Core Analysis)
- Combine with Stage 3 (ML Analysis)
- Look for consensus across all stages
- Consider your own research and judgment

---

## File Reference

**Location:** `pipeline/sentiment_analysis.py`

**Class:** `SentimentAnalysisEngine`

**Main Methods:**
1. `analyze_sentiment()` - Entry point
2. `fetch_news_for_asset()` - Article generation
3. `analyze_asset_sentiment()` - Core analysis
4. `extract_key_themes()` - Theme identification
5. `calculate_sentiment_impact_score()` - Impact scoring

**Lines of Code:** ~279

**Integration:** Called by main pipeline in `app.py` after Stage 2 (Core Analysis)
