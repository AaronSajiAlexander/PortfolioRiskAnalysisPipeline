# Mock Data Generation Guide

This guide explains how the test/demo data is created for the portfolio risk analysis application, written in simple, everyday language.

---

## Why Do We Need Mock Data?

### The Problem

This application is designed to analyze real investment portfolios using data from Bloomberg (a professional financial data service). But:

- Bloomberg charges thousands of dollars per month for access
- Most users don't have Bloomberg accounts
- We want you to try the application without spending money
- We need realistic data to demonstrate all features

### The Solution

We created a "mock data generator" - a system that creates fake but realistic financial data that looks and behaves like real market data. Think of it like:

- **Movie props**: They look real on camera but aren't actual items
- **Flight simulator**: Feels like flying a real plane without leaving the ground
- **Practice exam**: Has the same format as the real test but different questions

The mock data is realistic enough to demonstrate how the system works, but it's completely made up.

---

## How the Mock Data System Works

### Overview: The MockBloombergData Class

The entire mock data system is contained in a single Python class called `MockBloombergData`. Think of it as a factory that manufactures different types of financial data.

**Location**: `utils/mock_data.py`

**What it creates:**
1. Individual asset/stock information
2. Historical price data (6-12 months of daily prices)
3. Trading volumes
4. Market index data (S&P 500, NASDAQ, Dow Jones)
5. Economic indicators (interest rates, GDP, inflation)
6. News headlines for sentiment analysis

Let's break down each one.

---

## 1. Generating Asset/Stock Information

### What Gets Created

For each fake company in your portfolio, the system creates:
- Company name (e.g., "Global Technologies")
- Stock symbol (e.g., "GTCH")
- Industry sector (Technology, Healthcare, etc.)
- Current stock price
- Market capitalization (total company value)
- Other financial metrics

### How Company Names Are Generated

**The System Uses Two Lists:**

**Prefixes** (First word):
- Global, Advanced, United, American, International, Pacific
- National, Premier, Superior, Dynamic, Strategic, Innovative
- First, Capital, Metro, Regional, Consolidated, Alliance

**Suffixes** (Second word):
- Corp, Inc, LLC, Group, Holdings, Systems
- Technologies, Solutions, Industries, Enterprises
- Partners, Capital, Resources, Services, International, Global

**Process:**
1. Randomly pick one prefix: "Global"
2. Randomly pick one suffix: "Technologies"
3. Combine them: "Global Technologies"

**Result**: Realistic-sounding company names that don't exist in real life.

**Examples:**
- "Advanced Solutions"
- "Pacific Holdings"
- "Dynamic Systems"
- "Strategic Resources"

### How Stock Symbols Are Generated

**Real Examples:**
- Apple = AAPL (4 letters)
- Microsoft = MSFT (4 letters)
- Google = GOOG (4 letters)
- AT&T = T (1 letter)

**Our System:**
1. Pick a length: 2, 3, or 4 characters (most commonly 3)
2. Start with a random letter from a curated list
3. Add more random letters
4. Result: "GTCH", "ADV", "PSL", "DYN"

**Why this works:** Real symbols are just random letter combinations, so our fake ones look identical.

### How Prices Are Determined

Prices vary by sector to match reality:

**Technology & Healthcare stocks:**
- Price range: $50 - $400
- Why: Tech stocks tend to be expensive (think Apple, Google)
- Volatility: Higher (15-45% annual fluctuation)

**Financial Services & Energy stocks:**
- Price range: $30 - $150
- Why: Banks and oil companies typically mid-range
- Volatility: Moderate to high (20-50%)

**Utilities & Consumer Staples:**
- Price range: $40 - $120
- Why: Electric companies, food producers are stable
- Volatility: Lower (10-25%)

**Other sectors:**
- Price range: $35 - $200
- Volatility: Moderate (15-35%)

**Example:**
```
Sector: Technology
Base Price: $156.23 (randomly chosen from $50-$400)
Volatility: 32% (randomly chosen from 15-45%)
```

### How Market Cap Is Calculated

**Market Cap** = Stock Price × Number of Shares Outstanding

**Process:**
1. Generate random shares outstanding: 50 million to 2 billion
2. Multiply by stock price
3. Result = Market capitalization

**Example:**
```
Stock Price: $156.23
Shares Outstanding: 500,000,000
Market Cap: $156.23 × 500,000,000 = $78,115,000,000 (78 billion)
```

This mirrors real companies:
- Small companies: $300M - $2B market cap
- Medium companies: $2B - $10B
- Large companies: $10B - $200B+

### Additional Financial Metrics

**P/E Ratio (Price-to-Earnings):**
- Range: 8 to 35
- 10% of companies have no P/E (unprofitable)
- Realistic because profitable companies typically have P/E in this range

**Dividend Yield:**
- Range: 0% to 6%
- 70% of companies pay dividends (others don't)
- Realistic for mature companies

**Exchange:**
- NYSE or NASDAQ (the two major U.S. stock exchanges)
- Weighted toward these major exchanges

---

## 2. Generating Historical Prices

### The Challenge

We need to create 252 days (one trading year) of historical prices that:
- Look realistic with ups and downs
- End at today's current price
- Show proper volatility patterns
- Avoid impossible scenarios (negative prices, wild swings)

### The Solution: Geometric Brownian Motion

This sounds complicated, but it's actually how real stock prices behave.

**Simple Explanation:**

Imagine a drunk person walking. Each step they take:
- Is somewhat random (could go left or right)
- Is influenced by a general direction (trying to walk home)
- Has some consistent randomness (their level of drunkenness)

Stock prices work similarly:
- **Random component**: Daily market noise, news, sentiment
- **General direction (drift)**: Overall trend up or down
- **Volatility**: How much the price bounces around

### The Math (Simplified)

**Formula Components:**

1. **Drift (μ - mu)**:
   - Range: -5% to +15% annually
   - Represents the overall trend
   - Negative = stock tends downward, Positive = tends upward
   - Example: μ = 0.10 means 10% annual growth trend

2. **Volatility (σ - sigma)**:
   - Range: 15% to 45% annually
   - Represents how wild the swings are
   - Higher σ = more dramatic daily changes
   - Example: σ = 0.30 means 30% annual volatility

3. **Time Step (dt)**:
   - 1/252 (one trading day)
   - There are 252 trading days in a year (365 minus weekends/holidays)

4. **Random Shock (dW)**:
   - Random number from a "normal distribution"
   - Think of it like rolling weighted dice
   - Most results near zero (small changes)
   - Occasionally large positive or negative (big price moves)

### The Process

**Step 1: Start with Today's Price**
- Current price: $156.23

**Step 2: Work Backwards 252 Days**

For each day going backward:

```
Previous Price = Current Price × e^((μ - 0.5×σ²)×dt + σ×random_shock)
```

Don't worry about the exact math. What this does:
- Takes the current price
- Applies the drift (general trend)
- Adds random movement based on volatility
- Ensures prices can't go negative

**Step 3: Add Safety Guards**

Prevent unrealistic scenarios:
- If price goes above 2× starting price → cap it
- If price goes below 0.3× starting price → floor it
- Ensure all prices stay positive (minimum $0.01)

**Step 4: Reverse the List**

We worked backwards, so flip the list to get chronological order.

### Example Output

**Input:**
- Current price: $100
- Drift: 10% annual
- Volatility: 25% annual

**Output (sample):**
```
Day 1 (252 days ago): $98.50
Day 2: $99.10
Day 3: $97.80
...
Day 250: $99.50
Day 251: $100.20
Day 252 (today): $100.00
```

Notice:
- Daily fluctuations (up and down)
- Overall slight upward trend (10% drift)
- Realistic volatility
- Ends exactly at current price

### Why This Works

Geometric Brownian Motion is the **actual mathematical model** used by financial professionals to simulate stock prices. We're using the real formula, just with made-up starting parameters.

---

## 3. Generating Trading Volumes

### What Are Trading Volumes?

Trading volume = How many shares were bought/sold on a given day.

**Real-world pattern:**
- Calm days: Lower volume (people just watching)
- Volatile days: Higher volume (everyone rushing to trade)
- Big news days: Spikes in volume

### How We Simulate This

**Step 1: Pick a Base Volume**
- Random number between 100,000 and 10,000,000 shares/day
- Represents typical daily trading for this stock

**Step 2: Correlate with Price Movements**

For each day:

```
Price Change % = |Today's Price - Yesterday's Price| / Yesterday's Price
Volume Multiplier = 1 + (Price Change × 3)
Daily Volume = Base Volume × Multiplier × Random Factor
```

**What this means:**

**Calm day** (price changed 1%):
- Volume Multiplier = 1 + (0.01 × 3) = 1.03
- Volume ≈ base volume (normal trading)

**Volatile day** (price changed 5%):
- Volume Multiplier = 1 + (0.05 × 3) = 1.15
- Volume = 15% higher than normal

**Crazy day** (price changed 10%):
- Volume Multiplier = 1 + (0.10 × 3) = 1.30
- Volume = 30% higher (panic/excitement)

**Step 3: Add Randomness**

Multiply by a random factor (0.5 to 2.0) to ensure day-to-day variation.

### Example

**Base Volume**: 1,000,000 shares/day

**Calm Day:**
- Price change: 0.5%
- Multiplier: 1.015
- Random factor: 1.2
- Volume: 1,000,000 × 1.015 × 1.2 = 1,218,000 shares

**Volatile Day:**
- Price change: 8%
- Multiplier: 1.24
- Random factor: 1.8
- Volume: 1,000,000 × 1.24 × 1.8 = 2,232,000 shares

This matches real behavior: volume spikes when prices move sharply.

---

## 4. Generating Market Index Data

### What Are Market Indices?

**Market indices** track overall market performance:
- **S&P 500**: 500 largest U.S. companies
- **NASDAQ**: Technology-heavy index
- **Dow Jones**: 30 major industrial companies

### Why We Need Them

The application calculates **Beta** - how a stock moves compared to the overall market. To calculate Beta, we need market index data.

### How We Generate Index Data

**Predefined Starting Points:**
- S&P 500: Starts at 4,200 (realistic 2024 level)
- NASDAQ: Starts at 13,000 (realistic level)
- Dow Jones: Starts at 34,000 (realistic level)

**Volatility Levels:**
- S&P 500: 16% annual (moderate)
- NASDAQ: 20% annual (higher - tech is volatile)
- Dow Jones: 15% annual (lower - blue chips are stable)

**Process:**

Use the **same Geometric Brownian Motion** process as individual stocks:
1. Start with current index level
2. Apply drift and volatility
3. Generate 252 days of historical data
4. Ensure realistic movements

**Result:** Three market indices with 252 days of data each, used for Beta calculations.

---

## 5. Generating Economic Indicators

### What Gets Created

**Interest Rates:**
- Fed Funds Rate: 0.25% - 5.5%
- 10-Year Treasury: 1.5% - 6.0%
- 30-Year Treasury: 2.0% - 6.5%

**Economic Metrics:**
- GDP Growth: -2.0% to 4.0%
- Unemployment: 3.5% to 8.0%
- Inflation: 1.0% to 6.0%
- Consumer Confidence: 80 to 140

**Market Metrics:**
- VIX (volatility index): 12 to 35
- Dollar Index: 95 to 110
- Oil Price: $60 - $120/barrel
- Gold Price: $1,800 - $2,100/ounce

### How They're Generated

Simple random number generation within realistic ranges:

```python
fed_funds_rate = random.uniform(0.25, 5.5)
unemployment = random.uniform(3.5, 8.0)
oil_price = random.uniform(60, 120)
```

**Why these ranges?**

Based on historical data from the past 20 years:
- Fed Funds Rate was near 0% in 2020, peaked at 5.5% in 2007
- Unemployment ranged from 3.5% (2019) to 14.7% (2020 COVID)
- Oil has been as low as $20 and as high as $140

Our ranges capture realistic scenarios.

---

## 6. Generating News Headlines

### The Challenge

For sentiment analysis, we need financial news headlines that:
- Sound realistic
- Have clear positive, negative, or neutral sentiment
- Vary by asset
- Cover different time periods (up to 1 year)

### How Headlines Are Created

**Step 1: Template Categories**

We have three template lists:

**Positive Templates:**
```
"{symbol} reports strong quarterly earnings, beats estimates"
"{symbol} announces major partnership with industry leader"
"Analysts upgrade {symbol} target price following positive outlook"
"{symbol} receives FDA approval for breakthrough product"
"{symbol} CEO outlines ambitious growth strategy at investor day"
```

**Negative Templates:**
```
"{symbol} misses quarterly expectations, guidance lowered"
"Regulatory concerns weigh on {symbol} stock price"
"{symbol} faces increased competition in core markets"
"Credit rating agency places {symbol} on negative watch"
"{symbol} announces restructuring, job cuts expected"
```

**Neutral Templates:**
```
"{symbol} announces quarterly dividend payment"
"{symbol} schedules earnings call for next week"
"Trading volume in {symbol} remains elevated"
"{symbol} stock included in new ESG index"
"Analyst maintains neutral rating on {symbol} shares"
```

**Step 2: Sentiment Bias Selection**

When generating news for an asset, we can specify sentiment bias:

**For RED (high-risk) stocks:**
- 70% negative headlines
- 20% neutral headlines
- 10% positive headlines
- **Why:** RED stocks have problems, so news should reflect that

**For YELLOW stocks:**
- 30% negative
- 40% neutral
- 30% positive
- **Why:** Mixed situation, mixed news

**For GREEN stocks:**
- 60% positive
- 30% neutral
- 10% negative
- **Why:** Healthy companies get good news

**If unspecified (mixed):**
- Equal distribution across all three

**Step 3: Headline Generation Process**

For each asset:

1. **Decide how many headlines**: Random number between 10-25
   - Reflects varying news coverage
   - Some stocks are newsworthy (25 articles), others not (10 articles)

2. **For each headline:**
   - Pick template category based on sentiment bias
   - Select random template from that category
   - Replace {symbol} with actual stock symbol
   - Assign random publication date (1-365 days ago)
   - Assign random news source

3. **Add metadata:**
   - Publication date
   - Source (Reuters, Bloomberg, WSJ, etc.)

### Example Output

**Stock Symbol: GTCH (Currently RED-rated)**

**Generated Headlines:**
```
1. "GTCH misses quarterly expectations, guidance lowered"
   - Date: 2024-08-15
   - Source: Reuters
   - Sentiment: Negative

2. "Regulatory concerns weigh on GTCH stock price"
   - Date: 2024-07-22
   - Source: Financial Times
   - Sentiment: Negative

3. "GTCH announces quarterly dividend payment"
   - Date: 2024-06-10
   - Source: Bloomberg
   - Sentiment: Neutral

4. "Credit rating agency places GTCH on negative watch"
   - Date: 2024-09-01
   - Source: Wall Street Journal
   - Sentiment: Negative

... (6-21 more headlines)
```

**Total:** 10-25 headlines spread over the past year, weighted toward negative (because it's RED).

### News Source Variety

Headlines are randomly assigned to real news sources:
- Reuters
- Bloomberg
- MarketWatch
- Yahoo Finance
- CNBC
- Financial Times
- Wall Street Journal
- Seeking Alpha

This adds realism - real stocks get coverage from multiple outlets.

---

## How Realistic Is the Mock Data?

### What's Realistic ✅

**Mathematical Models:**
- Uses **actual financial formulas** (Geometric Brownian Motion)
- Volatility ranges match real markets
- Price distributions follow real patterns

**Relationships:**
- Volume correlates with price changes (just like reality)
- Tech stocks are more volatile (matches reality)
- Utilities are stable (matches reality)

**Ranges:**
- Stock prices, P/E ratios, dividend yields all within real bounds
- Economic indicators match historical ranges
- News templates based on real financial journalism

**Behavior:**
- Prices can go up or down (randomness)
- Trends exist but aren't guaranteed (drift)
- Occasional big moves (fat-tail distribution)

### What's Not Realistic ❌

**No Real Events:**
- Mock data doesn't respond to actual news
- Real stocks react to earnings, Fed announcements, wars, pandemics
- Our stocks just follow mathematical patterns

**No Correlations:**
- Real stocks in same sector move together
- Our stocks are independent (don't influence each other)
- Real market has correlations we don't simulate

**Simplified Dynamics:**
- Real markets have market makers, high-frequency trading, institutional flows
- We just use mathematical formulas
- Missing complex microstructure

**Company Names:**
- Obviously fake (no real company called "Global Technologies Corp")
- Symbols are random combinations

**News Headlines:**
- Template-based, not real journalism
- Same basic stories repeated with different symbols
- Real news is more varied and specific

### The Bottom Line

The mock data is:
- **Mathematically accurate**: Uses real financial models
- **Visually realistic**: Looks like real market data
- **Functionally adequate**: Good enough to demonstrate the system
- **Clearly fake**: Won't fool anyone who examines it closely

Think of it like a **flight simulator**:
- Physics are accurate (how planes respond to controls)
- Visuals are realistic (looks like flying)
- But you're not actually in the air

Similarly, our mock data:
- Math is accurate (how stock prices behave)
- Data is realistic (looks like Bloomberg)
- But it's not actually connected to real markets

---

## How the Application Uses Mock Data

### Portfolio Generation

When you run the application:

**Step 1: Generate 20-40 Assets**
```python
num_assets = random.randint(20, 40)
portfolio = []
for i in range(num_assets):
    asset = generate_asset_data()
    portfolio.append(asset)
```

**Step 2: Generate Historical Data for Each**
```python
for asset in portfolio:
    historical = generate_historical_prices(
        current_price=asset['current_price'],
        days=252
    )
    asset['historical_data'] = historical
```

**Step 3: Generate News Based on Risk**

After Stage 2 (Core Analysis) determines risk ratings:
```python
for asset in portfolio:
    if asset['risk_rating'] == 'RED':
        sentiment_bias = 'negative'
    elif asset['risk_rating'] == 'YELLOW':
        sentiment_bias = 'mixed'
    else:  # GREEN
        sentiment_bias = 'positive'
    
    news = generate_news_headlines(
        symbol=asset['symbol'],
        sentiment_bias=sentiment_bias
    )
    asset['news'] = news
```

**Result**: A complete, realistic-looking portfolio ready for all 5 analysis stages.

### Data Quality Simulation

The system also simulates **data quality scores** (0-1 scale):

```python
quality_score = random.uniform(0.85, 1.0)
```

Why 0.85-1.0?
- Simulates that Bloomberg data is generally high quality
- Occasional minor gaps (0.85) but usually perfect (1.0)
- The pipeline checks this to ensure data reliability

---

## Could We Use Real Data Instead?

### Yes, With Modifications

The application is designed so you could swap in real data:

**What You'd Need:**
1. Bloomberg API subscription ($2,000+/month)
2. Or Alpha Vantage API (free tier available)
3. Or Yahoo Finance API (free but limited)

**Code Changes:**
Replace `MockBloombergData` calls with real API calls:

```python
# Current (Mock):
mock_data = MockBloombergData()
asset_data = mock_data.generate_asset_data()

# Real (Bloomberg):
bloomberg_api = BloombergAPI(api_key='your_key')
asset_data = bloomberg_api.get_asset_data('AAPL')
```

**Everything Else Stays the Same:**
- All 5 pipeline stages work identically
- Analysis algorithms don't care if data is mock or real
- Reports would show real insights

### Why We Use Mock Data

**Cost**: Bloomberg is expensive
**Accessibility**: Anyone can try the app
**Demonstration**: Shows all features without real accounts
**Testing**: Developers can test without API limits
**Education**: Learn the system risk-free

For a production/commercial version, you'd definitely want real data. But for learning, demonstration, and testing, mock data is perfect.

---

## Key Takeaways

### Understanding Mock Data

1. **It's Fake but Realistic**
   - Uses real mathematical models
   - Matches real market patterns
   - Good enough for demonstration

2. **Systematic Generation**
   - Names: Random combinations of realistic words
   - Prices: Geometric Brownian Motion (real model)
   - Volumes: Correlated with price changes
   - News: Template-based with sentiment bias

3. **Appropriate Ranges**
   - All numbers within realistic bounds
   - Sector-specific characteristics
   - Historical data patterns

4. **Serves Its Purpose**
   - Demonstrates all features
   - Tests all 5 pipeline stages
   - Provides learning environment

### When Using the Application

**Remember:**
- Results are based on simulated data
- Patterns and predictions are for demonstration only
- Don't make real investment decisions based on mock data
- The analysis techniques are real, the data is not

**But Also:**
- The algorithms are real (ML, sentiment analysis, risk scoring)
- The methodology is professional-grade
- The system architecture is production-ready
- With real data, this would be a powerful tool

Think of it as **practicing surgery on a mannequin** - the techniques are real, the skills transfer, but you're not operating on an actual person. Similarly, the analysis techniques here are real and transferable, but the data is practice material.

---

## File Reference

All mock data generation code is in:
- **File**: `utils/mock_data.py`
- **Class**: `MockBloombergData`
- **Lines**: ~280 lines of code

Main methods:
- `generate_asset_data()`: Creates individual stock info
- `generate_historical_prices()`: Creates price history using GBM
- `generate_market_indices()`: Creates S&P 500, NASDAQ, Dow data
- `generate_economic_indicators()`: Creates macro indicators
- `generate_news_headlines()`: Creates sentiment analysis data

You can examine this file to see exactly how everything works!
