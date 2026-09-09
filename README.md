# 📈 Indian Stock Sentiment Analyser (NSE Nifty 50)

A financial NLP and time-series analytics application that investigates whether accumulated news headlines for **NSE Nifty 50** companies correlate with and predict **next-day stock price movements**.

The application combines structured **Google News RSS** ingestion, sentiment classification via HuggingFace's **`ProsusAI/finbert`** transformer model, historical daily price action from **`yfinance`**, and statistical correlation analysis with a $-1$ day price shift to evaluate directional accuracy.

---

## 📸 Application Preview

```
+----------------------------------------------------------------------------------------------------+
|  📈 Indian Stock Sentiment Analyser                                                                 |
|  Analyze how news headlines for NSE Nifty 50 companies correlate with next-day price returns.       |
|----------------------------------------------------------------------------------------------------|
|  📊 Reliance Industries (RELIANCE.NS)                ₹1,279.00  -15.90 (-1.23%) Today               |
|  NSE Ticker: RELIANCE.NS • Nifty 50 Index            🔒 Official Close • 09 Sep 2026               |
|----------------------------------------------------------------------------------------------------|
|  [ Pearson r: +0.2841 ]  [ Directional Accuracy: 66.7% ]  [ Non-Neutral: 6 / 12 ]  [ Headlines: 38 ]|
|----------------------------------------------------------------------------------------------------|
|  📈 Plotly Dual-Axis: Mean Sentiment Score (Bars) vs. Next-Day Price Return % (Line)               |
|  📰 Filterable Headlines Table with FinBERT Sentiment Badges and Direct Article Links              |
+----------------------------------------------------------------------------------------------------+
```
*(Place your application screenshot at `docs/screenshot.png`)*

---

## 🎯 Interview Talking Points & Design Choices

This project was built with production-grade data engineering and statistical rigor rather than as a superficial wrapper. Below are the key architectural decisions and engineering rationales:

### 1. Google News RSS vs. Direct Web Scraping
* **Anti-Scraping Resistance:** Top Indian financial portals (*Moneycontrol*, *The Economic Times*, *Mint*, *Business Standard*) employ aggressive Cloudflare protections, dynamic JavaScript hydration, CAPTCHAs, and frequently changing DOM structures. Direct scraping is brittle and violates Terms of Service.
* **Structured XML Protocol:** Google News provides a standardized, reliable XML RSS feed (`news.google.com/rss/search`) parameterized for the Indian financial market (`hl=en-IN&gl=IN&ceid=IN:IN`). It guarantees clean metadata (title, publication timestamp, source publisher, and canonical URL) with low latency and zero headless browser overhead.

### 2. FinBERT vs. Generic Sentiment Models (VADER / Standard BERT)
* **Domain-Specific Vocabulary:** Standard NLP models fail on financial terminology. Words like `"risk"`, `"liability"`, `"hedging"`, `"drag"`, or `"plunge"` are interpreted as negative in colloquial English, whereas in financial statements and corporate news they are standard descriptive terms.
* **Trained on Financial Corpora:** `ProsusAI/finbert` was fine-tuned on the Financial PhraseBank dataset, enabling calibrated classification into positive, negative, and neutral categories.
* **Continuous Signed Metric:** We map FinBERT's discrete class probabilities into a continuous signed score:
  $$\text{Score} = \begin{cases} +\text{confidence}, & \text{if label is Positive} \\ -\text{confidence}, & \text{if label is Negative} \\ 0.0, & \text{if label is Neutral} \end{cases}$$

### 3. The -1 Day Return Shift (Eliminating Lookahead Bias)
* **Direction of Causality:** Markets react to news *after* it is published. Aligning day $t$ sentiment with day $t$ return creates simultaneous lookahead bias — an afternoon intraday price drop often triggers negative evening headlines, which falsely inflates same-day correlation.
* **Predictive Signal Testing:** Shifting price returns by $-1$ day aligns day $t$ accumulated sentiment with day $t+1$ close-to-close return:
  $$\text{Next-Day Return}_t = \frac{\text{Close}_{t+1} - \text{Close}_t}{\text{Close}_t} \times 100\%$$
  This tests whether sentiment has an actual **leading, predictive relationship** with market movement.

### 4. Directional Accuracy & Active Non-Neutral Day Counts
* **Pearson Correlation ($r$):** Measures linear co-movement across all days, including subtle fractional price shifts.
* **Directional Accuracy (%):** Measures how often the sign of sentiment matched the sign of the next day's price move:
  $$\text{Directional Match} = (\text{Mean Sentiment}_t \times \text{Next-Day Return}_t) > 0$$
* **Statistical Honesty:** A directional accuracy of 80% evaluated over only 3 active days is statistical noise. We explicitly filter out neutral days ($|\text{sentiment}| < 0.05$) and report the exact count of active days (e.g. *"4 of 6 active days"*) to prevent deceptive percentage inflation.

### 5. Intraday vs. Official Closing Prices
* **The Problem:** During market hours, `yfinance` returns the live Last Traded Price (LTP) in the `Close` column. Treating an unfinalized session as a completed close causes premature next-day return calculations against an incomplete day.
* **The Solution:** The `is_market_closed_for_date()` guard evaluates the exchange timezone (`Asia/Kolkata`):
  * Prior days ($\text{date} < \text{today}$) are marked **Closed**.
  * Today ($\text{date} == \text{today}$) is **In Progress** until 15:30 IST.
  * Incomplete intraday rows are excluded from return and correlation calculations. Yesterday's row displays `NaN` for Next-Day Return until today's session officially settles.
  * The top of the dashboard displays a prominent live price card distinguishing `Live Market (In Progress)` from `Official Close`.

---

## 🏗️ Architecture Overview

The pipeline follows a modular, decoupled architecture where each layer can be tested and executed independently:

```mermaid
flowchart TD
    subgraph Data Ingestion
        A[Google News RSS<br/>news.google.com] -->|XML Feed| B[news_fetcher.py]
        C[Yahoo Finance<br/>NSE: .NS] -->|Historical / Live| D[price_fetcher.py]
    end

    subgraph Transformation & NLP
        B -->|Headlines DataFrame| E[sentiment.py<br/>ProsusAI/finbert]
        E -->|Signed Sentiment Scores| F[Daily Aggregation]
        D -->|Market Close Validation| G[Filtered Closing Prices]
    end

    subgraph Analytics Engine
        F --> H[correlation.py]
        G --> H
        H -->|Weekend News Roll-Forward| I[Effective Trading Day Alignment]
        I -->|-1 Day Price Shift| J[Pearson r & Directional Accuracy]
    end

    subgraph User Interface
        J --> K[app.py<br/>Streamlit Dashboard]
        K --> L[Interactive Plotly Dual-Axis Chart]
        K --> M[Filterable Headlines & Metrics Table]
        K --> N[Real-Time Price & Status Header]
    end
```

### Module Breakdown

| Module | Primary Responsibility | Key Functions / Classes |
|:---|:---|:---|
| [`news_fetcher.py`](news_fetcher.py) | Ingests and parses XML feeds from Google News RSS | `fetch_headlines(company_name, days)` |
| [`price_fetcher.py`](price_fetcher.py) | Fetches NSE prices, validates market hours, drops incomplete intraday data | `fetch_prices()`, `is_market_closed_for_date()`, `get_current_price_summary()` |
| [`sentiment.py`](sentiment.py) | Runs HuggingFace FinBERT pipeline and produces signed continuous scores | `score_sentiment()`, `aggregate_daily_sentiment()` |
| [`correlation.py`](correlation.py) | Rolls weekend news, shifts returns by -1 day, computes Pearson $r$ and directional match | `correlate_sentiment_and_returns()`, `roll_to_next_trading_day()` |
| [`app.py`](app.py) | Streamlit web application with caching, dual-axis Plotly charts, and all 50 Nifty constituents | `st.set_page_config`, `@st.cache_data`, `@st.cache_resource` |

---

## 🚀 How to Run Locally

### 1. Prerequisites
* **Python 3.10 - 3.13** installed on your system.
* Active Internet connection (to query Google News RSS, Yahoo Finance, and download the FinBERT weights on initial run).

### 2. Clone and Setup Environment
```bash
# Navigate to project root
cd "d:/Stock Analysis"

# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
Install exact pinned dependencies:
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Running Individual Component Tests
Each module includes a standalone `__main__` test block for debugging:
```bash
# Test price fetching and market close logic
python price_fetcher.py

# Test Google News headline fetching
python news_fetcher.py

# Test sentiment scoring with FinBERT
python sentiment.py

# Test correlation and return shifts
python correlation.py
```

---

## 📂 Project Structure

```
Stock Analysis/
├── app.py               # Streamlit web application & Plotly visualizations
├── correlation.py       # Alignment, -1 day shift, Pearson r, and directional accuracy
├── news_fetcher.py      # Google News RSS ingestion and parsing
├── price_fetcher.py     # yfinance historical prices, market-close validation, price summary
├── sentiment.py         # HuggingFace ProsusAI/finbert sentiment scoring pipeline
├── requirements.txt     # Exact pinned dependencies from virtual environment
└── README.md            # Architecture documentation and interview reference
```

---

## ⚠️ Limitations & Analytical Caveats

1. **Sample Size Limitations:**
   Within a standard 30 to 60-day window, individual companies rarely have more than 5 to 15 days with active non-neutral news. Drawing definitive quantitative conclusions from small samples risks overfitting to random market noise.
2. **Single-Headline Days:**
   Days with a single isolated headline provide a weak, noisy signal compared to high-volume news days (e.g. earnings announcements, board meetings, or regulatory audits) where dozens of articles corroborate sentiment.
3. **Correlation Does Not Imply Causation:**
   Stock prices are driven by broader macroeconomic trends, RBI monetary policy decisions, FII/DII institutional liquidity flows, crude oil price fluctuations, and currency movements that news headlines alone cannot capture.
4. **Domain and Language Specificity:**
   `ProsusAI/finbert` was trained primarily on global financial English (SEC filings, international financial journals). It may occasionally miss Indian corporate idioms, regional business vernacular, or domestic regulatory shorthand (e.g., *NCLT*, *SEBI order*, *promoter pledge*).
5. **Exploratory Baseline, Not an Execution Strategy:**
   This tool is intended as an exploratory analytical baseline for feature engineering and hypothesis testing. It does not account for transaction slippage, bid-ask spreads, STT (Securities Transaction Tax), brokerage costs, or liquidity constraints.
