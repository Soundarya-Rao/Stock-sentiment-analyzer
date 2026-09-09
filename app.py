"""
app.py
------
Streamlit Web Application: Indian Stock Sentiment Analyser.

Combines:
1. Google News RSS ingestion via feedparser
2. HuggingFace ProsusAI/finbert sentiment scoring (cached via @st.cache_resource)
3. Historical NSE price fetching via yfinance (cached via @st.cache_data)
4. -1 day price shift correlation & directional accuracy analysis
5. Interactive Plotly dual-axis chart and filterable headlines table
"""

import datetime
import sys
from typing import Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from transformers import pipeline

from correlation import correlate_sentiment_and_returns
from news_fetcher import fetch_headlines
from price_fetcher import (
    fetch_prices,
    is_market_closed_for_date,
    get_current_price_summary
)
from sentiment import aggregate_daily_sentiment

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Curated Nifty 50 Large-Cap Companies (Full Index - 50 Constituents)
# -----------------------------------------------------------------------------
NIFTY_COMPANIES: Dict[str, str] = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports and Special Economic Zone": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharat Electronics (BEL)": "BEL.NS",
    "Bharat Petroleum (BPCL)": "BPCL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Infosys": "INFY.NS",
    "ITC Limited": "ITC.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nestlé India": "NESTLEIND.NS",
    "NTPC Limited": "NTPC.NS",
    "Oil & Natural Gas Corporation (ONGC)": "ONGC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "State Bank of India (SBI)": "SBIN.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors Passenger Vehicles": "TMPV.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan Company": "TITAN.NS",
    "Trent Limited": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS"
}


# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_finbert_pipeline():
    """
    Loads HuggingFace's ProsusAI/finbert pipeline once and caches in memory.
    Avoids expensive re-instantiation across user interactions.
    """
    return pipeline("text-classification", model="ProsusAI/finbert", device=-1)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_headlines(company_name: str, days: int) -> pd.DataFrame:
    """Cached wrapper around news_fetcher.fetch_headlines with 1-hour TTL."""
    return fetch_headlines(company_name=company_name, days=days)


@st.cache_data(ttl=300, show_spinner=False)
def get_cached_prices(ticker: str, days: int) -> pd.DataFrame:
    """
    Cached wrapper around price_fetcher.fetch_prices with 5-minute TTL.
    Guarantees that an in-progress intraday session is never returned as a closing price.
    """
    prices = fetch_prices(ticker=ticker, days=days)
    if not prices.empty:
        latest = prices["date"].iloc[-1]
        if not is_market_closed_for_date(latest, ticker=ticker):
            prices = prices.iloc[:-1].copy()
    return prices


@st.cache_data(ttl=60, show_spinner=False)
def get_cached_price_summary(ticker: str) -> dict:
    """Cached wrapper around price_fetcher.get_current_price_summary with 1-minute TTL."""
    return get_current_price_summary(ticker=ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def score_headlines_cached(headlines_list: list) -> list:
    """
    Scores headlines using the cached FinBERT pipeline.
    Takes a tuple/list of headline strings so it can be hashed by st.cache_data.
    """
    nlp = load_finbert_pipeline()
    if not headlines_list:
        return []
    return nlp(headlines_list, batch_size=16, truncation=True)


def generate_context_summary(
    company_name: str,
    news_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    neutral_threshold: float = 0.05
) -> str:
    """
    Generates a plain-English summary sentence based on recent news sentiment
    and historical next-day price movement tendencies in the analyzed window.

    Wording constraints strictly followed:
    - Never uses forecasting or certainty language ('will rise', 'will fall', 'predicts').
    - Strictly frames moves as historical tendencies: 'historically moved up/down X% of the time'.
    - Explicitly includes sample size (Y days) directly in the sentence body.
    - Explicitly caveats: 'this reflects a historical tendency in a small sample, not a prediction.'
    """
    neutral_msg = (
        f"No strong sentiment signal from recent news for {company_name} — "
        f"either no headlines were found, or coverage was neutral. "
        f"Historical data can't offer a directional lean on a day like this."
    )

    if news_df is None or news_df.empty or "sentiment_score" not in news_df.columns:
        return neutral_msg

    df = news_df.dropna(subset=["sentiment_score"]).copy()
    if df.empty:
        return neutral_msg

    df["date_parsed"] = [
        d if isinstance(d, datetime.date) else pd.to_datetime(d).date()
        for d in df["date"]
    ]

    max_date = df["date_parsed"].max()
    recent_news = df[df["date_parsed"] == max_date]
    recent_score = float(recent_news["sentiment_score"].mean())

    # If the latest single day was neutral or had few headlines, also check the latest 2 days
    if abs(recent_score) < neutral_threshold:
        two_day_cutoff = max_date - datetime.timedelta(days=1)
        recent_2d_news = df[df["date_parsed"] >= two_day_cutoff]
        two_day_score = float(recent_2d_news["sentiment_score"].mean())
        if abs(two_day_score) >= neutral_threshold:
            recent_score = two_day_score

    # Determine direction
    if recent_score >= neutral_threshold:
        direction = "positive"
    elif recent_score <= -neutral_threshold:
        direction = "negative"
    else:
        return neutral_msg

    # Compute historical statistics X and Y from merged_df (analyzed window)
    if merged_df is None or merged_df.empty or "next_day_pct_change" not in merged_df.columns:
        valid_df = pd.DataFrame()
    else:
        valid_df = merged_df.dropna(subset=["mean_sentiment", "next_day_pct_change"]).copy()

    if direction == "positive":
        similar_days = valid_df[valid_df["mean_sentiment"] >= neutral_threshold]
        y_count = len(similar_days)
        if y_count > 0:
            up_moves = (similar_days["next_day_pct_change"] > 0).sum()
            x_pct = (up_moves / y_count) * 100.0
            x_str = f"{x_pct:.0f}" if round(x_pct, 1).is_integer() else f"{x_pct:.1f}"
        else:
            x_str = "0"
        return (
            f"Recent news for {company_name} was mostly positive. "
            f"Historically, on days with positive sentiment, this stock moved up next-day {x_str}% "
            f"of the time (based on {y_count} similar days in the analyzed window) — "
            f"this reflects a historical tendency in a small sample, not a prediction."
        )
    else:  # direction == "negative"
        similar_days = valid_df[valid_df["mean_sentiment"] <= -neutral_threshold]
        y_count = len(similar_days)
        if y_count > 0:
            down_moves = (similar_days["next_day_pct_change"] < 0).sum()
            x_pct = (down_moves / y_count) * 100.0
            x_str = f"{x_pct:.0f}" if round(x_pct, 1).is_integer() else f"{x_pct:.1f}"
        else:
            x_str = "0"
        return (
            f"Recent news for {company_name} was mostly negative. "
            f"Historically, on days with negative sentiment, this stock moved down next-day {x_str}% "
            f"of the time (based on {y_count} similar days in the analyzed window) — "
            f"this reflects a historical tendency in a small sample, not a prediction."
        )


# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Stock Sentiment Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for metrics, cards, and badges
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .badge-pos {
        background-color: #0f5132;
        color: #d1e7dd;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neg {
        background-color: #842029;
        color: #f8d7da;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-neu {
        background-color: #495057;
        color: #dee2e6;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .headline-table th {
        background-color: rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation & Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
st.sidebar.title("Configuration")

selected_company = st.sidebar.selectbox(
    "Select Nifty 50 Company",
    options=list(NIFTY_COMPANIES.keys()),
    index=0,
    help="Pick an NSE-listed large-cap stock to fetch recent headlines and historical prices."
)
ticker_symbol = NIFTY_COMPANIES[selected_company]
st.sidebar.caption(f"**NSE Ticker:** `{ticker_symbol}`")

# Quick price preview in sidebar
sidebar_price = get_cached_price_summary(ticker_symbol)
if sidebar_price.get("current_price", 0) > 0:
    st.sidebar.metric(
        label=sidebar_price["status_label"],
        value=f"₹{sidebar_price['current_price']:,.2f}",
        delta=f"{sidebar_price['change']:+,.2f} ({sidebar_price['pct_change']:+.2f}%)"
    )

news_days = st.sidebar.slider(
    "News Lookback (Days)",
    min_value=7,
    max_value=60,
    value=30,
    step=1,
    help="Number of past calendar days of news headlines to retrieve from Google News RSS."
)

price_days = st.sidebar.slider(
    "Price History (Days)",
    min_value=30,
    max_value=120,
    value=60,
    step=5,
    help="Number of days of daily Close prices to fetch via yfinance."
)

analyze_btn = st.sidebar.button("🔍 Fetch & Analyze Sentiment", type="primary", use_container_width=True)

with st.sidebar.expander("ℹ️ Architecture & Methodology", expanded=False):
    st.markdown("""
    **Data Pipeline:**
    1. **Google News RSS:** Fetches structured XML news feeds (bypasses anti-scraping blocks).
    2. **ProsusAI/finbert:** Pre-trained financial transformer scores each headline to $+/-$ signed scores.
    3. **Weekend Rolling:** Saturday/Sunday news rolls forward to Monday's trading session.
    4. **-1 Day Shift:** Aligns day $t$ news sentiment with day $t+1$ price return to test predictive direction.
    5. **Metrics:** Pearson correlation coefficient $r$ and directional accuracy % over active non-neutral days.
    """)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.title("📈 Indian Stock Sentiment Analyser")
st.markdown(
    "Analyze how recent news headlines for **NSE Nifty 50** companies correlate with **next-day stock price movements**, "
    "powered by **Google News RSS**, **HuggingFace `ProsusAI/finbert`**, and **`yfinance`**."
)
st.divider()

# Session state to retain analysis across widget clicks
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.data = None

if analyze_btn or st.session_state.analysis_done:
    if analyze_btn:
        with st.spinner(f"Analyzing {selected_company} ({ticker_symbol})..."):
            # 1. Fetch news headlines
            news_df = get_cached_headlines(company_name=selected_company, days=news_days)

            # 2. Score sentiment with FinBERT
            if not news_df.empty:
                headlines_list = news_df["headline"].astype(str).tolist()
                preds = score_headlines_cached(headlines_list)

                labels = []
                signed_scores = []
                for p in preds:
                    lbl = p["label"].lower()
                    score = float(p["score"])
                    labels.append(lbl)
                    if lbl == "positive":
                        signed_scores.append(round(score, 4))
                    elif lbl == "negative":
                        signed_scores.append(round(-score, 4))
                    else:
                        signed_scores.append(0.0)

                news_df = news_df.copy()
                news_df["sentiment_label"] = labels
                news_df["sentiment_score"] = signed_scores
            else:
                news_df["sentiment_label"] = []
                news_df["sentiment_score"] = []

            # 3. Fetch historical prices
            prices_df = get_cached_prices(ticker=ticker_symbol, days=price_days)

            # 4. Correlate sentiment and returns with -1 day shift
            metrics = correlate_sentiment_and_returns(news_df, prices_df)

            # Store in session state
            st.session_state.analysis_done = True
            st.session_state.data = {
                "company": selected_company,
                "ticker": ticker_symbol,
                "news_df": news_df,
                "prices_df": prices_df,
                "metrics": metrics
            }

    # Retrieve from session state
    data = st.session_state.data
    news_df = data["news_df"]
    prices_df = data["prices_df"]
    metrics = data["metrics"]
    merged_df = metrics["merged_df"]

    # -------------------------------------------------------------------------
    # 1. Prominent Company & Current Price Header
    # -------------------------------------------------------------------------
    price_info = get_cached_price_summary(data["ticker"])

    header_col1, header_col2 = st.columns([3, 2])
    with header_col1:
        st.subheader(f"📊 {data['company']}")
        st.caption(f"**NSE Ticker:** `{data['ticker']}` • Nifty 50 Index Constituent")
    with header_col2:
        if price_info.get("current_price", 0) > 0:
            price_str = f"₹{price_info['current_price']:,.2f}"
            delta_str = f"{price_info['change']:+,.2f} ({price_info['pct_change']:+.2f}%)"
            st.metric(
                label=price_info["status_label"],
                value=price_str,
                delta=delta_str,
                help=f"Previous Day Close: ₹{price_info['previous_close']:,.2f}. As of {price_info['as_of_time_str']}."
            )
        else:
            st.metric(label="Price Status", value="Price Data Unavailable")

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. Correlation & Directional Accuracy Summary Metrics
    # -------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    # Metric 1: Pearson Correlation
    corr_val = metrics["correlation"]
    corr_str = f"{corr_val:+.4f}" if corr_val is not None else "N/A"
    col1.metric(
        label="Pearson Correlation (r)",
        value=corr_str,
        help="Measures linear correlation between daily mean sentiment and NEXT-day price % return."
    )

    # Metric 2: Directional Accuracy %
    acc_val = metrics["directional_accuracy"]
    acc_str = f"{acc_val:.1f}%" if acc_val is not None else "N/A"
    active_days_str = f"{metrics['matching_days_count']} of {metrics['directional_days_count']} active days"
    col2.metric(
        label="Directional Accuracy",
        value=acc_str,
        delta=active_days_str,
        delta_color="off",
        help="Percentage of non-neutral sentiment days where next-day stock price moved in the same direction."
    )

    # Metric 3: Active vs Neutral Days
    col3.metric(
        label="Non-Neutral vs Neutral Days",
        value=f"{metrics['directional_days_count']} / {metrics['neutral_days_count']}",
        help="Number of days with active (non-neutral) sentiment vs. neutral/no-movement days."
    )

    # Metric 4: Total Headlines
    col4.metric(
        label="Headlines Analyzed",
        value=len(news_df),
        help="Total Google News RSS headlines captured in the lookback window."
    )

    # Plain-English Summary & Key Finding Callout
    summary_sentence = generate_context_summary(
        company_name=data["company"],
        news_df=news_df,
        merged_df=merged_df
    )
    st.info(
        f"📌 **Summary:** {summary_sentence}\n\n"
        f"💡 **Key Finding:** {metrics['interpretation']}"
    )

    # -------------------------------------------------------------------------
    # 2. Interactive Plotly Overlay Line Chart
    # -------------------------------------------------------------------------
    st.subheader("📈 Daily Sentiment vs. Next-Day Price Return (%)")

    if not merged_df.empty and "next_day_pct_change" in merged_df.columns:
        # Filter rows that have valid next_day_pct_change for charting
        chart_df = merged_df.dropna(subset=["mean_sentiment", "next_day_pct_change"]).copy()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Trace 1: Next-Day Price Return (%)
        fig.add_trace(
            go.Scatter(
                x=chart_df["date"],
                y=chart_df["next_day_pct_change"],
                name="Next-Day Price Return (%)",
                mode="lines+markers",
                line=dict(color="#00D4B2", width=2.5),
                marker=dict(size=7, symbol="circle"),
                hovertemplate="<b>%{x}</b><br>Next-Day Return: %{y:.2f}%<extra></extra>"
            ),
            secondary_y=False
        )

        # Trace 2: Daily Mean Sentiment Score (Bar / Line)
        # Assign bar colors based on positive/negative
        bar_colors = ["#28a745" if s > 0 else "#dc3545" if s < 0 else "#6c757d" for s in chart_df["mean_sentiment"]]

        fig.add_trace(
            go.Bar(
                x=chart_df["date"],
                y=chart_df["mean_sentiment"],
                name="Mean Sentiment Score",
                marker=dict(color=bar_colors, opacity=0.65),
                hovertemplate="<b>%{x}</b><br>Mean Sentiment: %{y:.3f}<br>Headlines: %{customdata}<extra></extra>",
                customdata=chart_df["headline_count"]
            ),
            secondary_y=True
        )

        # Layout adjustments (optimized for 13-15\" laptop screens)
        fig.update_layout(
            template="plotly_dark",
            height=600,
            margin=dict(l=60, r=60, t=75, b=55),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                font=dict(size=14)
            ),
            hovermode="x unified",
            hoverlabel=dict(font_size=14),
            font=dict(size=14)
        )

        fig.update_xaxes(
            title_text="Date (Shifted to Align With Day t News)",
            title_font=dict(size=16),
            tickfont=dict(size=13),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        )
        fig.update_yaxes(
            title_text="Next-Day Price Return (%)",
            title_font=dict(size=16),
            tickfont=dict(size=13),
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)"
        )
        fig.update_yaxes(
            title_text="FinBERT Sentiment Score [-1 to +1]",
            title_font=dict(size=16),
            tickfont=dict(size=13),
            secondary_y=True,
            showgrid=False,
            range=[-1.15, 1.15]
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient overlapping trading days to generate the dual-axis chart.")

    # -------------------------------------------------------------------------
    # 3. Headlines Table with Sentiment Badges & Links
    # -------------------------------------------------------------------------
    st.subheader(f"📰 Recent Headlines & Sentiment Scoring ({len(news_df)} items)")

    if not news_df.empty:
        # Static tally of headlines across sentiment categories for the current lookback window
        pos_count = int((news_df["sentiment_label"] == "positive").sum()) if "sentiment_label" in news_df.columns else 0
        neg_count = int((news_df["sentiment_label"] == "negative").sum()) if "sentiment_label" in news_df.columns else 0
        neu_count = int((news_df["sentiment_label"] == "neutral").sum()) if "sentiment_label" in news_df.columns else 0

        filter_col1, filter_col2 = st.columns([3, 2], vertical_alignment="bottom")
        with filter_col1:
            label_filter = st.radio(
                "Filter by Sentiment:",
                options=["All", "Positive", "Negative", "Neutral"],
                horizontal=True
            )
        with filter_col2:
            st.markdown(
                f"<div style='padding-bottom: 8px; font-size: 0.95rem;'>"
                f"<span style='color: #22c55e; font-weight: 600;'>{pos_count} positive</span> &nbsp;·&nbsp; "
                f"<span style='color: #ef4444; font-weight: 600;'>{neg_count} negative</span> &nbsp;·&nbsp; "
                f"<span style='color: #94a3b8; font-weight: 600;'>{neu_count} neutral</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        filtered_news = news_df.copy()
        if label_filter != "All":
            filtered_news = filtered_news[filtered_news["sentiment_label"] == label_filter.lower()]

        # Format dataframe for display
        display_df = filtered_news[["date", "sentiment_label", "sentiment_score", "source", "headline", "link"]].copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "sentiment_label": st.column_config.TextColumn("Sentiment"),
                "sentiment_score": st.column_config.NumberColumn("Score", format="%.4f"),
                "source": st.column_config.TextColumn("Publisher"),
                "headline": st.column_config.TextColumn("Headline"),
                "link": st.column_config.LinkColumn("Article Link", display_text="Read Article")
            },
            hide_index=True
        )
    else:
        st.info("No headlines found for the selected company and lookback window.")

    # -------------------------------------------------------------------------
    # 4. Daily Merged Breakdown (Expandable)
    # -------------------------------------------------------------------------
    with st.expander("🔍 View Daily Merged Data & Directional Match Table", expanded=False):
        if not merged_df.empty:
            st.dataframe(
                merged_df,
                use_container_width=True,
                column_config={
                    "date": st.column_config.DateColumn("Trading Date", format="YYYY-MM-DD"),
                    "mean_sentiment": st.column_config.NumberColumn("Mean Sentiment", format="%.4f"),
                    "close": st.column_config.NumberColumn("Close Price (₹)", format="%.2f"),
                    "pct_change": st.column_config.NumberColumn("Same-Day Return (%)", format="%.2f%%"),
                    "next_day_pct_change": st.column_config.NumberColumn("Next-Day Return (%)", format="%.2f%%"),
                    "headline_count": st.column_config.NumberColumn("Headlines"),
                    "direction_match": st.column_config.CheckboxColumn("Direction Matched?")
                },
                hide_index=True
            )
        else:
            st.write("No merged data available.")

else:
    # Initial landing view before user clicks analyze
    quick_price = get_cached_price_summary(ticker_symbol)
    if quick_price.get("current_price", 0) > 0:
        preview_c1, preview_c2 = st.columns([3, 2])
        with preview_c1:
            st.subheader(f"🏢 {selected_company}")
            st.caption(f"**NSE Ticker:** `{ticker_symbol}` • Selected Nifty 50 Constituent")
        with preview_c2:
            st.metric(
                label=quick_price["status_label"],
                value=f"₹{quick_price['current_price']:,.2f}",
                delta=f"{quick_price['change']:+,.2f} ({quick_price['pct_change']:+.2f}%)",
                help=f"Prior Close: ₹{quick_price['previous_close']:,.2f}. As of {quick_price['as_of_time_str']}."
            )
    st.info("👈 Configure the lookback parameters in the sidebar and click **'Fetch & Analyze Sentiment'** to run the FinBERT NLP & return correlation model.")