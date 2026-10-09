"""
correlation.py
--------------
Analyzes the statistical and directional relationship between news sentiment and next-day NSE stock returns.

Design Choices & Interview Talking Points:
- Why the -1 day shift?
  1. Direction of Causality: Markets react to news after it is published. Aligning sentiment on day t
     with the price return of day t creates simultaneous lookahead bias (e.g., afternoon price drops causing
     negative evening news, rather than news leading price).
  2. Predictive Signal Testing: By shifting price percentage change by -1 day (or aligning day t sentiment
     with day t+1 close-to-close return), we test whether accumulated news sentiment has a leading or
     predictive relationship with next-day price movement.
- Rolling Weekend & Holiday News Forward:
  News published over weekends (Saturday/Sunday) or NSE market holidays is absorbed by the market on the
  next available trading session (usually Monday morning). Rolling non-trading day news forward to the next
  trading day ensures these high-signal headlines are not discarded by an inner join.
- Directional Accuracy vs. Raw Correlation:
  Pearson correlation measures linear association across all days (including subtle fractional moves).
  Directional accuracy measures how often the sign of non-neutral sentiment correctly predicted the direction
  (up/down) of the next-day move. To maintain statistical integrity, we always report the exact count of
  active days (e.g. 5 of 7 days) because a high percentage on 2 days is anecdotal, not statistically robust.
"""

import datetime
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

def save_to_bigquery(df, table_id="stock_data.sentiment_returns"):
    from google.cloud import bigquery
    client = bigquery.Client()
    job = client.load_table_from_dataframe(df, table_id)
    job.result()  # waits for it to finish
    print(f"Saved {len(df)} rows to {table_id}")


# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def roll_to_next_trading_day(news_date: datetime.date, trading_dates: List[datetime.date]) -> datetime.date:
    """
    Finds the earliest trading day >= news_date.
    If news_date is after the last available trading day (e.g. today while the session is still in progress),
    returns news_date so today's news is never rolled backwards to yesterday.
    """
    for td in trading_dates:
        if td >= news_date:
            return td
    return news_date


def correlate_sentiment_and_returns(
    sentiment_df: pd.DataFrame,
    price_df: pd.DataFrame,
    neutral_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Merges daily sentiment with price data, shifting price % change by -1 day (next-day return),
    and computes the Pearson correlation coefficient and directional accuracy.

    Parameters:
        sentiment_df (pd.DataFrame): Output from score_sentiment() or aggregate_daily_sentiment().
        price_df (pd.DataFrame): Output from fetch_prices(), sorted ascending by date.
        neutral_threshold (float): Minimum absolute sentiment score to be classified as non-neutral.
                                   Default is 0.05.

    Returns:
        Dict[str, Any] containing:
            - 'correlation': Pearson correlation float or None
            - 'directional_accuracy': Percentage float [0-100] or None
            - 'directional_days_count': Number of non-neutral sentiment days evaluated
            - 'matching_days_count': Number of days where sentiment direction matched next-day return
            - 'neutral_days_count': Number of days classified as neutral
            - 'total_trading_days': Total overlapping days with next-day price data
            - 'merged_df': Merged DataFrame with columns ['date', 'mean_sentiment', 'close',
                           'pct_change', 'next_day_pct_change', 'headline_count', 'direction_match']
            - 'interpretation': Plain-English summary string for display
    """
    empty_result = {
        "correlation": None,
        "directional_accuracy": None,
        "directional_days_count": 0,
        "matching_days_count": 0,
        "neutral_days_count": 0,
        "total_trading_days": 0,
        "merged_df": pd.DataFrame(),
        "interpretation": "Insufficient overlapping data between news sentiment and price returns."
    }

    if sentiment_df is None or sentiment_df.empty or price_df is None or price_df.empty:
        return empty_result

    # 1. Clean and sort price data
    prices = price_df.copy().sort_values(by="date", ascending=True).reset_index(drop=True)
    if "pct_change" not in prices.columns:
        if "close" in prices.columns:
            prices["pct_change"] = prices["close"].pct_change() * 100.0
        elif "Close" in prices.columns:
            prices["pct_change"] = prices["Close"].pct_change() * 100.0
        else:
            return empty_result

    # Ensure date column consists of standard datetime.date objects
    prices["date"] = [d if isinstance(d, datetime.date) else pd.to_datetime(d).date() for d in prices["date"]]

    # Defense-in-depth: Exclude any trailing in-progress session if market has not closed
    if not prices.empty:
        from price_fetcher import is_market_closed_for_date
        latest_date = prices["date"].iloc[-1]
        if not is_market_closed_for_date(latest_date):
            prices = prices.iloc[:-1].copy()

    trading_dates = sorted(prices["date"].unique().tolist())

    # 2. Prepare and aggregate sentiment data
    sent = sentiment_df.copy()
    sent["date"] = [d if isinstance(d, datetime.date) else pd.to_datetime(d).date() for d in sent["date"]]

    # Roll weekend/holiday news forward to the next available trading day
    sent["effective_date"] = [roll_to_next_trading_day(d, trading_dates) for d in sent["date"]]

    # Aggregate by effective trading date
    if "mean_sentiment" in sent.columns:
        daily_sent = (
            sent.groupby("effective_date")
            .agg(
                mean_sentiment=("mean_sentiment", "mean"),
                headline_count=("headline_count", "sum") if "headline_count" in sent.columns else ("mean_sentiment", "count")
            )
            .reset_index()
            .rename(columns={"effective_date": "date"})
        )
    elif "sentiment_score" in sent.columns:
        daily_sent = (
            sent.groupby("effective_date")
            .agg(
                mean_sentiment=("sentiment_score", "mean"),
                headline_count=("sentiment_score", "count")
            )
            .reset_index()
            .rename(columns={"effective_date": "date"})
        )
    else:
        return empty_result

    daily_sent["mean_sentiment"] = daily_sent["mean_sentiment"].round(4)

    # 3. Shift price % change by -1 day (align day t sentiment with day t+1 price return)
    prices["next_day_pct_change"] = prices["pct_change"].shift(-1)

    # 4. Merge on date
    cols_to_keep = ["date", "close", "pct_change", "next_day_pct_change"]
    available_cols = [c for c in cols_to_keep if c in prices.columns]
    merged = pd.merge(daily_sent, prices[available_cols], on="date", how="inner")

    if merged.empty:
        return empty_result

    # Filter rows with valid next_day_pct_change (the latest trading day does not have next-day return yet)
    valid_df = merged.dropna(subset=["mean_sentiment", "next_day_pct_change"]).copy()

    total_trading_days = len(valid_df)
    if total_trading_days < 2:
        empty_result["total_trading_days"] = total_trading_days
        empty_result["merged_df"] = merged
        return empty_result

    # 5. Pearson correlation coefficient
    corr_val = valid_df["mean_sentiment"].corr(valid_df["next_day_pct_change"])
    correlation = round(float(corr_val), 4) if not np.isnan(corr_val) else None

    # 6. Directional accuracy on non-neutral days
    non_neutral = valid_df[valid_df["mean_sentiment"].abs() >= neutral_threshold].copy()
    neutral_days_count = total_trading_days - len(non_neutral)
    directional_days_count = len(non_neutral)

    if directional_days_count > 0:
        # Same direction if: (sentiment > 0 and return > 0) OR (sentiment < 0 and return < 0)
        # Note: product > 0 encapsulates both conditions
        non_neutral["direction_match"] = (
            (non_neutral["mean_sentiment"] * non_neutral["next_day_pct_change"]) > 0
        )
        matching_days_count = int(non_neutral["direction_match"].sum())
        directional_accuracy = round((matching_days_count / directional_days_count) * 100.0, 1)

        # Merge direction_match flag back into merged_df for charting / inspection
        match_map = dict(zip(non_neutral["date"], non_neutral["direction_match"]))
        merged["direction_match"] = merged["date"].map(match_map)
    else:
        matching_days_count = 0
        directional_accuracy = None
        merged["direction_match"] = None

    # 7. Plain-English interpretation
    if directional_accuracy is not None:
        direction_text = (
            f"Sentiment and next-day price moved in the same direction on {directional_accuracy}% of days "
            f"({matching_days_count} of {directional_days_count} non-neutral days) over {total_trading_days} analyzed trading days."
        )
    else:
        direction_text = f"All {total_trading_days} analyzed days had neutral sentiment within the threshold."

    if correlation is not None:
        corr_strength = (
            "moderate positive" if correlation >= 0.3 else
            "mild positive" if correlation > 0.05 else
            "near zero / neutral" if abs(correlation) <= 0.05 else
            "mild negative" if correlation > -0.3 else
            "moderate negative"
        )
        interpretation = f"{direction_text} Pearson correlation is {correlation:+.4f} ({corr_strength} correlation)."
    else:
        interpretation = direction_text

    return {
        "correlation": correlation,
        "directional_accuracy": directional_accuracy,
        "directional_days_count": directional_days_count,
        "matching_days_count": matching_days_count,
        "neutral_days_count": neutral_days_count,
        "total_trading_days": total_trading_days,
        "merged_df": merged,
        "interpretation": interpretation
    }


if __name__ == "__main__":
    from news_fetcher import fetch_headlines
    from price_fetcher import fetch_prices
    from sentiment import score_sentiment

    test_company = "Reliance Industries"
    test_ticker = "RELIANCE.NS"

    print(f"=== Testing correlation.py on {test_company} ({test_ticker}) ===\n")
    print("1. Fetching news headlines (last 30 days)...")
    news_df = fetch_headlines(test_company, days=30)
    print(f"   Fetched {len(news_df)} headlines.")

    print("2. Scoring headlines with FinBERT...")
    scored_df = score_sentiment(news_df)

    print("3. Fetching stock prices (last 60 days)...")
    price_df = fetch_prices(test_ticker, days=60)
    print(f"   Fetched {len(price_df)} trading days.")

    print("4. Calculating correlation and directional accuracy (with -1 day shift)...")
    metrics = correlate_sentiment_and_returns(scored_df, price_df)

    print("\n" + "=" * 60)
    print(f"Pearson Correlation:     {metrics['correlation']}")
    print(f"Directional Accuracy:    {metrics['directional_accuracy']}%")
    print(f"Active Non-Neutral Days: {metrics['matching_days_count']} of {metrics['directional_days_count']} days")
    print(f"Neutral Days Count:      {metrics['neutral_days_count']}")
    print(f"Total Overlapping Days:  {metrics['total_trading_days']}")
    print("=" * 60)
    print("\nInterpretation:")
    print(metrics["interpretation"])

    print("\nMerged Data Sample:")
    merged_preview = metrics["merged_df"][
        ["date", "mean_sentiment", "pct_change", "next_day_pct_change", "headline_count", "direction_match"]
    ]
    print(merged_preview.head(10).to_string(index=False))

