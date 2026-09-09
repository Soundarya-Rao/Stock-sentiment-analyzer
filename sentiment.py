"""
sentiment.py
------------
Performs financial NLP sentiment scoring on news headlines using HuggingFace's ProsusAI/finbert.

Design Choices & Interview Talking Points:
- Why ProsusAI/finbert instead of generic sentiment models (e.g., VADER, standard BERT)?
  1. Domain Sensitivity: General NLP models misinterpret financial vocabulary. For example, words
     like 'risk', 'liability', 'hedging', 'plunge', or 'drag' often register as heavily negative in
     colloquial text, whereas in financial filings or news they are standard descriptive terminology.
  2. Domain Pretraining: FinBERT was specifically fine-tuned on the Financial PhraseBank dataset,
     enabling it to accurately classify financial statements into positive, negative, and neutral sentiment
     with calibrated probability scores.
  3. Signed Sentiment Metric: FinBERT outputs a discrete label (positive/negative/neutral) and a confidence
     probability [0, 1]. We map this into a continuous signed score:
       * Positive: +confidence
       * Negative: -confidence
       * Neutral:   0.0
     This allows daily mean sentiment aggregation and direct Pearson correlation with price returns.
- Performance & Caching:
  The transformer pipeline is cached via @functools.lru_cache so model weights (~440MB) are loaded into
  RAM exactly once during application runtime, preventing expensive multi-second reloading on every query.
"""

import datetime
import functools
import sys
from typing import List, Optional
import pandas as pd
from transformers import pipeline

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@functools.lru_cache(maxsize=1)
def get_sentiment_pipeline():
    """
    Loads and caches the HuggingFace text-classification pipeline for ProsusAI/finbert.
    Cached with lru_cache(maxsize=1) to ensure single model instance in memory.
    """
    return pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        device=-1  # CPU inference
    )


def score_sentiment(headlines_df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
    """
    Scores sentiment of headlines using FinBERT.

    Parameters:
        headlines_df (pd.DataFrame): DataFrame containing at least a 'headline' column.
        batch_size (int): Batch size for transformer inference (default: 16).

    Returns:
        pd.DataFrame: Original DataFrame with added columns:
                      - 'sentiment_label': 'positive', 'negative', or 'neutral'
                      - 'sentiment_score': Signed float:
                                           +score for positive,
                                           -score for negative,
                                            0.0 for neutral.
    """
    if headlines_df is None or headlines_df.empty or "headline" not in headlines_df.columns:
        res_df = headlines_df.copy() if headlines_df is not None else pd.DataFrame()
        res_df["sentiment_label"] = pd.Series(dtype=str)
        res_df["sentiment_score"] = pd.Series(dtype=float)
        return res_df

    df = headlines_df.copy()
    headlines = [str(h).strip() for h in df["headline"]]

    # Load cached pipeline
    nlp = get_sentiment_pipeline()

    # Run inference with truncation enabled
    predictions = nlp(headlines, batch_size=batch_size, truncation=True)

    labels = []
    signed_scores = []

    for pred in predictions:
        label = pred["label"].lower()  # 'positive', 'negative', 'neutral'
        score = float(pred["score"])

        labels.append(label)
        if label == "positive":
            signed_scores.append(round(score, 4))
        elif label == "negative":
            signed_scores.append(round(-score, 4))
        else:  # neutral
            signed_scores.append(0.0)

    df["sentiment_label"] = labels
    df["sentiment_score"] = signed_scores

    return df


def aggregate_daily_sentiment(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates scored headlines by date to compute mean daily sentiment score and headline volume.

    Parameters:
        scored_df (pd.DataFrame): DataFrame output from score_sentiment(),
                                  containing 'date' and 'sentiment_score'.

    Returns:
        pd.DataFrame: Columns ['date', 'mean_sentiment', 'headline_count'],
                      sorted ascending by date.
    """
    if scored_df is None or scored_df.empty or "date" not in scored_df.columns:
        return pd.DataFrame(columns=["date", "mean_sentiment", "headline_count"])

    # Group by date and calculate mean sentiment and count of headlines
    daily = (
        scored_df.groupby("date")
        .agg(
            mean_sentiment=("sentiment_score", "mean"),
            headline_count=("sentiment_score", "count")
        )
        .reset_index()
    )

    daily["mean_sentiment"] = daily["mean_sentiment"].round(4)
    # Sort ascending by date for chronological analysis
    daily = daily.sort_values(by="date", ascending=True).reset_index(drop=True)

    return daily


if __name__ == "__main__":
    from news_fetcher import fetch_headlines

    test_company = "Reliance Industries"
    print(f"Fetching headlines for '{test_company}' (last 30 days)...")
    raw_news = fetch_headlines(test_company, days=30)
    print(f"Fetched {len(raw_news)} headlines.")

    print("\nRunning FinBERT sentiment scoring...")
    scored = score_sentiment(raw_news)
    print("Scored DataFrame preview:")
    print(scored[["date", "sentiment_label", "sentiment_score", "headline"]].head(8).to_string(index=False))

    print("\nSentiment Label Distribution:")
    print(scored["sentiment_label"].value_counts().to_string())

    print("\nAggregating to mean daily sentiment...")
    daily_sentiment = aggregate_daily_sentiment(scored)
    print("Daily Aggregated Sentiment (first 10 days):")
    print(daily_sentiment.head(10).to_string(index=False))