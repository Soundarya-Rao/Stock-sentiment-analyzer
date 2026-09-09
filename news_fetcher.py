"""
news_fetcher.py
---------------
Fetches recent news headlines for Indian listed companies via Google News RSS.

Design Choice / Interview Talking Point:
- Why RSS over web scraping?
  1. Reliability & Legality: Indian financial news websites (Moneycontrol, Economic Times,
     Mint) employ anti-bot protections, CAPTCHAs, and frequently change HTML class names/DOM trees.
     Scraping them directly can violate Terms of Service.
  2. Structured Data: Google News RSS provides an open, standard XML feed containing clean
     metadata (title, published date, source publication, canonical link) without requiring
     fragile HTML parsing.
  3. Low Latency: Feeds can be parsed rapidly without browser automation overhead (e.g. Selenium/Playwright).
"""

import datetime
import urllib.parse
from typing import Optional
import feedparser
import pandas as pd
from dateutil import parser as date_parser
import sys

# Ensure UTF-8 output encoding for Windows terminals (e.g. Rupee symbol ₹)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def fetch_headlines(company_name: str, days: int = 30) -> pd.DataFrame:
    """
    Fetches news headlines for a company from Google News RSS (India edition),
    filtered to the last `days` days.

    Parameters:
        company_name (str): Name of the company (e.g. 'Reliance Industries', 'Tata Motors').
        days (int): Lookback window in days (default: 30).

    Returns:
        pd.DataFrame: Columns ['date', 'headline', 'source', 'link'],
                      sorted by date descending.
    """
    if not company_name or not company_name.strip():
        return pd.DataFrame(columns=["date", "headline", "source", "link"])

    query = f"{company_name.strip()} stock"
    encoded_query = urllib.parse.quote(query)
    rss_url = (
        f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:IN"
    )

    feed = feedparser.parse(rss_url)

    records = []
    today = datetime.date.today()
    cutoff_date = today - datetime.timedelta(days=days)

    for entry in feed.entries:
        # 1. Parse published date
        pub_date: Optional[datetime.date] = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                pub_date = datetime.date(*entry.published_parsed[:3])
            except Exception:
                pub_date = None

        if pub_date is None and hasattr(entry, "published"):
            try:
                pub_date = date_parser.parse(entry.published).date()
            except Exception:
                continue

        if pub_date is None:
            continue

        # Filter by cutoff window
        if pub_date < cutoff_date or pub_date > today:
            continue

        # 2. Extract source
        source_name = "Unknown"
        if hasattr(entry, "source") and entry.source:
            if isinstance(entry.source, dict):
                source_name = entry.source.get("title", "Unknown")
            elif hasattr(entry.source, "title"):
                source_name = entry.source.title

        # 3. Clean headline
        raw_title = entry.title if hasattr(entry, "title") else ""
        # Google News titles usually end with ' - Source Name'. Strip it for cleaner sentiment scoring.
        if " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            clean_headline = parts[0].strip()
            if source_name == "Unknown" and len(parts) > 1:
                source_name = parts[1].strip()
        else:
            clean_headline = raw_title.strip()

        # 4. Link
        link = entry.link if hasattr(entry, "link") else ""

        if clean_headline:
            records.append({
                "date": pub_date,
                "headline": clean_headline,
                "source": source_name,
                "link": link
            })

    if not records:
        return pd.DataFrame(columns=["date", "headline", "source", "link"])

    df = pd.DataFrame(records)

    # Deduplicate identical headlines on the same date (syndicated news wires)
    df = df.drop_duplicates(subset=["date", "headline"])

    # Sort descending by date
    df = df.sort_values(by=["date"], ascending=False).reset_index(drop=True)

    return df


if __name__ == "__main__":
    test_company = "Reliance Industries"
    print(f"Fetching headlines for '{test_company}' (last 30 days)...")
    df_headlines = fetch_headlines(test_company, days=30)
    print(f"Found {len(df_headlines)} headlines.")
    print("\nSample Headlines:")
    print(df_headlines.head(10).to_string(index=False))

    # Test another company
    test_company_2 = "Tata Motors"
    print(f"\nFetching headlines for '{test_company_2}' (last 30 days)...")
    df_2 = fetch_headlines(test_company_2, days=30)
    print(f"Found {len(df_2)} headlines.")
    if not df_2.empty:
        print(df_2.head(5)[["date", "source", "headline"]].to_string(index=False))
