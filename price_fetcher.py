"""
price_fetcher.py
----------------
Fetches historical stock prices for National Stock Exchange (NSE) listed companies using yfinance.

Design Choices and Interview Talking Points:
- NSE Suffix (.NS): Yahoo Finance tracks NSE stocks with the '.NS' suffix (e.g., RELIANCE.NS, TCS.NS).
  The fetcher automatically guarantees this suffix so callers can pass either 'RELIANCE' or 'RELIANCE.NS'.
- Data Alignment and Type Normalization:
  yfinance returns timestamps with timezone information (e.g., Asia/Kolkata DatetimeIndex).
  To ensure seamless merging with news headlines (from news_fetcher.py), the date column is explicitly
  normalized to standard Python datetime.date objects. This avoids silent empty joins caused by
  mismatched pandas Timestamp vs datetime.date types.
- Lookback Buffer:
  An extra 7-day lookback buffer is fetched prior to the requested cutoff date so that the percentage
  change (pct_change) calculation for the first trading day of the target window has a valid prior
  close price and does not evaluate to NaN.
- Market Close & Intraday Session Guard:
  When yfinance is queried during regular trading hours, its latest row contains the live, unfinalized
  intraday price (LTP). Treating this as an official daily close creates lookahead and calculation distortion,
  causing premature 'Next-Day Return' calculations against incomplete days. The fetcher verifies whether
  the latest session has officially closed (after 15:30 IST for NSE) and drops incomplete in-progress rows.
"""

import datetime
import sys
from typing import Optional
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# NSE regular trading session closing time (15:30 IST / 3:30 PM)
NSE_MARKET_CLOSE_TIME = datetime.time(15, 30)
NSE_TIMEZONE = ZoneInfo("Asia/Kolkata")


def is_market_closed_for_date(
    session_date: datetime.date,
    ticker: str = "",
    tz_info: Optional[datetime.tzinfo] = None,
    close_time: Optional[datetime.time] = None
) -> bool:
    """
    Determines whether the market has officially closed for a given trading session date.

    How 'closed' vs. 'still in progress' is checked:
    1. Resolve Exchange Timezone and Close Time:
       Uses the exchange's timezone (defaults to 'Asia/Kolkata' for NSE/BSE stocks, closing at 15:30 IST).
       Evaluating time in the exchange's timezone ensures consistent behavior regardless
       of the host system's local clock.
    2. Compare Session Date Against Exchange's 'Today':
       - If session_date < today_in_tz:
         The trading day is in the past, so the market has definitively CLOSED (returns True).
       - If session_date == today_in_tz:
         The trading day is today. Compare current time in exchange_tz against close_time (15:30 IST):
         * current_time >= 15:30: Market session is finished and closing price is final (returns True).
         * current_time < 15:30: Market is currently open / mid-trading-day; price is live intraday
                                 and not a genuine closing price (returns False).
       - If session_date > today_in_tz:
         Future date (returns False).

    Parameters:
        session_date (datetime.date): The trading date to check.
        ticker (str): Ticker symbol (used to detect exchange if tz_info not provided).
        tz_info (datetime.tzinfo, optional): Timezone of the exchange.
        close_time (datetime.time, optional): Market close time (defaults to 15:30 IST for NSE).

    Returns:
        bool: True if market has closed for session_date; False if still in progress.
    """
    clean = ticker.strip().upper() if ticker else ""
    if clean.endswith(".BO") or clean.endswith(".NS") or not clean or not ("." in clean):
        default_tz = NSE_TIMEZONE
        default_close = NSE_MARKET_CLOSE_TIME
    else:
        default_tz = ZoneInfo("America/New_York")
        default_close = datetime.time(16, 0)

    exchange_tz = tz_info if tz_info is not None else default_tz
    market_close = close_time if close_time is not None else default_close

    # Current moment in the exchange's timezone
    now_in_exchange = datetime.datetime.now(exchange_tz)
    today_in_exchange = now_in_exchange.date()

    if session_date < today_in_exchange:
        # Prior calendar day has already completed its session
        return True
    elif session_date == today_in_exchange:
        # Today's session is only closed once the current time reaches or passes the closing bell
        return now_in_exchange.time() >= market_close
    else:
        # Future date
        return False


# Ticker alias mappings for legacy symbols:
# TMPV.NS is the renamed original Tata Motors entity (Passenger Vehicles + JLR) which preserves
# the continuous pre-2025 historical price series. TMCV.NS is the separate commercial vehicle
# spinoff (a distinct company listing) and is strictly NOT aliased here.
TICKER_ALIASES = {
    "TATAMOTORS.NS": "TMPV.NS",
    "TATAMOTORS": "TMPV.NS",
    "TATAMOTORS.BO": "TMPV.BO",
}


def get_current_price_summary(ticker: str) -> dict:
    """
    Retrieves the most recent price summary for a ticker, distinguishing between
    a finalized official close and an active/unclosed intraday session.

    Returns:
        dict containing:
            - 'ticker': Standardized ticker string (e.g. 'RELIANCE.NS')
            - 'current_price': Latest price float (LTP or official close)
            - 'previous_close': Prior trading day's closing price float
            - 'change': Absolute price change (current_price - previous_close)
            - 'pct_change': Percentage change over previous close
            - 'as_of_date': Date of the latest price record
            - 'as_of_time_str': Formatted date or time string
            - 'is_market_closed': Boolean indicating if the session is closed
            - 'status_label': Short badge/label string for the UI
    """
    clean_ticker = ticker.strip().upper() if ticker else ""
    if clean_ticker in TICKER_ALIASES:
        clean_ticker = TICKER_ALIASES[clean_ticker]
    elif not clean_ticker.endswith(".NS") and not clean_ticker.endswith(".BO"):
        clean_ticker = f"{clean_ticker}.NS"

    empty_summary = {
        "ticker": clean_ticker,
        "current_price": 0.0,
        "previous_close": 0.0,
        "change": 0.0,
        "pct_change": 0.0,
        "as_of_date": datetime.date.today(),
        "as_of_time_str": "Unavailable",
        "is_market_closed": True,
        "status_label": "Price Data Unavailable",
    }

    if not clean_ticker:
        return empty_summary

    try:
        stock = yf.Ticker(clean_ticker)
        raw_df = stock.history(period="5d", auto_adjust=True)
    except Exception as e:
        print(f"Warning: Error fetching current price summary for {clean_ticker}: {e}", file=sys.stderr)
        return empty_summary

    if raw_df is None or raw_df.empty:
        return empty_summary

    latest_ts = raw_df.index[-1]
    latest_date = latest_ts.date() if hasattr(latest_ts, "date") else latest_ts
    exchange_tz = raw_df.index.tz if getattr(raw_df.index, "tz", None) is not None else NSE_TIMEZONE

    is_closed = is_market_closed_for_date(latest_date, ticker=clean_ticker, tz_info=exchange_tz)
    current_price = float(raw_df["Close"].iloc[-1])
    prev_close = float(raw_df["Close"].iloc[-2]) if len(raw_df) >= 2 else current_price
    change = current_price - prev_close
    pct_change = (change / prev_close) * 100.0 if prev_close else 0.0

    if is_closed:
        as_of_str = latest_date.strftime("%d %b %Y")
        status_label = f"Official Close • {as_of_str}"
    else:
        now_time = datetime.datetime.now(exchange_tz).strftime("%I:%M %p IST")
        as_of_str = f"{latest_date.strftime('%d %b %Y')} ({now_time})"
        status_label = f"Live Market • as of {now_time} (Session in progress)"

    return {
        "ticker": clean_ticker,
        "current_price": round(current_price, 2),
        "previous_close": round(prev_close, 2),
        "change": round(change, 2),
        "pct_change": round(pct_change, 2),
        "as_of_date": latest_date,
        "as_of_time_str": as_of_str,
        "is_market_closed": is_closed,
        "status_label": status_label,
    }


def fetch_prices(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    Fetches daily stock prices for an NSE ticker over the last `days` days.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'RELIANCE', 'RELIANCE.NS', 'TCS').
        days (int): Number of days of historical data to retrieve (default: 60).

    Returns:
        pd.DataFrame: Columns ['date', 'close', 'Close', 'pct_change'],
                      where 'date' contains datetime.date objects, sorted ascending.
                      In-progress (unclosed) trading days are dropped so live intraday prices
                      are never treated as final closing prices.
    """
    if not ticker or not ticker.strip():
        return pd.DataFrame(columns=["date", "close", "pct_change"])

    # Standardize ticker for NSE (with alias resolution)
    clean_ticker = ticker.strip().upper()
    if clean_ticker in TICKER_ALIASES:
        clean_ticker = TICKER_ALIASES[clean_ticker]
    elif not clean_ticker.endswith(".NS") and not clean_ticker.endswith(".BO"):
        clean_ticker = f"{clean_ticker}.NS"

    today = datetime.date.today()
    cutoff_date = today - datetime.timedelta(days=days)
    # Fetch with a 7-day buffer so that the first day in the cutoff has a valid pct_change
    fetch_start = cutoff_date - datetime.timedelta(days=7)
    fetch_end = today + datetime.timedelta(days=1)

    try:
        stock = yf.Ticker(clean_ticker)
        raw_df = stock.history(
            start=fetch_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            auto_adjust=True
        )
    except Exception as e:
        print(f"Warning: Error fetching data for {clean_ticker}: {e}", file=sys.stderr)
        return pd.DataFrame(columns=["date", "close", "pct_change"])

    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["date", "close", "pct_change"])

    # Extract date objects from raw_df index
    raw_dates = [idx.date() if hasattr(idx, "date") else idx for idx in raw_df.index]

    # Check if the most recent row represents a trading day that has NOT closed yet
    if raw_dates:
        latest_date = raw_dates[-1]
        exchange_tz = raw_df.index.tz if getattr(raw_df.index, "tz", None) is not None else None
        if not is_market_closed_for_date(latest_date, ticker=clean_ticker, tz_info=exchange_tz):
            # Market is still in session: drop the incomplete row so live intraday price
            # is not used in same-day pct_change or prior-day next_day_pct_change calculations
            print(
                f"Info: Market session for {latest_date} is still in progress (closes at 15:30 IST). "
                f"Excluding incomplete row from dataset.",
                file=sys.stderr
            )
            raw_df = raw_df.iloc[:-1].copy()
            raw_dates = raw_dates[:-1]

    if raw_df.empty:
        return pd.DataFrame(columns=["date", "close", "pct_change"])

    # Copy and calculate daily percentage change on confirmed closing prices only
    df = raw_df[["Close"]].copy()
    df["close"] = df["Close"].astype(float)
    df["pct_change"] = df["close"].pct_change() * 100.0

    # Assign normalized date to match news_fetcher.py
    df["date"] = raw_dates

    # Filter to requested cutoff window
    df = df[df["date"] >= cutoff_date].copy()

    # Reset index and order columns
    df = df.reset_index(drop=True)
    df = df[["date", "close", "pct_change"]]
    # Also provide 'Close' alias for convenience
    df["Close"] = df["close"]

    # Sort ascending by date for chronological time-series analysis
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

    return df


if __name__ == "__main__":
    test_tickers = ["RELIANCE.NS", "TCS", "INFY.NS"]
    print("Testing price_fetcher.py across sample NSE stocks (last 60 days)...\n")

    for sym in test_tickers:
        print(f"Fetching prices for {sym}...")
        prices_df = fetch_prices(sym, days=60)
        print(f"Retrieved {len(prices_df)} trading days.")
        if not prices_df.empty:
            print(prices_df.tail(5).to_string(index=False))
            print(f"Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
            print(f"Date column type: {type(prices_df['date'].iloc[0])}")
            print(f"Any NaN in pct_change: {prices_df['pct_change'].isna().any()}")
        print("-" * 60)