from news_fetcher import fetch_headlines
from price_fetcher import fetch_prices
from sentiment import score_sentiment
from correlation import correlate_sentiment_and_returns, save_to_bigquery

nifty_companies = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Larsen & Toubro": "LT.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS",
    "Nestle India": "NESTLEIND.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tech Mahindra": "TECHM.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Coal India": "COALINDIA.NS",
    "Grasim Industries": "GRASIM.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Dr Reddy's Laboratories": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Trent": "TRENT.NS",
    "LTIMindtree": "LTIM.NS",
    "Bharat Electronics": "BEL.NS",
    "Cholamandalam Investment": "CHOLAFIN.NS",
}

for company, ticker in nifty_companies.items():
    print(f"Processing {company}...")
    news_df = fetch_headlines(company, days=30)
    scored_df = score_sentiment(news_df)
    price_df = fetch_prices(ticker, days=60)
    metrics = correlate_sentiment_and_returns(scored_df, price_df)

    if metrics["merged_df"].empty:
        print(f"  Skipped {company} — no overlapping data.")
        continue

    df = metrics["merged_df"].copy()
    df["company"] = company
    df["ticker"] = ticker
    save_to_bigquery(df)

print("Done.")