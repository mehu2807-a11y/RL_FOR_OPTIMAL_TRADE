"""
Alternative to prep_data.py: pull data directly from yfinance instead of the
GitHub-hosted historical dump.

This sandbox's network egress blocks query1/query2.finance.yahoo.com, so the
main pipeline (prep_data.py) instead sources real NSE minute bars from a
public GitHub dataset (see README's "Data" section for the citation and the
tradeoffs). If you're running this on a machine with normal internet access,
this script is a drop-in replacement that will work directly with yfinance's
5-minute bars, per the original request -- the tradeoff is that yfinance
only serves ~60 days of 5-minute history, so you'll have far fewer days to
split into train/test than the 2017-2021 GitHub dataset gives you. Consider
running this weekly and accumulating a local archive if you want more
history over time.

Usage:
    pip install yfinance
    python3 fetch_yfinance.py
"""
import pandas as pd
import numpy as np
import os
import yfinance as yf

from paths import DATA_DIR

OUT_DIR = DATA_DIR
os.makedirs(OUT_DIR, exist_ok=True)

TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "TCS.NS"]
BARS_PER_DAY = 75

for tkr in TICKERS:
    df = yf.download(tkr, period="60d", interval="5m", progress=False)
    if df.empty:
        print(f"WARNING: no data returned for {tkr}, skipping")
        continue
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df["date"] = df.index.date

    counts = df.groupby("date").size()
    good_days = counts[counts == BARS_PER_DAY].index
    df = df[df["date"].isin(good_days)].copy()

    stem = tkr.replace(".NS", "")
    df.to_parquet(os.path.join(OUT_DIR, f"{stem}_5min.parquet"))
    print(f"{stem}: {len(good_days)} clean days out of {counts.shape[0]}, "
          f"range {df.index.min().date() if len(df) else 'n/a'} -> {df.index.max().date() if len(df) else 'n/a'}")

print("\nNote: with ~60 days of history you likely won't have enough days for a")
print("meaningful time-based train/test split per stock. Options: pool across more")
print("tickers, lower BARS_PER_DAY requirements, or accumulate data over several weeks.")
