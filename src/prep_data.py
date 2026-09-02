import pandas as pd
import numpy as np
import os
from paths import ROOT, DATA_DIR
SRC_DIR = os.environ.get(
    "NSE_DATA_SRC_DIR",
    str(ROOT.parent / "nse_repo" / "NSE Minute Data" / "NSE_Stocks_Data"),
)
OUT_DIR = DATA_DIR
os.makedirs(OUT_DIR, exist_ok=True)
TICKERS = {
    "RELIANCE": "RELIANCE__EQ__NSE__NSE__MINUTE.csv",
    "HDFCBANK": "HDFCBANK__EQ__NSE__NSE__MINUTE.csv",
    "ICICIBANK": "ICICIBANK__EQ__NSE__NSE__MINUTE.csv",
    "INFY": "INFY__EQ__NSE__NSE__MINUTE.csv",
    "SBIN": "SBIN__EQ__NSE__NSE__MINUTE.csv",
}
BAR = "5min"
SESSION_START = "09:15"
SESSION_END = "15:29"   # last 5-min bar covers 15:25-15:30
EXPECTED_BARS = 75
all_days_summary = []
for tkr, fname in TICKERS.items():
    path = os.path.join(SRC_DIR, fname)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Resample 1-min -> 5-min bars
    agg = df.resample(BAR, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    agg = agg.between_time(SESSION_START, SESSION_END)
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    agg["volume"] = agg["volume"].fillna(0)

    agg["date"] = agg.index.date
    counts = agg.groupby("date").size()
    good_days = counts[counts == EXPECTED_BARS].index
    agg = agg[agg["date"].isin(good_days)].copy()

    print(f"{tkr}: {len(good_days)} clean full-session days out of {counts.shape[0]} total days, "
          f"range {agg.index.min().date()} -> {agg.index.max().date()}")

    agg.to_parquet(os.path.join(OUT_DIR, f"{tkr}_5min.parquet"))

    daily = agg.groupby("date").agg(day_volume=("volume", "sum"), day_close=("close", "last"))
    daily["ticker"] = tkr
    all_days_summary.append(daily.reset_index())

summary = pd.concat(all_days_summary, ignore_index=True)
summary.to_parquet(os.path.join(OUT_DIR, "daily_summary.parquet"))
print("\nTotal clean stock-days across universe:", len(summary))
