import pandas as pd
import numpy as np
import os
from paths import DATA_DIR
TICKERS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "SBIN"]
BARS_PER_DAY = 75
TRAIN_TEST_CUTOFF = "2020-01-01"   # dates >= cutoff are test (out-of-sample)
ADV_WINDOW = 20
VOL_WINDOW = 10
class MarketData:
    def __init__(self):
        self.bars = {}          # ticker -> DataFrame indexed by timestamp (5-min bars)
        self.daily = {}         # ticker -> DataFrame indexed by date with day_volume, day_close, adv20, vol10
        self.profile = {}       # ticker -> np.array(75,) historical avg fraction of day volume per bin
        self.day_list = {}      # ticker -> sorted list of usable dates (features available)
        self.train_days = {}    # ticker -> list of usable train dates
        self.test_days = {}     # ticker -> list of usable test dates
        self._load()

    def _load(self):
        for tkr in TICKERS:
            bars = pd.read_parquet(os.path.join(DATA_DIR, f"{tkr}_5min.parquet"))
            bars.index = pd.to_datetime(bars.index)
            self.bars[tkr] = bars

            daily = bars.groupby("date").agg(day_volume=("volume", "sum"), day_close=("close", "last"))
            daily.index = pd.to_datetime(daily.index)
            daily = daily.sort_index()
            daily["log_ret"] = np.log(daily["day_close"]).diff()
            # trailing windows use .shift(1) so day t's feature only sees days < t
            daily["adv20"] = daily["day_volume"].shift(1).rolling(ADV_WINDOW).mean()
            daily["vol10"] = daily["log_ret"].shift(1).rolling(VOL_WINDOW).std()
            self.daily[tkr] = daily

            usable = daily.dropna(subset=["adv20", "vol10"]).index
            self.day_list[tkr] = list(usable)
            self.train_days[tkr] = [d for d in usable if d < pd.Timestamp(TRAIN_TEST_CUTOFF)]
            self.test_days[tkr] = [d for d in usable if d >= pd.Timestamp(TRAIN_TEST_CUTOFF)]

            # historical intraday volume profile, computed from TRAIN days only
            train_bars = bars[bars["date"].astype(str) < TRAIN_TEST_CUTOFF[:10]].copy()
            train_bars["bin"] = train_bars.groupby("date").cumcount()
            day_totals = train_bars.groupby("date")["volume"].transform("sum").replace(0, np.nan)
            train_bars["frac"] = train_bars["volume"] / day_totals
            profile = train_bars.groupby("bin")["frac"].mean().reindex(range(BARS_PER_DAY)).fillna(
                1.0 / BARS_PER_DAY
            )
            profile = profile.values
            profile = profile / profile.sum()  # renormalize to exactly 1.0
            self.profile[tkr] = profile

    def get_day_bars(self, ticker, date):
        """Return the 75x5 (open,high,low,close,volume) array for one trading day."""
        bars = self.bars[ticker]
        day_bars = bars[bars["date"] == date.date()] if hasattr(date, "date") else bars[bars["date"] == date]
        return day_bars[["open", "high", "low", "close", "volume"]].values

    def get_features(self, ticker, date):
        row = self.daily[ticker].loc[date]
        return float(row["adv20"]), float(row["vol10"])

    def sample_day(self, rng, split="train", ticker=None):
        if ticker is None:
            ticker = rng.choice(TICKERS)
        days = self.train_days[ticker] if split == "train" else self.test_days[ticker]
        date = days[rng.integers(len(days))]
        return ticker, date
if __name__ == "__main__":
    md = MarketData()
    for t in TICKERS:
        print(t, "train days:", len(md.train_days[t]), "test days:", len(md.test_days[t]),
              "profile sums to", md.profile[t].sum())
