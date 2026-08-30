# RL for Optimal Trade Execution — Indian Equities

A PPO agent that learns how to split a large sell order into a schedule of
child orders across a trading day, trained and backtested on real NSE
minute-bar data, and benchmarked against TWAP and VWAP.

## The problem

Given an order to sell `Q` shares of a stock over one trading day (75 bins
of 5 minutes each, 09:15–15:30 IST), decide how much to sell in each bin.
Sell too fast and you pay more market-impact cost; sell too slow and you
carry the unsold position, exposed to price risk, for longer. This is the
classic Almgren-Chriss (2000) optimal-execution trade-off, cast here as an
RL problem instead of solved in closed form.

## Data

**Source actually used in this build:** real 1-minute OHLCV bars (Jan 2017
– Jan 2021) for five liquid NSE large-caps — RELIANCE, HDFCBANK, ICICIBANK,
INFY, and SBIN — pulled from the public GitHub dataset
[`ShabbirHasan1/NSE-Data`](https://github.com/ShabbirHasan1/NSE-Data).

**Why not yfinance directly, as originally asked:** this sandbox's network
egress blocks Yahoo Finance's API domains, so `yfinance` returns nothing
here. `fetch_yfinance.py` is included as a drop-in alternative that *will*
work on a normal machine — the tradeoff is that yfinance only serves ~60
days of 5-minute history (vs. the 4 years used here), which isn't really
enough to build a proper time-based train/test split. If you have a paid
data feed or broker API (Zerodha/Kite, etc.) with deeper intraday history,
swapping it in only requires matching `prep_data.py`'s output schema
(a parquet per ticker with `open/high/low/close/volume` indexed by
timestamp, plus a `date` column).

**TCS** wasn't present in the GitHub dataset, so **SBIN** was substituted to
keep a 5-stock universe.

Bars are resampled from 1-minute to 5-minute, keeping only full 75-bar
trading days (984 of ~988 calendar days per stock survive this filter).
Split **by date**, not randomly, to avoid leakage:
- **Train:** 2017-01-02 to 2019-12-31 (714 days/stock)
- **Test:** 2020-01-01 to 2021-01-01 (250 days/stock, includes the COVID
  volatility shock — a genuine stress test, not cherry-picked)

## Environment (`execution_env.py`)

A Gymnasium env that replays the **real** historical price/volume path for
one (stock, day) and layers a simulated impact-cost model on top (the
agent's own trading is assumed too small to already be reflected in the
recorded volume — a standard simplification for a project at this scale).

- **Order size:** for training, randomized 3–8% of trailing 20-day ADV each
  episode (for robustness); backtests use a fixed 5% of ADV so every policy
  faces an identical, comparable order.
- **Action:** fraction of *remaining* inventory to sell this bin (continuous,
  `[0,1]`); the last bin always force-liquidates whatever is left.
- **Cost model, per bin `t`:**
  - Temporary impact (sqrt law, reverts): `ETA * sigma_bar * price * participation^0.5`
  - Permanent impact (persists across bins): `GAMMA * sigma_bar * price * (qty / avg_bin_volume)`, accumulated
  - `sigma_bar` = trailing 10-day daily realized vol, scaled to bin-level by `1/sqrt(75)`
  - **Reward** = `-(impact cost in bps of order notional) - lambda_risk * (remaining_inventory_fraction)^2 * sigma_bar^2`
  - `ETA`, `GAMMA` are illustrative constants tuned only so that, under TWAP,
    total impact cost and total risk penalty land in the same ballpark (see
    `sanity_check.py`) — **not** fitted to a real broker's cost curve. If you
    have real impact data, recalibrate these before trusting absolute bps
    numbers.
- **State (5 features):** time remaining, inventory remaining, trailing
  volatility regime, this bin's *historical* expected volume share (from a
  train-only intraday volume profile — no lookahead), and the previous bin's
  realized volume relative to its historical norm.

## Training (`train_ppo.py`)

PPO (Stable-Baselines3), `MlpPolicy` with two 64-unit hidden layers,
600k timesteps, ~130 seconds on a single CPU core. Trained **three agents**
at different risk-aversion levels (`lambda_risk` = 2, 6, 15 — "patient",
"moderate", "urgent") to trace out a risk-return spectrum rather than
reporting a single operating point.

## Results

All backtests: 1,250 held-out 2020 episodes (5 stocks × 250 days), fixed
5%-ADV order, identical (stock, day) pairs across every policy for a fair
paired comparison. Full per-episode numbers in `backtest_results.csv`.

### Efficient frontier (`figures/efficient_frontier.png`)

Mean vs. std. dev. of **raw** implementation shortfall (impact-independent
of any lambda choice — the number a trading desk actually cares about):

| Policy              | Mean shortfall (bps) | Std dev (bps) |
|---------------------|----------------------|----------------|
| TWAP                | 18.8                 | 153.0          |
| VWAP                | 18.6                 | 153.9          |
| PPO, λ=2 (patient)  | 25.5                 | 156.0          |
| PPO, λ=6 (moderate) | 22.1                 | 114.7          |
| PPO, λ=15 (urgent)  | 22.5                 | 89.9           |

The urgent and moderate agents trace out a real frontier: **~25–41% lower
risk (std dev) for a ~3.3 bp increase in mean cost** relative to TWAP/VWAP.
Whether that trade is worth it depends on the desk's actual risk aversion —
that's the point of training multiple λ.

### Why: the liquidation trajectories (`figures/liquidation_trajectory.png`)

All three PPO agents learned to front-load most of the order into the first
1–2 bins, and also to hold back a small remainder for the closing bins. NSE
volume is U-shaped intraday — highest at the open and close, thinnest
midday — so this schedule minimizes participation rate (hence impact) *and*
minimizes time spent holding inventory (hence risk) simultaneously, by
concentrating trading into the two periods with the most natural liquidity.
Higher λ pushes more of the order into the open specifically (faster
completion, more risk-averse); lower λ holds more back for later in the day.

### An honest negative result

The **patient agent (λ=2) is dominated** by TWAP/VWAP — worse on *both*
mean and variance of raw shortfall, and it also loses to TWAP/VWAP on its
own training objective (32.5 bps vs. ~23.3 bps, win rate 49%, essentially a
coin flip trending unfavorable). It didn't converge to a good policy. The
likely cause: at low risk-aversion, the risk term contributes very little
signal, so the reward is dominated by high-variance, unpredictable
price-drift noise — exactly the "training-stability" issue this class of
project is supposed to teach you to diagnose. Things worth trying if you
pick this up: more timesteps, a lower learning rate with more epochs for
finer updates in a low-signal regime, reward normalization, or explicitly
regularizing the policy toward TWAP as a prior at low λ.

## Repo layout

```
src/            all source files (see manifest below)
models/         trained PPO agents (ppo_execution_{urgent,moderate,patient}.zip)
figures/        the three results figures, already generated
results/        backtest_results.csv, already generated
data/           empty (.gitkeep only) -- generate locally, see below
```

All scripts resolve their paths relative to the repo root via `src/paths.py`,
so they work the same regardless of where you clone this.

## Reproducing

```
pip install -r requirements.txt

# get the data -- pick one:
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/ShabbirHasan1/NSE-Data.git ../nse_repo
cd ../nse_repo && git sparse-checkout init --no-cone && cat > .git/info/sparse-checkout << 'EOF'
NSE Minute Data/NSE_Stocks_Data/RELIANCE__EQ__NSE__NSE__MINUTE.csv
NSE Minute Data/NSE_Stocks_Data/HDFCBANK__EQ__NSE__NSE__MINUTE.csv
NSE Minute Data/NSE_Stocks_Data/ICICIBANK__EQ__NSE__NSE__MINUTE.csv
NSE Minute Data/NSE_Stocks_Data/INFY__EQ__NSE__NSE__MINUTE.csv
NSE Minute Data/NSE_Stocks_Data/SBIN__EQ__NSE__NSE__MINUTE.csv
EOF
git checkout main && cd -
python3 src/prep_data.py            # builds data/*.parquet from the clone above
#   -- or, on a machine with normal internet access --
python3 src/fetch_yfinance.py       # pulls ~60 days of 5-min bars directly (see Data section)

python3 src/sanity_check.py         # sanity-checks the impact-cost model
python3 src/train_ppo.py --lambda_risk 15.0 --out models/ppo_execution_urgent.zip
python3 src/train_ppo.py --lambda_risk 6.0  --out models/ppo_execution_moderate.zip
python3 src/train_ppo.py --lambda_risk 2.0  --out models/ppo_execution_patient.zip
python3 src/backtest.py             # writes results/backtest_results.csv, prints summary tables
python3 src/make_figures.py         # writes figures/*.png
```

`data/` is gitignored — the NSE-Data source repo's license isn't clearly
stated, so the derived parquet files aren't redistributed here. Regenerate
them locally with the commands above; it takes a couple of minutes.
The `models/`, `figures/`, and `results/` already in this repo are the
outputs from the run described below, so you can explore results
immediately without retraining anything.


## File manifest

```
src/paths.py             # repo-relative path constants used by every script
src/prep_data.py         # 1-min -> 5-min resampling, clean-day filtering, feature prep
src/fetch_yfinance.py    # alternative live-data loader (see Data section)
src/market_data.py       # loads parquet data, builds ADV/vol/volume-profile features
src/execution_env.py     # Gymnasium env: AC-style impact cost model
src/sanity_check.py      # validates the cost model's relative magnitudes
src/train_ppo.py         # PPO training, parameterized by lambda_risk
src/backtest.py          # evaluates all policies on the held-out test set
src/make_figures.py      # generates the three results figures
results/backtest_results.csv   # full per-episode results (1,250 x 5 policies)
models/ppo_execution_*.zip     # the three trained agents (urgent/moderate/patient)
figures/                 # liquidation_trajectory.png, efficient_frontier.png,
                          # shortfall_distributions.png
LICENSE                  # MIT, covers the code only -- not the third-party data
```
