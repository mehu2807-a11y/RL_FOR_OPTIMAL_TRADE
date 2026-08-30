"""
Optimal-execution Gymnasium environment.

Each episode = liquidate (sell) a quantity Q of one stock over one real
historical trading day (75 five-minute bins), on top of the ACTUAL observed
price/volume path for that stock-day. The agent's own trading is assumed not
to be present in the observed volume (a standard "price-taker" approximation
for a resume-scale project) and is charged a temporary + permanent market
impact cost, in the spirit of Almgren-Chriss (2000).

Action (continuous, shape (1,), in [0,1]):
    fraction of REMAINING inventory to sell in this bin.
    The final bin always force-liquidates whatever inventory is left.

Reward:
    -(implementation-shortfall cost this bin, in bps of order notional)
    -(risk-aversion penalty on remaining inventory's price-volatility exposure)

Observation (5 features):
    time_left_frac, inventory_left_frac, vol10_regime (trailing realized vol),
    expected_relative_volume (historical intraday volume-curve value for this
    bin), realized_liquidity_shock (previous bin's volume vs its historical
    norm).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from market_data import MarketData, BARS_PER_DAY

# ---- market-impact model constants (illustrative, not fitted to a broker's
# real cost curve -- tune ETA/GAMMA/LAMBDA_RISK if you have real impact data) ----
ETA = 0.6          # temporary-impact coefficient (sqrt-law)
IMPACT_BETA = 0.5  # concavity of temporary impact in participation rate
GAMMA = 0.08       # permanent-impact coefficient
DEFAULT_LAMBDA_RISK = 15.0  # risk-aversion weight on residual inventory variance, calibrated
                             # so TWAP's total risk penalty is roughly the same order of
                             # magnitude as its total impact cost -- see sanity_check.py.
                             # Pass a different lambda_risk to ExecutionEnv() to move along
                             # the Almgren-Chriss risk-aversion spectrum (higher = more urgent).

MIN_ORDER_PCT = 0.03   # order size as a fraction of trailing ADV20
MAX_ORDER_PCT = 0.08


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market_data: MarketData, split="train", ticker=None, seed=None,
                 lambda_risk=DEFAULT_LAMBDA_RISK):
        super().__init__()
        self.md = market_data
        self.split = split
        self.fixed_ticker = ticker
        self.rng = np.random.default_rng(seed)
        self.lambda_risk = lambda_risk

        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._episode_ready = False

    def reset(self, *, seed=None, options=None, fixed=None):
        """
        fixed: optional dict {"ticker":..., "date":..., "order_pct":...} to force
        a specific evaluation episode (used by the backtester for reproducibility).
        """
        super().reset(seed=seed)
        if fixed is not None:
            ticker, date, order_pct = fixed["ticker"], fixed["date"], fixed["order_pct"]
        else:
            ticker, date = self.md.sample_day(self.rng, split=self.split, ticker=self.fixed_ticker)
            order_pct = self.rng.uniform(MIN_ORDER_PCT, MAX_ORDER_PCT)

        self.ticker = ticker
        self.date = date
        bars = self.md.get_day_bars(ticker, date)
        assert bars.shape[0] == BARS_PER_DAY, f"expected {BARS_PER_DAY} bars, got {bars.shape[0]}"
        self.open_ = bars[:, 0]
        self.high_ = bars[:, 1]
        self.low_ = bars[:, 2]
        self.close_ = bars[:, 3]
        self.vol_ = bars[:, 4]

        adv20, vol10 = self.md.get_features(ticker, date)
        self.adv20 = adv20
        self.vol10 = vol10
        self.sigma_bar = vol10 / np.sqrt(BARS_PER_DAY)
        self.avg_bin_vol = max(adv20 / BARS_PER_DAY, 1.0)
        self.profile = self.md.profile[ticker]

        self.Q = max(order_pct * adv20, 1.0)
        self.arrival_price = float(self.open_[0])
        self.remaining = self.Q
        self.cum_permanent_impact = 0.0
        self.t = 0
        self.prev_bin_volume = self.avg_bin_vol  # neutral prior for bin 0

        self._episode_ready = True
        return self._obs(), {}

    def _obs(self):
        time_left_frac = (BARS_PER_DAY - self.t) / BARS_PER_DAY
        inv_left_frac = self.remaining / self.Q
        vol_regime = np.clip(self.vol10 / 0.02, 0.0, 5.0)          # ~1.0 at 2% daily vol
        expected_rel_vol = np.clip(self.profile[self.t] * BARS_PER_DAY, 0.0, 5.0)  # ~1.0 = typical
        liquidity_shock = np.clip(self.prev_bin_volume / self.avg_bin_vol, 0.0, 5.0)
        return np.array(
            [time_left_frac, inv_left_frac, vol_regime, expected_rel_vol, liquidity_shock],
            dtype=np.float32,
        )

    def step(self, action):
        assert self._episode_ready
        frac = float(np.clip(action[0], 0.0, 1.0))
        last_step = self.t == BARS_PER_DAY - 1
        qty = self.remaining if last_step else min(frac * self.remaining, self.remaining)

        bar_price = float(self.close_[self.t])
        bar_volume = max(float(self.vol_[self.t]), 1.0)
        participation = qty / bar_volume

        temp_impact = ETA * self.sigma_bar * bar_price * (participation ** IMPACT_BETA)
        perm_impact_increment = GAMMA * self.sigma_bar * bar_price * (qty / self.avg_bin_vol)

        exec_price = bar_price - self.cum_permanent_impact - temp_impact
        self.cum_permanent_impact += perm_impact_increment

        cost = qty * (self.arrival_price - exec_price)  # >0 = worse than arrival, for a sell
        cost_bps = (cost / (self.Q * self.arrival_price)) * 1e4

        self.remaining -= qty
        inv_frac_after = self.remaining / self.Q
        risk_pen_bps = self.lambda_risk * (inv_frac_after ** 2) * (self.sigma_bar ** 2) * 1e4

        reward = -(cost_bps + risk_pen_bps)

        self.prev_bin_volume = bar_volume
        self.t += 1
        terminated = self.t >= BARS_PER_DAY
        obs = self._obs() if not terminated else np.zeros(5, dtype=np.float32)
        info = {"cost_bps": cost_bps, "risk_pen_bps": risk_pen_bps, "qty": qty, "exec_price": exec_price}
        return obs, reward, terminated, False, info
