import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from market_data import MarketData, TICKERS, BARS_PER_DAY
from execution_env import ExecutionEnv
from backtest import twap_action, vwap_action, FIXED_ORDER_PCT, AGENTS
from paths import FIGURES_DIR, RESULTS_DIR

OUT_DIR = str(FIGURES_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
md = MarketData()
models = {name: PPO.load(path) for name, (path, _lam) in AGENTS.items()}
env = ExecutionEnv(md, split="test")
rng = np.random.default_rng(7)
sample_pairs = []
for tkr in TICKERS:
    dates = md.test_days[tkr]
    chosen = rng.choice(len(dates), size=40, replace=False)
    sample_pairs += [(tkr, dates[i]) for i in chosen]
policy_names = ["twap", "vwap", "ppo_patient", "ppo_moderate", "ppo_urgent"]
traj = {p: [] for p in policy_names}
for tkr, date in sample_pairs:
    fixed = {"ticker": tkr, "date": date, "order_pct": FIXED_ORDER_PCT}
    for policy in policy_names:
        obs, _ = env.reset(fixed=fixed)
        path = [1.0]
        done = False
        while not done:
            if policy == "twap":
                action = twap_action(env)
            elif policy == "vwap":
                action = vwap_action(env)
            else:
                action, _ = models[policy].predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(action)
            path.append(env.remaining / env.Q)
        traj[policy].append(path)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
x = np.arange(BARS_PER_DAY + 1)
colors = {"twap": "#9ca3af", "vwap": "#f59e0b", "ppo_patient": "#86efac",
          "ppo_moderate": "#60a5fa", "ppo_urgent": "#1e3a8a"}
labels = {"twap": "TWAP", "vwap": "VWAP", "ppo_patient": "PPO (low risk-aversion, \u03bb=2)",
          "ppo_moderate": "PPO (moderate, \u03bb=6)", "ppo_urgent": "PPO (high risk-aversion, \u03bb=15)"}
for policy in policy_names:
    arr = np.array(traj[policy])
    ax.plot(x, arr.mean(axis=0), label=labels[policy], color=colors[policy], linewidth=2.5)
ax.set_xlabel("5-minute bin (0 = market open, 75 = market close)")
ax.set_ylabel("Fraction of order remaining")
ax.set_title("Average liquidation trajectory by risk-aversion level\n(held-out 2020 test days, 5% ADV order)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "liquidation_trajectory.png"), dpi=150)
print("saved liquidation_trajectory.png")
results = pd.read_csv(str(RESULTS_DIR / "backtest_results.csv"))
frontier = results.groupby("policy")["cost_bps"].agg(["mean", "std"])

fig, ax = plt.subplots(figsize=(7, 6))
order = ["twap", "vwap", "ppo_patient", "ppo_moderate", "ppo_urgent"]
for p in order:
    row = frontier.loc[p]
    marker = "s" if p in ("twap", "vwap") else "o"
    ax.scatter(row["std"], row["mean"], s=140, color=colors[p], marker=marker,
               edgecolor="black", zorder=3, label=labels.get(p, p.upper()))
    ax.annotate(labels.get(p, p.upper()), (row["std"], row["mean"]),
                textcoords="offset points", xytext=(8, 6), fontsize=9)
ax.set_xlabel("Std. dev. of implementation shortfall (bps) \u2192 risk")
ax.set_ylabel("Mean implementation shortfall (bps) \u2192 cost")
ax.set_title("Execution risk-return frontier\n(1,250 held-out stock-days, 5% ADV order)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "efficient_frontier.png"), dpi=150)
print("saved efficient_frontier.png")
fig, ax = plt.subplots(figsize=(9, 5.5))
data = [results[results.policy == p]["cost_bps"].clip(-150, 400) for p in order]
bp = ax.boxplot(data, tick_labels=[labels.get(p, p) for p in order], showfliers=False, patch_artist=True)
for patch, p in zip(bp["boxes"], order):
    patch.set_facecolor(colors[p])
    patch.set_alpha(0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Raw implementation shortfall (bps of order notional)")
ax.set_title("Distribution of execution cost across 1,250 held-out test episodes")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "shortfall_distributions.png"), dpi=150)
print("saved shortfall_distributions.png")
