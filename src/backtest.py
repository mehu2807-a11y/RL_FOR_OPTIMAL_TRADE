"""
Backtest all policies on the held-out 2020 test set, using an IDENTICAL
order (same ticker, date, and order size) for every policy so comparisons
are paired and fair.

Headline comparison is the classic Almgren-Chriss efficient-frontier view:
mean vs. std of RAW implementation shortfall (cost_bps) across test episodes.
Raw shortfall is lambda-independent (it's just realized execution cost), so
it's the right axis for comparing agents trained at different risk-aversion
levels against each other and against TWAP/VWAP on equal footing. The shaped
risk-adjusted objective (cost + risk penalty) is also reported per-agent
using that agent's OWN training lambda, as a secondary diagnostic of whether
each agent is doing well on the thing it was actually optimized for.
"""
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from market_data import MarketData, TICKERS, BARS_PER_DAY
from execution_env import ExecutionEnv
from paths import MODELS_DIR, RESULTS_DIR

FIXED_ORDER_PCT = 0.05  # headline comparison: liquidate 5% of trailing ADV20 in a single day

AGENTS = {
    "ppo_urgent": (str(MODELS_DIR / "ppo_execution_urgent.zip"), 15.0),
    "ppo_moderate": (str(MODELS_DIR / "ppo_execution_moderate.zip"), 6.0),
    "ppo_patient": (str(MODELS_DIR / "ppo_execution_patient.zip"), 2.0),
}


def twap_action(env):
    remaining_bins = BARS_PER_DAY - env.t
    return np.array([1.0 / remaining_bins], dtype=np.float32)


def vwap_action(env):
    remaining_profile_mass = env.profile[env.t:].sum()
    if remaining_profile_mass <= 1e-9:
        return np.array([1.0], dtype=np.float32)
    frac_of_remaining = env.profile[env.t] / remaining_profile_mass
    return np.array([frac_of_remaining], dtype=np.float32)


def run_episode(env, fixed, action_fn):
    obs, _ = env.reset(fixed=fixed)
    done = False
    total_cost, total_risk = 0.0, 0.0
    while not done:
        action = action_fn(obs, env)
        obs, r, done, _, info = env.step(action)
        total_cost += info["cost_bps"]
        total_risk += info["risk_pen_bps"]
    return total_cost, total_risk


if __name__ == "__main__":
    md = MarketData()

    models = {name: PPO.load(path) for name, (path, _lam) in AGENTS.items()}
    policies = {
        "twap": (lambda obs, env: twap_action(env), 6.0),
        "vwap": (lambda obs, env: vwap_action(env), 6.0),
    }
    for name, (path, lam) in AGENTS.items():
        m = models[name]
        policies[name] = (lambda obs, env, m=m: m.predict(obs, deterministic=True)[0], lam)

    rows = []
    for policy_name, (action_fn, lam) in policies.items():
        env = ExecutionEnv(md, split="test", lambda_risk=lam)
        for tkr in TICKERS:
            for date in md.test_days[tkr]:
                fixed = {"ticker": tkr, "date": date, "order_pct": FIXED_ORDER_PCT}
                cost, risk = run_episode(env, fixed, action_fn)
                rows.append(dict(ticker=tkr, date=date, policy=policy_name,
                                  cost_bps=cost, risk_bps=risk, total_bps=cost + risk))

    results = pd.DataFrame(rows)
    results.to_csv(str(RESULTS_DIR / "backtest_results.csv"), index=False)

    n_days = len(md.test_days[TICKERS[0]])
    print(f"Test episodes per policy: {len(results)//len(policies)}  (5 stocks x {n_days} days)\n")

    print("=== Efficient-frontier view: raw implementation shortfall (lambda-independent) ===")
    frontier = results.groupby("policy")["cost_bps"].agg(["mean", "std"]).reindex(
        ["twap", "vwap", "ppo_patient", "ppo_moderate", "ppo_urgent"]
    )
    print(frontier)

    print("\n=== Each PPO agent's own risk-adjusted objective vs TWAP/VWAP under that SAME lambda ===")
    for name, (_path, lam) in AGENTS.items():
        sub = results[results.policy.isin([name, "twap", "vwap"])]
        # recompute twap/vwap risk_bps under this agent's lambda for a fair side-by-side
        env = ExecutionEnv(md, split="test", lambda_risk=lam)
        bench_rows = []
        for bench_name, action_fn in [("twap", lambda obs, env: twap_action(env)),
                                       ("vwap", lambda obs, env: vwap_action(env))]:
            for tkr in TICKERS:
                for date in md.test_days[tkr]:
                    fixed = {"ticker": tkr, "date": date, "order_pct": FIXED_ORDER_PCT}
                    cost, risk = run_episode(env, fixed, action_fn)
                    bench_rows.append(dict(ticker=tkr, date=date, policy=bench_name,
                                            total_bps=cost + risk))
        bench_df = pd.DataFrame(bench_rows)
        agent_df = results[results.policy == name][["ticker", "date", "total_bps"]].rename(
            columns={"total_bps": "agent"})
        piv = bench_df.pivot_table(index=["ticker", "date"], columns="policy", values="total_bps")
        piv["agent"] = agent_df.set_index(["ticker", "date"])["agent"]
        print(f"\n[{name}, lambda={lam}]  mean total_bps: "
              f"agent={piv['agent'].mean():.2f}  twap={piv['twap'].mean():.2f}  vwap={piv['vwap'].mean():.2f}")
        print(f"  win-rate vs TWAP: {(piv['agent'] < piv['twap']).mean():.3f}   "
              f"win-rate vs VWAP: {(piv['agent'] < piv['vwap']).mean():.3f}")
