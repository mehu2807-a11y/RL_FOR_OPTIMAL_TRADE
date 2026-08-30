import numpy as np
from market_data import MarketData, BARS_PER_DAY
from execution_env import ExecutionEnv

md = MarketData()
env = ExecutionEnv(md, split="train", seed=0)

def run_policy(policy_fn, n_episodes=200, seed=0):
    rng = np.random.default_rng(seed)
    costs = []
    env.rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_cost = 0.0
        done = False
        while not done:
            a = policy_fn(obs, env)
            obs, r, done, _, info = env.step(a)
            total_cost += info["cost_bps"] * (info["qty"])  # weight by qty for a clean re-aggregate
        costs.append(total_cost / env.Q)  # back to bps of total order (approx, qty-weighted)
    return np.array(costs)

def dump_all(obs, env):
    # sell everything on bin 0, nothing after
    return np.array([1.0]) if env.t == 0 else np.array([0.0])

def twap(obs, env):
    remaining_bins = BARS_PER_DAY - env.t
    return np.array([1.0 / remaining_bins])

dump_costs = run_policy(dump_all, 300, seed=1)
twap_costs = run_policy(twap, 300, seed=1)

print(f"Dump-all-at-open:  mean cost = {dump_costs.mean():7.2f} bps   (std {dump_costs.std():.2f})")
print(f"TWAP:              mean cost = {twap_costs.mean():7.2f} bps   (std {twap_costs.std():.2f})")
print(f"Ratio dump/twap: {dump_costs.mean()/twap_costs.mean():.1f}x")
