import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from market_data import MarketData
from execution_env import ExecutionEnv, DEFAULT_LAMBDA_RISK
from paths import MODELS_DIR

N_ENVS = 8
TOTAL_TIMESTEPS = 600_000


def make_env(seed, lambda_risk):
    def _init():
        md = MarketData()
        env = ExecutionEnv(md, split="train", seed=seed, lambda_risk=lambda_risk)
        return Monitor(env)
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_risk", type=float, default=DEFAULT_LAMBDA_RISK)
    parser.add_argument("--out", type=str, default=str(MODELS_DIR / "ppo_execution.zip"))
    args = parser.parse_args()

    t0 = time.time()
    env = DummyVecEnv([make_env(seed=i, lambda_risk=args.lambda_risk) for i in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
        seed=42,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    model.save(args.out)
    print(f"\nDone in {time.time()-t0:.0f}s. lambda_risk={args.lambda_risk} Saved to {args.out}")
