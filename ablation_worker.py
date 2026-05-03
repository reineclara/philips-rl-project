"""Isolated PPO training for entropy ablations (spawned by benchmarks_extended).

Must not import ``main`` until ``config`` flags are patched, so the parent
process can sweep ``ENTROPY_COEF`` / ``NUM_EPISODES`` without rewriting code.
Emits a single JSON line on stdout for parsing.
"""

from __future__ import annotations

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--episodes", type=int, default=900)
    args = parser.parse_args()

    import numpy as np
    import torch

    import config as cfg

    cfg.ENTROPY_COEF = float(args.entropy)
    cfg.NUM_EPISODES = int(args.episodes)
    sfx = format(args.entropy, ".6f").rstrip("0").rstrip(".") or "0"
    sfx = sfx.replace(".", "_")
    cfg.BEST_MODEL_PATH = f"_ablation_ppo_entropy_{sfx}.pt"

    # Import after config mutation so train()/evaluate honor the overrides.
    from agent import PPOAgent
    from environment import ContractEnv
    from main import evaluate, train

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    env = ContractEnv()
    agent = PPOAgent(env.obs_dim, env.action_dim)
    train_seeds = list(range(3000, 3000 + cfg.EVAL_EPISODES))
    train(env, agent, train_seeds)

    eval_seeds = list(range(5000, 5000 + cfg.EVAL_EPISODES_FINAL))
    res = evaluate(env, agent.act_greedy, eval_seeds)
    out = {
        "entropy": float(args.entropy),
        "mean_reward": float(res["reward"]),
        "episodes": int(args.episodes),
    }
    print(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
