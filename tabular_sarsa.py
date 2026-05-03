"""Classic tabular SARSA baseline on the original single-equipment simulator.

Answers review requests for ``another RL flavour'' besides PPO operating on exact
interaction data (no differentiable function approximators). States are hashed
via coarse buckets; training is stochastic because failure draws stay realistic.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

import config
from environment import ContractEnv


def _discrete_index(obs: np.ndarray, step_idx: int) -> tuple:
    cov = 1 if float(obs[0]) > 0.5 else 0
    days_float = round(float(obs[1]) * float(config.MAX_COVERED_DAYS))
    bucket_days = int(min(days_float // 5, 12))
    clev = min(int(round(obs[2] * 2.0)), 2)
    rlev = min(int(round(obs[3] * 2.0)), 2)
    tbin = int(min(step_idx // 24, max(config.MAX_STEPS // 24 - 1, 0)))
    return (cov, bucket_days, clev, rlev, tbin)


class TabularSARSAAgent:
    def __init__(self):
        self.q = defaultdict(lambda: np.zeros(3, dtype=np.float64))

    def _eps(self, ep: int, total: int, eps_start=0.4, eps_end=0.05) -> float:
        if total <= 1:
            return eps_end
        return float(np.clip(eps_end + (eps_start - eps_end) * (total - ep) / total,
                            eps_end, eps_start))

    def train(self, num_episodes: int, seed_rollout_start: int = 10_000) -> dict:
        np.random.seed(config.SEED)
        env = ContractEnv()

        gamma = float(config.GAMMA)
        alpha_floor = 0.05

        for ep_idx in range(num_episodes):
            seed = seed_rollout_start + ep_idx * 7937
            obs = env.reset(seed=seed)
            step_idx = 0
            s = _discrete_index(obs, step_idx)
            epsilon = self._eps(ep_idx, num_episodes)

            action = (
                int(np.random.randint(0, 3))
                if np.random.random() < epsilon
                else int(np.argmax(self.q[s]))
            )

            done = False
            alpha = alpha_floor + (
                0.25 * float(num_episodes - ep_idx) / max(float(num_episodes), 1.0)
            )

            while not done:
                next_obs, reward, done, info_env = env.step(action)
                step_idx += 1

                if done:
                    delta = reward - self.q[s][action]
                    self.q[s][action] += alpha * delta
                    break

                s_next = _discrete_index(next_obs, step_idx)
                epsilon_next = self._eps(ep_idx, num_episodes)
                action_next = (
                    int(np.random.randint(0, 3))
                    if np.random.random() < epsilon_next
                    else int(np.argmax(self.q[s_next]))
                )

                td = reward + gamma * self.q[s_next][action_next] - self.q[s][action]
                self.q[s][action] += alpha * td

                s = s_next
                action = action_next
                obs = next_obs

        mean_reward = float(
            self.evaluate_greedy(eval_seeds=range(2000, 2000 + config.EVAL_EPISODES_FINAL))[
                "reward"
            ]
        )
        return {"mean_eval_reward": mean_reward}

    def act_greedy(self, obs: np.ndarray, step_idx: int) -> int:
        s = _discrete_index(obs, step_idx)
        qa = self.q[s]
        return int(np.argmax(qa))

    def evaluate_greedy(self, eval_seeds) -> dict:
        env = ContractEnv()
        rew_list, unc, uf = [], [], []
        for seed in eval_seeds:
            obs = env.reset(seed=seed)
            ep_reward = ep_unc = ep_fails = step_idx = 0
            done = False
            while not done:
                act = self.act_greedy(obs, step_idx)
                obs, rew_step, done, info = env.step(act)
                ep_reward += rew_step
                ep_unc += int(info["uncovered"])
                ep_fails += int(info["uncovered_failure"])
                step_idx += 1
            rew_list.append(ep_reward)
            unc.append(ep_unc)
            uf.append(ep_fails)
        return {
            "reward": float(np.mean(rew_list)),
            "rewards": rew_list,
            "uncovered_steps": unc,
            "uncovered_failures": uf,
        }
