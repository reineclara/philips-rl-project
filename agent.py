"""Agents: Random, Heuristic, PPO (with observation normalization)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import config


# ============================================================================
# Observation normalization (Welford online mean/variance)
# ============================================================================


class RunningMeanStd:
    """Online estimation of running mean and variance for observation normalization."""

    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, eps=1e-8):
        return (x - self.mean) / np.sqrt(self.var + eps)


# ============================================================================
# Baselines
# ============================================================================


class RandomAgent:
    """Uniformly random policy over the discrete action space."""

    def __init__(self, num_actions, seed=None):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.integers(0, self.num_actions))


class HeuristicAgent:
    """Rule-based baseline for the single-equipment environment.

    Rules (priority order):
      1. If uncovered                 -> renew          (action 1)
      2. Else if close to expiration  -> extend         (action 2)
      3. Otherwise                    -> do nothing     (action 0)
    """

    def __init__(self, renew_threshold=5):
        self.renew_threshold = renew_threshold

    def act(self, obs):
        covered = obs[0]
        days = obs[1] * config.MAX_COVERED_DAYS  # denormalize
        if covered < 0.5:
            return 1
        if days <= self.renew_threshold:
            return 2
        return 0


# ============================================================================
# PPO actor-critic
# ============================================================================


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)


class PPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.model = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.obs_norm = RunningMeanStd(shape=(obs_dim,))

    def _preprocess(self, obs):
        x = self.obs_norm.normalize(np.asarray(obs, dtype=np.float32))
        x = np.clip(x, -config.OBS_CLIP, config.OBS_CLIP)
        return torch.as_tensor(x, dtype=torch.float32)

    @torch.no_grad()
    def act(self, obs):
        obs_t = self._preprocess(obs)
        logits, value = self.model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    @torch.no_grad()
    def act_greedy(self, obs):
        obs_t = self._preprocess(obs)
        logits, _ = self.model(obs_t)
        return int(torch.argmax(logits).item())

    def compute_gae(self, rewards, values, dones):
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        values_ext = np.append(values, 0.0)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + config.GAMMA * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, obs, actions, old_logprobs, returns, advantages):
        # Update running obs statistics from the collected rollouts.
        obs_np = np.asarray(obs, dtype=np.float32)
        self.obs_norm.update(obs_np)

        obs_norm = np.clip(
            self.obs_norm.normalize(obs_np), -config.OBS_CLIP, config.OBS_CLIP
        )
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        old_logprobs = torch.as_tensor(np.asarray(old_logprobs), dtype=torch.float32)
        returns = torch.as_tensor(np.asarray(returns), dtype=torch.float32)
        advantages = torch.as_tensor(np.asarray(advantages), dtype=torch.float32)

        # Advantage normalization (mandatory for PPO stability).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(obs_t)
        for _ in range(config.EPOCHS):
            idxs = np.random.permutation(n)
            for start in range(0, n, config.BATCH_SIZE):
                batch = idxs[start : start + config.BATCH_SIZE]

                logits, values = self.model(obs_t[batch])
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - old_logprobs[batch])
                surr1 = ratio * advantages[batch]
                surr2 = (
                    torch.clamp(ratio, 1 - config.CLIP_EPS, 1 + config.CLIP_EPS)
                    * advantages[batch]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values.squeeze(-1) - returns[batch]) ** 2).mean()

                loss = (
                    policy_loss
                    + config.VALUE_COEF * value_loss
                    - config.ENTROPY_COEF * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
