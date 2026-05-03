"""Train PPO, evaluate periodically, save the best model, compare baselines."""

import copy

import matplotlib

matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from agent import HeuristicAgent, PPOAgent, RandomAgent
from environment import ContractEnv


# ============================================================================
# Rollouts
# ============================================================================


def collect_training_episode(env, agent):
    """Run one training episode with stochastic (exploratory) actions."""
    obs = env.reset()
    obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
    ep_reward = 0.0

    done = False
    while not done:
        action, logp, value = agent.act(obs)
        next_obs, reward, done, _ = env.step(action)

        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        val_buf.append(value)
        rew_buf.append(reward)
        done_buf.append(float(done))

        ep_reward += reward
        obs = next_obs

    return {
        "obs": obs_buf,
        "actions": act_buf,
        "logprobs": logp_buf,
        "values": val_buf,
        "rewards": rew_buf,
        "dones": done_buf,
        "ep_reward": ep_reward,
    }


def evaluate(env, policy_fn, seeds):
    """Run one evaluation episode per seed. Same seeds => fair comparison."""
    rewards, uncovered_steps, failures = [], [], []
    for seed in seeds:
        obs = env.reset(seed=seed)
        ep_r, ep_u, ep_f = 0.0, 0, 0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, done, info = env.step(action)
            ep_r += reward
            ep_u += info["uncovered"]
            ep_f += info["uncovered_failure"]
        rewards.append(ep_r)
        uncovered_steps.append(ep_u)
        failures.append(ep_f)

    return {
        "reward": float(np.mean(rewards)),
        "uncovered_steps": float(np.mean(uncovered_steps)),
        "uncovered_failures": float(np.mean(failures)),
    }


# ============================================================================
# Training loop
# ============================================================================


def train(env, agent, eval_seeds):
    episode_rewards = []
    eval_points, eval_rewards = [], []
    best_eval_reward = -float("inf")
    best_state = None

    print(
        f"Training for {config.NUM_EPISODES} episodes "
        f"(lr={config.LEARNING_RATE}, entropy={config.ENTROPY_COEF}).\n"
    )

    for ep in range(1, config.NUM_EPISODES + 1):
        batch = collect_training_episode(env, agent)
        advantages, returns = agent.compute_gae(
            batch["rewards"], batch["values"], batch["dones"]
        )
        agent.update(
            batch["obs"], batch["actions"], batch["logprobs"], returns, advantages
        )
        episode_rewards.append(batch["ep_reward"])

        if ep % config.EVAL_INTERVAL == 0:
            eval_result = evaluate(env, agent.act_greedy, eval_seeds)
            eval_points.append(ep)
            eval_rewards.append(eval_result["reward"])

            if eval_result["reward"] > best_eval_reward:
                best_eval_reward = eval_result["reward"]
                best_state = copy.deepcopy(agent.model.state_dict())
                torch.save(best_state, config.BEST_MODEL_PATH)

            if ep % config.LOG_EVERY == 0:
                recent_train = np.mean(episode_rewards[-config.LOG_EVERY :])
                print(
                    f"Ep {ep:5d} | train avg = {recent_train:7.2f} "
                    f"| eval = {eval_result['reward']:7.2f} "
                    f"| best eval = {best_eval_reward:7.2f}"
                )

    if best_state is not None:
        agent.model.load_state_dict(best_state)
        print(f"\nRestored best model (eval reward = {best_eval_reward:.2f}).")

    return episode_rewards, eval_points, eval_rewards


# ============================================================================
# Reporting
# ============================================================================


def print_comparison(random_r, heuristic_r, ppo_r, n):
    print("\n" + "=" * 70)
    print(f" Final comparison (average over {n} evaluation episodes, same seeds)")
    print("=" * 70)
    print(f"{'Metric':<30} {'Random':>10} {'Heuristic':>12} {'PPO':>10}")
    print(f"{'-' * 30} {'-' * 10} {'-' * 12} {'-' * 10}")
    print(
        f"{'Average reward':<30} "
        f"{random_r['reward']:>10.2f} {heuristic_r['reward']:>12.2f} {ppo_r['reward']:>10.2f}"
    )
    print(
        f"{'Average uncovered steps':<30} "
        f"{random_r['uncovered_steps']:>10.2f} {heuristic_r['uncovered_steps']:>12.2f} "
        f"{ppo_r['uncovered_steps']:>10.2f}"
    )
    print(
        f"{'Average uncovered failures':<30} "
        f"{random_r['uncovered_failures']:>10.2f} {heuristic_r['uncovered_failures']:>12.2f} "
        f"{ppo_r['uncovered_failures']:>10.2f}"
    )


def plot_training_curve(
    episode_rewards, eval_points, eval_rewards, random_r, heuristic_r
):
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(episode_rewards) + 1)
    ax.plot(
        episodes,
        episode_rewards,
        color="lightgray",
        alpha=0.5,
        label="Training reward (raw)",
    )

    window = max(10, len(episode_rewards) // 50)
    if len(episode_rewards) >= window:
        smoothed = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            np.arange(window, len(episode_rewards) + 1),
            smoothed,
            color="tab:blue",
            linewidth=2,
            label=f"Training reward (MA {window})",
        )

    if eval_points:
        ax.plot(
            eval_points,
            eval_rewards,
            "o-",
            color="tab:orange",
            linewidth=2,
            label="Evaluation reward (greedy)",
        )

    ax.axhline(
        random_r["reward"],
        linestyle="--",
        color="tab:red",
        label=f"Random baseline ({random_r['reward']:.1f})",
    )
    ax.axhline(
        heuristic_r["reward"],
        linestyle="--",
        color="tab:green",
        label=f"Heuristic baseline ({heuristic_r['reward']:.1f})",
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("PPO training curve — contract-management MDP")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.TRAINING_CURVE_PATH, dpi=100)
    plt.close(fig)
    print(f"Training curve saved to '{config.TRAINING_CURVE_PATH}'.")


# ============================================================================
# Main
# ============================================================================


def main():
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    env = ContractEnv()
    agent = PPOAgent(env.obs_dim, env.action_dim)

    print(f"Observation dim: {env.obs_dim} | Action dim: {env.action_dim}")

    # Fixed evaluation seeds => every agent is evaluated on the same contracts.
    eval_seeds = [1000 + i for i in range(config.EVAL_EPISODES)]

    episode_rewards, eval_points, eval_rewards = train(env, agent, eval_seeds)

    # Final evaluation
    print(f"\nFinal evaluation on {config.EVAL_EPISODES} episodes...")
    ppo_r = evaluate(env, agent.act_greedy, eval_seeds)
    heuristic_r = evaluate(env, HeuristicAgent().act, eval_seeds)
    random_r = evaluate(env, RandomAgent(env.action_dim, seed=config.SEED).act, eval_seeds)

    print(f"\nPPO         : reward={ppo_r['reward']:.2f}  "
          f"uncovered_steps={ppo_r['uncovered_steps']:.2f}  "
          f"uncovered_failures={ppo_r['uncovered_failures']:.2f}")

    print_comparison(random_r, heuristic_r, ppo_r, config.EVAL_EPISODES)
    plot_training_curve(episode_rewards, eval_points, eval_rewards, random_r, heuristic_r)


if __name__ == "__main__":
    main()
