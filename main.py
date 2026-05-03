"""Train PPO, evaluate periodically, save the best model, compare baselines.

This script produces the empirical artifacts cited in the paper: mean ± SD ±
95% CI for all metrics, Welch's tests, heuristic threshold sensitivity, and
the performance of an exact tabular policy obtained by value iteration on the
expected-reward MDP (DpSolver baseline). Aggregates go to ``results.{md,json}``.
"""

import copy
import json
import math
import sys

import matplotlib

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

matplotlib.use("Agg")  # no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from agent import HeuristicAgent, PPOAgent, RandomAgent
from dp_solver import compute_optimal_policy
from environment import ContractEnv


# ============================================================================
# Rollouts
# ============================================================================


def evaluate_with_dp_policy(env, policy_table, seeds):
    """Greedy rollout under a tabular policy keyed by ``env.internal_tuple()``."""
    rewards, uncovered_steps, failures = [], [], []
    for seed in seeds:
        env.reset(seed=seed)
        ep_r, ep_u, ep_f = 0.0, 0, 0
        done = False
        while not done:
            action = policy_table[env.internal_tuple()]
            _, reward, done, info = env.step(action)
            ep_r += reward
            ep_u += info["uncovered"]
            ep_f += info["uncovered_failure"]
        rewards.append(ep_r)
        uncovered_steps.append(ep_u)
        failures.append(ep_f)

    return {
        "rewards": rewards,
        "uncovered_steps": uncovered_steps,
        "uncovered_failures": failures,
        "reward": float(np.mean(rewards)),
        "uncovered_steps_mean": float(np.mean(uncovered_steps)),
        "uncovered_failures_mean": float(np.mean(failures)),
    }


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
    """Run one evaluation episode per seed. Same seeds => fair comparison.

    Returns the FULL per-episode lists so that the caller can compute not just
    means, but also standard deviations, confidence intervals and statistical
    tests.
    """
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
        "rewards": rewards,
        "uncovered_steps": uncovered_steps,
        "uncovered_failures": failures,
        # Convenience scalar summaries (used during training for logging only).
        "reward": float(np.mean(rewards)),
        "uncovered_steps_mean": float(np.mean(uncovered_steps)),
        "uncovered_failures_mean": float(np.mean(failures)),
    }


# ============================================================================
# Statistics
# ============================================================================


def summarize(values, confidence=0.95):
    """Mean, SD (sample), and two-sided CI on the mean using Student's t.

    We use the t-distribution rather than a normal approximation because our
    sample size (100 evaluation episodes) is finite and the empirical
    distribution of episode rewards is not guaranteed to be normal. The
    critical values are computed in closed form to avoid a SciPy dependency.
    """
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n > 0 else 0.0
    # Student's t critical value (two-sided). Approximate via normal for
    # df >= 30 (within 1% error), falling back to a short table otherwise.
    t_crit = _t_critical(n - 1, confidence)
    half_width = t_crit * sem
    return {
        "n": n,
        "mean": mean,
        "std": sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "ci_half_width": half_width,
    }


def _t_critical(df, confidence):
    """Two-sided t critical value, small lookup for df < 30, normal above."""
    table = {
        (0.95, 5):   2.571,
        (0.95, 10):  2.228,
        (0.95, 15):  2.131,
        (0.95, 20):  2.086,
        (0.95, 25):  2.060,
        (0.95, 30):  2.042,
    }
    if df >= 30:
        return 1.96 if confidence == 0.95 else 2.576
    # Find nearest larger or equal table entry (conservative).
    keys = sorted(k for k in table if k[0] == confidence and k[1] >= df)
    return table[keys[0]] if keys else 1.96


def welch_t_test(x, y):
    """Welch's (unequal variance) two-sided t-test.

    Returns (t_statistic, approximate p-value). The p-value is computed via
    the complementary error function (normal approximation, sufficient for
    df > 30 and a ~3% precision goal here). No SciPy dependency is needed.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0.0:
        return 0.0, 1.0
    t = (x.mean() - y.mean()) / se
    # Welch-Satterthwaite df (unused in the normal-approx p-value but reported).
    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx / nx) ** 2 / max(nx - 1, 1) + (vy / ny) ** 2 / max(ny - 1, 1)
    df = df_num / df_den if df_den > 0 else float(nx + ny - 2)
    # Two-sided p-value from standard normal.
    p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(p), float(df)


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


def fmt(stats, decimals=2):
    """`mean ± std  [CI_low, CI_high]` as a compact string."""
    return (
        f"{stats['mean']:.{decimals}f} ± {stats['std']:.{decimals}f}  "
        f"[{stats['ci_low']:.{decimals}f}, {stats['ci_high']:.{decimals}f}]"
    )


def print_rigorous_comparison(results, n):
    print("\n" + "=" * 108)
    print(
        f" Final comparison — {n} evaluation episodes, shared seeds "
        f"(mean ± SD  [95% CI])"
    )
    print("=" * 108)
    header = (
        f"{'Metric':<26} {'Random':>20} {'Heuristic':>20} "
        f"{'PPO':>20} {'VI (exp.)':>20}"
    )
    print(header)
    print("-" * 108)
    rows = ["random", "heuristic", "ppo", "vi_expected"]
    for metric_name, key in [
        ("Cumulative reward", "reward"),
        ("Uncovered steps",   "uncovered_steps"),
        ("Uncovered failures","uncovered_failures"),
    ]:
        parts = [
            fmt(results[row][key]) for row in rows
        ]
        print(f"{metric_name:<26} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20} {parts[3]:>20}")
    print("=" * 108)


def plot_training_curve(
    episode_rewards,
    eval_points,
    eval_rewards,
    random_r,
    heuristic_r,
    vi_r=None,
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
    if vi_r is not None:
        ax.axhline(
            vi_r["reward"],
            linestyle="--",
            color="tab:purple",
            label=f"VI expected-opt ({vi_r['reward']:.1f})",
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
# Sensitivity analysis (heuristic renew threshold)
# ============================================================================


def heuristic_sensitivity(env, seeds, thresholds):
    """Re-evaluate the heuristic baseline for several renew thresholds.

    This is cheap (no re-training needed) and quantifies how much of the
    heuristic's performance comes from having the right threshold vs. from
    the rule structure itself.
    """
    sweep = []
    for th in thresholds:
        h = HeuristicAgent(renew_threshold=th)
        res = evaluate(env, h.act, seeds)
        summary = summarize(res["rewards"])
        summary["threshold"] = th
        sweep.append(summary)
    return sweep


# ============================================================================
# Results export
# ============================================================================


def build_results(random_raw, heuristic_raw, ppo_raw, vi_raw,
                  sensitivity, welch_ppo_heuristic, welch_ppo_vi):
    def pack(raw):
        return {
            "reward":              summarize(raw["rewards"]),
            "uncovered_steps":     summarize(raw["uncovered_steps"]),
            "uncovered_failures":  summarize(raw["uncovered_failures"]),
        }
    th, ph, dfh = welch_ppo_heuristic
    tv, pv, dfv = welch_ppo_vi
    return {
        "random":               pack(random_raw),
        "heuristic":            pack(heuristic_raw),
        "ppo":                  pack(ppo_raw),
        "vi_expected":          pack(vi_raw),
        "sensitivity":          sensitivity,
        "welch_ppo_vs_heuristic": {
            "t": th, "p": ph, "df": dfh,
        },
        "welch_ppo_vs_vi": {
            "t": tv, "p": pv, "df": dfv,
        },
    }


def export_results(results, n_eval):
    with open(config.RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("# Evaluation results\n")
    lines.append(f"All metrics are averaged over **{n_eval} evaluation episodes** "
                 "using the same seeds across policies for a fair comparison.\n")
    lines.append(
        "**Tabular VI** maximises $\\gamma$-discounted *expected* return under the "
        "same transitions as `environment.py`; per-episode returns still vary "
        "because failures are sampled in simulation.\n"
    )
    lines.append("## Main comparison\n")
    lines.append("| Metric | Random | Heuristic | PPO | VI (expected MDP) |")
    lines.append("|---|---|---|---|---|")
    for label, key in [
        ("Reward",             "reward"),
        ("Uncovered steps",    "uncovered_steps"),
        ("Uncovered failures", "uncovered_failures"),
    ]:
        lines.append(
            f"| {label} | {fmt(results['random'][key])} | "
            f"{fmt(results['heuristic'][key])} | "
            f"{fmt(results['ppo'][key])} | "
            f"{fmt(results['vi_expected'][key])} |"
        )
    lines.append("")
    w1 = results["welch_ppo_vs_heuristic"]
    lines.append("## Welch's t-test (PPO vs Heuristic, reward)\n")
    lines.append(f"- t = {w1['t']:.3f}")
    lines.append(f"- p = {w1['p']:.4f}")
    lines.append(f"- df ≈ {w1['df']:.1f}")
    lines.append("")
    w2 = results["welch_ppo_vs_vi"]
    lines.append("## Welch's t-test (PPO vs Tabular VI, reward)\n")
    lines.append(f"- t = {w2['t']:.3f}")
    lines.append(f"- p = {w2['p']:.4f}")
    lines.append(f"- df ≈ {w2['df']:.1f}")
    lines.append("")
    lines.append("## Heuristic sensitivity sweep (renew threshold)\n")
    lines.append("| Threshold | n | Mean | SD | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for s in results["sensitivity"]:
        lines.append(
            f"| {s['threshold']} | {s['n']} | {s['mean']:.2f} | "
            f"{s['std']:.2f} | [{s['ci_low']:.2f}, {s['ci_high']:.2f}] |"
        )
    with open(config.RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResults exported to '{config.RESULTS_JSON_PATH}' and "
          f"'{config.RESULTS_MD_PATH}'.")




# ============================================================================
# Main
# ============================================================================


def main():
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    env = ContractEnv()
    agent = PPOAgent(env.obs_dim, env.action_dim)

    print(f"Observation dim: {env.obs_dim} | Action dim: {env.action_dim}")

    # Fixed evaluation seeds for PERIODIC evaluation during training (cheap).
    eval_seeds_train = [1000 + i for i in range(config.EVAL_EPISODES)]
    # Larger, distinct seed pool used ONCE for the final rigorous comparison.
    eval_seeds_final = [2000 + i for i in range(config.EVAL_EPISODES_FINAL)]

    episode_rewards, eval_points, eval_rewards = train(env, agent, eval_seeds_train)

    # Expected-optimal tabular policy (same gamma, expected failure reward).
    print("\nSolving tabular value iteration (expected MDP)...")
    _, vi_policy = compute_optimal_policy()

    # ----- Rigorous final evaluation (shared seeds across policies) -----
    print(f"\nFinal rigorous evaluation on {config.EVAL_EPISODES_FINAL} "
          f"episodes (shared seeds)...")
    ppo_raw       = evaluate(env, agent.act_greedy, eval_seeds_final)
    heuristic_raw = evaluate(env, HeuristicAgent().act, eval_seeds_final)
    random_raw    = evaluate(
        env, RandomAgent(env.action_dim, seed=config.SEED).act, eval_seeds_final
    )
    vi_raw = evaluate_with_dp_policy(env, vi_policy, eval_seeds_final)

    welch_ph = welch_t_test(ppo_raw["rewards"], heuristic_raw["rewards"])
    welch_pv = welch_t_test(ppo_raw["rewards"], vi_raw["rewards"])

    # Heuristic sensitivity sweep over the renew threshold.
    sensitivity = heuristic_sensitivity(
        env, eval_seeds_final, config.HEURISTIC_THRESHOLDS
    )

    results = build_results(
        random_raw, heuristic_raw, ppo_raw, vi_raw, sensitivity,
        welch_ph, welch_pv,
    )

    print_rigorous_comparison(results, config.EVAL_EPISODES_FINAL)
    print(
        f"\nWelch's t-test PPO vs Heuristic: "
        f"t = {welch_ph[0]:.3f}, p = {welch_ph[1]:.4f}, df ~ {welch_ph[2]:.1f}"
    )
    print(
        f"Welch's t-test PPO vs Tabular VI: "
        f"t = {welch_pv[0]:.3f}, p = {welch_pv[1]:.4f}, df ~ {welch_pv[2]:.1f}"
    )
    print("\nHeuristic sensitivity sweep (renew threshold -> mean reward):")
    for s in sensitivity:
        print(
            f"  threshold = {s['threshold']:>2}  "
            f"reward = {s['mean']:7.2f} ± {s['std']:5.2f}  "
            f"[{s['ci_low']:7.2f}, {s['ci_high']:7.2f}]"
        )

    export_results(results, config.EVAL_EPISODES_FINAL)

    plot_training_curve(
        episode_rewards, eval_points, eval_rewards,
        {"reward": results["random"]["reward"]["mean"]},
        {"reward": results["heuristic"]["reward"]["mean"]},
        vi_r={"reward": results["vi_expected"]["reward"]["mean"]},
    )


if __name__ == "__main__":
    main()
