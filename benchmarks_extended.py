"""Post-hoc benchmark bundle addressing common RL review check-lists:

* **OOD / stress test** — inflate failure probabilities at evaluation while
  reusing the trained PPO checkpoint.
* **Tabular SARSA** — complementary classic RL baseline on the coarse state
  buckets used in ``tabular_sarsa.py``.
* **Two-equipment + daily budget MDP** — ``environment_two_budget.TwoEquipDailyBudgetEnv``
  exposes portfolio friction that naive per-device threshold rules mishandle when
  two renewals collide financially.
* **Reward-shape sensitivity** — vary ``UNCOVERED_PENALTY`` while benchmarking
  the heuristic (cheap diagnostic).
* **PPO entropy ablations** — short training runs spawned via ``ablation_worker``.

Run AFTER ``python main.py``::

    python benchmarks_extended.py

Artifacts: ``extended_benchmarks.{json,md}`` paths from ``config.py``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Callable

import numpy as np
import torch

import config
from agent import HeuristicAgent, PPOAgent, RandomAgent
from environment import ContractEnv
from environment_two_budget import TwoEquipDailyBudgetEnv
from main import evaluate, summarize, train, welch_t_test
from tabular_sarsa import TabularSARSAAgent


def _evaluate_contract_scaled(
    policy_fn: Callable[[np.ndarray], int],
    seeds,
    failure_scale_mult: float,
) -> dict:
    env = ContractEnv()
    rewards, uncovered_steps, failures = [], [], []
    for seed in seeds:
        env.reset(seed=seed, failure_scale_mult=failure_scale_mult)
        ep_r = ep_u = ep_f = 0.0
        done = False
        while not done:
            action = policy_fn(env._get_obs())
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
        "reward": float(np.mean(rewards)),
    }


class HeuristicJointNine:
    """Independent per-asset heuristics mapped to a single 3×3 index."""

    def __init__(self, renew_threshold: int = 5):
        self.h = HeuristicAgent(renew_threshold=renew_threshold)

    def act(self, obs: np.ndarray) -> int:
        o0 = obs[:4]
        o1 = obs[4:]
        a0 = self.h.act(o0)
        a1 = self.h.act(o1)
        return int(a0 * 3 + a1)


def _evaluate_joint_env(env, policy_fn: Callable[[np.ndarray], int], seeds) -> dict:
    rewards, uncovered_steps, failures = [], [], []
    for seed in seeds:
        env.reset(seed=seed)
        ep_r = ep_u = ep_f = 0.0
        done = False
        while not done:
            action = policy_fn(env._get_obs())
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
        "reward": float(np.mean(rewards)),
    }


def _reward_shape_sweep(seeds) -> list[dict]:
    base = float(config.UNCOVERED_PENALTY)
    rows = []
    for mult in (0.85, 1.0, 1.15):
        config.UNCOVERED_PENALTY = base * mult
        res = evaluate(ContractEnv(), HeuristicAgent().act, seeds)
        rows.append(
            {
                "uncovered_penalty_multiplier": mult,
                "summary": summarize(res["rewards"]),
            }
        )
    config.UNCOVERED_PENALTY = base
    return rows


def run_all() -> dict:
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    eval_seeds = [2000 + i for i in range(config.EVAL_EPISODES_FINAL)]
    report: dict = {}

    # --- Tabular SARSA -------------------------------------------------
    print("[1/5] Training tabular SARSA ...", flush=True)
    sarsa = TabularSARSAAgent()
    sarsa.train(config.TABULAR_SARSA_EPISODES)
    sarsa_raw = sarsa.evaluate_greedy(eval_seeds)
    report["tabular_sarsa"] = {
        "reward": summarize(sarsa_raw["rewards"]),
    }

    # --- OOD PPO -------------------------------------------------------
    print("[2/5] OOD evaluation of saved PPO (if checkpoint exists) ...", flush=True)
    ckpt = pathlib.Path(config.BEST_MODEL_PATH)
    if ckpt.is_file():
        env_ref = ContractEnv()
        ppo = PPOAgent(env_ref.obs_dim, env_ref.action_dim)
        ppo.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        in_dist = evaluate(env_ref, ppo.act_greedy, eval_seeds)
        ood = _evaluate_contract_scaled(
            ppo.act_greedy,
            eval_seeds,
            config.FAILURE_SCALE_OOD,
        )
        report["ppo_in_distribution"] = {"reward": summarize(in_dist["rewards"])}
        report["ppo_ood_failure_scale"] = {
            "scale": config.FAILURE_SCALE_OOD,
            "reward": summarize(ood["rewards"]),
        }
    else:
        report["ppo_ood_failure_scale"] = {
            "note": f"Missing {config.BEST_MODEL_PATH}; run main.py first.",
        }

    # --- Two-equipment portfolio --------------------------------------
    print("[3/5] Two-equipment PPO + baselines ...", flush=True)
    saved_eps = config.NUM_EPISODES
    config.NUM_EPISODES = config.TWO_AGENT_TRAIN_EPISODES
    env2 = TwoEquipDailyBudgetEnv()
    ag2 = PPOAgent(env2.obs_dim, env2.action_dim)
    train(env2, ag2, [1000 + i for i in range(config.EVAL_EPISODES)])
    config.NUM_EPISODES = saved_eps

    heur_joint = HeuristicJointNine()
    rand9 = RandomAgent(env2.action_dim, seed=config.SEED + 3)

    two_ppo = _evaluate_joint_env(env2, ag2.act_greedy, eval_seeds)
    two_heur = _evaluate_joint_env(env2, heur_joint.act, eval_seeds)
    two_rand = _evaluate_joint_env(env2, rand9.act, eval_seeds)
    report["two_equip_budget"] = {
        "ppo": {"reward": summarize(two_ppo["rewards"])},
        "heuristic_joint": {"reward": summarize(two_heur["rewards"])},
        "random": {"reward": summarize(two_rand["rewards"])},
        "welch_ppo_vs_heuristic": welch_t_test(
            two_ppo["rewards"], two_heur["rewards"]
        ),
    }

    # --- Reward coefficient probe -------------------------------------
    print("[4/5] Heuristic reward-shape sensitivity ...", flush=True)
    probe_seeds = eval_seeds[:40]
    report["reward_shape_heuristic"] = _reward_shape_sweep(probe_seeds)

    # --- Entropy ablations --------------------------------------------
    print("[5/5] PPO entropy ablations (subprocess) ...", flush=True)
    grid = [0.01, 0.02, 0.035]
    ab_rows = []
    for ent in grid:
        proc = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).with_name("ablation_worker.py")),
                "--entropy",
                str(ent),
                "--episodes",
                str(config.ABLATION_TRAIN_EPISODES),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            ab_rows.append(
                {
                    "entropy": ent,
                    "error": proc.stderr[-400:],
                }
            )
            continue
        line = proc.stdout.strip().splitlines()[-1]
        ab_rows.append(json.loads(line))
    report["ppo_entropy_ablation"] = ab_rows

    return report


def export_md(report: dict) -> str:
    lines = ["# Extended benchmarks", ""]
    lines.append("Generated by `benchmarks_extended.py` after `main.py`.\n")

    if "tabular_sarsa" in report:
        r = report["tabular_sarsa"]["reward"]
        lines.append(
            f"## Tabular SARSA (greedy, {config.EVAL_EPISODES_FINAL} seeds)\n"
            f"- mean reward: {r['mean']:.2f} ± {r['std']:.2f}  "
            f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]\n"
        )

    ood = report.get("ppo_ood_failure_scale", {})
    if "reward" in ood:
        r = ood["reward"]
        lines.append(
            f"## PPO OOD (failure scale ×{ood.get('scale', config.FAILURE_SCALE_OOD):.2f})\n"
            f"- mean reward: {r['mean']:.2f} ± {r['std']:.2f}\n"
        )
    elif "note" in ood:
        lines.append(f"## PPO OOD\n- {ood['note']}\n")

    if "two_equip_budget" in report:
        te = report["two_equip_budget"]
        w = te["welch_ppo_vs_heuristic"]
        lines.append("## Two equipments + daily budget\n")
        for name in ("ppo", "heuristic_joint", "random"):
            r = te[name]["reward"]
            lines.append(
                f"- **{name}**: {r['mean']:.2f} ± {r['std']:.2f}  "
                f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
            )
        lines.append(
            f"- Welch PPO vs heuristic joint: t={w[0]:.3f}, p={w[1]:.4f}, df≈{w[2]:.1f}\n"
        )

    lines.append("## Heuristic vs UNCOVERED_PENALTY multiplier (40 seeds)\n")
    for row in report.get("reward_shape_heuristic", []):
        s = row["summary"]
        lines.append(
            f"- mult={row['uncovered_penalty_multiplier']:.2f} → "
            f"{s['mean']:.2f} ± {s['std']:.2f}"
        )
    lines.append("")

    lines.append("## Entropy ablation (short PPO runs)\n")
    for row in report.get("ppo_entropy_ablation", []):
        if "mean_reward" in row:
            lines.append(
                f"- entropy={row['entropy']}: mean eval reward {row['mean_reward']:.2f}"
            )
        else:
            lines.append(f"- entropy={row.get('entropy')}: ERROR {row.get('error','')[:120]}")

    return "\n".join(lines) + "\n"


def main():
    payload = run_all()
    path_j = pathlib.Path(config.EXTENDED_BENCHMARK_JSON)
    path_m = pathlib.Path(config.EXTENDED_BENCHMARK_MD)
    path_j.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = export_md(payload)
    path_m.write_text(md, encoding="utf-8")
    print(f"\nWrote {path_j} and {path_m}", flush=True)


if __name__ == "__main__":
    main()
