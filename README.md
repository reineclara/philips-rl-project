# Contract-Management RL (PPO)

Minimal PyTorch implementation of Proximal Policy Optimization (PPO) applied to
a simulated single-equipment contract-management MDP.

The agent learns **when** to renew a contract or extend a warranty, given the
coverage status, days-to-expiration, contract cost level and risk level.

## Problem (MDP formulation)

| Component         | Definition                                                            |
| ----------------- | --------------------------------------------------------------------- |
| State (4-dim)     | `[covered, days_to_expiration, contract_cost_level, risk_level]`      |
| Actions (3)       | `0 = do nothing`, `1 = renew contract`, `2 = extend warranty`         |
| Horizon           | `MAX_STEPS = 120` days per episode                                    |

### Reward (balanced, shaped)

| Component               | Sign | Description                                                             |
| ----------------------- | ---- | ----------------------------------------------------------------------- |
| Coverage bonus          |  +   | Per-step reward while covered.                                          |
| Renewal-close bonus     |  +   | Bonus when renewing/extending close to expiration.                      |
| Uncovered penalty       |  -   | Strong per-step penalty while uncovered.                                |
| Expiration penalty      |  -   | One-shot penalty the exact step a contract expires without renewal.    |
| Early-renewal penalty   |  -   | Small penalty for renewing too early (waste of contract cost).         |
| Renewal/extension cost  |  -   | Scaled by `contract_cost_level`.                                        |
| Uncovered-failure cost  |  -   | Penalty on a failure happening while the equipment is uncovered.        |

Coefficients live in `config.py` and keep per-step rewards roughly in `[-5, +3]`.

## Project structure

```
config.py                  # hyperparameters (env + PPO + training + I/O)
environment.py             # ContractEnv (+ internal_tuple for VI alignment)
environment_two_budget.py  # optional two-equipment MDP with shared daily budget
tabular_sarsa.py           # bucketed SARSA baseline on ContractEnv
agent.py                   # RunningMeanStd, RandomAgent, HeuristicAgent, PPOAgent
dp_solver.py               # discounted value iteration for the expected MDP (exact tabular greedy policy)
main.py                    # train PPO + rigorous eval vs random/heuristic/VI + export results.md/json
benchmarks_extended.py     # OOD, SARSA summary, two-equip PPO, reward probe, entropy ablations
ablation_worker.py         # subprocess helper for entropy sweeps (used by benchmarks_extended.py)
requirements.txt           # numpy, torch, matplotlib
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## What the script does

1. **Trains PPO** for `NUM_EPISODES = 5000` episodes with:
   - observation normalization (`RunningMeanStd`, clipped to `±5`)
   - advantage normalization
   - entropy bonus (`ENTROPY_COEF = 0.02`)
   - learning rate `3e-4`
   - clipped surrogate objective (PPO standard).
2. **Evaluates every 100 episodes** (greedy policy, fixed eval seeds).
3. **Saves the best model** to `best_model.pt` whenever the evaluation reward
   improves.
4. After training, **reloads the best model** and runs the rigorous **100-seed**
   suite shared across `RandomAgent`, `HeuristicAgent`, PPO's greedy actor and
   the tabular oracle from **`dp_solver.py`**.
5. **Exports** Welch tests (PPO\,vs heuristic, PPO\,vs VI), heuristic threshold
   sensitivities and metric tables to **`results.md` / `results.json`**.
6. Saves **`training_curve.png`** showing training/eval traces plus dashed
   baseline means (remember to mirror the PNG inside **`figures/`** for LaTeX).

All stochastic policies confront the **same simulator seeds**, so aggregates are directly comparable (`config.EVAL_EPISODES_FINAL = 100` during the exported run).

### Optional extended benchmarks

After `main.py` has produced `best_model.pt`, run:

```bash
python benchmarks_extended.py
```

This bundles tabular SARSA training, an out-of-distribution failure-rate stress
test on the saved PPO policy, a short PPO run on the two-equipment budget
environment, a heuristic-only sweep over the non-coverage penalty multiplier,
and a few subprocessed entropy ablations (`ablation_worker.py`). Outputs land in
`extended_benchmarks.md` and `extended_benchmarks.json` (paths in `config.py`).


## Configuration

All hyperparameters are centralized in `config.py`:

- **Environment** — `MAX_STEPS`, initial-days range, renewal/extension lengths,
  reward coefficients.
- **PPO** — `LEARNING_RATE`, `GAMMA`, `GAE_LAMBDA`, `CLIP_EPS`, `EPOCHS`,
  `BATCH_SIZE`, `HIDDEN_DIM`, `ENTROPY_COEF`, `VALUE_COEF`, `OBS_CLIP`.
- **Training** — `NUM_EPISODES`, `LOG_EVERY`, `EVAL_INTERVAL`, `EVAL_EPISODES`,
  `SEED`.
- **I/O** — `BEST_MODEL_PATH`, `TRAINING_CURVE_PATH`, `EXTENDED_BENCHMARK_*`.
- **Extended suite** — `FAILURE_SCALE_OOD`, `TWO_DAILY_MAX_SPEND`,
  `BUDGET_VIOLATION_PENALTY`, `TABULAR_SARSA_EPISODES`, `TWO_AGENT_TRAIN_EPISODES`,
  `ABLATION_TRAIN_EPISODES`.

## Notes

- No external dataset: everything is simulated from the rules in `environment.py`.
- The implementation stays deliberately minimal (one file per component,
  no framework such as stable-baselines3) so the PPO mechanics remain easy to
  describe in an academic paper.
