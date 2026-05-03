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
config.py          # hyperparameters (env + PPO + training + I/O)
environment.py     # ContractEnv (+ internal_tuple for VI alignment)
agent.py           # RunningMeanStd, RandomAgent, HeuristicAgent, PPOAgent
dp_solver.py       # discounted value iteration for the expected MDP (exact tabular greedy policy)
main.py            # train PPO + rigorous eval vs random/heuristic/VI + export results.md/json
requirements.txt  # numpy, torch, matplotlib
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

1. **Trains PPO** for `NUM_EPISODES = 3000` episodes with:
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


## Configuration

All hyperparameters are centralized in `config.py`:

- **Environment** — `MAX_STEPS`, initial-days range, renewal/extension lengths,
  reward coefficients.
- **PPO** — `LEARNING_RATE`, `GAMMA`, `GAE_LAMBDA`, `CLIP_EPS`, `EPOCHS`,
  `BATCH_SIZE`, `HIDDEN_DIM`, `ENTROPY_COEF`, `VALUE_COEF`, `OBS_CLIP`.
- **Training** — `NUM_EPISODES`, `LOG_EVERY`, `EVAL_INTERVAL`, `EVAL_EPISODES`,
  `SEED`.
- **I/O** — `BEST_MODEL_PATH`, `TRAINING_CURVE_PATH`.

## Notes

- No external dataset: everything is simulated from the rules in `environment.py`.
- The implementation stays deliberately minimal (one file per component,
  no framework such as stable-baselines3) so the PPO mechanics remain easy to
  describe in an academic paper.
