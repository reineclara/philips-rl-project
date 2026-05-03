# Evaluation results

All metrics are averaged over **100 evaluation episodes** using the same seeds across policies for a fair comparison.

**Tabular VI** maximises $\gamma$-discounted *expected* return under the same transitions as `environment.py`; per-episode returns still vary because failures are sampled in simulation.

## Main comparison

| Metric | Random | Heuristic | PPO | VI (expected MDP) |
|---|---|---|---|---|
| Reward | -241.44 ± 92.66  [-259.60, -223.28] | 58.06 ± 3.93  [57.29, 58.84] | 44.83 ± 20.05  [40.90, 48.76] | 58.08 ± 3.96  [57.31, 58.86] |
| Uncovered steps | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] | 1.12 ± 1.81  [0.77, 1.47] | 0.00 ± 0.00  [0.00, 0.00] |
| Uncovered failures | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] | 0.11 ± 0.40  [0.03, 0.19] | 0.00 ± 0.00  [0.00, 0.00] |

## Welch's t-test (PPO vs Heuristic, reward)

- t = -6.477
- p = 0.0000
- df ≈ 106.6

## Welch's t-test (PPO vs Tabular VI, reward)

- t = -6.485
- p = 0.0000
- df ≈ 106.7

## Heuristic sensitivity sweep (renew threshold)

| Threshold | n | Mean | SD | 95% CI |
|---|---|---|---|---|
| 3 | 100 | 58.06 | 3.93 | [57.29, 58.84] |
| 5 | 100 | 58.06 | 3.93 | [57.29, 58.84] |
| 7 | 100 | 49.98 | 4.00 | [49.20, 50.76] |
| 10 | 100 | 49.74 | 4.23 | [48.91, 50.57] |
| 15 | 100 | 49.14 | 4.62 | [48.24, 50.04] |