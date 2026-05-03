# Evaluation results

All metrics are averaged over **100 evaluation episodes** using the same seeds across policies for a fair comparison.

**Tabular VI** maximises $\gamma$-discounted *expected* return under the same transitions as `environment.py`; per-episode returns still vary because failures are sampled in simulation.

## Main comparison

| Metric | Random | Heuristic | PPO | VI (expected MDP) |
|---|---|---|---|---|
| Reward | -241.44 ± 92.66  [-259.60, -223.28] | 58.06 ± 3.93  [57.29, 58.84] | 53.14 ± 4.68  [52.22, 54.06] | 58.08 ± 3.96  [57.31, 58.86] |
| Uncovered steps | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] |
| Uncovered failures | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] | 0.00 ± 0.00  [0.00, 0.00] |

## Welch's t-test (PPO vs Heuristic, reward)

- t = -8.057
- p = 0.0000
- df ≈ 192.4

## Welch's t-test (PPO vs Tabular VI, reward)

- t = -8.066
- p = 0.0000
- df ≈ 192.8

## Heuristic sensitivity sweep (renew threshold)

| Threshold | n | Mean | SD | 95% CI |
|---|---|---|---|---|
| 3 | 100 | 58.06 | 3.93 | [57.29, 58.84] |
| 5 | 100 | 58.06 | 3.93 | [57.29, 58.84] |
| 7 | 100 | 50.03 | 3.99 | [49.25, 50.81] |
| 10 | 100 | 49.87 | 4.10 | [49.07, 50.68] |
| 15 | 100 | 49.14 | 4.62 | [48.24, 50.04] |