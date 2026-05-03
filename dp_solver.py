"""Exact tabular value iteration for expected-return Bellman optimal policy.

The simulator samples Bernoulli failures each step; this solver maximises the
_policy_ that is optimal under the *expected* one-step reward (failure treated
as its expectation), which matches the long-run average return PPO is trained
to maximise (gamma-discounted with the same gamma as config.GAMMA).

State: (covered, days, contract_cost_level, risk_level, step_index)
  - covered in {0, 1}; if uncovered, days is always 0.
  - if covered, days in 1..MAX_COVERED_DAYS
  - step_index in 0..MAX_STEPS-1 is env.step_count at decision time.
"""

from __future__ import annotations

import config


def _expected_step(covered: int, days: int, cl: int, rl: int, action: int):
    """Return next (covered, days) and expected single-step reward (cl, rl fixed)."""
    r_cost = 0.0
    r_timing = 0.0
    c = covered
    d = days

    if action == 1:
        cost = config.RENEWAL_BASE_COST * (1 + cl)
        r_cost = -cost
        days_left = d if c == 1 else 0
        if days_left >= config.RENEWAL_EARLY_THRESHOLD:
            r_timing = -config.RENEWAL_EARLY_PENALTY
        elif days_left <= config.RENEWAL_CLOSE_THRESHOLD:
            r_timing = config.RENEWAL_CLOSE_BONUS
        d = config.RENEWAL_DAYS
        c = 1

    elif action == 2:
        if c == 1:
            cost = (
                config.RENEWAL_BASE_COST
                * config.EXTENSION_COST_FACTOR
                * (1 + cl)
            )
            r_cost = -cost
            if d <= config.RENEWAL_CLOSE_THRESHOLD:
                r_timing = config.RENEWAL_CLOSE_BONUS
            elif d >= config.RENEWAL_EARLY_THRESHOLD:
                r_timing = -config.RENEWAL_EARLY_PENALTY
            d = min(d + config.EXTENSION_DAYS, config.MAX_COVERED_DAYS)
        else:
            r_cost = -0.5

    just_expired = False
    if c == 1:
        d -= 1
        if d <= 0:
            c = 0
            d = 0
            just_expired = True

    p_fail = config.BASE_FAILURE_PROB * (1 + rl)
    r_cov = config.COVERAGE_BONUS if c == 1 else -config.UNCOVERED_PENALTY
    r_exp = -config.EXPIRATION_PENALTY if just_expired else 0.0
    r_fail_exp = -p_fail * config.FAILURE_PENALTY if c == 0 else 0.0

    r = r_cost + r_timing + r_cov + r_exp + r_fail_exp
    return (c, d), float(r)


def _iter_states():
    for s in range(config.MAX_STEPS):
        for cl in range(3):
            for rl in range(3):
                yield (0, 0, cl, rl, s)
                for d in range(1, config.MAX_COVERED_DAYS + 1):
                    yield (1, d, cl, rl, s)


def compute_optimal_policy(gamma=None, tol=1e-5, max_iterations=8000):
    """Run synchronous value iteration; return (V dict, policy dict)."""
    if gamma is None:
        gamma = config.GAMMA

    states = tuple(_iter_states())
    V = {st: 0.0 for st in states}
    pol = {st: 0 for st in states}

    for it in range(max_iterations):
        delta = 0.0
        V_new = {}
        pol_new = {}
        for st in states:
            c, d, cl, rl, s = st
            if s >= config.MAX_STEPS:
                V_new[st] = 0.0
                pol_new[st] = 0
                continue
            best_q = -1e30
            best_a = 0
            for a in range(3):
                (c2, d2), r = _expected_step(c, d, cl, rl, a)
                sp = s + 1
                if sp >= config.MAX_STEPS:
                    q = r
                else:
                    nxt = (c2, d2, cl, rl, sp)
                    q = r + gamma * V[nxt]
                if q > best_q:
                    best_q = q
                    best_a = a
            V_new[st] = best_q
            pol_new[st] = best_a
            delta = max(delta, abs(best_q - V[st]))
        V = V_new
        pol = pol_new
        if delta < tol:
            break
    else:
        raise RuntimeError("value iteration did not converge")

    return V, pol


def action_for_state(policy: dict, state_tuple) -> int:
    return int(policy[state_tuple])
