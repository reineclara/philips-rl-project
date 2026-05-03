"""Two parallel equipments with a *daily shared budget* (portfolio toy).

Motivation from review feedback (ChatGPT-class): coupling breaks naive
per-asset thresholds.  Each timestep the purse is refilled to
``config.TWO_DAILY_MAX_SPEND``.  Actions incur costs (renew / extend exactly
like :class:`~environment.ContractEnv`); insufficient funds degrade the action to
do-nothing and apply ``config.BUDGET_VIOLATION_PENALTY``.

Observations concatenate two 4-dimensional vectors per ``ContractEnv``.
Actions are flattened ``a0,a1 ∈ {0,1,2}`` as ``idx = a0 * 3 + a1``.
"""

from __future__ import annotations

import numpy as np

import config


class TwoEquipDailyBudgetEnv:
    def __init__(self):
        self.obs_dim = 8
        self.action_dim = 9
        self.rngs = (
            np.random.default_rng(config.SEED),
            np.random.default_rng(config.SEED + 911),
        )
        self.failure_scale_mult = 1.0
        self.covered = np.ones(2, dtype=np.int32)
        self.days = np.zeros(2, dtype=np.int32)
        self.cl = np.zeros(2, dtype=np.int32)
        self.rl = np.zeros(2, dtype=np.int32)
        self.step_count = 0

    # --------------------------------------------------------------------- #
    def reset(self, seed=None, failure_scale_mult: float | None = None):
        if seed is not None:
            self.rngs = (
                np.random.default_rng(int(seed)),
                np.random.default_rng(int(seed) + 911),
            )
        self.failure_scale_mult = (
            float(failure_scale_mult)
            if failure_scale_mult is not None
            else 1.0
        )
        self.step_count = 0
        self.covered[:] = 1

        def draw(r):
            days = int(
                r.integers(
                    config.MIN_INITIAL_DAYS,
                    config.MAX_INITIAL_DAYS + 1,
                )
            )
            cl = int(r.integers(0, 3))
            rl_v = int(r.integers(0, 3))
            return days, cl, rl_v

        d0, c0, r0 = draw(self.rngs[0])
        d1, c1, r1 = draw(self.rngs[1])
        self.days[0], self.cl[0], self.rl[0] = d0, c0, r0
        self.days[1], self.cl[1], self.rl[1] = d1, c1, r1

        return self._get_obs()

    def _get_obs(self):
        out = []
        for i in range(2):
            out.extend(
                [
                    float(self.covered[i]),
                    float(self.days[i]) / config.MAX_COVERED_DAYS,
                    float(self.cl[i]) / 2.0,
                    float(self.rl[i]) / 2.0,
                ]
            )
        return np.array(out, dtype=np.float32)

    def step(self, joint_action: int):
        a0, a1 = divmod(int(joint_action), 3)
        wallet = float(config.TWO_DAILY_MAX_SPEND)

        tot_u = tot_f = 0
        tot_r = 0.0

        for equip in (0, 1):
            action = (a0, a1)[equip]
            r, wallet, eu, uf = self._sub_step_one(equip, action, wallet)
            tot_r += r
            tot_u += eu
            tot_f += uf

        self.step_count += 1
        done = self.step_count >= config.MAX_STEPS
        info = {"uncovered": tot_u, "uncovered_failure": tot_f}

        return self._get_obs(), float(tot_r), done, info

    def _sub_step_one(self, equip: int, action: int, wallet: float):
        """Return (reward_increment, wallet_after, uncovered_steps_here, uncovered_fail_here)."""

        rng = self.rngs[equip]
        c = int(self.covered[equip])
        d = int(self.days[equip])
        cl = int(self.cl[equip])
        rl = int(self.rl[equip])

        def upfront_cost(act: int) -> float:
            """Cash outflow for renewing / extending (aligns with single-env spending)."""
            if act == 0:
                return 0.0
            if act == 1:
                return float(config.RENEWAL_BASE_COST * (1 + cl))
            if act == 2 and c == 1:
                return float(
                    config.RENEWAL_BASE_COST
                    * config.EXTENSION_COST_FACTOR
                    * (1 + cl)
                )
            return 0.0

        eff_action = int(action)

        tentative = upfront_cost(action)
        if tentative > wallet + 1e-12 and action != 0:
            eff_action = 0
            reward = -float(config.BUDGET_VIOLATION_PENALTY)
        else:
            reward = 0.0

        spend = upfront_cost(eff_action)
        wallet_after = wallet - spend

        r_cost = 0.0
        r_timing = 0.0

        if eff_action == 1:
            r_cost = -spend
            days_left = d if c == 1 else 0
            if days_left >= config.RENEWAL_EARLY_THRESHOLD:
                r_timing = -config.RENEWAL_EARLY_PENALTY
            elif days_left <= config.RENEWAL_CLOSE_THRESHOLD:
                r_timing = config.RENEWAL_CLOSE_BONUS
            d = config.RENEWAL_DAYS
            c = 1

        elif eff_action == 2:
            if c == 1:
                r_cost = -spend
                if d <= config.RENEWAL_CLOSE_THRESHOLD:
                    r_timing = config.RENEWAL_CLOSE_BONUS
                elif d >= config.RENEWAL_EARLY_THRESHOLD:
                    r_timing = -config.RENEWAL_EARLY_PENALTY
                d = min(d + config.EXTENSION_DAYS, config.MAX_COVERED_DAYS)
            else:
                r_cost -= 0.5

        just_expired = False
        if c == 1:
            d -= 1
            if d <= 0:
                c = 0
                d = 0
                just_expired = True

        r_cov = config.COVERAGE_BONUS if c == 1 else -config.UNCOVERED_PENALTY
        r_expiry = -config.EXPIRATION_PENALTY if just_expired else 0.0

        fail_prob = (
            config.BASE_FAILURE_PROB
            * float(self.failure_scale_mult)
            * (1 + rl)
        )
        failed = rng.random() < fail_prob

        uf = 0
        if failed and not c:
            uf += 1
            r_failure = -config.FAILURE_PENALTY
        else:
            r_failure = 0.0

        reward += (
            float(r_cost) + float(r_timing) + float(r_cov)
            + float(r_expiry) + float(r_failure)
        )

        self.covered[equip], self.days[equip] = c, d

        uncovered_steps_here = int(1 - c)
        return reward, wallet_after, uncovered_steps_here, uf
