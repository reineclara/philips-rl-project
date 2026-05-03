"""Simulated single-equipment contract-management MDP.

State (4-dimensional, all values fed to the policy are in [0, 1]):
    covered              in {0, 1}
    days_to_expiration   in [0, MAX_COVERED_DAYS] -> /MAX_COVERED_DAYS
    contract_cost_level  in {0, 1, 2}             -> /2
    risk_level           in {0, 1, 2}             -> /2

Actions (discrete, 3):
    0 = do nothing
    1 = renew contract    (sets days_to_expiration to RENEWAL_DAYS)
    2 = extend warranty   (adds EXTENSION_DAYS, requires active coverage)

Reward (balanced, shaped):
    + COVERAGE_BONUS                       while covered
    + RENEWAL_CLOSE_BONUS                  when renewing close to expiration
    - UNCOVERED_PENALTY                    while uncovered (strong)
    - EXPIRATION_PENALTY                   one-shot at the step the contract expires
    - RENEWAL_EARLY_PENALTY                when renewing way too early
    - renewal/extension cost               scaled by contract_cost_level
    - FAILURE_PENALTY                      on a failure that happens while uncovered
"""

import numpy as np

import config


class ContractEnv:
    def __init__(self):
        self.obs_dim = 4
        self.action_dim = 3
        self.rng = np.random.default_rng(config.SEED)

    def reset(self, seed=None, failure_scale_mult=1.0):
        """failure_scale_mult>1 stresses higher failure prevalence (distribution shift)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.failure_scale_mult = float(failure_scale_mult)
        self.covered = 1
        self.days_to_expiration = int(
            self.rng.integers(config.MIN_INITIAL_DAYS, config.MAX_INITIAL_DAYS + 1)
        )
        self.contract_cost_level = int(self.rng.integers(0, 3))  # 0..2
        self.risk_level = int(self.rng.integers(0, 3))           # 0..2
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array(
            [
                float(self.covered),
                self.days_to_expiration / config.MAX_COVERED_DAYS,
                self.contract_cost_level / 2.0,
                self.risk_level / 2.0,
            ],
            dtype=np.float32,
        )

    def internal_tuple(self):
        """Discrete MDP coordinate used by dp_solver (state at decision time)."""
        return (
            int(self.covered),
            int(self.days_to_expiration),
            int(self.contract_cost_level),
            int(self.risk_level),
            int(self.step_count),
        )

    def step(self, action):
        r_cost = 0.0
        r_timing = 0.0

        if action == 1:  # renew contract
            cost = config.RENEWAL_BASE_COST * (1 + self.contract_cost_level)
            r_cost = -cost
            days_left = self.days_to_expiration if self.covered else 0
            if days_left >= config.RENEWAL_EARLY_THRESHOLD:
                r_timing = -config.RENEWAL_EARLY_PENALTY
            elif days_left <= config.RENEWAL_CLOSE_THRESHOLD:
                r_timing = config.RENEWAL_CLOSE_BONUS
            self.days_to_expiration = config.RENEWAL_DAYS
            self.covered = 1

        elif action == 2:  # extend warranty (valid only while covered)
            if self.covered:
                cost = (
                    config.RENEWAL_BASE_COST
                    * config.EXTENSION_COST_FACTOR
                    * (1 + self.contract_cost_level)
                )
                r_cost = -cost
                if self.days_to_expiration <= config.RENEWAL_CLOSE_THRESHOLD:
                    r_timing = config.RENEWAL_CLOSE_BONUS
                elif self.days_to_expiration >= config.RENEWAL_EARLY_THRESHOLD:
                    r_timing = -config.RENEWAL_EARLY_PENALTY
                self.days_to_expiration = min(
                    self.days_to_expiration + config.EXTENSION_DAYS,
                    config.MAX_COVERED_DAYS,
                )
            else:
                # Extending an expired contract is invalid: small no-op penalty.
                r_cost = -0.5

        # --- Time passes ---
        just_expired = False
        if self.covered:
            self.days_to_expiration -= 1
            if self.days_to_expiration <= 0:
                self.covered = 0
                self.days_to_expiration = 0
                just_expired = True

        # --- Shaped components ---
        r_coverage = config.COVERAGE_BONUS if self.covered else -config.UNCOVERED_PENALTY
        r_expiry = -config.EXPIRATION_PENALTY if just_expired else 0.0

        fail_prob = (
            config.BASE_FAILURE_PROB
            * self.failure_scale_mult
            * (1 + self.risk_level)
        )
        failed = self.rng.random() < fail_prob
        uncovered_failure = failed and not self.covered
        r_failure = -config.FAILURE_PENALTY if uncovered_failure else 0.0

        reward = r_cost + r_timing + r_coverage + r_expiry + r_failure

        self.step_count += 1
        done = self.step_count >= config.MAX_STEPS

        info = {
            "uncovered": 0 if self.covered else 1,
            "uncovered_failure": 1 if uncovered_failure else 0,
            "just_expired": just_expired,
        }
        return self._get_obs(), float(reward), done, info
