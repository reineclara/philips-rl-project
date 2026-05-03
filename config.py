"""Hyperparameters for the contract-management RL project."""

# === Environment ===
MAX_STEPS = 120                 # episode length (days)
MIN_INITIAL_DAYS = 5
MAX_INITIAL_DAYS = 30
RENEWAL_DAYS = 30               # coverage added by a full renewal
EXTENSION_DAYS = 30             # coverage added by a warranty extension
MAX_COVERED_DAYS = 60           # cap on remaining-coverage days

# Reward coefficients (balanced, typical per-step range: [-5, +3])
COVERAGE_BONUS = 0.5            # per-step bonus while covered
UNCOVERED_PENALTY = 3.0         # per-step penalty while uncovered  (strong)
EXPIRATION_PENALTY = 5.0        # one-shot at the step a contract expires
RENEWAL_CLOSE_BONUS = 2.0       # renewing close to expiration
RENEWAL_EARLY_PENALTY = 1.0     # renewing too early
RENEWAL_CLOSE_THRESHOLD = 5     # "close to expiration" (<=)
RENEWAL_EARLY_THRESHOLD = 20    # "too early" (>=)
RENEWAL_BASE_COST = 1.5         # base renewal cost (scaled by cost_level)
EXTENSION_COST_FACTOR = 0.8     # extension is cheaper than full renewal
FAILURE_PENALTY = 5.0           # per uncovered failure
BASE_FAILURE_PROB = 0.02        # scaled by (1 + risk_level)
FAILURE_SCALE_OOD = 1.45        # distribution-shift stress for Section OOD experiments

# Two-equipment portfolio toy (daily shared budget renewed every step)
TWO_DAILY_MAX_SPEND = 2.15      # cash < 2 × min(renew heavy) ⇒ portfolio friction
BUDGET_VIOLATION_PENALTY = 1.35 # when a costly action cannot be funded today

# === PPO ===
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
EPOCHS = 4
BATCH_SIZE = 64
HIDDEN_DIM = 64
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5
OBS_CLIP = 5.0                  # clip normalized obs to [-OBS_CLIP, +OBS_CLIP]

# === Training ===
NUM_EPISODES = 5000             # longer budget (reviewer suggestion: extend beyond 3k)
TWO_AGENT_TRAIN_EPISODES = 1800 # portfolio PPO budget in benchmarks_extended.py
LOG_EVERY = 100
EVAL_INTERVAL = 100             # evaluate every N episodes
EVAL_EPISODES = 20              # used during training (cheap periodic check)
EVAL_EPISODES_FINAL = 100       # used once, at the end, for rigorous comparison
SEED = 42

# === Heuristic sensitivity analysis ===
# Thresholds tested for the heuristic "renew when uncovered, extend when
# days_to_expiration <= threshold". A sweep across these values quantifies
# the sensitivity of the heuristic baseline to its only tunable parameter.
HEURISTIC_THRESHOLDS = [3, 5, 7, 10, 15]

# === I/O ===
BEST_MODEL_PATH = "best_model.pt"
TRAINING_CURVE_PATH = "training_curve.png"
RESULTS_JSON_PATH = "results.json"
RESULTS_MD_PATH = "results.md"

# Tabular SARSA (extended benchmark)
TABULAR_SARSA_EPISODES = 3500

# Ablation quick runs (entropy grid); see ablation_worker.py
ABLATION_TRAIN_EPISODES = 900

# Extended benchmark bundle I/O
EXTENDED_BENCHMARK_JSON = "extended_benchmarks.json"
EXTENDED_BENCHMARK_MD = "extended_benchmarks.md"
