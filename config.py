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
NUM_EPISODES = 3000
LOG_EVERY = 100
EVAL_INTERVAL = 100             # evaluate every N episodes
EVAL_EPISODES = 20
SEED = 42

# === I/O ===
BEST_MODEL_PATH = "best_model.pt"
TRAINING_CURVE_PATH = "training_curve.png"
