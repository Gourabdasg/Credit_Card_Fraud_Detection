"""Central configuration for the Credit Card Fraud and Alert System (Review-3 state)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

SEED = 42

# --- Synthetic proxy dataset (mirrors class imbalance of the Kaggle data) ---
N_SAMPLES = 6000
N_FRAUD = 147          # ~2.45% fraud, gives 103 fraud in train / 22 in val / 22 in test
N_PCA_FEATURES = 28    # V1..V28, as in the Kaggle dataset

# --- Split (stratified 70:15:15) ---
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# --- Pseudo-sequence shaping: 29 numeric features -> pad to 30 -> 5 steps x 6 features ---
N_TIMESTEPS = 5
N_STEP_FEATURES = 6

# --- Deep model training ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 40
EARLY_STOP_PATIENCE = 5
LSTM_UNITS = 32
CNN_FILTERS = 32
CNN_KERNEL = 3
DENSE_UNITS = 16
DROPOUT = 0.3

# =====================================================================
# Review-4/5/6: real Kaggle data, alert module, final evaluation
# =====================================================================
DATA_DIR = ROOT / "data"
KAGGLE_CSV = DATA_DIR / "creditcard.csv"
MODEL_DIR = ROOT / "models"
FULL_OUTPUT_DIR = OUTPUT_DIR / "full"
ALERT_LOG = FULL_OUTPUT_DIR / "alerts.jsonl"

SEQ_WINDOW = 5                 # consecutive transactions per sequence (ordered by Time)
SPLIT_MODE = "stratified"      # "stratified" (as in the reports) or "chronological"
SMOTE_STRATEGY = 1.0           # minority/majority ratio after SMOTE (1.0 = fully balanced)
KAGGLE_BATCH_SIZE = 256        # larger batch for the 200k+ row training set

# Kaggle has no narration text: synthetic narration is generated (clearly flagged in reports).
SYNTH_TEXT_FRAUD_RATE = 0.70       # share of fraud rows that get fraud-style text
SYNTH_TEXT_GENUINE_FP_RATE = 0.02  # share of genuine rows that get fraud-style text (noise)
