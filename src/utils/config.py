"""
Project-wide configuration constants.
Edit these values to match your dataset paths and training preferences.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
NSLKDD_TRAIN_PATH = "data/NSL-KDD-Train.csv"
NSLKDD_TEST_PATH = "data/NSL-KDD-Test.csv"
CICIDS_PATH = "data/CICIDS2017.csv"
MODELS_DIR = "saved_models"

# ── Preprocessing ──────────────────────────────────────────────────────────────
TARGET_COL = "label"          # column name for attack labels
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Baseline ML ────────────────────────────────────────────────────────────────
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None

XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

SVM_C = 1.0
SVM_KERNEL = "rbf"

# ── Deep Learning ──────────────────────────────────────────────────────────────
DL_EPOCHS = 30
DL_BATCH_SIZE = 256
DL_LEARNING_RATE = 1e-3
DL_HIDDEN_DIM = 128
DL_NUM_LAYERS = 2
DL_DROPOUT = 0.3

# ── LSTM Forecasting ───────────────────────────────────────────────────────────
LSTM_LOOKBACK = 10      # number of past timesteps to use as input
LSTM_FORECAST_STEPS = 5 # how many future steps to predict

# ── Risk Scoring thresholds ────────────────────────────────────────────────────
RISK_HIGH_THRESHOLD = 0.8
RISK_MEDIUM_THRESHOLD = 0.5
