"""
Data loading and preprocessing for NSL-KDD and CICIDS datasets.

Pipeline:
  1. Load CSV
  2. Drop duplicates / fill NaN
  3. Encode categorical columns (LabelEncoder)
  4. Encode target label (binary: normal=0, attack=1)
  5. Normalise numeric features (StandardScaler)
  6. Return feature matrix X and label vector y
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from src.utils.helpers import setup_logger
from src.utils.config import TARGET_COL, TEST_SIZE, RANDOM_STATE, MODELS_DIR

logger = setup_logger(__name__)


# ── NSL-KDD column names (the raw file has no header) ─────────────────────────
NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

NSLKDD_CATEGORICAL = ["protocol_type", "service", "flag"]


def load_dataset(path: str, header: bool = True) -> pd.DataFrame:
    """
    Load a dataset from CSV.

    Args:
        path: Path to the CSV file.
        header: True if the file already has a header row.

    Returns:
        Cleaned DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    if header:
        df = pd.read_csv(path)
    else:
        # NSL-KDD raw format has no header
        df = pd.read_csv(path, header=None, names=NSLKDD_COLUMNS)

    before = len(df)
    df = df.drop_duplicates()
    df = df.fillna(0)
    logger.info("Loaded %s rows from %s (dropped %d duplicates)", len(df), path, before - len(df))
    return df


def encode_nslkdd_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert NSL-KDD multi-class labels to binary: 'normal' → 0, anything else → 1.
    Also drops the 'difficulty' column if present.
    """
    df = df.copy()
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])
    df[TARGET_COL] = (df[TARGET_COL] != "normal").astype(int)
    return df


def encode_cicids_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert CICIDS labels to binary: 'BENIGN' → 0, anything else → 1.
    Also strips whitespace from column names (CICIDS has leading spaces).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    if TARGET_COL not in df.columns:
        # CICIDS uses ' Label' or 'Label'
        possible = [c for c in df.columns if "label" in c.lower()]
        if possible:
            df = df.rename(columns={possible[0]: TARGET_COL})
        else:
            raise ValueError("Could not find a label column in CICIDS data.")
    df[TARGET_COL] = (df[TARGET_COL] != "BENIGN").astype(int)
    return df


def preprocess(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    scaler_path: str | None = None,
    fit_scaler: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Encode categoricals, scale numerics, and split into X / y.

    Args:
        df:          Cleaned DataFrame with a target column.
        target_col:  Name of the label column.
        scaler_path: If provided, save/load the scaler from this path.
        fit_scaler:  If True, fit a new scaler; if False, load from scaler_path.

    Returns:
        X (np.ndarray), y (np.ndarray), fitted StandardScaler
    """
    df = df.copy()

    # Encode any remaining object columns (e.g. categorical features)
    for col in df.select_dtypes(include=["object"]).columns:
        if col == target_col:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    y = df[target_col].values.astype(int)
    X_df = df.drop(columns=[target_col])

    # Keep only numeric columns
    X_df = X_df.select_dtypes(include=[np.number])

    # Replace inf values that can appear in CICIDS
    X_df = X_df.replace([np.inf, -np.inf], 0)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df)
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info("Scaler saved to %s", scaler_path)
    else:
        if not scaler_path or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded from %s", scaler_path)
        X = scaler.transform(X_df)

    logger.info("Feature matrix shape: %s | Positive (attack) rate: %.2f%%",
                X.shape, 100 * y.mean())
    return X, y, scaler


def get_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Thin wrapper around sklearn's train_test_split with project defaults."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=y)
