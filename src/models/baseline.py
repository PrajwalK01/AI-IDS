"""
Baseline ML models: RandomForest, XGBoost, SVM.

Each function follows the same interface:
  train_<model>(X_train, y_train, **kwargs) → fitted model
  predict(model, X) → np.ndarray of predictions
"""

import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.utils.helpers import setup_logger
from src.utils.config import (
    RF_N_ESTIMATORS, RF_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    SVM_C, SVM_KERNEL,
    RANDOM_STATE, MODELS_DIR,
)

logger = setup_logger(__name__)


# ── RandomForest ───────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = RF_N_ESTIMATORS,
    max_depth: int | None = RF_MAX_DEPTH,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train a RandomForest classifier."""
    logger.info("Training RandomForest (n_estimators=%d)…", n_estimators)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("RandomForest training complete.")
    return model


# ── XGBoost ────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = XGB_N_ESTIMATORS,
    max_depth: int = XGB_MAX_DEPTH,
    learning_rate: float = XGB_LEARNING_RATE,
    random_state: int = RANDOM_STATE,
) -> XGBClassifier:
    """Train an XGBoost classifier."""
    logger.info("Training XGBoost (n_estimators=%d, lr=%.3f)…", n_estimators, learning_rate)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return model


# ── SVM ───────────────────────────────────────────────────────────────────────

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = SVM_C,
    kernel: str = SVM_KERNEL,
) -> SVC:
    """
    Train an SVM classifier.
    Note: SVM is slow on large datasets – subsample if needed.
    """
    logger.info("Training SVM (C=%.2f, kernel=%s)…", C, kernel)
    model = SVC(C=C, kernel=kernel, probability=True)
    model.fit(X_train, y_train)
    logger.info("SVM training complete.")
    return model


# ── Generic helpers ────────────────────────────────────────────────────────────

def save_model(model, name: str) -> str:
    """Save a sklearn-compatible model to MODELS_DIR/<name>.pkl"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str):
    """Load a saved model from MODELS_DIR/<name>.pkl"""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at {path}")
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model
