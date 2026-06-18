"""
Unit tests for baseline ML models.
Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import train_random_forest, train_xgboost
from src.evaluation.metrics import evaluate_model, risk_score


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def small_dataset():
    """50-sample binary classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=50)
    split = 40
    return X[:split], X[split:], y[:split], y[split:]


# ── Baseline ML ────────────────────────────────────────────────────────────────

def test_random_forest(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    model = train_random_forest(X_train, y_train, n_estimators=10)
    assert model is not None
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(preds).issubset({0, 1})


def test_xgboost(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    model = train_xgboost(X_train, y_train, n_estimators=10)
    assert model is not None
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)


# ── Evaluation metrics ─────────────────────────────────────────────────────────

def test_evaluate_model():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    metrics = evaluate_model(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_model_with_proba():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    metrics = evaluate_model(y_true, y_pred, y_prob)
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] == 1.0


def test_risk_score():
    # Thresholds: High >= 0.8, Medium >= 0.5, else Low
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.4, 0.6], [0.7, 0.3]])
    # attack prob: 0.1 → Low, 0.9 → High, 0.6 → Medium, 0.3 → Low
    labels = risk_score(probs)
    assert labels[0] == "Low"
    assert labels[1] == "High"
    assert labels[2] == "Medium"
    assert labels[3] == "Low"
