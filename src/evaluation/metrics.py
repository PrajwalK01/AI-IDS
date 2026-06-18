"""
Evaluation utilities for IDS models.

Includes:
  - evaluate_model()   – standard classification metrics dict
  - risk_score()       – per-sample severity label (Low / Medium / High)
  - plot_confusion_matrix()
  - plot_roc_curve()
  - benchmark_models() – compare multiple models in one call
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)

from src.utils.helpers import setup_logger
from src.utils.config import RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD

logger = setup_logger(__name__)


# ── Classification metrics ─────────────────────────────────────────────────────

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray | None = None) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for the positive class (used for AUC).

    Returns:
        dict with accuracy, precision, recall, f1_score, and optionally roc_auc.
    """
    results = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="weighted",
                                        zero_division=0), 4),
        "f1_score":  round(f1_score(y_true, y_pred, average="weighted",
                                    zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            results["roc_auc"] = round(roc_auc_score(y_true, prob_pos), 4)
        except ValueError as e:
            logger.warning("Could not compute ROC-AUC: %s", e)
    return results


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print a metrics dict."""
    header = f"── {model_name} ──"
    print(header)
    for k, v in metrics.items():
        print(f"  {k:<15} {v}")
    print("─" * len(header))


# ── Risk Scoring ───────────────────────────────────────────────────────────────

def risk_score(y_prob: np.ndarray) -> np.ndarray:
    """
    Assign a severity label to each sample based on attack probability.

    Args:
        y_prob: Array of shape (N,) or (N, 2) with attack probabilities.

    Returns:
        np.ndarray of strings: 'Low', 'Medium', or 'High'.
    """
    prob_attack = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    labels = np.where(
        prob_attack >= RISK_HIGH_THRESHOLD, "High",
        np.where(prob_attack >= RISK_MEDIUM_THRESHOLD, "Medium", "Low")
    )
    return labels


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "Model",
                          save_path: str | None = None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Normal", "Attack"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved → %s", save_path)
    plt.close(fig)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str = "Model",
                   save_path: str | None = None):
    """Plot and optionally save a ROC curve."""
    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, prob_pos, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve – {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("ROC curve saved → %s", save_path)
    plt.close(fig)
    return fig


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_models(
    models: dict,           # {"model_name": model_object, ...}
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate multiple models and return a comparison table.

    Args:
        models: Dict mapping model names to fitted model objects.
                Each model must have a predict() method.
                Models with predict_proba() will also get ROC-AUC.
        X_test: Test features.
        y_test: True labels.

    Returns:
        Dict of {model_name: metrics_dict}.
    """
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        probs = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(X_test)
            except Exception:
                pass
        metrics = evaluate_model(y_test, preds, probs)
        results[name] = metrics
        print_metrics(metrics, name)
    return results
