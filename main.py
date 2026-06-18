"""
AI-IDS – Main pipeline entrypoint.

Usage:
    python main.py --dataset nslkdd --mode baseline
    python main.py --dataset nslkdd --mode deep
    python main.py --dataset nslkdd --mode deep --fast      # quick CPU run
    python main.py --dataset cicids  --mode baseline
    python main.py --dataset nslkdd --mode forecast

Flags:
    --dataset   nslkdd | cicids                      (default: nslkdd)
    --mode      baseline | deep | all | forecast     (default: all)
    --save      save trained models to disk
    --fast      DL quick-run: 5 epochs, batch 1024, 20k training rows
    --sample N  subsample N training rows (useful for quick experiments)
"""

import argparse
import os
import numpy as np

from src.utils.helpers import setup_logger, set_seed, Timer
from src.utils.config import (
    NSLKDD_TRAIN_PATH, NSLKDD_TEST_PATH, CICIDS_PATH,
    TARGET_COL, MODELS_DIR, RANDOM_STATE,
    DL_EPOCHS, DL_BATCH_SIZE, DL_LEARNING_RATE,
    DL_FAST_EPOCHS, DL_FAST_BATCH_SIZE, DL_FAST_SAMPLE,
)
from src.data.preprocess import (
    load_dataset, encode_nslkdd_labels, encode_cicids_labels,
    preprocess, get_train_test_split,
)
from src.models.baseline import (
    train_random_forest, train_xgboost, train_svm,
    save_model,
)
from src.models.deep_ids import (
    MLPClassifier, CNN1DClassifier, LSTMClassifier, TransformerClassifier,
    save_dl_model,
)
from src.evaluation.metrics import benchmark_models, plot_confusion_matrix, plot_roc_curve

logger = setup_logger("AI-IDS.main")


def load_and_preprocess(dataset: str):
    """Return (X_train, X_test, y_train, y_test, input_dim)."""
    if dataset == "nslkdd":
        logger.info("Loading NSL-KDD dataset…")
        train_df = load_dataset(NSLKDD_TRAIN_PATH, header=False)
        test_df  = load_dataset(NSLKDD_TEST_PATH,  header=False)
        train_df = encode_nslkdd_labels(train_df)
        test_df  = encode_nslkdd_labels(test_df)

        scaler_path = os.path.join(MODELS_DIR, "nslkdd_scaler.pkl")
        X_train, y_train, scaler = preprocess(train_df, fit_scaler=True, scaler_path=scaler_path)
        X_test,  y_test,  _      = preprocess(test_df,  fit_scaler=False, scaler_path=scaler_path)

    elif dataset == "cicids":
        logger.info("Loading CICIDS 2017 dataset…")
        df = load_dataset(CICIDS_PATH, header=True)
        df = encode_cicids_labels(df)

        scaler_path = os.path.join(MODELS_DIR, "cicids_scaler.pkl")
        X, y, _ = preprocess(df, fit_scaler=True, scaler_path=scaler_path)
        X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'nslkdd' or 'cicids'.")

    input_dim = X_train.shape[1]
    logger.info("Train: %s | Test: %s | Features: %d", X_train.shape, X_test.shape, input_dim)
    return X_train, X_test, y_train, y_test, input_dim


def run_baseline(X_train, X_test, y_train, y_test, save: bool = False):
    """Train and evaluate all baseline ML models."""
    logger.info("═══ BASELINE ML MODELS ═══")
    models = {}

    with Timer("RandomForest"):
        rf = train_random_forest(X_train, y_train)
        models["RandomForest"] = rf
        if save:
            save_model(rf, "random_forest")

    with Timer("XGBoost"):
        xgb = train_xgboost(X_train, y_train)
        models["XGBoost"] = xgb
        if save:
            save_model(xgb, "xgboost")

    # SVM is slow on >50k rows – subsample to 20k for practicality
    max_svm = 20_000
    X_svm = X_train[:max_svm]
    y_svm = y_train[:max_svm]
    with Timer("SVM"):
        svm = train_svm(X_svm, y_svm)
        models["SVM"] = svm
        if save:
            save_model(svm, "svm")

    results = benchmark_models(models, X_test, y_test)

    # Save confusion matrices and ROC curves
    os.makedirs("outputs", exist_ok=True)
    for name, model in models.items():
        preds = model.predict(X_test)
        plot_confusion_matrix(y_test, preds, name, f"outputs/cm_{name}.png")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            plot_roc_curve(y_test, probs, name, f"outputs/roc_{name}.png")

    return results


def run_deep(X_train, X_test, y_train, y_test, input_dim: int,
             save: bool = False, fast: bool = False, sample: int | None = None):
    """Train and evaluate deep learning models."""
    logger.info("═══ DEEP LEARNING MODELS ═══")

    # ── Resolve training settings ──────────────────────────────────────────
    if fast:
        epochs     = DL_FAST_EPOCHS
        batch_size = DL_FAST_BATCH_SIZE
        n_sample   = DL_FAST_SAMPLE
        logger.info("Fast mode: epochs=%d, batch=%d, sample=%d", epochs, batch_size, n_sample)
    else:
        epochs     = DL_EPOCHS
        batch_size = DL_BATCH_SIZE
        n_sample   = sample  # None means use all

    # ── Optional subsampling ───────────────────────────────────────────────
    if n_sample and n_sample < len(X_train):
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), size=n_sample, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
        logger.info("Subsampled training set: %d → %d rows", len(X_train), n_sample)
    else:
        X_tr, y_tr = X_train, y_train

    models = {}
    num_classes = len(np.unique(y_tr))

    for ModelClass, tag in [
        (MLPClassifier,         "MLP"),
        (CNN1DClassifier,       "CNN1D"),
        (LSTMClassifier,        "LSTM"),
        (TransformerClassifier, "Transformer"),
    ]:
        with Timer(tag):
            m = ModelClass(input_dim=input_dim, num_classes=num_classes)
            m.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                  lr=DL_LEARNING_RATE)
            models[tag] = m
            if save:
                save_dl_model(m, tag.lower())

    results = benchmark_models(models, X_test, y_test)

    os.makedirs("outputs", exist_ok=True)
    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        plot_confusion_matrix(y_test, preds, name, f"outputs/cm_{name}.png")
        plot_roc_curve(y_test, probs, name, f"outputs/roc_{name}.png")

    return results


def run_forecast(dataset: str):
    """Demo: aggregate daily attack counts and forecast the next week."""
    from src.models.forecasting import LSTMForecaster
    logger.info("═══ ATTACK FORECASTING ═══")

    if dataset == "nslkdd":
        df = load_dataset(NSLKDD_TRAIN_PATH, header=False)
        df = encode_nslkdd_labels(df)
    else:
        df = load_dataset(CICIDS_PATH, header=True)
        df = encode_cicids_labels(df)

    # Simulate a daily time series: count attacks per 500-sample window
    attacks = df[TARGET_COL].values
    window = 500
    series = np.array([attacks[i:i+window].sum() for i in range(0, len(attacks)-window, window)],
                      dtype=np.float32)
    logger.info("Time series length: %d windows", len(series))

    fc = LSTMForecaster(lookback=10, forecast_steps=5)
    fc.fit(series, epochs=50)
    forecast = fc.predict()
    logger.info("Forecast (next 5 windows): %s", np.round(forecast, 1))
    return forecast


def main():
    parser = argparse.ArgumentParser(description="AI-IDS pipeline")
    parser.add_argument("--dataset", default="nslkdd", choices=["nslkdd", "cicids"])
    parser.add_argument("--mode", default="all",
                        choices=["baseline", "deep", "all", "forecast"])
    parser.add_argument("--save",   action="store_true", help="Save trained models")
    parser.add_argument("--fast",   action="store_true",
                        help="Quick DL run: 5 epochs, large batch, 20k rows")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N training rows (e.g. --sample 30000)")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    logger.info("Dataset: %s | Mode: %s | Save: %s | Fast: %s | Sample: %s",
                args.dataset, args.mode, args.save, args.fast, args.sample)

    if args.mode == "forecast":
        run_forecast(args.dataset)
        return

    X_train, X_test, y_train, y_test, input_dim = load_and_preprocess(args.dataset)

    if args.mode in ("baseline", "all"):
        run_baseline(X_train, X_test, y_train, y_test, save=args.save)

    if args.mode in ("deep", "all"):
        run_deep(X_train, X_test, y_train, y_test, input_dim,
                 save=args.save, fast=args.fast, sample=args.sample)


if __name__ == "__main__":
    main()
