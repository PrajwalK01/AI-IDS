"""
Unit tests for deep learning IDS models.
Uses tiny data so tests run fast without a GPU.
"""

import numpy as np
import pytest

from src.models.deep_ids import (
    MLPClassifier, CNN1DClassifier, LSTMClassifier, TransformerClassifier,
)
from src.models.forecasting import LSTMForecaster


INPUT_DIM = 12
N_TRAIN = 80
N_TEST = 20
NUM_CLASSES = 2

rng = np.random.default_rng(7)
X_train = rng.standard_normal((N_TRAIN, INPUT_DIM)).astype(np.float32)
y_train = rng.integers(0, NUM_CLASSES, size=N_TRAIN)
X_test  = rng.standard_normal((N_TEST,  INPUT_DIM)).astype(np.float32)


@pytest.mark.parametrize("ModelClass,tag", [
    (MLPClassifier,         "MLP"),
    (CNN1DClassifier,       "CNN1D"),
    (LSTMClassifier,        "LSTM"),
    (TransformerClassifier, "Transformer"),
])
def test_dl_model_predict_shape(ModelClass, tag):
    model = ModelClass(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    model.fit(X_train, y_train, epochs=2, batch_size=16)
    preds = model.predict(X_test)
    assert preds.shape == (N_TEST,), f"{tag}: unexpected prediction shape"
    assert set(preds).issubset(set(range(NUM_CLASSES))), f"{tag}: unexpected class labels"


@pytest.mark.parametrize("ModelClass,tag", [
    (MLPClassifier,         "MLP"),
    (CNN1DClassifier,       "CNN1D"),
    (LSTMClassifier,        "LSTM"),
    (TransformerClassifier, "Transformer"),
])
def test_dl_model_proba_shape(ModelClass, tag):
    model = ModelClass(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    model.fit(X_train, y_train, epochs=2, batch_size=16)
    probs = model.predict_proba(X_test)
    assert probs.shape == (N_TEST, NUM_CLASSES), f"{tag}: unexpected proba shape"
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_lstm_forecaster():
    series = np.sin(np.linspace(0, 4 * np.pi, 60)).astype(np.float32) + 2
    fc = LSTMForecaster(lookback=10, forecast_steps=5)
    fc.fit(series, epochs=5, batch_size=8)
    forecast = fc.predict()
    assert forecast.shape == (5,)
    assert not np.any(np.isnan(forecast))
