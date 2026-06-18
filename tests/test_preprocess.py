"""
Unit tests for data preprocessing.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess, encode_nslkdd_labels, encode_cicids_labels


@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking preprocessed IDS data."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, 8)),
        columns=[f"feat_{i}" for i in range(8)],
    )
    df["label"] = rng.integers(0, 2, size=100)
    return df


def test_preprocess_shapes(sample_df):
    X, y, scaler = preprocess(sample_df, target_col="label", fit_scaler=True)
    assert X.shape == (100, 8)
    assert y.shape == (100,)


def test_preprocess_scaler_is_standard(sample_df):
    X, _, _ = preprocess(sample_df, target_col="label", fit_scaler=True)
    # After StandardScaler, mean ≈ 0 and std ≈ 1
    assert abs(X.mean()) < 0.1
    assert abs(X.std() - 1.0) < 0.2


def test_encode_nslkdd_labels():
    df = pd.DataFrame({"label": ["normal", "DoS", "normal", "Probe"], "difficulty": [1, 2, 1, 3]})
    out = encode_nslkdd_labels(df)
    assert "difficulty" not in out.columns
    assert list(out["label"]) == [0, 1, 0, 1]


def test_encode_cicids_labels():
    df = pd.DataFrame({"Label": ["BENIGN", "DoS Hulk", "BENIGN", "PortScan"]})
    out = encode_cicids_labels(df)
    assert list(out["label"]) == [0, 1, 0, 1]


def test_preprocess_no_inf(sample_df):
    """Verify that inf values are replaced before scaling."""
    sample_df["feat_0"] = np.inf
    X, _, _ = preprocess(sample_df, fit_scaler=True)
    assert not np.any(np.isinf(X))
