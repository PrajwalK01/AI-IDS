import pandas as pd
from src.models.baseline import train_baseline

def test_baseline_model():
    # Dummy dataset for testing
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "label": [0, 1, 0, 1, 0]
    })

    model = train_baseline(df, target_col="label")
    assert model is not None
