import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    Args:
        path (str): Path to dataset file.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.fillna(0)
    return df
