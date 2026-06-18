"""
Attack forecasting module.

Provides two approaches:
  1. LSTMForecaster  – PyTorch LSTM that predicts future attack counts
                       from a sliding window of historical counts.
  2. ProphetForecaster – Facebook Prophet wrapper for daily attack-rate trends.

Usage example (LSTM):
    from src.models.forecasting import LSTMForecaster
    series = np.array([...])   # 1-D array of daily attack counts
    fc = LSTMForecaster(lookback=10, forecast_steps=5)
    fc.fit(series, epochs=50)
    future = fc.predict()      # array of next 5 predicted counts
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.helpers import setup_logger
from src.utils.config import (
    LSTM_LOOKBACK, LSTM_FORECAST_STEPS,
    DL_EPOCHS, DL_BATCH_SIZE, DL_LEARNING_RATE,
    RANDOM_STATE,
)

logger = setup_logger(__name__)
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LSTM Forecaster ────────────────────────────────────────────────────────────

class _ForecastLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, forecast_steps: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_steps)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class LSTMForecaster:
    """
    Predict the next `forecast_steps` attack counts from a 1-D time series.

    The series is first normalised (min-max), then split into sliding windows
    of length `lookback`. After training, call predict() to get future values.
    """

    def __init__(self, lookback: int = LSTM_LOOKBACK,
                 forecast_steps: int = LSTM_FORECAST_STEPS,
                 hidden_size: int = 64, num_layers: int = 2):
        self.lookback = lookback
        self.forecast_steps = forecast_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model: _ForecastLSTM | None = None
        self._min: float = 0.0
        self._max: float = 1.0
        self._last_window: np.ndarray | None = None

    def _normalise(self, series: np.ndarray) -> np.ndarray:
        self._min = float(series.min())
        self._max = float(series.max()) or 1.0
        return (series - self._min) / (self._max - self._min)

    def _denormalise(self, values: np.ndarray) -> np.ndarray:
        return values * (self._max - self._min) + self._min

    def _make_windows(self, series: np.ndarray):
        X, y = [], []
        for i in range(len(series) - self.lookback - self.forecast_steps + 1):
            X.append(series[i: i + self.lookback])
            y.append(series[i + self.lookback: i + self.lookback + self.forecast_steps])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, series: np.ndarray,
            epochs: int = DL_EPOCHS,
            batch_size: int = DL_BATCH_SIZE,
            lr: float = DL_LEARNING_RATE) -> "LSTMForecaster":
        """
        Args:
            series: 1-D array of attack counts per time step.
        """
        norm = self._normalise(series)
        self._last_window = norm[-self.lookback:]

        X, y = self._make_windows(norm)
        if len(X) == 0:
            raise ValueError(
                f"Series too short: need at least {self.lookback + self.forecast_steps} points."
            )

        X_t = torch.tensor(X).unsqueeze(-1)   # (N, lookback, 1)
        y_t = torch.tensor(y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        self.model = _ForecastLSTM(1, self.hidden_size, self.num_layers, self.forecast_steps)
        self.model.to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        logger.info("Training LSTMForecaster on %d windows…", len(X))
        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if epoch % max(1, epochs // 5) == 0 or epoch == 1:
                logger.info("  Epoch %d/%d – MSE loss: %.6f", epoch, epochs, total / len(loader))
        return self

    def predict(self) -> np.ndarray:
        """Return the next `forecast_steps` predicted attack counts."""
        if self.model is None or self._last_window is None:
            raise RuntimeError("Call fit() before predict().")
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(self._last_window, dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(-1).to(DEVICE)   # (1, lookback, 1)
            out = self.model(x).cpu().numpy()[0]
        return self._denormalise(out)


# ── Prophet Forecaster ─────────────────────────────────────────────────────────

class ProphetForecaster:
    """
    Thin wrapper around Facebook Prophet for attack-rate time series forecasting.

    Expects a pandas DataFrame with columns ['ds', 'y'] where:
      ds = datetime (e.g., daily date)
      y  = number of attacks that day
    """

    def __init__(self, forecast_days: int = 30, **prophet_kwargs):
        self.forecast_days = forecast_days
        self._kwargs = prophet_kwargs
        self.model = None
        self._future = None

    def fit(self, df):
        """
        Args:
            df: pd.DataFrame with columns 'ds' (datetime) and 'y' (attack count).
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Install prophet: pip install prophet")

        logger.info("Fitting Prophet on %d observations…", len(df))
        self.model = Prophet(**self._kwargs)
        self.model.fit(df)
        return self

    def predict(self):
        """
        Returns:
            pd.DataFrame with forecast including 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        self._future = self.model.make_future_dataframe(periods=self.forecast_days)
        forecast = self.model.predict(self._future)
        logger.info("Prophet forecast generated for %d days ahead.", self.forecast_days)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def plot(self):
        """Plot the forecast (requires matplotlib)."""
        if self.model is None:
            raise RuntimeError("Call fit() and predict() first.")
        fig = self.model.plot(self.model.predict(self._future))
        return fig
