"""
Deep Learning IDS models built with PyTorch:
  - MLP           – simple fully-connected baseline
  - CNN1D         – 1-D convolutional network for tabular/sequence features
  - LSTMClassifier – recurrent network for sequential traffic data
  - TransformerIDS – self-attention based classifier

All models expose the same interface:
  model = <Class>(input_dim, num_classes)
  model.fit(X_train, y_train, ...)
  preds = model.predict(X)
  probs = model.predict_proba(X)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.helpers import setup_logger
from src.utils.config import (
    DL_EPOCHS, DL_BATCH_SIZE, DL_LEARNING_RATE,
    DL_HIDDEN_DIM, DL_NUM_LAYERS, DL_DROPOUT,
    RANDOM_STATE, MODELS_DIR,
)

logger = setup_logger(__name__)
torch.manual_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Utility ────────────────────────────────────────────────────────────────────

def _to_tensor(X: np.ndarray, y: np.ndarray | None = None):
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_t = torch.tensor(y, dtype=torch.long)
        return X_t, y_t
    return X_t


def _train_loop(model: nn.Module, loader: DataLoader, epochs: int, lr: float):
    """Generic training loop shared across all DL models."""
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % max(1, epochs // 5) == 0 or epoch == 1:
            logger.info("  Epoch %d/%d – loss: %.4f", epoch, epochs, total_loss / len(loader))
    return model


def _predict(model: nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_predictions, class_probabilities)."""
    model.eval()
    with torch.no_grad():
        X_t = _to_tensor(X).to(DEVICE)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs


# ── MLP ────────────────────────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, num_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_features, hidden_dim), nn.ReLU(),
                       nn.Dropout(dropout)]
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier:
    """Fully-connected MLP wrapper with sklearn-like API."""

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dim: int = DL_HIDDEN_DIM, num_layers: int = DL_NUM_LAYERS,
                 dropout: float = DL_DROPOUT):
        self.model = _MLPNet(input_dim, hidden_dim, num_layers, dropout, num_classes)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = DL_EPOCHS, batch_size: int = DL_BATCH_SIZE,
            lr: float = DL_LEARNING_RATE):
        logger.info("Training MLP on %d samples…", len(X))
        X_t, y_t = _to_tensor(X, y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        _train_loop(self.model, loader, epochs, lr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds, _ = _predict(self.model, X)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, probs = _predict(self.model, X)
        return probs


# ── CNN-1D ─────────────────────────────────────────────────────────────────────

class _CNN1DNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (B, 1, features)
        x = self.conv(x)
        return self.classifier(x)


class CNN1DClassifier:
    """1-D CNN for tabular network traffic features."""

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dim: int = DL_HIDDEN_DIM, dropout: float = DL_DROPOUT):
        self.model = _CNN1DNet(input_dim, hidden_dim, dropout, num_classes)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = DL_EPOCHS, batch_size: int = DL_BATCH_SIZE,
            lr: float = DL_LEARNING_RATE):
        logger.info("Training CNN-1D on %d samples…", len(X))
        X_t, y_t = _to_tensor(X, y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        _train_loop(self.model, loader, epochs, lr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds, _ = _predict(self.model, X)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, probs = _predict(self.model, X)
        return probs


# ── LSTM Classifier ────────────────────────────────────────────────────────────

class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                       # treat each row as a 1-step sequence
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])


class LSTMClassifier:
    """LSTM-based classifier for network traffic."""

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dim: int = DL_HIDDEN_DIM, num_layers: int = DL_NUM_LAYERS,
                 dropout: float = DL_DROPOUT):
        self.model = _LSTMNet(input_dim, hidden_dim, num_layers, dropout, num_classes)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = DL_EPOCHS, batch_size: int = DL_BATCH_SIZE,
            lr: float = DL_LEARNING_RATE):
        logger.info("Training LSTM on %d samples…", len(X))
        X_t, y_t = _to_tensor(X, y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        _train_loop(self.model, loader, epochs, lr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds, _ = _predict(self.model, X)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, probs = _predict(self.model, X)
        return probs


# ── Transformer IDS ────────────────────────────────────────────────────────────

class _TransformerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, num_classes: int, nhead: int = 4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)   # (B, 1, hidden_dim)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])   # CLS token


class TransformerClassifier:
    """Transformer encoder classifier for tabular IDS features."""

    def __init__(self, input_dim: int, num_classes: int = 2,
                 hidden_dim: int = DL_HIDDEN_DIM, num_layers: int = DL_NUM_LAYERS,
                 dropout: float = DL_DROPOUT, nhead: int = 4):
        # hidden_dim must be divisible by nhead
        if hidden_dim % nhead != 0:
            hidden_dim = (hidden_dim // nhead) * nhead
        self.model = _TransformerNet(input_dim, hidden_dim, num_layers, dropout,
                                     num_classes, nhead)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = DL_EPOCHS, batch_size: int = DL_BATCH_SIZE,
            lr: float = DL_LEARNING_RATE):
        logger.info("Training Transformer on %d samples…", len(X))
        X_t, y_t = _to_tensor(X, y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        _train_loop(self.model, loader, epochs, lr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds, _ = _predict(self.model, X)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, probs = _predict(self.model, X)
        return probs


# ── Model persistence ──────────────────────────────────────────────────────────

def save_dl_model(wrapper, name: str):
    """Save a DL model's state_dict."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save(wrapper.model.state_dict(), path)
    logger.info("DL model saved → %s", path)
    return path


def load_dl_model(wrapper, name: str):
    """Load a saved state_dict into an existing model wrapper."""
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved DL model at {path}")
    wrapper.model.load_state_dict(torch.load(path, map_location=DEVICE))
    logger.info("DL model loaded ← %s", path)
    return wrapper
