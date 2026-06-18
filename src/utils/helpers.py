"""
Shared utility helpers: logging, reproducibility, timing.
"""

import time
import logging
import random
import numpy as np


def setup_logger(name: str = "AI-IDS", level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger.  Idempotent – safe to call multiple times.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        logger = setup_logger()
        logger.info("%s took %.2f seconds.", self.label or "Block", elapsed)
