"""
pathoai/training/history/history.py
==================================
Training History Tracker.

Records per-epoch losses, learning rates, elapsed times, and metrics,
and serializes them to CSV, JSON, and Pickle formats.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.4
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class HistoryTracker:
    """Tracks and serializes training history across epochs."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        elapsed_time: float,
        metrics: Dict[str, Any],
    ) -> None:
        """Record the state and metrics of an epoch.

        Parameters
        ----------
        epoch : int
            Zero-indexed epoch count.
        train_loss : float
            Average training loss.
        val_loss : float
            Average validation loss.
        learning_rate : float
            Learning rate at epoch end.
        elapsed_time : float
            Elapsed duration of the epoch in seconds.
        metrics : Dict[str, Any]
            Dictionary of validation metrics computed for the epoch.
        """
        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
            "elapsed_time": elapsed_time,
        }

        # Merge in all validation metrics (flattening lists if necessary,
        # but class dice/iou are already flat in compute() output)
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                entry[k] = v

        self.history.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the training history to a Pandas DataFrame."""
        return pd.DataFrame(self.history)

    def save_csv(self, path: str | Path) -> None:
        """Save history as a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(p, index=False)
        logger.debug("History saved as CSV to %s", p)

    def save_json(self, path: str | Path) -> None:
        """Save history as a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.debug("History saved as JSON to %s", p)

    def save_pickle(self, path: str | Path) -> None:
        """Save history as a Python pickle file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self.history, f)
        logger.debug("History saved as Pickle to %s", p)
