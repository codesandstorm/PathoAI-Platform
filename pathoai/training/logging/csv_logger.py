"""
pathoai/training/logging/csv_logger.py
======================================
CSV Training Logger.

Appends per-epoch metrics and losses to a CSV file.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.5
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from pathoai.core.logger import get_logger
from pathoai.training.callbacks.base import Callback

if TYPE_CHECKING:
    from pathoai.training.trainer.trainer import Trainer

logger = get_logger(__name__)


class CSVLogger(Callback):
    """Callback that logs epoch results to a CSV file."""

    def __init__(self, filename: str | Path) -> None:
        """
        Parameters
        ----------
        filename : str | Path
            Path to the output CSV file.
        """
        self.filename = Path(filename)
        self.headers_written = False

    def on_train_begin(self, trainer: Trainer) -> None:
        """Create target directories and initialize CSV flags."""
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        # If resuming, we check if file already exists
        if self.filename.is_file():
            self.headers_written = True

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Record epoch stats to CSV."""
        row_dict = {
            "epoch": trainer.state.epoch + 1,
            "train_loss": trainer.state.train_loss,
            "val_loss": trainer.state.val_loss,
            "learning_rate": trainer.state.learning_rate,
            "elapsed_time": trainer.state.elapsed_time,
        }

        # Merge in validation metrics
        metrics = getattr(trainer, "current_epoch_metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                row_dict[k] = v

        keys = list(row_dict.keys())

        # Write to file
        file_mode = "a" if self.headers_written else "w"
        try:
            with open(self.filename, file_mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                if not self.headers_written:
                    writer.writeheader()
                    self.headers_written = True
                writer.writerow(row_dict)
        except Exception as exc:
            logger.error("CSVLogger: failed to write epoch stats to %s: %s", self.filename, exc)
