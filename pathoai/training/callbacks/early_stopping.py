"""
pathoai/training/callbacks/early_stopping.py
===========================================
Early stopping callback to halt training on plateau.

Monitors validation loss or segmentation metrics and triggers termination
if no progress occurs within a defined number of epochs.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathoai.core.logger import get_logger
from pathoai.training.callbacks.base import Callback

if TYPE_CHECKING:
    from pathoai.training.trainer.trainer import Trainer

logger = get_logger(__name__)


class EarlyStopping(Callback):
    """Halts model training if validation metrics cease to improve."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """
        Parameters
        ----------
        monitor : str
            The metric key to monitor (e.g. 'val_loss', 'val_dice').
        patience : int
            Number of epochs to wait without improvement before stopping.
        min_delta : float
            Minimum change in the monitored value to qualify as an improvement.
        mode : str
            One of {'min', 'max'}. In 'min' mode, training stops when the quantity
            monitored has stopped decreasing. In 'max' mode, it stops when increasing.
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'. Got: {mode}")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.wait = 0
        self.best_value = float("inf") if mode == "min" else -float("inf")

    def on_train_begin(self, trainer: Trainer) -> None:
        """Reset counters at start of training run."""
        self.wait = 0
        self.best_value = float("inf") if self.mode == "min" else -float("inf")

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Evaluate the monitored metric at the end of each epoch."""
        # Retrieve metric from trainer metrics dictionary or state
        metrics = getattr(trainer, "current_epoch_metrics", {})
        val = metrics.get(self.monitor)

        if val is None:
            # Check trainer state attributes as fallback (e.g., trainer.state.val_loss)
            val = getattr(trainer.state, self.monitor, None)

        if val is None:
            logger.warning(
                "EarlyStopping: monitored metric '%s' not found in current epoch. "
                "Skipping check for epoch %d.",
                self.monitor,
                trainer.state.epoch,
            )
            return

        # Check for improvement
        improved = False
        if self.mode == "min":
            if val < self.best_value - self.min_delta:
                improved = True
        else:
            if val > self.best_value + self.min_delta:
                improved = True

        if improved:
            self.best_value = val
            self.wait = 0
            logger.debug(
                "EarlyStopping: metric '%s' improved to %.4f. Resetting patience counter.",
                self.monitor,
                val,
            )
        else:
            self.wait += 1
            logger.info(
                "EarlyStopping: metric '%s' did not improve (best: %.4f, current: %.4f). "
                "Patience step: %d/%d",
                self.monitor,
                self.best_value,
                val,
                self.wait,
                self.patience,
            )
            if self.wait >= self.patience:
                logger.warning(
                    "EarlyStopping: patience of %d epochs reached. Triggering early stop.",
                    self.patience,
                )
                trainer.stop_training = True
