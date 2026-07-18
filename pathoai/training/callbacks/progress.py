"""
pathoai/training/callbacks/progress.py
=====================================
Progress logger callback.

Integrates tqdm progress bars for active feedback on training runs,
reporting epoch counts, steps, running losses, and metrics.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tqdm import tqdm

from pathoai.core.logger import get_logger
from pathoai.training.callbacks.base import Callback

if TYPE_CHECKING:
    from pathoai.training.trainer.trainer import Trainer

logger = get_logger(__name__)


class ProgressLogger(Callback):
    """Callback to render tqdm progress bars for epochs and batches."""

    def __init__(self) -> None:
        self._pbar: Optional[tqdm] = None

    def on_epoch_begin(self, trainer: Trainer) -> None:
        """Initialize progress bar at start of epoch."""
        epoch = trainer.state.epoch
        total_batches = getattr(trainer, "num_batches", 0) or 0
        self._pbar = tqdm(
            total=total_batches,
            desc=f"Epoch {epoch + 1:03d}",
            unit="batch",
            leave=True,
        )

    def on_batch_end(self, trainer: Trainer) -> None:
        """Step progress bar, updating running stats."""
        if self._pbar is not None:
            loss_val = 0.0
            if trainer.current_batch_loss is not None:
                loss_val = trainer.current_batch_loss.item()

            self._pbar.update(1)
            self._pbar.set_postfix(
                step=trainer.state.global_step,
                loss=f"{loss_val:.4f}",
                lr=f"{trainer.state.learning_rate:.6f}",
            )

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Close progress bar and print summary metrics."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

        # Print overall epoch summary
        s = trainer.state
        metrics = getattr(trainer, "current_epoch_metrics", {})
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))

        msg = f"Epoch {s.epoch + 1:03d} summary - train_loss: {s.train_loss:.4f}, val_loss: {s.val_loss:.4f}"
        if metric_str:
            msg += f", {metric_str}"

        logger.info(msg)
