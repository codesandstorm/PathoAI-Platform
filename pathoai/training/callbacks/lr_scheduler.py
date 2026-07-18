"""
pathoai/training/callbacks/lr_scheduler.py
=========================================
Learning rate scheduler callback.

Steps the learning rate scheduler at the end of each training epoch,
passing monitored metrics if required (e.g. for ReduceLROnPlateau).

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pathoai.core.logger import get_logger
from pathoai.training.callbacks.base import Callback

if TYPE_CHECKING:
    from pathoai.training.trainer.trainer import Trainer

logger = get_logger(__name__)


class LRSchedulerCallback(Callback):
    """Callback to step PyTorch learning rate schedulers at epoch end."""

    def __init__(
        self,
        scheduler: Any,
        monitor: str = "val_loss",
    ) -> None:
        """
        Parameters
        ----------
        scheduler : torch.optim.lr_scheduler._LRScheduler
            PyTorch learning rate scheduler.
        monitor : str
            Validation metric to monitor if using ReduceLROnPlateau.
        """
        self.scheduler = scheduler
        self.monitor = monitor

        # Check if the scheduler requires a metric argument (e.g. ReduceLROnPlateau)
        self.is_plateau = self.scheduler.__class__.__name__ == "ReduceLROnPlateau"

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Step the scheduler, passing monitored metrics if needed."""
        if self.is_plateau:
            # Query target metric
            metrics = getattr(trainer, "current_epoch_metrics", {})
            val = metrics.get(self.monitor)
            if val is None:
                val = getattr(trainer.state, self.monitor, None)

            if val is None:
                logger.warning(
                    "LRSchedulerCallback: plateau scheduler requires '%s' but it was not found. "
                    "Stepping with default 0.0 value.",
                    self.monitor,
                )
                val = 0.0

            self.scheduler.step(val)
        else:
            self.scheduler.step()

        # Update state learning rate
        for param_group in trainer.optimizer.param_groups:
            trainer.state.learning_rate = param_group["lr"]
            break

        logger.debug(
            "LRSchedulerCallback: stepped scheduler. Current learning rate: %.6f",
            trainer.state.learning_rate,
        )
