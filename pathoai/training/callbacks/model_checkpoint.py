"""
pathoai/training/callbacks/model_checkpoint.py
=============================================
Model checkpoint callback.

Interfaces with the CheckpointManager to save weights at the end of each epoch,
preserving top-K models and track validation improvements.

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


class ModelCheckpoint(Callback):
    """Callback to automatically save model parameters during training."""

    def __init__(self, checkpoint_manager: Any) -> None:
        """
        Parameters
        ----------
        checkpoint_manager : CheckpointManager
            Manager instance responsible for checking values and writing weights.
        """
        self.manager = checkpoint_manager

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Saves current state via checkpoint manager."""
        # Query monitored metric value from current epoch metrics
        metrics = getattr(trainer, "current_epoch_metrics", {})
        monitor = self.manager.monitor
        val = metrics.get(monitor)

        if val is None:
            # Fallback to state attributes
            val = getattr(trainer.state, monitor, None)

        if val is None:
            logger.warning(
                "ModelCheckpoint: monitored metric '%s' not found. "
                "Defaulting save evaluation value to 0.0.",
                monitor,
            )
            val = 0.0

        # Save checkpoint using the manager
        self.manager.save_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            state=trainer.state,
            current_value=float(val),
        )
