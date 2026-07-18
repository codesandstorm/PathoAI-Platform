"""
pathoai/training/logging/tensorboard.py
======================================
TensorBoard training metrics logger.

Streams losses, learning rates, and validation metrics to TensorBoard logs.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.5
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pathoai.core.logger import get_logger
from pathoai.training.callbacks.base import Callback

if TYPE_CHECKING:
    from pathoai.training.trainer.trainer import Trainer

logger = get_logger(__name__)


class TensorBoardLogger(Callback):
    """Callback to write training history and metrics to TensorBoard logs."""

    def __init__(self, log_dir: str | Path) -> None:
        """
        Parameters
        ----------
        log_dir : str | Path
            Directory where TensorBoard event logs will be saved.
        """
        self.log_dir = Path(log_dir)
        self.writer: Optional[Any] = None

    def on_train_begin(self, trainer: Trainer) -> None:
        """Initialize SummaryWriter at the start of training."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            logger.info("TensorBoardLogger: logging events to %s", self.log_dir)
        except ImportError:
            logger.warning(
                "TensorBoardLogger: torch.utils.tensorboard.SummaryWriter could not be imported. "
                "TensorBoard logging will be disabled."
            )
            self.writer = None

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Log epoch-level metrics and loss parameters."""
        if self.writer is None:
            return

        epoch = trainer.state.epoch + 1

        # Log losses
        self.writer.add_scalar("Loss/Train", trainer.state.train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", trainer.state.val_loss, epoch)

        # Log hyperparams
        self.writer.add_scalar("Hyperparams/LearningRate", trainer.state.learning_rate, epoch)
        self.writer.add_scalar("Time/EpochDuration", trainer.state.elapsed_time, epoch)

        # Log validation metrics
        metrics = getattr(trainer, "current_epoch_metrics", {})
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                # Clean up metric names for tensorboard folders (e.g. mean_dice -> Metrics/MeanDice)
                tb_folder = "Metrics"
                if "dice" in name.lower():
                    tb_folder = "Metrics/Dice"
                elif "iou" in name.lower():
                    tb_folder = "Metrics/IoU"
                elif "precision" in name.lower() or "recall" in name.lower() or "f1" in name.lower():
                    tb_folder = "Metrics/Classification"

                # Clean key name
                clean_name = "".join(part.capitalize() for part in name.split("_"))
                self.writer.add_scalar(f"{tb_folder}/{clean_name}", value, epoch)

    def on_train_end(self, trainer: Trainer) -> None:
        """Close SummaryWriter at the end of training."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
            logger.debug("TensorBoardLogger: flushed and closed writer.")
stream = None
