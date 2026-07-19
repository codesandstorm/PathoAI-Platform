"""
pathoai/training/callbacks/base.py
=================================
Abstract base class for all Trainer callback observers.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathoai.training.trainer.trainer import Trainer


class Callback:
    """Base observer interface to hook into the Trainer lifecycle.

    Inheriting callbacks can inspect the trainer state and alter execution flow
    (e.g., stopping training early).
    """

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at the start of the fit() training run."""
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        """Called at the end of the fit() training run."""
        pass

    def on_epoch_begin(self, trainer: Trainer) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: Trainer) -> None:
        """Called at the start of each training batch."""
        pass

    def on_batch_end(self, trainer: Trainer) -> None:
        """Called at the end of each training batch."""
        pass

    def on_validation_begin(self, trainer: Trainer) -> None:
        """Called at the start of the validation run."""
        pass

    def on_validation_end(self, trainer: Trainer) -> None:
        """Called at the end of the validation run."""
        pass

    def on_validation_batch_end(self, trainer: Trainer) -> None:
        """Called at the end of each validation batch."""
        pass
