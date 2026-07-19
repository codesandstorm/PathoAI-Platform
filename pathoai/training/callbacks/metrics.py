"""
pathoai/training/callbacks/metrics.py
=====================================
Metrics Evaluation Callback.

Aggregates predictions batch-by-batch during validation and testing,
updating a MetricCollection and storing results in trainer.current_epoch_metrics.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.5
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import torch

from pathoai.training.callbacks.base import Callback
from pathoai.training.metrics.aggregation import MetricCollection

if TYPE_CHECKING:
    from pathoai.training.trainer.trainer import Trainer


class MetricsCallback(Callback):
    """Callback that evaluates performance metrics batch-by-batch and at epoch end."""

    def __init__(self, n_classes: int) -> None:
        """
        Parameters
        ----------
        n_classes : int
            Number of target segmentation classes.
        """
        self.metrics = MetricCollection(n_classes=n_classes)

    def on_epoch_begin(self, trainer: Trainer) -> None:
        """Reset validation metrics and initialize metrics dict at epoch start."""
        self.metrics.reset()
        trainer.current_epoch_metrics = {}

    def on_validation_begin(self, trainer: Trainer) -> None:
        """Reset validation metrics state."""
        self.metrics.reset()

    def on_validation_batch_end(self, trainer: Trainer) -> None:
        """Accumulate validation batch predictions and targets."""
        if trainer.current_batch_pred is not None and trainer.current_batch_lbl is not None:
            self.metrics.update(trainer.current_batch_pred, trainer.current_batch_lbl)

    def on_validation_end(self, trainer: Trainer) -> None:
        """Compute final validation metrics and store them in the trainer."""
        computed = self.metrics.compute()
        trainer.current_epoch_metrics = computed
