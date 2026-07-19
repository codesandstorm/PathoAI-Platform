"""
pathoai/training/callbacks/metrics.py
=====================================
Metrics Evaluation Callback.

Aggregates predictions accumulated in trainer.epoch_preds/epoch_targets,
updates a MetricCollection, and sets trainer.current_epoch_metrics.

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
    """Callback that evaluates performance metrics at validation/epoch end."""

    def __init__(self, n_classes: int) -> None:
        """
        Parameters
        ----------
        n_classes : int
            Number of target segmentation classes.
        """
        self.metrics = MetricCollection(n_classes=n_classes)

    def on_epoch_begin(self, trainer: Trainer) -> None:
        """Reset validation metrics and prediction lists at epoch start."""
        self.metrics.reset()
        # Initialize dictionary to prevent getattr NameError in other callbacks
        trainer.current_epoch_metrics = {}

    def on_validation_begin(self, trainer: Trainer) -> None:
        """Reset validation metrics state."""
        self.metrics.reset()

    def on_validation_end(self, trainer: Trainer) -> None:
        """Aggregate prediction batches and compute validation metrics."""
        # 1. Update metric collection
        for preds, targets in zip(trainer.epoch_preds, trainer.epoch_targets):
            self.metrics.update(preds, targets)

        # 2. Compute final metrics dictionary
        computed = self.metrics.compute()

        # 3. Store metrics in trainer for loggers and early stopping callbacks
        trainer.current_epoch_metrics = computed
