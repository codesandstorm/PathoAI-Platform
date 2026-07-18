"""
pathoai/training/metrics/aggregation.py
======================================
MetricCollection aggregator.

Aggregates multiple metric calculation engines into a single collection,
routing update, compute, and reset signals.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.3
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from pathoai.training.metrics.confusion import ConfusionMatrixMetric
from pathoai.training.metrics.segmentation import SegmentationMetrics


class MetricCollection:
    """Aggregates and orchestrates multiple performance metric calculators."""

    def __init__(self, n_classes: int = 6) -> None:
        """
        Parameters
        ----------
        n_classes : int
            Number of target classes.
        """
        self.n_classes = n_classes
        self.metrics = [
            SegmentationMetrics(n_classes=n_classes),
            ConfusionMatrixMetric(n_classes=n_classes),
        ]

    def reset(self) -> None:
        """Reset all aggregated metrics."""
        for metric in self.metrics:
            metric.reset()

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update all aggregated metrics with a batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions (logits or class IDs).
        y_true : torch.Tensor
            Ground truth class IDs.
        """
        for metric in self.metrics:
            metric.update(y_pred, y_true)

    def compute(self) -> Dict[str, Any]:
        """Compute values for all metrics, returning a single merged dictionary."""
        results: Dict[str, Any] = {}
        for metric in self.metrics:
            results.update(metric.compute())
        return results
