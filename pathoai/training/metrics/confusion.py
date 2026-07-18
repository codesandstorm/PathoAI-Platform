"""
pathoai/training/metrics/confusion.py
====================================
Confusion Matrix and Cohen's Kappa calculation.

Accumulates predictions across validation batches and calculates raw/normalized
confusion matrices and Cohen's Kappa statistic.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.3
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch


class ConfusionMatrixMetric:
    """Accumulates predictions to compute confusion matrices and Cohen's Kappa."""

    def __init__(self, n_classes: int = 6) -> None:
        """
        Parameters
        ----------
        n_classes : int
            Number of target classes.
        """
        self.n_classes = n_classes
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated confusion matrix."""
        self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update confusion matrix counts from a batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predictions (logits or class IDs). Shape: (B, H, W) or (B, C, H, W).
        y_true : torch.Tensor
            Ground truth class IDs. Shape: (B, H, W).
        """
        if y_pred.ndim == 4:
            y_pred = y_pred.argmax(dim=1)

        pred = y_pred.detach().cpu().numpy().astype(np.int64).flatten()
        true = y_true.detach().cpu().numpy().astype(np.int64).flatten()

        # Vectorized confusion matrix computation
        # Use np.bincount for high-speed indexing
        # index = true * n_classes + pred
        indices = true * self.n_classes + pred
        valid_mask = (pred >= 0) & (pred < self.n_classes) & (true >= 0) & (true < self.n_classes)

        counts = np.bincount(indices[valid_mask], minlength=self.n_classes * self.n_classes)
        self.matrix += counts.reshape((self.n_classes, self.n_classes))

    def compute(self) -> Dict[str, Any]:
        """Compute the raw, normalized matrices and Cohen's Kappa."""
        matrix = self.matrix
        total = np.sum(matrix)

        if total == 0:
            return {
                "confusion_matrix": matrix.tolist(),
                "normalized_confusion_matrix": np.zeros_like(matrix, dtype=np.float64).tolist(),
                "cohens_kappa": 0.0,
            }

        # 1. Normalize row-wise (divide by support of true labels)
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        # Avoid division by zero
        normalized = np.where(row_sums > 0, matrix / row_sums, 0.0)

        # 2. Cohen's Kappa calculation
        # p_o = observed agreement = sum of diagonal / total
        p_o = float(np.trace(matrix) / total)

        # p_e = expected agreement = sum_c (pred_c * true_c) / total^2
        pred_sums = np.sum(matrix, axis=0)  # column sums
        true_sums = np.sum(matrix, axis=1)  # row sums
        p_e = float(np.sum(pred_sums * true_sums) / (total ** 2))

        # Kappa = (p_o - p_e) / (1 - p_e)
        if abs(1.0 - p_e) > 1e-6:
            cohens_kappa = float((p_o - p_e) / (1.0 - p_e))
        else:
            cohens_kappa = 1.0

        return {
            "confusion_matrix": matrix.tolist(),
            "normalized_confusion_matrix": normalized.tolist(),
            "cohens_kappa": cohens_kappa,
        }
