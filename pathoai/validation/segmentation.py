"""
pathoai/validation/segmentation.py
===================================
Segmentation Stage Evaluator.

Computes Dice Similarity Coefficient, IoU, Precision, Recall, Specificity,
Pixel Accuracy, and F1 score between predicted masks and ground-truth masks.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.3
"""

from __future__ import annotations

import numpy as np

from pathoai.core.types import SegmentationMetrics
from pathoai.validation.registry import register_evaluator


@register_evaluator("segmentation")
class SegmentationEvaluator:
    """Evaluates semantic segmentation quality metrics against ground truth."""

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> SegmentationMetrics:
        """Evaluates segmentation quality metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Binary ground-truth mask.
        y_pred : np.ndarray
            Binary predicted mask.

        Returns
        -------
        SegmentationMetrics
            Calculated segmentation metrics container DTO.
        """
        y_true_bin = (y_true > 0).astype(np.uint8)
        y_pred_bin = (y_pred > 0).astype(np.uint8)

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        dice = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
        iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

        return SegmentationMetrics(
            dice=round(dice, 4),
            iou=round(iou, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            specificity=round(specificity, 4),
            pixel_accuracy=round(accuracy, 4),
            f1=round(f1, 4),
        )
