"""
pathoai/training/metrics/segmentation.py
=======================================
Semantic segmentation metrics calculator.

Accumulates True Positive, False Positive, and False Negative pixel counts
across batches to compute mathematically correct epoch-level Dice, IoU,
Precision, Recall, and F1 scores.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.3
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates and computes semantic segmentation performance metrics."""

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
        """Clear accumulator counts."""
        self.tp = np.zeros(self.n_classes, dtype=np.int64)
        self.fp = np.zeros(self.n_classes, dtype=np.int64)
        self.fn = np.zeros(self.n_classes, dtype=np.int64)
        self.support = np.zeros(self.n_classes, dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Accumulate counts from a prediction/ground-truth batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted class IDs. Shape: (B, H, W) or logits (B, C, H, W).
        y_true : torch.Tensor
            Ground truth class IDs. Shape: (B, H, W).
        """
        # Convert logits/probabilities to class IDs if needed
        if y_pred.ndim == 4:
            y_pred = y_pred.argmax(dim=1)

        pred = y_pred.detach().cpu().numpy().astype(np.int64)
        true = y_true.detach().cpu().numpy().astype(np.int64)

        self.total_pixels += true.size
        self.correct_pixels += np.sum(pred == true)

        # Vectorized counts per class
        for c in range(self.n_classes):
            p_mask = pred == c
            t_mask = true == c
            self.tp[c] += np.sum(p_mask & t_mask)
            self.fp[c] += np.sum(p_mask & ~t_mask)
            self.fn[c] += np.sum(~p_mask & t_mask)
            self.support[c] += np.sum(t_mask)

    def compute(self) -> Dict[str, Any]:
        """Compute metrics based on accumulated counts."""
        if self.total_pixels == 0:
            return {}

        pixel_accuracy = float(self.correct_pixels / self.total_pixels)

        dice_per_class: List[float] = []
        iou_per_class: List[float] = []
        precision_per_class: List[float] = []
        recall_per_class: List[float] = []

        for c in range(self.n_classes):
            tp = self.tp[c]
            fp = self.fp[c]
            fn = self.fn[c]

            # Dice
            denom_dice = 2 * tp + fp + fn
            dice = float(2 * tp / denom_dice) if denom_dice > 0 else 1.0

            # IoU
            denom_iou = tp + fp + fn
            iou = float(tp / denom_iou) if denom_iou > 0 else 1.0

            # Precision
            denom_prec = tp + fp
            prec = float(tp / denom_prec) if denom_prec > 0 else 1.0

            # Recall
            denom_rec = tp + fn
            rec = float(tp / denom_rec) if denom_rec > 0 else 1.0

            dice_per_class.append(dice)
            iou_per_class.append(iou)
            precision_per_class.append(prec)
            recall_per_class.append(rec)

        mean_dice = float(np.mean(dice_per_class))
        mean_iou = float(np.mean(iou_per_class))
        mean_pixel_acc = float(np.mean(recall_per_class))

        # F1 Macro, Micro, Weighted
        macro_f1 = mean_dice
        # Micro F1 equals pixel accuracy for multiclass classification
        micro_f1 = pixel_accuracy

        # Weighted F1
        total_support = np.sum(self.support)
        if total_support > 0:
            weighted_f1 = float(np.sum(np.array(dice_per_class) * self.support) / total_support)
            weighted_prec = float(np.sum(np.array(precision_per_class) * self.support) / total_support)
            weighted_rec = float(np.sum(np.array(recall_per_class) * self.support) / total_support)
        else:
            weighted_f1 = 1.0
            weighted_prec = 1.0
            weighted_rec = 1.0

        res = {
            "pixel_accuracy": pixel_accuracy,
            "mean_pixel_accuracy": mean_pixel_acc,
            "mean_dice": mean_dice,
            "mean_iou": mean_iou,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "weighted_f1": weighted_f1,
            "weighted_precision": weighted_prec,
            "weighted_recall": weighted_rec,
            "dice_per_class": dice_per_class,
            "iou_per_class": iou_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "support_per_class": self.support.tolist(),
        }

        # Expose individual class dice and iou keys for flat dictionary loggers
        for c in range(self.n_classes):
            res[f"class_{c}_dice"] = dice_per_class[c]
            res[f"class_{c}_iou"] = iou_per_class[c]

        return res
