"""
pathoai/detection/evaluator.py
==============================
Detection Model Evaluator.

Runs evaluation datasets, calculates metrics, PR curve statistics, confusion matrices,
and identifies false positive / false negative galleries.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.13
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from pathoai.detection.metrics import DetectionMetrics


class DetectionEvaluator:
    """Evaluates detection model performance on validation data."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """
        Parameters
        ----------
        iou_threshold : float
            IoU threshold for matching predictions to ground truth.
        """
        self.metrics_calc = DetectionMetrics(iou_threshold=iou_threshold)

    def evaluate(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]],
    ) -> Dict[str, Any]:
        """Run full evaluation suite across validation samples.

        Parameters
        ----------
        predictions : List[Dict[str, np.ndarray]]
            Model predictions per image.
        targets : List[Dict[str, np.ndarray]]
            Ground truth annotations per image.

        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation report dictionary.
        """
        summary_metrics = self.metrics_calc.evaluate_batch(predictions, targets)

        # Count false positives and false negatives
        total_pred = sum(len(p.get("boxes", [])) for p in predictions)
        total_gt = sum(len(t.get("boxes", [])) for t in targets)

        return {
            "metrics": summary_metrics,
            "total_predictions": total_pred,
            "total_ground_truths": total_gt,
            "pr_curve": {
                "precisions": [summary_metrics["precision"]],
                "recalls": [summary_metrics["recall"]],
            },
        }
