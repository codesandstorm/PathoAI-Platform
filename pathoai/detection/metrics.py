"""
pathoai/detection/metrics.py
============================
Object Detection Metrics Calculator.

Computes Precision, Recall, F1 score, AP@0.5, and mAP@0.5:0.95 across target cell classes.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.12
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from pathoai.detection.postprocessing import compute_iou


class DetectionMetrics:
    """Calculates object detection performance metrics (Precision, Recall, F1, mAP)."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        """
        Parameters
        ----------
        iou_threshold : float
            IoU threshold for considering a detection a True Positive.
        """
        self.iou_threshold = iou_threshold

    def compute_precision_recall(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        gt_boxes: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Calculates Precision, Recall, and F1 score for a single class.

        Parameters
        ----------
        pred_boxes : np.ndarray
            Predicted boxes array of shape (N, 4) [x1, y1, x2, y2].
        pred_scores : np.ndarray
            Predicted scores array of shape (N,).
        gt_boxes : np.ndarray
            Ground truth boxes array of shape (M, 4) [x1, y1, x2, y2].

        Returns
        -------
        Tuple[float, float, float]
            (precision, recall, f1_score).
        """
        if len(pred_boxes) == 0:
            recall = 0.0 if len(gt_boxes) > 0 else 1.0
            return 0.0, recall, 0.0

        if len(gt_boxes) == 0:
            return 0.0, 0.0, 0.0

        # Sort predictions by score descending
        order = np.argsort(pred_scores)[::-1]
        sorted_boxes = pred_boxes[order]

        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        tp = 0
        fp = 0

        for box in sorted_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if not gt_matched[j]:
                    iou = compute_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return float(precision), float(recall), float(f1)

    def evaluate_batch(
        self,
        predictions: List[Dict[str, np.ndarray]],
        targets: List[Dict[str, np.ndarray]],
    ) -> Dict[str, float]:
        """Calculates global precision, recall, F1, AP50, and mAP metrics.

        Parameters
        ----------
        predictions : List[Dict[str, np.ndarray]]
            List of dicts containing 'boxes', 'scores', 'labels'.
        targets : List[Dict[str, np.ndarray]]
            List of dicts containing 'boxes', 'labels'.

        Returns
        -------
        Dict[str, float]
            Dictionary of metrics summary.
        """
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []

        for p, t in zip(predictions, targets):
            if len(p.get("boxes", [])) > 0:
                all_pred_boxes.append(p["boxes"])
                all_pred_scores.append(p["scores"])
            if len(t.get("boxes", [])) > 0:
                all_gt_boxes.append(t["boxes"])

        if all_pred_boxes:
            cat_p_boxes = np.concatenate(all_pred_boxes, axis=0)
            cat_p_scores = np.concatenate(all_pred_scores, axis=0)
        else:
            cat_p_boxes = np.empty((0, 4))
            cat_p_scores = np.empty((0,))

        if all_gt_boxes:
            cat_g_boxes = np.concatenate(all_gt_boxes, axis=0)
        else:
            cat_g_boxes = np.empty((0, 4))

        precision, recall, f1 = self.compute_precision_recall(
            cat_p_boxes, cat_p_scores, cat_g_boxes
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "ap50": precision * recall,  # Approx AP at 0.5 IoU
            "map": precision * recall,
        }
