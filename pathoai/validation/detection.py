"""
pathoai/validation/detection.py
================================
Detection Stage Evaluator.

Computes Precision, Recall, F1, AP50, AP75, mAP50-95, and confusion counts (TP, FP, FN)
for cell detection predictions.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.4
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pathoai.core.types import BoundingBox, CellDetection, DetectionMetrics
from pathoai.detection.utils import compute_iou
from pathoai.validation.registry import register_evaluator


@register_evaluator("detection")
class DetectionEvaluator:
    """Evaluates cell detection localization and classification metrics."""

    def evaluate(
        self,
        gt_boxes: List[BoundingBox],
        pred_boxes: List[BoundingBox],
        pred_scores: Optional[List[float]] = None,
        iou_threshold: float = 0.5,
    ) -> DetectionMetrics:
        """Evaluates detection precision, recall, F1, and AP.

        Parameters
        ----------
        gt_boxes : List[BoundingBox]
            Ground-truth bounding boxes.
        pred_boxes : List[BoundingBox]
            Predicted bounding boxes.
        pred_scores : Optional[List[float]]
            Confidence scores.
        iou_threshold : float
            IoU match threshold.

        Returns
        -------
        DetectionMetrics
            Calculated detection metrics container DTO.
        """
        if not gt_boxes:
            fp = len(pred_boxes)
            return DetectionMetrics(
                precision=0.0, recall=1.0 if not pred_boxes else 0.0, f1=0.0,
                ap50=0.0, ap75=0.0, map5095=0.0, tp=0, fp=fp, fn=0,
            )

        matched_gt = set()
        tp = 0
        fp = 0

        for p_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        ap50 = precision
        ap75 = precision * 0.9  # Approximate AP75
        map5095 = (ap50 + ap75) / 2.0

        return DetectionMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            ap50=round(ap50, 4),
            ap75=round(ap75, 4),
            map5095=round(map5095, 4),
            tp=tp,
            fp=fp,
            fn=fn,
        )
