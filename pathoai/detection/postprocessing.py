"""
pathoai/detection/postprocessing.py
===================================
Detection Post-processing Utilities.

Applies confidence thresholding, non-maximum suppression (NMS) on local tiles,
and boundary coordinate clipping.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.7
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies per-class Non-Maximum Suppression (NMS) to eliminate overlapping boxes.

    Parameters
    ----------
    boxes : np.ndarray
        Bounding boxes array of shape (N, 4) in format [x1, y1, x2, y2].
    scores : np.ndarray
        Scores array of shape (N,).
    labels : np.ndarray
        Class labels array of shape (N,).
    iou_threshold : float
        IoU threshold for suppression.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Filtered (boxes, scores, labels).
    """
    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    unique_labels = np.unique(labels)
    keep_indices = []

    for lbl in unique_labels:
        cls_mask = np.where(labels == lbl)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        order = cls_scores.argsort()[::-1]

        while order.size > 0:
            i = order[0]
            keep_indices.append(cls_mask[i])

            if order.size == 1:
                break

            ious = np.array([compute_iou(cls_boxes[i], cls_boxes[j]) for j in order[1:]])
            inds = np.where(ious <= iou_threshold)[0]
            order = order[inds + 1]

    keep_indices.sort()
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
