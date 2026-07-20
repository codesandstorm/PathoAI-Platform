"""
pathoai/detection/utils.py
==========================
Detection Utility Functions.

Helper routines for bounding box transformations, area calculations, and shape checks.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.16
"""

from __future__ import annotations

import numpy as np


def box_xywh_to_xyxy(box_xywh: np.ndarray) -> np.ndarray:
    """Converts bounding box from [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box_xywh[0], box_xywh[1], box_xywh[2], box_xywh[3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_xyxy_to_xywh(box_xyxy: np.ndarray) -> np.ndarray:
    """Converts bounding box from [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
    x1, y1, x2, y2 = box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return np.array([cx, cy, w, h], dtype=np.float32)


def clip_boxes_to_image(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clips boxes [x1, y1, x2, y2] to image bounds [0, width] and [0, height]."""
    if len(boxes) == 0:
        return boxes.copy()

    clipped = boxes.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height))
    clipped[:, 2] = np.clip(clipped[:, 2], 0.0, float(width))
    clipped[:, 3] = np.clip(clipped[:, 3], 0.0, float(height))
    return clipped


def compute_iou(boxA: Any, boxB: Any) -> float:
    """Computes IoU between two BoundingBox DTOs or numpy array coordinate bounds."""
    if hasattr(boxA, "min_x"):
        ax1, ay1, ax2, ay2 = boxA.min_x, boxA.min_y, boxA.max_x, boxA.max_y
    else:
        ax1, ay1, ax2, ay2 = boxA[0], boxA[1], boxA[2], boxA[3]

    if hasattr(boxB, "min_x"):
        bx1, by1, bx2, by2 = boxB.min_x, boxB.min_y, boxB.max_x, boxB.max_y
    else:
        bx1, by1, bx2, by2 = boxB[0], boxB[1], boxB[2], boxB[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)

