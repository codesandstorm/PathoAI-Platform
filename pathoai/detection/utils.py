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
