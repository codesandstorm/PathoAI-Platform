"""
pathoai/detection/visualization.py
==================================
Cell Detection Overlay & Visualizer.

Renders bounding box overlays, confidence heatmaps, and detection density maps
on ROI patches or slide thumbnails.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.11
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pathoai.core.constants import CELL_CLASS_COLORS
from pathoai.core.types import CellDetection


def draw_detection_overlay(
    image: np.ndarray,
    detections: List[CellDetection],
    confidence_threshold: float = 0.25,
    thickness: int = 2,
    draw_labels: bool = True,
) -> np.ndarray:
    """Renders bounding boxes and class labels onto an image.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8.
    detections : List[CellDetection]
        List of CellDetection objects.
    confidence_threshold : float
        Minimum score threshold for drawing.
    thickness : int
        Line thickness for box borders.
    draw_labels : bool
        Whether to render label and confidence text above boxes.

    Returns
    -------
    np.ndarray
        Annotated RGB image array of shape (H, W, 3).
    """
    canvas = image.copy()
    h, w, _ = canvas.shape

    for d in detections:
        if d.confidence < confidence_threshold:
            continue

        x1 = max(0, min(w - 1, int(d.bbox.min_x)))
        y1 = max(0, min(h - 1, int(d.bbox.min_y)))
        x2 = max(0, min(w - 1, int(d.bbox.max_x)))
        y2 = max(0, min(h - 1, int(d.bbox.max_y)))

        color = CELL_CLASS_COLORS.get(d.class_id, (0, 255, 0))
        # Convert RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, thickness)

        if draw_labels:
            text = f"{d.class_name}: {d.confidence:.2f}"
            cv2.putText(
                canvas,
                text,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_bgr,
                1,
                cv2.LINE_AA,
            )

    return canvas


def create_density_heatmap(
    image_shape: Tuple[int, int],
    detections: List[CellDetection],
    grid_size: int = 32,
) -> np.ndarray:
    """Generates a 2D density heatmap grid of cell counts.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        Height and width of the spatial grid.
    detections : List[CellDetection]
        List of detections.
    grid_size : int
        Size of spatial binning cells in pixels.

    Returns
    -------
    np.ndarray
        2D heatmap matrix of shape (H_grid, W_grid), float32.
    """
    img_h, img_w = image_shape
    grid_h = max(1, img_h // grid_size)
    grid_w = max(1, img_w // grid_size)

    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)

    for d in detections:
        gx = int(d.centroid.x // grid_size)
        gy = int(d.centroid.y // grid_size)

        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            heatmap[gy, gx] += 1.0

    return heatmap
