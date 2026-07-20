"""
pathoai/fusion/visualization.py
================================
Spatial Fusion Visualization Overlay Engine.

Renders TumorROI boundaries, cell detections, and spatial classification overlays
onto tissue images.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.8
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from pathoai.core.types import SpatialDetection, TumorROI


def draw_spatial_fusion_overlay(
    image: np.ndarray,
    spatial_detections: List[SpatialDetection],
    rois: List[TumorROI],
    draw_roi_contours: bool = True,
    draw_distance_lines: bool = False,
) -> np.ndarray:
    """Renders spatial fusion overlay onto an RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3), uint8.
    spatial_detections : List[SpatialDetection]
        List of spatial detections.
    rois : List[TumorROI]
        List of TumorROIs.
    draw_roi_contours : bool
        Whether to draw ROI polygon contours.
    draw_distance_lines : bool
        Whether to draw lines to nearest boundary points.

    Returns
    -------
    np.ndarray
        Annotated RGB image array.
    """
    canvas = image.copy()

    # 1. Draw ROI contours
    if draw_roi_contours:
        for roi in rois:
            for poly in roi.contours:
                pts = np.array([[int(p.x), int(p.y)] for p in poly.exterior], dtype=np.int32)
                if len(pts) >= 3:
                    cv2.polylines(canvas, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # 2. Draw spatial cell detections
    for sd in spatial_detections:
        cx, cy = int(sd.detection.centroid.x), int(sd.detection.centroid.y)

        if sd.inside_tumor:
            color = (0, 0, 255)  # Red for intratumoral
        elif sd.inside_stroma:
            color = (0, 255, 0)  # Green for stromal
        else:
            color = (255, 255, 0)  # Cyan for distant

        cv2.circle(canvas, (cx, cy), radius=4, color=color, thickness=-1)

        if draw_distance_lines and sd.nearest_boundary_point:
            bx, by = int(sd.nearest_boundary_point.x), int(sd.nearest_boundary_point.y)
            cv2.line(canvas, (cx, cy), (bx, by), color=(200, 200, 200), thickness=1)

    return canvas
