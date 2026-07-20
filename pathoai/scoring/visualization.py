"""
pathoai/scoring/visualization.py
================================
sTIL Scoring Visualizer Engine.

Renders sTIL density heatmaps, score histograms, and ROI statistics figures.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.12
"""

from __future__ import annotations

import cv2
import numpy as np

from pathoai.core.types import FusionResult, STILScore


def create_stil_density_heatmap(
    image_shape: tuple[int, int],
    fusion_result: FusionResult,
    grid_size: int = 128,
) -> np.ndarray:
    """Renders 2D density heatmap array for stromal lymphocytes.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Target (H, W) dimensions.
    fusion_result : FusionResult
        Spatial fusion result DTO.
    grid_size : int
        Grid cell stride in pixels.

    Returns
    -------
    np.ndarray
        Colorized RGB heatmap overlay of shape (H, W, 3).
    """
    h, w = image_shape[:2]
    density_map = np.zeros((h // grid_size + 1, w // grid_size + 1), dtype=np.float32)

    for sd in fusion_result.spatial_detections:
        if sd.inside_stroma:
            gx = int(sd.detection.centroid.x) // grid_size
            gy = int(sd.detection.centroid.y) // grid_size
            if 0 <= gy < density_map.shape[0] and 0 <= gx < density_map.shape[1]:
                density_map[gy, gx] += 1.0

    # Normalize density map
    max_val = np.max(density_map)
    if max_val > 0:
        norm = (density_map / max_val * 255.0).astype(np.uint8)
    else:
        norm = density_map.astype(np.uint8)

    # Resize to match full image dimensions
    heatmap_resized = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
    colorized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
