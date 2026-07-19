"""
pathoai/fusion/point_filter.py
==============================
Point Filtering Engine.

Filters cell centroid coordinates inside binary masks using custom downsamples.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 8.2
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def filter_points_in_mask(
    points: np.ndarray | List[Tuple[float, float]],
    mask: np.ndarray,
    downsample: float,
) -> Tuple[np.ndarray, int]:
    """Filters a set of 2D coordinates (e.g. centroids) to retain those inside the mask.

    Coordinates are expected to be at level 0 (slide resolution).
    This function converts them to mask resolution using the downsample factor.

    Parameters
    ----------
    points : np.ndarray | List[Tuple[float, float]]
        Array of shape (N, 2) representing (x, y) coordinates at level 0.
    mask : np.ndarray
        Binary mask at downsampled resolution.
    downsample : float
        Downsample factor between level 0 and the mask resolution.

    Returns
    -------
    Tuple[np.ndarray, int]
        - Filtered points array of shape (K, 2) at level 0.
        - Count of filtered points (K).
    """
    pts = np.asarray(points)
    if len(pts) == 0:
        return np.empty((0, 2)), 0

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be of shape (N, 2). Got: {pts.shape}")

    if downsample <= 0:
        raise ValueError(f"downsample factor must be positive. Got: {downsample}")

    h, w = mask.shape
    filtered_pts = []

    for pt in pts:
        x, y = pt
        # Map to mask coordinate indices
        x_idx = int(np.round(x / downsample))
        y_idx = int(np.round(y / downsample))

        # Check bounds
        if 0 <= x_idx < w and 0 <= y_idx < h:
            if mask[y_idx, x_idx] > 0:
                filtered_pts.append(pt)

    if not filtered_pts:
        return np.empty((0, 2)), 0

    filtered_arr = np.array(filtered_pts)
    return filtered_arr, len(filtered_arr)
