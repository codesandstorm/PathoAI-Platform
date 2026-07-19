"""
pathoai/tumor_bulk/contours.py
==============================
Tumor Bulk Contour Extraction.

Extracts polygonal boundary contours from binary or labeled masks using scikit-image.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.3
"""

from __future__ import annotations

from typing import List

import numpy as np
import skimage.measure


def extract_region_contours(binary_mask: np.ndarray) -> List[np.ndarray]:
    """Extracts boundary polygon contours from a binary mask.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask (boolean or uint8) of the region.

    Returns
    -------
    List[np.ndarray]
        List of contours, where each contour is a numpy array of shape (N, 2)
        representing [x, y] coordinates in pixel space.
    """
    if not np.any(binary_mask):
        return []

    # Find contours using marching squares algorithm in scikit-image
    # skimage find_contours returns coordinates as [row, col] (i.e. [y, x])
    raw_contours = skimage.measure.find_contours(binary_mask.astype(float), level=0.5)

    contours = []
    for c in raw_contours:
        # Convert from [row, col] to [col, row] (i.e. [x, y])
        c_xy = c[:, [1, 0]]
        contours.append(c_xy)

    return contours
