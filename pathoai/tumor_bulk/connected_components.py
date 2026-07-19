"""
pathoai/tumor_bulk/connected_components.py
==========================================
Tumor Bulk Connected Components.

Labels connected components of the tumor bed and filters out small noisy elements.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.2
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.ndimage


def label_and_filter_tumor_regions(
    tumor_bed: np.ndarray,
    mpp: float,
    min_area_um2: float = 10000.0,
) -> Tuple[np.ndarray, int]:
    """Labels connected components in the tumor bed and filters out small regions.

    Parameters
    ----------
    tumor_bed : np.ndarray
        Binary tumor bed mask.
    mpp : float
        Microns per pixel resolution.
    min_area_um2 : float
        Minimum physical area of a valid tumor region in square microns.

    Returns
    -------
    Tuple[np.ndarray, int]
        - Labeled regions mask of shape (H, W) where elements are 0, 1, 2, ...
        - Count of remaining regions.
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be positive. Got: {mpp}")
    if min_area_um2 < 0:
        raise ValueError(f"min_area_um2 must be non-negative. Got: {min_area_um2}")

    if not np.any(tumor_bed):
        return np.zeros(tumor_bed.shape, dtype=np.int32), 0

    # 1. Label components
    labeled_mask, num_features = scipy.ndimage.label(tumor_bed > 0)

    if num_features == 0:
        return labeled_mask, 0

    # 2. Filter components by area
    # Convert min_area_um2 to pixel count
    pixel_area_um2 = mpp**2
    min_pixels = min_area_um2 / pixel_area_um2

    # Find size of each label component
    label_sizes = scipy.ndimage.sum_labels(
        np.ones_like(labeled_mask),
        labeled_mask,
        index=np.arange(1, num_features + 1),
    )

    # Convert label_sizes to scalar or 1D array
    if num_features == 1:
        label_sizes = np.array([label_sizes])

    # Re-map valid labels to contiguous integers 1..K
    filtered_mask = np.zeros_like(labeled_mask, dtype=np.int32)
    next_label = 1

    for idx, size in enumerate(label_sizes):
        orig_label = idx + 1
        if size >= min_pixels:
            filtered_mask[labeled_mask == orig_label] = next_label
            next_label += 1

    return filtered_mask, next_label - 1
