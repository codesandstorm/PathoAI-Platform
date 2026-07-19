"""
pathoai/fusion/spatial_intersection.py
======================================
Spatial Intersection Engine.

Extracts tumor-associated stroma by performing intersections of tumor beds and stroma masks.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 8.1
"""

from __future__ import annotations

import numpy as np


def extract_tumor_associated_stroma(
    tumor_bed_mask: np.ndarray,
    stroma_mask: np.ndarray,
) -> np.ndarray:
    """Extracts stroma lying within the borders of the tumor bed.

    Parameters
    ----------
    tumor_bed_mask : np.ndarray
        Binary mask representing the contiguous tumor bed.
    stroma_mask : np.ndarray
        Binary mask representing the raw segmented stroma.

    Returns
    -------
    np.ndarray
        Binary mask representing the tumor-associated stroma (boolean type).
    """
    if tumor_bed_mask.shape != stroma_mask.shape:
        raise ValueError(
            f"Shape mismatch: tumor_bed_mask {tumor_bed_mask.shape} vs stroma_mask {stroma_mask.shape}"
        )

    return np.logical_and(tumor_bed_mask > 0, stroma_mask > 0)
