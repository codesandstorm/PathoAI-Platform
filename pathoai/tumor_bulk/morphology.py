"""
pathoai/tumor_bulk/morphology.py
================================
Tumor Bulk Morphology.

Implements morphological operations (dilation, hole filling, cleanups)
on tumor masks to identify the contiguous tumor bed.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.1
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def extract_tumor_bed(
    tumor_mask: np.ndarray,
    mpp: float,
    dilation_dist_um: float = 500.0,
) -> np.ndarray:
    """Dilates the tumor mask and fills holes to define the contiguous tumor bed.

    Parameters
    ----------
    tumor_mask : np.ndarray
        Binary mask (boolean or uint8) where 1 indicates invasive tumor nests.
    mpp : float
        Microns per pixel resolution of the mask.
    dilation_dist_um : float
        Clinical margin distance in microns to expand the tumor boundary.

    Returns
    -------
    np.ndarray
        Contiguous binary tumor bed mask (same shape as input, boolean type).
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be positive. Got: {mpp}")
    if dilation_dist_um < 0:
        raise ValueError(f"dilation_dist_um must be non-negative. Got: {dilation_dist_um}")

    if not np.any(tumor_mask):
        return np.zeros(tumor_mask.shape, dtype=bool)

    # Convert micron distance to pixel radius
    radius_px = int(np.round(dilation_dist_um / mpp))
    
    if radius_px == 0:
        # If dilation distance is too small to resolve, fill holes on raw mask
        return scipy.ndimage.binary_fill_holes(tumor_mask > 0)

    # Create a circular (disk) structuring element
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    disk = x**2 + y**2 <= radius_px**2

    # Perform dilation and fill holes to connect disjoint tumor nests
    dilated = scipy.ndimage.binary_dilation(tumor_mask > 0, structure=disk)
    tumor_bed = scipy.ndimage.binary_fill_holes(dilated)

    return tumor_bed
