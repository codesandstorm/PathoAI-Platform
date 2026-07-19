"""
pathoai/fusion/spatial_ops.py
==============================
Spatial Operations Engine.

Implements morphological operations (dilation, hole-filling, intersections)
to define the tumor bed (tumor bulk margin) and extract tumor-associated stroma,
along with scaling computations and cell centroid filters.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.1
"""

from __future__ import annotations

from typing import List, Tuple

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


def calculate_mask_area(mask: np.ndarray, mpp: float) -> float:
    """Computes the physical area of a binary mask in square millimeters (mm^2).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    mpp : float
        Microns per pixel resolution.

    Returns
    -------
    float
        Physical area in mm^2.
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be positive. Got: {mpp}")

    pixel_count = int(np.sum(mask > 0))
    # Area in um^2 = pixel_count * mpp^2
    # Area in mm^2 = Area in um^2 / 1,000,000
    area_mm2 = (pixel_count * (mpp**2)) / 1_000_000.0
    return float(area_mm2)


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
