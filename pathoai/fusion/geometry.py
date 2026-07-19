"""
pathoai/fusion/geometry.py
==========================
Geometry Area Calculations.

Calculates physical areas of binary masks in square millimeters (mm^2).

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 8.3
"""

from __future__ import annotations

import numpy as np


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
