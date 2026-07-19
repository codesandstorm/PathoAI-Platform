"""
pathoai/fusion/stil_computer.py
================================
sTIL Scoring and Density Computer.

Implements clinical algorithms to compute lymphocyte densities and estimated
sTIL percentage scores according to international TILs Working Group guidelines.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.2
"""

from __future__ import annotations

import math
from typing import Dict, Union


def compute_stil_score(
    n_lymphocytes: int,
    stroma_area_mm2: float,
    lymphocyte_diameter_um: float = 10.0,
) -> Dict[str, Union[float, int]]:
    """Calculates both lymphocyte density (cells/mm^2) and estimated sTIL percentage.

    Parameters
    ----------
    n_lymphocytes : int
        Number of lymphocytes detected in the tumor-associated stroma.
    stroma_area_mm2 : float
        Total area of tumor-associated stroma in square millimeters.
    lymphocyte_diameter_um : float
        Typical diameter of a lymphocyte in microns (used for area estimation).

    Returns
    -------
    Dict[str, Union[float, int]]
        A dictionary containing:
        - "n_lymphocytes": count of lymphocytes
        - "stroma_area_mm2": area of stroma
        - "density_per_mm2": density of lymphocytes (cells/mm^2)
        - "estimated_pct": estimated sTIL percentage (0.0 to 100.0)
    """
    if n_lymphocytes < 0:
        raise ValueError(f"n_lymphocytes must be non-negative. Got: {n_lymphocytes}")
    if stroma_area_mm2 < 0:
        raise ValueError(f"stroma_area_mm2 must be non-negative. Got: {stroma_area_mm2}")

    # Default metrics when stroma is zero to prevent division-by-zero
    if stroma_area_mm2 == 0.0:
        return {
            "n_lymphocytes": n_lymphocytes,
            "stroma_area_mm2": 0.0,
            "density_per_mm2": 0.0,
            "estimated_pct": 0.0,
        }

    # 1. Density Calculation (cells per mm^2)
    density = float(n_lymphocytes) / stroma_area_mm2

    # 2. Area-based percentage calculation
    # Area of one cell (um^2) = pi * (d / 2)^2
    cell_radius_um = lymphocyte_diameter_um / 2.0
    cell_area_um2 = math.pi * (cell_radius_um**2)
    
    # Total cells area (um^2)
    total_cells_area_um2 = n_lymphocytes * cell_area_um2
    # Convert stroma area from mm^2 to um^2
    stroma_area_um2 = stroma_area_mm2 * 1_000_000.0

    # Estimated sTIL percentage = (Total cells area / Total stroma area) * 100
    estimated_pct = (total_cells_area_um2 / stroma_area_um2) * 100.0

    # Clip percentage to 100.0 (in case of extremely high density / overlap)
    estimated_pct = min(100.0, max(0.0, estimated_pct))

    return {
        "n_lymphocytes": n_lymphocytes,
        "stroma_area_mm2": float(stroma_area_mm2),
        "density_per_mm2": float(density),
        "estimated_pct": float(estimated_pct),
    }
