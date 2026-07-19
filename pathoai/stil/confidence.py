"""
pathoai/stil/confidence.py
==========================
Clinical Confidence and Quality Flags.

Assigns clinical quality flags based on stroma size, cell count, and boundary values.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 9.4
"""

from __future__ import annotations

from typing import List


def assign_quality_flags(
    stroma_area_mm2: float,
    n_lymph_in_stroma: int,
    ci_lower: float,
    ci_upper: float,
    estimated_pct: float,
    min_stroma_area_mm2: float = 0.5,
    min_lymph_for_confidence: int = 50,
) -> List[str]:
    """Assigns clinical quality flags based on stroma and TIL measurements.

    Parameters
    ----------
    stroma_area_mm2 : float
        Total area of stroma in square millimeters.
    n_lymph_in_stroma : int
        Count of lymphocytes inside the stroma.
    ci_lower : float
        Bootstrap CI lower bound.
    ci_upper : float
        Bootstrap CI upper bound.
    estimated_pct : float
        Point estimate sTIL score percentage.
    min_stroma_area_mm2 : float
        Threshold below which INSUFFICIENT_STROMA is flagged.
    min_lymph_for_confidence : int
        Threshold below which INSUFFICIENT_LYMPHOCYTES is flagged.

    Returns
    -------
    List[str]
        List of assigned clinical quality flag strings.
    """
    flags = []

    if stroma_area_mm2 < min_stroma_area_mm2:
        flags.append("INSUFFICIENT_STROMA")

    if n_lymph_in_stroma < min_lymph_for_confidence:
        flags.append("INSUFFICIENT_LYMPHOCYTES")

    ci_width = ci_upper - ci_lower
    if ci_width > 20.0:
        flags.append("LOW_CONFIDENCE")

    # Clinical boundary cases (near 10% or 20% cutoffs)
    for boundary in [10.0, 20.0]:
        if abs(estimated_pct - boundary) <= 2.0:
            flags.append("SCORE_BOUNDARY")
            break

    return flags
