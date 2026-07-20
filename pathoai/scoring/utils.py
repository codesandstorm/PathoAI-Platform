"""
pathoai/scoring/utils.py
========================
Clinical Scoring Helper Routines.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.15
"""

from __future__ import annotations


def convert_um2_to_mm2(area_um2: float) -> float:
    """Converts area in square microns (um^2) to square millimeters (mm^2)."""
    return float(area_um2 / 1_000_000.0)


def convert_mm2_to_um2(area_mm2: float) -> float:
    """Converts area in square millimeters (mm^2) to square microns (um^2)."""
    return float(area_mm2 * 1_000_000.0)
