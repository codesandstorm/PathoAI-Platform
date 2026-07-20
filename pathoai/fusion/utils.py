"""
pathoai/fusion/utils.py
=======================
Spatial Fusion Helper Routines.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.10
"""

from __future__ import annotations

import math
from typing import Tuple

from pathoai.core.types import Point


def euclidean_distance_um(p1: Point, p2: Point, mpp: float) -> float:
    """Computes Euclidean distance between two Points in microns."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dist_px = math.sqrt(dx * dx + dy * dy)
    return dist_px * mpp
