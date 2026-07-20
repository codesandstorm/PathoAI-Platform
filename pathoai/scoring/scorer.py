"""
pathoai/scoring/scorer.py
=========================
International TIL Working Group sTIL Scorer Engine.

Computes stromal Tumor Infiltrating Lymphocyte (sTIL) percentage and cell density
following International TIL Working Group guidelines.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.4
"""

from __future__ import annotations

import math

from pathoai.core.types import FusionResult
from pathoai.scoring.registry import register_scorer
from pathoai.scoring.statistics import StatisticsEngine


@register_scorer("tiger_working_group")
class sTILScorer:
    """Clinical sTIL Scorer implementation following International Working Group guidelines."""

    def __init__(self, lymphocyte_diameter_um: float = 10.0) -> None:
        """
        Parameters
        ----------
        lymphocyte_diameter_um : float
            Average physical lymphocyte diameter in microns (default 10.0 um).
        """
        if lymphocyte_diameter_um <= 0:
            raise ValueError(f"lymphocyte_diameter_um must be positive. Got: {lymphocyte_diameter_um}")
        self.lymphocyte_diameter_um = lymphocyte_diameter_um
        self.radius_um = lymphocyte_diameter_um / 2.0
        # Single cell area in um^2
        self.single_cell_area_um2 = math.pi * (self.radius_um ** 2)
        self.stats_engine = StatisticsEngine()

    def compute_stil_score_percent(self, fusion_result: FusionResult) -> float:
        """Computes sTIL percentage score [0.0, 100.0].

        Parameters
        ----------
        fusion_result : FusionResult
            Spatial fusion results DTO.

        Returns
        -------
        float
            sTIL percentage score rounded to 2 decimal places.
        """
        stats = self.stats_engine.compute_statistics(fusion_result)
        stromal_cells = stats["stromal_lymphocytes"]
        stromal_area_mm2 = stats["stromal_area_mm2"]

        if stromal_area_mm2 <= 0.0 or stromal_cells == 0:
            return 0.0

        # Convert stroma area from mm^2 to um^2 (1 mm^2 = 1,000,000 um^2)
        stromal_area_um2 = stromal_area_mm2 * 1_000_000.0
        total_lymphocyte_area_um2 = stromal_cells * self.single_cell_area_um2

        # sTIL % = (Total Lymphocyte Area / Stromal Area) * 100
        raw_score = (total_lymphocyte_area_um2 / stromal_area_um2) * 100.0
        clamped_score = max(0.0, min(100.0, raw_score))
        return float(round(clamped_score, 2))
