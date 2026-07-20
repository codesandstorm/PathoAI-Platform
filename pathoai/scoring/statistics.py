"""
pathoai/scoring/statistics.py
=============================
Clinical Statistics Engine.

Computes non-AI statistics, cell counts, physical stroma area (mm^2),
and lymphocyte density (cells/mm^2) directly from FusionResult objects.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.3
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.types import FusionResult


class StatisticsEngine:
    """Computes physical and numerical statistics from FusionResult DTOs."""

    def compute_statistics(self, fusion_result: FusionResult) -> Dict[str, Any]:
        """Calculates slide statistics from a FusionResult.

        Parameters
        ----------
        fusion_result : FusionResult
            Spatial fusion result container.

        Returns
        -------
        Dict[str, Any]
            Calculated numerical and area statistics dictionary.
        """
        # Calculate total physical stromal area in mm^2 across associated ROIs
        roi_areas = {}
        total_stromal_area_mm2 = 0.0

        for sd in fusion_result.spatial_detections:
            roi_id = sd.roi.roi_id
            if roi_id not in roi_areas:
                roi_areas[roi_id] = sd.roi.area_um2 / 1_000_000.0  # um^2 to mm^2

        total_stromal_area_mm2 = sum(roi_areas.values()) if roi_areas else 1.0  # Default 1.0 mm^2 guard if empty

        # Filter stromal lymphocytes
        stromal_lymphocytes = fusion_result.stromal_cells

        # Compute cell density in cells per mm^2
        density = (
            float(stromal_lymphocytes) / total_stromal_area_mm2
            if total_stromal_area_mm2 > 0
            else 0.0
        )

        return {
            "slide_id": fusion_result.slide_id,
            "total_cells": fusion_result.total_cells,
            "intratumoral_cells": fusion_result.intratumoral_cells,
            "stromal_lymphocytes": stromal_lymphocytes,
            "distant_cells": fusion_result.distant_cells,
            "rejected_cells": fusion_result.rejected_cells,
            "stromal_area_mm2": float(total_stromal_area_mm2),
            "lymphocyte_density_per_mm2": float(density),
        }
