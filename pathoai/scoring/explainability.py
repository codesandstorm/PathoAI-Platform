"""
pathoai/scoring/explainability.py
=================================
sTIL Explainability Engine.

Generates human-readable, transparent clinical rationales detailing total cells,
stromal area, cell density, excluded cells, and 95% confidence intervals.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.9
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


class STILExplainability:
    """Generates explainable scoring summaries."""

    def generate_explanation(
        self,
        score_percent: float,
        stats: Dict[str, Any],
        ci: Tuple[float, float],
        category: str,
    ) -> str:
        """Generates detailed explanation string.

        Parameters
        ----------
        score_percent : float
            Primary sTIL score percentage.
        stats : Dict[str, Any]
            Calculated statistics.
        ci : Tuple[float, float]
            95% confidence interval bounds.
        category : str
            Clinical category.

        Returns
        -------
        str
            Formatted explanation text.
        """
        explanation = (
            f"Evaluated sTIL score of {score_percent:.2f}% (Category: {category}, 95% CI: [{ci[0]:.2f}%, {ci[1]:.2f}%]). "
            f"Calculation based on {stats.get('stromal_lymphocytes', 0):,} stromal lymphocytes "
            f"across {stats.get('stromal_area_mm2', 0.0):.3f} mm² of tumor-associated stroma "
            f"(density: {stats.get('lymphocyte_density_per_mm2', 0.0):.1f} cells/mm²). "
            f"Total cells evaluated: {stats.get('total_cells', 0):,}; "
            f"Excluded non-stromal/distant cells: {stats.get('distant_cells', 0) + stats.get('rejected_cells', 0):,}."
        )
        return explanation
