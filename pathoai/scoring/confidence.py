"""
pathoai/scoring/confidence.py
=============================
Scoring Confidence Estimator.

Calculates uncertainty width and quality confidence flags.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.6
"""

from __future__ import annotations

from typing import Dict, Tuple


class ConfidenceEstimator:
    """Evaluates scoring confidence metrics."""

    def evaluate_confidence(
        self, score_percent: float, ci: Tuple[float, float]
    ) -> Dict[str, Any]:
        """Evaluates confidence metrics from sTIL score and confidence interval bounds."""
        ci_lower, ci_upper = ci
        ci_width = round(ci_upper - ci_lower, 2)
        is_wide_ci = ci_width > 20.0
        is_boundary_score = (score_percent < 5.0) or (score_percent > 90.0)

        return {
            "ci_width": ci_width,
            "is_wide_ci": is_wide_ci,
            "is_boundary_score": is_boundary_score,
            "confidence_quality": "low" if is_wide_ci else "high",
        }
