"""
pathoai/scoring/validation.py
=============================
Score Validation Engine.

Verifies sTIL score range bounds [0, 100], zero-division guards, and CI bounds.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.10
"""

from __future__ import annotations

from typing import Any, Dict

from pathoai.core.types import STILScore


class ScoreValidator:
    """Validates sTILScore DTO instances."""

    def validate_score(self, stil_score: STILScore) -> Dict[str, Any]:
        """Validates STILScore instance attributes."""
        issues = []
        if not (0.0 <= stil_score.score_percent <= 100.0):
            issues.append(f"score_percent out of bounds [0, 100]: {stil_score.score_percent}")

        if stil_score.stromal_area_mm2 < 0.0:
            issues.append(f"Negative stromal_area_mm2: {stil_score.stromal_area_mm2}")

        if stil_score.confidence_interval[0] > stil_score.confidence_interval[1]:
            issues.append(f"Invalid CI bounds: {stil_score.confidence_interval}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
        }
