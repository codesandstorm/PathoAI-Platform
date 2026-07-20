"""
pathoai/scoring/clinical_rules.py
=================================
Clinical Guideline Rules Engine.

Implements guideline-based threshold rules and clinical risk stratification criteria.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.7
"""

from __future__ import annotations

from typing import Dict, List


class ClinicalRules:
    """Configurable clinical guideline threshold rules."""

    def __init__(self, low_threshold: float = 10.0, high_threshold: float = 50.0) -> None:
        """
        Parameters
        ----------
        low_threshold : float
            Upper limit for 'Low' sTIL percentage (default < 10.0%).
        high_threshold : float
            Lower limit for 'High' sTIL percentage (default >= 50.0%).
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def get_category(self, score_percent: float) -> str:
        """Maps sTIL percentage score to clinical category string."""
        if score_percent < self.low_threshold:
            return "Low"
        elif score_percent < self.high_threshold:
            return "Intermediate"
        else:
            return "High"

    def get_interpretation(self, score_percent: float) -> str:
        """Generates clinical interpretation string."""
        cat = self.get_category(score_percent)
        if cat == "Low":
            return (
                f"Low sTIL score ({score_percent:.1f}%). Tumor exhibits limited stromal immune infiltration (<{self.low_threshold}%)."
            )
        elif cat == "Intermediate":
            return (
                f"Intermediate sTIL score ({score_percent:.1f}%). Tumor exhibits moderate stromal immune infiltration ({self.low_threshold}-{self.high_threshold}%)."
            )
        else:
            return (
                f"High sTIL score ({score_percent:.1f}%). Tumor exhibits strong, prominent stromal immune infiltration (>={self.high_threshold}%)."
            )
