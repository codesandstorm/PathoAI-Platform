"""
pathoai/scoring/categorization.py
=================================
STIL Categorizer Engine.

Assigns clinical risk categories to sTIL scores using ClinicalRules.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.8
"""

from __future__ import annotations

from pathoai.scoring.clinical_rules import ClinicalRules


class STILCategorizer:
    """Categorizes sTIL scores into clinical buckets."""

    def __init__(self, rules: ClinicalRules = None) -> None:
        self.rules = rules or ClinicalRules()

    def categorize(self, score_percent: float) -> str:
        return self.rules.get_category(score_percent)
