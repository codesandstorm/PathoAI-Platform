"""
tests/unit/scoring/test_clinical_rules.py
==========================================
Unit tests for ClinicalRules.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.scoring.clinical_rules import ClinicalRules


class TestClinicalRules:
    """Test ClinicalRules thresholds."""

    def test_get_category(self):
        """Test score categorization thresholds."""
        rules = ClinicalRules(low_threshold=10.0, high_threshold=50.0)

        assert rules.get_category(5.0) == "Low"
        assert rules.get_category(25.0) == "Intermediate"
        assert rules.get_category(60.0) == "High"
