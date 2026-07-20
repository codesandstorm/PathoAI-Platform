"""
tests/unit/validation/test_correlation.py
==========================================
Unit tests for CorrelationEngine.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.validation.correlation import CorrelationEngine


class TestCorrelationEngine:
    """Test Pearson, Spearman, and R^2 correlation math."""

    def test_compute_correlations(self):
        """Test statistical correlation calculations."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 22.0, 29.0, 41.0, 51.0])

        engine = CorrelationEngine()
        corrs = engine.compute_correlations(y_true, y_pred)

        assert corrs["pearson_r"] > 0.95
        assert corrs["spearman_r"] > 0.95
        assert corrs["r2"] > 0.90
