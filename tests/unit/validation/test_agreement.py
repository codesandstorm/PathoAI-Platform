"""
tests/unit/validation/test_agreement.py
========================================
Unit tests for AgreementEngine.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.validation.agreement import AgreementEngine


class TestAgreementEngine:
    """Test ICC and Bland–Altman agreement calculations."""

    def test_compute_icc_and_bland_altman(self):
        """Test ICC score and Bland–Altman limits of agreement."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([12.0, 19.0, 31.0, 38.0, 52.0])

        engine = AgreementEngine()
        icc = engine.compute_icc(y_true, y_pred)
        ba = engine.compute_bland_altman(y_true, y_pred)

        assert 0.8 <= icc <= 1.0
        assert "bland_altman_bias" in ba
        assert ba["bland_altman_lower_limit"] <= ba["bland_altman_bias"] <= ba["bland_altman_upper_limit"]
