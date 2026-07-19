"""
tests/unit/fusion/test_stil_computer.py
======================================
Unit tests for sTIL computer module.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import pytest

from pathoai.fusion.stil_computer import compute_stil_score


class TestSTILComputer:
    """Test sTIL scoring computer."""

    def test_compute_stil_score_standard(self):
        """Test normal values."""
        # 100 lymphocytes, 2.0 mm^2 stroma area
        # density = 100 / 2.0 = 50.0 cells/mm^2
        # cell area = pi * 5^2 = 78.5398 um^2
        # total cells area = 100 * 78.5398 = 7853.98 um^2
        # stroma area in um^2 = 2.0 * 1,000,000 = 2,000,000 um^2
        # estimated_pct = (7853.98 / 2,000,000) * 100 = 0.392699%
        res = compute_stil_score(n_lymphocytes=100, stroma_area_mm2=2.0, lymphocyte_diameter_um=10.0)
        assert res["n_lymphocytes"] == 100
        assert res["stroma_area_mm2"] == 2.0
        assert res["density_per_mm2"] == 50.0
        assert pytest.approx(res["estimated_pct"], rel=1e-5) == 0.392699

    def test_compute_stil_score_zero_stroma(self):
        """Test zero stroma prevents division-by-zero."""
        res = compute_stil_score(n_lymphocytes=50, stroma_area_mm2=0.0)
        assert res["density_per_mm2"] == 0.0
        assert res["estimated_pct"] == 0.0

    def test_compute_stil_score_clipped(self):
        """Test percentage clipping at 100%."""
        # Extremely high count in tiny stroma area
        res = compute_stil_score(n_lymphocytes=1_000_000, stroma_area_mm2=0.01)
        assert res["estimated_pct"] == 100.0

    def test_compute_stil_score_invalid_inputs(self):
        """Test negative values raise ValueError."""
        with pytest.raises(ValueError, match="n_lymphocytes must be non-negative"):
            compute_stil_score(-1, 1.0)
        with pytest.raises(ValueError, match="stroma_area_mm2 must be non-negative"):
            compute_stil_score(1, -1.0)
