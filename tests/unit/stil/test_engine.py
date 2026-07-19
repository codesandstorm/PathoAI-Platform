"""
tests/unit/stil/test_engine.py
==============================
Unit tests for FusionEngine coordinator class in stil package.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.core.exceptions import ValidationError
from pathoai.stil.engine import FusionEngine


class TestFusionEngine:
    """Test FusionEngine workflow."""

    def test_fusion_engine_validation_mpp(self):
        """Test non-positive MPP validation."""
        with pytest.raises(ValidationError, match="mpp must be positive"):
            FusionEngine(mpp=0.0)

    def test_fusion_engine_shape_validation(self):
        """Test mask shape matching check."""
        engine = FusionEngine(mpp=0.5)
        tumor = np.ones((5, 5), dtype=np.uint8)
        stroma = np.ones((4, 4), dtype=np.uint8)
        with pytest.raises(ValidationError, match="Shape mismatch"):
            engine.process_slide(tumor, stroma, [])

    def test_process_slide_workflow(self):
        """Test complete workflow with mock masks and points."""
        engine = FusionEngine(
            mpp=0.5,
            dilation_dist_um=1.0,  # 2 pixels radius
            bootstrap_n=100,
            min_stroma_area_mm2=0.5,
            min_lymph_for_confidence=5,
        )

        tumor = np.zeros((10, 10), dtype=np.uint8)
        tumor[4, 4] = 1

        stroma = np.zeros((10, 10), dtype=np.uint8)
        stroma[4:7, 4:7] = 1

        centroids = [
            (8.0, 8.0),
            (12.0, 12.0),
            (18.0, 18.0),
        ]

        res = engine.process_slide(tumor, stroma, centroids)

        # Area in mm^2 = 6 * 0.25 / 1e6 = 1.5e-6 mm^2
        assert res["stroma_area_mm2"] == 1.5e-6
        assert res["n_lymphocytes_total"] == 3
        assert res["n_lymphocytes_in_stroma"] == 1

        assert "INSUFFICIENT_STROMA" in res["quality_flags"]
        assert "INSUFFICIENT_LYMPHOCYTES" in res["quality_flags"]

    def test_bootstrap_ci_calculation(self):
        """Test bootstrap confidence interval range check."""
        engine = FusionEngine(
            mpp=0.5,
            bootstrap_n=50,
            seed=42,
        )

        patch_coords = [
            {"x": 0, "y": 0, "score": 10.0, "stroma_area": 0.4},
            {"x": 256, "y": 0, "score": 20.0, "stroma_area": 0.6},
            {"x": 256, "y": 256, "score": 30.0, "stroma_area": 0.2},
        ]

        res = engine.process_slide(
            tumor_mask=np.zeros((5, 5), dtype=np.uint8),
            stroma_mask=np.zeros((5, 5), dtype=np.uint8),
            lymphocyte_centroids=[],
            patch_coords=patch_coords,
        )
        ci_lower, ci_upper = res["ci_95"]
        assert 10.0 <= ci_lower <= ci_upper <= 30.0
