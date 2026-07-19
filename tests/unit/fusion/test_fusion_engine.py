"""
tests/unit/fusion/test_fusion_engine.py
=======================================
Unit tests for FusionEngine coordinator class.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.core.exceptions import ValidationError
from pathoai.fusion.fusion_engine import FusionEngine


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

        # 10x10 masks
        tumor = np.zeros((10, 10), dtype=np.uint8)
        tumor[4, 4] = 1

        stroma = np.zeros((10, 10), dtype=np.uint8)
        stroma[4:7, 4:7] = 1

        # Centroids at level 0 (downsample from 0.5 to 0.25 is 2x)
        # points inside stroma index range [4:7, 4:7]:
        # (8.0, 8.0) -> mapped index (4, 4) -> inside stroma (Yes)
        # (12.0, 12.0) -> mapped index (6, 6) -> inside stroma (Yes)
        # (18.0, 18.0) -> mapped index (9, 9) -> outside stroma (No)
        centroids = [
            (8.0, 8.0),
            (12.0, 12.0),
            (18.0, 18.0),
        ]

        # Process slide without patch coords (bootstrap returns fallback score)
        res = engine.process_slide(tumor, stroma, centroids)

        # Stroma area: stroma_mask inside dilated tumor bed.
        # Dilated tumor bed with radius 2 from (4,4) covers:
        # y in [2:7], x in [2:7]
        # Intersection with stroma (4:7, 4:7) is 3x3 = 9 pixels.
        # Area in mm^2 = 6 * 0.25 / 1e6 = 1.5e-6 mm^2
        assert res["stroma_area_mm2"] == 1.5e-6
        assert res["n_lymphocytes_total"] == 3
        assert res["n_lymphocytes_in_stroma"] == 1

        # Both stroma and lymphocyte counts are below safety limits:
        # Expected flags: "INSUFFICIENT_STROMA" and "INSUFFICIENT_LYMPHOCYTES"
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

        ci_lower, ci_upper = engine._calculate_bootstrap_ci(patch_coords, fallback_score=20.0)
        assert 10.0 <= ci_lower <= ci_upper <= 30.0
