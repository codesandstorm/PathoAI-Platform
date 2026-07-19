"""
tests/unit/fusion/test_aggregator.py
====================================
Unit tests for patch aggregator module.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.fusion.aggregator import PatchAggregator


class TestPatchAggregator:
    """Test patch aggregator."""

    def test_patch_aggregator_empty(self):
        """Test with empty patches."""
        agg = PatchAggregator(stride=256)
        res = agg.aggregate()
        assert res["slide_score"] == 0.0
        assert res["total_stroma_area_mm2"] == 0.0
        assert res["total_lymphocytes"] == 0
        assert np.array_equal(res["heatmap"], np.zeros((1, 1), dtype=np.float32))

    def test_patch_aggregator_invalid_stride(self):
        """Test with non-positive stride."""
        with pytest.raises(ValueError, match="stride must be positive"):
            PatchAggregator(stride=0)

    def test_patch_aggregator_standard(self):
        """Test normal aggregation and heatmap layout."""
        agg = PatchAggregator(stride=256)

        # Add 3 patches:
        # Patch 1: grid (0, 0), score = 10.0, stroma = 400,000 um^2 (0.4 mm^2)
        # Patch 2: grid (0, 1), score = 20.0, stroma = 600,000 um^2 (0.6 mm^2)
        # Patch 3: grid (1, 1), score = 30.0, stroma = 0.0
        agg.add_patch(x_level0=0, y_level0=0, score=10.0, stroma_area_um2=400_000.0, n_lymphocytes=15)
        agg.add_patch(x_level0=256, y_level0=0, score=20.0, stroma_area_um2=600_000.0, n_lymphocytes=25)
        agg.add_patch(x_level0=256, y_level0=256, score=30.0, stroma_area_um2=0.0, n_lymphocytes=0)

        res = agg.aggregate()

        # Total stroma area = 1,000,000 um^2 = 1.0 mm^2
        assert res["total_stroma_area_mm2"] == 1.0
        assert res["total_lymphocytes"] == 40

        # Weighted score = (10 * 400,000 + 20 * 600,000) / 1,000,000 = 16.0
        assert res["slide_score"] == 16.0

        # Heatmap grid should be 2x2
        # grid coordinates:
        # (0,0) -> 10.0
        # (0,1) -> 20.0
        # (1,0) -> 0.0 (no patch added there)
        # (1,1) -> 30.0
        expected_heatmap = np.array([
            [10.0, 20.0],
            [0.0, 30.0]
        ], dtype=np.float32)

        assert np.array_equal(res["heatmap"], expected_heatmap)
