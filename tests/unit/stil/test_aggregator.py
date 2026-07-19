"""
tests/unit/stil/test_aggregator.py
==================================
Unit tests for patch aggregator in stil package.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.stil.aggregator import PatchAggregator


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

        agg.add_patch(x_level0=0, y_level0=0, score=10.0, stroma_area_um2=400_000.0, n_lymphocytes=15)
        agg.add_patch(x_level0=256, y_level0=0, score=20.0, stroma_area_um2=600_000.0, n_lymphocytes=25)
        agg.add_patch(x_level0=256, y_level0=256, score=30.0, stroma_area_um2=0.0, n_lymphocytes=0)

        res = agg.aggregate()

        assert res["total_stroma_area_mm2"] == 1.0
        assert res["total_lymphocytes"] == 40
        assert res["slide_score"] == 16.0

        expected_heatmap = np.array([
            [10.0, 20.0],
            [0.0, 30.0]
        ], dtype=np.float32)

        assert np.array_equal(res["heatmap"], expected_heatmap)
