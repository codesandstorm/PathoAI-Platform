"""
tests/unit/tumor_bulk/test_connected_components.py
==================================================
Unit tests for connected components labeling and filtering.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.tumor_bulk.connected_components import label_and_filter_tumor_regions


class TestConnectedComponents:
    """Test connected components labeling."""

    def test_label_and_filter_tumor_regions_empty(self):
        """Test with empty mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        res, count = label_and_filter_tumor_regions(mask, mpp=0.5)
        assert count == 0
        assert not np.any(res)

    def test_label_and_filter_tumor_regions_invalid_params(self):
        """Test validation check."""
        mask = np.ones((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="mpp must be positive"):
            label_and_filter_tumor_regions(mask, mpp=0.0)
        with pytest.raises(ValueError, match="min_area_um2 must be non-negative"):
            label_and_filter_tumor_regions(mask, mpp=0.5, min_area_um2=-1.0)

    def test_label_and_filter_tumor_regions_filtering(self):
        """Test filtering out regions below size threshold."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        # Region 1: 2x2 = 4 pixels (area at mpp=0.5: 4 * 0.25 = 1.0 um^2)
        mask[2:4, 2:4] = 1
        # Region 2: 8x8 = 64 pixels (area at mpp=0.5: 64 * 0.25 = 16.0 um^2)
        mask[10:18, 10:18] = 1

        # Case A: Filter area >= 2.0 um^2 (Region 1 should be dropped, Region 2 remains)
        res_a, count_a = label_and_filter_tumor_regions(mask, mpp=0.5, min_area_um2=2.0)
        assert count_a == 1
        assert not np.any(res_a[2:4, 2:4])
        assert np.all(res_a[10:18, 10:18] == 1)

        # Case B: Filter area >= 0.5 um^2 (Both regions remain, labeled 1 and 2)
        res_b, count_b = label_and_filter_tumor_regions(mask, mpp=0.5, min_area_um2=0.5)
        assert count_b == 2
        assert np.all(res_b[2:4, 2:4] == 1)
        assert np.all(res_b[10:18, 10:18] == 2)
