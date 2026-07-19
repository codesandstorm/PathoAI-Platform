"""
tests/unit/fusion/test_spatial_ops.py
=====================================
Unit tests for spatial intersection and point filtering operations.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.fusion.geometry import calculate_mask_area
from pathoai.fusion.point_filter import filter_points_in_mask
from pathoai.fusion.spatial_intersection import extract_tumor_associated_stroma


class TestSpatialOps:
    """Test spatial operations."""

    def test_extract_tumor_associated_stroma(self):
        """Test logical intersection of tumor bed and raw stroma."""
        bed = np.zeros((5, 5), dtype=bool)
        bed[1:4, 1:4] = True

        stroma = np.zeros((5, 5), dtype=bool)
        stroma[2:5, 2:5] = True

        expected = np.zeros((5, 5), dtype=bool)
        expected[2:4, 2:4] = True

        res = extract_tumor_associated_stroma(bed, stroma)
        assert np.array_equal(res, expected)

    def test_extract_tumor_associated_stroma_mismatched_shape(self):
        """Test shapes mismatch raises exception."""
        bed = np.ones((5, 5), dtype=bool)
        stroma = np.ones((4, 4), dtype=bool)
        with pytest.raises(ValueError, match="Shape mismatch"):
            extract_tumor_associated_stroma(bed, stroma)

    def test_calculate_mask_area(self):
        """Test area scaling in mm^2."""
        # 100 pixels, mpp = 0.5 um/px
        # area in um^2 = 100 * 0.25 = 25 um^2
        # area in mm^2 = 25 / 1e6 = 0.000025 mm^2
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0:10, 0:10] = 1

        res = calculate_mask_area(mask, mpp=0.5)
        assert pytest.approx(res, rel=1e-6) == 0.000025

    def test_filter_points_in_mask(self):
        """Test coordinate filter mapping from level 0 to mask resolution."""
        # Mask of size 10x10, downsample = 8.0
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 1

        points = [
            (24.0, 24.0),
            (16.0, 16.0),
            (8.0, 8.0),
        ]

        filtered, count = filter_points_in_mask(points, mask, downsample=8.0)
        assert count == 2
        assert np.array_equal(filtered, np.array([[24.0, 24.0], [16.0, 16.0]]))

    def test_filter_points_in_mask_empty(self):
        """Test empty points list returns empty array."""
        mask = np.ones((5, 5), dtype=np.uint8)
        filtered, count = filter_points_in_mask([], mask, downsample=2.0)
        assert count == 0
        assert filtered.shape == (0, 2)
