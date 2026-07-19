"""
tests/unit/fusion/test_spatial_ops.py
=====================================
Unit tests for spatial operations module.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.fusion.spatial_ops import (
    calculate_mask_area,
    extract_tumor_associated_stroma,
    extract_tumor_bed,
    filter_points_in_mask,
)


class TestSpatialOps:
    """Test spatial operations."""

    def test_extract_tumor_bed_empty(self):
        """Test with empty mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        res = extract_tumor_bed(mask, mpp=0.5, dilation_dist_um=1.0)
        assert not np.any(res)

    def test_extract_tumor_bed_invalid_mpp(self):
        """Test with non-positive mpp."""
        mask = np.ones((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="mpp must be positive"):
            extract_tumor_bed(mask, mpp=0.0)

    def test_extract_tumor_bed_invalid_dilation(self):
        """Test with negative dilation distance."""
        mask = np.ones((5, 5), dtype=np.uint8)
        with pytest.raises(ValueError, match="dilation_dist_um must be non-negative"):
            extract_tumor_bed(mask, mpp=0.5, dilation_dist_um=-1.0)

    def test_extract_tumor_bed_dilation_and_fill(self):
        """Test dilation and hole filling on a hollow square."""
        # Hollow square of size 7x7
        mask = np.zeros((9, 9), dtype=np.uint8)
        # Outer boundary of square
        mask[1:8, 1] = 1
        mask[1:8, 7] = 1
        mask[1, 1:8] = 1
        mask[7, 1:8] = 1
        # The center at (4,4) is hollow (0)

        # No dilation, just hole filling
        res_fill = extract_tumor_bed(mask, mpp=0.5, dilation_dist_um=0.0)
        # Center should be filled (True)
        assert res_fill[4, 4] == True

        # Dilate by 1 pixel (dilation_dist_um = 0.5 um, mpp = 0.5 um/pixel -> 1 pixel radius)
        res_dilate = extract_tumor_bed(mask, mpp=0.5, dilation_dist_um=0.5)
        # The entire square and its expanded border should be filled
        assert res_dilate[4, 4] == True
        assert res_dilate[0, 4] == True  # outer dilated row

    def test_extract_tumor_associated_stroma(self):
        """Test logical intersection of tumor bed and raw stroma."""
        bed = np.zeros((5, 5), dtype=bool)
        bed[1:4, 1:4] = True

        stroma = np.zeros((5, 5), dtype=bool)
        stroma[2:5, 2:5] = True

        # Intersection is (2,2) and (3,3) region
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
        mask[2:5, 2:5] = 1  # x_idx, y_idx in [2, 3, 4]

        # Points at level-0
        # point1: (24.0, 24.0) -> mapped indices (3, 3) -> inside mask (Yes)
        # point2: (16.0, 16.0) -> mapped indices (2, 2) -> inside mask (Yes)
        # point3: (8.0, 8.0) -> mapped indices (1, 1) -> outside mask (No)
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
