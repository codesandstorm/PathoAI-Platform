"""
tests/unit/tumor_bulk/test_roi_generator.py
===========================================
Unit tests for ROI generator.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.tumor_bulk.roi_generator import generate_rois


class TestROIGenerator:
    """Test ROI generator."""

    def test_generate_rois_empty(self):
        """Test with empty mask."""
        mask = np.zeros((10, 10), dtype=np.int32)
        rois = generate_rois(mask, mpp=0.5)
        assert len(rois) == 0

    def test_generate_rois_invalid_mpp(self):
        """Test validation check."""
        mask = np.ones((5, 5), dtype=np.int32)
        with pytest.raises(ValueError, match="mpp must be positive"):
            generate_rois(mask, mpp=0.0)

    def test_generate_rois_standard(self):
        """Test generating ROIs with correct metadata."""
        # Mask with two separate labeled components:
        # Region 1: label 1 at [2:5, 2:5] (9 pixels)
        # Region 2: label 2 at [10:14, 10:14] (16 pixels)
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[2:5, 2:5] = 1
        mask[10:14, 10:14] = 2

        rois = generate_rois(mask, mpp=0.5)
        assert len(rois) == 2

        # ROI 1
        roi1 = rois[0]
        assert roi1["roi_id"] == 1
        assert roi1["bbox_yxyx"] == [2, 2, 4, 4]
        assert roi1["centroid_xy"] == (3.0, 3.0)
        assert roi1["area_px"] == 9
        # Area = 9 * 0.25 = 2.25 um^2
        assert roi1["area_um2"] == 2.25
        assert roi1["perimeter_um"] > 0.0
        assert len(roi1["contours"]) == 1

        # ROI 2
        roi2 = rois[1]
        assert roi2["roi_id"] == 2
        assert roi2["bbox_yxyx"] == [10, 10, 13, 13]
        assert roi2["centroid_xy"] == (11.5, 11.5)
        assert roi2["area_px"] == 16
        # Area = 16 * 0.25 = 4.0 um^2
        assert roi2["area_um2"] == 4.0
