"""
tests/unit/tumor_bulk/test_roi_generator.py
===========================================
Unit tests for ROI generator and TumorROI mappings.

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
        """Test generating ROIs with correct metadata and advanced stats."""
        # Mask with two separate labeled components:
        # Region 1: label 1 at [2:5, 2:5] (9 pixels)
        # Region 2: label 2 at [10:14, 10:14] (16 pixels)
        mask = np.zeros((20, 20), dtype=np.int32)
        mask[2:5, 2:5] = 1
        mask[10:14, 10:14] = 2

        rois = generate_rois(mask, mpp=0.5)
        assert len(rois) == 2

        # ROI 1 (Square region)
        roi1 = rois[0]
        assert roi1.roi_id == 1
        assert roi1.bbox.to_yxyx() == [2, 2, 5, 5]
        assert roi1.centroid.x == 3.0
        assert roi1.centroid.y == 3.0
        assert roi1.area_px == 9
        assert roi1.area_um2 == 2.25
        assert roi1.perimeter_um > 0.0
        
        # Advanced statistics checks
        assert roi1.solidity == 1.0  # solid square has solidity 1.0
        assert 0.0 <= roi1.eccentricity <= 1.0
        assert roi1.compactness > 0.0
        assert roi1.equivalent_diameter_um > 0.0
        assert len(roi1.contours) == 1

        # ROI 2
        roi2 = rois[1]
        assert roi2.roi_id == 2
        assert roi2.bbox.to_yxyx() == [10, 10, 14, 14]
        assert roi2.centroid.x == 11.5
        assert roi2.centroid.y == 11.5
        assert roi2.area_px == 16
        assert roi2.area_um2 == 4.0
        assert roi2.solidity == 1.0
