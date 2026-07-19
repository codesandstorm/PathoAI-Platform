"""
tests/unit/tumor_bulk/test_morphology.py
========================================
Unit tests for tumor bed morphology operations.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np
import pytest

from pathoai.tumor_bulk.morphology import extract_tumor_bed


class TestMorphology:
    """Test tumor bed morphology."""

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
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[1:8, 1] = 1
        mask[1:8, 7] = 1
        mask[1, 1:8] = 1
        mask[7, 1:8] = 1

        # No dilation, just hole filling
        res_fill = extract_tumor_bed(mask, mpp=0.5, dilation_dist_um=0.0)
        assert res_fill[4, 4] == True

        # Dilate by 1 pixel (dilation_dist_um = 0.5 um, mpp = 0.5 um/pixel -> 1 pixel radius)
        res_dilate = extract_tumor_bed(mask, mpp=0.5, dilation_dist_um=0.5)
        assert res_dilate[4, 4] == True
        assert res_dilate[0, 4] == True
