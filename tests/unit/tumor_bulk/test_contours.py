"""
tests/unit/tumor_bulk/test_contours.py
======================================
Unit tests for contour coordinate extraction.

Author: PathoAI Research Team
Created: 2026-07-19
"""

import numpy as np

from pathoai.tumor_bulk.contours import extract_region_contours


class TestContours:
    """Test contour extraction."""

    def test_extract_region_contours_empty(self):
        """Test with empty mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        contours = extract_region_contours(mask)
        assert len(contours) == 0

    def test_extract_region_contours_simple_square(self):
        """Test extracting contours of a simple square."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 1

        contours = extract_region_contours(mask)
        assert len(contours) == 1
        c = contours[0]

        # Coordinates should form a closed loop of shape (N, 2)
        assert c.shape[1] == 2
        # The contour coordinates should be in the range [1.5, 5.5] (due to level=0.5 marching squares)
        assert np.min(c) >= 1.5
        assert np.max(c) <= 5.5
