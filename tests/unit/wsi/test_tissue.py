"""
tests/unit/wsi/test_tissue.py
=============================
Unit tests for the Tissue Detection Engine.

Tests cover:
- TissueDetector initialization parameters
- detect_tissue method with synthetic RGB thumbnails (tissue vs background)
- Morphological cleanup and stray component filtering
- Error handling on empty inputs and invalid shapes

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

import numpy as np
import pytest

from pathoai.core.exceptions import DataError
from pathoai.wsi.tissue.tissue import TissueDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_thumbnail(
    h: int = 100,
    w: int = 100,
    tissue_ratio: float = 0.5,
    add_noise_pixel_count: int = 0,
) -> np.ndarray:
    """Create a synthetic thumbnail.

    - Background is white/bright: RGB (240, 240, 240)
    - Tissue is darker/colored (e.g., purple/pink H&E): RGB (120, 50, 150)
    """
    img = np.full((h, w, 3), 240, dtype=np.uint8)

    # Fill tissue region
    boundary = int(h * tissue_ratio)
    img[:boundary, :, :] = [120, 50, 150]

    # Add small stray pixels (noise)
    if add_noise_pixel_count > 0:
        # Put noise in background region
        for i in range(add_noise_pixel_count):
            img[h - 5 - i, w // 2] = [120, 50, 150]

    return img


# ---------------------------------------------------------------------------
# TissueDetector
# ---------------------------------------------------------------------------

class TestTissueDetector:
    """Verifies classical tissue detection operations on thumbnails."""

    def test_initialization(self):
        detector = TissueDetector(method="otsu_hsv", morph_kernel_size=5, min_component_pixels=100)
        assert detector.method == "otsu_hsv"
        assert detector.morph_kernel_size == 5
        assert detector.min_component_pixels == 100

        with pytest.raises(ValueError, match="Unsupported tissue detection method"):
            TissueDetector(method="deep_learning")

    def test_detect_tissue_returns_correct_mask_shape_and_range(self):
        thumb = _make_synthetic_thumbnail(64, 64, tissue_ratio=0.4)
        detector = TissueDetector(morph_kernel_size=3, min_component_pixels=10)
        mask, ratio = detector.detect_tissue(thumb)

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        # Expected ratio close to 0.4
        assert abs(ratio - 0.4) < 0.05

    def test_tissue_segmentation_labels_correctly(self):
        """Tissue (darker) must be labeled 1, background (bright) labeled 0."""
        thumb = _make_synthetic_thumbnail(32, 32, tissue_ratio=0.5)
        detector = TissueDetector(morph_kernel_size=3, min_component_pixels=1)
        mask, _ = detector.detect_tissue(thumb)

        # Upper half (tissue) -> 1
        assert np.all(mask[:16, :] == 1)
        # Lower half (background) -> 0
        assert np.all(mask[16:, :] == 0)

    def test_connected_component_noise_filtering(self):
        """Stray components smaller than min_component_pixels are filtered out."""
        # Create thumbnail with 50% tissue and 2 noise pixels
        thumb = _make_synthetic_thumbnail(100, 100, tissue_ratio=0.5, add_noise_pixel_count=2)

        # 1. With small component filter threshold = 1, noise pixels are kept
        detector_keep = TissueDetector(morph_kernel_size=1, min_component_pixels=1)
        mask_keep, _ = detector_keep.detect_tissue(thumb)
        assert np.sum(mask_keep[50:, :]) > 0  # Noise is kept in the background region

        # 2. With component threshold = 10, the 2-pixel noise component is removed
        detector_drop = TissueDetector(morph_kernel_size=1, min_component_pixels=10)
        mask_drop, _ = detector_drop.detect_tissue(thumb)
        assert np.all(mask_drop[50:, :] == 0)  # Noise is filtered out

    def test_raises_on_empty_thumbnail(self):
        detector = TissueDetector()
        with pytest.raises(DataError, match="image is empty"):
            detector.detect_tissue(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_raises_on_invalid_image_channels(self):
        detector = TissueDetector()
        # Grayscale image (2D) instead of 3D RGB
        with pytest.raises(DataError, match="Expected RGB thumbnail"):
            detector.detect_tissue(np.zeros((64, 64), dtype=np.uint8))
        # 4-channel image
        with pytest.raises(DataError, match="Expected RGB thumbnail"):
            detector.detect_tissue(np.zeros((64, 64, 4), dtype=np.uint8))
class TestTissueDetectorMorphCleanups:
    """Verifies morphological cleaning operations fill holes and smooth mask."""

    def test_morphology_fills_isolated_holes(self):
        # Create a thumbnail with a 50% tissue region, but place a small 2x2 white hole in the tissue
        thumb = _make_synthetic_thumbnail(100, 100, tissue_ratio=0.5)
        thumb[10:12, 10:12] = [240, 240, 240]  # White hole

        # With a kernel of 5, the morphological close should close the 2x2 hole
        detector = TissueDetector(morph_kernel_size=5, min_component_pixels=10)
        mask, _ = detector.detect_tissue(thumb)
        assert np.all(mask[10:12, 10:12] == 1)  # The hole has been closed/filled
