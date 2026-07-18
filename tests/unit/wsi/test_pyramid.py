"""
tests/unit/wsi/test_pyramid.py
==============================
Unit tests for the WSI Pyramid Engine.

Tests cover:
- level_to_slide_coords & slide_to_level_coords coordinate conversions
- read_patch_at_mpp with matching and mismatched levels (checks BICUBIC resizing)
- get_slide_thumbnail scaling and BILINEAR resizing
- Error handling on closed readers and invalid parameters

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pathoai.core.exceptions import WSIReadError
from pathoai.wsi.metadata.metadata import SlideMetadata
from pathoai.wsi.pyramid.pyramid import (
    get_slide_thumbnail,
    level_to_slide_coords,
    read_patch_at_mpp,
    slide_to_level_coords,
)
from pathoai.wsi.readers.base import BaseWSI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_metadata():
    return SlideMetadata(
        path=Path("/tmp/slide.svs"),
        vendor="aperio",
        dimensions=(8000, 6000),
        level_count=3,
        level_dimensions=[(8000, 6000), (2000, 1500), (500, 375)],
        level_downsamples=[1.0, 4.0, 16.0],
        mpp_x=0.25,
        mpp_y=0.25,
        magnification=40.0,
        associated_images=[],
        properties={},
    )


@pytest.fixture()
def mock_reader():
    reader = MagicMock(spec=BaseWSI)
    reader.is_open = True
    reader.path = Path("/tmp/slide.svs")
    return reader


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

class TestCoordinateConversions:
    """Verifies rounding and scaling in bidirectional coordinate mapping."""

    def test_level_to_slide_coords(self):
        assert level_to_slide_coords((10, 20), 4.0) == (40, 80)
        assert level_to_slide_coords((15, 25), 3.125) == (47, 78)  # rounded

    def test_slide_to_level_coords(self):
        assert slide_to_level_coords((40, 80), 4.0) == (10, 20)
        assert slide_to_level_coords((47, 78), 3.125) == (15, 25)  # rounded


# ---------------------------------------------------------------------------
# read_patch_at_mpp
# ---------------------------------------------------------------------------

class TestReadPatchAtMpp:
    """Verifies patch extraction at target physical microns-per-pixel resolution."""

    def test_raises_if_closed(self, mock_reader, mock_metadata):
        mock_reader.is_open = False
        with pytest.raises(WSIReadError, match="Slide is closed"):
            read_patch_at_mpp(mock_reader, mock_metadata, (0, 0), 0.50, (256, 256))

    def test_raises_on_negative_mpp(self, mock_reader, mock_metadata):
        with pytest.raises(WSIReadError, match="Target MPP must be positive"):
            read_patch_at_mpp(mock_reader, mock_metadata, (0, 0), -0.50, (256, 256))

    def test_raises_on_invalid_patch_size(self, mock_reader, mock_metadata):
        with pytest.raises(WSIReadError, match="Patch size dimensions must be positive"):
            read_patch_at_mpp(mock_reader, mock_metadata, (0, 0), 0.50, (0, 256))
        with pytest.raises(WSIReadError, match="Patch size dimensions must be positive"):
            read_patch_at_mpp(mock_reader, mock_metadata, (0, 0), 0.50, (256, -10))

    def test_reads_region_at_exact_level_without_resize(self, mock_reader, mock_metadata):
        """When the selected level matches the target MPP exactly, no resize is done."""
        # target_mpp = 1.0 (level 1 has downsample 4.0 -> mpp is 0.25 * 4.0 = 1.0)
        # patch_size = (100, 100) -> w_level = 100, h_level = 100
        fake_patch = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_reader.read_region.return_value = fake_patch

        result = read_patch_at_mpp(
            mock_reader,
            mock_metadata,
            location_level0=(400, 800),
            target_mpp=1.0,
            patch_size=(100, 100),
        )

        assert result is fake_patch
        mock_reader.read_region.assert_called_once_with((400, 800), 1, (100, 100))

    def test_reads_region_and_resizes_when_mpp_mismatched(self, mock_reader, mock_metadata):
        """When target MPP lies between levels, reads best level and resizes to target."""
        # target_mpp = 2.0 (level 1 has mpp 1.0, level 2 has mpp 4.0. Closest is level 1, ds=4.0)
        # target patch_size = (100, 100)
        # size at level 1: w_level = 100 * (2.0 / 1.0) = 200 pixels
        fake_raw_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        mock_reader.read_region.return_value = fake_raw_img

        result = read_patch_at_mpp(
            mock_reader,
            mock_metadata,
            location_level0=(100, 100),
            target_mpp=2.0,
            patch_size=(100, 100),
        )

        assert result.shape == (100, 100, 3)
        assert np.all(result == 128)
        # Read call was at level 1, size (200, 200)
        mock_reader.read_region.assert_called_once_with((100, 100), 1, (200, 200))


# ---------------------------------------------------------------------------
# get_slide_thumbnail
# ---------------------------------------------------------------------------

class TestGetSlideThumbnail:
    """Verifies slide thumbnail creation and scaling."""

    def test_raises_if_closed(self, mock_reader, mock_metadata):
        mock_reader.is_open = False
        with pytest.raises(WSIReadError, match="Slide is closed"):
            get_slide_thumbnail(mock_reader, mock_metadata, 500)

    def test_raises_on_invalid_max_dim(self, mock_reader, mock_metadata):
        with pytest.raises(WSIReadError, match="max_dim must be positive"):
            get_slide_thumbnail(mock_reader, mock_metadata, 0)

    def test_reads_correct_level_and_rescales(self, mock_reader, mock_metadata):
        """Finds closest downsample level, reads the region, and scales to max_dim."""
        # max_dim = 1000. Dimensions at level 0 are 8000x6000.
        # target downsample is 8000/1000 = 8.0.
        # level downsamples are [1.0, 4.0, 16.0].
        # 4.0 is closer to 8.0 than 16.0, so it picks level 1.
        # Dimensions at level 1 are (2000, 1500).
        # Thumbnail output size should scale 2000x1500 to max_dim 1000 -> 1000x750.
        fake_level1_img = np.zeros((1500, 2000, 3), dtype=np.uint8)
        mock_reader.read_region.return_value = fake_level1_img

        result = get_slide_thumbnail(mock_reader, mock_metadata, max_dim=1000)

        assert result.shape == (750, 1000, 3)
        mock_reader.read_region.assert_called_once_with((0, 0), 1, (2000, 1500))
