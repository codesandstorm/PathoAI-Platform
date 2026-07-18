"""
tests/unit/wsi/test_patches.py
==============================
Unit tests for the Patch Engine.

Tests cover:
- PatchMetadata serialization (to_dict)
- PatchSampler parameter validation
- sample_patches gridding and tissue coverage filtering
- Error handling on closed slide readers and invalid mask shapes

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pathoai.core.exceptions import DataError
from pathoai.wsi.metadata.metadata import SlideMetadata
from pathoai.wsi.patches.patches import PatchMetadata, PatchSampler
from pathoai.wsi.readers.base import BaseWSI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_metadata():
    return SlideMetadata(
        path=Path("/tmp/slide.svs"),
        vendor="aperio",
        dimensions=(8000, 6000),  # width, height at level 0
        level_count=3,
        level_dimensions=[(8000, 6000), (2000, 1500), (500, 375)],
        level_downsamples=[1.0, 4.0, 16.0],
        mpp_x=0.5,  # 20x slide
        mpp_y=0.5,
        magnification=20.0,
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
# PatchMetadata
# ---------------------------------------------------------------------------

class TestPatchMetadata:
    """Verifies PatchMetadata container and serialization."""

    def test_construction_and_dict_serialization(self):
        meta = PatchMetadata(
            slide_path=Path("/fake/slide.svs"),
            x_level0=1000,
            y_level0=2000,
            patch_size=512,
            target_mpp=0.50,
            tissue_coverage=0.875,
        )
        assert meta.x_level0 == 1000
        assert meta.tissue_coverage == 0.875

        d = meta.to_dict()
        assert d["x_level0"] == 1000
        assert d["tissue_coverage"] == 0.875
        assert d["patch_size"] == 512
        assert json.loads(json.dumps(d)) is not None


# ---------------------------------------------------------------------------
# PatchSampler
# ---------------------------------------------------------------------------

class TestPatchSampler:
    """Verifies gridding coordinates generation and tissue mask filtering."""

    def test_initialization_validates_parameters(self):
        sampler = PatchSampler(patch_size=512, stride=256, target_mpp=0.5, min_tissue_coverage=0.2)
        assert sampler.patch_size == 512
        assert sampler.stride == 256

        with pytest.raises(ValueError, match="patch_size must be positive"):
            PatchSampler(patch_size=0)
        with pytest.raises(ValueError, match="stride must be positive"):
            PatchSampler(stride=-10)
        with pytest.raises(ValueError, match="target_mpp must be positive"):
            PatchSampler(target_mpp=0.0)
        with pytest.raises(ValueError, match="min_tissue_coverage must be in"):
            PatchSampler(min_tissue_coverage=1.5)

    def test_raises_if_reader_closed(self, mock_reader, mock_metadata):
        mock_reader.is_open = False
        sampler = PatchSampler()
        with pytest.raises(DataError, match="Slide is closed"):
            sampler.sample_patches(mock_reader, mock_metadata, np.zeros((10, 10)))

    def test_raises_on_invalid_mask_dimensions(self, mock_reader, mock_metadata):
        sampler = PatchSampler()
        # 3D mask instead of 2D
        with pytest.raises(DataError, match="Expected 2-D tissue mask"):
            sampler.sample_patches(mock_reader, mock_metadata, np.zeros((10, 10, 3)))
        # Empty mask
        with pytest.raises(DataError, match="Invalid tissue mask shape"):
            sampler.sample_patches(mock_reader, mock_metadata, np.zeros((0, 10)))

    def test_sample_patches_filters_by_tissue_coverage(self, mock_reader, mock_metadata):
        """Grids slide and keeps only patches with tissue coverage >= threshold."""
        # Slide size: 8000x6000.
        # Patch size: 1000, Stride: 1000 at target MPP = 0.50 (matches slide MPP 0.50, ratio = 1.0).
        # Patch size in level 0: 1000, Stride: 1000.
        # Grid dimensions:
        # x: range(0, 8000 - 1000 + 1, 1000) -> 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000 (8 coordinates)
        # y: range(0, 6000 - 1000 + 1, 1000) -> 0, 1000, 2000, 3000, 4000, 5000 (6 coordinates)
        # Total potential patches = 8 * 6 = 48.
        # Let's create a tissue mask of size 80x60 (downsample = 100x).
        # Upper half of mask (y: 0-30) is all tissue (1), lower half is background (0).
        # Thus, patches in rows y=0, y=1000, y=2000 (3 rows) are tissue (coverage 1.0).
        # Patches in row y=3000 mapping to rows 30-40 in mask will have 0.0 coverage (since y=30-40 is 0).
        # Expected patches kept: 8 cols * 3 rows = 24 patches.
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[:30, :] = 1  # Upper half is tissue

        sampler = PatchSampler(patch_size=1000, stride=1000, target_mpp=0.50, min_tissue_coverage=0.25)
        patches = sampler.sample_patches(mock_reader, mock_metadata, mask)

        assert len(patches) == 24
        # Verify coordinates of first patch
        assert patches[0].x_level0 == 0
        assert patches[0].y_level0 == 0
        assert patches[0].patch_size == 1000
        assert patches[0].tissue_coverage == 1.0

        # Verify that all returned patches have y_level0 < 3000
        assert all(p.y_level0 < 3000 for p in patches)
