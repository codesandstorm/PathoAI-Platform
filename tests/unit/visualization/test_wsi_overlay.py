"""
tests/unit/visualization/test_wsi_overlay.py
============================================
Unit tests for slide-level visualization overlays.

Tests cover:
- draw_patch_grid_overlay grid rendering, downsample scaling, and error bounds
- draw_tissue_mask_overlay transparency blending, color application, and shape checks

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathoai.core.exceptions import ValidationError
from pathoai.wsi.metadata.metadata import SlideMetadata
from pathoai.wsi.patches.patches import PatchMetadata
from pathoai.visualization.wsi_overlay import (
    draw_patch_grid_overlay,
    draw_tissue_mask_overlay,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_metadata():
    return SlideMetadata(
        path=Path("/tmp/slide.svs"),
        vendor="generic",
        dimensions=(4000, 3000),
        level_count=1,
        level_dimensions=[(4000, 3000)],
        level_downsamples=[1.0],
        mpp_x=0.5,
        mpp_y=0.5,
        magnification=20.0,
        associated_images=[],
        properties={},
    )


# ---------------------------------------------------------------------------
# draw_patch_grid_overlay
# ---------------------------------------------------------------------------

class TestDrawPatchGridOverlay:
    """Verifies that patch grids are drawn correctly on thumbnails."""

    def test_grid_overlay_drawn_successfully(self, mock_metadata):
        # Thumbnail: 400x300 (downsample = 10x)
        thumb = np.zeros((300, 400, 3), dtype=np.uint8)
        patch = PatchMetadata(
            slide_path=mock_metadata.path,
            x_level0=1000,
            y_level0=1000,
            patch_size=500,  # 500 px at target MPP 0.50
            target_mpp=0.50,
            tissue_coverage=1.0,
        )

        # Draw overlay in blue (0, 0, 255)
        result = draw_patch_grid_overlay(
            thumb,
            patches_metadata=[patch],
            metadata=mock_metadata,
            color=(0, 0, 255),
            thickness=1,
        )

        assert result.shape == (300, 400, 3)
        assert result.dtype == np.uint8
        # Mapped coord: x_thumb = 1000/10 = 100, y_thumb = 1000/10 = 100.
        # Patch size 500 -> size_thumb = 500/10 = 50.
        # So top edge at y=100, x=100 to 150 should contain blue color (0,0,255)
        assert np.array_equal(result[100, 100], [0, 0, 255])
        # Center of patch at y=120, x=120 should still be background black (0,0,0)
        assert np.array_equal(result[120, 120], [0, 0, 0])

    def test_does_not_modify_original_thumbnail(self, mock_metadata):
        thumb = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_patch_grid_overlay(thumb, [], mock_metadata)
        # Verify original thumb is untouched (all zeros)
        assert np.all(thumb == 0)

    def test_raises_on_empty_thumbnail(self, mock_metadata):
        with pytest.raises(ValidationError, match="image is empty"):
            draw_patch_grid_overlay(np.zeros((0, 0, 3), dtype=np.uint8), [], mock_metadata)

    def test_raises_on_invalid_thumbnail_channels(self, mock_metadata):
        with pytest.raises(ValidationError, match="Expected RGB thumbnail"):
            draw_patch_grid_overlay(np.zeros((100, 100), dtype=np.uint8), [], mock_metadata)


# ---------------------------------------------------------------------------
# draw_tissue_mask_overlay
# ---------------------------------------------------------------------------

class TestDrawTissueMaskOverlay:
    """Verifies that semi-transparent tissue overlays are blended correctly."""

    def test_mask_overlay_blended_successfully(self):
        # Thumbnail is all dark gray: 50
        thumb = np.full((100, 100, 3), 50, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:50, :] = 1  # Upper half has tissue

        # Blend with alpha=0.5, green color (0, 200, 0)
        result = draw_tissue_mask_overlay(
            thumb,
            mask,
            alpha=0.5,
            color=(0, 200, 0),
        )

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
        # Upper half (tissue): blended color = 50 * 0.5 + 200 * 0.5 = 125 in Green
        assert result[10, 10, 1] == 125
        # Lower half (background): unblended color = 50
        assert result[60, 10, 1] == 50

    def test_raises_on_shape_mismatch(self):
        thumb = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 50), dtype=np.uint8)
        with pytest.raises(ValidationError, match="does not match"):
            draw_tissue_mask_overlay(thumb, mask)

    def test_raises_on_invalid_alpha(self):
        thumb = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValidationError, match="alpha"):
            draw_tissue_mask_overlay(thumb, mask, alpha=1.2)
