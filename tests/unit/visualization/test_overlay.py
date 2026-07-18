"""
tests/unit/visualization/test_overlay.py
==========================================
Unit tests for pathoai.visualization.overlay.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.8
"""

from __future__ import annotations

import numpy as np
import pytest

from pathoai.visualization.overlay import (
    OverlayError,
    annotate_tissue_regions,
    blend_mask_overlay,
    draw_bounding_boxes,
    make_patch_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img(h: int = 16, w: int = 16) -> np.ndarray:
    """Create a valid (H, W, 3) uint8 image."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _mask(h: int = 16, w: int = 16, class_id: int = 0) -> np.ndarray:
    """Create a (H, W) uint8 mask."""
    return np.full((h, w), class_id, dtype=np.uint8)


# ---------------------------------------------------------------------------
# blend_mask_overlay
# ---------------------------------------------------------------------------

class TestBlendMaskOverlay:
    """Tests for blend_mask_overlay()."""

    def test_output_shape_matches_input(self):
        img = _img(32, 32)
        mask = _mask(32, 32)
        result = blend_mask_overlay(img, mask)
        assert result.shape == (32, 32, 3)

    def test_output_dtype_is_uint8(self):
        result = blend_mask_overlay(_img(), _mask())
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self):
        result = blend_mask_overlay(_img(), _mask())
        assert result.min() >= 0
        assert result.max() <= 255

    def test_alpha_zero_returns_image_unchanged(self):
        """With alpha=0, the output should equal the original image."""
        img = np.full((8, 8, 3), 128, dtype=np.uint8)
        mask = np.ones((8, 8), dtype=np.uint8)
        result = blend_mask_overlay(img, mask, alpha=0.0)
        np.testing.assert_array_equal(result, img)

    def test_raises_on_non_uint8_image(self):
        img = np.zeros((8, 8, 3), dtype=np.float32)
        mask = np.zeros((8, 8), dtype=np.uint8)
        with pytest.raises(OverlayError, match="dtype uint8"):
            blend_mask_overlay(img, mask)

    def test_raises_on_non_3channel_image(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        with pytest.raises(OverlayError):
            blend_mask_overlay(img, mask)

    def test_raises_on_3d_mask(self):
        img = _img()
        mask = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(OverlayError, match="2-D"):
            blend_mask_overlay(img, mask)

    def test_raises_on_shape_mismatch(self):
        img = _img(16, 16)
        mask = _mask(32, 32)
        with pytest.raises(OverlayError, match="does not match"):
            blend_mask_overlay(img, mask)

    def test_raises_on_invalid_alpha(self):
        with pytest.raises(OverlayError, match="alpha"):
            blend_mask_overlay(_img(), _mask(), alpha=1.5)

    def test_all_tissue_class_ids_work(self):
        """All 6 tissue class IDs must not raise."""
        img = _img(8, 8)
        for class_id in range(6):
            mask = _mask(8, 8, class_id=class_id)
            result = blend_mask_overlay(img, mask)
            assert result.shape == (8, 8, 3)


# ---------------------------------------------------------------------------
# draw_bounding_boxes
# ---------------------------------------------------------------------------

class TestDrawBoundingBoxes:
    """Tests for draw_bounding_boxes()."""

    def test_returns_copy_not_original(self):
        """Must not modify the input image."""
        img = _img(64, 64)
        original = img.copy()
        draw_bounding_boxes(img, [(5, 5, 20, 20)])
        np.testing.assert_array_equal(img, original)

    def test_output_shape_matches_input(self):
        img = _img(64, 64)
        result = draw_bounding_boxes(img, [(5, 5, 20, 20)])
        assert result.shape == img.shape

    def test_box_drawn_in_image(self):
        """The top edge of the bounding box must be drawn in the image."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = draw_bounding_boxes(img, [(10, 10, 30, 30)])
        # Top row of box at y=10 should not be all zeros
        assert not np.all(result[10, 10:31] == 0)

    def test_empty_boxes_list_returns_copy(self):
        img = _img(32, 32)
        result = draw_bounding_boxes(img, [])
        np.testing.assert_array_equal(result, img)

    def test_raises_on_labels_length_mismatch(self):
        with pytest.raises(OverlayError, match="must equal len"):
            draw_bounding_boxes(_img(), [(0, 0, 5, 5)], labels=["a", "b"])

    def test_boxes_clipped_to_image_bounds(self):
        """Out-of-bounds boxes must not cause IndexError."""
        img = _img(16, 16)
        # Box extends beyond image — must not raise
        result = draw_bounding_boxes(img, [(-10, -10, 100, 100)])
        assert result.shape == (16, 16, 3)

    def test_labels_and_colors_applied(self):
        """Boxes with matching label colors must be drawn in the right color."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = draw_bounding_boxes(
            img,
            boxes=[(5, 5, 20, 20)],
            labels=["tumor"],
            colors={"tumor": (0, 255, 0)},
        )
        # The box should have green pixels
        assert result[5, 5, 1] == 255  # Green channel at top-left corner

    def test_raises_on_invalid_image(self):
        with pytest.raises(OverlayError):
            draw_bounding_boxes(np.zeros((8, 8), dtype=np.uint8), [])


# ---------------------------------------------------------------------------
# make_patch_grid
# ---------------------------------------------------------------------------

class TestMakePatchGrid:
    """Tests for make_patch_grid()."""

    def test_output_is_numpy_array(self):
        patches = [_img(32, 32) for _ in range(4)]
        result = make_patch_grid(patches, n_cols=2)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_is_uint8(self):
        patches = [_img(8, 8) for _ in range(4)]
        result = make_patch_grid(patches, n_cols=2)
        assert result.dtype == np.uint8

    def test_correct_number_of_rows(self):
        """6 patches in 4 columns → 2 rows."""
        patches = [_img(8, 8) for _ in range(6)]
        result = make_patch_grid(patches, n_cols=4, pad=0)
        # 2 rows × 8px = 16px height
        assert result.shape[0] == 2 * 8

    def test_raises_on_empty_patches_list(self):
        with pytest.raises(OverlayError, match="empty"):
            make_patch_grid([])

    def test_raises_on_inconsistent_patch_shapes(self):
        p1 = _img(16, 16)
        p2 = _img(32, 32)
        with pytest.raises(OverlayError):
            make_patch_grid([p1, p2])

    def test_single_patch_grid(self):
        patches = [_img(16, 16)]
        result = make_patch_grid(patches, n_cols=1, pad=0)
        assert result.shape[0] == 16
        assert result.shape[1] == 16

    def test_output_is_3_channel(self):
        patches = [_img(8, 8) for _ in range(4)]
        result = make_patch_grid(patches, n_cols=2)
        assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# annotate_tissue_regions
# ---------------------------------------------------------------------------

class TestAnnotateTissueRegions:
    """Tests for annotate_tissue_regions()."""

    def test_output_shape_matches_input(self):
        thumbnail = _img(32, 32)
        tissue_mask = _mask(32, 32, class_id=1)
        result = annotate_tissue_regions(thumbnail, tissue_mask)
        assert result.shape == (32, 32, 3)

    def test_output_dtype_is_uint8(self):
        result = annotate_tissue_regions(_img(16, 16), _mask(16, 16))
        assert result.dtype == np.uint8

    def test_raises_on_shape_mismatch(self):
        with pytest.raises(OverlayError):
            annotate_tissue_regions(_img(16, 16), _mask(32, 32))
