"""
tests/unit/core/utils/test_math_utils.py
=========================================
Unit tests for pathoai.core.utils.math_utils.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import numpy as np
import pytest

from pathoai.core.utils.math_utils import (
    bootstrap_confidence_interval,
    clip_boxes_to_image,
    compute_box_areas,
    compute_box_centroids,
    compute_iou,
    find_best_pyramid_level,
    level_to_slide_coords,
    pixels_to_mm2,
    pixels_to_um2,
    slide_to_level_coords,
    um2_to_mm2,
)


class TestAreaConversions:
    """Tests for pixel-area to physical-area conversion functions."""

    def test_pixels_to_um2_basic(self):
        """1000 pixels at 0.5 μm/px = 250 μm²."""
        result = pixels_to_um2(1000, mpp=0.5)
        assert abs(result - 250.0) < 1e-6

    def test_pixels_to_um2_zero_pixels(self):
        """Zero pixels → zero area."""
        assert pixels_to_um2(0, mpp=0.5) == 0.0

    def test_pixels_to_mm2_one_mm2(self):
        """1 mm² = 4,000,000 pixels at 0.5 μm/px."""
        # 1 mm² = 1e6 μm². Each pixel = 0.25 μm². So pixels = 1e6 / 0.25 = 4e6
        assert abs(pixels_to_mm2(4_000_000, mpp=0.5) - 1.0) < 1e-6

    def test_um2_to_mm2_basic(self):
        """1,000,000 μm² = 1 mm²."""
        assert abs(um2_to_mm2(1_000_000.0) - 1.0) < 1e-9

    def test_um2_to_mm2_zero(self):
        """Zero μm² → zero mm²."""
        assert um2_to_mm2(0.0) == 0.0


class TestCoordinateTransforms:
    """Tests for slide-level to pyramid-level coordinate transforms."""

    def test_slide_to_level_at_level0(self):
        """At level 0, downsample=1.0, coordinates unchanged."""
        assert slide_to_level_coords(100, 200, level_downsample=1.0) == (100, 200)

    def test_slide_to_level_at_4x_downsample(self):
        """At 4× downsample, coordinates are divided by 4."""
        x, y = slide_to_level_coords(400, 800, level_downsample=4.0)
        assert x == 100
        assert y == 200

    def test_level_to_slide_inverse(self):
        """level_to_slide_coords is the inverse of slide_to_level_coords."""
        original_x, original_y = 1024, 2048
        downsample = 4.0
        lx, ly = slide_to_level_coords(original_x, original_y, downsample)
        restored_x, restored_y = level_to_slide_coords(lx, ly, downsample)
        # Note: integer truncation may cause small rounding (within 1 pixel)
        assert abs(restored_x - original_x) <= downsample
        assert abs(restored_y - original_y) <= downsample


class TestBoxCentroids:
    """Tests for bounding box centroid computation."""

    def test_centroid_of_unit_square(self):
        """Centroid of [0, 0, 2, 2] is [1, 1]."""
        boxes = np.array([[0, 0, 2, 2]], dtype=np.float32)
        centroids = compute_box_centroids(boxes)
        np.testing.assert_allclose(centroids, [[1.0, 1.0]], atol=1e-6)

    def test_centroids_batch(self):
        """Batch centroid computation is correct for multiple boxes."""
        boxes = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)
        centroids = compute_box_centroids(boxes)
        np.testing.assert_allclose(centroids, [[5.0, 5.0], [15.0, 15.0]], atol=1e-6)

    def test_centroid_output_shape(self):
        """Output shape must be (N, 2)."""
        boxes = np.zeros((5, 4), dtype=np.float32)
        centroids = compute_box_centroids(boxes)
        assert centroids.shape == (5, 2)

    def test_invalid_shape_raises(self):
        """Wrong input shape must raise ValueError."""
        with pytest.raises(ValueError):
            compute_box_centroids(np.zeros((5, 3)))  # Wrong: (N, 3) not (N, 4)


class TestBoxAreas:
    """Tests for bounding box area computation."""

    def test_area_unit_square(self):
        """Area of [0, 0, 2, 2] is 4."""
        boxes = np.array([[0, 0, 2, 2]], dtype=np.float32)
        areas = compute_box_areas(boxes)
        np.testing.assert_allclose(areas, [4.0])

    def test_zero_width_box_has_zero_area(self):
        """Box with zero width has zero area."""
        boxes = np.array([[5, 0, 5, 10]], dtype=np.float32)  # x1 == x2
        areas = compute_box_areas(boxes)
        np.testing.assert_allclose(areas, [0.0])

    def test_negative_dimension_box_has_zero_area(self):
        """Box with x1 > x2 (inverted) has zero area (clamped to 0)."""
        boxes = np.array([[10, 0, 5, 10]], dtype=np.float32)  # x1 > x2
        areas = compute_box_areas(boxes)
        np.testing.assert_allclose(areas, [0.0])


class TestIoU:
    """Tests for Intersection over Union computation."""

    def test_identical_boxes_iou_is_one(self):
        """Identical boxes have IoU = 1.0."""
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        iou = compute_iou(boxes, boxes)
        np.testing.assert_allclose(iou, [[1.0]], atol=1e-6)

    def test_non_overlapping_boxes_iou_is_zero(self):
        """Non-overlapping boxes have IoU = 0.0."""
        a = np.array([[0, 0, 5, 5]], dtype=np.float32)
        b = np.array([[10, 10, 20, 20]], dtype=np.float32)
        iou = compute_iou(a, b)
        np.testing.assert_allclose(iou, [[0.0]], atol=1e-6)

    def test_iou_output_shape(self):
        """IoU output shape must be (M, N) for M and N input boxes."""
        a = np.zeros((3, 4), dtype=np.float32)
        b = np.zeros((5, 4), dtype=np.float32)
        iou = compute_iou(a, b)
        assert iou.shape == (3, 5)

    def test_iou_is_symmetric(self):
        """IoU(A, B) == IoU(B, A).T."""
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 5, 15, 15]], dtype=np.float32)
        np.testing.assert_allclose(compute_iou(a, b), compute_iou(b, a).T, atol=1e-6)


class TestFindBestPyramidLevel:
    """Tests for pyramid level selection."""

    def test_finds_closest_mpp_level(self):
        """Selects level closest to target MPP."""
        # Level 0: 0.25 μm/px (40×), Level 1: 1.0 μm/px (10×), Level 2: 4.0 μm/px (2.5×)
        best = find_best_pyramid_level(
            level_downsamples=[1.0, 4.0, 16.0],
            target_mpp=0.5,
            slide_mpp=0.25,
        )
        # Level 0 MPP = 0.25, Level 1 MPP = 1.0. Target 0.5 is closest to Level 0.
        assert best == 0

    def test_selects_level1_for_20x(self):
        """At 40× slide, target 0.5 μm/px selects level with 4× downsample."""
        best = find_best_pyramid_level(
            level_downsamples=[1.0, 2.0, 4.0, 8.0],
            target_mpp=0.50,
            slide_mpp=0.25,
        )
        # Level 0: 0.25, Level 1: 0.5, Level 2: 1.0, Level 3: 2.0
        assert best == 1  # Level 1: MPP=0.50 — exact match


class TestClipBoxes:
    """Tests for bounding box clipping."""

    def test_boxes_within_image_unchanged(self):
        """Boxes fully within image boundaries are not modified."""
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        clipped = clip_boxes_to_image(boxes, image_width=100, image_height=100)
        np.testing.assert_allclose(clipped, boxes)

    def test_boxes_outside_image_are_clipped(self):
        """Boxes extending outside image are clipped to image boundaries."""
        boxes = np.array([[-5, -5, 110, 110]], dtype=np.float32)
        clipped = clip_boxes_to_image(boxes, image_width=100, image_height=100)
        np.testing.assert_allclose(clipped, [[0, 0, 100, 100]])


class TestBootstrapCI:
    """Tests for bootstrap confidence interval estimation."""

    def test_ci_bounds_are_ordered(self):
        """Lower CI bound must be ≤ upper bound."""
        values = np.array([10, 20, 15, 25, 18], dtype=np.float64)
        lower, upper = bootstrap_confidence_interval(values, n_resamples=200)
        assert lower <= upper

    def test_ci_contains_true_mean(self):
        """95% CI should contain the sample mean most of the time."""
        rng = np.random.RandomState(42)
        values = rng.normal(loc=20.0, scale=5.0, size=100).astype(np.float64)
        lower, upper = bootstrap_confidence_interval(
            values, n_resamples=500, confidence=0.95
        )
        sample_mean = np.mean(values)
        assert lower <= sample_mean <= upper, (
            f"CI [{lower:.2f}, {upper:.2f}] does not contain mean {sample_mean:.2f}"
        )

    def test_single_value_ci_width_is_zero(self):
        """CI of a single repeated value has width ≈ 0."""
        values = np.full(10, 42.0, dtype=np.float64)
        lower, upper = bootstrap_confidence_interval(values, n_resamples=100)
        assert abs(upper - lower) < 1e-6

    def test_ci_reproducible_with_same_seed(self):
        """Same seed produces identical CI bounds."""
        values = np.array([5, 10, 15, 20, 25], dtype=np.float64)
        lo1, hi1 = bootstrap_confidence_interval(values, seed=42)
        lo2, hi2 = bootstrap_confidence_interval(values, seed=42)
        assert lo1 == lo2
        assert hi1 == hi2
