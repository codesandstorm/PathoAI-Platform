"""
tests/unit/core/test_constants.py
==================================
Unit tests for pathoai.core.constants.

Verifies that all constants are defined, have the correct types,
and satisfy scientific constraints (e.g., class IDs are consecutive,
color values are in [0, 255]).

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1
"""

import pytest

from pathoai.core import constants as C


class TestTissueClassConstants:
    """Tests for tissue segmentation class definitions."""

    def test_tissue_classes_dict_is_complete(self):
        """TISSUE_CLASSES must define exactly N_TISSUE_CLASSES entries."""
        assert len(C.TISSUE_CLASSES) == C.N_TISSUE_CLASSES

    def test_tissue_class_ids_are_zero_indexed_consecutive(self):
        """Class IDs must start at 0 and be consecutive integers."""
        ids = sorted(C.TISSUE_CLASSES.keys())
        expected = list(range(C.N_TISSUE_CLASSES))
        assert ids == expected, f"Non-consecutive class IDs: {ids}"

    def test_tissue_class_inverse_mapping_is_consistent(self):
        """TISSUE_CLASS_IDS must be the exact inverse of TISSUE_CLASSES."""
        for class_id, name in C.TISSUE_CLASSES.items():
            assert C.TISSUE_CLASS_IDS[name] == class_id

    def test_background_class_id_is_zero(self):
        """Background must be class 0 (convention in segmentation)."""
        assert C.BACKGROUND_CLASS_ID == 0
        assert C.TISSUE_CLASSES[0] == "background"

    def test_stroma_class_id_matches_tiger_schema(self):
        """Stroma must be class 2 to match TIGER dataset annotation schema."""
        assert C.STROMA_CLASS_ID == 2
        assert C.TISSUE_CLASSES[C.STROMA_CLASS_ID] == "stroma"

    def test_all_class_names_are_strings(self):
        """All class names must be non-empty strings."""
        for name in C.TISSUE_CLASSES.values():
            assert isinstance(name, str) and len(name) > 0


class TestCellClassConstants:
    """Tests for cell detection class definitions."""

    def test_cell_classes_background_is_zero(self):
        """Cell background class must be 0."""
        assert C.CELL_BACKGROUND_ID == 0
        assert C.CELL_CLASSES[0] == "background"

    def test_lymphocyte_detection_class_exists(self):
        """Lymphocyte detection class must be defined."""
        assert C.LYMPHOCYTE_DET_CLASS_ID in C.CELL_CLASSES
        assert C.CELL_CLASSES[C.LYMPHOCYTE_DET_CLASS_ID] == "lymphocyte"

    def test_n_cell_classes_equals_dict_length(self):
        """N_CELL_CLASSES must equal number of entries in CELL_CLASSES."""
        assert C.N_CELL_CLASSES == len(C.CELL_CLASSES)


class TestColorMapConstants:
    """Tests for visualization color map constants."""

    def test_tissue_color_map_covers_all_classes(self):
        """Color map must have an entry for every tissue class."""
        for class_id in C.TISSUE_CLASSES:
            assert class_id in C.TISSUE_CLASS_COLORS, (
                f"Class {class_id} ({C.TISSUE_CLASSES[class_id]}) missing from color map"
            )

    def test_all_rgb_values_in_valid_range(self):
        """All RGB color values must be in [0, 255]."""
        for class_id, color in C.TISSUE_CLASS_COLORS.items():
            assert len(color) == 3, f"Color for class {class_id} must be (R, G, B)"
            for channel_val in color:
                assert 0 <= channel_val <= 255, (
                    f"Color channel {channel_val} out of range [0, 255] for class {class_id}"
                )

    def test_overlay_alpha_in_valid_range(self):
        """Segmentation overlay alpha must be between 0 (transparent) and 1 (opaque)."""
        assert 0.0 < C.SEGMENTATION_OVERLAY_ALPHA <= 1.0


class TestSpatialConstants:
    """Tests for spatial and optical constants."""

    def test_segmentation_mpp_is_positive(self):
        """Segmentation target MPP must be positive."""
        assert C.SEGMENTATION_TARGET_MPP > 0

    def test_detection_mpp_is_less_than_segmentation_mpp(self):
        """Detection at higher magnification → lower MPP than segmentation."""
        assert C.DETECTION_TARGET_MPP < C.SEGMENTATION_TARGET_MPP

    def test_um2_to_mm2_conversion_constant(self):
        """Verify unit conversion constant: 1 mm² = 1e6 μm²."""
        area_um2 = 1_000_000.0
        area_mm2 = area_um2 * C.UM2_TO_MM2
        assert abs(area_mm2 - 1.0) < 1e-9


class TestSTILConstants:
    """Tests for sTIL scoring constants."""

    def test_stil_score_range_is_valid(self):
        """sTIL score range must be [0, 100]."""
        assert C.STIL_SCORE_MIN == 0.0
        assert C.STIL_SCORE_MAX == 100.0

    def test_clinical_cutoffs_are_ordered(self):
        """Clinical cutoffs must be in ascending order."""
        assert C.STIL_SCORE_MIN < C.STIL_CUTOFF_LOW < C.STIL_CUTOFF_MODERATE < C.STIL_SCORE_MAX

    def test_min_stroma_area_is_positive(self):
        """Minimum stroma area must be positive."""
        assert C.MIN_STROMA_AREA_MM2 > 0
        assert C.MIN_STROMA_AREA_UM2 > 0

    def test_min_stroma_area_um2_matches_mm2(self):
        """UM2 and MM2 stroma constants must be consistent."""
        assert abs(C.MIN_STROMA_AREA_UM2 - C.MIN_STROMA_AREA_MM2 * 1e6) < 1.0


class TestPatchExtractionConstants:
    """Tests for patch extraction constants."""

    def test_default_patch_size_is_power_of_two(self):
        """Default patch size should be power of 2 for efficient GPU ops."""
        n = C.DEFAULT_PATCH_SIZE
        assert n > 0 and (n & (n - 1)) == 0, f"{n} is not a power of 2"

    def test_default_stride_less_than_patch_size(self):
        """Default stride must be ≤ patch_size (no gap between patches)."""
        assert C.DEFAULT_PATCH_STRIDE <= C.DEFAULT_PATCH_SIZE

    def test_blank_threshold_in_pixel_range(self):
        """Blank patch threshold must be valid pixel intensity value."""
        assert 0 < C.BLANK_PATCH_THRESHOLD <= 255

    def test_min_tissue_coverage_in_unit_range(self):
        """Tissue coverage ratio must be in (0, 1]."""
        assert 0 < C.MIN_TISSUE_COVERAGE_RATIO <= 1.0


class TestSupportedFormats:
    """Tests for WSI format support list."""

    def test_supported_formats_includes_tif(self):
        """Must support .tif for TIGER dataset."""
        assert ".tif" in C.SUPPORTED_WSI_FORMATS

    def test_supported_formats_includes_svs(self):
        """Must support .svs for TCGA dataset."""
        assert ".svs" in C.SUPPORTED_WSI_FORMATS

    def test_all_formats_start_with_dot(self):
        """All format strings must start with a dot."""
        for fmt in C.SUPPORTED_WSI_FORMATS:
            assert fmt.startswith("."), f"Format '{fmt}' must start with '.'"

    def test_supported_formats_is_frozenset(self):
        """SUPPORTED_WSI_FORMATS must be a frozenset (immutable)."""
        assert isinstance(C.SUPPORTED_WSI_FORMATS, frozenset)


class TestImageNetConstants:
    """Tests for ImageNet normalization constants."""

    def test_imagenet_mean_has_three_channels(self):
        """ImageNet mean must have exactly 3 channel values (RGB)."""
        assert len(C.IMAGENET_MEAN) == 3

    def test_imagenet_std_has_three_channels(self):
        """ImageNet std must have exactly 3 channel values."""
        assert len(C.IMAGENET_STD) == 3

    def test_imagenet_values_in_unit_range(self):
        """ImageNet mean and std must be normalized in (0, 1)."""
        for val in C.IMAGENET_MEAN + C.IMAGENET_STD:
            assert 0 < val < 1, f"ImageNet value {val} outside (0, 1)"
