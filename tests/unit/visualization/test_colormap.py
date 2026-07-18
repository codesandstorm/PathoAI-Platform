"""
tests/unit/visualization/test_colormap.py
==========================================
Unit tests for pathoai.visualization.colormap.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.8
"""

from __future__ import annotations

import numpy as np
import pytest

from pathoai.visualization.colormap import (
    ColorMapError,
    build_matplotlib_colormap,
    build_tissue_lut,
    colorize_mask,
    get_cell_colormap_uint8,
    get_class_color_legend,
    get_tissue_colormap_bgr,
    get_tissue_colormap_float,
    get_tissue_colormap_uint8,
    rgb_float_to_uint8,
    rgb_to_bgr,
    rgb_uint8_to_float,
)


# ---------------------------------------------------------------------------
# rgb_uint8_to_float
# ---------------------------------------------------------------------------

class TestRgbUint8ToFloat:
    def test_white_converts_to_one(self):
        assert rgb_uint8_to_float((255, 255, 255)) == (1.0, 1.0, 1.0)

    def test_black_converts_to_zero(self):
        assert rgb_uint8_to_float((0, 0, 0)) == (0.0, 0.0, 0.0)

    def test_mid_gray_approximation(self):
        r, g, b = rgb_uint8_to_float((128, 128, 128))
        assert abs(r - 128 / 255) < 1e-9

    def test_raises_on_out_of_range_channel(self):
        with pytest.raises(ColorMapError):
            rgb_uint8_to_float((256, 0, 0))

    def test_raises_on_negative_channel(self):
        with pytest.raises(ColorMapError):
            rgb_uint8_to_float((-1, 0, 0))


# ---------------------------------------------------------------------------
# rgb_float_to_uint8
# ---------------------------------------------------------------------------

class TestRgbFloatToUint8:
    def test_one_converts_to_255(self):
        assert rgb_float_to_uint8((1.0, 1.0, 1.0)) == (255, 255, 255)

    def test_zero_converts_to_zero(self):
        assert rgb_float_to_uint8((0.0, 0.0, 0.0)) == (0, 0, 0)

    def test_raises_on_out_of_range(self):
        with pytest.raises(ColorMapError):
            rgb_float_to_uint8((1.1, 0.0, 0.0))

    def test_roundtrip_uint8_to_float_to_uint8(self):
        original = (200, 100, 50)
        result = rgb_float_to_uint8(rgb_uint8_to_float(original))
        assert result == original


# ---------------------------------------------------------------------------
# rgb_to_bgr
# ---------------------------------------------------------------------------

class TestRgbToBgr:
    def test_pure_red_becomes_pure_blue_in_bgr(self):
        assert rgb_to_bgr((255, 0, 0)) == (0, 0, 255)

    def test_gray_unchanged(self):
        assert rgb_to_bgr((128, 128, 128)) == (128, 128, 128)

    def test_double_conversion_is_identity(self):
        original = (100, 150, 200)
        assert rgb_to_bgr(rgb_to_bgr(original)) == original


# ---------------------------------------------------------------------------
# get_tissue_colormap_uint8
# ---------------------------------------------------------------------------

class TestGetTissueColormapUint8:
    def test_returns_dict(self):
        cm = get_tissue_colormap_uint8()
        assert isinstance(cm, dict)

    def test_contains_background_class(self):
        cm = get_tissue_colormap_uint8()
        assert 0 in cm

    def test_all_values_are_rgb_tuples(self):
        cm = get_tissue_colormap_uint8()
        for class_id, color in cm.items():
            assert len(color) == 3
            assert all(0 <= v <= 255 for v in color)

    def test_contains_six_classes(self):
        cm = get_tissue_colormap_uint8()
        assert len(cm) == 6


# ---------------------------------------------------------------------------
# get_tissue_colormap_float
# ---------------------------------------------------------------------------

class TestGetTissueColormapFloat:
    def test_returns_float_values(self):
        cm = get_tissue_colormap_float()
        for class_id, color in cm.items():
            assert all(isinstance(v, float) for v in color)

    def test_all_values_in_unit_range(self):
        cm = get_tissue_colormap_float()
        for class_id, color in cm.items():
            for v in color:
                assert 0.0 <= v <= 1.0, f"class {class_id}: value {v} out of [0,1]"


# ---------------------------------------------------------------------------
# get_tissue_colormap_bgr
# ---------------------------------------------------------------------------

class TestGetTissueColormapBgr:
    def test_red_class_has_blue_in_channel_0(self):
        """Tumor class (1) is red in RGB → blue in BGR channel 0 should be high."""
        cm_rgb = get_tissue_colormap_uint8()
        cm_bgr = get_tissue_colormap_bgr()
        r_rgb, _, _ = cm_rgb[1]  # Red channel in RGB
        _, _, r_bgr = cm_bgr[1]  # Red channel is now at index 2 in BGR
        assert r_rgb == r_bgr

    def test_bgr_has_same_classes_as_rgb(self):
        assert set(get_tissue_colormap_bgr().keys()) == set(get_tissue_colormap_uint8().keys())


# ---------------------------------------------------------------------------
# build_tissue_lut
# ---------------------------------------------------------------------------

class TestBuildTissueLut:
    def test_returns_numpy_array(self):
        lut = build_tissue_lut()
        assert isinstance(lut, np.ndarray)

    def test_shape_is_n_classes_by_4(self):
        lut = build_tissue_lut()
        assert lut.shape[1] == 4  # RGBA

    def test_dtype_is_uint8(self):
        lut = build_tissue_lut()
        assert lut.dtype == np.uint8

    def test_alpha_channel_matches_input(self):
        lut = build_tissue_lut(alpha=180)
        # All non-zero class rows should have alpha=180
        assert lut[1, 3] == 180

    def test_raises_on_invalid_alpha(self):
        with pytest.raises(ColorMapError):
            build_tissue_lut(alpha=256)
        with pytest.raises(ColorMapError):
            build_tissue_lut(alpha=-1)


# ---------------------------------------------------------------------------
# colorize_mask
# ---------------------------------------------------------------------------

class TestColorizeMask:
    def test_output_shape_is_hw4(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        result = colorize_mask(mask)
        assert result.shape == (8, 8, 4)

    def test_output_dtype_is_uint8(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        assert colorize_mask(mask).dtype == np.uint8

    def test_background_gets_background_color(self):
        from pathoai.core.constants import TISSUE_CLASS_COLORS
        mask = np.zeros((4, 4), dtype=np.uint8)
        result = colorize_mask(mask)
        bg_color = TISSUE_CLASS_COLORS[0]
        # All pixels should be background color
        assert result[0, 0, 0] == bg_color[0]
        assert result[0, 0, 1] == bg_color[1]
        assert result[0, 0, 2] == bg_color[2]

    def test_raises_on_non_2d_mask(self):
        with pytest.raises(ColorMapError, match="2-D"):
            colorize_mask(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_raises_on_out_of_range_class_id(self):
        mask = np.array([[255]], dtype=np.uint8)
        with pytest.raises(ColorMapError, match="class ID"):
            colorize_mask(mask)


# ---------------------------------------------------------------------------
# build_matplotlib_colormap
# ---------------------------------------------------------------------------

class TestBuildMatplotlibColormap:
    def test_builds_listed_colormap(self):
        try:
            from matplotlib.colors import ListedColormap
            cmap = build_matplotlib_colormap()
            assert isinstance(cmap, ListedColormap)
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_colormap_has_n_classes_colors(self):
        try:
            cmap = build_matplotlib_colormap()
            assert cmap.N == 6
        except ImportError:
            pytest.skip("matplotlib not installed")


# ---------------------------------------------------------------------------
# get_class_color_legend
# ---------------------------------------------------------------------------

class TestGetClassColorLegend:
    def test_returns_list_of_dicts(self):
        legend = get_class_color_legend()
        assert isinstance(legend, list)
        assert all(isinstance(entry, dict) for entry in legend)

    def test_each_entry_has_required_keys(self):
        legend = get_class_color_legend()
        for entry in legend:
            for key in ("class_id", "class_name", "rgb_uint8", "hex_color"):
                assert key in entry

    def test_hex_color_format(self):
        legend = get_class_color_legend()
        for entry in legend:
            h = entry["hex_color"]
            assert h.startswith("#")
            assert len(h) == 7

    def test_contains_all_tissue_classes(self):
        from pathoai.core.constants import TISSUE_CLASSES
        legend = get_class_color_legend()
        legend_ids = {e["class_id"] for e in legend}
        assert legend_ids == set(TISSUE_CLASSES.keys())
