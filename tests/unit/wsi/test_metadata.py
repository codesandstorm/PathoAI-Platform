"""
tests/unit/wsi/test_metadata.py
================================
Unit tests for the WSI Metadata Engine.

Tests cover:
- SlideMetadata container & serialization
- extract_metadata function with standard property keys
- Vendor detection logic
- Standard and fallback MPP parser algorithms
- Standard, vendor-specific, and inferred objective magnification
- Metadata validation bounds and exception checking

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pathoai.core.exceptions import MetadataExtractionError
from pathoai.wsi.metadata.metadata import SlideMetadata, extract_metadata
from pathoai.wsi.readers.base import BaseWSI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_reader(
    properties: dict[str, str],
    dimensions: tuple[int, int] = (1000, 1000),
    level_count: int = 3,
    level_dimensions: list[tuple[int, int]] | None = None,
    level_downsamples: list[float] | None = None,
    associated_images: list[str] | None = None,
    is_open: bool = True,
    path: str = "/fake/slide.svs",
) -> MagicMock:
    """Create a mock reader implementing BaseWSI with defined properties."""
    reader = MagicMock(spec=BaseWSI)
    reader.is_open = is_open
    reader.path = Path(path)
    reader.properties = properties
    reader.level_count = level_count
    reader.level_dimensions = level_dimensions or [dimensions, (500, 500), (250, 250)]
    reader.level_downsamples = level_downsamples or [1.0, 2.0, 4.0]
    reader.associated_images = associated_images or ["label", "thumbnail"]
    return reader


# ---------------------------------------------------------------------------
# SlideMetadata Container
# ---------------------------------------------------------------------------

class TestSlideMetadataContainer:
    """Verifies SlideMetadata behaves as a correct frozen dataclass."""

    def test_construction_and_dict_serialization(self):
        meta = SlideMetadata(
            path=Path("/tmp/slide.svs"),
            vendor="leica",
            dimensions=(8000, 6000),
            level_count=2,
            level_dimensions=[(8000, 6000), (4000, 3000)],
            level_downsamples=[1.0, 2.0],
            mpp_x=0.5,
            mpp_y=0.5,
            magnification=20.0,
            associated_images=["macro"],
            properties={"openslide.vendor": "leica"},
        )
        assert meta.vendor == "leica"
        assert meta.mpp_x == 0.5
        assert meta.magnification == 20.0

        # Dict serialization
        d = meta.to_dict()
        assert d["vendor"] == "leica"
        assert d["dimensions"] == (8000, 6000)
        assert d["associated_images"] == ["macro"]
        # Must be JSON-serializable
        assert json.loads(json.dumps(d)) is not None


# ---------------------------------------------------------------------------
# extract_metadata
# ---------------------------------------------------------------------------

class TestExtractMetadata:
    """Verifies WSI metadata parsing under standard and edge cases."""

    def test_extract_raises_if_closed(self):
        """Raises MetadataExtractionError if trying to read closed slide."""
        reader = _make_mock_reader(properties={}, is_open=False)
        with pytest.raises(MetadataExtractionError, match="Slide is closed"):
            extract_metadata(reader)

    def test_standard_metadata_extraction_success(self):
        """Standard slide with standard OpenSlide properties extracts correctly."""
        props = {
            "openslide.vendor": "aperio",
            "openslide.mpp-x": "0.2520",
            "openslide.mpp-y": "0.2522",
            "openslide.objective-power": "40",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)

        assert meta.vendor == "aperio"
        assert meta.mpp_x == 0.2520
        assert meta.mpp_y == 0.2522
        assert meta.magnification == 40.0
        assert meta.level_count == 3
        assert meta.dimensions == (1000, 1000)

    def test_vendor_fallback_to_generic(self):
        """Vendor defaults to 'generic' if openslide.vendor property is absent."""
        props = {
            "openslide.mpp-x": "0.5",
            "openslide.mpp-y": "0.5",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)
        assert meta.vendor == "generic"

    def test_mpp_fallback_to_tiff_resolution_centimeter(self):
        """Resolves MPP from tiff.ResolutionUnit centimeter tags."""
        props = {
            "tiff.ResolutionUnit": "centimeter",
            "tiff.XResolution": "40000",  # 40000 pixels/cm = 0.25 MPP
            "tiff.YResolution": "40000",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)
        assert abs(meta.mpp_x - 0.25) < 1e-9
        assert abs(meta.mpp_y - 0.25) < 1e-9

    def test_mpp_fallback_to_tiff_resolution_inch(self):
        """Resolves MPP from tiff.ResolutionUnit inch tags."""
        props = {
            "tiff.ResolutionUnit": "inch",
            "tiff.XResolution": "100000",  # 100000 pixels/inch = 0.254 MPP
            "tiff.YResolution": "100000",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)
        assert abs(meta.mpp_x - 0.254) < 1e-9

    def test_mpp_fallback_to_description_comment(self):
        """Resolves MPP by regex searching the comment/description fields."""
        props = {
            "openslide.comment": "Aperio Image Library v10.0\nMPP = 0.4990\nObjective = 20",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)
        assert meta.mpp_x == 0.4990
        assert meta.mpp_y == 0.4990

    def test_raises_if_mpp_cannot_be_resolved(self):
        """Raises MetadataExtractionError if MPP is completely missing."""
        props = {
            "openslide.vendor": "generic",
        }
        reader = _make_mock_reader(properties=props)
        with pytest.raises(MetadataExtractionError, match="Microns-per-pixel .* could not be determined"):
            extract_metadata(reader)

    def test_magnification_fallback_aperio(self):
        """Extracts magnification from aperio.AppMag key if standard power is missing."""
        props = {
            "openslide.mpp-x": "0.50",
            "openslide.mpp-y": "0.50",
            "aperio.AppMag": "20",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)
        assert meta.magnification == 20.0

    def test_magnification_inferred_from_mpp_40x(self):
        """Infers 40x magnification if MPP is close to 0.25."""
        props = {
            "openslide.mpp-x": "0.253",
            "openslide.mpp-y": "0.253",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)
        assert meta.magnification == 40.0

    def test_magnification_inferred_from_mpp_20x(self):
        """Infers 20x magnification if MPP is close to 0.50."""
        props = {
            "openslide.mpp-x": "0.495",
            "openslide.mpp-y": "0.495",
        }
        reader = _make_mock_reader(properties=props)
        meta = extract_metadata(reader)
        assert meta.magnification == 20.0


# ---------------------------------------------------------------------------
# Metadata Bounds Validation
# ---------------------------------------------------------------------------

class TestMetadataValidationBounds:
    """Verifies that extract_metadata raises on corrupt or out-of-bounds metrics."""

    def test_raises_on_invalid_dimensions(self):
        props = {"openslide.mpp-x": "0.5", "openslide.mpp-y": "0.5"}
        reader = _make_mock_reader(properties=props, dimensions=(0, 1000))
        with pytest.raises(MetadataExtractionError, match="invalid full-res dimensions"):
            extract_metadata(reader)

    def test_raises_on_invalid_level_count(self):
        props = {"openslide.mpp-x": "0.5", "openslide.mpp-y": "0.5"}
        reader = _make_mock_reader(properties=props, level_count=0)
        with pytest.raises(MetadataExtractionError, match="invalid level count"):
            extract_metadata(reader)

    def test_raises_on_level_array_size_mismatch(self):
        props = {"openslide.mpp-x": "0.5", "openslide.mpp-y": "0.5"}
        # count=3, but downsamples array is length 2
        reader = _make_mock_reader(properties=props, level_count=3, level_downsamples=[1.0, 2.0])
        with pytest.raises(MetadataExtractionError, match="Inconsistent level count"):
            extract_metadata(reader)

    def test_raises_on_negative_mpp(self):
        props = {"openslide.mpp-x": "-0.5", "openslide.mpp-y": "0.5"}
        reader = _make_mock_reader(properties=props)
        with pytest.raises(MetadataExtractionError, match="Invalid negative or zero MPP"):
            extract_metadata(reader)
