"""
tests/unit/wsi/test_reader.py
==============================
Unit tests for the WSI Reader Layer.

Tests cover:
- BaseWSI interface compliance
- WSIReaderFactory format validation and instance creation
- OpenSlideWSI context manager, resource lifecycle, and error wrapping
- Input validation on read_region (bounds, sizes, levels)
- Channel conversion logic (RGBA -> RGB)

Uses sys.modules mocking to prevent loading OpenSlide C DLLs during test runs.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 2.1
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# 1. Stub the 'openslide' module in sys.modules BEFORE importing the reader code
# This completely bypasses the C library / DLL loading check.
mock_openslide_module = MagicMock()
sys.modules["openslide"] = mock_openslide_module

# Now we can safely import our WSI classes
from pathoai.core.exceptions import WSIReadError
from pathoai.wsi.base import BaseWSI
from pathoai.wsi.factory import get_wsi_reader
from pathoai.wsi.openslide_reader import OpenSlideWSI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_openslide_mocks():
    """Reset the mock openslide module's internal state before each test."""
    mock_openslide_module.reset_mock()
    # Configure default mock attributes
    mock_openslide_module.OpenSlide = MagicMock
    yield


# ---------------------------------------------------------------------------
# BaseWSI Interface Compliance
# ---------------------------------------------------------------------------

class TestBaseWSIInterface:
    """Verifies that BaseWSI enforces the abstract interface contract."""

    def test_cannot_instantiate_abstract_class(self):
        """Cannot instantiate BaseWSI directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseWSI()  # type: ignore

    def test_concrete_subclass_must_implement_abstract_methods(self):
        """Subclass missing abstract methods cannot be instantiated."""
        class IncompleteWSI(BaseWSI):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteWSI()  # type: ignore


# ---------------------------------------------------------------------------
# WSIReaderFactory
# ---------------------------------------------------------------------------

class TestWSIReaderFactory:
    """Verifies format validation and reader creation in WSIReaderFactory."""

    def test_factory_returns_openslide_wsi_for_supported_formats(self, tmp_path: Path):
        """Returns an OpenSlideWSI instance for standard formats like .svs or .tif."""
        slide_path = tmp_path / "slide.svs"
        reader = get_wsi_reader(slide_path)
        assert isinstance(reader, OpenSlideWSI)
        assert reader.path == slide_path.resolve()

    def test_factory_raises_for_unsupported_formats(self, tmp_path: Path):
        """Raises WSIReadError if the format/extension is not supported."""
        unsupported_path = tmp_path / "slide.png"
        with pytest.raises(WSIReadError, match="Unsupported WSI file format"):
            get_wsi_reader(unsupported_path)


# ---------------------------------------------------------------------------
# OpenSlideWSI Adapter Lifecycle & Context Manager
# ---------------------------------------------------------------------------

class TestOpenSlideWSILifecycle:
    """Verifies the state transitions and resource management of OpenSlideWSI."""

    def test_starts_closed(self, tmp_path: Path):
        """A new OpenSlideWSI instance is not open."""
        reader = OpenSlideWSI(tmp_path / "slide.svs")
        assert reader.is_open is False

    def test_open_raises_on_nonexistent_file(self, tmp_path: Path):
        """Raises WSIReadError if the slide file is missing from disk."""
        reader = OpenSlideWSI(tmp_path / "missing.svs")
        with pytest.raises(WSIReadError, match="Slide file does not exist"):
            reader.open()

    def test_open_success(self, tmp_path: Path):
        """Loads OpenSlide and creates the underlying slide object successfully."""
        slide_path = tmp_path / "slide.svs"
        slide_path.write_bytes(b"dummy")  # Create file to pass existence check

        mock_slide_instance = MagicMock()
        mock_openslide_module.OpenSlide = MagicMock(return_value=mock_slide_instance)

        reader = OpenSlideWSI(slide_path)
        reader.open()
        assert reader.is_open is True
        assert reader.path == slide_path.resolve()
        mock_openslide_module.OpenSlide.assert_called_once_with(str(slide_path.resolve()))

    def test_context_manager_lifecycle(self, tmp_path: Path):
        """Context manager opens and automatically closes the slide resource."""
        slide_path = tmp_path / "slide.svs"
        slide_path.write_bytes(b"dummy")

        mock_slide_instance = MagicMock()
        mock_openslide_module.OpenSlide = MagicMock(return_value=mock_slide_instance)

        with OpenSlideWSI(slide_path) as reader:
            assert reader.is_open is True
            mock_openslide_module.OpenSlide.assert_called_once()

        assert reader.is_open is False
        mock_slide_instance.close.assert_called_once()

    def test_double_open_is_noop(self, tmp_path: Path):
        """Calling open() on an already open slide does not re-open it."""
        slide_path = tmp_path / "slide.svs"
        slide_path.write_bytes(b"dummy")

        mock_slide_instance = MagicMock()
        mock_openslide_module.OpenSlide = MagicMock(return_value=mock_slide_instance)

        reader = OpenSlideWSI(slide_path)
        reader.open()
        reader.open()
        assert mock_openslide_module.OpenSlide.call_count == 1

    def test_double_close_is_noop(self, tmp_path: Path):
        """Calling close() on an already closed slide does not raise."""
        slide_path = tmp_path / "slide.svs"
        slide_path.write_bytes(b"dummy")

        mock_slide_instance = MagicMock()
        mock_openslide_module.OpenSlide = MagicMock(return_value=mock_slide_instance)

        reader = OpenSlideWSI(slide_path)
        reader.open()
        reader.close()
        reader.close()  # Should be safe and not raise


# ---------------------------------------------------------------------------
# OpenSlideWSI Region Reading & Input Validation
# ---------------------------------------------------------------------------

class TestOpenSlideWSIReading:
    """Verifies reading behavior, exception wrapping, and bounds validation."""

    @pytest.fixture()
    def open_reader(self, tmp_path: Path):
        """Fixture that returns an open reader with a mocked OpenSlide instance."""
        slide_path = tmp_path / "slide.svs"
        slide_path.write_bytes(b"dummy")

        mock_slide = MagicMock()
        mock_slide.level_count = 3
        mock_openslide_module.OpenSlide = MagicMock(return_value=mock_slide)

        reader = OpenSlideWSI(slide_path)
        reader.open()
        # Return the reader and its underlying mock slide for assertions
        return reader, mock_slide

    def test_read_region_raises_if_closed(self, tmp_path: Path):
        """Raises WSIReadError if trying to read a region from a closed slide."""
        reader = OpenSlideWSI(tmp_path / "slide.svs")
        with pytest.raises(WSIReadError, match="Slide is closed"):
            reader.read_region((0, 0), 0, (100, 100))

    def test_read_region_validates_coordinates(self, open_reader):
        """Coordinates must be non-negative integers."""
        reader, _ = open_reader
        with pytest.raises(WSIReadError, match="Coordinates must be non-negative"):
            reader.read_region((-1, 0), 0, (100, 100))

    def test_read_region_validates_size(self, open_reader):
        """Region size/dimensions must be positive integers."""
        reader, _ = open_reader
        with pytest.raises(WSIReadError, match="Region dimensions must be positive"):
            reader.read_region((0, 0), 0, (0, 100))
        with pytest.raises(WSIReadError, match="Region dimensions must be positive"):
            reader.read_region((0, 0), 0, (100, -5))

    def test_read_region_validates_negative_level(self, open_reader):
        """Pyramid level must be non-negative."""
        reader, _ = open_reader
        with pytest.raises(WSIReadError, match="Pyramid level must be non-negative"):
            reader.read_region((0, 0), -1, (100, 100))

    def test_read_region_validates_level_out_of_bounds(self, open_reader):
        """Raises WSIReadError if the level index is out of bounds."""
        reader, _ = open_reader
        with pytest.raises(WSIReadError, match="exceeds slide level count"):
            reader.read_region((0, 0), 3, (100, 100))  # level_count is 3, valid levels are 0,1,2

    def test_read_region_returns_rgb_array(self, open_reader):
        """PIL RGBA image returned by OpenSlide is converted to RGB NumPy array."""
        reader, mock_slide = open_reader

        # Create a mock PIL image (RGBA)
        width, height = 50, 40
        fake_img = Image.new("RGBA", (width, height), color=(255, 128, 64, 255))
        mock_slide.read_region.return_value = fake_img

        result = reader.read_region((10, 20), 1, (width, height))

        assert isinstance(result, np.ndarray)
        assert result.shape == (height, width, 3)
        assert result.dtype == np.uint8
        # Verify color conversion: RGBA -> RGB
        np.testing.assert_array_equal(result[0, 0], [255, 128, 64])
        mock_slide.read_region.assert_called_once_with((10, 20), 1, (width, height))

    def test_read_region_propagates_openslide_exceptions(self, open_reader):
        """Internal OpenSlide exceptions must be caught and wrapped in WSIReadError."""
        reader, mock_slide = open_reader
        mock_slide.read_region.side_effect = RuntimeError("Low-level C library error")

        with pytest.raises(WSIReadError, match="Failed to read region"):
            reader.read_region((0, 0), 0, (10, 10))


# ---------------------------------------------------------------------------
# Dependency loading checks
# ---------------------------------------------------------------------------

class TestOpenSlideImportFailures:
    """Verifies that missing openslide library raises informative WSIReadErrors."""

    def test_raises_wsi_read_error_on_import_failure(self, tmp_path: Path):
        """Simulates missing library to verify user-friendly instruction message."""
        slide_path = tmp_path / "slide.svs"
        slide_path.write_bytes(b"dummy")

        reader = OpenSlideWSI(slide_path)
        # Temporarily remove "openslide" from sys.modules to simulate missing/failed import
        sys.modules["openslide"] = None
        try:
            with pytest.raises(WSIReadError, match="Failed to import/load OpenSlide"):
                reader.open()
        finally:
            # Restore the mock module reference
            sys.modules["openslide"] = mock_openslide_module
