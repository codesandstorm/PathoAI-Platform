"""
tests/unit/detection/test_tiling.py
====================================
Unit tests for TileGenerator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np
import pytest

from pathoai.core.types import BoundingBox, Point, TumorROI
from pathoai.detection.tiling import TileGenerator


class TestTileGenerator:
    """Test tile generator."""

    def test_tile_generator_validation(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="tile_size must be positive"):
            TileGenerator(tile_size=0)
        with pytest.raises(ValueError, match="overlap must be less than tile_size"):
            TileGenerator(tile_size=128, overlap=128)

    def test_generate_tile_coords(self):
        """Test generating tile coordinates for a TumorROI."""
        roi = TumorROI(
            roi_id=1,
            bbox=BoundingBox(min_y=100, min_x=100, max_y=500, max_x=500),
            centroid=Point(300.0, 300.0),
            area_px=160000,
            area_um2=40000.0,
            perimeter_um=1600.0,
            contours=[],
        )

        gen = TileGenerator(tile_size=256, overlap=32)
        coords = gen.generate_tile_coords(roi)

        assert len(coords) > 0
        assert coords[0].tile_x0 == 100
        assert coords[0].tile_y0 == 100

    def test_extract_tiles_from_array(self):
        """Test streaming tiles from image array."""
        roi = TumorROI(
            roi_id=1,
            bbox=BoundingBox(min_y=10, min_x=10, max_y=100, max_x=100),
            centroid=Point(55.0, 55.0),
            area_px=8100,
            area_um2=2025.0,
            perimeter_um=360.0,
            contours=[],
        )

        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        gen = TileGenerator(tile_size=64, overlap=16)

        tiles = list(gen.extract_tiles_from_array(img, roi))
        assert len(tiles) > 0
        meta, patch = tiles[0]
        assert patch.shape == (64, 64, 3)
