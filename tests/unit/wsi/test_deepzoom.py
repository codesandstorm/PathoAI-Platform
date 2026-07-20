"""
tests/unit/wsi/test_deepzoom.py
===============================
Unit tests for DeepZoomTileGenerator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from pathoai.wsi.pyramid.deepzoom import DeepZoomTileGenerator
from pathoai.wsi.readers.base import BaseWSI


class DummyWSI(BaseWSI):
    """Dummy WSI reader for unit testing DeepZoomTileGenerator."""

    def __init__(self, width: int = 1000, height: int = 800) -> None:
        self._dims = (width, height)
        self._path = Path("dummy_slide.svs")
        self._open = True

    def __enter__(self) -> BaseWSI: return self
    def __exit__(self, exc_type, exc_val, exc_tb) -> None: pass
    def open(self) -> None: self._open = True
    def close(self) -> None: self._open = False
    @property
    def is_open(self) -> bool: return self._open
    @property
    def path(self) -> Path: return self._path
    @property
    def dimensions(me) -> Tuple[int, int]: return me._dims
    @property
    def level_count(me) -> int: return 1
    @property
    def level_dimensions(me) -> List[Tuple[int, int]]: return [me._dims]
    @property
    def level_downsamples(me) -> List[float]: return [1.0]
    @property
    def properties(me) -> Dict[str, Any]: return {}
    @property
    def associated_images(me) -> Dict[str, np.ndarray]: return {}
    @property
    def raw_metadata(me) -> Dict[str, Any]: return {}

    def read_region(me, location, level, size):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)


class TestDeepZoomTileGenerator:
    """Test DeepZoom XML descriptor and tile byte generation."""

    def test_get_dzi_xml(self):
        reader = DummyWSI(width=2000, height=1600)
        generator = DeepZoomTileGenerator(reader, tile_size=256)
        xml = generator.get_dzi_xml()

        assert 'TileSize="256"' in xml
        assert 'Width="2000"' in xml
        assert 'Height="1600"' in xml

    def test_get_tile_bytes(self):
        reader = DummyWSI(width=2000, height=1600)
        generator = DeepZoomTileGenerator(reader, tile_size=256)
        tile_bytes = generator.get_tile(level=10, col=0, row=0)

        assert isinstance(tile_bytes, bytes)
        assert len(tile_bytes) > 0
        assert tile_bytes[:4] == b"\x89PNG"
