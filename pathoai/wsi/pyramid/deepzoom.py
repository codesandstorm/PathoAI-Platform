"""
pathoai/wsi/pyramid/deepzoom.py
===============================
DeepZoom Tile Generator for OpenSeadragon.

Generates DZI (Deep Zoom Image) XML descriptors and extracts tile images (z, x, y)
from whole slide images using OpenSlideWSI and PIL.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: Phase 2 (WSI DeepZoom Integration)
"""

from __future__ import annotations

import io
import math
from typing import Tuple

import numpy as np
from PIL import Image

from pathoai.wsi.readers.base import BaseWSI


class DeepZoomTileGenerator:
    """Generates DZI metadata and extracts tile images for OpenSeadragon."""

    def __init__(self, reader: BaseWSI, tile_size: int = 256, overlap: int = 0) -> None:
        """
        Parameters
        ----------
        reader : BaseWSI
            Initialized OpenSlideWSI or mock slide reader.
        tile_size : int
            Individual tile width/height in pixels.
        overlap : int
            Tile border overlap in pixels.
        """
        self.reader = reader
        self.tile_size = tile_size
        self.overlap = overlap
        self.width, self.height = reader.dimensions

        # Calculate max zoom level
        max_dim = max(self.width, self.height)
        self.max_level = int(math.ceil(math.log2(max_dim))) if max_dim > 0 else 0

    def get_dzi_xml(self) -> str:
        """Generates DZI XML descriptor string for OpenSeadragon.

        Returns
        -------
        str
            DZI XML format string.
        """
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Image TileSize="{self.tile_size}" Overlap="{self.overlap}" Format="png" '
            f'xmlns="http://schemas.microsoft.com/deepzoom/2008">\n'
            f'  <Size Width="{self.width}" Height="{self.height}"/>\n'
            f'</Image>'
        )

    def get_tile(self, level: int, col: int, row: int) -> bytes:
        """Extracts tile image bytes (PNG format) at specific (level, col, row).

        Parameters
        ----------
        level : int
            DeepZoom pyramid level.
        col : int
            Tile column index (x).
        row : int
            Tile row index (y).

        Returns
        -------
        bytes
            PNG encoded tile image bytes.
        """
        # Calculate scale factor for requested level
        scale = 2 ** (self.max_level - level)
        level_width = max(1, self.width // scale)
        level_height = max(1, self.height // scale)

        # Calculate tile pixel coordinates at level
        x = col * self.tile_size
        y = row * self.tile_size
        w = min(self.tile_size, level_width - x)
        h = min(self.tile_size, level_height - y)

        if w <= 0 or h <= 0:
            # Empty tile fallback
            img = Image.new("RGB", (self.tile_size, self.tile_size), color=(240, 240, 240))
        else:
            # Map level coordinates to slide level 0 coordinates
            slide_x = int(x * scale)
            slide_y = int(y * scale)
            slide_w = int(w * scale)
            slide_h = int(h * scale)

            try:
                rgb = self.reader.read_region((slide_x, slide_y), 0, (slide_w, slide_h))
                img = Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR)
            except Exception:
                img = Image.new("RGB", (max(1, w), max(1, h)), color=(228, 228, 228))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
