"""
pathoai/detection/tiling.py
===========================
Streaming Tile Generator for Object Detection.

Extracts sliding-window tiles from TumorROI bounding boxes or WSI regions
with configurable tile size, overlap, stride, and padding.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 7.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple, Union

import numpy as np

from pathoai.core.types import BoundingBox, TumorROI


@dataclass(frozen=True)
class TileMetadata:
    """Metadata for an extracted inference tile.

    Attributes
    ----------
    tile_id : int
        Index of the tile.
    roi_id : str
        Parent TumorROI identifier.
    tile_x0 : int
        Top-left X coordinate of the tile in slide level-0 pixel space.
    tile_y0 : int
        Top-left Y coordinate of the tile in slide level-0 pixel space.
    width : int
        Width of the tile in pixels.
    height : int
        Height of the tile in pixels.
    local_x0 : int
        X offset of tile inside the ROI bounding box.
    local_y0 : int
        Y offset of tile inside the ROI bounding box.
    """

    tile_id: int
    roi_id: str
    tile_x0: int
    tile_y0: int
    width: int
    height: int
    local_x0: int
    local_y0: int


class TileGenerator:
    """Generates sliding window tiles over TumorROI regions."""

    def __init__(
        self,
        tile_size: int = 640,
        overlap: int = 64,
        padding: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        tile_size : int
            Square width and height of each generated tile in pixels.
        overlap : int
            Overlap in pixels between adjacent tiles.
        padding : int
            Border padding in pixels added around the ROI bounding box.
        """
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive. Got: {tile_size}")
        if overlap >= tile_size:
            raise ValueError(f"overlap must be less than tile_size. Got: {overlap} vs {tile_size}")
        if padding < 0:
            raise ValueError(f"padding must be non-negative. Got: {padding}")

        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.padding = padding

    def generate_tile_coords(
        self, roi: TumorROI
    ) -> List[TileMetadata]:
        """Calculate list of tile metadata entries covering the ROI.

        Parameters
        ----------
        roi : TumorROI
            Target Region of Interest.

        Returns
        -------
        List[TileMetadata]
            List of tile metadata objects covering the region.
        """
        min_y = max(0, roi.bbox.min_y - self.padding)
        min_x = max(0, roi.bbox.min_x - self.padding)
        max_y = roi.bbox.max_y + self.padding
        max_x = roi.bbox.max_x + self.padding

        roi_w = max_x - min_x
        roi_h = max_y - min_y

        if roi_w <= 0 or roi_h <= 0:
            return []

        tiles = []
        tile_id = 0

        y = min_y
        while y < max_y:
            x = min_x
            while x < max_x:
                tiles.append(
                    TileMetadata(
                        tile_id=tile_id,
                        roi_id=str(roi.roi_id),
                        tile_x0=x,
                        tile_y0=y,
                        width=self.tile_size,
                        height=self.tile_size,
                        local_x0=x - min_x,
                        local_y0=y - min_y,
                    )
                )
                tile_id += 1
                x += self.stride
                if x >= max_x and (x - self.stride + self.tile_size) >= max_x:
                    break

            y += self.stride
            if y >= max_y and (y - self.stride + self.tile_size) >= max_y:
                break

        return tiles

    def extract_tiles_from_array(
        self, image: np.ndarray, roi: TumorROI
    ) -> Generator[Tuple[TileMetadata, np.ndarray], None, None]:
        """Stream tiles from an in-memory image array covering an ROI.

        Parameters
        ----------
        image : np.ndarray
            Image array of shape (H, W, C).
        roi : TumorROI
            Target region.

        Yields
        ------
        Tuple[TileMetadata, np.ndarray]
            Tuple of (tile_metadata, tile_patch_array).
        """
        h_img, w_img, _ = image.shape
        tile_coords = self.generate_tile_coords(roi)

        for meta in tile_coords:
            y1 = meta.tile_y0
            x1 = meta.tile_x0
            y2 = min(h_img, y1 + meta.height)
            x2 = min(w_img, x1 + meta.width)

            patch = np.zeros((self.tile_size, self.tile_size, image.shape[2]), dtype=image.dtype)
            sub_patch = image[y1:y2, x1:x2]
            patch[: sub_patch.shape[0], : sub_patch.shape[1]] = sub_patch

            yield meta, patch
