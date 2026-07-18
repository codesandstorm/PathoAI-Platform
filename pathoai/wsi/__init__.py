"""pathoai.wsi — Whole Slide Image (WSI) reading and processing engine.

Exposes:
    BaseWSI: Abstract interface for all WSI readers.
    OpenSlideWSI: OpenSlide adapter implementation.
    get_wsi_reader: Factory function to retrieve the appropriate reader.
"""

from pathoai.wsi.base import BaseWSI
from pathoai.wsi.factory import get_wsi_reader
from pathoai.wsi.openslide_reader import OpenSlideWSI

__all__ = [
    "BaseWSI",
    "OpenSlideWSI",
    "get_wsi_reader",
]
