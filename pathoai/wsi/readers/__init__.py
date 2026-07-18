"""pathoai.wsi.readers — Whole Slide Image (WSI) reading abstractions and implementations.

Exposes:
    BaseWSI: Abstract interface for all WSI readers.
    OpenSlideWSI: OpenSlide adapter implementation.
    get_wsi_reader: Factory function to retrieve the appropriate reader.
"""

from pathoai.wsi.readers.base import BaseWSI
from pathoai.wsi.readers.factory import get_wsi_reader
from pathoai.wsi.readers.openslide_reader import OpenSlideWSI

__all__ = [
    "BaseWSI",
    "OpenSlideWSI",
    "get_wsi_reader",
]
