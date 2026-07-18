"""pathoai.wsi.metadata — WSI metadata structures and parser engine.

Exposes:
    SlideMetadata: Frozen dataclass container for WSI properties.
    extract_metadata: Extracts SlideMetadata from a BaseWSI reader.
"""

from pathoai.wsi.metadata.metadata import SlideMetadata, extract_metadata

__all__ = [
    "SlideMetadata",
    "extract_metadata",
]
