"""pathoai.wsi — Whole Slide Image (WSI) engine.

This package exposes the complete WSI processing API: readers, metadata extraction,
pyramid operations, tissue masking, and patch sampling.
"""

from pathoai.wsi.metadata.metadata import SlideMetadata, extract_metadata
from pathoai.wsi.patches.patches import PatchMetadata, PatchSampler
from pathoai.wsi.pyramid.pyramid import (
    get_slide_thumbnail,
    level_to_slide_coords,
    read_patch_at_mpp,
    slide_to_level_coords,
)
from pathoai.wsi.readers.base import BaseWSI
from pathoai.wsi.readers.factory import get_wsi_reader
from pathoai.wsi.readers.openslide_reader import OpenSlideWSI
from pathoai.wsi.tissue.tissue import TissueDetector

__all__ = [
    # Readers
    "BaseWSI",
    "OpenSlideWSI",
    "get_wsi_reader",
    # Metadata
    "SlideMetadata",
    "extract_metadata",
    # Pyramid
    "level_to_slide_coords",
    "slide_to_level_coords",
    "read_patch_at_mpp",
    "get_slide_thumbnail",
    # Tissue
    "TissueDetector",
    # Patches
    "PatchMetadata",
    "PatchSampler",
]
