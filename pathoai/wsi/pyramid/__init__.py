"""pathoai.wsi.pyramid — WSI pyramid coordinate mapping and image scaling engine.

Exposes:
    level_to_slide_coords: Maps coordinates at downsampled levels to level 0.
    slide_to_level_coords: Maps level 0 coordinates to downsampled levels.
    read_patch_at_mpp: Reads WSI regions rescaled to target physical spacing (MPP).
    get_slide_thumbnail: Generates a full slide thumbnail of specified size.
"""

from pathoai.wsi.pyramid.pyramid import (
    get_slide_thumbnail,
    level_to_slide_coords,
    read_patch_at_mpp,
    slide_to_level_coords,
)

__all__ = [
    "level_to_slide_coords",
    "slide_to_level_coords",
    "read_patch_at_mpp",
    "get_slide_thumbnail",
]
