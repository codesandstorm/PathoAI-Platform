"""
pathoai/wsi/pyramid/pyramid.py
==============================
Pyramid Engine for coordinate mapping, scaling, and patch/thumbnail generation.

Deals with pyramid coordinate conversions, selecting matching zoom levels,
reading slide regions at target physical resolutions (microns-per-pixel),
and generating downsampled thumbnails.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from pathoai.core.exceptions import WSIReadError
from pathoai.core.utils.math_utils import find_best_pyramid_level
from pathoai.wsi.metadata.metadata import SlideMetadata
from pathoai.wsi.readers.base import BaseWSI


def level_to_slide_coords(level_coord: Tuple[int, int], downsample: float) -> Tuple[int, int]:
    """Convert coordinates at a specific pyramid level to level 0 (slide) coordinates.

    Parameters
    ----------
    level_coord : Tuple[int, int]
        (x, y) coordinates at the downsampled level.
    downsample : float
        Downsample factor of that level.

    Returns
    -------
    Tuple[int, int]
        (x, y) coordinates at level 0 (full resolution).
    """
    return (
        int(round(level_coord[0] * downsample)),
        int(round(level_coord[1] * downsample)),
    )


def slide_to_level_coords(slide_coord: Tuple[int, int], downsample: float) -> Tuple[int, int]:
    """Convert level 0 (slide) coordinates to coordinates at a specific pyramid level.

    Parameters
    ----------
    slide_coord : Tuple[int, int]
        (x, y) coordinates at level 0 (full resolution).
    downsample : float
        Downsample factor of that level.

    Returns
    -------
    Tuple[int, int]
        (x, y) coordinates at the downsampled level.
    """
    return (
        int(round(slide_coord[0] / downsample)),
        int(round(slide_coord[1] / downsample)),
    )


def read_patch_at_mpp(
    reader: BaseWSI,
    metadata: SlideMetadata,
    location_level0: Tuple[int, int],
    target_mpp: float,
    patch_size: Tuple[int, int],
) -> np.ndarray:
    """Read a WSI region at a target physical resolution (microns per pixel).

    Checks the slide's pyramid structure, selects the closest downsample level,
    reads the region, and scales it to match the exact target size and MPP.

    Parameters
    ----------
    reader : BaseWSI
        An open slide reader instance.
    metadata : SlideMetadata
        Metadata for the slide being read.
    location_level0 : Tuple[int, int]
        (x, y) coordinates in the level 0 reference frame.
    target_mpp : float
        Target microns-per-pixel (e.g. 0.50 for 20x equivalent).
    patch_size : Tuple[int, int]
        (width, height) in pixels at the target MPP.

    Returns
    -------
    np.ndarray
        RGB image patch. Shape: (height, width, 3), dtype uint8.

    Raises
    ------
    WSIReadError
        If coordinates/size are out of bounds, target MPP is negative, or slide read fails.
    """
    if not reader.is_open:
        raise WSIReadError(f"Cannot read patch. Slide is closed: {reader.path}")
    if target_mpp <= 0.0:
        raise WSIReadError(f"Target MPP must be positive. Got: {target_mpp}")
    if patch_size[0] <= 0 or patch_size[1] <= 0:
        raise WSIReadError(f"Patch size dimensions must be positive. Got: {patch_size}")

    # 1. Find the best matching pyramid level
    level = find_best_pyramid_level(
        metadata.level_downsamples,
        target_mpp,
        metadata.mpp_x,
    )

    actual_ds = metadata.level_downsamples[level]
    mpp_level_x = metadata.mpp_x * actual_ds
    mpp_level_y = metadata.mpp_y * actual_ds

    # 2. Compute size of region to read at the selected level
    w_level = int(round(patch_size[0] * (target_mpp / mpp_level_x)))
    h_level = int(round(patch_size[1] * (target_mpp / mpp_level_y)))

    # Ensure dimension is at least 1 pixel
    w_level = max(1, w_level)
    h_level = max(1, h_level)

    # 3. Read the region
    img = reader.read_region(location_level0, level, (w_level, h_level))

    # 4. If read size differs from target size, resize to match target pixel size
    if (w_level, h_level) != patch_size:
        pil_img = Image.fromarray(img)
        resized_pil = pil_img.resize(patch_size, resample=Image.Resampling.BICUBIC)
        return np.array(resized_pil, dtype=np.uint8)

    return img


def get_slide_thumbnail(
    reader: BaseWSI,
    metadata: SlideMetadata,
    max_dim: int,
) -> np.ndarray:
    """Generate a downsampled thumbnail image of the entire slide.

    Finds the best matching pyramid level to read from (to prevent large memory
    allocation) and rescales the resulting image.

    Parameters
    ----------
    reader : BaseWSI
        An open slide reader instance.
    metadata : SlideMetadata
        Metadata for the slide.
    max_dim : int
        Maximum size (width or height) in pixels of the returned thumbnail.

    Returns
    -------
    np.ndarray
        RGB thumbnail image. Shape: (H, W, 3), dtype uint8.

    Raises
    ------
    WSIReadError
        If slide reading fails.
    """
    if not reader.is_open:
        raise WSIReadError(f"Cannot read thumbnail. Slide is closed: {reader.path}")
    if max_dim <= 0:
        raise WSIReadError(f"max_dim must be positive. Got: {max_dim}")

    # 1. Determine downsample factor needed
    w_0, h_0 = metadata.dimensions
    target_ds = max(w_0, h_0) / max_dim

    # 2. Find the level closest to target downsample
    differences = [abs(ds - target_ds) for ds in metadata.level_downsamples]
    level = int(np.argmin(differences))

    # 3. Read the whole level image
    level_w, level_h = metadata.level_dimensions[level]
    img = reader.read_region((0, 0), level, (level_w, level_h))

    # 4. Calculate exact resize dimensions
    ratio = max_dim / max(level_h, level_w)
    new_w = int(round(level_w * ratio))
    new_h = int(round(level_h * ratio))

    # Ensure sizes are positive
    new_w = max(1, new_w)
    new_h = max(1, new_h)

    # 5. Resize to exact target size
    pil_img = Image.fromarray(img)
    resized_pil = pil_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return np.array(resized_pil, dtype=np.uint8)
