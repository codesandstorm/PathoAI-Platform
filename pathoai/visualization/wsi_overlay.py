"""
pathoai/visualization/wsi_overlay.py
====================================
Slide-level visualization overlays for PathoAI-Platform.

Provides functions to overlay patch sampling grids and tissue masks on slide
thumbnail images, facilitating quality control (QC) and pipeline validation.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pathoai.core.exceptions import ValidationError
from pathoai.core.logger import get_logger
from pathoai.wsi.metadata.metadata import SlideMetadata
from pathoai.wsi.patches.patches import PatchMetadata

logger = get_logger(__name__)

RGBUint8 = Tuple[int, int, int]


def draw_patch_grid_overlay(
    thumbnail: np.ndarray,
    patches_metadata: List[PatchMetadata],
    metadata: SlideMetadata,
    color: RGBUint8 = (0, 0, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw sampled patch bounding boxes on a slide thumbnail.

    Parameters
    ----------
    thumbnail : np.ndarray
        RGB thumbnail image. Shape: (H, W, 3), dtype uint8.
    patches_metadata : List[PatchMetadata]
        List of sampled patch metadata.
    metadata : SlideMetadata
        Standardized metadata of the WSI.
    color : RGBUint8
        RGB color tuple for the grid lines. Defaults to red: (0, 0, 255).
    thickness : int
        Thickness of grid lines in pixels. Defaults to 1.

    Returns
    -------
    np.ndarray
        RGB image copy of the thumbnail with the grid drawn. Shape: (H, W, 3).

    Raises
    ------
    ValidationError
        If input parameters or shapes are invalid.
    """
    if thumbnail.size == 0:
        raise ValidationError("Thumbnail image is empty.")
    if thumbnail.ndim != 3 or thumbnail.shape[2] != 3:
        raise ValidationError(f"Expected RGB thumbnail, got shape {thumbnail.shape}")

    result = thumbnail.copy()
    h_thumb, w_thumb = result.shape[:2]
    w_slide_0, _ = metadata.dimensions

    # Compute thumbnail downsample factor
    downsample = w_slide_0 / w_thumb if w_thumb > 0 else 1.0

    logger.debug(
        "Drawing patch grid overlay",
        extra={
            "n_patches": len(patches_metadata),
            "thumbnail_size": f"{w_thumb}x{h_thumb}",
            "downsample": round(downsample, 2),
        },
    )

    for patch in patches_metadata:
        # Determine patch width/height in level 0 coords
        mpp_ratio = patch.target_mpp / metadata.mpp_x
        patch_size_0 = patch.patch_size * mpp_ratio

        # Map top-left and bottom-right corners to thumbnail coordinates
        x1 = int(round(patch.x_level0 / downsample))
        y1 = int(round(patch.y_level0 / downsample))
        x2 = int(round((patch.x_level0 + patch_size_0) / downsample))
        y2 = int(round((patch.y_level0 + patch_size_0) / downsample))

        # Clip values to thumbnail dimensions
        x1 = max(0, min(x1, w_thumb - 1))
        y1 = max(0, min(y1, h_thumb - 1))
        x2 = max(0, min(x2, w_thumb - 1))
        y2 = max(0, min(y2, h_thumb - 1))

        # Draw the rectangle lines
        result[y1:y1 + thickness, x1:x2 + 1] = color  # top
        result[y2 - thickness + 1:y2 + 1, x1:x2 + 1] = color  # bottom
        result[y1:y2 + 1, x1:x1 + thickness] = color  # left
        result[y1:y2 + 1, x2 - thickness + 1:x2 + 1] = color  # right

    return result


def draw_tissue_mask_overlay(
    thumbnail: np.ndarray,
    tissue_mask: np.ndarray,
    alpha: float = 0.35,
    color: RGBUint8 = (80, 200, 120),
) -> np.ndarray:
    """Overlay a semi-transparent tissue detection mask on a WSI thumbnail.

    Parameters
    ----------
    thumbnail : np.ndarray
        RGB thumbnail image. Shape: (H, W, 3), dtype uint8.
    tissue_mask : np.ndarray
        Binary mask (1 = tissue, 0 = background). Shape: (H, W), dtype uint8.
    alpha : float
        Opacity of the mask overlay. Range: [0.0, 1.0]. Defaults to 0.35.
    color : RGBUint8
        RGB color tuple for the overlay (defaults to emerald green: (80, 200, 120)).

    Returns
    -------
    np.ndarray
        RGB image copy with tissue overlay blended. Shape: (H, W, 3), dtype uint8.

    Raises
    ------
    ValidationError
        If thumbnail and mask shapes mismatch.
    """
    if thumbnail.shape[:2] != tissue_mask.shape:
        raise ValidationError(
            f"Thumbnail shape {thumbnail.shape[:2]} does not match "
            f"tissue mask shape {tissue_mask.shape}"
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValidationError(f"alpha must be in [0.0, 1.0], got {alpha}")

    mask_3d = (tissue_mask == 1)[..., np.newaxis]
    color_arr = np.array(color, dtype=np.uint8)

    # Vectorized alpha blending only on mask pixels
    blended_pixels = (thumbnail.astype(np.float32) * (1.0 - alpha) + color_arr * alpha).astype(np.uint8)
    result = np.where(mask_3d, blended_pixels, thumbnail)

    return result
