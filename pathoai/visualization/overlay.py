"""
pathoai/visualization/overlay.py
=====================================
Image overlay utilities for PathoAI-Platform.

Provides functions for blending segmentation masks and detection boxes
over WSI patch images for visualization and quality control.

Design principles:
    - Pure NumPy/PIL — no OpenCV dependency at this layer.
    - All functions are stateless and pure (no global state mutations).
    - Designed for quality control (QC) and report visualization, not
      real-time rendering.

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.8
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pathoai.core.constants import SEGMENTATION_OVERLAY_ALPHA as OVERLAY_ALPHA
from pathoai.core.exceptions import PathoAIException
from pathoai.core.logger import get_logger
from pathoai.visualization.colormap import build_tissue_lut, colorize_mask

logger = get_logger(__name__)

# Type aliases
BoundingBox = Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
RGBUint8 = Tuple[int, int, int]


class OverlayError(PathoAIException):
    """Raised when an overlay operation cannot complete."""


# ---------------------------------------------------------------------------
# Core overlay function
# ---------------------------------------------------------------------------

def blend_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
    lut: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Blend a colored segmentation mask over an RGB image.

    Uses alpha compositing:
        result = image * (1 - alpha) + colored_mask * alpha

    Background pixels (class ID = 0) in the mask are blended with
    a low alpha value to preserve the underlying tissue appearance.

    Args:
        image: RGB image array. Shape: (H, W, 3), dtype uint8.
        mask: Class-ID mask. Shape: (H, W), dtype uint8 or int.
            Values must be in [0, n_classes).
        alpha: Blend strength in [0.0, 1.0]. 0.0 = show only image,
            1.0 = show only mask. Defaults to ``OVERLAY_ALPHA`` from
            constants (typically 0.4).
        lut: Pre-built RGBA LUT from ``build_tissue_lut()``. If None,
            a fresh LUT is built.

    Returns:
        Blended RGB image. Shape: (H, W, 3), dtype uint8.

    Raises:
        OverlayError: If image and mask have incompatible shapes, or
            if image is not a 3-channel uint8 array.

    Example:
        >>> result = blend_mask_overlay(patch_rgb, seg_mask, alpha=0.4)
        >>> Image.fromarray(result).save("overlay.png")
    """
    _validate_image(image, name="image")
    _validate_mask(mask, name="mask")
    _validate_shapes_match(image, mask)
    if not (0.0 <= alpha <= 1.0):
        raise OverlayError(f"alpha must be in [0.0, 1.0], got {alpha}")

    if lut is None:
        lut = build_tissue_lut(alpha=255)

    # Colorize mask → RGBA
    colored = colorize_mask(mask, lut=lut)  # (H, W, 4) uint8
    colored_rgb = colored[..., :3].astype(np.float32)  # Drop alpha channel

    # Build per-pixel alpha map: lower alpha for background class
    background_alpha = alpha * 0.2
    pixel_alpha = np.where(mask == 0, background_alpha, alpha).astype(np.float32)
    pixel_alpha = pixel_alpha[..., np.newaxis]  # (H, W, 1) for broadcast

    image_f = image.astype(np.float32)
    result = image_f * (1.0 - pixel_alpha) + colored_rgb * pixel_alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Bounding box drawing
# ---------------------------------------------------------------------------

def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[BoundingBox],
    labels: Optional[List[str]] = None,
    colors: Optional[Dict[str, RGBUint8]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on an RGB image.

    Args:
        image: RGB image array. Shape: (H, W, 3), dtype uint8.
        boxes: List of bounding boxes as (x1, y1, x2, y2) tuples.
            Coordinates in pixel space, 0-indexed, inclusive.
        labels: Optional list of label strings, one per box. If provided,
            must have the same length as ``boxes``.
        colors: Optional dict mapping label string → RGB uint8 color.
            Boxes with unlabeled or unmapped labels use a default red.
        thickness: Line thickness in pixels. Defaults to 2.

    Returns:
        A copy of the image with boxes drawn. Shape: (H, W, 3), dtype uint8.

    Raises:
        OverlayError: If image shape is invalid or boxes/labels lengths mismatch.

    Example:
        >>> boxes = [(10, 10, 50, 50), (100, 100, 140, 140)]
        >>> labels = ["tumor_bulk", "lymphocyte"]
        >>> result = draw_bounding_boxes(image, boxes, labels)
    """
    _validate_image(image, name="image")
    if labels is not None and len(labels) != len(boxes):
        raise OverlayError(
            f"len(labels)={len(labels)} must equal len(boxes)={len(boxes)}"
        )

    result = image.copy()
    h, w = result.shape[:2]
    default_color: RGBUint8 = (255, 0, 0)  # Red

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Clip to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Determine color
        color = default_color
        if labels is not None and colors is not None:
            color = colors.get(labels[i], default_color)

        # Draw four rectangle edges
        result[y1:y1 + thickness, x1:x2 + 1] = color  # top
        result[y2:y2 + thickness, x1:x2 + 1] = color  # bottom
        result[y1:y2 + 1, x1:x1 + thickness] = color  # left
        result[y1:y2 + 1, x2:x2 + thickness] = color  # right

    return result


# ---------------------------------------------------------------------------
# Patch grid visualization
# ---------------------------------------------------------------------------

def make_patch_grid(
    patches: List[np.ndarray],
    n_cols: int = 8,
    pad: int = 2,
    background_color: RGBUint8 = (240, 240, 240),
) -> np.ndarray:
    """Arrange a list of RGB patches into a grid image.

    All patches must be the same size. Useful for quality-control
    inspection of extracted patches.

    Args:
        patches: List of RGB patch arrays. All must have identical shape
            (H, W, 3) and dtype uint8.
        n_cols: Number of columns in the grid. Defaults to 8.
        pad: Padding in pixels between patches. Defaults to 2.
        background_color: Background fill color as RGB uint8 tuple.

    Returns:
        Grid image as a single RGB array, dtype uint8.

    Raises:
        OverlayError: If patches list is empty, or patches have inconsistent shapes.

    Example:
        >>> grid = make_patch_grid(patches, n_cols=8, pad=2)
        >>> Image.fromarray(grid).save("qc_grid.png")
    """
    if not patches:
        raise OverlayError("patches list is empty")

    # Validate all patches have the same shape
    ref_shape = patches[0].shape
    for i, p in enumerate(patches):
        if p.shape != ref_shape:
            raise OverlayError(
                f"Patch {i} has shape {p.shape} but expected {ref_shape}"
            )
        _validate_image(p, name=f"patches[{i}]")

    patch_h, patch_w = ref_shape[:2]
    n_rows = (len(patches) + n_cols - 1) // n_cols

    cell_h = patch_h + pad
    cell_w = patch_w + pad
    grid_h = n_rows * cell_h + pad
    grid_w = n_cols * cell_w + pad

    grid = np.full((grid_h, grid_w, 3), background_color, dtype=np.uint8)

    for idx, patch in enumerate(patches):
        row = idx // n_cols
        col = idx % n_cols
        y0 = row * cell_h + pad
        x0 = col * cell_w + pad
        grid[y0:y0 + patch_h, x0:x0 + patch_w] = patch

    return grid


# ---------------------------------------------------------------------------
# Thumbnail annotation
# ---------------------------------------------------------------------------

def annotate_tissue_regions(
    thumbnail: np.ndarray,
    tissue_mask: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """Annotate a WSI thumbnail with a tissue detection overlay.

    A simplified wrapper around blend_mask_overlay specifically for
    low-resolution tissue detection masks.

    Args:
        thumbnail: Low-resolution RGB thumbnail. Shape: (H, W, 3), uint8.
        tissue_mask: Binary or class-ID tissue mask. Shape: (H, W), uint8.
            For binary masks, class 0 = background, class 1 = tissue.
        alpha: Overlay blend strength. Defaults to 0.35.

    Returns:
        Annotated thumbnail. Shape: (H, W, 3), dtype uint8.

    Raises:
        OverlayError: If thumbnail and tissue_mask shapes are incompatible.
    """
    return blend_mask_overlay(thumbnail, tissue_mask, alpha=alpha)


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------

def _validate_image(image: np.ndarray, name: str = "image") -> None:
    """Raise OverlayError if image is not a valid (H, W, 3) uint8 array."""
    if not isinstance(image, np.ndarray):
        raise OverlayError(f"'{name}' must be a NumPy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise OverlayError(
            f"'{name}' must have shape (H, W, 3), got {image.shape}"
        )
    if image.dtype != np.uint8:
        raise OverlayError(
            f"'{name}' must be dtype uint8, got {image.dtype}"
        )


def _validate_mask(mask: np.ndarray, name: str = "mask") -> None:
    """Raise OverlayError if mask is not a valid 2-D integer array."""
    if not isinstance(mask, np.ndarray):
        raise OverlayError(f"'{name}' must be a NumPy array, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise OverlayError(f"'{name}' must be 2-D (H, W), got shape {mask.shape}")


def _validate_shapes_match(image: np.ndarray, mask: np.ndarray) -> None:
    """Raise OverlayError if image and mask spatial dimensions differ."""
    if image.shape[:2] != mask.shape:
        raise OverlayError(
            f"image spatial shape {image.shape[:2]} does not match "
            f"mask shape {mask.shape}"
        )
