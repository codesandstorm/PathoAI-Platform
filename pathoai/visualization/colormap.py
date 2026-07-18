"""
pathoai/visualization/colormap.py
====================================
Color map utilities for PathoAI-Platform visualizations.

Provides standardized, publication-ready color maps for tissue class
overlays and cell detection results. All color definitions are sourced
from ``pathoai.core.constants`` to ensure consistency across modules.

Design principles:
    - All colormaps are deterministic and match the TIGER annotation schema.
    - Functions return pure Python/NumPy/Matplotlib objects — no side effects.
    - Color values are available as both normalized [0, 1] floats (Matplotlib)
      and uint8 [0, 255] integers (OpenCV/PIL).

Author: PathoAI Research Team
Created: 2026-07-18
Milestone: 1.8
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pathoai.core.constants import (
    CELL_CLASS_COLORS,
    TISSUE_CLASS_COLORS,
    TISSUE_CLASSES,
)
from pathoai.core.exceptions import PathoAIException
from pathoai.core.logger import get_logger

logger = get_logger(__name__)

# Type aliases
RGBUint8 = Tuple[int, int, int]
RGBFloat = Tuple[float, float, float]
RGBAFloat = Tuple[float, float, float, float]


class ColorMapError(PathoAIException):
    """Raised when a colormap operation fails due to invalid inputs."""


# ---------------------------------------------------------------------------
# Color format conversion helpers
# ---------------------------------------------------------------------------

def rgb_uint8_to_float(color: RGBUint8) -> RGBFloat:
    """Convert an RGB tuple from [0, 255] integers to [0.0, 1.0] floats.

    Args:
        color: RGB tuple with values in [0, 255].

    Returns:
        RGB tuple with values in [0.0, 1.0].

    Raises:
        ColorMapError: If any channel value is outside [0, 255].

    Example:
        >>> rgb_uint8_to_float((255, 128, 0))
        (1.0, 0.5019607843137255, 0.0)
    """
    for i, v in enumerate(color):
        if not (0 <= v <= 255):
            raise ColorMapError(
                f"RGB channel {i} value {v} is outside [0, 255]"
            )
    return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)


def rgb_float_to_uint8(color: RGBFloat) -> RGBUint8:
    """Convert an RGB tuple from [0.0, 1.0] floats to [0, 255] integers.

    Args:
        color: RGB tuple with values in [0.0, 1.0].

    Returns:
        RGB tuple with values in [0, 255].

    Raises:
        ColorMapError: If any channel value is outside [0.0, 1.0].

    Example:
        >>> rgb_float_to_uint8((1.0, 0.5, 0.0))
        (255, 127, 0)
    """
    for i, v in enumerate(color):
        if not (0.0 <= v <= 1.0):
            raise ColorMapError(
                f"RGB float channel {i} value {v:.4f} is outside [0.0, 1.0]"
            )
    return (round(color[0] * 255), round(color[1] * 255), round(color[2] * 255))


def rgb_to_bgr(color: RGBUint8) -> RGBUint8:
    """Convert an RGB tuple to BGR order (for OpenCV compatibility).

    Args:
        color: RGB tuple (R, G, B).

    Returns:
        BGR tuple (B, G, R).

    Example:
        >>> rgb_to_bgr((255, 0, 0))
        (0, 0, 255)
    """
    return (color[2], color[1], color[0])


# ---------------------------------------------------------------------------
# Tissue class colormap
# ---------------------------------------------------------------------------

def get_tissue_colormap_uint8() -> Dict[int, RGBUint8]:
    """Return the tissue class colormap as uint8 RGB tuples.

    Sourced from ``TISSUE_CLASS_COLORS`` in constants.py.

    Returns:
        Dict mapping tissue class ID → (R, G, B) uint8 tuple.

    Example:
        >>> cm = get_tissue_colormap_uint8()
        >>> cm[0]  # Background — light gray
        (220, 220, 220)
    """
    return dict(TISSUE_CLASS_COLORS)


def get_tissue_colormap_float() -> Dict[int, RGBFloat]:
    """Return the tissue class colormap as normalized [0, 1] float tuples.

    Suitable for use with Matplotlib colormaps and axes.

    Returns:
        Dict mapping tissue class ID → (R, G, B) float tuple.
    """
    return {
        class_id: rgb_uint8_to_float(color)
        for class_id, color in TISSUE_CLASS_COLORS.items()
    }


def get_tissue_colormap_bgr() -> Dict[int, RGBUint8]:
    """Return the tissue class colormap in BGR order for OpenCV.

    Returns:
        Dict mapping tissue class ID → (B, G, R) uint8 tuple.
    """
    return {
        class_id: rgb_to_bgr(color)
        for class_id, color in TISSUE_CLASS_COLORS.items()
    }


def get_cell_colormap_uint8() -> Dict[int, RGBUint8]:
    """Return the cell detection colormap as uint8 RGB tuples.

    Returns:
        Dict mapping cell class ID → (R, G, B) uint8 tuple.
    """
    return dict(CELL_CLASS_COLORS)


# ---------------------------------------------------------------------------
# NumPy lookup table (LUT) for fast colorization
# ---------------------------------------------------------------------------

def build_tissue_lut(alpha: int = 255) -> np.ndarray:
    """Build a NumPy lookup table (LUT) for fast tissue mask colorization.

    The LUT maps each class ID (0–N) to an RGBA color row. It is designed
    for use with vectorized indexing:

        colored = lut[mask_array]   # shape: (H, W, 4), uint8

    Args:
        alpha: Alpha channel value for all colors. Defaults to 255 (opaque).
            Use a lower value (e.g., 128) for semi-transparent overlays.

    Returns:
        NumPy array of shape (n_classes, 4), dtype uint8.
        Columns: [R, G, B, A].

    Raises:
        ColorMapError: If alpha is not in [0, 255].

    Example:
        >>> lut = build_tissue_lut(alpha=180)
        >>> colored = lut[mask_array]   # (H, W, 4) uint8 RGBA
    """
    if not (0 <= alpha <= 255):
        raise ColorMapError(f"alpha must be in [0, 255], got {alpha}")

    n_classes = max(TISSUE_CLASS_COLORS.keys()) + 1
    lut = np.zeros((n_classes, 4), dtype=np.uint8)

    for class_id, (r, g, b) in TISSUE_CLASS_COLORS.items():
        if 0 <= class_id < n_classes:
            lut[class_id] = [r, g, b, alpha]

    return lut


def colorize_mask(
    mask: np.ndarray,
    lut: Optional[np.ndarray] = None,
    alpha: int = 255,
) -> np.ndarray:
    """Apply a color lookup table to a segmentation mask.

    Args:
        mask: Integer class-ID mask. Shape: (H, W), dtype uint8 or int.
        lut: Pre-built LUT from ``build_tissue_lut()``. If None, a fresh
            LUT is built using the default tissue colormap.
        alpha: Alpha value used when building the LUT (only if lut is None).

    Returns:
        RGBA image array. Shape: (H, W, 4), dtype uint8.

    Raises:
        ColorMapError: If mask is not 2-D or contains out-of-range class IDs.

    Example:
        >>> mask = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        >>> colored = colorize_mask(mask)
        >>> colored.shape
        (2, 2, 4)
    """
    if mask.ndim != 2:
        raise ColorMapError(
            f"mask must be 2-D (H, W), got shape {mask.shape}"
        )

    if lut is None:
        lut = build_tissue_lut(alpha=alpha)

    max_id = mask.max()
    if max_id >= len(lut):
        raise ColorMapError(
            f"mask contains class ID {max_id} but LUT only covers 0–{len(lut) - 1}"
        )

    return lut[mask]


# ---------------------------------------------------------------------------
# Matplotlib colormap generation
# ---------------------------------------------------------------------------

def build_matplotlib_colormap(name: str = "pathoai_tissue"):
    """Build a Matplotlib ListedColormap from the tissue class colors.

    Args:
        name: Name for the Matplotlib colormap. Defaults to
            ``"pathoai_tissue"``.

    Returns:
        ``matplotlib.colors.ListedColormap`` with tissue class colors.

    Raises:
        ColorMapError: If Matplotlib is not installed.

    Example:
        >>> cmap = build_matplotlib_colormap()
        >>> plt.imshow(mask, cmap=cmap, vmin=0, vmax=5)
    """
    try:
        from matplotlib.colors import ListedColormap
    except ImportError as exc:
        raise ColorMapError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        ) from exc

    n_classes = max(TISSUE_CLASS_COLORS.keys()) + 1
    colors_rgba: List[RGBAFloat] = []

    for class_id in range(n_classes):
        if class_id in TISSUE_CLASS_COLORS:
            r, g, b = rgb_uint8_to_float(TISSUE_CLASS_COLORS[class_id])
            colors_rgba.append((r, g, b, 1.0))
        else:
            colors_rgba.append((0.5, 0.5, 0.5, 1.0))  # Gray for unknown

    cmap = ListedColormap(colors_rgba, name=name, N=n_classes)
    logger.debug("Built Matplotlib colormap '%s' with %d colors", name, n_classes)
    return cmap


def get_class_color_legend() -> List[Dict]:
    """Build a legend-ready list of class color entries.

    Returns:
        List of dicts, each with keys ``class_id``, ``class_name``,
        ``rgb_uint8``, and ``hex_color``. Suitable for generating
        plot legends or HTML reports.

    Example:
        >>> legend = get_class_color_legend()
        >>> for entry in legend:
        ...     print(entry["class_name"], entry["hex_color"])
    """
    legend = []
    for class_id, class_name in TISSUE_CLASSES.items():
        color = TISSUE_CLASS_COLORS.get(class_id, (128, 128, 128))
        r, g, b = color
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        legend.append({
            "class_id": class_id,
            "class_name": class_name,
            "rgb_uint8": color,
            "hex_color": hex_color,
        })
    return legend
