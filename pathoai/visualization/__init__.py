"""pathoai.visualization — Figure, overlay, and colormap generation engine.

Exposes:
    colormap: Color lookup tables and Matplotlib colormap builders.
    overlay: Mask blending, bounding box drawing, patch grid utilities.
    wsi_overlay: Slide-level visualization (grid overlays, tissue mask overlays).
"""

from pathoai.visualization.colormap import (
    ColorMapError,
    build_matplotlib_colormap,
    build_tissue_lut,
    colorize_mask,
    get_cell_colormap_uint8,
    get_class_color_legend,
    get_tissue_colormap_bgr,
    get_tissue_colormap_float,
    get_tissue_colormap_uint8,
    rgb_float_to_uint8,
    rgb_to_bgr,
    rgb_uint8_to_float,
)
from pathoai.visualization.overlay import (
    OverlayError,
    annotate_tissue_regions,
    blend_mask_overlay,
    draw_bounding_boxes,
    make_patch_grid,
)
from pathoai.visualization.wsi_overlay import (
    draw_patch_grid_overlay,
    draw_tissue_mask_overlay,
)

__all__ = [
    # colormap
    "ColorMapError",
    "build_matplotlib_colormap",
    "build_tissue_lut",
    "colorize_mask",
    "get_class_color_legend",
    "get_cell_colormap_uint8",
    "get_tissue_colormap_bgr",
    "get_tissue_colormap_float",
    "get_tissue_colormap_uint8",
    "rgb_float_to_uint8",
    "rgb_to_bgr",
    "rgb_uint8_to_float",
    # overlay
    "OverlayError",
    "annotate_tissue_regions",
    "blend_mask_overlay",
    "draw_bounding_boxes",
    "make_patch_grid",
    # wsi_overlay
    "draw_patch_grid_overlay",
    "draw_tissue_mask_overlay",
]
