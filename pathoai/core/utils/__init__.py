"""
pathoai/core/utils/__init__.py
================================
Core utility subpackage.

Exposes mathematical/coordinate helpers (math_utils) and
filesystem/path helpers (path_utils) at the package level.

Author: PathoAI Research Team
Milestone: 1.1
"""

from pathoai.core.utils.math_utils import (
    bootstrap_confidence_interval,
    clip_boxes_to_image,
    compute_box_areas,
    compute_box_centroids,
    compute_iou,
    find_best_pyramid_level,
    level_to_slide_coords,
    pixels_to_mm2,
    pixels_to_um2,
    slide_to_level_coords,
    um2_to_mm2,
)
from pathoai.core.utils.path_utils import (
    PathError,
    create_project_structure,
    ensure_directory,
    ensure_file_exists,
    ensure_parent_exists,
    get_file_size_bytes,
    get_free_disk_space_gb,
    list_files_with_extension,
    resolve_path,
    safe_copy_file,
    validate_project_structure,
)

__all__ = [
    # math_utils
    "bootstrap_confidence_interval",
    "clip_boxes_to_image",
    "compute_box_areas",
    "compute_box_centroids",
    "compute_iou",
    "find_best_pyramid_level",
    "level_to_slide_coords",
    "pixels_to_mm2",
    "pixels_to_um2",
    "slide_to_level_coords",
    "um2_to_mm2",
    # path_utils
    "PathError",
    "create_project_structure",
    "ensure_directory",
    "ensure_file_exists",
    "ensure_parent_exists",
    "get_file_size_bytes",
    "get_free_disk_space_gb",
    "list_files_with_extension",
    "resolve_path",
    "safe_copy_file",
    "validate_project_structure",
]
