"""pathoai.fusion — Spatial fusion engine (Milestone 8)."""

from pathoai.fusion.geometry import calculate_mask_area
from pathoai.fusion.point_filter import filter_points_in_mask
from pathoai.fusion.spatial_intersection import extract_tumor_associated_stroma

__all__ = [
    "extract_tumor_associated_stroma",
    "filter_points_in_mask",
    "calculate_mask_area",
]
