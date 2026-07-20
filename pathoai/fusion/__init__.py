"""pathoai.fusion — Spatial Fusion Engine (Milestone 8)."""

from pathoai.fusion.coordinate_index import SpatialIndex
from pathoai.fusion.exporter import (
    export_spatial_detections_to_csv,
    export_spatial_detections_to_json,
)
from pathoai.fusion.factory import create_fusion_engine
from pathoai.fusion.geometry import (
    calculate_mask_area,
    distance_to_polygon,
    nearest_boundary_point,
    point_in_polygon,
    polygon_area,
    polygon_perimeter,
)
from pathoai.fusion.pipeline import FusionPipeline
from pathoai.fusion.point_filter import filter_points_in_mask
from pathoai.fusion.registry import (
    get_fusion_op,
    list_registered_fusion_ops,
    register_fusion_op,
)
from pathoai.fusion.roi_mapper import ROIMapper
from pathoai.fusion.spatial_intersection import extract_tumor_associated_stroma
from pathoai.fusion.summary import generate_spatial_fusion_summary
from pathoai.fusion.validation import SpatialValidator
from pathoai.fusion.visualization import draw_spatial_fusion_overlay

from pathoai.core.types import FusionResult, SpatialLabel

__all__ = [
    "register_fusion_op",
    "get_fusion_op",
    "list_registered_fusion_ops",
    "create_fusion_engine",
    "SpatialIndex",
    "ROIMapper",
    "FusionPipeline",
    "FusionResult",
    "SpatialLabel",
    "SpatialValidator",
    "point_in_polygon",
    "nearest_boundary_point",
    "distance_to_polygon",
    "polygon_area",
    "polygon_perimeter",
    "extract_tumor_associated_stroma",
    "filter_points_in_mask",
    "calculate_mask_area",
    "export_spatial_detections_to_json",
    "export_spatial_detections_to_csv",
    "draw_spatial_fusion_overlay",
    "generate_spatial_fusion_summary",
]
