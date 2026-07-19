"""pathoai.fusion — Spatial fusion and sTIL computation engine."""

from pathoai.fusion.aggregator import PatchAggregator
from pathoai.fusion.fusion_engine import FusionEngine
from pathoai.fusion.spatial_ops import (
    calculate_mask_area,
    extract_tumor_associated_stroma,
    extract_tumor_bed,
    filter_points_in_mask,
)
from pathoai.fusion.stil_computer import compute_stil_score

__all__ = [
    "extract_tumor_bed",
    "extract_tumor_associated_stroma",
    "calculate_mask_area",
    "filter_points_in_mask",
    "compute_stil_score",
    "PatchAggregator",
    "FusionEngine",
]
