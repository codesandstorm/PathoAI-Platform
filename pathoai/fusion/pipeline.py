"""
pathoai/fusion/pipeline.py
==========================
Spatial Fusion Pipeline Coordinator.

Orchestrates spatial association between TumorROI objects and CellDetection objects,
returning typed SpatialDetection domain models.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 8.11
"""

from __future__ import annotations

from typing import Any, List, Optional

from pathoai.core.types import CellDetection, SpatialDetection, TumorROI
from pathoai.fusion.factory import create_fusion_engine
from pathoai.fusion.roi_mapper import ROIMapper
from pathoai.fusion.validation import SpatialValidator


class FusionPipeline:
    """Coordinating pipeline for spatial fusion reasoning."""

    def __init__(
        self,
        config: Optional[Any] = None,
        max_distance_um: float = 1000.0,
        grid_size: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        config : Optional[Any]
            Config object.
        max_distance_um : float
            Maximum distance threshold in microns.
        grid_size : int
            Grid size for spatial indexing.
        """
        if config is not None:
            engine_cfg = create_fusion_engine(config)
            max_distance_um = engine_cfg["max_distance_um"]
            grid_size = engine_cfg["grid_size"]

        self.max_distance_um = max_distance_um
        self.grid_size = grid_size
        self.validator = SpatialValidator()

    def process(
        self,
        rois: List[TumorROI],
        detections: List[CellDetection],
        mpp: float,
    ) -> List[SpatialDetection]:
        """Executes spatial fusion mapping between CellDetections and TumorROIs.

        Parameters
        ----------
        rois : List[TumorROI]
            Extracted tissue regions.
        detections : List[CellDetection]
            Detected cells.
        mpp : float
            Microns per pixel resolution.

        Returns
        -------
        List[SpatialDetection]
            List of typed SpatialDetection objects.
        """
        if not rois or not detections:
            return []

        mapper = ROIMapper(mpp=mpp, max_distance_um=self.max_distance_um, grid_size=self.grid_size)
        spatial_dets = mapper.map_detections(detections, rois)

        # Validate mappings
        val_status = self.validator.validate_spatial_detections(spatial_dets)
        if not val_status["passed"]:
            raise ValueError(f"Spatial fusion validation failed: {val_status['issues']}")

        return spatial_dets
