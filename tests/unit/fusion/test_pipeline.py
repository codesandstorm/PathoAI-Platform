"""
tests/unit/fusion/test_pipeline.py
===================================
Unit tests for FusionPipeline coordinator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, CellDetection, Point, Polygon, TumorROI, FusionResult
from pathoai.fusion.pipeline import FusionPipeline


class TestFusionPipeline:
    """Test FusionPipeline coordinator."""

    def test_process_workflow(self):
        """Test processing workflow from rois and detections to SpatialDetection list."""
        roi = TumorROI(
            roi_id=1,
            bbox=BoundingBox(0, 0, 50, 50),
            centroid=Point(25.0, 25.0),
            area_px=2500,
            area_um2=625.0,
            perimeter_um=200.0,
            contours=[Polygon(exterior=[Point(0.0, 0.0), Point(50.0, 0.0), Point(50.0, 50.0), Point(0.0, 50.0)])],
        )
        det = CellDetection("d1", "s1", "1", BoundingBox(20, 20, 30, 30), Point(25.0, 25.0), 0.9, 2, "lymphocyte")

        pipeline = FusionPipeline(max_distance_um=500.0)
        spatial_dets = pipeline.process([roi], [det], mpp=0.5)

        assert len(spatial_dets) == 1
        assert spatial_dets[0].inside_tumor is True

    def test_process_fusion_returns_result_dto(self):
        """Test process_fusion returning FusionResult container object."""
        roi = TumorROI(
            roi_id=1,
            bbox=BoundingBox(0, 0, 50, 50),
            centroid=Point(25.0, 25.0),
            area_px=2500,
            area_um2=625.0,
            perimeter_um=200.0,
            contours=[Polygon(exterior=[Point(0.0, 0.0), Point(50.0, 0.0), Point(50.0, 50.0), Point(0.0, 50.0)])],
        )
        det = CellDetection("d1", "s1", "1", BoundingBox(20, 20, 30, 30), Point(25.0, 25.0), 0.9, 2, "lymphocyte")

        pipeline = FusionPipeline(max_distance_um=500.0)
        result = pipeline.process_fusion([roi], [det], mpp=0.5, slide_id="s1")

        assert isinstance(result, FusionResult)
        assert result.slide_id == "s1"
        assert result.total_cells == 1
        assert result.intratumoral_cells == 1
        assert result.processing_time_s >= 0.0
