"""
tests/unit/scoring/test_pipeline.py
====================================
Unit tests for ScoringPipeline coordinator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, CellDetection, ClinicalReport, FusionResult, Point, SpatialDetection, STILScore, TumorROI
from pathoai.scoring.pipeline import ScoringPipeline


class TestScoringPipeline:
    """Test ScoringPipeline process flow."""

    def test_process(self):
        """Test process workflow from FusionResult to ClinicalReport DTO."""
        roi = TumorROI(1, BoundingBox(0, 0, 1000, 1000), Point(500.0, 500.0), 1000000, 250000.0, 4000.0, [])
        det = CellDetection("d1", "s1", "1", BoundingBox(0, 0, 10, 10), Point(5.0, 5.0), 0.9, 2, "lymphocyte")

        sd = SpatialDetection(
            detection=det,
            roi=roi,
            inside_tumor=False,
            inside_stroma=True,
            distance_to_tumor_boundary_um=10.0,
            distance_to_roi_centroid_um=500.0,
            nearest_boundary_point=Point(0.0, 5.0),
            spatial_label="peritumoral_stromal_lymphocyte",
        )

        res = FusionResult(
            slide_id="s1",
            spatial_detections=[sd],
            total_cells=1,
            intratumoral_cells=0,
            stromal_cells=1,
            distant_cells=0,
        )

        pipeline = ScoringPipeline(n_bootstrap_iterations=50)
        report = pipeline.process(res)

        assert isinstance(report, ClinicalReport)
        assert report.slide_id == "s1"
        assert isinstance(report.stil_score, STILScore)
        assert report.stil_score.clinical_category == "Low"
