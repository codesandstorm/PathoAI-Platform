"""
tests/unit/scoring/test_statistics.py
======================================
Unit tests for StatisticsEngine.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, CellDetection, FusionResult, Point, Polygon, SpatialDetection, TumorROI
from pathoai.scoring.statistics import StatisticsEngine


class TestStatisticsEngine:
    """Test StatisticsEngine physical count and density math."""

    def test_compute_statistics(self):
        """Test computing count and density statistics."""
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

        stats_engine = StatisticsEngine()
        stats = stats_engine.compute_statistics(res)

        assert stats["total_cells"] == 1
        assert stats["stromal_lymphocytes"] == 1
        assert stats["stromal_area_mm2"] == 0.25  # 250,000 um^2 = 0.25 mm^2
        assert stats["lymphocyte_density_per_mm2"] == 4.0  # 1 / 0.25 = 4.0
