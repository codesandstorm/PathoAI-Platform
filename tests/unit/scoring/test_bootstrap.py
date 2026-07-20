"""
tests/unit/scoring/test_bootstrap.py
=====================================
Unit tests for BootstrapEngine.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, CellDetection, FusionResult, Point, SpatialDetection, TumorROI
from pathoai.scoring.bootstrap import BootstrapEngine


class TestBootstrapEngine:
    """Test BootstrapEngine confidence interval calculation."""

    def test_compute_confidence_interval(self):
        """Test bootstrap confidence interval bounds."""
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

        boot = BootstrapEngine(n_iterations=100, seed=42)
        ci_lower, ci_upper = boot.compute_confidence_interval(res, base_score=15.0)

        assert 0.0 <= ci_lower <= ci_upper <= 100.0
