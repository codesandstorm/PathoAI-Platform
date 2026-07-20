"""
tests/unit/scoring/test_scorer.py
==================================
Unit tests for sTILScorer engine.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, CellDetection, FusionResult, Point, SpatialDetection, TumorROI
from pathoai.scoring.scorer import sTILScorer


class TestSTILScorer:
    """Test sTILScorer math."""

    def test_compute_stil_score_percent(self):
        """Test calculating sTIL percentage score."""
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

        scorer = sTILScorer(lymphocyte_diameter_um=10.0)
        score = scorer.compute_stil_score_percent(res)

        assert 0.0 <= score <= 100.0
        # 1 cell area = pi * 25 = 78.54 um^2. Stromal area = 250,000 um^2. % = (78.54 / 250000)*100 = 0.03%
        assert score == 0.03
