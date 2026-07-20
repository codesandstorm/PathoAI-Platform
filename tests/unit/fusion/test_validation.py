"""
tests/unit/fusion/test_validation.py
=====================================
Unit tests for SpatialValidator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, CellDetection, Point, SpatialDetection, TumorROI
from pathoai.fusion.validation import SpatialValidator


class TestSpatialValidator:
    """Test SpatialValidator consistency checker."""

    def test_validate_spatial_detections(self):
        """Test validating SpatialDetection objects."""
        roi = TumorROI(1, BoundingBox(0, 0, 10, 10), Point(5.0, 5.0), 100, 25.0, 40.0, [])
        det = CellDetection("d1", "s1", "1", BoundingBox(0, 0, 5, 5), Point(2.5, 2.5), 0.9, 2, "lymphocyte")

        sd = SpatialDetection(
            detection=det,
            roi=roi,
            inside_tumor=True,
            inside_stroma=False,
            distance_to_tumor_boundary_um=0.0,
            distance_to_roi_centroid_um=3.5,
            nearest_boundary_point=Point(5.0, 0.0),
            spatial_label="intratumoral_lymphocyte",
        )

        val = SpatialValidator()
        res = val.validate_spatial_detections([sd])

        assert res["passed"] is True
        assert res["valid_detections"] == 1
