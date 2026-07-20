"""
tests/unit/fusion/test_coordinate_index.py
===========================================
Unit tests for SpatialIndex.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, Point, TumorROI
from pathoai.fusion.coordinate_index import SpatialIndex


class TestSpatialIndex:
    """Test spatial index candidate lookup."""

    def test_spatial_index(self):
        """Test index building and query candidate ROIs."""
        roi = TumorROI(
            roi_id=1,
            bbox=BoundingBox(min_y=100, min_x=100, max_y=500, max_x=500),
            centroid=Point(300.0, 300.0),
            area_px=160000,
            area_um2=40000.0,
            perimeter_um=1600.0,
            contours=[],
        )

        idx = SpatialIndex(grid_size=256)
        idx.build_index([roi])

        candidates = idx.query_candidate_rois(Point(300.0, 300.0))
        assert len(candidates) == 1
        assert candidates[0].roi_id == 1
