"""
tests/unit/fusion/test_roi_mapper.py
=====================================
Unit tests for ROIMapper.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox, CellDetection, Point, Polygon, TumorROI
from pathoai.fusion.roi_mapper import ROIMapper


class TestROIMapper:
    """Test ROIMapper cell-to-ROI assignment."""

    def test_map_detections(self):
        """Test mapping CellDetection to TumorROI."""
        roi = TumorROI(
            roi_id=1,
            bbox=BoundingBox(min_y=0, min_x=0, max_y=50, max_x=50),
            centroid=Point(25.0, 25.0),
            area_px=2500,
            area_um2=625.0,
            perimeter_um=200.0,
            contours=[
                Polygon(exterior=[
                    Point(0.0, 0.0),
                    Point(50.0, 0.0),
                    Point(50.0, 50.0),
                    Point(0.0, 50.0),
                ])
            ],
            class_label="tumor_bulk",
        )

        det = CellDetection(
            detection_id="d1",
            slide_id="s1",
            roi_id="1",
            bbox=BoundingBox(min_y=20, min_x=20, max_y=30, max_x=30),
            centroid=Point(25.0, 25.0),
            confidence=0.9,
            class_id=2,
            class_name="lymphocyte",
            area_pixels=100.0,
            area_um2=25.0,
        )

        mapper = ROIMapper(mpp=0.5)
        spatial_dets = mapper.map_detections([det], [roi])

        assert len(spatial_dets) == 1
        sd = spatial_dets[0]
        assert sd.inside_tumor is True
        assert sd.roi.roi_id == 1
        assert "intratumoral_lymphocyte" in sd.spatial_label
