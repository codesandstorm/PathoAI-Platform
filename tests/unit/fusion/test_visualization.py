"""
tests/unit/fusion/test_visualization.py
========================================
Unit tests for spatial fusion visualizer.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.core.types import BoundingBox, CellDetection, Point, Polygon, SpatialDetection, TumorROI
from pathoai.fusion.visualization import draw_spatial_fusion_overlay


class TestSpatialVisualization:
    """Test spatial fusion visualizer."""

    def test_draw_spatial_fusion_overlay(self):
        """Test drawing spatial fusion overlay."""
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

        sd = SpatialDetection(
            detection=det,
            roi=roi,
            inside_tumor=True,
            inside_stroma=False,
            distance_to_tumor_boundary_um=0.0,
            distance_to_roi_centroid_um=0.0,
            nearest_boundary_point=Point(25.0, 0.0),
            spatial_label="intratumoral_lymphocyte",
        )

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        overlay = draw_spatial_fusion_overlay(img, [sd], [roi])

        assert overlay.shape == (100, 100, 3)
        assert np.any(overlay > 0)
