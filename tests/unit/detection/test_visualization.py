"""
tests/unit/detection/test_visualization.py
===========================================
Unit tests for detection visualizers.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.core.types import BoundingBox, CellDetection, Point
from pathoai.detection.visualization import (
    create_density_heatmap,
    draw_detection_overlay,
)


class TestDetectionVisualization:
    """Test detection visualizers."""

    def test_draw_detection_overlay(self):
        """Test drawing detection bounding boxes on image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            CellDetection(
                detection_id="d1",
                slide_id="s1",
                roi_id="r1",
                bbox=BoundingBox(min_y=10, min_x=10, max_y=50, max_x=50),
                centroid=Point(30.0, 30.0),
                confidence=0.85,
                class_id=2,
                class_name="lymphocyte",
            )
        ]

        overlay = draw_detection_overlay(img, detections)
        assert overlay.shape == (100, 100, 3)
        assert np.any(overlay > 0)

    def test_create_density_heatmap(self):
        """Test creating density heatmap."""
        detections = [
            CellDetection(
                detection_id="d1",
                slide_id="s1",
                roi_id="r1",
                bbox=BoundingBox(min_y=10, min_x=10, max_y=20, max_x=20),
                centroid=Point(15.0, 15.0),
                confidence=0.9,
                class_id=2,
                class_name="lymphocyte",
            )
        ]

        heatmap = create_density_heatmap((100, 100), detections, grid_size=10)
        assert heatmap.shape == (10, 10)
        assert heatmap[1, 1] == 1.0
