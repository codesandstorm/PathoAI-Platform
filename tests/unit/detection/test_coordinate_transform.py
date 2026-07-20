"""
tests/unit/detection/test_coordinate_transform.py
===================================================
Unit tests for CoordinateTransformer.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.detection.coordinate_transform import CoordinateTransformer


class TestCoordinateTransformer:
    """Test coordinate transformer."""

    def test_create_cell_detections(self):
        """Test generating typed CellDetection objects."""
        transformer = CoordinateTransformer(mpp=0.5)

        boxes = np.array([[10.0, 20.0, 30.0, 40.0]])
        scores = np.array([0.95])
        labels = np.array([2])

        detections = transformer.create_cell_detections(
            slide_boxes=boxes,
            scores=scores,
            labels=labels,
            slide_id="slide_001",
            roi_id="roi_001",
        )

        assert len(detections) == 1
        d = detections[0]
        assert d.slide_id == "slide_001"
        assert d.roi_id == "roi_001"
        assert d.class_id == 2
        assert d.class_name == "lymphocyte"
        assert d.centroid.x == 20.0
        assert d.centroid.y == 30.0
        assert d.area_pixels == 400.0
        assert d.area_um2 == 100.0  # 400 * 0.5^2
