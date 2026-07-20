"""
tests/unit/detection/test_pipeline.py
======================================
Unit tests for DetectionPipeline.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.core.types import BoundingBox, Point, TumorROI
from pathoai.detection.pipeline import DetectionPipeline


class TestDetectionPipeline:
    """Test DetectionPipeline orchestrator."""

    def test_process_roi(self):
        """Test processing single TumorROI."""
        pipeline = DetectionPipeline(
            tile_size=64,
            overlap=16,
            confidence_threshold=0.0,
        )

        roi = TumorROI(
            roi_id=1,
            bbox=BoundingBox(min_y=10, min_x=10, max_y=100, max_x=100),
            centroid=Point(55.0, 55.0),
            area_px=8100,
            area_um2=2025.0,
            perimeter_um=360.0,
            contours=[],
        )

        img = np.zeros((150, 150, 3), dtype=np.uint8)
        detections = pipeline.process_roi(roi, img, slide_id="slide_001", mpp=0.5)

        assert isinstance(detections, list)
