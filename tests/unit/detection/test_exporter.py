"""
tests/unit/detection/test_exporter.py
======================================
Unit tests for detection exporters.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import json
from pathlib import Path

from pathoai.core.types import BoundingBox, CellDetection, Point
from pathoai.detection.exporter import (
    export_to_coco,
    export_to_csv,
    export_to_json,
    export_to_yolo_txt,
)


class TestDetectionExporters:
    """Test detection exporters."""

    def test_exporters(self, tmp_path):
        """Test exporting to JSON, CSV, COCO, and YOLO TXT files."""
        detections = [
            CellDetection(
                detection_id="det_1",
                slide_id="slide_1",
                roi_id="roi_1",
                bbox=BoundingBox(min_y=10, min_x=10, max_y=30, max_x=30),
                centroid=Point(20.0, 20.0),
                confidence=0.9,
                class_id=2,
                class_name="lymphocyte",
                area_pixels=400.0,
                area_um2=100.0,
            )
        ]

        json_path = tmp_path / "det.json"
        csv_path = tmp_path / "det.csv"
        coco_path = tmp_path / "coco.json"
        yolo_path = tmp_path / "yolo.txt"

        export_to_json(detections, json_path)
        export_to_csv(detections, csv_path)
        export_to_coco(detections, coco_path)
        export_to_yolo_txt(detections, yolo_path)

        assert json_path.is_file()
        assert csv_path.is_file()
        assert coco_path.is_file()
        assert yolo_path.is_file()

        with open(json_path) as f:
            data = json.load(f)
            assert len(data["detections"]) == 1
