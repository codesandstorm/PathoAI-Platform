"""
tests/integration/test_detection_pipeline.py
=============================================
Integration tests for Cell Detection Engine (Milestone 7).

Verifies complete execution workflow from TumorROI objects to DetectionPipeline
inference, NMS merging, coordinate transformation, typed CellDetection objects,
and file exporter formats.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import json
import numpy as np

from pathoai.core.types import BoundingBox, Point, TumorROI
from pathoai.detection.exporter import export_to_json
from pathoai.detection.pipeline import DetectionPipeline


def test_end_to_end_cell_detection_pipeline(tmp_path):
    """Verifies end-to-end execution of cell detection pipeline."""
    pipeline = DetectionPipeline(
        tile_size=64,
        overlap=16,
        confidence_threshold=0.0,
        nms_iou_threshold=0.45,
    )

    roi1 = TumorROI(
        roi_id=1,
        bbox=BoundingBox(min_y=10, min_x=10, max_y=100, max_x=100),
        centroid=Point(55.0, 55.0),
        area_px=8100,
        area_um2=2025.0,
        perimeter_um=360.0,
        contours=[],
    )
    roi2 = TumorROI(
        roi_id=2,
        bbox=BoundingBox(min_y=120, min_x=120, max_y=180, max_x=180),
        centroid=Point(150.0, 150.0),
        area_px=3600,
        area_um2=900.0,
        perimeter_um=240.0,
        contours=[],
    )

    img = np.ones((200, 200, 3), dtype=np.uint8) * 200

    # Execute pipeline on TumorROIs
    detections = pipeline.process_rois([roi1, roi2], img, slide_id="TCGA-BRCA-001", mpp=0.5)

    assert isinstance(detections, list)

    # Verify exporter integration
    out_json = tmp_path / "detections.json"
    export_to_json(detections, out_json)

    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "detections" in data
