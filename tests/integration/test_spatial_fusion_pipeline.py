"""
tests/integration/test_spatial_fusion_pipeline.py
==================================================
Integration tests for Spatial Fusion Engine (Milestone 8).

Verifies complete execution workflow from TumorROI and CellDetection objects
to FusionPipeline spatial reasoning, SpatialDetection creation, and file export formats.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import json

from pathoai.core.types import BoundingBox, CellDetection, Point, Polygon, TumorROI
from pathoai.fusion.exporter import export_spatial_detections_to_json
from pathoai.fusion.pipeline import FusionPipeline


def test_end_to_end_spatial_fusion_pipeline(tmp_path):
    """Verifies end-to-end execution of spatial fusion pipeline."""
    roi1 = TumorROI(
        roi_id=1,
        bbox=BoundingBox(min_y=0, min_x=0, max_y=100, max_x=100),
        centroid=Point(50.0, 50.0),
        area_px=10000,
        area_um2=2500.0,
        perimeter_um=400.0,
        contours=[
            Polygon(exterior=[
                Point(0.0, 0.0),
                Point(100.0, 0.0),
                Point(100.0, 100.0),
                Point(0.0, 100.0),
            ])
        ],
    )

    det1 = CellDetection(
        detection_id="det_1",
        slide_id="slide_001",
        roi_id="1",
        bbox=BoundingBox(min_y=40, min_x=40, max_y=60, max_x=60),
        centroid=Point(50.0, 50.0),
        confidence=0.92,
        class_id=2,
        class_name="lymphocyte",
        area_pixels=400.0,
        area_um2=100.0,
    )
    det2 = CellDetection(
        detection_id="det_2",
        slide_id="slide_001",
        roi_id="1",
        bbox=BoundingBox(min_y=110, min_x=110, max_y=130, max_x=130),
        centroid=Point(120.0, 120.0),
        confidence=0.88,
        class_id=2,
        class_name="lymphocyte",
        area_pixels=400.0,
        area_um2=100.0,
    )

    pipeline = FusionPipeline(max_distance_um=500.0)

    # Process spatial fusion
    spatial_dets = pipeline.process(rois=[roi1], detections=[det1, det2], mpp=0.5)

    assert len(spatial_dets) == 2

    # Verify first detection is intratumoral
    assert spatial_dets[0].inside_tumor is True
    assert spatial_dets[0].spatial_label == "intratumoral_lymphocyte"

    # Verify second detection is peritumoral/distant
    assert spatial_dets[1].inside_tumor is False

    # Verify exporter
    out_json = tmp_path / "spatial_results.json"
    export_spatial_detections_to_json(spatial_dets, out_json)

    assert out_json.is_file()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "spatial_detections" in data
    assert len(data["spatial_detections"]) == 2
