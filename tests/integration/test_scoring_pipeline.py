"""
tests/integration/test_scoring_pipeline.py
===========================================
Integration tests for Clinical sTIL Scoring Engine (Milestone 9).

Verifies complete execution workflow from FusionResult inputs to ScoringPipeline,
STILScore and ClinicalReport outputs, and file exports.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import json

from pathoai.core.types import BoundingBox, CellDetection, ClinicalReport, FusionResult, Point, Polygon, SpatialDetection, STILScore, TumorROI
from pathoai.scoring.exporter import export_clinical_report_to_markdown, export_stil_score_to_json
from pathoai.scoring.pipeline import ScoringPipeline


def test_end_to_end_scoring_pipeline(tmp_path):
    """Verifies end-to-end execution of clinical sTIL scoring pipeline."""
    roi1 = TumorROI(
        roi_id=1,
        bbox=BoundingBox(min_y=0, min_x=0, max_y=1000, max_x=1000),
        centroid=Point(500.0, 500.0),
        area_px=1000000,
        area_um2=250000.0,
        perimeter_um=4000.0,
        contours=[
            Polygon(exterior=[
                Point(0.0, 0.0),
                Point(1000.0, 0.0),
                Point(1000.0, 1000.0),
                Point(0.0, 1000.0),
            ])
        ],
    )

    det1 = CellDetection(
        detection_id="det_1",
        slide_id="slide_001",
        roi_id="1",
        bbox=BoundingBox(min_y=400, min_x=400, max_y=420, max_x=420),
        centroid=Point(410.0, 410.0),
        confidence=0.92,
        class_id=2,
        class_name="lymphocyte",
        area_pixels=400.0,
        area_um2=100.0,
    )

    sd1 = SpatialDetection(
        detection=det1,
        roi=roi1,
        inside_tumor=False,
        inside_stroma=True,
        distance_to_tumor_boundary_um=15.0,
        distance_to_roi_centroid_um=100.0,
        nearest_boundary_point=Point(400.0, 410.0),
        spatial_label="peritumoral_stromal_lymphocyte",
    )

    fusion_res = FusionResult(
        slide_id="slide_001",
        spatial_detections=[sd1],
        total_cells=1,
        intratumoral_cells=0,
        stromal_cells=1,
        distant_cells=0,
    )

    pipeline = ScoringPipeline(n_bootstrap_iterations=50)

    # Process scoring
    report = pipeline.process(fusion_res)

    assert isinstance(report, ClinicalReport)
    assert report.slide_id == "slide_001"
    assert isinstance(report.stil_score, STILScore)
    assert report.stil_score.score_percent >= 0.0

    # Exporters
    out_json = tmp_path / "score.json"
    out_md = tmp_path / "clinical_report.md"

    export_stil_score_to_json(report.stil_score, out_json)
    export_clinical_report_to_markdown(report, out_md)

    assert out_json.is_file()
    assert out_md.is_file()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "score_percent" in data
    assert data["slide_id"] == "slide_001"
