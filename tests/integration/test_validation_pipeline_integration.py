"""
tests/integration/test_validation_pipeline_integration.py
===========================================================
Integration tests for Scientific Validation & Benchmarking Framework (Milestone 10).

Author: PathoAI Research Team
Created: 2026-07-20
"""

import json

import numpy as np

from pathoai.core.types import BoundingBox, ValidationReport
from pathoai.validation.exporter import export_validation_report_to_markdown, export_validation_result_to_json
from pathoai.validation.pipeline import ValidationPipeline


def test_end_to_end_validation_pipeline_integration(tmp_path):
    """Verifies end-to-end execution of Scientific Validation Framework."""
    pipeline = ValidationPipeline(experiment_name="publication_exp_01", dataset_name="TIGER_Benchmark")

    # Synthetic segmentation masks
    seg_gt = np.zeros((100, 100), dtype=np.uint8)
    seg_gt[20:80, 20:80] = 1
    seg_pred = np.zeros((100, 100), dtype=np.uint8)
    seg_pred[22:78, 22:78] = 1

    # Detection boxes
    box1 = BoundingBox(10, 10, 30, 30)
    box2 = BoundingBox(50, 50, 70, 70)

    # sTIL scores
    score_gt = np.array([15.0, 30.0, 45.0, 60.0])
    score_pred = np.array([16.5, 29.0, 46.0, 58.5])
    slides = ["slide_01", "slide_02", "slide_03", "slide_04"]

    report = pipeline.run_validation(
        seg_y_true=seg_gt,
        seg_y_pred=seg_pred,
        det_gt_boxes=[box1, box2],
        det_pred_boxes=[box1, box2],
        score_y_true=score_gt,
        score_y_pred=score_pred,
        slide_ids=slides,
    )

    assert isinstance(report, ValidationReport)
    assert report.experiment_name == "publication_exp_01"
    assert report.validation_result.scoring_metrics.icc > 0.95

    # Exporters
    out_json = tmp_path / "val_result.json"
    out_md = tmp_path / "val_report.md"

    export_validation_result_to_json(report.validation_result, out_json)
    export_validation_report_to_markdown(report, out_md)

    assert out_json.is_file()
    assert out_md.is_file()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "publication_exp_01"
    assert "scoring_metrics" in data
