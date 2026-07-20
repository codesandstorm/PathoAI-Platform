"""
tests/integration/test_end_to_end_orchestrator.py
===================================================
Integration test for top-level PipelineOrchestrator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.config.experiment_config import ExperimentConfig
from pathoai.core.types import ClinicalReport
from pathoai.pipeline.orchestrator import PipelineOrchestrator
from pathoai.scoring.exporter import export_clinical_report_to_markdown, export_stil_score_to_json


def test_end_to_end_pipeline_orchestrator(tmp_path):
    """Verifies complete top-level orchestrator pipeline run and report exports."""
    config = ExperimentConfig(
        experiment_id="exp_integration_001",
        wsi_mpp=0.5,
        segmentation_model="deeplabv3plus",
        segmentation_version="v1.2",
        detection_model="yolo",
        detection_version="v0.9",
        scoring_algorithm="tiger_working_group",
        scoring_version="v1.0",
        n_bootstrap_iterations=50,
    )

    orchestrator = PipelineOrchestrator(config=config)
    orchestrator.tumor_pipeline.dilation_dist_um = 10.0  # Small footprint for test speed

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    tumor_mask = np.zeros((200, 200), dtype=np.uint8)
    tumor_mask[30:170, 30:170] = 1

    report = orchestrator.run(
        slide_id="slide_full_test",
        image=image,
        tumor_mask=tumor_mask,
    )

    assert isinstance(report, ClinicalReport)
    assert report.slide_id == "slide_full_test"
    assert report.processing_metadata["segmentation_version"] == "v1.2"
    assert report.processing_metadata["detection_version"] == "v0.9"

    out_json = tmp_path / "orchestrated_score.json"
    out_md = tmp_path / "orchestrated_report.md"

    export_stil_score_to_json(report.stil_score, out_json)
    export_clinical_report_to_markdown(report, out_md)

    assert out_json.is_file()
    assert out_md.is_file()
