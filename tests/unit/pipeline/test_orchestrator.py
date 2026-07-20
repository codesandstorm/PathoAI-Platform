"""
tests/unit/pipeline/test_orchestrator.py
=========================================
Unit tests for PipelineOrchestrator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.config.experiment_config import ExperimentConfig
from pathoai.core.types import ClinicalReport
from pathoai.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator:
    """Test top-level PipelineOrchestrator."""

    def test_pipeline_orchestrator_execution(self):
        """Test executing PipelineOrchestrator run() method."""
        cfg = ExperimentConfig(
            experiment_id="test_exp_001",
            segmentation_version="v1.2",
            detection_version="v0.9",
            n_bootstrap_iterations=50,
        )
        orchestrator = PipelineOrchestrator(config=cfg)
        orchestrator.tumor_pipeline.dilation_dist_um = 10.0  # Small footprint for test speed

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tumor_mask = np.zeros((100, 100), dtype=np.uint8)
        tumor_mask[20:80, 20:80] = 1

        report = orchestrator.run(
            slide_id="slide_test_001",
            image=img,
            tumor_mask=tumor_mask,
        )

        assert isinstance(report, ClinicalReport)
        assert report.slide_id == "slide_test_001"
        assert report.processing_metadata["experiment_id"] == "test_exp_001"
        assert report.processing_metadata["segmentation_version"] == "v1.2"
        assert report.processing_metadata["detection_version"] == "v0.9"
