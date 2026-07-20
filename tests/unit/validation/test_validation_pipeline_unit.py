"""
tests/unit/validation/test_validation_pipeline_unit.py
======================================================
Unit tests for ValidationPipeline master coordinator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.core.types import BoundingBox, ValidationReport
from pathoai.validation.pipeline import ValidationPipeline


class TestValidationPipelineUnit:
    """Test ValidationPipeline execution."""

    def test_run_validation(self):
        """Test executing master ValidationPipeline run_validation method."""
        pipeline = ValidationPipeline(experiment_name="exp_test", dataset_name="Val_Set")

        seg_gt = np.zeros((30, 30), dtype=np.uint8)
        seg_gt[5:25, 5:25] = 1

        box1 = BoundingBox(5, 5, 20, 20)

        score_true = np.array([10.0, 25.0, 45.0])
        score_pred = np.array([12.0, 24.0, 47.0])

        report = pipeline.run_validation(
            seg_y_true=seg_gt,
            seg_y_pred=seg_gt,
            det_gt_boxes=[box1],
            det_pred_boxes=[box1],
            score_y_true=score_true,
            score_y_pred=score_pred,
            slide_ids=["s1", "s2", "s3"],
        )

        assert isinstance(report, ValidationReport)
        assert report.experiment_name == "exp_test"
        assert report.validation_result.segmentation_metrics.dice == 1.0
        assert report.validation_result.scoring_metrics.icc > 0.9
