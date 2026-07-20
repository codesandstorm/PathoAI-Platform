"""
tests/unit/validation/test_segmentation_eval.py
=================================================
Unit tests for SegmentationEvaluator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.validation.segmentation import SegmentationEvaluator


class TestSegmentationEvaluator:
    """Test SegmentationEvaluator metrics."""

    def test_evaluate_perfect_match(self):
        """Test metrics when predictions match ground truth perfectly."""
        y = np.zeros((50, 50), dtype=np.uint8)
        y[10:40, 10:40] = 1

        evaluator = SegmentationEvaluator()
        metrics = evaluator.evaluate(y, y)

        assert metrics.dice == 1.0
        assert metrics.iou == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.pixel_accuracy == 1.0
