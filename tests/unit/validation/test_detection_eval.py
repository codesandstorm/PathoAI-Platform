"""
tests/unit/validation/test_detection_eval.py
==============================================
Unit tests for DetectionEvaluator.

Author: PathoAI Research Team
Created: 2026-07-20
"""

from pathoai.core.types import BoundingBox
from pathoai.validation.detection import DetectionEvaluator


class TestDetectionEvaluator:
    """Test DetectionEvaluator metrics."""

    def test_evaluate_detection_boxes(self):
        """Test detection metrics calculation."""
        box1 = BoundingBox(10, 10, 30, 30)
        box2 = BoundingBox(40, 40, 60, 60)

        evaluator = DetectionEvaluator()
        metrics = evaluator.evaluate([box1, box2], [box1])

        assert metrics.tp == 1
        assert metrics.fn == 1
        assert metrics.fp == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 0.5
