"""
tests/unit/detection/test_metrics.py
=====================================
Unit tests for DetectionMetrics.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.detection.metrics import DetectionMetrics


class TestDetectionMetrics:
    """Test detection metrics calculation."""

    def test_compute_precision_recall(self):
        """Test precision, recall, and F1 computation."""
        calc = DetectionMetrics(iou_threshold=0.5)

        pred_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        pred_scores = np.array([0.9])
        gt_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])

        prec, rec, f1 = calc.compute_precision_recall(pred_boxes, pred_scores, gt_boxes)
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0
