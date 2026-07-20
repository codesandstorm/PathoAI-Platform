"""
tests/unit/detection/test_postprocessing.py
============================================
Unit tests for post-processing functions.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import numpy as np

from pathoai.detection.postprocessing import apply_nms, compute_iou


class TestPostprocessing:
    """Test NMS and IoU functions."""

    def test_compute_iou(self):
        """Test IoU calculation."""
        boxA = np.array([0.0, 0.0, 10.0, 10.0])
        boxB = np.array([5.0, 0.0, 15.0, 10.0])
        # Intersection width 5, height 10 = 50 area. Union area = 100 + 100 - 50 = 150. IoU = 50/150 = 0.333333
        iou = compute_iou(boxA, boxB)
        assert abs(iou - 0.333333) < 1e-4

    def test_apply_nms(self):
        """Test Non-Maximum Suppression."""
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 10.0, 10.0],  # High overlap with box 0
            [50.0, 50.0, 60.0, 60.0], # Disjoint
        ])
        scores = np.array([0.9, 0.8, 0.95])
        labels = np.array([1, 1, 1])

        clean_boxes, clean_scores, clean_labels = apply_nms(boxes, scores, labels, iou_threshold=0.5)
        assert len(clean_boxes) == 2
