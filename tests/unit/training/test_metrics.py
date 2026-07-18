"""
tests/unit/training/test_metrics.py
===================================
Unit tests for the Metrics Engine.

Verifies:
- SegmentationMetrics calculations for multi-class inputs
- ConfusionMatrixMetric mapping and Cohen's Kappa statistic correctness
- MetricCollection update, compute, and reset orchestration

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.3
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pathoai.training.metrics.aggregation import MetricCollection
from pathoai.training.metrics.confusion import ConfusionMatrixMetric
from pathoai.training.metrics.segmentation import SegmentationMetrics


class TestSegmentationMetrics:
    """Verifies Dice, IoU, precision, recall, and support calculations."""

    def test_basic_calculations(self):
        # 3 classes: 0, 1, 2
        # Setup synthetic masks of size (2, 2)
        # true:  [[0, 1], [2, 1]]
        # pred:  [[0, 1], [1, 1]]
        true = torch.tensor([[0, 1], [2, 1]])
        pred = torch.tensor([[0, 1], [1, 1]])

        metric = SegmentationMetrics(n_classes=3)
        metric.update(pred, true)
        res = metric.compute()

        # Class 0: True positive = 1, support = 1. Dice = 2 * 1 / (2*1 + 0 + 0) = 1.0
        assert res["class_0_dice"] == 1.0
        assert res["class_0_iou"] == 1.0

        # Class 1: True positive = 2 (indices [0,1] and [1,1]), False positive = 1 (index [1,0] predicted 1 but true is 2),
        #          False negative = 0 (all true class 1s were predicted 1).
        #          Dice = 2 * 2 / (2*2 + 1 + 0) = 4 / 5 = 0.80
        #          IoU = 2 / (2 + 1 + 0) = 2 / 3 = 0.6667
        assert abs(res["class_1_dice"] - 0.80) < 1e-4
        assert abs(res["class_1_iou"] - 2 / 3) < 1e-4

        # Class 2: True positive = 0, support = 1. Dice = 0
        assert res["class_2_dice"] == 0.0

        # Overall Pixel Accuracy: 3 correct / 4 total = 0.75
        assert res["pixel_accuracy"] == 0.75


class TestConfusionMatrixMetric:
    """Verifies raw/normalized confusion matrix elements and Cohen's Kappa."""

    def test_matrix_and_kappa(self):
        # 2 classes: 0 and 1
        true = torch.tensor([[0, 1], [0, 1]])
        pred = torch.tensor([[0, 1], [1, 0]])

        metric = ConfusionMatrixMetric(n_classes=2)
        metric.update(pred, true)
        res = metric.compute()

        matrix = np.array(res["confusion_matrix"])
        # True is rows (0, 1), Pred is columns (0, 1)
        # true=0: pred=0 once, pred=1 once. -> Row 0 is [1, 1]
        # true=1: pred=0 once, pred=1 once. -> Row 1 is [1, 1]
        assert np.array_equal(matrix, np.array([[1, 1], [1, 1]]))

        # Since it's symmetric 1s, agreement is 50% (2 / 4).
        # Expected agreement is 50% ((2*2 + 2*2)/16) = 8/16 = 50%.
        # Kappa = (0.5 - 0.5) / (1 - 0.5) = 0.0
        assert abs(res["cohens_kappa"] - 0.0) < 1e-4


class TestMetricCollection:
    """Verifies aggregator behavior across sub-metrics."""

    def test_aggregation(self):
        true = torch.tensor([[0, 1], [2, 1]])
        pred = torch.tensor([[0, 1], [1, 1]])

        collection = MetricCollection(n_classes=3)
        collection.update(pred, true)
        res = collection.compute()

        # Check key coverage from both metrics
        assert "pixel_accuracy" in res
        assert "class_0_dice" in res
        assert "cohens_kappa" in res
        assert "confusion_matrix" in res

        # Check reset
        collection.reset()
        res_reset = collection.compute()
        assert res_reset["cohens_kappa"] == 0.0
