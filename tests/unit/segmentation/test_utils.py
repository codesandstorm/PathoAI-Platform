"""
tests/unit/segmentation/test_utils.py
=====================================
Unit tests for the segmentation utilities.

Verifies:
- estimate_model_size_mb estimation math
- verify_output_shape input/output mapping checks
- check_backbone availability indicators in SMP

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pathoai.segmentation.utils import (
    check_backbone,
    estimate_model_size_mb,
    verify_output_shape,
)


class DummyLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 10 parameters (linear layer: weight 3x3, bias 3) = 12 total elements
        # Float32 takes 4 bytes. 12 * 4 = 48 bytes total
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestSegmentationUtils:
    """Verifies parameter count byte conversions and shape assertions."""

    def test_estimate_model_size(self):
        model = DummyLayerModel()
        size_mb = estimate_model_size_mb(model)
        # Size should be extremely small, around 48 / (1024^2) MB
        assert 0.0 < size_mb < 0.01

    def test_verify_output_shape(self):
        model = DummyLayerModel()
        out_shape = verify_output_shape(model, input_shape=(2, 3))
        assert out_shape == (2, 3)

    def test_check_backbone(self):
        # resnet34 is a standard SMP encoder
        assert check_backbone("resnet34")
        # should fail on dummy values
        assert not check_backbone("invalid_nonexistent_backbone_123")
        # case-insensitive check
        assert check_backbone("ResNet34")
