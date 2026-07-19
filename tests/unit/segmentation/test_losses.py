"""
tests/unit/segmentation/test_losses.py
======================================
Unit tests for the Loss Engine.

Verifies:
- Multiclass DiceLoss forward pass calculations
- Multiclass FocalLoss forward pass calculations
- CombinedLoss combination weights
- LossFactory config resolution and instantiation

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from pathoai.core.exceptions import ValidationError
from pathoai.segmentation.losses import (
    CombinedLoss,
    DiceLoss,
    FocalLoss,
    LossFactory,
)


class TestLossFunctions:
    """Verifies that individual losses output valid scalars and handle class weights."""

    def test_dice_loss_forward(self):
        logits = torch.randn(2, 3, 8, 8)  # 3 classes
        targets = torch.randint(0, 3, (2, 8, 8))

        loss_fn = DiceLoss()
        val = loss_fn(logits, targets)

        assert isinstance(val, torch.Tensor)
        assert val.ndim == 0  # scalar
        assert val.item() >= 0.0

    def test_focal_loss_forward(self):
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))

        alpha = torch.tensor([1.0, 2.0, 1.0])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        val = loss_fn(logits, targets)

        assert isinstance(val, torch.Tensor)
        assert val.ndim == 0
        assert val.item() >= 0.0

    def test_combined_loss_forward(self):
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))

        loss1 = DiceLoss()
        loss2 = nn.CrossEntropyLoss()
        combined = CombinedLoss([(1.0, loss1), (2.0, loss2)])

        val = combined(logits, targets)
        assert isinstance(val, torch.Tensor)
        assert val.ndim == 0


class TestLossFactory:
    """Verifies config-driven loss creation and error conditions."""

    def test_factory_resolves_correctly(self):
        from pathoai.core.config import ConfigNode

        # Cross entropy
        config = ConfigNode({"segmentation": {"loss_name": "ce"}})
        loss = LossFactory.create_loss(config)
        assert isinstance(loss, nn.CrossEntropyLoss)

        # Dice CE Combined
        config = ConfigNode({"segmentation": {"loss_name": "dice_ce"}})
        loss = LossFactory.create_loss(config)
        assert isinstance(loss, CombinedLoss)

    def test_factory_invalid_loss_name_raises(self):
        from pathoai.core.config import ConfigNode
        config = ConfigNode({"segmentation": {"loss_name": "invalid_name_123"}})

        with pytest.raises(ValidationError, match="Unsupported loss name"):
            LossFactory.create_loss(config)
