"""
tests/unit/segmentation/test_factory.py
=======================================
Unit tests for the model Factory creator.

Verifies:
- Config-driven instantiation of registered models (e.g. DeepLabV3+)
- Correct routing of backbone and pretrained encoder weight properties

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from pathoai.segmentation.architectures.deeplabv3plus import DeepLabV3PlusArch
from pathoai.segmentation.factory import create_model


class TestModelFactory:
    """Verifies that model creation dynamically loads and instantiates models."""

    def test_create_deeplabv3plus_success(self):
        # Create a mock configuration object
        config = MagicMock()
        config.segmentation.model_name = "deeplabv3plus"
        config.segmentation.n_classes = 4
        # Configure backbone properties
        config.segmentation.encoder_name = "resnet18"  # lightweight for testing
        config.segmentation.encoder_weights = None     # disable downloads during test

        # Instantiate model
        model = create_model(config)

        # Assert instantiation properties
        assert isinstance(model, DeepLabV3PlusArch)
        assert model.model.encoder.__class__.__name__ == "ResNetEncoder"

        # Check output classes via forward pass shape
        model.eval()
        dummy_input = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (1, 4, 64, 64)
