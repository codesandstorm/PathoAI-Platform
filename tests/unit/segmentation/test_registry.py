"""
tests/unit/segmentation/test_registry.py
========================================
Unit tests for the model architecture registry.

Verifies:
- register_model decorator
- get_model_class resolution and list_registered_models utility
- Exception handling for unknown architecture keys

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from pathoai.core.exceptions import ValidationError
from pathoai.segmentation.registry import (
    get_model_class,
    list_registered_models,
    register_model,
)


class TestModelRegistry:
    """Verifies decorator registration, lookup, list mapping, and validation checks."""

    def test_custom_architecture_registration(self):
        @register_model("mock_segmentor")
        class MockSegmentor(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
            def forward(self, x):
                return x

        # Lookup registered class
        cls = get_model_class("mock_segmentor")
        assert cls is MockSegmentor

        # Check list contains it
        registered = list_registered_models()
        assert "mock_segmentor" in registered

    def test_lookup_unknown_architecture_raises(self):
        with pytest.raises(ValidationError, match="not found in registry"):
            get_model_class("unknown_nonexistent_model")
