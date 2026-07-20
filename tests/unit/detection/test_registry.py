"""
tests/unit/detection/test_registry.py
======================================
Unit tests for detector registry.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import pytest
import torch.nn as nn

from pathoai.core.exceptions import ValidationError
from pathoai.detection.registry import (
    get_detector_class,
    list_registered_detectors,
    register_detector,
)


class TestDetectorRegistry:
    """Test detector architecture registry functionality."""

    def test_list_registered_detectors(self):
        """Test listing registered detectors includes yolo."""
        detectors = list_registered_detectors()
        assert "yolo" in detectors

    def test_get_detector_class_success(self):
        """Test retrieving registered detector class."""
        cls = get_detector_class("yolo")
        assert issubclass(cls, nn.Module)

    def test_get_detector_class_not_found(self):
        """Test exception raised for unregistered detector."""
        with pytest.raises(ValidationError, match="Detector architecture 'nonexistent' not found"):
            get_detector_class("nonexistent")

    def test_custom_detector_registration(self):
        """Test registering custom detector subclass."""
        @register_detector("dummy_detector")
        class DummyDetector(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

        assert "dummy_detector" in list_registered_detectors()
        fetched = get_detector_class("dummy_detector")
        assert fetched is DummyDetector
