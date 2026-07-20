"""
tests/unit/detection/test_factory.py
=====================================
Unit tests for detector factory.

Author: PathoAI Research Team
Created: 2026-07-20
"""

import torch.nn as nn

from pathoai.detection.factory import create_detector


class TestDetectorFactory:
    """Test detector factory creation logic."""

    def test_create_detector_from_dict(self):
        """Test creating detector from config dictionary."""
        config = {
            "detection": {
                "architecture": "yolo",
                "n_classes": 3,
                "in_channels": 3,
            }
        }
        model = create_detector(config)
        assert isinstance(model, nn.Module)
