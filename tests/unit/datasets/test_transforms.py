"""
tests/unit/datasets/test_transforms.py
======================================
Unit tests for the Albumentations transform pipelines.

Tests cover:
- get_transforms Compose pipeline generation for train vs. validation splits
- Execution of transforms on synthetic image/mask arrays
- Output tensor shapes and types (Image -> 3D Float, Mask -> 2D Long)

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import albumentations as A
import numpy as np
import pytest
import torch

from pathoai.datasets.transforms import get_transforms


class TestTransforms:
    """Verifies that the transform pipelines are constructed and execute correctly."""

    def test_transforms_generation_for_validation(self):
        """Validation transforms must contain only normalize and tensor conversion."""
        pipeline = get_transforms(split="val")
        assert isinstance(pipeline, A.Compose)
        # Check that it doesn't contain random augmentations like flips/rotations
        names = [t.__class__.__name__ for t in pipeline.transforms]
        assert "Normalize" in names
        assert "ToTensorV2" in names
        assert "HorizontalFlip" not in names
        assert "GaussNoise" not in names

    def test_transforms_generation_for_training(self):
        """Training transforms must contain augmentations."""
        pipeline = get_transforms(split="train")
        assert isinstance(pipeline, A.Compose)
        names = [t.__class__.__name__ for t in pipeline.transforms]
        assert "Normalize" in names
        assert "ToTensorV2" in names
        assert "HorizontalFlip" in names
        assert "RandomRotate90" in names

    def test_transforms_execution_on_synthetic_data(self):
        """Transforms must map numpy arrays to standard PyTorch tensors with correct shapes."""
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        mask = np.random.randint(0, 6, (128, 128), dtype=np.uint8)

        pipeline = get_transforms(split="train")
        transformed = pipeline(image=img, mask=mask)

        img_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        # Assert PyTorch tensor types and shapes
        assert isinstance(img_tensor, torch.Tensor)
        assert isinstance(mask_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 128, 128)
        assert mask_tensor.shape == (128, 128)
        assert img_tensor.dtype == torch.float32
        assert mask_tensor.dtype == torch.uint8
