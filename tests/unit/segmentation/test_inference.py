"""
tests/unit/segmentation/test_inference.py
=========================================
Unit tests for the Inference Engine.

Verifies:
- preprocess_numpy_image array conversions
- predict_patch, predict_batch, and predict_probabilities operations
- save_prediction_mask and save_prediction_overlay file saves

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pathoai.segmentation.inference import (
    SegmentationInference,
    preprocess_numpy_image,
)


class MockSimpleSegmentor(nn.Module):
    """Mock model that outputs fixed predictions for testing."""

    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.n_classes = n_classes
        # Dummy param to allow parameters() iteration
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Return logits: class 1 has maximum logit values
        logits = torch.zeros(B, self.n_classes, H, W, dtype=torch.float32)
        logits[:, 1, ...] = 5.0
        return logits


class TestSegmentationInference:
    """Verifies image preprocessing, prediction argmaxes, and image exports."""

    def test_preprocess_numpy_image(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        tensor = preprocess_numpy_image(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 32, 32)
        assert tensor.dtype == torch.float32

    def test_inference_predictions(self):
        model = MockSimpleSegmentor(n_classes=3)
        engine = SegmentationInference(model, device="cpu")

        # 1. Single patch prediction
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        pred = engine.predict_patch(img)

        assert isinstance(pred, np.ndarray)
        assert pred.shape == (16, 16)
        # Class 1 logits were set to 5.0, so argmax should resolve to 1
        assert np.all(pred == 1)

        # 2. Batch prediction
        batch_input = torch.zeros(2, 3, 16, 16, dtype=torch.float32)
        batch_pred = engine.predict_batch(batch_input)

        assert isinstance(batch_pred, torch.Tensor)
        assert batch_pred.shape == (2, 16, 16)
        assert torch.all(batch_pred == 1)

        # 3. Probabilities prediction
        probs = engine.predict_probabilities(img)
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (3, 16, 16)
        # Class 1 should have high probability
        assert np.all(probs[1, ...] > 0.8)

    def test_save_prediction_files(self, tmp_path: Path):
        model = MockSimpleSegmentor(n_classes=3)
        engine = SegmentationInference(model, device="cpu")

        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[2:8, 2:8] = 2  # class 2 stroma

        out_mask = tmp_path / "mask.png"
        out_overlay = tmp_path / "overlay.png"

        engine.save_prediction_mask(mask, out_mask)
        engine.save_prediction_overlay(mask, out_overlay, alpha=180)

        assert out_mask.is_file()
        assert out_overlay.is_file()
