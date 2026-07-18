"""
tests/unit/segmentation/test_model.py
=====================================
Unit tests for the SegmentationModel wrapper.

Verifies:
- Parameter counting math
- Encoder parameter freezing and unfreezing
- Model weights saving and loading lifecycle
- Device routing properties

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pathoai.segmentation.model import SegmentationModel


class DummyArch(nn.Module):
    """Synthetic model featuring a structured dummy encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(5, 5)
        self.classifier = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class TestSegmentationModelWrapper:
    """Verifies wrapper operations (gradients constraints, counts, save/load)."""

    @pytest.fixture()
    def wrapped_model(self):
        raw = DummyArch()
        return SegmentationModel(raw)

    def test_parameter_counts(self, wrapped_model):
        counts = wrapped_model.count_parameters()
        # total = (5*5 + 5) [encoder] + (5*2 + 2) [classifier] = 30 + 12 = 42
        assert counts["total"] == 42
        assert counts["trainable"] == 42
        assert counts["non_trainable"] == 0

    def test_encoder_freezing_unfreezing(self, wrapped_model):
        # Freeze
        wrapped_model.freeze_encoder()
        counts_frozen = wrapped_model.count_parameters()
        # Encoder is 30 params -> trainable should now be 12, non_trainable 30
        assert counts_frozen["trainable"] == 12
        assert counts_frozen["non_trainable"] == 30

        # Unfreeze
        wrapped_model.unfreeze_encoder()
        counts_unfrozen = wrapped_model.count_parameters()
        assert counts_unfrozen["trainable"] == 42
        assert counts_unfrozen["non_trainable"] == 0

    def test_save_and_load_weights(self, wrapped_model, tmp_path: Path):
        w_path = tmp_path / "weights.pt"

        # Modify parameter to test loading
        with torch.no_grad():
            wrapped_model.model.classifier.weight.fill_(5.0)

        # Save weights
        wrapped_model.save_weights(w_path)
        assert w_path.is_file()

        # Load into a clean model
        clean_model = SegmentationModel(DummyArch())
        clean_model.load_weights(w_path)

        # Assert weight matches
        assert torch.all(clean_model.model.classifier.weight == 5.0)

    def test_to_device(self, wrapped_model):
        # Simply checks return type and device property routing
        ret = wrapped_model.to_device("cpu")
        assert ret is wrapped_model
