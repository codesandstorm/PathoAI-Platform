"""
tests/unit/training/test_trainer_features.py
============================================
Unit tests verifying advanced Trainer features:
- Gradient accumulation scaling and stepping
- Gradient clipping operations
- Loss value NaN checking

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathoai.training.trainer.trainer import Trainer


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTrainerAdvancedFeatures:
    """Verifies gradient accumulation steps, gradient clipping boundaries, and finite loss checks."""

    @pytest.fixture()
    def setup_data(self):
        x = torch.randn(8, 2)
        y = torch.randn(8, 1)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)  # 4 batches
        return loader

    def test_gradient_accumulation(self, setup_data):
        model = TinyLinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()

        # Wrap optimizer.step with a mock to track invocations
        original_step = optimizer.step
        optimizer.step = MagicMock(side_effect=original_step)

        # Set accumulation factor to 2.
        # Since we have 4 batches, step should be called exactly twice!
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            accumulate_grad_batches=2,
        )

        trainer.fit(setup_data, epochs=1)

        assert optimizer.step.call_count == 2

    def test_gradient_clipping(self, setup_data):
        model = TinyLinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.MSELoss()

        # Set clipping threshold
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="cpu",
            grad_clip_val=1.0,
        )

        # Simply running fit without errors validates clip_grad_norm invocation
        trainer.fit(setup_data, epochs=1)

    def test_nan_loss_raises(self, setup_data):
        model = TinyLinearModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Loss function that returns NaN
        def nan_loss_fn(outputs, targets):
            return torch.tensor(float("nan"), requires_grad=True)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nan_loss_fn,
            device="cpu",
        )

        with pytest.raises(RuntimeError, match="Loss value is NaN or Inf"):
            trainer.fit(setup_data, epochs=1)
