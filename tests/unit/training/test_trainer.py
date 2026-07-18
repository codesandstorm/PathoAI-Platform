"""
tests/unit/training/test_trainer.py
===================================
Unit tests for Trainer and TrainerState.

Verifies:
- TrainerState serialization and restoration
- Trainer fitting with a synthetic simple model
- Correct sequence of callback trigger events

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.1
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathoai.training.callbacks.base import Callback
from pathoai.training.trainer.state import TrainerState
from pathoai.training.trainer.trainer import Trainer


class TestTrainerState:
    """Verifies TrainerState properties and conversion utility."""

    def test_state_to_from_dict(self):
        state = TrainerState(
            epoch=5,
            global_step=120,
            best_metric=0.88,
            learning_rate=0.001,
            train_loss=0.25,
            val_loss=0.28,
            elapsed_time=500.0,
        )

        d = state.to_dict()
        assert d["epoch"] == 5
        assert d["elapsed_time"] == 500.0

        restored = TrainerState.from_dict(d)
        assert restored.epoch == 5
        assert restored.best_metric == 0.88
        assert restored.elapsed_time == 500.0


class DummyModel(nn.Module):
    """Simple linear layer model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestTrainerLoop:
    """Verifies that the trainer runs fits and validation loops."""

    @pytest.fixture()
    def setup_data(self):
        # Create small synthetic datasets
        x_train = torch.randn(10, 10)
        # Class targets (0 or 1)
        y_train = torch.randint(0, 2, (10,))
        train_ds = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=2)

        x_val = torch.randn(4, 10)
        y_val = torch.randint(0, 2, (4,))
        val_ds = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=2)

        return train_loader, val_loader

    def test_trainer_fit_and_callbacks(self, setup_data):
        train_loader, val_loader = setup_data

        model = DummyModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Create a mock callback to check lifecycle order
        cb = MagicMock(spec=Callback)

        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=loss_fn,
            callbacks=[cb],
        )

        # Fit for 2 epochs
        trainer.fit(train_loader, val_loader, epochs=2)

        # Assert correct callback triggers
        assert cb.on_train_begin.call_count == 1
        assert cb.on_train_end.call_count == 1
        assert cb.on_epoch_begin.call_count == 2
        assert cb.on_epoch_end.call_count == 2
        assert cb.on_batch_begin.call_count == 10  # 5 batches/epoch * 2 epochs
        assert cb.on_batch_end.call_count == 10
        assert cb.on_validation_begin.call_count == 2
        assert cb.on_validation_end.call_count == 2

        # Check state progression
        assert trainer.state.epoch == 1  # 0-indexed: epoch 0 and epoch 1 completed
        assert trainer.state.global_step == 10
        assert trainer.state.train_loss > 0.0
        assert trainer.state.val_loss > 0.0
        assert trainer.state.elapsed_time > 0.0
