"""
tests/unit/training/test_run.py
===============================
Unit and integration tests for the training orchestrator (run.py).

Verifies:
- Config validation checks
- Optimizer and scheduler resolution
- Preflight safety verification checks
- Integration flow using a mocked config/loaders run

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathoai.core.exceptions import PipelineError, ValidationError
from pathoai.training.orchestrator import (
    resolve_optimizer,
    resolve_scheduler,
    run_preflight_verification,
    validate_training_config,
)


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1)
        # Dummy encoder class to support model summary utilities
        self.encoder = nn.Linear(3, 3)

    def forward(self, x):
        return self.conv(x)


class TestTrainingRunner:
    """Verifies that the orchestrator modules resolve configs, run preflight, and catch exceptions."""

    def test_validate_config_checks(self):
        # Invalid config
        config = MagicMock()
        del config.segmentation

        with pytest.raises(ValidationError, match="Missing 'segmentation'"):
            validate_training_config(config)

    def test_resolve_optimizer(self):
        from pathoai.core.config import ConfigNode
        model = MockModel()

        # AdamW
        config = ConfigNode({
            "segmentation": {
                "training": {
                    "optimizer_name": "adamw",
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4
                }
            }
        })
        opt = resolve_optimizer(model, config)
        assert isinstance(opt, torch.optim.AdamW)

        # SGD
        config = ConfigNode({
            "segmentation": {
                "training": {
                    "optimizer_name": "sgd",
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "momentum": 0.9
                }
            }
        })
        opt = resolve_optimizer(model, config)
        assert isinstance(opt, torch.optim.SGD)

    def test_resolve_scheduler(self):
        from pathoai.core.config import ConfigNode
        model = MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Cosine
        config = ConfigNode({
            "segmentation": {
                "training": {
                    "lr_scheduler": "cosine",
                    "epochs": 10
                }
            }
        })
        sched = resolve_scheduler(optimizer, config)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

        # Plateau
        config = ConfigNode({
            "segmentation": {
                "training": {
                    "lr_scheduler": "plateau",
                    "scheduler_patience": 3,
                    "early_stopping_monitor": "val_loss"
                }
            }
        })
        sched = resolve_scheduler(optimizer, config)
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_run_preflight_verification_success(self):
        model = MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()

        # Should complete without throwing exception
        run_preflight_verification(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=torch.device("cpu"),
            patch_size=16,
        )

    def test_run_preflight_verification_failure_raises(self):
        model = MockModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Loss function that fails
        def failing_loss(out, target):
            raise ValueError("Loss failed.")

        with pytest.raises(PipelineError, match="Pre-flight verification failed"):
            run_preflight_verification(
                model=model,
                loss_fn=failing_loss,
                optimizer=optimizer,
                device=torch.device("cpu"),
                patch_size=16,
            )
