"""
tests/unit/training/test_callbacks.py
=====================================
Unit tests for the callback components.

Verifies:
- EarlyStopping halting training on plateau
- LRSchedulerCallback stepping schedulers and updating trainer state LR
- ModelCheckpoint delegating saves correctly to CheckpointManager

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.2
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pathoai.training.callbacks.early_stopping import EarlyStopping
from pathoai.training.callbacks.lr_scheduler import LRSchedulerCallback
from pathoai.training.callbacks.model_checkpoint import ModelCheckpoint
from pathoai.training.trainer.trainer import Trainer


class TestEarlyStopping:
    """Verifies EarlyStopping monitoring behavior and termination flags."""

    def test_early_stopping_min_mode(self):
        trainer = MagicMock()
        trainer.state.epoch = 0
        trainer.stop_training = False
        trainer.current_epoch_metrics = {"val_loss": 0.5}

        cb = EarlyStopping(monitor="val_loss", patience=2, mode="min")
        cb.on_train_begin(trainer)

        # Epoch 1: val_loss 0.5 (best)
        cb.on_epoch_end(trainer)
        assert not trainer.stop_training

        # Epoch 2: val_loss 0.5 (no improvement)
        trainer.state.epoch = 1
        cb.on_epoch_end(trainer)
        assert not trainer.stop_training
        assert cb.wait == 1

        # Epoch 3: val_loss 0.6 (no improvement -> stop)
        trainer.state.epoch = 2
        trainer.current_epoch_metrics = {"val_loss": 0.6}
        cb.on_epoch_end(trainer)
        assert trainer.stop_training
        assert cb.wait == 2

    def test_early_stopping_max_mode(self):
        trainer = MagicMock()
        trainer.state.epoch = 0
        trainer.stop_training = False
        trainer.current_epoch_metrics = {"val_dice": 0.8}

        cb = EarlyStopping(monitor="val_dice", patience=2, mode="max")
        cb.on_train_begin(trainer)

        # Epoch 1: val_dice 0.8
        cb.on_epoch_end(trainer)
        assert not trainer.stop_training

        # Epoch 2: val_dice 0.85 (improvement -> reset)
        trainer.state.epoch = 1
        trainer.current_epoch_metrics = {"val_dice": 0.85}
        cb.on_epoch_end(trainer)
        assert not trainer.stop_training
        assert cb.wait == 0


class TestSchedulerCallback:
    """Verifies scheduling steps and learning rate state parsing."""

    def test_scheduler_callback_step(self):
        trainer = MagicMock()
        mock_group = {"lr": 0.005}
        trainer.optimizer.param_groups = [mock_group]

        mock_sched = MagicMock()
        cb = LRSchedulerCallback(mock_sched, monitor="val_loss")

        # Standard step
        cb.on_epoch_end(trainer)
        mock_sched.step.assert_called_once()
        assert trainer.state.learning_rate == 0.005


class TestModelCheckpointCallback:
    """Verifies that weights checkpoint triggers delegate correct parameters."""

    def test_model_checkpoint_callback(self):
        trainer = MagicMock()
        trainer.current_epoch_metrics = {"val_dice": 0.75}

        mock_mgr = MagicMock()
        mock_mgr.monitor = "val_dice"

        cb = ModelCheckpoint(checkpoint_manager=mock_mgr)
        cb.on_epoch_end(trainer)

        mock_mgr.save_checkpoint.assert_called_once_with(
            model=trainer.model,
            optimizer=trainer.optimizer,
            state=trainer.state,
            current_value=0.75,
        )
