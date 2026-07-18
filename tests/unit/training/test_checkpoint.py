"""
tests/unit/training/test_checkpoint.py
======================================
Unit tests for the CheckpointManager.

Verifies:
- CheckpointManager directory structure and metadata tracking
- Saving best/last checkpoints
- Top-K pruning of worst/oldest epochs
- Checkpoint state serialization and restore resuming

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.6
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pathoai.training.checkpoint.manager import CheckpointManager
from pathoai.training.trainer.state import TrainerState


class TestCheckpointManager:
    """Verifies checkpoint saving, restoration, top-K selection, and pruning."""

    @pytest.fixture()
    def setup_modules(self):
        model = nn.Linear(5, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        return model, opt

    def test_save_and_resume_checkpoints(self, setup_modules, tmp_path: Path):
        model, opt = setup_modules
        mgr = CheckpointManager(
            checkpoint_dir=tmp_path,
            monitor="val_dice",
            mode="max",
            save_top_k=2,
        )

        state = TrainerState(epoch=0, global_step=10, best_metric=-float("inf"))

        # Save Epoch 1: val_dice 0.70
        mgr.save_checkpoint(model, opt, state, current_value=0.70)

        assert (tmp_path / "last.pt").is_file()
        assert (tmp_path / "best.pt").is_file()
        assert (tmp_path / "epoch_001.pt").is_file()
        assert state.best_metric == 0.70

        # Save Epoch 2: val_dice 0.65 (not a new best, but saved as epoch/last)
        state.epoch = 1
        state.global_step = 20
        mgr.save_checkpoint(model, opt, state, current_value=0.65)

        assert (tmp_path / "epoch_002.pt").is_file()
        # Best should still be 0.70 (best.pt unmodified)
        assert state.best_metric == 0.70

        # Save Epoch 3: val_dice 0.75 (new best -> overwrites best.pt, epoch 3 saved, epoch 2 worst -> pruned)
        state.epoch = 2
        state.global_step = 30
        mgr.save_checkpoint(model, opt, state, current_value=0.75)

        assert (tmp_path / "epoch_003.pt").is_file()
        # Top-K = 2. Epoch 2 (0.65) was worst, so epoch_002.pt should be deleted!
        assert not (tmp_path / "epoch_002.pt").is_file()
        assert (tmp_path / "epoch_001.pt").is_file()  # Epoch 1 (0.70) kept

        # Test resume
        new_model = nn.Linear(5, 2)
        new_opt = torch.optim.SGD(new_model.parameters(), lr=0.01)

        # Resume from last.pt (corresponds to Epoch 3 state)
        res_state = mgr.resume_training(new_model, new_opt)

        assert res_state.epoch == 3  # Resume starts at epoch 3 (next epoch is 4)
        assert res_state.global_step == 30
        assert res_state.best_metric == 0.75
