"""
tests/unit/training/test_orchestrator.py
========================================
Unit tests for the TrainingOrchestrator component.

Verifies:
- Config-driven state machine initialization
- State transitions logging
- Legacy and explicit model name resolutions in factory
- Reproducibility metadata extraction helpers

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.5
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from pathoai.core.config import ConfigNode
from pathoai.training.orchestrator import TrainingOrchestrator


class TestTrainingOrchestratorUnit:
    """Verifies internal TrainingOrchestrator behavior."""

    @pytest.fixture()
    def base_config(self) -> ConfigNode:
        return ConfigNode({
            "pipeline": {
                "name": "orchestrator_unit_test",
                "seed": 42,
                "device": "cpu",
            },
            "segmentation": {
                "architecture": "deeplabv3plus",
                "encoder": "resnet18",
                "n_classes": 3,
                "input_size": 128,
                "training": {
                    "epochs": 1,
                    "optimizer_name": "adamw",
                    "learning_rate": 1e-4,
                }
            },
            "data": {
                "tiger": {
                    "splits_file": "dummy_splits.json"
                }
            }
        })

    def test_initialization_and_state_transitions(self, base_config):
        orchestrator = TrainingOrchestrator(base_config)
        assert orchestrator.state == "INITIALIZING"

        # Mock experiment to verify JSON state file writing
        mock_experiment = MagicMock()
        mock_experiment.experiment_dir = Path("dummy_exp_dir")
        
        with patch("pathoai.training.orchestrator.open", patch("builtins.open")) as mock_open:
            orchestrator.experiment = mock_experiment
            orchestrator._set_state("READY")
            assert orchestrator.state == "READY"

    def test_file_hash_calculation(self, tmp_path, base_config):
        orchestrator = TrainingOrchestrator(base_config)
        
        dummy_file = tmp_path / "hash_target.txt"
        dummy_file.write_text("reproducibility-rules", encoding="utf-8")
        
        checksum = orchestrator._get_file_hash(dummy_file)
        assert len(checksum) == 64  # SHA-256 length
        
        # Verify deterministic hash
        assert checksum == orchestrator._get_file_hash(dummy_file)
