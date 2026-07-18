"""
tests/unit/training/test_experiment.py
======================================
Unit tests for the Experiment directory structure orchestrator.

Verifies:
- Experiment directory structure creation
- Saving of configuration backup yaml files
- Correct logger class instantiations

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.10
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pathoai.training.experiment.experiment import Experiment
from pathoai.training.logging.csv_logger import CSVLogger
from pathoai.training.logging.tensorboard import TensorBoardLogger


class TestExperimentOrchestrator:
    """Verifies that experiments layout correct paths and initialize loggers."""

    def test_experiment_directory_setup(self, tmp_path: Path):
        config_dict = {
            "pipeline": {"name": "test_run", "seed": 42},
            "segmentation": {"model_name": "deeplabv3plus"},
        }

        # Instantiate Experiment orchestrator
        exp = Experiment(experiment_dir=tmp_path, config=config_dict)
        exp.setup()

        # Check directories exist
        assert (tmp_path / "checkpoints").is_dir()
        assert (tmp_path / "history").is_dir()
        assert (tmp_path / "curves").is_dir()
        assert (tmp_path / "confusion").is_dir()
        assert (tmp_path / "predictions").is_dir()
        assert (tmp_path / "reports").is_dir()
        assert (tmp_path / "logs").is_dir()
        assert (tmp_path / "tensorboard").is_dir()

        # Check configuration backup
        cfg_backup = tmp_path / "config.yaml"
        assert cfg_backup.is_file()

        with open(cfg_backup, encoding="utf-8") as f:
            saved_cfg = yaml.safe_load(f)
        assert saved_cfg["pipeline"]["name"] == "test_run"
        assert saved_cfg["segmentation"]["model_name"] == "deeplabv3plus"

        # Check loggers list
        loggers = exp.get_loggers()
        assert len(loggers) == 2
        assert isinstance(loggers[0], CSVLogger)
        assert isinstance(loggers[1], TensorBoardLogger)
