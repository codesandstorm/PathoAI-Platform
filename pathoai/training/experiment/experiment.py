"""
pathoai/training/experiment/experiment.py
=========================================
Experiment Directory Layout Orchestrator.

Initializes the experiment workspace folder structure (checkpoints, history,
curves, confusion matrices, logs, tensorboard) and saves configurations.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.10
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict

from pathoai.core.logger import get_logger
from pathoai.training.logging.csv_logger import CSVLogger
from pathoai.training.logging.tensorboard import TensorBoardLogger

logger = get_logger(__name__)


class Experiment:
    """Orchestrates directory creation and config backups for training runs."""

    def __init__(
        self,
        experiment_dir: str | Path,
        config: Any,
    ) -> None:
        """
        Parameters
        ----------
        experiment_dir : str | Path
            Base directory path for this training experiment.
        config : Any
            The global ConfigNode pipeline configuration.
        """
        self.experiment_dir = Path(experiment_dir)
        self.config = config

        # Setup standard subdirectories
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.history_dir = self.experiment_dir / "history"
        self.metrics_dir = self.experiment_dir / "metrics"
        self.curves_dir = self.experiment_dir / "curves"
        self.confusion_dir = self.experiment_dir / "confusion"
        self.predictions_dir = self.experiment_dir / "predictions"
        self.reports_dir = self.experiment_dir / "reports"
        self.logs_dir = self.experiment_dir / "logs"
        self.tensorboard_dir = self.experiment_dir / "tensorboard"

    def setup(self) -> None:
        """Create all subdirectories and copy the current configuration file."""
        dirs = [
            self.experiment_dir,
            self.checkpoints_dir,
            self.history_dir,
            self.metrics_dir,
            self.curves_dir,
            self.confusion_dir,
            self.predictions_dir,
            self.reports_dir,
            self.logs_dir,
            self.tensorboard_dir,
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Save config backup inside the experiment directory
        config_backup_path = self.experiment_dir / "config.yaml"
        try:
            # Check if config is a ConfigNode and supports serialize/to_dict
            if hasattr(self.config, "to_dict"):
                config_dict = self.config.to_dict()
            elif isinstance(self.config, dict):
                config_dict = self.config
            else:
                config_dict = vars(self.config)

            with open(config_backup_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info("Saved configuration backup to %s", config_backup_path)
        except Exception as exc:
            logger.error("Failed to backup configuration file: %s", exc)

    def get_loggers(self) -> list[Any]:
        """Instantiate CSV and TensorBoard loggers pointing to correct directories."""
        csv_filename = self.history_dir / "history.csv"
        return [
            CSVLogger(filename=csv_filename),
            TensorBoardLogger(log_dir=self.tensorboard_dir),
        ]
