"""
pathoai/experiments/tracker.py
===============================
Experiment Tracker Engine.

Logs experiment configurations, runtime manifests, and validation metrics into structured
directory artifacts.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pathoai.config.experiment_config import ExperimentConfig
from pathoai.core.types import ExperimentManifest, ValidationResult
from pathoai.experiments.manifest import ManifestGenerator


class ExperimentTracker:
    """Logs experiment artifacts, manifests, and validation results."""

    def __init__(self, base_dir: Union[str, Path] = "experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_gen = ManifestGenerator()

    def log_experiment(
        self,
        config: ExperimentConfig,
        val_result: ValidationResult,
        duration_s: float = 0.0,
    ) -> Path:
        """Logs an experiment run into an experiment artifact directory.

        Parameters
        ----------
        config : ExperimentConfig
            Experiment configuration object.
        val_result : ValidationResult
            Master validation result object.
        duration_s : float
            Total run time in seconds.

        Returns
        -------
        Path
            Output directory path.
        """
        exp_dir = self.base_dir / config.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.manifest_gen.create_manifest(config, val_result.dataset_name, duration_s)
        self.manifest_gen.export_manifest_to_json(manifest, exp_dir / "manifest.json")

        summary_metrics = {
            "experiment_id": config.experiment_id,
            "dice": val_result.segmentation_metrics.dice,
            "f1_detection": val_result.detection_metrics.f1,
            "icc": val_result.scoring_metrics.icc,
            "mae": val_result.scoring_metrics.mae,
            "rmse": val_result.scoring_metrics.rmse,
        }

        with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        return exp_dir
