"""
pathoai/experiments/manifest.py
================================
Experiment Manifest Generator.

Generates and serializes complete ExperimentManifest DTOs.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 10.5.2
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from pathoai.config.experiment_config import ExperimentConfig
from pathoai.core.types import ExperimentManifest
from pathoai.experiments.environment import EnvironmentAuditor


class ManifestGenerator:
    """Generates ExperimentManifest DTO objects."""

    def __init__(self) -> None:
        self.auditor = EnvironmentAuditor()

    def create_manifest(
        self, config: ExperimentConfig, dataset_name: str = "TIGER_Val", duration_s: float = 0.0
    ) -> ExperimentManifest:
        """Assembles ExperimentManifest from ExperimentConfig and runtime audit."""
        env = self.auditor.capture_environment()
        now_str = datetime.now().isoformat()

        return ExperimentManifest(
            experiment_id=config.experiment_id,
            dataset_name=dataset_name,
            git_commit=env["git_commit"],
            python_version=env["python_version"],
            pytorch_version=env["pytorch_version"],
            cuda_version=env["cuda_version"],
            gpu_name=env["gpu_name"],
            random_seed=42,
            segmentation_model=config.segmentation_model,
            segmentation_version=config.segmentation_version,
            detection_model=config.detection_model,
            detection_version=config.detection_version,
            scoring_version=config.scoring_version,
            validation_version="v1.0",
            execution_date=now_str,
            duration_s=float(round(duration_s, 2)),
            metadata={"os_platform": env["os_platform"]},
        )

    def export_manifest_to_json(
        self, manifest: ExperimentManifest, output_path: Union[str, Path]
    ) -> None:
        """Exports ExperimentManifest object to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "experiment_id": manifest.experiment_id,
            "dataset_name": manifest.dataset_name,
            "git_commit": manifest.git_commit,
            "python_version": manifest.python_version,
            "pytorch_version": manifest.pytorch_version,
            "cuda_version": manifest.cuda_version,
            "gpu_name": manifest.gpu_name,
            "random_seed": manifest.random_seed,
            "segmentation_model": manifest.segmentation_model,
            "segmentation_version": manifest.segmentation_version,
            "detection_model": manifest.detection_model,
            "detection_version": manifest.detection_version,
            "scoring_version": manifest.scoring_version,
            "validation_version": manifest.validation_version,
            "execution_date": manifest.execution_date,
            "duration_s": manifest.duration_s,
            "metadata": manifest.metadata,
        }

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
