"""
pathoai/config/experiment_config.py
===================================
Experiment Configuration & Provenance DTO.

Defines top-level unified experiment parameters and version tracking across engines.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.9 (Orchestration & Configuration)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ExperimentConfig:
    """Unified configuration for end-to-end computational pathology experiments.

    Attributes
    ----------
    experiment_id : str
        Unique experiment string identifier.
    wsi_mpp : float
        Microns per pixel slide resolution.
    segmentation_model : str
        Semantic segmentation model key (e.g. 'deeplabv3plus').
    segmentation_weights : str
        Path or key for segmentation model weights.
    segmentation_version : str
        Version string for segmentation weights (e.g. 'v1.2').
    detection_model : str
        Cell detection model key (e.g. 'yolo').
    detection_weights : str
        Path or key for detection model weights.
    detection_version : str
        Version string for detection weights (e.g. 'v0.9').
    fusion_grid_size : int
        Grid size in pixels for spatial indexing.
    fusion_max_distance_um : float
        Maximum cell-to-ROI association distance threshold in microns.
    scoring_algorithm : str
        Scoring algorithm key (e.g. 'tiger_working_group').
    scoring_version : str
        Scoring rules version (e.g. 'v1.0').
    lymphocyte_diameter_um : float
        Average physical lymphocyte diameter in microns (default 10.0 um).
    n_bootstrap_iterations : int
        Number of bootstrap iterations for 95% CI estimation.
    low_threshold : float
        Upper percentage limit for 'Low' sTIL category.
    high_threshold : float
        Lower percentage limit for 'High' sTIL category.
    metadata : Dict[str, Any]
        Additional key-value metadata parameters.
    """

    experiment_id: str = "exp_default"
    wsi_mpp: float = 0.5
    segmentation_model: str = "deeplabv3plus"
    segmentation_weights: str = "models/segmentation_v1.2.pth"
    segmentation_version: str = "v1.2"
    detection_model: str = "yolo"
    detection_weights: str = "models/yolo_v0.9.pt"
    detection_version: str = "v0.9"
    fusion_grid_size: int = 1024
    fusion_max_distance_um: float = 1000.0
    scoring_algorithm: str = "tiger_working_group"
    scoring_version: str = "v1.0"
    lymphocyte_diameter_um: float = 10.0
    n_bootstrap_iterations: int = 500
    low_threshold: float = 10.0
    high_threshold: float = 50.0
    metadata: Dict[str, Any] = field(default_factory=dict)
