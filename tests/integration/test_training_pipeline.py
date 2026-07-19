"""
tests/integration/test_training_pipeline.py
===========================================
Integration tests for the complete training orchestrator execution.

Runs a full end-to-end dry-run training pipeline on a small synthetic dataset:
- Instantiates configuration
- Loads splits and dataloaders
- Instantiates model architectures and losses
- Runs preflight verification and fit cycles
- Generates all artifact files in directories

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 5.10
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from pathoai.core.config import ConfigManager
from pathoai.training.run import run_experiment
from pathoai.wsi.metadata.metadata import SlideMetadata


@pytest.fixture()
def mock_dataset_on_disk(tmp_path: Path):
    """Set up a small fake slide dataset structure with splits and mask files."""
    data_dir = tmp_path / "raw_tiger"
    data_dir.mkdir()

    # Create dummy mask PNGs (256x256)
    masks_dir = data_dir / "masks"
    masks_dir.mkdir()
    
    from PIL import Image
    for pat in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        mask_path = masks_dir / f"slide_{pat}.png"
        mask_data = np.zeros((256, 256), dtype=np.uint8)
        mask_data[20:80, 20:80] = 1  # class 1
        Image.fromarray(mask_data, mode="L").save(mask_path)

    # Create the splits.json file to bypass manifest generation
    splits_dict = {
        "train": [
            {
                "slide_path": str(data_dir / "slide_A1.svs"),
                "mask_path": str(masks_dir / "slide_A1.png"),
                "x_level0": 0,
                "y_level0": 0,
                "patch_size": 256,
                "target_mpp": 0.5,
                "patient_id": "patientA",
            },
            {
                "slide_path": str(data_dir / "slide_A2.svs"),
                "mask_path": str(masks_dir / "slide_A2.png"),
                "x_level0": 0,
                "y_level0": 0,
                "patch_size": 256,
                "target_mpp": 0.5,
                "patient_id": "patientA",
            }
        ],
        "val": [
            {
                "slide_path": str(data_dir / "slide_B1.svs"),
                "mask_path": str(masks_dir / "slide_B1.png"),
                "x_level0": 0,
                "y_level0": 0,
                "patch_size": 256,
                "target_mpp": 0.5,
                "patient_id": "patientB",
            },
            {
                "slide_path": str(data_dir / "slide_B2.svs"),
                "mask_path": str(masks_dir / "slide_B2.png"),
                "x_level0": 0,
                "y_level0": 0,
                "patch_size": 256,
                "target_mpp": 0.5,
                "patient_id": "patientB",
            }
        ],
        "test": [
            {
                "slide_path": str(data_dir / "slide_C1.svs"),
                "mask_path": str(masks_dir / "slide_C1.png"),
                "x_level0": 0,
                "y_level0": 0,
                "patch_size": 256,
                "target_mpp": 0.5,
                "patient_id": "patientC",
            },
            {
                "slide_path": str(data_dir / "slide_C2.svs"),
                "mask_path": str(masks_dir / "slide_C2.png"),
                "x_level0": 0,
                "y_level0": 0,
                "patch_size": 256,
                "target_mpp": 0.5,
                "patient_id": "patientC",
            }
        ],
    }

    splits_file = tmp_path / "splits.json"
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=4)

    return data_dir, splits_file


@pytest.fixture()
def config_file_path(tmp_path: Path, mock_dataset_on_disk):
    """Generate a custom pipeline configuration YAML file with small values."""
    data_dir, splits_file = mock_dataset_on_disk

    cfg_dict = {
        "pipeline": {
            "name": "integration_test_pipeline",
            "version": "0.1.0",
            "seed": 42,
            "device": "cpu",
            "mixed_precision": False,
            "resume": False,
            "save_intermediates": True,
            "num_workers": 0,  # avoid multiprocessing overhead in test
            "pin_memory": False,
        },
        "wsi": {
            "supported_formats": [".svs"],
            "thumbnail": {"max_dim": 256},
            "tissue_detection": {
                "method": "otsu_hsv",
                "min_tissue_ratio": 0.01,
                "morph_kernel_size": 3,
                "min_component_pixels": 10,
            },
            "patch_extraction": {
                "patch_size": 256,
                "stride": 256,
                "target_mpp": 0.5,
                "blank_threshold": 250,
                "blur_threshold": 0.0,
                "min_tissue_coverage": 0.0,
            },
        },
        "stain_normalization": {
            "method": "none",
        },
        "segmentation": {
            "model_name": "deeplabv3plus_resnet18",  # ResNet18 encoder
            "n_classes": 2,
            "input_size": 256,
            "batch_size": 2,
            "checkpoint": None,
            "training": {
                "epochs": 1,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer_name": "adamw",
                "lr_scheduler": "cosine",
                "early_stopping_monitor": "val_loss",
                "early_stopping_patience": 2,
                "accumulate_grad_batches": 1,
            },
        },
        "logging": {
            "level": "INFO",
            "console": False,
        },
        "output": {
            "base_dir": str(tmp_path / "outputs"),
            "log_dir": "logs",
            "model_dir": "models",
            "save_segmentation_overlay": True,
        },
        "data": {
            "tiger": {
                "root_dir": str(data_dir),
                "splits_file": str(splits_file),
                "train_dir": str(data_dir),
                "val_dir": str(data_dir),
                "test_dir": str(data_dir),
            },
        },
        "dataset": {
            "train_ratio": 0.50,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "split_seed": 42,
            "augmentations": {
                "flip_prob": 0.0,
                "rotate_prob": 0.0,
                "brightness_contrast_prob": 0.0,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "gaussian_noise_prob": 0.0,
            },
        },
    }

    cfg_file = tmp_path / "test_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

    return cfg_file


@patch("pathoai.datasets.dataset.WSICache")
@patch("pathoai.datasets.dataset.extract_metadata")
@patch("pathoai.datasets.dataset.read_patch_at_mpp")
def test_end_to_end_training_pipeline_execution(
    mock_read_patch,
    mock_extract_metadata,
    mock_wsi_cache_class,
    config_file_path: Path,
    tmp_path: Path,
):
    """Executes run_experiment and asserts all directory artifacts are created."""
    # 1. Setup WSICache mock to return a dummy reader
    mock_cache_instance = MagicMock()
    mock_wsi_cache_class.return_value = mock_cache_instance
    mock_cache_instance.get.return_value = MagicMock()

    # 2. Setup SlideMetadata mock properties
    metadata = SlideMetadata(
        path=Path(config_file_path).parent / "raw_tiger" / "slide_A.svs",
        vendor="Aperio",
        dimensions=(256, 256),
        level_count=1,
        level_dimensions=[(256, 256)],
        level_downsamples=[1.0],
        mpp_x=0.5,
        mpp_y=0.5,
        magnification=20.0,
        associated_images=[],
        properties={},
    )
    mock_extract_metadata.return_value = metadata

    # 3. Setup read_patch_at_mpp mock output (256x256x3 RGB patch)
    dummy_patch = np.zeros((256, 256, 3), dtype=np.uint8)
    mock_read_patch.return_value = dummy_patch

    # Run the experiment orchestrator
    run_experiment(str(config_file_path))

    # Assert base directory and timestamped experiment folder exists
    outputs_base = tmp_path / "outputs"
    assert outputs_base.is_dir()

    # Find the timestamped folder
    exp_folders = list(outputs_base.glob("integration_test_pipeline_*"))
    assert len(exp_folders) == 1
    exp_dir = exp_folders[0]

    # Verify all MIC-mandated artifact directories are populated
    assert (exp_dir / "checkpoints").is_dir()
    assert (exp_dir / "history").is_dir()
    assert (exp_dir / "metrics").is_dir()
    assert (exp_dir / "reports").is_dir()
    assert (exp_dir / "logs").is_dir()
    assert (exp_dir / "exports").is_dir()

    # Verify files created
    assert (exp_dir / "config.yaml").is_file()
    assert (exp_dir / "Model_Summary.md").is_file()
    assert (exp_dir / "history" / "history.csv").is_file()
    assert (exp_dir / "metrics" / "metrics.json").is_file()
    assert (exp_dir / "experiment_summary.md").is_file()

    # Verify model was compiled and exported
    assert (exp_dir / "exports" / "model.ts").is_file()
