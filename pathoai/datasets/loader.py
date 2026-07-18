"""
pathoai/datasets/loader.py
==========================
Data loader factories and PyTorch DataModule for semantic segmentation.

Creates train, validation, and test PyTorch DataLoader instances configured with
augmentations, batch sizes, worker counts, and memory pinning.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from pathoai.core.logger import get_logger
from pathoai.datasets.dataset import SegmentationDataset
from pathoai.datasets.transforms import get_transforms

logger = get_logger(__name__)


def get_segmentation_dataloaders(
    train_entries: List[Dict[str, Any]],
    val_entries: List[Dict[str, Any]],
    test_entries: List[Dict[str, Any]],
    config: Any,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Build train, validation, and test PyTorch DataLoaders from manifest entries.

    Parameters
    ----------
    train_entries : List[Dict[str, Any]]
        Manifest entries for the training split.
    val_entries : List[Dict[str, Any]]
        Manifest entries for the validation split.
    test_entries : List[Dict[str, Any]]
        Manifest entries for the test split.
    config : Any
        Pipeline configuration (ConfigNode instance).

    Returns
    -------
    Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]
        (train_loader, val_loader, test_loader). Individual loaders are None
        if their corresponding splits are empty.
    """
    # 1. Extract data loader settings from config
    batch_size = config.segmentation.batch_size
    num_workers = config.pipeline.num_workers
    pin_memory = config.pipeline.pin_memory

    logger.info(
        "Creating dataloaders",
        extra={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
    )

    # 2. Extract augmentation parameters from config
    aug_cfg = config.dataset.augmentations
    train_transforms = get_transforms(
        split="train",
        flip_prob=aug_cfg.flip_prob,
        rotate_prob=aug_cfg.rotate_prob,
        brightness_contrast_prob=aug_cfg.brightness_contrast_prob,
        brightness_limit=aug_cfg.brightness_limit,
        contrast_limit=aug_cfg.contrast_limit,
        gaussian_noise_prob=aug_cfg.gaussian_noise_prob,
    )
    val_transforms = get_transforms(split="val")

    # Common DataLoader params
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    # 3. Create datasets and loaders
    train_loader = None
    if train_entries:
        train_ds = SegmentationDataset(train_entries, transforms=train_transforms)
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **loader_kwargs,
        )

    val_loader = None
    if val_entries:
        val_ds = SegmentationDataset(val_entries, transforms=val_transforms)
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            **loader_kwargs,
        )

    test_loader = None
    if test_entries:
        test_ds = SegmentationDataset(test_entries, transforms=val_transforms)
        test_loader = DataLoader(
            test_ds,
            shuffle=False,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader
