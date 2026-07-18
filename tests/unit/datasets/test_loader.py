"""
tests/unit/datasets/test_loader.py
==================================
Unit tests for the dataset loaders.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from torch.utils.data import DataLoader

from pathoai.datasets.loader import get_segmentation_dataloaders


class TestDataLoaderFactories:
    """Verifies dataloader construction, split configurations, and parameter passing."""

    def test_get_dataloaders_success(self):
        # Mock configuration node
        config = MagicMock()
        config.segmentation.batch_size = 4
        config.pipeline.num_workers = 0
        config.pipeline.pin_memory = False
        # Augmentations config
        aug = config.dataset.augmentations
        aug.flip_prob = 0.5
        aug.rotate_prob = 0.5
        aug.brightness_contrast_prob = 0.5
        aug.brightness_limit = 0.2
        aug.contrast_limit = 0.2
        aug.gaussian_noise_prob = 0.2

        train_entries = [{"slide_path": "s.tif", "x_level0": 0, "y_level0": 0, "patch_size": 64, "target_mpp": 0.5}]
        val_entries = [{"slide_path": "s.tif", "x_level0": 0, "y_level0": 0, "patch_size": 64, "target_mpp": 0.5}]
        test_entries = []

        train_loader, val_loader, test_loader = get_segmentation_dataloaders(
            train_entries,
            val_entries,
            test_entries,
            config,
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert test_loader is None

        # Verify batch size and worker counts
        assert train_loader.batch_size == 4
        assert train_loader.num_workers == 0
        # train loader must shuffle
        # In PyTorch, Dataloader handles shuffle by creating a RandomSampler
        assert train_loader.sampler.__class__.__name__ == "RandomSampler"
        # val loader does not shuffle
        assert val_loader.sampler.__class__.__name__ == "SequentialSampler"
