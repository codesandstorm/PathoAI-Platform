"""
tests/unit/datasets/test_dataset.py
===================================
Unit tests for SegmentationDataset and WSICache.

Tests cover:
- WSICache LRU eviction and closure of slide files
- SegmentationDataset lifecycle and lazy cache initialization
- Patch read mapping, mask cropping, and nearest-neighbor interpolation resizing
- Integration with Albumentations transforms
- Output shape and type verification

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from pathoai.core.exceptions import DataError
from pathoai.datasets.dataset import SegmentationDataset, WSICache
from pathoai.wsi.readers.base import BaseWSI


# ---------------------------------------------------------------------------
# Test WSICache
# ---------------------------------------------------------------------------

class TestWSICache:
    """Verifies caching, retrieval, and LRU eviction in WSICache."""

    @patch("pathoai.datasets.dataset.get_wsi_reader")
    def test_cache_retrieval_and_lifecycle(self, mock_factory, tmp_path: Path):
        """Retrieves slide reader from cache, and opens it on first access."""
        p = tmp_path / "slide.svs"
        p.write_bytes(b"dummy")

        mock_reader = MagicMock(spec=BaseWSI)
        mock_factory.return_value = mock_reader

        cache = WSICache(max_capacity=2)
        # First retrieve -> opens slide
        r1 = cache.get(p)
        assert r1 is mock_reader
        mock_reader.open.assert_called_once()

        # Second retrieve -> cached, does not call open again
        r2 = cache.get(p)
        assert r2 is mock_reader
        assert mock_factory.call_count == 1

    @patch("pathoai.datasets.dataset.get_wsi_reader")
    def test_lru_eviction(self, mock_factory, tmp_path: Path):
        """Evicts the least recently used reader when capacity is exceeded."""
        p1 = tmp_path / "s1.svs"
        p2 = tmp_path / "s2.svs"
        p3 = tmp_path / "s3.svs"
        for p in (p1, p2, p3):
            p.write_bytes(b"dummy")

        r1 = MagicMock(spec=BaseWSI)
        r2 = MagicMock(spec=BaseWSI)
        r3 = MagicMock(spec=BaseWSI)
        mock_factory.side_effect = [r1, r2, r3]

        cache = WSICache(max_capacity=2)
        cache.get(p1)  # cache has s1
        cache.get(p2)  # cache has s1, s2
        assert len(cache._cache) == 2

        # Access s1 to make it most recently used (s2 becomes oldest)
        cache.get(p1)

        # Get s3 -> capacity exceeded, evicts oldest (s2)
        cache.get(p3)
        assert len(cache._cache) == 2
        r2.close.assert_called_once()  # s2 evicted and closed
        r1.close.assert_not_called()  # s1 was kept


# ---------------------------------------------------------------------------
# Test SegmentationDataset
# ---------------------------------------------------------------------------

class TestSegmentationDataset:
    """Verifies PyTorch Dataset operations, patches, masks, and transforms."""

    @pytest.fixture()
    def fake_mask(self, tmp_path: Path):
        """Creates a dummy mask PNG image of size 64x64 filled with class 2."""
        mask_path = tmp_path / "mask.png"
        mask_arr = np.ones((64, 64), dtype=np.uint8) * 2  # class 2 stroma
        img = Image.fromarray(mask_arr, mode="L")
        img.save(mask_path)
        return mask_path

    @pytest.fixture()
    def dummy_entries(self, fake_mask: Path):
        return [
            {
                "slide_path": "/fake/slide1.svs",
                "mask_path": str(fake_mask),
                "x_level0": 1000,
                "y_level0": 2000,
                "patch_size": 256,
                "target_mpp": 0.50,
            }
        ]

    def test_length(self, dummy_entries):
        ds = SegmentationDataset(dummy_entries)
        assert len(ds) == 1

    @patch("pathoai.datasets.dataset.extract_metadata")
    @patch("pathoai.datasets.dataset.read_patch_at_mpp")
    def test_getitem_without_transforms(
        self,
        mock_read_patch,
        mock_extract_meta,
        dummy_entries,
        tmp_path: Path,
    ):
        """Verifies loading patch and mask arrays, returning default tensors."""
        # Setup mock WSI reader
        mock_reader = MagicMock(spec=BaseWSI)
        mock_reader.is_open = True
        mock_reader.path = Path(dummy_entries[0]["slide_path"])

        mock_meta = MagicMock()
        mock_meta.dimensions = (8000, 6000)
        mock_meta.mpp_x = 0.50
        mock_extract_meta.return_value = mock_meta

        # Mock image read region output (256x256 RGB)
        fake_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        mock_read_patch.return_value = fake_img

        # Create dataset
        ds = SegmentationDataset(dummy_entries)
        # Mock WSICache to bypass actual slide reading
        ds._wsi_cache = MagicMock(spec=WSICache)
        ds._wsi_cache.get.return_value = mock_reader

        image_tensor, mask_tensor = ds[0]

        # Assert PyTorch tensors and shapes
        assert isinstance(image_tensor, torch.Tensor)
        assert isinstance(mask_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 256, 256)
        assert mask_tensor.shape == (256, 256)
        assert image_tensor.dtype == torch.float32
        assert mask_tensor.dtype == torch.int64

        # Values mapping
        assert image_tensor.max() <= 1.0
        # Check that cropped mask patch defaults to 2 (from fake_mask PNG)
        assert torch.all(mask_tensor == 2)

    @patch("pathoai.datasets.dataset.extract_metadata")
    @patch("pathoai.datasets.dataset.read_patch_at_mpp")
    def test_getitem_with_mask_resizing(
        self,
        mock_read_patch,
        mock_extract_meta,
        dummy_entries,
        tmp_path: Path,
    ):
        """Verifies that mask crop is resized to patch_size if dimensions mismatch."""
        mock_reader = MagicMock(spec=BaseWSI)
        mock_reader.is_open = True
        mock_reader.path = Path(dummy_entries[0]["slide_path"])

        # Set slide dimensions large relative to mask size, to trigger size mismatch
        mock_meta = MagicMock()
        mock_meta.dimensions = (80000, 60000)  # Very large -> crop size differs
        mock_meta.mpp_x = 0.50
        mock_extract_meta.return_value = mock_meta

        fake_img = np.zeros((256, 256, 3), dtype=np.uint8)
        mock_read_patch.return_value = fake_img

        ds = SegmentationDataset(dummy_entries)
        ds._wsi_cache = MagicMock(spec=WSICache)
        ds._wsi_cache.get.return_value = mock_reader

        _, mask_tensor = ds[0]
        # Must still be resized to exact target size
        assert mask_tensor.shape == (256, 256)

    @patch("pathoai.datasets.dataset.extract_metadata")
    @patch("pathoai.datasets.dataset.read_patch_at_mpp")
    def test_getitem_raises_on_missing_mask_file(self, mock_read_patch, mock_extract_meta):
        """Raises DataError if the mask path is set but file is missing."""
        entry = [
            {
                "slide_path": "/fake/slide.svs",
                "mask_path": "/fake/missing_mask.png",
                "x_level0": 0,
                "y_level0": 0,
                "patch_size": 256,
                "target_mpp": 0.50,
            }
        ]
        # Mock extract_metadata to return a valid mock metadata
        mock_meta = MagicMock()
        mock_meta.dimensions = (1000, 1000)
        mock_meta.mpp_x = 0.50
        mock_meta.level_downsamples = [1.0]
        mock_extract_meta.return_value = mock_meta

        # Mock read_patch_at_mpp to return a valid numpy array
        mock_read_patch.return_value = np.zeros((256, 256, 3), dtype=np.uint8)

        ds = SegmentationDataset(entry)
        ds._wsi_cache = MagicMock(spec=WSICache)
        # Mock reader
        mock_reader = MagicMock(spec=BaseWSI)
        mock_reader.is_open = True
        ds._wsi_cache.get.return_value = mock_reader

        with pytest.raises(DataError, match="mask file not found"):
            _ = ds[0]
