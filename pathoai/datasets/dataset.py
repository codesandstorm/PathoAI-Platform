"""
pathoai/datasets/dataset.py
===========================
PyTorch Segmentation Dataset and Slide Reader Cache.

Implements the SegmentationDataset class for on-demand patch extraction,
applying Albumentations augmentations, and returning PyTorch tensors.
Includes an LRU slide reader cache to prevent file descriptor exhaustion
across PyTorch dataloader worker processes.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pathoai.core.exceptions import DataError
from pathoai.core.logger import get_logger
from pathoai.wsi.metadata.metadata import extract_metadata
from pathoai.wsi.pyramid.pyramid import read_patch_at_mpp
from pathoai.wsi.readers.base import BaseWSI
from pathoai.wsi.readers.factory import get_wsi_reader

logger = get_logger(__name__)


class WSICache:
    """Least Recently Used (LRU) cache for BaseWSI readers.

    Ensures that we do not exceed OS open file descriptor limits in multi-threaded
    or multi-worker dataloaders by automatically closing slide files when capacity is exceeded.
    """

    def __init__(self, max_capacity: int = 8) -> None:
        """
        Parameters
        ----------
        max_capacity : int
            Maximum number of open slide readers to keep in cache.
        """
        self.max_capacity = max_capacity
        self._cache: OrderedDict[str, BaseWSI] = OrderedDict()

    def get(self, path: Path | str) -> BaseWSI:
        """Retrieve the reader for the given path, opening it if not cached.

        Moves the retrieved slide to the most-recently-used position.
        """
        p_str = str(Path(path).resolve())

        if p_str in self._cache:
            self._cache.move_to_end(p_str)
            return self._cache[p_str]

        # Check capacity and evict oldest if needed
        if len(self._cache) >= self.max_capacity:
            oldest_path, oldest_reader = self._cache.popitem(last=False)
            logger.debug("WSICache: evicting open slide: %s", oldest_path)
            try:
                oldest_reader.close()
            except Exception as exc:
                logger.warning("WSICache: error closing evicted slide %s: %s", oldest_path, exc)

        # Load and cache new reader
        logger.debug("WSICache: opening and caching slide: %s", p_str)
        reader = get_wsi_reader(p_str)
        reader.open()
        self._cache[p_str] = reader
        return reader

    def clear(self) -> None:
        """Close all cached slide readers and empty the cache."""
        while self._cache:
            path, reader = self._cache.popitem(last=False)
            try:
                reader.close()
            except Exception as exc:
                logger.warning("WSICache: error closing slide during clear %s: %s", path, exc)


class SegmentationDataset(Dataset):
    """PyTorch Dataset for segmentation patch training and validation.

    Loads image patches and annotations mask crops on-demand from slide and mask files.
    """

    def __init__(
        self,
        manifest_entries: List[Dict[str, Any]],
        transforms: Optional[Any] = None,
        max_cache_capacity: int = 8,
    ) -> None:
        """
        Parameters
        ----------
        manifest_entries : List[Dict[str, Any]]
            List of patch manifest entry dictionaries.
        transforms : albumentations.Compose, optional
            Transforms to apply to the image and mask.
        max_cache_capacity : int
            Maximum size of the LRU slide reader cache.
        """
        self.entries = manifest_entries
        self.transforms = transforms
        self.max_cache_capacity = max_cache_capacity

        # Lazy initialize the cache inside __getitem__ to prevent open file descriptor
        # leakage during process spawning/forking in PyTorch Multi-worker DataLoader.
        self._wsi_cache: Optional[WSICache] = None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return the patch image and mask at the specified index.

        Parameters
        ----------
        index : int
            Index of the patch.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Normalized image tensor of shape (3, H, W), float32.
            - Mask tensor of shape (H, W), long (int64).
        """
        # Lazy initialize the WSI cache per worker process
        if self._wsi_cache is None:
            self._wsi_cache = WSICache(max_capacity=self.max_cache_capacity)

        entry = self.entries[index]
        slide_path_str = entry["slide_path"]
        mask_path_str = entry.get("mask_path")

        x = entry["x_level0"]
        y = entry["y_level0"]
        patch_size = entry["patch_size"]
        target_mpp = entry["target_mpp"]

        try:
            # 1. Fetch reader and extract metadata
            reader = self._wsi_cache.get(slide_path_str)
            metadata = extract_metadata(reader)

            # 2. Extract WSI RGB patch
            image_arr = read_patch_at_mpp(
                reader=reader,
                metadata=metadata,
                location_level0=(x, y),
                target_mpp=target_mpp,
                patch_size=(patch_size, patch_size),
            )

            # 3. Extract mask patch
            if mask_path_str is not None:
                mask_path = Path(mask_path_str)
                if not mask_path.is_file():
                    raise DataError(f"Dataset mask file not found: {mask_path}")

                with Image.open(mask_path) as mask_img:
                    mask_w, mask_h = mask_img.size

                    # Calculate downsample scale factor between WSI level 0 and mask dimensions
                    scale_x = metadata.dimensions[0] / mask_w
                    scale_y = metadata.dimensions[1] / mask_h

                    # Scale level 0 bounds to mask coords
                    mpp_ratio = target_mpp / metadata.mpp_x
                    patch_size_0 = int(round(patch_size * mpp_ratio))

                    mx1 = int(round(x / scale_x))
                    my1 = int(round(y / scale_y))
                    mx2 = int(round((x + patch_size_0) / scale_x))
                    my2 = int(round((y + patch_size_0) / scale_y))

                    # Clip to mask bounds
                    mx1 = max(0, min(mx1, mask_w))
                    my1 = max(0, min(my1, mask_h))
                    mx2 = max(0, min(mx2, mask_w))
                    my2 = max(0, min(my2, mask_h))

                    # Crop mask patch
                    cropped_mask = mask_img.crop((mx1, my1, mx2, my2))

                    # Nearest-neighbor resize mask if crop shape doesn't match patch_size
                    if cropped_mask.size != (patch_size, patch_size):
                        cropped_mask = cropped_mask.resize(
                            (patch_size, patch_size),
                            resample=Image.Resampling.NEAREST,
                        )
                    mask_arr = np.array(cropped_mask, dtype=np.uint8)
            else:
                # Return empty/background mask if no file is configured
                mask_arr = np.zeros((patch_size, patch_size), dtype=np.uint8)

        except Exception as exc:
            logger.error(
                "Failed loading dataset patch at index %d [slide=%s, coords=(%d, %d)]: %s",
                index,
                slide_path_str,
                x,
                y,
                exc,
            )
            raise DataError(
                f"Error loading patch at index {index} from slide {slide_path_str}: {exc}"
            ) from exc

        # 4. Apply transforms & augmentations
        if self.transforms is not None:
            # Albumentations expects 'image' and 'mask' keyword args
            augmented = self.transforms(image=image_arr, mask=mask_arr)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"].long()
        else:
            # Default fallback: manual tensor conversion
            image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask_arr).long()

        return image_tensor, mask_tensor
