"""
pathoai/wsi/patches/patches.py
==============================
Patch Engine for tissue-aware patch gridding and sampling.

Performs uniform sliding-window gridding of Whole Slide Images at target physical
micron resolution (MPP) and filters out background patches using downsampled
tissue masks.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from pathoai.core.exceptions import DataError
from pathoai.core.logger import get_logger
from pathoai.wsi.metadata.metadata import SlideMetadata
from pathoai.wsi.readers.base import BaseWSI

logger = get_logger(__name__)


@dataclass(frozen=True)
class PatchMetadata:
    """Dataclass holding location and scale parameters for a single slide patch.

    Downstream stages (normalization, model inference) consume a sequence of
    these metadata objects to load patches on-demand.

    Attributes:
        slide_path: Absolute Path to the parent slide file.
        x_level0: X coordinate of top-left corner in the level 0 reference frame.
        y_level0: Y coordinate of top-left corner in the level 0 reference frame.
        patch_size: Square patch size in pixels at target MPP.
        target_mpp: Target physical resolution (microns-per-pixel) of the patch.
        tissue_coverage: Fraction [0.0, 1.0] of patch area containing tissue.
    """
    slide_path: Path
    x_level0: int
    y_level0: int
    patch_size: int
    target_mpp: float
    tissue_coverage: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "slide_path": str(self.slide_path),
            "x_level0": self.x_level0,
            "y_level0": self.y_level0,
            "patch_size": self.patch_size,
            "target_mpp": self.target_mpp,
            "tissue_coverage": round(self.tissue_coverage, 4),
        }


class PatchSampler:
    """Grids Whole Slide Images and samples tissue patches using a tissue mask."""

    def __init__(
        self,
        patch_size: int = 512,
        stride: int = 256,
        target_mpp: float = 0.50,
        min_tissue_coverage: float = 0.25,
    ) -> None:
        """
        Parameters
        ----------
        patch_size : int
            Patch width and height in pixels (square patches).
        stride : int
            Grid step size in pixels at the target MPP.
        target_mpp : float
            Target microns-per-pixel.
        min_tissue_coverage : float
            Minimum fraction [0.0, 1.0] of tissue area required to keep a patch.
        """
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive. Got: {patch_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive. Got: {stride}")
        if target_mpp <= 0.0:
            raise ValueError(f"target_mpp must be positive. Got: {target_mpp}")
        if not (0.0 <= min_tissue_coverage <= 1.0):
            raise ValueError(f"min_tissue_coverage must be in [0.0, 1.0]. Got: {min_tissue_coverage}")

        self.patch_size = patch_size
        self.stride = stride
        self.target_mpp = target_mpp
        self.min_tissue_coverage = min_tissue_coverage

    def sample_patches(
        self,
        reader: BaseWSI,
        metadata: SlideMetadata,
        tissue_mask: np.ndarray,
    ) -> List[PatchMetadata]:
        """Sample tissue-containing patches from a slide.

        Grids the slide at target MPP, maps coordinates to the tissue mask,
        calculates tissue coverage, and returns metadata for valid patches.

        Parameters
        ----------
        reader : BaseWSI
            An open slide reader instance.
        metadata : SlideMetadata
            Standardized metadata for the slide.
        tissue_mask : np.ndarray
            Binary tissue mask (1 = tissue, 0 = background). Shape: (H_thumb, W_thumb).

        Returns
        -------
        List[PatchMetadata]
            List of metadata objects for patches meeting the tissue coverage criteria.

        Raises
        ------
        DataError
            If tissue mask shape is invalid.
        """
        if not reader.is_open:
            raise DataError(f"Cannot sample patches. Slide is closed: {reader.path}")
        if tissue_mask.ndim != 2:
            raise DataError(f"Expected 2-D tissue mask, got shape: {tissue_mask.shape}")

        slide_w_0, slide_h_0 = metadata.dimensions
        mask_h, mask_w = tissue_mask.shape

        if mask_w <= 0 or mask_h <= 0:
            raise DataError(f"Invalid tissue mask shape: {tissue_mask.shape}")

        # Compute scaling between level 0 and the downsampled tissue mask
        scale_x = slide_w_0 / mask_w
        scale_y = slide_h_0 / mask_h

        # Compute physical patch size and stride in level 0 pixels
        # patch_size_level0 = patch_size * (target_mpp / slide_mpp)
        mpp_ratio = self.target_mpp / metadata.mpp_x
        patch_size_0 = int(round(self.patch_size * mpp_ratio))
        stride_0 = int(round(self.stride * mpp_ratio))

        if patch_size_0 <= 0 or stride_0 <= 0:
            raise DataError("Calculated level 0 patch dimensions are non-positive.")

        logger.debug(
            "Gridding slide for patch sampling",
            extra={
                "patch_size_0": patch_size_0,
                "stride_0": stride_0,
                "target_mpp": self.target_mpp,
                "slide_mpp": metadata.mpp_x,
            },
        )

        patches_metadata: List[PatchMetadata] = []

        # Uniform gridding loop
        # We only generate patches fully contained inside the slide bounds
        x_coords = range(0, slide_w_0 - patch_size_0 + 1, stride_0)
        y_coords = range(0, slide_h_0 - patch_size_0 + 1, stride_0)

        for x in x_coords:
            for y in y_coords:
                # Map patch bounding box to the tissue mask coordinates
                mx_start = int(round(x / scale_x))
                my_start = int(round(y / scale_y))
                mx_end = int(round((x + patch_size_0) / scale_x))
                my_end = int(round((y + patch_size_0) / scale_y))

                # Clip mapped coordinates to mask boundaries
                mx_start = max(0, min(mx_start, mask_w))
                my_start = max(0, min(my_start, mask_h))
                mx_end = max(0, min(mx_end, mask_w))
                my_end = max(0, min(my_end, mask_h))

                # Extract the mask subregion
                sub_mask = tissue_mask[my_start:my_end, mx_start:mx_end]

                # Compute tissue coverage
                if sub_mask.size > 0:
                    coverage = float(np.mean(sub_mask == 1))
                else:
                    coverage = 0.0

                # Keep patch if it exceeds the minimum coverage threshold
                if coverage >= self.min_tissue_coverage:
                    patches_metadata.append(
                        PatchMetadata(
                            slide_path=metadata.path,
                            x_level0=x,
                            y_level0=y,
                            patch_size=self.patch_size,
                            target_mpp=self.target_mpp,
                            tissue_coverage=coverage,
                        )
                    )

        logger.info(
            "Completed patch sampling",
            extra={
                "n_patches_sampled": len(patches_metadata),
                "grid_size": f"{len(x_coords)}x{len(y_coords)}",
                "min_coverage": self.min_tissue_coverage,
            },
        )

        return patches_metadata
