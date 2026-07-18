"""
pathoai/datasets/manifest.py
============================
Manifest Generator for the PathoAI Dataset Engine.

Aggregates slide-mask pairs, runs uniform patch gridding, extracts WSI metadata,
computes patch-level class pixel distributions, and saves a dataset manifest file.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from pathoai.core.exceptions import DataError, DatasetValidationError
from pathoai.core.logger import get_logger
from pathoai.wsi.metadata.metadata import extract_metadata
from pathoai.wsi.patches.patches import PatchSampler
from pathoai.wsi.pyramid.pyramid import get_slide_thumbnail
from pathoai.wsi.readers.factory import get_wsi_reader
from pathoai.wsi.tissue.tissue import TissueDetector

logger = get_logger(__name__)


def generate_dataset_manifest(
    dataset_root: str | Path,
    patch_size: int = 512,
    stride: int = 256,
    target_mpp: float = 0.50,
    min_tissue_coverage: float = 0.25,
    slide_subdir: str = "images",
    mask_subdir: str = "masks",
    mask_extension: str = ".png",
    output_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Generate a dataset manifest mapping patch coordinates to slide/mask sources.

    Finds WSI files and matching annotation masks, runs tissue detection,
    grids tissue regions, and profiles the pixel-level class distribution
    for each patch.

    Parameters
    ----------
    dataset_root : str | Path
        Root directory of the dataset.
    patch_size : int
        Square patch size in pixels at target MPP.
    stride : int
        Grid stride in pixels at target MPP.
    target_mpp : float
        Target microns-per-pixel.
    min_tissue_coverage : float
        Minimum tissue area fraction to keep a patch.
    slide_subdir : str
        Subdirectory containing WSI files.
    mask_subdir : str
        Subdirectory containing PNG mask files.
    mask_extension : str
        Extension of mask files.
    output_path : str | Path, optional
        Path to save the generated manifest JSON file.

    Returns
    -------
    List[Dict[str, Any]]
        List of manifest dictionary entries.
    """
    root = Path(dataset_root).resolve()
    slides_dir = root / slide_subdir
    masks_dir = root / mask_subdir

    if not slides_dir.is_dir():
        raise DataError(f"Slides directory not found: {slides_dir}")

    # Discover all WSI files matching supported formats
    from pathoai.core.constants import SUPPORTED_WSI_FORMATS
    slide_paths: List[Path] = []
    for ext in SUPPORTED_WSI_FORMATS:
        slide_paths.extend(slides_dir.glob(f"**/*{ext}"))
    slide_paths = sorted(set(slide_paths))

    if not slide_paths:
        raise DataError(f"No whole slide images found in {slides_dir}")

    logger.info("Generating manifest", extra={"n_slides": len(slide_paths), "root": str(root)})

    # Initialize pre-requisites
    tissue_detector = TissueDetector(
        morph_kernel_size=15,
        min_component_pixels=1000,
    )
    sampler = PatchSampler(
        patch_size=patch_size,
        stride=stride,
        target_mpp=target_mpp,
        min_tissue_coverage=min_tissue_coverage,
    )

    manifest_entries: List[Dict[str, Any]] = []

    for slide_path in slide_paths:
        slide_stem = slide_path.stem
        mask_path = masks_dir / f"{slide_stem}{mask_extension}"

        # If mask is missing, log warning (can still generate entries for evaluation / infer)
        has_mask = mask_path.is_file()
        if not has_mask:
            logger.warning("Mask file not found for WSI: %s. Proceeding without mask.", slide_path)
            mask_path = None

        logger.debug("Profiling slide: %s", slide_stem)

        # Open WSI reader and extract metadata
        try:
            reader = get_wsi_reader(slide_path)
            with reader:
                metadata = extract_metadata(reader)

                # 1. Generate thumbnail and detect tissue regions
                thumb = get_slide_thumbnail(reader, metadata, max_dim=2048)
                tissue_mask, _ = tissue_detector.detect_tissue(thumb)

                # 2. Sample patch grid coordinates
                patches = sampler.sample_patches(reader, metadata, tissue_mask)

                if not patches:
                    logger.warning("No tissue patches sampled from slide: %s", slide_stem)
                    continue

                # 3. Load mask for class distribution computation
                mask_img: Optional[Image.Image] = None
                scale_x, scale_y = 1.0, 1.0
                if has_mask:
                    try:
                        mask_img = Image.open(mask_path)
                        mask_w, mask_h = mask_img.size
                        scale_x = metadata.dimensions[0] / mask_w
                        scale_y = metadata.dimensions[1] / mask_h
                    except Exception as exc:
                        raise DatasetValidationError(
                            f"Failed to open/parse mask image {mask_path}: {exc}"
                        ) from exc

                # 4. Map each patch and compile manifest entries
                mpp_ratio = target_mpp / metadata.mpp_x
                patch_size_0 = int(round(patch_size * mpp_ratio))

                for patch in patches:
                    dist: Dict[str, int] = {}

                    # If mask is available, compute the class distribution
                    if mask_img is not None:
                        # Map level 0 patch bounds to mask coordinate space
                        mx1 = int(round(patch.x_level0 / scale_x))
                        my1 = int(round(patch.y_level0 / scale_y))
                        mx2 = int(round((patch.x_level0 + patch_size_0) / scale_x))
                        my2 = int(round((patch.y_level0 + patch_size_0) / scale_y))

                        # Safe clipping to mask boundaries
                        mx1 = max(0, min(mx1, mask_img.size[0]))
                        my1 = max(0, min(my1, mask_img.size[1]))
                        mx2 = max(0, min(mx2, mask_img.size[0]))
                        my2 = max(0, min(my2, mask_img.size[1]))

                        # Extract the region and build class counts
                        try:
                            crop_box = mask_img.crop((mx1, my1, mx2, my2))
                            mask_arr = np.array(crop_box, dtype=np.uint8)
                            unique, counts = np.unique(mask_arr, return_counts=True)
                            dist = {str(c): int(count) for c, count in zip(unique, counts)}
                        except Exception as exc:
                            logger.warning(
                                "Failed to extract mask region for patch at (%d, %d): %s",
                                patch.x_level0,
                                patch.y_level0,
                                exc,
                            )

                    manifest_entries.append({
                        "slide_path": str(slide_path.resolve()),
                        "mask_path": str(mask_path.resolve()) if mask_path else None,
                        "x_level0": patch.x_level0,
                        "y_level0": patch.y_level0,
                        "patch_size": patch.patch_size,
                        "target_mpp": patch.target_mpp,
                        "tissue_coverage": patch.tissue_coverage,
                        "class_distribution": dist,
                    })

        except Exception as exc:
            # We catch exceptions to prevent failing the entire manifest generation
            # but raise critical ones if needed
            logger.error("Failed to generate manifest entries for slide %s: %s", slide_path, exc)
            raise DatasetValidationError(f"Manifest generation failed on slide {slide_path}: {exc}") from exc

    # Save manifest if output_path is provided
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest_entries, f, indent=2)
        logger.info("Saved dataset manifest", extra={"path": str(out), "n_entries": len(manifest_entries)})

    return manifest_entries
