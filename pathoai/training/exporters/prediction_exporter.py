"""
pathoai/training/exporters/prediction_exporter.py
=================================================
Prediction Exporter.

Saves model prediction masks and ground truth maps as PNG, TIFF, or NumPy files.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 4.8
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class PredictionExporter:
    """Exports model prediction masks and coordinates to disk."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def export_mask(
        self,
        mask: Union[torch.Tensor, np.ndarray],
        filename: str,
        save_png: bool = True,
        save_tiff: bool = False,
        save_npy: bool = True,
    ) -> None:
        """Export a prediction/ground-truth mask to disk.

        Parameters
        ----------
        mask : torch.Tensor | np.ndarray
            Mask array (H, W).
        filename : str
            Base filename (e.g. 'prediction_001').
        save_png : bool
            Save as 8-bit grayscale PNG.
        save_tiff : bool
            Save as TIFF.
        save_npy : bool
            Save as raw NumPy array.
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()

        mask_np = mask.astype(np.uint8)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base_path = self.output_dir / filename

        # 1. Save PNG
        if save_png:
            png_path = base_path.with_suffix(".png")
            try:
                img = Image.fromarray(mask_np, mode="L")
                img.save(png_path)
                logger.debug("Exported mask PNG to %s", png_path)
            except Exception as exc:
                logger.error("Failed to export mask PNG: %s", exc)

        # 2. Save TIFF
        if save_tiff:
            tiff_path = base_path.with_suffix(".tif")
            try:
                img = Image.fromarray(mask_np, mode="L")
                img.save(tiff_path)
                logger.debug("Exported mask TIFF to %s", tiff_path)
            except Exception as exc:
                logger.error("Failed to export mask TIFF: %s", exc)

        # 3. Save NumPy
        if save_npy:
            npy_path = base_path.with_suffix(".npy")
            try:
                np.save(npy_path, mask_np)
                logger.debug("Exported mask NumPy to %s", npy_path)
            except Exception as exc:
                logger.error("Failed to export mask NumPy: %s", exc)
