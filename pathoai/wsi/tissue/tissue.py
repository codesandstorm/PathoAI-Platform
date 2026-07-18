"""
pathoai/wsi/tissue/tissue.py
============================
Tissue detection engine using classical computer vision.

Provides automated tissue detection and masking (background removal) on Whole
Slide Image thumbnails. Implements Otsu thresholding in HSV color space
followed by morphological cleanup and small component removal.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 2
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from pathoai.core.exceptions import DataError
from pathoai.core.logger import get_logger

logger = get_logger(__name__)


class TissueDetector:
    """Tissue detector executing classical image segmentation on slide thumbnails."""

    def __init__(
        self,
        method: str = "otsu_hsv",
        morph_kernel_size: int = 15,
        min_component_pixels: int = 1000,
    ) -> None:
        """
        Parameters
        ----------
        method : str
            Thresholding method. Only 'otsu_hsv' is currently supported.
        morph_kernel_size : int
            Kernel size in pixels for morphological closing and opening.
        min_component_pixels : int
            Minimum pixel count for a connected tissue component to be kept.
        """
        if method != "otsu_hsv":
            raise ValueError(f"Unsupported tissue detection method: {method}")

        self.method = method
        self.morph_kernel_size = morph_kernel_size
        self.min_component_pixels = min_component_pixels

    def detect_tissue(self, thumbnail: np.ndarray) -> Tuple[np.ndarray, float]:
        """Generate a binary tissue mask from a slide thumbnail.

        Parameters
        ----------
        thumbnail : np.ndarray
            RGB thumbnail image. Shape: (H, W, 3), dtype uint8.

        Returns
        -------
        Tuple[np.ndarray, float]
            - Binary tissue mask. Shape: (H, W), dtype uint8 (1 = tissue, 0 = background).
            - Fraction of the thumbnail area covered by tissue (float between 0.0 and 1.0).

        Raises
        ------
        DataError
            If the input image is empty or does not match RGB shape expectations.
        """
        if thumbnail.size == 0:
            raise DataError("Thumbnail image is empty.")
        if thumbnail.ndim != 3 or thumbnail.shape[2] != 3:
            raise DataError(f"Expected RGB thumbnail of shape (H, W, 3), got {thumbnail.shape}")

        logger.debug(
            "Detecting tissue on thumbnail",
            extra={
                "shape": thumbnail.shape,
                "method": self.method,
                "morph_kernel": self.morph_kernel_size,
            },
        )

        # 1. Convert to HSV and extract the Saturation channel
        hsv = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]

        # 2. Otsu thresholding on Saturation channel
        # Otsu computes threshold automatically based on bimodal histogram
        _, binary = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Morphological closing (fills holes) and opening (removes dust)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 4. Remove small connected components (ink, noise, artifacts)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        # Create final mask (0 = background, 1 = tissue)
        mask = np.zeros_like(binary, dtype=np.uint8)

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= self.min_component_pixels:
                mask[labels == label] = 1

        # 5. Compute coverage fraction
        total_pixels = mask.size
        tissue_pixels = np.sum(mask == 1)
        coverage_ratio = float(tissue_pixels / total_pixels) if total_pixels > 0 else 0.0

        logger.info(
            "Tissue detection completed",
            extra={
                "tissue_ratio": round(coverage_ratio, 4),
                "total_pixels": total_pixels,
                "tissue_pixels": int(tissue_pixels),
            },
        )

        return mask, coverage_ratio
