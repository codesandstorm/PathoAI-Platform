"""
pathoai/tumor_bulk/pipeline.py
==============================
Tumor Bulk Pipeline Coordinator.

Coordinates the sequential stages of tumor bed morphology extraction,
filtering, region-property labeling, contouring, and ROI generation.

Author: PathoAI Research Team
Created: 2026-07-19
Milestone: 6.6
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from pathoai.core.types import TumorROI
from pathoai.tumor_bulk.connected_components import label_and_filter_tumor_regions
from pathoai.tumor_bulk.morphology import extract_tumor_bed
from pathoai.tumor_bulk.roi_generator import generate_rois


class TumorBulkPipeline:
    """Coordinating pipeline for Tumor Bulk extraction and ROI properties analysis."""

    def __init__(
        self,
        dilation_dist_um: float = 500.0,
        min_area_um2: float = 10000.0,
        class_label: str = "tumor_bulk",
    ) -> None:
        """
        Parameters
        ----------
        dilation_dist_um : float
            Clinical margin boundary in microns to expand the raw tumor cells.
        min_area_um2 : float
            Minimum area size threshold in square microns to filter out noise.
        class_label : str
            Clinical label assigned to the generated ROIs.
        """
        if dilation_dist_um < 0:
            raise ValueError(f"dilation_dist_um must be non-negative. Got: {dilation_dist_um}")
        if min_area_um2 < 0:
            raise ValueError(f"min_area_um2 must be non-negative. Got: {min_area_um2}")

        self.dilation_dist_um = dilation_dist_um
        self.min_area_um2 = min_area_um2
        self.class_label = class_label

    def process(
        self,
        tumor_mask: np.ndarray,
        mpp: float,
    ) -> Tuple[np.ndarray, List[TumorROI]]:
        """Executes the complete extraction pipeline on a tumor segmentation mask.

        Parameters
        ----------
        tumor_mask : np.ndarray
            Binary segmentation mask where 1 indicates invasive tumor nests.
        mpp : float
            Microns per pixel resolution.

        Returns
        -------
        Tuple[np.ndarray, List[TumorROI]]
            - Binary tumor bed mask of shape (H, W).
            - List of structured TumorROI objects.
        """
        # 1. Morphology: dilation and hole filling
        tumor_bed = extract_tumor_bed(
            tumor_mask=tumor_mask,
            mpp=mpp,
            dilation_dist_um=self.dilation_dist_um,
        )

        # 2. Components: Label and filter small noise components
        labeled_mask, _ = label_and_filter_tumor_regions(
            tumor_bed=tumor_bed,
            mpp=mpp,
            min_area_um2=self.min_area_um2,
        )

        # 3. ROI Generation: Bounding boxes and advanced morph metrics
        rois = generate_rois(
            labeled_mask=labeled_mask,
            mpp=mpp,
            class_label=self.class_label,
        )

        return tumor_bed, rois
