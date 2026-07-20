"""
pathoai/pipeline/orchestrator.py
================================
Top-Level Pipeline Orchestrator.

Provides a unified high-level interface executing end-to-end computational pathology
processing: WSI -> TumorROI -> CellDetection -> FusionResult -> STILScore -> ClinicalReport.

Author: PathoAI Research Team
Created: 2026-07-20
Milestone: 9.9 (Orchestration & Provenance)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from pathoai.config.experiment_config import ExperimentConfig
from pathoai.core.types import CellDetection, ClinicalReport, FusionResult, STILScore, TumorROI
from pathoai.detection.pipeline import DetectionPipeline
from pathoai.fusion.pipeline import FusionPipeline
from pathoai.scoring.pipeline import ScoringPipeline
from pathoai.tumor_bulk.pipeline import TumorBulkPipeline


class PipelineOrchestrator:
    """Top-level pipeline coordinator orchestrating end-to-end computational pathology execution."""

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        """
        Parameters
        ----------
        config : Optional[ExperimentConfig]
            Unified experiment configuration.
        """
        self.config = config or ExperimentConfig()
        self.tumor_pipeline = TumorBulkPipeline()
        self.detection_pipeline = DetectionPipeline()
        self.fusion_pipeline = FusionPipeline(
            max_distance_um=self.config.fusion_max_distance_um,
            grid_size=self.config.fusion_grid_size,
        )
        self.scoring_pipeline = ScoringPipeline(
            lymphocyte_diameter_um=self.config.lymphocyte_diameter_um,
            n_bootstrap_iterations=self.config.n_bootstrap_iterations,
            low_threshold=self.config.low_threshold,
            high_threshold=self.config.high_threshold,
        )

    def run(
        self,
        slide_id: str,
        image: np.ndarray,
        tumor_mask: np.ndarray,
        stroma_mask: Optional[np.ndarray] = None,
    ) -> ClinicalReport:
        """Executes end-to-end computational pathology pipeline on tissue slide input.

        Parameters
        ----------
        slide_id : str
            Source slide identifier.
        image : np.ndarray
            RGB image array of shape (H, W, 3).
        tumor_mask : np.ndarray
            Binary tumor bed mask of shape (H, W).
        stroma_mask : Optional[np.ndarray]
            Binary stroma mask of shape (H, W).

        Returns
        -------
        ClinicalReport
            Comprehensive clinical evaluation report DTO with model provenance.
        """
        # 1. Tumor Bulk Extraction
        _, rois = self.tumor_pipeline.process(tumor_mask, mpp=self.config.wsi_mpp)

        # 2. Cell Detection
        detections: List[CellDetection] = []
        if rois:
            for roi in rois:
                det_list = self.detection_pipeline.process_roi(image, roi, mpp=self.config.wsi_mpp)
                detections.extend(det_list)

        # 3. Spatial Fusion
        fusion_result: FusionResult = self.fusion_pipeline.process_fusion(
            rois=rois,
            detections=detections,
            mpp=self.config.wsi_mpp,
            slide_id=slide_id,
        )

        # 4. Clinical sTIL Scoring
        report: ClinicalReport = self.scoring_pipeline.process(fusion_result)

        # 5. Populate model & engine provenance versioning
        report.processing_metadata.update({
            "experiment_id": self.config.experiment_id,
            "segmentation_model": self.config.segmentation_model,
            "segmentation_version": self.config.segmentation_version,
            "detection_model": self.config.detection_model,
            "detection_version": self.config.detection_version,
            "scoring_algorithm": self.config.scoring_algorithm,
            "scoring_version": self.config.scoring_version,
        })

        return report
